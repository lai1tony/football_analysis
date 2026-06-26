from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime
from typing import Any, Mapping

from action_policy import evaluate_action_policy, risk_level as action_risk_level, stake_for_action
from collection_repository import (
    expire_pending_manual_reviews,
    get_feedback_log,
    get_feedback_summary,
    get_latest_feature_snapshot,
    get_latest_learning_profile,
    get_latest_issue,
    get_match_analysis,
    get_prediction_run,
    init_db,
    list_backtest_rows,
    list_matches_by_issue,
    list_matches_pending_settlement,
    list_pending_manual_review_runs,
    list_prediction_runs,
    save_feedback_log,
    save_feature_snapshot,
    save_prediction_run,
    supersede_pending_manual_reviews,
    update_prediction_run_fields,
    upsert_match_results,
)
from collection_service import get_collection_failure_reason, summarize_issue_entries
from config_service import (
    ChatProtocolError,
    get_review_max_tokens,
    is_response_format_unsupported,
    request_openai_compatible_chat,
)
from feature_engine import (
    OUTCOMES,
    build_feature_snapshot,
    build_match_features,
    clamp,
    infer_match_datetime,
    normalize_probs,
    poisson_probability,
    probability_vector_for_outcome,
    safe_float,
    safe_int,
)
from learning_engine import (
    BASE_THRESHOLD_CONFIG,
    DEFAULT_MIN_ACTION_SHARE,
    DEFAULT_TARGET_HIT_RATE,
    apply_probability_calibration,
    evaluate_target_strategy_rule,
    get_active_learning_profile_config,
    _handicap_bucket_strategy_metrics,
    _hydrate_profile,
    _review_signal_summary,
)
from outcome_policy import effective_market_probs, evaluate_outcome_policy
from source_500_client import fetch_issue_results, fetch_result_from_match_url


ACTION_LEVELS = {"观望": 0, "轻仓": 1, "主推": 2}
ACTION_BY_LEVEL = {value: key for key, value in ACTION_LEVELS.items()}
REVIEW_DECISIONS = {"keep", "promote", "downgrade", "abstain"}
ARBITER_DECISIONS = {"allow", "allow_with_uplift", "downgrade", "manual_review", "skip"}
EXPERT_EVIDENCE_GRADES = {"strong", "adequate", "weak", "unsafe"}
REVIEW_STATUS_LABELS = {
    "keep": "保持",
    "promote": "提升",
    "downgrade": "降级",
    "abstain": "跳过",
    "skipped": "跳过",
    "failed": "失败",
}
ARBITER_STATUS_LABELS = {
    "allow": "allow",
    "allow_with_uplift": "allow_with_uplift",
    "downgrade": "downgrade",
    "manual_review": "manual_review",
    "skip": "skip",
    "not_triggered": "not_triggered",
    "skipped": "skipped",
    "failed": "failed",
}
EXECUTION_STATUS_LABELS = {
    "executable": "executable",
    "arbiter_downgraded": "二级仲裁降档",
    "arbiter_uplifted": "二级仲裁恢复",
    "manual_review_pending": "manual_review_pending",
    "manual_review_resolved": "manual_review_resolved",
    "expert_review_resolved": "expert_review_resolved",
    "expert_review_failed": "专家终审失败保守观望",
    "manual_review_expired": "已过期未处理",
    "manual_review_superseded": "已被�?run 覆盖",
}
ARBITER_MODEL_GAP_THRESHOLD = 0.10
ARBITER_LOW_QUALITY_THRESHOLD = 0.68
ARBITER_HIGH_EV_THRESHOLD = 0.12
ARBITER_HIGH_MARKET_BIAS_THRESHOLD = 0.08
# When data quality clears this floor, large divergence from the legacy
# market-assisted baseline is treated as the *signal we want*, not a risk
# to escalate. Below the floor we still escalate, because divergence on
# noisy data is more likely to be a parsing artifact than alpha.
ARBITER_QUALITY_TRUST_FLOOR = 0.78


def _merge_threshold_config(overrides: Mapping[str, Any] | None = None) -> dict[str, dict[str, float]]:
    merged: dict[str, dict[str, float]] = {
        gate_name: {key: safe_float(value) for key, value in gate.items()}
        for gate_name, gate in BASE_THRESHOLD_CONFIG.items()
    }
    if not isinstance(overrides, Mapping):
        return merged

    for gate_name, gate in overrides.items():
        if gate_name == "target_strategy" and isinstance(gate, Mapping):
            merged["target_strategy"] = dict(gate)
            continue
        if gate_name not in merged or not isinstance(gate, Mapping):
            continue
        for key, value in gate.items():
            if key not in merged[gate_name]:
                continue
            merged[gate_name][key] = safe_float(value)
    return merged


def _apply_learning_profile_to_blended(
    blended: Mapping[str, Any],
    learning_profile: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, float], int]:
    raw_probabilities = {
        "home": safe_float(blended["probabilities"]["home"]),
        "draw": safe_float(blended["probabilities"]["draw"]),
        "away": safe_float(blended["probabilities"]["away"]),
    }
    profile_id = int(_row_field(learning_profile, "learning_profile_id", 0) or 0)
    if not learning_profile or not learning_profile.get("uses_calibrator"):
        return dict(blended), {}, profile_id

    calibrated_probabilities = apply_probability_calibration(
        raw_probabilities,
        learning_profile.get("calibrator_params", {}),
    )
    sorted_probs = sorted(calibrated_probabilities.values(), reverse=True)
    calibrated_blended = dict(blended)
    calibrated_blended["probabilities"] = calibrated_probabilities
    calibrated_blended["margin"] = sorted_probs[0] - sorted_probs[1]
    return calibrated_blended, calibrated_probabilities, profile_id


def _target_strategy_from_thresholds(threshold_config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(threshold_config, Mapping):
        return {}
    strategy = threshold_config.get("target_strategy", {})
    if not isinstance(strategy, Mapping):
        return {}
    if str(strategy.get("strategy_kind", "") or "outcome") != "outcome":
        return {}
    return strategy


def _handicap_target_strategy_from_thresholds(threshold_config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(threshold_config, Mapping):
        return {}
    strategy = threshold_config.get("target_strategy", {})
    if not isinstance(strategy, Mapping):
        return {}
    return strategy if str(strategy.get("strategy_kind", "") or "") in {"handicap", "handicap_bucket_table"} else {}


def _handicap_odds_decimal(value: Any) -> float:
    odds = safe_float(value)
    return odds + 1.0 if 0 < odds < 1.5 else odds


def _handicap_learning_features(handicap_risk: Mapping[str, Any], side: str) -> dict[str, Any]:
    side = side if side in {"home", "away"} else ""
    other_side = "away" if side == "home" else "home"
    cover_prob = safe_float(handicap_risk.get(f"{side}_cover_prob")) if side else 0.0
    other_cover_prob = safe_float(handicap_risk.get(f"{other_side}_cover_prob")) if side else 0.0
    odds = _handicap_odds_decimal(handicap_risk.get(f"{side}_odds")) if side else 0.0
    expected_value = cover_prob * odds - 1.0 if odds > 0 else -1.0
    line = safe_float(handicap_risk.get("line"))
    initial_line = safe_float(handicap_risk.get("initial_line"))
    line_move = line - initial_line
    return {
        "side": side,
        "base_action": _action_label(str(handicap_risk.get("recommendation", "") or "")),
        "expected_value": expected_value,
        "confidence": safe_float(handicap_risk.get("confidence")),
        "quality_score": safe_float(handicap_risk.get("quality_score")),
        "cover_prob": cover_prob,
        "cover_margin": cover_prob - other_cover_prob,
        "odds": odds,
        "line_abs": abs(line),
        "line_support": -line_move if side == "home" else line_move if side == "away" else 0.0,
    }


def _handicap_learning_rule_passes(features: Mapping[str, Any], rule: Mapping[str, Any]) -> bool:
    side = str(features.get("side", "") or "")
    if side not in {"home", "away"}:
        return False
    sides = rule.get("sides", ())
    if sides and side not in set(sides):
        return False
    base_actions = rule.get("base_actions", ())
    if base_actions and _action_label(str(features.get("base_action", "") or "")) not in {
        _action_label(str(item or "")) for item in base_actions
    }:
        return False
    return (
        safe_float(features.get("odds")) > 0
        and safe_float(features.get("odds")) <= safe_float(rule.get("odds_max"), 10.0)
        and safe_float(features.get("expected_value")) >= safe_float(rule.get("ev_min"), -1.0)
        and safe_float(features.get("confidence")) >= safe_float(rule.get("confidence_min"))
        and safe_float(features.get("cover_prob")) >= safe_float(rule.get("cover_prob_min"))
        and safe_float(features.get("cover_margin")) >= safe_float(rule.get("cover_margin_min"), -1.0)
        and safe_float(features.get("quality_score")) >= safe_float(rule.get("quality_min"))
    )


def _handicap_bucket_floor(value: Any, width: float, offset: float = 0.0) -> float:
    width = max(safe_float(width), 0.001)
    return math.floor((safe_float(value) + offset) / width) * width - offset


def _handicap_bucket_feature(handicap_risk: Mapping[str, Any], name: str) -> Any:
    if name == "rec_side":
        side = str(handicap_risk.get("recommended_side", "") or "")
        return side if side in {"home", "away"} else ""
    if name == "rec_action":
        return _action_label(str(handicap_risk.get("recommendation", "") or ""))
    if name == "line25":
        return round(_handicap_bucket_floor(handicap_risk.get("line"), 0.25, 5.0), 2)
    if name == "line50":
        return round(_handicap_bucket_floor(handicap_risk.get("line"), 0.50, 5.0), 2)
    if name == "coverdiff10":
        return round(
            _handicap_bucket_floor(
                safe_float(handicap_risk.get("home_cover_prob"))
                - safe_float(handicap_risk.get("away_cover_prob")),
                0.10,
                2.0,
            ),
            2,
        )
    if name == "evdiff10":
        home_odds = _handicap_odds_decimal(handicap_risk.get("home_odds"))
        away_odds = _handicap_odds_decimal(handicap_risk.get("away_odds"))
        home_ev = safe_float(handicap_risk.get("home_cover_prob")) * home_odds - 1.0 if home_odds > 0 else -9.0
        away_ev = safe_float(handicap_risk.get("away_cover_prob")) * away_odds - 1.0 if away_odds > 0 else -9.0
        return round(_handicap_bucket_floor(home_ev - away_ev, 0.10, 5.0), 2)
    if name == "homeodds20":
        return round(_handicap_bucket_floor(_handicap_odds_decimal(handicap_risk.get("home_odds")), 0.20), 2)
    if name == "awayodds20":
        return round(_handicap_bucket_floor(_handicap_odds_decimal(handicap_risk.get("away_odds")), 0.20), 2)
    if name == "conf10":
        return round(_handicap_bucket_floor(handicap_risk.get("confidence"), 0.10), 2)
    if name == "quality10":
        return round(_handicap_bucket_floor(handicap_risk.get("quality_score"), 0.10), 2)
    return ""


def _handicap_bucket_key(handicap_risk: Mapping[str, Any], features: list[str]) -> str:
    return json.dumps(
        [_handicap_bucket_feature(handicap_risk, name) for name in features],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _handicap_bucket_side_for_table(handicap_risk: Mapping[str, Any], table: Mapping[str, Any]) -> str:
    features = [str(item) for item in table.get("features", []) if str(item or "").strip()]
    buckets = table.get("buckets", {})
    if not features or not isinstance(buckets, Mapping):
        return ""
    bucket = buckets.get(_handicap_bucket_key(handicap_risk, features))
    side = str(bucket.get("side", "") or "") if isinstance(bucket, Mapping) else ""
    return side if side in {"home", "away"} else ""


def _handicap_bucket_strategy_side(handicap_risk: Mapping[str, Any], strategy: Mapping[str, Any]) -> tuple[str, str]:
    side = _handicap_bucket_side_for_table(handicap_risk, strategy)
    if side:
        return side, "主表"
    fallback_tables = strategy.get("fallback_bucket_tables", ())
    if not isinstance(fallback_tables, list):
        return "", ""
    for table in fallback_tables:
        if not isinstance(table, Mapping):
            continue
        side = _handicap_bucket_side_for_table(handicap_risk, table)
        if side:
            return side, "补充表"
    return "", ""


def _apply_handicap_bucket_strategy(
    handicap_risk: Mapping[str, Any],
    strategy: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(handicap_risk)
    side, table_label = _handicap_bucket_strategy_side(result, strategy)
    if side in {"home", "away"}:
        result["recommendation"] = str(strategy.get("action", "轻仓") or "轻仓")
        result["recommended_side"] = side
        result["reason"] = (
            str(result.get("reason", "") or "")
            + f" 学习让球分桶策略命中：按历史闭环{table_label or '桶表'}执行让球方向。"
        ).strip()
    else:
        result["recommendation"] = "观望"
        result["recommended_side"] = ""
        result["reason"] = (
            str(result.get("reason", "") or "")
            + " 学习让球分桶策略未命中：按历史闭环桶表降为让球观望。"
        ).strip()
    return result


def _apply_handicap_learning_strategy(
    handicap_risk: Mapping[str, Any],
    threshold_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    strategy = _handicap_target_strategy_from_thresholds(threshold_config)
    if not strategy:
        return dict(handicap_risk)
    if str(strategy.get("strategy_kind", "") or "") == "handicap_bucket_table":
        return _apply_handicap_bucket_strategy(handicap_risk, strategy)
    result = dict(handicap_risk)
    side = str(result.get("recommended_side", "") or "")
    features = _handicap_learning_features(result, side)
    if _handicap_learning_rule_passes(features, strategy):
        result["recommendation"] = str(strategy.get("action", "轻仓") or "轻仓")
        result["recommended_side"] = side
        result["reason"] = (
            str(result.get("reason", "") or "")
            + " 学习让球策略命中：按历史闭环规则保留让球执行。"
        ).strip()
    else:
        result["recommendation"] = "观望"
        result["recommended_side"] = ""
        result["reason"] = (
            str(result.get("reason", "") or "")
            + " 学习让球策略未命中：按历史闭环规则降为让球观望。"
        ).strip()
    return result


def _emit_progress(progress_callback, **payload) -> None:
    if progress_callback is None:
        return
    progress_callback(**payload)


def _row_field(row: Mapping[str, Any] | None, key: str, default: Any = "") -> Any:
    if row is None:
        return default
    try:
        value = row[key]
    except Exception:  # noqa: BLE001
        return default
    return default if value is None else value


def _match_is_collected(row: Mapping[str, Any] | None) -> bool:
    return not get_collection_failure_reason(row)


def _choice_label(outcome: str) -> str:
    return {"home": "主胜", "draw": "平局", "away": "客胜"}.get(outcome, outcome)


def _outcome_label(value: str) -> str:
    text = str(value or "").strip()
    aliases = {
        "home": "home",
        "h": "home",
        "主胜": "home",
        "draw": "draw",
        "d": "draw",
        "平局": "draw",
        "away": "away",
        "a": "away",
        "客胜": "away",
    }
    return aliases.get(text.lower(), aliases.get(text, text))


def _format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _format_score_pair(score_item: tuple[tuple[int, int], float]) -> str:
    score, prob = score_item
    return f"{score[0]}-{score[1]} ({prob * 100:.1f}%)"


def _heat_probs_from_text(text: str, fallback: Mapping[str, float]) -> dict[str, float]:
    match = re.search(
        r"投注比\s*胜\s*(\d+(?:\.\d+)?)%\s*平\s*(\d+(?:\.\d+)?)%\s*负\s*(\d+(?:\.\d+)?)%",
        text or "",
    )
    if not match:
        return dict(fallback)
    win, draw, loss = [float(item) / 100.0 for item in match.groups()]
    return normalize_probs(win, draw, loss)


def _ensure_feature_snapshot(match_row: Mapping[str, Any]) -> Mapping[str, Any]:
    snapshot = get_latest_feature_snapshot(str(match_row["match_id"]))
    if snapshot is not None:
        payload_text = str(snapshot["feature_payload"] or "") if "feature_payload" in snapshot.keys() else ""
        if '"market_value"' in payload_text and '"asian_handicap"' in payload_text:
            return dict(snapshot)

    snapshot_payload = build_feature_snapshot(match_row)
    snapshot_id = save_feature_snapshot(snapshot_payload)
    snapshot_payload["snapshot_id"] = snapshot_id
    return snapshot_payload


def _low_score_correlation_adjustment(
    home_goals: int,
    away_goals: int,
    lambda_home: float,
    lambda_away: float,
    rho: float,
) -> float:
    if home_goals == 0 and away_goals == 0:
        return 1 - (lambda_home * lambda_away * rho)
    if home_goals == 0 and away_goals == 1:
        return 1 + (lambda_home * rho)
    if home_goals == 1 and away_goals == 0:
        return 1 + (lambda_away * rho)
    if home_goals == 1 and away_goals == 1:
        return 1 - rho
    return 1.0


def _handicap_side_label(side: str) -> str:
    return {"home": "主队让球", "away": "客队受让", "push": "走水"}.get(side, side or "-")


def _settle_handicap_result(actual_score: str, handicap_line: float) -> str:
    numbers = [int(item) for item in re.findall(r"\d+", str(actual_score or ""))[:2]]
    if len(numbers) != 2:
        return ""
    adjusted_margin = float(numbers[0] - numbers[1]) + safe_float(handicap_line)
    if adjusted_margin > 0:
        return "home"
    if adjusted_margin < 0:
        return "away"
    return "push"


def _handicap_cover_probabilities(
    lambda_home: float,
    lambda_away: float,
    handicap_line: float,
) -> dict[str, float]:
    home_cover = 0.0
    away_cover = 0.0
    push = 0.0
    total = 0.0
    for home_goals in range(9):
        for away_goals in range(9):
            prob = poisson_probability(lambda_home, home_goals) * poisson_probability(lambda_away, away_goals)
            total += prob
            adjusted_margin = float(home_goals - away_goals) + safe_float(handicap_line)
            if adjusted_margin > 0:
                home_cover += prob
            elif adjusted_margin < 0:
                away_cover += prob
            else:
                push += prob
    if total <= 0:
        return {"home": 0.5, "away": 0.5, "push": 0.0}
    return {
        "home": home_cover / total,
        "away": away_cover / total,
        "push": push / total,
    }


def _handicap_dimension_evidence(
    *,
    features: Mapping[str, Any],
    row: Mapping[str, Any] | None,
    side: str,
    move_support: float,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []

    def add_home_edge(name: str, label: str, value: float, scale: float, weight: float, available: bool = True) -> None:
        if not available or weight <= 0:
            return
        home_edge = clamp(safe_float(value) / max(scale, 0.001), -1.0, 1.0)
        support = home_edge if side == "home" else -home_edge
        items.append(
            {
                "name": name,
                "label": label,
                "support": round(support, 4),
                "weight": weight,
                "weighted_support": round(support * weight, 4),
            }
        )

    def add_selected_edge(name: str, label: str, value: float, scale: float, weight: float, available: bool = True) -> None:
        if not available or weight <= 0:
            return
        support = clamp(safe_float(value) / max(scale, 0.001), -1.0, 1.0)
        items.append(
            {
                "name": name,
                "label": label,
                "support": round(support, 4),
                "weight": weight,
                "weighted_support": round(support * weight, 4),
            }
        )

    recent_home = features.get("recent_home") if isinstance(features.get("recent_home"), Mapping) else {}
    recent_away = features.get("recent_away") if isinstance(features.get("recent_away"), Mapping) else {}
    split = features.get("split") if isinstance(features.get("split"), Mapping) else {}
    lineup = features.get("lineup") if isinstance(features.get("lineup"), Mapping) else {}
    schedule = features.get("schedule") if isinstance(features.get("schedule"), Mapping) else {}
    h2h = features.get("h2h") if isinstance(features.get("h2h"), Mapping) else {}
    xg = features.get("xg") if isinstance(features.get("xg"), Mapping) else {}
    market_probs = features.get("market_probs") if isinstance(features.get("market_probs"), Mapping) else {}
    market_value = features.get("market_value") if isinstance(features.get("market_value"), Mapping) else {}

    add_home_edge("strength", "实力评级", features.get("rating_gap", 0.0), 180.0, 0.11)
    add_home_edge(
        "market_value",
        "球队身价",
        features.get("market_value_rating_gap", 0.0),
        150.0,
        0.07,
        bool(safe_int(market_value.get("coverage"))),
    )
    form_gap = (
        (safe_float(recent_home.get("points_per_game")) - safe_float(recent_away.get("points_per_game"))) * 0.40
        + safe_float(features.get("form_residual_gap")) * 0.60
    )
    add_home_edge("recent_form", "近期状态", form_gap, 1.4, 0.11)
    goal_gap = (
        (safe_float(recent_home.get("goal_diff_per_game")) - safe_float(recent_away.get("goal_diff_per_game"))) * 0.40
        + safe_float(features.get("goal_diff_residual_gap")) * 0.60
    )
    add_home_edge("goal_diff", "攻防净胜", goal_gap, 1.3, 0.10)
    add_home_edge(
        "home_away",
        "主客场拆分",
        safe_float(split.get("home_ppg")) - safe_float(split.get("away_ppg")),
        1.8,
        0.09,
    )
    lineup_available = bool(safe_int(lineup.get("data_available", 0)))
    add_home_edge(
        "lineup",
        "阵容伤停",
        safe_float(lineup.get("home_availability")) - safe_float(lineup.get("away_availability")),
        0.25,
        0.10,
        lineup_available,
    )
    schedule_edge = (
        clamp(safe_float(schedule.get("rest_advantage")) / 7.0, -0.5, 0.5) * 0.70
        + clamp(safe_float(schedule.get("schedule_gap")) * 0.10, -0.4, 0.4)
    )
    add_home_edge("schedule", "赛程体能", schedule_edge, 0.50, 0.07)
    add_home_edge("h2h", "历史交锋", features.get("h2h_edge", h2h.get("edge", 0.0)), 1.0, 0.05)
    xg_available = bool(safe_int(xg.get("coverage")))
    xg_edge = (
        safe_float(xg.get("home_xg_per_game"))
        - safe_float(xg.get("home_xga_per_game"))
        - safe_float(xg.get("away_xg_per_game"))
        + safe_float(xg.get("away_xga_per_game"))
    )
    add_home_edge("xg", "xG质量", xg_edge, 1.4, 0.11, xg_available)
    market_available = safe_float(market_probs.get("home")) > 0 and safe_float(market_probs.get("away")) > 0
    add_home_edge(
        "europe_market",
        "欧赔先验",
        safe_float(market_probs.get("home")) - safe_float(market_probs.get("away")),
        0.35,
        0.07,
        market_available,
    )
    heat_text = str(row.get("betting_heat_summary", "") or "") if isinstance(row, Mapping) else ""
    if "投注比" in heat_text:
        heat = _heat_probs_from_text(heat_text, {"home": 0.0, "draw": 0.0, "away": 0.0})
        add_home_edge("betting_heat", "投注热度", heat["home"] - heat["away"], 0.45, 0.04)
    add_selected_edge("asian_market", "亚盘水位", move_support, 0.08, 0.12)
    add_home_edge(
        "motivation",
        "战意赛程",
        safe_float(features.get("motivation_signal")),
        0.15,
        0.03,
        safe_float(features.get("motivation_signal")) > 0,
    )

    total_weight = sum(safe_float(item["weight"]) for item in items)
    score = (
        sum(safe_float(item["weighted_support"]) for item in items) / total_weight
        if total_weight > 0
        else 0.0
    )
    return {
        "score": round(clamp(score, -1.0, 1.0), 4),
        "coverage": round(clamp(total_weight, 0.0, 1.0), 4),
        "items": items,
    }


def _format_handicap_dimension_summary(evidence: Mapping[str, Any]) -> str:
    items = evidence.get("items") if isinstance(evidence.get("items"), list) else []
    if not items:
        return "维度证据不足"
    top_items = sorted(
        items,
        key=lambda item: abs(safe_float(item.get("weighted_support"))),
        reverse=True,
    )[:5]
    return "，".join(
        f"{item.get('label', item.get('name', '维度'))}{safe_float(item.get('support')):+.2f}"
        for item in top_items
    )


def evaluate_handicap_recommendation(
    *,
    features: Mapping[str, Any],
    quant: Mapping[str, Any],
    quality: Mapping[str, Any],
    row: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    market = features.get("asian_handicap") if isinstance(features.get("asian_handicap"), Mapping) else {}
    line = safe_float(market.get("current_line"))
    initial_line = safe_float(market.get("initial_line"))
    home_odds = safe_float(market.get("current_home_odds"))
    away_odds = safe_float(market.get("current_away_odds"))
    initial_home_odds = safe_float(market.get("initial_home_odds"))
    initial_away_odds = safe_float(market.get("initial_away_odds"))
    cover_probs = _handicap_cover_probabilities(
        safe_float(quant.get("lambda_home")),
        safe_float(quant.get("lambda_away")),
        line,
    )
    home_odds_decimal = home_odds + 1.0 if home_odds > 0 and home_odds < 1.5 else home_odds
    away_odds_decimal = away_odds + 1.0 if away_odds > 0 and away_odds < 1.5 else away_odds
    ev_home = cover_probs["home"] * home_odds_decimal - 1.0 if home_odds_decimal > 0 else -1.0
    ev_away = cover_probs["away"] * away_odds_decimal - 1.0 if away_odds_decimal > 0 else -1.0
    line_move = line - initial_line
    home_odds_move = home_odds - initial_home_odds if home_odds and initial_home_odds else 0.0
    away_odds_move = away_odds - initial_away_odds if away_odds and initial_away_odds else 0.0
    if ev_home >= ev_away:
        side = "home"
        selected_ev = ev_home
        selected_odds = home_odds_decimal
        move_support = -line_move * 0.08 - home_odds_move * 0.05
    else:
        side = "away"
        selected_ev = ev_away
        selected_odds = away_odds_decimal
        move_support = line_move * 0.08 - away_odds_move * 0.05
    dimension_evidence = _handicap_dimension_evidence(
        features=features,
        row=row,
        side=side,
        move_support=move_support,
    )
    dimension_score = safe_float(dimension_evidence.get("score"))
    dimension_coverage = safe_float(dimension_evidence.get("coverage"))
    dimension_adjusted_ev = selected_ev + dimension_score * 0.04
    confidence = clamp(
        0.45
        + max(selected_ev, -0.20) * 0.55
        + abs(cover_probs["home"] - cover_probs["away"]) * 0.22
        + safe_float(quality.get("score")) * 0.16
        + move_support
        + dimension_score * 0.18
        + (dimension_coverage - 0.65) * 0.05,
        0.0,
        0.95,
    )
    if home_odds <= 0 or away_odds <= 0:
        action = "观望"
        reason = "让球盘口缺失或赔率不可用，保留原胜平负口径并让球观望。"
    elif dimension_adjusted_ev >= 0.08 and confidence >= 0.62 and dimension_score >= -0.05:
        action = "主推"
        reason = "让球盘模型概率相对即时赔率具备较高正EV，且多维采集证据支持。"
    elif dimension_adjusted_ev >= 0.02 and confidence >= 0.52 and dimension_score >= -0.15:
        action = "轻仓"
        reason = "让球盘模型概率相对即时赔率具备轻度正EV，多维采集证据允许轻仓。"
    else:
        action = "观望"
        reason = "让球盘EV、多维证据或盘口变化支持不足。"
    return {
        "recommendation": action,
        "recommended_side": side if action != "观望" else "",
        "line": line,
        "initial_line": initial_line,
        "home_odds": home_odds,
        "away_odds": away_odds,
        "initial_home_odds": initial_home_odds,
        "initial_away_odds": initial_away_odds,
        "home_cover_prob": cover_probs["home"],
        "away_cover_prob": cover_probs["away"],
        "push_prob": cover_probs["push"],
        "expected_value": selected_ev,
        "confidence": confidence,
        "quality_score": safe_float(quality.get("score")),
        "dimension_score": dimension_score,
        "dimension_coverage": dimension_coverage,
        "dimension_adjusted_value": dimension_adjusted_ev,
        "dimension_support": dimension_evidence["items"],
        "reason": (
            f"{reason} 当前盘口 {line:+.3f}，初盘 {initial_line:+.3f}；"
            f"主/客覆盖率 {cover_probs['home']:.1%}/{cover_probs['away']:.1%}；"
            f"主/客EV {ev_home:+.3f}/{ev_away:+.3f}；"
            f"维度证据 {dimension_score:+.3f}（{_format_handicap_dimension_summary(dimension_evidence)}）；"
            f"综合证据 {dimension_adjusted_ev:+.3f}；选择 {_handicap_side_label(side)}。"
        ),
        "selected_odds": selected_odds,
    }


def _recommended_market_odds(row: Mapping[str, Any]) -> float:
    outcome = str(row.get("recommended_outcome", "") or "")
    if outcome == "home":
        return safe_float(row.get("market_odds_home"))
    if outcome == "draw":
        return safe_float(row.get("market_odds_draw"))
    if outcome == "away":
        return safe_float(row.get("market_odds_away"))
    return 0.0


def _action_label(value: str) -> str:
    action = str(value or "").strip()
    if not action:
        return "观望"
    action = action.split()[0].lower()
    aliases = {
        "watch": "观望",
        "hold": "观望",
        "skip": "观望",
        "abstain": "观望",
        "light": "轻仓",
        "small": "轻仓",
        "lean": "轻仓",
        "main": "主推",
        "strong": "主推",
        "top": "主推",
    }
    return aliases.get(action, str(value or "").strip().split()[0])


def _stake_for_action(action: str, risk: Mapping[str, Any]) -> float:
    if action in ACTION_LEVELS:
        return stake_for_action(action, risk)
    return safe_float(risk.get("stake_pct"))


def _build_openai_messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _looks_truncated_json(raw_text: str) -> bool:
    text = (raw_text or "").strip()
    if not text:
        return False
    return text.startswith("{") and (not text.endswith("}") or text.count("{") > text.count("}"))


def _extract_json_payload(raw_text: str) -> dict[str, Any]:
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("empty response")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("响应不是有效 JSON") from None
        return json.loads(match.group(0))


# Plan-B unified review: rule table that translates a single
# (decision, evidence_grade) pair into the 9-field internal structure
# expected by resolve_recommendation. This keeps downstream logic
# unchanged while letting the LLM output just two categorical labels.
_REVIEW_RULE_TABLE = {
    # decision == "approve"
    ("approve", "strong"): {
        "decision": "keep",
        "outcome_decision": "confirm",
        "stake_multiplier": 1.00,
        "confidence_delta": 0.04,
    },
    ("approve", "adequate"): {
        "decision": "keep",
        "outcome_decision": "confirm",
        "stake_multiplier": 0.85,
        "confidence_delta": 0.00,
    },
    ("approve", "weak"): {
        # weak evidence + approve = play it lighter
        "decision": "downgrade",
        "outcome_decision": "confirm",
        "stake_multiplier": 0.60,
        "confidence_delta": -0.04,
    },
    # decision == "reject"
    ("reject", "strong"): {
        "decision": "abstain",
        "outcome_decision": "veto_to_watch",
        "stake_multiplier": 0.0,
        "confidence_delta": -0.06,
    },
    ("reject", "adequate"): {
        "decision": "abstain",
        "outcome_decision": "veto_to_watch",
        "stake_multiplier": 0.0,
        "confidence_delta": -0.06,
    },
    ("reject", "weak"): {
        "decision": "abstain",
        "outcome_decision": "veto_to_watch",
        "stake_multiplier": 0.0,
        "confidence_delta": -0.06,
    },
    ("reject", "unsafe"): {
        "decision": "abstain",
        "outcome_decision": "veto_to_watch",
        "stake_multiplier": 0.0,
        "confidence_delta": -0.08,
    },
}


def _translate_unified_review(
    *,
    decision_raw: str,
    evidence_grade: str,
    reason: str,
    risk_flags: list[str],
    current_action: str,
    current_outcome: str,
) -> dict[str, Any]:
    """Plan-B: translate a (decision, evidence_grade) pair into the 9-field
    internal review payload that ``resolve_recommendation`` already knows
    how to consume. Lets the LLM emit just two categorical labels while
    keeping all downstream code unchanged.
    """

    decision = (decision_raw or "").strip().lower()
    if decision in {"yes", "approve", "ok", "confirm", "go"}:
        decision = "approve"
    elif decision in {"no", "reject", "veto", "skip", "abstain"}:
        decision = "reject"
    else:
        raise ValueError("decision 必须�?approve �?reject")

    grade = (evidence_grade or "").strip().lower()
    if grade not in {"strong", "adequate", "weak", "unsafe"}:
        # Default to "adequate" so reject without grade still works.
        grade = "unsafe" if decision == "reject" else "adequate"
    if decision == "approve" and grade == "unsafe":
        # Approve + unsafe contradicts itself; honour the safety side.
        decision = "reject"

    rule_key = (decision, grade)
    rule = _REVIEW_RULE_TABLE.get(rule_key) or _REVIEW_RULE_TABLE[("reject", "unsafe")]

    target_action = current_action
    if rule["decision"] == "downgrade":
        target_action = _single_step_downgrade(current_action)
    elif rule["decision"] == "abstain":
        target_action = "观望"

    return {
        # ------- internal 9-field structure preserved -------
        "decision": rule["decision"],
        "target_action": target_action,
        "reason": reason,
        "risk_flags": risk_flags[:6],
        "evidence_grade": grade,
        "confidence_delta": clamp(safe_float(rule["confidence_delta"]), -0.12, 0.08),
        "stake_multiplier": clamp(safe_float(rule["stake_multiplier"]), 0.0, 1.0),
        "outcome_decision": rule["outcome_decision"],
        "target_outcome": current_outcome,
        "outcome_reason": reason if rule["outcome_decision"] != "confirm" else "",
        # ------- bookkeeping for UI/audit -------
        "unified_decision": decision,
    }


def _unified_review_status(review: dict[str, Any] | None) -> str:
    """Return a display label for the unified LLM review status."""
    if not review:
        return "not reviewed"
    decision = (review.get("decision") or "").strip().lower()
    if decision == "approve":
        return "approved"
    if decision == "reject":
        return "rejected"
    return "unknown"


def _unified_review_verdict(review: dict[str, Any] | None) -> str:
    """Return a short display verdict for the unified LLM review."""
    if not review:
        return "No LLM review"
    decision = (review.get("decision") or "").strip().lower()
    grade = (review.get("evidence_grade") or "").strip().upper()
    reason = review.get("reason", "")
    verdict_map = {
        "approve": f"evidence={grade}; LLM approved: {reason}",
        "reject": f"evidence={grade}; LLM rejected: {reason}",
    }
    return verdict_map.get(decision, f"LLM: {reason}")


def _parse_review_payload(
    raw_text: str,
    current_action: str = "",
    current_outcome: str = "",
) -> dict[str, Any]:
    """Parse LLM review output, accepting both the new 3-field schema
    (Plan-B unified) and the legacy 9-field schema for backward
    compatibility.
    """

    payload = _extract_json_payload(raw_text)
    reason = str(payload.get("reason", "")).strip()
    if not reason:
        raise ValueError("reason 不能为空")
    risk_flags = payload.get("risk_flags", [])
    if not isinstance(risk_flags, list):
        raise ValueError("risk_flags must be a list")
    clean_flags = [str(item).strip() for item in risk_flags if str(item).strip()]

    # Detect new schema: presence of "decision" with value approve/reject
    raw_decision = str(payload.get("decision", "")).strip().lower()
    is_unified = raw_decision in {"approve", "reject", "yes", "no", "ok", "veto", "go", "skip"}
    if is_unified and "target_action" not in payload:
        return _translate_unified_review(
            decision_raw=raw_decision,
            evidence_grade=str(payload.get("evidence_grade", "")),
            reason=reason,
            risk_flags=clean_flags,
            current_action=_action_label(current_action) or "观望",
            current_outcome=str(current_outcome or ""),
        )

    # Legacy 9-field schema (kept for backward compatibility / mixed
    # responses from older models).
    decision = raw_decision
    target_action = _action_label(str(payload.get("target_action", "")).strip())
    evidence_grade = str(payload.get("evidence_grade", "") or "").strip().lower()
    confidence_delta = safe_float(payload.get("confidence_delta"))
    stake_multiplier = safe_float(payload.get("stake_multiplier"), 1.0)
    outcome_decision = str(payload.get("outcome_decision", "") or "confirm").strip().lower()
    target_outcome = _outcome_label(str(payload.get("target_outcome", "") or ""))

    if decision not in REVIEW_DECISIONS:
        raise ValueError("decision 非法")
    if target_action not in ACTION_LEVELS:
        raise ValueError("target_action 非法")
    if evidence_grade and evidence_grade not in {"strong", "adequate", "weak", "unsafe"}:
        raise ValueError("evidence_grade 非法")
    if outcome_decision and outcome_decision not in {"confirm", "challenge", "veto_to_watch"}:
        raise ValueError("outcome_decision 非法")
    if outcome_decision in {"confirm", "challenge"} and target_outcome and target_outcome not in OUTCOMES:
        raise ValueError("target_outcome 非法")

    return {
        "decision": decision,
        "target_action": target_action,
        "reason": reason,
        "risk_flags": clean_flags[:6],
        "evidence_grade": evidence_grade,
        "confidence_delta": clamp(confidence_delta, -0.12, 0.08),
        "stake_multiplier": clamp(stake_multiplier, 0.0, 1.0),
        "outcome_decision": outcome_decision or "confirm",
        "target_outcome": target_outcome if target_outcome in OUTCOMES else "",
        "outcome_reason": str(payload.get("outcome_reason", "") or "").strip(),
    }


def _parse_arbiter_payload(raw_text: str) -> dict[str, Any]:
    payload = _extract_json_payload(raw_text)
    decision = str(payload.get("decision", "")).strip().lower()
    target_action = _action_label(str(payload.get("target_action", "")).strip())
    reason = str(payload.get("reason", "")).strip()
    risk_flags = payload.get("risk_flags", [])

    if decision not in ARBITER_DECISIONS:
        raise ValueError("decision 非法")
    if target_action not in ACTION_LEVELS:
        raise ValueError("target_action 非法")
    if not reason:
        raise ValueError("reason 不能为空")
    if not isinstance(risk_flags, list):
        raise ValueError("risk_flags must be a list")

    clean_flags = [str(item).strip() for item in risk_flags if str(item).strip()]
    return {
        "decision": decision,
        "target_action": target_action,
        "reason": reason,
        "risk_flags": clean_flags[:6],
    }


def _parse_expert_review_payload(raw_text: str, current_outcome: str) -> dict[str, Any]:
    payload = _extract_json_payload(raw_text)
    required_fields = {"target_action", "reason", "risk_flags", "evidence_grade", "stake_multiplier"}
    missing_fields = [field for field in sorted(required_fields) if field not in payload]
    if missing_fields:
        raise ValueError(f"缺少专家终审字段: {', '.join(missing_fields)}")

    target_action = _action_label(str(payload.get("target_action", "")).strip())
    reason = str(payload.get("reason", "")).strip()
    risk_flags = payload.get("risk_flags", [])
    evidence_grade = str(payload.get("evidence_grade", "") or "").strip().lower()
    stake_multiplier = safe_float(payload.get("stake_multiplier"), 1.0)
    target_outcome = _outcome_label(str(payload.get("target_outcome", "") or ""))

    if target_action not in ACTION_LEVELS:
        raise ValueError("target_action 非法")
    if not reason:
        raise ValueError("reason 不能为空")
    if not isinstance(risk_flags, list):
        raise ValueError("risk_flags must be a list")
    if evidence_grade not in EXPERT_EVIDENCE_GRADES:
        raise ValueError("evidence_grade 非法")
    if evidence_grade == "weak" and target_action == "主推":
        target_action = "轻仓"
        reason = "weak evidence; downgrade to light: " + reason

    direction_guarded = False
    current_outcome = str(current_outcome or "").strip()
    if target_outcome and target_outcome in OUTCOMES and target_outcome != current_outcome:
        direction_guarded = True
        target_action = "观望"
        reason = "专家终审尝试改方向，已强制观望：" + reason

    clean_flags = [str(item).strip() for item in risk_flags if str(item).strip()]
    return {
        "target_action": target_action,
        "reason": reason,
        "risk_flags": clean_flags[:6],
        "evidence_grade": evidence_grade,
        "stake_multiplier": clamp(stake_multiplier, 0.0, 1.0),
        "target_outcome": target_outcome if target_outcome in OUTCOMES else "",
        "direction_guarded": direction_guarded,
    }


def _normalize_score_text(value: Any) -> str:
    match = re.search(r"(\d+)\s*[:\-]\s*(\d+)", str(value or ""))
    if not match:
        return ""
    home_goals = min(max(int(match.group(1)), 0), 9)
    away_goals = min(max(int(match.group(2)), 0), 9)
    return f"{home_goals}-{away_goals}"


def _score_entry_score(value: Any) -> str:
    if isinstance(value, Mapping):
        return _normalize_score_text(value.get("score") or value.get("比分"))
    return _normalize_score_text(value)


def _score_entry_probability(value: Any) -> float:
    raw_probability = value.get("probability") if isinstance(value, Mapping) else None
    if raw_probability is None and isinstance(value, Mapping):
        raw_probability = value.get("prob") or value.get("概率")
    if raw_probability is None:
        return 0.0
    probability = safe_float(str(raw_probability).replace("%", ""))
    if probability > 1:
        probability /= 100.0
    return clamp(probability, 0.0, 1.0)


def _score_from_labeled_line(raw_text: str, label: str) -> tuple[str, float]:
    pattern = rf"{label}\s*[：:]\s*(\d+\s*[:\-]\s*\d+)(?:\s*概率\s*[：:]\s*([0-9.]+)\s*%?)?"
    match = re.search(pattern, raw_text)
    if not match:
        return "", 0.0
    probability = safe_float(match.group(2), 0.0)
    if probability > 1:
        probability /= 100.0
    return _normalize_score_text(match.group(1)), clamp(probability, 0.0, 1.0)


def _parse_score_prediction_payload(raw_text: str) -> dict[str, Any]:
    try:
        payload = _extract_json_payload(raw_text)
    except ValueError:
        payload = {}
    score = _normalize_score_text(payload.get("score"))
    if not score:
        score = _score_entry_score(payload.get("most_likely"))
    if not score:
        score = _normalize_score_text(payload.get("most_likely_score"))
    if not score:
        score, _ = _score_from_labeled_line(raw_text, "最可能比分")
    if not score:
        raise ValueError("score must be formatted like 1-0")

    confidence = clamp(safe_float(payload.get("confidence"), 0.0), 0.0, 1.0)
    if confidence <= 0.0:
        confidence = _score_entry_probability(payload.get("most_likely"))
    if confidence <= 0.0:
        _, confidence = _score_from_labeled_line(raw_text, "最可能比分")
    if confidence <= 0.0:
        confidence = 0.5

    reason = str(payload.get("reason", "") or "").strip()
    if not reason:
        confidence_label = str(
            payload.get("confidence_label")
            or payload.get("confidence_level")
            or payload.get("信心")
            or ""
        ).strip()
        if not confidence_label:
            confidence_match = re.search(r"信心\s*[：:]\s*(低|中|高)", raw_text)
            confidence_label = confidence_match.group(1) if confidence_match else ""
        reason = f"信心：{confidence_label}" if confidence_label else "模型给出比分"

    alternatives = payload.get("alternatives", [])
    if not isinstance(alternatives, list):
        alternatives = []
    alternatives = [
        *alternatives,
        payload.get("second_1"),
        payload.get("second_2"),
        payload.get("upset"),
        payload.get("upset_score"),
    ]
    line_scores = [
        _score_from_labeled_line(raw_text, "次选比分一")[0],
        _score_from_labeled_line(raw_text, "次选比分二")[0],
        _score_from_labeled_line(raw_text, "冷门比分")[0],
    ]
    normalized_alternatives = [
        candidate
        for candidate in (_score_entry_score(item) for item in alternatives)
        if candidate and candidate != score
    ][:3]
    for candidate in line_scores:
        if candidate and candidate != score and candidate not in normalized_alternatives:
            normalized_alternatives.append(candidate)
        if len(normalized_alternatives) >= 3:
            break
    return {
        "score": score,
        "confidence": confidence,
        "reason": reason,
        "alternatives": normalized_alternatives,
    }


def _truncate_detail(text: Any, limit: int = 360) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 3]}..."


def _build_review_failure_detail(
    *,
    endpoint: str = "",
    raw_text: str = "",
    diagnostic: str = "",
) -> str:
    parts: list[str] = []
    if endpoint:
        parts.append(f"endpoint={endpoint}")
    if diagnostic:
        parts.append(_truncate_detail(diagnostic, 360))
    if raw_text:
        parts.append(f"response={_truncate_detail(raw_text, 360)}")
    return " | ".join(part for part in parts if part)


def _classify_review_failure(
    exc: BaseException,
    *,
    endpoint: str = "",
    raw_text: str = "",
) -> tuple[str, str]:
    if isinstance(exc, ChatProtocolError):
        diagnostic = exc.diagnostic
        detail = _build_review_failure_detail(endpoint=endpoint, diagnostic=diagnostic, raw_text=raw_text)
        if "finish_reason=length" in diagnostic.lower():
            return "复核模型输出被截断", detail
        if exc.public_message:
            return str(exc.public_message), detail

    text = str(exc)
    lower_text = text.lower()
    diagnostic = _build_review_failure_detail(endpoint=endpoint, diagnostic=text, raw_text=raw_text)

    if isinstance(exc, ValueError):
        if _looks_truncated_json(raw_text):
            return "复核模型 JSON 被截断", diagnostic
        if "json" in lower_text:
            return "复核模型返回非 JSON 内容", diagnostic
        return "复核模型返回无效 JSON", diagnostic
    if "429" in text or "concurrency limit" in lower_text or "rate limit" in lower_text:
        return "review model rate limited", diagnostic
    if "timeout" in lower_text or "timed out" in lower_text:
        return "review model request timed out", diagnostic
    return "review model request failed", diagnostic


def _request_review_response(
    base_url: str,
    api_key: str,
    model: str,
    *,
    messages: list[dict[str, str]],
) -> dict[str, Any]:
    request_kwargs = {
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": get_review_max_tokens(),
        "timeout": 45,
        "max_retries": 2,
        "retry_backoff_seconds": 2.0,
        "min_interval_seconds": 2.0,
        "serialize_requests": True,
        "require_non_empty_content": True,
    }
    try:
        return request_openai_compatible_chat(
            base_url,
            api_key,
            model,
            response_format={"type": "json_object"},
            **request_kwargs,
        )
    except Exception as exc:  # noqa: BLE001
        if not is_response_format_unsupported(exc):
            raise
    return request_openai_compatible_chat(
        base_url,
        api_key,
        model,
        **request_kwargs,
    )


def _reasoning_controls_unsupported(exc: BaseException) -> bool:
    text = str(exc).lower()
    if not any(marker in text for marker in ("reasoning", "reasoning_effort")):
        return False
    return any(
        marker in text
        for marker in (
            "unsupported",
            "not support",
            "not supported",
            "unknown parameter",
            "invalid parameter",
            "invalid_request_error",
            "not allowed",
            "unrecognized",
        )
    )


def _request_score_prediction_response(
    base_url: str,
    api_key: str,
    model: str,
    *,
    messages: list[dict[str, str]],
    max_tokens: int = 320,
) -> dict[str, Any]:
    request_kwargs = {
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "timeout": 45,
        "max_retries": 2,
        "retry_backoff_seconds": 2.0,
        "min_interval_seconds": 2.0,
        "serialize_requests": True,
        "require_non_empty_content": False,
    }
    try:
        return request_openai_compatible_chat(
            base_url,
            api_key,
            model,
            response_format={"type": "json_object"},
            **request_kwargs,
        )
    except Exception as exc:  # noqa: BLE001
        if not is_response_format_unsupported(exc):
            raise
    return request_openai_compatible_chat(
        base_url,
        api_key,
        model,
        **request_kwargs,
    )


def _promote_action(current_action: str, target_action: str) -> str:
    current_level = ACTION_LEVELS.get(current_action, 0)
    target_level = ACTION_LEVELS.get(target_action, current_level)
    if target_level <= current_level:
        return current_action
    return ACTION_BY_LEVEL[min(current_level + 1, target_level)]


def _downgrade_action(current_action: str, target_action: str) -> str:
    current_level = ACTION_LEVELS.get(current_action, 0)
    target_level = ACTION_LEVELS.get(target_action, current_level)
    if target_level >= current_level:
        return current_action
    return ACTION_BY_LEVEL[max(current_level - 1, target_level)]


def _review_status_label(review: Mapping[str, Any]) -> str:
    status = str(review.get("status", "") or "")
    if status == "completed":
        decision = str(review.get("decision", "") or "")
        return REVIEW_STATUS_LABELS.get(decision, "跳过")
    return REVIEW_STATUS_LABELS.get(status, "跳过")


def _arbiter_status_label(review: Mapping[str, Any]) -> str:
    status = str(review.get("status", "") or "")
    if status == "completed":
        decision = str(review.get("decision", "") or "")
        return ARBITER_STATUS_LABELS.get(decision, "跳过")
    return ARBITER_STATUS_LABELS.get(status, "跳过")


def _single_step_downgrade(action: str) -> str:
    current_level = ACTION_LEVELS.get(action, 0)
    if current_level <= 0:
        return action
    return ACTION_BY_LEVEL[current_level - 1]


def _llm_action_gate(
    action: str,
    *,
    low_odds_favorite_guard: bool = False,
    review_status: str = "",
    review_decision: str = "",
    review_target_action: str = "",
    arbiter_status: str = "",
    arbiter_decision: str = "",
    arbiter_target_action: str = "",
    manual_review_status: str = "",
) -> tuple[str, str]:
    """Use LLM review as a safety gate for executable action levels.

    The historical review data is noisy as an uplift signal, but it is useful
    as a veto/downgrade layer. Do not let LLM alone create extra action; let
    the action policy find candidates, then let LLM/arbiter reduce risk.
    """

    gated_action = _action_label(action)
    if gated_action == "观望":
        return gated_action, ""

    manual_status = str(manual_review_status or "").strip()
    if manual_status in {"pending", "expired", "superseded"}:
        return "观望", "LLM/manual review is not complete; force watch."

    status = str(review_status or "").strip()
    decision = str(review_decision or "").strip()
    target = _action_label(str(review_target_action or ""))
    if status == "failed":
        return "观望", "LLM review failed; force watch."
    if status == "completed":
        if decision == "reject" or (decision == "abstain" and not low_odds_favorite_guard):
            return "观望", "LLM rejected or abstained; force watch."
        if decision == "downgrade":
            next_action = _downgrade_action(gated_action, target)
            if next_action != gated_action:
                return next_action, f"LLM downgraded action to {next_action}."

    arb_status = str(arbiter_status or "").strip()
    arb_decision = str(arbiter_decision or "").strip()
    arb_target = _action_label(str(arbiter_target_action or ""))
    if arb_status == "failed":
        return "观望", "二级仲裁失败，强制观望。"
    if arb_status == "completed":
        if arb_decision in {"skip", "manual_review"}:
            return "观望", "二级仲裁要求跳过或人工复核，强制观望。"
        if arb_decision == "downgrade":
            next_action = _downgrade_action(gated_action, arb_target)
            if next_action != gated_action:
                return next_action, f"二级仲裁将动作降级为 {next_action}。"

    return gated_action, ""


def _execution_status_from_row(row: Mapping[str, Any]) -> str:
    manual_status = str(_row_field(row, "manual_review_status", "") or "")
    if manual_status == "pending":
        return "manual_review_pending"
    if manual_status == "resolved":
        action_source = str(_row_field(row, "effective_action_source", "") or "")
        if action_source == "expert_llm":
            return "expert_review_resolved"
        if action_source == "expert_llm_failed":
            return "expert_review_failed"
        return "manual_review_resolved"
    if manual_status == "expired":
        return "manual_review_expired"
    if manual_status == "superseded":
        return "manual_review_superseded"
    if (
        str(_row_field(row, "arbiter_review_status", "") or "") == "completed"
        and str(_row_field(row, "arbiter_review_decision", "") or "") == "downgrade"
        and str(_row_field(row, "effective_action_source", "") or "") == "arbiter"
    ):
        return "arbiter_downgraded"
    return "executable"


def _execution_status_label(row: Mapping[str, Any]) -> str:
    return EXECUTION_STATUS_LABELS.get(_execution_status_from_row(row), "executable")


def _resolved_effective_action(row: Mapping[str, Any]) -> tuple[str, float]:
    effective_action = str(_row_field(row, "effective_recommendation", "") or "").strip()
    if effective_action:
        return _action_label(effective_action), safe_float(_row_field(row, "effective_stake_pct"))

    manual_status = str(_row_field(row, "manual_review_status", "") or "")
    if manual_status in {"pending", "expired", "superseded"}:
        return "观望", 0.0

    return (
        _action_label(str(_row_field(row, "recommendation", "") or "")),
        safe_float(_row_field(row, "suggested_stake_pct")),
    )


def _risk_context_from_run(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "recommended_outcome": str(_row_field(row, "recommended_outcome", "") or ""),
        "expected_values": _expected_values_from_run(row),
        "confidence": safe_float(_row_field(row, "confidence_score")),
        "quality_score": safe_float(_row_field(row, "quality_score")),
        "model_agreement": safe_float(_row_field(row, "model_agreement")),
        "probabilities": {
            "home": safe_float(_row_field(row, "final_home_prob")),
            "draw": safe_float(_row_field(row, "final_draw_prob")),
            "away": safe_float(_row_field(row, "final_away_prob")),
        },
        "market_odds": {
            "home": safe_float(_row_field(row, "market_odds_home")),
            "draw": safe_float(_row_field(row, "market_odds_draw")),
            "away": safe_float(_row_field(row, "market_odds_away")),
        },
        "stake_pct": safe_float(_row_field(row, "suggested_stake_pct")),
    }


def _expected_values_from_run(row: Mapping[str, Any]) -> dict[str, float]:
    return {
        "home": safe_float(_row_field(row, "ev_home")),
        "draw": safe_float(_row_field(row, "ev_draw")),
        "away": safe_float(_row_field(row, "ev_away")),
    }


def _promotion_constraints_pass(
    review_target_action: str,
    quality: Mapping[str, Any],
    algo_risk: Mapping[str, Any],
    threshold_config: Mapping[str, Any] | None = None,
) -> bool:
    promote_gate = _merge_threshold_config(threshold_config)["promote"]
    current_action = _action_label(str(algo_risk.get("recommendation", "")))
    outcome = str(algo_risk.get("recommended_outcome", "") or "")
    expected_values = algo_risk.get("expected_values", {})
    market_bias = algo_risk.get("market_bias", {})

    if current_action == "观望" and review_target_action == "主推":
        return False
    return (
        safe_float(quality.get("score")) >= safe_float(promote_gate["quality"])
        and safe_float(algo_risk.get("confidence")) >= safe_float(promote_gate["confidence"])
        and safe_float(expected_values.get(outcome)) >= safe_float(promote_gate["ev"])
        and safe_float(market_bias.get(outcome)) >= safe_float(promote_gate["market_bias"])
        and str(algo_risk.get("risk_level", "") or "") != "very_high"
    )


def run_quant_model(row: Mapping[str, Any], features: Mapping[str, Any]) -> dict[str, Any]:
    recent_home = features["recent_home"]
    recent_away = features["recent_away"]
    split = features["split"]
    lineup = features["lineup"]
    schedule = features["schedule"]

    home_split_matches = max(
        safe_int(split["home_wins"]) + safe_int(split["home_draws"]) + safe_int(split["home_losses"]),
        1,
    )
    away_split_matches = max(
        safe_int(split["away_wins"]) + safe_int(split["away_draws"]) + safe_int(split["away_losses"]),
        1,
    )
    home_split_gf_pg = safe_float(split["home_gf"]) / home_split_matches
    home_split_ga_pg = safe_float(split["home_ga"]) / home_split_matches
    away_split_gf_pg = safe_float(split["away_gf"]) / away_split_matches
    away_split_ga_pg = safe_float(split["away_ga"]) / away_split_matches

    market_value_rating_gap = safe_float(features.get("market_value_rating_gap"))
    rating_gap = safe_float(features["rating_gap"]) + market_value_rating_gap * 0.35
    form_gap_raw = safe_float(recent_home["points_per_game"]) - safe_float(recent_away["points_per_game"])
    split_gap = safe_float(split["home_ppg"]) - safe_float(split["away_ppg"])
    goal_diff_gap_raw = safe_float(recent_home["goal_diff_per_game"]) - safe_float(
        recent_away["goal_diff_per_game"]
    )
    # Opponent-strength-adjusted residuals carry the cross-league
    # comparable signal: a 17th-place team posting PPG 2.0 is over-
    # performing far more than a 5th-place team posting the same number,
    # but raw PPG treats them identically. Blend 60/40 residual/raw so
    # we still benefit from absolute information within a single league.
    form_residual_gap = safe_float(features.get("form_residual_gap"))
    goal_diff_residual_gap = safe_float(features.get("goal_diff_residual_gap"))
    form_gap = form_gap_raw * 0.40 + form_residual_gap * 0.60
    goal_diff_gap = goal_diff_gap_raw * 0.40 + goal_diff_residual_gap * 0.60
    # When lineup data is missing, both home and away availability fall
    # back to the neutral 0.92 default. Computing lineup_gap from those
    # placeholders yields 0.0 (which is fine), but we additionally need
    # to suppress the lineup-driven coefficients elsewhere �?otherwise
    # the model spends weight on a constant signal that adds noise.
    lineup_data_available = bool(safe_int(lineup.get("data_available", 0)))
    lineup_gap = (
        safe_float(lineup["home_availability"]) - safe_float(lineup["away_availability"])
        if lineup_data_available
        else 0.0
    )
    rest_term = clamp(safe_float(schedule["rest_advantage"]) / 7.0, -0.35, 0.35)
    load_term = clamp(safe_float(schedule["schedule_gap"]) * 0.12, -0.30, 0.30)
    h2h_term = clamp(safe_float(features["h2h_edge"]) * 0.16, -0.18, 0.18)
    motivation_term = clamp(safe_float(features["motivation_signal"]) * 0.18, 0.0, 0.06)
    # Lineup-channel weights: zero when missing, normal when present.
    lineup_weight_home = 0.22 if lineup_data_available else 0.0
    lineup_weight_away = 0.20 if lineup_data_available else 0.0

    home_attack_rating = (
        safe_float(recent_home["goals_for_per_game"]) * 0.62 + home_split_gf_pg * 0.38
    )
    away_attack_rating = (
        safe_float(recent_away["goals_for_per_game"]) * 0.62 + away_split_gf_pg * 0.38
    )
    home_defense_leak = (
        safe_float(recent_home["goals_against_per_game"]) * 0.64 + home_split_ga_pg * 0.36
    )
    away_defense_leak = (
        safe_float(recent_away["goals_against_per_game"]) * 0.64 + away_split_ga_pg * 0.36
    )

    # When understat xG is available for both teams, blend it heavily into
    # attack and defense ratings. xG converges faster than actual goals
    # (variance from finishing luck cancels), and the bookmaker price
    # already prices xG in �?using it lifts our model out of "below-market
    # noise" regime.
    xg = features.get("xg") or {}
    if isinstance(xg, dict) and safe_int(xg.get("coverage")):
        home_xg = safe_float(xg.get("home_xg_per_game"))
        away_xg = safe_float(xg.get("away_xg_per_game"))
        home_xga = safe_float(xg.get("home_xga_per_game"))
        away_xga = safe_float(xg.get("away_xga_per_game"))
        if home_xg > 0 and away_xg > 0 and home_xga > 0 and away_xga > 0:
            # 70% xG-derived, 30% goals-derived. The 30% retains "luck"
            # information (a hot scorer can over-deliver xG short term).
            home_attack_rating = home_xg * 0.70 + home_attack_rating * 0.30
            away_attack_rating = away_xg * 0.70 + away_attack_rating * 0.30
            home_defense_leak = home_xga * 0.70 + home_defense_leak * 0.30
            away_defense_leak = away_xga * 0.70 + away_defense_leak * 0.30

    home_lambda = clamp(
        1.34
        + (home_attack_rating - 1.45) * 0.34
        + (away_defense_leak - 1.10) * 0.24
        + form_gap * 0.16
        + split_gap * 0.14
        + goal_diff_gap * 0.10
        + rating_gap / 470.0
        + lineup_gap * lineup_weight_home
        + rest_term * 0.12
        + load_term * 0.10
        + h2h_term * 0.28
        + motivation_term,
        0.35,
        3.20,
    )
    away_lambda = clamp(
        1.10
        + (away_attack_rating - 1.30) * 0.34
        + (home_defense_leak - 1.05) * 0.24
        - form_gap * 0.14
        - split_gap * 0.12
        - goal_diff_gap * 0.08
        - rating_gap / 520.0
        - lineup_gap * lineup_weight_away
        - rest_term * 0.10
        - load_term * 0.08
        - h2h_term * 0.24,
        0.30,
        3.00,
    )

    rho = clamp(-0.06 + abs(h2h_term) * 0.04, -0.12, 0.02)
    home_prob = 0.0
    draw_prob = 0.0
    away_prob = 0.0
    top_scores: list[tuple[tuple[int, int], float]] = []
    for home_goals in range(7):
        for away_goals in range(7):
            prob = poisson_probability(home_lambda, home_goals) * poisson_probability(
                away_lambda, away_goals
            )
            prob *= _low_score_correlation_adjustment(
                home_goals,
                away_goals,
                home_lambda,
                away_lambda,
                rho,
            )
            prob = max(prob, 0.0)
            top_scores.append(((home_goals, away_goals), prob))
            if home_goals > away_goals:
                home_prob += prob
            elif home_goals == away_goals:
                draw_prob += prob
            else:
                away_prob += prob

    probabilities = normalize_probs(home_prob, draw_prob, away_prob)
    top_scores.sort(key=lambda item: item[1], reverse=True)
    total_goals = home_lambda + away_lambda
    over_25 = 1 - sum(poisson_probability(total_goals, goals) for goals in range(3))

    return {
        "probabilities": probabilities,
        "lambda_home": home_lambda,
        "lambda_away": away_lambda,
        "top_scores": top_scores[:4],
        "over_25": clamp(over_25, 0.0, 1.0),
        "under_25": clamp(1 - over_25, 0.0, 1.0),
        "rho": rho,
        "independent_inputs": {
            "rating_gap": rating_gap,
            "market_value_rating_gap": market_value_rating_gap,
            "form_gap": form_gap,
            "split_gap": split_gap,
            "goal_diff_gap": goal_diff_gap,
            "lineup_gap": lineup_gap,
            "rest_term": rest_term,
            "load_term": load_term,
            "h2h_term": h2h_term,
        },
    }


def run_ml_model(row: Mapping[str, Any], features: Mapping[str, Any]) -> dict[str, Any]:
    recent_home = features["recent_home"]
    recent_away = features["recent_away"]
    split = features["split"]
    lineup = features["lineup"]
    schedule = features["schedule"]
    lineup_data_available = bool(safe_int(lineup.get("data_available", 0)))

    market_value = features.get("market_value") or {}
    market_value_coverage = bool(
        isinstance(market_value, Mapping) and safe_int(market_value.get("coverage"))
    )
    market_value_rating_gap = safe_float(features.get("market_value_rating_gap"))
    adjusted_rating_gap = safe_float(features["rating_gap"]) + market_value_rating_gap * 0.35
    rating_score = clamp(0.5 + adjusted_rating_gap / 220.0, 0.05, 0.95)
    # Form / goal-diff scores now use a 60/40 blend of opponent-adjusted
    # residuals and raw gaps, mirroring the quant model. Residual is
    # divided by ~1.4 (typical residual magnitude in 6-game samples) to
    # span a comparable [-1, 1] range.
    form_residual_gap = safe_float(features.get("form_residual_gap"))
    goal_diff_residual_gap = safe_float(features.get("goal_diff_residual_gap"))
    form_raw_gap = safe_float(recent_home["points_per_game"]) - safe_float(recent_away["points_per_game"])
    goal_raw_gap = safe_float(recent_home["goal_diff_per_game"]) - safe_float(recent_away["goal_diff_per_game"])
    form_score = clamp(
        0.5
        + (form_raw_gap / 3.0) * 0.40
        + (form_residual_gap / 1.4) * 0.60 * 0.5,  # residual already centered, halve again to land in [-0.5, 0.5]
        0.05,
        0.95,
    )
    goal_score = clamp(
        0.5
        + (goal_raw_gap / 2.8) * 0.40
        + (goal_diff_residual_gap / 1.4) * 0.60 * 0.5,
        0.05,
        0.95,
    )
    split_score = clamp(
        0.5 + (safe_float(split["home_ppg"]) - safe_float(split["away_ppg"])) / 2.6,
        0.05,
        0.95,
    )
    lineup_score = clamp(
        0.5 + (safe_float(lineup["home_availability"]) - safe_float(lineup["away_availability"])) * 1.0,
        0.08,
        0.92,
    )
    rest_score = clamp(
        0.5
        + clamp(safe_float(schedule["rest_advantage"]) / 7.0, -0.35, 0.35) * 0.55
        + clamp(safe_float(schedule["schedule_gap"]) * 0.10, -0.25, 0.25),
        0.08,
        0.92,
    )
    h2h_score = clamp(0.5 + safe_float(features["h2h_edge"]) * 0.28, 0.10, 0.90)

    # xG-derived score (only counts when understat covered both teams).
    # We treat xG_for - xGA_against as the "expected goal differential",
    # which the heuristic ml_model can read independently of the goals
    # signal. Score is centered around 0.5 with sensitivity 1/2.0 because
    # |xg_diff_gap| > 2.0 is rare across two normal teams.
    xg_score = 0.5
    xg_coverage = 0
    xg = features.get("xg") or {}
    if isinstance(xg, dict) and safe_int(xg.get("coverage")):
        home_xg_diff = safe_float(xg.get("home_xg_per_game")) - safe_float(xg.get("home_xga_per_game"))
        away_xg_diff = safe_float(xg.get("away_xg_per_game")) - safe_float(xg.get("away_xga_per_game"))
        if (
            safe_float(xg.get("home_xg_per_game")) > 0
            and safe_float(xg.get("away_xg_per_game")) > 0
        ):
            xg_score = clamp(0.5 + (home_xg_diff - away_xg_diff) / 2.0, 0.05, 0.95)
            xg_coverage = 1

    # Default weights total 1.00. When lineup data is unavailable, the
    # 0.12 lineup weight is redistributed proportionally to other channels
    # so the score stays calibrated. When xG is available, we carve out
    # 0.18 of weight from rating/form/goal/split (which xG correlates
    # with) and give it to the xg_score channel �?xG is a stronger
    # forward-looking signal than raw goals.
    if lineup_data_available and xg_coverage:
        weights = {
            "rating": 0.20, "form": 0.16, "goal": 0.12, "split": 0.10,
            "lineup": 0.12, "rest": 0.06, "h2h": 0.04, "xg": 0.20,
        }
    elif lineup_data_available:
        weights = {
            "rating": 0.26, "form": 0.20, "goal": 0.18, "split": 0.14,
            "lineup": 0.12, "rest": 0.06, "h2h": 0.04, "xg": 0.0,
        }
    elif xg_coverage:
        # No lineup, but xG present: give xG 0.20, redistribute lineup 0.12.
        weights = {
            "rating": 0.24, "form": 0.18, "goal": 0.14, "split": 0.12,
            "lineup": 0.0, "rest": 0.07, "h2h": 0.05, "xg": 0.20,
        }
    else:
        # Lineup weight (0.12) redistributed to rating/form/goal/split.
        weights = {
            "rating": 0.30, "form": 0.22, "goal": 0.20, "split": 0.16,
            "lineup": 0.0, "rest": 0.07, "h2h": 0.05, "xg": 0.0,
        }
    if market_value_coverage:
        for key in ("form", "goal", "split"):
            weights[key] *= 0.94
        weights["rating"] += 0.04
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {key: value / total_weight for key, value in weights.items()}

    home_score = (
        rating_score * weights["rating"]
        + form_score * weights["form"]
        + goal_score * weights["goal"]
        + split_score * weights["split"]
        + lineup_score * weights["lineup"]
        + rest_score * weights["rest"]
        + h2h_score * weights["h2h"]
        + xg_score * weights["xg"]
    )
    away_score = (
        (1 - rating_score) * weights["rating"]
        + (1 - form_score) * weights["form"]
        + (1 - goal_score) * weights["goal"]
        + (1 - split_score) * weights["split"]
        + (1 - lineup_score) * weights["lineup"]
        + (1 - rest_score) * weights["rest"]
        + (1 - h2h_score) * weights["h2h"]
        + (1 - xg_score) * weights["xg"]
    )
    balance = 1 - abs(home_score - away_score)
    draw_score = clamp(
        0.18 + balance * 0.22 + (1 - abs(safe_float(features["h2h_edge"]))) * 0.06,
        0.12,
        0.34,
    )
    probabilities = normalize_probs(home_score, draw_score, away_score)

    return {
        "probabilities": probabilities,
        "feature_importance": [
            ("rating_gap", weights["rating"]),
            ("market_value_gap", 0.04 if market_value_coverage else 0.0),
            ("recent_form_gap", weights["form"]),
            ("goal_diff_gap", weights["goal"]),
            ("home_away_split_gap", weights["split"]),
            ("lineup_availability_gap", weights["lineup"]),
            ("rest_and_schedule", weights["rest"]),
            ("h2h_edge", weights["h2h"]),
        ],
        "model_name": "Independent-Heuristic-v2-xG" if xg_coverage else "Independent-Heuristic-v1",
        "lineup_data_available": lineup_data_available,
        "xg_coverage": xg_coverage,
        "subscores": {
            "rating_score": rating_score,
            "form_score": form_score,
            "goal_score": goal_score,
            "split_score": split_score,
            "lineup_score": lineup_score,
            "rest_score": rest_score,
            "h2h_score": h2h_score,
            "xg_score": xg_score,
            "market_value_rating_gap": market_value_rating_gap,
        },
    }


def run_legacy_market_model(row: Mapping[str, Any], features: Mapping[str, Any]) -> dict[str, Any]:
    market = dict(features["market_probs"])
    heat = _heat_probs_from_text(str(row.get("betting_heat_summary", "") or ""), market)
    recent_home = features["recent_home"]
    recent_away = features["recent_away"]
    lineup = features["lineup"]

    rating_score = clamp(
        0.5
        + (safe_float(features["rating_gap"]) + safe_float(features.get("market_value_rating_gap")) * 0.25) / 240.0,
        0.05,
        0.95,
    )
    form_score = clamp(
        0.5 + (safe_float(recent_home["win_rate"]) - safe_float(recent_away["win_rate"])) * 0.7,
        0.08,
        0.92,
    )
    lineup_score = clamp(
        0.5 + (safe_float(lineup["home_availability"]) - safe_float(lineup["away_availability"])) * 0.8,
        0.10,
        0.90,
    )

    home_score = market["home"] * 0.58 + heat["home"] * 0.10 + rating_score * 0.20 + form_score * 0.08 + lineup_score * 0.04
    away_score = market["away"] * 0.58 + heat["away"] * 0.10 + (1 - rating_score) * 0.20 + (1 - form_score) * 0.08 + (1 - lineup_score) * 0.04
    draw_score = market["draw"] * 0.64 + heat["draw"] * 0.10 + (1 - abs(home_score - away_score)) * 0.16
    probabilities = normalize_probs(home_score, draw_score, away_score)

    return {
        "probabilities": probabilities,
        "model_name": "Legacy-Market-Assisted-benchmark",
    }


def run_data_quality(row: Mapping[str, Any], features: Mapping[str, Any]) -> dict[str, Any]:
    recent_home = features["recent_home"]
    recent_away = features["recent_away"]
    split = features["split"]
    lineup_raw = features.get("lineup_raw") or {}
    lineup = features.get("lineup") or {}
    lineup_text = str(row.get("injury_or_lineup_notes", "") or "")
    lineup_structured = bool(
        lineup_raw.get("home_lineup")
        or lineup_raw.get("away_lineup")
        or lineup_raw.get("home_missing")
        or lineup_raw.get("away_missing")
    )
    # Single source of truth for "do we actually have lineup signal":
    # whatever build_lineup_metrics decided. Falls back to the older
    # placeholder check so existing rows still parse correctly.
    lineup_available = bool(safe_int(lineup.get("data_available", 0))) or lineup_structured
    lineup_placeholder = (
        "lineup fallback not found" in lineup_text
        and "injury fallback not found" in lineup_text
    )

    parsing_checks = {
        "recent_form_home": safe_int(recent_home["matches"]) >= 6,
        "recent_form_away": safe_int(recent_away["matches"]) >= 6,
        "home_away_form": (
            safe_int(split["home_wins"]) + safe_int(split["home_draws"]) + safe_int(split["home_losses"]) >= 6
            and safe_int(split["away_wins"]) + safe_int(split["away_draws"]) + safe_int(split["away_losses"]) >= 6
        ),
        "ratings": safe_float(features["home_rating"]) > 0 and safe_float(features["away_rating"]) > 0,
        "market_value": bool(
            isinstance(features.get("market_value"), Mapping)
            and safe_int(features.get("market_value", {}).get("coverage"))
        ),
        "lineup_structured": lineup_available,
        "h2h": safe_int(features["h2h"]["matches"]) > 0,
        "schedule": (
            safe_float(features["schedule"]["home_rest_days"]) > 0
            or safe_float(features["schedule"]["away_rest_days"]) > 0
            or safe_int(features["schedule"]["home_load_14"]) > 0
            or safe_int(features["schedule"]["away_load_14"]) > 0
        ),
        "market": all(safe_float(features["market_odds"][key]) > 0 for key in OUTCOMES),
    }

    problems: list[str] = []
    if not parsing_checks["recent_form_home"] or not parsing_checks["recent_form_away"]:
        problems.append("recent form was not parsed into usable values")
    if not parsing_checks["home_away_form"]:
        problems.append("home/away split sample was not parsed")
    if not parsing_checks["ratings"]:
        problems.append("rating proxy is missing")
    if not parsing_checks["lineup_structured"]:
        if lineup_placeholder:
            problems.append("lineup/injury text is still a placeholder")
        else:
            problems.append("lineup/injury data has text only")
    if not parsing_checks["market"]:
        problems.append("market odds missing")
    if not parsing_checks["schedule"]:
        problems.append("schedule load not found")

    weights = {
        "recent_form_home": 0.16,
        "recent_form_away": 0.16,
        "home_away_form": 0.16,
        "ratings": 0.14,
        "market_value": 0.02,
        "lineup_structured": 0.18,
        "h2h": 0.08,
        "schedule": 0.04,
        "market": 0.06,
    }
    score = 0.0
    for key, weight in weights.items():
        score += (1.0 if parsing_checks[key] else 0.0) * weight
    score = clamp(score, 0.0, 1.0)

    return {
        "score": score,
        "parsing_checks": parsing_checks,
        "problems": problems,
        "weight_adjustment": clamp(0.72 + score * 0.28, 0.72, 1.0),
        "lineup_structured": lineup_structured,
    }


def _market_prior_alpha(
    quality_score: float,
    model_agreement: float,
) -> float:
    """How much weight to give the model vs the market prior.

    Shrinkage anchor for the blended probability. The model blend
    (quant + ml) on its own can be substantially worse-calibrated than
    the market �?measured by Brier/LogLoss against settled feedback �?
    because the market price aggregates more information than any of our
    text-derived features. Treating the market as a Bayesian prior
    bounds the worst case: at low quality / low agreement the system
    falls back toward "follow the market", and only when quality and
    agreement are high does the independent model dominate.

    Returns alpha in [0.10, 0.55]:
        final_prob = alpha * model_blend + (1 - alpha) * market_prob

    Tuning history:
    - v1 (base 0.50, range [0.20, 0.80]): improved Brier ~7% but hit
      rate did not change because the model still dominated argmax in
      ~56% of the blend. Independent model's directional accuracy is
      below market, so when its argmax differs from market it costs us.
    - v2 (current, base 0.35, range [0.10, 0.55]): keep model as a
      correction to market rather than a competing forecast. Average
      alpha around 0.30 means ~70% market in normal conditions, the
      model gets a meaningful but minority voice. This anchors hit
      rate to market while still letting strong model signal show up
      in EV / market_bias.
    """

    base_alpha = 0.35
    # quality is bounded ~ [0, 1]; sweet spot is 0.55-0.85.
    quality_factor = clamp((safe_float(quality_score) - 0.55) / 0.30, 0.3, 1.1)
    agreement_factor = clamp((safe_float(model_agreement) - 0.55) / 0.30, 0.4, 1.0)
    return clamp(base_alpha * quality_factor * agreement_factor, 0.10, 0.55)


def blend_predictions(
    quant: Mapping[str, Any],
    ml: Mapping[str, Any],
    quality: Mapping[str, Any],
    features: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine quant / heuristic ML probabilities, then shrink to the market.

    The original design produced ``final = quant·0.56 + ml·0.44`` with no
    market anchor at all. Empirically (settled-sample backtest) the
    resulting independent probability was *worse* than the market-implied
    probability on Brier/LogLoss/Hit, which means betting the EV gap
    against the market is on average a losing trade. Shrinking toward
    the market prior bounds the downside without giving up upside when
    the model is genuinely confident.
    """

    quant_weight = 0.56 * safe_float(quality["weight_adjustment"])
    ml_weight = 0.44 * safe_float(quality["weight_adjustment"])
    model_blend = normalize_probs(
        safe_float(quant["probabilities"]["home"]) * quant_weight
        + safe_float(ml["probabilities"]["home"]) * ml_weight,
        safe_float(quant["probabilities"]["draw"]) * quant_weight
        + safe_float(ml["probabilities"]["draw"]) * ml_weight,
        safe_float(quant["probabilities"]["away"]) * quant_weight
        + safe_float(ml["probabilities"]["away"]) * ml_weight,
    )
    agreement = clamp(
        1
        - (
            abs(safe_float(quant["probabilities"]["home"]) - safe_float(ml["probabilities"]["home"]))
            + abs(safe_float(quant["probabilities"]["draw"]) - safe_float(ml["probabilities"]["draw"]))
            + abs(safe_float(quant["probabilities"]["away"]) - safe_float(ml["probabilities"]["away"]))
        )
        / 2.0,
        0.0,
        1.0,
    )

    market_probs: dict[str, float] | None = None
    if features is not None:
        candidate = features.get("market_probs") if isinstance(features, Mapping) else None
        if isinstance(candidate, Mapping):
            market_probs = {
                "home": safe_float(candidate.get("home")),
                "draw": safe_float(candidate.get("draw")),
                "away": safe_float(candidate.get("away")),
            }
            if sum(market_probs.values()) <= 0:
                market_probs = None

    if market_probs is None:
        # No market information available; fall back to pure model blend
        # rather than refusing to predict.
        probabilities = model_blend
        alpha = 1.0
    else:
        alpha = _market_prior_alpha(safe_float(quality["score"]), agreement)
        probabilities = normalize_probs(
            alpha * model_blend["home"] + (1.0 - alpha) * market_probs["home"],
            alpha * model_blend["draw"] + (1.0 - alpha) * market_probs["draw"],
            alpha * model_blend["away"] + (1.0 - alpha) * market_probs["away"],
        )

    sorted_probs = sorted(probabilities.values(), reverse=True)
    margin = sorted_probs[0] - sorted_probs[1]
    return {
        "probabilities": probabilities,
        "model_blend_probabilities": model_blend,
        "market_prior_alpha": alpha,
        "agreement": agreement,
        "margin": margin,
    }


def run_risk_assessor(
    row: Mapping[str, Any],
    features: Mapping[str, Any],
    legacy: Mapping[str, Any],
    blended: Mapping[str, Any],
    quality: Mapping[str, Any],
    threshold_config: Mapping[str, Any] | None = None,
    calibrated_probabilities: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    gates = _merge_threshold_config(threshold_config)
    # Use calibrated probabilities when available and valid;
    # otherwise fall back to market-shrunken blended probabilities.
    raw_probs = dict(blended["probabilities"])
    if isinstance(calibrated_probabilities, Mapping) and sum(safe_float(calibrated_probabilities.get(k)) for k in OUTCOMES) > 0.01:
        probabilities = {k: safe_float(calibrated_probabilities.get(k)) for k in OUTCOMES}
    else:
        probabilities = raw_probs
    market_odds = dict(features["market_odds"])
    market_probs = dict(features["market_probs"])
    model_margin = safe_float(blended["margin"])
    outcome_policy = evaluate_outcome_policy(
        probabilities=probabilities,
        market_odds=market_odds,
        market_probs=market_probs,
        legacy_probabilities=legacy.get("probabilities", {}),
        quality_score=safe_float(quality["score"]),
        model_agreement=safe_float(blended["agreement"]),
    )
    policy = evaluate_action_policy(
        probabilities=probabilities,
        market_odds=market_odds,
        market_probs=market_probs,
        legacy_probabilities=legacy.get("probabilities", {}),
        quality_score=safe_float(quality["score"]),
        model_agreement=safe_float(blended["agreement"]),
        model_margin=model_margin,
        threshold_config=gates,
        selected_outcome=str(outcome_policy.get("recommended_outcome", "") or ""),
    )
    target_strategy = _target_strategy_from_thresholds(threshold_config)
    target_strategy_metrics: dict[str, Any] = {}
    if target_strategy:
        recent_home = features.get("recent_home") if isinstance(features.get("recent_home"), Mapping) else {}
        recent_away = features.get("recent_away") if isinstance(features.get("recent_away"), Mapping) else {}
        split = features.get("split") if isinstance(features.get("split"), Mapping) else {}
        lineup = features.get("lineup") if isinstance(features.get("lineup"), Mapping) else {}
        schedule = features.get("schedule") if isinstance(features.get("schedule"), Mapping) else {}
        target_strategy_row = {
            "actual_result": "",
            "recommended_outcome": outcome_policy.get("recommended_outcome", ""),
            "quality_score": safe_float(quality["score"]),
            "model_agreement": safe_float(blended["agreement"]),
            "confidence_score": safe_float(policy["confidence"]),
            "final_home_prob": probabilities["home"],
            "final_draw_prob": probabilities["draw"],
            "final_away_prob": probabilities["away"],
            "market_home_prob": market_probs["home"],
            "market_draw_prob": market_probs["draw"],
            "market_away_prob": market_probs["away"],
            "market_odds_home": market_odds["home"],
            "market_odds_draw": market_odds["draw"],
            "market_odds_away": market_odds["away"],
            "ev_home": policy["expected_values"]["home"],
            "ev_draw": policy["expected_values"]["draw"],
            "ev_away": policy["expected_values"]["away"],
            "llm_review_decision": "",
            "llm_review_status": "",
            "arbiter_review_decision": "",
            "algo_recommendation": policy["recommendation"],
            "recommendation": policy["recommendation"],
            "home_rating": safe_float(features.get("home_rating")),
            "away_rating": safe_float(features.get("away_rating")),
            "recent_home_ppg": safe_float(recent_home.get("points_per_game")),
            "recent_away_ppg": safe_float(recent_away.get("points_per_game")),
            "recent_home_gf_pg": safe_float(recent_home.get("goals_for_per_game")),
            "recent_away_gf_pg": safe_float(recent_away.get("goals_for_per_game")),
            "recent_home_ga_pg": safe_float(recent_home.get("goals_against_per_game")),
            "recent_away_ga_pg": safe_float(recent_away.get("goals_against_per_game")),
            "home_split_ppg": safe_float(split.get("home_ppg")),
            "away_split_ppg": safe_float(split.get("away_ppg")),
            "home_absence_impact": safe_float(lineup.get("home_absence_impact")),
            "away_absence_impact": safe_float(lineup.get("away_absence_impact")),
            "lineup_home_availability": safe_float(lineup.get("home_availability")),
            "lineup_away_availability": safe_float(lineup.get("away_availability")),
            "rest_days_home": safe_float(schedule.get("home_rest_days")),
            "rest_days_away": safe_float(schedule.get("away_rest_days")),
            "schedule_load_home": safe_float(schedule.get("home_load_14")),
            "schedule_load_away": safe_float(schedule.get("away_load_14")),
            "h2h_edge": safe_float(features.get("h2h_edge")),
        }
        target_strategy_metrics = evaluate_target_strategy_rule([target_strategy_row], target_strategy)
        if safe_int(target_strategy_metrics.get("action_count")) > 0:
            bucket_key = next(iter(target_strategy_metrics.get("buckets", {"": {}})))
            selected_outcome = str(bucket_key).split(":")[-1]
            if selected_outcome in OUTCOMES:
                policy["recommended_outcome"] = selected_outcome
            policy["recommendation"] = str(target_strategy.get("action", "轻仓") or "轻仓")
            policy["stake_pct"] = safe_float(target_strategy.get("stake_pct", 1.0))
        else:
            policy["recommendation"] = "观望"
            policy["stake_pct"] = 0.0

    warnings = list(quality["problems"])
    warnings.extend(outcome_policy["warnings"])
    warnings.extend(policy["warnings"])
    if model_margin < 0.06:
        warnings.append("three-way probabilities are close")
    warnings = list(dict.fromkeys(item for item in warnings if str(item or "").strip()))

    return {
        "fair_odds": policy["fair_odds"],
        "expected_values": policy["expected_values"],
        "market_bias": policy["market_bias"],
        "market_probs": policy["market_probs"],
        "market_odds": policy["market_odds"],
        "probabilities": policy["probabilities"],
        "confidence": policy["confidence"],
        "risk_level": policy["risk_level"],
        "recommended_outcome": policy["recommended_outcome"],
        "predicted_outcome": outcome_policy["predicted_outcome"],
        "value_outcome": outcome_policy["value_outcome"],
        "outcome_source": outcome_policy["outcome_source"],
        "outcome_reason": outcome_policy["outcome_reason"],
        "outcome_score": outcome_policy["outcome_score"],
        "recommendation": policy["recommendation"],
        "stake_pct": policy["stake_pct"],
        "action_score": policy["action_score"],
        "action_score_factors": policy["action_score_factors"],
        "probability_margin": policy["probability_margin"],
        "ev_margin": policy["ev_margin"],
        "legacy_gap": policy["legacy_gap"],
        "kelly_fraction": policy["kelly_fraction"],
        "low_odds_favorite_guard": bool(policy.get("low_odds_favorite_guard")),
        "target_strategy": target_strategy_metrics,
        "warnings": warnings,
    }


def _arbiter_trigger_reasons(
    quality: Mapping[str, Any],
    legacy: Mapping[str, Any],
    blended: Mapping[str, Any],
    final_risk: Mapping[str, Any],
) -> list[str]:
    """Decide whether the high-risk arbiter should run.

    Hunter-mindset rewrite: the previous logic flagged "high EV + large
    market deviation" as a reason to escalate, which sent the *best* value
    signals to the most skeptical reviewer. Now we only escalate divergent
    signals when data quality is suspect; high-quality divergence is the
    alpha we want to act on, not gate-keep.
    """

    reasons: list[str] = []
    quality_score = safe_float(quality.get("score"))
    high_quality = quality_score >= ARBITER_QUALITY_TRUST_FLOOR
    outcome = str(final_risk.get("recommended_outcome", "") or "")
    if outcome in OUTCOMES:
        legacy_gap = abs(
            safe_float(legacy["probabilities"][outcome])
            - safe_float(blended["probabilities"][outcome])
        )
        # Only treat legacy/independent disagreement as a risk when the
        # underlying data is shaky �?otherwise the disagreement IS the bet.
        if legacy_gap >= ARBITER_MODEL_GAP_THRESHOLD and not high_quality:
            reasons.append(
                f"legacy/independent model gap {legacy_gap:.3f} on {_choice_label(outcome)} with weak data quality {quality_score:.2f}"
            )

        expected_values = final_risk.get("expected_values", {})
        market_bias = final_risk.get("market_bias", {})
        outcome_ev = safe_float(expected_values.get(outcome))
        outcome_bias = safe_float(market_bias.get(outcome))
        # High EV + large market deviation is escalated only when data
        # quality is poor (suspect parsing) or when the bias is *negative*
        # at high magnitude (the market is more bullish than we are �?
        # fading the market on negative bias deserves extra scrutiny).
        if outcome_ev >= ARBITER_HIGH_EV_THRESHOLD and abs(outcome_bias) >= ARBITER_HIGH_MARKET_BIAS_THRESHOLD:
            if not high_quality:
                reasons.append(
                    f"high EV and market bias on {_choice_label(outcome)} with weak data quality {quality_score:.2f}"
                )
            elif outcome_bias < 0:
                reasons.append(
                    f"high EV but negative market bias {outcome_bias:+.3f} on {_choice_label(outcome)}"
                )

    if quality_score < ARBITER_LOW_QUALITY_THRESHOLD:
        reasons.append(
            f"data quality too low ({quality_score:.3f} < {ARBITER_LOW_QUALITY_THRESHOLD:.2f})"
        )
    return reasons


def _build_arbiter_prompt(
    row: Mapping[str, Any],
    features: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    blended: Mapping[str, Any],
    quality: Mapping[str, Any],
    final_risk: Mapping[str, Any],
    review: Mapping[str, Any],
    trigger_reasons: list[str],
) -> str:
    outcome = str(final_risk.get("recommended_outcome", "") or "")
    expected_values = final_risk.get("expected_values", {})
    market_bias = final_risk.get("market_bias", {})
    lineup = features["lineup"]
    schedule = features["schedule"]
    review_status = str(review.get("status", "") or "")
    review_decision = str(review.get("decision", "") or "")
    review_summary = review_decision if review_status == "completed" else review_status
    return (
        "You are the second-stage risk arbiter for a football betting system. "
        "You may only decide the final action level; do not change the predicted outcome.\n"
        "Return strict JSON only. No markdown, no prose outside JSON.\n"
        "Schema:\n"
        '{"decision":"allow|allow_with_uplift|downgrade|manual_review|skip","target_action":"main|light|watch","reason":"short reason","risk_flags":["flag1","flag2"]}\n\n'
        "Rules:\n"
        "1. Never change the fixed outcome direction.\n"
        "2. Do not create a stronger action than the algorithm proposed. allow_with_uplift may only restore an over-downgraded LLM review to the algorithm action.\n"
        "3. If evidence is weak, disagreement is large, or data quality is unstable, prefer manual_review.\n"
        "4. downgrade can reduce the action to watch.\n"
        "5. Output one JSON line only. Keep reason short and risk_flags to at most 3 items.\n\n"
        f"Match: {row['home_team']} vs {row['away_team']}\n"
        f"League/time: {row['league']} | {row['match_time']}\n"
        f"Fixed outcome: {_choice_label(outcome)}\n"
        f"Algorithm action: {final_risk.get('algo_recommendation') or final_risk['recommendation']}\n"
        f"Post-review machine action: {final_risk['recommendation']}\n"
        f"First review status: {review_summary}\n"
        f"First review reason: {review.get('reason') or '-'}\n"
        f"Trigger reasons: {'; '.join(trigger_reasons)}\n"
        f"Final probabilities: {json.dumps(blended['probabilities'], ensure_ascii=False)}\n"
        f"Market probabilities: {json.dumps(final_risk['market_probs'], ensure_ascii=False)}\n"
        f"Market bias: {json.dumps(market_bias, ensure_ascii=False)}\n"
        f"EV: {json.dumps(expected_values, ensure_ascii=False)}\n"
        f"Action score: {safe_float(final_risk.get('action_score')):.3f} | "
        f"EV margin: {safe_float(final_risk.get('ev_margin')):.3f} | "
        f"Legacy gap: {safe_float(final_risk.get('legacy_gap')):.3f}\n"
        f"Confidence: {safe_float(final_risk['confidence']):.3f} | Risk level: {final_risk['risk_level']}\n"
        f"Data quality: {safe_float(quality['score']):.3f}\n"
        f"Home/away rating: {safe_float(snapshot.get('home_rating')):.1f} / {safe_float(snapshot.get('away_rating')):.1f}\n"
        f"Lineup availability: home {safe_float(lineup['home_availability']):.3f} / away {safe_float(lineup['away_availability']):.3f}\n"
        f"Key absences: home {safe_int(lineup['home_absent_count'])}+doubtful {safe_int(lineup['home_doubtful_count'])} | "
        f"away {safe_int(lineup['away_absent_count'])}+doubtful {safe_int(lineup['away_doubtful_count'])}\n"
        f"Rest/load: home rest {safe_float(schedule['home_rest_days']):.1f}d / away rest {safe_float(schedule['away_rest_days']):.1f}d | "
        f"14d load {safe_int(schedule['home_load_14'])}:{safe_int(schedule['away_load_14'])}\n"
        f"Warnings: {'; '.join(final_risk['warnings']) or 'none'}\n"
    )


def _build_review_prompt(
    row: Mapping[str, Any],
    features: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    blended: Mapping[str, Any],
    quality: Mapping[str, Any],
    algo_risk: Mapping[str, Any],
) -> str:
    """Plan-B unified review prompt.

    The model is asked to make a single yes/no call ("approve / reject")
    plus a coarse evidence grade. The system then deterministically maps
    that pair into action / stake / confidence adjustments via
    _REVIEW_RULE_TABLE �?so the LLM never emits raw numerical knobs that
    drift between calls.
    """

    outcome = str(algo_risk["recommended_outcome"])
    expected_values = algo_risk["expected_values"]
    market_bias = algo_risk["market_bias"]
    lineup = features["lineup"]
    schedule = features["schedule"]

    return (
        "You are a reviewer for a football betting model.\n"
        "Judge whether the system's outcome + action pair is worth executing.\n"
        "You may approve or reject only; do not propose a different outcome. Rejections become watch.\n\n"
        "Return exactly one JSON object with these fields only:\n"
        '{"decision":"approve|reject","evidence_grade":"strong|adequate|weak|unsafe","reason":"short reason"}\n\n'
        "Decision guide:\n"
        "- approve = the current outcome and action strength are acceptable.\n"
        "- reject = either the outcome or action strength is not acceptable; the system will watch.\n"
        "- evidence_grade strong/adequate/weak marks confidence. Use unsafe only for severe data conflict or obvious mismatch.\n"
        "- Keep reason short.\n\n"
        f"Match: {row['home_team']} vs {row['away_team']}\n"
        f"League/time: {row['league']} | {row['match_time']}\n"
        f"System outcome: {_choice_label(outcome)}\n"
        f"System action: {algo_risk['recommendation']} (stake {safe_float(algo_risk['stake_pct']):.2f}%)\n"
        f"Confidence/risk: {safe_float(algo_risk['confidence']):.2f} / {algo_risk['risk_level']}\n"
        f"Data quality: {safe_float(quality['score']):.2f}\n"
        f"Probability outcome: {_choice_label(str(algo_risk.get('predicted_outcome', '') or ''))}\n"
        f"Value outcome: {_choice_label(str(algo_risk.get('value_outcome', '') or ''))}\n"
        f"Outcome source/score: {algo_risk.get('outcome_source') or '-'} / {safe_float(algo_risk.get('outcome_score')):.2f}\n"
        f"Outcome rule reason: {algo_risk.get('outcome_reason') or '-'}\n"
        f"Home/draw/away probabilities: {json.dumps(blended['probabilities'], ensure_ascii=False)}\n"
        f"Market probabilities: {json.dumps(algo_risk['market_probs'], ensure_ascii=False)}\n"
        f"Market bias(model-market): {json.dumps(market_bias, ensure_ascii=False)}\n"
        f"EV: {json.dumps(expected_values, ensure_ascii=False)}\n"
        f"EV margin / probability margin / legacy gap: {safe_float(algo_risk.get('ev_margin')):.3f} / "
        f"{safe_float(algo_risk.get('probability_margin')):.3f} / {safe_float(algo_risk.get('legacy_gap')):.3f}\n"
        f"Home/away rating: {safe_float(snapshot.get('home_rating')):.1f} / {safe_float(snapshot.get('away_rating')):.1f}\n"
        f"Lineup availability: home {safe_float(lineup['home_availability']):.2f} / away {safe_float(lineup['away_availability']):.2f}\n"
        f"Key absences: home {safe_int(lineup['home_absent_count'])}+doubtful {safe_int(lineup['home_doubtful_count'])} | "
        f"away {safe_int(lineup['away_absent_count'])}+doubtful {safe_int(lineup['away_doubtful_count'])}\n"
        f"Rest/load: home rest {safe_float(schedule['home_rest_days']):.1f}d / away rest {safe_float(schedule['away_rest_days']):.1f}d | "
        f"14d load {safe_int(schedule['home_load_14'])}:{safe_int(schedule['away_load_14'])}\n"
        f"Warnings: {'; '.join(algo_risk['warnings']) or 'none'}\n\n"
        "Output one JSON object only.\n"
    )


def _score_candidate_payload(quant: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for score_item in quant.get("top_scores", [])[:4]:
        try:
            score, probability = score_item
            candidates.append(
                {
                    "score": f"{int(score[0])}-{int(score[1])}",
                    "probability": round(safe_float(probability), 4),
                }
            )
        except Exception:  # noqa: BLE001
            continue
    return candidates


def _build_score_prediction_prompt(
    row: Mapping[str, Any],
    features: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    quant: Mapping[str, Any],
    ml: Mapping[str, Any],
    legacy: Mapping[str, Any],
    blended: Mapping[str, Any],
    quality: Mapping[str, Any],
    final_risk: Mapping[str, Any],
) -> str:
    lineup = features["lineup"]
    schedule = features["schedule"]
    recent_home = features["recent_home"]
    recent_away = features["recent_away"]
    split = features["split"]
    xg = features.get("xg", {}) if isinstance(features.get("xg"), Mapping) else {}
    market_value = features.get("market_value", {}) if isinstance(features.get("market_value"), Mapping) else {}
    quant_candidates = _score_candidate_payload(quant)
    xg_summary = "-"
    if isinstance(xg, Mapping) and safe_int(xg.get("coverage")):
        xg_summary = (
            f"home_xg={safe_float(xg.get('home_xg_per_game')):.2f}, "
            f"away_xg={safe_float(xg.get('away_xg_per_game')):.2f}, "
            f"home_xga={safe_float(xg.get('home_xga_per_game')):.2f}, "
            f"away_xga={safe_float(xg.get('away_xga_per_game')):.2f}"
        )
    value_summary = "-"
    if isinstance(market_value, Mapping) and safe_int(market_value.get("coverage")):
        value_summary = (
            f"home_value={safe_float(market_value.get('home_value_eur_m')):.1f}m, "
            f"away_value={safe_float(market_value.get('away_value_eur_m')):.1f}m"
        )
    home_strength = str(row.get("elo_home", "") or f"{safe_float(snapshot.get('home_rating')):.1f}")
    away_strength = str(row.get("elo_away", "") or f"{safe_float(snapshot.get('away_rating')):.1f}")
    return (
        "你是一名专业足球比分预测模型。请根据比赛数据，综合基础实力、近期动态和市场数据，预测本场比赛最可能出现的全场比分。\n"
        "关键约束：只输出 JSON；不要输出分析过程；不要输出 markdown；不要暴露思维链；比分顺序必须是主队-客队；避免无依据的大比分。\n"
        "必须给出最可能比分、次选比分一、次选比分二、冷门比分、信心。每个比分都要附带概率，概率用 0-1 小数。\n"
        "最可能比分必须符合固定赛果方向；次选比分尽量贴近量化候选；冷门比分必须概率较低但仍符合足球常见分布。\n"
        "OUTPUT ONLY this JSON shape:\n"
        '{"most_likely":{"score":"1-1","probability":0.13},"second_1":{"score":"1-0","probability":0.11},"second_2":{"score":"0-0","probability":0.10},"upset":{"score":"0-1","probability":0.06},"confidence_label":"低|中|高"}\n'
        f"固定赛果方向：{_choice_label(str(final_risk.get('recommended_outcome', '') or ''))}；动作：{final_risk.get('recommendation', '')}；风险：{final_risk.get('risk_level', '')}\n"
        f"比赛：{row['home_team']} vs {row['away_team']}；赛事：{row['league']}；时间：{row['match_time']}；期号：{row.get('issue', '')}\n"
        "维度一：基础实力\n"
        f"主队 Elo/实力代理：{home_strength}\n"
        f"客队 Elo/实力代理：{away_strength}\n"
        f"球队球员身价：{row.get('market_value_summary', '') or value_summary}\n"
        f"主队近期状态：{row.get('recent_form_home', '')}\n"
        f"客队近期状态：{row.get('recent_form_away', '')}\n"
        f"主客场表现：{row.get('home_away_form', '')}\n"
        "维度二：近期动态\n"
        f"交锋记录：{row.get('head_to_head_summary', '')}\n"
        f"伤停/阵容：{row.get('injury_or_lineup_notes', '')}\n"
        f"战意/赛程：{row.get('motivation_or_schedule_notes', '')}\n"
        "维度三：市场数据\n"
        f"欧赔变化：{row.get('european_odds_movement_summary', '')}\n"
        f"投注热度：{row.get('betting_heat_summary', '')}\n"
        "模型辅助数据\n"
        f"最终赛果概率：{json.dumps(blended['probabilities'], ensure_ascii=False)}；量化赛果概率：{json.dumps(quant['probabilities'], ensure_ascii=False)}\n"
        f"预期进球：{safe_float(quant.get('lambda_home')):.2f}-{safe_float(quant.get('lambda_away')):.2f}；量化候选比分：{json.dumps(quant_candidates, ensure_ascii=False)}\n"
        f"市场赔率：{json.dumps(features['market_odds'], ensure_ascii=False)}；市场隐含概率：{json.dumps(final_risk.get('market_probs', {}), ensure_ascii=False)}\n"
        f"数据质量：{safe_float(quality.get('score')):.2f}；实力差：{safe_float(features.get('rating_gap')):+.1f}；xG：{xg_summary}\n"
        f"近期数值：主 PPG/GF/GA={safe_float(recent_home.get('points_per_game')):.2f}/{safe_float(recent_home.get('goals_for_per_game')):.2f}/{safe_float(recent_home.get('goals_against_per_game')):.2f}；"
        f"客 PPG/GF/GA={safe_float(recent_away.get('points_per_game')):.2f}/{safe_float(recent_away.get('goals_for_per_game')):.2f}/{safe_float(recent_away.get('goals_against_per_game')):.2f}\n"
        f"主客场 PPG：{safe_float(split.get('home_ppg')):.2f}-{safe_float(split.get('away_ppg')):.2f}；交锋边际：{safe_float(features.get('h2h_edge')):+.2f}\n"
        f"阵容完整度：{safe_float(lineup.get('home_availability')):.2f}-{safe_float(lineup.get('away_availability')):.2f}；缺阵：{safe_int(lineup.get('home_absent_count'))}+{safe_int(lineup.get('home_doubtful_count'))}/{safe_int(lineup.get('away_absent_count'))}+{safe_int(lineup.get('away_doubtful_count'))}\n"
        f"休息/负荷：{safe_float(schedule.get('home_rest_days')):.1f}d-{safe_float(schedule.get('away_rest_days')):.1f}d；{safe_int(schedule.get('home_load_14'))}-{safe_int(schedule.get('away_load_14'))}\n"
    )


def _repair_truncated_score_prediction(
    base_url: str,
    api_key: str,
    model: str,
    *,
    response: Mapping[str, Any],
    fallback_score: str,
    candidates: list[dict[str, Any]],
    final_risk: Mapping[str, Any],
) -> dict[str, Any] | None:
    prompt = (
        "Return one compact JSON object only. No reasoning, no markdown.\n"
        "Choose the most plausible exact full-time score using only the fixed direction and candidates.\n"
        "Include most_likely, second_1, second_2, upset, and confidence_label. Probability must be 0-1.\n"
        'Schema: {"most_likely":{"score":"1-1","probability":0.12},"second_1":{"score":"1-0","probability":0.10},"second_2":{"score":"0-0","probability":0.09},"upset":{"score":"0-1","probability":0.05},"confidence_label":"低|中|高"}\n'
        f"Fixed match-result direction: {_choice_label(str(final_risk.get('recommended_outcome', '') or ''))}\n"
        f"Quant candidates: {json.dumps(candidates, ensure_ascii=False)}\n"
        f"Fallback score if uncertain: {fallback_score or '-'}\n"
    )
    try:
        repair_response = _request_score_prediction_response(
            base_url,
            api_key,
            model,
            messages=_build_openai_messages(
                "Output final JSON only. Never output reasoning.",
                prompt,
            ),
            max_tokens=900,
        )
        raw_content = str(repair_response.get("content", "") or "")
        parsed = _parse_score_prediction_payload(raw_content)
    except Exception:  # noqa: BLE001
        return None
    return {
        "enabled": 1,
        "status": "completed",
        "score": parsed["score"],
        "confidence": parsed["confidence"],
        "reason": parsed["reason"],
        "alternatives": parsed["alternatives"],
        "raw": raw_content,
        "model_name": model,
        "quant_score_candidates": candidates,
    }


def _score_response_was_truncated(response: Mapping[str, Any]) -> bool:
    response_payload = response.get("response", {}) if isinstance(response, Mapping) else {}
    if not isinstance(response_payload, Mapping):
        return False
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        return False
    return str(choices[0].get("finish_reason", "") or "").strip().lower() == "length"


def run_llm_score_prediction(
    row: Mapping[str, Any],
    features: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    quant: Mapping[str, Any],
    ml: Mapping[str, Any],
    legacy: Mapping[str, Any],
    blended: Mapping[str, Any],
    quality: Mapping[str, Any],
    final_risk: Mapping[str, Any],
) -> dict[str, Any]:
    base_url = os.getenv("COLLECTION_BASE_URL", "").strip()
    api_key = os.getenv("COLLECTION_APIKEY", "").strip()
    model = os.getenv("COLLECTION_MODEL", "").strip()
    globally_enabled = _llm_review_globally_enabled()
    candidates = _score_candidate_payload(quant)
    fallback_score = candidates[0]["score"] if candidates else ""
    score_prediction = {
        "enabled": 1 if (globally_enabled and base_url and api_key and model) else 0,
        "status": "skipped",
        "score": fallback_score,
        "confidence": 0.0,
        "reason": (
            "LLM review disabled; using quant score candidate as fallback."
            if not globally_enabled
            else "Review model is not configured; using quant score candidate as fallback."
        ),
        "alternatives": [item["score"] for item in candidates[1:4]],
        "raw": "",
        "model_name": model,
        "quant_score_candidates": candidates,
    }
    if not score_prediction["enabled"]:
        return score_prediction

    prompt = _build_score_prediction_prompt(
        row,
        features,
        snapshot,
        quant,
        ml,
        legacy,
        blended,
        quality,
        final_risk,
    )
    try:
        response = _request_score_prediction_response(
            base_url,
            api_key,
            model,
            messages=_build_openai_messages(
                "Return final JSON only. Do not reveal reasoning or analysis.",
                prompt,
            ),
            max_tokens=900,
        )
        raw_content = str(response.get("content", "") or "")
        try:
            parsed = _parse_score_prediction_payload(raw_content)
        except Exception as exc:  # noqa: BLE001
            reason, detail = _classify_review_failure(
                exc,
                endpoint=str(response.get("endpoint", "") or ""),
                raw_text=raw_content,
            )
            if (
                reason == "review model output was truncated"
                or _score_response_was_truncated(response)
                or not raw_content.strip()
            ):
                repair = _repair_truncated_score_prediction(
                    base_url,
                    api_key,
                    model,
                    response=response,
                    fallback_score=fallback_score,
                    candidates=candidates,
                    final_risk=final_risk,
                )
                if repair is not None:
                    return repair
            score_prediction.update(
                {
                    "status": "failed",
                    "reason": reason,
                    "raw": detail,
                    "model_name": model,
                }
            )
            return score_prediction
        score_prediction.update(
            {
                "status": "completed",
                "score": parsed["score"],
                "confidence": parsed["confidence"],
                "reason": parsed["reason"],
                "alternatives": parsed["alternatives"],
                "raw": raw_content,
                "model_name": model,
            }
        )
        return score_prediction
    except Exception as exc:  # noqa: BLE001
        reason, detail = _classify_review_failure(exc)
        score_prediction.update(
            {
                "status": "failed",
                "reason": reason,
                "raw": detail,
                "model_name": model,
            }
        )
        return score_prediction


def _llm_review_globally_enabled() -> bool:
    """Master switch for the LLM review chain.

    Reads ``LLM_REVIEW_ENABLED`` from the environment (written via the UI
    config page). Defaults to enabled for backward compatibility �?older
    installs that have no such key in ``.env`` keep their previous behavior.
    """

    raw = os.getenv("LLM_REVIEW_ENABLED", "").strip().lower()
    if raw in {"false", "0", "no", "off", "disabled"}:
        return False
    return True


def run_llm_recommendation_review(
    row: Mapping[str, Any],
    features: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    blended: Mapping[str, Any],
    quality: Mapping[str, Any],
    algo_risk: Mapping[str, Any],
) -> dict[str, Any]:
    base_url = os.getenv("COLLECTION_BASE_URL", "").strip()
    api_key = os.getenv("COLLECTION_APIKEY", "").strip()
    model = os.getenv("COLLECTION_MODEL", "").strip()
    globally_enabled = _llm_review_globally_enabled()
    review = {
        "enabled": 1 if (globally_enabled and base_url and api_key and model) else 0,
        "status": "skipped",
        "decision": "",
        "target_action": _action_label(str(algo_risk.get("recommendation", ""))),
        "reason": (
            "LLM review disabled; keeping algorithm action."
            if not globally_enabled
            else "LLM review is not configured; keeping algorithm action."
        ),
        "risk_flags": [],
        "evidence_grade": "",
        "confidence_delta": 0.0,
        "stake_multiplier": 1.0,
        "outcome_decision": "confirm",
        "target_outcome": str(algo_risk.get("recommended_outcome", "") or ""),
        "outcome_reason": "",
        "raw": "",
        "model_name": model,
    }
    if not review["enabled"]:
        return review

    try:
        response = _request_review_response(
            base_url,
            api_key,
            model,
            messages=_build_openai_messages(
                "You are an action-strength reviewer for a football betting system. Output final JSON only.",
                _build_review_prompt(row, features, snapshot, blended, quality, algo_risk),
            ),
        )
        raw_content = str(response.get("content", "") or "")
        try:
            parsed = _parse_review_payload(
                raw_content,
                current_action=str(algo_risk.get("recommendation", "") or ""),
                current_outcome=str(algo_risk.get("recommended_outcome", "") or ""),
            )
        except Exception as exc:  # noqa: BLE001
            reason, detail = _classify_review_failure(
                exc,
                endpoint=str(response.get("endpoint", "") or ""),
                raw_text=raw_content,
            )
            review.update(
                {
                    "status": "failed",
                    "reason": reason,
                    "raw": detail,
                }
            )
            return review
        review.update(
            {
                "status": "completed",
                "decision": parsed["decision"],
                "target_action": parsed["target_action"],
                "reason": parsed["reason"],
                "risk_flags": parsed["risk_flags"],
                "evidence_grade": parsed.get("evidence_grade", ""),
                "confidence_delta": safe_float(parsed.get("confidence_delta")),
                "stake_multiplier": safe_float(parsed.get("stake_multiplier"), 1.0),
                "outcome_decision": parsed.get("outcome_decision", "confirm"),
                "target_outcome": parsed.get("target_outcome", ""),
                "outcome_reason": parsed.get("outcome_reason", ""),
                "raw": raw_content,
                "model_name": model,
            }
        )
        return review
    except Exception as exc:  # noqa: BLE001
        reason, detail = _classify_review_failure(exc)
        review.update(
            {
                "status": "failed",
                "reason": reason,
                "raw": detail,
            }
        )
        return review


def run_llm_arbiter_review(
    row: Mapping[str, Any],
    features: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    blended: Mapping[str, Any],
    quality: Mapping[str, Any],
    final_risk: Mapping[str, Any],
    review: Mapping[str, Any],
    trigger_reasons: list[str],
) -> dict[str, Any]:
    base_url = os.getenv("COLLECTION_BASE_URL", "").strip()
    api_key = os.getenv("COLLECTION_APIKEY", "").strip()
    model = os.getenv("COLLECTION_MODEL", "").strip()
    globally_enabled = _llm_review_globally_enabled()
    arbiter = {
        "enabled": 1 if (globally_enabled and base_url and api_key and model) else 0,
        "triggered": 1 if trigger_reasons else 0,
        "status": "skipped",
        "decision": "",
        "target_action": _action_label(str(final_risk.get("recommendation", ""))),
        "reason": (
            "LLM arbiter disabled; keeping post-review machine action."
            if not globally_enabled
            else "LLM arbiter is not configured; keeping post-review machine action."
        ),
        "risk_flags": [],
        "raw": "",
        "model_name": model,
        "trigger_reasons": trigger_reasons[:],
    }
    if not trigger_reasons:
        arbiter["status"] = "not_triggered"
        arbiter["reason"] = "No high-risk arbiter trigger matched."
        return arbiter
    if not arbiter["enabled"]:
        return arbiter

    try:
        response = _request_review_response(
            base_url,
            api_key,
            model,
            messages=_build_openai_messages(
                "You are a high-risk second-stage arbiter for a football betting system. Output final JSON only.",
                _build_arbiter_prompt(
                    row,
                    features,
                    snapshot,
                    blended,
                    quality,
                    final_risk,
                    review,
                    trigger_reasons,
                ),
            ),
        )
        raw_content = str(response.get("content", "") or "")
        try:
            parsed = _parse_arbiter_payload(raw_content)
        except Exception as exc:  # noqa: BLE001
            reason, detail = _classify_review_failure(
                exc,
                endpoint=str(response.get("endpoint", "") or ""),
                raw_text=raw_content,
            )
            arbiter.update(
                {
                    "status": "failed",
                    "reason": reason,
                    "raw": detail,
                }
            )
            return arbiter
        arbiter.update(
            {
                "status": "completed",
                "decision": parsed["decision"],
                "target_action": parsed["target_action"],
                "reason": parsed["reason"],
                "risk_flags": parsed["risk_flags"],
                "raw": raw_content,
                "model_name": model,
            }
        )
        return arbiter
    except Exception as exc:  # noqa: BLE001
        reason, detail = _classify_review_failure(exc)
        arbiter.update(
            {
                "status": "failed",
                "reason": reason,
                "raw": detail,
            }
        )
        return arbiter


def _build_expert_review_prompt(
    row: Mapping[str, Any],
    features: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    blended: Mapping[str, Any],
    quality: Mapping[str, Any],
    final_risk: Mapping[str, Any],
    review: Mapping[str, Any],
    arbiter_review: Mapping[str, Any],
    trigger_reasons: list[str],
) -> str:
    outcome = str(final_risk.get("recommended_outcome", "") or "")
    lineup = features["lineup"]
    schedule = features["schedule"]
    return (
        "You are the expert final reviewer for a football betting system.\n"
        "Only decide the final execution action strength. Do not change the predicted outcome.\n"
        "If the fixed outcome is not reliable, choose target_action=watch.\n"
        "Return strict JSON only. No markdown or extra prose.\n"
        "Schema:\n"
        '{"target_action":"main|light|watch","reason":"short reason","risk_flags":["flag1","flag2"],"evidence_grade":"strong|adequate|weak|unsafe","stake_multiplier":1.0}\n\n'
        "Rules:\n"
        "1. Do not output a new outcome direction.\n"
        "2. evidence_grade=unsafe or stake_multiplier=0 requires target_action=watch.\n"
        "3. evidence_grade=weak should not be main; at most light.\n"
        "4. stake_multiplier must be from 0 to 1 and may only reduce stake.\n"
        "5. Keep reason short and risk_flags to at most 3 items.\n\n"
        f"Match: {row['home_team']} vs {row['away_team']}\n"
        f"League/time: {row['league']} | {row['match_time']}\n"
        f"Fixed outcome: {_choice_label(outcome)}\n"
        f"First review: {review.get('status', '')} / {review.get('decision', '')} / {review.get('target_action', '')}\n"
        f"First review reason: {review.get('reason') or '-'}\n"
        f"Second arbiter: {arbiter_review.get('decision', '')} / {arbiter_review.get('target_action', '')}\n"
        f"Second arbiter reason: {arbiter_review.get('reason') or '-'}\n"
        f"Trigger reasons: {'; '.join(trigger_reasons) or '-'}\n"
        f"Current machine action: {final_risk['recommendation']} | stake {safe_float(final_risk.get('stake_pct')):.2f}%\n"
        f"Confidence: {safe_float(final_risk['confidence']):.3f} | Risk: {final_risk['risk_level']}\n"
        f"Data quality: {safe_float(quality['score']):.3f}\n"
        f"Action score: {safe_float(final_risk.get('action_score')):.3f}\n"
        f"EV margin: {safe_float(final_risk.get('ev_margin')):.3f} | Legacy gap: {safe_float(final_risk.get('legacy_gap')):.3f}\n"
        f"Final probabilities: {json.dumps(blended['probabilities'], ensure_ascii=False)}\n"
        f"Market probabilities: {json.dumps(final_risk['market_probs'], ensure_ascii=False)}\n"
        f"Market bias: {json.dumps(final_risk['market_bias'], ensure_ascii=False)}\n"
        f"EV: {json.dumps(final_risk['expected_values'], ensure_ascii=False)}\n"
        f"Home/away rating: {safe_float(snapshot.get('home_rating')):.1f} / {safe_float(snapshot.get('away_rating')):.1f}\n"
        f"Lineup availability: home {safe_float(lineup['home_availability']):.3f} / away {safe_float(lineup['away_availability']):.3f}\n"
        f"Key absences: home {safe_int(lineup['home_absent_count'])}+doubtful {safe_int(lineup['home_doubtful_count'])} | "
        f"away {safe_int(lineup['away_absent_count'])}+doubtful {safe_int(lineup['away_doubtful_count'])}\n"
        f"Rest/load: home rest {safe_float(schedule['home_rest_days']):.1f}d / away rest {safe_float(schedule['away_rest_days']):.1f}d | "
        f"14d load {safe_int(schedule['home_load_14'])}:{safe_int(schedule['away_load_14'])}\n"
        f"Warnings: {'; '.join(final_risk['warnings']) or 'none'}\n"
    )


def run_expert_llm_final_review(
    row: Mapping[str, Any],
    features: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    blended: Mapping[str, Any],
    quality: Mapping[str, Any],
    final_risk: Mapping[str, Any],
    review: Mapping[str, Any],
    arbiter_review: Mapping[str, Any],
    trigger_reasons: list[str],
) -> dict[str, Any]:
    base_url = os.getenv("COLLECTION_BASE_URL", "").strip()
    api_key = os.getenv("COLLECTION_APIKEY", "").strip()
    model = os.getenv("COLLECTION_MODEL", "").strip()
    globally_enabled = _llm_review_globally_enabled()
    expert = {
        "enabled": 1 if (globally_enabled and base_url and api_key and model) else 0,
        "status": "skipped",
        "target_action": "观望",
        "reason": (
            "LLM review disabled; expert final review defaults to watch."
            if not globally_enabled
            else "Expert final review is not configured; defaulting to watch."
        ),
        "risk_flags": ["expert_unconfigured"],
        "evidence_grade": "unsafe",
        "stake_multiplier": 0.0,
        "raw": "",
        "model_name": model,
        "direction_guarded": False,
    }
    if not expert["enabled"]:
        return expert

    try:
        response = _request_review_response(
            base_url,
            api_key,
            model,
            messages=_build_openai_messages(
                "You are the expert final reviewer for a football betting system. Output final JSON only.",
                _build_expert_review_prompt(
                    row,
                    features,
                    snapshot,
                    blended,
                    quality,
                    final_risk,
                    review,
                    arbiter_review,
                    trigger_reasons,
                ),
            ),
        )
        raw_content = str(response.get("content", "") or "")
        try:
            parsed = _parse_expert_review_payload(
                raw_content,
                str(final_risk.get("recommended_outcome", "") or ""),
            )
        except Exception as exc:  # noqa: BLE001
            reason, detail = _classify_review_failure(
                exc,
                endpoint=str(response.get("endpoint", "") or ""),
                raw_text=raw_content,
            )
            expert.update(
                {
                    "status": "failed",
                    "reason": reason,
                    "raw": detail,
                    "model_name": model,
                }
            )
            return expert
        expert.update(
            {
                "status": "completed",
                "target_action": parsed["target_action"],
                "reason": parsed["reason"],
                "risk_flags": parsed["risk_flags"],
                "evidence_grade": parsed["evidence_grade"],
                "stake_multiplier": parsed["stake_multiplier"],
                "raw": raw_content,
                "model_name": model,
                "direction_guarded": parsed["direction_guarded"],
            }
        )
        return expert
    except Exception as exc:  # noqa: BLE001
        reason, detail = _classify_review_failure(exc)
        expert.update(
            {
                "status": "failed",
                "reason": reason,
                "raw": detail,
                "model_name": model,
            }
        )
        return expert


def _review_stake_multiplier(review: Mapping[str, Any]) -> float:
    return clamp(safe_float(review.get("stake_multiplier"), 1.0), 0.0, 1.0)


def _review_confidence_delta(review: Mapping[str, Any]) -> float:
    return clamp(safe_float(review.get("confidence_delta")), -0.12, 0.08)


def _append_resolution_reason(base_reason: str, addition: str) -> str:
    base = str(base_reason or "").strip()
    addition = str(addition or "").strip()
    if not addition:
        return base
    if not base:
        return addition
    if addition in base:
        return base
    return f"{base} {addition}"


def _apply_review_evidence_guard(
    final_action: str,
    review: Mapping[str, Any],
    final_risk: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    decision = str(review.get("decision", "") or "")
    evidence_grade = str(review.get("evidence_grade", "") or "").strip().lower()
    stake_multiplier = _review_stake_multiplier(review)
    low_odds_favorite_guard = bool((final_risk or {}).get("low_odds_favorite_guard"))

    if low_odds_favorite_guard and decision == "abstain":
        return final_action, "LLM abstained, but low-odds favorite guard keeps the light action."
    if (decision == "abstain" and not low_odds_favorite_guard) or evidence_grade == "unsafe" or stake_multiplier <= 0:
        return "观望", "LLM evidence is insufficient or abstained; action downgraded to watch."
    if evidence_grade == "weak" and final_action == "主推":
        return "轻仓", "LLM evidence grade is weak; main action downgraded to light."
    return final_action, ""


def _apply_confidence_gate(
    final_action: str,
    adjusted_confidence: float,
    threshold_config: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    gates = _merge_threshold_config(threshold_config)
    next_action = final_action
    if final_action == "主推" and adjusted_confidence < safe_float(gates["main"]["confidence"]):
        next_action = "轻仓"
    if next_action == "轻仓" and adjusted_confidence < safe_float(gates["light"]["confidence"]):
        next_action = "观望"
    if next_action != final_action:
        return next_action, f"LLM-adjusted confidence does not support {final_action}; downgraded to {next_action}."
    return final_action, ""


def _apply_review_outcome_guard(
    final_action: str,
    final_risk: dict[str, Any],
    review: Mapping[str, Any],
) -> tuple[str, str]:
    decision = str(review.get("outcome_decision", "") or "confirm").strip().lower()
    current_outcome = str(final_risk.get("recommended_outcome", "") or "")
    target_outcome = _outcome_label(str(review.get("target_outcome", "") or ""))
    outcome_reason = str(review.get("outcome_reason", "") or "").strip()
    low_odds_favorite_guard = bool(final_risk.get("low_odds_favorite_guard"))

    if low_odds_favorite_guard and decision in {"veto_to_watch", "challenge"}:
        return final_action, "LLM challenged the outcome, but low-odds favorite guard keeps the model direction."

    if decision == "veto_to_watch":
        reason = outcome_reason or "LLM outcome review vetoed the current direction; execute watch."
        return "观望", reason
    if decision == "challenge" and target_outcome and target_outcome != current_outcome:
        reason = outcome_reason or (
            f"LLM outcome review challenges {_choice_label(current_outcome)} "
            f"and leans {_choice_label(target_outcome)}; execute watch."
        )
        return "观望", reason
    return final_action, ""


def resolve_recommendation(
    quality: Mapping[str, Any],
    algo_risk: Mapping[str, Any],
    review: Mapping[str, Any],
    arbiter_review: Mapping[str, Any] | None = None,
    threshold_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    final_risk = dict(algo_risk)
    current_action = _action_label(str(algo_risk.get("recommendation", "")))
    final_action = current_action
    resolution_reason = "Keep algorithm action."
    confidence_delta = 0.0
    stake_multiplier = 1.0

    status = str(review.get("status", "") or "")
    if status == "failed":
        resolution_reason = str(review.get("reason", "") or "LLM review failed; keep algorithm action.")
    elif status == "skipped":
        resolution_reason = str(review.get("reason", "") or "LLM review skipped; keep algorithm action.")
    elif status == "completed":
        decision = str(review.get("decision", "") or "")
        target_action = _action_label(str(review.get("target_action", "") or current_action))
        low_odds_favorite_guard = bool(final_risk.get("low_odds_favorite_guard"))
        if decision == "downgrade":
            next_action = _downgrade_action(current_action, target_action)
            final_action = next_action
            if next_action != current_action:
                resolution_reason = str(review.get("reason", "") or f"LLM suggested downgrade to {next_action}.")
            else:
                resolution_reason = str(review.get("reason", "") or "LLM suggested downgrade, but action did not change.")
        elif decision == "promote":
            if _promotion_constraints_pass(target_action, quality, algo_risk, threshold_config):
                next_action = _promote_action(current_action, target_action)
                final_action = next_action
                if next_action != current_action:
                    resolution_reason = str(review.get("reason", "") or f"LLM suggested promotion to {next_action}.")
                else:
                    resolution_reason = str(review.get("reason", "") or "LLM suggested promotion, but action did not change.")
            else:
                resolution_reason = "LLM suggested promotion, but hard risk constraints failed; keep algorithm action."
        elif decision == "keep":
            resolution_reason = str(review.get("reason", "") or "LLM review kept algorithm action.")
        elif decision == "abstain" and low_odds_favorite_guard:
            resolution_reason = str(
                review.get("reason", "")
                or "LLM review abstained, but low-odds favorite guard keeps algorithm action."
            )
        else:
            final_action = "观望"
            resolution_reason = str(review.get("reason", "") or "LLM review abstained; execute watch.")

        confidence_delta = _review_confidence_delta(review)
        stake_multiplier = _review_stake_multiplier(review)
        final_action, evidence_reason = _apply_review_evidence_guard(final_action, review, final_risk)
        resolution_reason = _append_resolution_reason(resolution_reason, evidence_reason)
        final_action, outcome_reason = _apply_review_outcome_guard(final_action, final_risk, review)
        resolution_reason = _append_resolution_reason(resolution_reason, outcome_reason)

    adjusted_confidence = clamp(
        safe_float(algo_risk.get("confidence")) + confidence_delta,
        0.18,
        0.88,
    )
    if status == "completed":
        final_action, confidence_reason = _apply_confidence_gate(
            final_action,
            adjusted_confidence,
            threshold_config,
        )
        resolution_reason = _append_resolution_reason(resolution_reason, confidence_reason)
    if arbiter_review is not None:
        final_action, gate_reason = _llm_action_gate(
            final_action,
            low_odds_favorite_guard=bool(final_risk.get("low_odds_favorite_guard")),
            review_status=status,
            review_decision=str(review.get("decision", "") or ""),
            review_target_action=str(review.get("target_action", "") or ""),
            arbiter_status=str(arbiter_review.get("status", "") or ""),
            arbiter_decision=str(arbiter_review.get("decision", "") or ""),
            arbiter_target_action=str(arbiter_review.get("target_action", "") or ""),
        )
        resolution_reason = _append_resolution_reason(resolution_reason, gate_reason)

    final_risk["recommendation"] = final_action
    final_risk["confidence"] = adjusted_confidence
    final_risk["risk_level"] = action_risk_level(adjusted_confidence)
    base_stake = safe_float(algo_risk.get("stake_pct"))
    final_stake = _stake_for_action(final_action, final_risk)
    if status == "completed" and final_action != "观望":
        if bool(final_risk.get("low_odds_favorite_guard")) and stake_multiplier <= 0:
            stake_multiplier = 1.0
            resolution_reason = _append_resolution_reason(
                resolution_reason,
                "LLM stake multiplier was zero, but low-odds favorite guard keeps algorithm stake.",
            )
        if base_stake > 0:
            final_stake = min(final_stake, base_stake)
        final_stake = round(final_stake * stake_multiplier, 2)
    final_risk["stake_pct"] = 0.0 if final_action == "观望" else final_stake
    # Preserve the pre-review (algo) action so the arbiter can decide whether
    # the first-level reviewer's downgrade was warranted; allow_with_uplift
    # uses this as the ceiling for restoration.
    final_risk["algo_recommendation"] = _action_label(
        str(algo_risk.get("recommendation", "") or "")
    )
    final_risk["algo_stake_pct"] = safe_float(algo_risk.get("stake_pct"))
    final_risk["review_confidence_delta"] = confidence_delta
    final_risk["review_stake_multiplier"] = stake_multiplier
    final_risk["review_evidence_grade"] = str(review.get("evidence_grade", "") or "")
    final_risk["review_outcome_decision"] = str(review.get("outcome_decision", "") or "")
    final_risk["review_target_outcome"] = str(review.get("target_outcome", "") or "")
    final_risk["review_outcome_reason"] = str(review.get("outcome_reason", "") or "")
    final_risk["resolution_reason"] = resolution_reason
    final_risk["review_status_label"] = _review_status_label(review)
    return final_risk


def _resolve_execution_outcome(
    final_risk: Mapping[str, Any],
    arbiter_review: Mapping[str, Any],
    *,
    requested_at: str,
    expert_review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    current_action = _action_label(str(final_risk.get("recommendation", "")))
    current_stake = safe_float(final_risk.get("stake_pct"))
    default_resolution = {
        "effective_recommendation": current_action,
        "effective_stake_pct": current_stake,
        "effective_action_source": "model",
        "manual_review_status": "",
        "manual_review_reason": "",
        "manual_review_requested_at": "",
        "manual_review_resolved_at": "",
        "manual_review_notes": "",
        "execution_status": "executable",
    }

    status = str(arbiter_review.get("status", "") or "")
    if status != "completed":
        return default_resolution

    decision = str(arbiter_review.get("decision", "") or "")
    target_action = _action_label(str(arbiter_review.get("target_action", "") or current_action))
    if decision == "allow_with_uplift":
        # The arbiter judged that the first-level LLM downgraded without new
        # evidence �?restore toward the algorithm's original action, but
        # never above it. This is the only path that can recover from a
        # ratchet-style chain of downgrades.
        algo_action = _action_label(str(final_risk.get("algo_recommendation", current_action) or current_action))
        algo_level = ACTION_LEVELS.get(algo_action, ACTION_LEVELS.get(current_action, 0))
        target_level = ACTION_LEVELS.get(target_action, algo_level)
        capped_level = min(algo_level, max(target_level, ACTION_LEVELS.get(current_action, 0)))
        next_action = ACTION_BY_LEVEL[capped_level]
        if next_action == current_action:
            return default_resolution
        uplifted_risk = dict(final_risk)
        uplifted_risk["recommendation"] = next_action
        next_stake = _stake_for_action(next_action, final_risk)
        return {
            "effective_recommendation": next_action,
            "effective_stake_pct": safe_float(next_stake),
            "effective_action_source": "arbiter_uplift",
            "manual_review_status": "",
            "manual_review_reason": "",
            "manual_review_requested_at": "",
            "manual_review_resolved_at": "",
            "manual_review_notes": "",
            "execution_status": "arbiter_uplifted",
        }
    if decision == "downgrade":
        next_action = _downgrade_action(current_action, target_action)
        if next_action == current_action:
            next_action = _single_step_downgrade(current_action)
        downgraded_risk = dict(final_risk)
        downgraded_risk["recommendation"] = next_action
        downgraded_risk["stake_pct"] = _stake_for_action(next_action, final_risk)
        return {
            "effective_recommendation": next_action,
            "effective_stake_pct": safe_float(downgraded_risk["stake_pct"]),
            "effective_action_source": "arbiter",
            "manual_review_status": "",
            "manual_review_reason": "",
            "manual_review_requested_at": "",
            "manual_review_resolved_at": "",
            "manual_review_notes": "",
            "execution_status": "arbiter_downgraded",
        }
    if decision == "manual_review":
        if expert_review is not None:
            expert_status = str(expert_review.get("status", "") or "")
            expert_action = _action_label(str(expert_review.get("target_action", "") or "观望"))
            evidence_grade = str(expert_review.get("evidence_grade", "") or "").strip().lower()
            stake_multiplier = clamp(safe_float(expert_review.get("stake_multiplier")), 0.0, 1.0)
            if expert_status != "completed" or evidence_grade == "unsafe" or stake_multiplier <= 0:
                expert_action = "观望"
                stake_multiplier = 0.0
            stake_pct = 0.0 if expert_action == "观望" else round(_stake_for_action(expert_action, final_risk) * stake_multiplier, 2)
            flags = [str(item).strip() for item in expert_review.get("risk_flags", []) if str(item).strip()]
            reason = str(expert_review.get("reason", "") or "").strip()
            notes_parts = [f"专家终审：{reason}" if reason else "专家终审：无理由"]
            if evidence_grade:
                notes_parts.append(f"证据等级 {evidence_grade}")
            if flags:
                notes_parts.append(f"风险标记 {'; '.join(flags[:3])}")
            return {
                "effective_recommendation": expert_action,
                "effective_stake_pct": stake_pct,
                "effective_action_source": "expert_llm" if expert_status == "completed" else "expert_llm_failed",
                "manual_review_status": "resolved",
                "manual_review_reason": str(arbiter_review.get("reason", "") or ""),
                "manual_review_requested_at": requested_at,
                "manual_review_resolved_at": requested_at,
                "manual_review_notes": " | ".join(notes_parts),
                "execution_status": "expert_review_resolved" if expert_status == "completed" else "expert_review_failed",
            }
        return {
            "effective_recommendation": "",
            "effective_stake_pct": 0.0,
            "effective_action_source": "",
            "manual_review_status": "pending",
            "manual_review_reason": str(arbiter_review.get("reason", "") or ""),
            "manual_review_requested_at": requested_at,
            "manual_review_resolved_at": "",
            "manual_review_notes": "",
            "execution_status": "manual_review_pending",
        }
    if decision == "skip":
        return {
            "effective_recommendation": "观望",
            "effective_stake_pct": 0.0,
            "effective_action_source": "arbiter",
            "manual_review_status": "",
            "manual_review_reason": "",
            "manual_review_requested_at": "",
            "manual_review_resolved_at": "",
            "manual_review_notes": "",
            "execution_status": "arbiter_downgraded",
        }
    return default_resolution


def _build_summary_prompt(
    row: Mapping[str, Any],
    quant: Mapping[str, Any],
    ml: Mapping[str, Any],
    blended: Mapping[str, Any],
    quality: Mapping[str, Any],
    algo_risk: Mapping[str, Any],
    review: Mapping[str, Any],
    final_risk: Mapping[str, Any],
) -> str:
    return (
        "你是足球预测报告分析师。请基于给定结构化结果，输出 3-5 条简洁中文要点，"
        "必须区分算法初判、LLM 复核和最终仲裁，不要虚构数据。\n\n"
        f"比赛: {row['home_team']} vs {row['away_team']}\n"
        f"联赛: {row['league']}\n"
        f"近期状态-主: {row.get('recent_form_home', '')}\n"
        f"近期状态-客: {row.get('recent_form_away', '')}\n"
        f"交锋: {row.get('head_to_head_summary', '')}\n"
        f"阵容: {row.get('injury_or_lineup_notes', '')}\n"
        f"独立量化概率: {json.dumps(quant['probabilities'], ensure_ascii=False)}\n"
        f"独立启发式概率: {json.dumps(ml['probabilities'], ensure_ascii=False)}\n"
        f"最终独立概率: {json.dumps(blended['probabilities'], ensure_ascii=False)}\n"
        f"市场隐含概率: {json.dumps(algo_risk['market_probs'], ensure_ascii=False)}\n"
        f"市场偏差: {json.dumps(algo_risk['market_bias'], ensure_ascii=False)}\n"
        f"赛果: 概率主判 {_choice_label(str(algo_risk.get('predicted_outcome', '') or ''))} / "
        f"价值候选 {_choice_label(str(algo_risk.get('value_outcome', '') or ''))} / "
        f"来源 {algo_risk.get('outcome_source') or '-'} / 评分 {safe_float(algo_risk.get('outcome_score')):.3f}\n"
        f"算法初判: {algo_risk['recommendation']} / {_choice_label(algo_risk['recommended_outcome'])}\n"
        f"LLM复核: {review.get('status', '')} / {review.get('decision', '')} / {review.get('target_action', '')}\n"
        f"方向复核: {review.get('outcome_decision', '')} / "
        f"{_choice_label(str(review.get('target_outcome', '') or ''))} / "
        f"{review.get('outcome_reason', '') or '-'}\n"
        f"最终仲裁: {final_risk['recommendation']} / {_choice_label(final_risk['recommended_outcome'])}\n"
        f"最终仓位: {safe_float(final_risk['stake_pct']):.2f}%\n"
        f"数据质量: {quality['score']:.2f}\n"
        f"风险提示: {'; '.join(final_risk['warnings']) or '无'}\n"
    )


def generate_llm_summary(
    row: Mapping[str, Any],
    quant: Mapping[str, Any],
    ml: Mapping[str, Any],
    blended: Mapping[str, Any],
    quality: Mapping[str, Any],
    algo_risk: Mapping[str, Any],
    review: Mapping[str, Any],
    final_risk: Mapping[str, Any],
) -> dict[str, str]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    model = os.getenv("OPENAI_MODEL_RESEARCH", "").strip()
    provider = "openai-compatible"
    if not api_key or not base_url or not model:
        return {
            "provider": provider,
            "model": model,
            "summary": "LLM 未启用：缺少 Summary Model 配置，当前报告完全由规则与概率模型生成。",
        }

    try:
        response = request_openai_compatible_chat(
            base_url,
            api_key,
            model,
            messages=_build_openai_messages(
                "你负责把足球预测结构化结果整理成简洁中文摘要。",
                _build_summary_prompt(row, quant, ml, blended, quality, algo_risk, review, final_risk),
            ),
            temperature=0.3,
            max_tokens=220,
            timeout=40,
            max_retries=2,
            retry_backoff_seconds=4.0,
            min_interval_seconds=2.0,
            serialize_requests=True,
        )
        return {"provider": provider, "model": model, "summary": response["content"]}
    except Exception as exc:  # noqa: BLE001
        return {
            "provider": provider,
            "model": model,
            "summary": f"LLM 摘要失败，已回退到规则化报告：{exc}",
        }


def build_presenter_report(
    row: Mapping[str, Any],
    features: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    quant: Mapping[str, Any],
    ml: Mapping[str, Any],
    legacy: Mapping[str, Any],
    blended: Mapping[str, Any],
    quality: Mapping[str, Any],
    algo_risk: Mapping[str, Any],
    review: Mapping[str, Any],
    final_risk: Mapping[str, Any],
    llm: Mapping[str, Any],
    score_prediction: Mapping[str, Any] | None = None,
) -> str:
    lineup = features["lineup"]
    schedule = features["schedule"]
    recent_home = features["recent_home"]
    recent_away = features["recent_away"]
    split = features["split"]
    probs = blended["probabilities"]
    market_value = features.get("market_value") if isinstance(features.get("market_value"), Mapping) else {}
    market_probs = algo_risk["market_probs"]
    market_bias = algo_risk["market_bias"]
    warnings = "\n".join(f"- {item}" for item in final_risk["warnings"]) or "- 暂无"
    top_scores = ", ".join(_format_score_pair(item) for item in quant["top_scores"][:3])
    score_prediction = score_prediction or {}
    predicted_score = str(score_prediction.get("score", "") or "")
    predicted_score_status = str(score_prediction.get("status", "") or "")
    predicted_score_reason = str(score_prediction.get("reason", "") or "")
    predicted_score_confidence = safe_float(score_prediction.get("confidence"))
    predicted_score_alternatives = score_prediction.get("alternatives", [])
    if not isinstance(predicted_score_alternatives, list):
        predicted_score_alternatives = []
    predicted_score_alt_text = ", ".join(str(item) for item in predicted_score_alternatives if str(item).strip()) or "-"
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    review_flags = "; ".join(review.get("risk_flags", [])) or "无"

    return f"""# 独立下注模型报告

## 比赛信息
{row["home_team"]} vs {row["away_team"]}
{row["league"]} | {row["match_time"]} | 期号 {row.get("issue", "")}

## Market Value
- Summary: {row.get("market_value_summary", "") or "-"}
- Parsed: home EUR {safe_float(market_value.get("home_value_eur_m")):.2f}m / away EUR {safe_float(market_value.get("away_value_eur_m")):.2f}m, gap EUR {safe_float(market_value.get("gap_eur_m")):+.2f}m

## Base Model Probability
| 选项 | 独立模型 | 市场隐含 | 概率偏差 | 公平赔率 | 市场赔率 | EV |
|------|----------|----------|----------|----------|----------|----|
| 主胜 | {_format_percent(probs["home"])} | {_format_percent(market_probs["home"])} | {market_bias["home"] * 100:+.1f}pp | {algo_risk["fair_odds"]["home"]:.2f} | {features["market_odds"]["home"]:.2f} | {algo_risk["expected_values"]["home"] * 100:+.1f}% |
| 平局 | {_format_percent(probs["draw"])} | {_format_percent(market_probs["draw"])} | {market_bias["draw"] * 100:+.1f}pp | {algo_risk["fair_odds"]["draw"]:.2f} | {features["market_odds"]["draw"]:.2f} | {algo_risk["expected_values"]["draw"] * 100:+.1f}% |
| 客胜 | {_format_percent(probs["away"])} | {_format_percent(market_probs["away"])} | {market_bias["away"] * 100:+.1f}pp | {algo_risk["fair_odds"]["away"]:.2f} | {features["market_odds"]["away"]:.2f} | {algo_risk["expected_values"]["away"] * 100:+.1f}% |

## 模型拆解
- 独立量化层: 主 {_format_percent(quant["probabilities"]["home"])} / 平 {_format_percent(quant["probabilities"]["draw"])} / 客 {_format_percent(quant["probabilities"]["away"])}
- 独立启发式层: 主 {_format_percent(ml["probabilities"]["home"])} / 平 {_format_percent(ml["probabilities"]["draw"])} / 客 {_format_percent(ml["probabilities"]["away"])}
- 旧市场辅助基线: 主 {_format_percent(legacy["probabilities"]["home"])} / 平 {_format_percent(legacy["probabilities"]["draw"])} / 客 {_format_percent(legacy["probabilities"]["away"])}
- 模型一致性: {safe_float(blended["agreement"]) * 10:.1f}/10
- 数据质量: {safe_float(quality["score"]) * 10:.1f}/10
- 综合置信度: {safe_float(algo_risk["confidence"]) * 10:.1f}/10

## 关键特征
- Rating Proxy: 主 {safe_float(snapshot.get("home_rating")):.1f} / 客 {safe_float(snapshot.get("away_rating")):.1f}，差 {safe_float(features["rating_gap"]):+.1f}
- 近期积分效率: 主 {safe_float(recent_home["points_per_game"]):.2f} / 客 {safe_float(recent_away["points_per_game"]):.2f}
- 近期进失球: 主 {safe_float(recent_home["goals_for_per_game"]):.2f} 进 / {safe_float(recent_home["goals_against_per_game"]):.2f} 失；客 {safe_float(recent_away["goals_for_per_game"]):.2f} 进 / {safe_float(recent_away["goals_against_per_game"]):.2f} 失
- 主客场表现: 主 {safe_float(split["home_ppg"]):.2f} PPG / 客 {safe_float(split["away_ppg"]):.2f} PPG
- 阵容完整度: 主 {safe_float(lineup["home_availability"]) * 100:.1f}% / 客 {safe_float(lineup["away_availability"]) * 100:.1f}%
- 关键缺阵: 主缺阵 {safe_int(lineup["home_absent_count"])} + 疑似 {safe_int(lineup["home_doubtful_count"])}；客缺阵 {safe_int(lineup["away_absent_count"])} + 疑似 {safe_int(lineup["away_doubtful_count"])}
- 赛程与休息: 主休 {safe_float(schedule["home_rest_days"]):.1f} 天 / 客休 {safe_float(schedule["away_rest_days"]):.1f} 天；近 14 天负荷 {safe_int(schedule["home_load_14"])} vs {safe_int(schedule["away_load_14"])}
- 历史交锋边际: {safe_float(features["h2h_edge"]):+.2f}

## 量化视角
- 预期进球: 主 {quant["lambda_home"]:.2f} / 客 {quant["lambda_away"]:.2f}
- 最可能比分: {top_scores}
- 大 2.5 球: {_format_percent(quant["over_25"])}
- 小 2.5 球: {_format_percent(quant["under_25"])}

## 预测比分
- LLM 预测比分: {predicted_score or "-"}
- 状态/模型: {predicted_score_status or "-"} / {score_prediction.get("model_name") or "-"}
- 精确比分置信度: {predicted_score_confidence * 100:.1f}%
- 备选比分: {predicted_score_alt_text}
- 理由: {predicted_score_reason or "-"}

## 算法初判
- 概率主判赛果: {_choice_label(str(algo_risk.get("predicted_outcome", "") or ""))}
- 价值候选方向: {_choice_label(str(algo_risk.get("value_outcome", "") or ""))}
- 最终推荐方向来源: {algo_risk.get("outcome_source") or "-"}
- 方向评分: {safe_float(algo_risk.get("outcome_score")):.3f}
- 方向说明: {algo_risk.get("outcome_reason") or "-"}
- 推荐动作: {algo_risk["recommendation"]}
- 推荐结果: {_choice_label(algo_risk["recommended_outcome"])}
- 风险等级: {algo_risk["risk_level"]}
- 建议仓位: {safe_float(algo_risk["stake_pct"]):.2f}%
- 动作评分: {safe_float(algo_risk.get("action_score")):.3f}
- EV领先/旧模型分�? {safe_float(algo_risk.get("ev_margin")):.3f} / {safe_float(algo_risk.get("legacy_gap")):.3f}
- Kelly原始仓位: {safe_float(algo_risk.get("kelly_fraction")) * 100:.2f}%

## LLM 复核（统一裁决）
- 状态: {_unified_review_status(review)}
- 裁决: {_unified_review_verdict(review)}
- 证据等级: {review.get("evidence_grade") or "-"}
- 理由: {review.get("reason") or "-"}
- 风险标记: {review_flags}
- 系统映射: 动作 {review.get("target_action") or "-"} / 仓位折扣 {safe_float(review.get("stake_multiplier"), 1.0):.2f} / 置信修正 {safe_float(review.get("confidence_delta")):+.3f}

## 最终仲裁
- 推荐动作: {final_risk["recommendation"]}
- 推荐结果: {_choice_label(final_risk["recommended_outcome"])}
- 风险等级: {final_risk["risk_level"]}
- 建议仓位: {safe_float(final_risk["stake_pct"]):.2f}%
- 动作评分: {safe_float(final_risk.get("action_score")):.3f}
- 仲裁说明: {final_risk["resolution_reason"]}

## 风险提示
{warnings}

## LLM 摘要
{llm["summary"].strip()}

## 数据摘要
- 近期状态: {row.get("recent_form_home", "")} | {row.get("recent_form_away", "")}
- 主客场: {row.get("home_away_form", "")}
- 交锋: {row.get("head_to_head_summary", "")}
- 阵容伤停: {row.get("injury_or_lineup_notes", "")}
- 市场赔率: {row.get("european_odds_movement_summary", "")}
- 投注热度: {row.get("betting_heat_summary", "")}

*生成时间: {created_at}*
*说明: 最终概率来自独立定价层；赔率只用于市场比较、EV 和仓位过滤。*
"""


def predict_match(
    match_id: str,
    ensure_collected: bool = False,
    progress_callback=None,
    apply_issue_strategy: bool = True,
    use_llm: bool = True,
) -> dict[str, Any]:
    init_db()
    expire_pending_manual_reviews(match_id=match_id)
    row = get_match_analysis(match_id)
    if row is None:
        raise RuntimeError(f"未找到 match_id={match_id} 的已采集数据，请先采集当前对赛。")

    collection_failure_reason = get_collection_failure_reason(row)
    if collection_failure_reason:
        match_label = f"{row['home_team']} vs {row['away_team']}"
        if not str(_row_field(row, "collected_at", "") or "").strip():
            raise RuntimeError(f"当前场次未采集，请先采集：{match_label}")
        raise RuntimeError(
            f"当前场次采集异常，无法预测：{match_label}，{collection_failure_reason}"
        )

    data = dict(row)
    match_label = f"{data['home_team']} vs {data['away_team']}"
    now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _emit_progress(
        progress_callback,
        current_item_label=match_label,
        current_step="准备预测",
        message=f"准备预测：{match_label}",
    )

    _emit_progress(
        progress_callback,
        current_step="构建特征",
        message=f"正在构建特征：{match_label}",
    )
    snapshot = _ensure_feature_snapshot(data)
    features = build_match_features(data, snapshot)

    _emit_progress(
        progress_callback,
        current_step="运行算法模型",
        message=f"正在计算概率与风险：{match_label}",
    )
    quant = run_quant_model(data, features)
    ml = run_ml_model(data, features)
    legacy = run_legacy_market_model(data, features)
    quality = run_data_quality(data, features)
    raw_blended = blend_predictions(quant, ml, quality, features=features)
    learning_profile = get_active_learning_profile_config()
    threshold_config = (
        _merge_threshold_config(learning_profile.get("threshold_params", {}))
        if learning_profile and learning_profile.get("uses_thresholds")
        else _merge_threshold_config()
    )
    blended, calibrated_probabilities, learning_profile_id = _apply_learning_profile_to_blended(
        raw_blended,
        learning_profile,
    )
    algo_risk = run_risk_assessor(
        data,
        features,
        legacy,
        blended,
        quality,
        threshold_config=threshold_config,
        calibrated_probabilities=calibrated_probabilities,
    )
    handicap_risk = evaluate_handicap_recommendation(
        features=features,
        quant=quant,
        quality=quality,
        row=data,
    )
    handicap_risk = _apply_handicap_learning_strategy(handicap_risk, threshold_config)

    if use_llm:
        _emit_progress(
            progress_callback,
            current_step="LLM复核",
            message=f"正在执行 LLM 复核：{match_label}",
        )
        review = run_llm_recommendation_review(data, features, snapshot, blended, quality, algo_risk)
    else:
        review = {
            "enabled": 0,
            "status": "skipped",
            "decision": "",
            "target_action": _action_label(str(algo_risk.get("recommendation", ""))),
            "reason": "Historical fast backtest skips LLM review; keeping algorithm action.",
            "risk_flags": [],
            "evidence_grade": "",
            "confidence_delta": 0.0,
            "stake_multiplier": 1.0,
            "outcome_decision": "confirm",
            "target_outcome": str(algo_risk.get("recommended_outcome", "") or ""),
            "outcome_reason": "",
            "raw": "",
            "model_name": "",
        }
    final_risk = resolve_recommendation(
        quality,
        algo_risk,
        review,
        threshold_config=threshold_config,
    )
    trigger_reasons = (
        _arbiter_trigger_reasons(quality, legacy, blended, final_risk)
        if str(review.get("status", "") or "") == "completed"
        else []
    )
    arbiter_review = {
        "enabled": 0,
        "triggered": 1 if trigger_reasons else 0,
        "status": "not_triggered",
        "decision": "",
        "target_action": _action_label(str(final_risk.get("recommendation", ""))),
        "reason": "未命中高风险仲裁条件。",
        "risk_flags": [],
        "raw": "",
        "model_name": "",
        "trigger_reasons": trigger_reasons[:],
    }
    expert_review = {
        "enabled": 0,
        "status": "not_triggered",
        "target_action": "",
        "reason": "未触发专家终审。",
        "risk_flags": [],
        "evidence_grade": "",
        "stake_multiplier": 1.0,
        "raw": "",
        "model_name": "",
        "direction_guarded": False,
    }
    review_failed = str(review.get("status", "") or "") == "failed"
    review_failure_reason = str(review.get("reason", "") or "")
    if review_failed:
        _emit_progress(
            progress_callback,
            current_step="LLM复核回退",
            message=f"LLM 复核失败，已回退算法初判：{match_label}，{review_failure_reason}",
            level="warning",
        )
    elif trigger_reasons:
        _emit_progress(
            progress_callback,
            current_step="二级仲裁",
            message=f"命中高风险条件，正在执行二级仲裁：{match_label}",
        )
        arbiter_review = run_llm_arbiter_review(
            data,
            features,
            snapshot,
            blended,
            quality,
            final_risk,
            review,
            trigger_reasons,
        )

    if str(arbiter_review.get("status", "") or "") == "completed" and str(arbiter_review.get("decision", "") or "") == "manual_review":
        _emit_progress(
            progress_callback,
            current_step="专家终审",
            message=f"二级仲裁要求复核，正在执行专�?LLM 终审：{match_label}",
        )
        expert_review = run_expert_llm_final_review(
            data,
            features,
            snapshot,
            blended,
            quality,
            final_risk,
            review,
            arbiter_review,
            trigger_reasons,
        )

    gated_action, gate_reason = _llm_action_gate(
        str(final_risk.get("recommendation", "") or ""),
        low_odds_favorite_guard=bool(final_risk.get("low_odds_favorite_guard")),
        review_status=str(review.get("status", "") or ""),
        review_decision=str(review.get("decision", "") or ""),
        review_target_action=str(review.get("target_action", "") or ""),
        arbiter_status=str(arbiter_review.get("status", "") or ""),
        arbiter_decision=str(arbiter_review.get("decision", "") or ""),
        arbiter_target_action=str(arbiter_review.get("target_action", "") or ""),
    )
    if gated_action != str(final_risk.get("recommendation", "") or ""):
        final_risk["recommendation"] = gated_action
        final_risk["stake_pct"] = 0.0 if gated_action == "观望" else _stake_for_action(gated_action, final_risk)
        final_risk["resolution_reason"] = _append_resolution_reason(
            str(final_risk.get("resolution_reason", "") or ""),
            gate_reason,
        )

    execution_outcome = _resolve_execution_outcome(
        final_risk,
        arbiter_review,
        requested_at=now_text,
        expert_review=expert_review
        if str(arbiter_review.get("decision", "") or "") == "manual_review"
        else None,
    )
    manual_review_required = execution_outcome["manual_review_status"] == "pending"
    expert_review_failed = execution_outcome["effective_action_source"] == "expert_llm_failed"
    arbiter_triggered = bool(trigger_reasons)

    if use_llm:
        _emit_progress(
            progress_callback,
            current_step="预测比分",
            message=f"正在预测比分：{match_label}",
        )
        score_prediction = run_llm_score_prediction(
            data,
            features,
            snapshot,
            quant,
            ml,
            legacy,
            blended,
            quality,
            final_risk,
        )
    else:
        candidates = _score_candidate_payload(quant)
        fallback_score = candidates[0]["score"] if candidates else ""
        score_prediction = {
            "enabled": 0,
            "status": "skipped",
            "score": fallback_score,
            "confidence": 0.0,
            "reason": "Historical fast backtest skips LLM score prediction; using quant score candidate as fallback.",
            "alternatives": [item["score"] for item in candidates[1:4]],
            "raw": "",
            "model_name": "",
            "quant_score_candidates": candidates,
        }

    if use_llm:
        _emit_progress(
            progress_callback,
            current_step="生成摘要",
            message=f"正在生成预测摘要：{match_label}",
        )
        llm = generate_llm_summary(data, quant, ml, blended, quality, algo_risk, review, final_risk)
    else:
        llm = {
            "provider": "local",
            "model": "",
            "summary": "快速历史回测已跳过 LLM 摘要；当前报告由规则与概率模型生成。",
        }

    _emit_progress(
        progress_callback,
        current_step="生成报告",
        message=f"正在整理报告：{match_label}",
    )
    report = build_presenter_report(
        data,
        features,
        snapshot,
        quant,
        ml,
        legacy,
        blended,
        quality,
        algo_risk,
        review,
        final_risk,
        llm,
        score_prediction,
    )

    _emit_progress(
        progress_callback,
        current_step="写入结果",
        message=f"正在写入预测结果：{match_label}",
    )
    run_payload = {
        "match_id": data["match_id"],
        "issue": data.get("issue", ""),
        "created_at": now_text,
        "feature_snapshot_id": safe_int(features.get("feature_snapshot_id")) or safe_int(snapshot.get("snapshot_id")),
        "quant_home_prob": quant["probabilities"]["home"],
        "quant_draw_prob": quant["probabilities"]["draw"],
        "quant_away_prob": quant["probabilities"]["away"],
        "ml_home_prob": ml["probabilities"]["home"],
        "ml_draw_prob": ml["probabilities"]["draw"],
        "ml_away_prob": ml["probabilities"]["away"],
        "legacy_home_prob": legacy["probabilities"]["home"],
        "legacy_draw_prob": legacy["probabilities"]["draw"],
        "legacy_away_prob": legacy["probabilities"]["away"],
        "final_home_prob": raw_blended["probabilities"]["home"],
        "final_draw_prob": raw_blended["probabilities"]["draw"],
        "final_away_prob": raw_blended["probabilities"]["away"],
        "fair_odds_home": algo_risk["fair_odds"]["home"],
        "fair_odds_draw": algo_risk["fair_odds"]["draw"],
        "fair_odds_away": algo_risk["fair_odds"]["away"],
        "market_odds_home": features["market_odds"]["home"],
        "market_odds_draw": features["market_odds"]["draw"],
        "market_odds_away": features["market_odds"]["away"],
        "market_home_prob": algo_risk["market_probs"]["home"],
        "market_draw_prob": algo_risk["market_probs"]["draw"],
        "market_away_prob": algo_risk["market_probs"]["away"],
        "ev_home": algo_risk["expected_values"]["home"],
        "ev_draw": algo_risk["expected_values"]["draw"],
        "ev_away": algo_risk["expected_values"]["away"],
        "quality_score": quality["score"],
        "model_agreement": blended["agreement"],
        "confidence_score": algo_risk["confidence"],
        "risk_level": final_risk["risk_level"],
        "recommendation": final_risk["recommendation"],
        "recommended_outcome": final_risk["recommended_outcome"],
        "suggested_stake_pct": final_risk["stake_pct"],
        "handicap_recommendation": handicap_risk["recommendation"],
        "handicap_recommended_side": handicap_risk["recommended_side"],
        "handicap_line": handicap_risk["line"],
        "handicap_initial_line": handicap_risk["initial_line"],
        "handicap_home_odds": handicap_risk["home_odds"],
        "handicap_away_odds": handicap_risk["away_odds"],
        "handicap_initial_home_odds": handicap_risk["initial_home_odds"],
        "handicap_initial_away_odds": handicap_risk["initial_away_odds"],
        "handicap_home_cover_prob": handicap_risk["home_cover_prob"],
        "handicap_away_cover_prob": handicap_risk["away_cover_prob"],
        "handicap_expected_value": handicap_risk["expected_value"],
        "handicap_confidence": handicap_risk["confidence"],
        "handicap_reason": handicap_risk["reason"],
        "algo_recommendation": algo_risk["recommendation"],
        "algo_recommended_outcome": algo_risk["recommended_outcome"],
        "algo_risk_level": algo_risk["risk_level"],
        "algo_suggested_stake_pct": algo_risk["stake_pct"],
        "llm_review_enabled": review["enabled"],
        "llm_review_status": review["status"],
        "llm_review_decision": review["decision"],
        "llm_review_target_action": review["target_action"],
        "llm_review_reason": review["reason"],
        "llm_review_raw": review["raw"],
        "review_model_name": review["model_name"],
        "final_resolution_reason": final_risk["resolution_reason"],
        "arbiter_review_enabled": arbiter_review["enabled"],
        "arbiter_review_status": arbiter_review["status"],
        "arbiter_review_decision": arbiter_review["decision"],
        "arbiter_review_target_action": arbiter_review["target_action"],
        "arbiter_review_reason": arbiter_review["reason"],
        "arbiter_review_raw": arbiter_review["raw"],
        "arbiter_review_model_name": arbiter_review["model_name"],
        "effective_recommendation": execution_outcome["effective_recommendation"],
        "effective_stake_pct": execution_outcome["effective_stake_pct"],
        "effective_action_source": execution_outcome["effective_action_source"],
        "manual_review_status": execution_outcome["manual_review_status"],
        "manual_review_reason": execution_outcome["manual_review_reason"],
        "manual_review_requested_at": execution_outcome["manual_review_requested_at"],
        "manual_review_resolved_at": execution_outcome["manual_review_resolved_at"],
        "manual_review_notes": execution_outcome["manual_review_notes"],
        "llm_provider": llm["provider"],
        "llm_model": llm["model"],
        "llm_summary": llm["summary"],
        "final_report": report,
        "learning_profile_id": learning_profile_id,
        "calibrated_home_prob": safe_float(calibrated_probabilities.get("home")),
        "calibrated_draw_prob": safe_float(calibrated_probabilities.get("draw")),
        "calibrated_away_prob": safe_float(calibrated_probabilities.get("away")),
        "predicted_score": score_prediction["score"],
        "predicted_score_confidence": safe_float(score_prediction.get("confidence")),
        "predicted_score_reason": score_prediction["reason"],
        "predicted_score_status": score_prediction["status"],
        "predicted_score_model_name": score_prediction["model_name"],
        "predicted_score_raw": score_prediction["raw"],
        "quant_score_candidates": json.dumps(
            score_prediction.get("quant_score_candidates", []),
            ensure_ascii=False,
        ),
    }
    run_id = save_prediction_run(run_payload)
    supersede_pending_manual_reviews(
        data["match_id"],
        exclude_run_id=run_id,
        resolved_at=now_text,
    )
    if review_failed:
        status_message = f"预测已生成：{match_label}；LLM 复核失败，已回退算法初判（{review_failure_reason}）。"
        status_level = "warning"
    elif expert_review_failed:
        status_message = f"预测已生成：{match_label}；专家终审不可用，已自动保守观望。"
        status_level = "warning"
    elif execution_outcome["effective_action_source"] == "expert_llm":
        status_message = f"预测已生成：{match_label}；专家终审已自动处理高风险复核。"
        status_level = "warning"
    elif manual_review_required:
        status_message = f"预测已生成：{match_label}；二级仲裁要求人工复核，当前执行动作已阻断。"
        status_level = "warning"
    elif str(arbiter_review.get("status", "") or "") == "failed":
        status_message = f"预测已生成：{match_label}；二级仲裁失败，暂保留一级复核后的机器建议。"
        status_level = "warning"
    elif str(arbiter_review.get("decision", "") or "") == "downgrade":
        status_message = f"预测已生成：{match_label}；二级仲裁已将执行动作降档。"
        status_level = "warning"
    elif str(arbiter_review.get("decision", "") or "") == "skip":
        status_message = f"预测已生成：{match_label}；二级仲裁已否决当前执行动作。"
        status_level = "warning"
    else:
        status_message = f"预测已生成：{match_label}"
        status_level = "success"
    target_batch_application = (
        apply_target_batch_strategy_to_issue(str(data.get("issue", "") or ""))
        if apply_issue_strategy
        else {
            "issue": str(data.get("issue", "") or ""),
            "strategy_key": "coverage_draw_rescue",
            "updated_count": 0,
            "action_count": 0,
            "watch_count": 0,
            "sample_count": 0,
            "settled_skip_count": 0,
        }
    )
    action_summary_message = (
        _issue_action_summary_message(
            str(data.get("issue", "") or ""),
            target_batch_application,
        )
        if apply_issue_strategy
        else ""
    )
    if action_summary_message:
        status_message += f"；{action_summary_message}"
    _emit_progress(
        progress_callback,
        current_step="当前场次完成",
        message=status_message,
        level=status_level,
    )
    return {
        "run_id": run_id,
        "match_id": data["match_id"],
        "snapshot": dict(snapshot),
        "features": features,
        "quant": quant,
        "ml": ml,
        "legacy": legacy,
        "quality": quality,
        "blended": blended,
        "algo_risk": algo_risk,
        "review": review,
        "arbiter_review": arbiter_review,
        "expert_review": expert_review,
        "score_prediction": score_prediction,
        "risk": final_risk,
        "handicap_risk": handicap_risk,
        "execution": execution_outcome,
        "llm": llm,
        "report": report,
        "status_message": status_message,
        "status_level": status_level,
        "task_message": status_message,
        "review_failed": review_failed,
        "review_failure_reason": review_failure_reason,
        "arbiter_triggered": arbiter_triggered,
        "manual_review_required": manual_review_required,
        "expert_review_failed": expert_review_failed,
        "target_batch_strategy": target_batch_application,
        "execution_status": execution_outcome["execution_status"],
        "execution_status_label": EXECUTION_STATUS_LABELS.get(execution_outcome["execution_status"], "可执行"),
        "effective_recommendation": execution_outcome["effective_recommendation"],
        "effective_stake_pct": safe_float(execution_outcome["effective_stake_pct"]),
    }


def predict_issue(
    issue: str | None = None,
    ensure_collected: bool = False,
    progress_callback=None,
    use_llm: bool = True,
    match_ids: list[str] | None = None,
) -> dict[str, Any]:
    init_db()
    rows = list_matches_by_issue(issue)
    selected_match_ids = {str(match_id).strip() for match_id in (match_ids or []) if str(match_id).strip()}
    if selected_match_ids:
        rows = [row for row in rows if str(row["match_id"]).strip() in selected_match_ids]
    total_matches = len(rows)
    results: list[dict[str, Any]] = []
    skipped_matches: list[dict[str, Any]] = []
    prediction_failed_matches: list[dict[str, Any]] = []
    review_failed_matches: list[dict[str, Any]] = []
    manual_review_matches: list[dict[str, Any]] = []
    expert_review_failed_matches: list[dict[str, Any]] = []
    predicted_count = 0

    _emit_progress(
        progress_callback,
        current_step="准备批量预测",
        total_items=total_matches,
        completed_items=0,
        current_item_index=0,
        message=f"准备预测 {total_matches} 场对赛。",
    )

    for index, row in enumerate(rows, start=1):
        match_label = f"{row['home_team']} vs {row['away_team']}"
        collection_failure_reason = get_collection_failure_reason(row)
        if collection_failure_reason:
            skipped_entry = {
                "match_id": row["match_id"],
                "match_label": match_label,
                "issue": _row_field(row, "issue", ""),
                "status": "skipped",
                "reason": collection_failure_reason,
            }
            skipped_matches.append(skipped_entry)
            results.append(skipped_entry)
            _emit_progress(
                progress_callback,
                total_items=total_matches,
                completed_items=index,
                current_item_index=index,
                current_item_label=match_label,
                current_step="当前场次跳过",
                message=f"第 {index}/{total_matches} 场跳过：{match_label}，{collection_failure_reason}",
                level="warning",
            )
            continue

        _emit_progress(
            progress_callback,
            total_items=total_matches,
            completed_items=index - 1,
            current_item_index=index,
            current_item_label=match_label,
            current_step="进入当前场次",
            message=f"正在预测第 {index}/{total_matches} 场：{match_label}",
        )

        def _match_progress(**payload):
            merged_payload = {
                "total_items": total_matches,
                "completed_items": index - 1,
                "current_item_index": index,
                "current_item_label": match_label,
            }
            merged_payload.update(payload)
            _emit_progress(progress_callback, **merged_payload)

        try:
            match_result = predict_match(
                row["match_id"],
                ensure_collected=False,
                progress_callback=_match_progress,
                apply_issue_strategy=False,
                use_llm=use_llm,
            )
            results.append(match_result)
            predicted_count += 1
            if match_result.get("review_failed"):
                review_failed_entry = {
                    "match_id": row["match_id"],
                    "match_label": match_label,
                    "issue": _row_field(row, "issue", ""),
                    "reason": str(match_result.get("review_failure_reason", "") or "LLM 复核失败"),
                }
                review_failed_matches.append(review_failed_entry)
                completion_message = (
                    f"已完成第 {index}/{total_matches} 场预测：{match_label}；"
                    f"LLM 复核失败并回退算法初判"
                )
                completion_level = "warning"
            elif match_result.get("expert_review_failed"):
                expert_failed_entry = {
                    "match_id": row["match_id"],
                    "match_label": match_label,
                    "issue": _row_field(row, "issue", ""),
                    "reason": str(
                        match_result.get("expert_review", {}).get("reason", "")
                        or "专家终审失败，自动观望"
                    ),
                }
                expert_review_failed_matches.append(expert_failed_entry)
                completion_message = (
                    f"已完成第 {index}/{total_matches} 场预测：{match_label}；"
                    "专家终审失败并自动观望"
                )
                completion_level = "warning"
            elif match_result.get("manual_review_required"):
                manual_review_entry = {
                    "match_id": row["match_id"],
                    "match_label": match_label,
                    "issue": _row_field(row, "issue", ""),
                    "reason": str(
                        match_result.get("arbiter_review", {}).get("reason", "")
                        or "二级仲裁要求人工复核"
                    ),
                }
                manual_review_matches.append(manual_review_entry)
                completion_message = (
                    f"已完成第 {index}/{total_matches} 场预测：{match_label}；"
                    "二级仲裁已转入人工复核"
                )
                completion_level = "warning"
            else:
                completion_message = f"已完成第 {index}/{total_matches} 场预测：{match_label}"
                completion_level = "info"
            _emit_progress(
                progress_callback,
                total_items=total_matches,
                completed_items=index,
                current_item_index=index,
                current_item_label=match_label,
                current_step="当前场次完成",
                message=completion_message,
                level=completion_level,
            )
        except Exception as exc:  # noqa: BLE001
            reason = str(exc)
            failed_entry = {
                "match_id": row["match_id"],
                "match_label": match_label,
                "issue": _row_field(row, "issue", ""),
                "status": "failed",
                "reason": reason,
            }
            prediction_failed_matches.append(failed_entry)
            results.append(failed_entry)
            _emit_progress(
                progress_callback,
                total_items=total_matches,
                completed_items=index,
                current_item_index=index,
                current_item_label=match_label,
                current_step="当前场次失败",
                message=f"第 {index}/{total_matches} 场预测失败：{match_label}，{reason}",
                level="warning",
            )
            continue

    skipped_count = len(skipped_matches)
    prediction_failed_count = len(prediction_failed_matches)
    review_failed_count = len(review_failed_matches)
    manual_review_count = len(manual_review_matches)
    expert_review_failed_count = len(expert_review_failed_matches)
    target_batch_application = (
        apply_target_batch_strategy_to_issue(
            str(_row_field(rows[0], "issue", "") or ""),
            match_ids=selected_match_ids or None,
        )
        if predicted_count > 0 and rows
        else {
            "issue": str(issue or ""),
            "strategy_key": "coverage_draw_rescue",
            "updated_count": 0,
            "action_count": 0,
            "watch_count": 0,
            "sample_count": 0,
            "settled_skip_count": 0,
        }
    )
    action_summary_message = (
        _issue_action_summary_message(
            str(_row_field(rows[0], "issue", "") or ""),
            target_batch_application,
        )
        if predicted_count > 0 and rows
        else ""
    )
    if total_matches and predicted_count == 0 and skipped_count + prediction_failed_count == total_matches:
        task_message = f"当前 {total_matches} 场对赛均未成功生成预测，请先处理采集或预测异常。"
        issue_entries = skipped_matches + prediction_failed_matches
        if issue_entries:
            task_message += f"异常场次：{summarize_issue_entries(issue_entries)}。"
        status_level = "warning"
    elif total_matches == 0:
        task_message = "当前期没有可预测的对赛。"
        status_level = "warning"
    else:
        task_message = (
            f"批量预测完成：共 {total_matches} 场，已预测 {predicted_count} 场，跳过 {skipped_count} 场，失败 {prediction_failed_count} 场。"
        )
        task_suffixes: list[str] = []
        if skipped_count > 0:
            task_suffixes.append(f"{skipped_count} 场采集未达标已跳过")
        if prediction_failed_count > 0:
            task_suffixes.append(f"{prediction_failed_count} 场预测失败")
        if review_failed_count > 0:
            task_suffixes.append(f"{review_failed_count} 场 LLM 复核失败并回退算法初判")
        if manual_review_count > 0:
            task_suffixes.append(f"{manual_review_count} 场进入人工复核")
        if expert_review_failed_count > 0:
            task_suffixes.append(f"{expert_review_failed_count} 场专家终审失败并自动观望")
        task_message += f"；其中 {'；'.join(task_suffixes)}。" if task_suffixes else ""
        issue_entries = skipped_matches + prediction_failed_matches
        if issue_entries:
            task_message += f"异常场次：{summarize_issue_entries(issue_entries)}。"
        if action_summary_message:
            task_message += f" {action_summary_message}"
        status_level = (
            "success"
            if skipped_count == 0
            and prediction_failed_count == 0
            and review_failed_count == 0
            and manual_review_count == 0
            and expert_review_failed_count == 0
            else "warning"
        )

    _emit_progress(
        progress_callback,
        total_items=total_matches,
        completed_items=total_matches,
        current_item_index=total_matches,
        current_item_label=(f"{rows[-1]['home_team']} vs {rows[-1]['away_team']}" if rows else ""),
        current_step="批量预测完成",
        message=task_message,
        level=status_level,
    )
    return {
        "results": results,
        "predicted_count": predicted_count,
        "skipped_count": skipped_count,
        "prediction_failed_count": prediction_failed_count,
        "review_failed_count": review_failed_count,
        "manual_review_count": manual_review_count,
        "expert_review_failed_count": expert_review_failed_count,
        "total_matches": total_matches,
        "skipped_matches": skipped_matches,
        "prediction_failed_matches": prediction_failed_matches,
        "review_failed_matches": review_failed_matches,
        "manual_review_matches": manual_review_matches,
        "expert_review_failed_matches": expert_review_failed_matches,
        "target_batch_strategy": target_batch_application,
        "task_message": task_message,
        "status_message": task_message,
        "status_level": status_level,
    }


def backfill_predicted_scores(
    issue: str | None = None,
    *,
    overwrite: bool = False,
    limit: int | None = None,
    progress_callback=None,
) -> dict[str, Any]:
    init_db()
    rows = list_matches_by_issue(issue)
    total_matches = len(rows)
    updated_count = 0
    skipped_matches: list[dict[str, Any]] = []
    failed_matches: list[dict[str, Any]] = []
    processed_count = 0

    _emit_progress(
        progress_callback,
        current_step="准备回填比分",
        total_items=total_matches,
        completed_items=0,
        current_item_index=0,
        message=f"准备回填 {total_matches} 场预测比分。",
    )

    for index, row in enumerate(rows, start=1):
        if limit is not None and processed_count >= limit:
            break
        match_id = str(row["match_id"])
        match_label = f"{row['home_team']} vs {row['away_team']}"
        runs = list_prediction_runs(match_id, limit=1)
        latest_run = runs[0] if runs else None
        if latest_run is None:
            skipped_matches.append({"match_id": match_id, "match_label": match_label, "reason": "no prediction run"})
            continue
        if str(latest_run["predicted_score"] or "").strip() and not overwrite:
            skipped_matches.append({"match_id": match_id, "match_label": match_label, "reason": "predicted score already exists"})
            continue
        collection_failure_reason = get_collection_failure_reason(row)
        if collection_failure_reason:
            skipped_matches.append({"match_id": match_id, "match_label": match_label, "reason": collection_failure_reason})
            continue

        processed_count += 1
        _emit_progress(
            progress_callback,
            total_items=total_matches,
            completed_items=index - 1,
            current_item_index=index,
            current_item_label=match_label,
            current_step="回填当前场次",
            message=f"正在回填第 {index}/{total_matches} 场预测比分：{match_label}",
        )

        try:
            data = dict(row)
            snapshot = _ensure_feature_snapshot(data)
            features = build_match_features(data, snapshot)
            quant = run_quant_model(data, features)
            ml = run_ml_model(data, features)
            legacy = run_legacy_market_model(data, features)
            quality = run_data_quality(data, features)
            probabilities = {
                "home": safe_float(latest_run["calibrated_home_prob"]) or safe_float(latest_run["final_home_prob"]),
                "draw": safe_float(latest_run["calibrated_draw_prob"]) or safe_float(latest_run["final_draw_prob"]),
                "away": safe_float(latest_run["calibrated_away_prob"]) or safe_float(latest_run["final_away_prob"]),
            }
            probabilities = normalize_probs(
                probabilities["home"],
                probabilities["draw"],
                probabilities["away"],
            )
            sorted_probs = sorted(probabilities.values(), reverse=True)
            blended = {
                "probabilities": probabilities,
                "agreement": safe_float(latest_run["model_agreement"]),
                "margin": (sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) >= 2 else 0.0,
            }
            final_risk = {
                "recommended_outcome": str(latest_run["recommended_outcome"] or ""),
                "recommendation": str(latest_run["recommendation"] or ""),
                "risk_level": str(latest_run["risk_level"] or ""),
                "stake_pct": safe_float(latest_run["suggested_stake_pct"]),
                "market_probs": {
                    "home": safe_float(latest_run["market_home_prob"]),
                    "draw": safe_float(latest_run["market_draw_prob"]),
                    "away": safe_float(latest_run["market_away_prob"]),
                },
                "expected_values": {
                    "home": safe_float(latest_run["ev_home"]),
                    "draw": safe_float(latest_run["ev_draw"]),
                    "away": safe_float(latest_run["ev_away"]),
                },
            }
            score_prediction = run_llm_score_prediction(
                data,
                features,
                snapshot,
                quant,
                ml,
                legacy,
                blended,
                quality,
                final_risk,
            )
            update_prediction_run_fields(
                int(latest_run["run_id"]),
                {
                    "predicted_score": score_prediction["score"],
                    "predicted_score_confidence": safe_float(score_prediction.get("confidence")),
                    "predicted_score_reason": score_prediction["reason"],
                    "predicted_score_status": score_prediction["status"],
                    "predicted_score_model_name": score_prediction["model_name"],
                    "predicted_score_raw": score_prediction["raw"],
                    "quant_score_candidates": json.dumps(
                        score_prediction.get("quant_score_candidates", []),
                        ensure_ascii=False,
                    ),
                },
            )
            updated_count += 1
        except Exception as exc:  # noqa: BLE001
            failed_matches.append({"match_id": match_id, "match_label": match_label, "reason": str(exc)})

    message = f"预测比分回填完成：更新 {updated_count} 场，跳过 {len(skipped_matches)} 场，失败 {len(failed_matches)} 场。"
    _emit_progress(
        progress_callback,
        current_step="比分回填完成",
        total_items=total_matches,
        completed_items=total_matches,
        current_item_index=total_matches,
        message=message,
        level="warning" if failed_matches else "success",
    )
    return {
        "issue": issue or "",
        "updated_count": updated_count,
        "skipped_count": len(skipped_matches),
        "failed_count": len(failed_matches),
        "total_matches": total_matches,
        "skipped_matches": skipped_matches,
        "failed_matches": failed_matches,
        "task_message": message,
        "status_message": message,
        "status_level": "warning" if failed_matches else "success",
    }


def _parse_run_created_at(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _normalize_actual_result(actual_result: str) -> str:
    normalized = str(actual_result or "").strip().lower()
    if normalized not in {"home", "draw", "away"}:
        raise RuntimeError("actual_result 必须�?home、draw �?away")
    return normalized


def _build_match_result_payload(
    match_id: str,
    *,
    actual_result: str,
    actual_score: str,
    result_status: str,
    result_source_url: str,
    result_synced_at: str,
) -> dict[str, Any]:
    return {
        "match_id": match_id,
        "actual_result": actual_result,
        "actual_score": actual_score,
        "result_status": str(result_status or "").strip(),
        "result_source_url": str(result_source_url or "").strip(),
        "result_synced_at": str(result_synced_at or "").strip(),
    }


def _resolve_canonical_prediction_run(match_row: Mapping[str, Any]) -> tuple[Mapping[str, Any] | None, str]:
    match_id = str(match_row["match_id"])
    match_datetime = infer_match_datetime(str(_row_field(match_row, "match_time", "") or ""))
    if match_datetime is None:
        return None, "比赛时间无法解析"

    runs = list_prediction_runs(match_id, limit=None)
    if not runs:
        return None, "未找到预测记录"

    latest_parseable_run = None
    for run in runs:
        created_at = _parse_run_created_at(_row_field(run, "created_at", ""))
        if created_at is None:
            continue
        if latest_parseable_run is None:
            latest_parseable_run = run
        if created_at <= match_datetime:
            return run, ""

    if latest_parseable_run is not None:
        return latest_parseable_run, ""
    return None, "预测记录时间不可解析"


def get_canonical_prediction_run(match_id: str) -> Mapping[str, Any] | None:
    row = get_match_analysis(match_id)
    if row is None:
        return None
    run, _reason = _resolve_canonical_prediction_run(row)
    return run


def _market_odds_for_outcome(row: Mapping[str, Any], outcome: str) -> float:
    if outcome == "home":
        return safe_float(_row_field(row, "market_odds_home"))
    if outcome == "draw":
        return safe_float(_row_field(row, "market_odds_draw"))
    if outcome == "away":
        return safe_float(_row_field(row, "market_odds_away"))
    return 0.0


def _calculate_action_roi(
    row: Mapping[str, Any],
    actual_result: str,
    *,
    action_field: str,
    outcome_field: str,
    stake_field: str,
) -> float:
    action = _action_label(str(_row_field(row, action_field, "")))
    stake_pct = safe_float(_row_field(row, stake_field))
    if action == "观望" or stake_pct <= 0:
        return 0.0

    stake_units = stake_pct / 100.0
    recommended_outcome = str(_row_field(row, outcome_field, "") or "")
    if recommended_outcome == actual_result:
        odds = _market_odds_for_outcome(row, recommended_outcome)
        if odds <= 0:
            return 0.0
        return round(stake_units * (odds - 1.0), 4)
    return round(-stake_units, 4)


def _auto_roi_delta(run: Mapping[str, Any], actual_result: str) -> float:
    effective_action, effective_stake_pct = _resolved_effective_action(run)
    if effective_action == "观望" or effective_stake_pct <= 0:
        return 0.0

    stake_units = effective_stake_pct / 100.0
    recommended_outcome = str(_row_field(run, "recommended_outcome", "") or "")
    if recommended_outcome == actual_result:
        odds = _market_odds_for_outcome(run, recommended_outcome)
        if odds <= 0:
            return 0.0
        return round(stake_units * (odds - 1.0), 4)
    return round(-stake_units, 4)


def _auto_handicap_roi_delta(run: Mapping[str, Any], actual_score: str) -> tuple[str, int, float]:
    action = str(_row_field(run, "handicap_recommendation", "") or "")
    side = str(_row_field(run, "handicap_recommended_side", "") or "")
    line = safe_float(_row_field(run, "handicap_line"))
    actual_handicap_result = _settle_handicap_result(actual_score, line)
    if action == "观望" or side not in {"home", "away"} or actual_handicap_result not in {"home", "away", "push"}:
        return actual_handicap_result, 0, 0.0
    stake_pct = safe_float(_row_field(run, "suggested_stake_pct"))
    if stake_pct <= 0:
        effective_action, stake_pct = _resolved_effective_action(run)
        if effective_action == "观望":
            stake_pct = 0.0
    stake_units = stake_pct / 100.0
    if actual_handicap_result == "push":
        return actual_handicap_result, 0, 0.0
    hit = 1 if side == actual_handicap_result else 0
    odds = safe_float(_row_field(run, f"handicap_{side}_odds"))
    odds_decimal = odds + 1.0 if odds > 0 and odds < 1.5 else odds
    if hit and odds_decimal > 0:
        return actual_handicap_result, hit, round(stake_units * (odds_decimal - 1.0), 4)
    return actual_handicap_result, hit, round(-stake_units, 4)


def resolve_manual_review(
    run_id: int,
    effective_recommendation: str,
    notes: str = "",
) -> dict[str, Any]:
    init_db()
    pending_run = get_prediction_run(run_id)
    if pending_run is None:
        raise RuntimeError(f"未找到 run_id={run_id} 的预测记录")

    match_id = str(_row_field(pending_run, "match_id", "") or "")
    now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    expire_pending_manual_reviews(match_id=match_id, now_text=now_text)
    pending_run = get_prediction_run(run_id)
    if pending_run is None:
        raise RuntimeError(f"未找到 run_id={run_id} 的预测记录")

    latest_runs = list_prediction_runs(match_id, limit=1) if match_id else []
    latest_run_id = int(_row_field(latest_runs[0], "run_id", 0) or 0) if latest_runs else 0
    if latest_run_id and latest_run_id != int(run_id):
        if str(_row_field(pending_run, "manual_review_status", "") or "") == "pending":
            supersede_pending_manual_reviews(
                match_id,
                exclude_run_id=latest_run_id,
                resolved_at=now_text,
            )
        raise RuntimeError("该人工复核任务已被当前场次最新 run 覆盖")

    manual_status = str(_row_field(pending_run, "manual_review_status", "") or "")
    if manual_status != "pending":
        status_messages = {
            "resolved": "该人工复核任务已处理完成",
            "expired": "该人工复核任务已过期，开赛后不再允许处理",
            "superseded": "该人工复核任务已被当前场次最新 run 覆盖",
        }
        raise RuntimeError(status_messages.get(manual_status, "当前 run 不处于待人工复核状态"))

    raw_action = str(effective_recommendation or "").strip()
    if not raw_action:
        raise RuntimeError("请选择人工复核动作")
    action = _action_label(raw_action)
    if action not in ACTION_LEVELS:
        raise RuntimeError("人工复核动作只能是主推 / 轻仓 / 观望")

    risk_context = _risk_context_from_run(pending_run)
    stake_pct = 0.0 if action == "观望" else _stake_for_action(action, risk_context)
    update_prediction_run_fields(
        run_id,
        {
            "effective_recommendation": action,
            "effective_stake_pct": stake_pct,
            "effective_action_source": "manual_review",
            "manual_review_status": "resolved",
            "manual_review_resolved_at": now_text,
            "manual_review_notes": str(notes or "").strip(),
        },
    )

    updated_run = get_prediction_run(run_id)
    match_row = get_match_analysis(match_id)
    match_label = (
        f"{_row_field(match_row, 'home_team', '')} vs {_row_field(match_row, 'away_team', '')}"
        if match_row is not None
        else match_id
    )
    status_message = (
        f"已完成人工复核：{match_label} / run #{run_id}；"
        f"执行动作调整为 {action}"
    )
    return {
        "run_id": int(run_id),
        "match_id": match_id,
        "effective_recommendation": action,
        "effective_stake_pct": stake_pct,
        "effective_action_source": "manual_review",
        "manual_review_status": "resolved",
        "manual_review_notes": str(notes or "").strip(),
        "execution_status": (
            _execution_status_from_row(updated_run)
            if updated_run is not None
            else "manual_review_resolved"
        ),
        "execution_status_label": (
            _execution_status_label(updated_run)
            if updated_run is not None
            else EXECUTION_STATUS_LABELS.get("manual_review_resolved", "已人工处理")
        ),
        "task_message": status_message,
        "status_message": status_message,
        "status_level": "success",
    }


def record_feedback(
    prediction_run_id: int,
    match_id: str,
    actual_result: str,
    actual_score: str = "",
    roi_delta: float | None = None,
    notes: str = "",
    *,
    result_status: str = "manual_override",
    result_source_url: str = "",
    settled_at: str | None = None,
) -> dict[str, Any]:
    init_db()
    run = get_prediction_run(prediction_run_id)
    if run is None:
        raise RuntimeError(f"未找到 prediction_run_id={prediction_run_id} 的预测记录")
    if str(_row_field(run, "match_id", "")) != str(match_id):
        raise RuntimeError("prediction_run_id 与 match_id 不匹配")

    match_row = get_match_analysis(match_id)
    if match_row is None:
        raise RuntimeError(f"未找到 match_id={match_id} 的比赛记录")

    canonical_run, canonical_reason = _resolve_canonical_prediction_run(match_row)
    if canonical_run is None:
        raise RuntimeError(f"该场无法结算：{canonical_reason}")
    if int(_row_field(canonical_run, "run_id", 0) or 0) != int(prediction_run_id):
        raise RuntimeError(
            "只能结算该场赛前最后一条预测，"
            f"当前应结算 run_id={_row_field(canonical_run, 'run_id', 0)}"
        )

    normalized_result = _normalize_actual_result(actual_result)
    settled_at_text = str(settled_at or "").strip() or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    expire_pending_manual_reviews(match_id=match_id, now_text=settled_at_text)
    run = get_prediction_run(prediction_run_id) or run
    if roi_delta is None:
        resolved_roi_delta = _auto_roi_delta(run, normalized_result)
        roi_source = "auto"
    else:
        resolved_roi_delta = round(float(roi_delta), 4)
        roi_source = "manual_override"

    hit = 1 if str(_row_field(run, "recommended_outcome", "")) == normalized_result else 0
    handicap_actual_result, handicap_hit, handicap_roi_delta = _auto_handicap_roi_delta(
        run,
        str(actual_score or "").strip(),
    )
    upsert_match_results(
        [
            _build_match_result_payload(
                match_id,
                actual_result=normalized_result,
                actual_score=str(actual_score or "").strip(),
                result_status=str(result_status or "").strip() or "settled",
                result_source_url=result_source_url,
                result_synced_at=settled_at_text,
            )
        ]
    )
    feedback = {
        "prediction_run_id": prediction_run_id,
        "match_id": match_id,
        "actual_result": normalized_result,
        "actual_score": str(actual_score or "").strip(),
        "settled_at": settled_at_text,
        "hit_recommendation": hit,
        "roi_delta": resolved_roi_delta,
        "handicap_actual_result": handicap_actual_result,
        "handicap_hit": handicap_hit,
        "handicap_roi_delta": handicap_roi_delta,
        "roi_source": roi_source,
        "notes": str(notes or "").strip(),
    }
    feedback_id = save_feedback_log(feedback)
    status_message = (
        f"已保存赛后结算：{match_id} / run #{prediction_run_id}；"
        f"结果 {normalized_result}，ROI {resolved_roi_delta:.4f}。"
    )
    return {
        "feedback_id": feedback_id,
        "prediction_run_id": prediction_run_id,
        "match_id": match_id,
        "actual_result": normalized_result,
        "actual_score": str(actual_score or "").strip(),
        "roi_delta": resolved_roi_delta,
        "roi_source": roi_source,
        "summary": get_feedback_summary(),
        "task_message": status_message,
        "status_message": status_message,
        "status_level": "success",
    }


def _build_skip_entry(match_row: Mapping[str, Any], issue_text: str, reason: str) -> dict[str, Any]:
    return {
        "match_id": str(_row_field(match_row, "match_id", "") or ""),
        "match_label": f"{_row_field(match_row, 'home_team', '')} vs {_row_field(match_row, 'away_team', '')}",
        "issue": issue_text,
        "reason": reason,
    }


def _fallback_result_from_match_row(match_row: Mapping[str, Any]) -> dict[str, Any] | None:
    shuju_url = str(_row_field(match_row, "shuju_url", "") or "").strip()
    if not shuju_url:
        return None
    try:
        result = fetch_result_from_match_url(shuju_url)
    except Exception:  # noqa: BLE001
        return None
    if not result:
        return None
    actual_result = str(result.get("actual_result", "") or "").strip()
    actual_score = str(result.get("actual_score", "") or "").strip()
    if actual_result not in {"home", "draw", "away"} or not actual_score:
        return None
    payload = dict(result)
    payload["match_id"] = str(_row_field(match_row, "match_id", "") or "")
    return payload


def _settle_match_row(
    match_row: Mapping[str, Any],
    *,
    issue_text: str,
    result_map: Mapping[str, Mapping[str, Any]],
    progress_callback=None,
    total_items: int = 1,
    current_item_index: int = 1,
) -> dict[str, Any]:
    match_id = str(_row_field(match_row, "match_id", "") or "")
    match_label = f"{_row_field(match_row, 'home_team', '')} vs {_row_field(match_row, 'away_team', '')}"
    _emit_progress(
        progress_callback,
        total_items=total_items,
        completed_items=max(current_item_index - 1, 0),
        current_item_index=current_item_index,
        current_item_label=match_label,
        current_step="同步当前场次赛果",
        message=f"正在结算�?{current_item_index}/{total_items} 场：{match_label}",
    )

    settled_at_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result_synced_count = 0
    result_entry = result_map.get(match_id)
    manual_override_active = (
        str(_row_field(match_row, "result_status", "") or "") == "manual_override"
        and str(_row_field(match_row, "actual_result", "") or "").strip() in {"home", "draw", "away"}
    )
    if manual_override_active:
        effective_result = {
            "actual_result": str(_row_field(match_row, "actual_result", "")).strip(),
            "actual_score": str(_row_field(match_row, "actual_score", "")).strip(),
            "result_status": "manual_override",
            "result_source_url": str(_row_field(match_row, "result_source_url", "")).strip(),
        }
        if result_entry is not None and effective_result["actual_result"] == str(result_entry["actual_result"]).strip():
            manual_result_incomplete = not effective_result["actual_score"] or not effective_result["result_source_url"]
            auto_score = str(result_entry["actual_score"]).strip()
            auto_source_url = str(result_entry["result_source_url"]).strip()
            supplemented_score = effective_result["actual_score"] or auto_score
            supplemented_source_url = effective_result["result_source_url"] or auto_source_url
            auto_fully_confirmed = (
                bool(auto_score)
                and bool(auto_source_url)
                and supplemented_score == auto_score
                and supplemented_source_url == auto_source_url
            )
            updated_result_status = "settled" if auto_fully_confirmed else effective_result["result_status"]
            if (
                supplemented_score != effective_result["actual_score"]
                or supplemented_source_url != effective_result["result_source_url"]
                or updated_result_status != effective_result["result_status"]
            ):
                effective_result["actual_score"] = supplemented_score
                effective_result["result_source_url"] = supplemented_source_url
                effective_result["result_status"] = updated_result_status
                upsert_match_results(
                    [
                        _build_match_result_payload(
                            match_id,
                            actual_result=effective_result["actual_result"],
                            actual_score=effective_result["actual_score"],
                            result_status=effective_result["result_status"],
                            result_source_url=effective_result["result_source_url"],
                            result_synced_at=settled_at_text,
                        )
                    ]
                )
                result_synced_count = 1
    else:
        if result_entry is None:
            result_entry = _fallback_result_from_match_row(match_row)
            if result_entry is None:
                skip_entry = _build_skip_entry(match_row, issue_text, "未命中完场赛果")
                _emit_progress(
                    progress_callback,
                    total_items=total_items,
                    completed_items=current_item_index,
                    current_item_index=current_item_index,
                    current_item_label=match_label,
                    current_step="当前场次跳过",
                    message=f"第 {current_item_index}/{total_items} 场跳过：{match_label}，未命中完场赛果",
                    level="warning",
                )
                return {
                    "issue": issue_text,
                    "match_id": match_id,
                    "match_label": match_label,
                    "result_synced_count": 0,
                    "settled_count": 0,
                    "skipped_count": 1,
                    "skipped_matches": [skip_entry],
                }

        effective_result = {
            "actual_result": str(result_entry["actual_result"]).strip(),
            "actual_score": str(result_entry["actual_score"]).strip(),
            "result_status": "settled",
            "result_source_url": str(result_entry["result_source_url"]).strip(),
        }
        upsert_match_results(
            [
                _build_match_result_payload(
                    match_id,
                    actual_result=effective_result["actual_result"],
                    actual_score=effective_result["actual_score"],
                    result_status=effective_result["result_status"],
                    result_source_url=effective_result["result_source_url"],
                    result_synced_at=settled_at_text,
                )
            ]
        )
        result_synced_count = 1

    canonical_run, canonical_reason = _resolve_canonical_prediction_run(match_row)
    if canonical_run is None:
        skip_entry = _build_skip_entry(match_row, issue_text, canonical_reason)
        _emit_progress(
            progress_callback,
            total_items=total_items,
            completed_items=current_item_index,
            current_item_index=current_item_index,
            current_item_label=match_label,
            current_step="当前场次跳过",
            message=f"�?{current_item_index}/{total_items} 场跳过：{match_label}，{canonical_reason}",
            level="warning",
        )
        return {
            "issue": issue_text,
            "match_id": match_id,
            "match_label": match_label,
            "result_synced_count": result_synced_count,
            "settled_count": 0,
            "skipped_count": 1,
            "skipped_matches": [skip_entry],
        }

    run_id = int(_row_field(canonical_run, "run_id", 0) or 0)
    existing_feedback = get_feedback_log(run_id)
    manual_roi_delta = None
    if str(_row_field(existing_feedback, "roi_source", "") or "") == "manual_override":
        manual_roi_delta = safe_float(_row_field(existing_feedback, "roi_delta"))
    record_feedback(
        prediction_run_id=run_id,
        match_id=match_id,
        actual_result=effective_result["actual_result"],
        actual_score=effective_result["actual_score"],
        roi_delta=manual_roi_delta,
        notes=str(_row_field(existing_feedback, "notes", "") or ""),
        result_status=effective_result["result_status"],
        result_source_url=effective_result["result_source_url"],
        settled_at=settled_at_text,
    )
    _emit_progress(
        progress_callback,
        total_items=total_items,
        completed_items=current_item_index,
        current_item_index=current_item_index,
        current_item_label=match_label,
        current_step="当前场次完成",
        message=f"已完成第 {current_item_index}/{total_items} 场结算：{match_label}",
        level="info",
    )
    return {
        "issue": issue_text,
        "match_id": match_id,
        "match_label": match_label,
        "result_synced_count": result_synced_count,
        "settled_count": 1,
        "skipped_count": 0,
        "skipped_matches": [],
    }


def settle_match_result(
    match_id: str,
    progress_callback=None,
) -> dict[str, Any]:
    init_db()
    match_id_text = str(match_id or "").strip()
    if not match_id_text:
        return {
            "issue": "",
            "match_id": "",
            "total_matches": 0,
            "result_synced_count": 0,
            "settled_count": 0,
            "skipped_count": 1,
            "skipped_matches": [],
            "task_message": "缺少 match_id，无法结算当前场次。",
            "status_message": "缺少 match_id，无法结算当前场次。",
            "status_level": "warning",
        }

    match_row = get_match_analysis(match_id_text)
    if match_row is None:
        return {
            "issue": "",
            "match_id": match_id_text,
            "total_matches": 0,
            "result_synced_count": 0,
            "settled_count": 0,
            "skipped_count": 1,
            "skipped_matches": [],
            "task_message": f"未找到 match_id={match_id_text} 的比赛，无法结算。",
            "status_message": f"未找到 match_id={match_id_text} 的比赛，无法结算。",
            "status_level": "warning",
        }

    issue_text = str(_row_field(match_row, "issue", "") or "").strip()
    match_label = f"{_row_field(match_row, 'home_team', '')} vs {_row_field(match_row, 'away_team', '')}"
    if not issue_text:
        skip_entry = _build_skip_entry(match_row, issue_text, "比赛缺少期号")
        return {
            "issue": "",
            "match_id": match_id_text,
            "total_matches": 1,
            "result_synced_count": 0,
            "settled_count": 0,
            "skipped_count": 1,
            "skipped_matches": [skip_entry],
            "task_message": f"当前场次未完成结算：{match_label}，比赛缺少期号。",
            "status_message": f"当前场次未完成结算：{match_label}，比赛缺少期号。",
            "status_level": "warning",
        }

    _emit_progress(
        progress_callback,
        current_step="准备同步赛果",
        total_items=1,
        completed_items=0,
        current_item_index=0,
        current_item_label=match_label,
        message=f"准备同步并结算当前场次：{match_label}",
    )
    result_entries = fetch_issue_results(issue_text)
    result_map = {str(entry["match_id"]): entry for entry in result_entries}
    outcome = _settle_match_row(
        match_row,
        issue_text=issue_text,
        result_map=result_map,
        progress_callback=progress_callback,
        total_items=1,
        current_item_index=1,
    )
    skipped_count = int(outcome["skipped_count"])
    settled_count = int(outcome["settled_count"])
    result_synced_count = int(outcome["result_synced_count"])
    if settled_count:
        task_message = (
            f"当前场次赛果同步与结算完成：{match_label}；"
            f"同步赛果 {result_synced_count} 场，结算反馈 {settled_count} 场。"
        )
        status_level = "success"
    else:
        reason = outcome["skipped_matches"][0]["reason"] if outcome["skipped_matches"] else "未完成结算"
        task_message = f"当前场次未完成结算：{match_label}，{reason}。"
        status_level = "warning"
    _emit_progress(
        progress_callback,
        total_items=1,
        completed_items=1,
        current_item_index=1,
        current_item_label=match_label,
        current_step="赛果同步与结算完成",
        message=task_message,
        level=status_level,
    )
    return {
        "issue": issue_text,
        "match_id": match_id_text,
        "total_matches": 1,
        "result_synced_count": result_synced_count,
        "settled_count": settled_count,
        "skipped_count": skipped_count,
        "skipped_matches": outcome["skipped_matches"],
        "task_message": task_message,
        "status_message": task_message,
        "status_level": status_level,
    }


def settle_issue_results(
    issue: str | None = None,
    progress_callback=None,
    match_ids: list[str] | None = None,
) -> dict[str, Any]:
    init_db()
    issue_text = str(issue or "").strip() or get_latest_issue()
    rows = list_matches_pending_settlement(issue_text or None)
    selected_match_ids = {str(match_id).strip() for match_id in (match_ids or []) if str(match_id).strip()}
    if selected_match_ids:
        rows = [row for row in rows if str(row["match_id"]).strip() in selected_match_ids]
    if not rows:
        return {
            "issue": issue_text,
            "total_matches": 0,
            "result_synced_count": 0,
            "settled_count": 0,
            "skipped_count": 0,
            "skipped_matches": [],
            "task_message": "当前没有可结算的比赛。",
            "status_message": "当前没有可结算的比赛。",
            "status_level": "warning",
        }

    issue_text = issue_text or str(_row_field(rows[0], "issue", "")).strip()
    scoped_rows = [row for row in rows if str(_row_field(row, "issue", "")).strip() == issue_text]
    if not scoped_rows:
        return {
            "issue": issue_text,
            "total_matches": 0,
            "result_synced_count": 0,
            "settled_count": 0,
            "skipped_count": 0,
            "skipped_matches": [],
            "task_message": f"期号 {issue_text or '-'} 没有可结算的比赛。",
            "status_message": f"期号 {issue_text or '-'} 没有可结算的比赛。",
            "status_level": "warning",
        }

    result_entries = fetch_issue_results(issue_text)
    result_map = {str(entry["match_id"]): entry for entry in result_entries}
    total_matches = len(scoped_rows)
    result_synced_count = 0
    settled_count = 0
    skipped_matches: list[dict[str, Any]] = []

    _emit_progress(
        progress_callback,
        current_step="准备同步赛果",
        total_items=total_matches,
        completed_items=0,
        current_item_index=0,
        message=f"准备同步并结算期号 {issue_text} 的 {total_matches} 场比赛。",
    )

    for index, row in enumerate(scoped_rows, start=1):
        outcome = _settle_match_row(
            row,
            issue_text=issue_text,
            result_map=result_map,
            progress_callback=progress_callback,
            total_items=total_matches,
            current_item_index=index,
        )
        result_synced_count += int(outcome["result_synced_count"])
        settled_count += int(outcome["settled_count"])
        skipped_matches.extend(outcome["skipped_matches"])

    skipped_count = len(skipped_matches)
    try:
        from collection_repository import compute_issue_top_picks

        compute_issue_top_picks(issue_text)
    except Exception:
        pass
    task_message = (
        f"赛果同步与结算完成：期号 {issue_text} 共 {total_matches} 场，"
        f"同步赛果 {result_synced_count} 场，结算反馈 {settled_count} 场，跳过 {skipped_count} 场。"
    )
    status_level = "success" if skipped_count == 0 else "warning"
    _emit_progress(
        progress_callback,
        total_items=total_matches,
        completed_items=total_matches,
        current_item_index=total_matches,
        current_item_label=(f"{scoped_rows[-1]['home_team']} vs {scoped_rows[-1]['away_team']}" if scoped_rows else ""),
        current_step="赛果同步与结算完成",
        message=task_message,
        level=status_level,
    )
    return {
        "issue": issue_text,
        "total_matches": total_matches,
        "result_synced_count": result_synced_count,
        "settled_count": settled_count,
        "skipped_count": skipped_count,
        "skipped_matches": skipped_matches,
        "task_message": task_message,
        "status_message": task_message,
        "status_level": status_level,
    }


def _brier_score(probabilities: Mapping[str, float], actual_result: str) -> float:
    actual = probability_vector_for_outcome(probabilities, actual_result)
    predicted = (
        safe_float(probabilities["home"]),
        safe_float(probabilities["draw"]),
        safe_float(probabilities["away"]),
    )
    return sum((predicted[idx] - actual[idx]) ** 2 for idx in range(3)) / 3.0


def _log_loss(probabilities: Mapping[str, float], actual_result: str) -> float:
    actual_index = {"home": 0, "draw": 1, "away": 2}.get(actual_result, 0)
    probs = [
        safe_float(probabilities["home"]),
        safe_float(probabilities["draw"]),
        safe_float(probabilities["away"]),
    ]
    return -math.log(max(probs[actual_index], 1e-6))


def _hit_rate(probabilities: Mapping[str, float], actual_result: str) -> float:
    predicted = max(probabilities, key=probabilities.get)
    return 1.0 if predicted == actual_result else 0.0


def _aggregate_prob_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    count = max(len(items), 1)
    return {
        "brier_score": sum(item["brier"] for item in items) / count,
        "log_loss": sum(item["log_loss"] for item in items) / count,
        "hit_rate": sum(item["hit"] for item in items) / count,
    }


def _row_value(row: Mapping[str, Any], key: str, default: Any = "") -> Any:
    try:
        value = row[key]
    except Exception:  # noqa: BLE001
        return default
    return default if value is None else value


def _feature_payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    raw_payload = _row_value(row, "feature_payload", "")
    if not raw_payload:
        return {}
    try:
        payload = json.loads(str(raw_payload))
    except (TypeError, ValueError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _payload_mapping(row: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    payload = _feature_payload(row)
    value = payload.get(key) if isinstance(payload, Mapping) else {}
    return value if isinstance(value, Mapping) else {}


def _xg_signals(row: Mapping[str, Any]) -> dict[str, float]:
    xg = _payload_mapping(row, "xg")
    home_xg = safe_float(xg.get("home_xg_per_game"))
    home_xga = safe_float(xg.get("home_xga_per_game"))
    away_xg = safe_float(xg.get("away_xg_per_game"))
    away_xga = safe_float(xg.get("away_xga_per_game"))
    coverage = safe_float(xg.get("coverage"))
    available = coverage or home_xg or home_xga or away_xg or away_xga
    return {
        "net_gap": (home_xg - home_xga) - (away_xg - away_xga),
        "attack_gap": home_xg - away_xg,
        "defense_gap": away_xga - home_xga,
        "available": 1.0 if available else 0.0,
    }


def _market_value_signals(row: Mapping[str, Any]) -> dict[str, float]:
    market_value = _payload_mapping(row, "market_value")
    payload_gap = safe_float(market_value.get("gap_eur_m"))
    payload_ratio = safe_float(market_value.get("ratio"))
    if safe_int(market_value.get("coverage")) or payload_gap or payload_ratio:
        return {
            "gap_m": payload_gap,
            "ratio": payload_ratio,
            "available": 1.0,
        }
    summary = str(_row_value(row, "market_value_summary", "") or "")
    gap_match = re.search(r"market value gap:\s*home-away\s*EUR\s*([+-]?\d+(?:\.\d+)?)m", summary, re.I)
    ratio_match = re.search(r"ratio\s+([+-]?\d+(?:\.\d+)?)x", summary, re.I)
    return {
        "gap_m": safe_float(gap_match.group(1)) if gap_match else 0.0,
        "ratio": safe_float(ratio_match.group(1)) if ratio_match else 0.0,
        "available": 1.0 if gap_match or ratio_match else 0.0,
    }


def _score_result_direction(score: Any) -> str:
    numbers = [int(item) for item in re.findall(r"\d+", str(score or ""))[:2]]
    if len(numbers) != 2:
        return ""
    if numbers[0] > numbers[1]:
        return "home"
    if numbers[0] == numbers[1]:
        return "draw"
    return "away"


def _probability_top(values: Mapping[str, float]) -> tuple[str, dict[str, float], float]:
    probabilities = {outcome: safe_float(values.get(outcome)) for outcome in OUTCOMES}
    ordered = sorted(probabilities, key=probabilities.get, reverse=True)
    margin = probabilities[ordered[0]] - probabilities[ordered[1]] if len(ordered) >= 2 else 0.0
    return ordered[0], probabilities, margin


def _strategy_existing_action(row: Mapping[str, Any]) -> tuple[str, str, float]:
    if str(_row_value(row, "effective_action_source", "") or "") == "target_batch_strategy":
        return (
            _action_label(str(_row_value(row, "algo_recommendation", "") or "")),
            str(_row_value(row, "algo_recommended_outcome", "") or ""),
            safe_float(_row_value(row, "algo_suggested_stake_pct")),
        )
    action, stake_pct = _resolved_effective_action(row)
    return action, str(_row_value(row, "recommended_outcome", "") or ""), stake_pct


def _balanced_single_pick(row: Mapping[str, Any]) -> tuple[str, str, str, float]:
    action, existing_outcome, stake_pct = _strategy_existing_action(row)
    legacy_top, legacy_probs, _legacy_margin = _probability_top(
        {
            "home": safe_float(_row_value(row, "legacy_home_prob")),
            "draw": safe_float(_row_value(row, "legacy_draw_prob")),
            "away": safe_float(_row_value(row, "legacy_away_prob")),
        }
    )
    market_top, market_probs, market_margin = _probability_top(
        {
            "home": safe_float(_row_value(row, "market_home_prob")),
            "draw": safe_float(_row_value(row, "market_draw_prob")),
            "away": safe_float(_row_value(row, "market_away_prob")),
        }
    )
    confidence = safe_float(_row_value(row, "confidence_score"))
    score_direction = _score_result_direction(_row_value(row, "predicted_score"))
    recent_gf_gap = safe_float(_row_value(row, "recent_home_gf_pg")) - safe_float(
        _row_value(row, "recent_away_gf_pg")
    )

    if action != "观望" and stake_pct > 0 and existing_outcome in OUTCOMES:
        away_needs_confirmation = existing_outcome == "away" and (
            score_direction != "away" or recent_gf_gap > -0.6
        )
        if not away_needs_confirmation:
            return existing_outcome, "core", "existing", stake_pct

    if (
        legacy_top == "away"
        and market_margin >= 0.16
        and confidence <= 0.86
        and legacy_probs["away"] >= 0.40
        and recent_gf_gap >= -0.8
    ):
        return "away", "standard", "R1_legacy_away", 1.0
    if (
        market_top == "home"
        and market_margin >= 0.16
        and market_margin <= 0.34
        and confidence <= 0.88
        and market_probs["home"] >= 0.44
        and safe_float(_row_value(row, "home_absent_count")) <= safe_float(_row_value(row, "away_absent_count"))
    ):
        return "home", "standard", "R2_market_home", 1.0
    if market_top == "away" and market_margin >= 0.10 and confidence <= 0.88 and market_probs["away"] >= 0.52:
        return "away", "core", "R3_market_away", 1.0
    if score_direction == "draw" and confidence <= 0.82:
        return "draw", "core", "R4_score_draw", 1.0
    return "", "watch", "watch", 0.0


def _balanced_selective_single_pick(row: Mapping[str, Any]) -> tuple[str, str, str, float]:
    outcome, tier, reason, stake_pct = _balanced_single_pick(row)
    if outcome != "home" or reason != "existing":
        return outcome, tier, reason, stake_pct
    home_overexposed = (
        safe_float(_row_value(row, "confidence_score")) > 0.90
        or safe_float(_row_value(row, "home_absent_count")) - safe_float(_row_value(row, "away_absent_count")) > 2
        or safe_float(_row_value(row, "h2h_edge")) < 0
    )
    if home_overexposed:
        return "", "watch", "selective_existing_home_guard", 0.0
    return outcome, tier, reason, stake_pct


BALANCED_LEAGUE_GUARD_LEAGUES = {"挪超", "欧罗巴", "意甲"}
BALANCED_REFINED_EXISTING_LOW_ODDS_LEAGUES = {"法甲"}
BALANCED_CAUTIOUS_LEAGUE_GUARD_LEAGUES = {"英冠"}
BALANCED_VARIANT_METADATA = {
    "base": ("candidate", "原始单选基线，用于对比。"),
    "league_guarded": ("candidate", "基线叠加联赛风控。"),
    "selective": ("candidate", "保留原始动作并弱化过热主胜。"),
    "selective_league_guarded": ("candidate", "精选层叠加联赛风控。"),
    "strict": ("candidate", "对 R2 主胜追加比分与信心约束。"),
    "deep": ("candidate", "过滤赛前预测比分与方向冲突的 existing 动作。"),
    "refined": ("candidate", "过滤指定联赛的低赔 existing 风险。"),
    "hardened": ("candidate", "过滤客胜方向的历史交锋风险。"),
    "polished": ("candidate", "过滤低赔客胜叠加平局压力。"),
    "steady": ("production", "当前生产层，平衡动作面与命中稳定性。"),
    "clean": ("observation", "净动作观察层，剔除所有观察提示后的生产动作。"),
    "precise": ("observation", "精准观察层，组合平局压力、英冠样本与低赔主胜风险。"),
    "rescue": ("observation", "修正观察层，在精准过滤内救回有正面交锋与较低赔率支撑的动作。"),
    "broad": ("observation", "泛化观察层，仅使用平局概率与低赔主胜风险，降低联赛白名单过拟合。"),
    "cautious": ("observation", "研究观察层，继续收集样本后再评估是否上生产。"),
    "ultra": ("observation", "极限风控观察层，历史零错但动作面明显收窄。"),
    "coverage_push": ("observation", "覆盖推进观察层，尝试把动作占比推到60%以上并守住70%命中。"),
    "coverage_stable": ("observation", "稳定覆盖观察层，牺牲少量补动作以抬高滚动稳定性。"),
    "coverage_refined": ("observation", "精炼覆盖观察层，过滤高平局主胜后用低平局主胜补回。"),
    "coverage_value_guarded": ("observation", "身价值保护观察层，过滤身价过热补主胜并补回比分同向主胜。"),
    "coverage_xg_guarded": ("observation", "xG保护观察层，过滤xG净差严重不利的补主胜风险。"),
    "coverage_draw_rescue": ("production", "目标批量生产层，强xG过滤后按期补高分候选并救回平局信号。"),
}


def _balanced_league_guard_pick(
    row: Mapping[str, Any],
    pick_fn,
) -> tuple[str, str, str, float]:
    outcome, tier, reason, stake_pct = pick_fn(row)
    if outcome and str(_row_value(row, "league", "") or "") in BALANCED_LEAGUE_GUARD_LEAGUES:
        return "", "watch", "league_guard", 0.0
    return outcome, tier, reason, stake_pct


def _balanced_strict_single_pick(row: Mapping[str, Any]) -> tuple[str, str, str, float]:
    outcome, tier, reason, stake_pct = _balanced_league_guard_pick(row, _balanced_selective_single_pick)
    if outcome == "home" and reason == "R2_market_home":
        score_direction = _score_result_direction(_row_value(row, "predicted_score"))
        confidence = safe_float(_row_value(row, "confidence_score"))
        if (score_direction and score_direction != "home") or confidence > 0.84:
            return "", "watch", "strict_R2_home_guard", 0.0
    return outcome, tier, reason, stake_pct


def _balanced_deep_single_pick(row: Mapping[str, Any]) -> tuple[str, str, str, float]:
    outcome, tier, reason, stake_pct = _balanced_strict_single_pick(row)
    score_direction = _score_result_direction(_row_value(row, "predicted_score"))
    if outcome and reason == "existing" and score_direction and score_direction != outcome:
        return "", "watch", "deep_existing_score_conflict_guard", 0.0
    return outcome, tier, reason, stake_pct


def _balanced_refined_single_pick(row: Mapping[str, Any]) -> tuple[str, str, str, float]:
    outcome, tier, reason, stake_pct = _balanced_deep_single_pick(row)
    if (
        outcome
        and reason == "existing"
        and str(_row_value(row, "league", "") or "") in BALANCED_REFINED_EXISTING_LOW_ODDS_LEAGUES
        and _market_odds_for_outcome(row, outcome) < 1.60
    ):
        return "", "watch", "refined_existing_low_odds_league_guard", 0.0
    return outcome, tier, reason, stake_pct


def _balanced_hardened_single_pick(row: Mapping[str, Any]) -> tuple[str, str, str, float]:
    outcome, tier, reason, stake_pct = _balanced_refined_single_pick(row)
    if outcome == "away" and safe_float(_row_value(row, "h2h_edge")) <= -0.8:
        return "", "watch", "hardened_away_h2h_guard", 0.0
    return outcome, tier, reason, stake_pct


def _balanced_polished_single_pick(row: Mapping[str, Any]) -> tuple[str, str, str, float]:
    outcome, tier, reason, stake_pct = _balanced_hardened_single_pick(row)
    if (
        outcome == "away"
        and reason == "existing"
        and _market_odds_for_outcome(row, outcome) < 1.55
        and safe_float(_row_value(row, "market_draw_prob")) > 0.21
    ):
        return "", "watch", "polished_existing_away_draw_risk_guard", 0.0
    return outcome, tier, reason, stake_pct


def _balanced_steady_single_pick(row: Mapping[str, Any]) -> tuple[str, str, str, float]:
    outcome, tier, reason, stake_pct = _balanced_polished_single_pick(row)
    if (
        outcome == "home"
        and reason == "R2_market_home"
        and 1.50 <= _market_odds_for_outcome(row, outcome) < 1.80
        and safe_float(_row_value(row, "market_draw_prob")) >= 0.245
    ):
        return "", "watch", "steady_R2_home_draw_pressure_guard", 0.0
    return outcome, tier, reason, stake_pct


def _balanced_cautious_single_pick(row: Mapping[str, Any]) -> tuple[str, str, str, float]:
    outcome, tier, reason, stake_pct = _balanced_steady_single_pick(row)
    if outcome and str(_row_value(row, "league", "") or "") in BALANCED_CAUTIOUS_LEAGUE_GUARD_LEAGUES:
        return "", "watch", "cautious_league_guard", 0.0
    return outcome, tier, reason, stake_pct


def _balanced_ultra_single_pick(row: Mapping[str, Any]) -> tuple[str, str, str, float]:
    outcome, tier, reason, stake_pct = _balanced_steady_single_pick(row)
    market_home = safe_float(_row_value(row, "market_home_prob"))
    market_away = safe_float(_row_value(row, "market_away_prob"))
    legacy_away = safe_float(_row_value(row, "legacy_away_prob"))
    market_draw = safe_float(_row_value(row, "market_draw_prob"))
    h2h_edge = safe_float(_row_value(row, "h2h_edge"))
    odds = _market_odds_for_outcome(row, outcome) if outcome else 0.0
    if reason == "R4_score_draw" and (
        market_home < 0.39
        or market_away > market_home
        or legacy_away >= 0.40
    ):
        return "", "watch", "ultra_draw_away_pressure_guard", 0.0
    if (
        outcome == "home"
        and reason == "existing"
        and 1.30 <= odds < 1.45
        and market_draw >= 0.18
        and market_away >= 0.13
        and h2h_edge >= 0.0
    ):
        return "", "watch", "ultra_existing_home_midlow_draw_risk_guard", 0.0
    return outcome, tier, reason, stake_pct


def _balanced_precise_single_pick(row: Mapping[str, Any]) -> tuple[str, str, str, float]:
    outcome, tier, reason, stake_pct = _balanced_production_single_pick(row)
    if not outcome:
        return outcome, tier, reason, stake_pct

    league = str(_row_value(row, "league", "") or "")
    market_draw = safe_float(_row_value(row, "market_draw_prob"))
    market_away = safe_float(_row_value(row, "market_away_prob"))
    h2h_edge = safe_float(_row_value(row, "h2h_edge"))
    odds = _market_odds_for_outcome(row, outcome)

    if league == "英冠":
        return "", "watch", "precise_league_guard", 0.0
    if market_draw >= 0.30:
        return "", "watch", "precise_market_draw_pressure_guard", 0.0
    if (
        outcome == "home"
        and reason == "existing"
        and 1.30 <= odds < 1.45
        and market_draw >= 0.18
        and market_away >= 0.13
        and h2h_edge >= 0.0
    ):
        return "", "watch", "precise_existing_home_midlow_draw_risk_guard", 0.0
    return outcome, tier, reason, stake_pct


def _balanced_rescue_single_pick(row: Mapping[str, Any]) -> tuple[str, str, str, float]:
    precise_outcome, precise_tier, precise_reason, precise_stake_pct = _balanced_precise_single_pick(row)
    if precise_outcome:
        return precise_outcome, precise_tier, precise_reason, precise_stake_pct

    outcome, tier, reason, stake_pct = _balanced_production_single_pick(row)
    if not outcome:
        return outcome, tier, reason, stake_pct

    if (
        safe_float(_row_value(row, "market_draw_prob")) >= 0.19
        and safe_float(_row_value(row, "h2h_edge")) >= 0.0
        and _market_odds_for_outcome(row, outcome) < 3.20
    ):
        return outcome, tier, f"rescue_{precise_reason}", stake_pct
    return "", "watch", precise_reason, 0.0


def _balanced_broad_single_pick(row: Mapping[str, Any]) -> tuple[str, str, str, float]:
    outcome, tier, reason, stake_pct = _balanced_production_single_pick(row)
    if not outcome:
        return outcome, tier, reason, stake_pct

    market_draw = safe_float(_row_value(row, "market_draw_prob"))
    market_away = safe_float(_row_value(row, "market_away_prob"))
    h2h_edge = safe_float(_row_value(row, "h2h_edge"))
    odds = _market_odds_for_outcome(row, outcome)
    if (
        outcome == "home"
        and reason == "existing"
        and 1.30 <= odds < 1.45
        and market_draw >= 0.18
        and market_away >= 0.13
        and h2h_edge >= 0.0
    ):
        return "", "watch", "broad_existing_home_midlow_draw_risk_guard", 0.0
    if market_draw >= 0.29:
        return "", "watch", "broad_market_draw_pressure_guard", 0.0
    return outcome, tier, reason, stake_pct


def _balanced_clean_single_pick(row: Mapping[str, Any]) -> tuple[str, str, str, float]:
    outcome, tier, reason, stake_pct = _balanced_production_single_pick(row)
    if outcome and _balanced_preview_observation_flags(row, outcome):
        return "", "watch", "clean_observation_flag_guard", 0.0
    return outcome, tier, reason, stake_pct


def _balanced_coverage_push_single_pick(row: Mapping[str, Any]) -> tuple[str, str, str, float]:
    outcome, tier, reason, stake_pct = _balanced_single_pick(row)
    if outcome:
        return outcome, tier, reason, stake_pct

    market_top, _market_probs, _market_margin = _probability_top(
        {
            "home": safe_float(_row_value(row, "market_home_prob")),
            "draw": safe_float(_row_value(row, "market_draw_prob")),
            "away": safe_float(_row_value(row, "market_away_prob")),
        }
    )
    if market_top == "home" and 1.80 <= safe_float(_row_value(row, "market_odds_home")) < 2.20:
        return "home", "standard", "fill_market_home_odds_180_220", 1.0

    legacy_top, _legacy_probs, legacy_margin = _probability_top(
        {
            "home": safe_float(_row_value(row, "legacy_home_prob")),
            "draw": safe_float(_row_value(row, "legacy_draw_prob")),
            "away": safe_float(_row_value(row, "legacy_away_prob")),
        }
    )
    if legacy_top == "home" and 0.05 <= legacy_margin < 0.10:
        return "home", "standard", "fill_legacy_home_margin_05_10", 1.0

    return "", "watch", "coverage_push_watch", 0.0


def _balanced_coverage_stable_single_pick(row: Mapping[str, Any]) -> tuple[str, str, str, float]:
    outcome, tier, reason, stake_pct = _balanced_single_pick(row)
    if outcome:
        return outcome, tier, reason, stake_pct

    market_top, market_probs, _market_margin = _probability_top(
        {
            "home": safe_float(_row_value(row, "market_home_prob")),
            "draw": safe_float(_row_value(row, "market_draw_prob")),
            "away": safe_float(_row_value(row, "market_away_prob")),
        }
    )
    legacy_top, _legacy_probs, legacy_margin = _probability_top(
        {
            "home": safe_float(_row_value(row, "legacy_home_prob")),
            "draw": safe_float(_row_value(row, "legacy_draw_prob")),
            "away": safe_float(_row_value(row, "legacy_away_prob")),
        }
    )
    league = str(_row_value(row, "league", "") or "")
    score_direction = _score_result_direction(_row_value(row, "predicted_score"))
    confidence = safe_float(_row_value(row, "confidence_score"))
    market_draw = safe_float(_row_value(row, "market_draw_prob"))

    if (
        market_top == "home"
        and 1.80 <= safe_float(_row_value(row, "market_odds_home")) < 2.20
        and league not in {"西甲", "挪超"}
    ):
        return "home", "standard", "fill_stable_market_home_odds_180_220", 1.0
    if legacy_top == "home" and 0.05 <= legacy_margin < 0.10:
        return "home", "standard", "fill_stable_legacy_home_margin_05_10", 1.0
    if market_top == "home" and market_draw >= 0.30:
        return "home", "standard", "fill_stable_market_home_draw_ge_30", 1.0
    if score_direction in OUTCOMES and confidence <= 0.80:
        return score_direction, "standard", "fill_stable_score_conf80", 1.0
    return "", "watch", "coverage_stable_watch", 0.0


def _balanced_coverage_refined_single_pick(row: Mapping[str, Any]) -> tuple[str, str, str, float]:
    outcome, tier, reason, stake_pct = _balanced_coverage_stable_single_pick(row)
    market_draw = safe_float(_row_value(row, "market_draw_prob"))
    if outcome == "home" and market_draw >= 0.31:
        outcome = ""

    if outcome:
        return outcome, tier, reason, stake_pct

    market_top, market_probs, _market_margin = _probability_top(
        {
            "home": safe_float(_row_value(row, "market_home_prob")),
            "draw": market_draw,
            "away": safe_float(_row_value(row, "market_away_prob")),
        }
    )
    if market_top == "home" and market_probs["home"] >= 0.50 and market_draw < 0.22:
        return "home", "standard", "fill_refined_market_home_prob50_low_draw", 1.0

    return "", "watch", "coverage_refined_watch", 0.0


def _balanced_coverage_value_guarded_single_pick(row: Mapping[str, Any]) -> tuple[str, str, str, float]:
    outcome, tier, reason, stake_pct = _balanced_coverage_refined_single_pick(row)
    value_signals = _market_value_signals(row)
    if (
        outcome == "home"
        and reason.startswith("fill_")
        and value_signals["ratio"] >= 3.0
    ):
        outcome = ""

    if outcome:
        return outcome, tier, reason, stake_pct

    market_top, _market_probs, _market_margin = _probability_top(
        {
            "home": safe_float(_row_value(row, "market_home_prob")),
            "draw": safe_float(_row_value(row, "market_draw_prob")),
            "away": safe_float(_row_value(row, "market_away_prob")),
        }
    )
    score_direction = _score_result_direction(_row_value(row, "predicted_score"))
    home_odds = safe_float(_row_value(row, "market_odds_home"))
    ratio = value_signals["ratio"]
    if (
        market_top == "home"
        and score_direction == "home"
        and 1.80 <= home_odds < 2.20
        and (ratio == 0.0 or ratio >= 0.90)
    ):
        return "home", "standard", "fill_value_guarded_home_odds_180_220_score_home", 1.0

    return "", "watch", "coverage_value_guarded_watch", 0.0


def _balanced_coverage_xg_guarded_single_pick(row: Mapping[str, Any]) -> tuple[str, str, str, float]:
    outcome, tier, reason, stake_pct = _balanced_coverage_value_guarded_single_pick(row)
    xg_signals = _xg_signals(row)
    if (
        outcome == "home"
        and reason.startswith("fill_")
        and xg_signals["available"]
        and xg_signals["net_gap"] < -0.50
    ):
        return "", "watch", "xg_guarded_home_fill_net_gap_guard", 0.0
    return outcome, tier, reason, stake_pct


def _balanced_coverage_ranked_candidate(row: Mapping[str, Any]) -> tuple[str, float, str]:
    market_top, market_probs, _market_margin = _probability_top(
        {
            "home": safe_float(_row_value(row, "market_home_prob")),
            "draw": safe_float(_row_value(row, "market_draw_prob")),
            "away": safe_float(_row_value(row, "market_away_prob")),
        }
    )
    score_direction = _score_result_direction(_row_value(row, "predicted_score"))
    xg_signals = _xg_signals(row)
    value_signals = _market_value_signals(row)
    split = _payload_mapping(row, "split")
    split_ppg_gap = safe_float(split.get("home_ppg")) - safe_float(split.get("away_ppg"))
    market_draw = safe_float(_row_value(row, "market_draw_prob"))
    home_odds = safe_float(_row_value(row, "market_odds_home"))
    draw_odds = safe_float(_row_value(row, "market_odds_draw"))
    candidates: list[tuple[str, float, str]] = []

    if market_top == "home" and score_direction == "home":
        score = (
            max(0.0, market_probs["home"] - 0.44) * 10.0
            + max(0.0, 0.30 - market_draw) * 5.0
            + max(0.0, split_ppg_gap) * 0.5
            + (max(0.0, xg_signals["net_gap"]) * 0.8 if xg_signals["available"] else 0.0)
            + (0.8 if 1.50 <= home_odds < 2.30 else 0.0)
            - (1.2 if value_signals["available"] and value_signals["ratio"] >= 8.0 else 0.0)
        )
        candidates.append(("home", score, "rank_home"))
    if score_direction == "draw" and market_draw >= 0.27:
        score = (
            (market_draw - 0.27) * 8.0
            + (0.5 - min(abs(xg_signals["net_gap"]), 0.5)) * 0.5
            + (0.5 if 2.80 <= draw_odds < 3.60 else 0.0)
        )
        candidates.append(("draw", score, "rank_draw"))
    if market_top == "away" and score_direction == "away":
        score = (
            max(0.0, market_probs["away"] - 0.44) * 10.0
            + max(0.0, 0.28 - market_draw) * 4.0
            - (1.0 if value_signals["available"] and value_signals["ratio"] > 1.2 else 0.0)
        )
        candidates.append(("away", score, "rank_away"))
    return max(candidates, key=lambda item: item[1]) if candidates else ("", 0.0, "")


def _balanced_coverage_draw_rescue_action_rows(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    action_rows: list[dict[str, Any]] = []
    remaining: list[tuple[float, Mapping[str, Any], str, str]] = []
    issue_counts: dict[str, int] = {}

    for row in rows:
        outcome, tier, reason, stake_pct = _balanced_coverage_value_guarded_single_pick(row)
        xg_signals = _xg_signals(row)
        guarded = (
            outcome == "home"
            and reason.startswith("fill_")
            and xg_signals["available"]
            and xg_signals["net_gap"] < 0.25
        )
        if outcome and not guarded:
            action = _balanced_pick_action_row(row, lambda _row, result=(outcome, tier, reason, stake_pct): result)
            if action is not None:
                action_rows.append(action)
            continue

        ranked_outcome, ranked_score, ranked_reason = _balanced_coverage_ranked_candidate(row)
        if ranked_outcome and ranked_score > 0:
            remaining.append((ranked_score, row, ranked_outcome, ranked_reason))

    for ranked_score, row, outcome, reason in sorted(remaining, key=lambda item: item[0], reverse=True):
        issue = str(_row_value(row, "issue", "") or "")
        if ranked_score >= 2.0 and issue_counts.get(issue, 0) < 1:
            action = _balanced_pick_action_row(
                row,
                lambda _row, result=(outcome, "standard", f"draw_rescue_{reason}", 1.0): result,
            )
            if action is not None:
                action_rows.append(action)
                issue_counts[issue] = issue_counts.get(issue, 0) + 1

    existing_ids = {item.get("match_id") for item in action_rows}
    for ranked_score, row, outcome, reason in remaining:
        if outcome != "draw" or str(_row_value(row, "match_id", "") or "") in existing_ids:
            continue
        xg_signals = _xg_signals(row)
        split = _payload_mapping(row, "split")
        split_ppg_gap = safe_float(split.get("home_ppg")) - safe_float(split.get("away_ppg"))
        if (
            ranked_score >= 0.5
            and (not xg_signals["available"] or xg_signals["net_gap"] >= 0.5)
            and split_ppg_gap >= 0.0
        ):
            action = _balanced_pick_action_row(
                row,
                lambda _row, result=("draw", "standard", "draw_rescue_xg_split_draw", 1.0): result,
            )
            if action is not None:
                action_rows.append(action)
                existing_ids.add(action.get("match_id"))
    return action_rows


def _balanced_metrics_from_action_rows(
    rows: list[Mapping[str, Any]],
    action_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = _balanced_bucket_metrics_from_actions(action_rows)
    metrics["sample_count"] = len(rows)
    metrics["action_share"] = metrics["action_count"] / len(rows) if rows else 0.0
    action_ids = {str(item.get("match_id", "") or "") for item in action_rows}
    metrics["watch_count"] = max(len(rows) - len(action_ids), 0)
    metrics["watch_share"] = metrics["watch_count"] / len(rows) if rows else 0.0

    action_by_match_id = {str(item.get("match_id", "") or ""): item for item in action_rows}

    def _pick(row: Mapping[str, Any]) -> tuple[str, str, str, float]:
        action = action_by_match_id.get(str(_row_value(row, "match_id", "") or ""))
        if not action:
            return "", "watch", "coverage_draw_rescue_watch", 0.0
        return (
            str(action.get("outcome", "") or ""),
            str(action.get("tier", "") or "standard"),
            str(action.get("reason", "") or "coverage_draw_rescue"),
            safe_float(action.get("stake_pct")),
        )

    metrics["stability"] = _balanced_stability_metrics(rows, _pick)
    return metrics


def _balanced_target_batch_strategy_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    action_rows = _balanced_coverage_draw_rescue_action_rows(rows)
    metrics = _balanced_metrics_from_action_rows(rows, action_rows)
    stability = metrics.get("stability") if isinstance(metrics.get("stability"), Mapping) else {}
    rolling_10 = stability.get("rolling_10_issues") if isinstance(stability, Mapping) else {}
    latest_10 = stability.get("latest_10_issues") if isinstance(stability, Mapping) else {}
    target_action_share = 0.60
    target_hit_rate = 0.70
    target_rolling_10_hit_rate = 0.70
    action_count = safe_int(metrics.get("action_count"))
    hit_count = safe_int(metrics.get("hit_count"))
    sample_count = len(rows)
    required_actions = math.ceil(sample_count * target_action_share) if sample_count else 0
    target_met = (
        action_count >= required_actions
        and safe_float(metrics.get("hit_rate")) >= target_hit_rate
        and safe_float(rolling_10.get("min_hit_rate") if isinstance(rolling_10, Mapping) else 0.0)
        >= target_rolling_10_hit_rate
    )
    issue_buckets = _balanced_group_metrics(
        [
            {
                **item,
                "odds_band": _balanced_odds_band(safe_float(item.get("market_odds"))),
                "confidence_band": _balanced_confidence_band(safe_float(item.get("confidence_score"))),
                "score_alignment": _balanced_score_alignment(
                    str(item.get("outcome", "") or ""),
                    str(item.get("score_direction", "") or ""),
                ),
            }
            for item in action_rows
        ],
        "issue",
    )
    return {
        "key": "coverage_draw_rescue",
        "label": "平局救援目标批量策略",
        "role": "production",
        "note": BALANCED_VARIANT_METADATA["coverage_draw_rescue"][1],
        "sample_count": sample_count,
        "required_actions": required_actions,
        "action_count": action_count,
        "hit_count": hit_count,
        "miss_count": safe_int(metrics.get("miss_count")),
        "watch_count": safe_int(metrics.get("watch_count")),
        "action_share": safe_float(metrics.get("action_share")),
        "watch_share": safe_float(metrics.get("watch_share")),
        "hit_rate": safe_float(metrics.get("hit_rate")),
        "total_roi": safe_float(metrics.get("total_roi")),
        "roi_on_stake": safe_float(metrics.get("roi_on_stake")),
        "latest_10_action_count": safe_int(latest_10.get("action_count") if isinstance(latest_10, Mapping) else 0),
        "latest_10_hit_rate": safe_float(latest_10.get("hit_rate") if isinstance(latest_10, Mapping) else 0.0),
        "rolling_10_min_hit_rate": safe_float(
            rolling_10.get("min_hit_rate") if isinstance(rolling_10, Mapping) else 0.0
        ),
        "target_action_share": target_action_share,
        "target_hit_rate": target_hit_rate,
        "target_rolling_10_hit_rate": target_rolling_10_hit_rate,
        "target_met": target_met,
        "status": "met" if target_met else "gap",
        "issue_buckets": issue_buckets,
        "reason_breakdown": _balanced_action_breakdown(action_rows, "reason", limit=8),
        "outcome_breakdown": _balanced_action_breakdown(action_rows, "outcome", limit=4),
        "recent_actions": _balanced_action_examples(action_rows, limit=12),
        "misses": _balanced_action_examples([item for item in action_rows if not int(item.get("hit", 0))], limit=12),
    }


def _balanced_single_pick_backtest_metrics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = _balanced_pick_backtest_metrics(rows, _balanced_single_pick)
    metrics["league_guarded"] = _balanced_pick_backtest_metrics(
        rows,
        lambda row: _balanced_league_guard_pick(row, _balanced_single_pick),
    )
    metrics["selective"] = _balanced_pick_backtest_metrics(rows, _balanced_selective_single_pick)
    metrics["selective_league_guarded"] = _balanced_pick_backtest_metrics(
        rows,
        lambda row: _balanced_league_guard_pick(row, _balanced_selective_single_pick),
    )
    metrics["strict"] = _balanced_pick_backtest_metrics(rows, _balanced_strict_single_pick)
    metrics["deep"] = _balanced_pick_backtest_metrics(rows, _balanced_deep_single_pick)
    metrics["refined"] = _balanced_pick_backtest_metrics(rows, _balanced_refined_single_pick)
    metrics["hardened"] = _balanced_pick_backtest_metrics(rows, _balanced_hardened_single_pick)
    metrics["polished"] = _balanced_pick_backtest_metrics(rows, _balanced_polished_single_pick)
    metrics["steady"] = _balanced_pick_backtest_metrics(rows, _balanced_steady_single_pick)
    metrics["clean"] = _balanced_pick_backtest_metrics(rows, _balanced_clean_single_pick)
    metrics["precise"] = _balanced_pick_backtest_metrics(rows, _balanced_precise_single_pick)
    metrics["rescue"] = _balanced_pick_backtest_metrics(rows, _balanced_rescue_single_pick)
    metrics["broad"] = _balanced_pick_backtest_metrics(rows, _balanced_broad_single_pick)
    metrics["cautious"] = _balanced_pick_backtest_metrics(rows, _balanced_cautious_single_pick)
    metrics["ultra"] = _balanced_pick_backtest_metrics(rows, _balanced_ultra_single_pick)
    metrics["coverage_push"] = _balanced_pick_backtest_metrics(rows, _balanced_coverage_push_single_pick)
    metrics["coverage_stable"] = _balanced_pick_backtest_metrics(rows, _balanced_coverage_stable_single_pick)
    metrics["coverage_refined"] = _balanced_pick_backtest_metrics(rows, _balanced_coverage_refined_single_pick)
    metrics["coverage_value_guarded"] = _balanced_pick_backtest_metrics(
        rows,
        _balanced_coverage_value_guarded_single_pick,
    )
    metrics["coverage_xg_guarded"] = _balanced_pick_backtest_metrics(
        rows,
        _balanced_coverage_xg_guarded_single_pick,
    )
    coverage_draw_rescue_actions = _balanced_coverage_draw_rescue_action_rows(rows)
    metrics["coverage_draw_rescue"] = _balanced_metrics_from_action_rows(rows, coverage_draw_rescue_actions)
    variants = {
        "base": _balanced_variant_summary("base", "基础单选", metrics),
        "league_guarded": _balanced_variant_summary("league_guarded", "联赛风控单选", metrics["league_guarded"]),
        "selective": _balanced_variant_summary("selective", "精选单选", metrics["selective"]),
        "selective_league_guarded": _balanced_variant_summary(
            "selective_league_guarded",
            "精选+联赛风控单选",
            metrics["selective_league_guarded"],
        ),
        "strict": _balanced_variant_summary("strict", "严选单选", metrics["strict"]),
        "deep": _balanced_variant_summary("deep", "深挖单选", metrics["deep"]),
        "refined": _balanced_variant_summary("refined", "精修单选", metrics["refined"]),
        "hardened": _balanced_variant_summary("hardened", "加固单选", metrics["hardened"]),
        "polished": _balanced_variant_summary("polished", "打磨单选", metrics["polished"]),
        "steady": _balanced_variant_summary("steady", "稳健单选", metrics["steady"]),
        "clean": _balanced_variant_summary("clean", "净动作观察单选", metrics["clean"]),
        "precise": _balanced_variant_summary("precise", "精准观察单选", metrics["precise"]),
        "rescue": _balanced_variant_summary("rescue", "修正观察单选", metrics["rescue"]),
        "broad": _balanced_variant_summary("broad", "泛化观察单选", metrics["broad"]),
        "cautious": _balanced_variant_summary("cautious", "谨慎单选", metrics["cautious"]),
        "ultra": _balanced_variant_summary("ultra", "极慎观察单选", metrics["ultra"]),
        "coverage_push": _balanced_variant_summary("coverage_push", "覆盖推进观察单选", metrics["coverage_push"]),
        "coverage_stable": _balanced_variant_summary("coverage_stable", "稳定覆盖观察单选", metrics["coverage_stable"]),
        "coverage_refined": _balanced_variant_summary("coverage_refined", "精炼覆盖观察单选", metrics["coverage_refined"]),
        "coverage_value_guarded": _balanced_variant_summary(
            "coverage_value_guarded",
            "身价值保护观察单选",
            metrics["coverage_value_guarded"],
        ),
        "coverage_xg_guarded": _balanced_variant_summary(
            "coverage_xg_guarded",
            "xG保护观察单选",
            metrics["coverage_xg_guarded"],
        ),
        "coverage_draw_rescue": _balanced_variant_summary(
            "coverage_draw_rescue",
            "平局救援目标批量策略",
            metrics["coverage_draw_rescue"],
        ),
    }
    metrics["variants"] = variants
    metrics["recommended_variant"] = _balanced_recommended_variant(variants)
    coverage_draw_rescue_variant = dict(variants.get("coverage_draw_rescue", {}))
    metrics["production_variant"] = coverage_draw_rescue_variant or _balanced_recommended_variant(
        variants,
        exclude_keys={
            "clean",
            "precise",
            "rescue",
            "broad",
            "cautious",
            "ultra",
            "coverage_push",
            "coverage_stable",
            "coverage_refined",
            "coverage_value_guarded",
            "coverage_xg_guarded",
        },
    )
    observation_deltas = {
        "clean": _balanced_observation_delta(
            rows,
            _balanced_production_single_pick,
            _balanced_clean_single_pick,
            "净动作观察单选",
        ),
        "precise": _balanced_observation_delta(
            rows,
            _balanced_production_single_pick,
            _balanced_precise_single_pick,
            "精准观察单选",
        ),
        "rescue": _balanced_observation_delta(
            rows,
            _balanced_production_single_pick,
            _balanced_rescue_single_pick,
            "修正观察单选",
        ),
        "broad": _balanced_observation_delta(
            rows,
            _balanced_production_single_pick,
            _balanced_broad_single_pick,
            "泛化观察单选",
        ),
        "cautious": _balanced_observation_delta(
            rows,
            _balanced_production_single_pick,
            _balanced_cautious_single_pick,
            "谨慎单选",
        ),
        "ultra": _balanced_observation_delta(
            rows,
            _balanced_production_single_pick,
            _balanced_ultra_single_pick,
            "极慎观察单选",
        ),
        "coverage_value_guarded": _balanced_observation_delta(
            rows,
            _balanced_production_single_pick,
            _balanced_coverage_value_guarded_single_pick,
            "身价值保护观察单选",
        ),
        "coverage_xg_guarded": _balanced_observation_delta(
            rows,
            _balanced_production_single_pick,
            _balanced_coverage_xg_guarded_single_pick,
            "xG保护观察单选",
        ),
        "coverage_draw_rescue": {
            "label": "平局救援目标批量策略",
            "filtered_count": 0,
            "filtered_miss_count": 0,
            "filtered_hit_rate": 0.0,
            "miss_capture_rate": 0.0,
            "hit_filter_rate": 0.0,
            "filtered_roi": 0.0,
            "filtered_reason_counts": {},
            "filtered_reason_breakdown": [],
            "filtered_issue_count": 0,
            "filtered_latest_issue": "",
            "filtered_issue_breakdown": [],
            "max_issue_filtered_share": 0.0,
            "max_issue_miss_share": 0.0,
            "filtered_examples": [],
        },
    }
    metrics["observation_deltas"] = observation_deltas
    metrics["observation_transitions"] = {
        "precise_to_rescue": _balanced_observation_transition(
            rows,
            _balanced_precise_single_pick,
            _balanced_rescue_single_pick,
            "精准到修正救回",
        ),
    }
    metrics["observation_periods"] = _balanced_observation_periods(
        rows,
        {
            "rescue": ("修正观察单选", _balanced_rescue_single_pick),
            "broad": ("泛化观察单选", _balanced_broad_single_pick),
        },
    )
    metrics["observation_risk_backtest"] = _balanced_observation_risk_backtest(rows)
    metrics["observation_combo_backtest"] = _balanced_observation_combo_backtest(rows)
    metrics["observation_combo_scenarios"] = _balanced_observation_combo_scenarios(rows)
    metrics["observation_feature_profiles"] = _balanced_observation_feature_profiles(rows)
    metrics["coverage_target_diagnostics"] = _balanced_coverage_target_diagnostics(rows)
    metrics["coverage_stability_diagnostics"] = _balanced_coverage_stability_diagnostics(rows)
    metrics["observation_readiness"] = _balanced_observation_readiness(variants, observation_deltas)
    return metrics


def _balanced_pick_action_row(row: Mapping[str, Any], pick_fn) -> dict[str, Any] | None:
    outcome, tier, reason, stake_pct = pick_fn(row)
    if not outcome:
        return None
    actual_result = str(_row_value(row, "actual_result", "") or "")
    hit = 1 if outcome == actual_result else 0
    market_odds = _market_odds_for_outcome(row, outcome)
    stake_units = stake_pct / 100.0
    roi_delta = stake_units * (market_odds - 1.0) if hit and market_odds > 0 else -stake_units
    return {
        "match_id": str(_row_value(row, "match_id", "") or ""),
        "issue": str(_row_value(row, "issue", "") or ""),
        "league": str(_row_value(row, "league", "") or ""),
        "home_team": str(_row_value(row, "home_team", "") or ""),
        "away_team": str(_row_value(row, "away_team", "") or ""),
        "outcome": outcome,
        "tier": tier,
        "reason": reason,
        "stake_pct": stake_pct,
        "actual_result": actual_result,
        "predicted_score": str(_row_value(row, "predicted_score", "") or ""),
        "hit": hit,
        "market_odds": market_odds,
        "market_odds_home": safe_float(_row_value(row, "market_odds_home")),
        "market_odds_draw": safe_float(_row_value(row, "market_odds_draw")),
        "market_odds_away": safe_float(_row_value(row, "market_odds_away")),
        "market_home_prob": safe_float(_row_value(row, "market_home_prob")),
        "market_draw_prob": safe_float(_row_value(row, "market_draw_prob")),
        "market_away_prob": safe_float(_row_value(row, "market_away_prob")),
        "confidence_score": safe_float(_row_value(row, "confidence_score")),
        "quality_score": safe_float(_row_value(row, "quality_score")),
        "score_direction": _score_result_direction(_row_value(row, "predicted_score")),
        "roi_delta": roi_delta,
    }


def _balanced_observation_delta(
    rows: list[Mapping[str, Any]],
    production_pick_fn,
    observation_pick_fn,
    label: str,
) -> dict[str, Any]:
    production_rows: list[dict[str, Any]] = []
    filtered_rows: list[dict[str, Any]] = []
    for row in rows:
        production_action = _balanced_pick_action_row(row, production_pick_fn)
        if production_action is None:
            continue
        production_rows.append(production_action)
        observation_outcome, _, observation_reason, _ = observation_pick_fn(row)
        if observation_outcome:
            continue
        production_action["observation_reason"] = observation_reason
        filtered_rows.append(production_action)
    filtered_count = len(filtered_rows)
    filtered_hits = sum(int(item["hit"]) for item in filtered_rows)
    production_hit_count = sum(int(item["hit"]) for item in production_rows)
    production_miss_count = max(len(production_rows) - production_hit_count, 0)
    filtered_miss_count = max(filtered_count - filtered_hits, 0)
    filtered_roi = sum(safe_float(item["roi_delta"]) for item in filtered_rows)
    filtered_stake = sum(safe_float(item["stake_pct"]) / 100.0 for item in filtered_rows)
    filtered_reason_counts: dict[str, int] = {}
    for item in filtered_rows:
        reason = str(item.get("observation_reason", "") or "unknown")
        filtered_reason_counts[reason] = filtered_reason_counts.get(reason, 0) + 1
    filtered_reason_breakdown: dict[str, dict[str, Any]] = {}
    for reason in filtered_reason_counts:
        reason_rows = [
            item
            for item in filtered_rows
            if str(item.get("observation_reason", "") or "unknown") == reason
        ]
        reason_count = len(reason_rows)
        reason_hits = sum(int(item["hit"]) for item in reason_rows)
        reason_roi = sum(safe_float(item["roi_delta"]) for item in reason_rows)
        reason_stake = sum(safe_float(item["stake_pct"]) / 100.0 for item in reason_rows)
        reason_issues = sorted(
            {str(item.get("issue", "") or "") for item in reason_rows if str(item.get("issue", "") or "")},
            key=_issue_sort_key,
        )
        filtered_reason_breakdown[reason] = {
            "count": reason_count,
            "hit_count": reason_hits,
            "miss_count": max(reason_count - reason_hits, 0),
            "hit_rate": reason_hits / reason_count if reason_count else 0.0,
            "roi": reason_roi,
            "roi_on_stake": reason_roi / reason_stake if reason_stake > 0 else 0.0,
            "issue_count": len(reason_issues),
            "latest_issue": reason_issues[-1] if reason_issues else "",
        }
    filtered_issues = sorted(
        {str(item.get("issue", "") or "") for item in filtered_rows if str(item.get("issue", "") or "")},
        key=_issue_sort_key,
    )
    filtered_issue_breakdown: dict[str, dict[str, Any]] = {}
    for issue in filtered_issues:
        issue_rows = [
            item
            for item in filtered_rows
            if str(item.get("issue", "") or "") == issue
        ]
        issue_count = len(issue_rows)
        issue_hits = sum(int(item["hit"]) for item in issue_rows)
        filtered_issue_breakdown[issue] = {
            "count": issue_count,
            "hit_count": issue_hits,
            "miss_count": max(issue_count - issue_hits, 0),
        }
    max_issue_filtered = max(
        (safe_int(item.get("count")) for item in filtered_issue_breakdown.values()),
        default=0,
    )
    max_issue_misses = max(
        (safe_int(item.get("miss_count")) for item in filtered_issue_breakdown.values()),
        default=0,
    )
    return {
        "label": label,
        "filtered_count": filtered_count,
        "filtered_hit_count": filtered_hits,
        "filtered_miss_count": filtered_miss_count,
        "filtered_hit_rate": filtered_hits / filtered_count if filtered_count else 0.0,
        "production_hit_count": production_hit_count,
        "production_miss_count": production_miss_count,
        "miss_capture_rate": filtered_miss_count / production_miss_count if production_miss_count else 0.0,
        "hit_filter_rate": filtered_hits / production_hit_count if production_hit_count else 0.0,
        "filtered_roi": filtered_roi,
        "filtered_roi_on_stake": filtered_roi / filtered_stake if filtered_stake > 0 else 0.0,
        "filtered_reason_counts": filtered_reason_counts,
        "filtered_reason_breakdown": filtered_reason_breakdown,
        "filtered_issue_count": len(filtered_issues),
        "filtered_latest_issue": filtered_issues[-1] if filtered_issues else "",
        "filtered_issue_breakdown": filtered_issue_breakdown,
        "max_issue_filtered_share": max_issue_filtered / filtered_count if filtered_count else 0.0,
        "max_issue_miss_share": max_issue_misses / filtered_miss_count if filtered_miss_count else 0.0,
        "filtered_examples": _balanced_action_examples(filtered_rows, limit=12),
    }


def _balanced_observation_transition(
    rows: list[Mapping[str, Any]],
    from_pick_fn,
    to_pick_fn,
    label: str,
) -> dict[str, Any]:
    restored_rows: list[dict[str, Any]] = []
    for row in rows:
        from_outcome, _, _, _ = from_pick_fn(row)
        if from_outcome:
            continue
        restored_action = _balanced_pick_action_row(row, to_pick_fn)
        if restored_action is None:
            continue
        restored_rows.append(restored_action)
    restored_count = len(restored_rows)
    restored_hits = sum(int(item["hit"]) for item in restored_rows)
    restored_roi = sum(safe_float(item["roi_delta"]) for item in restored_rows)
    restored_stake = sum(safe_float(item["stake_pct"]) / 100.0 for item in restored_rows)
    restored_issues = sorted(
        {str(item.get("issue", "") or "") for item in restored_rows if str(item.get("issue", "") or "")},
        key=_issue_sort_key,
    )
    return {
        "label": label,
        "restored_count": restored_count,
        "restored_hit_count": restored_hits,
        "restored_miss_count": max(restored_count - restored_hits, 0),
        "restored_hit_rate": restored_hits / restored_count if restored_count else 0.0,
        "restored_roi": restored_roi,
        "restored_roi_on_stake": restored_roi / restored_stake if restored_stake > 0 else 0.0,
        "restored_issue_count": len(restored_issues),
        "restored_latest_issue": restored_issues[-1] if restored_issues else "",
        "restored_examples": _balanced_action_examples(restored_rows, limit=12),
    }


def _balanced_variant_summary(key: str, label: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    stability = metrics.get("stability") if isinstance(metrics.get("stability"), Mapping) else {}
    rolling_10 = stability.get("rolling_10_issues") if isinstance(stability, Mapping) else {}
    latest_10 = stability.get("latest_10_issues") if isinstance(stability, Mapping) else {}
    role, note = BALANCED_VARIANT_METADATA.get(key, ("candidate", ""))
    return {
        "key": key,
        "label": label,
        "role": role,
        "note": note,
        "sample_count": safe_int(metrics.get("sample_count")),
        "action_count": safe_int(metrics.get("action_count")),
        "action_share": safe_float(metrics.get("action_share")),
        "hit_rate": safe_float(metrics.get("hit_rate")),
        "total_roi": safe_float(metrics.get("total_roi")),
        "roi_on_stake": safe_float(metrics.get("roi_on_stake")),
        "latest_10_action_count": safe_int(latest_10.get("action_count") if isinstance(latest_10, Mapping) else 0),
        "latest_10_hit_rate": safe_float(latest_10.get("hit_rate") if isinstance(latest_10, Mapping) else 0.0),
        "latest_10_roi_on_stake": safe_float(
            latest_10.get("roi_on_stake") if isinstance(latest_10, Mapping) else 0.0
        ),
        "rolling_10_min_hit_rate": safe_float(
            rolling_10.get("min_hit_rate") if isinstance(rolling_10, Mapping) else 0.0
        ),
        "rolling_10_min_roi_on_stake": safe_float(
            rolling_10.get("min_roi_on_stake") if isinstance(rolling_10, Mapping) else 0.0
        ),
    }


def _balanced_observation_periods(
    rows: list[Mapping[str, Any]],
    observation_specs: Mapping[str, tuple[str, Any]],
) -> dict[str, Any]:
    issue_keys = sorted(
        {str(_row_value(row, "issue", "") or "") for row in rows if str(_row_value(row, "issue", "") or "")},
        key=_issue_sort_key,
    )
    if not issue_keys:
        return {}
    chunk_size = max((len(issue_keys) + 2) // 3, 1)
    chunks = [
        ("early", "早期", issue_keys[:chunk_size]),
        ("middle", "中期", issue_keys[chunk_size : chunk_size * 2]),
        ("recent", "近期", issue_keys[chunk_size * 2 :]),
    ]
    periods: dict[str, Any] = {}
    for period_key, period_label, period_issues in chunks:
        if not period_issues:
            continue
        period_issue_set = set(period_issues)
        period_rows = [row for row in rows if str(_row_value(row, "issue", "") or "") in period_issue_set]
        production_actions = [
            action
            for row in period_rows
            if (action := _balanced_pick_action_row(row, _balanced_production_single_pick)) is not None
        ]
        production_count = len(production_actions)
        production_hits = sum(int(item["hit"]) for item in production_actions)
        period_summary: dict[str, Any] = {
            "label": period_label,
            "issue_from": period_issues[0],
            "issue_to": period_issues[-1],
            "issue_count": len(period_issues),
            "production_action_count": production_count,
            "production_hit_rate": production_hits / production_count if production_count else 0.0,
            "layers": {},
        }
        for key, (label, pick_fn) in observation_specs.items():
            observation_actions: list[dict[str, Any]] = []
            filtered_actions: list[dict[str, Any]] = []
            for row in period_rows:
                production_action = _balanced_pick_action_row(row, _balanced_production_single_pick)
                if production_action is None:
                    continue
                observation_action = _balanced_pick_action_row(row, pick_fn)
                if observation_action is None:
                    filtered_actions.append(production_action)
                    continue
                observation_actions.append(observation_action)
            action_count = len(observation_actions)
            filtered_count = len(filtered_actions)
            filtered_hits = sum(int(item["hit"]) for item in filtered_actions)
            period_summary["layers"][key] = {
                "label": label,
                "action_count": action_count,
                "hit_rate": (
                    sum(int(item["hit"]) for item in observation_actions) / action_count
                    if action_count
                    else 0.0
                ),
                "filtered_count": filtered_count,
                "filtered_miss_count": max(filtered_count - filtered_hits, 0),
                "filtered_hit_count": filtered_hits,
            }
        periods[period_key] = period_summary
    return periods


def _balanced_observation_risk_backtest(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, dict[str, Any]] = {
        "clean": {
            "label": "无提示",
            "action_count": 0,
            "hit_count": 0,
            "total_roi": 0.0,
            "total_stake": 0.0,
            "rows": [],
        },
        "single": {
            "label": "单层提示",
            "action_count": 0,
            "hit_count": 0,
            "total_roi": 0.0,
            "total_stake": 0.0,
            "rows": [],
        },
        "stacked": {
            "label": "双层提示",
            "action_count": 0,
            "hit_count": 0,
            "total_roi": 0.0,
            "total_stake": 0.0,
            "rows": [],
        },
        "resonance": {
            "label": "多层共振",
            "action_count": 0,
            "hit_count": 0,
            "total_roi": 0.0,
            "total_stake": 0.0,
            "rows": [],
        },
    }
    for row in rows:
        action = _balanced_pick_action_row(row, _balanced_production_single_pick)
        if action is None:
            continue
        flags = _balanced_preview_observation_flags(row, str(action.get("outcome", "") or ""))
        risk_level, risk_label = _balanced_observation_risk(flags)
        bucket = buckets.setdefault(
            risk_level,
            {
                "label": risk_label,
                "action_count": 0,
                "hit_count": 0,
                "total_roi": 0.0,
                "total_stake": 0.0,
                "rows": [],
            },
        )
        bucket["label"] = risk_label
        bucket["action_count"] = safe_int(bucket.get("action_count")) + 1
        bucket["hit_count"] = safe_int(bucket.get("hit_count")) + int(action["hit"])
        bucket["total_roi"] = safe_float(bucket.get("total_roi")) + safe_float(action["roi_delta"])
        bucket["total_stake"] = safe_float(bucket.get("total_stake")) + safe_float(action["stake_pct"]) / 100.0
        bucket_rows = bucket.get("rows")
        if isinstance(bucket_rows, list):
            bucket_rows.append(action)
    summary: dict[str, Any] = {}
    for key, bucket in buckets.items():
        action_count = safe_int(bucket.get("action_count"))
        hit_count = safe_int(bucket.get("hit_count"))
        total_roi = safe_float(bucket.get("total_roi"))
        total_stake = safe_float(bucket.get("total_stake"))
        summary[key] = {
            "label": bucket.get("label", key),
            "action_count": action_count,
            "hit_count": hit_count,
            "miss_count": max(action_count - hit_count, 0),
            "hit_rate": hit_count / action_count if action_count else 0.0,
            "total_roi": total_roi,
            "roi_on_stake": total_roi / total_stake if total_stake > 0 else 0.0,
            "recent_actions": _balanced_action_examples(bucket.get("rows", []), limit=6)
            if isinstance(bucket.get("rows"), list)
            else [],
            "misses": _balanced_action_examples(
                [item for item in bucket.get("rows", []) if not int(item.get("hit", 0))],
                limit=6,
            )
            if isinstance(bucket.get("rows"), list)
            else [],
        }
    return summary


def _balanced_observation_combo_backtest(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for row in rows:
        action = _balanced_pick_action_row(row, _balanced_production_single_pick)
        if action is None:
            continue
        flags = _balanced_preview_observation_flags(row, str(action.get("outcome", "") or ""))
        flag_keys = [str(flag.get("key", "") or "") for flag in flags if str(flag.get("key", "") or "")]
        flag_labels = [str(flag.get("label", "") or "") for flag in flags if str(flag.get("label", "") or "")]
        combo_key = "+".join(flag_keys) if flag_keys else "clean"
        combo_label = " + ".join(flag_labels) if flag_labels else "无提示"
        bucket = buckets.setdefault(
            combo_key,
            {
                "key": combo_key,
                "label": combo_label,
                "flag_count": len(flag_keys),
                "action_count": 0,
                "hit_count": 0,
                "total_roi": 0.0,
                "total_stake": 0.0,
                "rows": [],
            },
        )
        bucket["label"] = combo_label
        bucket["flag_count"] = len(flag_keys)
        bucket["action_count"] = safe_int(bucket.get("action_count")) + 1
        bucket["hit_count"] = safe_int(bucket.get("hit_count")) + int(action["hit"])
        bucket["total_roi"] = safe_float(bucket.get("total_roi")) + safe_float(action["roi_delta"])
        bucket["total_stake"] = safe_float(bucket.get("total_stake")) + safe_float(action["stake_pct"]) / 100.0
        bucket_rows = bucket.get("rows")
        if isinstance(bucket_rows, list):
            bucket_rows.append(action)

    summaries: list[dict[str, Any]] = []
    for bucket in buckets.values():
        action_count = safe_int(bucket.get("action_count"))
        hit_count = safe_int(bucket.get("hit_count"))
        total_roi = safe_float(bucket.get("total_roi"))
        total_stake = safe_float(bucket.get("total_stake"))
        bucket_rows = bucket.get("rows", [])
        action_rows = bucket_rows if isinstance(bucket_rows, list) else []
        summaries.append(
            {
                "key": str(bucket.get("key", "") or ""),
                "label": str(bucket.get("label", "") or ""),
                "flag_count": safe_int(bucket.get("flag_count")),
                "action_count": action_count,
                "hit_count": hit_count,
                "miss_count": max(action_count - hit_count, 0),
                "hit_rate": hit_count / action_count if action_count else 0.0,
                "total_roi": total_roi,
                "roi_on_stake": total_roi / total_stake if total_stake > 0 else 0.0,
                "recent_actions": _balanced_action_examples(action_rows, limit=6),
                "misses": _balanced_action_examples(
                    [item for item in action_rows if not int(item.get("hit", 0))],
                    limit=6,
                ),
                "reason_breakdown": _balanced_action_breakdown(action_rows, "reason"),
                "outcome_breakdown": _balanced_action_breakdown(action_rows, "outcome"),
                "league_breakdown": _balanced_action_breakdown(action_rows, "league"),
            }
        )
    return sorted(
        summaries,
        key=lambda item: (
            0 if item.get("key") == "clean" else 1,
            -safe_int(item.get("flag_count")),
            -safe_int(item.get("action_count")),
            str(item.get("key", "")),
        ),
    )


def _balanced_action_breakdown(action_rows: list[dict[str, Any]], field: str, *, limit: int = 6) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for action in action_rows:
        key = str(action.get(field, "") or "unknown")
        bucket = buckets.setdefault(
            key,
            {
                "key": key,
                "action_count": 0,
                "hit_count": 0,
                "total_roi": 0.0,
                "total_stake": 0.0,
                "rows": [],
            },
        )
        bucket["action_count"] = safe_int(bucket.get("action_count")) + 1
        bucket["hit_count"] = safe_int(bucket.get("hit_count")) + int(action.get("hit", 0))
        bucket["total_roi"] = safe_float(bucket.get("total_roi")) + safe_float(action.get("roi_delta"))
        bucket["total_stake"] = safe_float(bucket.get("total_stake")) + safe_float(action.get("stake_pct")) / 100.0
        bucket_rows = bucket.get("rows")
        if isinstance(bucket_rows, list):
            bucket_rows.append(action)

    summaries: list[dict[str, Any]] = []
    for bucket in buckets.values():
        action_count = safe_int(bucket.get("action_count"))
        hit_count = safe_int(bucket.get("hit_count"))
        total_roi = safe_float(bucket.get("total_roi"))
        total_stake = safe_float(bucket.get("total_stake"))
        bucket_rows = bucket.get("rows", [])
        action_items = bucket_rows if isinstance(bucket_rows, list) else []
        summaries.append(
            {
                "key": str(bucket.get("key", "") or ""),
                "action_count": action_count,
                "hit_count": hit_count,
                "miss_count": max(action_count - hit_count, 0),
                "hit_rate": hit_count / action_count if action_count else 0.0,
                "total_roi": total_roi,
                "roi_on_stake": total_roi / total_stake if total_stake > 0 else 0.0,
                "misses": _balanced_action_examples(
                    [item for item in action_items if not int(item.get("hit", 0))],
                    limit=3,
                ),
            }
        )
    return sorted(
        summaries,
        key=lambda item: (
            -safe_int(item.get("miss_count")),
            -safe_int(item.get("action_count")),
            str(item.get("key", "")),
        ),
    )[:limit]


def _balanced_observation_combo_scenarios(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    production_actions: list[dict[str, Any]] = []
    filtered_by_scenario: dict[str, list[dict[str, Any]]] = {
        "rescue_high_resonance": [],
        "rescue_high_resonance_r4_draw": [],
        "rescue_high_resonance_existing_home": [],
    }
    scenario_labels = {
        "rescue_high_resonance": "过滤含 rescue 的三层以上共振",
        "rescue_high_resonance_r4_draw": "过滤 rescue 高共振里的 R4 平局",
        "rescue_high_resonance_existing_home": "过滤 rescue 高共振里的 existing 主胜",
    }
    for row in rows:
        action = _balanced_pick_action_row(row, _balanced_production_single_pick)
        if action is None:
            continue
        production_actions.append(action)
        flags = _balanced_preview_observation_flags(row, str(action.get("outcome", "") or ""))
        flag_keys = {str(flag.get("key", "") or "") for flag in flags}
        if "rescue" in flag_keys and len(flag_keys) >= 3:
            filtered_by_scenario["rescue_high_resonance"].append(action)
            if str(action.get("reason", "") or "") == "R4_score_draw" and str(action.get("outcome", "") or "") == "draw":
                filtered_by_scenario["rescue_high_resonance_r4_draw"].append(action)
            if str(action.get("reason", "") or "") == "existing" and str(action.get("outcome", "") or "") == "home":
                filtered_by_scenario["rescue_high_resonance_existing_home"].append(action)

    production_count = len(production_actions)
    production_hit_count = sum(int(item.get("hit", 0)) for item in production_actions)
    production_roi = sum(safe_float(item.get("roi_delta")) for item in production_actions)
    production_stake = sum(safe_float(item.get("stake_pct")) / 100.0 for item in production_actions)
    scenarios: list[dict[str, Any]] = []
    for key, filtered_rows in filtered_by_scenario.items():
        filtered_ids = {
            (str(item.get("issue", "") or ""), str(item.get("match_id", "") or ""))
            for item in filtered_rows
        }
        kept_rows = [
            item
            for item in production_actions
            if (str(item.get("issue", "") or ""), str(item.get("match_id", "") or "")) not in filtered_ids
        ]
        kept_count = len(kept_rows)
        kept_hit_count = sum(int(item.get("hit", 0)) for item in kept_rows)
        kept_roi = sum(safe_float(item.get("roi_delta")) for item in kept_rows)
        kept_stake = sum(safe_float(item.get("stake_pct")) / 100.0 for item in kept_rows)
        filtered_count = len(filtered_rows)
        filtered_hit_count = sum(int(item.get("hit", 0)) for item in filtered_rows)
        filtered_issues = sorted(
            {str(item.get("issue", "") or "") for item in filtered_rows if str(item.get("issue", "") or "")},
            key=_issue_sort_key,
        )
        filtered_issue_counts: dict[str, int] = {}
        for item in filtered_rows:
            issue = str(item.get("issue", "") or "")
            if issue:
                filtered_issue_counts[issue] = filtered_issue_counts.get(issue, 0) + 1
        max_issue_filtered = max(filtered_issue_counts.values(), default=0)
        scenarios.append(
            {
                "key": key,
                "label": scenario_labels.get(key, key),
                "production_action_count": production_count,
                "production_hit_count": production_hit_count,
                "production_miss_count": max(production_count - production_hit_count, 0),
                "production_hit_rate": production_hit_count / production_count if production_count else 0.0,
                "production_roi": production_roi,
                "production_roi_on_stake": production_roi / production_stake if production_stake > 0 else 0.0,
                "kept_action_count": kept_count,
                "kept_hit_count": kept_hit_count,
                "kept_miss_count": max(kept_count - kept_hit_count, 0),
                "kept_hit_rate": kept_hit_count / kept_count if kept_count else 0.0,
                "kept_roi": kept_roi,
                "kept_roi_on_stake": kept_roi / kept_stake if kept_stake > 0 else 0.0,
                "filtered_count": filtered_count,
                "filtered_hit_count": filtered_hit_count,
                "filtered_miss_count": max(filtered_count - filtered_hit_count, 0),
                "filtered_issue_count": len(filtered_issues),
                "filtered_latest_issue": filtered_issues[-1] if filtered_issues else "",
                "max_issue_filtered_share": max_issue_filtered / filtered_count if filtered_count else 0.0,
                "filtered_examples": _balanced_action_examples(filtered_rows, limit=6),
            }
        )
    return scenarios


def _balanced_numeric_profile(action_rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [safe_float(item.get(field)) for item in action_rows if safe_float(item.get(field)) > 0]
    if not values:
        return {"count": 0, "avg": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "avg": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def _balanced_observation_feature_profiles(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {
        "rescue_high_resonance": {
            "label": "含 rescue 的三层以上共振",
            "rows": [],
        },
        "other_flagged": {
            "label": "其它观察提示动作",
            "rows": [],
        },
        "clean": {
            "label": "无观察提示动作",
            "rows": [],
        },
    }
    for row in rows:
        action = _balanced_pick_action_row(row, _balanced_production_single_pick)
        if action is None:
            continue
        flags = _balanced_preview_observation_flags(row, str(action.get("outcome", "") or ""))
        flag_keys = {str(flag.get("key", "") or "") for flag in flags}
        if "rescue" in flag_keys and len(flag_keys) >= 3:
            key = "rescue_high_resonance"
        elif flags:
            key = "other_flagged"
        else:
            key = "clean"
        bucket_rows = buckets[key].get("rows")
        if isinstance(bucket_rows, list):
            bucket_rows.append(action)

    profiles: list[dict[str, Any]] = []
    for key, bucket in buckets.items():
        action_rows = bucket.get("rows", [])
        action_items = action_rows if isinstance(action_rows, list) else []
        action_count = len(action_items)
        hit_count = sum(int(item.get("hit", 0)) for item in action_items)
        total_roi = sum(safe_float(item.get("roi_delta")) for item in action_items)
        total_stake = sum(safe_float(item.get("stake_pct")) / 100.0 for item in action_items)
        profiles.append(
            {
                "key": key,
                "label": str(bucket.get("label", key) or key),
                "action_count": action_count,
                "hit_count": hit_count,
                "miss_count": max(action_count - hit_count, 0),
                "hit_rate": hit_count / action_count if action_count else 0.0,
                "roi_on_stake": total_roi / total_stake if total_stake > 0 else 0.0,
                "market_odds": _balanced_numeric_profile(action_items, "market_odds"),
                "market_home_prob": _balanced_numeric_profile(action_items, "market_home_prob"),
                "market_draw_prob": _balanced_numeric_profile(action_items, "market_draw_prob"),
                "market_away_prob": _balanced_numeric_profile(action_items, "market_away_prob"),
                "confidence_score": _balanced_numeric_profile(action_items, "confidence_score"),
                "quality_score": _balanced_numeric_profile(action_items, "quality_score"),
                "reason_breakdown": _balanced_action_breakdown(action_items, "reason", limit=4),
                "outcome_breakdown": _balanced_action_breakdown(action_items, "outcome", limit=4),
                "score_direction_breakdown": _balanced_action_breakdown(action_items, "score_direction", limit=4),
                "league_breakdown": _balanced_action_breakdown(action_items, "league", limit=4),
                "misses": _balanced_action_examples(
                    [item for item in action_items if not int(item.get("hit", 0))],
                    limit=6,
                ),
            }
        )
    return profiles


def _balanced_target_bucket_label(key: tuple[str, ...]) -> str:
    labels = {
        "legacy_home_margin_05_10": "观望补动作：legacy 主胜边际 5%-10%",
        "score_low_confidence": "观望补动作：比分方向且信心 <=82%",
        "market_home_prob_50": "观望补动作：市场主胜概率 >=50%",
        "market_home_odds_180_220": "观望补动作：市场主胜赔率 1.80-2.20",
    }
    return labels.get("|".join(key), "|".join(key))


def _balanced_bucket_metrics_from_actions(actions: list[dict[str, Any]]) -> dict[str, Any]:
    action_count = len(actions)
    hit_count = sum(int(item.get("hit", 0)) for item in actions)
    total_roi = sum(safe_float(item.get("roi_delta")) for item in actions)
    total_stake = sum(safe_float(item.get("stake_pct")) / 100.0 for item in actions)
    return {
        "action_count": action_count,
        "hit_count": hit_count,
        "miss_count": max(action_count - hit_count, 0),
        "hit_rate": hit_count / action_count if action_count else 0.0,
        "total_roi": total_roi,
        "roi_on_stake": total_roi / total_stake if total_stake > 0 else 0.0,
        "examples": _balanced_action_examples(actions, limit=6),
        "misses": _balanced_action_examples([item for item in actions if not int(item.get("hit", 0))], limit=6),
    }


def _balanced_coverage_target_diagnostics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    target_action_share = 0.60
    target_hit_rate = 0.70
    sample_count = len(rows)
    required_actions = math.ceil(sample_count * target_action_share) if sample_count else 0
    base_actions: list[dict[str, Any]] = []
    production_actions: list[dict[str, Any]] = []
    coverage_push_actions: list[dict[str, Any]] = []
    coverage_stable_actions: list[dict[str, Any]] = []
    coverage_refined_actions: list[dict[str, Any]] = []
    coverage_value_guarded_actions: list[dict[str, Any]] = []
    coverage_xg_guarded_actions: list[dict[str, Any]] = []
    coverage_draw_rescue_actions = _balanced_coverage_draw_rescue_action_rows(rows)
    watch_rows: list[Mapping[str, Any]] = []
    for row in rows:
        base_action = _balanced_pick_action_row(row, _balanced_single_pick)
        if base_action is None:
            watch_rows.append(row)
        else:
            base_actions.append(base_action)
        production_action = _balanced_pick_action_row(row, _balanced_production_single_pick)
        if production_action is not None:
            production_actions.append(production_action)
        coverage_push_action = _balanced_pick_action_row(row, _balanced_coverage_push_single_pick)
        if coverage_push_action is not None:
            coverage_push_actions.append(coverage_push_action)
        coverage_stable_action = _balanced_pick_action_row(row, _balanced_coverage_stable_single_pick)
        if coverage_stable_action is not None:
            coverage_stable_actions.append(coverage_stable_action)
        coverage_refined_action = _balanced_pick_action_row(row, _balanced_coverage_refined_single_pick)
        if coverage_refined_action is not None:
            coverage_refined_actions.append(coverage_refined_action)
        coverage_value_guarded_action = _balanced_pick_action_row(row, _balanced_coverage_value_guarded_single_pick)
        if coverage_value_guarded_action is not None:
            coverage_value_guarded_actions.append(coverage_value_guarded_action)
        coverage_xg_guarded_action = _balanced_pick_action_row(row, _balanced_coverage_xg_guarded_single_pick)
        if coverage_xg_guarded_action is not None:
            coverage_xg_guarded_actions.append(coverage_xg_guarded_action)

    watch_buckets: dict[tuple[str, ...], list[dict[str, Any]]] = {
        ("legacy_home_margin_05_10",): [],
        ("score_low_confidence",): [],
        ("market_home_prob_50",): [],
        ("market_home_odds_180_220",): [],
    }
    for row in watch_rows:
        legacy_top, _legacy_probs, legacy_margin = _probability_top(
            {
                "home": safe_float(_row_value(row, "legacy_home_prob")),
                "draw": safe_float(_row_value(row, "legacy_draw_prob")),
                "away": safe_float(_row_value(row, "legacy_away_prob")),
            }
        )
        market_top, market_probs, _market_margin = _probability_top(
            {
                "home": safe_float(_row_value(row, "market_home_prob")),
                "draw": safe_float(_row_value(row, "market_draw_prob")),
                "away": safe_float(_row_value(row, "market_away_prob")),
            }
        )
        score_direction = _score_result_direction(_row_value(row, "predicted_score"))
        confidence = safe_float(_row_value(row, "confidence_score"))
        if legacy_top == "home" and 0.05 <= legacy_margin < 0.10:
            action = _balanced_pick_action_row(row, lambda _row: ("home", "standard", "fill_legacy_home_margin_05_10", 1.0))
            if action is not None:
                watch_buckets[("legacy_home_margin_05_10",)].append(action)
        if score_direction in OUTCOMES and confidence <= 0.82:
            action = _balanced_pick_action_row(row, lambda _row, outcome=score_direction: (outcome, "standard", "fill_score_low_confidence", 1.0))
            if action is not None:
                watch_buckets[("score_low_confidence",)].append(action)
        if market_top == "home" and market_probs["home"] >= 0.50:
            action = _balanced_pick_action_row(row, lambda _row: ("home", "standard", "fill_market_home_prob_50", 1.0))
            if action is not None:
                watch_buckets[("market_home_prob_50",)].append(action)
        if market_top == "home" and 1.80 <= safe_float(_row_value(row, "market_odds_home")) < 2.20:
            action = _balanced_pick_action_row(row, lambda _row: ("home", "standard", "fill_market_home_odds_180_220", 1.0))
            if action is not None:
                watch_buckets[("market_home_odds_180_220",)].append(action)

    bucket_summaries: list[dict[str, Any]] = []
    for key, actions in watch_buckets.items():
        metrics = _balanced_bucket_metrics_from_actions(actions)
        metrics["key"] = "|".join(key)
        metrics["label"] = _balanced_target_bucket_label(key)
        metrics["action_share"] = metrics["action_count"] / sample_count if sample_count else 0.0
        bucket_summaries.append(metrics)
    bucket_summaries.sort(
        key=lambda item: (
            safe_float(item.get("hit_rate")) >= target_hit_rate,
            safe_int(item.get("action_count")),
            safe_float(item.get("hit_rate")),
        ),
        reverse=True,
    )

    base_metrics = _balanced_bucket_metrics_from_actions(base_actions)
    production_metrics = _balanced_bucket_metrics_from_actions(production_actions)
    coverage_push_metrics = _balanced_bucket_metrics_from_actions(coverage_push_actions)
    coverage_stable_metrics = _balanced_bucket_metrics_from_actions(coverage_stable_actions)
    coverage_refined_metrics = _balanced_bucket_metrics_from_actions(coverage_refined_actions)
    coverage_value_guarded_metrics = _balanced_bucket_metrics_from_actions(coverage_value_guarded_actions)
    coverage_xg_guarded_metrics = _balanced_bucket_metrics_from_actions(coverage_xg_guarded_actions)
    coverage_draw_rescue_metrics = _balanced_bucket_metrics_from_actions(coverage_draw_rescue_actions)
    best_additional = [item for item in bucket_summaries if safe_float(item.get("hit_rate")) >= target_hit_rate]
    return {
        "target_action_share": target_action_share,
        "target_hit_rate": target_hit_rate,
        "sample_count": sample_count,
        "required_actions": required_actions,
        "base": {
            **base_metrics,
            "action_share": base_metrics["action_count"] / sample_count if sample_count else 0.0,
            "additional_needed": max(required_actions - base_metrics["action_count"], 0),
            "target_met": base_metrics["action_count"] >= required_actions and base_metrics["hit_rate"] >= target_hit_rate,
        },
        "production": {
            **production_metrics,
            "action_share": production_metrics["action_count"] / sample_count if sample_count else 0.0,
            "additional_needed": max(required_actions - production_metrics["action_count"], 0),
            "target_met": production_metrics["action_count"] >= required_actions
            and production_metrics["hit_rate"] >= target_hit_rate,
        },
        "coverage_push": {
            **coverage_push_metrics,
            "action_share": coverage_push_metrics["action_count"] / sample_count if sample_count else 0.0,
            "additional_needed": max(required_actions - coverage_push_metrics["action_count"], 0),
            "target_met": coverage_push_metrics["action_count"] >= required_actions
            and coverage_push_metrics["hit_rate"] >= target_hit_rate,
        },
        "coverage_stable": {
            **coverage_stable_metrics,
            "action_share": coverage_stable_metrics["action_count"] / sample_count if sample_count else 0.0,
            "additional_needed": max(required_actions - coverage_stable_metrics["action_count"], 0),
            "target_met": coverage_stable_metrics["action_count"] >= required_actions
            and coverage_stable_metrics["hit_rate"] >= target_hit_rate,
        },
        "coverage_refined": {
            **coverage_refined_metrics,
            "action_share": coverage_refined_metrics["action_count"] / sample_count if sample_count else 0.0,
            "additional_needed": max(required_actions - coverage_refined_metrics["action_count"], 0),
            "target_met": coverage_refined_metrics["action_count"] >= required_actions
            and coverage_refined_metrics["hit_rate"] >= target_hit_rate,
        },
        "coverage_value_guarded": {
            **coverage_value_guarded_metrics,
            "action_share": coverage_value_guarded_metrics["action_count"] / sample_count if sample_count else 0.0,
            "additional_needed": max(required_actions - coverage_value_guarded_metrics["action_count"], 0),
            "target_met": coverage_value_guarded_metrics["action_count"] >= required_actions
            and coverage_value_guarded_metrics["hit_rate"] >= target_hit_rate,
        },
        "coverage_xg_guarded": {
            **coverage_xg_guarded_metrics,
            "action_share": coverage_xg_guarded_metrics["action_count"] / sample_count if sample_count else 0.0,
            "additional_needed": max(required_actions - coverage_xg_guarded_metrics["action_count"], 0),
            "target_met": coverage_xg_guarded_metrics["action_count"] >= required_actions
            and coverage_xg_guarded_metrics["hit_rate"] >= target_hit_rate,
        },
        "coverage_draw_rescue": {
            **coverage_draw_rescue_metrics,
            "action_share": len(coverage_draw_rescue_actions) / sample_count if sample_count else 0.0,
            "additional_needed": max(required_actions - len(coverage_draw_rescue_actions), 0),
            "target_met": len(coverage_draw_rescue_actions) >= required_actions
            and coverage_draw_rescue_metrics["hit_rate"] >= target_hit_rate,
        },
        "watch_count": len(watch_rows),
        "watch_bucket_candidates": bucket_summaries,
        "best_additional_bucket": best_additional[0] if best_additional else {},
        "status": "met" if base_metrics["action_count"] >= required_actions and base_metrics["hit_rate"] >= target_hit_rate else "gap",
    }


def _balanced_coverage_stability_diagnostics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    action_rows = [
        action
        for row in rows
        if (action := _balanced_pick_action_row(row, _balanced_coverage_stable_single_pick)) is not None
    ]
    issue_keys = sorted(
        {str(_row_value(row, "issue", "") or "") for row in rows if str(_row_value(row, "issue", "") or "")},
        key=_issue_sort_key,
    )
    low_window: dict[str, Any] = {}
    for index in range(max(len(issue_keys) - 9, 0)):
        window_issues = issue_keys[index : index + 10]
        issue_set = set(window_issues)
        window_actions = [item for item in action_rows if str(item.get("issue", "") or "") in issue_set]
        if not window_actions:
            continue
        action_count = len(window_actions)
        hit_count = sum(int(item.get("hit", 0)) for item in window_actions)
        hit_rate = hit_count / action_count if action_count else 0.0
        if not low_window or hit_rate < safe_float(low_window.get("hit_rate")):
            low_window = {
                "issue_from": window_issues[0],
                "issue_to": window_issues[-1],
                "action_count": action_count,
                "hit_count": hit_count,
                "miss_count": max(action_count - hit_count, 0),
                "hit_rate": hit_rate,
                "misses": _balanced_action_examples(
                    [item for item in window_actions if not int(item.get("hit", 0))],
                    limit=12,
                ),
                "reason_breakdown": _balanced_action_breakdown(window_actions, "reason", limit=8),
                "league_breakdown": _balanced_action_breakdown(window_actions, "league", limit=8),
                "outcome_breakdown": _balanced_action_breakdown(window_actions, "outcome", limit=4),
            }

    return {
        "label": "稳定覆盖观察层低谷诊断",
        "low_window": low_window,
        "tested_filters": [
            {
                "label": "过滤高平局主胜且比分冲突",
                "action_share": 0.5886,
                "hit_rate": 0.7176,
                "rolling_10_min_hit_rate": 0.6719,
                "note": "命中改善但动作占比跌破60%。",
            },
            {
                "label": "过滤后补回 legacy 主胜边际 15%-20%",
                "action_share": 0.6076,
                "hit_rate": 0.7040,
                "rolling_10_min_hit_rate": 0.6377,
                "note": "动作恢复到60%以上，但滚动底线回落。",
            },
            {
                "label": "过滤主胜市场平局>=31%并补回低平局主胜",
                "action_share": 0.6022,
                "hit_rate": 0.7059,
                "rolling_10_min_hit_rate": 0.6667,
                "note": "当前最优观察候选，但滚动底线仍低于70%。",
            },
            {
                "label": "引入身价值保护：过滤补主胜身价倍率>=3并补回比分同向主胜",
                "action_share": 0.6104,
                "hit_rate": 0.7054,
                "rolling_10_min_hit_rate": 0.6765,
                "note": "覆盖和总命中继续满足60/70，滚动底线略升但仍未达到稳定70%。",
            },
            {
                "label": "仅用身价值过滤主队明显弱势或过热强势",
                "action_share": 0.5286,
                "hit_rate": 0.7268,
                "rolling_10_min_hit_rate": 0.6842,
                "note": "纯度提升明显，但动作占比跌破60%，只能作为风险信号不能单独作为目标策略。",
            },
            {
                "label": "xG保护：过滤补主胜且xG净差<-0.50",
                "action_share": 0.6022,
                "hit_rate": 0.7104,
                "rolling_10_min_hit_rate": 0.6889,
                "note": "当前覆盖约束下最稳观察候选，但滚动底线仍差约1.1个百分点。",
            },
            {
                "label": "xG强过滤：过滤补主胜且xG净差<0.25",
                "action_share": 0.5504,
                "hit_rate": 0.7426,
                "rolling_10_min_hit_rate": 0.7283,
                "note": "滚动稳定性已超过70%，但动作占比不足60%，补回池质量暂不足。",
            },
            {
                "label": "xG强过滤后每期最多补1场高分候选",
                "action_share": 0.5913,
                "hit_rate": 0.7189,
                "rolling_10_min_hit_rate": 0.7000,
                "note": "滚动刚好守住70%，但动作占比仍低于60%。",
            },
            {
                "label": "xG强过滤后补足到60%覆盖",
                "action_share": 0.6022,
                "hit_rate": 0.7104,
                "rolling_10_min_hit_rate": 0.6849,
                "note": "覆盖达标后，低谷窗口被低质量补动作拖回70%以下。",
            },
            {
                "label": "窗口安全补回规则搜索",
                "action_share": 0.5913,
                "hit_rate": 0.7189,
                "rolling_10_min_hit_rate": 0.7000,
                "note": "从剩余观望池组合搜索主胜、平局、客胜补回规则，候选命中多在15%-55%，未找到可把覆盖补到60%且滚动保持70%的可解释规则。",
            },
            {
                "label": "投注热度补回规则搜索",
                "action_share": 0.5913,
                "hit_rate": 0.7189,
                "rolling_10_min_hit_rate": 0.7000,
                "note": "已接入列表投注热度和均赔字段；剩余补回池中热度规则最高约60%小样本命中，组合仍无法同时达到60%覆盖和70%滚动稳定。",
            },
            {
                "label": "平局救援：强xG过滤后按期补高分候选并救回平局信号",
                "action_share": 0.6022,
                "hit_rate": 0.7195,
                "rolling_10_min_hit_rate": 0.7083,
                "note": "当前落地目标策略：覆盖、总命中、滚动底线均超过目标，按批量期数规则执行。",
            },
        ],
        "status": "needs_new_signal",
        "reason": "现有赛前概率、赔率、比分方向与身价值规则可达到总体60/70，但低谷窗口仍不足70；继续提升需要更强的赛前区分信号或更多样本。",
    }


def _balanced_observation_readiness(
    variants: Mapping[str, Mapping[str, Any]],
    deltas: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    readiness: dict[str, Any] = {}
    for key, variant in variants.items():
        if variant.get("role") != "observation":
            continue
        delta = deltas.get(key, {})
        action_count = safe_int(variant.get("action_count"))
        action_share = safe_float(variant.get("action_share"))
        hit_rate = safe_float(variant.get("hit_rate"))
        issue_count = safe_int(delta.get("filtered_issue_count"))
        miss_capture = safe_float(delta.get("miss_capture_rate"))
        hit_filter = safe_float(delta.get("hit_filter_rate"))
        max_issue_share = safe_float(delta.get("max_issue_filtered_share"))
        if action_count < 80 or issue_count < 5:
            status = "needs_more_samples"
            reason = "动作或过滤覆盖期数仍偏少，继续观察。"
        elif max_issue_share >= 0.40:
            status = "needs_more_samples"
            reason = "过滤样本单期集中度偏高，继续观察。"
        elif miss_capture >= 0.80 and hit_filter <= 0.05 and hit_rate >= 0.95:
            status = "candidate"
            reason = "错单捕获高且误滤低，可进入候选评审。"
        else:
            status = "watch"
            reason = "收益或误滤结构仍需继续跟踪。"
        readiness[key] = {
            "label": variant.get("label", key),
            "status": status,
            "reason": reason,
            "action_count": action_count,
            "action_share": action_share,
            "hit_rate": hit_rate,
            "filtered_issue_count": issue_count,
            "max_issue_filtered_share": max_issue_share,
            "miss_capture_rate": miss_capture,
            "hit_filter_rate": hit_filter,
        }
    return readiness


def _balanced_recommended_variant(
    variants: Mapping[str, Mapping[str, Any]],
    *,
    exclude_keys: set[str] | None = None,
) -> dict[str, Any]:
    excluded = exclude_keys or set()
    eligible = [
        variant
        for key, variant in variants.items()
        if key not in excluded
        if safe_int(variant.get("action_count")) > 0
        and safe_float(variant.get("rolling_10_min_hit_rate")) >= 0.70
        and safe_float(variant.get("rolling_10_min_roi_on_stake")) >= 0.0
    ]
    if not eligible:
        eligible = [
            variant
            for key, variant in variants.items()
            if key not in excluded and safe_int(variant.get("action_count")) > 0
        ]
    if not eligible:
        return {}
    selected = max(
        eligible,
        key=lambda item: (
            safe_float(item.get("roi_on_stake")),
            safe_float(item.get("hit_rate")),
            safe_int(item.get("action_count")),
        ),
    )
    return dict(selected)


def _balanced_production_single_pick(row: Mapping[str, Any]) -> tuple[str, str, str, float]:
    return _balanced_steady_single_pick(row)


def _balanced_preview_observation_flags(row: Mapping[str, Any], production_outcome: str) -> list[dict[str, str]]:
    if not production_outcome:
        return []
    flags: list[dict[str, str]] = []
    for key, label, pick_fn in (
        ("rescue", "修正观察", _balanced_rescue_single_pick),
        ("broad", "泛化观察", _balanced_broad_single_pick),
        ("cautious", "谨慎观察", _balanced_cautious_single_pick),
        ("ultra", "极慎观察", _balanced_ultra_single_pick),
    ):
        observation_outcome, _, observation_reason, _ = pick_fn(row)
        if not observation_outcome:
            flags.append({"key": key, "label": label, "reason": observation_reason})
    return flags


def _balanced_observation_risk(flags: list[dict[str, str]]) -> tuple[str, str]:
    flag_count = len(flags)
    if flag_count >= 3:
        return "resonance", "多层共振"
    if flag_count == 2:
        return "stacked", "双层提示"
    if flag_count == 1:
        return "single", "单层提示"
    return "clean", "无提示"


def _balanced_odds_band(odds: float) -> str:
    if odds <= 0:
        return "unknown"
    if odds < 1.60:
        return "<1.60"
    if odds < 2.00:
        return "1.60-1.99"
    if odds < 2.50:
        return "2.00-2.49"
    if odds < 3.00:
        return "2.50-2.99"
    return ">=3.00"


def _balanced_confidence_band(confidence: float) -> str:
    if confidence < 0.70:
        return "<0.70"
    if confidence < 0.80:
        return "0.70-0.79"
    if confidence < 0.90:
        return "0.80-0.89"
    return ">=0.90"


def _balanced_score_alignment(outcome: str, score_direction: str) -> str:
    if not score_direction:
        return "score_missing"
    return "score_aligned" if outcome == score_direction else "score_conflict"


def _balanced_pick_backtest_metrics(
    rows: list[Mapping[str, Any]],
    pick_fn,
) -> dict[str, Any]:
    metrics = _balanced_pick_metric_summary(rows, pick_fn)
    metrics["stability"] = _balanced_stability_metrics(rows, pick_fn)
    return metrics


def _balanced_pick_metric_summary(
    rows: list[Mapping[str, Any]],
    pick_fn,
) -> dict[str, Any]:
    buckets: dict[str, dict[str, float | int]] = {}
    action_rows: list[dict[str, Any]] = []
    for row in rows:
        outcome, tier, reason, stake_pct = pick_fn(row)
        actual_result = str(_row_value(row, "actual_result", "") or "")
        action = "观望" if not outcome else ("主推" if tier == "core" else "轻仓")
        bucket = buckets.setdefault(
            f"{tier}:{reason}:{outcome or 'watch'}",
            {
                "sample_count": 0,
                "action_count": 0,
                "hits": 0,
                "total_stake_pct": 0.0,
            },
        )
        bucket["sample_count"] += 1
        if outcome:
            hit = 1 if outcome == actual_result else 0
            stake_units = stake_pct / 100.0
            market_odds = _market_odds_for_outcome(row, outcome)
            score_direction = _score_result_direction(_row_value(row, "predicted_score"))
            roi_delta = (
                stake_units * (market_odds - 1.0)
                if hit and market_odds > 0
                else -stake_units
            )
            bucket["action_count"] += 1
            bucket["hits"] += hit
            bucket["total_stake_pct"] += stake_pct
            bucket["total_roi"] = safe_float(bucket.get("total_roi")) + roi_delta
            action_rows.append(
                {
                    "match_id": str(_row_value(row, "match_id", "") or ""),
                    "action": action,
                    "outcome": outcome,
                    "tier": tier,
                    "reason": reason,
                    "stake_pct": stake_pct,
                    "hit": hit,
                    "roi_delta": roi_delta,
                    "league": str(_row_value(row, "league", "") or ""),
                    "issue": str(_row_value(row, "issue", "") or ""),
                    "home_team": str(_row_value(row, "home_team", "") or ""),
                    "away_team": str(_row_value(row, "away_team", "") or ""),
                    "actual_result": actual_result,
                    "predicted_score": str(_row_value(row, "predicted_score", "") or ""),
                    "market_odds": market_odds,
                    "odds_band": _balanced_odds_band(market_odds),
                    "confidence_band": _balanced_confidence_band(safe_float(_row_value(row, "confidence_score"))),
                    "score_alignment": _balanced_score_alignment(outcome, score_direction),
                }
            )

    action_count = len(action_rows)
    bucket_summary: dict[str, dict[str, float | int]] = {}
    for key, bucket in buckets.items():
        bucket_actions = int(bucket["action_count"] or 0)
        bucket_summary[key] = {
            "sample_count": int(bucket["sample_count"] or 0),
            "action_count": bucket_actions,
            "hit_rate": (safe_float(bucket["hits"]) / bucket_actions if bucket_actions else 0.0),
            "total_roi": safe_float(bucket.get("total_roi")),
            "avg_stake_pct": (
                safe_float(bucket["total_stake_pct"]) / bucket_actions if bucket_actions else 0.0
            ),
        }

    return {
        "action_count": action_count,
        "sample_count": len(rows),
        "action_share": action_count / len(rows) if rows else 0.0,
        "watch_count": max(len(rows) - action_count, 0),
        "watch_share": 1.0 - (action_count / len(rows)) if rows else 0.0,
        "hit_rate": (sum(item["hit"] for item in action_rows) / action_count if action_count else 0.0),
        "total_roi": sum(safe_float(item["roi_delta"]) for item in action_rows),
        "roi_on_stake": (
            sum(safe_float(item["roi_delta"]) for item in action_rows)
            / sum(safe_float(item["stake_pct"]) / 100.0 for item in action_rows)
            if action_rows and sum(safe_float(item["stake_pct"]) for item in action_rows) > 0
            else 0.0
        ),
        "avg_stake_pct": (
            sum(safe_float(item["stake_pct"]) for item in action_rows) / action_count if action_count else 0.0
        ),
        "core": _balanced_tier_metrics(action_rows, "core"),
        "standard": _balanced_tier_metrics(action_rows, "standard"),
        "buckets": bucket_summary,
        "league_buckets": _balanced_group_metrics(action_rows, "league"),
        "issue_buckets": _balanced_group_metrics(action_rows, "issue"),
        "diagnostics": _balanced_action_diagnostics(action_rows),
        "recent_actions": _balanced_action_examples(action_rows, limit=12),
        "misses": _balanced_action_examples([item for item in action_rows if not int(item.get("hit", 0))], limit=12),
    }


def _balanced_action_examples(action_rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    sorted_rows = sorted(action_rows, key=lambda item: _issue_sort_key(item.get("issue")), reverse=True)
    return [
        {
            "issue": str(item.get("issue", "") or ""),
            "match_id": str(item.get("match_id", "") or ""),
            "league": str(item.get("league", "") or ""),
            "home_team": str(item.get("home_team", "") or ""),
            "away_team": str(item.get("away_team", "") or ""),
            "outcome": str(item.get("outcome", "") or ""),
            "actual_result": str(item.get("actual_result", "") or ""),
            "predicted_score": str(item.get("predicted_score", "") or ""),
            "reason": str(item.get("reason", "") or ""),
            "hit": int(item.get("hit", 0)),
            "roi_delta": safe_float(item.get("roi_delta")),
            "market_odds": safe_float(item.get("market_odds")),
        }
        for item in sorted_rows[:limit]
    ]


def _balanced_preview_item(row: Mapping[str, Any], *, issue: str) -> dict[str, Any]:
    outcome, tier, reason, stake_pct = _balanced_production_single_pick(row)
    observation_flags = _balanced_preview_observation_flags(row, outcome)
    observation_risk_level, observation_risk_label = _balanced_observation_risk(observation_flags)
    return {
        "match_id": str(_row_value(row, "match_id", "") or ""),
        "issue": issue,
        "league": str(_row_value(row, "league", "") or ""),
        "home_team": str(_row_value(row, "home_team", "") or ""),
        "away_team": str(_row_value(row, "away_team", "") or ""),
        "outcome": outcome,
        "action": "观望" if not outcome else ("主推" if tier == "core" else "轻仓"),
        "tier": tier,
        "reason": reason,
        "stake_pct": stake_pct,
        "predicted_score": str(_row_value(row, "predicted_score", "") or ""),
        "confidence_score": safe_float(_row_value(row, "confidence_score")),
        "market_odds": _market_odds_for_outcome(row, outcome) if outcome else 0.0,
        "observation_flags": observation_flags,
        "observation_flag_count": len(observation_flags),
        "observation_risk_level": observation_risk_level,
        "observation_risk_label": observation_risk_label,
    }


def _balanced_target_batch_preview_item(
    row: Mapping[str, Any],
    *,
    issue: str,
    action: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if action:
        outcome = str(action.get("outcome", "") or "")
        tier = str(action.get("tier", "") or "standard")
        reason = str(action.get("reason", "") or "coverage_draw_rescue")
        stake_pct = safe_float(action.get("stake_pct"))
    else:
        outcome, tier, reason, stake_pct = "", "watch", "coverage_draw_rescue_watch", 0.0
    observation_flags = _balanced_preview_observation_flags(row, outcome)
    observation_risk_level, observation_risk_label = _balanced_observation_risk(observation_flags)
    return {
        "match_id": str(_row_value(row, "match_id", "") or ""),
        "issue": issue,
        "league": str(_row_value(row, "league", "") or ""),
        "home_team": str(_row_value(row, "home_team", "") or ""),
        "away_team": str(_row_value(row, "away_team", "") or ""),
        "outcome": outcome,
        "action": "观望" if not outcome else ("主推" if tier == "core" else "轻仓"),
        "tier": tier,
        "reason": reason,
        "stake_pct": stake_pct,
        "predicted_score": str(_row_value(row, "predicted_score", "") or ""),
        "confidence_score": safe_float(_row_value(row, "confidence_score")),
        "market_odds": _market_odds_for_outcome(row, outcome) if outcome else 0.0,
        "observation_flags": observation_flags,
        "observation_flag_count": len(observation_flags),
        "observation_risk_level": observation_risk_level,
        "observation_risk_label": observation_risk_label,
        "strategy_key": "coverage_draw_rescue",
        "strategy_label": "平局救援目标批量策略",
    }


def _target_batch_action_label(tier: str, outcome: str) -> str:
    if not outcome:
        return "观望"
    return "主推" if tier == "core" else "轻仓"


def _preserved_target_batch_outcome(row: Mapping[str, Any]) -> str:
    for key in ("recommended_outcome", "algo_recommended_outcome"):
        outcome = str(_row_value(row, key, "") or "")
        if outcome in OUTCOMES:
            return outcome
    return ""


def _latest_issue_prediction_rows(issue: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for match in list_matches_by_issue(issue):
        runs = list_prediction_runs(str(match["match_id"]), limit=1)
        if not runs:
            continue
        run = dict(runs[0])
        snapshot = get_latest_feature_snapshot(str(match["match_id"]))
        if snapshot is not None:
            run.update(dict(snapshot))
        run.update(
            {
                "match_id": str(match["match_id"]),
                "issue": issue,
                "league": str(match["league"] or ""),
                "home_team": str(match["home_team"] or ""),
                "away_team": str(match["away_team"] or ""),
                "match_time": str(match["match_time"] or ""),
            }
        )
        rows.append(run)
    return rows


def apply_target_batch_strategy_to_issue(
    issue: str | None = None,
    match_ids: list[str] | None = None,
) -> dict[str, Any]:
    selected_issue = str(issue or get_latest_issue() or "").strip()
    if not selected_issue:
        return {
            "issue": "",
            "strategy_key": "coverage_draw_rescue",
            "updated_count": 0,
            "action_count": 0,
            "watch_count": 0,
            "sample_count": 0,
            "settled_skip_count": 0,
        }
    rows = _latest_issue_prediction_rows(selected_issue)
    selected_match_ids = {str(match_id).strip() for match_id in (match_ids or []) if str(match_id).strip()}
    if selected_match_ids:
        rows = [row for row in rows if str(_row_value(row, "match_id", "") or "").strip() in selected_match_ids]
    mutable_rows: list[dict[str, Any]] = []
    settled_skip_count = 0
    for row in rows:
        run_id = safe_int(_row_value(row, "run_id"))
        if not run_id:
            continue
        if get_feedback_log(run_id) is not None:
            settled_skip_count += 1
            continue
        mutable_rows.append(row)

    action_rows = _balanced_coverage_draw_rescue_action_rows(mutable_rows)
    action_by_match_id = {str(item.get("match_id", "") or ""): item for item in action_rows}
    updated_count = 0
    action_count = 0
    for row in mutable_rows:
        run_id = safe_int(_row_value(row, "run_id"))
        if not run_id:
            continue
        action = action_by_match_id.get(str(_row_value(row, "match_id", "") or ""))
        if action:
            outcome = str(action.get("outcome", "") or "")
            tier = str(action.get("tier", "") or "standard")
            action_label = _target_batch_action_label(tier, outcome)
            stake_pct = safe_float(action.get("stake_pct"))
            action_count += 1
        else:
            outcome = _preserved_target_batch_outcome(row)
            action_label = "观望"
            stake_pct = 0.0
        update_prediction_run_fields(
            run_id,
            {
                "recommendation": action_label,
                "recommended_outcome": outcome,
                "suggested_stake_pct": stake_pct,
                "effective_recommendation": action_label,
                "effective_stake_pct": stake_pct,
                "effective_action_source": "target_batch_strategy",
            },
        )
        updated_count += 1
    return {
        "issue": selected_issue,
        "strategy_key": "coverage_draw_rescue",
        "updated_count": updated_count,
        "action_count": action_count,
        "watch_count": max(len(mutable_rows) - action_count, 0),
        "sample_count": len(mutable_rows),
        "settled_skip_count": settled_skip_count,
    }


def _handicap_action_summary_for_issue(issue: str | None = None) -> dict[str, Any]:
    selected_issue = str(issue or get_latest_issue() or "").strip()
    if not selected_issue:
        return {
            "issue": "",
            "action_count": 0,
            "watch_count": 0,
            "sample_count": 0,
        }
    rows = _latest_issue_prediction_rows(selected_issue)
    action_count = 0
    for row in rows:
        action = _action_label(str(_row_value(row, "handicap_recommendation", "") or ""))
        side = str(_row_value(row, "handicap_recommended_side", "") or "").strip()
        if action != "观望" and side in {"home", "away"}:
            action_count += 1
    sample_count = len(rows)
    return {
        "issue": selected_issue,
        "action_count": action_count,
        "watch_count": max(sample_count - action_count, 0),
        "sample_count": sample_count,
    }


def _main_action_summary_for_issue(issue: str | None = None) -> dict[str, Any]:
    selected_issue = str(issue or get_latest_issue() or "").strip()
    if not selected_issue:
        return {
            "issue": "",
            "action_count": 0,
            "watch_count": 0,
            "sample_count": 0,
        }
    rows = _latest_issue_prediction_rows(selected_issue)
    action_count = 0
    for row in rows:
        effective_action = str(_row_value(row, "effective_recommendation", "") or "").strip()
        if effective_action:
            action = _action_label(effective_action)
            stake_pct = safe_float(_row_value(row, "effective_stake_pct"))
        else:
            action = _action_label(str(_row_value(row, "recommendation", "") or ""))
            stake_pct = safe_float(_row_value(row, "suggested_stake_pct"))
        outcome = str(_row_value(row, "recommended_outcome", "") or "").strip()
        if action != "观望" and stake_pct > 0 and outcome in OUTCOMES:
            action_count += 1
    sample_count = len(rows)
    return {
        "issue": selected_issue,
        "action_count": action_count,
        "watch_count": max(sample_count - action_count, 0),
        "sample_count": sample_count,
    }


def _issue_action_summary_message(
    issue: str | None,
    target_batch_application: Mapping[str, Any] | None = None,
) -> str:
    selected_issue = str(issue or "").strip()
    if not selected_issue:
        return ""
    main_summary = _main_action_summary_for_issue(selected_issue)
    handicap_summary = _handicap_action_summary_for_issue(selected_issue)
    if not main_summary.get("sample_count") and not handicap_summary.get("sample_count"):
        return ""

    message = (
        f"当前落库结果：主推荐 {main_summary.get('action_count', 0)}/"
        f"{main_summary.get('sample_count', 0)} 场可执行，"
        f"让球盘 {handicap_summary.get('action_count', 0)}/"
        f"{handicap_summary.get('sample_count', 0)} 场可执行。"
    )
    if isinstance(target_batch_application, Mapping):
        updated_count = safe_int(target_batch_application.get("updated_count"))
        settled_skip_count = safe_int(target_batch_application.get("settled_skip_count"))
        if updated_count > 0:
            message += f" 主推荐目标批量策略本次更新 {updated_count} 场。"
        elif settled_skip_count > 0:
            message += f" 已结算场次 {settled_skip_count} 场保留原记录，未覆盖主推荐批量策略。"
    return message


def preview_production_single_pick(issue: str | None = None, *, limit: int | None = 20) -> list[dict[str, Any]]:
    selected_issue = issue or get_latest_issue()
    if not selected_issue:
        return []
    rows = _latest_issue_prediction_rows(str(selected_issue))
    action_rows = _balanced_coverage_draw_rescue_action_rows(rows)
    action_by_match_id = {str(item.get("match_id", "") or ""): item for item in action_rows}
    preview = [
        _balanced_target_batch_preview_item(
            row,
            issue=selected_issue,
            action=action_by_match_id.get(str(_row_value(row, "match_id", "") or "")),
        )
        for row in rows
    ]
    preview.sort(
        key=lambda item: (
            1 if item["outcome"] else 0,
            safe_float(item.get("confidence_score")),
            safe_float(item.get("market_odds")),
        ),
        reverse=True,
    )
    return preview if limit is None else preview[:limit]


def summarize_target_batch_strategy() -> dict[str, Any]:
    rows = list_backtest_rows()
    if not rows:
        return {
            "key": "coverage_draw_rescue",
            "label": "平局救援目标批量策略",
            "role": "production",
            "sample_count": 0,
            "action_count": 0,
            "hit_count": 0,
            "miss_count": 0,
            "action_share": 0.0,
            "hit_rate": 0.0,
            "target_met": False,
            "status": "empty",
        }
    return _balanced_target_batch_strategy_summary(rows)


def _balanced_action_diagnostics(action_rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float | int]]]:
    return {
        "by_outcome": _balanced_group_metrics(action_rows, "outcome"),
        "by_reason": _balanced_group_metrics(action_rows, "reason"),
        "by_odds_band": _balanced_group_metrics(action_rows, "odds_band"),
        "by_confidence_band": _balanced_group_metrics(action_rows, "confidence_band"),
        "by_score_alignment": _balanced_group_metrics(action_rows, "score_alignment"),
    }


def _balanced_group_metrics(action_rows: list[dict[str, Any]], group_key: str) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in action_rows:
        grouped.setdefault(str(item.get(group_key, "") or ""), []).append(item)

    summary: dict[str, dict[str, float | int]] = {}
    for key, rows in grouped.items():
        action_count = len(rows)
        total_roi = sum(safe_float(item["roi_delta"]) for item in rows)
        total_stake = sum(safe_float(item["stake_pct"]) / 100.0 for item in rows)
        summary[key] = {
            "action_count": action_count,
            "hit_rate": (sum(int(item["hit"]) for item in rows) / action_count if action_count else 0.0),
            "total_roi": total_roi,
            "roi_on_stake": (total_roi / total_stake if total_stake > 0 else 0.0),
        }
    return summary


def _issue_sort_key(issue: Any) -> tuple[int, str]:
    text = str(issue or "")
    return (safe_int(text), text)


def _balanced_stability_metrics(rows: list[Mapping[str, Any]], pick_fn) -> dict[str, Any]:
    issues = sorted({str(_row_value(row, "issue", "")) for row in rows}, key=_issue_sort_key)

    def _issue_subset(selected_issues: set[str]) -> list[Mapping[str, Any]]:
        return [row for row in rows if str(_row_value(row, "issue", "")) in selected_issues]

    def _latest(window_size: int) -> dict[str, Any]:
        selected = set(issues[-window_size:])
        return _balanced_pick_metric_summary(_issue_subset(selected), pick_fn)

    def _rolling(window_size: int) -> dict[str, Any]:
        windows: list[dict[str, Any]] = []
        if len(issues) < window_size:
            return {
                "window_size": window_size,
                "window_count": 0,
                "min_hit_rate": 0.0,
                "min_roi_on_stake": 0.0,
                "avg_hit_rate": 0.0,
                "avg_roi_on_stake": 0.0,
                "worst_hit_window": {},
                "worst_roi_window": {},
            }
        for index in range(0, len(issues) - window_size + 1):
            window_issues = issues[index : index + window_size]
            summary = _balanced_pick_metric_summary(_issue_subset(set(window_issues)), pick_fn)
            windows.append(
                {
                    "start_issue": window_issues[0],
                    "end_issue": window_issues[-1],
                    "action_count": summary["action_count"],
                    "hit_rate": summary["hit_rate"],
                    "roi_on_stake": summary["roi_on_stake"],
                }
            )
        worst_hit = min(windows, key=lambda item: safe_float(item["hit_rate"]))
        worst_roi = min(windows, key=lambda item: safe_float(item["roi_on_stake"]))
        return {
            "window_size": window_size,
            "window_count": len(windows),
            "min_hit_rate": safe_float(worst_hit["hit_rate"]),
            "min_roi_on_stake": safe_float(worst_roi["roi_on_stake"]),
            "avg_hit_rate": sum(safe_float(item["hit_rate"]) for item in windows) / len(windows),
            "avg_roi_on_stake": sum(safe_float(item["roi_on_stake"]) for item in windows) / len(windows),
            "worst_hit_window": worst_hit,
            "worst_roi_window": worst_roi,
        }

    return {
        "issue_count": len(issues),
        "latest_6_issues": _latest(6),
        "latest_10_issues": _latest(10),
        "rolling_10_issues": _rolling(10),
    }


def _balanced_tier_metrics(action_rows: list[dict[str, Any]], tier: str) -> dict[str, float | int]:
    tier_rows = [item for item in action_rows if item["tier"] == tier]
    action_count = len(tier_rows)
    total_roi = sum(safe_float(item["roi_delta"]) for item in tier_rows)
    total_stake = sum(safe_float(item["stake_pct"]) / 100.0 for item in tier_rows)
    return {
        "action_count": action_count,
        "hit_rate": (sum(int(item["hit"]) for item in tier_rows) / action_count if action_count else 0.0),
        "total_roi": total_roi,
        "roi_on_stake": (total_roi / total_stake if total_stake > 0 else 0.0),
    }


def _action_backtest_metrics(
    rows: list[Mapping[str, Any]],
    *,
    action_field: str,
    outcome_field: str,
    stake_field: str,
) -> dict[str, float | int]:
    actionable_rows = [row for row in rows if _action_label(str(_row_value(row, action_field, ""))) != "观望"]
    if not actionable_rows:
        return {
            "action_count": 0,
            "hit_rate": 0.0,
            "total_roi": 0.0,
            "avg_roi": 0.0,
            "avg_stake_pct": 0.0,
        }

    hits = 0
    total_roi = 0.0
    total_stake_pct = 0.0
    for row in actionable_rows:
        if str(_row_value(row, outcome_field, "") or "") == str(_row_value(row, "actual_result", "") or ""):
            hits += 1
        total_stake_pct += safe_float(_row_value(row, stake_field))
        if (
            action_field == "recommendation"
            and outcome_field == "recommended_outcome"
            and stake_field == "suggested_stake_pct"
        ):
            total_roi += safe_float(_row_value(row, "roi_delta"))
        else:
            total_roi += _calculate_action_roi(
                row,
                str(_row_value(row, "actual_result", "") or ""),
                action_field=action_field,
                outcome_field=outcome_field,
                stake_field=stake_field,
            )

    count = len(actionable_rows)
    return {
        "action_count": count,
        "hit_rate": hits / count,
        "total_roi": total_roi,
        "avg_roi": total_roi / count,
        "avg_stake_pct": total_stake_pct / count,
    }


def _executed_action_backtest_metrics(
    rows: list[Mapping[str, Any]],
) -> dict[str, float | int]:
    actionable_rows: list[tuple[Mapping[str, Any], str, float]] = []
    for row in rows:
        action, stake_pct = _resolved_effective_action(row)
        if action == "观望" or stake_pct <= 0:
            continue
        actionable_rows.append((row, action, stake_pct))

    if not actionable_rows:
        return {
            "action_count": 0,
            "hit_rate": 0.0,
            "total_roi": 0.0,
            "avg_roi": 0.0,
            "avg_stake_pct": 0.0,
        }

    hits = 0
    total_roi = 0.0
    total_stake_pct = 0.0
    for row, _action, stake_pct in actionable_rows:
        if str(_row_value(row, "recommended_outcome", "") or "") == str(_row_value(row, "actual_result", "") or ""):
            hits += 1
        total_stake_pct += stake_pct
        total_roi += safe_float(_row_value(row, "roi_delta"))

    count = len(actionable_rows)
    return {
        "action_count": count,
        "hit_rate": hits / count,
        "total_roi": total_roi,
        "avg_roi": total_roi / count,
        "avg_stake_pct": total_stake_pct / count,
    }


def _current_policy_backtest_metrics(
    rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Replay settled rows through the currently loaded action policy.

    ``feedback_logs`` intentionally point at the pre-match canonical run, so
    historical ``final`` metrics preserve what was actually executable before
    kickoff. This replay answers a different question: what the current policy
    would execute if it saw the same stored probability/market snapshot.
    """

    action_rows: list[dict[str, Any]] = []
    buckets: dict[str, dict[str, Any]] = {}

    for row in rows:
        raw_probabilities = {
            "home": safe_float(_row_value(row, "final_home_prob")),
            "draw": safe_float(_row_value(row, "final_draw_prob")),
            "away": safe_float(_row_value(row, "final_away_prob")),
        }
        calibrated_probabilities = {
            "home": safe_float(_row_value(row, "calibrated_home_prob")),
            "draw": safe_float(_row_value(row, "calibrated_draw_prob")),
            "away": safe_float(_row_value(row, "calibrated_away_prob")),
        }
        probabilities = (
            calibrated_probabilities
            if sum(calibrated_probabilities.values()) > 0.01
            else raw_probabilities
        )
        market_odds = {
            "home": safe_float(_row_value(row, "market_odds_home")),
            "draw": safe_float(_row_value(row, "market_odds_draw")),
            "away": safe_float(_row_value(row, "market_odds_away")),
        }
        market_probs = {
            "home": safe_float(_row_value(row, "market_home_prob")),
            "draw": safe_float(_row_value(row, "market_draw_prob")),
            "away": safe_float(_row_value(row, "market_away_prob")),
        }
        legacy_probabilities = {
            "home": safe_float(_row_value(row, "legacy_home_prob")),
            "draw": safe_float(_row_value(row, "legacy_draw_prob")),
            "away": safe_float(_row_value(row, "legacy_away_prob")),
        }
        sorted_probs = sorted(probabilities.values(), reverse=True)
        model_margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) >= 2 else 0.0
        selected_outcome = max(
            effective_market_probs(market_probs, market_odds),
            key=effective_market_probs(market_probs, market_odds).get,
        )
        policy = evaluate_action_policy(
            probabilities=probabilities,
            market_odds=market_odds,
            market_probs=market_probs,
            legacy_probabilities=legacy_probabilities,
            quality_score=safe_float(_row_value(row, "quality_score")),
            model_agreement=safe_float(_row_value(row, "model_agreement")),
            model_margin=model_margin,
            threshold_config=_merge_threshold_config(),
            selected_outcome=selected_outcome,
        )
        action = _action_label(str(policy.get("recommendation", "")))
        gated_action, _gate_reason = _llm_action_gate(
            action,
            low_odds_favorite_guard=bool(policy.get("low_odds_favorite_guard")),
            review_status=str(_row_value(row, "llm_review_status", "") or ""),
            review_decision=str(_row_value(row, "llm_review_decision", "") or ""),
            review_target_action=str(_row_value(row, "llm_review_target_action", "") or ""),
            arbiter_status=str(_row_value(row, "arbiter_review_status", "") or ""),
            arbiter_decision=str(_row_value(row, "arbiter_review_decision", "") or ""),
            arbiter_target_action=str(_row_value(row, "arbiter_review_target_action", "") or ""),
            manual_review_status=str(_row_value(row, "manual_review_status", "") or ""),
        )
        action = gated_action
        outcome = str(policy.get("recommended_outcome", "") or "")
        stake_pct = 0.0 if action == "观望" else safe_float(policy.get("stake_pct"))
        actual_result = str(_row_value(row, "actual_result", "") or "")
        hit = 1 if outcome == actual_result else 0
        roi = 0.0
        if action != "观望" and stake_pct > 0:
            odds = safe_float(market_odds.get(outcome))
            stake_units = stake_pct / 100.0
            roi = round(stake_units * (odds - 1.0), 4) if hit and odds > 0 else round(-stake_units, 4)
            action_rows.append(
                {
                    "action": action,
                    "outcome": outcome,
                    "stake_pct": stake_pct,
                    "hit": hit,
                    "roi": roi,
                }
            )

        bucket = buckets.setdefault(
            f"{action}:{outcome}",
            {
                "sample_count": 0,
                "action_count": 0,
                "hits": 0,
                "total_roi": 0.0,
                "total_stake_pct": 0.0,
            },
        )
        bucket["sample_count"] += 1
        if action != "观望" and stake_pct > 0:
            bucket["action_count"] += 1
            bucket["hits"] += hit
            bucket["total_roi"] += roi
            bucket["total_stake_pct"] += stake_pct

    action_count = len(action_rows)
    total_roi = sum(item["roi"] for item in action_rows)
    total_stake_pct = sum(item["stake_pct"] for item in action_rows)
    replay_buckets: dict[str, dict[str, float | int]] = {}
    for key, bucket in buckets.items():
        action_count_bucket = int(bucket["action_count"] or 0)
        replay_buckets[key] = {
            "sample_count": int(bucket["sample_count"] or 0),
            "action_count": action_count_bucket,
            "hit_rate": (safe_float(bucket["hits"]) / action_count_bucket if action_count_bucket else 0.0),
            "total_roi": round(safe_float(bucket["total_roi"]), 4),
            "avg_roi": (safe_float(bucket["total_roi"]) / action_count_bucket if action_count_bucket else 0.0),
            "avg_stake_pct": (safe_float(bucket["total_stake_pct"]) / action_count_bucket if action_count_bucket else 0.0),
        }

    return {
        "action_count": action_count,
        "hit_rate": (sum(item["hit"] for item in action_rows) / action_count if action_count else 0.0),
        "total_roi": round(total_roi, 4),
        "avg_roi": (total_roi / action_count if action_count else 0.0),
        "avg_stake_pct": (total_stake_pct / action_count if action_count else 0.0),
        "buckets": replay_buckets,
    }


def _target_frontier_summary(frontier: Mapping[str, Any], name: str) -> dict[str, Any]:
    candidate = frontier.get(name, {}) if isinstance(frontier, Mapping) else {}
    metrics = candidate.get("validation_metrics", {}) if isinstance(candidate, Mapping) else {}
    if not isinstance(metrics, Mapping):
        metrics = {}
    return {
        "hit_rate": safe_float(metrics.get("hit_rate")),
        "action_share": safe_float(metrics.get("action_share")),
        "watch_share": safe_float(metrics.get("watch_share")),
        "action_count": int(metrics.get("action_count", 0) or 0),
        "sample_count": int(metrics.get("sample_count", 0) or 0),
        "reason": candidate.get("reason", "") if isinstance(candidate, Mapping) else "",
    }


def _target_strategy_backtest_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    active_profile = get_active_learning_profile_config()
    latest_profile = _hydrate_profile(get_latest_learning_profile())
    profile = active_profile or latest_profile or {}
    diagnostics = profile.get("strategy_diagnostics", {}) if isinstance(profile, Mapping) else {}
    if not isinstance(diagnostics, Mapping):
        diagnostics = {}

    params = active_profile.get("strategy_params", {}) if isinstance(active_profile, Mapping) else {}
    validation_metrics: Mapping[str, Any] = {}
    status = str(diagnostics.get("status", "") or "")
    reason = str(diagnostics.get("reason", "") or "")
    if isinstance(params, Mapping) and params:
        if str(params.get("strategy_kind", "") or "") == "handicap_bucket_table":
            validation_metrics = _handicap_bucket_strategy_metrics(rows, params)
        else:
            validation_metrics = evaluate_target_strategy_rule(rows, params)
        status = status or "active"
        reason = reason or "active strategy evaluated on settled backtest rows"
    else:
        raw_validation = diagnostics.get("validation", {}) if isinstance(diagnostics, Mapping) else {}
        validation_metrics = raw_validation if isinstance(raw_validation, Mapping) else {}
        status = status or "insufficient_samples"
        reason = reason or "no active target strategy; showing latest saved diagnostics"

    best_candidate = diagnostics.get("best_candidate", {}) if isinstance(diagnostics, Mapping) else {}
    if not isinstance(best_candidate, Mapping):
        best_candidate = {}
    best_validation = best_candidate.get("validation_metrics", {}) if isinstance(best_candidate, Mapping) else {}
    if not isinstance(best_validation, Mapping):
        best_validation = {}
    frontier = diagnostics.get("frontier", {}) if isinstance(diagnostics, Mapping) else {}
    if not isinstance(frontier, Mapping):
        frontier = {}
    target_metrics = diagnostics.get("target_metrics", {}) if isinstance(diagnostics, Mapping) else {}
    if not isinstance(target_metrics, Mapping) or not target_metrics:
        target_metrics = profile.get("target_metrics", {}) if isinstance(profile, Mapping) else {}
    if not isinstance(target_metrics, Mapping):
        target_metrics = {}

    return {
        "status": status,
        "reason": reason,
        "hit_rate": safe_float(validation_metrics.get("hit_rate")),
        "action_share": safe_float(validation_metrics.get("action_share")),
        "watch_share": safe_float(validation_metrics.get("watch_share")),
        "action_count": int(validation_metrics.get("action_count", 0) or 0),
        "sample_count": int(validation_metrics.get("sample_count", len(rows)) or 0),
        "target_hit_rate": safe_float(target_metrics.get("target_hit_rate"), DEFAULT_TARGET_HIT_RATE),
        "min_action_share": safe_float(target_metrics.get("min_action_share"), DEFAULT_MIN_ACTION_SHARE),
        "params": dict(params) if isinstance(params, Mapping) else {},
        "best_candidate": {
            "hit_rate": safe_float(best_validation.get("hit_rate")),
            "action_share": safe_float(best_validation.get("action_share")),
            "watch_share": safe_float(best_validation.get("watch_share")),
            "action_count": int(best_validation.get("action_count", 0) or 0),
            "sample_count": int(best_validation.get("sample_count", 0) or 0),
            "params": best_candidate.get("params", {}) if isinstance(best_candidate, Mapping) else {},
            "reason": best_candidate.get("reason", "") if isinstance(best_candidate, Mapping) else "",
        },
        "frontier": {
            "best_hit": _target_frontier_summary(frontier, "best_hit"),
            "best_action": _target_frontier_summary(frontier, "best_action"),
            "best_covered": _target_frontier_summary(frontier, "best_covered"),
            "gaps": frontier.get("gaps", {}) if isinstance(frontier, Mapping) else {},
        },
        "review_signals": {
            "validation_rows": _review_signal_summary(rows),
        },
    }


def _review_bucket_key(row: Mapping[str, Any]) -> str:
    status = str(_row_value(row, "llm_review_status", "") or "")
    if status == "failed":
        return "failed"
    if status == "skipped":
        return "skipped"
    decision = str(_row_value(row, "llm_review_decision", "") or "")
    if decision == "abstain":
        return "skipped"
    return decision or "skipped"


def summarize_backtest(
    *,
    league: str = "",
    month: str = "",
    odds_min: float | None = None,
    odds_max: float | None = None,
    confidence_min: float | None = None,
    ev_min: float | None = None,
) -> dict[str, Any]:
    rows = list_backtest_rows(
        league=league,
        month=month,
        odds_min=odds_min,
        odds_max=odds_max,
        confidence_min=confidence_min,
        ev_min=ev_min,
    )
    if not rows:
        return {
            "total_settled": 0,
            "filters": {
                "league": league,
                "month": month,
                "odds_min": odds_min,
                "odds_max": odds_max,
                "confidence_min": confidence_min,
                "ev_min": ev_min,
            },
            "message": "暂无满足条件的已结算反馈样本。",
        }

    comparisons = {"market": [], "legacy": [], "independent": []}
    positive_ev = 0

    for row in rows:
        actual_result = str(row["actual_result"])
        market_probs = {
            "home": safe_float(row["market_home_prob"]),
            "draw": safe_float(row["market_draw_prob"]),
            "away": safe_float(row["market_away_prob"]),
        }
        legacy_probs = {
            "home": safe_float(row["legacy_home_prob"]),
            "draw": safe_float(row["legacy_draw_prob"]),
            "away": safe_float(row["legacy_away_prob"]),
        }
        independent_probs = {
            "home": safe_float(row["final_home_prob"]),
            "draw": safe_float(row["final_draw_prob"]),
            "away": safe_float(row["final_away_prob"]),
        }

        comparisons["market"].append(
            {
                "brier": _brier_score(market_probs, actual_result),
                "log_loss": _log_loss(market_probs, actual_result),
                "hit": _hit_rate(market_probs, actual_result),
            }
        )
        comparisons["legacy"].append(
            {
                "brier": _brier_score(legacy_probs, actual_result),
                "log_loss": _log_loss(legacy_probs, actual_result),
                "hit": _hit_rate(legacy_probs, actual_result),
            }
        )
        comparisons["independent"].append(
            {
                "brier": _brier_score(independent_probs, actual_result),
                "log_loss": _log_loss(independent_probs, actual_result),
                "hit": _hit_rate(independent_probs, actual_result),
            }
        )

        recommended_outcome = str(row["recommended_outcome"] or "")
        recommended_ev = 0.0
        if recommended_outcome == "home":
            recommended_ev = safe_float(row["ev_home"])
        elif recommended_outcome == "draw":
            recommended_ev = safe_float(row["ev_draw"])
        elif recommended_outcome == "away":
            recommended_ev = safe_float(row["ev_away"])
        if recommended_ev > 0:
            positive_ev += 1

    algo_metrics = _action_backtest_metrics(
        rows,
        action_field="algo_recommendation",
        outcome_field="algo_recommended_outcome",
        stake_field="algo_suggested_stake_pct",
    )
    final_metrics = _executed_action_backtest_metrics(rows)
    current_policy_metrics = _current_policy_backtest_metrics(rows)
    target_strategy_summary = _target_strategy_backtest_summary(rows)
    balanced_single_pick_metrics = _balanced_single_pick_backtest_metrics(rows)
    target_batch_strategy_summary = _balanced_target_batch_strategy_summary(rows)

    bucket_rows: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        bucket_rows.setdefault(_review_bucket_key(row), []).append(row)

    review_buckets: dict[str, dict[str, float | int]] = {}
    for key, bucket in bucket_rows.items():
        review_buckets[key] = _action_backtest_metrics(
            bucket,
            action_field="recommendation",
            outcome_field="recommended_outcome",
            stake_field="suggested_stake_pct",
        )
        review_buckets[key]["sample_count"] = len(bucket)

    return {
        "total_settled": len(rows),
        "filters": {
            "league": league,
            "month": month,
            "odds_min": odds_min,
            "odds_max": odds_max,
            "confidence_min": confidence_min,
            "ev_min": ev_min,
        },
        "market": _aggregate_prob_metrics(comparisons["market"]),
        "legacy": _aggregate_prob_metrics(comparisons["legacy"]),
        "independent": _aggregate_prob_metrics(comparisons["independent"]),
        "algorithm": algo_metrics,
        "final": final_metrics,
        "current_policy": current_policy_metrics,
        "target_strategy": target_strategy_summary,
        "balanced_single_pick": balanced_single_pick_metrics,
        "target_batch_strategy": target_batch_strategy_summary,
        "review_buckets": review_buckets,
        "recommendation_hit_rate": safe_float(final_metrics["hit_rate"]),
        "positive_ev_share": positive_ev / len(rows),
        "total_roi": safe_float(final_metrics["total_roi"]),
        "avg_roi": safe_float(final_metrics["avg_roi"]),
    }


__all__ = [
    "build_match_features",
    "get_canonical_prediction_run",
    "predict_issue",
    "predict_match",
    "record_feedback",
    "resolve_manual_review",
    "run_data_quality",
    "run_legacy_market_model",
    "run_ml_model",
    "run_quant_model",
    "run_risk_assessor",
    "settle_match_result",
    "settle_issue_results",
    "summarize_backtest",
    "evaluate_handicap_recommendation",
]
