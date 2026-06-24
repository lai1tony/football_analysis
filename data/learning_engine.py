from __future__ import annotations

import json
import math
import re
from datetime import datetime
from itertools import product
from typing import Any, Mapping

from action_policy import evaluate_action_policy, expected_values_for, stake_for_action
from collection_repository import (
    DEFAULT_ISSUE_RETENTION_COUNT,
    get_active_learning_profile,
    get_learning_profile,
    get_latest_learning_profile,
    init_db,
    list_backtest_rows,
    list_learning_profiles,
    list_recent_issues,
    save_learning_profile,
)
from feature_engine import clamp, safe_float
from outcome_policy import evaluate_outcome_policy


OUTCOMES = ("home", "draw", "away")
ACTION_LEVELS = {"观望": 0, "轻仓": 1, "主推": 2}
ACTION_BY_LEVEL = {value: key for key, value in ACTION_LEVELS.items()}
PROFILE_STATUS_LABELS = {
    "ready_candidate": "待启用候选",
    "active": "启用中",
    "archived": "已归档",
    "insufficient_samples": "样本不足",
    "no_gain": "无增益",
}
COMPONENT_STATUS_LABELS = {
    "ready": "可用",
    "not_ready": "未启用",
    "insufficient_samples": "样本不足",
    "no_gain": "无增益",
}
# Sample-size floors for the temperature/bias calibrator. Older versions
# allowed fitting on as few as 60 samples (8 per class); with the rolling
# 90-issue retention window that meant validation splits of ~12 samples,
# which produces noisy "ready" calibrators that fail to generalize. The
# realistic minimum for a 3-class temperature scaling fit with a holdout
# split is ~150-200 settled samples; we set the floor accordingly.
CALIBRATION_MIN_TOTAL_SAMPLES = 200
CALIBRATION_MIN_CLASS_SAMPLES = 25
THRESHOLD_MIN_TRAIN_ACTIONS = 40
THRESHOLD_MIN_VALIDATION_ACTIONS = 10
DEFAULT_LEARNING_WINDOW_ISSUE_COUNT = DEFAULT_ISSUE_RETENTION_COUNT
MIN_LEARNING_WINDOW_ISSUE_COUNT = 1
MAX_LEARNING_WINDOW_ISSUE_COUNT = DEFAULT_ISSUE_RETENTION_COUNT
DEFAULT_TARGET_HIT_RATE = 0.75
DEFAULT_MIN_ACTION_SHARE = 0.50
DEFAULT_HANDICAP_TARGET_HIT_RATE = 0.70
DEFAULT_HANDICAP_MIN_ACTION_SHARE = 0.60
TARGET_STRATEGY_MIN_TRAIN_SAMPLES = 40
TARGET_STRATEGY_MIN_VALIDATION_SAMPLES = 10
EPSILON = 1e-6
TEMPERATURE_GRID = (0.75, 0.9, 1.0, 1.15, 1.3, 1.5)
BIAS_SCALE_GRID = (0.0, 0.5, 1.0, 1.5, 2.0)
BASE_THRESHOLD_CONFIG = {
    "main": {"ev": 0.10, "confidence": 0.62, "market_bias": 0.030, "quality": 0.68},
    "light": {"ev": 0.04, "confidence": 0.52, "market_bias": 0.015, "quality": 0.58},
    "promote": {"ev": 0.08, "confidence": 0.58, "market_bias": 0.030, "quality": 0.70},
}
TARGET_STRATEGY_DIRECTION_SOURCES = ("market", "model", "current", "algo", "low_odds", "ev")
TARGET_STRATEGY_SEARCH_SPACE = {
    "odds_max": (1.4, 1.5, 1.8, 2.0, 10.0),
    "prob_min": (0.0, 0.50, 0.60),
    "market_prob_min": (0.0, 0.55, 0.65),
    "quality_min": (0.0, 0.70, 0.80),
    "confidence_min": (0.0, 0.60),
    "ev_min": (-1.0, 0.0),
    "prob_margin_min": (-1.0, 0.0),
    "ev_margin_min": (-2.0, 0.0),
}
TARGET_STRATEGY_CATEGORY_FILTERS = (
    {},
    {"outcomes": ("home",)},
    {"outcomes": ("away",)},
    {"outcomes": ("home", "away")},
    {"llm_decisions": ("keep",)},
    {"llm_decisions": ("", "skipped")},
    {"arbiter_decisions": ("keep",)},
    {"arbiter_decisions": ("downgrade", "watch")},
    {"effective_sources": ("expert_llm", "arbiter", "manual")},
    {"algo_actions": ("轻仓", "主推", "杞讳粨", "涓绘帹")},
    {"algo_actions": ("观望", "瑙傛湜")},
)
TARGET_STRATEGY_CATEGORY_EXPANSION_LIMIT = 220
TARGET_STRATEGY_UNION_RULE_LIMIT = 60
TARGET_STRATEGY_UNION_CHILD_MIN_TRAIN_ACTIONS = 12
TARGET_STRATEGY_COMPLEMENT_RULE_LIMIT = 120
TARGET_STRATEGY_COMPLEMENT_CHILDREN_MAX = 4
TARGET_STRATEGY_MAX_CANDIDATE_RULES = 25000
TARGET_STRATEGY_STRUCTURAL_FILTERS = (
    {},
    {"rating_gap_min": -100.0},
    {"rating_gap_min": 0.0},
    {"rating_gap_min": 50.0},
    {"ppg_gap_min": -0.30},
    {"ppg_gap_min": 0.0},
    {"ppg_gap_min": 0.30},
    {"split_ppg_gap_min": -0.30},
    {"split_ppg_gap_min": 0.0},
    {"lineup_gap_min": -0.05},
    {"lineup_gap_min": 0.0},
    {"absence_gap_min": -0.05},
    {"absence_gap_min": 0.0},
    {"h2h_edge_min": -0.20},
    {"h2h_edge_min": 0.0},
    {"rating_gap_min": 0.0, "ppg_gap_min": 0.0},
    {"rating_gap_min": 0.0, "lineup_gap_min": 0.0},
    {"ppg_gap_min": 0.0, "split_ppg_gap_min": 0.0},
)
HANDICAP_TARGET_SEARCH_SPACE = {
    "ev_min": (-0.20, -0.05, 0.0, 0.03, 0.06),
    "confidence_min": (0.0, 0.52, 0.58, 0.62),
    "cover_prob_min": (0.0, 0.50, 0.54),
    "cover_margin_min": (-1.0, 0.0, 0.04),
    "quality_min": (0.0, 0.70),
    "odds_max": (1.8, 2.2, 10.0),
    "sides": ((), ("home",), ("away",)),
    "base_actions": ((), ("轻仓", "主推"), ("主推",)),
}
HANDICAP_BUCKET_STRATEGY_CANDIDATES = (
    (("line50", "coverdiff10", "evdiff10", "awayodds20"), 3, 0.55),
    (("line50", "coverdiff10", "evdiff10"), 3, 0.55),
    (("line25", "evdiff10"), 3, 0.55),
)


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _emit_progress(progress_callback, **payload) -> None:
    if progress_callback is None:
        return
    progress_callback(**payload)


def _row_value(row: Mapping[str, Any], key: str, default: Any = "") -> Any:
    try:
        value = row[key]
    except Exception:  # noqa: BLE001
        return default
    return default if value is None else value


def _parse_datetime_text(value: Any) -> datetime:
    text = str(value or "").strip()
    if not text:
        return datetime.min
    for pattern in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(text, pattern)
        except ValueError:
            continue
    return datetime.min


def _loads_json(text: Any, default: Any) -> Any:
    normalized = str(text or "").strip()
    if not normalized:
        return default
    try:
        return json.loads(normalized)
    except json.JSONDecodeError:
        return default


def _dumps_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _status_label(status: str) -> str:
    return PROFILE_STATUS_LABELS.get(status, status or "-")


def _component_status_label(status: str) -> str:
    return COMPONENT_STATUS_LABELS.get(status, status or "-")


def normalize_learning_window_issue_count(value: Any | None = None) -> int:
    try:
        count = int(value) if value not in (None, "") else DEFAULT_LEARNING_WINDOW_ISSUE_COUNT
    except (TypeError, ValueError):
        count = DEFAULT_LEARNING_WINDOW_ISSUE_COUNT
    return max(MIN_LEARNING_WINDOW_ISSUE_COUNT, min(count, MAX_LEARNING_WINDOW_ISSUE_COUNT))


def _normalize_probs(home: float, draw: float, away: float) -> dict[str, float]:
    values = [max(home, EPSILON), max(draw, EPSILON), max(away, EPSILON)]
    total = sum(values)
    return {
        "home": values[0] / total,
        "draw": values[1] / total,
        "away": values[2] / total,
    }


def apply_probability_calibration(
    probabilities: Mapping[str, Any],
    calibrator_params: Mapping[str, Any] | None,
) -> dict[str, float]:
    raw = _normalize_probs(
        safe_float(probabilities.get("home")),
        safe_float(probabilities.get("draw")),
        safe_float(probabilities.get("away")),
    )
    if not calibrator_params:
        return raw

    temperature = max(safe_float(calibrator_params.get("temperature"), 1.0), 0.35)
    biases = calibrator_params.get("biases", {}) if isinstance(calibrator_params, Mapping) else {}
    scores = []
    for outcome in OUTCOMES:
        bias = safe_float(biases.get(outcome))
        score = (math.log(max(raw[outcome], EPSILON)) + bias) / temperature
        scores.append(score)
    max_score = max(scores)
    exps = [math.exp(score - max_score) for score in scores]
    total = sum(exps) or 1.0
    return {
        "home": exps[0] / total,
        "draw": exps[1] / total,
        "away": exps[2] / total,
    }


def _prob_vector_from_row(
    row: Mapping[str, Any],
    calibrator_params: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    raw = {
        "home": safe_float(_row_value(row, "final_home_prob")),
        "draw": safe_float(_row_value(row, "final_draw_prob")),
        "away": safe_float(_row_value(row, "final_away_prob")),
    }
    return apply_probability_calibration(raw, calibrator_params)


def _probability_actual_index(actual_result: str) -> int:
    return {"home": 0, "draw": 1, "away": 2}.get(str(actual_result or "").strip(), 0)


def _brier_score(probabilities: Mapping[str, float], actual_result: str) -> float:
    actual_index = _probability_actual_index(actual_result)
    actual = [0.0, 0.0, 0.0]
    actual[actual_index] = 1.0
    predicted = [
        safe_float(probabilities["home"]),
        safe_float(probabilities["draw"]),
        safe_float(probabilities["away"]),
    ]
    return sum((predicted[idx] - actual[idx]) ** 2 for idx in range(3)) / 3.0


def _log_loss(probabilities: Mapping[str, float], actual_result: str) -> float:
    actual_index = _probability_actual_index(actual_result)
    predicted = [
        safe_float(probabilities["home"]),
        safe_float(probabilities["draw"]),
        safe_float(probabilities["away"]),
    ]
    return -math.log(max(predicted[actual_index], EPSILON))


def _hit_rate(probabilities: Mapping[str, float], actual_result: str) -> float:
    predicted = max(probabilities, key=probabilities.get)
    return 1.0 if predicted == actual_result else 0.0


def _aggregate_prob_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    count = max(len(items), 1)
    return {
        "sample_count": len(items),
        "brier_score": sum(item["brier"] for item in items) / count,
        "log_loss": sum(item["log_loss"] for item in items) / count,
        "hit_rate": sum(item["hit"] for item in items) / count,
    }


def _probability_metrics(
    rows: list[Mapping[str, Any]],
    *,
    calibrator_params: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    metrics = []
    for row in rows:
        actual_result = str(_row_value(row, "actual_result", "") or "")
        probabilities = _prob_vector_from_row(row, calibrator_params)
        metrics.append(
            {
                "brier": _brier_score(probabilities, actual_result),
                "log_loss": _log_loss(probabilities, actual_result),
                "hit": _hit_rate(probabilities, actual_result),
            }
        )
    return _aggregate_prob_metrics(metrics)


def _sorted_learning_rows(window_issue_count: Any | None = None) -> list[Mapping[str, Any]]:
    window_count = normalize_learning_window_issue_count(window_issue_count)
    rows = list_backtest_rows(limit=None)
    issue_groups = _rows_by_issue([dict(row) for row in rows])
    selected_issues = {
        issue
        for issue, _issue_rows in issue_groups[-window_count:]
    }
    if selected_issues:
        rows = [row for row in rows if str(_row_value(row, "issue", "") or "").strip() in selected_issues]
    return sorted(
        rows,
        key=lambda row: (
            _parse_datetime_text(_row_value(row, "created_at", "")),
            int(_row_value(row, "prediction_run_id", 0) or 0),
        ),
    )


def _issue_sort_key(issue: str) -> tuple[int, int | str]:
    try:
        return (0, int(issue))
    except ValueError:
        return (1, issue)


def _rows_by_issue(rows: list[Mapping[str, Any]]) -> list[tuple[str, list[Mapping[str, Any]]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        issue = str(_row_value(row, "issue", "") or "").strip()
        if issue:
            grouped.setdefault(issue, []).append(row)
    return [(issue, grouped[issue]) for issue in sorted(grouped, key=_issue_sort_key)]


def _flatten_issue_groups(groups: list[tuple[str, list[Mapping[str, Any]]]]) -> list[Mapping[str, Any]]:
    return [row for _, issue_rows in groups for row in issue_rows]


def _latest_issue_holdout_split(
    rows: list[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    groups = _rows_by_issue(rows)
    if len(groups) <= 1:
        return rows[:], []
    return _flatten_issue_groups(groups[:-1]), groups[-1][1][:]


def _actual_result_counts(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    counts = {outcome: 0 for outcome in OUTCOMES}
    for row in rows:
        outcome = str(_row_value(row, "actual_result", "") or "")
        if outcome in counts:
            counts[outcome] += 1
    return counts


def _resolved_effective_action(row: Mapping[str, Any]) -> tuple[str, float]:
    raw_action = str(_row_value(row, "effective_recommendation", "") or "").strip()
    if raw_action:
        return _action_label(raw_action), safe_float(_row_value(row, "effective_stake_pct"))

    manual_status = str(_row_value(row, "manual_review_status", "") or "")
    if manual_status in {"pending", "expired", "superseded"}:
        return "观望", 0.0

    return (
        _action_label(str(_row_value(row, "recommendation", "") or "")),
        safe_float(_row_value(row, "suggested_stake_pct")),
    )


def _count_actionable_rows(rows: list[Mapping[str, Any]]) -> int:
    count = 0
    for row in rows:
        action, stake_pct = _resolved_effective_action(row)
        if action != "观望" and stake_pct > 0:
            count += 1
    return count


def _sample_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "issue_range": ["", ""],
            "created_at_range": ["", ""],
            "class_counts": {outcome: 0 for outcome in OUTCOMES},
            "actionable_samples": 0,
        }

    issues = [str(_row_value(row, "issue", "") or "") for row in rows if str(_row_value(row, "issue", "") or "")]
    created_values = [str(_row_value(row, "created_at", "") or "") for row in rows if str(_row_value(row, "created_at", "") or "")]
    return {
        "issue_range": [issues[0] if issues else "", issues[-1] if issues else ""],
        "created_at_range": [created_values[0] if created_values else "", created_values[-1] if created_values else ""],
        "class_counts": _actual_result_counts(rows),
        "actionable_samples": _count_actionable_rows(rows),
    }


def _review_signal_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    sample_count = len(rows)
    llm_completed = 0
    llm_keep = 0
    llm_downgrade = 0
    llm_abstain = 0
    arbiter_completed = 0
    arbiter_actioned = 0
    expert_source = 0
    arbiter_source = 0
    for row in rows:
        llm_status = str(_row_value(row, "llm_review_status", "") or "")
        llm_decision = str(_row_value(row, "llm_review_decision", "") or "")
        arbiter_status = str(_row_value(row, "arbiter_review_status", "") or "")
        arbiter_decision = str(_row_value(row, "arbiter_review_decision", "") or "")
        source = str(_row_value(row, "effective_action_source", "") or "")
        if llm_status == "completed":
            llm_completed += 1
        if llm_decision == "keep":
            llm_keep += 1
        elif llm_decision == "downgrade":
            llm_downgrade += 1
        elif llm_decision == "abstain":
            llm_abstain += 1
        if arbiter_status == "completed":
            arbiter_completed += 1
        if arbiter_decision in {"allow", "downgrade", "skip", "manual_review"}:
            arbiter_actioned += 1
        if source == "expert_llm":
            expert_source += 1
        elif source == "arbiter":
            arbiter_source += 1

    def _share(count: int) -> float:
        return count / sample_count if sample_count else 0.0

    return {
        "sample_count": sample_count,
        "llm_completed": llm_completed,
        "llm_completed_share": _share(llm_completed),
        "llm_keep": llm_keep,
        "llm_downgrade": llm_downgrade,
        "llm_abstain": llm_abstain,
        "arbiter_completed": arbiter_completed,
        "arbiter_completed_share": _share(arbiter_completed),
        "arbiter_actioned": arbiter_actioned,
        "expert_source": expert_source,
        "expert_source_share": _share(expert_source),
        "arbiter_source": arbiter_source,
        "arbiter_source_share": _share(arbiter_source),
    }


def _fit_calibrator(train_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    total_samples = len(train_rows)
    class_counts = _actual_result_counts(train_rows)
    if total_samples < CALIBRATION_MIN_TOTAL_SAMPLES:
        return {
            "status": "insufficient_samples",
            "reason": f"已结算样本 {total_samples} < {CALIBRATION_MIN_TOTAL_SAMPLES}",
            "params": {},
            "train_metrics": {"baseline": _probability_metrics(train_rows), "candidate": _probability_metrics(train_rows)},
        }
    if min(class_counts.values()) < CALIBRATION_MIN_CLASS_SAMPLES:
        return {
            "status": "insufficient_samples",
            "reason": "训练段三向赛果分布不足，无法稳定校准",
            "params": {},
            "train_metrics": {"baseline": _probability_metrics(train_rows), "candidate": _probability_metrics(train_rows)},
        }

    average_predicted = {
        outcome: sum(_prob_vector_from_row(row)[outcome] for row in train_rows) / total_samples
        for outcome in OUTCOMES
    }
    empirical = {outcome: class_counts[outcome] / total_samples for outcome in OUTCOMES}
    base_biases = {
        outcome: math.log(max(empirical[outcome], EPSILON) / max(average_predicted[outcome], EPSILON))
        for outcome in OUTCOMES
    }
    mean_bias = sum(base_biases.values()) / len(OUTCOMES)
    centered_biases = {outcome: base_biases[outcome] - mean_bias for outcome in OUTCOMES}

    best_candidate: dict[str, Any] | None = None
    baseline_metrics = _probability_metrics(train_rows)
    for bias_scale, temperature in product(BIAS_SCALE_GRID, TEMPERATURE_GRID):
        params = {
            "temperature": temperature,
            "biases": {outcome: centered_biases[outcome] * bias_scale for outcome in OUTCOMES},
        }
        candidate_metrics = _probability_metrics(train_rows, calibrator_params=params)
        if best_candidate is None or candidate_metrics["log_loss"] < best_candidate["train_metrics"]["candidate"]["log_loss"]:
            best_candidate = {
                "status": "ready",
                "reason": "",
                "params": params,
                "train_metrics": {
                    "baseline": baseline_metrics,
                    "candidate": candidate_metrics,
                },
            }

    return best_candidate or {
        "status": "no_gain",
        "reason": "未找到可用校准器",
        "params": {},
        "train_metrics": {"baseline": baseline_metrics, "candidate": baseline_metrics},
    }


def _validate_calibrator(
    validation_rows: list[Mapping[str, Any]],
    calibrator_fit: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_metrics = _probability_metrics(validation_rows)
    if calibrator_fit.get("status") != "ready":
        return {
            "status": str(calibrator_fit.get("status", "not_ready") or "not_ready"),
            "reason": str(calibrator_fit.get("reason", "") or ""),
            "baseline": baseline_metrics,
            "candidate": baseline_metrics,
        }

    params = calibrator_fit.get("params", {})
    candidate_metrics = _probability_metrics(validation_rows, calibrator_params=params)
    if not validation_rows:
        return {
            "status": "insufficient_samples",
            "reason": "验证样本为空，暂不启用校准器",
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
        }
    if candidate_metrics["log_loss"] + 1e-9 < baseline_metrics["log_loss"] and candidate_metrics["brier_score"] <= baseline_metrics["brier_score"] + 1e-9:
        return {
            "status": "ready",
            "reason": "",
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
        }
    return {
        "status": "no_gain",
        "reason": "验证段未同时满足 LogLoss 改善且 Brier 不变差",
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
    }


def _action_label(value: str) -> str:
    action = str(value or "").strip()
    if not action:
        return "观望"
    token = action.split()[0].lower()
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
    return aliases.get(token, action.split()[0])


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


def _market_odds(row: Mapping[str, Any]) -> dict[str, float]:
    return {
        "home": safe_float(_row_value(row, "market_odds_home")),
        "draw": safe_float(_row_value(row, "market_odds_draw")),
        "away": safe_float(_row_value(row, "market_odds_away")),
    }


def _market_probs(row: Mapping[str, Any]) -> dict[str, float]:
    return {
        "home": safe_float(_row_value(row, "market_home_prob")),
        "draw": safe_float(_row_value(row, "market_draw_prob")),
        "away": safe_float(_row_value(row, "market_away_prob")),
    }


def _stake_for_action(action: str, expected_values: Mapping[str, float], confidence: float, recommended_outcome: str) -> float:
    return stake_for_action(
        action,
        {
            "recommended_outcome": recommended_outcome,
            "expected_values": expected_values,
            "confidence": confidence,
        },
    )


def _risk_level(confidence: float) -> str:
    if confidence >= 0.78:
        return "low"
    if confidence >= 0.62:
        return "medium"
    if confidence >= 0.48:
        return "high"
    return "very_high"


def _build_algo_risk(
    row: Mapping[str, Any],
    probabilities: Mapping[str, float],
    threshold_config: Mapping[str, Any],
) -> dict[str, Any]:
    market_probs = _market_probs(row)
    market_odds = _market_odds(row)
    legacy_probabilities = {
        "home": safe_float(_row_value(row, "legacy_home_prob")),
        "draw": safe_float(_row_value(row, "legacy_draw_prob")),
        "away": safe_float(_row_value(row, "legacy_away_prob")),
    }
    sorted_probs = sorted(probabilities.values(), reverse=True)
    margin = sorted_probs[0] - sorted_probs[1]
    outcome_policy = evaluate_outcome_policy(
        probabilities=probabilities,
        market_odds=market_odds,
        market_probs=market_probs,
        legacy_probabilities=legacy_probabilities,
        quality_score=safe_float(_row_value(row, "quality_score")),
        model_agreement=safe_float(_row_value(row, "model_agreement")),
    )
    policy = evaluate_action_policy(
        probabilities=probabilities,
        market_odds=market_odds,
        market_probs=market_probs,
        legacy_probabilities=legacy_probabilities,
        quality_score=safe_float(_row_value(row, "quality_score")),
        model_agreement=safe_float(_row_value(row, "model_agreement")),
        model_margin=margin,
        threshold_config=threshold_config,
        selected_outcome=str(outcome_policy.get("recommended_outcome", "") or ""),
    )

    return {
        "market_probs": market_probs,
        "market_odds": market_odds,
        "probabilities": dict(probabilities),
        "expected_values": policy["expected_values"],
        "market_bias": policy["market_bias"],
        "confidence": policy["confidence"],
        "risk_level": policy["risk_level"],
        "recommended_outcome": policy["recommended_outcome"],
        "predicted_outcome": outcome_policy["predicted_outcome"],
        "value_outcome": outcome_policy["value_outcome"],
        "outcome_source": outcome_policy["outcome_source"],
        "outcome_score": outcome_policy["outcome_score"],
        "recommendation": policy["recommendation"],
        "stake_pct": policy["stake_pct"],
        "action_score": policy["action_score"],
        "probability_margin": policy["probability_margin"],
        "ev_margin": policy["ev_margin"],
        "legacy_gap": policy["legacy_gap"],
        "kelly_fraction": policy["kelly_fraction"],
    }


def _promotion_constraints_pass(
    target_action: str,
    algo_risk: Mapping[str, Any],
    threshold_config: Mapping[str, Any],
    quality_score: float,
) -> bool:
    current_action = _action_label(str(algo_risk.get("recommendation", "")))
    if current_action == "观望" and target_action == "主推":
        return False
    promote_gate = threshold_config["promote"]
    outcome = str(algo_risk.get("recommended_outcome", "") or "")
    expected_values = algo_risk.get("expected_values", {})
    market_bias = algo_risk.get("market_bias", {})
    return (
        quality_score >= safe_float(promote_gate["quality"])
        and safe_float(algo_risk.get("confidence")) >= safe_float(promote_gate["confidence"])
        and safe_float(expected_values.get(outcome)) >= safe_float(promote_gate["ev"])
        and safe_float(market_bias.get(outcome)) >= safe_float(promote_gate["market_bias"])
        and str(algo_risk.get("risk_level", "") or "") != "very_high"
    )


def _build_final_risk(
    row: Mapping[str, Any],
    algo_risk: Mapping[str, Any],
    threshold_config: Mapping[str, Any],
) -> dict[str, Any]:
    final_action = _action_label(str(algo_risk.get("recommendation", "")))
    status = str(_row_value(row, "llm_review_status", "") or "")
    confidence_delta = 0.0
    stake_multiplier = 1.0
    if status == "completed":
        decision = str(_row_value(row, "llm_review_decision", "") or "")
        target_action = _action_label(str(_row_value(row, "llm_review_target_action", "") or final_action))
        if decision == "downgrade":
            final_action = _downgrade_action(final_action, target_action)
        elif decision == "promote":
            if _promotion_constraints_pass(
                target_action,
                algo_risk,
                threshold_config,
                safe_float(_row_value(row, "quality_score")),
            ):
                final_action = _promote_action(final_action, target_action)
        elif decision == "abstain":
            final_action = "观望"

        review_payload = _loads_json(_row_value(row, "llm_review_raw", ""), {})
        if isinstance(review_payload, Mapping):
            confidence_delta = clamp(safe_float(review_payload.get("confidence_delta")), -0.12, 0.08)
            stake_multiplier = clamp(safe_float(review_payload.get("stake_multiplier"), 1.0), 0.0, 1.0)
            evidence_grade = str(review_payload.get("evidence_grade", "") or "").strip().lower()
            if evidence_grade == "unsafe" or stake_multiplier <= 0:
                final_action = "观望"
            elif evidence_grade == "weak" and final_action == "主推":
                final_action = "轻仓"

    adjusted_confidence = clamp(
        safe_float(algo_risk.get("confidence")) + confidence_delta,
        0.18,
        0.88,
    )
    if status == "completed":
        if final_action == "主推" and adjusted_confidence < safe_float(threshold_config["main"]["confidence"]):
            final_action = "轻仓"
        if final_action == "轻仓" and adjusted_confidence < safe_float(threshold_config["light"]["confidence"]):
            final_action = "观望"

    recommended_outcome = str(algo_risk.get("recommended_outcome", "") or "")
    risk_context = dict(algo_risk)
    risk_context["confidence"] = adjusted_confidence
    stake_pct = stake_for_action(final_action, risk_context)
    if status == "completed" and final_action != "观望":
        stake_pct = round(stake_pct * stake_multiplier, 2)
    return {
        "recommendation": final_action,
        "recommended_outcome": recommended_outcome,
        "stake_pct": 0.0 if final_action == "观望" else stake_pct,
    }


def _roi_for_action(
    *,
    action: str,
    recommended_outcome: str,
    stake_pct: float,
    actual_result: str,
    market_odds: Mapping[str, float],
) -> float:
    normalized_action = _action_label(action)
    if normalized_action == "观望" or stake_pct <= 0:
        return 0.0
    units = stake_pct / 100.0
    if recommended_outcome == actual_result:
        odds = safe_float(market_odds.get(recommended_outcome))
        if odds <= 0:
            return 0.0
        return round(units * (odds - 1.0), 4)
    return round(-units, 4)


def _empty_action_metrics() -> dict[str, float | int]:
    return {
        "action_count": 0,
        "hit_rate": 0.0,
        "total_roi": 0.0,
        "avg_roi": 0.0,
        "avg_stake_pct": 0.0,
    }


def _empty_target_strategy_metrics(sample_count: int = 0) -> dict[str, Any]:
    return {
        "sample_count": sample_count,
        "action_count": 0,
        "watch_count": sample_count,
        "hit_count": 0,
        "hit_rate": 0.0,
        "action_share": 0.0,
        "watch_share": 1.0 if sample_count else 0.0,
        "total_roi": 0.0,
        "avg_roi": 0.0,
        "avg_stake_pct": 0.0,
        "buckets": {},
    }


def _combine_target_strategy_metrics(items: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not items:
        return _empty_target_strategy_metrics()
    sample_count = sum(int(item.get("sample_count", 0) or 0) for item in items)
    action_count = sum(int(item.get("action_count", 0) or 0) for item in items)
    hit_count = sum(int(item.get("hit_count", 0) or 0) for item in items)
    total_roi = sum(safe_float(item.get("total_roi")) for item in items)
    total_stake_pct = sum(
        safe_float(item.get("avg_stake_pct")) * int(item.get("action_count", 0) or 0)
        for item in items
    )
    buckets: dict[str, dict[str, float | int]] = {}
    for item in items:
        for key, bucket in dict(item.get("buckets", {}) or {}).items():
            aggregate = buckets.setdefault(
                str(key),
                {"sample_count": 0, "action_count": 0, "hits": 0, "total_roi": 0.0},
            )
            aggregate["sample_count"] += int(bucket.get("sample_count", 0) or 0)
            aggregate["action_count"] += int(bucket.get("action_count", 0) or 0)
            aggregate["hits"] += int(bucket.get("hits", 0) or 0)
            aggregate["total_roi"] += safe_float(bucket.get("total_roi"))
    action_share = action_count / sample_count if sample_count else 0.0
    return {
        "sample_count": sample_count,
        "action_count": action_count,
        "watch_count": max(sample_count - action_count, 0),
        "hit_count": hit_count,
        "hit_rate": hit_count / action_count if action_count else 0.0,
        "action_share": action_share,
        "watch_share": 1.0 - action_share if sample_count else 0.0,
        "total_roi": round(total_roi, 4),
        "avg_roi": total_roi / action_count if action_count else 0.0,
        "avg_stake_pct": total_stake_pct / action_count if action_count else 0.0,
        "buckets": buckets,
    }


def _replay_threshold_metrics(
    rows: list[Mapping[str, Any]],
    threshold_config: Mapping[str, Any],
    calibrator_params: Mapping[str, Any] | None,
) -> dict[str, dict[str, float | int]]:
    aggregates = {
        "algorithm": {"action_count": 0, "hits": 0, "total_roi": 0.0, "total_stake_pct": 0.0},
        "final": {"action_count": 0, "hits": 0, "total_roi": 0.0, "total_stake_pct": 0.0},
    }

    for row in rows:
        probabilities = _prob_vector_from_row(row, calibrator_params)
        algo_risk = _build_algo_risk(row, probabilities, threshold_config)
        final_risk = _build_final_risk(row, algo_risk, threshold_config)
        market_odds = _market_odds(row)
        actual_result = str(_row_value(row, "actual_result", "") or "")

        for bucket_name, action_payload in (("algorithm", algo_risk), ("final", final_risk)):
            action = _action_label(str(action_payload.get("recommendation", "")))
            stake_pct = safe_float(action_payload.get("stake_pct"))
            if action == "观望" or stake_pct <= 0:
                continue
            aggregates[bucket_name]["action_count"] += 1
            aggregates[bucket_name]["total_stake_pct"] += stake_pct
            if str(action_payload.get("recommended_outcome", "") or "") == actual_result:
                aggregates[bucket_name]["hits"] += 1
            aggregates[bucket_name]["total_roi"] += _roi_for_action(
                action=action,
                recommended_outcome=str(action_payload.get("recommended_outcome", "") or ""),
                stake_pct=stake_pct,
                actual_result=actual_result,
                market_odds=market_odds,
            )

    metrics: dict[str, dict[str, float | int]] = {}
    for bucket_name, values in aggregates.items():
        action_count = int(values["action_count"])
        if action_count <= 0:
            metrics[bucket_name] = _empty_action_metrics()
            continue
        total_roi = safe_float(values["total_roi"])
        metrics[bucket_name] = {
            "action_count": action_count,
            "hit_rate": safe_float(values["hits"]) / action_count,
            "total_roi": total_roi,
            "avg_roi": total_roi / action_count,
            "avg_stake_pct": safe_float(values["total_stake_pct"]) / action_count,
        }
    return metrics


def _target_strategy_outcome(
    row: Mapping[str, Any],
    probabilities: Mapping[str, float],
    rule: Mapping[str, Any],
) -> str:
    source = str(rule.get("direction_source", "market") or "market")
    market_probs = _market_probs(row)
    market_odds = _market_odds(row)
    ev_values = expected_values_for(probabilities, market_odds)
    if source == "model":
        return max(probabilities, key=probabilities.get)
    if source == "current":
        current = str(_row_value(row, "recommended_outcome", "") or "")
        return current if current in OUTCOMES else max(market_probs, key=market_probs.get)
    if source == "algo":
        algo = str(_row_value(row, "algo_recommended_outcome", "") or "")
        return algo if algo in OUTCOMES else max(market_probs, key=market_probs.get)
    if source == "low_odds":
        return min(
            OUTCOMES,
            key=lambda outcome: safe_float(market_odds.get(outcome)) if safe_float(market_odds.get(outcome)) > 0 else 99.0,
        )
    if source == "ev":
        return max(ev_values, key=ev_values.get)
    return max(market_probs, key=market_probs.get)


def _target_strategy_features(
    row: Mapping[str, Any],
    probabilities: Mapping[str, float],
    outcome: str,
) -> dict[str, float]:
    market_probs = _market_probs(row)
    market_odds = _market_odds(row)
    ev_values = expected_values_for(probabilities, market_odds)
    probability = safe_float(probabilities.get(outcome))
    ev_value = safe_float(ev_values.get(outcome))
    alternative_probabilities = [
        safe_float(probabilities.get(item))
        for item in OUTCOMES
        if item != outcome
    ]
    alternative_evs = [
        safe_float(ev_values.get(item))
        for item in OUTCOMES
        if item != outcome
    ]
    home_rating = safe_float(_row_value(row, "home_rating"))
    away_rating = safe_float(_row_value(row, "away_rating"))
    home_ppg = safe_float(_row_value(row, "recent_home_ppg"))
    away_ppg = safe_float(_row_value(row, "recent_away_ppg"))
    home_gf = safe_float(_row_value(row, "recent_home_gf_pg"))
    away_gf = safe_float(_row_value(row, "recent_away_gf_pg"))
    home_ga = safe_float(_row_value(row, "recent_home_ga_pg"))
    away_ga = safe_float(_row_value(row, "recent_away_ga_pg"))
    home_split_ppg = safe_float(_row_value(row, "home_split_ppg"))
    away_split_ppg = safe_float(_row_value(row, "away_split_ppg"))
    home_absence_impact = safe_float(_row_value(row, "home_absence_impact"))
    away_absence_impact = safe_float(_row_value(row, "away_absence_impact"))
    home_lineup = safe_float(_row_value(row, "lineup_home_availability"))
    away_lineup = safe_float(_row_value(row, "lineup_away_availability"))
    home_rest = safe_float(_row_value(row, "rest_days_home"))
    away_rest = safe_float(_row_value(row, "rest_days_away"))
    home_load = safe_float(_row_value(row, "schedule_load_home"))
    away_load = safe_float(_row_value(row, "schedule_load_away"))
    h2h_edge = safe_float(_row_value(row, "h2h_edge"))
    predicted_score = str(_row_value(row, "predicted_score", "") or "")
    score_numbers = [int(item) for item in re.findall(r"\d+", predicted_score)[:2]]
    score_outcome = ""
    if len(score_numbers) == 2:
        if score_numbers[0] > score_numbers[1]:
            score_outcome = "home"
        elif score_numbers[0] == score_numbers[1]:
            score_outcome = "draw"
        else:
            score_outcome = "away"
    feature_ready = 1.0 if home_rating > 0 and away_rating > 0 else 0.0
    if outcome == "home":
        rating_gap = home_rating - away_rating
        ppg_gap = home_ppg - away_ppg
        split_ppg_gap = home_split_ppg - away_split_ppg
        goal_for_gap = home_gf - away_gf
        goal_against_gap = away_ga - home_ga
        absence_gap = away_absence_impact - home_absence_impact
        lineup_gap = home_lineup - away_lineup
        rest_gap = home_rest - away_rest
        schedule_gap = away_load - home_load
        h2h_for_outcome = h2h_edge
    elif outcome == "away":
        rating_gap = away_rating - home_rating
        ppg_gap = away_ppg - home_ppg
        split_ppg_gap = away_split_ppg - home_split_ppg
        goal_for_gap = away_gf - home_gf
        goal_against_gap = home_ga - away_ga
        absence_gap = home_absence_impact - away_absence_impact
        lineup_gap = away_lineup - home_lineup
        rest_gap = away_rest - home_rest
        schedule_gap = home_load - away_load
        h2h_for_outcome = -h2h_edge
    else:
        rating_gap = -abs(home_rating - away_rating)
        ppg_gap = -abs(home_ppg - away_ppg)
        split_ppg_gap = -abs(home_split_ppg - away_split_ppg)
        goal_for_gap = -abs(home_gf - away_gf)
        goal_against_gap = -abs(home_ga - away_ga)
        absence_gap = -abs(home_absence_impact - away_absence_impact)
        lineup_gap = -abs(home_lineup - away_lineup)
        rest_gap = -abs(home_rest - away_rest)
        schedule_gap = -abs(home_load - away_load)
        h2h_for_outcome = -abs(h2h_edge)
    return {
        "outcome": outcome,
        "probability": probability,
        "market_probability": safe_float(market_probs.get(outcome)),
        "odds": safe_float(market_odds.get(outcome)),
        "ev": ev_value,
        "probability_margin": probability - max(alternative_probabilities or [0.0]),
        "ev_margin": ev_value - max(alternative_evs or [-1.0]),
        "quality_score": safe_float(_row_value(row, "quality_score")),
        "model_agreement": safe_float(_row_value(row, "model_agreement")),
        "confidence_score": safe_float(_row_value(row, "confidence_score")),
        "score_outcome": score_outcome,
        "score_confidence": safe_float(_row_value(row, "predicted_score_confidence")),
        "llm_decision": str(_row_value(row, "llm_review_decision", "") or ""),
        "llm_status": str(_row_value(row, "llm_review_status", "") or ""),
        "arbiter_decision": str(_row_value(row, "arbiter_review_decision", "") or ""),
        "arbiter_status": str(_row_value(row, "arbiter_review_status", "") or ""),
        "effective_source": str(_row_value(row, "effective_action_source", "") or ""),
        "algo_action": _normalized_action(_row_value(row, "algo_recommendation", "")),
        "final_action": _normalized_action(_row_value(row, "recommendation", "")),
        "feature_ready": feature_ready,
        "rating_gap": rating_gap,
        "abs_rating_gap": abs(home_rating - away_rating),
        "ppg_gap": ppg_gap,
        "split_ppg_gap": split_ppg_gap,
        "goal_for_gap": goal_for_gap,
        "goal_against_gap": goal_against_gap,
        "absence_gap": absence_gap,
        "lineup_gap": lineup_gap,
        "rest_gap": rest_gap,
        "schedule_gap": schedule_gap,
        "h2h_edge": h2h_for_outcome,
    }


def _normalized_action(value: Any) -> str:
    action = str(value or "").strip()
    if action in {"轻仓", "杞讳粨"}:
        return "轻仓"
    if action in {"主推", "涓绘帹"}:
        return "主推"
    if action in {"观望", "瑙傛湜"}:
        return "观望"
    return action


def _target_strategy_passes(features: Mapping[str, Any], rule: Mapping[str, Any]) -> bool:
    any_rules = rule.get("any_rules")
    if isinstance(any_rules, list) and any_rules:
        return any(
            isinstance(item, Mapping) and _target_strategy_passes(features, item)
            for item in any_rules
        )
    odds = safe_float(features.get("odds"))
    if not (
        odds > 0
        and odds <= safe_float(rule.get("odds_max"), 10.0)
        and safe_float(features.get("probability")) >= safe_float(rule.get("prob_min"))
        and safe_float(features.get("market_probability")) >= safe_float(rule.get("market_prob_min"))
        and safe_float(features.get("quality_score")) >= safe_float(rule.get("quality_min"))
        and safe_float(features.get("model_agreement")) >= safe_float(rule.get("agreement_min"))
        and safe_float(features.get("confidence_score")) >= safe_float(rule.get("confidence_min"))
        and safe_float(features.get("ev")) >= safe_float(rule.get("ev_min"), -1.0)
        and safe_float(features.get("probability_margin")) >= safe_float(rule.get("prob_margin_min"), -1.0)
        and safe_float(features.get("ev_margin")) >= safe_float(rule.get("ev_margin_min"), -2.0)
    ):
        return False
    outcomes = rule.get("outcomes", ())
    if outcomes and str(features.get("outcome", "")) not in set(outcomes):
        return False
    score_outcomes = rule.get("score_outcomes", ())
    if score_outcomes and str(features.get("score_outcome", "")) not in set(score_outcomes):
        return False
    if safe_float(features.get("score_confidence")) < safe_float(rule.get("score_confidence_min")):
        return False
    llm_decisions = rule.get("llm_decisions", ())
    if llm_decisions and str(features.get("llm_decision", "")) not in set(llm_decisions):
        return False
    arbiter_decisions = rule.get("arbiter_decisions", ())
    if arbiter_decisions and str(features.get("arbiter_decision", "")) not in set(arbiter_decisions):
        return False
    effective_sources = rule.get("effective_sources", ())
    if effective_sources and str(features.get("effective_source", "")) not in set(effective_sources):
        return False
    algo_actions = rule.get("algo_actions", ())
    if algo_actions and _normalized_action(features.get("algo_action")) not in {
        _normalized_action(item) for item in algo_actions
    }:
        return False
    final_actions = rule.get("final_actions", ())
    if final_actions and _normalized_action(features.get("final_action")) not in {
        _normalized_action(item) for item in final_actions
    }:
        return False
    structural_thresholds = {
        "rating_gap_min": "rating_gap",
        "ppg_gap_min": "ppg_gap",
        "split_ppg_gap_min": "split_ppg_gap",
        "goal_for_gap_min": "goal_for_gap",
        "goal_against_gap_min": "goal_against_gap",
        "absence_gap_min": "absence_gap",
        "lineup_gap_min": "lineup_gap",
        "rest_gap_min": "rest_gap",
        "schedule_gap_min": "schedule_gap",
        "h2h_edge_min": "h2h_edge",
    }
    for rule_key, feature_key in structural_thresholds.items():
        if rule_key in rule and safe_float(features.get(feature_key), -999.0) < safe_float(rule.get(rule_key), -999.0):
            return False
    return True


def evaluate_target_strategy_rule(
    rows: list[Mapping[str, Any]],
    rule: Mapping[str, Any],
    *,
    calibrator_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    sample_count = len(rows)
    action_count = 0
    hits = 0
    total_roi = 0.0
    total_stake_pct = 0.0
    buckets: dict[str, dict[str, float | int]] = {}
    for row in rows:
        probabilities = _prob_vector_from_row(row, calibrator_params)
        any_rules = rule.get("any_rules")
        if isinstance(any_rules, list) and any_rules:
            action = "瑙傛湜"
            outcome = "home"
            fallback_rule = any_rules[0] if isinstance(any_rules[0], Mapping) else {"direction_source": "market"}
            fallback_outcome = _target_strategy_outcome(row, probabilities, fallback_rule)
            if fallback_outcome in OUTCOMES:
                outcome = fallback_outcome
            for child_rule in any_rules:
                if not isinstance(child_rule, Mapping):
                    continue
                child_outcome = _target_strategy_outcome(row, probabilities, child_rule)
                if child_outcome not in OUTCOMES:
                    continue
                child_features = _target_strategy_features(row, probabilities, child_outcome)
                if _target_strategy_passes(child_features, child_rule):
                    action = "杞讳粨"
                    outcome = child_outcome
                    break

            actual_result = str(_row_value(row, "actual_result", "") or "")
            hit = 1 if outcome == actual_result else 0
            stake_pct = safe_float(rule.get("stake_pct", 1.0)) if action != "瑙傛湜" else 0.0
            roi = _roi_for_action(
                action=action,
                recommended_outcome=outcome,
                stake_pct=stake_pct,
                actual_result=actual_result,
                market_odds=_market_odds(row),
            )
            bucket = buckets.setdefault(
                f"{action}:{outcome}",
                {"sample_count": 0, "action_count": 0, "hits": 0, "total_roi": 0.0},
            )
            bucket["sample_count"] += 1
            if action != "瑙傛湜":
                action_count += 1
                hits += hit
                total_roi += roi
                total_stake_pct += stake_pct
                bucket["action_count"] += 1
                bucket["hits"] += hit
                bucket["total_roi"] += roi
            continue
        outcome = _target_strategy_outcome(row, probabilities, rule)
        if outcome not in OUTCOMES:
            continue
        features = _target_strategy_features(row, probabilities, outcome)
        action = "轻仓" if _target_strategy_passes(features, rule) else "观望"
        actual_result = str(_row_value(row, "actual_result", "") or "")
        hit = 1 if outcome == actual_result else 0
        stake_pct = safe_float(rule.get("stake_pct", 1.0)) if action != "观望" else 0.0
        roi = _roi_for_action(
            action=action,
            recommended_outcome=outcome,
            stake_pct=stake_pct,
            actual_result=actual_result,
            market_odds=_market_odds(row),
        )
        bucket = buckets.setdefault(
            f"{action}:{outcome}",
            {"sample_count": 0, "action_count": 0, "hits": 0, "total_roi": 0.0},
        )
        bucket["sample_count"] += 1
        if action != "观望":
            action_count += 1
            hits += hit
            total_roi += roi
            total_stake_pct += stake_pct
            bucket["action_count"] += 1
            bucket["hits"] += hit
            bucket["total_roi"] += roi

    action_share = action_count / sample_count if sample_count else 0.0
    hit_rate = hits / action_count if action_count else 0.0
    return {
        "sample_count": sample_count,
        "action_count": action_count,
        "watch_count": max(sample_count - action_count, 0),
        "hit_count": hits,
        "hit_rate": hit_rate,
        "action_share": action_share,
        "watch_share": 1.0 - action_share if sample_count else 0.0,
        "total_roi": round(total_roi, 4),
        "avg_roi": total_roi / action_count if action_count else 0.0,
        "avg_stake_pct": total_stake_pct / action_count if action_count else 0.0,
        "buckets": buckets,
    }


def target_strategy_status(
    metrics: Mapping[str, Any],
    *,
    target_hit_rate: float = DEFAULT_TARGET_HIT_RATE,
    min_action_share: float = DEFAULT_MIN_ACTION_SHARE,
) -> tuple[str, str]:
    sample_count = int(metrics.get("sample_count", 0) or 0)
    action_count = int(metrics.get("action_count", 0) or 0)
    if sample_count < TARGET_STRATEGY_MIN_VALIDATION_SAMPLES:
        return "insufficient_samples", f"validation samples {sample_count} < {TARGET_STRATEGY_MIN_VALIDATION_SAMPLES}"
    if action_count <= 0:
        return "target_unreachable", "no executable validation actions"
    hit_rate = safe_float(metrics.get("hit_rate"))
    action_share = safe_float(metrics.get("action_share"))
    if hit_rate >= target_hit_rate and action_share >= min_action_share:
        return "ready", "target satisfied on validation"
    return (
        "target_unreachable",
        f"validation hit {hit_rate * 100:.1f}% / action {action_share * 100:.1f}% below target",
    )


def _target_strategy_score(metrics: Mapping[str, Any]) -> tuple[float, float, float, float]:
    hit_rate = safe_float(metrics.get("hit_rate"))
    action_share = safe_float(metrics.get("action_share"))
    hit_gap = max(DEFAULT_TARGET_HIT_RATE - hit_rate, 0.0)
    action_gap = max(DEFAULT_MIN_ACTION_SHARE - action_share, 0.0)
    return (
        -(hit_gap * 2.0 + action_gap),
        hit_rate,
        action_share,
        safe_float(metrics.get("total_roi")),
    )


def _target_strategy_gap_summary(
    metrics: Mapping[str, Any],
    *,
    target_hit_rate: float = DEFAULT_TARGET_HIT_RATE,
    min_action_share: float = DEFAULT_MIN_ACTION_SHARE,
) -> dict[str, float]:
    hit_rate = safe_float(metrics.get("hit_rate"))
    action_share = safe_float(metrics.get("action_share"))
    return {
        "hit_rate_gap": max(target_hit_rate - hit_rate, 0.0),
        "action_share_gap": max(min_action_share - action_share, 0.0),
        "watch_share_gap": max(safe_float(metrics.get("watch_share")) - (1.0 - min_action_share), 0.0),
    }


def _target_strategy_candidate_summary(candidate: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(candidate, Mapping):
        return {}
    return {
        "params": candidate.get("params", {}),
        "train_metrics": candidate.get("train_metrics", {}),
        "validation_metrics": candidate.get("validation_metrics", {}),
        "reason": str(candidate.get("reason", "") or ""),
    }


def _target_strategy_rules() -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    space = TARGET_STRATEGY_SEARCH_SPACE
    for (
        direction_source,
        odds_max,
        prob_min,
        market_prob_min,
        quality_min,
        confidence_min,
        ev_min,
        prob_margin_min,
        ev_margin_min,
    ) in product(
        TARGET_STRATEGY_DIRECTION_SOURCES,
        space["odds_max"],
        space["prob_min"],
        space["market_prob_min"],
        space["quality_min"],
        space["confidence_min"],
        space["ev_min"],
        space["prob_margin_min"],
        space["ev_margin_min"],
    ):
        rules.append(
            {
                "direction_source": direction_source,
                "action": "轻仓",
                "stake_pct": 1.0,
                "odds_max": odds_max,
                "prob_min": prob_min,
                "market_prob_min": market_prob_min,
                "quality_min": quality_min,
                "agreement_min": 0.0,
                "confidence_min": confidence_min,
                "ev_min": ev_min,
                "prob_margin_min": prob_margin_min,
                "ev_margin_min": ev_margin_min,
                "outcomes": (),
                "llm_decisions": (),
                "algo_actions": (),
            }
        )
    return rules


def _target_strategy_rule_key(rule: Mapping[str, Any]) -> str:
    def _normalize(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: _normalize(value[key]) for key in sorted(value)}
        if isinstance(value, (set, tuple, list)):
            normalized_items = [_normalize(item) for item in value]
            return sorted(normalized_items, key=lambda item: _dumps_json(item))
        return value

    return _dumps_json(_normalize(rule))


def _target_strategy_action_hits(items: list[dict[str, Any]], rule: Mapping[str, Any]) -> dict[int, int]:
    source = str(rule.get("direction_source", "market") or "market")
    hits: dict[int, int] = {}
    for item in items:
        if item["source"] != source:
            continue
        if not _target_strategy_passes(item["features"], rule):
            continue
        row_index = int(item.get("row_index", 0))
        hits[row_index] = 1 if item["outcome"] == item["actual_result"] else 0
    return hits


def _target_strategy_combo_rule(children: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "direction_source": children[0].get("direction_source", "market") if children else "market",
        "action": "杞讳粨",
        "stake_pct": 1.0,
        "any_rules": children,
    }


def _target_strategy_complement_rules(
    seed_rules: list[dict[str, Any]],
    train_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    indexed: list[tuple[tuple[float, float, int, int], dict[str, Any], dict[int, int]]] = []
    for rule in seed_rules[:TARGET_STRATEGY_COMPLEMENT_RULE_LIMIT]:
        hits_by_row = _target_strategy_action_hits(train_items, rule)
        action_count = len(hits_by_row)
        if action_count < TARGET_STRATEGY_UNION_CHILD_MIN_TRAIN_ACTIONS:
            continue
        hit_count = sum(hits_by_row.values())
        hit_rate = hit_count / action_count if action_count else 0.0
        indexed.append(((hit_rate, action_count / max(len({int(item.get("row_index", 0)) for item in train_items}), 1), action_count, hit_count), rule, hits_by_row))
    indexed.sort(key=lambda item: item[0], reverse=True)

    for _score, base_rule, base_hits in indexed[:TARGET_STRATEGY_UNION_RULE_LIMIT]:
        children = [base_rule]
        covered = dict(base_hits)
        used = {_target_strategy_rule_key(base_rule)}
        for _depth in range(TARGET_STRATEGY_COMPLEMENT_CHILDREN_MAX - 1):
            best_next: tuple[float, dict[str, Any], dict[int, int]] | None = None
            for _candidate_score, candidate_rule, candidate_hits in indexed:
                key = _target_strategy_rule_key(candidate_rule)
                if key in used:
                    continue
                added = {idx: hit for idx, hit in candidate_hits.items() if idx not in covered}
                if not added:
                    continue
                added_count = len(added)
                added_hit_rate = sum(added.values()) / added_count
                combined_actions = len(covered) + added_count
                combined_hits = sum(covered.values()) + sum(added.values())
                combined_hit_rate = combined_hits / combined_actions if combined_actions else 0.0
                score = combined_hit_rate * 3.0 + added_hit_rate + min(added_count / 20.0, 1.0)
                if combined_hit_rate < 0.72:
                    continue
                if best_next is None or score > best_next[0]:
                    best_next = (score, candidate_rule, added)
            if best_next is None:
                break
            _score_value, next_rule, added_hits = best_next
            children.append(next_rule)
            used.add(_target_strategy_rule_key(next_rule))
            covered.update(added_hits)
            if len(children) >= 2:
                candidates.append(_target_strategy_combo_rule(list(children)))
    return candidates


def _expand_target_strategy_rules(base_rules: list[dict[str, Any]], train_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored: list[tuple[tuple[float, float, float, float], dict[str, Any]]] = []
    for rule in base_rules:
        metrics = _target_strategy_metrics_from_items(train_items, rule)
        if int(metrics["action_count"]) < TARGET_STRATEGY_MIN_TRAIN_SAMPLES:
            continue
        scored.append((_target_strategy_score(metrics), rule))
    expanded: list[dict[str, Any]] = []
    for _score, rule in sorted(scored, key=lambda item: item[0], reverse=True)[:TARGET_STRATEGY_CATEGORY_EXPANSION_LIMIT]:
        expanded.append(rule)
        for filters in TARGET_STRATEGY_CATEGORY_FILTERS:
            if not filters:
                continue
            candidate = dict(rule)
            candidate.update(filters)
            expanded.append(candidate)
        for filters in TARGET_STRATEGY_STRUCTURAL_FILTERS:
            if not filters:
                continue
            candidate = dict(rule)
            candidate.update(filters)
            expanded.append(candidate)
        for category_filters in TARGET_STRATEGY_CATEGORY_FILTERS:
            if not category_filters:
                continue
            for structural_filters in TARGET_STRATEGY_STRUCTURAL_FILTERS:
                if not structural_filters:
                    continue
                candidate = dict(rule)
                candidate.update(category_filters)
                candidate.update(structural_filters)
                expanded.append(candidate)
    pre_union_rules: list[dict[str, Any]] = []
    seen_pre_union: set[str] = set()
    for rule in expanded:
        key = _target_strategy_rule_key(rule)
        if key in seen_pre_union:
            continue
        seen_pre_union.add(key)
        pre_union_rules.append(rule)

    union_scored: list[tuple[tuple[float, float, float, float], dict[str, Any]]] = []
    for rule in pre_union_rules:
        metrics = _target_strategy_metrics_from_items(train_items, rule)
        if int(metrics["action_count"]) < TARGET_STRATEGY_UNION_CHILD_MIN_TRAIN_ACTIONS:
            continue
        union_scored.append((_target_strategy_score(metrics), rule))
    seed_rules = [
        rule
        for _score, rule in sorted(union_scored, key=lambda item: item[0], reverse=True)[:TARGET_STRATEGY_UNION_RULE_LIMIT]
    ]
    for left_index, left in enumerate(seed_rules):
        for right in seed_rules[left_index + 1:]:
            expanded.append(
                {
                    "direction_source": left.get("direction_source", "market"),
                    "action": "杞讳粨",
                    "stake_pct": 1.0,
                    "any_rules": [left, right],
                }
            )
    expanded.extend(_target_strategy_complement_rules(seed_rules, train_items))
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rule in expanded:
        key = _target_strategy_rule_key(rule)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rule)
        if len(deduped) >= TARGET_STRATEGY_MAX_CANDIDATE_RULES:
            break
    return deduped


def _target_strategy_precompute(
    rows: list[Mapping[str, Any]],
    calibrator_params: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        probabilities = _prob_vector_from_row(row, calibrator_params)
        for source in TARGET_STRATEGY_DIRECTION_SOURCES:
            rule = {"direction_source": source}
            outcome = _target_strategy_outcome(row, probabilities, rule)
            if outcome not in OUTCOMES:
                continue
            items.append(
                {
                    "source": source,
                    "row_index": row_index,
                    "outcome": outcome,
                    "actual_result": str(_row_value(row, "actual_result", "") or ""),
                    "market_odds": _market_odds(row),
                    "features": _target_strategy_features(row, probabilities, outcome),
                }
            )
    return items


def _target_strategy_metrics_from_items(items: list[dict[str, Any]], rule: Mapping[str, Any]) -> dict[str, Any]:
    source = str(rule.get("direction_source", "market") or "market")
    sample_count = sum(1 for item in items if item["source"] == source)
    action_count = 0
    hits = 0
    total_roi = 0.0
    total_stake_pct = 0.0
    buckets: dict[str, dict[str, float | int]] = {}
    for item in items:
        if item["source"] != source:
            continue
        outcome = str(item["outcome"])
        action = "轻仓" if _target_strategy_passes(item["features"], rule) else "观望"
        hit = 1 if outcome == item["actual_result"] else 0
        stake_pct = safe_float(rule.get("stake_pct", 1.0)) if action != "观望" else 0.0
        roi = _roi_for_action(
            action=action,
            recommended_outcome=outcome,
            stake_pct=stake_pct,
            actual_result=str(item["actual_result"]),
            market_odds=item["market_odds"],
        )
        bucket = buckets.setdefault(
            f"{action}:{outcome}",
            {"sample_count": 0, "action_count": 0, "hits": 0, "total_roi": 0.0},
        )
        bucket["sample_count"] += 1
        if action != "观望":
            action_count += 1
            hits += hit
            total_roi += roi
            total_stake_pct += stake_pct
            bucket["action_count"] += 1
            bucket["hits"] += hit
            bucket["total_roi"] += roi
    action_share = action_count / sample_count if sample_count else 0.0
    hit_rate = hits / action_count if action_count else 0.0
    return {
        "sample_count": sample_count,
        "action_count": action_count,
        "watch_count": max(sample_count - action_count, 0),
        "hit_count": hits,
        "hit_rate": hit_rate,
        "action_share": action_share,
        "watch_share": 1.0 - action_share if sample_count else 0.0,
        "total_roi": round(total_roi, 4),
        "avg_roi": total_roi / action_count if action_count else 0.0,
        "avg_stake_pct": total_stake_pct / action_count if action_count else 0.0,
        "buckets": buckets,
    }


def _target_strategy_metrics_v2(items: list[dict[str, Any]], rule: Mapping[str, Any]) -> dict[str, Any]:
    any_rules = rule.get("any_rules")
    if not isinstance(any_rules, list) or not any_rules:
        return _target_strategy_metrics_from_items(items, rule)

    by_row: dict[int, list[dict[str, Any]]] = {}
    for item in items:
        by_row.setdefault(int(item.get("row_index", 0)), []).append(item)

    action_count = 0
    hits = 0
    total_roi = 0.0
    total_stake_pct = 0.0
    buckets: dict[str, dict[str, float | int]] = {}
    for row_items in by_row.values():
        fallback = row_items[0] if row_items else None
        selected_item = fallback
        action = "观望"
        for child_rule in any_rules:
            if not isinstance(child_rule, Mapping):
                continue
            child_source = str(child_rule.get("direction_source", "market") or "market")
            for item in row_items:
                if item["source"] == child_source and _target_strategy_passes(item["features"], child_rule):
                    selected_item = item
                    action = "轻仓"
                    break
            if action != "观望":
                break
        if selected_item is None:
            continue
        outcome = str(selected_item["outcome"])
        hit = 1 if outcome == selected_item["actual_result"] else 0
        stake_pct = safe_float(rule.get("stake_pct", 1.0)) if action != "观望" else 0.0
        roi = _roi_for_action(
            action=action,
            recommended_outcome=outcome,
            stake_pct=stake_pct,
            actual_result=str(selected_item["actual_result"]),
            market_odds=selected_item["market_odds"],
        )
        bucket = buckets.setdefault(
            f"{action}:{outcome}",
            {"sample_count": 0, "action_count": 0, "hits": 0, "total_roi": 0.0},
        )
        bucket["sample_count"] += 1
        if action != "观望":
            action_count += 1
            hits += hit
            total_roi += roi
            total_stake_pct += stake_pct
            bucket["action_count"] += 1
            bucket["hits"] += hit
            bucket["total_roi"] += roi

    sample_count = len(by_row)
    action_share = action_count / sample_count if sample_count else 0.0
    hit_rate = hits / action_count if action_count else 0.0
    return {
        "sample_count": sample_count,
        "action_count": action_count,
        "watch_count": max(sample_count - action_count, 0),
        "hit_count": hits,
        "hit_rate": hit_rate,
        "action_share": action_share,
        "watch_share": 1.0 - action_share if sample_count else 0.0,
        "total_roi": round(total_roi, 4),
        "avg_roi": total_roi / action_count if action_count else 0.0,
        "avg_stake_pct": total_stake_pct / action_count if action_count else 0.0,
        "buckets": buckets,
    }


def fit_target_strategy(
    train_rows: list[Mapping[str, Any]],
    validation_rows: list[Mapping[str, Any]],
    calibrator_params: Mapping[str, Any] | None,
    *,
    target_hit_rate: float = DEFAULT_TARGET_HIT_RATE,
    min_action_share: float = DEFAULT_MIN_ACTION_SHARE,
) -> dict[str, Any]:
    target_hit_rate = clamp(safe_float(target_hit_rate, DEFAULT_TARGET_HIT_RATE), 0.01, 1.0)
    min_action_share = clamp(safe_float(min_action_share, DEFAULT_MIN_ACTION_SHARE), 0.01, 0.99)
    target_metrics = {
        "target_hit_rate": target_hit_rate,
        "min_action_share": min_action_share,
        "max_watch_share": 1.0 - min_action_share,
    }
    if len(train_rows) < TARGET_STRATEGY_MIN_TRAIN_SAMPLES or len(validation_rows) < TARGET_STRATEGY_MIN_VALIDATION_SAMPLES:
        empty_rule = {"direction_source": "market", "action": "观望"}
        return {
            "status": "insufficient_samples",
            "reason": (
                f"train/validation samples {len(train_rows)}/{len(validation_rows)} below "
                f"{TARGET_STRATEGY_MIN_TRAIN_SAMPLES}/{TARGET_STRATEGY_MIN_VALIDATION_SAMPLES}"
            ),
            "params": {},
            "target_metrics": target_metrics,
            "train_metrics": evaluate_target_strategy_rule(train_rows, empty_rule, calibrator_params=calibrator_params),
            "validation_metrics": evaluate_target_strategy_rule(validation_rows, empty_rule, calibrator_params=calibrator_params),
            "best_candidate": {},
        }

    train_items = _target_strategy_precompute(train_rows, calibrator_params)
    validation_items = _target_strategy_precompute(validation_rows, calibrator_params)
    candidate_rules = _expand_target_strategy_rules(_target_strategy_rules(), train_items)
    best_ready: dict[str, Any] | None = None
    best_candidate: dict[str, Any] | None = None
    best_hit_candidate: dict[str, Any] | None = None
    best_action_candidate: dict[str, Any] | None = None
    best_covered_candidate: dict[str, Any] | None = None
    for rule in candidate_rules:
        train_metrics = _target_strategy_metrics_v2(train_items, rule)
        if int(train_metrics["action_count"]) < TARGET_STRATEGY_MIN_TRAIN_SAMPLES:
            continue
        validation_metrics = _target_strategy_metrics_v2(validation_items, rule)
        status, reason = target_strategy_status(
            validation_metrics,
            target_hit_rate=target_hit_rate,
            min_action_share=min_action_share,
        )
        candidate = {
            "status": status,
            "reason": reason,
            "params": rule,
            "target_metrics": target_metrics,
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
        }
        if best_candidate is None or _target_strategy_score(validation_metrics) > _target_strategy_score(best_candidate["validation_metrics"]):
            best_candidate = candidate
        if best_hit_candidate is None or (
            safe_float(validation_metrics.get("hit_rate")),
            safe_float(validation_metrics.get("action_share")),
            safe_float(validation_metrics.get("total_roi")),
        ) > (
            safe_float(best_hit_candidate["validation_metrics"].get("hit_rate")),
            safe_float(best_hit_candidate["validation_metrics"].get("action_share")),
            safe_float(best_hit_candidate["validation_metrics"].get("total_roi")),
        ):
            best_hit_candidate = candidate
        if best_action_candidate is None or (
            safe_float(validation_metrics.get("action_share")),
            safe_float(validation_metrics.get("hit_rate")),
            safe_float(validation_metrics.get("total_roi")),
        ) > (
            safe_float(best_action_candidate["validation_metrics"].get("action_share")),
            safe_float(best_action_candidate["validation_metrics"].get("hit_rate")),
            safe_float(best_action_candidate["validation_metrics"].get("total_roi")),
        ):
            best_action_candidate = candidate
        if safe_float(validation_metrics.get("action_share")) > min_action_share and (
            best_covered_candidate is None
            or (
                safe_float(validation_metrics.get("hit_rate")),
                safe_float(validation_metrics.get("action_share")),
                safe_float(validation_metrics.get("total_roi")),
            )
            > (
                safe_float(best_covered_candidate["validation_metrics"].get("hit_rate")),
                safe_float(best_covered_candidate["validation_metrics"].get("action_share")),
                safe_float(best_covered_candidate["validation_metrics"].get("total_roi")),
            )
        ):
            best_covered_candidate = candidate
        if status == "ready" and (
            best_ready is None
            or _target_strategy_score(validation_metrics) > _target_strategy_score(best_ready["validation_metrics"])
        ):
            best_ready = candidate

    selected = best_ready or best_candidate
    if selected is None:
        return {
            "status": "target_unreachable",
            "reason": "no target strategy candidate produced enough executable train actions",
            "params": {},
            "target_metrics": target_metrics,
            "train_metrics": _empty_target_strategy_metrics(len(train_rows)),
            "validation_metrics": _empty_target_strategy_metrics(len(validation_rows)),
            "best_candidate": {},
            "frontier": {},
        }
    if best_ready is not None:
        return selected
    frontier = {
        "best_hit": _target_strategy_candidate_summary(best_hit_candidate),
        "best_action": _target_strategy_candidate_summary(best_action_candidate),
        "best_covered": _target_strategy_candidate_summary(best_covered_candidate),
        "gaps": _target_strategy_gap_summary(
            selected["validation_metrics"],
            target_hit_rate=target_hit_rate,
            min_action_share=min_action_share,
        ),
    }
    return {
        "status": "target_unreachable",
        "reason": selected["reason"],
        "params": {},
        "target_metrics": target_metrics,
        "train_metrics": selected["train_metrics"],
        "validation_metrics": selected["validation_metrics"],
        "best_candidate": {
            "params": selected["params"],
            "train_metrics": selected["train_metrics"],
            "validation_metrics": selected["validation_metrics"],
            "reason": selected["reason"],
        },
        "frontier": frontier,
    }


def fit_target_strategy_walk_forward(
    rows: list[Mapping[str, Any]],
    calibrator_params: Mapping[str, Any] | None,
    *,
    target_hit_rate: float = DEFAULT_TARGET_HIT_RATE,
    min_action_share: float = DEFAULT_MIN_ACTION_SHARE,
) -> dict[str, Any]:
    target_hit_rate = clamp(safe_float(target_hit_rate, DEFAULT_TARGET_HIT_RATE), 0.01, 1.0)
    min_action_share = clamp(safe_float(min_action_share, DEFAULT_MIN_ACTION_SHARE), 0.01, 0.99)
    target_metrics = {
        "target_hit_rate": target_hit_rate,
        "min_action_share": min_action_share,
        "max_watch_share": 1.0 - min_action_share,
    }
    groups = _rows_by_issue(rows)
    fold_results: list[dict[str, Any]] = []
    skipped_folds: list[dict[str, Any]] = []
    best_ready_candidate: dict[str, Any] | None = None
    best_candidate: dict[str, Any] | None = None
    for index in range(1, len(groups)):
        issue, validation_rows = groups[index]
        train_rows = _flatten_issue_groups(groups[:index])
        if len(train_rows) < TARGET_STRATEGY_MIN_TRAIN_SAMPLES or len(validation_rows) < TARGET_STRATEGY_MIN_VALIDATION_SAMPLES:
            skipped_folds.append(
                {
                    "issue": issue,
                    "train_samples": len(train_rows),
                    "validation_samples": len(validation_rows),
                    "reason": "fold below train/validation sample floor",
                }
            )
            continue
        fold = fit_target_strategy(
            train_rows,
            validation_rows,
            calibrator_params,
            target_hit_rate=target_hit_rate,
            min_action_share=min_action_share,
        )
        fold_result = {
            "issue": issue,
            "status": fold["status"],
            "reason": str(fold.get("reason", "") or ""),
            "train_samples": len(train_rows),
            "validation_samples": len(validation_rows),
            "params": fold.get("params", {}),
            "train_metrics": fold["train_metrics"],
            "validation_metrics": fold["validation_metrics"],
        }
        fold_results.append(fold_result)
        candidate_score = _target_strategy_score(fold["validation_metrics"])
        if fold["status"] == "ready" and fold.get("params") and (
            best_ready_candidate is None
            or candidate_score > _target_strategy_score(best_ready_candidate["validation_metrics"])
        ):
            best_ready_candidate = fold_result
        if best_candidate is None or candidate_score > _target_strategy_score(best_candidate["validation_metrics"]):
            best_candidate = fold_result
    validation_metrics = _combine_target_strategy_metrics(
        [fold["validation_metrics"] for fold in fold_results]
    )
    status, reason = target_strategy_status(
        validation_metrics,
        target_hit_rate=target_hit_rate,
        min_action_share=min_action_share,
    )
    if not fold_results:
        status = "insufficient_samples"
        reason = "no walk-forward fold has enough historical train and validation samples"
    return {
        "status": status if status == "ready" else "target_unreachable" if fold_results else status,
        "reason": reason,
        "params": best_ready_candidate.get("params", {}) if status == "ready" and best_ready_candidate else {},
        "target_metrics": target_metrics,
        "train_metrics": _combine_target_strategy_metrics(
            [fold["train_metrics"] for fold in fold_results]
        ),
        "validation_metrics": validation_metrics,
        "best_candidate": _target_strategy_candidate_summary(best_candidate),
        "folds": fold_results,
        "skipped_folds": skipped_folds,
        "frontier": {
            "gaps": _target_strategy_gap_summary(
                validation_metrics,
                target_hit_rate=target_hit_rate,
                min_action_share=min_action_share,
            )
        },
    }


def _handicap_odds_decimal(value: Any) -> float:
    odds = safe_float(value)
    return odds + 1.0 if 0 < odds < 1.5 else odds


def _handicap_side_features(row: Mapping[str, Any], side: str) -> dict[str, Any]:
    side = side if side in {"home", "away"} else ""
    other_side = "away" if side == "home" else "home"
    cover_prob = safe_float(_row_value(row, f"handicap_{side}_cover_prob")) if side else 0.0
    other_cover_prob = safe_float(_row_value(row, f"handicap_{other_side}_cover_prob")) if side else 0.0
    odds = _handicap_odds_decimal(_row_value(row, f"handicap_{side}_odds")) if side else 0.0
    expected_value = cover_prob * odds - 1.0 if odds > 0 else -1.0
    line = safe_float(_row_value(row, "handicap_line"))
    initial_line = safe_float(_row_value(row, "handicap_initial_line"))
    line_move = line - initial_line
    if side == "home":
        line_support = -line_move
    elif side == "away":
        line_support = line_move
    else:
        line_support = 0.0
    return {
        "side": side,
        "base_action": _normalized_action(_row_value(row, "handicap_recommendation", "")),
        "expected_value": expected_value,
        "confidence": safe_float(_row_value(row, "handicap_confidence")),
        "quality_score": safe_float(_row_value(row, "quality_score")),
        "cover_prob": cover_prob,
        "cover_margin": cover_prob - other_cover_prob,
        "odds": odds,
        "line_abs": abs(line),
        "line_support": line_support,
    }


def _handicap_strategy_passes(features: Mapping[str, Any], rule: Mapping[str, Any]) -> bool:
    side = str(features.get("side", "") or "")
    if side not in {"home", "away"}:
        return False
    sides = rule.get("sides", ())
    if sides and side not in set(sides):
        return False
    base_actions = rule.get("base_actions", ())
    if base_actions and _normalized_action(features.get("base_action")) not in {
        _normalized_action(item) for item in base_actions
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


def _handicap_roi_for_action(row: Mapping[str, Any], side: str, stake_pct: float) -> float:
    actual = str(_row_value(row, "handicap_actual_result", "") or "")
    if side not in {"home", "away"} or actual not in {"home", "away", "push"} or stake_pct <= 0:
        return 0.0
    if actual == "push":
        return 0.0
    odds = _handicap_odds_decimal(_row_value(row, f"handicap_{side}_odds"))
    stake_units = stake_pct / 100.0
    if side == actual:
        return round(stake_units * (odds - 1.0), 4) if odds > 1.0 else 0.0
    return round(-stake_units, 4)


def _handicap_strategy_metrics(rows: list[Mapping[str, Any]], rule: Mapping[str, Any]) -> dict[str, Any]:
    sample_count = 0
    action_count = 0
    hits = 0
    total_roi = 0.0
    total_stake_pct = 0.0
    buckets: dict[str, dict[str, float | int]] = {}
    stake_pct = safe_float(rule.get("stake_pct", 1.0))
    for row in rows:
        actual = str(_row_value(row, "handicap_actual_result", "") or "")
        if actual not in {"home", "away", "push"}:
            continue
        sample_count += 1
        side = str(_row_value(row, "handicap_recommended_side", "") or "")
        features = _handicap_side_features(row, side)
        action = "轻仓" if _handicap_strategy_passes(features, rule) else "观望"
        bucket = buckets.setdefault(
            f"{action}:{side or '-'}",
            {"sample_count": 0, "action_count": 0, "hits": 0, "total_roi": 0.0},
        )
        bucket["sample_count"] += 1
        if action == "观望":
            continue
        hit = 1 if side == actual else 0
        roi = _handicap_roi_for_action(row, side, stake_pct)
        action_count += 1
        hits += hit
        total_roi += roi
        total_stake_pct += stake_pct
        bucket["action_count"] += 1
        bucket["hits"] += hit
        bucket["total_roi"] += roi
    action_share = action_count / sample_count if sample_count else 0.0
    return {
        "sample_count": sample_count,
        "action_count": action_count,
        "watch_count": max(sample_count - action_count, 0),
        "hit_count": hits,
        "hit_rate": hits / action_count if action_count else 0.0,
        "action_share": action_share,
        "watch_share": 1.0 - action_share if sample_count else 0.0,
        "total_roi": round(total_roi, 4),
        "avg_roi": total_roi / action_count if action_count else 0.0,
        "avg_stake_pct": total_stake_pct / action_count if action_count else 0.0,
        "buckets": buckets,
    }


def _bucket_floor(value: Any, width: float, offset: float = 0.0) -> float:
    width = max(safe_float(width), 0.001)
    return math.floor((safe_float(value) + offset) / width) * width - offset


def _handicap_ev_for_side(row: Mapping[str, Any], side: str) -> float:
    if side not in {"home", "away"}:
        return -9.0
    prob = safe_float(_row_value(row, f"handicap_{side}_cover_prob"))
    odds = _handicap_odds_decimal(_row_value(row, f"handicap_{side}_odds"))
    return prob * odds - 1.0 if odds > 0 else -9.0


def _handicap_bucket_feature(row: Mapping[str, Any], name: str) -> Any:
    if name == "rec_side":
        side = str(_row_value(row, "handicap_recommended_side", "") or "")
        return side if side in {"home", "away"} else ""
    if name == "rec_action":
        return _normalized_action(_row_value(row, "handicap_recommendation", ""))
    if name == "line25":
        return round(_bucket_floor(_row_value(row, "handicap_line"), 0.25, 5.0), 2)
    if name == "line50":
        return round(_bucket_floor(_row_value(row, "handicap_line"), 0.50, 5.0), 2)
    if name == "coverdiff10":
        return round(
            _bucket_floor(
                safe_float(_row_value(row, "handicap_home_cover_prob"))
                - safe_float(_row_value(row, "handicap_away_cover_prob")),
                0.10,
                2.0,
            ),
            2,
        )
    if name == "evdiff10":
        return round(_bucket_floor(_handicap_ev_for_side(row, "home") - _handicap_ev_for_side(row, "away"), 0.10, 5.0), 2)
    if name == "homeodds20":
        return round(_bucket_floor(_handicap_odds_decimal(_row_value(row, "handicap_home_odds")), 0.20), 2)
    if name == "awayodds20":
        return round(_bucket_floor(_handicap_odds_decimal(_row_value(row, "handicap_away_odds")), 0.20), 2)
    if name == "conf10":
        return round(_bucket_floor(_row_value(row, "handicap_confidence"), 0.10), 2)
    if name == "quality10":
        return round(_bucket_floor(_row_value(row, "quality_score"), 0.10), 2)
    return ""


def _handicap_bucket_key(row: Mapping[str, Any], features: tuple[str, ...] | list[str]) -> str:
    return _dumps_json([_handicap_bucket_feature(row, name) for name in features])


def _handicap_bucket_strategy_metrics(rows: list[Mapping[str, Any]], strategy: Mapping[str, Any]) -> dict[str, Any]:
    features = [str(item) for item in strategy.get("features", []) if str(item or "").strip()]
    buckets = strategy.get("buckets", {})
    if not isinstance(buckets, Mapping):
        buckets = {}
    sample_count = 0
    action_count = 0
    hits = 0
    bucket_summary: dict[str, dict[str, float | int]] = {}
    for row in rows:
        actual = str(_row_value(row, "handicap_actual_result", "") or "")
        if actual not in {"home", "away", "push"}:
            continue
        sample_count += 1
        bucket = buckets.get(_handicap_bucket_key(row, features))
        side = str(bucket.get("side", "") or "") if isinstance(bucket, Mapping) else ""
        action = str(strategy.get("action", "轻仓") or "轻仓") if side in {"home", "away"} else "观望"
        key = f"{action}:{side or '-'}"
        summary = bucket_summary.setdefault(
            key,
            {"sample_count": 0, "action_count": 0, "hits": 0, "total_roi": 0.0},
        )
        summary["sample_count"] += 1
        if action == "观望":
            continue
        hit = 1 if side == actual else 0
        action_count += 1
        hits += hit
        summary["action_count"] += 1
        summary["hits"] += hit
    action_share = action_count / sample_count if sample_count else 0.0
    return {
        "sample_count": sample_count,
        "action_count": action_count,
        "watch_count": max(sample_count - action_count, 0),
        "hit_count": hits,
        "hit_rate": hits / action_count if action_count else 0.0,
        "action_share": action_share,
        "watch_share": 1.0 - action_share if sample_count else 0.0,
        "total_roi": 0.0,
        "avg_roi": 0.0,
        "avg_stake_pct": safe_float(strategy.get("stake_pct", 1.0)) if action_count else 0.0,
        "buckets": bucket_summary,
    }


def fit_handicap_bucket_strategy(
    train_rows: list[Mapping[str, Any]],
    validation_rows: list[Mapping[str, Any]] | None = None,
    *,
    target_hit_rate: float = DEFAULT_HANDICAP_TARGET_HIT_RATE,
    min_action_share: float = DEFAULT_HANDICAP_MIN_ACTION_SHARE,
) -> dict[str, Any]:
    target_hit_rate = clamp(safe_float(target_hit_rate, DEFAULT_HANDICAP_TARGET_HIT_RATE), 0.01, 1.0)
    min_action_share = clamp(safe_float(min_action_share, DEFAULT_HANDICAP_MIN_ACTION_SHARE), 0.01, 0.99)
    target_metrics = {
        "target_hit_rate": target_hit_rate,
        "min_action_share": min_action_share,
        "max_watch_share": 1.0 - min_action_share,
        "strategy_kind": "handicap_bucket_table",
    }
    validation_rows = train_rows if validation_rows is None else validation_rows
    best_ready: dict[str, Any] | None = None
    best_candidate: dict[str, Any] | None = None
    for features, min_group, min_bucket_hit in HANDICAP_BUCKET_STRATEGY_CANDIDATES:
        groups: dict[str, dict[str, int]] = {}
        for row in train_rows:
            actual = str(_row_value(row, "handicap_actual_result", "") or "")
            if actual not in {"home", "away"}:
                continue
            key = _handicap_bucket_key(row, features)
            group = groups.setdefault(key, {"home": 0, "away": 0})
            group[actual] += 1

        buckets: dict[str, dict[str, Any]] = {}
        for key, counts in groups.items():
            total = int(counts["home"] + counts["away"])
            if total < min_group:
                continue
            side = "home" if counts["home"] >= counts["away"] else "away"
            hit_rate = counts[side] / total if total else 0.0
            if hit_rate < min_bucket_hit:
                continue
            buckets[key] = {
                "side": side,
                "sample_count": total,
                "hit_rate": hit_rate,
                "home_hits": counts["home"],
                "away_hits": counts["away"],
            }

        strategy = {
            "strategy_kind": "handicap_bucket_table",
            "action": "轻仓",
            "stake_pct": 1.0,
            "features": list(features),
            "min_group": min_group,
            "min_bucket_hit": min_bucket_hit,
            "bucket_count": len(buckets),
            "buckets": buckets,
        }
        train_metrics = _handicap_bucket_strategy_metrics(train_rows, strategy)
        validation_metrics = _handicap_bucket_strategy_metrics(validation_rows, strategy)
        status, reason = target_strategy_status(
            validation_metrics,
            target_hit_rate=target_hit_rate,
            min_action_share=min_action_share,
        )
        candidate = {
            "status": status,
            "reason": reason,
            "params": strategy,
            "target_metrics": target_metrics,
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
        }
        if best_candidate is None or _target_strategy_score(validation_metrics) > _target_strategy_score(best_candidate["validation_metrics"]):
            best_candidate = candidate
        if status == "ready" and (
            best_ready is None
            or _target_strategy_score(validation_metrics) > _target_strategy_score(best_ready["validation_metrics"])
        ):
            best_ready = candidate

    if best_ready is not None:
        best_ready["reason"] = "handicap bucket validation reached target"
        return best_ready
    if best_candidate is not None:
        return {
            "status": "target_unreachable",
            "reason": best_candidate["reason"],
            "params": {},
            "target_metrics": target_metrics,
            "train_metrics": best_candidate["train_metrics"],
            "validation_metrics": best_candidate["validation_metrics"],
            "best_candidate": {
                "params": best_candidate["params"],
                "train_metrics": best_candidate["train_metrics"],
                "validation_metrics": best_candidate["validation_metrics"],
                "reason": best_candidate["reason"],
            },
            "frontier": {
                "gaps": _target_strategy_gap_summary(
                    best_candidate["validation_metrics"],
                    target_hit_rate=target_hit_rate,
                    min_action_share=min_action_share,
                )
            },
        }
    return {
        "status": "target_unreachable",
        "reason": "no handicap bucket candidate produced executable rows",
        "params": {},
            "target_metrics": target_metrics,
            "train_metrics": _empty_target_strategy_metrics(len(train_rows)),
            "validation_metrics": _empty_target_strategy_metrics(len(validation_rows)),
            "best_candidate": {},
            "frontier": {},
        }


def _handicap_target_rules() -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    space = HANDICAP_TARGET_SEARCH_SPACE
    for (
        ev_min,
        confidence_min,
        cover_prob_min,
        cover_margin_min,
        quality_min,
        odds_max,
        sides,
        base_actions,
    ) in product(
        space["ev_min"],
        space["confidence_min"],
        space["cover_prob_min"],
        space["cover_margin_min"],
        space["quality_min"],
        space["odds_max"],
        space["sides"],
        space["base_actions"],
    ):
        rules.append(
            {
                "strategy_kind": "handicap",
                "action": "轻仓",
                "stake_pct": 1.0,
                "ev_min": ev_min,
                "confidence_min": confidence_min,
                "cover_prob_min": cover_prob_min,
                "cover_margin_min": cover_margin_min,
                "quality_min": quality_min,
                "odds_max": odds_max,
                "sides": sides,
                "base_actions": base_actions,
            }
        )
    return rules


def fit_handicap_target_strategy(
    train_rows: list[Mapping[str, Any]],
    validation_rows: list[Mapping[str, Any]],
    *,
    target_hit_rate: float = DEFAULT_HANDICAP_TARGET_HIT_RATE,
    min_action_share: float = DEFAULT_HANDICAP_MIN_ACTION_SHARE,
) -> dict[str, Any]:
    target_hit_rate = clamp(safe_float(target_hit_rate, DEFAULT_HANDICAP_TARGET_HIT_RATE), 0.01, 1.0)
    min_action_share = clamp(safe_float(min_action_share, DEFAULT_HANDICAP_MIN_ACTION_SHARE), 0.01, 0.99)
    target_metrics = {
        "target_hit_rate": target_hit_rate,
        "min_action_share": min_action_share,
        "max_watch_share": 1.0 - min_action_share,
        "strategy_kind": "handicap",
    }
    if len(train_rows) < TARGET_STRATEGY_MIN_TRAIN_SAMPLES or len(validation_rows) < TARGET_STRATEGY_MIN_VALIDATION_SAMPLES:
        empty_rule = {"strategy_kind": "handicap", "action": "观望"}
        return {
            "status": "insufficient_samples",
            "reason": (
                f"train/validation samples {len(train_rows)}/{len(validation_rows)} below "
                f"{TARGET_STRATEGY_MIN_TRAIN_SAMPLES}/{TARGET_STRATEGY_MIN_VALIDATION_SAMPLES}"
            ),
            "params": {},
            "target_metrics": target_metrics,
            "train_metrics": _handicap_strategy_metrics(train_rows, empty_rule),
            "validation_metrics": _handicap_strategy_metrics(validation_rows, empty_rule),
            "best_candidate": {},
            "frontier": {},
        }

    best_ready: dict[str, Any] | None = None
    best_candidate: dict[str, Any] | None = None
    best_hit_candidate: dict[str, Any] | None = None
    best_action_candidate: dict[str, Any] | None = None
    best_covered_candidate: dict[str, Any] | None = None
    for rule in _handicap_target_rules():
        train_metrics = _handicap_strategy_metrics(train_rows, rule)
        if int(train_metrics["action_count"]) < TARGET_STRATEGY_MIN_TRAIN_SAMPLES:
            continue
        validation_metrics = _handicap_strategy_metrics(validation_rows, rule)
        status, reason = target_strategy_status(
            validation_metrics,
            target_hit_rate=target_hit_rate,
            min_action_share=min_action_share,
        )
        candidate = {
            "status": status,
            "reason": reason,
            "params": rule,
            "target_metrics": target_metrics,
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
        }
        if best_candidate is None or _target_strategy_score(validation_metrics) > _target_strategy_score(best_candidate["validation_metrics"]):
            best_candidate = candidate
        if best_hit_candidate is None or (
            safe_float(validation_metrics.get("hit_rate")),
            safe_float(validation_metrics.get("action_share")),
            safe_float(validation_metrics.get("total_roi")),
        ) > (
            safe_float(best_hit_candidate["validation_metrics"].get("hit_rate")),
            safe_float(best_hit_candidate["validation_metrics"].get("action_share")),
            safe_float(best_hit_candidate["validation_metrics"].get("total_roi")),
        ):
            best_hit_candidate = candidate
        if best_action_candidate is None or (
            safe_float(validation_metrics.get("action_share")),
            safe_float(validation_metrics.get("hit_rate")),
            safe_float(validation_metrics.get("total_roi")),
        ) > (
            safe_float(best_action_candidate["validation_metrics"].get("action_share")),
            safe_float(best_action_candidate["validation_metrics"].get("hit_rate")),
            safe_float(best_action_candidate["validation_metrics"].get("total_roi")),
        ):
            best_action_candidate = candidate
        if safe_float(validation_metrics.get("action_share")) >= min_action_share and (
            best_covered_candidate is None
            or (
                safe_float(validation_metrics.get("hit_rate")),
                safe_float(validation_metrics.get("action_share")),
                safe_float(validation_metrics.get("total_roi")),
            )
            > (
                safe_float(best_covered_candidate["validation_metrics"].get("hit_rate")),
                safe_float(best_covered_candidate["validation_metrics"].get("action_share")),
                safe_float(best_covered_candidate["validation_metrics"].get("total_roi")),
            )
        ):
            best_covered_candidate = candidate
        if status == "ready" and (
            best_ready is None
            or _target_strategy_score(validation_metrics) > _target_strategy_score(best_ready["validation_metrics"])
        ):
            best_ready = candidate

    if best_ready is not None:
        return best_ready

    bucket_candidate = fit_handicap_bucket_strategy(
        train_rows,
        validation_rows,
        target_hit_rate=target_hit_rate,
        min_action_share=min_action_share,
    )
    if bucket_candidate["status"] == "ready":
        return bucket_candidate

    selected = best_candidate
    if selected is None:
        return {
            "status": "target_unreachable",
            "reason": "no handicap strategy candidate produced enough executable train actions",
            "params": {},
            "target_metrics": target_metrics,
            "train_metrics": bucket_candidate.get("train_metrics", _empty_target_strategy_metrics(len(train_rows))),
            "validation_metrics": bucket_candidate.get("validation_metrics", _empty_target_strategy_metrics(len(validation_rows))),
            "best_candidate": bucket_candidate.get("best_candidate", {}),
            "frontier": bucket_candidate.get("frontier", {}),
        }
    return {
        "status": "target_unreachable",
        "reason": selected["reason"],
        "params": {},
        "target_metrics": target_metrics,
        "train_metrics": selected["train_metrics"],
        "validation_metrics": selected["validation_metrics"],
        "best_candidate": {
            "params": selected["params"],
            "train_metrics": selected["train_metrics"],
            "validation_metrics": selected["validation_metrics"],
            "reason": selected["reason"],
        },
        "frontier": {
            "best_hit": _target_strategy_candidate_summary(best_hit_candidate),
            "best_action": _target_strategy_candidate_summary(best_action_candidate),
            "best_covered": _target_strategy_candidate_summary(best_covered_candidate),
            "gaps": _target_strategy_gap_summary(
                selected["validation_metrics"],
                target_hit_rate=target_hit_rate,
                min_action_share=min_action_share,
            ),
        },
    }


def _threshold_search_space() -> list[dict[str, dict[str, float]]]:
    candidates: list[dict[str, dict[str, float]]] = []
    for (
        main_ev,
        main_confidence,
        main_bias,
        light_ev,
        light_confidence,
        light_bias,
        promote_ev,
        promote_confidence,
        promote_bias,
    ) in product(
        (0.08, 0.10, 0.12),
        (0.58, 0.62, 0.68),
        (0.020, 0.030, 0.040),
        (0.03, 0.04, 0.06),
        (0.48, 0.52, 0.58),
        (0.010, 0.015, 0.025),
        (0.06, 0.08, 0.10),
        (0.54, 0.58, 0.64),
        (0.020, 0.030, 0.040),
    ):
        candidates.append(
            {
                "main": {"ev": main_ev, "confidence": main_confidence, "market_bias": main_bias, "quality": 0.68},
                "light": {"ev": light_ev, "confidence": light_confidence, "market_bias": light_bias, "quality": 0.58},
                "promote": {
                    "ev": promote_ev,
                    "confidence": promote_confidence,
                    "market_bias": promote_bias,
                    "quality": 0.70,
                },
            }
        )
    return candidates


def _passes_action_floor(candidate_count: int, baseline_count: int) -> bool:
    if baseline_count <= 0:
        return candidate_count > 0
    return candidate_count >= max(1, math.ceil(baseline_count * 0.35))


def _fit_thresholds(
    train_rows: list[Mapping[str, Any]],
    validation_rows: list[Mapping[str, Any]],
    calibrator_params: Mapping[str, Any] | None,
) -> dict[str, Any]:
    baseline_train = _replay_threshold_metrics(train_rows, BASE_THRESHOLD_CONFIG, calibrator_params)
    baseline_validation = _replay_threshold_metrics(validation_rows, BASE_THRESHOLD_CONFIG, calibrator_params)
    baseline_train_action_count = max(
        int(baseline_train["algorithm"]["action_count"]),
        int(baseline_train["final"]["action_count"]),
    )
    baseline_validation_action_count = max(
        int(baseline_validation["algorithm"]["action_count"]),
        int(baseline_validation["final"]["action_count"]),
    )

    if baseline_train_action_count < THRESHOLD_MIN_TRAIN_ACTIONS:
        return {
            "status": "insufficient_samples",
            "reason": f"训练段动作样本 {baseline_train_action_count} < {THRESHOLD_MIN_TRAIN_ACTIONS}",
            "params": BASE_THRESHOLD_CONFIG,
            "train_metrics": {"baseline": baseline_train, "candidate": baseline_train},
            "validation_metrics": {"baseline": baseline_validation, "candidate": baseline_validation},
        }
    if baseline_validation_action_count < THRESHOLD_MIN_VALIDATION_ACTIONS:
        return {
            "status": "insufficient_samples",
            "reason": f"验证段动作样本 {baseline_validation_action_count} < {THRESHOLD_MIN_VALIDATION_ACTIONS}",
            "params": BASE_THRESHOLD_CONFIG,
            "train_metrics": {"baseline": baseline_train, "candidate": baseline_train},
            "validation_metrics": {"baseline": baseline_validation, "candidate": baseline_validation},
        }

    best_candidate: dict[str, Any] | None = None
    for threshold_config in _threshold_search_space():
        candidate_train = _replay_threshold_metrics(train_rows, threshold_config, calibrator_params)
        candidate_train_action_count = max(
            int(candidate_train["algorithm"]["action_count"]),
            int(candidate_train["final"]["action_count"]),
        )
        if candidate_train_action_count < THRESHOLD_MIN_TRAIN_ACTIONS:
            continue

        objective = (
            max(
                safe_float(candidate_train["final"]["total_roi"]),
                safe_float(candidate_train["algorithm"]["total_roi"]),
            ),
            safe_float(candidate_train["final"]["total_roi"]),
            safe_float(candidate_train["algorithm"]["total_roi"]),
            -abs(
                int(candidate_train["final"]["action_count"])
                - int(baseline_train["final"]["action_count"])
            ),
        )
        if best_candidate is None or objective > best_candidate["objective"]:
            best_candidate = {
                "objective": objective,
                "params": threshold_config,
                "train_metrics": candidate_train,
            }

    if best_candidate is None:
        return {
            "status": "no_gain",
            "reason": "未搜索到满足训练段动作样本下限的阈值候选",
            "params": BASE_THRESHOLD_CONFIG,
            "train_metrics": {"baseline": baseline_train, "candidate": baseline_train},
            "validation_metrics": {"baseline": baseline_validation, "candidate": baseline_validation},
        }

    candidate_validation = _replay_threshold_metrics(validation_rows, best_candidate["params"], calibrator_params)
    algorithm_improved = (
        safe_float(candidate_validation["algorithm"]["total_roi"]) > safe_float(baseline_validation["algorithm"]["total_roi"]) + 1e-9
        and _passes_action_floor(
            int(candidate_validation["algorithm"]["action_count"]),
            int(baseline_validation["algorithm"]["action_count"]),
        )
    )
    final_improved = (
        safe_float(candidate_validation["final"]["total_roi"]) > safe_float(baseline_validation["final"]["total_roi"]) + 1e-9
        and _passes_action_floor(
            int(candidate_validation["final"]["action_count"]),
            int(baseline_validation["final"]["action_count"]),
        )
    )
    if algorithm_improved or final_improved:
        return {
            "status": "ready",
            "reason": "",
            "params": best_candidate["params"],
            "train_metrics": {"baseline": baseline_train, "candidate": best_candidate["train_metrics"]},
            "validation_metrics": {"baseline": baseline_validation, "candidate": candidate_validation},
        }

    return {
        "status": "no_gain",
        "reason": "验证段 ROI 未优于基线，或动作样本量低于基线的 35%",
        "params": best_candidate["params"],
        "train_metrics": {"baseline": baseline_train, "candidate": best_candidate["train_metrics"]},
        "validation_metrics": {"baseline": baseline_validation, "candidate": candidate_validation},
    }


def _hydrate_profile(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None

    profile = dict(row)
    profile["learning_profile_id"] = int(profile.get("learning_profile_id", 0) or 0)
    profile["retention_issue_count"] = int(profile.get("retention_issue_count", DEFAULT_ISSUE_RETENTION_COUNT) or DEFAULT_ISSUE_RETENTION_COUNT)
    profile["window_value"] = int(profile.get("window_value", DEFAULT_ISSUE_RETENTION_COUNT) or DEFAULT_ISSUE_RETENTION_COUNT)
    profile["total_samples"] = int(profile.get("total_samples", 0) or 0)
    profile["training_samples"] = int(profile.get("training_samples", 0) or 0)
    profile["validation_samples"] = int(profile.get("validation_samples", 0) or 0)
    profile["training_action_samples"] = int(profile.get("training_action_samples", 0) or 0)
    profile["validation_action_samples"] = int(profile.get("validation_action_samples", 0) or 0)
    profile["calibrator_params"] = _loads_json(profile.get("calibrator_params"), {})
    profile["threshold_params"] = _loads_json(profile.get("threshold_params"), {})
    profile["train_metrics"] = _loads_json(profile.get("train_metrics"), {})
    profile["validation_metrics"] = _loads_json(profile.get("validation_metrics"), {})
    profile["sample_summary"] = _loads_json(profile.get("sample_summary"), {})
    profile["target_metrics"] = profile["validation_metrics"].get("target_metrics", {})
    strategy_section = profile["validation_metrics"].get("target_strategy", {})
    if not isinstance(strategy_section, Mapping):
        strategy_section = {}
    profile["strategy_params"] = strategy_section.get("params", {})
    profile["strategy_metrics"] = strategy_section.get("validation", {})
    profile["strategy_diagnostics"] = strategy_section
    profile["strategy_status"] = str(strategy_section.get("status", "") or "")
    profile["strategy_reason"] = str(strategy_section.get("reason", "") or "")
    profile["status_label"] = _status_label(str(profile.get("status", "") or ""))
    profile["calibrator_status_label"] = _component_status_label(str(profile.get("calibrator_status", "") or ""))
    profile["threshold_status_label"] = _component_status_label(str(profile.get("threshold_status", "") or ""))
    profile["ready_for_activation"] = str(profile.get("status", "") or "") in {"ready_candidate", "active"}
    profile["uses_calibrator"] = str(profile.get("calibrator_status", "") or "") == "ready"
    profile["uses_thresholds"] = str(profile.get("threshold_status", "") or "") == "ready"
    profile["uses_target_strategy"] = profile["strategy_status"] == "ready" and bool(profile["strategy_params"])
    return profile


def get_active_learning_profile_config() -> dict[str, Any] | None:
    init_db()
    profile = _hydrate_profile(get_active_learning_profile())
    if profile is None:
        return None
    if str(profile.get("status", "") or "") != "active":
        return None
    return profile


def get_learning_overview(window_issue_count: Any | None = None) -> dict[str, Any]:
    init_db()
    latest_candidate = _hydrate_profile(
        get_latest_learning_profile(
            statuses=("ready_candidate", "insufficient_samples", "no_gain"),
        )
    )
    active_profile = _hydrate_profile(get_active_learning_profile())
    default_window = (
        latest_candidate.get("retention_issue_count")
        if latest_candidate
        else (active_profile.get("retention_issue_count") if active_profile else DEFAULT_LEARNING_WINDOW_ISSUE_COUNT)
    )
    explicit_window = window_issue_count not in (None, "")
    window_count = normalize_learning_window_issue_count(window_issue_count if explicit_window else default_window)
    rows = _sorted_learning_rows(window_count)
    settled_samples = len(rows)
    settled_issue_count = len(_rows_by_issue(rows))
    active_strategy_replay: dict[str, Any] = {}
    if active_profile and active_profile.get("uses_target_strategy"):
        strategy_params = active_profile.get("strategy_params", {})
        if (
            isinstance(strategy_params, Mapping)
            and str(strategy_params.get("strategy_kind", "") or "") == "handicap_bucket_table"
        ):
            replay_rows = list_backtest_rows(limit=None)
            active_strategy_replay = {
                **_handicap_bucket_strategy_metrics(replay_rows, strategy_params),
                "strategy_kind": "handicap_bucket_table",
                "profile_id": int(active_profile.get("learning_profile_id", 0) or 0),
                "target_hit_rate": safe_float(
                    active_profile.get("target_metrics", {}).get(
                        "target_hit_rate",
                        DEFAULT_HANDICAP_TARGET_HIT_RATE,
                    )
                ),
                "min_action_share": safe_float(
                    active_profile.get("target_metrics", {}).get(
                        "min_action_share",
                        DEFAULT_HANDICAP_MIN_ACTION_SHARE,
                    )
                ),
            }
    recent_profiles = [
        _hydrate_profile(item)
        for item in list_learning_profiles(limit=5)
    ]
    return {
        "retention_issue_count": window_count,
        "min_retention_issue_count": MIN_LEARNING_WINDOW_ISSUE_COUNT,
        "max_retention_issue_count": MAX_LEARNING_WINDOW_ISSUE_COUNT,
        "settled_samples": settled_samples,
        "settled_issue_count": settled_issue_count,
        "required_issue_count": window_count,
        "action_samples": _count_actionable_rows(rows),
        "active_strategy_replay": active_strategy_replay,
        "active_profile": active_profile,
        "latest_candidate": latest_candidate,
        "recent_profiles": [item for item in recent_profiles if item is not None],
    }


def train_learning_profile(
    window_issue_count: Any | None = None,
    progress_callback=None,
    *,
    target_hit_rate: float = DEFAULT_HANDICAP_TARGET_HIT_RATE,
    min_action_share: float = DEFAULT_HANDICAP_MIN_ACTION_SHARE,
) -> dict[str, Any]:
    init_db()
    window_count = normalize_learning_window_issue_count(window_issue_count)
    _emit_progress(
        progress_callback,
        total_items=1,
        completed_items=0,
        current_item_index=1,
        current_item_label="学习闭环",
        current_step="准备训练样本",
        message="正在整理最近滚动窗口内的已结算样本",
    )
    rows = _sorted_learning_rows(window_count)
    issue_groups = _rows_by_issue(rows)
    settled_issue_count = len(issue_groups)
    train_rows, validation_rows = _latest_issue_holdout_split(rows)
    full_window_ready = settled_issue_count >= window_count
    summary = {
        "all_rows": _sample_summary(rows),
        "train_rows": _sample_summary(train_rows),
        "validation_rows": _sample_summary(validation_rows),
        "settled_issue_count": settled_issue_count,
        "required_issue_count": window_count,
        "review_signals": {
            "all_rows": _review_signal_summary(rows),
            "train_rows": _review_signal_summary(train_rows),
            "validation_rows": _review_signal_summary(validation_rows),
        },
    }

    _emit_progress(
        progress_callback,
        total_items=1,
        completed_items=0,
        current_item_index=1,
        current_item_label="学习闭环",
        current_step="训练概率校准器",
        message="正在训练 final 概率校准器",
    )
    calibrator_fit = _fit_calibrator(train_rows)
    calibrator_validation = _validate_calibrator(validation_rows, calibrator_fit)
    calibrator_status = calibrator_validation["status"]
    calibrator_params = calibrator_fit["params"] if calibrator_status == "ready" else {}

    _emit_progress(
        progress_callback,
        total_items=1,
        completed_items=0,
        current_item_index=1,
        current_item_label="学习闭环",
        current_step="搜索动作阈值",
        message="正在回放 algorithm/final 动作阈值候选",
    )
    threshold_fit = _fit_thresholds(train_rows, validation_rows, calibrator_params or None)
    threshold_status = threshold_fit["status"]
    threshold_params = threshold_fit["params"] if threshold_status == "ready" else {}

    _emit_progress(
        progress_callback,
        total_items=1,
        completed_items=0,
        current_item_index=1,
        current_item_label="学习闭环",
        current_step="搜索让球盘策略",
        message="正在验证让球盘执行覆盖与命中约束",
    )
    if full_window_ready:
        target_strategy = fit_handicap_target_strategy(
            train_rows,
            validation_rows,
            target_hit_rate=target_hit_rate,
            min_action_share=min_action_share,
        )
    else:
        target_strategy = {
            "status": "insufficient_samples",
            "reason": f"完整已结算期数 {settled_issue_count} < {window_count}",
            "params": {},
            "target_metrics": {
                "target_hit_rate": target_hit_rate,
                "min_action_share": min_action_share,
                "max_watch_share": 1.0 - min_action_share,
                "strategy_kind": "handicap",
            },
            "train_metrics": _empty_target_strategy_metrics(len(train_rows)),
            "validation_metrics": _empty_target_strategy_metrics(len(validation_rows)),
            "best_candidate": {},
            "folds": [],
            "skipped_folds": [],
            "frontier": {},
        }
    target_strategy["review_signals"] = summary["review_signals"]
    strategy_status = str(target_strategy.get("status", "") or "")
    strategy_params = target_strategy.get("params", {}) if strategy_status == "ready" else {}

    overall_status = "ready_candidate" if strategy_status == "ready" else strategy_status
    notes: list[str] = []
    if calibrator_status == "ready":
        notes.append("概率校准通过验证")
    elif calibrator_validation["reason"]:
        notes.append(f"概率校准：{calibrator_validation['reason']}")
    if threshold_status == "ready":
        notes.append("动作阈值通过验证")
    elif threshold_fit["reason"]:
        notes.append(f"动作阈值：{threshold_fit['reason']}")

    if strategy_status == "ready":
        validation_strategy = target_strategy["validation_metrics"]
        notes.append(
            "让球盘策略验证通过："
            f"{safe_float(validation_strategy.get('hit_rate')) * 100:.1f}% 命中，"
            f"{safe_float(validation_strategy.get('action_share')) * 100:.1f}% 执行"
        )
    else:
        notes.append(f"让球盘策略未启用：{target_strategy.get('reason', '')}")
    notes.append(f"闭环期数：{settled_issue_count}/{window_count}")

    if strategy_status == "ready":
        overall_status = "ready_candidate"
    elif strategy_status == "insufficient_samples":
        overall_status = "insufficient_samples"
    elif strategy_status == "target_unreachable":
        overall_status = "no_gain"
    elif calibrator_status != "ready" and threshold_status != "ready":
        if calibrator_status == "insufficient_samples" and threshold_status == "insufficient_samples":
            overall_status = "insufficient_samples"
        else:
            overall_status = "no_gain"

    now_text = _now_text()
    profile_id = save_learning_profile(
        {
            "status": overall_status,
            "created_at": now_text,
            "updated_at": now_text,
            "activated_at": "",
            "archived_at": "",
            "retention_issue_count": window_count,
            "window_type": "rolling_issues",
            "window_value": window_count,
            "total_samples": len(rows),
            "training_samples": len(rows),
            "validation_samples": int(target_strategy["validation_metrics"].get("sample_count", 0)) if target_strategy.get("validation_metrics") else 0,
            "training_action_samples": max(
                _count_actionable_rows(rows),
                int(threshold_fit["train_metrics"]["baseline"]["final"]["action_count"]) if threshold_fit.get("train_metrics") else 0,
            ),
            "validation_action_samples": max(
                _count_actionable_rows(validation_rows),
                int(threshold_fit["validation_metrics"]["baseline"]["final"]["action_count"]) if threshold_fit.get("validation_metrics") else 0,
                int(target_strategy["validation_metrics"].get("action_count", 0)) if target_strategy.get("validation_metrics") else 0,
            ),
            "calibrator_status": calibrator_status,
            "threshold_status": "ready" if strategy_status == "ready" else threshold_status,
            "calibrator_params": _dumps_json(calibrator_params),
            "threshold_params": _dumps_json(
                {
                    **threshold_params,
                    "target_strategy": strategy_params,
                }
                if strategy_status == "ready"
                else threshold_params
            ),
            "train_metrics": _dumps_json(
                {
                    "calibration": calibrator_fit["train_metrics"],
                    "thresholds": threshold_fit["train_metrics"],
                    "target_strategy": target_strategy["train_metrics"],
                }
            ),
            "validation_metrics": _dumps_json(
                {
                    "calibration": {
                        "baseline": calibrator_validation["baseline"],
                        "candidate": calibrator_validation["candidate"],
                    },
                    "thresholds": threshold_fit["validation_metrics"],
                    "target_metrics": target_strategy["target_metrics"],
                    "target_strategy": {
                        "status": strategy_status,
                        "reason": str(target_strategy.get("reason", "") or ""),
                        "params": strategy_params,
                        "train": target_strategy["train_metrics"],
                        "validation": target_strategy["validation_metrics"],
                        "best_candidate": target_strategy.get("best_candidate", {}),
                        "frontier": target_strategy.get("frontier", {}),
                        "folds": target_strategy.get("folds", []),
                        "skipped_folds": target_strategy.get("skipped_folds", []),
                        "review_signals": summary["review_signals"],
                    },
                }
            ),
            "sample_summary": _dumps_json(summary),
            "notes": "；".join(note for note in notes if note),
        }
    )

    if overall_status == "ready_candidate":
        task_message = (
            f"学习训练完成：生成候选配置 #{profile_id}，"
            f"校准器 {COMPONENT_STATUS_LABELS.get(calibrator_status, calibrator_status)}，"
            f"阈值器 {COMPONENT_STATUS_LABELS.get(threshold_status, threshold_status)}。"
        )
        status_level = "success"
    elif overall_status == "insufficient_samples":
        task_message = (
            f"学习训练完成：当前仅有 {settled_issue_count}/{window_count} 个完整已结算期，"
            f"候选配置 #{profile_id} 标记为样本不足。"
        )
        status_level = "warning"
    else:
        task_message = (
            f"学习训练完成：候选配置 #{profile_id} 未通过验证增益门槛，"
            "已保留离线对比结果供复核。"
        )
        status_level = "warning"

    _emit_progress(
        progress_callback,
        total_items=1,
        completed_items=1,
        current_item_index=1,
        current_item_label="学习闭环",
        current_step="学习训练完成",
        message=task_message,
        level=status_level,
    )
    return {
        "learning_profile_id": profile_id,
        "status": overall_status,
        "strategy_status": strategy_status,
        "strategy_params": strategy_params,
        "strategy_metrics": target_strategy["validation_metrics"],
        "strategy_diagnostics": target_strategy,
        "task_message": task_message,
        "status_message": task_message,
        "status_level": status_level,
    }


def activate_learning_profile(profile_id: int, progress_callback=None) -> dict[str, Any]:
    init_db()
    _emit_progress(
        progress_callback,
        total_items=1,
        completed_items=0,
        current_item_index=1,
        current_item_label="学习闭环",
        current_step="切换启用配置",
        message=f"正在启用学习配置 #{profile_id}",
    )
    profile = _hydrate_profile(get_learning_profile(profile_id))
    if profile is None:
        task_message = f"未找到 learning_profile_id={profile_id}，无法启用。"
        return {
            "learning_profile_id": profile_id,
            "task_message": task_message,
            "status_message": task_message,
            "status_level": "warning",
        }
    if str(profile.get("status", "") or "") not in {"ready_candidate", "active"}:
        task_message = f"学习配置 #{profile_id} 当前状态为 {profile['status_label']}，不能启用。"
        return {
            "learning_profile_id": profile_id,
            "task_message": task_message,
            "status_message": task_message,
            "status_level": "warning",
        }

    now_text = _now_text()
    for active_row in list_learning_profiles(limit=20, statuses=("active",)):
        active_profile = _hydrate_profile(active_row)
        if active_profile is None:
            continue
        if int(active_profile["learning_profile_id"]) == int(profile_id):
            continue
        active_profile["status"] = "archived"
        active_profile["updated_at"] = now_text
        active_profile["archived_at"] = now_text
        active_profile["ready_for_activation"] = False
        save_learning_profile(
            {
                "learning_profile_id": active_profile["learning_profile_id"],
                "status": active_profile["status"],
                "created_at": active_profile["created_at"],
                "updated_at": active_profile["updated_at"],
                "activated_at": active_profile["activated_at"],
                "archived_at": active_profile["archived_at"],
                "retention_issue_count": active_profile["retention_issue_count"],
                "window_type": active_profile["window_type"],
                "window_value": active_profile["window_value"],
                "total_samples": active_profile["total_samples"],
                "training_samples": active_profile["training_samples"],
                "validation_samples": active_profile["validation_samples"],
                "training_action_samples": active_profile["training_action_samples"],
                "validation_action_samples": active_profile["validation_action_samples"],
                "calibrator_status": active_profile["calibrator_status"],
                "threshold_status": active_profile["threshold_status"],
                "calibrator_params": _dumps_json(active_profile["calibrator_params"]),
                "threshold_params": _dumps_json(active_profile["threshold_params"]),
                "train_metrics": _dumps_json(active_profile["train_metrics"]),
                "validation_metrics": _dumps_json(active_profile["validation_metrics"]),
                "sample_summary": _dumps_json(active_profile["sample_summary"]),
                "notes": str(active_profile.get("notes", "") or ""),
            }
        )

    profile["status"] = "active"
    profile["updated_at"] = now_text
    profile["activated_at"] = now_text
    profile["archived_at"] = ""
    save_learning_profile(
        {
            "learning_profile_id": profile["learning_profile_id"],
            "status": profile["status"],
            "created_at": profile["created_at"],
            "updated_at": profile["updated_at"],
            "activated_at": profile["activated_at"],
            "archived_at": profile["archived_at"],
            "retention_issue_count": profile["retention_issue_count"],
            "window_type": profile["window_type"],
            "window_value": profile["window_value"],
            "total_samples": profile["total_samples"],
            "training_samples": profile["training_samples"],
            "validation_samples": profile["validation_samples"],
            "training_action_samples": profile["training_action_samples"],
            "validation_action_samples": profile["validation_action_samples"],
            "calibrator_status": profile["calibrator_status"],
            "threshold_status": profile["threshold_status"],
            "calibrator_params": _dumps_json(profile["calibrator_params"]),
            "threshold_params": _dumps_json(profile["threshold_params"]),
            "train_metrics": _dumps_json(profile["train_metrics"]),
            "validation_metrics": _dumps_json(profile["validation_metrics"]),
            "sample_summary": _dumps_json(profile["sample_summary"]),
            "notes": str(profile.get("notes", "") or ""),
        }
    )
    task_message = f"学习配置 #{profile_id} 已启用，后续预测将按可用组件应用校准与阈值。"
    _emit_progress(
        progress_callback,
        total_items=1,
        completed_items=1,
        current_item_index=1,
        current_item_label="学习闭环",
        current_step="学习配置已启用",
        message=task_message,
        level="success",
    )
    return {
        "learning_profile_id": profile_id,
        "task_message": task_message,
        "status_message": task_message,
        "status_level": "success",
    }


def deactivate_learning_profile(progress_callback=None) -> dict[str, Any]:
    init_db()
    _emit_progress(
        progress_callback,
        total_items=1,
        completed_items=0,
        current_item_index=1,
        current_item_label="学习闭环",
        current_step="停用启用配置",
        message="正在停用当前学习配置",
    )
    profile = _hydrate_profile(get_active_learning_profile())
    if profile is None:
        task_message = "当前没有已启用的学习配置。"
        return {
            "learning_profile_id": 0,
            "task_message": task_message,
            "status_message": task_message,
            "status_level": "warning",
        }

    now_text = _now_text()
    profile["status"] = "archived"
    profile["updated_at"] = now_text
    profile["archived_at"] = now_text
    save_learning_profile(
        {
            "learning_profile_id": profile["learning_profile_id"],
            "status": profile["status"],
            "created_at": profile["created_at"],
            "updated_at": profile["updated_at"],
            "activated_at": profile["activated_at"],
            "archived_at": profile["archived_at"],
            "retention_issue_count": profile["retention_issue_count"],
            "window_type": profile["window_type"],
            "window_value": profile["window_value"],
            "total_samples": profile["total_samples"],
            "training_samples": profile["training_samples"],
            "validation_samples": profile["validation_samples"],
            "training_action_samples": profile["training_action_samples"],
            "validation_action_samples": profile["validation_action_samples"],
            "calibrator_status": profile["calibrator_status"],
            "threshold_status": profile["threshold_status"],
            "calibrator_params": _dumps_json(profile["calibrator_params"]),
            "threshold_params": _dumps_json(profile["threshold_params"]),
            "train_metrics": _dumps_json(profile["train_metrics"]),
            "validation_metrics": _dumps_json(profile["validation_metrics"]),
            "sample_summary": _dumps_json(profile["sample_summary"]),
            "notes": str(profile.get("notes", "") or ""),
        }
    )
    task_message = f"学习配置 #{profile['learning_profile_id']} 已停用，后续预测恢复基线策略。"
    _emit_progress(
        progress_callback,
        total_items=1,
        completed_items=1,
        current_item_index=1,
        current_item_label="学习闭环",
        current_step="学习配置已停用",
        message=task_message,
        level="success",
    )
    return {
        "learning_profile_id": profile["learning_profile_id"],
        "task_message": task_message,
        "status_message": task_message,
        "status_level": "success",
    }


__all__ = [
    "activate_learning_profile",
    "apply_probability_calibration",
    "deactivate_learning_profile",
    "get_active_learning_profile_config",
    "get_learning_overview",
    "normalize_learning_window_issue_count",
    "train_learning_profile",
]
