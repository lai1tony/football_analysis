from __future__ import annotations

from typing import Any, Mapping

from feature_engine import clamp, safe_float
from outcome_policy import effective_market_probs


OUTCOMES = ("home", "draw", "away")
ACTION_LEVELS = {"观望": 0, "轻仓": 1, "主推": 2}
ACTION_BY_LEVEL = {value: key for key, value in ACTION_LEVELS.items()}
DEFAULT_MAIN_GATE = {"ev": 0.08, "confidence": 0.58, "market_bias": 0.020, "quality": 0.68}
DEFAULT_LIGHT_GATE = {"ev": 0.02, "confidence": 0.48, "market_bias": 0.005, "quality": 0.58}
DEFAULT_PROMOTE_GATE = {"ev": 0.06, "confidence": 0.55, "market_bias": 0.020, "quality": 0.68}
ACTION_CONTROL = {
    "main_score": 0.74,
    "light_score": 0.50,
    "main_ev_margin": 0.030,
    "light_ev_margin": 0.005,
    "main_probability": 0.20,
    "light_probability": 0.14,
    "main_agreement": 0.70,
    "light_agreement": 0.55,
    # legacy_gap thresholds are only enforced when data quality is poor.
    # When quality clears legacy_gap_quality_floor we deliberately allow
    # large divergence from the legacy market-assisted baseline — that gap
    # is the signal we want to bet on, not punish.
    "main_legacy_gap_low_quality": 0.16,
    "light_legacy_gap_low_quality": 0.24,
    "legacy_gap_quality_floor": 0.70,
    "main_probability_margin": -0.060,
    "light_probability_margin": -0.140,
    # Hit-rate-first execution guards. Historical feedback showed that
    # actionable picks were much worse than watch-only directions, especially
    # draws, long-away picks, and higher-odds home picks. Keep the direction
    # recommendation, but require a much cleaner edge before staking.
    "home_soft_odds_cap": 1.65,
    "home_high_odds_light_ev": 0.70,
    "home_high_odds_light_market_bias": 0.350,
    "home_high_odds_light_ev_margin": 0.800,
    "home_high_odds_light_probability_margin": 0.450,
    "home_high_odds_main_score": 0.980,
    "away_light_odds_cap": 2.00,
    "away_main_odds_cap": 2.00,
    "away_light_ev": 0.32,
    "away_light_market_bias": 0.220,
    "away_light_ev_margin": 0.700,
    "away_light_probability_margin": 0.500,
    "away_light_score": 0.950,
    "away_main_ev": 0.32,
    "away_main_market_bias": 0.220,
    "away_main_ev_margin": 0.700,
    "away_main_probability_margin": 0.500,
    "away_main_score": 0.950,
    "draw_light_ev": 0.18,
    "draw_light_market_bias": 0.080,
    "draw_light_ev_margin": 0.160,
    "draw_light_probability_margin": 0.060,
    "draw_light_score": 0.900,
    "draw_odds_cap": 4.20,
    "favorite_safe_odds_cap": 1.55,
    "favorite_safe_min_quality": 0.58,
    "favorite_safe_min_agreement": 0.55,
    "favorite_safe_min_probability": 0.52,
    "favorite_safe_main_score": 0.70,
}


def _gate_value(gate: Mapping[str, Any], defaults: Mapping[str, float], key: str) -> float:
    return safe_float(gate.get(key, defaults[key]))


def _threshold_gate(
    threshold_config: Mapping[str, Any],
    gate_name: str,
    defaults: Mapping[str, float],
) -> dict[str, float]:
    raw_gate = threshold_config.get(gate_name, {}) if isinstance(threshold_config, Mapping) else {}
    if not isinstance(raw_gate, Mapping):
        raw_gate = {}
    return {key: _gate_value(raw_gate, defaults, key) for key in defaults}


def risk_level(confidence: float) -> str:
    if confidence >= 0.78:
        return "low"
    if confidence >= 0.62:
        return "medium"
    if confidence >= 0.48:
        return "high"
    return "very_high"


def action_confidence(
    *,
    quality_score: float,
    model_agreement: float,
    model_margin: float,
) -> float:
    # Confidence reflects "信号一致性 + 数据可信度" rather than how big the
    # gap between top and second probability is. Three-way close matches
    # (35/30/35) are not necessarily *less* confident than skewed ones —
    # what matters is whether quality and agreement are high. Margin still
    # contributes but no longer dominates.
    return clamp(
        0.30
        + safe_float(quality_score) * 0.34
        + safe_float(model_agreement) * 0.26
        + safe_float(model_margin) * 0.14,
        0.18,
        0.92,
    )


def expected_values_for(
    probabilities: Mapping[str, Any],
    market_odds: Mapping[str, Any],
) -> dict[str, float]:
    return {
        outcome: (
            safe_float(probabilities.get(outcome)) * safe_float(market_odds.get(outcome)) - 1.0
            if safe_float(market_odds.get(outcome)) > 0
            else -1.0
        )
        for outcome in OUTCOMES
    }


def market_bias_for(
    probabilities: Mapping[str, Any],
    market_probs: Mapping[str, Any],
) -> dict[str, float]:
    return {
        outcome: safe_float(probabilities.get(outcome)) - safe_float(market_probs.get(outcome))
        for outcome in OUTCOMES
    }


def fair_odds_for(probabilities: Mapping[str, Any]) -> dict[str, float]:
    return {
        outcome: round(1.0 / max(safe_float(probabilities.get(outcome)), 0.001), 2)
        for outcome in OUTCOMES
    }


def probability_margin_for(probabilities: Mapping[str, Any], outcome: str) -> float:
    selected = safe_float(probabilities.get(outcome))
    alternatives = [
        safe_float(probabilities.get(item))
        for item in OUTCOMES
        if item != outcome
    ]
    return selected - max(alternatives or [0.0])


def ev_margin_for(expected_values: Mapping[str, Any], outcome: str) -> float:
    selected = safe_float(expected_values.get(outcome))
    alternatives = [
        safe_float(expected_values.get(item))
        for item in OUTCOMES
        if item != outcome
    ]
    return selected - max(alternatives or [-1.0])


def legacy_gap_for(
    legacy_probabilities: Mapping[str, Any] | None,
    probabilities: Mapping[str, Any],
    outcome: str,
) -> float:
    if not isinstance(legacy_probabilities, Mapping):
        return 0.0
    legacy_values = [safe_float(legacy_probabilities.get(item)) for item in OUTCOMES]
    if sum(legacy_values) <= 0:
        return 0.0
    return abs(safe_float(legacy_probabilities.get(outcome)) - safe_float(probabilities.get(outcome)))


def _score_component(value: float, target: float) -> float:
    if target <= 0:
        return 1.0 if value > 0 else 0.0
    return clamp(value / target, 0.0, 1.0)


def action_evidence_score(
    *,
    best_ev: float,
    best_bias: float,
    confidence: float,
    quality_score: float,
    model_agreement: float,
    ev_margin: float,
    outcome_probability: float,
    legacy_gap: float,
    threshold_config: Mapping[str, Any],
) -> tuple[float, dict[str, float]]:
    main_gate = _threshold_gate(threshold_config, "main", DEFAULT_MAIN_GATE)
    light_gate = _threshold_gate(threshold_config, "light", DEFAULT_LIGHT_GATE)
    # legacy_alignment was a flat penalty for divergence from the legacy
    # market-assisted baseline. Because that baseline is ~70% market, the
    # penalty was effectively "punish disagreeing with the market" — which
    # is the opposite of what a value-betting system should do. Neutralize
    # it when quality is solid; only keep a soft tilt when quality is poor.
    quality_floor = safe_float(ACTION_CONTROL["legacy_gap_quality_floor"])
    if safe_float(quality_score) >= quality_floor:
        legacy_alignment_factor = 1.0
    else:
        legacy_alignment_factor = clamp(1.0 - max(legacy_gap - 0.10, 0.0) / 0.20, 0.0, 1.0)

    factors = {
        "ev": _score_component(best_ev, max(main_gate["ev"], 0.001)),
        "confidence": _score_component(
            confidence - light_gate["confidence"],
            max(main_gate["confidence"] - light_gate["confidence"], 0.001),
        ),
        "quality": _score_component(
            quality_score - light_gate["quality"],
            max(main_gate["quality"] - light_gate["quality"], 0.001),
        ),
        "market_bias": _score_component(best_bias, max(main_gate["market_bias"], 0.001)),
        "ev_margin": _score_component(ev_margin, 0.08),
        "agreement": _score_component(model_agreement - 0.56, 0.28),
        "legacy_alignment": legacy_alignment_factor,
        "outcome_probability": _score_component(outcome_probability - 0.18, 0.18),
    }
    score = (
        factors["ev"] * 0.26
        + factors["confidence"] * 0.14
        + factors["quality"] * 0.14
        + factors["market_bias"] * 0.16
        + factors["ev_margin"] * 0.12
        + factors["agreement"] * 0.08
        + factors["legacy_alignment"] * 0.04
        + factors["outcome_probability"] * 0.06
    )
    return round(clamp(score, 0.0, 1.0), 4), factors


def kelly_fraction_for(
    *,
    probability: float,
    odds: float,
) -> float:
    probability = safe_float(probability)
    odds = safe_float(odds)
    if probability <= 0 or odds <= 1.0:
        return 0.0
    return clamp((probability * odds - 1.0) / (odds - 1.0), 0.0, 1.0)


def stake_for_action(action: str, risk: Mapping[str, Any]) -> float:
    """Compute the bankroll % to stake.

    Design notes (post-rewrite):
    - Use ~1/4 Kelly as the base ("fractional Kelly" is industry standard
      for sports betting; full Kelly is too aggressive given parameter
      uncertainty). The previous implementation stacked five multipliers
      that collapsed even strong signals to ~1% — the reciprocal of value
      betting's whole point.
    - Soft tilt by data quality / model agreement / action_score (±25%).
      These are evidence-driven scalars, not hard cuts.
    - Only penalize legacy_gap when data quality is poor; otherwise
      divergence from the legacy market-assisted baseline is the alpha
      we are paying for, not something to discount.
    - Cap main 推 at 4% bankroll, 轻仓 at 1.5%. Floors stay tight so a
      qualifying main 推 still differs from a 轻仓.
    """

    if action == "观望":
        return 0.0

    outcome = str(risk.get("recommended_outcome", "") or "")
    if outcome not in OUTCOMES:
        return 0.0

    kelly_fraction = safe_float(risk.get("kelly_fraction"))
    if kelly_fraction <= 0:
        probabilities = risk.get("probabilities", {})
        market_odds = risk.get("market_odds", {})
        if isinstance(probabilities, Mapping) and isinstance(market_odds, Mapping):
            kelly_fraction = kelly_fraction_for(
                probability=safe_float(probabilities.get(outcome)),
                odds=safe_float(market_odds.get(outcome)),
            )

    confidence = safe_float(risk.get("confidence"))
    quality_score = safe_float(risk.get("quality_score", risk.get("quality", 0.70)))
    model_agreement = safe_float(risk.get("model_agreement", risk.get("agreement", 0.70)))
    action_score = safe_float(risk.get("action_score", 0.60))
    legacy_gap = safe_float(risk.get("legacy_gap"))

    # EV/confidence fallback when kelly cannot be computed (e.g. odds missing).
    if kelly_fraction <= 0:
        expected_values = risk.get("expected_values", {})
        ev = safe_float(expected_values.get(outcome)) if isinstance(expected_values, Mapping) else 0.0
        base = max(ev, 0.0) * confidence
        if action == "主推":
            return round(clamp(base * 12.0, 0.5, 4.0), 2)
        if action == "轻仓":
            return round(clamp(base * 6.0, 0.2, 1.5), 2)
        return 0.0

    # 1/4 Kelly anchor. Scale slightly with action_score / confidence /
    # agreement — strong evidence approaches 1/3 Kelly, weak evidence
    # falls back toward ~1/6 Kelly. Multipliers operate on a base that
    # is no longer crushed by an additive 0.06 floor.
    evidence_blend = clamp(
        0.55
        + (confidence - 0.55) * 0.45
        + (action_score - 0.55) * 0.40
        + (model_agreement - 0.65) * 0.25
        + (quality_score - 0.65) * 0.20,
        0.40,
        1.30,
    )
    quarter_kelly = kelly_fraction * 0.25

    # Only when quality is poor do we fade the stake against legacy divergence.
    quality_floor = safe_float(ACTION_CONTROL["legacy_gap_quality_floor"])
    if quality_score < quality_floor and legacy_gap > 0.10:
        legacy_multiplier = clamp(1.0 - (legacy_gap - 0.10) * 1.6, 0.55, 1.0)
    else:
        legacy_multiplier = 1.0

    stake_pct = quarter_kelly * 100.0 * evidence_blend * legacy_multiplier
    if action == "主推":
        return round(clamp(stake_pct, 0.5, 4.0), 2)
    if action == "轻仓":
        return round(clamp(stake_pct, 0.2, 1.5), 2)
    return 0.0


def evaluate_action_policy(
    *,
    probabilities: Mapping[str, Any],
    market_odds: Mapping[str, Any],
    market_probs: Mapping[str, Any],
    legacy_probabilities: Mapping[str, Any] | None,
    quality_score: float,
    model_agreement: float,
    model_margin: float,
    threshold_config: Mapping[str, Any],
    selected_outcome: str | None = None,
) -> dict[str, Any]:
    probabilities = {outcome: safe_float(probabilities.get(outcome)) for outcome in OUTCOMES}
    market_odds = {outcome: safe_float(market_odds.get(outcome)) for outcome in OUTCOMES}
    market_probs = effective_market_probs(market_probs, market_odds)
    expected_values = expected_values_for(probabilities, market_odds)
    market_bias = market_bias_for(probabilities, market_probs)
    fair_odds = fair_odds_for(probabilities)
    best_outcome = str(selected_outcome or "").strip()
    if best_outcome not in OUTCOMES:
        best_outcome = max(expected_values, key=expected_values.get)
    best_ev = safe_float(expected_values.get(best_outcome))
    best_bias = safe_float(market_bias.get(best_outcome))
    outcome_probability = safe_float(probabilities.get(best_outcome))
    ev_margin = ev_margin_for(expected_values, best_outcome)
    probability_margin = probability_margin_for(probabilities, best_outcome)
    legacy_gap = legacy_gap_for(legacy_probabilities, probabilities, best_outcome)
    confidence = action_confidence(
        quality_score=quality_score,
        model_agreement=model_agreement,
        model_margin=model_margin,
    )
    score, score_factors = action_evidence_score(
        best_ev=best_ev,
        best_bias=best_bias,
        confidence=confidence,
        quality_score=safe_float(quality_score),
        model_agreement=safe_float(model_agreement),
        ev_margin=ev_margin,
        outcome_probability=outcome_probability,
        legacy_gap=legacy_gap,
        threshold_config=threshold_config,
    )
    kelly_fraction = kelly_fraction_for(
        probability=outcome_probability,
        odds=safe_float(market_odds.get(best_outcome)),
    )

    main_gate = _threshold_gate(threshold_config, "main", DEFAULT_MAIN_GATE)
    light_gate = _threshold_gate(threshold_config, "light", DEFAULT_LIGHT_GATE)
    quality_value = safe_float(quality_score)
    quality_floor = safe_float(ACTION_CONTROL["legacy_gap_quality_floor"])
    high_quality = quality_value >= quality_floor
    # legacy_gap is only enforced as a hard cap when data quality is poor.
    # When quality is healthy, divergence from the legacy market-assisted
    # baseline is the source of value, not a risk to gate against.
    light_legacy_cap = (
        float("inf") if high_quality else safe_float(ACTION_CONTROL["light_legacy_gap_low_quality"])
    )
    main_legacy_cap = (
        float("inf") if high_quality else safe_float(ACTION_CONTROL["main_legacy_gap_low_quality"])
    )
    selected_odds = safe_float(market_odds.get(best_outcome))
    execution_guard_reasons: list[str] = []
    light_execution_guard = True
    main_execution_guard = True
    low_odds_favorite_guard = (
        best_outcome in {"home", "away"}
        and 0 < selected_odds <= ACTION_CONTROL["favorite_safe_odds_cap"]
        and quality_value >= ACTION_CONTROL["favorite_safe_min_quality"]
        and safe_float(model_agreement) >= ACTION_CONTROL["favorite_safe_min_agreement"]
        and outcome_probability >= ACTION_CONTROL["favorite_safe_min_probability"]
    )
    if best_outcome == "draw":
        draw_guard = (
            selected_odds <= ACTION_CONTROL["draw_odds_cap"]
            and best_ev >= ACTION_CONTROL["draw_light_ev"]
            and best_bias >= ACTION_CONTROL["draw_light_market_bias"]
            and ev_margin >= ACTION_CONTROL["draw_light_ev_margin"]
            and probability_margin >= ACTION_CONTROL["draw_light_probability_margin"]
            and score >= ACTION_CONTROL["draw_light_score"]
        )
        light_execution_guard = draw_guard
        main_execution_guard = False
        if not draw_guard:
            execution_guard_reasons.append("平局历史执行命中偏低，未达到高置信保护门槛。")
    elif best_outcome == "away" and not low_odds_favorite_guard:
        light_execution_guard = (
            selected_odds <= ACTION_CONTROL["away_light_odds_cap"]
            and best_ev >= ACTION_CONTROL["away_light_ev"]
            and best_bias >= ACTION_CONTROL["away_light_market_bias"]
            and ev_margin >= ACTION_CONTROL["away_light_ev_margin"]
            and probability_margin >= ACTION_CONTROL["away_light_probability_margin"]
            and score >= ACTION_CONTROL["away_light_score"]
        )
        main_execution_guard = (
            selected_odds <= ACTION_CONTROL["away_main_odds_cap"]
            and best_ev >= ACTION_CONTROL["away_main_ev"]
            and best_bias >= ACTION_CONTROL["away_main_market_bias"]
            and ev_margin >= ACTION_CONTROL["away_main_ev_margin"]
            and probability_margin >= ACTION_CONTROL["away_main_probability_margin"]
            and score >= ACTION_CONTROL["away_main_score"]
        )
        if not light_execution_guard:
            execution_guard_reasons.append("客胜执行分桶历史命中偏低，未达到低赔率强优势门槛。")
    elif best_outcome == "home" and selected_odds > ACTION_CONTROL["home_soft_odds_cap"] and not low_odds_favorite_guard:
        high_odds_home_guard = (
            best_ev >= ACTION_CONTROL["home_high_odds_light_ev"]
            and best_bias >= ACTION_CONTROL["home_high_odds_light_market_bias"]
            and ev_margin >= ACTION_CONTROL["home_high_odds_light_ev_margin"]
            and probability_margin >= ACTION_CONTROL["home_high_odds_light_probability_margin"]
            and score >= ACTION_CONTROL["home_high_odds_main_score"]
        )
        light_execution_guard = high_odds_home_guard
        main_execution_guard = high_odds_home_guard
        if not high_odds_home_guard:
            execution_guard_reasons.append("高赔率主胜执行分桶历史命中偏低，未达到强优势门槛。")
    recommendation = "观望"

    light_pass = (
        best_ev >= light_gate["ev"]
        and best_bias >= light_gate["market_bias"]
        and confidence >= light_gate["confidence"]
        and quality_value >= light_gate["quality"]
        and ev_margin >= ACTION_CONTROL["light_ev_margin"]
        and outcome_probability >= ACTION_CONTROL["light_probability"]
        and probability_margin >= ACTION_CONTROL["light_probability_margin"]
        and safe_float(model_agreement) >= ACTION_CONTROL["light_agreement"]
        and legacy_gap <= light_legacy_cap
        and score >= ACTION_CONTROL["light_score"]
        and light_execution_guard
    )
    # main_pass no longer requires probability_margin to be near zero or
    # positive — that constraint forced the betting target to coincide
    # with argmax(probability) and effectively killed value-direction
    # main bets. We still demand a wide EV margin and a strong market bias.
    main_pass = (
        light_pass
        and best_ev >= main_gate["ev"]
        and best_bias >= main_gate["market_bias"]
        and confidence >= main_gate["confidence"]
        and quality_value >= main_gate["quality"]
        and ev_margin >= ACTION_CONTROL["main_ev_margin"]
        and outcome_probability >= ACTION_CONTROL["main_probability"]
        and probability_margin >= ACTION_CONTROL["main_probability_margin"]
        and safe_float(model_agreement) >= ACTION_CONTROL["main_agreement"]
        and legacy_gap <= main_legacy_cap
        and score >= ACTION_CONTROL["main_score"]
        and main_execution_guard
    )
    if low_odds_favorite_guard and not light_pass:
        light_pass = True
        main_pass = score >= ACTION_CONTROL["favorite_safe_main_score"]

    risk_context = {
        "recommended_outcome": best_outcome,
        "probabilities": probabilities,
        "market_odds": market_odds,
        "expected_values": expected_values,
        "confidence": confidence,
        "quality_score": quality_value,
        "model_agreement": safe_float(model_agreement),
        "action_score": score,
        "legacy_gap": legacy_gap,
        "kelly_fraction": kelly_fraction,
    }
    if main_pass:
        recommendation = "主推"
    elif light_pass:
        recommendation = "轻仓"

    warnings: list[str] = []
    if best_ev <= 0:
        warnings.append("独立概率没有跑赢市场赔率，当前不具备正 EV。")
    if best_bias < light_gate["market_bias"]:
        warnings.append("推荐方向相对市场没有足够正向概率偏差。")
    if ev_margin < ACTION_CONTROL["light_ev_margin"]:
        warnings.append("最佳 EV 与次优选项差距过小，动作强度降温。")
    if probability_margin < 0:
        warnings.append("推荐方向不是最高概率方向，主要依赖赔率价值。")
    if legacy_gap >= 0.10 and not high_quality:
        warnings.append("数据质量偏弱时与旧市场辅助模型分歧较大，建议人工复核。")
    if score < ACTION_CONTROL["light_score"]:
        warnings.append("动作评分不足，建议观望。")
    warnings.extend(execution_guard_reasons)

    return {
        "fair_odds": fair_odds,
        "expected_values": expected_values,
        "market_bias": market_bias,
        "market_probs": market_probs,
        "market_odds": market_odds,
        "probabilities": probabilities,
        "confidence": confidence,
        "risk_level": risk_level(confidence),
        "recommended_outcome": best_outcome,
        "recommendation": recommendation,
        "stake_pct": stake_for_action(recommendation, risk_context),
        "action_score": score,
        "action_score_factors": score_factors,
        "probability_margin": probability_margin,
        "ev_margin": ev_margin,
        "legacy_gap": legacy_gap,
        "kelly_fraction": kelly_fraction,
        "low_odds_favorite_guard": low_odds_favorite_guard,
        "warnings": warnings,
    }
