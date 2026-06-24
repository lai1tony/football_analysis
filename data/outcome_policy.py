from __future__ import annotations

from typing import Any, Mapping

from feature_engine import clamp, safe_float


OUTCOMES = ("home", "draw", "away")
OUTCOME_CONTROL = {
    "same_direction_score": 0.5,
    "switch_direction_score": 0.6,
    "same_direction_ev": 0.015,
    "switch_direction_ev": 0.04,
    "switch_market_bias": 0.01,
    "switch_ev_margin": 0.025,
    "switch_probability": 0.18,
    "switch_probability_margin": -0.16,
    "switch_agreement": 0.58,
    "draw_switch_score": 0.48,
    "draw_switch_ev": 0.015,
    "draw_switch_market_bias": 0.005,
    "draw_switch_ev_margin": 0.02,
    "draw_switch_probability": 0.2,
    "draw_switch_probability_margin": -0.28,
    "min_quality": 0.55,
    # legacy_gap is only treated as a hard veto when data quality is low.
    # Above this quality threshold a large gap with the legacy market-assisted
    # baseline is *not* punished: divergence from the market is exactly where
    # the value lives.
    "legacy_gap_quality_floor": 0.70,
    "legacy_gap_low_quality_cap": 0.18,
}


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


def market_probs_from_odds(market_odds: Mapping[str, Any]) -> dict[str, float]:
    implied = {
        outcome: (1.0 / safe_float(market_odds.get(outcome)) if safe_float(market_odds.get(outcome)) > 0 else 0.0)
        for outcome in OUTCOMES
    }
    total = sum(implied.values())
    if total <= 0:
        return {outcome: 1.0 / 3.0 for outcome in OUTCOMES}
    return {outcome: implied[outcome] / total for outcome in OUTCOMES}


def effective_market_probs(
    market_probs: Mapping[str, Any],
    market_odds: Mapping[str, Any],
) -> dict[str, float]:
    normalized = {outcome: safe_float(market_probs.get(outcome)) for outcome in OUTCOMES}
    if sum(normalized.values()) <= 0:
        return market_probs_from_odds(market_odds)
    total = sum(max(value, 0.0) for value in normalized.values())
    if total <= 0:
        return market_probs_from_odds(market_odds)
    return {outcome: max(normalized[outcome], 0.0) / total for outcome in OUTCOMES}


def _margin_for(values: Mapping[str, Any], outcome: str) -> float:
    selected = safe_float(values.get(outcome))
    alternatives = [
        safe_float(values.get(item))
        for item in OUTCOMES
        if item != outcome
    ]
    return selected - max(alternatives or [0.0])


def _legacy_gap(
    legacy_probabilities: Mapping[str, Any] | None,
    probabilities: Mapping[str, Any],
    outcome: str,
) -> float:
    if not isinstance(legacy_probabilities, Mapping):
        return 0.0
    values = [safe_float(legacy_probabilities.get(item)) for item in OUTCOMES]
    if sum(values) <= 0:
        return 0.0
    return abs(safe_float(legacy_probabilities.get(outcome)) - safe_float(probabilities.get(outcome)))


def _score_component(value: float, target: float) -> float:
    if target <= 0:
        return 1.0 if value > 0 else 0.0
    return clamp(value / target, 0.0, 1.0)


def _value_score(
    *,
    value_ev: float,
    value_bias: float,
    value_probability: float,
    value_ev_margin: float,
    value_probability_margin: float,
    legacy_gap: float,
    quality_score: float,
    model_agreement: float,
    outcome: str = "",
) -> float:
    # legacy_alignment is no longer a flat penalty for divergence from the
    # legacy market-assisted baseline (which is itself ~70% market). When data
    # quality is solid we *want* to disagree with the market — that's where
    # the value comes from. Only punish divergence when quality is poor.
    quality_floor = 0.70
    if safe_float(quality_score) >= quality_floor:
        legacy_alignment_factor = 1.0
    else:
        legacy_alignment_factor = clamp(1.0 - max(legacy_gap - 0.10, 0.0) / 0.20, 0.0, 1.0)

    ev_benchmark = 0.08 if outcome == "draw" else 0.12
    prob_benchmark = 0.28 if outcome == "draw" else 0.36
    factors = {
        "ev": _score_component(value_ev, ev_benchmark),
        "market_bias": _score_component(value_bias, 0.05),
        "probability": _score_component(value_probability, prob_benchmark),
        "ev_margin": _score_component(value_ev_margin, 0.10),
        "probability_margin": clamp((value_probability_margin + 0.10) / 0.20, 0.0, 1.0),
        "legacy_alignment": legacy_alignment_factor,
        "quality": _score_component(quality_score, 0.82),
        "agreement": _score_component(model_agreement, 0.86),
    }
    return round(
        clamp(
            factors["ev"] * 0.24
            + factors["market_bias"] * 0.16
            + factors["probability"] * 0.16
            + factors["ev_margin"] * 0.16
            + factors["probability_margin"] * 0.10
            + factors["legacy_alignment"] * 0.04
            + factors["quality"] * 0.08
            + factors["agreement"] * 0.06,
            0.0,
            1.0,
        ),
        4,
    )


def evaluate_outcome_policy(
    *,
    probabilities: Mapping[str, Any],
    market_odds: Mapping[str, Any],
    market_probs: Mapping[str, Any],
    legacy_probabilities: Mapping[str, Any] | None,
    quality_score: float,
    model_agreement: float,
) -> dict[str, Any]:
    probabilities = {outcome: safe_float(probabilities.get(outcome)) for outcome in OUTCOMES}
    market_odds = {outcome: safe_float(market_odds.get(outcome)) for outcome in OUTCOMES}
    market_probs = effective_market_probs(market_probs, market_odds)
    expected_values = expected_values_for(probabilities, market_odds)
    market_bias = market_bias_for(probabilities, market_probs)

    # ---------- Market-Aware Strategy (v4) ----------
    # After 179-match backtest analysis:
    #   Market favorite alone:         50.3% hit rate
    #   Model favorite alone:          46.9% hit rate
    #   Model override of market:      17.2% hit rate (worse than random!)
    #   Market odds<2.0 filter:        56.3% hit rate on 58% of matches
    #
    # Strategy: follow market direction always. The model's value is
    # in EV/market_bias/confidence computation, NOT directional prediction.
    # Draws (~25% of matches) are structurally unpreditable by any model;
    # the market itself can't pick them either (0 draw picks in 179 matches).
    
    market_pick = max(market_probs, key=market_probs.get)
    model_pick = max(probabilities, key=probabilities.get)
    predictions = {outcome: safe_float(probabilities.get(outcome)) for outcome in OUTCOMES}
    
    # Always follow market direction for the recommended outcome.
    # This gives us the highest achievable three-way hit rate (~50%).
    recommended_outcome = market_pick
    outcome_source = "market_direction"
    
    # Legacy fields for backward compatibility
    predicted_outcome = max(probabilities, key=probabilities.get)
    value_outcome = max(expected_values, key=expected_values.get)
    value_ev = safe_float(expected_values.get(value_outcome))
    value_bias = safe_float(market_bias.get(value_outcome))
    value_probability = safe_float(probabilities.get(value_outcome))
    value_ev_margin = _margin_for(expected_values, value_outcome)
    value_probability_margin = _margin_for(probabilities, value_outcome)
    value_legacy_gap = _legacy_gap(legacy_probabilities, probabilities, value_outcome)
    score = _value_score(
        value_ev=value_ev, value_bias=value_bias, value_probability=value_probability,
        value_ev_margin=value_ev_margin, value_probability_margin=value_probability_margin,
        legacy_gap=value_legacy_gap, quality_score=safe_float(quality_score),
        model_agreement=safe_float(model_agreement), outcome=value_outcome,
    )
    
    reason = "market_aware_v3: " + outcome_source
    
    warnings: list[str] = []
    if value_ev <= 0:
        warnings.append("value EV is not positive")
    if safe_float(quality_score) < OUTCOME_CONTROL["min_quality"]:
        warnings.append("data quality is below outcome policy minimum")
    
    return {
        "predicted_outcome": predicted_outcome,
        "value_outcome": value_outcome,
        "recommended_outcome": recommended_outcome,
        "outcome_source": outcome_source,
        "outcome_reason": reason,
        "outcome_score": score,
        "value_ev": value_ev,
        "value_market_bias": value_bias,
        "value_probability": value_probability,
        "value_probability_margin": value_probability_margin,
        "value_ev_margin": value_ev_margin,
        "value_legacy_gap": value_legacy_gap,
        "expected_values": expected_values,
        "market_bias": market_bias,
        "warnings": warnings,
    }

