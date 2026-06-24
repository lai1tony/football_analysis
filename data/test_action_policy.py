from __future__ import annotations

import unittest

from action_policy import evaluate_action_policy


BASE_THRESHOLDS = {
    "main": {"ev": 0.12, "confidence": 0.66, "market_bias": 0.04, "quality": 0.70},
    "light": {"ev": 0.06, "confidence": 0.56, "market_bias": 0.025, "quality": 0.60},
    "promote": {"ev": 0.10, "confidence": 0.62, "market_bias": 0.04, "quality": 0.72},
}


class ActionPolicyTests(unittest.TestCase):
    def _evaluate(self, **overrides):
        payload = {
            "probabilities": {"home": 0.52, "draw": 0.25, "away": 0.23},
            "market_odds": {"home": 2.40, "draw": 3.30, "away": 3.50},
            "market_probs": {"home": 0.43, "draw": 0.30, "away": 0.27},
            "legacy_probabilities": {"home": 0.50, "draw": 0.27, "away": 0.23},
            "quality_score": 0.90,
            "model_agreement": 0.90,
            "model_margin": 0.27,
            "threshold_config": BASE_THRESHOLDS,
        }
        payload.update(overrides)
        return evaluate_action_policy(**payload)

    def test_strong_aligned_positive_edge_can_be_main(self) -> None:
        result = self._evaluate(
            probabilities={"home": 0.70, "draw": 0.18, "away": 0.12},
            market_odds={"home": 1.62, "draw": 4.60, "away": 7.20},
            market_probs={"home": 0.56, "draw": 0.25, "away": 0.19},
            legacy_probabilities={"home": 0.68, "draw": 0.19, "away": 0.13},
            model_margin=0.52,
        )

        self.assertEqual(result["recommended_outcome"], "home")
        self.assertEqual(result["recommendation"], "主推")
        self.assertGreaterEqual(result["action_score"], 0.74)
        self.assertGreater(result["stake_pct"], 0)

    def test_positive_ev_without_positive_market_bias_stays_watch(self) -> None:
        result = self._evaluate(
            probabilities={"home": 0.44, "draw": 0.30, "away": 0.26},
            market_odds={"home": 2.60, "draw": 3.05, "away": 3.20},
            market_probs={"home": 0.48, "draw": 0.28, "away": 0.24},
            legacy_probabilities={"home": 0.43, "draw": 0.31, "away": 0.26},
            model_margin=0.14,
        )

        self.assertEqual(result["recommended_outcome"], "home")
        self.assertGreater(result["expected_values"]["home"], 0)
        self.assertLess(result["market_bias"]["home"], 0)
        self.assertEqual(result["recommendation"], "观望")

    def test_large_legacy_gap_blocks_main_action(self) -> None:
        # When data quality is high, divergence from the legacy
        # market-assisted baseline is the *signal*, not a risk to gate
        # against. Only enforce the legacy_gap cap when quality is poor.
        result = self._evaluate(
            legacy_probabilities={"home": 0.30, "draw": 0.38, "away": 0.32},
            quality_score=0.55,
            model_agreement=0.85,
        )

        self.assertGreaterEqual(result["legacy_gap"], 0.20)
        self.assertNotEqual(result["recommendation"], "主推")

    def test_tight_ev_margin_blocks_main_action(self) -> None:
        result = self._evaluate(
            probabilities={"home": 0.50, "draw": 0.25, "away": 0.25},
            market_odds={"home": 2.26, "draw": 4.48, "away": 4.30},
            market_probs={"home": 0.43, "draw": 0.28, "away": 0.29},
            legacy_probabilities={"home": 0.49, "draw": 0.26, "away": 0.25},
            model_margin=0.25,
        )

        self.assertLess(result["ev_margin"], 0.060)
        self.assertNotEqual(result["recommendation"], "主推")

    def test_value_pick_below_top_probability_is_not_main(self) -> None:
        result = self._evaluate(
            probabilities={"home": 0.42, "draw": 0.25, "away": 0.33},
            market_odds={"home": 2.25, "draw": 3.60, "away": 4.20},
            market_probs={"home": 0.47, "draw": 0.28, "away": 0.25},
            legacy_probabilities={"home": 0.43, "draw": 0.25, "away": 0.32},
            model_margin=0.09,
        )

        self.assertEqual(result["recommended_outcome"], "away")
        self.assertLess(result["probability_margin"], 0)
        self.assertNotEqual(result["recommendation"], "主推")

    def test_main_stake_is_fractional_kelly_capped(self) -> None:
        # Strong-edge bets now scale up to the 4% bankroll cap (1/4 Kelly
        # base, post-rewrite) rather than being crushed to 1%. The cap is
        # what protects against parameter uncertainty on extreme signals.
        result = self._evaluate(
            probabilities={"home": 0.72, "draw": 0.17, "away": 0.11},
            market_odds={"home": 1.60, "draw": 4.80, "away": 7.50},
            market_probs={"home": 0.57, "draw": 0.25, "away": 0.18},
            legacy_probabilities={"home": 0.70, "draw": 0.18, "away": 0.12},
            model_margin=0.55,
        )

        self.assertEqual(result["recommendation"], "主推")
        self.assertGreater(result["stake_pct"], 1.0)
        self.assertLessEqual(result["stake_pct"], 4.0)

    def test_away_pick_requires_low_odds_strong_edge(self) -> None:
        result = self._evaluate(
            probabilities={"home": 0.23, "draw": 0.24, "away": 0.53},
            market_odds={"home": 3.70, "draw": 3.40, "away": 2.75},
            market_probs={"home": 0.31, "draw": 0.29, "away": 0.40},
            legacy_probabilities={"home": 0.25, "draw": 0.25, "away": 0.50},
            model_margin=0.29,
            selected_outcome="away",
        )

        self.assertEqual(result["recommended_outcome"], "away")
        self.assertGreater(result["expected_values"]["away"], 0)
        self.assertEqual(result["recommendation"], "观望")

    def test_draw_pick_requires_exceptional_edge(self) -> None:
        result = self._evaluate(
            probabilities={"home": 0.32, "draw": 0.36, "away": 0.32},
            market_odds={"home": 2.70, "draw": 4.60, "away": 2.80},
            market_probs={"home": 0.39, "draw": 0.22, "away": 0.39},
            legacy_probabilities={"home": 0.33, "draw": 0.35, "away": 0.32},
            model_margin=0.04,
            selected_outcome="draw",
        )

        self.assertEqual(result["recommended_outcome"], "draw")
        self.assertGreater(result["expected_values"]["draw"], 0)
        self.assertEqual(result["recommendation"], "观望")

    def test_high_odds_home_pick_is_guarded(self) -> None:
        result = self._evaluate(
            probabilities={"home": 0.50, "draw": 0.25, "away": 0.25},
            market_odds={"home": 2.35, "draw": 3.40, "away": 3.10},
            market_probs={"home": 0.45, "draw": 0.29, "away": 0.26},
            legacy_probabilities={"home": 0.49, "draw": 0.26, "away": 0.25},
            model_margin=0.25,
            selected_outcome="home",
        )

        self.assertEqual(result["recommended_outcome"], "home")
        self.assertGreater(result["expected_values"]["home"], 0)
        self.assertEqual(result["recommendation"], "观望")

    def test_low_odds_market_favorite_can_be_light_even_without_positive_ev(self) -> None:
        result = self._evaluate(
            probabilities={"home": 0.66, "draw": 0.21, "away": 0.13},
            market_odds={"home": 1.45, "draw": 4.50, "away": 7.20},
            market_probs={"home": 0.67, "draw": 0.21, "away": 0.12},
            legacy_probabilities={"home": 0.65, "draw": 0.22, "away": 0.13},
            model_margin=0.45,
            selected_outcome="home",
        )

        self.assertEqual(result["recommended_outcome"], "home")
        self.assertLess(result["expected_values"]["home"], 0)
        self.assertEqual(result["recommendation"], "轻仓")
        self.assertGreater(result["stake_pct"], 0)


if __name__ == "__main__":
    unittest.main()
