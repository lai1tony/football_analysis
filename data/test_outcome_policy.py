from __future__ import annotations

import unittest

from outcome_policy import evaluate_outcome_policy


class OutcomePolicyTests(unittest.TestCase):
    def _evaluate(self, **overrides):
        payload = {
            "probabilities": {"home": 0.50, "draw": 0.26, "away": 0.24},
            "market_odds": {"home": 2.35, "draw": 3.40, "away": 3.90},
            "market_probs": {"home": 0.44, "draw": 0.29, "away": 0.27},
            "legacy_probabilities": {"home": 0.49, "draw": 0.27, "away": 0.24},
            "quality_score": 0.90,
            "model_agreement": 0.88,
        }
        payload.update(overrides)
        return evaluate_outcome_policy(**payload)

    def test_uses_market_direction_when_value_switch_is_weak(self) -> None:
        result = self._evaluate(
            probabilities={"home": 0.44, "draw": 0.29, "away": 0.27},
            market_odds={"home": 2.10, "draw": 3.40, "away": 4.20},
            market_probs={"home": 0.47, "draw": 0.29, "away": 0.24},
            legacy_probabilities={"home": 0.43, "draw": 0.30, "away": 0.27},
        )

        self.assertEqual(result["predicted_outcome"], "home")
        self.assertEqual(result["value_outcome"], "away")
        self.assertEqual(result["recommended_outcome"], "home")
        self.assertEqual(result["outcome_source"], "market_direction")

    def test_keeps_market_direction_even_when_value_direction_is_strong(self) -> None:
        result = self._evaluate(
            probabilities={"home": 0.42, "draw": 0.24, "away": 0.34},
            market_odds={"home": 2.15, "draw": 3.70, "away": 4.10},
            market_probs={"home": 0.47, "draw": 0.28, "away": 0.25},
            legacy_probabilities={"home": 0.41, "draw": 0.25, "away": 0.34},
            quality_score=0.92,
            model_agreement=0.90,
        )

        self.assertEqual(result["predicted_outcome"], "home")
        self.assertEqual(result["value_outcome"], "away")
        self.assertEqual(result["recommended_outcome"], "home")
        self.assertEqual(result["outcome_source"], "market_direction")

    def test_low_quality_warns_but_still_uses_market_direction(self) -> None:
        result = self._evaluate(
            probabilities={"home": 0.42, "draw": 0.24, "away": 0.34},
            market_odds={"home": 2.15, "draw": 3.70, "away": 4.10},
            market_probs={"home": 0.47, "draw": 0.28, "away": 0.25},
            legacy_probabilities={"home": 0.41, "draw": 0.25, "away": 0.34},
            quality_score=0.52,
            model_agreement=0.90,
        )

        self.assertEqual(result["recommended_outcome"], "home")
        self.assertEqual(result["outcome_source"], "market_direction")
        self.assertIn("data quality is below", " ".join(result["warnings"]))

    def test_draw_value_direction_does_not_override_market_direction(self) -> None:
        result = self._evaluate(
            probabilities={"home": 0.40, "draw": 0.31, "away": 0.29},
            market_odds={"home": 2.15, "draw": 3.55, "away": 3.10},
            market_probs={"home": 0.47, "draw": 0.27, "away": 0.26},
            legacy_probabilities={"home": 0.39, "draw": 0.31, "away": 0.30},
            quality_score=0.92,
            model_agreement=0.88,
        )

        self.assertEqual(result["predicted_outcome"], "home")
        self.assertEqual(result["value_outcome"], "draw")
        self.assertEqual(result["recommended_outcome"], "home")
        self.assertEqual(result["outcome_source"], "market_direction")


if __name__ == "__main__":
    unittest.main()
