from __future__ import annotations

import json
import tempfile
import subprocess
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import app as web_app_module
import collection_repository
import historical_import
import learning_engine
import prediction_engine
import replay_backfill
import replay_profile43_issues
import progress_service


class TemporaryDatabaseTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.primary_db = Path(self.tempdir.name) / "football_data.db"
        self.recovery_db = Path(self.tempdir.name) / "football_data_live.db"
        self.patches = [
            patch.object(collection_repository, "PRIMARY_DB_PATH", self.primary_db),
            patch.object(collection_repository, "RECOVERY_DB_PATH", self.recovery_db),
            patch.object(collection_repository, "DB_PATH", self.primary_db),
            patch.object(
                collection_repository,
                "READONLY_PRIMARY_URI",
                f"file:{self.primary_db.as_posix()}?mode=ro&immutable=1",
            ),
        ]
        for item in self.patches:
            item.start()
        collection_repository._ACTIVE_RW_PATH = None
        web_app_module._DB_INITIALIZED = False
        progress_service._TASKS.clear()
        collection_repository.init_db()

    def tearDown(self) -> None:
        collection_repository._ACTIVE_RW_PATH = None
        web_app_module._DB_INITIALIZED = False
        progress_service._TASKS.clear()
        for item in reversed(self.patches):
            item.stop()
        self.tempdir.cleanup()

    def insert_match(
        self,
        match_id: str,
        *,
        issue: str,
        match_no: str = "1",
        match_time: str = "2026-04-29 20:00:00",
        home_team: str = "Home",
        away_team: str = "Away",
        league: str = "Test League",
    ) -> None:
        collection_repository.upsert_matches(
            [
                {
                    "match_id": match_id,
                    "issue": issue,
                    "league": league,
                    "match_no": match_no,
                    "match_time": match_time,
                    "home_team": home_team,
                    "away_team": away_team,
                    "source_match_url": "",
                    "shuju_url": "",
                    "ouzhi_url": "",
                    "touzhu_url": "",
                    "list_odds_win": "",
                    "list_odds_draw": "",
                    "list_odds_loss": "",
                    "list_heat_win": "",
                    "list_heat_draw": "",
                    "list_heat_loss": "",
                    "sync_time": "",
                }
            ]
        )

    def insert_run(
        self,
        match_id: str,
        *,
        issue: str,
        created_at: str,
        recommendation: str = "轻仓",
        recommended_outcome: str = "home",
        suggested_stake_pct: float = 10.0,
        algo_recommendation: str | None = None,
        algo_recommended_outcome: str | None = None,
        algo_suggested_stake_pct: float | None = None,
        market_odds_home: float = 2.5,
        market_odds_draw: float = 3.2,
        market_odds_away: float = 3.8,
        market_home_prob: float = 0.44,
        market_draw_prob: float = 0.30,
        market_away_prob: float = 0.26,
        legacy_home_prob: float = 0.46,
        legacy_draw_prob: float = 0.28,
        legacy_away_prob: float = 0.26,
        final_home_prob: float = 0.50,
        final_draw_prob: float = 0.27,
        final_away_prob: float = 0.23,
        ev_home: float = 0.08,
        ev_draw: float = -0.03,
        ev_away: float = -0.08,
        handicap_recommendation: str = "轻仓",
        handicap_recommended_side: str = "home",
        handicap_line: float = -0.25,
        handicap_initial_line: float = -0.25,
        handicap_home_odds: float = 1.85,
        handicap_away_odds: float = 1.95,
        handicap_home_cover_prob: float = 0.62,
        handicap_away_cover_prob: float = 0.38,
        handicap_expected_value: float = 0.147,
        handicap_confidence: float = 0.65,
        llm_review_status: str = "completed",
        llm_review_decision: str = "keep",
    ) -> int:
        with closing(collection_repository.get_connection()) as conn:
            cursor = conn.execute(
                """
                INSERT INTO prediction_runs (
                    match_id, issue, created_at,
                    feature_snapshot_id,
                    legacy_home_prob, legacy_draw_prob, legacy_away_prob,
                    final_home_prob, final_draw_prob, final_away_prob,
                    market_home_prob, market_draw_prob, market_away_prob,
                    market_odds_home, market_odds_draw, market_odds_away,
                    ev_home, ev_draw, ev_away,
                    quality_score, model_agreement, confidence_score,
                    handicap_recommendation, handicap_recommended_side,
                    handicap_line, handicap_initial_line,
                    handicap_home_odds, handicap_away_odds,
                    handicap_home_cover_prob, handicap_away_cover_prob,
                    handicap_expected_value, handicap_confidence,
                    recommendation, recommended_outcome, suggested_stake_pct,
                    algo_recommendation, algo_recommended_outcome, algo_suggested_stake_pct,
                    llm_review_enabled, llm_review_status, llm_review_decision,
                    final_report
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    match_id,
                    issue,
                    created_at,
                    0,
                    legacy_home_prob,
                    legacy_draw_prob,
                    legacy_away_prob,
                    final_home_prob,
                    final_draw_prob,
                    final_away_prob,
                    market_home_prob,
                    market_draw_prob,
                    market_away_prob,
                    market_odds_home,
                    market_odds_draw,
                    market_odds_away,
                    ev_home,
                    ev_draw,
                    ev_away,
                    0.82,
                    0.74,
                    0.71,
                    handicap_recommendation,
                    handicap_recommended_side,
                    handicap_line,
                    handicap_initial_line,
                    handicap_home_odds,
                    handicap_away_odds,
                    handicap_home_cover_prob,
                    handicap_away_cover_prob,
                    handicap_expected_value,
                    handicap_confidence,
                    recommendation,
                    recommended_outcome,
                    suggested_stake_pct,
                    algo_recommendation if algo_recommendation is not None else recommendation,
                    algo_recommended_outcome if algo_recommended_outcome is not None else recommended_outcome,
                    algo_suggested_stake_pct if algo_suggested_stake_pct is not None else suggested_stake_pct,
                    1,
                    llm_review_status,
                    llm_review_decision,
                    "report",
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def save_learning_profile(self, **overrides: object) -> int:
        payload = {
            "status": "ready_candidate",
            "created_at": "2026-04-29 12:00:00",
            "updated_at": "2026-04-29 12:00:00",
            "activated_at": "",
            "archived_at": "",
            "retention_issue_count": 90,
            "window_type": "rolling_issues",
            "window_value": 90,
            "total_samples": 80,
            "training_samples": 64,
            "validation_samples": 16,
            "training_action_samples": 48,
            "validation_action_samples": 12,
            "calibrator_status": "ready",
            "threshold_status": "ready",
            "calibrator_params": "{}",
            "threshold_params": "{}",
            "train_metrics": "{}",
            "validation_metrics": "{}",
            "sample_summary": "{}",
            "notes": "",
        }
        payload.update(overrides)
        return collection_repository.save_learning_profile(payload)


class LearningEngineTests(TemporaryDatabaseTestCase):
    def test_target_strategy_can_filter_by_predicted_score_outcome(self) -> None:
        rows = [
            {
                "actual_result": "draw",
                "predicted_score": "1-1",
                "predicted_score_confidence": 0.66,
                "recommended_outcome": "draw",
                "quality_score": 0.82,
                "model_agreement": 0.74,
                "confidence_score": 0.88,
                "final_home_prob": 0.34,
                "final_draw_prob": 0.32,
                "final_away_prob": 0.34,
                "market_home_prob": 0.36,
                "market_draw_prob": 0.31,
                "market_away_prob": 0.33,
                "market_odds_home": 2.70,
                "market_odds_draw": 3.20,
                "market_odds_away": 2.80,
                "ev_home": -0.08,
                "ev_draw": 0.02,
                "ev_away": -0.05,
            },
            {
                "actual_result": "home",
                "predicted_score": "2-1",
                "predicted_score_confidence": 0.70,
                "recommended_outcome": "draw",
                "quality_score": 0.82,
                "model_agreement": 0.74,
                "confidence_score": 0.88,
                "final_home_prob": 0.34,
                "final_draw_prob": 0.32,
                "final_away_prob": 0.34,
                "market_home_prob": 0.36,
                "market_draw_prob": 0.31,
                "market_away_prob": 0.33,
                "market_odds_home": 2.70,
                "market_odds_draw": 3.20,
                "market_odds_away": 2.80,
                "ev_home": -0.08,
                "ev_draw": 0.02,
                "ev_away": -0.05,
            },
            {
                "actual_result": "draw",
                "predicted_score": "0-0",
                "predicted_score_confidence": 0.40,
                "recommended_outcome": "draw",
                "quality_score": 0.82,
                "model_agreement": 0.74,
                "confidence_score": 0.88,
                "final_home_prob": 0.34,
                "final_draw_prob": 0.32,
                "final_away_prob": 0.34,
                "market_home_prob": 0.36,
                "market_draw_prob": 0.31,
                "market_away_prob": 0.33,
                "market_odds_home": 2.70,
                "market_odds_draw": 3.20,
                "market_odds_away": 2.80,
                "ev_home": -0.08,
                "ev_draw": 0.02,
                "ev_away": -0.05,
            },
        ]
        rule = {
            "direction_source": "current",
            "stake_pct": 1.0,
            "outcomes": ("draw",),
            "score_outcomes": ("draw",),
            "score_confidence_min": 0.60,
            "odds_max": 10.0,
            "prob_min": 0.0,
            "market_prob_min": 0.0,
            "quality_min": 0.0,
            "agreement_min": 0.0,
            "confidence_min": 0.0,
            "ev_min": -1.0,
            "prob_margin_min": -1.0,
            "ev_margin_min": -2.0,
        }

        metrics = learning_engine.evaluate_target_strategy_rule(rows, rule)

        self.assertEqual(metrics["sample_count"], 3)
        self.assertEqual(metrics["action_count"], 1)
        self.assertEqual(metrics["hit_count"], 1)
        self.assertEqual(metrics["buckets"]["轻仓:draw"]["action_count"], 1)

    def test_sorted_learning_rows_filters_to_recent_issues_and_keeps_time_order(self) -> None:
        rows = [
            {"issue": "26067", "created_at": "2026-04-27 20:00:00", "prediction_run_id": 3},
            {"issue": "26069", "created_at": "2026-04-29 21:00:00", "prediction_run_id": 2},
            {"issue": "26068", "created_at": "2026-04-28 20:00:00", "prediction_run_id": 1},
        ]
        with patch("learning_engine.list_backtest_rows", return_value=rows):
            ordered_rows = learning_engine._sorted_learning_rows(window_issue_count=2)

        self.assertEqual([row["issue"] for row in ordered_rows], ["26068", "26069"])

    def test_apply_probability_calibration_returns_valid_three_way_probs(self) -> None:
        calibrated = learning_engine.apply_probability_calibration(
            {"home": 0.52, "draw": 0.26, "away": 0.22},
            {
                "temperature": 0.85,
                "biases": {"home": 0.10, "draw": -0.04, "away": -0.06},
            },
        )

        self.assertAlmostEqual(sum(calibrated.values()), 1.0, places=6)
        self.assertTrue(all(0.0 < value < 1.0 for value in calibrated.values()))

    def test_train_learning_profile_marks_insufficient_samples_on_small_live_set(self) -> None:
        self.insert_match("M1", issue="26069", match_no="1")
        self.insert_match("M2", issue="26069", match_no="2")
        run_id_1 = self.insert_run("M1", issue="26069", created_at="2026-04-29 18:00:00")
        run_id_2 = self.insert_run("M2", issue="26069", created_at="2026-04-29 18:30:00")
        prediction_engine.record_feedback(run_id_1, "M1", "home", actual_score="2-1", result_status="settled")
        prediction_engine.record_feedback(run_id_2, "M2", "away", actual_score="0-1", result_status="settled")

        result = learning_engine.train_learning_profile(window_issue_count=4)

        self.assertEqual(result["status"], "insufficient_samples")
        profile = collection_repository.get_learning_profile(result["learning_profile_id"])
        self.assertIsNotNone(profile)
        self.assertEqual(str(profile["status"]), "insufficient_samples")
        self.assertEqual(int(profile["retention_issue_count"]), 4)
        self.assertEqual(int(profile["window_value"]), 4)

    def test_target_strategy_marks_unreachable_when_validation_cannot_hit_target(self) -> None:
        train_rows = [
            {
                "actual_result": "home",
                "recommended_outcome": "home",
                "quality_score": 0.90,
                "model_agreement": 0.80,
                "confidence_score": 0.75,
                "final_home_prob": 0.66,
                "final_draw_prob": 0.20,
                "final_away_prob": 0.14,
                "market_home_prob": 0.62,
                "market_draw_prob": 0.24,
                "market_away_prob": 0.14,
                "market_odds_home": 1.35,
                "market_odds_draw": 4.2,
                "market_odds_away": 7.0,
            }
            for _ in range(40)
        ]
        validation_rows = [dict(item, actual_result="draw") for item in train_rows[:10]]

        result = learning_engine.fit_target_strategy(train_rows, validation_rows, None)

        self.assertEqual(result["status"], "target_unreachable")
        self.assertEqual(result["params"], {})
        self.assertLess(result["validation_metrics"]["hit_rate"], learning_engine.DEFAULT_TARGET_HIT_RATE)
        self.assertIn("frontier", result)
        self.assertIn("best_hit", result["frontier"])
        self.assertIn("best_action", result["frontier"])
        self.assertGreaterEqual(result["frontier"]["gaps"]["hit_rate_gap"], 0.0)

    def test_target_strategy_ready_requires_hit_and_action_share(self) -> None:
        base_row = {
            "actual_result": "home",
            "recommended_outcome": "home",
            "quality_score": 0.90,
            "model_agreement": 0.80,
            "confidence_score": 0.75,
            "final_home_prob": 0.66,
            "final_draw_prob": 0.20,
            "final_away_prob": 0.14,
            "market_home_prob": 0.62,
            "market_draw_prob": 0.24,
            "market_away_prob": 0.14,
            "market_odds_home": 1.35,
            "market_odds_draw": 4.2,
            "market_odds_away": 7.0,
        }
        train_rows = [dict(base_row) for _ in range(40)]
        validation_rows = [dict(base_row) for _ in range(10)]

        result = learning_engine.fit_target_strategy(train_rows, validation_rows, None)

        self.assertEqual(result["status"], "ready")
        self.assertGreaterEqual(result["validation_metrics"]["hit_rate"], learning_engine.DEFAULT_TARGET_HIT_RATE)
        self.assertGreater(result["validation_metrics"]["action_share"], 0.50)
        self.assertLess(result["validation_metrics"]["watch_share"], 0.50)
        self.assertTrue(result["params"])

    def test_handicap_target_strategy_ready_requires_hit_and_action_share(self) -> None:
        base_row = {
            "handicap_actual_result": "home",
            "handicap_recommended_side": "home",
            "handicap_recommendation": "轻仓",
            "handicap_line": -0.25,
            "handicap_initial_line": -0.25,
            "handicap_home_odds": 1.85,
            "handicap_away_odds": 1.95,
            "handicap_home_cover_prob": 0.62,
            "handicap_away_cover_prob": 0.38,
            "handicap_expected_value": 0.147,
            "handicap_confidence": 0.65,
            "quality_score": 0.82,
        }
        train_rows = [dict(base_row) for _ in range(40)]
        validation_rows = [dict(base_row) for _ in range(10)]

        result = learning_engine.fit_handicap_target_strategy(train_rows, validation_rows)

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["params"]["strategy_kind"], "handicap")
        self.assertGreaterEqual(
            result["validation_metrics"]["hit_rate"],
            learning_engine.DEFAULT_HANDICAP_TARGET_HIT_RATE,
        )
        self.assertGreaterEqual(
            result["validation_metrics"]["action_share"],
            learning_engine.DEFAULT_HANDICAP_MIN_ACTION_SHARE,
        )

    def test_handicap_bucket_strategy_trains_buckets_then_validates(self) -> None:
        base_row = {
            "handicap_recommended_side": "home",
            "handicap_recommendation": "轻仓",
            "handicap_line": -0.25,
            "handicap_home_odds": 1.85,
            "handicap_away_odds": 1.95,
            "handicap_home_cover_prob": 0.62,
            "handicap_away_cover_prob": 0.38,
            "handicap_confidence": 0.65,
            "quality_score": 0.82,
        }
        train_rows = [dict(base_row, handicap_actual_result="home") for _ in range(8)]
        train_rows.extend(dict(base_row, handicap_actual_result="away") for _ in range(2))
        validation_rows = [dict(base_row, handicap_actual_result="home") for _ in range(8)]
        validation_rows.extend(dict(base_row, handicap_actual_result="away") for _ in range(2))

        result = learning_engine.fit_handicap_bucket_strategy(train_rows, validation_rows)

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["params"]["strategy_kind"], "handicap_bucket_table")
        self.assertEqual(result["train_metrics"]["sample_count"], 10)
        self.assertEqual(result["validation_metrics"]["sample_count"], 10)
        self.assertGreaterEqual(result["validation_metrics"]["hit_rate"], 0.70)
        self.assertGreaterEqual(result["validation_metrics"]["action_share"], 0.60)

    def test_handicap_bucket_strategy_does_not_learn_validation_only_bucket(self) -> None:
        train_row = {
            "handicap_recommended_side": "home",
            "handicap_recommendation": "轻仓",
            "handicap_line": -0.25,
            "handicap_home_odds": 1.85,
            "handicap_away_odds": 1.95,
            "handicap_home_cover_prob": 0.62,
            "handicap_away_cover_prob": 0.38,
            "handicap_confidence": 0.65,
            "quality_score": 0.82,
            "handicap_actual_result": "home",
        }
        validation_row = dict(train_row, handicap_line=-1.25, handicap_actual_result="home")
        train_rows = [dict(train_row) for _ in range(10)]
        validation_rows = [dict(validation_row) for _ in range(10)]

        result = learning_engine.fit_handicap_bucket_strategy(train_rows, validation_rows)

        self.assertEqual(result["status"], "target_unreachable")
        self.assertEqual(result["validation_metrics"]["sample_count"], 10)
        self.assertEqual(result["validation_metrics"]["action_count"], 0)

    def test_backtest_summary_evaluates_active_handicap_bucket_strategy(self) -> None:
        features = ["line50", "coverdiff10", "evdiff10", "awayodds20"]
        base_row = {
            "handicap_recommended_side": "home",
            "handicap_recommendation": "轻仓",
            "handicap_line": -0.25,
            "handicap_initial_line": -0.25,
            "handicap_home_odds": 1.85,
            "handicap_away_odds": 1.95,
            "handicap_home_cover_prob": 0.62,
            "handicap_away_cover_prob": 0.38,
            "handicap_confidence": 0.65,
            "quality_score": 0.82,
        }
        key = learning_engine._handicap_bucket_key(base_row, features)
        strategy = {
            "strategy_kind": "handicap_bucket_table",
            "action": "轻仓",
            "stake_pct": 1.0,
            "features": features,
            "buckets": {key: {"side": "home", "sample_count": 3, "hit_rate": 0.75}},
        }
        self.save_learning_profile(
            status="active",
            activated_at="2026-04-29 13:00:00",
            threshold_params=json.dumps({"target_strategy": strategy}, ensure_ascii=False),
            validation_metrics=json.dumps(
                {
                    "target_metrics": {
                        "target_hit_rate": 0.70,
                        "min_action_share": 0.60,
                        "strategy_kind": "handicap_bucket_table",
                    },
                    "target_strategy": {
                        "status": "ready",
                        "params": strategy,
                        "validation": {},
                    },
                },
                ensure_ascii=False,
            ),
        )

        summary = prediction_engine._target_strategy_backtest_summary(
            [
                dict(base_row, handicap_actual_result="home"),
                dict(base_row, handicap_actual_result="away"),
            ]
        )

        self.assertEqual(summary["params"]["strategy_kind"], "handicap_bucket_table")
        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["action_count"], 2)
        self.assertEqual(summary["hit_rate"], 0.5)
        self.assertEqual(summary["target_hit_rate"], 0.70)
        self.assertEqual(summary["min_action_share"], 0.60)

    def test_train_learning_profile_stores_handicap_target_strategy(self) -> None:
        def handicap_row(issue: str, index: int) -> dict[str, object]:
            return {
                "issue": issue,
                "created_at": f"2026-04-29 18:{index:02d}:00",
                "actual_result": "home",
                "recommended_outcome": "home",
                "recommendation": "轻仓",
                "suggested_stake_pct": 1.0,
                "market_odds_home": 1.85,
                "market_odds_draw": 3.4,
                "market_odds_away": 4.2,
                "market_home_prob": 0.54,
                "market_draw_prob": 0.27,
                "market_away_prob": 0.19,
                "legacy_home_prob": 0.52,
                "legacy_draw_prob": 0.28,
                "legacy_away_prob": 0.20,
                "final_home_prob": 0.58,
                "final_draw_prob": 0.25,
                "final_away_prob": 0.17,
                "ev_home": 0.073,
                "ev_draw": -0.15,
                "ev_away": -0.29,
                "quality_score": 0.82,
                "model_agreement": 0.74,
                "confidence_score": 0.71,
                "handicap_actual_result": "home",
                "handicap_recommended_side": "home",
                "handicap_recommendation": "轻仓",
                "handicap_line": -0.25,
                "handicap_initial_line": -0.25,
                "handicap_home_odds": 1.85,
                "handicap_away_odds": 1.95,
                "handicap_home_cover_prob": 0.62,
                "handicap_away_cover_prob": 0.38,
                "handicap_expected_value": 0.147,
                "handicap_confidence": 0.65,
            }

        rows = [handicap_row("26001", index) for index in range(40)]
        rows.extend(handicap_row("26002", index) for index in range(10))
        saved_profiles: list[dict[str, object]] = []
        threshold_metrics = {
            "baseline": {"final": {"action_count": 0}},
        }

        def fake_save_profile(profile: dict[str, object]) -> int:
            saved_profiles.append(profile)
            return 77

        with (
            patch("learning_engine._sorted_learning_rows", return_value=rows),
            patch(
                "learning_engine._fit_calibrator",
                return_value={"status": "insufficient_samples", "params": {}, "train_metrics": {}},
            ),
            patch(
                "learning_engine._validate_calibrator",
                return_value={
                    "status": "insufficient_samples",
                    "reason": "test calibration skipped",
                    "baseline": {},
                    "candidate": {},
                },
            ),
            patch(
                "learning_engine._fit_thresholds",
                return_value={
                    "status": "insufficient_samples",
                    "reason": "test thresholds skipped",
                    "params": {},
                    "train_metrics": threshold_metrics,
                    "validation_metrics": threshold_metrics,
                },
            ),
            patch("learning_engine.save_learning_profile", side_effect=fake_save_profile),
        ):
            result = learning_engine.train_learning_profile(window_issue_count=2)

        self.assertEqual(result["status"], "ready_candidate")
        self.assertEqual(result["strategy_status"], "ready")
        self.assertEqual(result["strategy_params"]["strategy_kind"], "handicap")
        self.assertEqual(saved_profiles[0]["status"], "ready_candidate")
        threshold_params = json.loads(str(saved_profiles[0]["threshold_params"]))
        validation_metrics = json.loads(str(saved_profiles[0]["validation_metrics"]))
        self.assertEqual(threshold_params["target_strategy"]["strategy_kind"], "handicap")
        self.assertEqual(validation_metrics["target_metrics"]["strategy_kind"], "handicap")
        self.assertGreaterEqual(
            validation_metrics["target_strategy"]["validation"]["hit_rate"],
            learning_engine.DEFAULT_HANDICAP_TARGET_HIT_RATE,
        )
        self.assertGreaterEqual(
            validation_metrics["target_strategy"]["validation"]["action_share"],
            learning_engine.DEFAULT_HANDICAP_MIN_ACTION_SHARE,
        )

    def test_target_strategy_walk_forward_uses_only_previous_issues(self) -> None:
        rows = []
        for issue_no in range(1, 5):
            issue = f"2600{issue_no}"
            rows.extend(
                {
                    "issue": issue,
                    "actual_result": "home",
                    "recommended_outcome": "home",
                    "market_odds_home": 1.8,
                    "market_odds_draw": 3.2,
                    "market_odds_away": 4.5,
                }
                for _ in range(20)
            )
        calls = []

        def fake_fit(train_rows, validation_rows, calibrator_params, *, target_hit_rate, min_action_share):
            calls.append(
                (
                    sorted({row["issue"] for row in train_rows}),
                    sorted({row["issue"] for row in validation_rows}),
                )
            )
            return {
                "status": "ready",
                "reason": "",
                "params": {"direction_source": "current"},
                "target_metrics": {
                    "target_hit_rate": target_hit_rate,
                    "min_action_share": min_action_share,
                    "max_watch_share": 1.0 - min_action_share,
                },
                "train_metrics": {
                    "sample_count": len(train_rows),
                    "action_count": len(train_rows),
                    "hit_count": len(train_rows),
                    "hit_rate": 1.0,
                    "action_share": 1.0,
                    "watch_share": 0.0,
                    "total_roi": 1.0,
                    "avg_stake_pct": 1.0,
                    "buckets": {},
                },
                "validation_metrics": {
                    "sample_count": len(validation_rows),
                    "action_count": len(validation_rows),
                    "hit_count": len(validation_rows),
                    "hit_rate": 1.0,
                    "action_share": 1.0,
                    "watch_share": 0.0,
                    "total_roi": 1.0,
                    "avg_stake_pct": 1.0,
                    "buckets": {},
                },
                "best_candidate": {},
            }

        with patch("learning_engine.fit_target_strategy", side_effect=fake_fit):
            result = learning_engine.fit_target_strategy_walk_forward(rows, None)

        self.assertEqual(result["status"], "ready")
        self.assertEqual(
            calls,
            [
                (["26001", "26002"], ["26003"]),
                (["26001", "26002", "26003"], ["26004"]),
            ],
        )
        self.assertEqual(result["validation_metrics"]["sample_count"], 40)
        self.assertEqual(len(result["skipped_folds"]), 1)

    def test_target_strategy_rule_can_filter_stored_review_and_action_fields(self) -> None:
        rows = [
            {
                "actual_result": "home",
                "recommended_outcome": "home",
                "algo_recommended_outcome": "home",
                "algo_recommendation": "轻仓",
                "llm_review_decision": "keep",
                "arbiter_review_decision": "keep",
                "effective_action_source": "expert_llm",
                "quality_score": 0.90,
                "model_agreement": 0.80,
                "confidence_score": 0.75,
                "final_home_prob": 0.66,
                "final_draw_prob": 0.20,
                "final_away_prob": 0.14,
                "market_home_prob": 0.62,
                "market_draw_prob": 0.24,
                "market_away_prob": 0.14,
                "market_odds_home": 1.35,
                "market_odds_draw": 4.2,
                "market_odds_away": 7.0,
            },
            {
                "actual_result": "away",
                "recommended_outcome": "home",
                "algo_recommended_outcome": "home",
                "algo_recommendation": "观望",
                "llm_review_decision": "downgrade",
                "arbiter_review_decision": "downgrade",
                "effective_action_source": "baseline",
                "quality_score": 0.90,
                "model_agreement": 0.80,
                "confidence_score": 0.75,
                "final_home_prob": 0.66,
                "final_draw_prob": 0.20,
                "final_away_prob": 0.14,
                "market_home_prob": 0.62,
                "market_draw_prob": 0.24,
                "market_away_prob": 0.14,
                "market_odds_home": 1.35,
                "market_odds_draw": 4.2,
                "market_odds_away": 7.0,
            },
        ]
        rule = {
            "direction_source": "market",
            "odds_max": 10.0,
            "prob_min": 0.0,
            "market_prob_min": 0.0,
            "quality_min": 0.0,
            "agreement_min": 0.0,
            "confidence_min": 0.0,
            "ev_min": -1.0,
            "prob_margin_min": -1.0,
            "ev_margin_min": -2.0,
            "llm_decisions": ("keep",),
            "arbiter_decisions": ("keep",),
            "effective_sources": ("expert_llm",),
            "algo_actions": ("轻仓",),
        }

        metrics = learning_engine.evaluate_target_strategy_rule(rows, rule)

        self.assertEqual(metrics["action_count"], 1)
        self.assertEqual(metrics["hit_count"], 1)

    def test_target_strategy_rule_can_filter_structural_snapshot_fields(self) -> None:
        rows = [
            {
                "actual_result": "home",
                "recommended_outcome": "home",
                "quality_score": 0.90,
                "model_agreement": 0.80,
                "confidence_score": 0.75,
                "final_home_prob": 0.66,
                "final_draw_prob": 0.20,
                "final_away_prob": 0.14,
                "market_home_prob": 0.62,
                "market_draw_prob": 0.24,
                "market_away_prob": 0.14,
                "market_odds_home": 1.35,
                "market_odds_draw": 4.2,
                "market_odds_away": 7.0,
                "home_rating": 1550,
                "away_rating": 1480,
                "recent_home_ppg": 2.0,
                "recent_away_ppg": 1.2,
                "home_split_ppg": 2.1,
                "away_split_ppg": 1.0,
            },
            {
                "actual_result": "away",
                "recommended_outcome": "home",
                "quality_score": 0.90,
                "model_agreement": 0.80,
                "confidence_score": 0.75,
                "final_home_prob": 0.66,
                "final_draw_prob": 0.20,
                "final_away_prob": 0.14,
                "market_home_prob": 0.62,
                "market_draw_prob": 0.24,
                "market_away_prob": 0.14,
                "market_odds_home": 1.35,
                "market_odds_draw": 4.2,
                "market_odds_away": 7.0,
                "home_rating": 1490,
                "away_rating": 1530,
                "recent_home_ppg": 1.0,
                "recent_away_ppg": 1.8,
                "home_split_ppg": 1.0,
                "away_split_ppg": 2.0,
            },
        ]
        rule = {
            "direction_source": "current",
            "odds_max": 10.0,
            "prob_min": 0.0,
            "market_prob_min": 0.0,
            "quality_min": 0.0,
            "agreement_min": 0.0,
            "confidence_min": 0.0,
            "ev_min": -1.0,
            "prob_margin_min": -1.0,
            "ev_margin_min": -2.0,
            "rating_gap_min": 0.0,
            "ppg_gap_min": 0.0,
        }

        metrics = learning_engine.evaluate_target_strategy_rule(rows, rule)

        self.assertEqual(metrics["action_count"], 1)
        self.assertEqual(metrics["hit_count"], 1)

    def test_target_strategy_or_rule_combines_complementary_children(self) -> None:
        rows = [
            {
                "actual_result": "home",
                "recommended_outcome": "home",
                "algo_recommended_outcome": "home",
                "quality_score": 0.90,
                "model_agreement": 0.80,
                "confidence_score": 0.75,
                "final_home_prob": 0.66,
                "final_draw_prob": 0.20,
                "final_away_prob": 0.14,
                "market_home_prob": 0.62,
                "market_draw_prob": 0.24,
                "market_away_prob": 0.14,
                "market_odds_home": 1.35,
                "market_odds_draw": 4.2,
                "market_odds_away": 7.0,
            },
            {
                "actual_result": "away",
                "recommended_outcome": "home",
                "algo_recommended_outcome": "away",
                "quality_score": 0.90,
                "model_agreement": 0.80,
                "confidence_score": 0.75,
                "final_home_prob": 0.46,
                "final_draw_prob": 0.20,
                "final_away_prob": 0.34,
                "market_home_prob": 0.40,
                "market_draw_prob": 0.24,
                "market_away_prob": 0.36,
                "market_odds_home": 2.4,
                "market_odds_draw": 4.2,
                "market_odds_away": 1.85,
            },
        ]
        rule = {
            "action": "杞讳粨",
            "stake_pct": 1.0,
            "any_rules": [
                {
                    "direction_source": "market",
                    "odds_max": 1.5,
                    "prob_min": 0.0,
                    "market_prob_min": 0.0,
                    "quality_min": 0.0,
                    "agreement_min": 0.0,
                    "confidence_min": 0.0,
                    "ev_min": -1.0,
                    "prob_margin_min": -1.0,
                    "ev_margin_min": -2.0,
                    "outcomes": ("home",),
                },
                {
                    "direction_source": "algo",
                    "odds_max": 2.0,
                    "prob_min": 0.0,
                    "market_prob_min": 0.0,
                    "quality_min": 0.0,
                    "agreement_min": 0.0,
                    "confidence_min": 0.0,
                    "ev_min": -1.0,
                    "prob_margin_min": -1.0,
                    "ev_margin_min": -2.0,
                    "outcomes": ("away",),
                },
            ],
        }

        metrics = learning_engine.evaluate_target_strategy_rule(rows, rule)

        self.assertEqual(metrics["action_count"], 2)
        self.assertEqual(metrics["hit_count"], 2)

    def test_learning_overview_defaults_to_ninety_issue_window(self) -> None:
        with patch("learning_engine.list_backtest_rows", return_value=[]) as mock_backtest_rows:
            overview = learning_engine.get_learning_overview()

        self.assertEqual(overview["retention_issue_count"], 90)
        self.assertEqual(overview["min_retention_issue_count"], 1)
        self.assertEqual(overview["max_retention_issue_count"], 90)
        self.assertEqual(overview["settled_issue_count"], 0)
        self.assertEqual(overview["required_issue_count"], 90)
        mock_backtest_rows.assert_called_once_with(limit=None)

    def test_learning_overview_uses_latest_profile_window_when_not_explicit(self) -> None:
        self.save_learning_profile(
            status="insufficient_samples",
            retention_issue_count=90,
            window_value=90,
            created_at="2026-04-29 13:00:00",
            updated_at="2026-04-29 13:00:00",
        )
        with patch("learning_engine.list_backtest_rows", return_value=[]) as mock_backtest_rows:
            overview = learning_engine.get_learning_overview()

        self.assertEqual(overview["retention_issue_count"], 90)
        mock_backtest_rows.assert_called_once_with(limit=None)

    def test_learning_overview_shows_active_handicap_strategy_replay(self) -> None:
        features = ["line50", "coverdiff10", "evdiff10", "awayodds20"]
        base_row = {
            "issue": "26001",
            "created_at": "2026-04-29 18:00:00",
            "prediction_run_id": 1,
            "handicap_actual_result": "home",
            "handicap_recommended_side": "home",
            "handicap_recommendation": "轻仓",
            "handicap_line": -0.25,
            "handicap_initial_line": -0.25,
            "handicap_home_odds": 1.85,
            "handicap_away_odds": 1.95,
            "handicap_home_cover_prob": 0.62,
            "handicap_away_cover_prob": 0.38,
            "handicap_confidence": 0.65,
            "quality_score": 0.82,
        }
        key = learning_engine._handicap_bucket_key(base_row, features)
        strategy = {
            "strategy_kind": "handicap_bucket_table",
            "action": "轻仓",
            "stake_pct": 1.0,
            "features": features,
            "buckets": {key: {"side": "home", "sample_count": 3, "hit_rate": 0.75}},
        }
        self.save_learning_profile(
            status="active",
            activated_at="2026-04-29 13:00:00",
            retention_issue_count=1,
            window_value=1,
            threshold_status="ready",
            threshold_params=json.dumps({"target_strategy": strategy}, ensure_ascii=False),
            validation_metrics=json.dumps(
                {
                    "target_metrics": {
                        "target_hit_rate": 0.70,
                        "min_action_share": 0.60,
                        "strategy_kind": "handicap_bucket_table",
                    },
                    "target_strategy": {
                        "status": "ready",
                        "params": strategy,
                        "validation": {},
                    },
                },
                ensure_ascii=False,
            ),
        )
        rows = [
            dict(base_row, handicap_actual_result="home", prediction_run_id=1),
            dict(base_row, issue="26002", handicap_actual_result="away", prediction_run_id=2),
        ]

        with patch("learning_engine.list_backtest_rows", return_value=rows):
            overview = learning_engine.get_learning_overview()

        replay = overview["active_strategy_replay"]
        self.assertEqual(replay["strategy_kind"], "handicap_bucket_table")
        self.assertEqual(replay["sample_count"], 2)
        self.assertEqual(replay["action_count"], 2)
        self.assertEqual(replay["hit_rate"], 0.5)
        self.assertEqual(replay["action_share"], 1.0)
        self.assertEqual(replay["target_hit_rate"], 0.70)
        self.assertEqual(replay["min_action_share"], 0.60)

    def test_activate_learning_profile_archives_previous_active_profile(self) -> None:
        active_id = self.save_learning_profile(status="active", activated_at="2026-04-29 09:00:00")
        candidate_id = self.save_learning_profile(
            status="ready_candidate",
            created_at="2026-04-29 13:00:00",
            updated_at="2026-04-29 13:00:00",
        )

        result = learning_engine.activate_learning_profile(candidate_id)

        self.assertEqual(result["status_level"], "success")
        archived = collection_repository.get_learning_profile(active_id)
        activated = collection_repository.get_learning_profile(candidate_id)
        self.assertEqual(str(archived["status"]), "archived")
        self.assertEqual(str(activated["status"]), "active")
        self.assertTrue(str(activated["activated_at"]))

    def test_deactivate_learning_profile_archives_current_active_profile(self) -> None:
        active_id = self.save_learning_profile(status="active", activated_at="2026-04-29 09:00:00")

        result = learning_engine.deactivate_learning_profile()

        self.assertEqual(result["status_level"], "success")
        profile = collection_repository.get_learning_profile(active_id)
        self.assertEqual(str(profile["status"]), "archived")


class PredictionLearningIntegrationTests(unittest.TestCase):
    def test_predict_match_saves_learning_profile_and_calibrated_probs(self) -> None:
        match_row = {
            "match_id": "M1",
            "issue": "20260428",
            "home_team": "主队",
            "away_team": "客队",
            "league": "测试联赛",
            "match_time": "2026-04-28 20:00:00",
            "collected_at": "2026-04-28 10:00:00",
            "elo_home": "测试联赛 第3/16 20分",
            "elo_away": "测试联赛 第8/16 14分",
            "recent_form_home": "WDL",
            "recent_form_away": "LDW",
            "head_to_head_summary": "主队稍优",
            "injury_or_lineup_notes": "阵容完整",
            "motivation_or_schedule_notes": "测试联赛 主队 vs 客队",
            "home_away_form": "正常",
            "european_odds_movement_summary": "平稳",
            "asian_handicap_summary": "平稳",
            "betting_heat_summary": "主胜偏热",
        }
        snapshot = {"snapshot_id": 7, "home_rating": 1510, "away_rating": 1485}
        features = {
            "feature_snapshot_id": 7,
            "market_odds": {"home": 2.1, "draw": 3.2, "away": 3.8},
            "market_probs": {"home": 0.45, "draw": 0.29, "away": 0.26},
            "lineup": {
                "home_availability": 0.95,
                "away_availability": 0.91,
                "home_absent_count": 1,
                "home_doubtful_count": 0,
                "away_absent_count": 2,
                "away_doubtful_count": 1,
            },
            "schedule": {
                "home_rest_days": 6,
                "away_rest_days": 5,
                "home_load_14": 2,
                "away_load_14": 3,
            },
            "recent_home": {
                "points_per_game": 2.0,
                "goals_for_per_game": 1.8,
                "goals_against_per_game": 0.9,
            },
            "recent_away": {
                "points_per_game": 1.2,
                "goals_for_per_game": 1.1,
                "goals_against_per_game": 1.4,
            },
            "split": {"home_ppg": 2.1, "away_ppg": 1.0},
            "rating_gap": 25,
            "h2h_edge": 0.12,
        }
        quant = {
            "probabilities": {"home": 0.52, "draw": 0.27, "away": 0.21},
            "top_scores": [((1, 0), 0.18)],
            "lambda_home": 1.45,
            "lambda_away": 0.92,
            "over_25": 0.44,
            "under_25": 0.56,
        }
        ml = {"probabilities": {"home": 0.50, "draw": 0.28, "away": 0.22}}
        legacy = {"probabilities": {"home": 0.47, "draw": 0.29, "away": 0.24}}
        quality = {"score": 0.84, "weight_adjustment": 1.0, "problems": []}
        raw_blended = {"probabilities": {"home": 0.51, "draw": 0.28, "away": 0.21}, "agreement": 0.73, "margin": 0.23}
        algo_risk = {
            "fair_odds": {"home": 1.90, "draw": 3.60, "away": 4.80},
            "market_probs": {"home": 0.45, "draw": 0.29, "away": 0.26},
            "expected_values": {"home": 0.10, "draw": -0.02, "away": -0.12},
            "market_bias": {"home": 0.07, "draw": -0.01, "away": -0.06},
            "confidence": 0.70,
            "risk_level": "medium",
            "recommended_outcome": "home",
            "recommendation": "轻仓",
            "stake_pct": 1.35,
            "warnings": [],
        }
        review = {
            "enabled": 0,
            "status": "skipped",
            "decision": "",
            "target_action": "轻仓",
            "reason": "未启用复核",
            "risk_flags": [],
            "raw": "",
            "model_name": "",
        }
        final_risk = dict(algo_risk)
        final_risk.update({"resolution_reason": "保留算法初判。", "review_status_label": "跳过"})
        llm = {"provider": "openai-compatible", "model": "summary-model", "summary": "摘要正常"}
        active_profile = {
            "learning_profile_id": 12,
            "uses_calibrator": True,
            "uses_thresholds": True,
            "calibrator_params": {
                "temperature": 0.85,
                "biases": {"home": 0.12, "draw": -0.04, "away": -0.08},
            },
            "threshold_params": learning_engine.BASE_THRESHOLD_CONFIG,
        }

        with (
            patch("prediction_engine.init_db"),
            patch("prediction_engine.expire_pending_manual_reviews"),
            patch("prediction_engine.get_match_analysis", return_value=match_row),
            patch("prediction_engine._ensure_feature_snapshot", return_value=snapshot),
            patch("prediction_engine.build_match_features", return_value=features),
            patch("prediction_engine.run_quant_model", return_value=quant),
            patch("prediction_engine.run_ml_model", return_value=ml),
            patch("prediction_engine.run_legacy_market_model", return_value=legacy),
            patch("prediction_engine.run_data_quality", return_value=quality),
            patch("prediction_engine.blend_predictions", return_value=raw_blended),
            patch("prediction_engine.get_active_learning_profile_config", return_value=active_profile),
            patch("prediction_engine.run_risk_assessor", return_value=algo_risk),
            patch("prediction_engine.run_llm_recommendation_review", return_value=review),
            patch("prediction_engine.resolve_recommendation", return_value=final_risk),
            patch("prediction_engine.generate_llm_summary", return_value=llm),
            patch("prediction_engine.build_presenter_report", return_value="report"),
            patch("prediction_engine.save_prediction_run", return_value=88) as mock_save,
            patch("prediction_engine.supersede_pending_manual_reviews"),
        ):
            prediction_engine.predict_match("M1")

        saved_payload = mock_save.call_args.args[0]
        expected_calibrated = learning_engine.apply_probability_calibration(
            raw_blended["probabilities"],
            active_profile["calibrator_params"],
        )
        self.assertEqual(saved_payload["learning_profile_id"], 12)
        self.assertAlmostEqual(saved_payload["final_home_prob"], raw_blended["probabilities"]["home"], places=6)
        self.assertAlmostEqual(saved_payload["calibrated_home_prob"], expected_calibrated["home"], places=6)
        self.assertAlmostEqual(saved_payload["calibrated_draw_prob"], expected_calibrated["draw"], places=6)
        self.assertAlmostEqual(saved_payload["calibrated_away_prob"], expected_calibrated["away"], places=6)

    def test_handicap_learning_strategy_filters_handicap_only(self) -> None:
        handicap_risk = {
            "recommendation": "轻仓",
            "recommended_side": "home",
            "home_cover_prob": 0.62,
            "away_cover_prob": 0.38,
            "home_odds": 1.85,
            "away_odds": 1.95,
            "line": -0.25,
            "initial_line": -0.25,
            "confidence": 0.65,
            "quality_score": 0.82,
            "reason": "base",
        }
        strategy = {
            "strategy_kind": "handicap",
            "action": "轻仓",
            "ev_min": 0.0,
            "confidence_min": 0.60,
            "cover_prob_min": 0.60,
            "cover_margin_min": 0.0,
            "quality_min": 0.80,
            "odds_max": 2.2,
            "sides": ("home",),
            "base_actions": ("轻仓", "主推"),
        }

        passed = prediction_engine._apply_handicap_learning_strategy(
            handicap_risk,
            {"target_strategy": strategy},
        )
        failed = prediction_engine._apply_handicap_learning_strategy(
            handicap_risk,
            {"target_strategy": {**strategy, "quality_min": 0.90}},
        )

        self.assertEqual(passed["recommendation"], "轻仓")
        self.assertEqual(passed["recommended_side"], "home")
        self.assertIn("学习让球策略命中", passed["reason"])
        self.assertEqual(failed["recommendation"], "观望")
        self.assertEqual(failed["recommended_side"], "")
        self.assertIn("学习让球策略未命中", failed["reason"])

    def test_handicap_bucket_strategy_applies_learned_side(self) -> None:
        handicap_risk = {
            "recommendation": "轻仓",
            "recommended_side": "home",
            "home_cover_prob": 0.62,
            "away_cover_prob": 0.38,
            "home_odds": 1.85,
            "away_odds": 1.95,
            "line": -0.25,
            "initial_line": -0.25,
            "confidence": 0.65,
            "quality_score": 0.82,
            "reason": "base",
        }
        features = ["line50", "coverdiff10", "evdiff10", "awayodds20"]
        key = prediction_engine._handicap_bucket_key(handicap_risk, features)
        strategy = {
            "strategy_kind": "handicap_bucket_table",
            "action": "轻仓",
            "features": features,
            "buckets": {key: {"side": "away", "sample_count": 3, "hit_rate": 0.75}},
        }

        result = prediction_engine._apply_handicap_learning_strategy(
            handicap_risk,
            {"target_strategy": strategy},
        )
        missed = prediction_engine._apply_handicap_learning_strategy(
            {**handicap_risk, "line": 1.25},
            {"target_strategy": strategy},
        )

        self.assertEqual(result["recommendation"], "轻仓")
        self.assertEqual(result["recommended_side"], "away")
        self.assertIn("学习让球分桶策略命中", result["reason"])
        self.assertEqual(missed["recommendation"], "观望")
        self.assertEqual(missed["recommended_side"], "")

    def test_handicap_target_strategy_does_not_override_outcome_policy(self) -> None:
        features = {
            "market_odds": {"home": 2.10, "draw": 3.20, "away": 3.80},
            "market_probs": {"home": 0.45, "draw": 0.29, "away": 0.26},
        }
        legacy = {"probabilities": {"home": 0.47, "draw": 0.29, "away": 0.24}}
        blended = {"probabilities": {"home": 0.54, "draw": 0.26, "away": 0.20}, "agreement": 0.74, "margin": 0.28}
        quality = {"score": 0.84, "problems": []}
        handicap_strategy = {
            "strategy_kind": "handicap",
            "action": "轻仓",
            "quality_min": 0.95,
            "sides": ("away",),
        }

        baseline = prediction_engine.run_risk_assessor(
            {},
            features,
            legacy,
            blended,
            quality,
        )
        with_handicap_strategy = prediction_engine.run_risk_assessor(
            {},
            features,
            legacy,
            blended,
            quality,
            threshold_config={"target_strategy": handicap_strategy},
        )

        self.assertEqual(with_handicap_strategy["recommendation"], baseline["recommendation"])
        self.assertEqual(with_handicap_strategy["recommended_outcome"], baseline["recommended_outcome"])
        self.assertEqual(with_handicap_strategy["stake_pct"], baseline["stake_pct"])
        self.assertEqual(with_handicap_strategy["target_strategy"], {})


class Profile43ReplayTests(TemporaryDatabaseTestCase):
    def _activate_profile43(self) -> None:
        features = ["line50", "coverdiff10", "evdiff10", "awayodds20"]
        base_row = {
            "handicap_line": -0.25,
            "handicap_home_odds": 1.85,
            "handicap_away_odds": 1.95,
            "handicap_home_cover_prob": 0.62,
            "handicap_away_cover_prob": 0.38,
            "handicap_recommended_side": "home",
            "quality_score": 0.82,
        }
        key = learning_engine._handicap_bucket_key(base_row, features)
        strategy = {
            "strategy_kind": "handicap_bucket_table",
            "action": "轻仓",
            "stake_pct": 1.0,
            "features": features,
            "bucket_count": 1,
            "buckets": {key: {"side": "home", "sample_count": 10, "hit_rate": 0.8}},
        }
        self.save_learning_profile(
            learning_profile_id=43,
            status="active",
            activated_at="2026-04-29 13:00:00",
            threshold_params=json.dumps({"target_strategy": strategy}, ensure_ascii=False),
            validation_metrics=json.dumps(
                {
                    "target_metrics": {
                        "target_hit_rate": 0.70,
                        "min_action_share": 0.60,
                        "strategy_kind": "handicap_bucket_table",
                    },
                    "target_strategy": {
                        "status": "ready",
                        "params": strategy,
                        "validation": {"sample_count": 10, "action_count": 10, "hit_rate": 0.8},
                    },
                },
                ensure_ascii=False,
            ),
        )

    def test_profile43_replay_runs_each_issue_in_order_and_writes_audit(self) -> None:
        self._activate_profile43()
        self.insert_match("P4326001", issue="26001")
        self.insert_match("P4326002", issue="26002")
        calls: list[tuple[str, str]] = []
        run_ids: dict[str, int] = {}

        def fake_sync(issue: str):
            calls.append(("sync", issue))
            self.insert_match(f"P43{issue}", issue=issue)
            return {"issue": issue, "matches": [{"match_id": f"P43{issue}"}]}

        def fake_predict(issue: str, ensure_collected: bool = False):
            calls.append(("predict", issue))
            match_id = f"P43{issue}"
            run_ids[issue] = self.insert_run(
                match_id,
                issue=issue,
                created_at="2026-04-29 12:00:00",
                handicap_recommendation="轻仓",
                handicap_recommended_side="home",
                handicap_line=-0.25,
            )
            return {"predicted_count": 1}

        def fake_settle(issue: str):
            calls.append(("settle", issue))
            match_id = f"P43{issue}"
            prediction_engine.record_feedback(
                run_ids[issue],
                match_id,
                "home",
                actual_score="1-0",
                result_status="settled",
            )
            return {"settled_count": 1}

        report_path = Path(self.tempdir.name) / "profile43_report.md"
        with (
            patch.object(replay_profile43_issues, "sync_issue_matches", side_effect=fake_sync),
            patch.object(replay_profile43_issues, "predict_issue", side_effect=fake_predict),
            patch.object(replay_profile43_issues, "settle_issue_results", side_effect=fake_settle),
        ):
            result = replay_profile43_issues.run_profile43_replay(
                backup_database=False,
                report_path=report_path,
            )

        self.assertEqual(
            calls,
            [
                ("sync", "26001"),
                ("predict", "26001"),
                ("settle", "26001"),
                ("sync", "26002"),
                ("predict", "26002"),
                ("settle", "26002"),
            ],
        )
        self.assertEqual(result["totals"]["sample_count"], 2)
        self.assertEqual(result["totals"]["action_count"], 2)
        self.assertEqual(result["totals"]["hit_count"], 2)
        self.assertTrue(report_path.exists())


class ReplayBackfillTests(TemporaryDatabaseTestCase):
    def _seed_feedback_issue(self, issue: str) -> None:
        match_id = f"EX{issue}"
        self.insert_match(match_id, issue=issue)
        run_id = self.insert_run(match_id, issue=issue, created_at="2026-04-28 12:00:00")
        prediction_engine.record_feedback(run_id, match_id, "home", actual_score="2-1", result_status="settled")

    def test_replay_backfill_generates_missing_older_issues_and_marks_feedback(self) -> None:
        self._seed_feedback_issue("26059")
        self._seed_feedback_issue("26060")
        run_ids: dict[str, int] = {}

        def fake_sync(issue: str, return_details: bool = False):
            match_id = f"RB{issue}"
            self.insert_match(match_id, issue=issue)
            matches = [{"match_id": match_id, "issue": issue}]
            return {"matches": matches, "issue": issue} if return_details else matches

        def fake_predict(issue: str, ensure_collected: bool = False):
            match_id = f"RB{issue}"
            run_ids[str(issue)] = self.insert_run(
                match_id,
                issue=str(issue),
                created_at="2026-04-28 12:00:00",
            )
            return {"predicted_count": 1}

        def fake_settle(issue: str):
            issue_text = str(issue)
            match_id = f"RB{issue_text}"
            prediction_engine.record_feedback(
                run_ids[issue_text],
                match_id,
                "home",
                actual_score="1-0",
                result_status="settled",
            )
            return {"settled_count": 1}

        with (
            patch.object(replay_backfill, "fetch_sfc_issue_sequence", return_value=[]),
            patch.object(replay_backfill, "sync_issue_matches", side_effect=fake_sync),
            patch.object(replay_backfill, "_collect_issue_matches_safely", return_value={"collected_count": 1}),
            patch.object(replay_backfill, "predict_issue", side_effect=fake_predict),
            patch.object(replay_backfill, "settle_issue_results", side_effect=fake_settle),
        ):
            result = replay_backfill.replay_backfill_learning_feedback(required_issue_count=4)

        self.assertEqual(result["planned_issues"], ["26057", "26058"])
        self.assertEqual(result["completed_issues"], ["26057", "26058"])
        self.assertEqual(result["feedback_marked"], 2)
        rows = {
            str(row["match_id"]): row
            for row in collection_repository.list_backtest_rows(limit=None)
        }
        self.assertEqual(rows["RB26057"]["roi_source"], replay_backfill.REPLAY_BACKFILL_SOURCE)
        self.assertIn(replay_backfill.REPLAY_BACKFILL_NOTE_PREFIX, rows["RB26057"]["notes"])
        self.assertEqual(rows["RB26058"]["roi_source"], replay_backfill.REPLAY_BACKFILL_SOURCE)

    def test_replay_backfill_plan_uses_real_issue_sequence_across_year_boundary(self) -> None:
        self._seed_feedback_issue("26001")
        issue_sequence = ["25194", "25195", "25196", "26001"]

        with patch.object(replay_backfill, "fetch_sfc_issue_sequence", return_value=issue_sequence):
            plan = replay_backfill.build_replay_backfill_issue_plan(
                required_issue_count=4,
            )

        self.assertEqual(plan, ["25194", "25195", "25196"])

    def test_replay_backfill_plan_does_not_invent_cross_year_issue_numbers(self) -> None:
        self._seed_feedback_issue("26001")

        with patch.object(replay_backfill, "fetch_sfc_issue_sequence", return_value=[]):
            plan = replay_backfill.build_replay_backfill_issue_plan(
                required_issue_count=4,
            )

        self.assertEqual(plan, [])

    def test_replay_backfill_skips_existing_feedback_only_when_collection_complete(self) -> None:
        self.insert_match("RB26058", issue="26058")
        run_id = self.insert_run("RB26058", issue="26058", created_at="2026-04-28 12:00:00")
        prediction_engine.record_feedback(run_id, "RB26058", "home", actual_score="2-1", result_status="settled")
        complete = {
            "match_id": "RB26058",
            "collected_at": "2026-04-28 12:00:00",
            "collection_status": "success",
            "media_source_links": "",
            "collected_sources": "",
            "collection_quality_summary": "",
            "remarks": "",
        }
        complete.update({field: "ok" for field in collection_repository.REQUIRED_ANALYSIS_FIELDS})
        collection_repository.save_analysis(complete)

        def fake_sync(issue: str, return_details: bool = False):
            return {"matches": [{"match_id": "RB26058", "issue": issue}], "issue": issue}

        with (
            patch.object(replay_backfill, "build_replay_backfill_issue_plan", return_value=["26058"]),
            patch.object(replay_backfill, "sync_issue_matches", side_effect=fake_sync),
            patch.object(replay_backfill, "_collect_issue_matches_safely") as mock_collect,
        ):
            result = replay_backfill.replay_backfill_learning_feedback(required_issue_count=3)

        self.assertEqual(result["skipped_issues"], [{"issue": "26058", "reason": "已有反馈闭环记录且采集完整"}])
        mock_collect.assert_not_called()

    def test_replay_backfill_recollects_existing_feedback_issue_with_collection_failure(self) -> None:
        self.insert_match("RB26058", issue="26058")
        run_id = self.insert_run("RB26058", issue="26058", created_at="2026-04-28 12:00:00")
        prediction_engine.record_feedback(run_id, "RB26058", "home", actual_score="2-1", result_status="settled")
        collection_repository.save_failed_analysis(
            "RB26058",
            "采集失败：缺少采集维度：维度二：近期动态/伤停/阵容",
            "2026-04-28 12:00:00",
        )

        def fake_sync(issue: str, return_details: bool = False):
            return {"matches": [{"match_id": "RB26058", "issue": issue}], "issue": issue}

        with (
            patch.object(replay_backfill, "build_replay_backfill_issue_plan", return_value=["26058"]),
            patch.object(replay_backfill, "sync_issue_matches", side_effect=fake_sync),
            patch.object(
                replay_backfill,
                "_collect_issue_matches_safely",
                return_value={"collected_count": 0, "failed_count": 1, "failed_matches": [{"issue": "26058", "match_label": "Home vs Away", "reason": "自动补采 2 次仍失败"}]},
            ) as mock_collect,
        ):
            result = replay_backfill.replay_backfill_learning_feedback(required_issue_count=3)

        mock_collect.assert_called_once_with(
            "26058",
            timeout_seconds=replay_backfill.MATCH_COLLECTION_TIMEOUT_SECONDS,
            progress_callback=None,
        )
        self.assertEqual(result["skipped_issues"], [])
        self.assertEqual(result["collection_failed_issues"][0]["issue"], "26058")
        self.assertIn("采集异常 1 期", result["status_message"])

    def test_collect_match_timeout_marks_failure_without_hanging_issue(self) -> None:
        self.insert_match("TMO1", issue="26057")

        with patch.object(
            replay_backfill.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(cmd="collect", timeout=1),
        ):
            result = replay_backfill._collect_issue_matches_safely(
                "26057",
                timeout_seconds=1,
            )

        self.assertEqual(result["total_matches"], 1)
        self.assertEqual(result["failed_count"], 1)
        row = collection_repository.get_match_analysis("TMO1")
        self.assertIn("历史回放补数超时", str(row["remarks"]))

    def test_collect_issue_resumes_after_existing_successful_rows(self) -> None:
        self.insert_match("OK1", issue="26057")
        self.insert_match("NEW1", issue="26057", match_no="2")
        collection_repository.save_analysis(
            {
                "match_id": "OK1",
                "collected_at": "2026-04-28 12:00:00",
                "elo_home": "主队强度正常",
                "elo_away": "客队强度正常",
                "recent_form_home": "近况正常",
                "recent_form_away": "近况正常",
                "home_away_form": "主客表现正常",
                "head_to_head_summary": "交锋正常",
                "injury_or_lineup_notes": "阵容正常",
                "motivation_or_schedule_notes": "赛程正常",
                "european_odds_movement_summary": "欧赔正常",
                "asian_handicap_summary": "亚盘正常",
                "betting_heat_summary": "热度正常",
                "media_source_links": "",
                "collected_sources": "",
                "collection_quality_summary": "",
                "remarks": "",
            }
        )
        calls: list[str] = []

        def fake_collect(match_id: str, *, timeout_seconds: int):
            calls.append(match_id)
            return {
                "match_id": match_id,
                "collected_at": "2026-04-28 12:01:00",
                "elo_home": "主队强度正常",
                "elo_away": "客队强度正常",
                "recent_form_home": "近况正常",
                "recent_form_away": "近况正常",
                "home_away_form": "主客表现正常",
                "head_to_head_summary": "交锋正常",
                "injury_or_lineup_notes": "阵容正常",
                "motivation_or_schedule_notes": "赛程正常",
                "european_odds_movement_summary": "欧赔正常",
                "asian_handicap_summary": "亚盘正常",
                "betting_heat_summary": "热度正常",
                "remarks": "",
            }

        with patch.object(replay_backfill, "_collect_match_with_timeout", side_effect=fake_collect):
            result = replay_backfill._collect_issue_matches_safely("26057")

        self.assertEqual(calls, ["NEW1"])
        self.assertEqual(result["collected_count"], 2)
        self.assertEqual(result["failed_count"], 0)

    def test_collect_issue_retries_existing_failed_rows(self) -> None:
        self.insert_match("MISS1", issue="26057")
        collection_repository.save_failed_analysis(
            "MISS1",
            "采集失败：缺少采集维度：维度二：近期动态/伤停/阵容",
            "2026-04-28 12:00:00",
        )
        calls: list[str] = []

        def fake_collect(match_id: str, *, timeout_seconds: int):
            calls.append(match_id)
            result = {
                "match_id": match_id,
                "collected_at": "2026-04-28 12:01:00",
                "collection_status": "success",
                "remarks": "",
            }
            result.update({field: "ok" for field in collection_repository.REQUIRED_ANALYSIS_FIELDS})
            return result

        with patch.object(replay_backfill, "_collect_match_with_timeout", side_effect=fake_collect):
            result = replay_backfill._collect_issue_matches_safely("26057")

        self.assertEqual(calls, ["MISS1"])
        self.assertEqual(result["collected_count"], 1)
        self.assertEqual(result["failed_count"], 0)

    def test_collect_issue_reports_failure_after_two_auto_retries(self) -> None:
        self.insert_match("MISS2", issue="26057")
        failure = {
            "match_id": "MISS2",
            "collected_at": "2026-04-28 12:01:00",
            "collection_status": "failed",
            "remarks": "采集失败：缺少采集维度：维度二：近期动态/伤停/阵容",
        }
        failure.update({field: "" for field in collection_repository.REQUIRED_ANALYSIS_FIELDS})

        with patch.object(replay_backfill, "_collect_match_with_timeout", return_value=failure) as mock_collect:
            result = replay_backfill._collect_issue_matches_safely("26057")

        self.assertEqual(mock_collect.call_count, 3)
        self.assertEqual(result["failed_count"], 1)
        self.assertEqual(result["failed_matches"][0]["auto_retry_count"], 2)


class HistoricalImportTests(TemporaryDatabaseTestCase):
    def _build_source_db(self) -> Path:
        source_path = Path(self.tempdir.name) / "source_history.db"
        with (
            patch.object(collection_repository, "PRIMARY_DB_PATH", source_path),
            patch.object(collection_repository, "RECOVERY_DB_PATH", source_path),
            patch.object(collection_repository, "DB_PATH", source_path),
            patch.object(collection_repository, "READONLY_PRIMARY_URI", f"file:{source_path.as_posix()}?mode=ro&immutable=1"),
        ):
            collection_repository._ACTIVE_RW_PATH = None
            collection_repository.init_db()
            collection_repository.upsert_matches(
                [
                    {
                        "match_id": "OLD1",
                        "issue": "26001",
                        "league": "Test League",
                        "match_no": "1",
                        "match_time": "2026-01-01 20:00:00",
                        "home_team": "Old Home",
                        "away_team": "Old Away",
                        "source_match_url": "",
                        "shuju_url": "",
                        "ouzhi_url": "",
                        "touzhu_url": "",
                        "yazhi_url": "",
                        "list_odds_win": "",
                        "list_odds_draw": "",
                        "list_odds_loss": "",
                        "list_heat_win": "",
                        "list_heat_draw": "",
                        "list_heat_loss": "",
                        "sync_time": "",
                    }
                ]
            )
            run_id = self.insert_run("OLD1", issue="26001", created_at="2026-01-01 10:00:00")
            prediction_engine.record_feedback(run_id, "OLD1", "home", actual_score="2-1", result_status="settled")
        collection_repository._ACTIVE_RW_PATH = None
        return source_path

    def test_import_historical_learning_feedback_appends_missing_rows_once(self) -> None:
        source_path = self._build_source_db()

        result = historical_import.import_historical_learning_feedback([source_path])
        second = historical_import.import_historical_learning_feedback([source_path])

        self.assertEqual(result["imported_rows"], 1)
        self.assertEqual(result["imported_issue_count"], 1)
        self.assertEqual(second["imported_rows"], 0)
        self.assertGreaterEqual(second["skipped_existing"], 1)
        rows = collection_repository.list_backtest_rows(limit=None)
        self.assertEqual(len(rows), 1)
        self.assertEqual(str(rows[0]["match_id"]), "OLD1")


class LearningWebTests(TemporaryDatabaseTestCase):
    def test_learning_train_route_returns_async_task_payload(self) -> None:
        client = web_app_module.app.test_client()
        with patch.object(
            web_app_module,
            "train_learning_profile",
            return_value={"task_message": "训练完成", "status_message": "训练完成", "status_level": "success"},
        ):
            response = client.post(
                "/learning/train",
                data={"match_id": "M1", "issue": "26069", "learning_window_issue_count": "6"},
                headers={"X-Requested-With": "XMLHttpRequest", "Accept": "application/json"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("task_id", payload)
        task = progress_service.get_task(payload["task_id"])
        self.assertIsNotNone(task)
        self.assertEqual(task["kind"], "learning-train")

    def test_learning_train_route_passes_custom_window(self) -> None:
        client = web_app_module.app.test_client()
        with patch.object(
            web_app_module,
            "train_learning_profile",
            return_value={"task_message": "训练完成", "status_message": "训练完成", "status_level": "success"},
        ) as mock_train:
            response = client.post(
                "/learning/train",
                data={"match_id": "M1", "issue": "26069", "learning_window_issue_count": "5"},
            )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(mock_train.call_args.kwargs["window_issue_count"], "5")

    def test_learning_import_history_route_returns_async_task_payload(self) -> None:
        client = web_app_module.app.test_client()
        with patch.object(
            web_app_module,
            "import_historical_learning_feedback",
            return_value={"task_message": "导入完成", "status_message": "导入完成", "status_level": "success"},
        ):
            response = client.post(
                "/learning/import-history",
                data={"match_id": "M1", "issue": "26069"},
                headers={"X-Requested-With": "XMLHttpRequest", "Accept": "application/json"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("task_id", payload)
        task = progress_service.get_task(payload["task_id"])
        self.assertIsNotNone(task)
        self.assertEqual(task["kind"], "learning-import-history")

    def test_learning_replay_backfill_route_returns_async_task_payload(self) -> None:
        client = web_app_module.app.test_client()
        with patch.object(
            web_app_module,
            "replay_backfill_learning_feedback",
            return_value={"task_message": "回放完成", "status_message": "回放完成", "status_level": "success"},
        ):
            response = client.post(
                "/learning/replay-backfill",
                data={"match_id": "M1", "issue": "26069"},
                headers={"X-Requested-With": "XMLHttpRequest", "Accept": "application/json"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("task_id", payload)
        task = progress_service.get_task(payload["task_id"])
        self.assertIsNotNone(task)
        self.assertEqual(task["kind"], "learning-replay-backfill")

    def test_index_renders_learning_section(self) -> None:
        self.insert_match("M1", issue="26069", home_team="日尔曼", away_team="拜仁")
        client = web_app_module.app.test_client()
        overview = {
            "retention_issue_count": 90,
            "min_retention_issue_count": 1,
            "max_retention_issue_count": 90,
            "settled_samples": 2,
            "settled_issue_count": 1,
            "required_issue_count": 90,
            "action_samples": 2,
            "active_profile": None,
            "latest_candidate": {
                "learning_profile_id": 3,
                "status_label": "样本不足",
                "ready_for_activation": False,
                "calibrator_status_label": "样本不足",
                "threshold_status_label": "样本不足",
                "training_samples": 1,
                "validation_samples": 1,
                "validation_metrics": {},
                "sample_summary": {"all_rows": {"issue_range": ["26069", "26069"]}},
                "notes": "当前仅有 2 条已结算样本",
            },
            "recent_profiles": [],
        }
        with patch.object(web_app_module, "get_learning_overview", return_value=overview):
            response = client.get("/?match_id=M1&issue=26069")

        body = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("学习闭环", body)
        self.assertIn("训练学习候选", body)
        self.assertIn("回放补齐历史期", body)
        self.assertIn("样本不足", body)


if __name__ == "__main__":
    unittest.main()
