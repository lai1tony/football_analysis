import unittest
from unittest.mock import patch

import app as web_app_module
import prediction_engine


class MatchSelectionRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        web_app_module.app.config["TESTING"] = True
        self.client = web_app_module.app.test_client()

    def test_collect_all_passes_selected_match_ids(self):
        calls = []

        def fake_collect_all(issue, *, progress_callback=None, return_details=False, match_ids=None):
            calls.append((issue, return_details, match_ids))
            return {"status_message": "ok", "status_level": "success"}

        with patch.object(web_app_module, "_ensure_db_initialized"), patch.object(
            web_app_module,
            "collect_all_matches",
            side_effect=fake_collect_all,
        ):
            response = self.client.post(
                "/collect-all",
                data={
                    "issue": "26001",
                    "match_id": "M1",
                    "selected_match_ids": ["M2", "M3"],
                },
            )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(calls, [("26001", True, ["M2", "M3"])])

    def test_predict_all_passes_selected_match_ids(self):
        calls = []

        def fake_predict(issue, *, progress_callback=None, match_ids=None):
            calls.append((issue, match_ids))
            return {"status_message": "ok", "status_level": "success"}

        with patch.object(web_app_module, "_ensure_db_initialized"), patch.object(
            web_app_module,
            "_predict_issue_and_refresh_top_picks",
            side_effect=fake_predict,
        ):
            response = self.client.post(
                "/predict-all",
                data={
                    "issue": "26001",
                    "match_id": "M1",
                    "selected_match_ids": ["M2", "M3"],
                },
            )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(calls, [("26001", ["M2", "M3"])])

    def test_settle_all_uses_full_scope_without_selection(self):
        calls = []

        def fake_settle(issue, *, progress_callback=None, match_ids=None):
            calls.append((issue, match_ids))
            return {"status_message": "ok", "status_level": "success"}

        with patch.object(web_app_module, "_ensure_db_initialized"), patch.object(
            web_app_module,
            "settle_issue_results",
            side_effect=fake_settle,
        ):
            response = self.client.post(
                "/settle-all",
                data={"issue": "26001", "match_id": "M1"},
            )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(calls, [("26001", None)])

    def test_target_batch_strategy_updates_only_selected_matches(self):
        rows = [
            {"run_id": 1, "match_id": "M1"},
            {"run_id": 2, "match_id": "M2"},
        ]
        updated_run_ids = []

        def fake_action_rows(mutable_rows):
            return [
                {
                    "match_id": row["match_id"],
                    "outcome": "home",
                    "tier": "standard",
                    "stake_pct": 1.0,
                }
                for row in mutable_rows
            ]

        def fake_update(run_id, fields):
            updated_run_ids.append(run_id)

        with patch.object(
            prediction_engine,
            "_latest_issue_prediction_rows",
            return_value=rows,
        ), patch.object(
            prediction_engine,
            "get_feedback_log",
            return_value=None,
        ), patch.object(
            prediction_engine,
            "_balanced_coverage_draw_rescue_action_rows",
            side_effect=fake_action_rows,
        ), patch.object(
            prediction_engine,
            "update_prediction_run_fields",
            side_effect=fake_update,
        ):
            result = prediction_engine.apply_target_batch_strategy_to_issue(
                "26001",
                match_ids=["M2"],
            )

        self.assertEqual(updated_run_ids, [2])
        self.assertEqual(result["updated_count"], 1)


if __name__ == "__main__":
    unittest.main()
