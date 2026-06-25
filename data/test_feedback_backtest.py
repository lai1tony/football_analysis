from __future__ import annotations

import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import app as web_app_module
import collection_service
import collection_repository
import prediction_engine
import source_500_client


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
        collection_repository.init_db()

    def tearDown(self) -> None:
        collection_repository._ACTIVE_RW_PATH = None
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
        handicap_line: float = 0.0,
        handicap_home_odds: float = 2.0,
        handicap_away_odds: float = 2.0,
        handicap_expected_value: float = 0.05,
        handicap_confidence: float = 0.60,
        llm_review_status: str = "completed",
        llm_review_decision: str = "keep",
        effective_recommendation: str = "",
        effective_stake_pct: float = 0.0,
        effective_action_source: str = "",
        predicted_score: str = "",
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
                    confidence_score,
                    recommendation, recommended_outcome, suggested_stake_pct,
                    handicap_recommendation, handicap_recommended_side,
                    handicap_line, handicap_home_odds, handicap_away_odds,
                    handicap_expected_value, handicap_confidence,
                    algo_recommendation, algo_recommended_outcome, algo_suggested_stake_pct,
                    llm_review_enabled, llm_review_status, llm_review_decision,
                    effective_recommendation, effective_stake_pct, effective_action_source,
                    predicted_score,
                    final_report
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    0.72,
                    recommendation,
                    recommended_outcome,
                    suggested_stake_pct,
                    handicap_recommendation,
                    handicap_recommended_side,
                    handicap_line,
                    handicap_home_odds,
                    handicap_away_odds,
                    handicap_expected_value,
                    handicap_confidence,
                    algo_recommendation if algo_recommendation is not None else recommendation,
                    algo_recommended_outcome if algo_recommended_outcome is not None else recommended_outcome,
                    algo_suggested_stake_pct if algo_suggested_stake_pct is not None else suggested_stake_pct,
                    1,
                    llm_review_status,
                    llm_review_decision,
                    effective_recommendation,
                    effective_stake_pct,
                    effective_action_source,
                    predicted_score,
                    "report",
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def feedback_count(self) -> int:
        with closing(collection_repository.get_connection()) as conn:
            return int(conn.execute("SELECT COUNT(*) FROM feedback_logs").fetchone()[0])

    def save_snapshot(
        self,
        match_id: str,
        *,
        issue: str,
        recent_home_gf_pg: float = 1.6,
        recent_away_gf_pg: float = 1.1,
        h2h_edge: float = 0.2,
        market_home_prob: float = 0.56,
        market_draw_prob: float = 0.25,
        market_away_prob: float = 0.19,
    ) -> int:
        return collection_repository.save_feature_snapshot(
            {
                "match_id": match_id,
                "issue": issue,
                "snapshot_at": "2026-04-29 18:00:00",
                "home_rating": 1500.0,
                "away_rating": 1500.0,
                "recent_home_ppg": 1.8,
                "recent_away_ppg": 1.2,
                "recent_home_gf_pg": recent_home_gf_pg,
                "recent_away_gf_pg": recent_away_gf_pg,
                "recent_home_ga_pg": 1.0,
                "recent_away_ga_pg": 1.4,
                "home_split_ppg": 2.0,
                "away_split_ppg": 1.0,
                "home_absent_count": 0,
                "away_absent_count": 0,
                "home_doubtful_count": 0,
                "away_doubtful_count": 0,
                "home_absence_impact": 0.0,
                "away_absence_impact": 0.0,
                "lineup_home_availability": 0.92,
                "lineup_away_availability": 0.92,
                "rest_days_home": 0.0,
                "rest_days_away": 0.0,
                "schedule_load_home": 0,
                "schedule_load_away": 0,
                "h2h_edge": h2h_edge,
                "market_home_prob": market_home_prob,
                "market_draw_prob": market_draw_prob,
                "market_away_prob": market_away_prob,
                "feature_payload": "{}",
            }
        )

    def prediction_payload(
        self,
        match_id: str,
        *,
        issue: str,
        created_at: str,
        recommendation: str = "轻仓",
        recommended_outcome: str = "home",
    ) -> dict:
        return {
            "match_id": match_id,
            "issue": issue,
            "created_at": created_at,
            "feature_snapshot_id": 0,
            "quant_home_prob": 0.50,
            "quant_draw_prob": 0.27,
            "quant_away_prob": 0.23,
            "ml_home_prob": 0.49,
            "ml_draw_prob": 0.28,
            "ml_away_prob": 0.23,
            "legacy_home_prob": 0.46,
            "legacy_draw_prob": 0.28,
            "legacy_away_prob": 0.26,
            "final_home_prob": 0.50,
            "final_draw_prob": 0.27,
            "final_away_prob": 0.23,
            "fair_odds_home": 2.0,
            "fair_odds_draw": 3.7,
            "fair_odds_away": 4.3,
            "market_odds_home": 2.5,
            "market_odds_draw": 3.2,
            "market_odds_away": 3.8,
            "market_home_prob": 0.44,
            "market_draw_prob": 0.30,
            "market_away_prob": 0.26,
            "ev_home": 0.08,
            "ev_draw": -0.03,
            "ev_away": -0.08,
            "quality_score": 0.91,
            "model_agreement": 0.82,
            "confidence_score": 0.72,
            "risk_level": "medium",
            "recommendation": recommendation,
            "recommended_outcome": recommended_outcome,
            "suggested_stake_pct": 10.0,
            "algo_recommendation": recommendation,
            "algo_recommended_outcome": recommended_outcome,
            "algo_risk_level": "medium",
            "algo_suggested_stake_pct": 10.0,
            "llm_review_enabled": 1,
            "llm_review_status": "completed",
            "llm_review_decision": "keep",
            "llm_review_target_action": recommendation,
            "llm_review_reason": "",
            "llm_review_raw": "",
            "review_model_name": "",
            "final_resolution_reason": "",
            "arbiter_review_enabled": 0,
            "arbiter_review_status": "skipped",
            "arbiter_review_decision": "",
            "arbiter_review_target_action": "",
            "arbiter_review_reason": "",
            "arbiter_review_raw": "",
            "arbiter_review_model_name": "",
            "effective_recommendation": recommendation,
            "effective_stake_pct": 10.0,
            "effective_action_source": "algorithm",
            "manual_review_status": "",
            "manual_review_reason": "",
            "manual_review_requested_at": "",
            "manual_review_resolved_at": "",
            "manual_review_notes": "",
            "llm_provider": "",
            "llm_model": "",
            "llm_summary": "",
            "final_report": "report",
            "learning_profile_id": 0,
            "calibrated_home_prob": 0.50,
            "calibrated_draw_prob": 0.27,
            "calibrated_away_prob": 0.23,
        }


class ResultParsingTests(unittest.TestCase):
    def test_parse_issue_results_html_extracts_match_score_and_outcome(self) -> None:
        html = """
        <table class="bet-tb">
          <tr class="bet-tb-tr">
            <td class="td-data"><a href="https://odds.500.com/fenxi/shuju-123456.shtml">数据</a></td>
            <td class="td-team">
              <a class="team-l">主队</a>
              <i class="team-vs">2:1</i>
              <a class="team-r">客队</a>
            </td>
          </tr>
        </table>
        """

        rows = source_500_client.parse_issue_results_html(html, issue="20260429")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["match_id"], "123456")
        self.assertEqual(rows[0]["actual_score"], "2-1")
        self.assertEqual(rows[0]["actual_result"], "home")
        self.assertEqual(
            rows[0]["result_source_url"],
            "https://trade.500.com/rj/?expect=20260429",
        )

    def test_parse_issue_results_html_falls_back_to_shuju_header_score(self) -> None:
        html = """
        <table id="vsTable" class="bet-tb bet-tb-dg">
          <tr class="bet-tb-tr bet-tb-end" data-fixtureid="1407201">
            <td class="td-data"><a href="https://odds.500.com/fenxi/shuju-1407201.shtml">析</a></td>
            <td class="td-team">
              <a class="team-l">日尔曼</a>
              <i class="team-vs">VS</i>
              <a class="team-r">拜仁</a>
            </td>
          </tr>
        </table>
        """
        shuju_html = """
        <div class="odds_header">
          <div class="odds_hd_center">
            <p class="odds_hd_bf"><strong>5:4</strong></p>
          </div>
        </div>
        """

        with patch("source_500_client.fetch_html", return_value=shuju_html):
            rows = source_500_client.parse_issue_results_html(html, issue="26069")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["match_id"], "1407201")
        self.assertEqual(rows[0]["actual_score"], "5-4")
        self.assertEqual(rows[0]["actual_result"], "home")
        self.assertEqual(
            rows[0]["result_source_url"],
            "https://odds.500.com/fenxi/shuju-1407201.shtml",
        )

    def test_parse_issue_results_html_skips_unfinished_shuju_header(self) -> None:
        html = """
        <table id="vsTable" class="bet-tb bet-tb-dg">
          <tr class="bet-tb-tr bet-tb-end" data-fixtureid="1407200">
            <td class="td-data"><a href="https://odds.500.com/fenxi/shuju-1407200.shtml">析</a></td>
            <td class="td-team">
              <a class="team-l">马竞技</a>
              <i class="team-vs">VS</i>
              <a class="team-r">阿森纳</a>
            </td>
          </tr>
        </table>
        """
        shuju_html = """
        <div class="odds_header">
          <div class="odds_hd_center">
            <p class="odds_hd_bf"><strong>VS</strong></p>
          </div>
        </div>
        """

        with patch("source_500_client.fetch_html", return_value=shuju_html):
            rows = source_500_client.parse_issue_results_html(html, issue="26069")

        self.assertEqual(rows, [])

    def test_fetch_issue_results_uses_main_page_and_shuju_fallback(self) -> None:
        main_html = """
        <table id="vsTable" class="bet-tb bet-tb-dg">
          <tr class="bet-tb-tr bet-tb-end" data-fixtureid="1407201">
            <td class="td-data"><a href="https://odds.500.com/fenxi/shuju-1407201.shtml">析</a></td>
            <td class="td-team">
              <a class="team-l">日尔曼</a>
              <i class="team-vs">VS</i>
              <a class="team-r">拜仁</a>
            </td>
          </tr>
        </table>
        """
        shuju_html = """
        <div class="odds_header">
          <div class="odds_hd_center">
            <p class="odds_hd_bf"><strong>5:4</strong></p>
          </div>
        </div>
        """
        seen_urls: list[str] = []

        def _fake_fetch_html(url: str) -> str:
            seen_urls.append(url)
            if "rj/?expect=26069" in url:
                return main_html
            if "shuju-1407201.shtml" in url:
                return shuju_html
            raise AssertionError(f"unexpected url: {url}")

        with patch("source_500_client.fetch_html", side_effect=_fake_fetch_html):
            rows = source_500_client.fetch_issue_results("26069")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["actual_score"], "5-4")
        self.assertEqual(rows[0]["actual_result"], "home")
        self.assertTrue(any("rj/?expect=26069" in url for url in seen_urls))
        self.assertTrue(any("shuju-1407201.shtml" in url for url in seen_urls))

    def test_fetch_issue_matches_parses_requested_issue_list(self) -> None:
        rows_html = "\n".join(
            f"""
            <tr class="bet-tb-tr" data-bjpl="{2 + index}.10,3.20,4.30" data-pjgl="40,30,30">
              <td class="td-no">{index}</td>
              <td class="td-evt">Test League</td>
              <td class="td-endtime">2026-04-{index:02d} 20:00</td>
              <td class="td-team">
                <a class="team-l">Home {index}</a>
                <i class="team-vs">VS</i>
                <a class="team-r">Away {index}</a>
              </td>
              <td class="td-data">
                <a href="https://odds.500.com/fenxi/shuju-260690{index:02d}.shtml">析</a>
              </td>
            </tr>
            """
            for index in range(1, 15)
        )
        html = f"""
        <select><option value="26069" selected>26069</option></select>
        <table id="vsTable">{rows_html}</table>
        """
        seen_urls: list[str] = []

        def _fake_fetch_html(url: str) -> str:
            seen_urls.append(url)
            return html

        with patch("source_500_client.fetch_html", side_effect=_fake_fetch_html):
            rows = source_500_client.fetch_issue_matches("26069")

        self.assertEqual(len(rows), 14)
        self.assertEqual(rows[0]["issue"], "26069")
        self.assertEqual(rows[0]["match_no"], "1")
        self.assertEqual(rows[0]["home_team"], "Home 1")
        self.assertEqual(rows[0]["away_team"], "Away 1")
        self.assertEqual(rows[0]["source_match_url"], "https://trade.500.com/sfc/?expect=26069")
        self.assertTrue(rows[0]["shuju_url"].endswith("shuju-26069001.shtml"))
        self.assertEqual(seen_urls, ["https://trade.500.com/sfc/?expect=26069"])

    def test_fetch_issue_matches_rejects_invalid_issue_input(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "不能为空"):
            source_500_client.fetch_issue_matches("")
        with self.assertRaisesRegex(RuntimeError, "数字期号"):
            source_500_client.fetch_issue_matches("abc")

    def test_fetch_issue_matches_rejects_mismatched_returned_issue(self) -> None:
        html = """
        <select><option value="26070" selected>26070</option></select>
        <table id="vsTable"></table>
        """

        with patch("source_500_client.fetch_html", return_value=html):
            with self.assertRaisesRegex(RuntimeError, "不一致"):
                source_500_client.fetch_issue_matches("26069")

    def test_parse_live_selectable_matches_builds_collection_urls(self) -> None:
        html = """
        <table>
          <tr fid="1407734">
            <td><input type="checkbox" name="check_id[]" value="1407734"></td>
            <td class="ssbox_01"><a>中女超</a></td>
            <td>第7轮</td>
            <td>06-24&nbsp;17:00</td>
            <td>&nbsp;</td>
            <td><span class="gray">[07]</span><a>长春女足</a></td>
            <td><div class="pk"><a>1</a><a>半球</a><a>1</a></div></td>
            <td><a>山东女足</a><span class="gray">[11]</span></td>
            <td>1 - 1</td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td><a href="//odds.500.com/fenxi/shuju-1407734.shtml">析</a></td>
          </tr>
        </table>
        """

        rows = source_500_client.parse_live_selectable_matches_html(html, issue="26099")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["match_id"], "1407734")
        self.assertEqual(rows[0]["issue"], "26099")
        self.assertEqual(rows[0]["league"], "中女超")
        self.assertEqual(rows[0]["home_team"], "长春女足")
        self.assertEqual(rows[0]["away_team"], "山东女足")
        self.assertEqual(rows[0]["match_no"], "9001")
        self.assertTrue(rows[0]["match_time"].endswith("06-24 17:00"))
        self.assertEqual(rows[0]["shuju_url"], "https://odds.500.com/fenxi/shuju-1407734.shtml")
        self.assertEqual(rows[0]["ouzhi_url"], "https://odds.500.com/fenxi/ouzhi-1407734.shtml?ctype=2")


class FeedbackBacktestTests(TemporaryDatabaseTestCase):
    def test_add_selectable_matches_upserts_selected_candidates(self) -> None:
        candidates = [
            {
                "match_id": "1407734",
                "issue": "26099",
                "league": "中女超",
                "match_no": "9001",
                "match_time": "2026-06-24 17:00",
                "home_team": "长春女足",
                "away_team": "山东女足",
                "source_match_url": "https://live.500.com/2h1.php",
                "shuju_url": "https://odds.500.com/fenxi/shuju-1407734.shtml",
                "ouzhi_url": "https://odds.500.com/fenxi/ouzhi-1407734.shtml?ctype=2",
                "touzhu_url": "https://odds.500.com/fenxi/touzhu-1407734.shtml",
                "yazhi_url": "https://odds.500.com/fenxi/yazhi-1407734.shtml",
                "list_odds_win": "",
                "list_odds_draw": "",
                "list_odds_loss": "",
                "list_heat_win": "",
                "list_heat_draw": "",
                "list_heat_loss": "",
                "sync_time": "2026-06-24 17:10:00",
            }
        ]

        with patch("collection_service.fetch_live_selectable_matches", return_value=candidates):
            result = collection_service.add_selectable_matches(
                ["1407734"],
                issue="26099",
                return_details=True,
            )

        row = collection_repository.get_match("1407734")
        self.assertEqual(result["status_level"], "success")
        self.assertIsNotNone(row)
        self.assertEqual(row["issue"], "26099")
        self.assertEqual(row["home_team"], "长春女足")
        self.assertTrue(row["yazhi_url"].endswith("yazhi-1407734.shtml"))

    def test_remove_selectable_match_deletes_only_live_selected_match(self) -> None:
        collection_repository.upsert_matches(
            [
                {
                    "match_id": "1407734",
                    "issue": "26099",
                    "league": "中女超",
                    "match_no": "9001",
                    "match_time": "2026-06-24 17:00",
                    "home_team": "长春女足",
                    "away_team": "山东女足",
                    "source_match_url": "https://live.500.com/2h1.php",
                    "shuju_url": "https://odds.500.com/fenxi/shuju-1407734.shtml",
                    "ouzhi_url": "https://odds.500.com/fenxi/ouzhi-1407734.shtml?ctype=2",
                    "touzhu_url": "https://odds.500.com/fenxi/touzhu-1407734.shtml",
                    "yazhi_url": "https://odds.500.com/fenxi/yazhi-1407734.shtml",
                    "list_odds_win": "",
                    "list_odds_draw": "",
                    "list_odds_loss": "",
                    "list_heat_win": "",
                    "list_heat_draw": "",
                    "list_heat_loss": "",
                    "sync_time": "2026-06-24 17:10:00",
                }
            ]
        )
        self.insert_match("regular1", issue="26099", match_no="1")

        blocked = collection_service.remove_selectable_match(
            "regular1",
            issue="26099",
            return_details=True,
        )
        deleted = collection_service.remove_selectable_match(
            "1407734",
            issue="26099",
            return_details=True,
        )

        self.assertFalse(blocked["deleted"])
        self.assertTrue(deleted["deleted"])
        self.assertIsNotNone(collection_repository.get_match("regular1"))
        self.assertIsNone(collection_repository.get_match("1407734"))

    def test_save_prediction_run_replaces_existing_match_prediction(self) -> None:
        issue = "26088"
        self.insert_match("M1", issue=issue)
        first_run_id = collection_repository.save_prediction_run(
            self.prediction_payload(
                "M1",
                issue=issue,
                created_at="2026-04-01 10:00:00",
                recommendation="轻仓",
            )
        )
        feedback_id = collection_repository.save_feedback_log(
            {
                "prediction_run_id": first_run_id,
                "match_id": "M1",
                "actual_result": "home",
                "actual_score": "2-1",
                "settled_at": "2026-04-01 22:00:00",
                "hit_recommendation": 1,
                "roi_delta": 0.15,
                "roi_source": "auto",
                "notes": "",
            }
        )
        collection_repository.compute_issue_top_picks(issue)
        self.assertGreater(feedback_id, 0)
        self.assertTrue(collection_repository.get_issue_top_picks(issue))

        second_run_id = collection_repository.save_prediction_run(
            self.prediction_payload(
                "M1",
                issue=issue,
                created_at="2026-04-01 11:00:00",
                recommendation="观望",
            )
        )

        runs = collection_repository.list_prediction_runs("M1", limit=None)
        self.assertEqual([int(run["run_id"]) for run in runs], [second_run_id])
        self.assertIsNone(collection_repository.get_prediction_run(first_run_id))
        self.assertIsNone(collection_repository.get_feedback_log(first_run_id))
        self.assertEqual(collection_repository.get_issue_top_picks(issue), [])

    def test_post_kickoff_prediction_replaces_settled_pre_match_baseline(self) -> None:
        issue = "26088"
        self.insert_match("M1", issue=issue, match_time="2026-04-01 20:00:00")
        pre_match_run_id = collection_repository.save_prediction_run(
            self.prediction_payload(
                "M1",
                issue=issue,
                created_at="2026-04-01 19:00:00",
                recommendation="轻仓",
                recommended_outcome="home",
            )
        )
        prediction_engine.record_feedback(
            pre_match_run_id,
            "M1",
            "home",
            actual_score="2-1",
            result_status="settled",
            settled_at="2026-04-01 22:00:00",
        )

        post_match_run_id = collection_repository.save_prediction_run(
            self.prediction_payload(
                "M1",
                issue=issue,
                created_at="2026-04-01 23:00:00",
                recommendation="轻仓",
                recommended_outcome="away",
            )
        )

        canonical = prediction_engine.get_canonical_prediction_run("M1")
        runs = collection_repository.list_prediction_runs("M1", limit=None)

        self.assertEqual([int(run["run_id"]) for run in runs], [post_match_run_id])
        self.assertIsNone(collection_repository.get_prediction_run(pre_match_run_id))
        self.assertIsNone(collection_repository.get_feedback_log(pre_match_run_id))
        self.assertIsNotNone(collection_repository.get_feedback_log(post_match_run_id))
        self.assertIsNotNone(canonical)
        self.assertEqual(int(canonical["run_id"]), post_match_run_id)
        self.assertEqual(collection_repository.get_feedback_summary(issue)["miss_predictions"], 1)

    def test_post_kickoff_prediction_replaces_unsettled_pre_match_run(self) -> None:
        issue = "26088"
        self.insert_match("M2", issue=issue, match_time="2026-04-01 20:00:00")
        pre_match_run_id = collection_repository.save_prediction_run(
            self.prediction_payload(
                "M2",
                issue=issue,
                created_at="2026-04-01 19:00:00",
                recommendation="轻仓",
                recommended_outcome="home",
            )
        )
        post_match_run_id = collection_repository.save_prediction_run(
            self.prediction_payload(
                "M2",
                issue=issue,
                created_at="2026-04-01 23:00:00",
                recommendation="轻仓",
                recommended_outcome="away",
            )
        )

        result = prediction_engine.record_feedback(
            post_match_run_id,
            "M2",
            "away",
            result_status="settled",
        )

        self.assertIsNone(collection_repository.get_prediction_run(pre_match_run_id))
        self.assertEqual(result["prediction_run_id"], post_match_run_id)
        self.assertEqual(collection_repository.get_feedback_summary(issue)["hit_predictions"], 1)

    def test_init_db_removes_historical_duplicate_prediction_runs(self) -> None:
        issue = "26088"
        self.insert_match("M3", issue=issue, match_time="2026-04-01 20:00:00")
        with closing(collection_repository.get_connection()) as conn:
            conn.execute("DROP INDEX IF EXISTS idx_prediction_runs_match_id_unique")
            conn.commit()
        first_run_id = self.insert_run("M3", issue=issue, created_at="2026-04-01 19:00:00")
        second_run_id = self.insert_run("M3", issue=issue, created_at="2026-04-01 23:05:00")

        collection_repository.init_db()
        runs = collection_repository.list_prediction_runs("M3", limit=None)

        self.assertEqual([int(run["run_id"]) for run in runs], [second_run_id])
        self.assertIsNone(collection_repository.get_prediction_run(first_run_id))

    def test_llm_action_gate_vetoes_and_downgrades_actions(self) -> None:
        rejected, reject_reason = prediction_engine._llm_action_gate(
            "主推",
            review_status="completed",
            review_decision="reject",
            review_target_action="观望",
        )
        downgraded, downgrade_reason = prediction_engine._llm_action_gate(
            "主推",
            review_status="completed",
            review_decision="downgrade",
            review_target_action="轻仓",
        )
        arbiter_skipped, arbiter_reason = prediction_engine._llm_action_gate(
            "轻仓",
            review_status="completed",
            review_decision="keep",
            review_target_action="轻仓",
            arbiter_status="completed",
            arbiter_decision="skip",
            arbiter_target_action="观望",
        )

        self.assertEqual(rejected, "观望")
        self.assertIn("LLM", reject_reason)
        self.assertEqual(downgraded, "轻仓")
        self.assertIn("LLM", downgrade_reason)
        self.assertEqual(arbiter_skipped, "观望")
        self.assertIn("仲裁", arbiter_reason)

    def test_low_odds_favorite_guard_survives_llm_abstain_veto(self) -> None:
        algo_risk = {
            "recommendation": "轻仓",
            "recommended_outcome": "home",
            "confidence": 0.62,
            "stake_pct": 0.2,
            "risk_level": "medium",
            "expected_values": {"home": -0.2, "draw": 0.1, "away": 0.2},
            "market_bias": {"home": -0.1, "draw": 0.05, "away": 0.06},
            "market_probs": {"home": 0.7, "draw": 0.2, "away": 0.1},
            "market_odds": {"home": 1.45, "draw": 4.8, "away": 6.0},
            "probabilities": {"home": 0.56, "draw": 0.24, "away": 0.20},
            "fair_odds": {"home": 1.78, "draw": 4.17, "away": 5.0},
            "action_score": 0.52,
            "action_score_factors": {},
            "probability_margin": -0.1,
            "ev_margin": 0.02,
            "legacy_gap": 0.1,
            "kelly_fraction": 0.0,
            "low_odds_favorite_guard": True,
            "warnings": [],
        }
        review = {
            "status": "completed",
            "decision": "abstain",
            "target_action": "观望",
            "reason": "LLM says no value.",
            "evidence_grade": "unsafe",
            "stake_multiplier": 0.0,
            "confidence_delta": -0.08,
            "outcome_decision": "veto_to_watch",
            "target_outcome": "home",
            "outcome_reason": "LLM says outcome is risky.",
        }

        resolved = prediction_engine.resolve_recommendation(
            quality={"score": 1.0},
            algo_risk=algo_risk,
            review=review,
        )

        self.assertEqual(resolved["recommendation"], "轻仓")
        self.assertEqual(resolved["stake_pct"], 0.2)
        self.assertIn("low-odds favorite guard", resolved["resolution_reason"])

    def test_issue_top_picks_prioritize_handicap_recommendations(self) -> None:
        issue = "26099"
        self.insert_match("m1", issue=issue, home_team="Win Main", away_team="Away")
        self.insert_match("m2", issue=issue, home_team="Handicap Main", away_team="Away")
        self.insert_match("m3", issue=issue, home_team="Handicap Light", away_team="Away")
        self.insert_match("m4", issue=issue, home_team="Handicap Watch", away_team="Away")
        self.insert_run(
            "m1",
            issue=issue,
            created_at="2026-04-01 10:00:00",
            recommendation="主推",
            effective_recommendation="主推",
            suggested_stake_pct=12,
            effective_stake_pct=12,
            handicap_recommendation="观望",
            handicap_recommended_side="",
            handicap_expected_value=0.20,
            handicap_confidence=0.80,
        )
        self.insert_run(
            "m2",
            issue=issue,
            created_at="2026-04-01 10:01:00",
            recommendation="观望",
            effective_recommendation="观望",
            suggested_stake_pct=0,
            effective_stake_pct=0,
            handicap_recommendation="主推",
            handicap_recommended_side="away",
            handicap_expected_value=0.04,
            handicap_confidence=0.62,
        )
        self.insert_run(
            "m3",
            issue=issue,
            created_at="2026-04-01 10:02:00",
            recommendation="观望",
            effective_recommendation="观望",
            suggested_stake_pct=0,
            effective_stake_pct=0,
            handicap_recommendation="轻仓",
            handicap_recommended_side="home",
            handicap_expected_value=0.08,
            handicap_confidence=0.58,
        )
        self.insert_run(
            "m4",
            issue=issue,
            created_at="2026-04-01 10:03:00",
            recommendation="轻仓",
            effective_recommendation="轻仓",
            suggested_stake_pct=0.2,
            effective_stake_pct=0.2,
            handicap_recommendation="观望",
            handicap_recommended_side="",
            handicap_expected_value=0.12,
            handicap_confidence=0.70,
        )

        computed = collection_repository.compute_issue_top_picks(issue)
        saved = collection_repository.get_issue_top_picks(issue)

        self.assertEqual([row["match_id"] for row in computed[:2]], ["m2", "m3"])
        self.assertEqual([row["recommendation"] for row in saved[:2]], ["主推", "轻仓"])
        self.assertEqual(saved[0]["recommended_outcome"], "away")
        self.assertIn("让球盘主推客队", saved[0]["reason"])

    def test_issue_top_picks_include_settlement_status(self) -> None:
        issue = "26099"
        self.insert_match("hit", issue=issue, home_team="Hit Home", away_team="Away")
        self.insert_match("miss", issue=issue, home_team="Miss Home", away_team="Away")
        self.insert_match("pending", issue=issue, home_team="Pending Home", away_team="Away")
        hit_run = self.insert_run("hit", issue=issue, created_at="2026-04-01 10:00:00")
        miss_run = self.insert_run("miss", issue=issue, created_at="2026-04-01 10:01:00")
        self.insert_run("pending", issue=issue, created_at="2026-04-01 10:02:00")
        prediction_engine.record_feedback(
            hit_run,
            "hit",
            "home",
            actual_score="2-1",
            result_status="settled",
        )
        prediction_engine.record_feedback(
            miss_run,
            "miss",
            "away",
            actual_score="0-1",
            result_status="settled",
        )

        collection_repository.compute_issue_top_picks(issue)
        saved = {
            row["match_id"]: row
            for row in collection_repository.get_issue_top_picks(issue)
        }

        self.assertTrue(saved["hit"]["is_settled"])
        self.assertEqual(saved["hit"]["top_pick_result_status"], "hit")
        self.assertEqual(saved["hit"]["hit_recommendation"], 1)
        self.assertEqual(saved["hit"]["actual_score"], "2-1")
        self.assertTrue(saved["miss"]["is_settled"])
        self.assertEqual(saved["miss"]["top_pick_result_status"], "miss")
        self.assertEqual(saved["miss"]["hit_recommendation"], 0)
        self.assertFalse(saved["pending"]["is_settled"])
        self.assertEqual(saved["pending"]["top_pick_result_status"], "pending")

    def test_index_renders_top_pick_settlement_summary(self) -> None:
        issue = "26099"
        self.insert_match("hit", issue=issue, home_team="Hit Home", away_team="Away")
        self.insert_match("miss", issue=issue, home_team="Miss Home", away_team="Away")
        self.insert_match("pending", issue=issue, home_team="Pending Home", away_team="Away")
        hit_run = self.insert_run("hit", issue=issue, created_at="2026-04-01 10:00:00")
        miss_run = self.insert_run("miss", issue=issue, created_at="2026-04-01 10:01:00")
        self.insert_run("pending", issue=issue, created_at="2026-04-01 10:02:00")
        prediction_engine.record_feedback(
            hit_run,
            "hit",
            "home",
            actual_score="2-1",
            result_status="settled",
        )
        prediction_engine.record_feedback(
            miss_run,
            "miss",
            "away",
            actual_score="0-1",
            result_status="settled",
        )
        collection_repository.compute_issue_top_picks(issue)

        web_app_module.app.config["TESTING"] = True
        response = web_app_module.app.test_client().get(f"/?issue={issue}")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("top-picks-summary", html)
        self.assertIn("top-pick-result-hit", html)
        self.assertIn("top-pick-result-miss", html)
        self.assertIn("top-pick-result-pending", html)
        self.assertIn("2/3", html)
        self.assertIn("50.0%", html)

    def test_preview_production_single_pick_is_read_only_and_uses_target_batch_layer(self) -> None:
        issue = "26100"
        self.insert_match("p1", issue=issue, home_team="Home Strong", away_team="Away Weak")
        self.insert_match("p2", issue=issue, home_team="Watch Home", away_team="Watch Away")
        self.save_snapshot(
            "p1",
            issue=issue,
            market_home_prob=0.56,
            market_draw_prob=0.24,
            market_away_prob=0.20,
        )
        self.save_snapshot("p2", issue=issue, market_home_prob=0.34, market_draw_prob=0.33, market_away_prob=0.33)
        self.insert_run(
            "p1",
            issue=issue,
            created_at="2026-04-29 18:00:00",
            recommendation="观望",
            effective_recommendation="观望",
            suggested_stake_pct=0.0,
            effective_stake_pct=0.0,
            market_odds_home=1.75,
            market_odds_draw=3.2,
            market_odds_away=4.8,
            market_home_prob=0.56,
            market_draw_prob=0.24,
            market_away_prob=0.19,
            legacy_home_prob=0.55,
            legacy_draw_prob=0.24,
            legacy_away_prob=0.21,
        )
        self.insert_run(
            "p2",
            issue=issue,
            created_at="2026-04-29 18:05:00",
            recommendation="观望",
            effective_recommendation="观望",
            suggested_stake_pct=0.0,
            effective_stake_pct=0.0,
            market_home_prob=0.34,
            market_draw_prob=0.33,
            market_away_prob=0.33,
            legacy_home_prob=0.34,
            legacy_draw_prob=0.33,
            legacy_away_prob=0.33,
        )

        full_preview = prediction_engine.preview_production_single_pick(issue, limit=None)
        preview = prediction_engine.preview_production_single_pick(issue, limit=10)
        limited_preview = prediction_engine.preview_production_single_pick(issue, limit=1)

        self.assertEqual(collection_repository.get_issue_top_picks(issue), [])
        self.assertEqual(len(full_preview), 2)
        self.assertEqual(len(preview), 2)
        self.assertEqual(len(limited_preview), 1)
        self.assertEqual(preview[0]["match_id"], "p1")
        self.assertEqual(preview[0]["action"], "轻仓")
        self.assertEqual(preview[0]["outcome"], "home")
        self.assertEqual(preview[0]["reason"], "R2_market_home")
        self.assertEqual(preview[0]["strategy_key"], "coverage_draw_rescue")
        self.assertEqual(preview[1]["action"], "观望")
        self.assertEqual(preview[1]["reason"], "coverage_draw_rescue_watch")

    def test_apply_target_batch_strategy_updates_latest_runs(self) -> None:
        issue = "26102"
        self.insert_match("tb1", issue=issue, home_team="Batch Home", away_team="Batch Away")
        self.insert_match("tb2", issue=issue, home_team="Watch Home", away_team="Watch Away")
        self.save_snapshot("tb1", issue=issue, market_home_prob=0.56, market_draw_prob=0.24, market_away_prob=0.20)
        self.save_snapshot("tb2", issue=issue, market_home_prob=0.33, market_draw_prob=0.34, market_away_prob=0.33)
        self.insert_run(
            "tb1",
            issue=issue,
            created_at="2026-04-29 18:00:00",
            recommendation="观望",
            recommended_outcome="",
            suggested_stake_pct=0.0,
            effective_recommendation="观望",
            effective_stake_pct=0.0,
            market_odds_home=1.75,
            market_odds_draw=3.2,
            market_odds_away=4.8,
            market_home_prob=0.56,
            market_draw_prob=0.24,
            market_away_prob=0.20,
            legacy_home_prob=0.55,
            legacy_draw_prob=0.24,
            legacy_away_prob=0.21,
            predicted_score="2-0",
        )
        self.insert_run(
            "tb2",
            issue=issue,
            created_at="2026-04-29 18:05:00",
            recommendation="观望",
            recommended_outcome="",
            algo_recommended_outcome="draw",
            suggested_stake_pct=0.0,
            effective_recommendation="观望",
            effective_stake_pct=0.0,
            market_home_prob=0.33,
            market_draw_prob=0.34,
            market_away_prob=0.33,
            legacy_home_prob=0.33,
            legacy_draw_prob=0.34,
            legacy_away_prob=0.33,
            predicted_score="",
        )

        result = prediction_engine.apply_target_batch_strategy_to_issue(issue)
        tb1 = dict(collection_repository.list_prediction_runs("tb1", limit=1)[0])
        tb2 = dict(collection_repository.list_prediction_runs("tb2", limit=1)[0])

        self.assertEqual(result["updated_count"], 2)
        self.assertEqual(result["action_count"], 1)
        self.assertEqual(tb1["recommendation"], "轻仓")
        self.assertEqual(tb1["recommended_outcome"], "home")
        self.assertEqual(tb1["effective_recommendation"], "轻仓")
        self.assertEqual(tb1["effective_action_source"], "target_batch_strategy")
        self.assertGreater(float(tb1["effective_stake_pct"] or 0), 0.0)
        self.assertEqual(tb2["recommendation"], "观望")
        self.assertEqual(tb2["recommended_outcome"], "draw")
        self.assertEqual(tb2["effective_recommendation"], "观望")
        self.assertEqual(tb2["effective_stake_pct"], 0.0)
        self.assertEqual(tb2["effective_action_source"], "target_batch_strategy")

    def test_handicap_action_summary_counts_only_actionable_handicap_rows(self) -> None:
        issue = "26102H"
        self.insert_match("h1", issue=issue, home_team="Handicap Home", away_team="Handicap Away")
        self.insert_match("h2", issue=issue, home_team="Watch Home", away_team="Watch Away")
        self.insert_match("h3", issue=issue, home_team="No Side Home", away_team="No Side Away")
        self.insert_run(
            "h1",
            issue=issue,
            created_at="2026-04-29 18:00:00",
            handicap_recommendation="轻仓",
            handicap_recommended_side="home",
        )
        self.insert_run(
            "h2",
            issue=issue,
            created_at="2026-04-29 18:05:00",
            handicap_recommendation="观望",
            handicap_recommended_side="",
        )
        self.insert_run(
            "h3",
            issue=issue,
            created_at="2026-04-29 18:10:00",
            handicap_recommendation="轻仓",
            handicap_recommended_side="",
        )

        summary = prediction_engine._handicap_action_summary_for_issue(issue)

        self.assertEqual(summary["sample_count"], 3)
        self.assertEqual(summary["action_count"], 1)
        self.assertEqual(summary["watch_count"], 2)

    def test_issue_action_summary_message_reports_settled_history_counts(self) -> None:
        issue = "26102S"
        self.insert_match("s1", issue=issue, home_team="Settled Home", away_team="Settled Away")
        self.insert_match("s2", issue=issue, home_team="Watch Home", away_team="Watch Away")
        self.insert_run(
            "s1",
            issue=issue,
            created_at="2026-04-29 18:00:00",
            recommendation="主推",
            recommended_outcome="home",
            suggested_stake_pct=5.0,
            handicap_recommendation="轻仓",
            handicap_recommended_side="away",
        )
        self.insert_run(
            "s2",
            issue=issue,
            created_at="2026-04-29 18:05:00",
            recommendation="观望",
            recommended_outcome="draw",
            suggested_stake_pct=0.0,
            handicap_recommendation="观望",
            handicap_recommended_side="",
        )

        message = prediction_engine._issue_action_summary_message(
            issue,
            {"updated_count": 0, "settled_skip_count": 2},
        )

        self.assertIn("主推荐 1/2 场可执行", message)
        self.assertIn("让球盘 1/2 场可执行", message)
        self.assertIn("已结算场次 2 场保留原记录", message)

    def test_predict_issue_applies_issue_strategy_once_after_batch(self) -> None:
        issue = "26102B"
        self.insert_match("b1", issue=issue, home_team="Batch One", away_team="Away One")
        self.insert_match("b2", issue=issue, home_team="Batch Two", away_team="Away Two")

        def fake_predict_match(match_id: str, **kwargs) -> dict:
            return {"match_id": match_id, "status": "ok", "kwargs": kwargs}

        with patch.object(prediction_engine, "get_collection_failure_reason", return_value=""):
            with patch.object(prediction_engine, "predict_match", side_effect=fake_predict_match) as predict_mock:
                with patch.object(
                    prediction_engine,
                    "apply_target_batch_strategy_to_issue",
                    return_value={
                        "issue": issue,
                        "strategy_key": "coverage_draw_rescue",
                        "updated_count": 2,
                        "action_count": 1,
                        "watch_count": 1,
                        "sample_count": 2,
                        "settled_skip_count": 0,
                    },
                ) as apply_mock:
                    result = prediction_engine.predict_issue(issue, ensure_collected=False)

        self.assertEqual(result["predicted_count"], 2)
        self.assertEqual(predict_mock.call_count, 2)
        self.assertTrue(
            all(call.kwargs.get("apply_issue_strategy") is False for call in predict_mock.call_args_list)
        )
        apply_mock.assert_called_once_with(issue)

    def test_apply_target_batch_strategy_skips_settled_runs(self) -> None:
        issue = "26103"
        self.insert_match(
            "settled",
            issue=issue,
            match_time="2026-04-29 20:00:00",
            home_team="Settled Home",
            away_team="Settled Away",
        )
        self.save_snapshot(
            "settled",
            issue=issue,
            market_home_prob=0.56,
            market_draw_prob=0.24,
            market_away_prob=0.20,
        )
        run_id = self.insert_run(
            "settled",
            issue=issue,
            created_at="2026-04-29 18:00:00",
            recommendation="观望",
            recommended_outcome="",
            suggested_stake_pct=0.0,
            effective_recommendation="观望",
            effective_stake_pct=0.0,
            market_odds_home=1.75,
            market_odds_draw=3.2,
            market_odds_away=4.8,
            market_home_prob=0.56,
            market_draw_prob=0.24,
            market_away_prob=0.20,
            legacy_home_prob=0.55,
            legacy_draw_prob=0.24,
            legacy_away_prob=0.21,
            predicted_score="2-0",
        )
        prediction_engine.record_feedback(
            run_id,
            "settled",
            "home",
            actual_score="2-0",
            result_status="settled",
            settled_at="2026-04-29 22:00:00",
        )
        before_summary = collection_repository.get_feedback_summary(issue)

        result = prediction_engine.apply_target_batch_strategy_to_issue(issue)
        settled_run = collection_repository.get_prediction_run(run_id)
        after_summary = collection_repository.get_feedback_summary(issue)

        self.assertEqual(result["updated_count"], 0)
        self.assertEqual(result["sample_count"], 0)
        self.assertEqual(result["settled_skip_count"], 1)
        self.assertEqual(before_summary, after_summary)
        self.assertIsNotNone(settled_run)
        self.assertEqual(settled_run["recommendation"], "观望")
        self.assertEqual(settled_run["recommended_outcome"], "")
        self.assertNotEqual(settled_run["effective_action_source"], "target_batch_strategy")

    def test_preview_production_single_pick_uses_target_batch_layer(self) -> None:
        issue = "26101"
        self.insert_match("safe", issue=issue, home_team="Safe Home", away_team="Away")
        self.insert_match("guarded", issue=issue, home_team="Guarded Home", away_team="Away")
        self.insert_match("ultra_flagged", issue=issue, home_team="Ultra Home", away_team="Away")
        self.save_snapshot(
            "safe",
            issue=issue,
            market_home_prob=0.56,
            market_draw_prob=0.24,
            market_away_prob=0.20,
        )
        self.save_snapshot(
            "guarded",
            issue=issue,
            market_home_prob=0.56,
            market_draw_prob=0.25,
            market_away_prob=0.19,
        )
        self.save_snapshot(
            "ultra_flagged",
            issue=issue,
            market_home_prob=0.67,
            market_draw_prob=0.19,
            market_away_prob=0.14,
            h2h_edge=0.3,
        )
        self.insert_run(
            "safe",
            issue=issue,
            created_at="2026-04-29 18:00:00",
            recommendation="观望",
            effective_recommendation="观望",
            suggested_stake_pct=0.0,
            effective_stake_pct=0.0,
            market_odds_home=1.75,
            market_odds_draw=3.2,
            market_odds_away=4.8,
            market_home_prob=0.56,
            market_draw_prob=0.24,
            market_away_prob=0.20,
            legacy_home_prob=0.55,
            legacy_draw_prob=0.24,
            legacy_away_prob=0.21,
        )
        self.insert_run(
            "guarded",
            issue=issue,
            created_at="2026-04-29 18:05:00",
            recommendation="观望",
            effective_recommendation="观望",
            suggested_stake_pct=0.0,
            effective_stake_pct=0.0,
            market_odds_home=1.75,
            market_odds_draw=3.2,
            market_odds_away=4.8,
            market_home_prob=0.56,
            market_draw_prob=0.25,
            market_away_prob=0.19,
            legacy_home_prob=0.55,
            legacy_draw_prob=0.25,
            legacy_away_prob=0.20,
        )
        self.insert_run(
            "ultra_flagged",
            issue=issue,
            created_at="2026-04-29 18:10:00",
            recommendation="主推",
            recommended_outcome="home",
            effective_recommendation="主推",
            suggested_stake_pct=0.2,
            effective_stake_pct=0.2,
            market_odds_home=1.40,
            market_odds_draw=5.0,
            market_odds_away=6.5,
            market_home_prob=0.67,
            market_draw_prob=0.19,
            market_away_prob=0.14,
            legacy_home_prob=0.65,
            legacy_draw_prob=0.19,
            legacy_away_prob=0.16,
        )

        preview = {
            item["match_id"]: item
            for item in prediction_engine.preview_production_single_pick(issue, limit=None)
        }

        self.assertEqual(preview["safe"]["action"], "轻仓")
        self.assertEqual(preview["safe"]["reason"], "R2_market_home")
        self.assertEqual(preview["safe"]["strategy_key"], "coverage_draw_rescue")
        self.assertEqual(preview["safe"]["observation_flag_count"], 0)
        self.assertEqual(preview["safe"]["observation_risk_level"], "clean")
        self.assertEqual(preview["guarded"]["action"], "轻仓")
        self.assertEqual(preview["guarded"]["reason"], "R2_market_home")
        self.assertGreater(preview["guarded"]["observation_flag_count"], 0)
        self.assertEqual(preview["ultra_flagged"]["action"], "主推")
        self.assertEqual(preview["ultra_flagged"]["reason"], "existing")
        flagged_keys = {flag["key"] for flag in preview["ultra_flagged"]["observation_flags"]}
        self.assertIn("broad", flagged_keys)
        self.assertIn("ultra", flagged_keys)
        self.assertEqual(preview["ultra_flagged"]["observation_risk_level"], "stacked")

    def test_sync_issue_matches_upserts_matches_and_returns_status(self) -> None:
        issue = "26069"
        matches = [
            {
                "match_id": "26069001",
                "issue": issue,
                "league": "Test League",
                "match_no": "1",
                "match_time": "2026-04-01 20:00",
                "home_team": "Home",
                "away_team": "Away",
                "source_match_url": "https://trade.500.com/sfc/?expect=26069",
                "shuju_url": "https://odds.500.com/fenxi/shuju-26069001.shtml",
                "ouzhi_url": "https://odds.500.com/fenxi/ouzhi-26069001.shtml?ctype=2",
                "touzhu_url": "https://odds.500.com/fenxi/touzhu-26069001.shtml",
                "list_odds_win": "2.10",
                "list_odds_draw": "3.20",
                "list_odds_loss": "4.30",
                "list_heat_win": "40",
                "list_heat_draw": "30",
                "list_heat_loss": "30",
                "sync_time": "2026-04-01 10:00:00",
            }
        ]

        with patch.object(collection_service, "fetch_issue_matches", return_value=matches):
            result = collection_service.sync_issue_matches(issue, return_details=True)

        stored = collection_repository.get_match_analysis("26069001")
        self.assertIsNotNone(stored)
        self.assertEqual(str(stored["issue"]), issue)
        self.assertEqual(result["issue"], issue)
        self.assertEqual(len(result["matches"]), 1)
        self.assertEqual(result["status_level"], "success")
        self.assertIn("已补入期号 26069", result["status_message"])

    def test_canonical_run_is_current_unique_prediction(self) -> None:
        self.insert_match("M1", issue="20260429")
        run_id = self.insert_run("M1", issue="20260429", created_at="2026-04-29 21:00:00")

        canonical = prediction_engine.get_canonical_prediction_run("M1")
        result = prediction_engine.record_feedback(run_id, "M1", "home", result_status="settled")

        self.assertIsNotNone(canonical)
        self.assertEqual(int(canonical["run_id"]), run_id)
        self.assertEqual(result["prediction_run_id"], run_id)

    def test_record_feedback_upserts_and_switches_roi_source(self) -> None:
        self.insert_match("M2", issue="20260429")
        run_id = self.insert_run("M2", issue="20260429", created_at="2026-04-29 18:00:00")

        auto_result = prediction_engine.record_feedback(
            run_id,
            "M2",
            "home",
            actual_score="2-1",
            result_status="settled",
        )
        manual_result = prediction_engine.record_feedback(
            run_id,
            "M2",
            "home",
            actual_score="2-1",
            roi_delta=0.42,
            notes="manual override",
        )

        feedback_row = collection_repository.get_feedback_log(run_id)
        self.assertEqual(self.feedback_count(), 1)
        self.assertAlmostEqual(auto_result["roi_delta"], 0.15, places=4)
        self.assertEqual(auto_result["roi_source"], "auto")
        self.assertAlmostEqual(manual_result["roi_delta"], 0.42, places=4)
        self.assertEqual(manual_result["roi_source"], "manual_override")
        self.assertAlmostEqual(float(feedback_row["roi_delta"]), 0.42, places=4)
        self.assertEqual(str(feedback_row["roi_source"]), "manual_override")

    def test_settle_defaults_to_latest_issue(self) -> None:
        self.insert_match("OLD1", issue="20260428", match_no="1")
        self.insert_match("NEW1", issue="20260429", match_no="1")
        self.insert_run("OLD1", issue="20260428", created_at="2026-04-28 18:00:00")
        latest_run = self.insert_run("NEW1", issue="20260429", created_at="2026-04-29 18:00:00")

        with patch(
            "prediction_engine.fetch_issue_results",
            side_effect=lambda issue: [
                {
                    "match_id": "NEW1",
                    "actual_result": "home",
                    "actual_score": "1-0",
                    "result_status": "settled",
                    "result_source_url": f"https://trade.500.com/rj/?expect={issue}",
                }
            ],
        ):
            result = prediction_engine.settle_issue_results()

        self.assertEqual(result["issue"], "20260429")
        self.assertEqual(result["settled_count"], 1)
        self.assertEqual(self.feedback_count(), 1)
        feedback_row = collection_repository.get_feedback_log(latest_run)
        self.assertIsNotNone(feedback_row)

    def test_settle_issue_results_refreshes_top_picks_status(self) -> None:
        issue = "20260429"
        self.insert_match("TOP1", issue=issue, match_time="2026-04-29 20:00:00")
        self.insert_match("TOP2", issue=issue, match_time="2026-04-29 21:00:00")
        self.insert_run("TOP1", issue=issue, created_at="2026-04-29 18:00:00")
        self.insert_run("TOP2", issue=issue, created_at="2026-04-29 18:01:00")
        collection_repository.compute_issue_top_picks(issue)

        with patch(
            "prediction_engine.fetch_issue_results",
            return_value=[
                {
                    "match_id": "TOP1",
                    "actual_result": "home",
                    "actual_score": "2-1",
                    "result_status": "settled",
                    "result_source_url": "https://trade.500.com/rj/?expect=20260429",
                },
                {
                    "match_id": "TOP2",
                    "actual_result": "away",
                    "actual_score": "0-1",
                    "result_status": "settled",
                    "result_source_url": "https://trade.500.com/rj/?expect=20260429",
                },
            ],
        ):
            result = prediction_engine.settle_issue_results(issue)

        top_picks = {
            row["match_id"]: row
            for row in collection_repository.get_issue_top_picks(issue)
        }

        self.assertEqual(result["settled_count"], 2)
        self.assertEqual(top_picks["TOP1"]["top_pick_result_status"], "hit")
        self.assertEqual(top_picks["TOP2"]["top_pick_result_status"], "miss")

    def test_settle_issue_results_falls_back_to_match_detail_result(self) -> None:
        issue = "20260429"
        self.insert_match("1407734", issue=issue, home_team="长春女足", away_team="山东女足")
        with closing(collection_repository.get_connection()) as conn:
            conn.execute(
                "UPDATE matches SET shuju_url = ? WHERE match_id = ?",
                ("https://odds.500.com/fenxi/shuju-1407734.shtml", "1407734"),
            )
            conn.commit()
        run_id = self.insert_run("1407734", issue=issue, created_at="2026-04-29 18:00:00")

        with patch("prediction_engine.fetch_issue_results", return_value=[]), patch(
            "prediction_engine.fetch_result_from_match_url",
            return_value={
                "match_id": "1407734",
                "actual_result": "home",
                "actual_score": "2-1",
                "result_status": "settled",
                "result_source_url": "https://odds.500.com/fenxi/shuju-1407734.shtml",
            },
        ):
            result = prediction_engine.settle_issue_results(issue)

        match_row = collection_repository.get_match_analysis("1407734")
        feedback_row = collection_repository.get_feedback_log(run_id)
        self.assertEqual(result["settled_count"], 1)
        self.assertEqual(result["skipped_count"], 0)
        self.assertEqual(str(match_row["actual_score"]), "2-1")
        self.assertEqual(str(match_row["result_source_url"]), "https://odds.500.com/fenxi/shuju-1407734.shtml")
        self.assertIsNotNone(feedback_row)

    def test_settle_is_idempotent_and_preserves_manual_roi_override(self) -> None:
        self.insert_match("M3", issue="20260429")
        run_id = self.insert_run("M3", issue="20260429", created_at="2026-04-29 18:00:00")

        prediction_engine.record_feedback(
            run_id,
            "M3",
            "home",
            actual_score="2-1",
            roi_delta=0.77,
            notes="manual settlement",
        )

        with patch(
            "prediction_engine.fetch_issue_results",
            return_value=[
                {
                    "match_id": "M3",
                    "actual_result": "away",
                    "actual_score": "0-1",
                    "result_status": "settled",
                    "result_source_url": "https://trade.500.com/rj/?expect=20260429",
                }
            ],
        ):
            first = prediction_engine.settle_issue_results("20260429")
            second = prediction_engine.settle_issue_results("20260429")

        feedback_row = collection_repository.get_feedback_log(run_id)
        match_row = collection_repository.get_match_analysis("M3")
        self.assertEqual(first["settled_count"], 1)
        self.assertEqual(second["settled_count"], 1)
        self.assertEqual(self.feedback_count(), 1)
        self.assertAlmostEqual(float(feedback_row["roi_delta"]), 0.77, places=4)
        self.assertEqual(str(feedback_row["roi_source"]), "manual_override")
        self.assertEqual(str(match_row["actual_result"]), "home")
        self.assertEqual(str(match_row["result_status"]), "manual_override")

    def test_settle_enriches_manual_override_with_missing_score_and_source(self) -> None:
        self.insert_match("M36", issue="20260429")
        run_id = self.insert_run("M36", issue="20260429", created_at="2026-04-29 18:00:00")

        prediction_engine.record_feedback(
            run_id,
            "M36",
            "home",
            actual_score="",
            notes="manual result only",
        )

        with patch(
            "prediction_engine.fetch_issue_results",
            return_value=[
                {
                    "match_id": "M36",
                    "actual_result": "home",
                    "actual_score": "5-4",
                    "result_status": "settled",
                    "result_source_url": "https://odds.500.com/fenxi/shuju-1407201.shtml",
                }
            ],
        ):
            result = prediction_engine.settle_match_result("M36")

        match_row = collection_repository.get_match_analysis("M36")
        feedback_row = collection_repository.get_feedback_log(run_id)
        self.assertEqual(result["settled_count"], 1)
        self.assertEqual(result["result_synced_count"], 1)
        self.assertEqual(str(match_row["actual_result"]), "home")
        self.assertEqual(str(match_row["actual_score"]), "5-4")
        self.assertEqual(str(match_row["result_status"]), "settled")
        self.assertEqual(str(match_row["result_source_url"]), "https://odds.500.com/fenxi/shuju-1407201.shtml")
        self.assertEqual(str(feedback_row["actual_score"]), "5-4")

    def test_settle_normalizes_stale_manual_override_when_auto_result_fully_matches(self) -> None:
        self.insert_match("M37", issue="20260429")
        run_id = self.insert_run("M37", issue="20260429", created_at="2026-04-29 18:00:00")

        prediction_engine.record_feedback(
            run_id,
            "M37",
            "home",
            actual_score="",
            notes="manual result only",
        )
        collection_repository.upsert_match_results(
            [
                {
                    "match_id": "M37",
                    "actual_result": "home",
                    "actual_score": "5-4",
                    "result_status": "manual_override",
                    "result_source_url": "https://odds.500.com/fenxi/shuju-1407201.shtml",
                    "result_synced_at": "2026-04-29 20:30:00",
                }
            ]
        )

        with patch(
            "prediction_engine.fetch_issue_results",
            return_value=[
                {
                    "match_id": "M37",
                    "actual_result": "home",
                    "actual_score": "5-4",
                    "result_status": "settled",
                    "result_source_url": "https://odds.500.com/fenxi/shuju-1407201.shtml",
                }
            ],
        ):
            result = prediction_engine.settle_match_result("M37")

        match_row = collection_repository.get_match_analysis("M37")
        feedback_row = collection_repository.get_feedback_log(run_id)
        self.assertEqual(result["settled_count"], 1)
        self.assertEqual(result["result_synced_count"], 1)
        self.assertEqual(str(match_row["result_status"]), "settled")
        self.assertEqual(str(match_row["actual_score"]), "5-4")
        self.assertEqual(str(match_row["result_source_url"]), "https://odds.500.com/fenxi/shuju-1407201.shtml")
        self.assertEqual(str(feedback_row["actual_score"]), "5-4")

    def test_settle_match_result_only_affects_target_match_and_uses_match_issue(self) -> None:
        self.insert_match("M31", issue="20260429", match_no="1", home_team="A", away_team="B")
        self.insert_match("M32", issue="20260429", match_no="2", home_team="C", away_team="D")
        target_run = self.insert_run("M31", issue="20260429", created_at="2026-04-29 18:00:00")
        other_run = self.insert_run("M32", issue="20260429", created_at="2026-04-29 18:05:00")
        requested_issues: list[str] = []

        def _fake_fetch(issue: str):
            requested_issues.append(issue)
            return [
                {
                    "match_id": "M31",
                    "actual_result": "home",
                    "actual_score": "2-1",
                    "result_status": "settled",
                    "result_source_url": f"https://trade.500.com/rj/?expect={issue}",
                },
                {
                    "match_id": "M32",
                    "actual_result": "away",
                    "actual_score": "0-1",
                    "result_status": "settled",
                    "result_source_url": f"https://trade.500.com/rj/?expect={issue}",
                },
            ]

        with patch("prediction_engine.fetch_issue_results", side_effect=_fake_fetch):
            result = prediction_engine.settle_match_result("M31")

        self.assertEqual(requested_issues, ["20260429"])
        self.assertEqual(result["settled_count"], 1)
        self.assertEqual(result["result_synced_count"], 1)
        self.assertEqual(self.feedback_count(), 1)
        self.assertIsNotNone(collection_repository.get_feedback_log(target_run))
        self.assertIsNone(collection_repository.get_feedback_log(other_run))

    def test_settle_match_result_returns_skip_reason_when_result_missing(self) -> None:
        self.insert_match("M33", issue="20260429")
        self.insert_run("M33", issue="20260429", created_at="2026-04-29 18:00:00")

        with patch("prediction_engine.fetch_issue_results", return_value=[]):
            result = prediction_engine.settle_match_result("M33")

        self.assertEqual(result["settled_count"], 0)
        self.assertEqual(result["skipped_count"], 1)
        self.assertEqual(result["skipped_matches"][0]["reason"], "未命中完场赛果")

    def test_settle_match_result_returns_skip_reason_when_canonical_run_missing(self) -> None:
        self.insert_match("M34", issue="20260429")

        with patch(
            "prediction_engine.fetch_issue_results",
            return_value=[
                {
                    "match_id": "M34",
                    "actual_result": "draw",
                    "actual_score": "1-1",
                    "result_status": "settled",
                    "result_source_url": "https://trade.500.com/rj/?expect=20260429",
                }
            ],
        ):
            result = prediction_engine.settle_match_result("M34")

        self.assertEqual(result["settled_count"], 0)
        self.assertEqual(result["skipped_count"], 1)
        self.assertIn("未找到预测记录", result["skipped_matches"][0]["reason"])

    def test_settle_match_result_matches_batch_settlement_behavior(self) -> None:
        self.insert_match("M35", issue="20260429")
        self.insert_run("M35", issue="20260429", created_at="2026-04-29 18:00:00")
        result_entry = [
            {
                "match_id": "M35",
                "actual_result": "home",
                "actual_score": "3-1",
                "result_status": "settled",
                "result_source_url": "https://trade.500.com/rj/?expect=20260429",
            }
        ]

        with patch("prediction_engine.fetch_issue_results", return_value=result_entry):
            single_result = prediction_engine.settle_match_result("M35")
            batch_result = prediction_engine.settle_issue_results("20260429")

        feedback_row = collection_repository.get_feedback_log(1)
        self.assertEqual(single_result["settled_count"], 1)
        self.assertEqual(batch_result["settled_count"], 1)
        self.assertEqual(self.feedback_count(), 1)
        self.assertIsNotNone(feedback_row)
        self.assertAlmostEqual(float(feedback_row["roi_delta"]), 0.15, places=4)

    def test_backtest_uses_final_roi_for_feedback_summary_and_recomputes_algorithm_roi(self) -> None:
        self.insert_match("M4", issue="20260429", match_no="1")
        self.insert_match("M5", issue="20260429", match_no="2")

        watch_run = self.insert_run(
            "M4",
            issue="20260429",
            created_at="2026-04-29 18:00:00",
            recommendation="观望",
            recommended_outcome="home",
            suggested_stake_pct=0.0,
            algo_recommendation="轻仓",
            algo_recommended_outcome="home",
            algo_suggested_stake_pct=10.0,
        )
        final_run = self.insert_run(
            "M5",
            issue="20260429",
            created_at="2026-04-29 18:05:00",
            recommendation="轻仓",
            recommended_outcome="home",
            suggested_stake_pct=10.0,
            algo_recommendation="轻仓",
            algo_recommended_outcome="home",
            algo_suggested_stake_pct=10.0,
        )

        prediction_engine.record_feedback(watch_run, "M4", "home", actual_score="2-1", result_status="settled")
        prediction_engine.record_feedback(final_run, "M5", "home", actual_score="1-0", result_status="settled")

        feedback_summary = collection_repository.get_feedback_summary()
        backtest_summary = prediction_engine.summarize_backtest()

        self.assertEqual(feedback_summary["total_predictions"], 2)
        self.assertEqual(feedback_summary["hit_predictions"], 2)
        self.assertEqual(feedback_summary["miss_predictions"], 0)
        self.assertAlmostEqual(float(feedback_summary["total_roi"]), 0.15, places=4)
        self.assertAlmostEqual(float(backtest_summary["total_roi"]), 0.15, places=4)
        self.assertEqual(backtest_summary["final"]["action_count"], 1)
        self.assertEqual(backtest_summary["algorithm"]["action_count"], 2)
        self.assertAlmostEqual(float(backtest_summary["algorithm"]["total_roi"]), 0.30, places=4)
        self.assertIn("current_policy", backtest_summary)
        self.assertIn("action_count", backtest_summary["current_policy"])
        self.assertIn("buckets", backtest_summary["current_policy"])
        self.assertIn("target_strategy", backtest_summary)
        self.assertIn(backtest_summary["target_strategy"]["status"], {"insufficient_samples", "target_unreachable", "ready"})
        self.assertIn("hit_rate", backtest_summary["target_strategy"])
        self.assertIn("action_share", backtest_summary["target_strategy"])
        self.assertIn("frontier", backtest_summary["target_strategy"])
        self.assertIn("gaps", backtest_summary["target_strategy"]["frontier"])
        self.assertIn("review_signals", backtest_summary["target_strategy"])
        self.assertIn("validation_rows", backtest_summary["target_strategy"]["review_signals"])
        self.assertIn("balanced_single_pick", backtest_summary)
        balanced_summary = backtest_summary["balanced_single_pick"]
        self.assertIn("action_count", balanced_summary)
        self.assertIn("sample_count", balanced_summary)
        self.assertIn("action_share", balanced_summary)
        self.assertIn("hit_rate", balanced_summary)
        self.assertIn("total_roi", balanced_summary)
        self.assertIn("roi_on_stake", balanced_summary)
        self.assertIn("stability", balanced_summary)
        self.assertIn("latest_10_issues", balanced_summary["stability"])
        self.assertIn("rolling_10_issues", balanced_summary["stability"])
        self.assertIn("core", balanced_summary)
        self.assertIn("standard", balanced_summary)
        self.assertIn("roi_on_stake", balanced_summary["core"])
        self.assertIn("roi_on_stake", balanced_summary["standard"])
        self.assertIn("buckets", balanced_summary)
        self.assertIn("league_buckets", balanced_summary)
        self.assertIn("issue_buckets", balanced_summary)
        self.assertIn("diagnostics", balanced_summary)
        self.assertIn("recent_actions", balanced_summary)
        self.assertIn("misses", balanced_summary)
        self.assertIn("by_outcome", balanced_summary["diagnostics"])
        self.assertIn("by_reason", balanced_summary["diagnostics"])
        self.assertIn("by_odds_band", balanced_summary["diagnostics"])
        self.assertIn("by_confidence_band", balanced_summary["diagnostics"])
        self.assertIn("by_score_alignment", balanced_summary["diagnostics"])
        self.assertIn("league_guarded", balanced_summary)
        self.assertIn("action_count", balanced_summary["league_guarded"])
        self.assertIn("roi_on_stake", balanced_summary["league_guarded"])
        self.assertIn("selective", balanced_summary)
        self.assertIn("action_count", balanced_summary["selective"])
        self.assertIn("hit_rate", balanced_summary["selective"])
        self.assertIn("total_roi", balanced_summary["selective"])
        self.assertIn("roi_on_stake", balanced_summary["selective"])
        self.assertIn("stability", balanced_summary["selective"])
        self.assertIn("roi_on_stake", balanced_summary["selective"]["core"])
        self.assertIn("roi_on_stake", balanced_summary["selective"]["standard"])
        self.assertIn("league_buckets", balanced_summary["selective"])
        self.assertIn("issue_buckets", balanced_summary["selective"])
        self.assertIn("selective_league_guarded", balanced_summary)
        self.assertIn("action_count", balanced_summary["selective_league_guarded"])
        self.assertIn("roi_on_stake", balanced_summary["selective_league_guarded"])
        self.assertIn("strict", balanced_summary)
        self.assertIn("action_count", balanced_summary["strict"])
        self.assertIn("roi_on_stake", balanced_summary["strict"])
        self.assertIn("deep", balanced_summary)
        self.assertIn("action_count", balanced_summary["deep"])
        self.assertIn("roi_on_stake", balanced_summary["deep"])
        self.assertIn("refined", balanced_summary)
        self.assertIn("action_count", balanced_summary["refined"])
        self.assertIn("roi_on_stake", balanced_summary["refined"])
        self.assertIn("hardened", balanced_summary)
        self.assertIn("action_count", balanced_summary["hardened"])
        self.assertIn("roi_on_stake", balanced_summary["hardened"])
        self.assertIn("polished", balanced_summary)
        self.assertIn("action_count", balanced_summary["polished"])
        self.assertIn("roi_on_stake", balanced_summary["polished"])
        self.assertIn("steady", balanced_summary)
        self.assertIn("action_count", balanced_summary["steady"])
        self.assertIn("roi_on_stake", balanced_summary["steady"])
        self.assertIn("clean", balanced_summary)
        self.assertIn("action_count", balanced_summary["clean"])
        self.assertIn("roi_on_stake", balanced_summary["clean"])
        self.assertIn("precise", balanced_summary)
        self.assertIn("action_count", balanced_summary["precise"])
        self.assertIn("roi_on_stake", balanced_summary["precise"])
        self.assertIn("rescue", balanced_summary)
        self.assertIn("action_count", balanced_summary["rescue"])
        self.assertIn("roi_on_stake", balanced_summary["rescue"])
        self.assertIn("broad", balanced_summary)
        self.assertIn("action_count", balanced_summary["broad"])
        self.assertIn("roi_on_stake", balanced_summary["broad"])
        self.assertIn("cautious", balanced_summary)
        self.assertIn("action_count", balanced_summary["cautious"])
        self.assertIn("roi_on_stake", balanced_summary["cautious"])
        self.assertIn("ultra", balanced_summary)
        self.assertIn("action_count", balanced_summary["ultra"])
        self.assertIn("roi_on_stake", balanced_summary["ultra"])
        self.assertIn("variants", balanced_summary)
        self.assertEqual(
            set(balanced_summary["variants"]),
            {
                "base",
                "league_guarded",
                "selective",
                "selective_league_guarded",
                "strict",
                "deep",
                "refined",
                "hardened",
                "polished",
                "steady",
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
                "coverage_draw_rescue",
            },
        )
        for variant in balanced_summary["variants"].values():
            self.assertIn("label", variant)
            self.assertIn("role", variant)
            self.assertIn("note", variant)
            self.assertIn("action_count", variant)
            self.assertIn("action_share", variant)
            self.assertIn("hit_rate", variant)
            self.assertIn("roi_on_stake", variant)
            self.assertIn("rolling_10_min_hit_rate", variant)
            self.assertIn("rolling_10_min_roi_on_stake", variant)
        self.assertEqual(balanced_summary["variants"]["steady"]["role"], "production")
        self.assertEqual(balanced_summary["variants"]["clean"]["role"], "observation")
        self.assertEqual(balanced_summary["variants"]["precise"]["role"], "observation")
        self.assertEqual(balanced_summary["variants"]["rescue"]["role"], "observation")
        self.assertEqual(balanced_summary["variants"]["broad"]["role"], "observation")
        self.assertEqual(balanced_summary["variants"]["cautious"]["role"], "observation")
        self.assertEqual(balanced_summary["variants"]["ultra"]["role"], "observation")
        self.assertEqual(balanced_summary["variants"]["coverage_push"]["role"], "observation")
        self.assertEqual(balanced_summary["variants"]["coverage_stable"]["role"], "observation")
        self.assertEqual(balanced_summary["variants"]["coverage_refined"]["role"], "observation")
        self.assertEqual(balanced_summary["variants"]["coverage_value_guarded"]["role"], "observation")
        self.assertEqual(balanced_summary["variants"]["coverage_xg_guarded"]["role"], "observation")
        self.assertEqual(balanced_summary["variants"]["coverage_draw_rescue"]["role"], "production")
        self.assertIn("recommended_variant", balanced_summary)
        self.assertIn(
            balanced_summary["recommended_variant"].get("key"),
            set(balanced_summary["variants"]),
        )
        self.assertIn("production_variant", balanced_summary)
        self.assertIn(
            balanced_summary["production_variant"].get("key"),
            set(balanced_summary["variants"]),
        )
        self.assertNotEqual(balanced_summary["production_variant"].get("key"), "ultra")
        self.assertNotEqual(balanced_summary["production_variant"].get("key"), "precise")
        self.assertNotEqual(balanced_summary["production_variant"].get("key"), "rescue")
        self.assertNotEqual(balanced_summary["production_variant"].get("key"), "broad")
        self.assertNotEqual(balanced_summary["production_variant"].get("key"), "coverage_push")
        self.assertNotEqual(balanced_summary["production_variant"].get("key"), "coverage_stable")
        self.assertNotEqual(balanced_summary["production_variant"].get("key"), "coverage_refined")
        self.assertNotEqual(balanced_summary["production_variant"].get("key"), "coverage_value_guarded")
        self.assertNotEqual(balanced_summary["production_variant"].get("key"), "coverage_xg_guarded")
        self.assertEqual(balanced_summary["production_variant"].get("key"), "coverage_draw_rescue")
        self.assertIn("observation_deltas", balanced_summary)
        self.assertIn("clean", balanced_summary["observation_deltas"])
        self.assertIn("precise", balanced_summary["observation_deltas"])
        self.assertIn("rescue", balanced_summary["observation_deltas"])
        self.assertIn("broad", balanced_summary["observation_deltas"])
        self.assertIn("cautious", balanced_summary["observation_deltas"])
        self.assertIn("ultra", balanced_summary["observation_deltas"])
        self.assertIn("coverage_value_guarded", balanced_summary["observation_deltas"])
        self.assertIn("coverage_xg_guarded", balanced_summary["observation_deltas"])
        self.assertIn("coverage_draw_rescue", balanced_summary["observation_deltas"])
        for delta in balanced_summary["observation_deltas"].values():
            self.assertIn("filtered_count", delta)
            self.assertIn("filtered_miss_count", delta)
            self.assertIn("filtered_hit_rate", delta)
            self.assertIn("miss_capture_rate", delta)
            self.assertIn("hit_filter_rate", delta)
            self.assertIn("filtered_roi", delta)
            self.assertIn("filtered_reason_counts", delta)
            self.assertIn("filtered_reason_breakdown", delta)
            self.assertIn("filtered_issue_count", delta)
            self.assertIn("filtered_latest_issue", delta)
            self.assertIn("filtered_issue_breakdown", delta)
            self.assertIn("max_issue_filtered_share", delta)
            self.assertIn("max_issue_miss_share", delta)
            self.assertIn("filtered_examples", delta)
        self.assertIn("observation_transitions", balanced_summary)
        self.assertIn("precise_to_rescue", balanced_summary["observation_transitions"])
        transition = balanced_summary["observation_transitions"]["precise_to_rescue"]
        self.assertIn("restored_count", transition)
        self.assertIn("restored_hit_rate", transition)
        self.assertIn("restored_roi", transition)
        self.assertIn("restored_issue_count", transition)
        self.assertIn("restored_latest_issue", transition)
        self.assertIn("restored_examples", transition)
        self.assertIn("observation_risk_backtest", balanced_summary)
        for risk_key in ("clean", "single", "stacked", "resonance"):
            self.assertIn(risk_key, balanced_summary["observation_risk_backtest"])
            self.assertIn("action_count", balanced_summary["observation_risk_backtest"][risk_key])
            self.assertIn("hit_rate", balanced_summary["observation_risk_backtest"][risk_key])
            self.assertIn("recent_actions", balanced_summary["observation_risk_backtest"][risk_key])
            self.assertIn("misses", balanced_summary["observation_risk_backtest"][risk_key])
        self.assertIn("observation_combo_backtest", balanced_summary)
        self.assertTrue(balanced_summary["observation_combo_backtest"])
        combo = balanced_summary["observation_combo_backtest"][0]
        self.assertIn("key", combo)
        self.assertIn("label", combo)
        self.assertIn("action_count", combo)
        self.assertIn("hit_rate", combo)
        self.assertIn("recent_actions", combo)
        self.assertIn("misses", combo)
        self.assertIn("reason_breakdown", combo)
        self.assertIn("outcome_breakdown", combo)
        self.assertIn("league_breakdown", combo)
        self.assertIn("observation_combo_scenarios", balanced_summary)
        self.assertTrue(balanced_summary["observation_combo_scenarios"])
        scenario_keys = {scenario.get("key") for scenario in balanced_summary["observation_combo_scenarios"]}
        self.assertIn("rescue_high_resonance", scenario_keys)
        self.assertIn("rescue_high_resonance_r4_draw", scenario_keys)
        self.assertIn("rescue_high_resonance_existing_home", scenario_keys)
        scenario = balanced_summary["observation_combo_scenarios"][0]
        self.assertIn("production_action_count", scenario)
        self.assertIn("kept_action_count", scenario)
        self.assertIn("kept_hit_rate", scenario)
        self.assertIn("filtered_issue_count", scenario)
        self.assertIn("filtered_latest_issue", scenario)
        self.assertIn("max_issue_filtered_share", scenario)
        self.assertIn("filtered_examples", scenario)
        self.assertIn("observation_feature_profiles", balanced_summary)
        self.assertTrue(balanced_summary["observation_feature_profiles"])
        profile = balanced_summary["observation_feature_profiles"][0]
        self.assertIn("market_odds", profile)
        self.assertIn("market_draw_prob", profile)
        self.assertIn("confidence_score", profile)
        self.assertIn("reason_breakdown", profile)
        self.assertIn("score_direction_breakdown", profile)
        self.assertIn("coverage_target_diagnostics", balanced_summary)
        coverage = balanced_summary["coverage_target_diagnostics"]
        self.assertIn("target_action_share", coverage)
        self.assertIn("target_hit_rate", coverage)
        self.assertIn("base", coverage)
        self.assertIn("production", coverage)
        self.assertIn("coverage_push", coverage)
        self.assertIn("coverage_stable", coverage)
        self.assertIn("coverage_refined", coverage)
        self.assertIn("coverage_value_guarded", coverage)
        self.assertIn("coverage_xg_guarded", coverage)
        self.assertIn("coverage_draw_rescue", coverage)
        self.assertIn("watch_bucket_candidates", coverage)
        self.assertIn("target_batch_strategy", backtest_summary)
        target_batch = backtest_summary["target_batch_strategy"]
        self.assertEqual(target_batch["key"], "coverage_draw_rescue")
        self.assertEqual(target_batch["role"], "production")
        self.assertIn("target_met", target_batch)
        self.assertIn("required_actions", target_batch)
        self.assertIn("rolling_10_min_hit_rate", target_batch)
        self.assertIn("issue_buckets", target_batch)
        self.assertIn("recent_actions", target_batch)
        self.assertIn("misses", target_batch)
        self.assertIn("coverage_stability_diagnostics", balanced_summary)
        stability = balanced_summary["coverage_stability_diagnostics"]
        self.assertIn("low_window", stability)
        self.assertIn("tested_filters", stability)
        self.assertIn("status", stability)
        self.assertIn("observation_periods", balanced_summary)
        self.assertTrue(balanced_summary["observation_periods"])
        period = next(iter(balanced_summary["observation_periods"].values()))
        self.assertIn("production_action_count", period)
        self.assertIn("layers", period)
        self.assertIn("rescue", period["layers"])
        self.assertIn("broad", period["layers"])
        self.assertIn("filtered_miss_count", period["layers"]["broad"])
        self.assertIn("observation_readiness", balanced_summary)
        self.assertIn("rescue", balanced_summary["observation_readiness"])
        self.assertIn("broad", balanced_summary["observation_readiness"])
        rescue_readiness = balanced_summary["observation_readiness"]["rescue"]
        self.assertIn("status", rescue_readiness)
        self.assertIn("reason", rescue_readiness)
        self.assertIn("filtered_issue_count", rescue_readiness)
        self.assertEqual(rescue_readiness["status"], "needs_more_samples")

    def test_feedback_summary_filters_by_issue_and_counts_misses(self) -> None:
        self.insert_match("M41", issue="20260429", match_no="1")
        self.insert_match("M42", issue="20260429", match_no="2")
        self.insert_match("M43", issue="20260430", match_no="1")

        hit_run = self.insert_run("M41", issue="20260429", created_at="2026-04-29 18:00:00")
        miss_run = self.insert_run("M42", issue="20260429", created_at="2026-04-29 18:05:00")
        other_issue_run = self.insert_run(
            "M43",
            issue="20260430",
            created_at="2026-04-29 18:10:00",
            recommended_outcome="away",
        )

        prediction_engine.record_feedback(hit_run, "M41", "home", actual_score="2-1", result_status="settled")
        prediction_engine.record_feedback(miss_run, "M42", "away", actual_score="0-1", result_status="settled")
        prediction_engine.record_feedback(other_issue_run, "M43", "away", actual_score="1-2", result_status="settled")

        all_summary = collection_repository.get_feedback_summary()
        issue_summary = collection_repository.get_feedback_summary("20260429")
        other_issue_summary = collection_repository.get_feedback_summary("20260430")
        empty_summary = collection_repository.get_feedback_summary("missing")

        self.assertEqual(all_summary["total_predictions"], 3)
        self.assertEqual(all_summary["hit_predictions"], 2)
        self.assertEqual(all_summary["miss_predictions"], 1)
        self.assertEqual(issue_summary["total_predictions"], 2)
        self.assertEqual(issue_summary["hit_predictions"], 1)
        self.assertEqual(issue_summary["miss_predictions"], 1)
        self.assertAlmostEqual(issue_summary["hit_rate"], 0.5, places=4)
        self.assertEqual(other_issue_summary["total_predictions"], 1)
        self.assertEqual(other_issue_summary["hit_predictions"], 1)
        self.assertEqual(other_issue_summary["miss_predictions"], 0)
        self.assertEqual(empty_summary["total_predictions"], 0)
        self.assertEqual(empty_summary["hit_predictions"], 0)
        self.assertEqual(empty_summary["miss_predictions"], 0)
        self.assertEqual(empty_summary["hit_rate"], 0.0)

    def test_feedback_summary_counts_only_actionable_handicap_recommendations(self) -> None:
        issue = "20260429"
        self.insert_match("M44", issue=issue, match_no="1")
        self.insert_match("M45", issue=issue, match_no="2")
        self.insert_match("M46", issue=issue, match_no="3")

        action_run = self.insert_run(
            "M44",
            issue=issue,
            created_at="2026-04-29 18:00:00",
            handicap_recommendation="轻仓",
            handicap_recommended_side="home",
            handicap_line=-1.0,
        )
        watch_run = self.insert_run(
            "M45",
            issue=issue,
            created_at="2026-04-29 18:05:00",
            handicap_recommendation="观望",
            handicap_recommended_side="",
            handicap_line=-1.0,
        )
        no_side_run = self.insert_run(
            "M46",
            issue=issue,
            created_at="2026-04-29 18:10:00",
            handicap_recommendation="轻仓",
            handicap_recommended_side="",
            handicap_line=-1.0,
        )

        prediction_engine.record_feedback(action_run, "M44", "home", actual_score="2-0", result_status="settled")
        prediction_engine.record_feedback(watch_run, "M45", "home", actual_score="3-0", result_status="settled")
        prediction_engine.record_feedback(no_side_run, "M46", "away", actual_score="0-2", result_status="settled")

        summary = collection_repository.get_feedback_summary(issue)

        self.assertEqual(summary["total_predictions"], 3)
        self.assertEqual(summary["handicap_total_predictions"], 1)
        self.assertEqual(summary["handicap_hit_predictions"], 1)
        self.assertEqual(summary["handicap_miss_predictions"], 0)
        self.assertAlmostEqual(summary["handicap_hit_rate"], 1.0, places=4)

    def test_manual_review_resolution_updates_effective_action_and_backtest_uses_it(self) -> None:
        self.insert_match("M6", issue="20260429", match_no="1", match_time="2099-04-30 20:00:00")
        run_id = self.insert_run(
            "M6",
            issue="20260429",
            created_at="2026-04-29 18:00:00",
            recommendation="主推",
            suggested_stake_pct=10.0,
            algo_recommendation="主推",
            algo_suggested_stake_pct=10.0,
        )
        collection_repository.update_prediction_run_fields(
            run_id,
            {
                "manual_review_status": "pending",
                "manual_review_reason": "二级仲裁要求人工复核",
                "manual_review_requested_at": "2026-04-29 19:00:00",
                "effective_recommendation": "",
                "effective_stake_pct": 0.0,
                "effective_action_source": "",
            },
        )

        result = prediction_engine.resolve_manual_review(run_id, "观望", "人工改为放弃执行")
        updated_run = collection_repository.get_prediction_run(run_id)
        prediction_engine.record_feedback(
            run_id,
            "M6",
            "home",
            actual_score="2-1",
            result_status="settled",
        )
        feedback_row = collection_repository.get_feedback_log(run_id)
        backtest_summary = prediction_engine.summarize_backtest()

        self.assertEqual(result["manual_review_status"], "resolved")
        self.assertEqual(str(updated_run["effective_recommendation"]), "观望")
        self.assertEqual(float(updated_run["effective_stake_pct"]), 0.0)
        self.assertEqual(str(updated_run["effective_action_source"]), "manual_review")
        self.assertEqual(str(updated_run["manual_review_notes"]), "人工改为放弃执行")
        self.assertAlmostEqual(float(feedback_row["roi_delta"]), 0.0, places=4)
        self.assertEqual(backtest_summary["final"]["action_count"], 0)
        self.assertEqual(backtest_summary["algorithm"]["action_count"], 1)

    def test_record_feedback_expires_pending_manual_review_after_kickoff(self) -> None:
        self.insert_match("M7", issue="20260429", match_no="1", match_time="2026-04-29 20:00:00")
        run_id = self.insert_run("M7", issue="20260429", created_at="2026-04-29 18:00:00")
        collection_repository.update_prediction_run_fields(
            run_id,
            {
                "manual_review_status": "pending",
                "manual_review_reason": "二级仲裁要求人工复核",
                "manual_review_requested_at": "2026-04-29 19:00:00",
                "effective_recommendation": "",
                "effective_stake_pct": 0.0,
                "effective_action_source": "",
            },
        )

        prediction_engine.record_feedback(
            run_id,
            "M7",
            "home",
            actual_score="1-0",
            result_status="settled",
            settled_at="2026-04-29 21:00:00",
        )

        updated_run = collection_repository.get_prediction_run(run_id)
        feedback_row = collection_repository.get_feedback_log(run_id)
        self.assertEqual(str(updated_run["manual_review_status"]), "expired")
        self.assertAlmostEqual(float(feedback_row["roi_delta"]), 0.0, places=4)


class ManualReviewWebTests(TemporaryDatabaseTestCase):
    def test_manual_review_route_resolves_pending_run(self) -> None:
        self.insert_match("MW1", issue="20260429", match_no="1", match_time="2099-04-30 20:00:00")
        run_id = self.insert_run("MW1", issue="20260429", created_at="2026-04-29 18:00:00")
        collection_repository.update_prediction_run_fields(
            run_id,
            {
                "manual_review_status": "pending",
                "manual_review_reason": "二级仲裁要求人工复核",
                "manual_review_requested_at": "2026-04-29 19:00:00",
                "effective_recommendation": "",
                "effective_stake_pct": 0.0,
                "effective_action_source": "",
            },
        )

        client = web_app_module.app.test_client()
        response = client.post(
            f"/manual-review/{run_id}/resolve",
            data={
                "match_id": "MW1",
                "issue": "20260429",
                "effective_recommendation": "轻仓",
                "notes": "人工确认轻仓",
            },
        )

        updated_run = collection_repository.get_prediction_run(run_id)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(str(updated_run["manual_review_status"]), "resolved")
        self.assertEqual(str(updated_run["effective_recommendation"]), "轻仓")
        self.assertEqual(str(updated_run["effective_action_source"]), "manual_review")
        self.assertEqual(str(updated_run["manual_review_notes"]), "人工确认轻仓")


class WebSettlementTests(unittest.TestCase):
    def test_delete_selectable_match_route_deletes_custom_match(self) -> None:
        client = web_app_module.app.test_client()
        with (
            patch.object(web_app_module, "_ensure_db_initialized"),
            patch.object(
                web_app_module,
                "remove_selectable_match",
                return_value={
                    "deleted": True,
                    "issue": "26099",
                    "match_id": "1407734",
                    "status_message": "已删除自选对赛 match_id 1407734。",
                    "status_level": "success",
                },
            ) as mock_remove,
        ):
            response = client.post(
                "/select-matches/1407734/delete",
                data={"issue": "26099", "current_match_id": "1407734"},
            )

        self.assertEqual(response.status_code, 302)
        self.assertIn("issue=26099", response.headers["Location"])
        self.assertIn("level=success", response.headers["Location"])
        self.assertNotIn("match_id=1407734", response.headers["Location"])
        mock_remove.assert_called_once_with("1407734", issue="26099", return_details=True)

    def test_select_matches_route_adds_checked_matches_and_redirects_to_issue(self) -> None:
        client = web_app_module.app.test_client()
        with (
            patch.object(web_app_module, "_ensure_db_initialized"),
            patch.object(
                web_app_module,
                "add_selectable_matches",
                return_value={
                    "issue": "26099",
                    "matches": [{"match_id": "1407734"}],
                    "status_message": "已添加自选对赛 1 场到期号 26099。",
                    "status_level": "success",
                },
            ) as mock_add,
        ):
            response = client.post(
                "/select-matches",
                data={
                    "issue": "26099",
                    "match_id": "existing",
                    "selected_match_ids": ["1407734"],
                },
            )

        self.assertEqual(response.status_code, 302)
        self.assertIn("issue=26099", response.headers["Location"])
        self.assertIn("match_id=1407734", response.headers["Location"])
        self.assertIn("level=success", response.headers["Location"])
        mock_add.assert_called_once_with(["1407734"], issue="26099", return_details=True)

    def test_sync_issue_route_redirects_to_backfilled_issue(self) -> None:
        client = web_app_module.app.test_client()
        with (
            patch.object(web_app_module, "_ensure_db_initialized"),
            patch.object(
                web_app_module,
                "sync_issue_matches",
                return_value={
                    "issue": "26069",
                    "matches": [],
                    "status_message": "已补入期号 26069 对赛 14 场",
                    "status_level": "success",
                },
            ) as mock_sync,
        ):
            response = client.post("/sync-issue", data={"manual_issue": "26069"})

        self.assertEqual(response.status_code, 302)
        self.assertIn("issue=26069", response.headers["Location"])
        self.assertIn("level=success", response.headers["Location"])
        mock_sync.assert_called_once_with("26069", return_details=True)

    def test_sync_issue_route_redirects_error_for_bad_issue(self) -> None:
        client = web_app_module.app.test_client()
        with (
            patch.object(web_app_module, "_ensure_db_initialized"),
            patch.object(
                web_app_module,
                "sync_issue_matches",
                side_effect=RuntimeError("issue 必须是数字期号"),
            ),
        ):
            response = client.post("/sync-issue", data={"manual_issue": "abc"})

        self.assertEqual(response.status_code, 302)
        self.assertIn("level=error", response.headers["Location"])

    def test_index_renders_single_settlement_button(self) -> None:
        client = web_app_module.app.test_client()
        current_row = {
            "match_id": "M1",
            "issue": "20260429",
            "league": "Test League",
            "match_no": "1",
            "match_time": "2026-04-29 20:00:00",
            "home_team": "Home",
            "away_team": "Away",
            "collected_at": "2026-04-29 10:00:00",
            "actual_result": "",
            "actual_score": "",
            "result_status": "",
            "result_source_url": "",
            "result_synced_at": "",
        }
        with (
            patch.object(web_app_module, "_ensure_db_initialized"),
            patch.object(web_app_module, "get_database_status", return_value={"read_only": False, "message": "", "level": "info"}),
            patch.object(web_app_module, "expire_pending_manual_reviews"),
            patch.object(web_app_module, "list_matches_by_issue", return_value=[current_row]),
            patch.object(web_app_module, "list_issues", return_value=["20260429"]),
            patch.object(web_app_module, "get_match_analysis", return_value=current_row),
            patch.object(web_app_module, "list_prediction_runs", return_value=[]),
            patch.object(web_app_module, "list_pending_manual_review_runs", return_value=[]),
            patch.object(web_app_module, "get_canonical_prediction_run", return_value=None),
            patch.object(web_app_module, "get_collection_stats", return_value={"total_matches": 1, "success_analyses": 1, "failed_analyses": 0}),
            patch.object(web_app_module, "get_feedback_summary", return_value={"total_predictions": 0, "hit_predictions": 0, "miss_predictions": 0, "hit_rate": 0.0, "total_roi": 0.0, "avg_roi": 0.0}),
            patch.object(web_app_module, "summarize_backtest", return_value={"total_settled": 0, "message": "暂无样本"}),
            patch.object(web_app_module, "list_backtest_rows", return_value=[]),
            patch.object(
                web_app_module,
                "get_learning_overview",
                return_value={
                    "retention_issue_count": 90,
                    "settled_samples": 0,
                    "action_samples": 0,
                    "active_profile": None,
                    "latest_candidate": None,
                    "recent_profiles": [],
                },
            ),
            patch.object(web_app_module, "build_sections", return_value=[]),
        ):
            response = client.get("/?match_id=M1&issue=20260429")

        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("/sync-issue", html)
        self.assertIn("manual_issue", html)
        self.assertIn("补入错过期号", html)
        self.assertIn("赛后结果与赛前预测对比", html)
        self.assertIn("同步赛果并结算当前场次", html)
        self.assertIn("/history/backtest", html)
        self.assertIn("回测历史", html)
        self.assertIn("/history/settle", html)
        self.assertIn("结算历史", html)
        self.assertIn("/settle/M1", html)

    def test_historical_prediction_runner_skips_current_issue(self) -> None:
        with (
            patch.object(web_app_module, "list_issues", return_value=["20260430", "20260429", "20260428"]),
            patch.object(
                web_app_module,
                "predict_issue",
                return_value={
                    "predicted_count": 2,
                    "skipped_count": 1,
                    "prediction_failed_count": 0,
                },
            ) as mock_predict,
        ):
            result = web_app_module._run_historical_predictions("20260430")

        self.assertEqual(result["issue_count"], 2)
        self.assertEqual(result["predicted_count"], 4)
        self.assertEqual(result["skipped_count"], 2)
        self.assertEqual(result["status_level"], "success")
        self.assertEqual(
            [call.args[0] for call in mock_predict.call_args_list],
            ["20260429", "20260428"],
        )

    def test_historical_settlement_runner_skips_current_issue(self) -> None:
        with (
            patch.object(web_app_module, "list_issues", return_value=["20260430", "20260429", "20260428"]),
            patch.object(
                web_app_module,
                "settle_issue_results",
                return_value={
                    "result_synced_count": 2,
                    "settled_count": 2,
                    "skipped_count": 0,
                },
            ) as mock_settle,
        ):
            result = web_app_module._run_historical_settlement("20260430")

        self.assertEqual(result["issue_count"], 2)
        self.assertEqual(result["result_synced_count"], 4)
        self.assertEqual(result["settled_count"], 4)
        self.assertEqual(result["status_level"], "success")
        self.assertEqual(
            [call.args[0] for call in mock_settle.call_args_list],
            ["20260429", "20260428"],
        )

    def test_index_renders_pending_manual_review_queue(self) -> None:
        client = web_app_module.app.test_client()
        current_row = {
            "match_id": "M9",
            "issue": "20260429",
            "league": "Test League",
            "match_no": "1",
            "match_time": "2026-04-29 20:00:00",
            "home_team": "Home",
            "away_team": "Away",
            "collected_at": "2026-04-29 10:00:00",
            "actual_result": "",
            "actual_score": "",
            "result_status": "",
            "result_source_url": "",
            "result_synced_at": "",
        }
        pending_rows = [
            {
                "run_id": 91,
                "match_id": "M9",
                "issue": "20260429",
                "created_at": "2026-04-29 18:00:00",
                "recommendation": "轻仓",
                "recommended_outcome": "home",
                "suggested_stake_pct": 1.2,
                "effective_recommendation": "",
                "effective_stake_pct": 0.0,
                "effective_action_source": "",
                "arbiter_review_status": "completed",
                "arbiter_review_decision": "manual_review",
                "arbiter_review_target_action": "轻仓",
                "arbiter_review_reason": "模型分歧较大",
                "manual_review_status": "pending",
                "manual_review_reason": "模型分歧较大",
                "manual_review_requested_at": "2026-04-29 19:00:00",
                "manual_review_notes": "",
                "match_time": "2026-04-29 20:00:00",
                "home_team": "Home",
                "away_team": "Away",
                "league": "Test League",
            }
        ]
        with (
            patch.object(web_app_module, "_ensure_db_initialized"),
            patch.object(web_app_module, "get_database_status", return_value={"read_only": False, "message": "", "level": "info"}),
            patch.object(web_app_module, "expire_pending_manual_reviews"),
            patch.object(web_app_module, "list_matches_by_issue", return_value=[current_row]),
            patch.object(web_app_module, "list_issues", return_value=["20260429"]),
            patch.object(web_app_module, "get_match_analysis", return_value=current_row),
            patch.object(web_app_module, "list_prediction_runs", return_value=[]),
            patch.object(web_app_module, "list_pending_manual_review_runs", return_value=pending_rows),
            patch.object(web_app_module, "get_canonical_prediction_run", return_value=None),
            patch.object(web_app_module, "get_collection_stats", return_value={"total_matches": 1, "success_analyses": 1, "failed_analyses": 0}),
            patch.object(web_app_module, "get_feedback_summary", return_value={"total_predictions": 0, "hit_predictions": 0, "miss_predictions": 0, "hit_rate": 0.0, "total_roi": 0.0, "avg_roi": 0.0}),
            patch.object(web_app_module, "summarize_backtest", return_value={"total_settled": 0, "message": "暂无样本"}),
            patch.object(web_app_module, "list_backtest_rows", return_value=[]),
            patch.object(
                web_app_module,
                "get_learning_overview",
                return_value={
                    "retention_issue_count": 90,
                    "settled_samples": 0,
                    "action_samples": 0,
                    "active_profile": None,
                    "latest_candidate": None,
                    "recent_profiles": [],
                },
            ),
            patch.object(web_app_module, "build_sections", return_value=[]),
        ):
            response = client.get("/?match_id=M9&issue=20260429")

        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("专家复核记录", html)
        self.assertIn("历史 pending 记录仍可手工处理", html)
        self.assertIn("/manual-review/91/resolve", html)
        self.assertIn("提交人工处理", html)

    def test_settle_single_route_returns_async_payload(self) -> None:
        client = web_app_module.app.test_client()
        payload = {
            "task_id": "task-1",
            "status_url": "/tasks/task-1",
            "view_url": "/?task_id=task-1",
            "complete_url": "/",
        }
        with (
            patch.object(web_app_module, "_ensure_db_initialized"),
            patch.object(web_app_module, "_start_background_task", return_value=payload) as mock_start,
        ):
            response = client.post(
                "/settle/M1",
                data={"issue": "20260429"},
                headers={"X-Requested-With": "XMLHttpRequest", "Accept": "application/json"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), payload)
        self.assertEqual(mock_start.call_args.kwargs["kind"], "settle-single")
        self.assertEqual(mock_start.call_args.kwargs["match_id"], "M1")

    def test_history_backtest_route_returns_async_payload(self) -> None:
        client = web_app_module.app.test_client()
        payload = {
            "task_id": "task-2",
            "status_url": "/tasks/task-2",
            "view_url": "/?task_id=task-2",
            "complete_url": "/",
        }
        with (
            patch.object(web_app_module, "_ensure_db_initialized"),
            patch.object(web_app_module, "_start_background_task", return_value=payload) as mock_start,
        ):
            response = client.post(
                "/history/backtest",
                data={"match_id": "M1", "issue": "20260430"},
                headers={"X-Requested-With": "XMLHttpRequest", "Accept": "application/json"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), payload)
        self.assertEqual(mock_start.call_args.kwargs["kind"], "history-backtest")
        self.assertEqual(mock_start.call_args.kwargs["match_id"], "M1")
        self.assertEqual(mock_start.call_args.kwargs["issue"], "20260430")

    def test_history_settle_route_returns_async_payload(self) -> None:
        client = web_app_module.app.test_client()
        payload = {
            "task_id": "task-3",
            "status_url": "/tasks/task-3",
            "view_url": "/?task_id=task-3",
            "complete_url": "/",
        }
        with (
            patch.object(web_app_module, "_ensure_db_initialized"),
            patch.object(web_app_module, "_start_background_task", return_value=payload) as mock_start,
        ):
            response = client.post(
                "/history/settle",
                data={"match_id": "M1", "issue": "20260430"},
                headers={"X-Requested-With": "XMLHttpRequest", "Accept": "application/json"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), payload)
        self.assertEqual(mock_start.call_args.kwargs["kind"], "history-settle")
        self.assertEqual(mock_start.call_args.kwargs["match_id"], "M1")
        self.assertEqual(mock_start.call_args.kwargs["issue"], "20260430")


if __name__ == "__main__":
    unittest.main()
