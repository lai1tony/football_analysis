import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import app as web_app_module
import backfill_collection_quality
import collection_repository
import collection_strategy
from feature_engine import build_match_features, extract_market_value_metrics
import collection_service
import prediction_engine
import source_500_client
import source_lineup_client
import source_market_value_client


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

    def insert_match(self, match_id: str = "M1", issue: str = "20260429") -> None:
        collection_repository.upsert_matches(
            [
                {
                    "match_id": match_id,
                    "issue": issue,
                    "league": "Test League",
                    "match_no": "1",
                    "match_time": "2026-04-29 20:00:00",
                    "home_team": "Home",
                    "away_team": "Away",
                    "source_match_url": "https://trade.500.com/sfc/",
                    "shuju_url": "https://odds.500.com/fenxi/shuju-M1.shtml",
                    "ouzhi_url": "https://odds.500.com/fenxi/ouzhi-M1.shtml",
                    "touzhu_url": "https://odds.500.com/fenxi/touzhu-M1.shtml",
                    "list_odds_win": "2.10",
                    "list_odds_draw": "3.20",
                    "list_odds_loss": "3.80",
                    "list_heat_win": "40",
                    "list_heat_draw": "30",
                    "list_heat_loss": "30",
                    "sync_time": "2026-04-29 10:00:00",
                }
            ]
        )


class CollectionQualityTests(TemporaryDatabaseTestCase):
    def test_sfc_issue_sequence_parser_keeps_real_cross_year_order(self):
        html = """
        <a href="/shtml/sfc/25195.shtml">25195</a>
        <a href="/shtml/sfc/26001.shtml">26001</a>
        <a href="/shtml/sfc/25196.shtml">25196</a>
        """

        self.assertEqual(
            source_500_client.parse_sfc_issue_sequence_html(html),
            ["25195", "25196", "26001"],
        )

    def test_market_value_parser_accepts_squad_worth_language(self):
        value = source_market_value_client.extract_market_value_eur_m(
            "Leeds United are heading back to the Premier League - "
            "with a squad worth an estimated €175.5m in the eyes of Transfermarkt."
        )
        self.assertAlmostEqual(value, 175.5)

    def test_market_value_parser_accepts_squad_value_stands_at_language(self):
        value = source_market_value_client.extract_market_value_eur_m(
            "Chelsea's current squad value stands at €1.16 billion - "
            "the sixth most valuable squad in the world."
        )
        self.assertAlmostEqual(value, 1160.0)

    def test_market_value_parser_accepts_transfermarkt_value_before_total_label(self):
        value = source_market_value_client.extract_market_value_eur_m(
            "Chelsea FC Premier League € 1.11 bn Total market value Squad size: 30 "
            "Average age: 23.7 ø-Market value: €37.00m"
        )
        self.assertAlmostEqual(value, 1110.0)

    def test_market_value_parser_does_not_treat_player_list_as_team_total(self):
        value = source_market_value_client.extract_market_value_eur_m(
            "# Player Market value 25 Moises Caicedo Defensive Midfield "
            "€100.00m 45 Romeo Lavia Defensive Midfield €22.00m"
        )
        self.assertEqual(value, 0.0)

    def test_market_value_parser_does_not_treat_average_value_as_team_total(self):
        value = source_market_value_client.extract_market_value_eur_m(
            "Squad size: 37 Average age: 24.2 ø-Market value: €30.00m"
        )
        self.assertEqual(value, 0.0)

    def test_transfermarkt_page_parser_requires_total_or_squad_value_context(self):
        player_list_html = (
            "<html><body># Player Market value 25 Moises Caicedo Defensive Midfield "
            "€100.00m 45 Romeo Lavia Defensive Midfield €22.00m</body></html>"
        )
        value, snippet = source_market_value_client._parse_transfermarkt_team_page(player_list_html, "Chelsea")
        self.assertEqual(value, 0.0)
        self.assertEqual(snippet, "")

    def test_anysearch_market_value_extracts_transfermarkt_total_from_url(self):
        search_output = """
## Search Results (1 results, 10ms)

### 1. Leeds United - Detailed squad 25/26 | Transfermarkt
- **URL**: https://www.transfermarkt.com/leeds-united/kader/verein/399/saison_id/2025
- This page displays a detailed overview of the club's current squad.
"""

        with (
            patch.object(source_market_value_client, "_run_anysearch_search", return_value=search_output),
            patch.object(
                source_market_value_client,
                "_run_anysearch_extract",
                return_value="Squad Leeds United Total market value: €175.50m Average age 25.7",
            ),
        ):
            value = source_market_value_client.fetch_team_market_value_via_anysearch("Leeds United", "英超")

        self.assertAlmostEqual(value.value_eur_m, 175.5)
        self.assertEqual(value.source_label, "AnySearch")
        self.assertIn("squad market value EUR 175.50m", value.summary)

    def test_anysearch_market_value_uses_partial_transfermarkt_sum_when_total_missing(self):
        search_output = """
## Search Results (1 results, 10ms)

### 1. Chelsea FC - Club profile - Transfermarkt
- **URL**: https://www.transfermarkt.us/fc-chelsea/startseite/verein/631
- # Player Market value 25 Moises Caicedo Defensive Midfield €100.00m 45 Romeo Lavia €22.00m
"""

        extracted = (
            "Chelsea FC current squad with market values "
            "Moises Caicedo €100.00m Enzo Fernandez €90.00m Romeo Lavia €22.00m "
            "Dario Essugo €15.00m Marc Cucurella €40.00m"
        )
        with (
            patch.object(source_market_value_client, "_run_anysearch_search", return_value=search_output),
            patch.object(source_market_value_client, "_run_anysearch_extract", return_value=extracted),
        ):
            value = source_market_value_client.fetch_team_market_value_via_anysearch("Chelsea", "英超")

        self.assertAlmostEqual(value.value_eur_m, 267.0)
        self.assertEqual(value.source_label, "AnySearch partial Transfermarkt")
        self.assertIn("source: AnySearch partial Transfermarkt", value.summary)

    def test_market_value_metrics_parse_and_feed_strength_gap(self):
        summary = (
            "Home FC: squad market value EUR 120.50m (source: Transfermarkt)\n"
            "Away FC: squad market value EUR 80.00m (source: Transfermarkt)\n"
            "market value gap: home-away EUR +40.50m; ratio 1.51x"
        )
        metrics = extract_market_value_metrics(summary)
        self.assertEqual(metrics["coverage"], 1)
        self.assertAlmostEqual(metrics["home_value_eur_m"], 120.5)
        self.assertAlmostEqual(metrics["away_value_eur_m"], 80.0)
        row = {
            "match_id": "mv-1",
            "issue": "1",
            "league": "Test League",
            "match_time": "2026-06-12 20:00",
            "home_team": "Home FC",
            "away_team": "Away FC",
            "list_odds_win": "2.10",
            "list_odds_draw": "3.20",
            "list_odds_loss": "3.40",
            "elo_home": "联赛 第5/20，30分，9胜3平6负，进28，失20，净胜8",
            "elo_away": "联赛 第5/20，30分，9胜3平6负，进28，失20，净胜8",
            "market_value_summary": summary,
            "recent_form_home": "近6场 3胜2平1负，进9球失5球，胜率50%，赢盘率50%，大球率33%",
            "recent_form_away": "近6场 3胜2平1负，进9球失5球，胜率50%，赢盘率50%，大球率33%",
            "home_away_form": "主场 3胜2平1负，进9失5 | 客场 3胜2平1负，进9失5",
            "head_to_head_summary": "",
            "injury_or_lineup_notes": "",
            "motivation_or_schedule_notes": "",
            "european_odds_movement_summary": "",
            "betting_heat_summary": "",
        }
        features = build_match_features(row)
        self.assertGreater(features["market_value_rating_gap"], 0)
        self.assertGreater(features["strength_gap"], 0)

    def test_unified_strategy_uses_anysearch_after_playwright_miss(self):
        calls = []

        def fake_market_value(match, analysis):
            return [], []

        def fake_supplement(match, analysis, missing_fields):
            return [], []

        def fake_playwright(field, match):
            calls.append(("playwright", field))
            return "", [], ""

        def fake_anysearch(field, match):
            calls.append(("anysearch", field))
            return "AnySearch fallback form summary", ["AnySearch: https://example.test/form"], "query"

        with (
            patch.object(collection_strategy, "_apply_market_value", side_effect=fake_market_value),
            patch.object(collection_strategy, "_apply_playwright_supplement", side_effect=fake_supplement),
            patch.object(collection_strategy, "_playwright_search_field", side_effect=fake_playwright),
            patch.object(collection_strategy, "_anysearch_field", side_effect=fake_anysearch),
        ):
            analysis = {
                "recent_form_home": "",
                "media_source_links": "",
                "collected_sources": "",
                "collection_quality_summary": "",
                "remarks": "",
            }
            collection_strategy.apply_unified_collection_strategy(
                {
                    "home_team": "Home FC",
                    "away_team": "Away FC",
                    "league": "Test League",
                    "match_time": "",
                },
                analysis,
                required_fields=["recent_form_home"],
            )

        self.assertEqual(analysis["recent_form_home"], "AnySearch fallback form summary")
        self.assertIn(("playwright", "recent_form_home"), calls)
        self.assertIn(("anysearch", "recent_form_home"), calls)
        self.assertIn("stage=anysearch", analysis["collection_quality_summary"])
        self.assertIn("https://example.test/form", analysis["media_source_links"])

    def test_unified_strategy_can_skip_external_fallbacks_for_batch_replay(self):
        analysis = {
            "recent_form_home": "",
            "media_source_links": "",
            "collected_sources": "",
            "collection_quality_summary": "",
            "remarks": "",
        }
        with (
            patch.dict(
                "os.environ",
                {"FOOTBALL_SKIP_EXTERNAL_SUPPLEMENT": "1"},
            ),
            patch.object(collection_strategy, "_apply_market_value") as mock_market,
            patch.object(collection_strategy, "_apply_playwright_supplement") as mock_supplement,
            patch.object(collection_strategy, "_playwright_search_field") as mock_playwright,
            patch.object(collection_strategy, "_anysearch_field") as mock_anysearch,
        ):
            collection_strategy.apply_unified_collection_strategy(
                {
                    "home_team": "Home FC",
                    "away_team": "Away FC",
                    "league": "Test League",
                    "match_time": "",
                },
                analysis,
                required_fields=["recent_form_home"],
            )

        mock_market.assert_not_called()
        mock_supplement.assert_not_called()
        mock_playwright.assert_not_called()
        mock_anysearch.assert_not_called()
        self.assertIn("status=missing", analysis["collection_quality_summary"])

    def test_quality_backfill_rewrites_field_rows_without_duplicates(self) -> None:
        self.insert_match()
        analysis = {
            "match_id": "M1",
            "collected_at": "2026-04-29 10:00:00",
            "media_source_links": "https://odds.500.com/fenxi/shuju-M1.shtml",
            "collected_sources": "500数据分析页",
            "collection_quality_summary": "\n".join(
                [
                    "strategy: stale",
                    "dim1_strength.recent_form_home: status=missing; stage=old; source=old; quality=0.00",
                ]
            ),
            "remarks": "",
        }
        analysis.update(
            {field: "ok" for field in collection_repository.REQUIRED_ANALYSIS_FIELDS}
        )
        analysis["market_value_summary"] = "market ok"
        collection_repository.save_analysis(analysis)

        stats = backfill_collection_quality.backfill_issue("20260429", force=True)
        row = collection_repository.get_match_analysis("M1")
        summary = str(row["collection_quality_summary"])

        self.assertEqual(stats["updated"], 1)
        self.assertIn("strategy: primary -> playwright-cli -> anysearch", summary)
        self.assertNotIn("stage=old", summary)
        self.assertEqual(summary.count("dim1_strength.recent_form_home:"), 1)

    def test_lineup_supplement_records_flashscore_failure_before_rotowire_fallback(self) -> None:
        class FakeBrowser:
            def close(self):
                pass

        rotowire_result = source_lineup_client.LineupSupplementResult(
            notes="rotowire notes",
            source_links=["https://rotowire.test/match"],
            source_labels=["RotoWire"],
            remarks=["伤停/阵容替补源命中：RotoWire（搜索引擎 bing）"],
            structured_data={"raw_summary": "team news"},
            lineup_found=True,
            injury_found=True,
        )

        with (
            patch.object(source_lineup_client, "extract_team_ids", return_value=[]),
            patch.object(source_lineup_client, "PlaywrightCliBrowser", return_value=FakeBrowser()),
            patch.object(source_lineup_client.PlaywrightCliSettings, "from_env", return_value=object()),
            patch.object(
                source_lineup_client,
                "discover_flashscore_match_url",
                side_effect=RuntimeError("flashscore timeout"),
            ),
            patch.object(
                source_lineup_client,
                "supplement_injury_or_lineup_notes_from_rotowire",
                return_value=rotowire_result,
            ),
        ):
            result = source_lineup_client.supplement_injury_or_lineup_notes(
                {
                    "match_time": "2026-04-29 20:00:00",
                    "home_team": "Home",
                    "away_team": "Away",
                },
                "<html></html>",
            )

        self.assertEqual(result.source_labels, ["RotoWire"])
        self.assertTrue(
            any("Flashscore lineup source failed: flashscore timeout" in item for item in result.remarks)
        )
        self.assertTrue(any("RotoWire" in item for item in result.remarks))

    def test_stats_treat_placeholder_in_any_required_field_as_failed(self) -> None:
        self.insert_match()
        analysis = {
            "match_id": "M1",
            "collected_at": "2026-04-29 10:00:00",
            "media_source_links": "",
            "collected_sources": "",
            "remarks": "",
        }
        analysis.update(
            {field: "ok" for field in collection_repository.REQUIRED_ANALYSIS_FIELDS}
        )
        analysis["recent_form_home"] = collection_repository.ANALYSIS_PLACEHOLDER_PATTERNS[0]
        collection_repository.save_analysis(analysis)

        row = collection_repository.get_match_analysis("M1")
        stats = collection_repository.get_collection_stats("20260429")

        self.assertEqual(row["collection_status"], "failed")
        self.assertEqual(stats["success_analyses"], 0)
        self.assertEqual(stats["failed_analyses"], 1)

    def test_one_side_without_public_injury_hit_does_not_fail_collection(self) -> None:
        self.insert_match()
        analysis = {
            "match_id": "M1",
            "collected_at": "2026-04-29 10:00:00",
            "media_source_links": "https://example.com/lineups",
            "collected_sources": "Flashscore",
            "remarks": "采集失败：缺少采集维度：维度二：近期动态/伤停/阵容",
        }
        analysis.update(
            {field: "ok" for field in collection_repository.REQUIRED_ANALYSIS_FIELDS}
        )
        analysis["injury_or_lineup_notes"] = "\n".join(
            [
                "预计首发",
                "主队：A / B / C / D / E / F / G / H / I / J / K",
                "客队：L / M / N / O / P / Q / R / S / T / U / V",
                "关键伤停：主队未在公开来源中命中明确伤停；客队Player X(膝部伤)",
                "来源：Flashscore",
            ]
        )
        collection_repository.save_analysis(analysis)

        row = collection_repository.get_match_analysis("M1")
        stats = collection_repository.get_collection_stats("20260429")

        self.assertEqual(row["collection_status"], "success")
        self.assertEqual(collection_service.get_missing_required_dimensions(row), [])
        self.assertEqual(collection_service.get_collection_failure_reason(row), "")
        self.assertEqual(stats["success_analyses"], 1)
        self.assertEqual(stats["failed_analyses"], 0)

    def test_collect_all_auto_retries_until_collection_succeeds(self) -> None:
        self.insert_match()
        failure = {
            "match_id": "M1",
            "collected_at": "2026-04-29 10:00:00",
            "collection_status": "failed",
            "remarks": "采集失败：缺少采集维度：维度二：近期动态/伤停/阵容",
        }
        failure.update({field: "" for field in collection_repository.REQUIRED_ANALYSIS_FIELDS})
        success = {
            "match_id": "M1",
            "collected_at": "2026-04-29 10:02:00",
            "collection_status": "success",
            "remarks": "",
        }
        success.update({field: "ok" for field in collection_repository.REQUIRED_ANALYSIS_FIELDS})

        with patch("collection_service.collect_match", side_effect=[failure, failure, success]) as mock_collect:
            result = collection_service.collect_all_matches("20260429", return_details=True)

        self.assertEqual(mock_collect.call_count, 3)
        self.assertTrue(mock_collect.call_args_list[0].kwargs["fast_mode"])
        self.assertNotIn("fast_mode", mock_collect.call_args_list[1].kwargs)
        self.assertNotIn("fast_mode", mock_collect.call_args_list[2].kwargs)
        self.assertEqual(result["collected_count"], 1)
        self.assertEqual(result["failed_count"], 0)
        self.assertEqual(result["results"][0]["auto_retry_count"], 2)

    def test_collect_all_auto_retries_fast_mode_optional_dimension_gaps(self) -> None:
        self.insert_match()
        fast_success = {
            "match_id": "M1",
            "collected_at": "2026-04-29 10:00:00",
            "collection_status": "success",
            "remarks": "",
            "market_value_summary": "",
        }
        fast_success.update(
            {field: "ok" for field in collection_repository.REQUIRED_ANALYSIS_FIELDS}
        )
        full_success = dict(fast_success)
        full_success["collected_at"] = "2026-04-29 10:02:00"
        full_success["market_value_summary"] = "Home €10.0m | Away €8.0m"

        with patch(
            "collection_service.collect_match",
            side_effect=[fast_success, full_success],
        ) as mock_collect:
            result = collection_service.collect_all_matches("20260429", return_details=True)

        self.assertEqual(mock_collect.call_count, 2)
        self.assertTrue(mock_collect.call_args_list[0].kwargs["fast_mode"])
        self.assertNotIn("fast_mode", mock_collect.call_args_list[1].kwargs)
        self.assertEqual(result["failed_count"], 0)
        self.assertEqual(result["results"][0]["auto_retry_count"], 1)
        self.assertEqual(
            result["results"][0]["market_value_summary"],
            "Home €10.0m | Away €8.0m",
        )

    def test_collect_all_marks_failure_after_two_auto_retries(self) -> None:
        self.insert_match()
        failure = {
            "match_id": "M1",
            "collected_at": "2026-04-29 10:00:00",
            "collection_status": "failed",
            "remarks": "采集失败：缺少采集维度：维度二：近期动态/伤停/阵容",
        }
        failure.update({field: "" for field in collection_repository.REQUIRED_ANALYSIS_FIELDS})

        with patch("collection_service.collect_match", return_value=failure) as mock_collect:
            result = collection_service.collect_all_matches("20260429", return_details=True)

        self.assertEqual(mock_collect.call_count, 3)
        self.assertTrue(mock_collect.call_args_list[0].kwargs["fast_mode"])
        self.assertNotIn("fast_mode", mock_collect.call_args_list[1].kwargs)
        self.assertNotIn("fast_mode", mock_collect.call_args_list[2].kwargs)
        self.assertEqual(result["failed_count"], 1)
        self.assertEqual(result["failed_matches"][0]["auto_retry_count"], 2)
        self.assertIn("自动补采 2 次仍失败", result["results"][0]["status_message"])

    def test_generic_failure_remark_after_previous_success_stays_failed(self) -> None:
        self.insert_match()
        analysis = {
            "match_id": "M1",
            "collected_at": "2026-04-29 10:00:00",
            "media_source_links": "https://example.com/lineups",
            "collected_sources": "Flashscore",
            "remarks": "",
        }
        analysis.update(
            {field: "ok" for field in collection_repository.REQUIRED_ANALYSIS_FIELDS}
        )
        collection_repository.save_analysis(analysis)
        collection_repository.save_failed_analysis(
            "M1",
            "采集失败：网络异常",
            "2026-04-29 10:05:00",
        )

        row = collection_repository.get_match_analysis("M1")
        stats = collection_repository.get_collection_stats("20260429")

        self.assertEqual(row["collection_status"], "failed")
        self.assertEqual(collection_service.get_collection_failure_reason(row), "采集失败：网络异常")
        self.assertEqual(stats["success_analyses"], 0)
        self.assertEqual(stats["failed_analyses"], 1)

    def test_collect_match_marks_missing_required_dimension_and_stats_failed(self) -> None:
        self.insert_match()
        lineup_result = SimpleNamespace(
            notes="\n".join(
                [
                    "预计首发",
                    "主队：A / B",
                    "客队：C / D",
                    "关键伤停：主队无；客队无",
                    "来源：Flashscore",
                ]
            ),
            source_links=["https://example.com/lineups"],
            source_labels=["Flashscore"],
            remarks=["伤停/阵容外部补采命中：Flashscore"],
            structured_data={"home_lineup": ["A"], "away_lineup": ["C"]},
        )

        with (
            patch("collection_service.fetch_html", return_value="<html></html>"),
            patch("collection_service.extract_recent_forms", return_value=("主队近况", "")),
            patch("collection_service.extract_h2h", return_value="双方近 3 次交锋主队 2 胜 1 平"),
            patch("collection_service.extract_strength_snapshots", return_value={
                "elo_home": "Test League 第3/16 20分",
                "elo_away": "Test League 第8/16 14分",
                "source_url": "https://liansai.500.com/zuqiu-1/jifen-1/",
                "source_label": "联赛积分",
                "note": "",
            }),
            patch("collection_service.extract_home_away_form", return_value="主场 6 胜2平2负 | 客场 3胜3平4负"),
            patch("collection_service.build_odds_summary", return_value="初赔 2.10/3.20/3.80 -> 即时 2.05/3.25/3.90"),
            patch("collection_service.build_heat_summary", return_value="投注比 胜40% 平30% 负30%"),
            patch("collection_service.supplement_injury_or_lineup_notes", return_value=lineup_result),
            patch("collection_service.apply_unified_collection_strategy", return_value=None),
            patch("collection_service.build_feature_snapshot", return_value=None),
        ):
            result = collection_service.collect_match("M1")

        self.assertEqual(result["status_level"], "warning")
        self.assertIn("维度一：基础实力/客队近期状态", result["missing_required_dimensions"])
        row = collection_repository.get_match_analysis("M1")
        self.assertEqual(row["collection_status"], "failed")
        self.assertTrue(str(row["remarks"]).startswith("采集失败：缺少采集维度"))
        stats = collection_repository.get_collection_stats("20260429")
        self.assertEqual(stats["success_analyses"], 0)
        self.assertEqual(stats["failed_analyses"], 1)

    def test_collect_match_fast_mode_skips_slow_external_supplements(self) -> None:
        self.insert_match()

        with (
            patch("collection_service.fetch_html", return_value="<html></html>"),
            patch("collection_service.extract_recent_forms", return_value=("主队近况", "")),
            patch("collection_service.extract_h2h", return_value="双方近 3 次交锋主队 2 胜 1 平"),
            patch("collection_service.extract_lineup_notes", return_value="预计首发已从 500 页面提取"),
            patch("collection_service.extract_strength_snapshots", return_value={
                "elo_home": "Test League 第3/16 20分",
                "elo_away": "Test League 第8/16 14分",
                "source_url": "https://liansai.500.com/zuqiu-1/jifen-1/",
                "source_label": "联赛积分",
                "note": "",
            }),
            patch("collection_service.extract_home_away_form", return_value="主场 6 胜2平2负 | 客场 3胜3平4负"),
            patch("collection_service.build_odds_summary", return_value="初赔 2.10/3.20/3.80 -> 即时 2.05/3.25/3.90"),
            patch("collection_service.build_asian_handicap_summary", return_value="亚盘主让平半"),
            patch("collection_service.build_heat_summary", return_value="投注比 胜40% 平30% 负30%"),
            patch("collection_service.supplement_injury_or_lineup_notes") as mock_lineup,
            patch("collection_service.apply_unified_collection_strategy", return_value=None) as mock_strategy,
            patch("collection_service._supplement_missing_dimensions") as mock_missing_supplement,
            patch("collection_service.build_feature_snapshot", return_value=None),
        ):
            result = collection_service.collect_match("M1", fast_mode=True)

        mock_lineup.assert_not_called()
        mock_missing_supplement.assert_not_called()
        self.assertFalse(mock_strategy.call_args.kwargs["allow_fallback"])
        self.assertEqual(result["collection_status"], "failed")
        self.assertIn("快速采集仍缺少维度", result["status_message"])

    def test_build_odds_summary_uses_company_row_medians_when_average_missing(self) -> None:
        match_row = {
            "list_odds_win": "2.90",
            "list_odds_draw": "3.20",
            "list_odds_loss": "2.40",
        }
        ouzhi_html = """
        <table id="datatb">
            <tr class="tr1">
                <td>1</td><td>Alpha</td>
                <td>2.80 3.05 2.25 3.35 3.15 1.95</td>
            </tr>
            <tr class="tr2">
                <td>2</td><td>Beta</td>
                <td>3.20 3.20 2.20 3.30 3.10 2.20</td>
            </tr>
            <tr><td>3.35</td><td>3.15</td><td>1.95</td></tr>
        </table>
        """

        summary = collection_service.build_odds_summary(match_row, ouzhi_html)

        self.assertEqual(
            summary,
            "初赔 3.00/3.12/2.23 -> 即时 3.33/3.12/2.08；变化 +0.33/+0.00/-0.15",
        )

    def test_predict_match_rejects_incomplete_collection(self) -> None:
        match_row = {
            "match_id": "M1",
            "issue": "20260429",
            "home_team": "Home",
            "away_team": "Away",
            "collected_at": "2026-04-29 10:00:00",
            "collection_status": "failed",
            "remarks": "采集失败：缺少采集维度：维度三：市场数据/投注热度",
        }
        with (
            patch("prediction_engine.init_db"),
            patch("prediction_engine.expire_pending_manual_reviews"),
            patch("prediction_engine.get_match_analysis", return_value=match_row),
        ):
            with self.assertRaisesRegex(RuntimeError, "缺少采集维度"):
                prediction_engine.predict_match("M1")

    def test_predict_issue_continues_after_skips_and_prediction_failures(self) -> None:
        rows = [
            {
                "match_id": "M1",
                "issue": "20260429",
                "home_team": "主队1",
                "away_team": "客队1",
                "collected_at": "2026-04-29 10:00:00",
                "collection_status": "success",
            },
            {
                "match_id": "M2",
                "issue": "20260429",
                "home_team": "主队2",
                "away_team": "客队2",
                "collected_at": "2026-04-29 10:01:00",
                "collection_status": "success",
            },
            {
                "match_id": "M3",
                "issue": "20260429",
                "home_team": "主队3",
                "away_team": "客队3",
                "collected_at": "",
                "collection_status": "uncollected",
            },
            {
                "match_id": "M4",
                "issue": "20260429",
                "home_team": "主队4",
                "away_team": "客队4",
                "collected_at": "2026-04-29 10:03:00",
                "collection_status": "failed",
                "remarks": "采集失败：缺少采集维度：维度一：基础实力/主队 Elo/实力代理",
            },
        ]

        def _fake_predict_match(match_id: str, ensure_collected: bool = False, progress_callback=None, **kwargs):
            if match_id == "M2":
                raise RuntimeError("模型计算失败")
            return {
                "match_id": match_id,
                "review_failed": False,
                "expert_review_failed": False,
                "manual_review_required": False,
                "status_level": "success",
                "status_message": "ok",
                "task_message": "ok",
            }

        with (
            patch("prediction_engine.init_db"),
            patch("prediction_engine.list_matches_by_issue", return_value=rows),
            patch("prediction_engine.predict_match", side_effect=_fake_predict_match),
        ):
            result = prediction_engine.predict_issue("20260429")

        self.assertEqual(result["predicted_count"], 1)
        self.assertEqual(result["skipped_count"], 2)
        self.assertEqual(result["prediction_failed_count"], 1)
        self.assertEqual(result["status_level"], "warning")
        self.assertIn("异常场次", result["task_message"])
        self.assertIn("缺少采集维度", result["skipped_matches"][1]["reason"])


class CollectionQualityWebTests(unittest.TestCase):
    def test_index_renders_collection_failure_badge(self) -> None:
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
            "collection_status": "failed",
            "remarks": "采集失败：缺少采集维度：维度三：市场数据/投注热度",
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
            patch.object(web_app_module, "get_collection_stats", return_value={"total_matches": 1, "success_analyses": 0, "failed_analyses": 1}),
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
        self.assertIn("采集异常", html)
        self.assertIn("缺少采集维度", html)


if __name__ == "__main__":
    unittest.main()
