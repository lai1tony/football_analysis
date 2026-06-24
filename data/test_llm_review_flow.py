from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import config_service
import prediction_engine


class FakeHTTPResponse:
    def __init__(self, payload: str) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return self.payload.encode("utf-8")

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class ConfigServiceTests(unittest.TestCase):
    def test_request_chat_falls_back_to_choice_text(self) -> None:
        payload = json.dumps({"choices": [{"text": "  fallback text  "}]}, ensure_ascii=False)
        with patch("config_service.urllib.request.urlopen", return_value=FakeHTTPResponse(payload)):
            response = config_service.request_openai_compatible_chat(
                "https://example.com/v1",
                "test-key",
                "test-model",
                messages=[{"role": "user", "content": "hello"}],
                require_non_empty_content=True,
            )
        self.assertEqual(response["content"], "fallback text")

    def test_request_chat_retries_empty_response(self) -> None:
        payload = json.dumps(
            {"choices": [{"message": {"content": "   "}, "finish_reason": "stop"}]},
            ensure_ascii=False,
        )
        call_count = 0

        def _fake_urlopen(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return FakeHTTPResponse(payload)

        with patch("config_service.urllib.request.urlopen", side_effect=_fake_urlopen):
            with self.assertRaises(config_service.ChatProtocolError) as ctx:
                config_service.request_openai_compatible_chat(
                    "https://example.com/v1",
                    "test-key",
                    "test-model",
                    messages=[{"role": "user", "content": "hello"}],
                    require_non_empty_content=True,
                    max_retries=1,
                    retry_backoff_seconds=0,
                )
        self.assertEqual(call_count, 2)
        self.assertEqual(ctx.exception.public_message, "模型返回空响应")

    def test_collection_api_rejects_invalid_review_responses(self) -> None:
        with self.subTest("non_json"):
            with patch(
                "config_service.request_openai_compatible_chat",
                return_value={"content": "not json"},
            ):
                result = config_service.test_collection_api("https://example.com/v1", "key", "model")
            self.assertFalse(result["success"])
            self.assertIn("非 JSON", result["message"])

        with self.subTest("empty"):
            with patch(
                "config_service.request_openai_compatible_chat",
                side_effect=config_service.ChatProtocolError(
                    "模型返回空响应",
                    "endpoint=https://example.com/v1/chat/completions",
                ),
            ):
                result = config_service.test_collection_api("https://example.com/v1", "key", "model")
            self.assertFalse(result["success"])
            self.assertIn("空响应", result["message"])

    def test_collection_api_uses_configurable_review_max_tokens(self) -> None:
        payload = json.dumps(
            {
                "decision": "keep",
                "target_action": "轻仓",
                "reason": "保持原动作",
                "risk_flags": [],
            },
            ensure_ascii=False,
        )
        with patch(
            "config_service.request_openai_compatible_chat",
            return_value={"content": payload},
        ) as mock_request:
            result = config_service.test_collection_api(
                "https://example.com/v1",
                "key",
                "model",
                max_tokens="1234",
            )

        self.assertTrue(result["success"])
        self.assertEqual(mock_request.call_args.kwargs["max_tokens"], 1234)


class PredictionEngineTests(unittest.TestCase):
    def _handicap_features(self, direction: int) -> dict:
        positive = direction > 0
        return {
            "asian_handicap": {
                "initial_line": 0.0,
                "current_line": 0.0,
                "initial_home_odds": 2.62,
                "current_home_odds": 2.60,
                "initial_away_odds": 2.05,
                "current_away_odds": 2.08,
            },
            "rating_gap": 150.0 * direction,
            "market_value": {"coverage": 1},
            "market_value_rating_gap": 110.0 * direction,
            "recent_home": {
                "points_per_game": 2.1 if positive else 0.8,
                "goal_diff_per_game": 1.1 if positive else -0.9,
            },
            "recent_away": {
                "points_per_game": 0.9 if positive else 2.0,
                "goal_diff_per_game": -0.8 if positive else 1.0,
            },
            "form_residual_gap": 0.8 * direction,
            "goal_diff_residual_gap": 0.9 * direction,
            "split": {
                "home_ppg": 2.2 if positive else 0.9,
                "away_ppg": 0.8 if positive else 2.1,
            },
            "lineup": {
                "data_available": 1,
                "home_availability": 0.95 if positive else 0.74,
                "away_availability": 0.75 if positive else 0.96,
            },
            "schedule": {
                "rest_advantage": 3.0 * direction,
                "schedule_gap": 2.0 * direction,
            },
            "h2h": {"edge": 0.5 * direction},
            "h2h_edge": 0.5 * direction,
            "xg": {
                "coverage": 1,
                "home_xg_per_game": 1.8 if positive else 0.9,
                "home_xga_per_game": 0.9 if positive else 1.7,
                "away_xg_per_game": 1.0 if positive else 1.9,
                "away_xga_per_game": 1.7 if positive else 0.8,
            },
            "market_probs": {
                "home": 0.52 if positive else 0.24,
                "draw": 0.24,
                "away": 0.24 if positive else 0.52,
            },
            "motivation_signal": 0.15 if positive else 0.0,
        }

    def test_handicap_recommendation_uses_explicit_dimension_evidence(self) -> None:
        quant = {"lambda_home": 1.34, "lambda_away": 1.10}
        quality = {"score": 0.50}
        positive = prediction_engine.evaluate_handicap_recommendation(
            features=self._handicap_features(1),
            quant=quant,
            quality=quality,
            row={"betting_heat_summary": "投注比 胜55% 平20% 负25%"},
        )
        negative = prediction_engine.evaluate_handicap_recommendation(
            features=self._handicap_features(-1),
            quant=quant,
            quality=quality,
            row={"betting_heat_summary": "投注比 胜25% 平20% 负55%"},
        )

        self.assertGreater(positive["dimension_score"], 0.55)
        self.assertLess(negative["dimension_score"], -0.55)
        self.assertGreater(
            prediction_engine.ACTION_LEVELS[positive["recommendation"]],
            prediction_engine.ACTION_LEVELS[negative["recommendation"]],
        )
        self.assertNotIn("盘口缺失", positive["reason"])
        self.assertIn("维度证据", positive["reason"])
        dimension_names = {item["name"] for item in positive["dimension_support"]}
        self.assertTrue(
            {
                "strength",
                "market_value",
                "recent_form",
                "goal_diff",
                "home_away",
                "lineup",
                "schedule",
                "h2h",
                "xg",
                "europe_market",
                "betting_heat",
                "asian_market",
                "motivation",
            }.issubset(dimension_names)
        )

    def test_parse_score_prediction_payload_accepts_ranked_score_json(self) -> None:
        payload = (
            '{"most_likely":{"score":"1-1","probability":0.13},'
            '"second_1":{"score":"1-0","probability":0.11},'
            '"second_2":{"score":"0-0","probability":0.10},'
            '"upset":{"score":"0-1","probability":0.06},'
            '"confidence_label":"中"}'
        )

        parsed = prediction_engine._parse_score_prediction_payload(payload)

        self.assertEqual(parsed["score"], "1-1")
        self.assertAlmostEqual(parsed["confidence"], 0.13)
        self.assertEqual(parsed["alternatives"], ["1-0", "0-0", "0-1"])
        self.assertEqual(parsed["reason"], "信心：中")

    def test_parse_score_prediction_payload_accepts_chinese_lines(self) -> None:
        payload = "\n".join(
            [
                "最可能比分：2-1 概率：18%",
                "次选比分一：1-1 概率：14%",
                "次选比分二：1-0 概率：12%",
                "冷门比分：0-1 概率：6%",
                "信心：中",
            ]
        )

        parsed = prediction_engine._parse_score_prediction_payload(payload)

        self.assertEqual(parsed["score"], "2-1")
        self.assertAlmostEqual(parsed["confidence"], 0.18)
        self.assertEqual(parsed["alternatives"], ["1-1", "1-0", "0-1"])
        self.assertEqual(parsed["reason"], "信心：中")

    def test_parse_review_payload_accepts_joint_decision_fields(self) -> None:
        raw = json.dumps(
            {
                "decision": "keep",
                "target_action": "主推",
                "outcome_decision": "challenge",
                "target_outcome": "home",
                "outcome_reason": "主胜概率更高",
                "confidence_delta": -0.04,
                "stake_multiplier": 0.65,
                "evidence_grade": "adequate",
                "reason": "证据够用但需降仓",
                "risk_flags": ["仓位保守"],
            },
            ensure_ascii=False,
        )

        parsed = prediction_engine._parse_review_payload(raw)

        self.assertEqual(parsed["decision"], "keep")
        self.assertEqual(parsed["confidence_delta"], -0.04)
        self.assertEqual(parsed["stake_multiplier"], 0.65)
        self.assertEqual(parsed["evidence_grade"], "adequate")
        self.assertEqual(parsed["outcome_decision"], "challenge")
        self.assertEqual(parsed["target_outcome"], "home")

    def test_parse_expert_review_payload_accepts_strict_json(self) -> None:
        raw = json.dumps(
            {
                "target_action": "轻仓",
                "reason": "高风险但证据仍可轻仓",
                "risk_flags": ["model_gap", "hot_market"],
                "evidence_grade": "adequate",
                "stake_multiplier": 0.6,
            },
            ensure_ascii=False,
        )

        parsed = prediction_engine._parse_expert_review_payload(raw, "home")

        self.assertEqual(parsed["target_action"], "轻仓")
        self.assertEqual(parsed["reason"], "高风险但证据仍可轻仓")
        self.assertEqual(parsed["risk_flags"], ["model_gap", "hot_market"])
        self.assertEqual(parsed["evidence_grade"], "adequate")
        self.assertEqual(parsed["stake_multiplier"], 0.6)
        self.assertFalse(parsed["direction_guarded"])

    def test_parse_expert_review_payload_guards_direction_changes(self) -> None:
        raw = json.dumps(
            {
                "target_action": "主推",
                "target_outcome": "away",
                "reason": "客胜方向更强",
                "risk_flags": ["direction_conflict"],
                "evidence_grade": "strong",
                "stake_multiplier": 1.0,
            },
            ensure_ascii=False,
        )

        parsed = prediction_engine._parse_expert_review_payload(raw, "home")

        self.assertEqual(parsed["target_action"], "观望")
        self.assertTrue(parsed["direction_guarded"])
        self.assertIn("改方向", parsed["reason"])

    def test_parse_expert_review_payload_requires_all_fields(self) -> None:
        raw = json.dumps(
            {
                "target_action": "观望",
                "reason": "证据不足",
                "risk_flags": [],
            },
            ensure_ascii=False,
        )

        with self.assertRaisesRegex(ValueError, "缺少专家终审字段"):
            prediction_engine._parse_expert_review_payload(raw, "home")

    def test_resolve_recommendation_vetoes_action_on_outcome_challenge(self) -> None:
        algo_risk = {
            "recommendation": "主推",
            "recommended_outcome": "away",
            "confidence": 0.76,
            "risk_level": "medium",
            "expected_values": {"home": 0.05, "draw": -0.10, "away": 0.20},
            "market_bias": {"home": 0.03, "draw": -0.02, "away": 0.08},
            "probabilities": {"home": 0.42, "draw": 0.24, "away": 0.34},
            "market_odds": {"home": 2.20, "draw": 3.30, "away": 4.10},
            "quality_score": 0.90,
            "model_agreement": 0.90,
            "action_score": 0.90,
            "legacy_gap": 0.02,
            "kelly_fraction": 0.08,
            "stake_pct": 1.0,
        }
        review = {
            "status": "completed",
            "decision": "keep",
            "target_action": "主推",
            "reason": "动作可保留",
            "risk_flags": [],
            "outcome_decision": "challenge",
            "target_outcome": "home",
            "outcome_reason": "主胜概率更高，方向冲突",
            "confidence_delta": 0.0,
            "stake_multiplier": 1.0,
            "evidence_grade": "adequate",
        }

        final_risk = prediction_engine.resolve_recommendation({"score": 0.90}, algo_risk, review)

        self.assertEqual(final_risk["recommended_outcome"], "away")
        self.assertEqual(final_risk["recommendation"], "观望")
        self.assertEqual(final_risk["stake_pct"], 0.0)
        self.assertIn("方向冲突", final_risk["resolution_reason"])

    def test_resolve_recommendation_treats_abstain_as_watch(self) -> None:
        algo_risk = {
            "recommendation": "主推",
            "recommended_outcome": "home",
            "confidence": 0.76,
            "risk_level": "medium",
            "expected_values": {"home": 0.20, "draw": -0.10, "away": -0.20},
            "market_bias": {"home": 0.08, "draw": -0.02, "away": -0.06},
            "probabilities": {"home": 0.52, "draw": 0.26, "away": 0.22},
            "market_odds": {"home": 2.40, "draw": 3.20, "away": 3.80},
            "quality_score": 0.90,
            "model_agreement": 0.90,
            "action_score": 0.90,
            "legacy_gap": 0.02,
            "kelly_fraction": 0.08,
            "stake_pct": 1.0,
        }
        review = {
            "status": "completed",
            "decision": "abstain",
            "target_action": "观望",
            "reason": "证据不足",
            "risk_flags": [],
        }

        final_risk = prediction_engine.resolve_recommendation({"score": 0.90}, algo_risk, review)

        self.assertEqual(final_risk["recommendation"], "观望")
        self.assertEqual(final_risk["stake_pct"], 0.0)
        self.assertIn("证据不足", final_risk["resolution_reason"])

    def test_resolve_recommendation_applies_confidence_and_stake_adjustments(self) -> None:
        algo_risk = {
            "recommendation": "主推",
            "recommended_outcome": "home",
            "confidence": 0.76,
            "risk_level": "medium",
            "expected_values": {"home": 0.20, "draw": -0.10, "away": -0.20},
            "market_bias": {"home": 0.08, "draw": -0.02, "away": -0.06},
            "probabilities": {"home": 0.52, "draw": 0.26, "away": 0.22},
            "market_odds": {"home": 2.40, "draw": 3.20, "away": 3.80},
            "quality_score": 0.90,
            "model_agreement": 0.90,
            "action_score": 0.90,
            "legacy_gap": 0.02,
            "kelly_fraction": 0.08,
            "stake_pct": 1.0,
        }
        review = {
            "status": "completed",
            "decision": "keep",
            "target_action": "主推",
            "reason": "方向可保留但仓位打折",
            "risk_flags": [],
            "confidence_delta": -0.08,
            "stake_multiplier": 0.50,
            "evidence_grade": "adequate",
        }

        final_risk = prediction_engine.resolve_recommendation({"score": 0.90}, algo_risk, review)

        self.assertEqual(final_risk["recommendation"], "主推")
        self.assertAlmostEqual(final_risk["confidence"], 0.68)
        self.assertGreater(final_risk["stake_pct"], 0.0)
        self.assertLessEqual(final_risk["stake_pct"], 0.50)
        self.assertEqual(final_risk["review_stake_multiplier"], 0.50)

    def test_review_failure_classifies_length_empty_response_as_truncated(self) -> None:
        exc = config_service.ChatProtocolError(
            "模型返回空响应",
            "endpoint=https://example.com/v1/chat/completions | finish_reason=length | response={...}",
        )

        reason, detail = prediction_engine._classify_review_failure(exc)

        self.assertEqual(reason, "复核模型输出被截断")
        self.assertIn("finish_reason=length", detail)

    def test_review_failure_classifies_partial_json_as_truncated(self) -> None:
        reason, detail = prediction_engine._classify_review_failure(
            ValueError("响应不是有效 JSON"),
            raw_text='{"decision":"keep","target_action":"主推"',
        )

        self.assertEqual(reason, "复核模型 JSON 被截断")
        self.assertIn("response=", detail)

    def test_predict_match_returns_warning_when_review_fails(self) -> None:
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
            "betting_heat_summary": "主胜偏热",
        }
        snapshot = {"snapshot_id": 7, "home_rating": 1510, "away_rating": 1485}
        features = {
            "feature_snapshot_id": 7,
            "market_odds": {"home": 2.1, "draw": 3.2, "away": 3.8},
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
        ml = {"probabilities": {"home": 0.5, "draw": 0.28, "away": 0.22}}
        legacy = {"probabilities": {"home": 0.47, "draw": 0.29, "away": 0.24}}
        quality = {"score": 0.84}
        blended = {"probabilities": {"home": 0.51, "draw": 0.28, "away": 0.21}, "agreement": 0.73}
        algo_risk = {
            "fair_odds": {"home": 1.96, "draw": 3.57, "away": 4.76},
            "market_probs": {"home": 0.45, "draw": 0.29, "away": 0.26},
            "expected_values": {"home": 0.08, "draw": -0.03, "away": -0.11},
            "market_bias": {"home": 0.06, "draw": -0.01, "away": -0.05},
            "confidence": 0.69,
            "risk_level": "medium",
            "recommended_outcome": "home",
            "recommendation": "轻仓",
            "stake_pct": 1.25,
            "warnings": ["热度偏高"],
        }
        review = {
            "enabled": 1,
            "status": "failed",
            "decision": "",
            "target_action": "轻仓",
            "reason": "复核模型返回空响应",
            "risk_flags": [],
            "raw": "endpoint=https://example.com/v1/chat/completions",
            "model_name": "review-model",
        }
        final_risk = dict(algo_risk)
        final_risk.update(
            {
                "resolution_reason": "复核模型返回空响应",
                "review_status_label": "失败",
            }
        )
        llm = {"provider": "openai-compatible", "model": "summary-model", "summary": "摘要正常"}

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
            patch("prediction_engine.blend_predictions", return_value=blended),
            patch("prediction_engine.get_active_learning_profile_config", return_value=None),
            patch("prediction_engine.run_risk_assessor", return_value=algo_risk),
            patch("prediction_engine.run_llm_recommendation_review", return_value=review),
            patch("prediction_engine.resolve_recommendation", return_value=final_risk),
            patch("prediction_engine.generate_llm_summary", return_value=llm),
            patch("prediction_engine.build_presenter_report", return_value="report"),
            patch("prediction_engine.save_prediction_run", return_value=88) as mock_save,
            patch("prediction_engine.supersede_pending_manual_reviews"),
        ):
            result = prediction_engine.predict_match("M1")

        self.assertTrue(result["review_failed"])
        self.assertEqual(result["status_level"], "warning")
        self.assertIn("回退算法初判", result["status_message"])
        saved_payload = mock_save.call_args.args[0]
        self.assertEqual(saved_payload["llm_review_status"], "failed")
        self.assertEqual(saved_payload["llm_review_reason"], "复核模型返回空响应")

    def test_predict_match_uses_expert_llm_when_arbiter_blocks_execution(self) -> None:
        match_row = {
            "match_id": "M9",
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
            "betting_heat_summary": "主胜偏热",
        }
        snapshot = {"snapshot_id": 7, "home_rating": 1510, "away_rating": 1485}
        features = {
            "feature_snapshot_id": 7,
            "market_odds": {"home": 2.1, "draw": 3.2, "away": 3.8},
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
        legacy = {"probabilities": {"home": 0.39, "draw": 0.33, "away": 0.28}}
        quality = {"score": 0.84}
        blended = {"probabilities": {"home": 0.51, "draw": 0.28, "away": 0.21}, "agreement": 0.73}
        algo_risk = {
            "fair_odds": {"home": 1.96, "draw": 3.57, "away": 4.76},
            "market_probs": {"home": 0.45, "draw": 0.29, "away": 0.26},
            "expected_values": {"home": 0.08, "draw": -0.03, "away": -0.11},
            "market_bias": {"home": 0.06, "draw": -0.01, "away": -0.05},
            "confidence": 0.69,
            "risk_level": "medium",
            "recommended_outcome": "home",
            "recommendation": "轻仓",
            "stake_pct": 1.25,
            "warnings": ["热度偏高"],
        }
        review = {
            "enabled": 1,
            "status": "completed",
            "decision": "keep",
            "target_action": "轻仓",
            "reason": "保持原动作",
            "risk_flags": [],
            "raw": "{}",
            "model_name": "review-model",
        }
        final_risk = dict(algo_risk)
        final_risk.update(
            {
                "resolution_reason": "一级复核保持原动作",
                "review_status_label": "保持",
            }
        )
        arbiter_review = {
            "enabled": 1,
            "triggered": 1,
            "status": "completed",
            "decision": "manual_review",
            "target_action": "轻仓",
            "reason": "新旧模型分歧较大，转专家终审",
            "risk_flags": ["model_gap"],
            "raw": "{}",
            "model_name": "arbiter-model",
            "trigger_reasons": ["模型分歧较大"],
        }
        expert_review = {
            "enabled": 1,
            "status": "completed",
            "target_action": "主推",
            "reason": "证据足够但仓位打折",
            "risk_flags": ["model_gap"],
            "evidence_grade": "strong",
            "stake_multiplier": 0.5,
            "raw": "{}",
            "model_name": "expert-model",
            "direction_guarded": False,
        }
        llm = {"provider": "openai-compatible", "model": "summary-model", "summary": "摘要正常"}

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
            patch("prediction_engine.blend_predictions", return_value=blended),
            patch("prediction_engine.get_active_learning_profile_config", return_value=None),
            patch("prediction_engine.run_risk_assessor", return_value=algo_risk),
            patch("prediction_engine.run_llm_recommendation_review", return_value=review),
            patch("prediction_engine.resolve_recommendation", return_value=final_risk),
            patch("prediction_engine.run_llm_arbiter_review", return_value=arbiter_review),
            patch("prediction_engine.run_expert_llm_final_review", return_value=expert_review) as mock_expert,
            patch("prediction_engine.generate_llm_summary", return_value=llm),
            patch("prediction_engine.build_presenter_report", return_value="report"),
            patch("prediction_engine.save_prediction_run", return_value=99) as mock_save,
            patch("prediction_engine.supersede_pending_manual_reviews"),
        ):
            result = prediction_engine.predict_match("M9")

        saved_payload = mock_save.call_args.args[0]
        self.assertTrue(result["arbiter_triggered"])
        self.assertTrue(mock_expert.called)
        self.assertFalse(result["manual_review_required"])
        self.assertEqual(result["execution_status"], "expert_review_resolved")
        self.assertEqual(saved_payload["manual_review_status"], "resolved")
        self.assertEqual(saved_payload["manual_review_reason"], "新旧模型分歧较大，转专家终审")
        self.assertEqual(saved_payload["effective_recommendation"], "主推")
        self.assertGreater(saved_payload["effective_stake_pct"], 0.0)
        self.assertLessEqual(saved_payload["effective_stake_pct"], 0.5)
        self.assertEqual(saved_payload["effective_action_source"], "expert_llm")
        self.assertIn("专家终审", saved_payload["manual_review_notes"])
        self.assertIn("证据足够", saved_payload["manual_review_notes"])
        self.assertEqual(saved_payload["arbiter_review_decision"], "manual_review")
        self.assertEqual(result["status_level"], "warning")

    def test_expert_llm_unconfigured_fails_closed_without_pending(self) -> None:
        final_risk = {
            "recommendation": "主推",
            "recommended_outcome": "home",
            "stake_pct": 1.0,
            "confidence": 0.70,
            "risk_level": "medium",
        }
        arbiter_review = {
            "status": "completed",
            "decision": "manual_review",
            "target_action": "主推",
            "reason": "高风险转专家终审",
        }

        with patch.dict(
            "os.environ",
            {"COLLECTION_BASE_URL": "", "COLLECTION_APIKEY": "", "COLLECTION_MODEL": ""},
            clear=False,
        ):
            expert_review = prediction_engine.run_expert_llm_final_review(
                {},
                {},
                {},
                {},
                {},
                final_risk,
                {},
                arbiter_review,
                [],
            )

        outcome = prediction_engine._resolve_execution_outcome(
            final_risk,
            arbiter_review,
            requested_at="2026-04-28 12:00:00",
            expert_review=expert_review,
        )

        self.assertEqual(expert_review["status"], "skipped")
        self.assertEqual(outcome["manual_review_status"], "resolved")
        self.assertEqual(outcome["effective_action_source"], "expert_llm_failed")
        self.assertEqual(outcome["effective_recommendation"], "观望")
        self.assertEqual(outcome["effective_stake_pct"], 0.0)
        self.assertEqual(outcome["execution_status"], "expert_review_failed")

    def test_expert_llm_invalid_json_fails_closed_without_pending(self) -> None:
        row = {
            "home_team": "主队",
            "away_team": "客队",
            "league": "测试联赛",
            "match_time": "2026-04-28 20:00:00",
        }
        features = {
            "lineup": {
                "home_availability": 0.95,
                "away_availability": 0.92,
                "home_absent_count": 1,
                "home_doubtful_count": 0,
                "away_absent_count": 1,
                "away_doubtful_count": 1,
            },
            "schedule": {
                "home_rest_days": 6,
                "away_rest_days": 5,
                "home_load_14": 2,
                "away_load_14": 3,
            },
        }
        final_risk = {
            "recommendation": "轻仓",
            "recommended_outcome": "home",
            "stake_pct": 0.45,
            "confidence": 0.70,
            "risk_level": "medium",
            "action_score": 0.72,
            "ev_margin": 0.08,
            "legacy_gap": 0.02,
            "market_probs": {"home": 0.45, "draw": 0.29, "away": 0.26},
            "market_bias": {"home": 0.06, "draw": -0.01, "away": -0.05},
            "expected_values": {"home": 0.08, "draw": -0.03, "away": -0.11},
            "warnings": ["模型分歧"],
        }
        arbiter_review = {
            "status": "completed",
            "decision": "manual_review",
            "target_action": "轻仓",
            "reason": "高风险转专家终审",
        }

        with (
            patch.dict(
                "os.environ",
                {
                    "COLLECTION_BASE_URL": "https://example.com/v1",
                    "COLLECTION_APIKEY": "key",
                    "COLLECTION_MODEL": "expert-model",
                },
                clear=False,
            ),
            patch(
                "prediction_engine._request_review_response",
                return_value={"content": "not json", "endpoint": "https://example.com/v1/chat/completions"},
            ),
        ):
            expert_review = prediction_engine.run_expert_llm_final_review(
                row,
                features,
                {"home_rating": 1510, "away_rating": 1485},
                {"probabilities": {"home": 0.51, "draw": 0.28, "away": 0.21}},
                {"score": 0.84},
                final_risk,
                {"status": "completed", "decision": "keep", "target_action": "轻仓", "reason": "保持原动作"},
                arbiter_review,
                ["模型分歧较大"],
            )

        outcome = prediction_engine._resolve_execution_outcome(
            final_risk,
            arbiter_review,
            requested_at="2026-04-28 12:00:00",
            expert_review=expert_review,
        )

        self.assertEqual(expert_review["status"], "failed")
        self.assertIn("JSON", expert_review["reason"])
        self.assertEqual(outcome["manual_review_status"], "resolved")
        self.assertEqual(outcome["effective_action_source"], "expert_llm_failed")
        self.assertEqual(outcome["effective_recommendation"], "观望")
        self.assertEqual(outcome["effective_stake_pct"], 0.0)

    def test_score_prediction_repairs_truncated_reasoning_response(self) -> None:
        calls = []

        def fake_score_response(*args, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return {
                    "endpoint": "https://example.com/v1/chat/completions",
                    "content": "",
                    "response": {
                        "choices": [
                            {
                                "finish_reason": "length",
                                "message": {
                                    "content": "",
                                    "reasoning_content": "Canada edge is small; 1-1 remains plausible.",
                                },
                            }
                        ]
                    },
                }
            return {
                "endpoint": "https://example.com/v1/chat/completions",
                "content": '{"score":"1-1","confidence":0.13,"reason":"走势接近平局","alternatives":["1-0","0-0"]}',
                "response": {},
            }

        row = {
            "match_id": "M-score",
            "issue": "26085",
            "home_team": "加拿大",
            "away_team": "波黑",
            "league": "世界杯",
            "match_time": "06-13 03:00",
            "market_value_summary": "",
            "recent_form_home": "",
            "recent_form_away": "",
            "home_away_form": "",
            "head_to_head_summary": "",
            "injury_or_lineup_notes": "",
            "motivation_or_schedule_notes": "",
            "european_odds_movement_summary": "",
            "betting_heat_summary": "",
            "collected_sources": "",
            "collection_quality_summary": "",
        }
        features = {
            "lineup": {
                "home_availability": 0.92,
                "away_availability": 0.92,
                "home_absent_count": 0,
                "home_doubtful_count": 0,
                "away_absent_count": 0,
                "away_doubtful_count": 0,
            },
            "schedule": {
                "home_rest_days": 5,
                "away_rest_days": 5,
                "home_load_14": 1,
                "away_load_14": 1,
            },
            "recent_home": {"points_per_game": 1.4, "goals_for_per_game": 1.1, "goals_against_per_game": 1.0},
            "recent_away": {"points_per_game": 1.2, "goals_for_per_game": 1.0, "goals_against_per_game": 1.1},
            "split": {"home_ppg": 1.5, "away_ppg": 1.1},
            "xg": {},
            "market_value": {},
            "market_odds": {"home": 1.93, "draw": 3.7, "away": 4.75},
            "rating_gap": 20,
            "h2h_edge": 0.0,
        }
        quant = {
            "top_scores": [((1, 1), 0.1393), ((1, 0), 0.1133), ((0, 0), 0.1051)],
            "probabilities": {"home": 0.46, "draw": 0.27, "away": 0.26},
            "lambda_home": 1.2,
            "lambda_away": 1.0,
            "over_25": 0.44,
            "under_25": 0.56,
        }
        ml = {"probabilities": {"home": 0.45, "draw": 0.28, "away": 0.27}}
        legacy = {"probabilities": {"home": 0.52, "draw": 0.27, "away": 0.21}}
        blended = {"probabilities": {"home": 0.46, "draw": 0.27, "away": 0.26}}
        quality = {"score": 0.82}
        final_risk = {
            "recommended_outcome": "home",
            "recommendation": "观望",
            "risk_level": "medium",
            "stake_pct": 0.0,
            "market_probs": {"home": 0.52, "draw": 0.27, "away": 0.21},
            "expected_values": {"home": -0.12, "draw": -0.02, "away": 0.18},
        }

        with (
            patch.dict(
                "os.environ",
                {
                    "COLLECTION_BASE_URL": "https://example.com/v1",
                    "COLLECTION_APIKEY": "key",
                    "COLLECTION_MODEL": "score-model",
                    "LLM_REVIEW_ENABLED": "true",
                },
                clear=False,
            ),
            patch("prediction_engine._request_score_prediction_response", side_effect=fake_score_response),
        ):
            score_prediction = prediction_engine.run_llm_score_prediction(
                row,
                features,
                {"home_rating": 1510, "away_rating": 1490},
                quant,
                ml,
                legacy,
                blended,
                quality,
                final_risk,
            )

        self.assertEqual(score_prediction["status"], "completed")
        self.assertEqual(score_prediction["score"], "1-1")
        self.assertEqual(len(calls), 2)

    def test_predict_issue_summarizes_review_failures_and_skips(self) -> None:
        rows = [
            {
                "match_id": "M1",
                "issue": "20260428",
                "home_team": "主队1",
                "away_team": "客队1",
                "collected_at": "2026-04-28 10:00:00",
            },
            {
                "match_id": "M2",
                "issue": "20260428",
                "home_team": "主队2",
                "away_team": "客队2",
                "collected_at": "2026-04-28 10:05:00",
            },
            {
                "match_id": "M3",
                "issue": "20260428",
                "home_team": "主队3",
                "away_team": "客队3",
                "collected_at": "",
            },
        ]
        progress_calls: list[dict[str, object]] = []

        def _fake_predict_match(match_id: str, ensure_collected: bool = False, progress_callback=None):
            if progress_callback is not None:
                progress_callback(current_step="单场处理中", message=f"{match_id} processing")
            if match_id == "M2":
                return {
                    "match_id": match_id,
                    "review_failed": True,
                    "review_failure_reason": "复核模型返回空响应",
                    "status_level": "warning",
                    "status_message": "warning",
                    "task_message": "warning",
                }
            return {
                "match_id": match_id,
                "review_failed": False,
                "review_failure_reason": "",
                "status_level": "success",
                "status_message": "ok",
                "task_message": "ok",
            }

        def _progress_callback(**payload):
            progress_calls.append(payload)

        with (
            patch("prediction_engine.init_db"),
            patch("prediction_engine.list_matches_by_issue", return_value=rows),
            patch("prediction_engine.predict_match", side_effect=_fake_predict_match),
        ):
            result = prediction_engine.predict_issue("20260428", progress_callback=_progress_callback)

        self.assertEqual(result["predicted_count"], 2)
        self.assertEqual(result["skipped_count"], 1)
        self.assertEqual(result["review_failed_count"], 1)
        self.assertEqual(result["status_level"], "warning")
        self.assertIn("1 场 LLM 复核失败", result["task_message"])
        self.assertTrue(any(call.get("current_step") == "当前场次跳过" for call in progress_calls))

    def test_predict_issue_summarizes_expert_review_failures(self) -> None:
        rows = [
            {
                "match_id": "M1",
                "issue": "20260428",
                "home_team": "主队1",
                "away_team": "客队1",
                "collected_at": "2026-04-28 10:00:00",
            },
            {
                "match_id": "M2",
                "issue": "20260428",
                "home_team": "主队2",
                "away_team": "客队2",
                "collected_at": "2026-04-28 10:05:00",
            },
        ]

        def _fake_predict_match(match_id: str, ensure_collected: bool = False, progress_callback=None):
            if match_id == "M2":
                return {
                    "match_id": match_id,
                    "review_failed": False,
                    "expert_review_failed": True,
                    "expert_review": {"reason": "未配置专家终审模型，自动保守观望。"},
                    "manual_review_required": False,
                    "status_level": "warning",
                    "status_message": "warning",
                    "task_message": "warning",
                }
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
            result = prediction_engine.predict_issue("20260428")

        self.assertEqual(result["predicted_count"], 2)
        self.assertEqual(result["manual_review_count"], 0)
        self.assertEqual(result["expert_review_failed_count"], 1)
        self.assertEqual(result["expert_review_failed_matches"][0]["reason"], "未配置专家终审模型，自动保守观望。")
        self.assertIn("1 场专家终审失败并自动观望", result["task_message"])


if __name__ == "__main__":
    unittest.main()
