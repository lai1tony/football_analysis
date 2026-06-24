import argparse
import json
import sys

from collection_repository import get_feedback_summary, list_matches_by_issue
from collection_service import collect_match, init_db
from prediction_engine import (
    backfill_predicted_scores,
    predict_match,
    record_feedback,
    settle_issue_results,
    summarize_backtest,
)


def _resolve_match_id(match_id: str | None, issue: str | None) -> str:
    if match_id:
        return match_id
    matches = list_matches_by_issue(issue)
    if not matches:
        raise RuntimeError("当前没有可预测的比赛，请先同步当前对赛并完成采集。")
    return matches[0]["match_id"]


def _print_summary(summary: dict) -> None:
    print("反馈学习汇总")
    print(f"- 总预测场次: {summary['total_predictions']}")
    print(f"- 命中场次: {summary['hit_predictions']}")
    print(f"- 命中率: {summary['hit_rate'] * 100:.1f}%")
    print(f"- 累计 ROI: {summary['total_roi']:.3f}")
    print(f"- 单场平均 ROI: {summary['avg_roi']:.3f}")


def _print_backtest(result: dict) -> None:
    if not result.get("total_settled"):
        print(result.get("message", "暂无已结算反馈样本。"))
        return

    print("独立下注模型回测")
    print(f"- 已结算样本: {result['total_settled']}")
    print(
        f"- 最终仲裁命中率: {result['recommendation_hit_rate'] * 100:.1f}% | "
        f"正 EV 占比: {result['positive_ev_share'] * 100:.1f}%"
    )
    print(
        f"- 最终仲裁 ROI: {result['total_roi']:.3f} | "
        f"最终仲裁单场平均 ROI: {result['avg_roi']:.3f}"
    )
    print(
        f"- 算法初判: 动作 {result['algorithm']['action_count']} 场 | "
        f"命中率 {result['algorithm']['hit_rate'] * 100:.1f}% | "
        f"ROI {result['algorithm']['total_roi']:.3f}"
    )
    print(
        f"- 最终仲裁: 动作 {result['final']['action_count']} 场 | "
        f"命中率 {result['final']['hit_rate'] * 100:.1f}% | "
        f"ROI {result['final']['total_roi']:.3f}"
    )
    current_policy = result.get("current_policy", {})
    if current_policy:
        print(
            f"- current_policy: action {current_policy['action_count']} | "
            f"hit {current_policy['hit_rate'] * 100:.1f}% | "
            f"ROI {current_policy['total_roi']:.3f}"
        )
    for label in ("market", "legacy", "independent"):
        bucket = result[label]
        print(
            f"- {label}: "
            f"Brier {bucket['brier_score']:.4f}, "
            f"LogLoss {bucket['log_loss']:.4f}, "
            f"Hit {bucket['hit_rate'] * 100:.1f}%"
        )
    for label in ("promote", "downgrade", "keep", "skipped", "failed"):
        bucket = result["review_buckets"].get(label)
        if not bucket:
            continue
        print(
            f"- {label}: 样本 {bucket['sample_count']} | "
            f"动作 {bucket['action_count']} | "
            f"命中率 {bucket['hit_rate'] * 100:.1f}% | "
            f"ROI {bucket['total_roi']:.3f}"
        )


def _print_settlement(result: dict) -> None:
    print("赛果同步与结算")
    print(f"- 期号: {result['issue'] or '-'}")
    print(f"- 总场次: {result['total_matches']}")
    print(f"- 同步赛果: {result['result_synced_count']}")
    print(f"- 完成结算: {result['settled_count']}")
    print(f"- 跳过场次: {result['skipped_count']}")
    for entry in result.get("skipped_matches", [])[:10]:
        print(f"  - {entry['match_label']}: {entry['reason']}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="足球独立下注模型 CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict_parser = subparsers.add_parser("predict", help="生成单场预测报告")
    predict_parser.add_argument("--match-id", default="", help="指定 match_id")
    predict_parser.add_argument("--issue", default="", help="指定期号，用于未传 match_id 时选择首场")
    predict_parser.add_argument("--collect", action="store_true", help="预测前强制重采该场比赛")
    predict_parser.add_argument("--json", action="store_true", help="输出 JSON")

    feedback_parser = subparsers.add_parser("feedback", help="手动补录或覆盖赛后反馈")
    feedback_parser.add_argument("--run-id", type=int, required=True, help="prediction_runs.run_id")
    feedback_parser.add_argument("--match-id", required=True, help="match_id")
    feedback_parser.add_argument("--actual-result", required=True, choices=["home", "draw", "away"])
    feedback_parser.add_argument("--actual-score", default="")
    feedback_parser.add_argument("--roi-delta", type=float, default=None)
    feedback_parser.add_argument("--notes", default="")

    summary_parser = subparsers.add_parser("summary", help="查看反馈学习汇总")
    summary_parser.add_argument("--json", action="store_true")

    settle_parser = subparsers.add_parser("settle", help="同步赛果并自动结算当前期")
    settle_parser.add_argument("--issue", default="", help="指定期号；默认使用当前期")
    settle_parser.add_argument("--json", action="store_true")

    backtest_parser = subparsers.add_parser("backtest", help="输出独立模型回测摘要")
    backtest_parser.add_argument("--league", default="")
    backtest_parser.add_argument("--month", default="", help="格式 YYYY-MM")
    backtest_parser.add_argument("--odds-min", type=float, default=None)
    backtest_parser.add_argument("--odds-max", type=float, default=None)
    backtest_parser.add_argument("--confidence-min", type=float, default=None)
    backtest_parser.add_argument("--ev-min", type=float, default=None)
    backtest_parser.add_argument("--json", action="store_true")

    score_parser = subparsers.add_parser("backfill-scores", help="用复核模型为历史预测回填比分")
    score_parser.add_argument("--issue", default="", help="指定期号；默认当前/全部筛选范围")
    score_parser.add_argument("--overwrite", action="store_true", help="覆盖已有预测比分")
    score_parser.add_argument("--limit", type=int, default=None, help="最多处理场次数")
    score_parser.add_argument("--json", action="store_true")

    args = parser.parse_args()
    init_db()

    if args.command == "predict":
        selected_match_id = _resolve_match_id(args.match_id or None, args.issue or None)
        if args.collect:
            collect_match(selected_match_id)
        result = predict_match(selected_match_id, ensure_collected=False)
        if args.json:
            print(
                json.dumps(
                    {
                        "run_id": result["run_id"],
                        "match_id": result["match_id"],
                        "snapshot": result["snapshot"],
                        "quality": result["quality"],
                        "algo_risk": result["algo_risk"],
                        "review": result["review"],
                        "risk": result["risk"],
                        "blended": result["blended"],
                        "score_prediction": result["score_prediction"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(result["report"])
        return

    if args.command == "feedback":
        result = record_feedback(
            prediction_run_id=args.run_id,
            match_id=args.match_id,
            actual_result=args.actual_result,
            actual_score=args.actual_score,
            roi_delta=args.roi_delta,
            notes=args.notes,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "summary":
        summary = get_feedback_summary()
        if args.json:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            _print_summary(summary)
        return

    if args.command == "settle":
        result = settle_issue_results(args.issue or None)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            _print_settlement(result)
        return

    if args.command == "backtest":
        result = summarize_backtest(
            league=args.league,
            month=args.month,
            odds_min=args.odds_min,
            odds_max=args.odds_max,
            confidence_min=args.confidence_min,
            ev_min=args.ev_min,
        )
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            _print_backtest(result)
        return

    if args.command == "backfill-scores":
        result = backfill_predicted_scores(
            args.issue or None,
            overwrite=args.overwrite,
            limit=args.limit,
        )
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(result["status_message"])
            for entry in result.get("failed_matches", [])[:10]:
                print(f"  - {entry['match_label']}: {entry['reason']}")


if __name__ == "__main__":
    main()
