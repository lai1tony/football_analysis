"""一键脚本：用新 blend 重跑过往所有期数的预测，再重新结算并跑诊断。

为什么需要：
    self_diagnose 的 [3/6] 段读的是 prediction_runs 表里 stored 概率。
    只有重新调用 predict_match 才会用新的 blend_predictions（含市场先验）
    生成新行。这个脚本把这个流程批量化。

什么不做：
    - 不改采集数据（matches/analyses 不动）
    - 不删旧 prediction_runs（保留历史，新行追加；canonical 自动漂移）
    - LLM 复核状态由 .env 的 LLM_REVIEW_ENABLED 决定（你在 /config 已关）

用法：
    python data/replay_all_issues.py
"""
from __future__ import annotations

import sys
import time

# 与 app.py 保持同样的扁平 import
import collection_repository as repo
from collection_service import collect_all_matches, init_db, sync_issue_matches
from prediction_engine import predict_issue, settle_issue_results


def _print_section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _collect_counts(result) -> tuple[int, int]:
    if isinstance(result, dict):
        collected = int(result.get("success_count", 0) or result.get("collected_count", 0) or 0)
        failed = int(result.get("failed_count", 0) or 0)
        return collected, failed
    if isinstance(result, list):
        collected = 0
        failed = 0
        for entry in result:
            if isinstance(entry, dict):
                status = str(entry.get("status") or entry.get("collection_status") or "").lower()
                if status in {"success", "ok", "collected", "done"} or entry.get("success") is True:
                    collected += 1
                elif status in {"failed", "error"} or entry.get("success") is False:
                    failed += 1
                else:
                    collected += 1
            else:
                collected += 1
        return collected, failed
    return 0, 0


def main() -> int:
    init_db()
    issues = repo.list_issues()
    if not issues:
        print("没有可重放的期号。先在 UI 同步对赛后再来。")
        return 1

    # 按期号正序（旧→新）跑，便于 prediction_runs.run_id 与时间一致
    issues_sorted = sorted(issues)

    _print_section(f"准备重放 {len(issues_sorted)} 期：{', '.join(issues_sorted)}")
    started_at = time.time()

    overall = {
        "collect_ok": 0,
        "collect_fail": 0,
        "predict_ok": 0,
        "predict_fail": 0,
        "settle_synced": 0,
        "settle_settled": 0,
        "settle_skipped": 0,
    }

    for idx, issue in enumerate(issues_sorted, start=1):
        _print_section(f"[{idx}/{len(issues_sorted)}] 期号 {issue}")

        # 0. 重新同步并采集，让新让球盘字段覆盖历史样本
        print("  → 重新同步并采集对赛（补采 500 亚盘初盘/即时盘）...")
        try:
            sync_issue_matches(issue)
            collect_result = collect_all_matches(issue)
            collected, failed = _collect_counts(collect_result)
            overall["collect_ok"] += collected
            overall["collect_fail"] += failed
            print(f"     ✓ 采集成功 {collected} 场，失败 {failed} 场")
        except Exception as exc:  # noqa: BLE001
            overall["collect_fail"] += 1
            print(f"     ✗ 采集整体失败: {exc}")

        # 1. 重新预测
        print("  → 重新预测全部对赛（含让球盘推荐）...")
        try:
            predict_result = predict_issue(issue, ensure_collected=False)
            ok_count = predict_result.get("predicted_count", 0) if isinstance(predict_result, dict) else 0
            failed_entries = (
                predict_result.get("prediction_failed_matches", [])
                if isinstance(predict_result, dict)
                else []
            )
            skipped_entries = (
                predict_result.get("skipped_matches", [])
                if isinstance(predict_result, dict)
                else []
            )
            overall["predict_ok"] += int(ok_count)
            overall["predict_fail"] += len(failed_entries) + len(skipped_entries)
            print(
                f"     ✓ 预测成功 {ok_count} 场，"
                f"失败 {len(failed_entries)} 场，跳过 {len(skipped_entries)} 场"
            )
            for entry in (failed_entries + skipped_entries)[:3]:
                label = entry.get("match_label", "")
                reason = entry.get("reason", "")
                print(f"       - {label}: {reason}")
        except Exception as exc:  # noqa: BLE001
            print(f"     ✗ 预测整体失败: {exc}")
            continue

        # 2. 重新结算
        # 此时 canonical run 已经漂移到新的 run_id，
        # save_feedback_log 内部会自动删除同 match 的旧 feedback。
        print("  → 重新结算（让 feedback_logs 绑定到新 run）...")
        try:
            settle_result = settle_issue_results(issue)
            synced = int(settle_result.get("result_synced_count", 0))
            settled = int(settle_result.get("settled_count", 0))
            skipped = int(settle_result.get("skipped_count", 0))
            overall["settle_synced"] += synced
            overall["settle_settled"] += settled
            overall["settle_skipped"] += skipped
            print(f"     ✓ 同步赛果 {synced} 场，反馈结算 {settled} 场，跳过 {skipped} 场")
            for entry in (settle_result.get("skipped_matches") or [])[:3]:
                label = entry.get("match_label", "")
                reason = entry.get("reason", "")
                print(f"       - {label}: {reason}")
        except Exception as exc:  # noqa: BLE001
            print(f"     ✗ 结算失败: {exc}")

    elapsed = time.time() - started_at

    _print_section("全部期数重放完成")
    print(f"  耗时:                {elapsed:.1f} 秒")
    print(f"  采集成功合计:        {overall['collect_ok']} 场")
    print(f"  采集失败合计:        {overall['collect_fail']} 场")
    print(f"  预测成功合计:        {overall['predict_ok']} 场")
    print(f"  预测失败/跳过合计:   {overall['predict_fail']} 场")
    print(f"  同步赛果合计:        {overall['settle_synced']} 场")
    print(f"  反馈结算合计:        {overall['settle_settled']} 场")
    print(f"  跳过结算合计:        {overall['settle_skipped']} 场")

    _print_section("接下来执行 self_diagnose")
    try:
        import self_diagnose

        self_diagnose.main()
    except Exception as exc:  # noqa: BLE001
        print(f"诊断执行失败: {exc}")
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
