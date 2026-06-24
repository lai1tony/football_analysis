from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime
from pathlib import Path
import shutil
import sqlite3
from typing import Any, Mapping

import collection_repository as repo
from collection_service import get_collection_failure_reason, sync_issue_matches
import learning_engine
from prediction_engine import (
    _resolve_canonical_prediction_run,
    _settle_handicap_result,
    predict_issue,
    settle_issue_results,
)


PROFILE_ID = 43
OLD_PROFILE43_BASELINE = {
    "sample_count": 1245,
    "action_count": 753,
    "hit_count": 544,
}


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _issue_sort_key(issue: str) -> tuple[int, str]:
    text = str(issue or "")
    try:
        return (int(text), text)
    except ValueError:
        return (0, text)


def _row_value(row: Mapping[str, Any], key: str, default: Any = "") -> Any:
    try:
        value = row[key]
    except Exception:  # noqa: BLE001
        return default
    return default if value is None else value


def _pct(numerator: int, denominator: int) -> float:
    return (numerator / denominator) if denominator else 0.0


def _backup_database() -> Path:
    backup_dir = repo.BASE_DIR / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"football_data_before_profile43_replay_{_now_stamp()}.db"
    shutil.copy2(repo.PRIMARY_DB_PATH, backup_path)
    return backup_path


def _active_profile_strategy(profile_id: int = PROFILE_ID) -> dict[str, Any]:
    profile = learning_engine.get_active_learning_profile_config()
    if profile is None:
        raise RuntimeError("当前没有启用中的学习配置，无法执行 #43 重统。")
    active_id = int(profile.get("learning_profile_id", 0) or 0)
    if active_id != int(profile_id):
        raise RuntimeError(f"当前启用配置为 #{active_id}，不是要求的 #{profile_id}。")
    strategy = profile.get("strategy_params", {})
    if not isinstance(strategy, Mapping) or str(strategy.get("strategy_kind", "")) != "handicap_bucket_table":
        raise RuntimeError("启用配置 #43 不是 handicap_bucket_table 策略，无法按计划重统。")
    return dict(strategy)


def _fetch_scalar(conn: sqlite3.Connection, query: str, params: tuple[Any, ...] = ()) -> int:
    row = conn.execute(query, params).fetchone()
    return int(row[0] or 0) if row else 0


def audit_database_before(profile_id: int = PROFILE_ID) -> list[str]:
    findings: list[str] = []
    strategy = _active_profile_strategy(profile_id)
    findings.append(
        f"启用配置 #{profile_id} 已确认，策略类型 {strategy.get('strategy_kind')}，bucket_count={strategy.get('bucket_count', 0)}。"
    )
    with closing(repo.get_connection()) as conn:
        duplicate_feedback = _fetch_scalar(
            conn,
            """
            SELECT COUNT(*)
            FROM (
                SELECT match_id
                FROM feedback_logs
                GROUP BY match_id
                HAVING COUNT(*) > 1
            )
            """,
        )
        duplicate_runs = _fetch_scalar(
            conn,
            """
            SELECT COUNT(*)
            FROM (
                SELECT match_id
                FROM prediction_runs
                GROUP BY match_id
                HAVING COUNT(*) > 1
            )
            """,
        )
        orphan_runs = _fetch_scalar(
            conn,
            """
            SELECT COUNT(*)
            FROM prediction_runs p
            LEFT JOIN matches m ON m.match_id = p.match_id
            WHERE m.match_id IS NULL
            """,
        )
        orphan_feedback = _fetch_scalar(
            conn,
            """
            SELECT COUNT(*)
            FROM feedback_logs f
            LEFT JOIN prediction_runs p ON p.run_id = f.prediction_run_id
            LEFT JOIN matches m ON m.match_id = f.match_id
            WHERE p.run_id IS NULL OR m.match_id IS NULL
            """,
        )
        blank_issues = _fetch_scalar(
            conn,
            "SELECT COUNT(*) FROM matches WHERE TRIM(IFNULL(issue, '')) = ''",
        )
    if duplicate_feedback:
        findings.append(f"修复计划：发现 {duplicate_feedback} 个 match 存在多条 feedback，先运行 init_db 去重后重查。")
        repo.init_db()
    if duplicate_runs:
        findings.append(f"修复计划：发现 {duplicate_runs} 个 match 存在多条 prediction_runs，重放预测会按 match 覆盖为最新 canonical run。")
    if orphan_runs:
        findings.append(f"数据错误：发现 {orphan_runs} 条 prediction_runs 缺少 matches 父记录，相关样本不纳入最终有效统计。")
    if orphan_feedback:
        findings.append(f"数据错误：发现 {orphan_feedback} 条 feedback_logs 缺少 run 或 match 父记录，相关样本不纳入最终有效统计。")
    if blank_issues:
        findings.append(f"数据错误：发现 {blank_issues} 场比赛缺少期号，无法进入期号筛选逐期重统。")
    if not any(item.startswith(("修复计划", "数据错误")) for item in findings):
        findings.append("执行前数据链审核未发现阻断性结构问题。")
    return findings


def _rows_for_issue(issue: str) -> list[Mapping[str, Any]]:
    return [
        row
        for row in repo.list_backtest_rows(limit=None)
        if str(_row_value(row, "issue", "") or "") == str(issue)
    ]


def _strategy_action_for_row(row: Mapping[str, Any], strategy: Mapping[str, Any]) -> tuple[str, str]:
    features = [str(item) for item in strategy.get("features", []) if str(item or "").strip()]
    buckets = strategy.get("buckets", {})
    bucket = buckets.get(learning_engine._handicap_bucket_key(row, features)) if isinstance(buckets, Mapping) else None
    side = str(bucket.get("side", "") or "") if isinstance(bucket, Mapping) else ""
    action = str(strategy.get("action", "轻仓") or "轻仓") if side in {"home", "away"} else "观望"
    return action, side


def _audit_issue_chain(issue: str, strategy: Mapping[str, Any]) -> dict[str, Any]:
    findings: list[str] = []
    invalid_match_ids: set[str] = set()
    matches = repo.list_matches_by_issue(issue)
    for row in matches:
        match_id = str(_row_value(row, "match_id", "") or "")
        failure_reason = get_collection_failure_reason(row)
        if failure_reason:
            findings.append(f"{match_id} 采集异常：{failure_reason}")
        canonical_run, canonical_reason = _resolve_canonical_prediction_run(row)
        if canonical_run is None:
            invalid_match_ids.add(match_id)
            findings.append(f"{match_id} 无法定位赛前 canonical run：{canonical_reason}")
    rows = _rows_for_issue(issue)
    for row in rows:
        match_id = str(_row_value(row, "match_id", "") or "")
        match_row = repo.get_match_analysis(match_id)
        if match_row is not None:
            canonical_run, canonical_reason = _resolve_canonical_prediction_run(match_row)
            canonical_run_id = int(_row_value(canonical_run, "run_id", 0) or 0) if canonical_run is not None else 0
            feedback_run_id = int(_row_value(row, "prediction_run_id", 0) or 0)
            if canonical_run_id and feedback_run_id != canonical_run_id:
                invalid_match_ids.add(match_id)
                findings.append(
                    f"{match_id} feedback 未绑定赛前 canonical run：feedback={feedback_run_id}，canonical={canonical_run_id}"
                )
            elif canonical_run is None:
                invalid_match_ids.add(match_id)
                findings.append(f"{match_id} feedback 无法复核 canonical run：{canonical_reason}")
        expected_handicap = _settle_handicap_result(
            str(_row_value(row, "actual_score", "") or ""),
            float(_row_value(row, "handicap_line", 0.0) or 0.0),
        )
        stored_handicap = str(_row_value(row, "handicap_actual_result", "") or "")
        if expected_handicap and stored_handicap and expected_handicap != stored_handicap:
            invalid_match_ids.add(match_id)
            findings.append(
                f"{match_id} 让球结算不一致：score+line={expected_handicap}，stored={stored_handicap}"
            )
        expected_action, expected_side = _strategy_action_for_row(row, strategy)
        stored_action = learning_engine._normalized_action(_row_value(row, "handicap_recommendation", ""))
        stored_side = str(_row_value(row, "handicap_recommended_side", "") or "")
        if stored_action != learning_engine._normalized_action(expected_action) or stored_side != expected_side:
            invalid_match_ids.add(match_id)
            findings.append(
                f"{match_id} #43 策略动作复算不一致：expected={expected_action}/{expected_side or '-'}，"
                f"stored={stored_action}/{stored_side or '-'}"
            )
    valid_rows = [row for row in rows if str(_row_value(row, "match_id", "") or "") not in invalid_match_ids]
    metrics = learning_engine._handicap_bucket_strategy_metrics(valid_rows, strategy)
    return {
        "issue": issue,
        "match_count": len(matches),
        "settled_rows": len(rows),
        "valid_rows": len(valid_rows),
        "invalid_rows": len(invalid_match_ids),
        "metrics": metrics,
        "findings": findings,
    }


def _combine_issue_metrics(issue_results: list[dict[str, Any]]) -> dict[str, int | float]:
    sample_count = sum(int(item["metrics"].get("sample_count", 0) or 0) for item in issue_results)
    action_count = sum(int(item["metrics"].get("action_count", 0) or 0) for item in issue_results)
    hit_count = sum(int(item["metrics"].get("hit_count", 0) or 0) for item in issue_results)
    return {
        "sample_count": sample_count,
        "action_count": action_count,
        "hit_count": hit_count,
        "watch_count": max(sample_count - action_count, 0),
        "action_share": _pct(action_count, sample_count),
        "hit_rate": _pct(hit_count, action_count),
    }


def _write_report(
    *,
    report_path: Path,
    backup_path: Path | None,
    pre_audit: list[str],
    issue_results: list[dict[str, Any]],
    totals: Mapping[str, Any],
) -> None:
    lines = [
        "# Profile #43 Replay Audit",
        "",
        f"- Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Database backup: {backup_path if backup_path else 'not created'}",
        f"- Old saved baseline: samples={OLD_PROFILE43_BASELINE['sample_count']}, actions={OLD_PROFILE43_BASELINE['action_count']}, hits={OLD_PROFILE43_BASELINE['hit_count']}",
        f"- Strict replay totals: samples={totals['sample_count']}, actions={totals['action_count']}, hits={totals['hit_count']}",
        f"- Strict action share: {totals['action_share'] * 100:.2f}%",
        f"- Strict hit rate: {totals['hit_rate'] * 100:.2f}%",
        f"- Target status: {'PASS' if totals['action_share'] >= 0.60 and totals['hit_rate'] >= 0.70 else 'FAIL'}",
        "",
        "## Pre-run Audit",
        "",
    ]
    lines.extend(f"- {item}" for item in pre_audit)
    lines.extend(["", "## Issue Results", "", "| Issue | Matches | Settled | Valid | Actions | Hits | Action Share | Hit Rate | Findings |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for item in issue_results:
        metrics = item["metrics"]
        findings = len(item["findings"])
        lines.append(
            f"| {item['issue']} | {item['match_count']} | {item['settled_rows']} | {item['valid_rows']} | "
            f"{int(metrics.get('action_count', 0) or 0)} | {int(metrics.get('hit_count', 0) or 0)} | "
            f"{float(metrics.get('action_share', 0.0) or 0.0) * 100:.2f}% | "
            f"{float(metrics.get('hit_rate', 0.0) or 0.0) * 100:.2f}% | {findings} |"
        )
    findings = [
        f"{item['issue']} {finding}"
        for item in issue_results
        for finding in item["findings"]
    ]
    lines.extend(["", "## Findings And Remediation", ""])
    if findings:
        lines.append("- 修复计划：上述问题中可由同步/预测/结算自动修复的已经在逐期重放中整改；仍不一致的 match 已排除出严格有效统计。")
        lines.extend(f"- {item}" for item in findings)
    else:
        lines.append("- 未发现需要额外整改的数据链问题。")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_profile43_replay(
    *,
    profile_id: int = PROFILE_ID,
    backup_database: bool = True,
    report_path: Path | None = None,
) -> dict[str, Any]:
    repo.init_db()
    backup_path = _backup_database() if backup_database else None
    pre_audit = audit_database_before(profile_id)
    strategy = _active_profile_strategy(profile_id)
    issues = sorted(repo.list_issues(), key=_issue_sort_key)
    issue_results: list[dict[str, Any]] = []
    for issue in issues:
        sync_result = sync_issue_matches(issue)
        predict_result = predict_issue(issue, ensure_collected=False)
        settle_result = settle_issue_results(issue)
        audit = _audit_issue_chain(issue, strategy)
        audit["sync_result"] = sync_result
        audit["predict_result"] = predict_result
        audit["settle_result"] = settle_result
        issue_results.append(audit)
    totals = _combine_issue_metrics(issue_results)
    report_path = report_path or repo.BASE_DIR / f"profile43_replay_audit_{_now_stamp()}.md"
    _write_report(
        report_path=report_path,
        backup_path=backup_path,
        pre_audit=pre_audit,
        issue_results=issue_results,
        totals=totals,
    )
    return {
        "profile_id": profile_id,
        "backup_path": str(backup_path) if backup_path else "",
        "report_path": str(report_path),
        "issues": issues,
        "issue_results": issue_results,
        "totals": totals,
        "status_level": "success" if totals["action_share"] >= 0.60 and totals["hit_rate"] >= 0.70 else "warning",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay all retained issues and audit active profile #43.")
    parser.add_argument("--profile-id", type=int, default=PROFILE_ID)
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args(argv)
    result = run_profile43_replay(
        profile_id=args.profile_id,
        backup_database=not args.no_backup,
    )
    totals = result["totals"]
    print(f"report={result['report_path']}")
    if result.get("backup_path"):
        print(f"backup={result['backup_path']}")
    print(
        "profile43 strict replay: "
        f"samples={totals['sample_count']} actions={totals['action_count']} hits={totals['hit_count']} "
        f"action_share={totals['action_share'] * 100:.2f}% hit_rate={totals['hit_rate'] * 100:.2f}%"
    )
    return 0 if result["status_level"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
