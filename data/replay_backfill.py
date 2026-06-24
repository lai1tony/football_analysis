from __future__ import annotations

from contextlib import closing
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from collection_repository import (
    BASE_DIR,
    COLLECTION_FAILURE_PREFIX,
    get_connection,
    init_db,
    list_matches_by_issue,
    save_failed_analysis,
)
from collection_service import (
    collect_match_with_auto_retry,
    get_collection_failure_reason,
    sync_issue_matches,
)
from prediction_engine import predict_issue, settle_issue_results
from source_500_client import fetch_sfc_issue_sequence


REPLAY_BACKFILL_SOURCE = "replay_backfill"
REPLAY_BACKFILL_NOTE_PREFIX = "历史回放补数：使用补数时点重新采集可用数据并回放生成预测，不等同于当期真实赛前运行记录。"
MATCH_COLLECTION_TIMEOUT_SECONDS = 180
CHILD_RESULT_PREFIX = "REPLAY_BACKFILL_COLLECT_RESULT="


def _issue_number(issue: str) -> int | None:
    text = str(issue or "").strip()
    if not text.isdigit():
        return None
    return int(text)


def _settled_issue_numbers() -> list[int]:
    with closing(get_connection()) as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT p.issue
            FROM feedback_logs f
            JOIN prediction_runs p ON p.run_id = f.prediction_run_id
            WHERE TRIM(IFNULL(p.issue, '')) <> ''
            ORDER BY p.issue
            """
        ).fetchall()
    numbers = [_issue_number(str(row[0] or "")) for row in rows]
    return sorted(number for number in numbers if number is not None)


def build_replay_backfill_issue_plan(
    *,
    required_issue_count: int = 90,
    target_issue_count: int | None = None,
) -> list[str]:
    init_db()
    settled_issues = _settled_issue_numbers()
    if not settled_issues:
        return []

    missing_count = max(int(required_issue_count or 0) - len(settled_issues), 0)
    if target_issue_count is not None:
        missing_count = min(max(int(target_issue_count or 0), 0), missing_count or int(target_issue_count or 0))
    if missing_count <= 0:
        return []

    first_issue = min(settled_issues)
    try:
        issue_sequence = fetch_sfc_issue_sequence()
    except Exception:  # noqa: BLE001
        issue_sequence = []
    if issue_sequence:
        first_issue_text = str(first_issue)
        if first_issue_text in issue_sequence:
            first_index = issue_sequence.index(first_issue_text)
            return issue_sequence[max(0, first_index - missing_count) : first_index]

    issue_ordinal = first_issue % 1000
    if issue_ordinal <= missing_count:
        return []

    start_issue = first_issue - missing_count
    return [str(issue) for issue in range(start_issue, first_issue)]


def _feedback_count_for_issue(issue: str) -> int:
    with closing(get_connection()) as conn:
        row = conn.execute(
            """
            SELECT COUNT(*)
            FROM feedback_logs f
            JOIN prediction_runs p ON p.run_id = f.prediction_run_id
            WHERE p.issue = ?
            """,
            (str(issue),),
        ).fetchone()
    return int(row[0] or 0) if row else 0


def _mark_issue_feedback_as_replay_backfill(issue: str) -> int:
    note_like = f"{REPLAY_BACKFILL_NOTE_PREFIX}%"
    with closing(get_connection()) as conn:
        cursor = conn.execute(
            """
            UPDATE feedback_logs
            SET roi_source = ?,
                notes = CASE
                    WHEN notes LIKE ? THEN notes
                    WHEN TRIM(IFNULL(notes, '')) = '' THEN ?
                    ELSE ? || ' ' || notes
                END
            WHERE feedback_id IN (
                SELECT f.feedback_id
                FROM feedback_logs f
                JOIN prediction_runs p ON p.run_id = f.prediction_run_id
                WHERE p.issue = ?
            )
            """,
            (
                REPLAY_BACKFILL_SOURCE,
                note_like,
                REPLAY_BACKFILL_NOTE_PREFIX,
                REPLAY_BACKFILL_NOTE_PREFIX,
                str(issue),
            ),
        )
        conn.commit()
        return int(cursor.rowcount or 0)


def _emit_progress(progress_callback, **payload: Any) -> None:
    if progress_callback is not None:
        progress_callback(**payload)


def _count_result(result: Any, key: str, fallback_key: str = "") -> int:
    if isinstance(result, dict):
        return int(result.get(key, 0) or (result.get(fallback_key, 0) if fallback_key else 0) or 0)
    if isinstance(result, list):
        return len(result)
    return 0


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _row_text(row: Any, key: str, default: str = "") -> str:
    try:
        value = row[key]
    except (KeyError, IndexError, TypeError):
        value = default
    return str(value or default).strip()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _run_collect_match_child(match_id: str) -> int:
    from collection_service import collect_match

    result = collect_match(match_id)
    print(CHILD_RESULT_PREFIX + json.dumps(_json_safe(result), ensure_ascii=False))
    return 0


def _parse_child_result(stdout: str) -> dict[str, Any]:
    for line in reversed(str(stdout or "").splitlines()):
        if line.startswith(CHILD_RESULT_PREFIX):
            payload = line[len(CHILD_RESULT_PREFIX) :]
            parsed = json.loads(payload)
            return parsed if isinstance(parsed, dict) else {}
    return {}


def _collect_match_with_timeout(
    match_id: str,
    *,
    timeout_seconds: int = MATCH_COLLECTION_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    try:
        completed = subprocess.run(
            [sys.executable, str(script_path), "collect-match", str(match_id)],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=max(int(timeout_seconds or 0), 1),
            check=False,
        )
    except subprocess.TimeoutExpired:
        remarks = f"{COLLECTION_FAILURE_PREFIX}历史回放补数超时：单场采集超过 {timeout_seconds} 秒"
        save_failed_analysis(match_id, remarks, _now_text())
        return {
            "match_id": match_id,
            "collection_status": "failed",
            "reason": remarks,
            "timed_out": True,
        }
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        remarks = f"{COLLECTION_FAILURE_PREFIX}历史回放补数子进程失败：{detail[:500]}"
        save_failed_analysis(match_id, remarks, _now_text())
        return {"match_id": match_id, "collection_status": "failed", "reason": remarks}
    result = _parse_child_result(completed.stdout)
    if not result:
        remarks = f"{COLLECTION_FAILURE_PREFIX}历史回放补数子进程未返回采集结果"
        save_failed_analysis(match_id, remarks, _now_text())
        return {"match_id": match_id, "collection_status": "failed", "reason": remarks}
    return result


def _issue_collection_failures(issue: str) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in list_matches_by_issue(issue):
        reason = get_collection_failure_reason(row)
        if not reason:
            continue
        failures.append(
            {
                "match_id": _row_text(row, "match_id"),
                "match_label": f"{_row_text(row, 'home_team')} vs {_row_text(row, 'away_team')}",
                "issue": issue,
                "reason": reason,
            }
        )
    return failures


def _summarize_collection_failures(failures: list[dict[str, Any]], limit: int = 3) -> str:
    parts = []
    for item in failures[:limit]:
        issue = str(item.get("issue") or "")
        label = str(item.get("match_label") or item.get("match_id") or "")
        reason = str(item.get("reason") or "")
        parts.append(f"{issue} {label}（{reason}）")
    if len(failures) > limit:
        parts.append(f"等 {len(failures)} 场")
    return "；".join(parts)


def _collect_issue_matches_safely(
    issue: str,
    *,
    timeout_seconds: int = MATCH_COLLECTION_TIMEOUT_SECONDS,
    progress_callback=None,
) -> dict[str, Any]:
    rows = list_matches_by_issue(issue)
    total_matches = len(rows)
    collected_count = 0
    failed_matches: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        match_id = _row_text(row, "match_id")
        match_label = f"{_row_text(row, 'home_team')} vs {_row_text(row, 'away_team')}"
        status = _row_text(row, "collection_status")
        if status == "success":
            collected_count += 1
            continue
        _emit_progress(
            progress_callback,
            total_items=total_matches,
            completed_items=index - 1,
            current_item_index=index,
            current_item_label=match_label,
            current_step="历史回放单场采集",
            message=f"期号 {issue} 正在采集第 {index}/{total_matches} 场：{match_label}",
        )
        try:
            result = collect_match_with_auto_retry(
                match_id,
                collector=lambda item: _collect_match_with_timeout(
                    item,
                    timeout_seconds=timeout_seconds,
                ),
                progress_callback=progress_callback,
                context_label="历史回放自动补采",
            )
        except Exception as exc:  # noqa: BLE001
            remarks = f"{COLLECTION_FAILURE_PREFIX}历史回放补数异常：{exc}"
            save_failed_analysis(match_id, remarks, _now_text())
            result = {"match_id": match_id, "collection_status": "failed", "reason": remarks}
        failure_reason = get_collection_failure_reason(result)
        if failure_reason:
            failed_matches.append({
                "match_id": match_id,
                "match_label": match_label,
                "issue": issue,
                "reason": failure_reason,
                "auto_retry_count": int(result.get("auto_retry_count", 0) or 0),
            })
        else:
            collected_count += 1
    failed_count = len(failed_matches)
    return {
        "total_matches": total_matches,
        "collected_count": collected_count,
        "failed_count": failed_count,
        "failed_matches": failed_matches,
        "status_level": "success" if failed_count == 0 else "warning",
    }


def replay_backfill_learning_feedback(
    *,
    required_issue_count: int = 90,
    target_issue_count: int | None = None,
    force: bool = False,
    match_timeout_seconds: int = MATCH_COLLECTION_TIMEOUT_SECONDS,
    progress_callback=None,
) -> dict[str, Any]:
    init_db()
    issue_plan = build_replay_backfill_issue_plan(
        required_issue_count=required_issue_count,
        target_issue_count=target_issue_count,
    )
    total_issues = len(issue_plan)
    if not issue_plan:
        message = "历史回放补数无需执行：当前没有可推导的更早期号缺口。"
        return {
            "planned_issues": [],
            "completed_issues": [],
            "failed_issues": [],
            "partial_issues": [],
            "collection_failed_issues": [],
            "skipped_issues": [],
            "feedback_marked": 0,
            "status_message": message,
            "task_message": message,
            "status_level": "info",
        }

    completed_issues: list[str] = []
    failed_issues: list[dict[str, str]] = []
    partial_issues: list[dict[str, int | str]] = []
    collection_failed_issues: list[dict[str, int | str]] = []
    skipped_issues: list[dict[str, str]] = []
    feedback_marked = 0
    synced_matches = 0
    collected_matches = 0
    predicted_matches = 0
    settled_matches = 0

    for index, issue in enumerate(issue_plan, start=1):
        _emit_progress(
            progress_callback,
            total_items=total_issues,
            completed_items=index - 1,
            current_item_index=index,
            current_item_label=f"期号 {issue}",
            current_step="准备历史回放补数",
            message=f"准备补入第 {index}/{total_issues} 个历史期号：{issue}",
        )
        try:
            sync_result = sync_issue_matches(issue, return_details=True)
            issue_matches = sync_result.get("matches", []) if isinstance(sync_result, dict) else sync_result
            match_count = len(issue_matches) if isinstance(issue_matches, list) else 0
            synced_matches += match_count
            if match_count <= 0:
                skipped_issues.append({"issue": issue, "reason": "未同步到对赛"})
                continue

            existing_feedback = _feedback_count_for_issue(issue)
            collection_failures = _issue_collection_failures(issue)
            if existing_feedback > 0 and not force and not collection_failures:
                skipped_issues.append({"issue": issue, "reason": "已有反馈闭环记录且采集完整"})
                continue
            if existing_feedback > 0 and collection_failures:
                _emit_progress(
                    progress_callback,
                    total_items=total_issues,
                    completed_items=index - 1,
                    current_item_index=index,
                    current_item_label=f"期号 {issue}",
                    current_step="历史回放采集完整性补查",
                    message=f"期号 {issue} 已有反馈但存在 {len(collection_failures)} 场采集异常，进入自动补采。", 
                    level="warning",
                )

            collect_result = _collect_issue_matches_safely(
                issue,
                timeout_seconds=match_timeout_seconds,
                progress_callback=progress_callback,
            )
            collected_matches += _count_result(collect_result, "collected_count", "success_count")
            if collect_result.get("failed_count", 0):
                failed = collect_result.get("failed_matches", [])
                collection_failed_issues.append(
                    {
                        "issue": issue,
                        "failed_count": int(collect_result.get("failed_count", 0) or 0),
                        "reason": _summarize_collection_failures(failed if isinstance(failed, list) else []),
                    }
                )
                continue

            predict_result = predict_issue(issue, ensure_collected=False)
            predicted_matches += _count_result(predict_result, "predicted_count")

            settle_result = settle_issue_results(issue)
            settled_matches += _count_result(settle_result, "settled_count")

            marked = _mark_issue_feedback_as_replay_backfill(issue)
            feedback_marked += marked
            if marked <= 0:
                skipped_issues.append({"issue": issue, "reason": "未生成可标记反馈"})
                continue
            if marked < match_count:
                partial_issues.append(
                    {"issue": issue, "match_count": match_count, "feedback_count": marked}
                )
                continue
            completed_issues.append(issue)

            _emit_progress(
                progress_callback,
                total_items=total_issues,
                completed_items=index,
                current_item_index=index,
                current_item_label=f"期号 {issue}",
                current_step="历史回放补数完成",
                message=f"期号 {issue} 已回放并标记 {marked} 条反馈。",
            )
        except Exception as exc:  # noqa: BLE001
            failed_issues.append({"issue": issue, "reason": str(exc)})
            _emit_progress(
                progress_callback,
                total_items=total_issues,
                completed_items=index,
                current_item_index=index,
                current_item_label=f"期号 {issue}",
                current_step="历史回放补数失败",
                message=f"期号 {issue} 补数失败：{exc}",
                level="warning",
            )

    message = (
        f"历史回放补数完成：计划 {total_issues} 期，完成 {len(completed_issues)} 期，"
        f"标记反馈 {feedback_marked} 条，部分闭环 {len(partial_issues)} 期，"
        f"采集异常 {len(collection_failed_issues)} 期，跳过 {len(skipped_issues)} 期，失败 {len(failed_issues)} 期。"
    )
    if collection_failed_issues:
        examples = _summarize_collection_failures(
            [
                {
                    "issue": item.get("issue", ""),
                    "match_label": "采集异常",
                    "reason": item.get("reason", ""),
                }
                for item in collection_failed_issues
            ],
            limit=3,
        )
        message += f" 采集异常明细：{examples}。"
    status_level = "success" if completed_issues and not failed_issues else "warning"
    if collection_failed_issues:
        status_level = "warning"
    elif not completed_issues:
        status_level = "warning" if failed_issues else "info"
    return {
        "planned_issues": issue_plan,
        "completed_issues": completed_issues,
        "failed_issues": failed_issues,
        "partial_issues": partial_issues,
        "collection_failed_issues": collection_failed_issues,
        "skipped_issues": skipped_issues,
        "synced_matches": synced_matches,
        "collected_matches": collected_matches,
        "predicted_matches": predicted_matches,
        "settled_matches": settled_matches,
        "feedback_marked": feedback_marked,
        "status_message": message,
        "task_message": message,
        "status_level": status_level,
    }


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if len(args) == 2 and args[0] == "collect-match":
        return _run_collect_match_child(args[1])
    print("Usage: python replay_backfill.py collect-match <match_id>")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
