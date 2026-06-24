from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable

from collection_repository import BASE_DIR, DB_PATH, get_connection, init_db


DEPENDENT_TABLES = ("matches", "analyses")
OPTIONAL_SNAPSHOT_TABLE = "feature_snapshots"
SOURCE_NAME_PATTERNS = ("football_data*.db*",)


def _candidate_source_paths() -> list[Path]:
    current = DB_PATH.resolve()
    candidates: list[Path] = []
    for pattern in SOURCE_NAME_PATTERNS:
        candidates.extend(BASE_DIR.glob(pattern))
    return [
        path
        for path in sorted(set(candidates))
        if path.is_file()
        and path.resolve() != current
        and not path.name.endswith("-journal")
        and "journal" not in path.name
        and "recovered" not in path.name
    ]


def _connect_readonly(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def _insert_or_ignore_row(
    dest: sqlite3.Connection,
    table: str,
    payload: dict[str, Any],
) -> None:
    if not payload:
        return
    columns = list(payload.keys())
    placeholders = ", ".join("?" for _ in columns)
    column_sql = ", ".join(columns)
    dest.execute(
        f"INSERT OR IGNORE INTO {table} ({column_sql}) VALUES ({placeholders})",
        tuple(payload[column] for column in columns),
    )


def _copy_match_dependencies(
    source: sqlite3.Connection,
    dest: sqlite3.Connection,
    match_id: str,
) -> None:
    for table in DEPENDENT_TABLES:
        source_columns = set(_columns(source, table))
        dest_columns = set(_columns(dest, table))
        common_columns = [column for column in _columns(dest, table) if column in source_columns]
        row = source.execute(f"SELECT * FROM {table} WHERE match_id = ?", (match_id,)).fetchone()
        if row is None:
            continue
        _insert_or_ignore_row(
            dest,
            table,
            {column: row[column] for column in common_columns},
        )


def _copy_feature_snapshot(
    source: sqlite3.Connection,
    dest: sqlite3.Connection,
    snapshot_id: int,
) -> int:
    if snapshot_id <= 0:
        return 0
    if OPTIONAL_SNAPSHOT_TABLE not in _table_names(source) or OPTIONAL_SNAPSHOT_TABLE not in _table_names(dest):
        return 0
    row = source.execute(
        "SELECT * FROM feature_snapshots WHERE snapshot_id = ?",
        (snapshot_id,),
    ).fetchone()
    if row is None:
        return 0
    source_columns = set(_columns(source, OPTIONAL_SNAPSHOT_TABLE))
    dest_columns = set(_columns(dest, OPTIONAL_SNAPSHOT_TABLE))
    common_columns = [
        column
        for column in _columns(dest, OPTIONAL_SNAPSHOT_TABLE)
        if column in source_columns and column != "snapshot_id"
    ]
    payload = {column: row[column] for column in common_columns}
    if not payload:
        return 0
    columns = list(payload.keys())
    dest.execute(
        f"INSERT INTO feature_snapshots ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})",
        tuple(payload[column] for column in columns),
    )
    return int(dest.execute("SELECT last_insert_rowid()").fetchone()[0] or 0)


def _insert_prediction_run(
    source: sqlite3.Connection,
    dest: sqlite3.Connection,
    source_run_id: int,
    copied_snapshot_id: int,
) -> int:
    row = source.execute("SELECT * FROM prediction_runs WHERE run_id = ?", (source_run_id,)).fetchone()
    if row is None:
        return 0
    source_columns = set(_columns(source, "prediction_runs"))
    common_columns = [
        column
        for column in _columns(dest, "prediction_runs")
        if column in source_columns and column != "run_id"
    ]
    payload = {column: row[column] for column in common_columns}
    if "feature_snapshot_id" in payload:
        payload["feature_snapshot_id"] = copied_snapshot_id
    columns = list(payload.keys())
    dest.execute(
        f"INSERT INTO prediction_runs ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})",
        tuple(payload[column] for column in columns),
    )
    return int(dest.execute("SELECT last_insert_rowid()").fetchone()[0] or 0)


def _insert_feedback_log(
    source: sqlite3.Connection,
    dest: sqlite3.Connection,
    source_feedback_id: int,
    new_run_id: int,
) -> None:
    row = source.execute("SELECT * FROM feedback_logs WHERE feedback_id = ?", (source_feedback_id,)).fetchone()
    if row is None or new_run_id <= 0:
        return
    source_columns = set(_columns(source, "feedback_logs"))
    common_columns = [
        column
        for column in _columns(dest, "feedback_logs")
        if column in source_columns and column != "feedback_id"
    ]
    payload = {column: row[column] for column in common_columns}
    payload["prediction_run_id"] = new_run_id
    columns = list(payload.keys())
    dest.execute(
        f"INSERT INTO feedback_logs ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})",
        tuple(payload[column] for column in columns),
    )


def summarize_historical_learning_sources(
    source_paths: Iterable[Path] | None = None,
) -> dict[str, Any]:
    init_db()
    paths = list(source_paths) if source_paths is not None else _candidate_source_paths()
    sources: list[dict[str, Any]] = []
    for path in paths:
        try:
            source = _connect_readonly(path)
            try:
                tables = _table_names(source)
                if not {"prediction_runs", "feedback_logs", "matches"} <= tables:
                    sources.append({"path": str(path), "status": "skipped", "reason": "missing required tables"})
                    continue
                row = source.execute(
                    """
                    SELECT COUNT(*) AS feedback_rows,
                           COUNT(DISTINCT p.issue) AS feedback_issues,
                           MIN(p.issue) AS min_issue,
                           MAX(p.issue) AS max_issue
                    FROM feedback_logs f
                    JOIN prediction_runs p ON p.run_id = f.prediction_run_id
                    WHERE TRIM(IFNULL(p.issue, '')) <> ''
                    """
                ).fetchone()
                sources.append(
                    {
                        "path": str(path),
                        "name": path.name,
                        "status": "ready",
                        "feedback_rows": int(row["feedback_rows"] or 0),
                        "feedback_issues": int(row["feedback_issues"] or 0),
                        "min_issue": str(row["min_issue"] or ""),
                        "max_issue": str(row["max_issue"] or ""),
                    }
                )
            finally:
                source.close()
        except Exception as exc:  # noqa: BLE001
            sources.append({"path": str(path), "name": path.name, "status": "error", "reason": str(exc)})
    return {"source_count": len(sources), "sources": sources}


def import_historical_learning_feedback(
    source_paths: Iterable[Path] | None = None,
    progress_callback=None,
) -> dict[str, Any]:
    init_db()
    paths = list(source_paths) if source_paths is not None else _candidate_source_paths()
    imported_rows = 0
    skipped_existing = 0
    skipped_invalid = 0
    source_summaries: list[dict[str, Any]] = []
    imported_issues: set[str] = set()

    dest = get_connection()
    try:
        dest.row_factory = sqlite3.Row
        for path in paths:
            summary = {"name": path.name, "imported_rows": 0, "skipped_existing": 0, "skipped_invalid": 0}
            try:
                source = _connect_readonly(path)
                try:
                    if not {"prediction_runs", "feedback_logs", "matches"} <= _table_names(source):
                        summary["status"] = "skipped"
                        summary["reason"] = "missing required tables"
                        source_summaries.append(summary)
                        continue
                    rows = source.execute(
                        """
                        SELECT f.feedback_id, f.match_id, f.prediction_run_id, p.issue, p.feature_snapshot_id
                        FROM feedback_logs f
                        JOIN prediction_runs p ON p.run_id = f.prediction_run_id
                        WHERE TRIM(IFNULL(p.issue, '')) <> ''
                        ORDER BY p.issue, f.feedback_id
                        """
                    ).fetchall()
                    for row in rows:
                        match_id = str(row["match_id"] or "").strip()
                        if not match_id:
                            skipped_invalid += 1
                            summary["skipped_invalid"] += 1
                            continue
                        exists = dest.execute(
                            "SELECT 1 FROM feedback_logs WHERE match_id = ? LIMIT 1",
                            (match_id,),
                        ).fetchone()
                        if exists:
                            skipped_existing += 1
                            summary["skipped_existing"] += 1
                            continue
                        _copy_match_dependencies(source, dest, match_id)
                        snapshot_id = _copy_feature_snapshot(
                            source,
                            dest,
                            int(row["feature_snapshot_id"] or 0),
                        )
                        new_run_id = _insert_prediction_run(
                            source,
                            dest,
                            int(row["prediction_run_id"] or 0),
                            snapshot_id,
                        )
                        if new_run_id <= 0:
                            skipped_invalid += 1
                            summary["skipped_invalid"] += 1
                            continue
                        _insert_feedback_log(source, dest, int(row["feedback_id"] or 0), new_run_id)
                        imported_rows += 1
                        summary["imported_rows"] += 1
                        imported_issues.add(str(row["issue"] or "").strip())
                    summary["status"] = "done"
                finally:
                    source.close()
            except Exception as exc:  # noqa: BLE001
                summary["status"] = "error"
                summary["reason"] = str(exc)
            source_summaries.append(summary)
        dest.commit()

    finally:
        dest.close()

    issue_count = len({issue for issue in imported_issues if issue})
    message = f"历史闭环导入完成：新增 {imported_rows} 条反馈，覆盖 {issue_count} 个期号，跳过已有 {skipped_existing} 条。"
    return {
        "imported_rows": imported_rows,
        "imported_issue_count": issue_count,
        "skipped_existing": skipped_existing,
        "skipped_invalid": skipped_invalid,
        "sources": source_summaries,
        "status_message": message,
        "task_message": message,
        "status_level": "success" if imported_rows else "info",
    }
