import re
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypeVar


BASE_DIR = Path(__file__).resolve().parent
PRIMARY_DB_PATH = BASE_DIR / "football_data.db"
RECOVERY_DB_PATH = BASE_DIR / "football_data_live.db"
DB_PATH = PRIMARY_DB_PATH
READONLY_PRIMARY_URI = f"file:{PRIMARY_DB_PATH.as_posix()}?mode=ro&immutable=1"
T = TypeVar("T")
_ACTIVE_RW_PATH: Path | None = None
DEFAULT_ISSUE_RETENTION_COUNT = 90
COLLECTION_FAILURE_PREFIX = "采集失败："
MISSING_DIMENSION_PREFIX = f"{COLLECTION_FAILURE_PREFIX}缺少采集维度："
REQUIRED_ANALYSIS_FIELDS = (
    "elo_home",
    "elo_away",
    "recent_form_home",
    "recent_form_away",
    "home_away_form",
    "head_to_head_summary",
    "injury_or_lineup_notes",
    "motivation_or_schedule_notes",
    "european_odds_movement_summary",
    "asian_handicap_summary",
    "betting_heat_summary",
)

# Fields that are best-effort (may be empty for lower division leagues)
OPTIONAL_ANALYSIS_FIELDS = frozenset({"elo_home", "elo_away"})
ANALYSIS_PLACEHOLDER_PATTERNS = (
    "未在公开来源中命中预计首发",
    "外部公开来源补采未命中",
    "伤停/阵容补采失败",
)


def _settle_handicap_result_from_score(actual_score: str, handicap_line: float) -> str:
    parts = [int(item) for item in re.findall(r"\d+", str(actual_score or ""))[:2]]
    if len(parts) != 2:
        return ""
    adjusted_margin = float(parts[0] - parts[1]) + float(handicap_line)
    if adjusted_margin > 0:
        return "home"
    if adjusted_margin < 0:
        return "away"
    return "push"


class DatabaseWriteUnavailableError(RuntimeError):
    """Raised when SQLite cannot accept writes."""


def _analysis_success_condition(alias: str = "a") -> str:
    prefix = f"{alias}." if alias else ""
    required_fields = [f for f in REQUIRED_ANALYSIS_FIELDS if f not in OPTIONAL_ANALYSIS_FIELDS]
    field_checks = [
        f"IFNULL({prefix}{field}, '') <> ''" for field in required_fields
    ]
    placeholder_checks = [
        f"IFNULL({prefix}{field}, '') NOT LIKE '%{pattern}%'"
        for field in REQUIRED_ANALYSIS_FIELDS
        for pattern in ANALYSIS_PLACEHOLDER_PATTERNS
    ]
    remarks_check = (
        f"(IFNULL({prefix}remarks, '') NOT LIKE '{COLLECTION_FAILURE_PREFIX}%' "
        f"OR IFNULL({prefix}remarks, '') LIKE '{MISSING_DIMENSION_PREFIX}%')"
    )
    return " AND ".join(
        [
            f"IFNULL({prefix}collected_at, '') <> ''",
            remarks_check,
            *field_checks,
            *placeholder_checks,
        ]
    )


def _collection_status_sql(alias: str = "a") -> str:
    prefix = f"{alias}." if alias else ""
    return (
        "CASE "
        f"WHEN IFNULL({prefix}collected_at, '') = '' THEN 'uncollected' "
        f"WHEN {_analysis_success_condition(alias)} THEN 'success' "
        "ELSE 'failed' "
        "END AS collection_status"
    )


def _apply_rw_pragmas(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=MEMORY").fetchone()
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")


def _connect_path(path: Path, *, writable: bool) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    if writable:
        _apply_rw_pragmas(conn)
    return conn


def _connect_primary_snapshot() -> sqlite3.Connection:
    conn = sqlite3.connect(READONLY_PRIMARY_URI, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _is_healthy_rw_path(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with closing(_connect_path(path, writable=True)) as conn:
            conn.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()
        return True
    except sqlite3.OperationalError:
        return False


def _resolve_rw_path(force_refresh: bool = False) -> Path | None:
    global _ACTIVE_RW_PATH

    if not force_refresh and _ACTIVE_RW_PATH and _is_healthy_rw_path(_ACTIVE_RW_PATH):
        return _ACTIVE_RW_PATH

    for candidate in (PRIMARY_DB_PATH, RECOVERY_DB_PATH):
        if _is_healthy_rw_path(candidate):
            _ACTIVE_RW_PATH = candidate
            return candidate

    _ACTIVE_RW_PATH = None
    return None


def _supports_readonly_fallback(exc: sqlite3.OperationalError) -> bool:
    message = str(exc).lower()
    return "disk i/o error" in message or "readonly" in message


def _readonly_message(exc: sqlite3.OperationalError) -> str:
    return (
        "数据库当前只能以只读方式打开，现有对赛和历史预测仍可查看，"
        "但同步、采集、预测暂时不可用。通常是 "
        "data/football_data.db-journal 残留或 SQLite 日志异常导致。"
        f" 原始错误: {exc}"
    )


def _recovery_message() -> str:
    return (
        "已自动切换到恢复库 football_data_live.db，写入功能已恢复。"
        "原始 football_data.db 和 football_data.db-journal 仍保留，方便后续手工核查。"
    )


def _run_read(operation: Callable[[sqlite3.Connection], T]) -> T:
    rw_path = _resolve_rw_path()
    if rw_path is not None:
        with closing(_connect_path(rw_path, writable=True)) as conn:
            return operation(conn)

    try:
        with closing(_connect_path(PRIMARY_DB_PATH, writable=True)) as conn:
            return operation(conn)
    except sqlite3.OperationalError as exc:
        if not PRIMARY_DB_PATH.exists() or not _supports_readonly_fallback(exc):
            raise
        with closing(_connect_primary_snapshot()) as conn:
            return operation(conn)


def _run_write(operation: Callable[[sqlite3.Connection], T]) -> T:
    rw_path = _resolve_rw_path()
    if rw_path is None:
        try:
            with closing(_connect_path(PRIMARY_DB_PATH, writable=True)) as conn:
                result = operation(conn)
                conn.commit()
                return result
        except sqlite3.OperationalError as exc:
            raise DatabaseWriteUnavailableError(_readonly_message(exc)) from exc

    try:
        with closing(_connect_path(rw_path, writable=True)) as conn:
            result = operation(conn)
            conn.commit()
            return result
    except sqlite3.OperationalError as exc:
        if rw_path != PRIMARY_DB_PATH:
            _resolve_rw_path(force_refresh=True)
        raise DatabaseWriteUnavailableError(_readonly_message(exc)) from exc


def get_database_status() -> dict:
    rw_path = _resolve_rw_path(force_refresh=True)
    if rw_path == PRIMARY_DB_PATH:
        return {
            "read_only": False,
            "message": "",
            "level": "info",
            "active_path": str(PRIMARY_DB_PATH),
        }
    if rw_path == RECOVERY_DB_PATH:
        return {
            "read_only": False,
            "message": _recovery_message(),
            "level": "success",
            "active_path": str(RECOVERY_DB_PATH),
        }

    try:
        with closing(_connect_primary_snapshot()) as conn:
            conn.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()
        return {
            "read_only": True,
            "message": _readonly_message(sqlite3.OperationalError("disk I/O error")),
            "level": "warning",
            "active_path": str(PRIMARY_DB_PATH),
        }
    except sqlite3.OperationalError as exc:
        return {
            "read_only": True,
            "message": f"数据库不可用: {exc}",
            "level": "error",
            "active_path": str(PRIMARY_DB_PATH),
        }


def get_connection() -> sqlite3.Connection:
    rw_path = _resolve_rw_path()
    if rw_path is None:
        return _connect_path(PRIMARY_DB_PATH, writable=True)
    return _connect_path(rw_path, writable=True)


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row[1]) for row in rows}


def _ensure_columns(
    conn: sqlite3.Connection,
    table_name: str,
    column_definitions: dict[str, str],
) -> None:
    existing = _table_columns(conn, table_name)
    for column_name, definition in column_definitions.items():
        if column_name in existing:
            continue
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")


def _dedup_feedback_logs_inplace(conn: sqlite3.Connection) -> int:
    """Collapse duplicate feedback_logs to one row per match_id.

    Historical bug: ``feedback_logs`` was unique only on ``prediction_run_id``,
    so a second ``predict_match`` followed by a fresh ``settle`` could leave an
    extra row for the same match (one per prediction_run_id). Statistics that
    aggregate by ``COUNT(*)`` / ``SUM`` would then double-count.

    The cleanup keeps the most recently written feedback row per match_id
    (``MAX(feedback_id)``) — that is the row produced by the latest settle and
    the one users see on screen, so dropping the rest matches user intent.
    Returns the number of stale rows removed.
    """

    cursor = conn.execute(
        """
        DELETE FROM feedback_logs
        WHERE feedback_id NOT IN (
            SELECT MAX(feedback_id)
            FROM feedback_logs
            GROUP BY match_id
        )
        """
    )
    return int(cursor.rowcount or 0)


def _dedup_feature_snapshots_inplace(conn: sqlite3.Connection) -> int:
    """Keep only the latest feature_snapshot per match_id.

    ``collect_match`` writes a new snapshot row each time, but predictions only
    read ``get_latest_feature_snapshot``. Older rows are dead weight that grow
    without bound across re-collections within the retention window.
    """

    cursor = conn.execute(
        """
        DELETE FROM feature_snapshots
        WHERE snapshot_id NOT IN (
            SELECT MAX(snapshot_id)
            FROM feature_snapshots
            GROUP BY match_id
        )
        """
    )
    return int(cursor.rowcount or 0)


def _dedup_prediction_runs_inplace(conn: sqlite3.Connection) -> int:
    """Keep only the latest prediction_run per match_id."""

    conn.execute(
        """
        UPDATE feedback_logs
        SET prediction_run_id = (
            SELECT pr.run_id
            FROM prediction_runs pr
            WHERE pr.match_id = feedback_logs.match_id
            ORDER BY pr.created_at DESC, pr.run_id DESC
            LIMIT 1
        )
        WHERE EXISTS (
            SELECT 1
            FROM prediction_runs pr
            WHERE pr.match_id = feedback_logs.match_id
        )
        """
    )
    conn.execute(
        """
        DELETE FROM issue_top_picks
        WHERE run_id IN (
            SELECT stale.run_id
            FROM prediction_runs stale
            WHERE stale.run_id NOT IN (
                SELECT latest.run_id
                FROM prediction_runs latest
                WHERE latest.run_id = (
                    SELECT candidate.run_id
                    FROM prediction_runs candidate
                    WHERE candidate.match_id = latest.match_id
                    ORDER BY candidate.created_at DESC, candidate.run_id DESC
                    LIMIT 1
                )
            )
        )
        """
    )
    cursor = conn.execute(
        """
        DELETE FROM prediction_runs
        WHERE run_id NOT IN (
            SELECT latest.run_id
            FROM prediction_runs latest
            WHERE latest.run_id = (
                SELECT candidate.run_id
                FROM prediction_runs candidate
                WHERE candidate.match_id = latest.match_id
                ORDER BY candidate.created_at DESC, candidate.run_id DESC
                LIMIT 1
            )
        )
        """
    )
    return int(cursor.rowcount or 0)


def _ensure_indexes(conn: sqlite3.Connection) -> None:
    # Dedup before creating UNIQUE indexes so the migration cannot fail on
    # historical duplicates.
    _dedup_feedback_logs_inplace(conn)
    _dedup_feature_snapshots_inplace(conn)
    _dedup_prediction_runs_inplace(conn)
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_feedback_logs_prediction_run_unique
        ON feedback_logs(prediction_run_id)
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_feedback_logs_match_id_unique
        ON feedback_logs(match_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_prediction_runs_match_created_at
        ON prediction_runs(match_id, created_at, run_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_matches_issue_match_no
        ON matches(issue, match_no, match_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_learning_profiles_status_created_at
        ON learning_profiles(status, created_at, learning_profile_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_prediction_runs_manual_review_status
        ON prediction_runs(manual_review_status, match_id, run_id)
        """
    )


def init_db() -> None:
    def _operation(conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS matches (
                match_id TEXT PRIMARY KEY,
                issue TEXT NOT NULL DEFAULT '',
                league TEXT NOT NULL DEFAULT '',
                match_no TEXT NOT NULL DEFAULT '',
                match_time TEXT NOT NULL DEFAULT '',
                home_team TEXT NOT NULL DEFAULT '',
                away_team TEXT NOT NULL DEFAULT '',
                source_match_url TEXT NOT NULL DEFAULT '',
                shuju_url TEXT NOT NULL DEFAULT '',
                ouzhi_url TEXT NOT NULL DEFAULT '',
                touzhu_url TEXT NOT NULL DEFAULT '',
                yazhi_url TEXT NOT NULL DEFAULT '',
                list_odds_win TEXT NOT NULL DEFAULT '',
                list_odds_draw TEXT NOT NULL DEFAULT '',
                list_odds_loss TEXT NOT NULL DEFAULT '',
                list_heat_win TEXT NOT NULL DEFAULT '',
                list_heat_draw TEXT NOT NULL DEFAULT '',
                list_heat_loss TEXT NOT NULL DEFAULT '',
                sync_time TEXT NOT NULL DEFAULT '',
                actual_result TEXT NOT NULL DEFAULT '',
                actual_score TEXT NOT NULL DEFAULT '',
                result_status TEXT NOT NULL DEFAULT '',
                result_source_url TEXT NOT NULL DEFAULT '',
                result_synced_at TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS analyses (
                match_id TEXT PRIMARY KEY,
                collected_at TEXT NOT NULL DEFAULT '',
                elo_home TEXT NOT NULL DEFAULT '',
                elo_away TEXT NOT NULL DEFAULT '',
                market_value_summary TEXT NOT NULL DEFAULT '',
                recent_form_home TEXT NOT NULL DEFAULT '',
                recent_form_away TEXT NOT NULL DEFAULT '',
                home_away_form TEXT NOT NULL DEFAULT '',
                head_to_head_summary TEXT NOT NULL DEFAULT '',
                injury_or_lineup_notes TEXT NOT NULL DEFAULT '',
                motivation_or_schedule_notes TEXT NOT NULL DEFAULT '',
                european_odds_movement_summary TEXT NOT NULL DEFAULT '',
                asian_handicap_summary TEXT NOT NULL DEFAULT '',
                betting_heat_summary TEXT NOT NULL DEFAULT '',
                media_source_links TEXT NOT NULL DEFAULT '',
                collected_sources TEXT NOT NULL DEFAULT '',
                collection_quality_summary TEXT NOT NULL DEFAULT '',
                remarks TEXT NOT NULL DEFAULT '',
                FOREIGN KEY(match_id) REFERENCES matches(match_id)
            );

            CREATE TABLE IF NOT EXISTS prediction_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id TEXT NOT NULL,
                issue TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT '',
                quant_home_prob REAL NOT NULL DEFAULT 0,
                quant_draw_prob REAL NOT NULL DEFAULT 0,
                quant_away_prob REAL NOT NULL DEFAULT 0,
                ml_home_prob REAL NOT NULL DEFAULT 0,
                ml_draw_prob REAL NOT NULL DEFAULT 0,
                ml_away_prob REAL NOT NULL DEFAULT 0,
                final_home_prob REAL NOT NULL DEFAULT 0,
                final_draw_prob REAL NOT NULL DEFAULT 0,
                final_away_prob REAL NOT NULL DEFAULT 0,
                fair_odds_home REAL NOT NULL DEFAULT 0,
                fair_odds_draw REAL NOT NULL DEFAULT 0,
                fair_odds_away REAL NOT NULL DEFAULT 0,
                market_odds_home REAL NOT NULL DEFAULT 0,
                market_odds_draw REAL NOT NULL DEFAULT 0,
                market_odds_away REAL NOT NULL DEFAULT 0,
                ev_home REAL NOT NULL DEFAULT 0,
                ev_draw REAL NOT NULL DEFAULT 0,
                ev_away REAL NOT NULL DEFAULT 0,
                quality_score REAL NOT NULL DEFAULT 0,
                model_agreement REAL NOT NULL DEFAULT 0,
                confidence_score REAL NOT NULL DEFAULT 0,
                risk_level TEXT NOT NULL DEFAULT '',
                recommendation TEXT NOT NULL DEFAULT '',
                recommended_outcome TEXT NOT NULL DEFAULT '',
                suggested_stake_pct REAL NOT NULL DEFAULT 0,
                handicap_recommendation TEXT NOT NULL DEFAULT '',
                handicap_recommended_side TEXT NOT NULL DEFAULT '',
                handicap_line REAL NOT NULL DEFAULT 0,
                handicap_initial_line REAL NOT NULL DEFAULT 0,
                handicap_home_odds REAL NOT NULL DEFAULT 0,
                handicap_away_odds REAL NOT NULL DEFAULT 0,
                handicap_initial_home_odds REAL NOT NULL DEFAULT 0,
                handicap_initial_away_odds REAL NOT NULL DEFAULT 0,
                handicap_home_cover_prob REAL NOT NULL DEFAULT 0,
                handicap_away_cover_prob REAL NOT NULL DEFAULT 0,
                handicap_expected_value REAL NOT NULL DEFAULT 0,
                handicap_confidence REAL NOT NULL DEFAULT 0,
                handicap_reason TEXT NOT NULL DEFAULT '',
                algo_recommendation TEXT NOT NULL DEFAULT '',
                algo_recommended_outcome TEXT NOT NULL DEFAULT '',
                algo_risk_level TEXT NOT NULL DEFAULT '',
                algo_suggested_stake_pct REAL NOT NULL DEFAULT 0,
                llm_review_enabled INTEGER NOT NULL DEFAULT 0,
                llm_review_status TEXT NOT NULL DEFAULT '',
                llm_review_decision TEXT NOT NULL DEFAULT '',
                llm_review_target_action TEXT NOT NULL DEFAULT '',
                llm_review_reason TEXT NOT NULL DEFAULT '',
                llm_review_raw TEXT NOT NULL DEFAULT '',
                review_model_name TEXT NOT NULL DEFAULT '',
                final_resolution_reason TEXT NOT NULL DEFAULT '',
                arbiter_review_enabled INTEGER NOT NULL DEFAULT 0,
                arbiter_review_status TEXT NOT NULL DEFAULT '',
                arbiter_review_decision TEXT NOT NULL DEFAULT '',
                arbiter_review_target_action TEXT NOT NULL DEFAULT '',
                arbiter_review_reason TEXT NOT NULL DEFAULT '',
                arbiter_review_raw TEXT NOT NULL DEFAULT '',
                arbiter_review_model_name TEXT NOT NULL DEFAULT '',
                effective_recommendation TEXT NOT NULL DEFAULT '',
                effective_stake_pct REAL NOT NULL DEFAULT 0,
                effective_action_source TEXT NOT NULL DEFAULT '',
                manual_review_status TEXT NOT NULL DEFAULT '',
                manual_review_reason TEXT NOT NULL DEFAULT '',
                manual_review_requested_at TEXT NOT NULL DEFAULT '',
                manual_review_resolved_at TEXT NOT NULL DEFAULT '',
                manual_review_notes TEXT NOT NULL DEFAULT '',
                llm_provider TEXT NOT NULL DEFAULT '',
                llm_model TEXT NOT NULL DEFAULT '',
                llm_summary TEXT NOT NULL DEFAULT '',
                final_report TEXT NOT NULL DEFAULT '',
                learning_profile_id INTEGER NOT NULL DEFAULT 0,
                calibrated_home_prob REAL NOT NULL DEFAULT 0,
                calibrated_draw_prob REAL NOT NULL DEFAULT 0,
                calibrated_away_prob REAL NOT NULL DEFAULT 0,
                predicted_score TEXT NOT NULL DEFAULT '',
                predicted_score_confidence REAL NOT NULL DEFAULT 0,
                predicted_score_reason TEXT NOT NULL DEFAULT '',
                predicted_score_status TEXT NOT NULL DEFAULT '',
                predicted_score_model_name TEXT NOT NULL DEFAULT '',
                predicted_score_raw TEXT NOT NULL DEFAULT '',
                quant_score_candidates TEXT NOT NULL DEFAULT '',
                FOREIGN KEY(match_id) REFERENCES matches(match_id)
            );

            CREATE TABLE IF NOT EXISTS issue_top_picks (
                pick_id INTEGER PRIMARY KEY AUTOINCREMENT,
                issue TEXT NOT NULL,
                rank INTEGER NOT NULL CHECK(rank BETWEEN 1 AND 3),
                match_id TEXT NOT NULL,
                run_id INTEGER,
                composite_score REAL NOT NULL DEFAULT 0,
                confidence_score REAL NOT NULL DEFAULT 0,
                ev_score REAL NOT NULL DEFAULT 0,
                quality_score REAL NOT NULL DEFAULT 0,
                reason TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT '',
                FOREIGN KEY(match_id) REFERENCES matches(match_id),
                FOREIGN KEY(run_id) REFERENCES prediction_runs(run_id),
                UNIQUE(issue, rank)
            );

            CREATE TABLE IF NOT EXISTS feature_snapshots (
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id TEXT NOT NULL,
                issue TEXT NOT NULL DEFAULT '',
                snapshot_at TEXT NOT NULL DEFAULT '',
                home_rating REAL NOT NULL DEFAULT 0,
                away_rating REAL NOT NULL DEFAULT 0,
                recent_home_ppg REAL NOT NULL DEFAULT 0,
                recent_away_ppg REAL NOT NULL DEFAULT 0,
                recent_home_gf_pg REAL NOT NULL DEFAULT 0,
                recent_away_gf_pg REAL NOT NULL DEFAULT 0,
                recent_home_ga_pg REAL NOT NULL DEFAULT 0,
                recent_away_ga_pg REAL NOT NULL DEFAULT 0,
                home_split_ppg REAL NOT NULL DEFAULT 0,
                away_split_ppg REAL NOT NULL DEFAULT 0,
                home_absent_count INTEGER NOT NULL DEFAULT 0,
                away_absent_count INTEGER NOT NULL DEFAULT 0,
                home_doubtful_count INTEGER NOT NULL DEFAULT 0,
                away_doubtful_count INTEGER NOT NULL DEFAULT 0,
                home_absence_impact REAL NOT NULL DEFAULT 0,
                away_absence_impact REAL NOT NULL DEFAULT 0,
                lineup_home_availability REAL NOT NULL DEFAULT 0,
                lineup_away_availability REAL NOT NULL DEFAULT 0,
                rest_days_home REAL NOT NULL DEFAULT 0,
                rest_days_away REAL NOT NULL DEFAULT 0,
                schedule_load_home INTEGER NOT NULL DEFAULT 0,
                schedule_load_away INTEGER NOT NULL DEFAULT 0,
                h2h_edge REAL NOT NULL DEFAULT 0,
                market_home_prob REAL NOT NULL DEFAULT 0,
                market_draw_prob REAL NOT NULL DEFAULT 0,
                market_away_prob REAL NOT NULL DEFAULT 0,
                feature_payload TEXT NOT NULL DEFAULT '',
                FOREIGN KEY(match_id) REFERENCES matches(match_id)
            );

            CREATE TABLE IF NOT EXISTS feedback_logs (
                feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_run_id INTEGER NOT NULL,
                match_id TEXT NOT NULL,
                actual_result TEXT NOT NULL DEFAULT '',
                actual_score TEXT NOT NULL DEFAULT '',
                settled_at TEXT NOT NULL DEFAULT '',
                hit_recommendation INTEGER NOT NULL DEFAULT 0,
                roi_delta REAL NOT NULL DEFAULT 0,
                handicap_actual_result TEXT NOT NULL DEFAULT '',
                handicap_hit INTEGER NOT NULL DEFAULT 0,
                handicap_roi_delta REAL NOT NULL DEFAULT 0,
                roi_source TEXT NOT NULL DEFAULT 'auto',
                notes TEXT NOT NULL DEFAULT '',
                FOREIGN KEY(prediction_run_id) REFERENCES prediction_runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS learning_profiles (
                learning_profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
                status TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL DEFAULT '',
                activated_at TEXT NOT NULL DEFAULT '',
                archived_at TEXT NOT NULL DEFAULT '',
                retention_issue_count INTEGER NOT NULL DEFAULT 90,
                window_type TEXT NOT NULL DEFAULT 'rolling_issues',
                window_value INTEGER NOT NULL DEFAULT 90,
                total_samples INTEGER NOT NULL DEFAULT 0,
                training_samples INTEGER NOT NULL DEFAULT 0,
                validation_samples INTEGER NOT NULL DEFAULT 0,
                training_action_samples INTEGER NOT NULL DEFAULT 0,
                validation_action_samples INTEGER NOT NULL DEFAULT 0,
                calibrator_status TEXT NOT NULL DEFAULT '',
                threshold_status TEXT NOT NULL DEFAULT '',
                calibrator_params TEXT NOT NULL DEFAULT '',
                threshold_params TEXT NOT NULL DEFAULT '',
                train_metrics TEXT NOT NULL DEFAULT '',
                validation_metrics TEXT NOT NULL DEFAULT '',
                sample_summary TEXT NOT NULL DEFAULT '',
                notes TEXT NOT NULL DEFAULT ''
            );
            """
        )
        _ensure_columns(
            conn,
            "matches",
            {
                "actual_result": "TEXT NOT NULL DEFAULT ''",
                "actual_score": "TEXT NOT NULL DEFAULT ''",
                "result_status": "TEXT NOT NULL DEFAULT ''",
                "result_source_url": "TEXT NOT NULL DEFAULT ''",
                "result_synced_at": "TEXT NOT NULL DEFAULT ''",
                "yazhi_url": "TEXT NOT NULL DEFAULT ''",
            },
        )
        _ensure_columns(
            conn,
            "analyses",
            {
                "market_value_summary": "TEXT NOT NULL DEFAULT ''",
                "asian_handicap_summary": "TEXT NOT NULL DEFAULT ''",
                "collection_quality_summary": "TEXT NOT NULL DEFAULT ''",
            },
        )
        _ensure_columns(
            conn,
            "prediction_runs",
            {
                "feature_snapshot_id": "INTEGER NOT NULL DEFAULT 0",
                "legacy_home_prob": "REAL NOT NULL DEFAULT 0",
                "legacy_draw_prob": "REAL NOT NULL DEFAULT 0",
                "legacy_away_prob": "REAL NOT NULL DEFAULT 0",
                "market_home_prob": "REAL NOT NULL DEFAULT 0",
                "market_draw_prob": "REAL NOT NULL DEFAULT 0",
                "market_away_prob": "REAL NOT NULL DEFAULT 0",
                "algo_recommendation": "TEXT NOT NULL DEFAULT ''",
                "algo_recommended_outcome": "TEXT NOT NULL DEFAULT ''",
                "algo_risk_level": "TEXT NOT NULL DEFAULT ''",
                "algo_suggested_stake_pct": "REAL NOT NULL DEFAULT 0",
                "llm_review_enabled": "INTEGER NOT NULL DEFAULT 0",
                "llm_review_status": "TEXT NOT NULL DEFAULT ''",
                "llm_review_decision": "TEXT NOT NULL DEFAULT ''",
                "llm_review_target_action": "TEXT NOT NULL DEFAULT ''",
                "llm_review_reason": "TEXT NOT NULL DEFAULT ''",
                "llm_review_raw": "TEXT NOT NULL DEFAULT ''",
                "review_model_name": "TEXT NOT NULL DEFAULT ''",
                "final_resolution_reason": "TEXT NOT NULL DEFAULT ''",
                "arbiter_review_enabled": "INTEGER NOT NULL DEFAULT 0",
                "arbiter_review_status": "TEXT NOT NULL DEFAULT ''",
                "arbiter_review_decision": "TEXT NOT NULL DEFAULT ''",
                "arbiter_review_target_action": "TEXT NOT NULL DEFAULT ''",
                "arbiter_review_reason": "TEXT NOT NULL DEFAULT ''",
                "arbiter_review_raw": "TEXT NOT NULL DEFAULT ''",
                "arbiter_review_model_name": "TEXT NOT NULL DEFAULT ''",
                "effective_recommendation": "TEXT NOT NULL DEFAULT ''",
                "effective_stake_pct": "REAL NOT NULL DEFAULT 0",
                "effective_action_source": "TEXT NOT NULL DEFAULT ''",
                "manual_review_status": "TEXT NOT NULL DEFAULT ''",
                "manual_review_reason": "TEXT NOT NULL DEFAULT ''",
                "manual_review_requested_at": "TEXT NOT NULL DEFAULT ''",
                "manual_review_resolved_at": "TEXT NOT NULL DEFAULT ''",
                "manual_review_notes": "TEXT NOT NULL DEFAULT ''",
                "learning_profile_id": "INTEGER NOT NULL DEFAULT 0",
                "calibrated_home_prob": "REAL NOT NULL DEFAULT 0",
                "calibrated_draw_prob": "REAL NOT NULL DEFAULT 0",
                "calibrated_away_prob": "REAL NOT NULL DEFAULT 0",
                "predicted_score": "TEXT NOT NULL DEFAULT ''",
                "predicted_score_confidence": "REAL NOT NULL DEFAULT 0",
                "predicted_score_reason": "TEXT NOT NULL DEFAULT ''",
                "predicted_score_status": "TEXT NOT NULL DEFAULT ''",
                "predicted_score_model_name": "TEXT NOT NULL DEFAULT ''",
                "predicted_score_raw": "TEXT NOT NULL DEFAULT ''",
                "quant_score_candidates": "TEXT NOT NULL DEFAULT ''",
                "handicap_recommendation": "TEXT NOT NULL DEFAULT ''",
                "handicap_recommended_side": "TEXT NOT NULL DEFAULT ''",
                "handicap_line": "REAL NOT NULL DEFAULT 0",
                "handicap_initial_line": "REAL NOT NULL DEFAULT 0",
                "handicap_home_odds": "REAL NOT NULL DEFAULT 0",
                "handicap_away_odds": "REAL NOT NULL DEFAULT 0",
                "handicap_initial_home_odds": "REAL NOT NULL DEFAULT 0",
                "handicap_initial_away_odds": "REAL NOT NULL DEFAULT 0",
                "handicap_home_cover_prob": "REAL NOT NULL DEFAULT 0",
                "handicap_away_cover_prob": "REAL NOT NULL DEFAULT 0",
                "handicap_expected_value": "REAL NOT NULL DEFAULT 0",
                "handicap_confidence": "REAL NOT NULL DEFAULT 0",
                "handicap_reason": "TEXT NOT NULL DEFAULT ''",
            },
        )
        _ensure_columns(
            conn,
            "feedback_logs",
            {
                "roi_source": "TEXT NOT NULL DEFAULT 'auto'",
                "handicap_actual_result": "TEXT NOT NULL DEFAULT ''",
                "handicap_hit": "INTEGER NOT NULL DEFAULT 0",
                "handicap_roi_delta": "REAL NOT NULL DEFAULT 0",
            },
        )
        _ensure_columns(
            conn,
            "learning_profiles",
            {
                "status": "TEXT NOT NULL DEFAULT ''",
                "created_at": "TEXT NOT NULL DEFAULT ''",
                "updated_at": "TEXT NOT NULL DEFAULT ''",
                "activated_at": "TEXT NOT NULL DEFAULT ''",
                "archived_at": "TEXT NOT NULL DEFAULT ''",
                "retention_issue_count": "INTEGER NOT NULL DEFAULT 90",
                "window_type": "TEXT NOT NULL DEFAULT 'rolling_issues'",
                "window_value": "INTEGER NOT NULL DEFAULT 90",
                "total_samples": "INTEGER NOT NULL DEFAULT 0",
                "training_samples": "INTEGER NOT NULL DEFAULT 0",
                "validation_samples": "INTEGER NOT NULL DEFAULT 0",
                "training_action_samples": "INTEGER NOT NULL DEFAULT 0",
                "validation_action_samples": "INTEGER NOT NULL DEFAULT 0",
                "calibrator_status": "TEXT NOT NULL DEFAULT ''",
                "threshold_status": "TEXT NOT NULL DEFAULT ''",
                "calibrator_params": "TEXT NOT NULL DEFAULT ''",
                "threshold_params": "TEXT NOT NULL DEFAULT ''",
                "train_metrics": "TEXT NOT NULL DEFAULT ''",
                "validation_metrics": "TEXT NOT NULL DEFAULT ''",
                "sample_summary": "TEXT NOT NULL DEFAULT ''",
                "notes": "TEXT NOT NULL DEFAULT ''",
            },
        )
        _ensure_indexes(conn)

    _run_write(_operation)


def upsert_matches(matches: list[dict]) -> None:
    def _operation(conn: sqlite3.Connection) -> None:
        normalized_matches = []
        for match in matches:
            payload = dict(match)
            payload.setdefault("yazhi_url", "")
            normalized_matches.append(payload)
        conn.executemany(
            """
            INSERT INTO matches (
                match_id, issue, league, match_no, match_time, home_team, away_team,
                source_match_url, shuju_url, ouzhi_url, touzhu_url, yazhi_url,
                list_odds_win, list_odds_draw, list_odds_loss,
                list_heat_win, list_heat_draw, list_heat_loss, sync_time
            ) VALUES (
                :match_id, :issue, :league, :match_no, :match_time, :home_team, :away_team,
                :source_match_url, :shuju_url, :ouzhi_url, :touzhu_url, :yazhi_url,
                :list_odds_win, :list_odds_draw, :list_odds_loss,
                :list_heat_win, :list_heat_draw, :list_heat_loss, :sync_time
            )
            ON CONFLICT(match_id) DO UPDATE SET
                issue=excluded.issue,
                league=excluded.league,
                match_no=excluded.match_no,
                match_time=excluded.match_time,
                home_team=excluded.home_team,
                away_team=excluded.away_team,
                source_match_url=excluded.source_match_url,
                shuju_url=excluded.shuju_url,
                ouzhi_url=excluded.ouzhi_url,
                touzhu_url=excluded.touzhu_url,
                yazhi_url=excluded.yazhi_url,
                list_odds_win=excluded.list_odds_win,
                list_odds_draw=excluded.list_odds_draw,
                list_odds_loss=excluded.list_odds_loss,
                list_heat_win=excluded.list_heat_win,
                list_heat_draw=excluded.list_heat_draw,
                list_heat_loss=excluded.list_heat_loss,
                sync_time=excluded.sync_time
            """,
            normalized_matches,
        )

    _run_write(_operation)


def upsert_match_results(results: list[dict]) -> None:
    if not results:
        return

    def _operation(conn: sqlite3.Connection) -> None:
        conn.executemany(
            """
            UPDATE matches
            SET
                actual_result = :actual_result,
                actual_score = :actual_score,
                result_status = :result_status,
                result_source_url = :result_source_url,
                result_synced_at = :result_synced_at
            WHERE match_id = :match_id
            """,
            results,
        )

    _run_write(_operation)


def _delete_matches_by_ids(conn: sqlite3.Connection, match_ids: list[str]) -> int:
    if not match_ids:
        return 0
    placeholders = ",".join("?" for _ in match_ids)
    conn.execute(
        f"""
        DELETE FROM feedback_logs
        WHERE match_id IN ({placeholders})
           OR prediction_run_id IN (
                SELECT run_id
                FROM prediction_runs
                WHERE match_id IN ({placeholders})
           )
        """,
        tuple(match_ids + match_ids),
    )
    conn.execute(
        f"""
        DELETE FROM issue_top_picks
        WHERE match_id IN ({placeholders})
           OR run_id IN (
                SELECT run_id
                FROM prediction_runs
                WHERE match_id IN ({placeholders})
           )
        """,
        tuple(match_ids + match_ids),
    )
    conn.execute(f"DELETE FROM prediction_runs WHERE match_id IN ({placeholders})", tuple(match_ids))
    conn.execute(f"DELETE FROM feature_snapshots WHERE match_id IN ({placeholders})", tuple(match_ids))
    conn.execute(f"DELETE FROM analyses WHERE match_id IN ({placeholders})", tuple(match_ids))
    cursor = conn.execute(f"DELETE FROM matches WHERE match_id IN ({placeholders})", tuple(match_ids))
    return int(cursor.rowcount or 0)


def delete_matches(match_ids: list[str]) -> int:
    normalized_ids = []
    seen = set()
    for raw_match_id in match_ids:
        match_id = str(raw_match_id or "").strip()
        if match_id and match_id not in seen:
            normalized_ids.append(match_id)
            seen.add(match_id)
    if not normalized_ids:
        return 0

    return _run_write(lambda conn: _delete_matches_by_ids(conn, normalized_ids))


def delete_custom_match(match_id: str, *, source_match_url: str) -> bool:
    match_id_text = str(match_id or "").strip()
    source_url_text = str(source_match_url or "").strip()
    if not match_id_text or not source_url_text:
        return False

    def _operation(conn: sqlite3.Connection) -> bool:
        row = conn.execute(
            """
            SELECT match_id
            FROM matches
            WHERE match_id = ?
              AND source_match_url = ?
            """,
            (match_id_text, source_url_text),
        ).fetchone()
        if row is None:
            return False
        return _delete_matches_by_ids(conn, [match_id_text]) > 0

    return _run_write(_operation)


def get_match(match_id: str) -> sqlite3.Row | None:
    return _run_read(
        lambda conn: conn.execute(
            "SELECT * FROM matches WHERE match_id = ?", (match_id,)
        ).fetchone()
    )


def save_analysis(analysis: dict) -> None:
    def _operation(conn: sqlite3.Connection) -> None:
        payload = dict(analysis)
        payload.setdefault("market_value_summary", "")
        payload.setdefault("asian_handicap_summary", "")
        payload.setdefault("collection_quality_summary", "")
        conn.execute(
            """
            INSERT INTO analyses (
                match_id, collected_at, elo_home, elo_away, market_value_summary, recent_form_home, recent_form_away,
                home_away_form, head_to_head_summary, injury_or_lineup_notes,
                motivation_or_schedule_notes, european_odds_movement_summary,
                asian_handicap_summary,
                betting_heat_summary, media_source_links, collected_sources, collection_quality_summary, remarks
            ) VALUES (
                :match_id, :collected_at, :elo_home, :elo_away, :market_value_summary, :recent_form_home, :recent_form_away,
                :home_away_form, :head_to_head_summary, :injury_or_lineup_notes,
                :motivation_or_schedule_notes, :european_odds_movement_summary,
                :asian_handicap_summary,
                :betting_heat_summary, :media_source_links, :collected_sources, :collection_quality_summary, :remarks
            )
            ON CONFLICT(match_id) DO UPDATE SET
                collected_at=excluded.collected_at,
                elo_home=excluded.elo_home,
                elo_away=excluded.elo_away,
                market_value_summary=excluded.market_value_summary,
                recent_form_home=excluded.recent_form_home,
                recent_form_away=excluded.recent_form_away,
                home_away_form=excluded.home_away_form,
                head_to_head_summary=excluded.head_to_head_summary,
                injury_or_lineup_notes=excluded.injury_or_lineup_notes,
                motivation_or_schedule_notes=excluded.motivation_or_schedule_notes,
                european_odds_movement_summary=excluded.european_odds_movement_summary,
                asian_handicap_summary=excluded.asian_handicap_summary,
                betting_heat_summary=excluded.betting_heat_summary,
                media_source_links=excluded.media_source_links,
                collected_sources=excluded.collected_sources,
                collection_quality_summary=excluded.collection_quality_summary,
                remarks=excluded.remarks
            """,
            payload,
        )

    _run_write(_operation)


def save_failed_analysis(match_id: str, remarks: str, failed_at: str) -> None:
    def _operation(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            INSERT INTO analyses (match_id, collected_at, collection_quality_summary, remarks)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(match_id) DO UPDATE SET
                collected_at=excluded.collected_at,
                collection_quality_summary=excluded.collection_quality_summary,
                remarks=excluded.remarks
            """,
            (match_id, failed_at, "", remarks),
        )

    _run_write(_operation)


def save_feature_snapshot(snapshot: dict[str, Any]) -> int:
    def _operation(conn: sqlite3.Connection) -> int:
        # Only one feature snapshot per match is meaningful at a time —
        # ``get_latest_feature_snapshot`` always reads MAX(snapshot_id), so
        # older rows are dead data. Drop them before inserting to keep the
        # table from growing on every re-collect.
        match_id = snapshot.get("match_id")
        if match_id:
            conn.execute(
                "DELETE FROM feature_snapshots WHERE match_id = ?",
                (match_id,),
            )
        cursor = conn.execute(
            """
            INSERT INTO feature_snapshots (
                match_id, issue, snapshot_at,
                home_rating, away_rating,
                recent_home_ppg, recent_away_ppg,
                recent_home_gf_pg, recent_away_gf_pg,
                recent_home_ga_pg, recent_away_ga_pg,
                home_split_ppg, away_split_ppg,
                home_absent_count, away_absent_count,
                home_doubtful_count, away_doubtful_count,
                home_absence_impact, away_absence_impact,
                lineup_home_availability, lineup_away_availability,
                rest_days_home, rest_days_away,
                schedule_load_home, schedule_load_away,
                h2h_edge,
                market_home_prob, market_draw_prob, market_away_prob,
                feature_payload
            ) VALUES (
                :match_id, :issue, :snapshot_at,
                :home_rating, :away_rating,
                :recent_home_ppg, :recent_away_ppg,
                :recent_home_gf_pg, :recent_away_gf_pg,
                :recent_home_ga_pg, :recent_away_ga_pg,
                :home_split_ppg, :away_split_ppg,
                :home_absent_count, :away_absent_count,
                :home_doubtful_count, :away_doubtful_count,
                :home_absence_impact, :away_absence_impact,
                :lineup_home_availability, :lineup_away_availability,
                :rest_days_home, :rest_days_away,
                :schedule_load_home, :schedule_load_away,
                :h2h_edge,
                :market_home_prob, :market_draw_prob, :market_away_prob,
                :feature_payload
            )
            """,
            snapshot,
        )
        return int(cursor.lastrowid)

    return _run_write(_operation)


def get_latest_feature_snapshot(match_id: str) -> sqlite3.Row | None:
    return _run_read(
        lambda conn: conn.execute(
            """
            SELECT *
            FROM feature_snapshots
            WHERE match_id = ?
            ORDER BY snapshot_id DESC
            LIMIT 1
            """,
            (match_id,),
        ).fetchone()
    )


def list_feature_snapshots(
    match_id: str | None = None,
    limit: int = 50,
) -> list[sqlite3.Row]:
    def _operation(conn: sqlite3.Connection) -> list[sqlite3.Row]:
        if match_id:
            return conn.execute(
                """
                SELECT *
                FROM feature_snapshots
                WHERE match_id = ?
                ORDER BY snapshot_id DESC
                LIMIT ?
                """,
                (match_id, limit),
            ).fetchall()
        return conn.execute(
            """
            SELECT *
            FROM feature_snapshots
            ORDER BY snapshot_id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return _run_read(_operation)


def list_matches() -> list[sqlite3.Row]:
    return _run_read(
        lambda conn: conn.execute(
            f"""
            SELECT m.*, a.collected_at, a.remarks, {_collection_status_sql("a")}
            FROM matches m
            LEFT JOIN analyses a ON a.match_id = m.match_id
            ORDER BY CAST(m.match_no AS INTEGER), m.match_id
            """
        ).fetchall()
    )


def list_issues() -> list[str]:
    def _operation(conn: sqlite3.Connection) -> list[str]:
        rows = conn.execute(
            "SELECT DISTINCT issue FROM matches WHERE issue <> '' ORDER BY issue DESC"
        ).fetchall()
        return [row[0] for row in rows]

    return _run_read(_operation)


def get_latest_issue() -> str:
    def _operation(conn: sqlite3.Connection) -> str:
        row = conn.execute(
            """
            SELECT issue
            FROM matches
            WHERE TRIM(issue) <> ''
            GROUP BY issue
            ORDER BY issue DESC
            LIMIT 1
            """
        ).fetchone()
        return row[0] if row else ""

    return _run_read(_operation)


def list_recent_issues(limit: int = DEFAULT_ISSUE_RETENTION_COUNT) -> list[str]:
    limit = max(int(limit or 0), 0)
    if limit <= 0:
        return []

    def _operation(conn: sqlite3.Connection) -> list[str]:
        rows = conn.execute(
            """
            SELECT issue
            FROM matches
            WHERE TRIM(issue) <> ''
            GROUP BY issue
            ORDER BY issue DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [str(row[0]).strip() for row in rows if str(row[0] or "").strip()]

    return _run_read(_operation)


def _empty_prune_result(keep_issues: list[str]) -> dict[str, Any]:
    kept_issue = keep_issues[0] if len(keep_issues) == 1 else ""
    return {
        "kept_issue": kept_issue,
        "kept_issues": keep_issues,
        "retained_issue_count": len(keep_issues),
        "deleted_issue_count": 0,
        "deleted_matches": 0,
        "deleted_analyses": 0,
        "deleted_feature_snapshots": 0,
        "deleted_prediction_runs": 0,
        "deleted_feedback_logs": 0,
        "deleted_total": 0,
        "trimmed": False,
    }


def _prune_to_issues(keep_issues: list[str]) -> dict[str, Any]:
    normalized_keep_issues: list[str] = []
    seen_issues: set[str] = set()
    for issue in keep_issues:
        issue_text = str(issue or "").strip()
        if not issue_text or issue_text in seen_issues:
            continue
        normalized_keep_issues.append(issue_text)
        seen_issues.add(issue_text)

    if not normalized_keep_issues:
        return _empty_prune_result([])

    issue_placeholders = ", ".join("?" for _ in normalized_keep_issues)
    keep_matches_subquery = (
        f"SELECT match_id FROM matches WHERE issue IN ({issue_placeholders})"
    )
    keep_issues_params = tuple(normalized_keep_issues)
    keep_issues_params_twice = keep_issues_params + keep_issues_params

    def _operation(conn: sqlite3.Connection) -> dict[str, Any]:
        stored_issue_rows = conn.execute(
            """
            SELECT issue
            FROM matches
            WHERE TRIM(issue) <> ''
            GROUP BY issue
            ORDER BY issue DESC
            """
        ).fetchall()
        stored_issues = [str(row[0]).strip() for row in stored_issue_rows if str(row[0] or "").strip()]
        existing_keep_issues = [issue for issue in normalized_keep_issues if issue in stored_issues]
        if not existing_keep_issues:
            return _empty_prune_result(normalized_keep_issues)

        deleted_matches = conn.execute(
            f"SELECT COUNT(*) FROM matches WHERE issue NOT IN ({issue_placeholders})",
            keep_issues_params,
        ).fetchone()[0]
        deleted_analyses = conn.execute(
            f"""
            SELECT COUNT(*)
            FROM analyses
            WHERE match_id NOT IN ({keep_matches_subquery})
            """,
            keep_issues_params,
        ).fetchone()[0]
        deleted_prediction_runs = conn.execute(
            f"""
            SELECT COUNT(*)
            FROM prediction_runs
            WHERE match_id NOT IN ({keep_matches_subquery})
            """,
            keep_issues_params,
        ).fetchone()[0]
        deleted_feature_snapshots = conn.execute(
            f"""
            SELECT COUNT(*)
            FROM feature_snapshots
            WHERE match_id NOT IN ({keep_matches_subquery})
            """,
            keep_issues_params,
        ).fetchone()[0]
        deleted_feedback_logs = conn.execute(
            f"""
            SELECT COUNT(*)
            FROM feedback_logs
            WHERE match_id NOT IN ({keep_matches_subquery})
               OR prediction_run_id IN (
                    SELECT run_id
                    FROM prediction_runs
                    WHERE match_id NOT IN ({keep_matches_subquery})
               )
            """,
            keep_issues_params_twice,
        ).fetchone()[0]

        conn.execute(
            f"""
            DELETE FROM feedback_logs
            WHERE match_id NOT IN ({keep_matches_subquery})
               OR prediction_run_id IN (
                    SELECT run_id
                    FROM prediction_runs
                    WHERE match_id NOT IN ({keep_matches_subquery})
               )
            """,
            keep_issues_params_twice,
        )
        conn.execute(
            f"""
            DELETE FROM prediction_runs
            WHERE match_id NOT IN ({keep_matches_subquery})
            """,
            keep_issues_params,
        )
        conn.execute(
            f"""
            DELETE FROM feature_snapshots
            WHERE match_id NOT IN ({keep_matches_subquery})
            """,
            keep_issues_params,
        )
        conn.execute(
            f"""
            DELETE FROM analyses
            WHERE match_id NOT IN ({keep_matches_subquery})
            """,
            keep_issues_params,
        )
        conn.execute(
            f"DELETE FROM matches WHERE issue NOT IN ({issue_placeholders})",
            keep_issues_params,
        )

        deleted_issue_count = max(len(stored_issues) - len(existing_keep_issues), 0)
        deleted_total = (
            deleted_matches
            + deleted_analyses
            + deleted_feature_snapshots
            + deleted_prediction_runs
            + deleted_feedback_logs
        )
        return {
            "kept_issue": existing_keep_issues[0] if len(existing_keep_issues) == 1 else "",
            "kept_issues": existing_keep_issues,
            "retained_issue_count": len(existing_keep_issues),
            "deleted_issue_count": deleted_issue_count,
            "deleted_matches": deleted_matches,
            "deleted_analyses": deleted_analyses,
            "deleted_feature_snapshots": deleted_feature_snapshots,
            "deleted_prediction_runs": deleted_prediction_runs,
            "deleted_feedback_logs": deleted_feedback_logs,
            "deleted_total": deleted_total,
            "trimmed": bool(deleted_issue_count or deleted_total),
        }

    return _run_write(_operation)


def prune_to_issue(keep_issue: str) -> dict[str, Any]:
    keep_issue = str(keep_issue or "").strip()
    if not keep_issue:
        return _empty_prune_result([])
    return _prune_to_issues([keep_issue])


def prune_to_recent_issues(
    keep_count: int = DEFAULT_ISSUE_RETENTION_COUNT,
) -> dict[str, Any]:
    keep_count = max(int(keep_count or 0), 0)
    if keep_count <= 0:
        return _empty_prune_result([])

    return _prune_to_issues(list_recent_issues(keep_count))


def list_matches_by_issue(issue: str | None = None) -> list[sqlite3.Row]:
    def _operation(conn: sqlite3.Connection) -> list[sqlite3.Row]:
        if issue:
            return conn.execute(
                f"""
                SELECT m.*, a.collected_at, a.remarks, {_collection_status_sql("a")}
                FROM matches m
                LEFT JOIN analyses a ON a.match_id = m.match_id
                WHERE m.issue = ?
                ORDER BY CAST(m.match_no AS INTEGER), m.match_id
                """,
                (issue,),
            ).fetchall()
        return conn.execute(
            f"""
            SELECT m.*, a.collected_at, a.remarks, {_collection_status_sql("a")}
            FROM matches m
            LEFT JOIN analyses a ON a.match_id = m.match_id
            ORDER BY CAST(m.match_no AS INTEGER), m.match_id
            """
        ).fetchall()

    return _run_read(_operation)


def list_matches_pending_settlement(issue: str | None = None) -> list[sqlite3.Row]:
    return list_matches_by_issue(issue)


def get_match_analysis(match_id: str) -> sqlite3.Row | None:
    return _run_read(
        lambda conn: conn.execute(
            f"""
            SELECT
                m.match_id, m.issue, m.league, m.match_no, m.match_time, m.home_team, m.away_team,
                m.source_match_url, m.shuju_url, m.ouzhi_url, m.touzhu_url, m.yazhi_url, m.sync_time,
                m.actual_result, m.actual_score, m.result_status, m.result_source_url, m.result_synced_at,
                a.collected_at, a.elo_home, a.elo_away, a.recent_form_home, a.recent_form_away,
                a.market_value_summary,
                a.home_away_form, a.head_to_head_summary, a.injury_or_lineup_notes,
                a.motivation_or_schedule_notes, a.european_odds_movement_summary,
                a.asian_handicap_summary,
                a.betting_heat_summary, a.media_source_links, a.collected_sources,
                a.collection_quality_summary, a.remarks,
                {_collection_status_sql("a")}
            FROM matches m
            LEFT JOIN analyses a ON a.match_id = m.match_id
            WHERE m.match_id = ?
            """,
            (match_id,),
        ).fetchone()
    )


def get_collection_stats(issue: str | None = None) -> dict:
    success_condition = _analysis_success_condition("a")

    def _operation(conn: sqlite3.Connection) -> dict:
        if issue:
            total_matches = conn.execute(
                "SELECT COUNT(*) FROM matches WHERE issue = ?", (issue,)
            ).fetchone()[0]
            total_analyses = conn.execute(
                """
                SELECT COUNT(*)
                FROM analyses a
                JOIN matches m ON m.match_id = a.match_id
                WHERE m.issue = ?
                """,
                (issue,),
            ).fetchone()[0]
            success_analyses = conn.execute(
                f"""
                SELECT COUNT(*)
                FROM analyses a
                JOIN matches m ON m.match_id = a.match_id
                WHERE m.issue = ?
                  AND {success_condition}
                """,
                (issue,),
            ).fetchone()[0]
        else:
            total_matches = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
            total_analyses = conn.execute(
                "SELECT COUNT(*) FROM analyses a"
            ).fetchone()[0]
            success_analyses = conn.execute(
                f"""
                SELECT COUNT(*)
                FROM analyses a
                WHERE {success_condition}
                """
            ).fetchone()[0]
        failed_analyses = max(int(total_analyses or 0) - int(success_analyses or 0), 0)
        return {
            "total_matches": total_matches,
            "total_analyses": total_analyses,
            "success_analyses": success_analyses,
            "failed_analyses": failed_analyses,
        }

    return _run_read(_operation)


def serialize_match(row: sqlite3.Row) -> dict:
    return dict(row) if row is not None else {}


def save_prediction_run(run: dict) -> int:
    def _operation(conn: sqlite3.Connection) -> int:
        match_id = str(run.get("match_id", "") or "")
        issue = str(run.get("issue", "") or "")
        payload = dict(run)
        payload.setdefault("predicted_score", "")
        payload.setdefault("predicted_score_confidence", 0.0)
        payload.setdefault("predicted_score_reason", "")
        payload.setdefault("predicted_score_status", "")
        payload.setdefault("predicted_score_model_name", "")
        payload.setdefault("predicted_score_raw", "")
        payload.setdefault("quant_score_candidates", "")
        payload.setdefault("handicap_recommendation", "")
        payload.setdefault("handicap_recommended_side", "")
        payload.setdefault("handicap_line", 0.0)
        payload.setdefault("handicap_initial_line", 0.0)
        payload.setdefault("handicap_home_odds", 0.0)
        payload.setdefault("handicap_away_odds", 0.0)
        payload.setdefault("handicap_initial_home_odds", 0.0)
        payload.setdefault("handicap_initial_away_odds", 0.0)
        payload.setdefault("handicap_home_cover_prob", 0.0)
        payload.setdefault("handicap_away_cover_prob", 0.0)
        payload.setdefault("handicap_expected_value", 0.0)
        payload.setdefault("handicap_confidence", 0.0)
        payload.setdefault("handicap_reason", "")
        carried_feedback: dict[str, Any] | None = None
        if match_id:
            feedback_row = conn.execute(
                """
                SELECT *
                FROM feedback_logs
                WHERE match_id = ?
                ORDER BY feedback_id DESC
                LIMIT 1
                """,
                (match_id,),
            ).fetchone()
            if feedback_row is not None:
                carried_feedback = {
                    "match_id": str(feedback_row["match_id"] or ""),
                    "actual_result": str(feedback_row["actual_result"] or ""),
                    "actual_score": str(feedback_row["actual_score"] or ""),
                    "settled_at": str(feedback_row["settled_at"] or ""),
                    "roi_delta": float(feedback_row["roi_delta"] or 0.0),
                    "handicap_actual_result": str(feedback_row["handicap_actual_result"] or "") if "handicap_actual_result" in feedback_row.keys() else "",
                    "handicap_hit": int(feedback_row["handicap_hit"] or 0) if "handicap_hit" in feedback_row.keys() else 0,
                    "handicap_roi_delta": float(feedback_row["handicap_roi_delta"] or 0.0) if "handicap_roi_delta" in feedback_row.keys() else 0.0,
                    "roi_source": str(feedback_row["roi_source"] or "auto"),
                    "notes": str(feedback_row["notes"] or ""),
                }
            conn.execute(
                """
                DELETE FROM issue_top_picks
                WHERE match_id = ?
                   OR run_id IN (
                        SELECT run_id
                        FROM prediction_runs
                        WHERE match_id = ?
                   )
                   OR (? <> '' AND issue = ?)
                """,
                (match_id, match_id, issue, issue),
            )
            conn.execute(
                """
                DELETE FROM feedback_logs
                WHERE match_id = ?
                   OR prediction_run_id IN (
                        SELECT run_id
                        FROM prediction_runs
                        WHERE match_id = ?
                   )
                """,
                (match_id, match_id),
            )
            conn.execute(
                "DELETE FROM prediction_runs WHERE match_id = ?",
                (match_id,),
            )
        cursor = conn.execute(
            """
            INSERT INTO prediction_runs (
                match_id, issue, created_at,
                feature_snapshot_id,
                quant_home_prob, quant_draw_prob, quant_away_prob,
                ml_home_prob, ml_draw_prob, ml_away_prob,
                legacy_home_prob, legacy_draw_prob, legacy_away_prob,
                final_home_prob, final_draw_prob, final_away_prob,
                fair_odds_home, fair_odds_draw, fair_odds_away,
                market_odds_home, market_odds_draw, market_odds_away,
                market_home_prob, market_draw_prob, market_away_prob,
                ev_home, ev_draw, ev_away,
                quality_score, model_agreement, confidence_score,
                risk_level, recommendation, recommended_outcome,
                suggested_stake_pct,
                handicap_recommendation, handicap_recommended_side,
                handicap_line, handicap_initial_line,
                handicap_home_odds, handicap_away_odds,
                handicap_initial_home_odds, handicap_initial_away_odds,
                handicap_home_cover_prob, handicap_away_cover_prob,
                handicap_expected_value, handicap_confidence, handicap_reason,
                algo_recommendation, algo_recommended_outcome,
                algo_risk_level, algo_suggested_stake_pct,
                llm_review_enabled, llm_review_status, llm_review_decision,
                llm_review_target_action, llm_review_reason, llm_review_raw,
                review_model_name, final_resolution_reason,
                arbiter_review_enabled, arbiter_review_status, arbiter_review_decision,
                arbiter_review_target_action, arbiter_review_reason, arbiter_review_raw,
                arbiter_review_model_name,
                effective_recommendation, effective_stake_pct, effective_action_source,
                manual_review_status, manual_review_reason,
                manual_review_requested_at, manual_review_resolved_at, manual_review_notes,
                llm_provider, llm_model, llm_summary, final_report,
                learning_profile_id,
                calibrated_home_prob, calibrated_draw_prob, calibrated_away_prob,
                predicted_score, predicted_score_confidence, predicted_score_reason,
                predicted_score_status, predicted_score_model_name,
                predicted_score_raw, quant_score_candidates
            ) VALUES (
                :match_id, :issue, :created_at,
                :feature_snapshot_id,
                :quant_home_prob, :quant_draw_prob, :quant_away_prob,
                :ml_home_prob, :ml_draw_prob, :ml_away_prob,
                :legacy_home_prob, :legacy_draw_prob, :legacy_away_prob,
                :final_home_prob, :final_draw_prob, :final_away_prob,
                :fair_odds_home, :fair_odds_draw, :fair_odds_away,
                :market_odds_home, :market_odds_draw, :market_odds_away,
                :market_home_prob, :market_draw_prob, :market_away_prob,
                :ev_home, :ev_draw, :ev_away,
                :quality_score, :model_agreement, :confidence_score,
                :risk_level, :recommendation, :recommended_outcome,
                :suggested_stake_pct,
                :handicap_recommendation, :handicap_recommended_side,
                :handicap_line, :handicap_initial_line,
                :handicap_home_odds, :handicap_away_odds,
                :handicap_initial_home_odds, :handicap_initial_away_odds,
                :handicap_home_cover_prob, :handicap_away_cover_prob,
                :handicap_expected_value, :handicap_confidence, :handicap_reason,
                :algo_recommendation, :algo_recommended_outcome,
                :algo_risk_level, :algo_suggested_stake_pct,
                :llm_review_enabled, :llm_review_status, :llm_review_decision,
                :llm_review_target_action, :llm_review_reason, :llm_review_raw,
                :review_model_name, :final_resolution_reason,
                :arbiter_review_enabled, :arbiter_review_status, :arbiter_review_decision,
                :arbiter_review_target_action, :arbiter_review_reason, :arbiter_review_raw,
                :arbiter_review_model_name,
                :effective_recommendation, :effective_stake_pct, :effective_action_source,
                :manual_review_status, :manual_review_reason,
                :manual_review_requested_at, :manual_review_resolved_at, :manual_review_notes,
                :llm_provider, :llm_model, :llm_summary, :final_report,
                :learning_profile_id,
                :calibrated_home_prob, :calibrated_draw_prob, :calibrated_away_prob,
                :predicted_score, :predicted_score_confidence, :predicted_score_reason,
                :predicted_score_status, :predicted_score_model_name,
                :predicted_score_raw, :quant_score_candidates
            )
            """,
            payload,
        )
        run_id = int(cursor.lastrowid)
        if carried_feedback is not None:
            actual_result = carried_feedback["actual_result"]
            recommended_outcome = str(payload.get("recommended_outcome", "") or "")
            hit = 1 if recommended_outcome == actual_result else 0
            handicap_side = str(payload.get("handicap_recommended_side", "") or "")
            handicap_line = float(payload.get("handicap_line") or 0.0)
            handicap_result = _settle_handicap_result_from_score(
                str(carried_feedback.get("actual_score", "") or ""),
                handicap_line,
            )
            handicap_hit = 1 if handicap_side and handicap_side == handicap_result else 0
            if carried_feedback["roi_source"] == "manual_override":
                roi_delta = float(carried_feedback.get("roi_delta", 0.0) or 0.0)
                handicap_roi_delta = float(carried_feedback.get("handicap_roi_delta", 0.0) or 0.0)
            else:
                action = str(payload.get("effective_recommendation") or payload.get("recommendation") or "")
                stake_pct = float(payload.get("effective_stake_pct") or payload.get("suggested_stake_pct") or 0.0)
                odds_key = f"market_odds_{recommended_outcome}"
                odds = float(payload.get(odds_key) or 0.0)
                if action == "观望" or stake_pct <= 0 or recommended_outcome not in {"home", "draw", "away"}:
                    roi_delta = 0.0
                elif recommended_outcome == actual_result:
                    roi_delta = round((odds - 1.0) * stake_pct / 100.0, 4)
                else:
                    roi_delta = round(-stake_pct / 100.0, 4)
                handicap_odds = float(payload.get(f"handicap_{handicap_side}_odds") or 0.0)
                if not handicap_side or handicap_result not in {"home", "away"} or handicap_odds <= 0:
                    handicap_roi_delta = 0.0
                elif handicap_side == handicap_result:
                    handicap_roi_delta = round((handicap_odds - 1.0) * stake_pct / 100.0, 4)
                else:
                    handicap_roi_delta = round(-stake_pct / 100.0, 4)
            conn.execute(
                """
                INSERT INTO feedback_logs (
                    prediction_run_id, match_id, actual_result, actual_score,
                    settled_at, hit_recommendation, roi_delta,
                    handicap_actual_result, handicap_hit, handicap_roi_delta,
                    roi_source, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    match_id,
                    actual_result,
                    carried_feedback["actual_score"],
                    carried_feedback["settled_at"],
                    hit,
                    roi_delta,
                    handicap_result,
                    handicap_hit,
                    handicap_roi_delta,
                    carried_feedback["roi_source"],
                    carried_feedback["notes"],
                ),
            )
        return run_id

    return _run_write(_operation)


def get_prediction_run(run_id: int) -> sqlite3.Row | None:
    return _run_read(
        lambda conn: conn.execute(
            "SELECT * FROM prediction_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
    )


def list_prediction_runs(
    match_id: str | None = None, limit: int | None = 20
) -> list[sqlite3.Row]:
    def _operation(conn: sqlite3.Connection) -> list[sqlite3.Row]:
        if match_id:
            if limit is None:
                return conn.execute(
                    """
                    SELECT *
                    FROM prediction_runs
                    WHERE match_id = ?
                    ORDER BY run_id DESC
                    """,
                    (match_id,),
                ).fetchall()
            return conn.execute(
                """
                SELECT *
                FROM prediction_runs
                WHERE match_id = ?
                ORDER BY run_id DESC
                LIMIT ?
                """,
                (match_id, limit),
            ).fetchall()
        if limit is None:
            return conn.execute(
                """
                SELECT *
                FROM prediction_runs
                ORDER BY run_id DESC
                """
            ).fetchall()
        return conn.execute(
            """
            SELECT *
            FROM prediction_runs
            ORDER BY run_id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return _run_read(_operation)


_PREDICTION_RUN_MUTABLE_FIELDS = {
    "arbiter_review_enabled": int,
    "arbiter_review_status": str,
    "arbiter_review_decision": str,
    "arbiter_review_target_action": str,
    "arbiter_review_reason": str,
    "arbiter_review_raw": str,
    "arbiter_review_model_name": str,
    "recommendation": str,
    "recommended_outcome": str,
    "suggested_stake_pct": float,
    "effective_recommendation": str,
    "effective_stake_pct": float,
    "effective_action_source": str,
    "manual_review_status": str,
    "manual_review_reason": str,
    "manual_review_requested_at": str,
    "manual_review_resolved_at": str,
    "manual_review_notes": str,
    "predicted_score": str,
    "predicted_score_confidence": float,
    "predicted_score_reason": str,
    "predicted_score_status": str,
    "predicted_score_model_name": str,
    "predicted_score_raw": str,
    "quant_score_candidates": str,
}


def update_prediction_run_fields(run_id: int, updates: dict[str, Any]) -> None:
    filtered_items: list[tuple[str, Any]] = []
    for field_name, caster in _PREDICTION_RUN_MUTABLE_FIELDS.items():
        if field_name not in updates:
            continue
        value = updates[field_name]
        if caster is float:
            filtered_items.append((field_name, float(value or 0.0)))
        elif caster is int:
            filtered_items.append((field_name, int(value or 0)))
        else:
            filtered_items.append((field_name, str(value or "")))

    if not filtered_items:
        return

    def _operation(conn: sqlite3.Connection) -> None:
        assignments = ", ".join(f"{field_name} = ?" for field_name, _value in filtered_items)
        values = [value for _field_name, value in filtered_items]
        values.append(int(run_id))
        conn.execute(
            f"""
            UPDATE prediction_runs
            SET {assignments}
            WHERE run_id = ?
            """,
            tuple(values),
        )

    _run_write(_operation)


def supersede_pending_manual_reviews(
    match_id: str,
    *,
    exclude_run_id: int = 0,
    resolved_at: str = "",
) -> int:
    resolved_at_text = str(resolved_at or "").strip()

    def _operation(conn: sqlite3.Connection) -> int:
        params: list[Any] = [
            "superseded",
            "有更新预测 run 生成，旧人工复核任务自动失效",
            resolved_at_text,
            str(match_id),
        ]
        exclude_clause = ""
        if exclude_run_id > 0:
            exclude_clause = " AND run_id <> ?"
            params.append(int(exclude_run_id))
        cursor = conn.execute(
            f"""
            UPDATE prediction_runs
            SET manual_review_status = ?,
                manual_review_reason = CASE
                    WHEN IFNULL(manual_review_reason, '') = '' THEN ?
                    ELSE manual_review_reason
                END,
                manual_review_resolved_at = CASE
                    WHEN IFNULL(manual_review_resolved_at, '') = '' THEN ?
                    ELSE manual_review_resolved_at
                END
            WHERE match_id = ?
              AND manual_review_status = 'pending'
              {exclude_clause}
            """,
            tuple(params),
        )
        return int(cursor.rowcount or 0)

    return _run_write(_operation)


def expire_pending_manual_reviews(
    *,
    match_id: str | None = None,
    now_text: str = "",
) -> int:
    timestamp = str(now_text or "").strip()
    if not timestamp:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _operation(conn: sqlite3.Connection) -> int:
        match_filter = ""
        params: list[Any] = [
            "expired",
            "已开赛且未完成人工复核，执行动作自动失效",
            timestamp,
            timestamp,
        ]
        if match_id:
            match_filter = " AND match_id = ?"
            params.append(str(match_id))
        cursor = conn.execute(
            f"""
            UPDATE prediction_runs
            SET manual_review_status = ?,
                manual_review_reason = CASE
                    WHEN IFNULL(manual_review_reason, '') = '' THEN ?
                    ELSE manual_review_reason
                END,
                manual_review_resolved_at = CASE
                    WHEN IFNULL(manual_review_resolved_at, '') = '' THEN ?
                    ELSE manual_review_resolved_at
                END
            WHERE manual_review_status = 'pending'
              AND match_id IN (
                  SELECT match_id
                  FROM matches
                  WHERE IFNULL(match_time, '') <> ''
                    AND match_time <= ?
                    {match_filter}
              )
            """,
            tuple(params),
        )
        return int(cursor.rowcount or 0)

    return _run_write(_operation)


def list_backtest_rows(
    *,
    league: str = "",
    month: str = "",
    odds_min: float | None = None,
    odds_max: float | None = None,
    confidence_min: float | None = None,
    ev_min: float | None = None,
    limit: int | None = None,
) -> list[sqlite3.Row]:
    def _operation(conn: sqlite3.Connection) -> list[sqlite3.Row]:
        rows = conn.execute(
            """
            SELECT
                f.feedback_id,
                f.prediction_run_id,
                f.match_id,
                f.actual_result,
                f.actual_score,
                f.settled_at,
                f.hit_recommendation,
                f.roi_delta,
                f.handicap_actual_result,
                f.handicap_hit,
                f.handicap_roi_delta,
                f.roi_source,
                f.notes,
                p.issue,
                p.created_at,
                p.feature_snapshot_id,
                p.quality_score,
                p.model_agreement,
                p.legacy_home_prob,
                p.legacy_draw_prob,
                p.legacy_away_prob,
                p.final_home_prob,
                p.final_draw_prob,
                p.final_away_prob,
                p.calibrated_home_prob,
                p.calibrated_draw_prob,
                p.calibrated_away_prob,
                p.predicted_score,
                p.predicted_score_confidence,
                p.predicted_score_reason,
                p.predicted_score_status,
                p.predicted_score_model_name,
                p.quant_score_candidates,
                p.learning_profile_id,
                p.market_home_prob,
                p.market_draw_prob,
                p.market_away_prob,
                p.market_odds_home,
                p.market_odds_draw,
                p.market_odds_away,
                p.ev_home,
                p.ev_draw,
                p.ev_away,
                p.confidence_score,
                p.algo_recommendation,
                p.algo_recommended_outcome,
                p.algo_risk_level,
                p.algo_suggested_stake_pct,
                p.recommendation,
                p.recommended_outcome,
                p.risk_level,
                p.suggested_stake_pct,
                p.handicap_recommendation,
                p.handicap_recommended_side,
                p.handicap_line,
                p.handicap_initial_line,
                p.handicap_home_odds,
                p.handicap_away_odds,
                p.handicap_initial_home_odds,
                p.handicap_initial_away_odds,
                p.handicap_home_cover_prob,
                p.handicap_away_cover_prob,
                p.handicap_expected_value,
                p.handicap_confidence,
                p.handicap_reason,
                p.llm_review_enabled,
                p.llm_review_status,
                p.llm_review_decision,
                p.llm_review_target_action,
                p.llm_review_reason,
                p.arbiter_review_enabled,
                p.arbiter_review_status,
                p.arbiter_review_decision,
                p.arbiter_review_target_action,
                p.arbiter_review_reason,
                p.effective_recommendation,
                p.effective_stake_pct,
                p.effective_action_source,
                p.manual_review_status,
                p.manual_review_reason,
                p.manual_review_requested_at,
                p.manual_review_resolved_at,
                p.manual_review_notes,
                p.final_resolution_reason,
                fs.home_rating,
                fs.away_rating,
                fs.recent_home_ppg,
                fs.recent_away_ppg,
                fs.recent_home_gf_pg,
                fs.recent_away_gf_pg,
                fs.recent_home_ga_pg,
                fs.recent_away_ga_pg,
                fs.home_split_ppg,
                fs.away_split_ppg,
                fs.home_absent_count,
                fs.away_absent_count,
                fs.home_absence_impact,
                fs.away_absence_impact,
                fs.lineup_home_availability,
                fs.lineup_away_availability,
                fs.rest_days_home,
                fs.rest_days_away,
                fs.schedule_load_home,
                fs.schedule_load_away,
                fs.h2h_edge,
                fs.feature_payload,
                a.market_value_summary,
                a.recent_form_home AS analysis_recent_form_home,
                a.recent_form_away AS analysis_recent_form_away,
                a.home_away_form AS analysis_home_away_form,
                a.head_to_head_summary AS analysis_head_to_head_summary,
                a.injury_or_lineup_notes,
                a.motivation_or_schedule_notes,
                a.european_odds_movement_summary,
                a.asian_handicap_summary,
                a.betting_heat_summary,
                a.collected_sources,
                a.collection_quality_summary,
                m.league,
                m.match_time,
                m.home_team,
                m.away_team,
                m.list_odds_win,
                m.list_odds_draw,
                m.list_odds_loss,
                m.list_heat_win,
                m.list_heat_draw,
                m.list_heat_loss,
                m.result_status
            FROM feedback_logs f
            JOIN prediction_runs p ON p.run_id = f.prediction_run_id
            LEFT JOIN feature_snapshots fs ON fs.snapshot_id = p.feature_snapshot_id
            LEFT JOIN analyses a ON a.match_id = f.match_id
            JOIN matches m ON m.match_id = f.match_id
            ORDER BY f.feedback_id DESC
            """
        ).fetchall()

        filtered: list[sqlite3.Row] = []
        for row in rows:
            if league and str(row["league"] or "") != league:
                continue
            if month and not str(row["created_at"] or "").startswith(month):
                continue

            recommended_outcome = str(row["recommended_outcome"] or "")
            recommended_odds = 0.0
            recommended_ev = 0.0
            if recommended_outcome == "home":
                recommended_odds = float(row["market_odds_home"] or 0.0)
                recommended_ev = float(row["ev_home"] or 0.0)
            elif recommended_outcome == "draw":
                recommended_odds = float(row["market_odds_draw"] or 0.0)
                recommended_ev = float(row["ev_draw"] or 0.0)
            elif recommended_outcome == "away":
                recommended_odds = float(row["market_odds_away"] or 0.0)
                recommended_ev = float(row["ev_away"] or 0.0)

            if odds_min is not None and recommended_odds < odds_min:
                continue
            if odds_max is not None and recommended_odds > odds_max:
                continue
            if confidence_min is not None and float(row["confidence_score"] or 0.0) < confidence_min:
                continue
            if ev_min is not None and recommended_ev < ev_min:
                continue

            filtered.append(row)
            if limit is not None and len(filtered) >= limit:
                break
        return filtered

    return _run_read(_operation)


def list_pending_manual_review_runs(
    issue: str | None = None,
    *,
    limit: int | None = 20,
) -> list[sqlite3.Row]:
    def _operation(conn: sqlite3.Connection) -> list[sqlite3.Row]:
        query = """
            SELECT
                p.run_id,
                p.match_id,
                p.issue,
                p.created_at,
                p.recommendation,
                p.recommended_outcome,
                p.suggested_stake_pct,
                p.effective_recommendation,
                p.effective_stake_pct,
                p.effective_action_source,
                p.arbiter_review_status,
                p.arbiter_review_decision,
                p.arbiter_review_target_action,
                p.arbiter_review_reason,
                p.manual_review_status,
                p.manual_review_reason,
                p.manual_review_requested_at,
                p.manual_review_notes,
                m.match_time,
                m.home_team,
                m.away_team,
                m.league
            FROM prediction_runs p
            JOIN matches m ON m.match_id = p.match_id
            WHERE (
                p.manual_review_status = 'pending'
                OR p.effective_action_source IN ('expert_llm', 'expert_llm_failed')
            )
        """
        params: list[Any] = []
        if issue:
            query += " AND p.issue = ?"
            params.append(str(issue))
        query += """
            ORDER BY
                CASE WHEN p.manual_review_status = 'pending' THEN 0 ELSE 1 END,
                p.run_id DESC
        """
        if limit is not None:
            query += " LIMIT ?"
            params.append(int(limit))
        return conn.execute(query, tuple(params)).fetchall()

    return _run_read(_operation)


def save_feedback_log(feedback: dict) -> int:
    def _operation(conn: sqlite3.Connection) -> int:
        # Drop stale feedback rows that point at the same match but a different
        # prediction_run_id. Without this, repeating "predict → settle" for the
        # same match leaves multiple rows behind (the canonical run drifts to
        # whichever run_id is newest), and downstream summary / backtest /
        # learning aggregates over feedback_logs would double-count that match.
        feedback_payload = dict(feedback)
        feedback_payload.setdefault("handicap_actual_result", "")
        feedback_payload.setdefault("handicap_hit", 0)
        feedback_payload.setdefault("handicap_roi_delta", 0.0)
        conn.execute(
            """
            DELETE FROM feedback_logs
            WHERE match_id = :match_id
              AND prediction_run_id <> :prediction_run_id
            """,
            feedback_payload,
        )
        conn.execute(
            """
            INSERT INTO feedback_logs (
                prediction_run_id, match_id, actual_result, actual_score,
                settled_at, hit_recommendation, roi_delta,
                handicap_actual_result, handicap_hit, handicap_roi_delta,
                roi_source, notes
            ) VALUES (
                :prediction_run_id, :match_id, :actual_result, :actual_score,
                :settled_at, :hit_recommendation, :roi_delta,
                :handicap_actual_result, :handicap_hit, :handicap_roi_delta,
                :roi_source, :notes
            )
            ON CONFLICT(prediction_run_id) DO UPDATE SET
                match_id=excluded.match_id,
                actual_result=excluded.actual_result,
                actual_score=excluded.actual_score,
                settled_at=excluded.settled_at,
                hit_recommendation=excluded.hit_recommendation,
                roi_delta=excluded.roi_delta,
                handicap_actual_result=excluded.handicap_actual_result,
                handicap_hit=excluded.handicap_hit,
                handicap_roi_delta=excluded.handicap_roi_delta,
                roi_source=excluded.roi_source,
                notes=excluded.notes
            """,
            feedback_payload,
        )
        row = conn.execute(
            "SELECT feedback_id FROM feedback_logs WHERE prediction_run_id = ?",
            (feedback["prediction_run_id"],),
        ).fetchone()
        return int(row[0]) if row else 0

    return _run_write(_operation)


def get_feedback_log(prediction_run_id: int) -> sqlite3.Row | None:
    return _run_read(
        lambda conn: conn.execute(
            """
            SELECT *
            FROM feedback_logs
            WHERE prediction_run_id = ?
            """,
            (prediction_run_id,),
        ).fetchone()
    )


def get_feedback_summary(issue: str | None = None) -> dict:
    def _operation(conn: sqlite3.Connection) -> dict:
        issue_text = str(issue or "").strip()
        params: tuple[Any, ...] = ()
        query = """
            WITH scoped_feedback AS (
                SELECT
                    f.*,
                    p.issue,
                    CASE
                        WHEN COALESCE(p.handicap_recommendation, '') <> '观望'
                             AND p.handicap_recommended_side IN ('home', 'away')
                             AND f.handicap_actual_result IN ('home', 'away', 'push')
                        THEN 1
                        ELSE 0
                    END AS has_handicap_action
                FROM feedback_logs f
                JOIN prediction_runs p ON p.run_id = f.prediction_run_id
        """
        if issue_text:
            query += """
                WHERE p.issue = ?
            """
            params = (issue_text,)

        query += """
            )
            SELECT
                COUNT(*) AS total,
                IFNULL(SUM(CASE WHEN hit_recommendation = 1 THEN 1 ELSE 0 END), 0) AS hits,
                IFNULL(SUM(roi_delta), 0) AS total_roi,
                IFNULL(SUM(has_handicap_action), 0) AS handicap_total,
                IFNULL(SUM(CASE WHEN has_handicap_action = 1 AND handicap_hit = 1 THEN 1 ELSE 0 END), 0) AS handicap_hits,
                IFNULL(SUM(CASE WHEN has_handicap_action = 1 THEN handicap_roi_delta ELSE 0 END), 0) AS handicap_total_roi
            FROM scoped_feedback
        """

        row = conn.execute(query, params).fetchone()
        total = int(row["total"] or 0)
        hits = int(row["hits"] or 0)
        total_roi = float(row["total_roi"] or 0.0)
        handicap_total = int(row["handicap_total"] or 0)
        handicap_hits = int(row["handicap_hits"] or 0)
        handicap_total_roi = float(row["handicap_total_roi"] or 0.0)
        misses = max(total - hits, 0)
        hit_rate = (hits / total) if total else 0.0
        avg_roi = (total_roi / total) if total else 0.0
        return {
            "total_predictions": total,
            "hit_predictions": hits,
            "miss_predictions": misses,
            "hit_rate": hit_rate,
            "total_roi": total_roi,
            "avg_roi": avg_roi,
            "handicap_total_predictions": handicap_total,
            "handicap_hit_predictions": handicap_hits,
            "handicap_miss_predictions": max(handicap_total - handicap_hits, 0),
            "handicap_hit_rate": (handicap_hits / handicap_total) if handicap_total else 0.0,
            "handicap_total_roi": handicap_total_roi,
            "handicap_avg_roi": (handicap_total_roi / handicap_total) if handicap_total else 0.0,
        }

    return _run_read(_operation)


def save_learning_profile(profile: dict[str, Any]) -> int:
    def _operation(conn: sqlite3.Connection) -> int:
        learning_profile_id = int(profile.get("learning_profile_id", 0) or 0)
        if learning_profile_id > 0:
            conn.execute(
                """
                INSERT INTO learning_profiles (
                    learning_profile_id,
                    status, created_at, updated_at, activated_at, archived_at,
                    retention_issue_count, window_type, window_value,
                    total_samples, training_samples, validation_samples,
                    training_action_samples, validation_action_samples,
                    calibrator_status, threshold_status,
                    calibrator_params, threshold_params,
                    train_metrics, validation_metrics, sample_summary, notes
                ) VALUES (
                    :learning_profile_id,
                    :status, :created_at, :updated_at, :activated_at, :archived_at,
                    :retention_issue_count, :window_type, :window_value,
                    :total_samples, :training_samples, :validation_samples,
                    :training_action_samples, :validation_action_samples,
                    :calibrator_status, :threshold_status,
                    :calibrator_params, :threshold_params,
                    :train_metrics, :validation_metrics, :sample_summary, :notes
                )
                ON CONFLICT(learning_profile_id) DO UPDATE SET
                    status=excluded.status,
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at,
                    activated_at=excluded.activated_at,
                    archived_at=excluded.archived_at,
                    retention_issue_count=excluded.retention_issue_count,
                    window_type=excluded.window_type,
                    window_value=excluded.window_value,
                    total_samples=excluded.total_samples,
                    training_samples=excluded.training_samples,
                    validation_samples=excluded.validation_samples,
                    training_action_samples=excluded.training_action_samples,
                    validation_action_samples=excluded.validation_action_samples,
                    calibrator_status=excluded.calibrator_status,
                    threshold_status=excluded.threshold_status,
                    calibrator_params=excluded.calibrator_params,
                    threshold_params=excluded.threshold_params,
                    train_metrics=excluded.train_metrics,
                    validation_metrics=excluded.validation_metrics,
                    sample_summary=excluded.sample_summary,
                    notes=excluded.notes
                """,
                profile,
            )
            return learning_profile_id

        conn.execute(
            """
            INSERT INTO learning_profiles (
                status, created_at, updated_at, activated_at, archived_at,
                retention_issue_count, window_type, window_value,
                total_samples, training_samples, validation_samples,
                training_action_samples, validation_action_samples,
                calibrator_status, threshold_status,
                calibrator_params, threshold_params,
                train_metrics, validation_metrics, sample_summary, notes
            ) VALUES (
                :status, :created_at, :updated_at, :activated_at, :archived_at,
                :retention_issue_count, :window_type, :window_value,
                :total_samples, :training_samples, :validation_samples,
                :training_action_samples, :validation_action_samples,
                :calibrator_status, :threshold_status,
                :calibrator_params, :threshold_params,
                :train_metrics, :validation_metrics, :sample_summary, :notes
            )
            """,
            profile,
        )
        row = conn.execute("SELECT last_insert_rowid()").fetchone()
        return int(row[0]) if row else 0

    payload = {
        "learning_profile_id": int(profile.get("learning_profile_id", 0) or 0),
        "status": str(profile.get("status", "") or ""),
        "created_at": str(profile.get("created_at", "") or ""),
        "updated_at": str(profile.get("updated_at", "") or ""),
        "activated_at": str(profile.get("activated_at", "") or ""),
        "archived_at": str(profile.get("archived_at", "") or ""),
        "retention_issue_count": int(profile.get("retention_issue_count", DEFAULT_ISSUE_RETENTION_COUNT) or DEFAULT_ISSUE_RETENTION_COUNT),
        "window_type": str(profile.get("window_type", "rolling_issues") or "rolling_issues"),
        "window_value": int(profile.get("window_value", DEFAULT_ISSUE_RETENTION_COUNT) or DEFAULT_ISSUE_RETENTION_COUNT),
        "total_samples": int(profile.get("total_samples", 0) or 0),
        "training_samples": int(profile.get("training_samples", 0) or 0),
        "validation_samples": int(profile.get("validation_samples", 0) or 0),
        "training_action_samples": int(profile.get("training_action_samples", 0) or 0),
        "validation_action_samples": int(profile.get("validation_action_samples", 0) or 0),
        "calibrator_status": str(profile.get("calibrator_status", "") or ""),
        "threshold_status": str(profile.get("threshold_status", "") or ""),
        "calibrator_params": str(profile.get("calibrator_params", "") or ""),
        "threshold_params": str(profile.get("threshold_params", "") or ""),
        "train_metrics": str(profile.get("train_metrics", "") or ""),
        "validation_metrics": str(profile.get("validation_metrics", "") or ""),
        "sample_summary": str(profile.get("sample_summary", "") or ""),
        "notes": str(profile.get("notes", "") or ""),
    }
    return _run_write(_operation)


def get_learning_profile(learning_profile_id: int) -> sqlite3.Row | None:
    return _run_read(
        lambda conn: conn.execute(
            """
            SELECT *
            FROM learning_profiles
            WHERE learning_profile_id = ?
            """,
            (learning_profile_id,),
        ).fetchone()
    )


def get_active_learning_profile() -> sqlite3.Row | None:
    return _run_read(
        lambda conn: conn.execute(
            """
            SELECT *
            FROM learning_profiles
            WHERE status = 'active'
            ORDER BY learning_profile_id DESC
            LIMIT 1
            """
        ).fetchone()
    )


def get_latest_learning_profile(
    *,
    statuses: tuple[str, ...] | None = None,
) -> sqlite3.Row | None:
    def _operation(conn: sqlite3.Connection) -> sqlite3.Row | None:
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            return conn.execute(
                f"""
                SELECT *
                FROM learning_profiles
                WHERE status IN ({placeholders})
                ORDER BY learning_profile_id DESC
                LIMIT 1
                """,
                tuple(statuses),
            ).fetchone()
        return conn.execute(
            """
            SELECT *
            FROM learning_profiles
            ORDER BY learning_profile_id DESC
            LIMIT 1
            """
        ).fetchone()

    return _run_read(_operation)


def list_learning_profiles(
    *,
    limit: int = 20,
    statuses: tuple[str, ...] | None = None,
) -> list[sqlite3.Row]:
    limit_value = max(int(limit or 0), 1)

    def _operation(conn: sqlite3.Connection) -> list[sqlite3.Row]:
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            return conn.execute(
                f"""
                SELECT *
                FROM learning_profiles
                WHERE status IN ({placeholders})
                ORDER BY learning_profile_id DESC
                LIMIT ?
                """,
                tuple(statuses) + (limit_value,),
            ).fetchall()
        return conn.execute(
            """
            SELECT *
            FROM learning_profiles
            ORDER BY learning_profile_id DESC
            LIMIT ?
            """,
            (limit_value,),
        ).fetchall()

    return _run_read(_operation)


# ---------- Issue Top Picks ----------

def _get_latest_prediction_run_for_match(match_id: str):
    """Get the most recent prediction run for a match."""
    with closing(get_connection()) as conn:
        row = conn.execute(
            """SELECT * FROM prediction_runs
               WHERE match_id = ?
               ORDER BY created_at DESC LIMIT 1""",
            (match_id,),
        ).fetchone()
        return row


def _top_pick_handicap_action_weight(action: str) -> int:
    action_text = str(action or "").strip()
    return {"主推": 2, "轻仓": 1}.get(action_text, 0)


def _top_pick_handicap_side_label(side: str) -> str:
    return {"home": "主队", "away": "客队", "push": "走水"}.get(str(side or "").strip(), "-")


def compute_issue_top_picks(issue: str) -> list[dict[str, Any]]:
    """Compute and persist TOP3 issue picks from Asian-handicap recommendations.

    The TOP3 panel is intentionally driven by ``handicap_recommendation`` and
    ``handicap_recommended_side`` rather than the older win/draw/loss action
    fields. Actionable handicap recommendations (``主推``/``轻仓`` with a home
    or away handicap side) are always ranked ahead of watch-only rows. If an
    issue has fewer than three actionable handicap picks, the best handicap
    watch rows are kept as explicit fallbacks so the page can still surface the
    strongest available handicap candidates.
    """
    from feature_engine import safe_float

    matches = list_matches_by_issue(issue)
    scored: list[dict[str, Any]] = []

    for match in matches:
        match_id = match["match_id"]
        run = _get_latest_prediction_run_for_match(match_id)
        if run is None:
            continue

        action = str(run["handicap_recommendation"] or "").strip() or "观望"
        side = str(run["handicap_recommended_side"] or "").strip()
        action_weight = _top_pick_handicap_action_weight(action)
        actionable = action_weight > 0 and side in {"home", "away"}
        confidence = safe_float(run["handicap_confidence"])
        ev = safe_float(run["handicap_expected_value"])
        quality = safe_float(run["quality_score"])
        agreement = safe_float(run["model_agreement"])

        composite = (
            action_weight * 1.00
            + max(ev, 0.0) * 0.45
            + confidence * 0.35
            + quality * 0.10
            + agreement * 0.05
        )

        scored.append(
            {
                "match_id": match_id,
                "run_id": run["run_id"],
                "composite_score": round(composite, 4),
                "confidence_score": round(confidence, 4),
                "ev_score": round(max(ev, 0.0), 4),
                "quality_score": round(quality, 4),
                "recommended_outcome": side,
                "recommendation": action,
                "handicap_recommendation": action,
                "handicap_recommended_side": side,
                "handicap_line": safe_float(run["handicap_line"]),
                "handicap_expected_value": round(ev, 4),
                "handicap_confidence": round(confidence, 4),
                "handicap_reason": str(run["handicap_reason"] or ""),
                "actionable": actionable,
                "action_weight": action_weight,
                "home_team": match["home_team"],
                "away_team": match["away_team"],
                "league": match["league"],
            }
        )

    scored.sort(
        key=lambda x: (
            bool(x["actionable"]),
            int(x["action_weight"]),
            float(x["ev_score"]),
            float(x["confidence_score"]),
            float(x["quality_score"]),
            float(x["composite_score"]),
        ),
        reverse=True,
    )

    top3 = scored[:3]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with closing(get_connection()) as conn:
        conn.execute("DELETE FROM issue_top_picks WHERE issue = ?", (issue,))
        for rank, pick in enumerate(top3, start=1):
            reason_parts = []
            side_label = _top_pick_handicap_side_label(pick["handicap_recommended_side"])
            if pick["actionable"]:
                reason_parts.append(f"让球盘{pick['handicap_recommendation']}{side_label}")
            else:
                reason_parts.append("让球盘可执行不足三场，按EV/信心候补")
            if pick["ev_score"] >= 0.02:
                reason_parts.append(f"让球EV {pick['handicap_expected_value']:+.3f}")
            if pick["confidence_score"] >= 0.52:
                reason_parts.append(f"让球信心 {pick['confidence_score']:.0%}")
            if pick["quality_score"] >= 0.80:
                reason_parts.append("数据质量好")
            reason = "；".join(reason_parts) if reason_parts else "让球盘综合评分领先"

            conn.execute(
                """INSERT OR REPLACE INTO issue_top_picks
                   (issue, rank, match_id, run_id, composite_score,
                    confidence_score, ev_score, quality_score, reason, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    issue,
                    rank,
                    pick["match_id"],
                    pick["run_id"],
                    pick["composite_score"],
                    pick["confidence_score"],
                    pick["ev_score"],
                    pick["quality_score"],
                    reason,
                    now,
                ),
            )

        conn.commit()
    return top3


def get_issue_top_picks(issue: str | None = None) -> list[dict[str, Any]]:
    """Retrieve saved Asian-handicap TOP3 picks for an issue.

    Returns up to 3 rows ordered by rank, enriched with match metadata,
    handicap recommendation fields, and handicap settlement status.
    """
    if issue is None:
        issue = get_latest_issue()
    if not issue:
        return []

    with closing(get_connection()) as conn:
        rows = conn.execute(
            """SELECT tp.*, m.home_team, m.away_team, m.league, m.match_time,
                      m.actual_result AS match_actual_result,
                      m.actual_score AS match_actual_score,
                      pr.recommended_outcome AS win_draw_loss_outcome,
                      pr.recommendation AS win_draw_loss_recommendation,
                      pr.handicap_recommendation,
                      pr.handicap_recommended_side,
                      pr.handicap_line,
                      pr.handicap_initial_line,
                      pr.handicap_home_odds,
                      pr.handicap_away_odds,
                      pr.handicap_home_cover_prob,
                      pr.handicap_away_cover_prob,
                      pr.handicap_expected_value,
                      pr.handicap_confidence,
                      pr.handicap_reason,
                      pr.final_home_prob, pr.final_draw_prob, pr.final_away_prob,
                      pr.market_odds_home, pr.market_odds_draw, pr.market_odds_away,
                      pr.confidence_score AS run_confidence,
                      f.actual_result,
                      f.actual_score,
                      f.hit_recommendation,
                      f.roi_delta,
                      f.handicap_actual_result,
                      f.handicap_hit,
                      f.handicap_roi_delta,
                      f.settled_at
               FROM issue_top_picks tp
               LEFT JOIN matches m ON tp.match_id = m.match_id
               LEFT JOIN prediction_runs pr ON tp.run_id = pr.run_id
               LEFT JOIN feedback_logs f ON f.prediction_run_id = pr.run_id
               WHERE tp.issue = ?
               ORDER BY tp.rank""",
            (issue,),
        ).fetchall()

    picks: list[dict[str, Any]] = []
    for row in rows:
        pick = dict(row)
        handicap_action = str(pick.get("handicap_recommendation") or "").strip() or "观望"
        handicap_side = str(pick.get("handicap_recommended_side") or "").strip()
        handicap_result = str(pick.get("handicap_actual_result") or "").strip()
        is_actionable = handicap_action != "观望" and handicap_side in {"home", "away"}
        if not handicap_result and is_actionable:
            handicap_result = _settle_handicap_result_from_score(
                str(pick.get("match_actual_score") or ""),
                float(pick.get("handicap_line") or 0.0),
            )
        is_settled = is_actionable and handicap_result in {"home", "away", "push"}
        handicap_hit = int(pick.get("handicap_hit") or (1 if is_settled and handicap_side == handicap_result else 0))
        pick["recommended_outcome"] = handicap_side
        pick["recommendation"] = handicap_action
        pick["handicap_side_label"] = _top_pick_handicap_side_label(handicap_side)
        pick["is_settled"] = bool(is_settled)
        pick["actual_result"] = handicap_result if is_settled else ""
        pick["actual_score"] = pick.get("actual_score") or pick.get("match_actual_score") or ""
        pick["hit_recommendation"] = handicap_hit if is_settled else 0
        pick["top_pick_result_status"] = (
            "hit" if is_settled and handicap_hit else "miss" if is_settled else "pending"
        )
        picks.append(pick)

    return picks


def has_issue_top_picks(issue: str | None = None) -> bool:
    """Check if top picks have been computed for the given issue."""
    conn = get_connection()
    if issue is None:
        issue = get_latest_issue()
    if not issue:
        return False
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM issue_top_picks WHERE issue = ?",
        (issue,),
    ).fetchone()
    return bool(row and row['cnt'] > 0)
