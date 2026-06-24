from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from collection_repository import DB_PATH, get_database_status, init_db
from collection_service import build_asian_handicap_summary
from feature_engine import build_feature_snapshot, build_match_features, safe_float
from prediction_engine import (
    _settle_handicap_result,
    evaluate_handicap_recommendation,
    run_data_quality,
    run_quant_model,
)
from source_500_client import fetch_html


HANDICAP_RUN_FIELDS = (
    "handicap_recommendation",
    "handicap_recommended_side",
    "handicap_line",
    "handicap_initial_line",
    "handicap_home_odds",
    "handicap_away_odds",
    "handicap_initial_home_odds",
    "handicap_initial_away_odds",
    "handicap_home_cover_prob",
    "handicap_away_cover_prob",
    "handicap_expected_value",
    "handicap_confidence",
    "handicap_reason",
)


def _connect() -> sqlite3.Connection:
    status = get_database_status()
    active_path = status.get("active_path") or DB_PATH
    conn = sqlite3.connect(Path(active_path))
    conn.execute("PRAGMA journal_mode=MEMORY")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_yazhi_url(conn: sqlite3.Connection, match: sqlite3.Row) -> str:
    current = str(match["yazhi_url"] or "") if "yazhi_url" in match.keys() else ""
    if current:
        return current
    url = f"https://odds.500.com/fenxi/yazhi-{match['match_id']}.shtml"
    conn.execute("UPDATE matches SET yazhi_url = ? WHERE match_id = ?", (url, match["match_id"]))
    return url


def _update_handicap_summary(conn: sqlite3.Connection, match: sqlite3.Row, *, force_fetch: bool) -> str:
    existing = str(match["asian_handicap_summary"] or "") if "asian_handicap_summary" in match.keys() else ""
    if existing and not force_fetch:
        return existing
    url = _ensure_yazhi_url(conn, match)
    html = fetch_html(url)
    summary = build_asian_handicap_summary(html)
    if summary:
        conn.execute(
            """
            UPDATE analyses
            SET asian_handicap_summary = ?,
                media_source_links = CASE
                    WHEN IFNULL(media_source_links, '') = '' THEN ?
                    WHEN media_source_links LIKE '%' || ? || '%' THEN media_source_links
                    ELSE media_source_links || char(10) || ?
                END,
                collected_sources = CASE
                    WHEN IFNULL(collected_sources, '') = '' THEN '500亚盘页'
                    WHEN collected_sources LIKE '%500亚盘页%' THEN collected_sources
                    ELSE collected_sources || char(10) || '500亚盘页'
                END
            WHERE match_id = ?
            """,
            (summary, url, url, url, match["match_id"]),
        )
    return summary


def _select_target_run(conn: sqlite3.Connection, match_id: str) -> sqlite3.Row | None:
    feedback_run = conn.execute(
        """
        SELECT pr.*
        FROM feedback_logs f
        JOIN prediction_runs pr ON pr.run_id = f.prediction_run_id
        WHERE f.match_id = ?
        ORDER BY f.feedback_id DESC
        LIMIT 1
        """,
        (match_id,),
    ).fetchone()
    if feedback_run is not None:
        return feedback_run
    return conn.execute(
        """
        SELECT *
        FROM prediction_runs
        WHERE match_id = ?
        ORDER BY run_id DESC
        LIMIT 1
        """,
        (match_id,),
    ).fetchone()


def _decimal_odds(raw: float) -> float:
    return raw + 1.0 if 0 < raw < 1.5 else raw


def _update_feedback(conn: sqlite3.Connection, run_id: int, risk: dict[str, Any]) -> None:
    feedback = conn.execute("SELECT * FROM feedback_logs WHERE prediction_run_id = ?", (run_id,)).fetchone()
    if feedback is None:
        return
    actual_score = str(feedback["actual_score"] or "")
    result = _settle_handicap_result(actual_score, safe_float(risk.get("line")))
    side = str(risk.get("recommended_side") or "")
    hit = 1 if side and result == side else 0
    roi = 0.0
    if side in {"home", "away"} and result in {"home", "away"}:
        run = conn.execute("SELECT * FROM prediction_runs WHERE run_id = ?", (run_id,)).fetchone()
        stake_pct = safe_float(run["suggested_stake_pct"] if run is not None else 0.0)
        odds = _decimal_odds(safe_float(risk.get(f"{side}_odds")))
        stake_units = stake_pct / 100.0
        roi = round(stake_units * (odds - 1.0), 4) if hit and odds > 0 else round(-stake_units, 4)
    conn.execute(
        """
        UPDATE feedback_logs
        SET handicap_actual_result = ?,
            handicap_hit = ?,
            handicap_roi_delta = ?
        WHERE prediction_run_id = ?
        """,
        (result, hit, roi, run_id),
    )


def _update_all_feedback_for_match(conn: sqlite3.Connection, match_id: str, risk: dict[str, Any]) -> int:
    feedback_rows = conn.execute(
        "SELECT prediction_run_id FROM feedback_logs WHERE match_id = ?",
        (match_id,),
    ).fetchall()
    for row in feedback_rows:
        _update_feedback(conn, int(row["prediction_run_id"]), risk)
    return len(feedback_rows)


def _update_all_prediction_runs_for_match(
    conn: sqlite3.Connection,
    match_id: str,
    payload: dict[str, Any],
) -> int:
    assignments = ", ".join(f"{field} = :{field}" for field in HANDICAP_RUN_FIELDS)
    update_payload = dict(payload)
    update_payload["match_id"] = match_id
    cursor = conn.execute(
        f"UPDATE prediction_runs SET {assignments} WHERE match_id = :match_id",
        update_payload,
    )
    return int(cursor.rowcount or 0)


def _backfill_one(conn: sqlite3.Connection, match: sqlite3.Row, *, force_fetch: bool) -> dict[str, Any]:
    summary = _update_handicap_summary(conn, match, force_fetch=force_fetch)
    if not summary:
        return {"match_id": match["match_id"], "status": "missing_handicap"}

    analysis = conn.execute(
        """
        SELECT
            m.*, a.collected_at, a.elo_home, a.elo_away, a.market_value_summary,
            a.recent_form_home, a.recent_form_away, a.home_away_form,
            a.head_to_head_summary, a.injury_or_lineup_notes,
            a.motivation_or_schedule_notes, a.european_odds_movement_summary,
            a.asian_handicap_summary, a.betting_heat_summary,
            a.media_source_links, a.collected_sources, a.collection_quality_summary, a.remarks
        FROM matches m
        JOIN analyses a ON a.match_id = m.match_id
        WHERE m.match_id = ?
        """,
        (match["match_id"],),
    ).fetchone()
    if analysis is None:
        return {"match_id": match["match_id"], "status": "missing_analysis"}

    snapshot = build_feature_snapshot(dict(analysis))
    snapshot["snapshot_id"] = 0
    features = build_match_features(dict(analysis), snapshot)
    quant = run_quant_model(dict(analysis), features)
    quality = run_data_quality(dict(analysis), features)
    risk = evaluate_handicap_recommendation(
        features=features,
        quant=quant,
        quality=quality,
        row=dict(analysis),
    )
    run = _select_target_run(conn, str(match["match_id"]))
    if run is None:
        return {"match_id": match["match_id"], "status": "missing_run"}

    payload = {
        "handicap_recommendation": risk["recommendation"],
        "handicap_recommended_side": risk["recommended_side"],
        "handicap_line": risk["line"],
        "handicap_initial_line": risk["initial_line"],
        "handicap_home_odds": risk["home_odds"],
        "handicap_away_odds": risk["away_odds"],
        "handicap_initial_home_odds": risk["initial_home_odds"],
        "handicap_initial_away_odds": risk["initial_away_odds"],
        "handicap_home_cover_prob": risk["home_cover_prob"],
        "handicap_away_cover_prob": risk["away_cover_prob"],
        "handicap_expected_value": risk["expected_value"],
        "handicap_confidence": risk["confidence"],
        "handicap_reason": risk["reason"],
    }
    updated_run_count = _update_all_prediction_runs_for_match(conn, str(match["match_id"]), payload)
    updated_feedback_count = _update_all_feedback_for_match(conn, str(match["match_id"]), risk)
    return {
        "match_id": match["match_id"],
        "run_id": int(run["run_id"]),
        "updated_run_count": updated_run_count,
        "updated_feedback_count": updated_feedback_count,
        "status": "updated",
        "recommendation": risk["recommendation"],
        "side": risk["recommended_side"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill Asian handicap collection, recommendation, and feedback fields.")
    parser.add_argument("--issue", default="", help="Only process one issue.")
    parser.add_argument("--limit", type=int, default=0, help="Limit matches for chunked runs.")
    parser.add_argument("--offset", type=int, default=0, help="Offset for chunked runs.")
    parser.add_argument("--force-fetch", action="store_true", help="Re-fetch handicap page even if summary exists.")
    args = parser.parse_args()

    init_db()
    with _connect() as conn:
        where = "WHERE a.match_id IS NOT NULL"
        params: list[Any] = []
        if args.issue:
            where += " AND m.issue = ?"
            params.append(args.issue)
        query = f"""
            SELECT m.*, a.asian_handicap_summary
            FROM matches m
            LEFT JOIN analyses a ON a.match_id = m.match_id
            {where}
            ORDER BY m.issue, CAST(m.match_no AS INTEGER), m.match_id
        """
        if args.limit:
            query += " LIMIT ? OFFSET ?"
            params.extend([args.limit, args.offset])
        rows = conn.execute(query, tuple(params)).fetchall()
        started = datetime.now()
        counts: dict[str, int] = {}
        updated_runs = 0
        updated_feedback = 0
        for index, match in enumerate(rows, start=1):
            try:
                result = _backfill_one(conn, match, force_fetch=args.force_fetch)
                conn.commit()
            except Exception as exc:  # noqa: BLE001
                result = {"match_id": match["match_id"], "status": "failed", "error": str(exc)}
                conn.rollback()
            status = str(result.get("status"))
            counts[status] = counts.get(status, 0) + 1
            updated_runs += int(result.get("updated_run_count", 0) or 0)
            updated_feedback += int(result.get("updated_feedback_count", 0) or 0)
            print(
                f"[{index}/{len(rows)}] {match['issue']} {match['match_id']} {status} "
                f"{result.get('recommendation', '')} {result.get('side', '')} "
                f"runs={result.get('updated_run_count', 0)} feedback={result.get('updated_feedback_count', 0)} "
                f"{result.get('error', '')}"
            )
        elapsed = (datetime.now() - started).total_seconds()
        print(
            f"done rows={len(rows)} counts={counts} updated_runs={updated_runs} "
            f"updated_feedback={updated_feedback} elapsed={elapsed:.1f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
