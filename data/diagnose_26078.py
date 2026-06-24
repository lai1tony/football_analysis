#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose issue 26078 Freiburg vs Villa collection anomaly."""

import sqlite3
import sys
from pathlib import Path

# Use the same DB path as the project
DB_PATH = Path(__file__).resolve().parent / "football_data.db"


def main():
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # 1. Check if issue 26078 exists
    print("=" * 60)
    print("ISSUE 26078 MATCHES")
    print("=" * 60)
    rows = conn.execute(
        "SELECT * FROM matches WHERE issue = ? ORDER BY CAST(match_no AS INTEGER)",
        ("26078",),
    ).fetchall()
    print(f"Total matches in 26078: {len(rows)}")
    for row in rows:
        print(f"  [{row['match_no']}] {row['home_team']} vs {row['away_team']} | league={row['league']} | match_id={row['match_id']}")

    # 2. Look for Freiburg/Villa specifically
    print()
    print("=" * 60)
    print("FREIBURG / VILLA SEARCH")
    print("=" * 60)
    freiburg_rows = conn.execute(
        "SELECT * FROM matches WHERE (home_team LIKE '%弗赖堡%' OR away_team LIKE '%弗赖堡%' OR home_team LIKE '%Freiburg%' OR away_team LIKE '%Freiburg%')"
    ).fetchall()
    villa_rows = conn.execute(
        "SELECT * FROM matches WHERE (home_team LIKE '%维拉%' OR away_team LIKE '%维拉%' OR home_team LIKE '%Villa%' OR away_team LIKE '%Villa%')"
    ).fetchall()
    print(f"Freiburg matches: {len(freiburg_rows)}")
    for row in freiburg_rows:
        print(f"  issue={row['issue']} [{row['match_no']}] {row['home_team']} vs {row['away_team']} | match_id={row['match_id']}")
    print(f"Villa matches: {len(villa_rows)}")
    for row in villa_rows:
        print(f"  issue={row['issue']} [{row['match_no']}] {row['home_team']} vs {row['away_team']} | match_id={row['match_id']}")

    # 3. If found, check collection status
    print()
    print("=" * 60)
    print("COLLECTION STATUS")
    print("=" * 60)
    all_candidates = freiburg_rows + villa_rows
    seen_ids = set()
    for row in all_candidates:
        if row["match_id"] in seen_ids:
            continue
        seen_ids.add(row["match_id"])
        analysis = conn.execute(
            "SELECT * FROM analyses WHERE match_id = ?", (row["match_id"],)
        ).fetchone()
        if analysis:
            print(f"\n  match_id={row['match_id']} ({row['home_team']} vs {row['away_team']})")
            print(f"  collected_at={analysis['collected_at']}")
            print(f"  elo_home={analysis['elo_home'][:80] if analysis['elo_home'] else '(empty)'}")
            print(f"  elo_away={analysis['elo_away'][:80] if analysis['elo_away'] else '(empty)'}")
            print(f"  recent_form_home={analysis['recent_form_home'][:80] if analysis['recent_form_home'] else '(empty)'}")
            print(f"  recent_form_away={analysis['recent_form_away'][:80] if analysis['recent_form_away'] else '(empty)'}")
            print(f"  home_away_form={analysis['home_away_form'][:80] if analysis['home_away_form'] else '(empty)'}")
            print(f"  head_to_head_summary={analysis['head_to_head_summary'][:80] if analysis['head_to_head_summary'] else '(empty)'}")
            print(f"  injury_or_lineup_notes={analysis['injury_or_lineup_notes'][:80] if analysis['injury_or_lineup_notes'] else '(empty)'}")
            print(f"  motivation_or_schedule_notes={analysis['motivation_or_schedule_notes'][:80] if analysis['motivation_or_schedule_notes'] else '(empty)'}")
            print(f"  european_odds={analysis['european_odds_movement_summary'][:80] if analysis['european_odds_movement_summary'] else '(empty)'}")
            print(f"  betting_heat={analysis['betting_heat_summary'][:80] if analysis['betting_heat_summary'] else '(empty)'}")
            print(f"  remarks={analysis['remarks'][:200] if analysis['remarks'] else '(empty)'}")
        else:
            print(f"\n  match_id={row['match_id']} ({row['home_team']} vs {row['away_team']}): NO ANALYSIS RECORD")

    # 4. Check collection stats for 26078
    print()
    print("=" * 60)
    print("COLLECTION STATS FOR 26078")
    print("=" * 60)
    total = conn.execute("SELECT COUNT(*) FROM matches WHERE issue = '26078'").fetchone()[0]
    collected = conn.execute(
        "SELECT COUNT(*) FROM analyses a JOIN matches m ON m.match_id = a.match_id WHERE m.issue = '26078' AND a.collected_at <> ''"
    ).fetchone()[0]
    print(f"Total matches: {total}")
    print(f"Collected (any): {collected}")

    # Check which matches have missing dimensions
    print()
    print("=" * 60)
    print("DIMENSION ANALYSIS FOR 26078")
    print("=" * 60)
    all_26078 = conn.execute(
        "SELECT m.match_id, m.home_team, m.away_team, m.match_no, a.* FROM matches m LEFT JOIN analyses a ON a.match_id = m.match_id WHERE m.issue = '26078' ORDER BY CAST(m.match_no AS INTEGER)"
    ).fetchall()

    required_fields = [
        ("elo_home", "Dim1: 主队Elo"),
        ("elo_away", "Dim1: 客队Elo"),
        ("recent_form_home", "Dim1: 主队近期状态"),
        ("recent_form_away", "Dim1: 客队近期状态"),
        ("home_away_form", "Dim1: 主客场效应"),
        ("head_to_head_summary", "Dim2: 交锋记录"),
        ("injury_or_lineup_notes", "Dim2: 伤停阵容"),
        ("motivation_or_schedule_notes", "Dim2: 战意赛程"),
        ("european_odds_movement_summary", "Dim3: 欧赔变化"),
        ("betting_heat_summary", "Dim3: 投注热度"),
    ]

    placeholder_tokens = [
        "未在公开来源中命中预计首发",
        "外部公开来源补采未命中",
        "伤停/阵容补采失败",
    ]

    for row in all_26078:
        missing = []
        for field, label in required_fields:
            val = row[field] if field in row.keys() else None
            val_str = str(val or "").strip()
            if not val_str:
                missing.append(label)
            elif any(tok in val_str for tok in placeholder_tokens):
                missing.append(f"{label} (placeholder)")
        status = "FAIL - missing: " + ", ".join(missing) if missing else "OK"
        print(f"  [{row['match_no']}] {row['home_team']} vs {row['away_team']}: {status}")

    conn.close()


if __name__ == "__main__":
    main()
