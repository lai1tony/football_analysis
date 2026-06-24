#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose issue 26079 collection status - detailed."""

import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "football_data.db"


def main():
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    print("=" * 60)
    print("FAILED MATCHES IN 26079 - DETAILED")
    print("=" * 60)

    rows = conn.execute(
        "SELECT * FROM matches WHERE issue = '26079' AND match_no IN ('8', '11', '14') ORDER BY CAST(match_no AS INTEGER)"
    ).fetchall()

    for r in rows:
        print()
        print(f"Match [{r['match_no']}]: {r['home_team']} vs {r['away_team']}")
        print(f"  league: {r['league']}")
        print(f"  match_time: {r['match_time']}")
        print(f"  match_id: {r['match_id']}")
        print(f"  shuju_url: {r['shuju_url']}")
        print(f"  ouzhi_url: {r['ouzhi_url']}")
        print(f"  touzhu_url: {r['touzhu_url']}")

        a = conn.execute(
            "SELECT * FROM analyses WHERE match_id = ?", (r["match_id"],)
        ).fetchone()
        if a:
            print(f"  collected_at: {a['collected_at']}")
            print(f"  elo_home: {a['elo_home']!r}")
            print(f"  elo_away: {a['elo_away']!r}")
            print(f"  recent_form_home: {a['recent_form_home']!r}")
            print(f"  recent_form_away: {a['recent_form_away']!r}")
            print(f"  home_away_form: {a['home_away_form']!r}")
            print(f"  head_to_head_summary: {a['head_to_head_summary']!r}")
            print(f"  injury_or_lineup_notes: {a['injury_or_lineup_notes']!r}")
            print(f"  motivation_or_schedule_notes: {a['motivation_or_schedule_notes']!r}")
            print(f"  european_odds_movement_summary: {a['european_odds_movement_summary']!r}")
            print(f"  betting_heat_summary: {a['betting_heat_summary']!r}")
            remarks = a['remarks'] or ''
            print(f"  remarks: {remarks!r}")
        else:
            print("  NO ANALYSIS RECORD")

    conn.close()


if __name__ == "__main__":
    main()
