#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证补充采集模块语法，并对弗赖堡 vs 维拉执行补采测试。"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 1. 语法检查
print("=" * 60)
print("STEP 1: 语法检查")
print("=" * 60)
try:
    import py_compile
    py_compile.compile(os.path.join(os.path.dirname(__file__), "source_supplement_client.py"), doraise=True)
    print("  source_supplement_client.py: OK")
    py_compile.compile(os.path.join(os.path.dirname(__file__), "collection_service.py"), doraise=True)
    print("  collection_service.py: OK")
except Exception as exc:
    print(f"  语法错误: {exc}")
    sys.exit(1)

# 2. 模块导入检查
print()
print("=" * 60)
print("STEP 2: 模块导入检查")
print("=" * 60)
try:
    from source_supplement_client import (
        supplement_match_data,
        supplement_xg_from_understat,
        supplement_from_flashscore,
        UNDERSTAT_LEAGUES,
        TEAM_LEAGUE_MAP,
        _infer_league,
    )
    print("  supplement_match_data: OK")
    print("  supplement_xg_from_understat: OK")
    print("  supplement_from_flashscore: OK")
    print(f"  UNDERSTAT_LEAGUES: {list(UNDERSTAT_LEAGUES.keys())}")
except Exception as exc:
    print(f"  导入错误: {exc}")
    sys.exit(1)

# 3. 球队联赛推断测试
print()
print("=" * 60)
print("STEP 3: 球队联赛推断")
print("=" * 60)
test_cases = [
    ("弗赖堡", "Bundesliga"),
    ("SC Freiburg", "Bundesliga"),
    ("维拉", "EPL"),
    ("Aston Villa", "EPL"),
    ("阿森纳", "EPL"),
    ("皇马", "La_liga"),
    ("拜仁", "Bundesliga"),
]
for team, expected in test_cases:
    result = _infer_league(team)
    status = "OK" if result == expected else f"FAIL (got {result})"
    print(f"  {team} -> {result} [{status}]")

# 4. 数据库诊断
print()
print("=" * 60)
print("STEP 4: 数据库诊断")
print("=" * 60)
import sqlite3
db_path = os.path.join(os.path.dirname(__file__), "football_data.db")
if not os.path.exists(db_path):
    print(f"  数据库不存在: {db_path}")
    sys.exit(1)

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row

# 查找 26078 期弗赖堡/维拉
rows = conn.execute(
    "SELECT match_id, home_team, away_team, issue, league, match_time FROM matches "
    "WHERE issue='26078' AND (home_team LIKE '%赖堡%' OR away_team LIKE '%维拉%' "
    "OR home_team LIKE '%Freiburg%' OR away_team LIKE '%Aston%')"
).fetchall()

if not rows:
    print("  未找到 26078 期弗赖堡 vs 维拉")
    # 搜索其他期号
    rows = conn.execute(
        "SELECT match_id, home_team, away_team, issue, league, match_time FROM matches "
        "WHERE home_team LIKE '%赖堡%' OR away_team LIKE '%维拉%' "
        "OR home_team LIKE '%Freiburg%' OR away_team LIKE '%Aston%' "
        "ORDER BY issue DESC LIMIT 5"
    ).fetchall()
    print("  搜索其他期号中的弗赖堡/维拉:")

for r in rows:
    print(f"  issue={r['issue']} match_id={r['match_id']} | {r['home_team']} vs {r['away_team']} | league={r['league']}")

    # 检查采集状态
    analysis = conn.execute("SELECT * FROM analyses WHERE match_id=?", (r['match_id'],)).fetchone()
    if analysis:
        print(f"    collected_at: {analysis['collected_at']}")
        print(f"    elo_home: {analysis['elo_home'][:60] if analysis['elo_home'] else '(空)'}")
        print(f"    elo_away: {analysis['elo_away'][:60] if analysis['elo_away'] else '(空)'}")
        print(f"    head_to_head: {analysis['head_to_head_summary'][:60] if analysis['head_to_head_summary'] else '(空)'}")
        print(f"    injury: {analysis['injury_or_lineup_notes'][:60] if analysis['injury_or_lineup_notes'] else '(空)'}")
        print(f"    remarks: {analysis['remarks'][:100] if analysis['remarks'] else '(空)'}")
    else:
        print("    无采集记录")

conn.close()

print()
print("=" * 60)
print("诊断完成。")
print("=" * 60)
print()
print("如需执行补采测试（需要 playwright-cli），运行：")
print("  python data/test_supplement.py --run-supplement")
print()
print("补采命令示例：")
print("  python data/run_prediction.py predict --match-id <match_id> --collect")
