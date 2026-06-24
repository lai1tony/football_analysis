"""xG 命中率检查 + 队名映射缺口扫描。

用法：
    python data/inspect_xg_match.py            # 用所有 settled 样本
    python data/inspect_xg_match.py --refresh  # 强制重新抓 understat

输出：
    - 每个联赛的 xG 命中率（成功匹配两队的比例）
    - 主队/客队各自的命中率
    - 完整列出*没匹配上*的中文队名，方便手工加进 team_name_aliases.py

只读不写。
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path

import collection_repository as repo
from source_understat_client import (
    UNDERSTAT_LEAGUES,
    load_league_xg,
    lookup_team_in_league,
)
from team_name_aliases import league_codes_for_500_label, resolve_team_aliases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="忽略本地缓存")
    parser.add_argument("--season", default=None)
    args = parser.parse_args()

    db_path: Path = repo.PRIMARY_DB_PATH
    if not db_path.exists():
        print(f"找不到数据库：{db_path}")
        return 1

    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT m.match_id, m.league, m.home_team, m.away_team
            FROM feedback_logs f
            JOIN matches m ON m.match_id = f.match_id
            ORDER BY m.match_id
            """
        ).fetchall()

    if not rows:
        print("没有 settled 样本可检查。")
        return 0

    # 一次性把所有相关联赛预热到本地缓存
    needed_leagues = set()
    for r in rows:
        for code in league_codes_for_500_label(str(r["league"] or "")):
            needed_leagues.add(code)

    print(f"准备加载 {len(needed_leagues)} 个 understat 联赛...")
    league_payloads: dict[str, dict] = {}
    for code in sorted(needed_leagues):
        try:
            league_payloads[code] = load_league_xg(code, args.season, force_refresh=args.refresh)
            teams_count = len(league_payloads[code].get("teams") or {})
            print(f"  ✓ {code} ({UNDERSTAT_LEAGUES[code]['label']}): {teams_count} 支球队")
        except Exception as exc:  # noqa: BLE001
            league_payloads[code] = {"teams": {}, "error": str(exc)}
            print(f"  ✗ {code}: 抓取失败 {exc}")

    # 逐场扫描
    per_league_total: Counter = Counter()
    per_league_hit: Counter = Counter()
    home_misses: list[tuple[str, str, str]] = []  # (league, chinese_name, candidates)
    away_misses: list[tuple[str, str, str]] = []
    unsupported_league: list[tuple[str, str, str]] = []

    for r in rows:
        league_label = str(r["league"] or "")
        candidate_codes = league_codes_for_500_label(league_label)
        if not candidate_codes:
            unsupported_league.append((league_label, str(r["home_team"] or ""), str(r["away_team"] or "")))
            continue
        per_league_total[candidate_codes[0]] += 1

        home_aliases = resolve_team_aliases(str(r["home_team"] or ""))
        away_aliases = resolve_team_aliases(str(r["away_team"] or ""))
        matched = False
        for code in candidate_codes:
            payload = league_payloads.get(code) or {}
            home_metrics = lookup_team_in_league(payload, home_aliases)
            away_metrics = lookup_team_in_league(payload, away_aliases)
            if home_metrics and away_metrics:
                per_league_hit[code] += 1
                matched = True
                break
            if home_metrics and not away_metrics:
                away_misses.append((code, str(r["away_team"] or ""), ", ".join(away_aliases[:3])))
            elif away_metrics and not home_metrics:
                home_misses.append((code, str(r["home_team"] or ""), ", ".join(home_aliases[:3])))
            elif not home_metrics and not away_metrics:
                home_misses.append((code, str(r["home_team"] or ""), ", ".join(home_aliases[:3])))
                away_misses.append((code, str(r["away_team"] or ""), ", ".join(away_aliases[:3])))
        if not matched and not (home_misses and away_misses):
            pass  # 已记录到缺口

    total = sum(per_league_total.values())
    hit = sum(per_league_hit.values())
    overall = hit / total * 100 if total else 0
    print()
    print("=" * 60)
    print(f"xG 命中率（成功匹配两队同时存在）")
    print("=" * 60)
    print(f"  整体: {hit}/{total} = {overall:.1f}%")
    if unsupported_league:
        print(f"  联赛非 understat 覆盖: {len(unsupported_league)} 场（已跳过统计）")
    print()
    print(f"  按联赛:")
    for code, total_count in per_league_total.most_common():
        h = per_league_hit.get(code, 0)
        rate = h / total_count * 100 if total_count else 0
        label = UNDERSTAT_LEAGUES.get(code, {}).get("label", code)
        print(f"    {code:12s} ({label}): {h}/{total_count} ({rate:.1f}%)")

    # 输出缺口清单（按出现次数排）
    home_miss_counter = Counter((cn, alias) for _, cn, alias in home_misses)
    away_miss_counter = Counter((cn, alias) for _, cn, alias in away_misses)
    miss_total = home_miss_counter + away_miss_counter

    if miss_total:
        print()
        print(f"未匹配上的中文队名（按频次排）:")
        print("  → 把这些加到 team_name_aliases.py 的 TEAM_ALIASES 字典就能补齐")
        for (chinese, candidates), count in miss_total.most_common(30):
            print(f"    {count:3d}× {chinese:20s}  当前候选: [{candidates}]")

    if unsupported_league:
        print()
        print(f"非 understat 覆盖的联赛分布（自动 fallback，不影响 xG 流程）:")
        league_unmatched = Counter(lg for lg, _, _ in unsupported_league)
        for lg, count in league_unmatched.most_common(15):
            print(f"    {count:3d}× {lg}")

    print()
    print("=" * 60)
    if overall >= 80:
        print("  ✓ 命中率充足，xG 接入按预期工作")
    elif overall >= 60:
        print("  ⚠ 命中率中等，建议把上面的'未匹配队名'加到映射表后重跑")
    else:
        print("  ✗ 命中率不足，先扩展映射表")
    return 0


if __name__ == "__main__":
    sys.exit(main())
