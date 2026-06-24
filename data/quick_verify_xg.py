"""一行验证：抓 EPL，看新解析是否能拿到 20 支球队 + xG 数值。

用法：
    python data/quick_verify_xg.py

输出会告诉你：
- playwright-cli 是否能跑
- understat 是否还活着
- 新解析逻辑是否拿到完整 20 队 + xG/xGA

任何一步失败都打印 root cause，便于直接贴回来诊断。
"""
from __future__ import annotations

import sys
import traceback


def main() -> int:
    print("[1/3] 准备抓取 EPL 当前赛季...")
    try:
        from source_understat_client import load_league_xg
    except Exception as exc:  # noqa: BLE001
        print(f"  ✗ 导入失败：{exc}")
        traceback.print_exc()
        return 2

    try:
        # force_refresh=True 跳过本地缓存，确保拿真实抓取结果
        payload = load_league_xg("EPL", force_refresh=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  ✗ 抓取/解析失败：{exc}")
        print()
        print("  ↑ 上面这条错误就是关键诊断。")
        print("  → 如果错误指向 .empty.html 文件路径，把那个路径贴出来；")
        print("  → 如果错误提到 playwright-cli 不可用，说明 PLAYWRIGHT_CLI_BIN 未配置。")
        return 3

    teams = payload.get("teams") or {}
    print(f"  ✓ 解析成功，抓到 {len(teams)} 支球队")
    print(f"  联赛: {payload.get('league_label')}  赛季: {payload.get('season')}")

    print()
    print("[2/3] 抽 5 支球队看 xG 数值是否合理（典型范围 0.6 - 2.5）...")
    sample = list(teams.values())[:5]
    for m in sample:
        print(
            f"    {m['title']:30s}  matches={m['matches']:2d}  "
            f"xG/G={m['xg_per_game']:.2f}  xGA/G={m['xga_per_game']:.2f}"
        )
    if any(m["xg_per_game"] <= 0 for m in sample):
        print("    ⚠ 出现 xG 为 0 的球队 — 解析可能漏了 xG 列")
        return 4

    print()
    print("[3/3] 验证查找接口（从 team_name_aliases 拿候选名能查到）...")
    try:
        from source_understat_client import lookup_team_in_league
        from team_name_aliases import resolve_team_aliases

        candidates = resolve_team_aliases("阿森纳")
        match = lookup_team_in_league(payload, candidates)
        if match:
            print(f"  ✓ '阿森纳' → {match['title']}, xG/G {match['xg_per_game']:.2f}")
        else:
            print(f"  ✗ '阿森纳' 没匹配上，候选: {candidates}")
            return 5
    except Exception as exc:  # noqa: BLE001
        print(f"  ✗ 查找失败：{exc}")
        traceback.print_exc()
        return 6

    print()
    print("=" * 56)
    print("  ✓ 全部通过。可以接着跑：")
    print("    python data/inspect_xg_match.py    # 全联赛覆盖率")
    print("    python data/replay_all_issues.py   # 重跑全部 7 期")
    return 0


if __name__ == "__main__":
    sys.exit(main())
