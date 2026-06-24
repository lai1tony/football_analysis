"""第 0 步：先扫一遍数据库，看 understat 能覆盖多少场。

工作量预算 6-10 小时之前，先用 1 分钟验证假设。
understat 只覆盖五大联赛 + 俄超 + MLS（部分），如果你 97 条样本里
60% 是中超 / 北欧 / 南美杯赛，那 xG 接入是赔本生意。

只读 matches 表，不写库。
    python data/inspect_xg_coverage.py
"""
from __future__ import annotations

import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path

import collection_repository as repo


# understat 实际覆盖（中文别名 + 各国语言名）
UNDERSTAT_COVERED = {
    # 英超
    "英超", "英格兰超级联赛", "EPL", "Premier League",
    # 西甲
    "西甲", "西班牙甲级联赛", "西班牙足球甲级联赛", "La Liga",
    # 意甲
    "意甲", "意大利甲级联赛", "Serie A",
    # 德甲
    "德甲", "德国甲级联赛", "Bundesliga",
    # 法甲
    "法甲", "法国甲级联赛", "Ligue 1",
    # 俄超
    "俄超", "俄罗斯超级联赛", "RPL", "Russian Premier League",
}

# 部分覆盖（understat 历史只覆盖到某一年/部分赛季）
UNDERSTAT_PARTIAL = {
    "MLS", "美职足", "美国职业足球大联盟",
}

# 明确不覆盖的常见 500.com 联赛
NOT_COVERED_HINTS = (
    "中超", "中甲",
    "日职", "J1", "J联赛",
    "韩职", "K联赛",
    "巴甲", "巴乙",
    "阿甲", "阿乙",
    "墨西", "墨甲",
    "瑞典", "瑞超", "瑞典超",
    "挪威", "挪超",
    "丹麦", "丹超",
    "芬兰", "芬超",
    "冰岛",
    "智利", "智甲",
    "哥伦比亚", "哥甲",
    "葡超", "葡萄牙",
    "荷甲",  # eredivisie understat 没有官方页
    "比甲",
    "苏超",  # 苏格兰超
    "土超",
    "希腊",
    "瑞士",
    "奥甲", "奥地利",
    "捷克", "捷甲",
    "波兰",
    "乌克兰",
    "塞浦路斯",
    "塞尔维亚",
    "克罗地亚",
    "罗马尼亚",
    "保加利亚",
    "斯洛文",
    "斯洛伐克",
    "匈牙利",
    # 欧洲赛事 / 杯赛
    "欧冠", "欧联", "欧协", "亚冠", "世俱杯", "南美解放者", "美洲杯",
    "杯", "附加", "资格", "友谊",
)


def _classify(league: str) -> str:
    text = (league or "").strip()
    if not text:
        return "unknown_empty"
    if any(token in text for token in UNDERSTAT_COVERED):
        return "covered"
    if any(token in text for token in UNDERSTAT_PARTIAL):
        return "partial"
    if any(token in text for token in NOT_COVERED_HINTS):
        return "not_covered"
    return "unknown_other"


def main() -> None:
    db_path: Path = repo.PRIMARY_DB_PATH
    if not db_path.exists():
        print(f"找不到数据库：{db_path}")
        return

    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        # 1) 全部 matches 的联赛分布
        all_leagues = [
            str(r["league"] or "")
            for r in conn.execute("SELECT league FROM matches").fetchall()
        ]
        # 2) 已结算 feedback 关联的联赛分布（这是真正能用来回测的样本）
        settled_leagues = [
            str(r["league"] or "")
            for r in conn.execute(
                """
                SELECT m.league
                FROM feedback_logs f
                JOIN matches m ON m.match_id = f.match_id
                """
            ).fetchall()
        ]

    def _summarize(name: str, leagues: list[str]) -> None:
        total = len(leagues)
        if total == 0:
            print(f"\n{name}: 无样本")
            return

        buckets = Counter(_classify(lg) for lg in leagues)
        league_counter = Counter(leagues)
        coverage_pct = (buckets["covered"] + buckets["partial"]) / total * 100

        print(f"\n{name}: 共 {total} 场")
        print(f"  ✓ understat 覆盖:        {buckets['covered']} 场 ({buckets['covered']/total*100:.1f}%)")
        print(f"  ~ understat 部分覆盖:    {buckets['partial']} 场 ({buckets['partial']/total*100:.1f}%)")
        print(f"  ✗ understat 不覆盖:      {buckets['not_covered']} 场 ({buckets['not_covered']/total*100:.1f}%)")
        print(f"  ? 未识别:                {buckets['unknown_other'] + buckets['unknown_empty']} 场")
        print(f"  → 估算覆盖率:            {coverage_pct:.1f}%")

        # 列出 top 10 联赛
        print(f"  详细分布（top 12）:")
        for league, count in league_counter.most_common(12):
            cls = _classify(league)
            mark = {"covered": "✓", "partial": "~", "not_covered": "✗"}.get(cls, "?")
            print(f"    {mark} {league:20s}  {count} 场")

    _summarize("全部 matches", all_leagues)
    _summarize("已结算（可回测）", settled_leagues)

    print()
    print("=" * 60)
    print("决策建议")
    print("=" * 60)

    settled_total = len(settled_leagues)
    if settled_total == 0:
        print("没有已结算样本，无法判断。")
        return

    settled_buckets = Counter(_classify(lg) for lg in settled_leagues)
    coverage = (settled_buckets["covered"] + settled_buckets["partial"]) / settled_total

    if coverage >= 0.70:
        print(f"  ✓ 覆盖率 {coverage*100:.1f}% — 值得做 D（xG 接入）")
        print("    可以直接开始写 source_understat_client.py")
    elif coverage >= 0.45:
        print(f"  ⚠ 覆盖率 {coverage*100:.1f}% — 边缘地带")
        print("    做 xG 接入只能改善一半场次，模型行为会出现两套表现。")
        print("    建议先把策略层稳定好，等样本量上去再说。")
    else:
        print(f"  ✗ 覆盖率 {coverage*100:.1f}% — 不建议做 D")
        print("    你的胜负彩组合主要是非主流联赛，understat 帮不上。")
        print("    可考虑：")
        print("    - 改用更广覆盖的赔率衍生特征（closing line value）")
        print("    - 或者集成 SofaScore / FlashScore 的非官方 xG 估算")
        print("    - 或者承认独立模型有上限，把 α 进一步降到市场为主")


if __name__ == "__main__":
    main()
