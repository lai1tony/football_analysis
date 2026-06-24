"""500.com 中文队名 → understat 英文队名映射。

设计：
- 6 大联赛（英超 / 西甲 / 意甲 / 德甲 / 法甲 / 俄超）当前赛季所有球队
  + 常见上赛季降级队（用于回溯历史比赛）。
- 每个中文名映射到 *候选英文名列表*：把 understat 上可能出现的写法都列出来，
  让 lookup_team_in_league 按顺序尝试匹配。
- 不做赛季过滤，understat 当前赛季页面里没有的球队会自然 miss，不会乱匹配。

用法：
    from team_name_aliases import resolve_team_aliases
    titles = resolve_team_aliases("巴塞罗那")  # ["Barcelona", "FC Barcelona"]

500.com 队名爬自 source_500_client。常见的不规范写法（"巴萨" vs "巴塞罗那"）
都做了归一。
"""
from __future__ import annotations


# 主键 = 500.com 用的标准中文名；值 = understat 标题候选列表。
TEAM_ALIASES: dict[str, list[str]] = {
    # ============ 英超 EPL ============
    "阿森纳": ["Arsenal"],
    "阿斯顿维拉": ["Aston Villa"],
    "维拉": ["Aston Villa"],
    "伯恩茅斯": ["Bournemouth"],
    "伯恩茅": ["Bournemouth"],  # 500.com 截断
    "布伦特福德": ["Brentford"],
    "布伦特": ["Brentford"],  # 500.com 截断
    "布莱顿": ["Brighton"],
    "切尔西": ["Chelsea"],
    "水晶宫": ["Crystal Palace"],
    "埃弗顿": ["Everton"],
    "富勒姆": ["Fulham"],
    "伊普斯维奇": ["Ipswich"],
    "莱斯特": ["Leicester"],
    "莱斯特城": ["Leicester"],
    "利物浦": ["Liverpool"],
    "曼城": ["Manchester City"],
    "曼彻斯特城": ["Manchester City"],
    "曼联": ["Manchester United"],
    "曼彻斯特联": ["Manchester United"],
    "纽卡": ["Newcastle United", "Newcastle"],
    "纽卡斯尔": ["Newcastle United", "Newcastle"],
    "纽卡斯尔联": ["Newcastle United", "Newcastle"],
    "诺丁汉森林": ["Nottingham Forest"],
    "诺丁汉": ["Nottingham Forest"],
    "南安普敦": ["Southampton"],
    "热刺": ["Tottenham"],
    "托特纳姆": ["Tottenham"],
    "西汉姆": ["West Ham"],
    "西汉姆联": ["West Ham"],
    "狼队": ["Wolverhampton Wanderers", "Wolves"],
    "卢顿": ["Luton"],
    "谢菲尔德联": ["Sheffield United"],
    "伯恩利": ["Burnley"],
    "利兹": ["Leeds"],
    "利兹联": ["Leeds"],
    "桑德兰": ["Sunderland"],

    # ============ 西甲 La Liga ============
    "皇家马德里": ["Real Madrid"],
    "皇马": ["Real Madrid"],
    "巴塞罗那": ["Barcelona"],
    "巴萨": ["Barcelona"],
    "马德里竞技": ["Atletico Madrid"],
    "马竞": ["Atletico Madrid"],
    "马竞技": ["Atletico Madrid"],  # 500.com 截断
    "毕尔巴鄂竞技": ["Athletic Club"],
    "毕尔巴鄂": ["Athletic Club"],
    "毕尔巴": ["Athletic Club"],  # 500.com 截断写法
    "塞维利亚": ["Sevilla"],
    "塞维利": ["Sevilla"],  # 500.com 截断
    "皇家社会": ["Real Sociedad"],
    "皇社": ["Real Sociedad"],
    "社会": ["Real Sociedad"],  # 500.com 截断（"皇家社会"）
    "贝蒂斯": ["Real Betis"],
    "皇家贝蒂斯": ["Real Betis"],
    "比利亚雷亚尔": ["Villarreal"],
    "比利亚雷": ["Villarreal"],  # 500.com 截断
    "瓦伦西亚": ["Valencia"],
    "巴伦西亚": ["Valencia"],
    "巴伦西": ["Valencia"],  # 500.com 截断
    "西班牙人": ["Espanyol"],
    "西班人": ["Espanyol"],  # 500.com 截断
    "赫塔费": ["Getafe"],
    "赫塔菲": ["Getafe"],  # 异写
    "马略卡": ["Mallorca"],
    "马洛卡": ["Mallorca"],
    "拉斯帕尔马斯": ["Las Palmas"],
    "拉斯帕尔": ["Las Palmas"],  # 500.com 截断
    "塞尔塔": ["Celta Vigo"],
    "塞尔塔维戈": ["Celta Vigo"],
    "奥萨苏纳": ["Osasuna"],
    "奥萨苏": ["Osasuna"],  # 500.com 截断
    "莱加内斯": ["Leganes"],
    "巴利亚多利德": ["Real Valladolid"],
    "巴拉多利德": ["Real Valladolid"],
    "巴利亚多": ["Real Valladolid"],  # 500.com 截断
    "赫罗纳": ["Girona"],
    "拉约": ["Rayo Vallecano"],
    "拉约·巴列卡诺": ["Rayo Vallecano"],
    "巴列卡诺": ["Rayo Vallecano"],
    "巴列卡": ["Rayo Vallecano"],  # 500.com 截断
    "格拉纳达": ["Granada"],
    "阿尔梅里亚": ["Almeria"],
    "阿尔梅里": ["Almeria"],  # 截断
    "加的斯": ["Cadiz"],
    "埃尔切": ["Elche"],
    "莱万特": ["Levante"],
    "奥维耶多": ["Real Oviedo"],
    "奥维耶": ["Real Oviedo"],  # 500.com 截断
    "皇家奥维耶多": ["Real Oviedo"],
    "阿拉维斯": ["Alaves"],
    "阿拉维": ["Alaves"],  # 500.com 截断

    # ============ 意甲 Serie A ============
    "国际米兰": ["Inter"],
    "国米": ["Inter"],
    "AC米兰": ["AC Milan", "Milan"],
    "AC米": ["AC Milan", "Milan"],
    "米兰": ["AC Milan", "Milan"],
    "尤文图斯": ["Juventus"],
    "尤文": ["Juventus"],
    "罗马": ["Roma"],
    "拉齐奥": ["Lazio"],
    "那不勒斯": ["Napoli"],
    "那不勒": ["Napoli"],  # 500.com 截断
    "亚特兰大": ["Atalanta"],
    "亚特兰": ["Atalanta"],  # 500.com 截断
    "佛罗伦萨": ["Fiorentina"],
    "博洛尼亚": ["Bologna"],
    "都灵": ["Torino"],
    "热那亚": ["Genoa"],
    "维罗纳": ["Hellas Verona", "Verona"],
    "莱切": ["Lecce"],
    "卡利亚里": ["Cagliari"],
    "乌迪内斯": ["Udinese"],
    "恩波利": ["Empoli"],
    "蒙扎": ["Monza"],
    "威尼斯": ["Venezia"],
    "科莫": ["Como"],
    "帕尔马": ["Parma"],
    "桑普多利亚": ["Sampdoria"],
    "斯佩齐亚": ["Spezia"],
    "克雷莫纳": ["Cremonese"],
    "弗罗西诺内": ["Frosinone"],
    "萨索洛": ["Sassuolo"],
    "皮萨": ["Pisa"],
    "比萨": ["Pisa"],

    # ============ 德甲 Bundesliga ============
    "拜仁": ["Bayern Munich"],
    "拜仁慕尼黑": ["Bayern Munich"],
    "多特": ["Borussia Dortmund"],
    "多特蒙德": ["Borussia Dortmund"],
    "勒沃库森": ["Bayer Leverkusen"],
    "勒沃": ["Bayer Leverkusen"],  # 500.com 截断
    "莱比锡": ["RasenBallsport Leipzig", "RB Leipzig"],
    "RB莱比锡": ["RasenBallsport Leipzig", "RB Leipzig"],
    "莱比锡红牛": ["RasenBallsport Leipzig", "RB Leipzig"],
    "法兰克福": ["Eintracht Frankfurt"],
    "斯图加特": ["VfB Stuttgart", "Stuttgart"],
    "斯图加": ["VfB Stuttgart", "Stuttgart"],  # 500.com 截断
    "沃尔夫斯堡": ["Wolfsburg"],
    "沃尔夫": ["Wolfsburg"],  # 500.com 截断
    "门兴": ["Borussia M.Gladbach", "Borussia Monchengladbach"],
    "门兴格拉德巴赫": ["Borussia M.Gladbach", "Borussia Monchengladbach"],
    "弗赖堡": ["SC Freiburg", "Freiburg"],
    "美因茨": ["Mainz 05", "Mainz"],
    "霍芬海姆": ["Hoffenheim"],
    "霍芬海": ["Hoffenheim"],  # 500.com 截断
    "奥格斯堡": ["FC Augsburg", "Augsburg"],
    "奥格斯": ["FC Augsburg", "Augsburg"],  # 500.com 截断
    "海登海姆": ["1. FC Heidenheim 1846", "FC Heidenheim", "Heidenheim"],
    "海登": ["1. FC Heidenheim 1846", "FC Heidenheim", "Heidenheim"],
    "云达不莱梅": ["Werder Bremen"],
    "不莱梅": ["Werder Bremen"],
    "不来梅": ["Werder Bremen"],  # 500.com 用 "来" 不用 "莱"
    "波鸿": ["VfL Bochum", "Bochum"],
    "圣保利": ["FC St. Pauli", "St. Pauli"],
    "基尔": ["Holstein Kiel"],
    "联合柏林": ["Union Berlin"],
    "柏林联合": ["Union Berlin"],
    "柏林联": ["Union Berlin"],  # 500.com 截断
    "科隆": ["FC Cologne", "Cologne"],
    "汉堡": ["Hamburger SV", "Hamburg"],
    "达姆斯塔特": ["SV Darmstadt 98"],
    "沙尔克": ["Schalke 04"],
    "比勒费尔德": ["Arminia Bielefeld"],
    "汉诺威": ["Hannover 96"],

    # ============ 法甲 Ligue 1 ============
    "巴黎圣日耳曼": ["Paris Saint Germain", "PSG"],
    "巴黎圣日尔曼": ["Paris Saint Germain", "PSG"],
    "日耳曼": ["Paris Saint Germain", "PSG"],
    "日尔曼": ["Paris Saint Germain", "PSG"],  # 500.com 别字
    "巴黎": ["Paris Saint Germain", "PSG"],
    "PSG": ["Paris Saint Germain", "PSG"],
    "马赛": ["Marseille"],
    "里昂": ["Lyon"],
    "摩纳哥": ["Monaco"],
    "里尔": ["Lille"],
    "尼斯": ["Nice"],
    "雷恩": ["Rennes"],
    "斯特拉斯堡": ["Strasbourg"],
    "斯特堡": ["Strasbourg"],  # 500.com 截断（斯特拉斯堡）
    "图卢兹": ["Toulouse"],
    "南特": ["Nantes"],
    "欧塞尔": ["Auxerre"],
    "蒙彼利埃": ["Montpellier"],
    "兰斯": ["Reims"],
    "勒阿弗尔": ["Le Havre"],
    "圣埃蒂安": ["Saint-Etienne"],
    "昂热": ["Angers"],
    "布雷斯特": ["Brest"],
    "布雷斯": ["Brest"],  # 500.com 截断
    "朗斯": ["Lens"],
    "梅斯": ["Metz"],
    "克莱蒙": ["Clermont Foot"],
    "洛里昂": ["Lorient"],
    "巴黎FC": ["Paris FC"],

    # ============ 俄超 RPL ============
    "泽尼特": ["Zenit"],
    "圣彼得堡泽尼特": ["Zenit"],
    "莫斯科斯巴达": ["Spartak Moscow"],
    "斯巴达克": ["Spartak Moscow"],
    "莫斯科中央陆军": ["CSKA Moscow"],
    "中央陆军": ["CSKA Moscow"],
    "莫斯科火车头": ["Lokomotiv Moscow"],
    "火车头": ["Lokomotiv Moscow"],
    "莫斯科迪纳摩": ["Dinamo Moscow", "Dynamo Moscow"],
    "迪纳摩": ["Dinamo Moscow", "Dynamo Moscow"],
    "克拉斯诺达尔": ["Krasnodar"],
    "罗斯托夫": ["FC Rostov", "Rostov"],
    "格罗兹尼": ["Akhmat Grozny"],
    "阿赫马特": ["Akhmat Grozny"],
    "下诺夫哥罗德": ["Pari Nizhny Novgorod"],
    "奥伦堡": ["Orenburg"],
    "下塔吉尔": ["Ural"],
    "乌拉尔": ["Ural"],
    "图拉兵工厂": ["Arsenal Tula"],
    "莫斯科火车头2": ["Lokomotiv Moscow"],
    "喀山红宝石": ["Rubin Kazan"],
    "鲁宾": ["Rubin Kazan"],
    "金环": ["Krylya Sovetov", "Krylia Sovetov"],
    "翼": ["Krylya Sovetov", "Krylia Sovetov"],
    "下诺夫哥罗德帕里": ["Pari Nizhny Novgorod"],
    "希姆基": ["Khimki"],
    "索契": ["Sochi"],
    "伏尔加格勒": ["Rotor Volgograd"],
    "罗托尔": ["Rotor Volgograd"],
}


# 把每条联赛归类，给后续 league-aware 查找用（可选）
LEAGUE_SHORT_NAME_KEYS = {
    "EPL": ("英超",),
    "La_liga": ("西甲",),
    "Serie_A": ("意甲",),
    "Bundesliga": ("德甲",),
    "Ligue_1": ("法甲",),
    "RFPL": ("俄超",),
}


def resolve_team_aliases(chinese_name: str) -> list[str]:
    """中文队名 → understat 候选英文名列表。

    返回空列表表示这个名字不在我们维护的范围内（比如非主流联赛）。
    `lookup_team_in_league` 会自动 fallback 到原名 lowercase 直接匹配，
    所以即使没列在这里，球队名是单一英文 token 时仍可能命中。
    """

    text = (chinese_name or "").strip()
    if not text:
        return []
    if text in TEAM_ALIASES:
        return TEAM_ALIASES[text]
    # 偶尔 500.com 队名带括号注释，剥掉再试
    cleaned = text.split("（")[0].split("(")[0].strip()
    if cleaned and cleaned != text and cleaned in TEAM_ALIASES:
        return TEAM_ALIASES[cleaned]
    return [text]  # 让上层用原名做最后一次模糊匹配


def league_codes_for_500_label(league_label: str) -> list[str]:
    """从 500.com 的联赛中文标签推断对应的 understat 联赛代码。"""

    label = (league_label or "").strip()
    if not label:
        return []
    out: list[str] = []
    for code, keywords in LEAGUE_SHORT_NAME_KEYS.items():
        if any(keyword in label for keyword in keywords):
            out.append(code)
    return out


__all__ = [
    "LEAGUE_SHORT_NAME_KEYS",
    "TEAM_ALIASES",
    "league_codes_for_500_label",
    "resolve_team_aliases",
]
