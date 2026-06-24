"""补充采集客户端：当 500.com 主源缺失维度数据时，通过免费公开资源补采。

设计原则：
1. 免费公开资源优先，不使用付费 API
2. 数据时效性：1年以内
3. 使用 playwright-cli 渲染 JS 重度页面
4. 仅作为主采集流程的后置补充，不替代 500.com 主源

当前支持的补充源：
- Flashscore (flashscore.com)：交锋记录、近期状态、伤停阵容
- Understat (understat.com)：xG 实力代理（替代 Elo）
- Soccerway (soccerway.com)：交锋记录、近期战绩

补充维度映射：
- 维度一（基础实力）：用 Understat xG/xGA 替代 Elo
- 维度二（近期动态）：用 Flashscore/Soccerway 补采交锋记录、伤停
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Mapping
from urllib.parse import quote_plus

from bs4 import BeautifulSoup

from playwright_cli_client import PlaywrightCliBrowser, PlaywrightCliSettings
from source_market_value_client import fetch_match_market_values

# ─── 工具函数 ───────────────────────────────────────────────────────────────


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def ascii_fold(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    return normalized.encode("ascii", "ignore").decode("ascii").lower()


def unique_non_empty(values: list[str]) -> list[str]:
    seen: set[str] = set()
    results: list[str] = []
    for v in values:
        t = normalize_text(v)
        if not t or t in seen:
            continue
        seen.add(t)
        results.append(t)
    return results


def to_float(text: str) -> float:
    if not text:
        return 0.0
    m = re.match(r"\s*(-?\d+(?:\.\d+)?)", text)
    if not m:
        return 0.0
    try:
        return float(m.group(1))
    except ValueError:
        return 0.0


def to_int(text: str) -> int:
    if not text:
        return 0
    m = re.match(r"\s*(-?\d+)", text)
    if not m:
        return 0
    try:
        return int(m.group(1))
    except ValueError:
        return 0


def current_season() -> str:
    from datetime import datetime
    now = datetime.now()
    return str(now.year if now.month >= 7 else now.year - 1)


# ─── 球队中文名 -> 英文名映射（用于 Flashscore 搜索）──────────────────────────

# 常见球队中文名到英文名的映射（覆盖胜负彩常见球队）
_TEAM_CN_EN_MAP: dict[str, str] = {
    # 荷乙
    "福伦丹": "FC Volendam",
    "威廉二世": "Willem II Tilburg",
    "坎布尔": "SC Cambuur",
    "海牙": "ADO Den Haag",
    "格拉夫夏普": "De Graafschap",
    "阿尔梅勒城": "Almere City",
    "瓦尔韦克": "RKC Waalwijk",
    "罗达JC": "Roda JC",
    "邓博施": "FC Den Bosch",
    "多德勒支": "FC Dordrecht",
    "埃门": "FC Emmen",
    "维特斯": "Vitesse",
    "海尔蒙特": "Helmond Sport",
    "马斯特里赫特": "MVV Maastricht",
    "奥斯": "VVV Venlo",
    "特尔斯达": "Telstar",
    "埃因霍温青年队": "Jong PSV",
    "埃因霍温FC": "FC Eindhoven",
    "乌德勒支青年队": "Jong Utrecht",
    "阿贾克斯青年队": "Jong Ajax",
    "阿尔克马尔青年队": "Jong AZ",
    # 芬超
    "库普斯": "KuPS",
    "赫尔辛基": "HJK Helsinki",
    "国际图尔库": "Inter Turku",
    # 瑞典超
    "马尔默": "Malmö FF",
    "赫根": "BK Häcken",
    "哈马比": "Hammarby IF",
    "佐加顿斯": "Djurgårdens IF",
    "索尔纳": "AIK",
    "埃尔夫斯堡": "IF Elfsborg",
    "哥德堡": "IFK Göteborg",
    # 挪超
    "博德闪耀": "Bodø/Glimt",
    "莫尔德": "Molde",
    "罗森博格": "Rosenborg",
    "维京": "Viking",
    "布兰": "Brann",
    # 丹超
    "哥本哈根": "FC Copenhagen",
    "布隆德比": "Brøndby",
    "奥胡斯": "AGF",
    "中日德兰": "FC Midtjylland",
    # 瑞士超
    "年轻人": "Young Boys",
    "巴塞尔": "FC Basel",
    "苏黎世": "FC Zürich",
    # 奥超
    "萨尔茨堡": "Red Bull Salzburg",
    "维也纳快速": "Rapid Wien",
    # 苏超
    "凯尔特人": "Celtic",
    "流浪者": "Rangers",
    # 比甲
    "布鲁日": "Club Brugge",
    "安德莱赫特": "Anderlecht",
    # 土超
    "加拉塔萨雷": "Galatasaray",
    "费内巴切": "Fenerbahçe",
    "贝西克塔斯": "Beşiktaş",
    # 希超
    "奥林匹亚科斯": "Olympiacos",
    "帕纳辛纳科斯": "Panathinaikos",
    # 俄超
    "泽尼特": "Zenit",
    "莫斯科斯巴达": "Spartak Moscow",
    "莫斯科中央陆军": "CSKA Moscow",
    # 乌超
    "基辅迪纳摩": "Dynamo Kyiv",
    "顿涅茨克矿工": "Shakhtar Donetsk",
    # 波超
    "华沙莱吉亚": "Legia Warsaw",
    # 捷超
    "布拉格斯拉维亚": "Slavia Prague",
    # 罗超
    "布加勒斯特星": "FCSB",
    # 塞尔超",
    "贝尔格莱德红星": "Red Star Belgrade",
    "贝尔格莱德游击": "Partizan",
    # 克超
    "萨格勒布迪纳摩": "Dinamo Zagreb",
    # 保超
    "卢多戈雷茨": "Ludogorets",
    # 爱超
    "沙姆洛克流浪者": "Shamrock Rovers",
    # 威超
    "新圣徒": "The New Saints",
}


def _resolve_team_en_name(team_name: str) -> str:
    """尝试将球队中文名解析为英文名（用于 Flashscore 搜索）。"""
    # 直接匹配
    if team_name in _TEAM_CN_EN_MAP:
        return _TEAM_CN_EN_MAP[team_name]
    # 模糊匹配（去掉"FC"等前缀）
    cleaned = team_name.replace("FC ", "").replace(" AFC", "").strip()
    if cleaned in _TEAM_CN_EN_MAP:
        return _TEAM_CN_EN_MAP[cleaned]
    # 返回原名（可能已经是英文名）
    return team_name


# ─── Flashscore 补采 ────────────────────────────────────────────────────────

FLASHSCORE_DOMAINS = (
    "flashscore.com",
    "flashscore.de",
    "flashscore.co.uk",
    "t.flashscore.com",
)

FIND_LINEUP_TAB_JS = (
    "(() => { "
    "const anchors = Array.from(document.querySelectorAll('a[href]')); "
    "const hit = anchors.find(a => /kader|lineup|aufstellung|startaufstellung|aufstellungen/i.test(((a.textContent||'').trim())+' '+(a.href||''))); "
    "return hit ? hit.href : ''; "
    "})()"
)

SEARCH_LINKS_JS = (
    "JSON.stringify("
    "[...document.querySelectorAll('a[href]')]"
    ".slice(0, 180)"
    ".map(a => ({ text: (a.textContent||'').trim(), href: a.href }))"
    ")"
)


def _normalize_fs_url(url: str) -> str:
    from urllib.parse import urlparse, urlunparse
    parsed = urlparse(url)
    host = parsed.netloc.lower().lstrip("www.")
    if not any(host == d or host.endswith(f".{d}") for d in FLASHSCORE_DOMAINS):
        return ""
    path = parsed.path.rstrip("/")
    path = re.sub(r"/(?:bericht|summary)(?:/.*)?$", "", path, flags=re.I)
    if not re.search(r"/(spiel|match|game)/", path, re.I):
        return ""
    return urlunparse(("https", parsed.netloc, f"{path}/", "", "", ""))


def _score_fs_candidate(href: str, text: str, home: str, away: str) -> int:
    haystack = ascii_fold(f"{href} {text}")
    score = 0
    home_parts = ascii_fold(home).split()
    away_parts = ascii_fold(away).split()
    if any(p in haystack for p in home_parts if len(p) > 2):
        score += 4
    if any(p in haystack for p in away_parts if len(p) > 2):
        score += 4
    if re.search(r"/(spiel|match|game)/", href, re.I):
        score += 3
    if "flashscore" in haystack:
        score += 1
    if re.search(r"/team/|/standings|/fixtures", href, re.I):
        score -= 5
    return score


def _search_fs_match(browser: PlaywrightCliBrowser, home: str, away: str, match_date: str = "") -> str:
    """搜索 Flashscore 找到对赛页 URL。"""
    stopwords = {"fc", "cf", "sc", "ac", "afc", "bk", "fk", "sv", "ud", "cd", "club", "team"}
    home_parts = [p for p in ascii_fold(home).split() if p not in stopwords and len(p) > 2]
    away_parts = [p for p in ascii_fold(away).split() if p not in stopwords and len(p) > 2]

    queries = []
    for hp in home_parts[:2]:
        for ap in away_parts[:2]:
            if match_date:
                queries.append(f"{hp} {ap} {match_date} flashscore")
            queries.append(f"{hp} {ap} flashscore")
    queries = unique_non_empty(queries)[:6]

    for query in queries:
        for engine, tpl in [
            ("bing", f"https://www.bing.com/search?q={quote_plus(query)}"),
            ("yahoo", f"https://search.yahoo.com/search?p={quote_plus(query)}"),
        ]:
            try:
                browser.goto(tpl)
                title = browser.title()
                body = browser.body_text()[:3000]
                if "one last step" in body.lower() or "solve this puzzle" in body.lower():
                    continue
                raw = browser.eval(SEARCH_LINKS_JS) or []
                if isinstance(raw, str):
                    try:
                        raw = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                scored = []
                for item in raw:
                    href = normalize_text(str(item.get("href", "")))
                    text = normalize_text(str(item.get("text", "")))
                    if not href:
                        continue
                    from urllib.parse import urlparse
                    if not any(
                        urlparse(href).netloc.lower().lstrip("www.") == d
                        or urlparse(href).netloc.lower().lstrip("www.").endswith(f".{d}")
                        for d in FLASHSCORE_DOMAINS
                    ):
                        continue
                    norm = _normalize_fs_url(href)
                    if not norm:
                        continue
                    scored.append((_score_fs_candidate(href, text, home, away), norm))
                scored.sort(key=lambda x: x[0], reverse=True)
                if scored:
                    return scored[0][1]
            except Exception:
                continue
    return ""


def _section_side_texts_js(pattern: str) -> str:
    escaped = pattern.replace("\\", "\\\\").replace("'", "\\'")
    return (
        "(() => { "
        "const sections = Array.from(document.querySelectorAll('.section, section')); "
        f"const section = sections.find(item => /{escaped}/i.test((item.innerText||''))); "
        "if (!section) return '[]'; "
        "return JSON.stringify(Array.from(section.querySelectorAll('.lf__side')).slice(0,2).map(s=>s.innerText||'')); "
        "})()"
    )


def _parse_lineup_side(text: str) -> list[str]:
    players = []
    for raw_line in (text or "").splitlines():
        line = normalize_text(raw_line).replace(" (G)", "")
        folded = ascii_fold(line)
        if not line or line.isdigit() or re.fullmatch(r"\([A-Z]{1,3}\)", line):
            continue
        if any(kw in folded for kw in ("formation", "predicted", "starting", "lineups")):
            continue
        players.append(line)
    return unique_non_empty(players)[:11]


def _parse_missing_side(text: str, status: str) -> list[dict]:
    lines = [normalize_text(l) for l in (text or "").splitlines() if normalize_text(l)]
    results = []
    i = 0
    while i < len(lines):
        player = normalize_text(lines[i]).replace(" (G)", "")
        reason = ""
        if i + 1 < len(lines):
            cand = normalize_text(lines[i + 1])
            if not cand.isdigit() and not re.search(r"\.$", cand):
                reason = cand
                i += 1
        if player and not player.isdigit():
            results.append({"player": player, "reason": reason, status: status})
        i += 1
    return results


def _extract_fs_h2h(browser: PlaywrightCliBrowser) -> str:
    """从 Flashscore 对赛页提取交锋记录文本。"""
    try:
        js = (
            "(() => { "
            "const tab = Array.from(document.querySelectorAll('a[href]')).find(a => "
            "/head.?to.?head|direktvergleich|confrontation|historial/i.test((a.textContent||'').trim()+' '+(a.href||''))); "
            "return tab ? tab.href : ''; "
            "})()"
        )
        h2h_url = normalize_text(str(browser.eval(js) or ""))
        if h2h_url:
            browser.goto(h2h_url)

        # 尝试多种选择器提取交锋记录
        js_code = (
            "(() => { "
            "const rows = document.querySelectorAll("
            "'.h2h__row, .head-to-head tr, .h2h tbody tr, [class*=\"h2h\"] tr, [class*=\"head\"] tr'"
            "); "
            "const out = []; "
            "rows.forEach(r => { const t = (r.innerText||'').trim(); "
            "if(t && t.length > 10 && t.length < 200) out.push(t); }); "
            "return JSON.stringify(out.slice(0, 10)); "
            "})()"
        )
        result = browser.eval(js_code)
        if isinstance(result, str):
            try:
                rows = json.loads(result or "[]")
            except json.JSONDecodeError:
                rows = []
        else:
            rows = result or []

        if rows:
            return "；".join(str(r) for r in rows[:5])

        # 回退：用 body_text 中找交锋关键词附近的内容
        body = browser.body_text()[:8000]
        patterns = [
            r"(?:近|Past|Recent|Head-to-Head|H2H)[^\n]*\n((?:[^\n]+\n){3,8})",
            r"(?:交战|meetings|Matches)[^\n]*\n((?:[^\n]+\n){3,8})",
        ]
        for pat in patterns:
            m = re.search(pat, body, re.I)
            if m:
                lines = [normalize_text(l) for l in m.group(1).splitlines() if normalize_text(l)]
                return "；".join(lines[:5])

        return ""
    except Exception:
        return ""


def _extract_fs_recent_form(browser: PlaywrightCliBrowser, team_name: str) -> str:
    """从 Flashscore 对赛页提取某队近期状态。"""
    try:
        body = browser.body_text()[:10000]
        team_key = ascii_fold(team_name)
        # 找包含队名关键词的近期战绩段落
        patterns = [
            rf"{re.escape(team_key)}[^\n]*(?:近|recent|last)[^\n]*\n((?:[^\n]+胜[^\n]+负[^\n]*\n){{1,5}})",
            rf"(?:近|recent|last)\s*\d+\s*(?:场|matches|games)[^\n]*{re.escape(team_key)}[^\n]*\n((?:[^\n]+\n){{1,5}})",
        ]
        for pat in patterns:
            m = re.search(pat, body, re.I)
            if m:
                text = normalize_text(m.group(1))
                if text:
                    return text
        return ""
    except Exception:
        return ""


def supplement_from_flashscore(
    home_team: str,
    away_team: str,
    match_time: str = "",
) -> dict[str, Any]:
    """通过 Flashscore 补采维度二数据。

    Returns:
        {
            "head_to_head_summary": str,
            "recent_form_home": str,
            "recent_form_away": str,
            "injury_or_lineup_notes": str,
            "home_lineup": list,
            "away_lineup": list,
            "home_missing": list,
            "away_missing": list,
            "source_url": str,
            "success": bool,
            "error": str,
        }
    """
    result = {
        "head_to_head_summary": "",
        "recent_form_home": "",
        "recent_form_away": "",
        "injury_or_lineup_notes": "",
        "home_lineup": [],
        "away_lineup": [],
        "home_missing": [],
        "away_missing": [],
        "source_url": "",
        "success": False,
        "error": "",
    }

    try:
        browser = PlaywrightCliBrowser(PlaywrightCliSettings.from_env())
    except Exception as exc:
        result["error"] = f"playwright-cli 不可用：{exc}"
        return result

    try:
        # 0. 解析英文名（Flashscore 搜索需要英文名）
        home_en = _resolve_team_en_name(home_team)
        away_en = _resolve_team_en_name(away_team)
        search_home = home_en if home_en != home_team else home_team
        search_away = away_en if away_en != away_team else away_team

        # 1. 搜索对赛页（优先英文名，失败时回退到原名）
        match_url = _search_fs_match(browser, search_home, search_away, match_time)
        if not match_url and (home_en != home_team or away_en != away_team):
            # 英文名搜索失败，回退到原名
            match_url = _search_fs_match(browser, home_team, away_team, match_time)
        if not match_url:
            result["error"] = f"Flashscore 未找到对赛页：{home_team} vs {away_team}"
            return result

        result["source_url"] = match_url

        # 2. 提取交锋记录
        result["head_to_head_summary"] = _extract_fs_h2h(browser)

        # 3. 提取伤停/阵容
        try:
            browser.goto(match_url)
            lineup_url = normalize_text(str(browser.eval(FIND_LINEUP_TAB_JS) or ""))
            if not lineup_url:
                current = browser.page_url()
                if re.search(r"/bericht/kader|/summary/lineups", current, re.I):
                    lineup_url = current
            if lineup_url:
                browser.goto(lineup_url)
                sides = browser.eval(
                    _section_side_texts_js("startaufstellung|predicted lineup|predicted lineups|lineups|aufstellungen")
                )
                missing = browser.eval(
                    _section_side_texts_js("wird nicht spielen|missing players|injuries|suspensions|will not play")
                )
                doubtful = browser.eval(
                    _section_side_texts_js("fraglich|questionable|doubtful")
                )
                if isinstance(sides, str):
                    sides = json.loads(sides or "[]")
                if isinstance(missing, str):
                    missing = json.loads(missing or "[]")
                if isinstance(doubtful, str):
                    doubtful = json.loads(doubtful or "[]")

                result["home_lineup"] = _parse_lineup_side(sides[0] if len(sides) > 0 else "")
                result["away_lineup"] = _parse_lineup_side(sides[1] if len(sides) > 1 else "")
                result["home_missing"] = _parse_missing_side(missing[0] if len(missing) > 0 else "", "out")
                result["away_missing"] = _parse_missing_side(missing[1] if len(missing) > 1 else "", "out")
                result["home_missing"].extend(_parse_missing_side(doubtful[0] if len(doubtful) > 0 else "", "doubtful"))
                result["away_missing"].extend(_parse_missing_side(doubtful[1] if len(doubtful) > 1 else "", "doubtful"))

                # 生成伤停文本
                parts = ["预计首发"]
                parts.append(f"主队：{' / '.join(result['home_lineup']) if result['home_lineup'] else '未命中'}")
                parts.append(f"客队：{' / '.join(result['away_lineup']) if result['away_lineup'] else '未命中'}")
                hm = [f"{m['player']}({m['reason'] or '缺阵'})" for m in result["home_missing"][:8]]
                am = [f"{m['player']}({m['reason'] or '缺阵'})" for m in result["away_missing"][:8]]
                parts.append(f"关键伤停：主队{'；'.join(hm) or '无明确伤停'}；客队{'；'.join(am) or '无明确伤停'}")
                parts.append(f"来源：Flashscore {match_url}")
                result["injury_or_lineup_notes"] = "\n".join(parts)
        except Exception:
            pass  # 阵容提取失败不阻断整体流程

        result["success"] = True

    except Exception as exc:
        result["error"] = str(exc)
    finally:
        browser.close()

    return result


# ─── Understat 补采（维度一：xG 实力代理）────────────────────────────────────

UNDERSTAT_LEAGUES = {
    "EPL": {"slug": "EPL", "label": "英超"},
    "La_liga": {"slug": "La_liga", "label": "西甲"},
    "Serie_A": {"slug": "Serie_A", "label": "意甲"},
    "Bundesliga": {"slug": "Bundesliga", "label": "德甲"},
    "Ligue_1": {"slug": "Ligue_1", "label": "法甲"},
}

# 球队名 -> understat league 映射（用于欧联杯等跨联赛比赛）
# key: 球队中文名或英文名片段, value: understat league slug
TEAM_LEAGUE_MAP = {
    # 法甲
    "lens": "Ligue_1",
    "朗斯": "Ligue_1",
    "marseille": "Ligue_1",
    "马赛": "Ligue_1",
    # 荷乙 (Dutch Eerste Divisie) - not in Understat
    "荷乙": "",
    "volendam": "",
    "福伦丹": "",
    "willem": "",
    "威廉二世": "",
    "willem ii": "",
    "cambuur": "",
    "坎布尔": "",
    "den haag": "",
    "海牙": "",
    "ado den haag": "",
    "graafschap": "",
    "格拉夫夏普": "",
    "de graafschap": "",
    "almere": "",
    "阿尔梅勒": "",
    "almere city": "",
    "valk": "",
    "瓦尔韦克": "",
    "roda": "",
    "罗达": "",
    "roda jc": "",
    "dordrecht": "",
    "多德勒支": "",
    "den bosch": "",
    "邓博施": "",
    "fc den bosch": "",
    "emmen": "",
    "埃门": "",
    "telstar": "",
    "特尔斯特": "",
    "vitesse": "",
    "维特斯": "",
    "heracles": "",
    "赫拉克勒斯": "",
    "excelsior": "",
    "excellence": "",
    # 芬超 (Finnish Veikkausliiga) - not in Understat
    "芬超": "",
    # 德甲
    "freiburg": "Bundesliga",
    "弗赖堡": "Bundesliga",
    "bayern": "Bundesliga",
    "拜仁": "Bundesliga",
    "dortmund": "Bundesliga",
    "多特": "Bundesliga",
    "leverkusen": "Bundesliga",
    "勒沃库森": "Bundesliga",
    "rb leipzig": "Bundesliga",
    "莱比锡": "Bundesliga",
    # 英超
    "aston villa": "EPL",
    "维拉": "EPL",
    "arsenal": "EPL",
    "阿森纳": "EPL",
    "chelsea": "EPL",
    "切尔西": "EPL",
    "liverpool": "EPL",
    "利物浦": "EPL",
    "manchester city": "EPL",
    "曼城": "EPL",
    "manchester united": "EPL",
    "曼联": "EPL",
    "tottenham": "EPL",
    "热刺": "EPL",
    "newcastle": "EPL",
    "纽卡": "EPL",
    # 西甲
    "real madrid": "La_liga",
    "皇马": "La_liga",
    "barcelona": "La_liga",
    "巴萨": "La_liga",
    "atletico": "La_liga",
    "马竞": "La_liga",
    # 意甲
    "inter": "Serie_A",
    "国米": "Serie_A",
    "milan": "Serie_A",
    "ac米兰": "Serie_A",
    "juventus": "Serie_A",
    "尤文": "Serie_A",
    "napoli": "Serie_A",
    "那不勒斯": "Serie_A",
    # 法甲
    "psg": "Ligue_1",
    "巴黎": "Ligue_1",
    "marseille": "Ligue_1",
    "马赛": "Ligue_1",
}


def _infer_league(team_name: str, league_hint: str = "") -> str:
    """根据球队名推断所属联赛。"""
    # 先尝试 ASCII 折叠匹配（适用于英文名）
    key = ascii_fold(team_name)
    for pattern, league in TEAM_LEAGUE_MAP.items():
        if pattern in key:
            return league
    # 再尝试原始名称匹配（适用于中文名）
    raw_lower = team_name.lower()
    for pattern, league in TEAM_LEAGUE_MAP.items():
        if pattern in raw_lower:
            return league
    # 尝试用赛事名推断
    if league_hint:
        hint_key = ascii_fold(league_hint)
        for pattern, league in TEAM_LEAGUE_MAP.items():
            if pattern in hint_key:
                return league
        hint_raw = league_hint.lower()
        for pattern, league in TEAM_LEAGUE_MAP.items():
            if pattern in hint_raw:
                return league
    return ""


def _fetch_understat_html(league: str, season: str) -> str:
    """通过 playwright-cli 渲染抓 understat 联赛页面。"""
    from playwright_cli_client import fetch_html_via_playwright_cli
    url = f"https://understat.com/league/{league}/{season}"
    return fetch_html_via_playwright_cli(url)


def _parse_understat_table(html: str) -> dict[str, dict]:
    """从 #league-chemp 表格提取 team-level xG 数据。"""
    soup = BeautifulSoup(html, "html.parser")
    container = soup.find("div", id="league-chemp")
    if container is None:
        return {}
    table = container.find("table")
    if table is None:
        return {}
    tbody = table.find("tbody")
    if tbody is None:
        return {}

    teams: dict[str, dict] = {}
    for tr in tbody.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) < 12:
            continue
        anchor = cells[1].find("a")
        if anchor is None:
            continue
        title = normalize_text(anchor.get_text(strip=True))
        if not title:
            continue

        matches = to_int(cells[2].get_text(strip=True))
        if matches <= 0:
            continue
        wins = to_int(cells[3].get_text(strip=True))
        draws = to_int(cells[4].get_text(strip=True))
        loses = to_int(cells[5].get_text(strip=True))
        gf = to_int(cells[6].get_text(strip=True))
        ga = to_int(cells[7].get_text(strip=True))
        pts = to_int(cells[8].get_text(strip=True))
        xg = to_float(cells[9].get_text(strip=True))
        xga = to_float(cells[10].get_text(strip=True))
        xpts = to_float(cells[11].get_text(strip=True))

        teams[title.lower()] = {
            "title": title,
            "matches": matches,
            "wins": wins,
            "draws": draws,
            "loses": loses,
            "goals_for": gf,
            "goals_against": ga,
            "points": pts,
            "xg_total": round(xg, 2),
            "xga_total": round(xga, 2),
            "xpts_total": round(xpts, 2),
            "xg_per_game": round(xg / matches, 2),
            "xga_per_game": round(xga / matches, 2),
            "win_rate": round(wins / matches, 3),
        }
    return teams


def _lookup_team(teams: dict, team_name: str) -> dict | None:
    """在 understat 表格中按球队名查找。"""
    key = ascii_fold(team_name)
    # 精确匹配
    if key in teams:
        return teams[key]
    # 模糊匹配
    for stored, metrics in teams.items():
        if stored.startswith(key) or key.startswith(stored):
            return metrics
        # 含队名片段
        parts = key.split()
        for p in parts:
            if len(p) > 3 and p in stored:
                return metrics
    return None


def _format_xg_summary(league_label: str, metrics: dict) -> str:
    """将 xG 数据格式化为实力代理文本。"""
    if not metrics:
        return ""
    title = league_label or metrics.get("title", "")
    segments = []
    if metrics.get("points"):
        segments.append(f"{metrics['points']}分")
    record = []
    if metrics.get("wins"):
        record.append(f"{metrics['wins']}胜")
    if metrics.get("draws"):
        record.append(f"{metrics['draws']}平")
    if metrics.get("loses"):
        record.append(f"{metrics['loses']}负")
    if record:
        segments.append("".join(record))
    if metrics.get("xg_per_game"):
        segments.append(f"xG {metrics['xg_per_game']}/场")
    if metrics.get("xga_per_game"):
        segments.append(f"xGA {metrics['xga_per_game']}/场")
    if not segments:
        return title
    return f"{title} {'，'.join(segments)}"


def supplement_xg_from_understat(
    home_team: str,
    away_team: str,
    league_hint: str = "",
) -> dict[str, Any]:
    """通过 Understat 补采维度一 xG 实力代理数据。

    Args:
        home_team: 主队名
        away_team: 客队名
        league_hint: 赛事名提示（如"欧联"、"德甲"）

    Returns:
        {
            "elo_home": str,
            "elo_away": str,
            "source_url": str,
            "success": bool,
            "error": str,
        }
    """
    result = {
        "elo_home": "",
        "elo_away": "",
        "source_url": "",
        "success": False,
        "error": "",
    }

    season = current_season()

    # 推断联赛
    home_league = _infer_league(home_team, league_hint)
    away_league = _infer_league(away_team, league_hint)

    if not home_league and not away_league:
        # 尝试用赛事名推断
        league_key = ascii_fold(league_hint)
        for pattern, lg in TEAM_LEAGUE_MAP.items():
            if pattern in league_key:
                home_league = away_league = lg
                break

    if not home_league:
        result["error"] = f"无法推断球队所属联赛：{home_team} / {away_team}"
        return result

    try:
        # 抓主队联赛
        home_html = _fetch_understat_html(home_league, season)
        home_teams = _parse_understat_table(home_html)
        home_metrics = _lookup_team(home_teams, home_team)

        # 抓客队联赛（可能同联赛）
        if away_league == home_league:
            away_teams = home_teams
        else:
            away_html = _fetch_understat_html(away_league, season)
            away_teams = _parse_understat_table(away_html)
        away_metrics = _lookup_team(away_teams, away_team)

        lg_info = UNDERSTAT_LEAGUES.get(home_league, {})
        lg_label = lg_info.get("label", home_league)

        result["elo_home"] = _format_xg_summary(lg_label, home_metrics) if home_metrics else ""
        result["elo_away"] = _format_xg_summary(
            UNDERSTAT_LEAGUES.get(away_league, {}).get("label", away_league),
            away_metrics,
        ) if away_metrics else ""
        result["source_url"] = f"https://understat.com/league/{home_league}/{season}"
        result["success"] = True

    except Exception as exc:
        result["error"] = f"Understat 补采失败：{exc}"

    return result



# ─── Soccerway 补采（维度二降级源）──────────────────────────────────────────

SOCCERWAY_DOMAINS = (
    "soccerway.com",
    "int.soccerway.com",
    "uk.soccerway.com",
    "us.soccerway.com",
)


def _normalize_sw_url(url: str) -> str:
    from urllib.parse import urlparse, urlunparse
    parsed = urlparse(url)
    host = parsed.netloc.lower().lstrip("www.")
    if not any(host == d or host.endswith("." + d) for d in SOCCERWAY_DOMAINS):
        return ""
    path = parsed.path.rstrip("/")
    return urlunparse(("https", parsed.netloc, path + "/", "", "", ""))


def _score_sw_candidate(href: str, text: str, home: str, away: str) -> int:
    from urllib.parse import urlparse
    haystack = ascii_fold(href + " " + text)
    score = 0
    home_key = ascii_fold(home)
    away_key = ascii_fold(away)
    for part in home_key.split():
        if len(part) > 2 and part in haystack:
            score += 4
    for part in away_key.split():
        if len(part) > 2 and part in haystack:
            score += 4
    if "/match/" in href or "/game/" in href:
        score += 3
    if "soccerway" in haystack:
        score += 1
    if re.search(r"/team/|/standings|/fixtures|/squads", href, re.I):
        score -= 5
    return score


def _search_sw_match(browser, home: str, away: str, match_date: str = "") -> str:
    """搜索 Soccerway 找到对赛页 URL。"""
    from urllib.parse import quote_plus
    stopwords = {"fc", "cf", "sc", "ac", "afc", "bk", "fk", "sv", "ud", "cd", "club", "team"}
    home_parts = [p for p in ascii_fold(home).split() if p not in stopwords and len(p) > 2]
    away_parts = [p for p in ascii_fold(away).split() if p not in stopwords and len(p) > 2]

    queries = []
    for hp in home_parts[:2]:
        for ap in away_parts[:2]:
            if match_date:
                queries.append(hp + " " + ap + " " + match_date + " soccerway")
            queries.append(hp + " " + ap + " soccerway")
    queries = list(dict.fromkeys(queries))[:6]

    for query in queries:
        for engine, tpl in [
            ("bing", "https://www.bing.com/search?q=" + quote_plus(query)),
            ("yahoo", "https://search.yahoo.com/search?p=" + quote_plus(query)),
        ]:
            try:
                browser.goto(tpl)
                title = browser.title()
                body = browser.body_text()[:3000]
                if "one last step" in body.lower() or "solve this puzzle" in body.lower():
                    continue
                raw = browser.eval(
                    "JSON.stringify("
                    "[...document.querySelectorAll('a[href]')]"
                    ".slice(0, 180)"
                    ".map(a => ({ text: (a.textContent||'').trim(), href: a.href }))"
                    ")"
                ) or []
                if isinstance(raw, str):
                    try:
                        raw = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                scored = []
                for item in raw:
                    href = (item.get("href") or "").strip()
                    text = (item.get("text") or "").strip()
                    if not href:
                        continue
                    norm = _normalize_sw_url(href)
                    if not norm:
                        continue
                    scored.append((_score_sw_candidate(href, text, home, away), norm))
                scored.sort(key=lambda x: x[0], reverse=True)
                if scored:
                    return scored[0][1]
            except Exception:
                continue
    return ""


def _extract_sw_injury_notes(browser, match_url: str) -> str:
    """从 Soccerway 对赛页提取伤停/阵容信息。"""
    try:
        browser.goto(match_url)
        body = browser.body_text()[:12000]

        # 尝试找伤停段落
        injury_patterns = [
            r"(?:伤停|Injuries|Suspensions)[^\n]*\n((?:[^\n]+\n){2,10})",
            r"(?:缺席|Absent|Missing)[^\n]*\n((?:[^\n]+\n){2,10})",
        ]
        for pat in injury_patterns:
            m = re.search(pat, body, re.I)
            if m:
                lines = [l.strip() for l in m.group(1).splitlines() if l.strip()]
                if lines:
                    return "Soccerway 伤停参考：" + "；".join(lines[:8])

        # 尝试找阵容段落
        lineup_patterns = [
            r"(?:预计首发|Starting Lineups|Line-ups)[^\n]*\n((?:[^\n]+\n){2,15})",
        ]
        for pat in lineup_patterns:
            m = re.search(pat, body, re.I)
            if m:
                lines = [l.strip() for l in m.group(1).splitlines() if l.strip()]
                if lines:
                    return "Soccerway 阵容参考：" + "；".join(lines[:10])

        return ""
    except Exception:
        return ""


def supplement_from_soccerway(
    home_team: str,
    away_team: str,
    match_time: str = "",
) -> dict:
    """通过 Soccerway 补采维度二数据（Flashscore 失败时的降级源）。"""
    result = {
        "head_to_head_summary": "",
        "injury_or_lineup_notes": "",
        "source_url": "",
        "success": False,
        "error": "",
    }

    try:
        browser = PlaywrightCliBrowser(PlaywrightCliSettings.from_env())
    except Exception as exc:
        result["error"] = f"playwright-cli 不可用：{exc}"
        return result

    try:
        match_url = _search_sw_match(browser, home_team, away_team, match_time)
        if not match_url:
            result["error"] = f"Soccerway 未找到对赛页：{home_team} vs {away_team}"
            return result

        result["source_url"] = match_url

        # 提取伤停/阵容
        injury_notes = _extract_sw_injury_notes(browser, match_url)
        if injury_notes:
            result["injury_or_lineup_notes"] = injury_notes

        result["success"] = True

    except Exception as exc:
        result["error"] = str(exc)
    finally:
        browser.close()

    return result


# ─── Web 搜索降级补采 ──────────────────────────────────────────────────────


def _web_search_team_strength(browser, team_name: str, league: str = "") -> str:
    """通过搜索引擎查找球队实力/排名信息（用于低级别联赛降级补采）。"""
    from urllib.parse import quote_plus

    queries = []
    if league:
        queries.append(quote_plus(f"{team_name} {league} 积分榜 排名 赛季"))
        queries.append(quote_plus(f"{team_name} {league} standings"))
    queries.append(quote_plus(f"{team_name} 联赛 排名 积分"))
    queries.append(quote_plus(f"{team_name} league standings table"))

    for query in queries[:3]:
        for engine_url in [
            "https://www.bing.com/search?q=" + query,
        ]:
            try:
                browser.goto(engine_url)
                body = browser.body_text()[:6000]
                if "one last step" in body.lower():
                    continue

                # 尝试提取包含排名/积分的文本片段
                patterns = [
                    r"(\d+[/\s]\d+\s*(?:分|points|pts))",
                    r"(?:第\s*\d+\s*名|排名\s*[:：]\s*\d+|rank\s*[:：]\s*\d+)",
                    r"(\d+\s*胜\s*\d+\s*平\s*\d+\s*负)",
                ]
                found = []
                for pat in patterns:
                    import re as _re
                    m = _re.search(pat, body, _re.I)
                    if m:
                        found.append(m.group(0).strip())

                if found:
                    source = f"Web搜索({engine_url[:40]}...)"
                    return f"Web参考：{'；'.join(found[:3])}（来源：{source}）"
            except Exception:
                continue
    return ""


def _direct_fetch_500_jifen(browser, team_name: str, league: str = "") -> str:
    """直接访问 500.com 联赛积分页获取球队排名数据（降级补采）。"""
    from urllib.parse import quote_plus
    import re as _re

    # 构造搜索查询定位 500.com 积分页
    queries = []
    if league:
        queries.append(quote_plus(f"site:liansai.500.com {league} 积分榜"))
        queries.append(quote_plus(f"site:500.com {team_name} {league} 积分"))
    queries.append(quote_plus(f"site:liansai.500.com {team_name} 积分榜"))

    for query in queries[:3]:
        try:
            browser.goto(f"https://www.bing.com/search?q={query}")
            body = browser.body_text()[:6000]
            if "one last step" in body.lower():
                continue

            # 找 500.com 积分页链接
            links_js = (
                "JSON.stringify("
                "[...document.querySelectorAll('a[href]')]"
                ".slice(0, 100)"
                ".filter(a => a.href.includes('liansai.500.com') && a.href.includes('jifen'))"
                ".map(a => ({ text: (a.textContent||'').trim(), href: a.href }))"
                ")"
            )
            raw = browser.eval(links_js)
            if isinstance(raw, str):
                import json as _json
                try:
                    raw = _json.loads(raw)
                except _json.JSONDecodeError:
                    continue
            if not raw:
                continue

            # 访问第一个积分页链接
            jifen_url = raw[0]["href"]
            if not jifen_url.startswith("http"):
                continue

            # 用 requests 直接抓取积分页（不需要 JS 渲染）
            try:
                resp = browser.goto(jifen_url)
                page_body = browser.body_text()[:8000]

                # 在积分表中查找球队名
                if team_name in page_body or _re.search(team_name, page_body):
                    # 提取包含球队名的表格行
                    # 尝试多种模式
                    row_patterns = [
                        rf"{_re.escape(team_name)}[^\n]{{0,200}}",
                        rf"{_re.escape(team_name)}.*?第\s*\d+\s*[/\s]\d+",
                    ]
                    for rp in row_patterns:
                        m = _re.search(rp, page_body, _re.S)
                        if m:
                            text = m.group(0).strip()
                            # 清理 HTML 标签
                            text = _re.sub(r"<[^>]+>", "", text)
                            text = _re.sub(r"\s+", " ", text).strip()
                            if text and len(text) > 10:
                                return f"500.com积分页参考：{text[:200]}（来源：{jifen_url}）"
            except Exception:
                continue
        except Exception:
            continue
    return ""


def _web_search_injury(browser, home: str, away: str) -> str:
    """通过搜索引擎查找伤停信息（用于低级别联赛降级补采）。"""
    from urllib.parse import quote_plus

    query = quote_plus(f"{home} vs {away} 伤停 阵容 首发")
    try:
        browser.goto("https://www.bing.com/search?q=" + query)
        body = browser.body_text()[:5000]
        if "one last step" in body.lower():
            return ""

        # 找伤停相关段落
        import re as _re
        patterns = [
            r"(?:伤停|injur|suspension|missing)[^\n]{10,100}",
            r"(?:缺席|缺阵|无法出场)[^\n]{10,100}",
        ]
        found = []
        for pat in patterns:
            for m in _re.finditer(pat, body, _re.I):
                text = m.group(0).strip()
                if len(text) > 15 and len(text) < 120:
                    found.append(text)
                if len(found) >= 3:
                    break

        if found:
            return "Web伤停参考：" + "；".join(found[:3])
    except Exception:
        pass
    return ""


def supplement_from_web_search(home_team: str, away_team: str, league: str = "") -> dict:
    """通过搜索引擎降级补采（所有专业源失败后的最后手段）。"""
    result = {
        "elo_home": "",
        "elo_away": "",
        "injury_or_lineup_notes": "",
        "head_to_head_summary": "",
        "source_url": "",
        "success": False,
        "error": "",
    }

    try:
        browser = PlaywrightCliBrowser(PlaywrightCliSettings.from_env())
    except Exception as exc:
        result["error"] = f"playwright-cli 不可用：{exc}"
        return result

    try:
        # 搜索球队实力（先常规搜索，再尝试直接访问 500.com 积分页）
        home_strength = _web_search_team_strength(browser, home_team, league)
        if not home_strength:
            home_strength = _direct_fetch_500_jifen(browser, home_team, league)
        away_strength = _web_search_team_strength(browser, away_team, league)
        if not away_strength:
            away_strength = _direct_fetch_500_jifen(browser, away_team, league)
        if home_strength:
            result["elo_home"] = home_strength
        if away_strength:
            result["elo_away"] = away_strength

        # 搜索伤停
        injury = _web_search_injury(browser, home_team, away_team)
        if injury:
            result["injury_or_lineup_notes"] = injury

        if result["elo_home"] or result["elo_away"] or result["injury_or_lineup_notes"]:
            result["success"] = True
            result["source_url"] = "web-search-fallback"
        else:
            result["error"] = "Web搜索未找到有效数据"

    except Exception as exc:
        result["error"] = str(exc)
    finally:
        browser.close()

    return result

# ─── 综合补采入口 ────────────────────────────────────────────────────────────


def supplement_match_data(
    home_team: str,
    away_team: str,
    league: str = "",
    match_time: str = "",
    missing_dimensions: list[str] | None = None,
) -> dict[str, Any]:
    """综合补采入口：根据缺失维度自动选择补充源。

    Args:
        home_team: 主队名
        away_team: 客队名
        league: 赛事名
        match_time: 比赛时间
        missing_dimensions: 缺失维度列表，如 ["维度一：基础实力", "维度二：近期动态"]
            None 表示全部维度都补采

    Returns:
        {
            "elo_home": str,
            "elo_away": str,
            "head_to_head_summary": str,
            "recent_form_home": str,
            "recent_form_away": str,
            "injury_or_lineup_notes": str,
            "sources": list[str],
            "errors": list[str],
        }
    """
    result = {
        "elo_home": "",
        "elo_away": "",
        "market_value_summary": "",
        "head_to_head_summary": "",
        "recent_form_home": "",
        "recent_form_away": "",
        "injury_or_lineup_notes": "",
        "sources": [],
        "errors": [],
    }

    need_dim1 = missing_dimensions is None or any("维度一" in d for d in missing_dimensions)
    need_dim2 = missing_dimensions is None or any("维度二" in d for d in missing_dimensions)

    # 维度一：xG 实力代理
    if need_dim1:
        dim1_filled = False
        try:
            market_value_result = fetch_match_market_values(home_team, away_team, league)
            if market_value_result.get("market_value_summary"):
                result["market_value_summary"] = market_value_result["market_value_summary"]
                for src in market_value_result.get("sources", []):
                    result["sources"].append(src)
            for err in market_value_result.get("errors", []):
                if err:
                    result["errors"].append(f"MarketValue: {err}")
        except Exception as exc:
            result["errors"].append(f"MarketValue 异常: {exc}")

        try:
            xg_result = supplement_xg_from_understat(home_team, away_team, league)
            if xg_result["success"]:
                result["elo_home"] = xg_result["elo_home"]
                result["elo_away"] = xg_result["elo_away"]
                if xg_result["source_url"]:
                    result["sources"].append(f"Understat: {xg_result['source_url']}")
                if result["elo_home"] or result["elo_away"]:
                    dim1_filled = True
            else:
                result["errors"].append(f"Understat: {xg_result.get('error', '未知错误')}")
        except Exception as exc:
            result["errors"].append(f"Understat 异常: {exc}")

        # Understat 失败时，降级到搜索引擎
        if not dim1_filled:
            try:
                web_result = supplement_from_web_search(home_team, away_team, league)
                if web_result["success"]:
                    if web_result["elo_home"] and not result["elo_home"]:
                        result["elo_home"] = web_result["elo_home"]
                    if web_result["elo_away"] and not result["elo_away"]:
                        result["elo_away"] = web_result["elo_away"]
                    result["sources"].append("WebSearch: fallback")
                else:
                    result["errors"].append(f"WebSearch: {web_result.get('error', '未知错误')}")
            except Exception as exc:
                result["errors"].append(f"WebSearch 异常: {exc}")

    # 维度二：交锋记录 + 伤停阵容
    if need_dim2:
        dim2_filled = False
        try:
            fs_result = supplement_from_flashscore(home_team, away_team, match_time)
            if fs_result["success"]:
                if fs_result["head_to_head_summary"]:
                    result["head_to_head_summary"] = fs_result["head_to_head_summary"]
                if fs_result["recent_form_home"]:
                    result["recent_form_home"] = fs_result["recent_form_home"]
                if fs_result["recent_form_away"]:
                    result["recent_form_away"] = fs_result["recent_form_away"]
                if fs_result["injury_or_lineup_notes"]:
                    result["injury_or_lineup_notes"] = fs_result["injury_or_lineup_notes"]
                if fs_result["source_url"]:
                    result["sources"].append(f"Flashscore: {fs_result['source_url']}")
                if result["head_to_head_summary"] or result["injury_or_lineup_notes"]:
                    dim2_filled = True
            else:
                result["errors"].append(f"Flashscore: {fs_result.get('error', '未知错误')}")
        except Exception as exc:
            result["errors"].append(f"Flashscore 异常: {exc}")

        # Flashscore 未命中关键数据时，降级到 Soccerway
        if not dim2_filled:
            try:
                sw_result = supplement_from_soccerway(home_team, away_team, match_time)
                if sw_result["success"]:
                    if sw_result["head_to_head_summary"] and not result["head_to_head_summary"]:
                        result["head_to_head_summary"] = sw_result["head_to_head_summary"]
                    if sw_result["injury_or_lineup_notes"] and not result["injury_or_lineup_notes"]:
                        result["injury_or_lineup_notes"] = sw_result["injury_or_lineup_notes"]
                    if sw_result["source_url"]:
                        result["sources"].append(f"Soccerway: {sw_result['source_url']}")
                    if result["head_to_head_summary"] or result["injury_or_lineup_notes"]:
                        dim2_filled = True
                else:
                    result["errors"].append(f"Soccerway: {sw_result.get('error', '未知错误')}")
            except Exception as exc:
                result["errors"].append(f"Soccerway 异常: {exc}")

        # 所有专业源都失败时，降级到搜索引擎
        if not dim2_filled:
            try:
                web_result = supplement_from_web_search(home_team, away_team, league)
                if web_result["success"]:
                    if web_result["injury_or_lineup_notes"] and not result["injury_or_lineup_notes"]:
                        result["injury_or_lineup_notes"] = web_result["injury_or_lineup_notes"]
                    if web_result["head_to_head_summary"] and not result["head_to_head_summary"]:
                        result["head_to_head_summary"] = web_result["head_to_head_summary"]
                    result["sources"].append("WebSearch: fallback")
                else:
                    result["errors"].append(f"WebSearch: {web_result.get('error', '未知错误')}")
            except Exception as exc:
                result["errors"].append(f"WebSearch 异常: {exc}")

    return result


__all__ = [
    "supplement_match_data",
    "supplement_xg_from_understat",
    "supplement_from_flashscore",
    "supplement_from_soccerway",
    "supplement_from_web_search",
    "UNDERSTAT_LEAGUES",
    "TEAM_LEAGUE_MAP",
]
