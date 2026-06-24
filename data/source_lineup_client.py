from __future__ import annotations

import json
import re
import unicodedata
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Mapping
from urllib.parse import quote_plus, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

from playwright_cli_client import PlaywrightCliBrowser, PlaywrightCliSettings


SEARCH_LINKS_JS = (
    "JSON.stringify("
    "[...document.querySelectorAll('a[href]')]"
    ".slice(0, 180)"
    ".map(a => ({ text: (a.textContent || '').trim(), href: a.href }))"
    ")"
)

FIND_LINEUP_TAB_JS = (
    "(() => { "
    "const anchors = Array.from(document.querySelectorAll('a[href]')); "
    "const hit = anchors.find(anchor => /kader|lineup|aufstellung|startaufstellung|aufstellungen/i.test(((anchor.textContent || '').trim()) + ' ' + (anchor.href || ''))); "
    "return hit ? hit.href : ''; "
    "})()"
)

FLASH_SCORE_DOMAINS = (
    "flashscore.de",
    "flashscore.com",
    "flashscore.co.uk",
    "flashscore.in",
    "flashscore.ca",
    "flashscoreusa.com",
    "flashscore.info",
    "flashscore.com.ng",
    "flashscore.co.za",
    "flashscore.at",
    "flashscore.es",
    "flashscore.it",
    "flashscore.fr",
    "flashscore.nl",
    "flashscore.pl",
    "flashscore.ro",
    "flashscore.tr",
    "flashscore.cz",
    "flashscore.sk",
    "t.flashscore.com",
)

ROTOWIRE_DOMAINS = (
    "rotowire.com",
    "www.rotowire.com",
)

BING_CHALLENGE_MARKERS = (
    "one last step",
    "solve this puzzle",
    "最后一步",
    "请完成验证",
)

TEAM_NAME_STOPWORDS = {
    "fc",
    "cf",
    "sc",
    "ac",
    "afc",
    "bk",
    "fk",
    "sv",
    "ud",
    "cd",
    "club",
    "team",
}

REASON_TRANSLATIONS = {
    "ankle injury": "踝关节伤",
    "back injury": "背部伤",
    "calf injury": "小腿伤",
    "concussion": "脑震荡",
    "foot injury": "脚部伤",
    "groin injury": "腹股沟伤",
    "hamstring injury": "腿筋伤",
    "heel injury": "脚跟伤",
    "hip injury": "髋部伤",
    "illness": "身体不适",
    "inactive": "未进入名单",
    "injury": "伤病",
    "knee injury": "膝部伤",
    "knock": "轻伤",
    "lacking match fitness": "状态未满",
    "muscle injury": "肌肉伤",
    "suspended": "停赛",
    "thigh injury": "大腿伤",
    "wrist injury": "手腕伤",
    "fersenverletzung": "脚跟伤",
    "hüftverletzung": "髋部伤",
    "inaktiv": "未进入名单",
    "knieverletzung": "膝部伤",
    "muskelverletzung": "肌肉伤",
    "oberschenkelverletzung": "大腿伤",
    "sprunggelenksverletzung": "踝关节伤",
    "wadenverletzung": "小腿伤",
}

HTTP = requests.Session()
HTTP.trust_env = False
HTTP.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/147.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9,de-DE;q=0.8",
    }
)


@dataclass
class LineupSupplementResult:
    notes: str
    source_links: list[str] = field(default_factory=list)
    source_labels: list[str] = field(default_factory=list)
    remarks: list[str] = field(default_factory=list)
    structured_data: dict[str, Any] = field(default_factory=dict)
    lineup_found: bool = False
    injury_found: bool = False


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def unique_non_empty(values: list[str]) -> list[str]:
    seen: set[str] = set()
    results: list[str] = []
    for value in values:
        text = normalize_text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        results.append(text)
    return results


def mapping_get(item: Mapping[str, Any], key: str, default: Any = "") -> Any:
    try:
        return item[key]
    except Exception:  # noqa: BLE001
        return default


def ascii_fold(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    return normalized.encode("ascii", "ignore").decode("ascii").lower()


def extract_team_ids(shuju_html: str) -> list[str]:
    team_ids: list[str] = []
    for team_id in re.findall(r"liansai\.500\.com/team/(\d+)/?", shuju_html or ""):
        if team_id not in team_ids:
            team_ids.append(team_id)
    return team_ids[:2]


@lru_cache(maxsize=256)
def fetch_text(url: str) -> str:
    response = HTTP.get(url, timeout=30)
    response.raise_for_status()
    return response.text


@lru_cache(maxsize=256)
def resolve_team_english_name(team_id: str) -> str:
    html = fetch_text(f"https://liansai.500.com/team/{team_id}/")
    soup = BeautifulSoup(html, "html.parser")
    node = soup.select_one("div.itm_name_en")
    if node is not None:
        return normalize_text(node.get_text(" ", strip=True))
    return ""


def infer_match_date(match_time: str) -> str:
    text = normalize_text(match_time)
    now = datetime.now()
    for pattern in ("%m-%d %H:%M", "%Y-%m-%d", "%y-%m-%d"):
        try:
            parsed = datetime.strptime(text, pattern)
        except ValueError:
            continue
        if pattern == "%m-%d %H:%M":
            candidate = parsed.replace(year=now.year)
            if candidate - now > timedelta(days=200):
                candidate = candidate.replace(year=now.year - 1)
            elif now - candidate > timedelta(days=200):
                candidate = candidate.replace(year=now.year + 1)
            return candidate.strftime("%Y-%m-%d")
        return parsed.strftime("%Y-%m-%d")
    return ""


def build_name_variants(name: str) -> list[str]:
    raw = normalize_text(name)
    ascii_name = normalize_text(ascii_fold(raw))
    variants = [raw, ascii_name]
    parts = [part for part in ascii_name.split() if part not in TEAM_NAME_STOPWORDS]
    if parts:
        variants.append(" ".join(parts))
        variants.append(parts[-1])
    return unique_non_empty(variants)


def build_search_queries(home_name: str, away_name: str, match_date: str) -> list[str]:
    queries: list[str] = []
    home_variants = build_name_variants(home_name) or [home_name]
    away_variants = build_name_variants(away_name) or [away_name]
    for home_variant in home_variants[:3]:
        for away_variant in away_variants[:3]:
            if match_date:
                queries.append(f"{home_variant} {away_variant} {match_date} flashscore")
            queries.append(f"{home_variant} {away_variant} flashscore")
    return unique_non_empty(queries)[:8]


def build_rotowire_search_queries(home_name: str, away_name: str, match_date: str) -> list[str]:
    queries: list[str] = []
    home_variants = build_name_variants(home_name) or [home_name]
    away_variants = build_name_variants(away_name) or [away_name]
    for home_variant in home_variants[:3]:
        for away_variant in away_variants[:3]:
            if match_date:
                queries.append(f"{home_variant} {away_variant} {match_date} rotowire soccer lineups injuries")
            queries.append(f"{home_variant} {away_variant} rotowire soccer expected lineups injuries")
    queries.append(f"{home_name} {away_name} site:rotowire.com/soccer lineups injuries")
    return unique_non_empty(queries)[:8]


def is_bing_challenge(title: str, body_text: str) -> bool:
    haystack = f"{title} {body_text}".lower()
    return any(marker in haystack for marker in BING_CHALLENGE_MARKERS)


def domain_matches(host: str, allowed_domains: tuple[str, ...]) -> bool:
    normalized = host.lower().lstrip("www.")
    return any(
        normalized == domain or normalized.endswith(f".{domain}")
        for domain in allowed_domains
    )


def normalize_flashscore_match_url(url: str) -> str:
    parsed = urlparse(url)
    if not domain_matches(parsed.netloc, FLASH_SCORE_DOMAINS):
        return ""
    path = parsed.path.rstrip("/")
    path = re.sub(r"/(?:bericht|summary)(?:/.*)?$", "", path, flags=re.I)
    if not re.search(r"/(spiel|match|game)/", path, re.I):
        return ""
    normalized_path = f"{path}/"
    return urlunparse(("https", parsed.netloc, normalized_path, "", "", ""))


def normalize_rotowire_url(url: str) -> str:
    parsed = urlparse(url)
    if not domain_matches(parsed.netloc, ROTOWIRE_DOMAINS):
        return ""
    path = parsed.path.rstrip("/")
    folded_path = path.lower()
    if "/soccer" not in folded_path:
        return ""
    if any(skip in folded_path for skip in ("/news", "/player", "/daily", "/rankings")):
        return ""
    return urlunparse(("https", parsed.netloc or "www.rotowire.com", path or "/soccer", "", "", ""))


def _section_side_texts_js(pattern: str) -> str:
    escaped = pattern.replace("\\", "\\\\").replace("'", "\\'")
    return (
        "(() => { "
        "const sections = Array.from(document.querySelectorAll('.section, section')); "
        f"const section = sections.find(item => /{escaped}/i.test((item.innerText || ''))); "
        "if (!section) return '[]'; "
        "return JSON.stringify(Array.from(section.querySelectorAll('.lf__side')).slice(0, 2).map(side => side.innerText || '')); "
        "})()"
    )


def score_flashscore_candidate(
    href: str,
    text: str,
    home_name: str,
    away_name: str,
) -> int:
    haystack = ascii_fold(f"{href} {text}")
    score = 0
    if any(keyword in haystack for keyword in build_name_variants(home_name)):
        score += 4
    if any(keyword in haystack for keyword in build_name_variants(away_name)):
        score += 4
    if re.search(r"/(spiel|match|game)/", href, re.I):
        score += 3
    if re.search(r"/bericht/kader|/summary/lineups", href, re.I):
        score += 2
    if "flashscore" in haystack:
        score += 1
    if re.search(r"/team/|/standings|/fixtures", href, re.I):
        score -= 5
    return score


def search_links(
    browser: PlaywrightCliBrowser,
    query: str,
    allowed_domains: tuple[str, ...],
) -> tuple[list[dict[str, str]], str]:
    engines = [
        ("bing", f"https://www.bing.com/search?q={quote_plus(query)}"),
        ("yahoo", f"https://search.yahoo.com/search?p={quote_plus(query)}"),
    ]
    last_engine = ""
    for engine, url in engines:
        last_engine = engine
        browser.goto(url)
        title = browser.title()
        body = browser.body_text()[:4000]
        if engine == "bing" and is_bing_challenge(title, body):
            continue
        items = browser.eval(SEARCH_LINKS_JS) or []
        if isinstance(items, str):
            with suppress(json.JSONDecodeError):
                items = json.loads(items)
        filtered: list[dict[str, str]] = []
        for item in items:
            href = normalize_text(str(item.get("href", "")))
            text = normalize_text(str(item.get("text", "")))
            if not href:
                continue
            if not domain_matches(urlparse(href).netloc, allowed_domains):
                continue
            filtered.append({"href": href, "text": text})
        if filtered:
            return filtered, engine
    return [], last_engine


def discover_flashscore_match_url(
    browser: PlaywrightCliBrowser,
    home_name: str,
    away_name: str,
    match_date: str,
) -> tuple[str, str]:
    for query in build_search_queries(home_name, away_name, match_date):
        links, engine = search_links(browser, query, FLASH_SCORE_DOMAINS)
        scored: list[tuple[int, str]] = []
        for item in links:
            normalized = normalize_flashscore_match_url(item["href"])
            if not normalized:
                continue
            scored.append(
                (
                    score_flashscore_candidate(
                        item["href"],
                        item["text"],
                        home_name,
                        away_name,
                    ),
                    normalized,
                )
            )
        scored.sort(key=lambda item: item[0], reverse=True)
        if scored:
            return scored[0][1], engine
    return "", ""


def score_rotowire_candidate(
    href: str,
    text: str,
    home_name: str,
    away_name: str,
) -> int:
    haystack = ascii_fold(f"{href} {text}")
    score = 0
    if any(keyword in haystack for keyword in build_name_variants(home_name)):
        score += 4
    if any(keyword in haystack for keyword in build_name_variants(away_name)):
        score += 4
    if "rotowire" in haystack:
        score += 2
    if any(keyword in haystack for keyword in ("lineup", "injur", "suspension", "team-news")):
        score += 3
    if any(keyword in haystack for keyword in ("player", "news", "rankings", "daily")):
        score -= 4
    return score


def discover_rotowire_url(
    browser: PlaywrightCliBrowser,
    home_name: str,
    away_name: str,
    match_date: str,
) -> tuple[str, str]:
    for query in build_rotowire_search_queries(home_name, away_name, match_date):
        links, engine = search_links(browser, query, ROTOWIRE_DOMAINS)
        scored: list[tuple[int, str]] = []
        for item in links:
            normalized = normalize_rotowire_url(item["href"])
            if not normalized:
                continue
            scored.append(
                (
                    score_rotowire_candidate(item["href"], item["text"], home_name, away_name),
                    normalized,
                )
            )
        scored.sort(key=lambda item: item[0], reverse=True)
        if scored:
            return scored[0][1], engine
    return "", ""


def translate_reason(reason: str) -> str:
    clean = normalize_text(reason)
    return REASON_TRANSLATIONS.get(clean.lower(), clean)


def normalize_player_name(name: str) -> str:
    return normalize_text(name).replace(" (G)", "")


def parse_lineup_side_text(text: str) -> list[str]:
    players: list[str] = []
    for raw_line in (text or "").splitlines():
        line = normalize_player_name(raw_line)
        folded = ascii_fold(line)
        if not line:
            continue
        if line.isdigit():
            continue
        if re.fullmatch(r"\([A-Z]{1,3}\)", line):
            continue
        if any(keyword in folded for keyword in ("formation", "predicted", "starting", "lineups")):
            continue
        players.append(line)
    return unique_non_empty(players)[:11]


def _compact_preview_lines(text: str, limit: int = 18) -> list[str]:
    lines: list[str] = []
    for raw_line in (text or "").splitlines():
        line = normalize_text(raw_line)
        if not line:
            continue
        folded = ascii_fold(line)
        if folded in {"advertisement", "more soccer", "soccer", "lineups", "injuries"}:
            continue
        if len(line) > 180:
            line = line[:177].rstrip() + "..."
        lines.append(line)
        if len(lines) >= limit:
            break
    return unique_non_empty(lines)


def _rotowire_section_lines(body_text: str) -> list[str]:
    keywords = (
        "expected lineups",
        "probable lineups",
        "confirmed lineups",
        "projected lineups",
        "injuries",
        "suspensions",
        "team news",
        "out",
        "doubtful",
        "questionable",
    )
    lines = [normalize_text(line) for line in (body_text or "").splitlines()]
    lines = [line for line in lines if line]
    selected: list[str] = []
    for index, line in enumerate(lines):
        folded = ascii_fold(line)
        if not any(keyword in folded for keyword in keywords):
            continue
        start = max(0, index - 2)
        end = min(len(lines), index + 10)
        selected.extend(lines[start:end])
    return _compact_preview_lines("\n".join(selected), limit=22)


def parse_missing_side_text(text: str, status: str) -> list[dict[str, str]]:
    lines = [normalize_text(line) for line in (text or "").splitlines() if normalize_text(line)]
    results: list[dict[str, str]] = []
    index = 0
    while index < len(lines):
        player = normalize_player_name(lines[index])
        reason = ""
        if index + 1 < len(lines):
            candidate_reason = normalize_text(lines[index + 1])
            if not candidate_reason.isdigit() and not re.search(r"\.$", candidate_reason):
                reason = candidate_reason
                index += 1
        if player and not player.isdigit():
            results.append(
                {
                    "player": player,
                    "reason": translate_reason(reason),
                    "status": status,
                }
            )
        index += 1
    return results


def normalize_missing_items(items: list[dict[str, str]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items:
        player = normalize_player_name(str(item.get("player", "")))
        if not player:
            continue
        reason_raw = normalize_text(str(item.get("reason", "")))
        status = normalize_text(str(item.get("status", "") or "out")).lower()
        translated = translate_reason(reason_raw)
        key = (player, translated, status)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(
            {
                "player": player,
                "reason": translated,
                "status": "doubtful" if status == "doubtful" else "out",
            }
        )
    return normalized


def build_structured_lineup_data(raw_payload: Mapping[str, Any]) -> dict[str, Any]:
    home_lineup = unique_non_empty(
        [normalize_player_name(item) for item in raw_payload.get("lineup_home", []) or []]
    )[:11]
    away_lineup = unique_non_empty(
        [normalize_player_name(item) for item in raw_payload.get("lineup_away", []) or []]
    )[:11]
    home_missing = normalize_missing_items(
        list(raw_payload.get("missing_home") or []) + list(raw_payload.get("doubtful_home") or [])
    )
    away_missing = normalize_missing_items(
        list(raw_payload.get("missing_away") or []) + list(raw_payload.get("doubtful_away") or [])
    )
    home_absent_count = sum(1 for item in home_missing if item["status"] != "doubtful")
    away_absent_count = sum(1 for item in away_missing if item["status"] != "doubtful")
    home_doubtful_count = sum(1 for item in home_missing if item["status"] == "doubtful")
    away_doubtful_count = sum(1 for item in away_missing if item["status"] == "doubtful")
    home_absence_impact = min(
        0.36,
        home_absent_count * 0.06 + home_doubtful_count * 0.03 + max(0, 11 - len(home_lineup)) * 0.015,
    )
    away_absence_impact = min(
        0.36,
        away_absent_count * 0.06 + away_doubtful_count * 0.03 + max(0, 11 - len(away_lineup)) * 0.015,
    )
    return {
        "provider": normalize_text(str(raw_payload.get("provider", ""))) or "Flashscore",
        "page_title": normalize_text(str(raw_payload.get("page_title", ""))),
        "page_url": normalize_text(str(raw_payload.get("page_url", ""))),
        "home_lineup": home_lineup,
        "away_lineup": away_lineup,
        "home_missing": home_missing,
        "away_missing": away_missing,
        "home_starters": len(home_lineup),
        "away_starters": len(away_lineup),
        "home_absent_count": home_absent_count,
        "away_absent_count": away_absent_count,
        "home_doubtful_count": home_doubtful_count,
        "away_doubtful_count": away_doubtful_count,
        "home_absence_impact": round(home_absence_impact, 4),
        "away_absence_impact": round(away_absence_impact, 4),
    }


def build_rotowire_structured_data(body_text: str, page_url: str, page_title: str) -> dict[str, Any]:
    section_lines = _rotowire_section_lines(body_text)
    joined = "\n".join(section_lines)
    missing_items: list[dict[str, str]] = []
    for line in section_lines:
        folded = ascii_fold(line)
        if not any(keyword in folded for keyword in ("injur", "out", "doubtful", "questionable", "suspended")):
            continue
        status = "doubtful" if "doubt" in folded or "questionable" in folded else "out"
        missing_items.extend(parse_missing_side_text(line, status))

    structured = build_structured_lineup_data(
        {
            "provider": "RotoWire",
            "lineup_home": parse_lineup_side_text(joined),
            "lineup_away": [],
            "missing_home": missing_items[:8],
            "missing_away": [],
            "page_url": page_url,
            "page_title": page_title,
        }
    )
    structured["raw_summary"] = joined
    structured["lineup_url"] = page_url
    return structured


def format_lineup_side(players: list[str]) -> str:
    return " / ".join(players) if players else "未在公开来源中命中预计首发"


def format_missing_side(items: list[dict[str, str]]) -> str:
    if not items:
        return "未在公开来源中命中明确伤停"
    parts: list[str] = []
    for item in items[:8]:
        player = item["player"]
        reason = item["reason"]
        if item["status"] == "doubtful":
            parts.append(f"{player}(出战成疑{f'，{reason}' if reason else ''})")
        else:
            parts.append(f"{player}({reason or '缺阵'})")
    return "；".join(parts)


def build_notes(structured_data: Mapping[str, Any], source_labels: list[str]) -> str:
    if str(structured_data.get("provider", "") or "") == "RotoWire":
        raw_summary = normalize_text(str(structured_data.get("raw_summary", "") or ""))
        summary_lines = _compact_preview_lines(raw_summary, limit=16)
        return "\n".join(
            [
                "RotoWire 伤停/阵容替补源",
                *(summary_lines or ["未在 RotoWire 页面中解析到明确伤停/预计阵容文本。"]),
                f"来源：{'；'.join(unique_non_empty(source_labels)) or 'RotoWire'}",
            ]
        )
    return "\n".join(
        [
            "预计首发",
            f"主队：{format_lineup_side(list(structured_data.get('home_lineup', []) or []))}",
            f"客队：{format_lineup_side(list(structured_data.get('away_lineup', []) or []))}",
            (
                "关键伤停："
                f"主队{format_missing_side(list(structured_data.get('home_missing', []) or []))}；"
                f"客队{format_missing_side(list(structured_data.get('away_missing', []) or []))}"
            ),
            f"来源：{'；'.join(unique_non_empty(source_labels)) or '外部公开来源补采未命中'}",
        ]
    )


def build_failure_notes() -> str:
    return "\n".join(
        [
            "预计首发",
            "主队：未在公开来源中命中预计首发",
            "客队：未在公开来源中命中预计首发",
            "关键伤停：主队未在公开来源中命中明确伤停；客队未在公开来源中命中明确伤停",
            "来源：外部公开来源补采未命中",
        ]
    )


def fetch_flashscore_structured_data(
    browser: PlaywrightCliBrowser,
    match_url: str,
) -> tuple[dict[str, Any], str]:
    browser.goto(match_url)
    lineup_url = normalize_text(str(browser.eval(FIND_LINEUP_TAB_JS) or ""))
    if not lineup_url:
        current_url = browser.page_url()
        if re.search(r"/bericht/kader|/summary/lineups", current_url, re.I):
            lineup_url = current_url
    if not lineup_url:
        raise RuntimeError("Flashscore 页面未找到阵容标签")

    browser.goto(lineup_url)
    lineup_sides = browser.eval(
        _section_side_texts_js("startaufstellung|predicted lineup|predicted lineups|lineups|aufstellungen")
    )
    missing_sides = browser.eval(
        _section_side_texts_js("wird nicht spielen|missing players|injuries|suspensions|will not play")
    )
    doubtful_sides = browser.eval(
        _section_side_texts_js("fraglich|questionable|doubtful")
    )

    if isinstance(lineup_sides, str):
        lineup_sides = json.loads(lineup_sides or "[]")
    if isinstance(missing_sides, str):
        missing_sides = json.loads(missing_sides or "[]")
    if isinstance(doubtful_sides, str):
        doubtful_sides = json.loads(doubtful_sides or "[]")

    payload = {
        "lineup_home": parse_lineup_side_text(lineup_sides[0] if len(lineup_sides) > 0 else ""),
        "lineup_away": parse_lineup_side_text(lineup_sides[1] if len(lineup_sides) > 1 else ""),
        "missing_home": parse_missing_side_text(missing_sides[0] if len(missing_sides) > 0 else "", "out"),
        "missing_away": parse_missing_side_text(missing_sides[1] if len(missing_sides) > 1 else "", "out"),
        "doubtful_home": parse_missing_side_text(doubtful_sides[0] if len(doubtful_sides) > 0 else "", "doubtful"),
        "doubtful_away": parse_missing_side_text(doubtful_sides[1] if len(doubtful_sides) > 1 else "", "doubtful"),
        "page_url": browser.page_url(),
        "page_title": browser.title(),
    }

    structured = build_structured_lineup_data(payload)
    structured["match_url"] = match_url
    structured["lineup_url"] = lineup_url
    return structured, lineup_url


def fetch_rotowire_structured_data(
    browser: PlaywrightCliBrowser,
    rotowire_url: str,
) -> tuple[dict[str, Any], str]:
    browser.goto(rotowire_url)
    structured = build_rotowire_structured_data(
        browser.body_text(),
        browser.page_url(),
        browser.title(),
    )
    if not structured.get("raw_summary"):
        raise RuntimeError("RotoWire 页面未解析到伤停/阵容相关文本")
    return structured, browser.page_url()


def supplement_injury_or_lineup_notes_from_rotowire(
    browser: PlaywrightCliBrowser,
    home_name: str,
    away_name: str,
    match_date: str,
) -> LineupSupplementResult:
    rotowire_url, engine = discover_rotowire_url(browser, home_name, away_name, match_date)
    if not rotowire_url:
        return LineupSupplementResult(
            notes=build_failure_notes(),
            remarks=[f"伤停/阵容替补源 RotoWire 未命中：查询={home_name} vs {away_name}"],
        )

    structured_data, page_url = fetch_rotowire_structured_data(browser, rotowire_url)
    source_links = unique_non_empty([rotowire_url, page_url])
    source_labels = ["RotoWire"]
    notes = build_notes(structured_data, source_labels)
    remarks = [f"伤停/阵容替补源命中：RotoWire（搜索引擎 {engine or 'unknown'}）"]
    return LineupSupplementResult(
        notes=notes,
        source_links=source_links,
        source_labels=source_labels,
        remarks=remarks,
        structured_data=structured_data,
        lineup_found=bool(structured_data.get("raw_summary") or structured_data.get("home_lineup")),
        injury_found=bool(structured_data.get("home_missing") or structured_data.get("raw_summary")),
    )


def supplement_injury_or_lineup_notes(
    match: Mapping[str, Any],
    shuju_html: str,
) -> LineupSupplementResult:
    team_ids = extract_team_ids(shuju_html)
    match_date = infer_match_date(str(mapping_get(match, "match_time", "") or ""))
    home_name = str(mapping_get(match, "home_team", "") or "")
    away_name = str(mapping_get(match, "away_team", "") or "")
    remarks: list[str] = [
        f"Lineup probe: input={home_name} vs {away_name}; match_date={match_date or 'unknown'}"
    ]

    if len(team_ids) >= 2:
        with suppress(Exception):
            home_name = resolve_team_english_name(team_ids[0]) or home_name
        with suppress(Exception):
            away_name = resolve_team_english_name(team_ids[1]) or away_name
    if team_ids:
        remarks.append(
            f"Lineup probe: resolved team_ids={','.join(team_ids)}; search={home_name} vs {away_name}"
        )
    else:
        remarks.append("Lineup probe: no 500.com team ids found; using listed team names")

    browser = PlaywrightCliBrowser(PlaywrightCliSettings.from_env())
    try:
        match_url, engine = discover_flashscore_match_url(
            browser,
            home_name,
            away_name,
            match_date,
        )
        if not match_url:
            remarks.append(
                f"Flashscore lineup source missed: engine={engine or 'unknown'}; query={home_name} vs {away_name}"
            )
            rotowire_result = supplement_injury_or_lineup_notes_from_rotowire(
                browser,
                home_name,
                away_name,
                match_date,
            )
            rotowire_result.remarks = remarks + rotowire_result.remarks
            return rotowire_result

        remarks.append(
            f"Flashscore lineup source hit: engine={engine or 'unknown'}; url={match_url}"
        )
        structured_data, lineup_url = fetch_flashscore_structured_data(browser, match_url)
        remarks.append(f"Flashscore lineup page parsed: url={lineup_url}")
    except Exception as exc:  # noqa: BLE001
        remarks.append(f"Flashscore lineup source failed: {exc}")
        try:
            rotowire_result = supplement_injury_or_lineup_notes_from_rotowire(
                browser,
                home_name,
                away_name,
                match_date,
            )
            rotowire_result.remarks = remarks + rotowire_result.remarks
            return rotowire_result
        except Exception as rotowire_exc:  # noqa: BLE001
            return LineupSupplementResult(
                notes=build_failure_notes(),
                remarks=remarks
                + [f"Lineup supplement failed: RotoWire={rotowire_exc}"],
            )
    finally:
        browser.close()

    source_links = unique_non_empty(
        [match_url, structured_data.get("lineup_url", ""), structured_data.get("page_url", "")]
    )
    source_labels = ["Flashscore"]
    notes = build_notes(structured_data, source_labels)
    lineup_found = bool(structured_data.get("home_lineup") or structured_data.get("away_lineup"))
    injury_found = bool(structured_data.get("home_missing") or structured_data.get("away_missing"))
    remarks.append(f"伤停/阵容外部补采命中：Flashscore（搜索引擎 {engine or 'unknown'}）")
    if not lineup_found and not injury_found:
        remarks.append(
            "Flashscore lineup source parsed but no concrete lineup/injury items were found"
        )

    return LineupSupplementResult(
        notes=notes,
        source_links=source_links,
        source_labels=source_labels,
        remarks=remarks,
        structured_data=structured_data,
        lineup_found=lineup_found,
        injury_found=injury_found,
    )


__all__ = [
    "LineupSupplementResult",
    "build_failure_notes",
    "supplement_injury_or_lineup_notes_from_rotowire",
    "supplement_injury_or_lineup_notes",
]
