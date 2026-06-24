from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from bs4 import BeautifulSoup

from playwright_cli_client import PlaywrightCliBrowser, PlaywrightCliSettings


ANYSEARCH_COMMAND = [
    "node",
    "C:/Users/15696/.codex/skills/anysearch/scripts/anysearch_cli.js",
]

TEAM_SEARCH_ALIASES = {
    "墨西哥": ["Mexico"],
    "南非": ["South Africa"],
    "韩国": ["South Korea", "Korea Republic"],
    "捷克": ["Czechia", "Czech Republic"],
    "加拿大": ["Canada"],
    "波黑": ["Bosnia and Herzegovina"],
    "美国": ["United States", "USA"],
    "巴拉圭": ["Paraguay"],
    "卡塔尔": ["Qatar"],
    "瑞士": ["Switzerland"],
    "巴西": ["Brazil"],
    "摩洛哥": ["Morocco"],
    "海地": ["Haiti"],
    "苏格兰": ["Scotland"],
    "澳大利": ["Australia"],
    "澳大利亚": ["Australia"],
    "土耳其": ["Turkey", "Turkiye"],
    "荷兰": ["Netherlands"],
    "日本": ["Japan"],
    "科特迪": ["Ivory Coast", "Cote d'Ivoire"],
    "科特迪瓦": ["Ivory Coast", "Cote d'Ivoire"],
    "厄瓜多": ["Ecuador"],
    "厄瓜多尔": ["Ecuador"],
    "瑞典": ["Sweden"],
    "突尼斯": ["Tunisia"],
    "西班牙": ["Spain"],
    "佛得角": ["Cape Verde"],
    "比利时": ["Belgium"],
    "埃及": ["Egypt"],
    "沙特": ["Saudi Arabia"],
    "乌拉圭": ["Uruguay"],
    "斯洛伐": ["Slovakia"],
    "斯洛伐克": ["Slovakia"],
    "黑山": ["Montenegro"],
    "匈牙利": ["Hungary"],
    "芬兰": ["Finland"],
    "爱尔兰": ["Ireland"],
    "葡萄牙": ["Portugal"],
    "智利": ["Chile"],
    "罗马尼": ["Romania"],
    "罗马尼亚": ["Romania"],
    "威尔士": ["Wales"],
    "德国": ["Germany"],
    "巴拿马": ["Panama"],
    "英格兰": ["England"],
    "新西兰": ["New Zealand"],
    "克罗地": ["Croatia"],
    "克罗地亚": ["Croatia"],
    "斯洛文": ["Slovenia"],
    "斯洛文尼亚": ["Slovenia"],
    "希腊": ["Greece"],
    "意大利": ["Italy"],
    "挪威": ["Norway"],
}


@dataclass
class TeamMarketValue:
    team_name: str
    value_eur_m: float = 0.0
    summary: str = ""
    source_url: str = ""
    source_label: str = ""
    error: str = ""


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _value_to_eur_m(amount: str, unit: str) -> float:
    number = float(str(amount).replace(",", "."))
    unit_text = (unit or "").lower()
    if unit_text in {"bn", "b", "billion"}:
        return number * 1000.0
    if unit_text in {"k", "th", "thousand"}:
        return number / 1000.0
    return number


_CURRENCY_PATTERN = r"(?:€|\u20ac|EUR|eur|£|GBP|gbp|\$|USD|usd)"
_UNIT_PATTERN = r"(bn|b|billion|m|mil|million|k|th|thousand)?"
_SQUAD_VALUE_CONTEXT_PATTERN = re.compile(
    r"(?:squad\s+(?:worth|value|valuation)|club\s+(?:worth|value|valuation)|team\s+(?:worth|value|valuation)|cumulative|estimated|worth|valuation|total\s+market\s+value|total\s*:)",
    flags=re.I,
)


def _coerce_market_value_to_eur_m(amount: str, unit: str, currency_hint: str = "") -> float:
    value = _value_to_eur_m(amount, unit or "m")
    hint = (currency_hint or "").lower()
    if "£" in currency_hint or "gbp" in hint:
        return value * 1.17
    if "$" in currency_hint or "usd" in hint:
        return value * 0.92
    return value


def _context_window(text: str, start: int, end: int, radius: int = 90) -> str:
    return text[max(0, start - radius): min(len(text), end + radius)]


def _is_average_or_player_value_context(context: str) -> bool:
    folded = (context or "").lower()
    if "total market value" in folded or "squad value" in folded or "squad worth" in folded:
        return False
    return any(
        token in folded
        for token in (
            "ø-market value",
            "average market value",
            "avg market value",
            "average value",
            "# player",
            "date of birth",
            "position",
        )
    )


def extract_market_value_eur_m(text: str) -> float:
    cleaned = normalize_text(text)
    if not cleaned:
        return 0.0

    total_patterns = [
        rf"({_CURRENCY_PATTERN})\s*([0-9]+(?:[.,][0-9]+)?)\s*{_UNIT_PATTERN}\s*(?:Total\s*market\s*value|Squad\s+value|Squad\s+worth|Cumulative\s+market\s+value)",
        rf"\bTotal\s*(?:market\s*value)?\s*:[^{_CURRENCY_PATTERN}0-9]*({_CURRENCY_PATTERN})\s*([0-9]+(?:[.,][0-9]+)?)\s*{_UNIT_PATTERN}",
        rf"\bTotal\s*(?:market\s*value)?\s*:\s*([0-9]+(?:[.,][0-9]+)?)\s*{_UNIT_PATTERN}\s*({_CURRENCY_PATTERN})",
        rf"(?:squad|club|team)\s+(?:worth|value|valuation)[^0-9€£$]{{0,80}}(?:estimated\s*)?(?:stands\s+)?(?:at\s*)?({_CURRENCY_PATTERN})?\s*([0-9]+(?:[.,][0-9]+)?)\s*{_UNIT_PATTERN}",
        rf"(?:cumulative|total)\s+(?:market\s+)?value[^0-9€£$]{{0,80}}({_CURRENCY_PATTERN})?\s*([0-9]+(?:[.,][0-9]+)?)\s*{_UNIT_PATTERN}",
    ]
    for pattern in total_patterns:
        total_match = re.search(pattern, cleaned, flags=re.I)
        if not total_match:
            continue
        if _is_average_or_player_value_context(_context_window(cleaned, total_match.start(), total_match.end())):
            continue
        groups = [item or "" for item in total_match.groups()]
        currency = next((item for item in groups if re.search(_CURRENCY_PATTERN, item, flags=re.I)), "")
        amount = next((item for item in groups if re.fullmatch(r"[0-9]+(?:[.,][0-9]+)?", item)), "")
        unit = next(
            (
                item
                for item in groups
                if item.lower() in {"bn", "b", "billion", "m", "mil", "million", "k", "th", "thousand"}
            ),
            "",
        )
        if amount:
            try:
                return round(_coerce_market_value_to_eur_m(amount, unit or "m", currency), 3)
            except (TypeError, ValueError):
                pass

    values: list[float] = []
    value_patterns = [
        rf"({_CURRENCY_PATTERN})\s*([0-9]+(?:[.,][0-9]+)?)\s*{_UNIT_PATTERN}",
        rf"([0-9]+(?:[.,][0-9]+)?)\s*(bn|b|billion|m|mil|million|k|th|thousand)\s*({_CURRENCY_PATTERN})",
        rf"([0-9]+(?:[.,][0-9]+)?)\s*(bn|b|billion|m|mil|million)\b",
    ]
    for pattern in value_patterns:
        for match in re.finditer(pattern, cleaned, flags=re.I):
            context = _context_window(cleaned, match.start(), match.end())
            if _is_average_or_player_value_context(context):
                continue
            if not _SQUAD_VALUE_CONTEXT_PATTERN.search(context):
                continue
            groups = [item or "" for item in match.groups()]
            currency = next((item for item in groups if re.search(_CURRENCY_PATTERN, item, flags=re.I)), "")
            amount = next((item for item in groups if re.fullmatch(r"[0-9]+(?:[.,][0-9]+)?", item)), "")
            unit = next(
                (
                    item
                    for item in groups
                    if item.lower() in {"bn", "b", "billion", "m", "mil", "million", "k", "th", "thousand"}
                ),
                "",
            )
            try:
                values.append(_coerce_market_value_to_eur_m(amount, unit or "m", currency))
            except (TypeError, ValueError):
                continue
    if not values:
        return 0.0
    return round(max(values), 3)


def extract_partial_squad_value_eur_m(text: str) -> float:
    cleaned = normalize_text(text)
    if not cleaned:
        return 0.0
    values: list[float] = []
    for match in re.finditer(
        rf"({_CURRENCY_PATTERN})\s*([0-9]+(?:[.,][0-9]+)?)\s*(bn|b|billion|m|mil|million|k|th|thousand)?",
        cleaned,
        flags=re.I,
    ):
        try:
            values.append(_coerce_market_value_to_eur_m(match.group(2), match.group(3) or "m", match.group(1)))
        except (TypeError, ValueError):
            continue
    meaningful = [value for value in values if value > 0]
    if len(meaningful) < 4:
        return 0.0
    return round(sum(meaningful), 3)


def format_market_value_summary(team_name: str, value_eur_m: float, source_label: str) -> str:
    if value_eur_m <= 0:
        return ""
    return f"{team_name}: squad market value EUR {value_eur_m:.2f}m (source: {source_label})"


def build_market_value_summary(home: TeamMarketValue, away: TeamMarketValue) -> str:
    lines = []
    for item in (home, away):
        if item.summary:
            lines.append(item.summary)
    if home.value_eur_m > 0 and away.value_eur_m > 0:
        gap = home.value_eur_m - away.value_eur_m
        ratio = home.value_eur_m / away.value_eur_m if away.value_eur_m > 0 else 0.0
        lines.append(f"market value gap: home-away EUR {gap:+.2f}m; ratio {ratio:.2f}x")
    errors = [item.error for item in (home, away) if item.error]
    for error in errors:
        lines.append(f"market value collection note: {error}")
    return "\n".join(lines)


def _tm_search_url(team_name: str) -> str:
    return "https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche?query=" + quote_plus(team_name)


def _name_candidates(team_name: str) -> list[str]:
    candidates = [team_name]
    candidates.extend(TEAM_SEARCH_ALIASES.get(team_name, []))
    try:
        from source_supplement_client import _resolve_team_en_name  # type: ignore

        resolved = _resolve_team_en_name(team_name)
        if resolved:
            candidates.append(resolved)
    except Exception:  # noqa: BLE001
        pass
    try:
        from team_name_aliases import resolve_team_aliases

        candidates.extend(resolve_team_aliases(team_name))
    except Exception:  # noqa: BLE001
        pass
    seen = set()
    result = []
    for item in candidates:
        clean = normalize_text(str(item or ""))
        if clean and clean.lower() not in seen:
            seen.add(clean.lower())
            result.append(clean)
    return result


def _extract_transfermarkt_candidate(html: str, team_name: str) -> tuple[str, str]:
    soup = BeautifulSoup(html or "", "html.parser")
    best_href = ""
    best_text = ""
    folded_name = team_name.lower()
    for anchor in soup.find_all("a", href=True):
        href = str(anchor.get("href", ""))
        text = normalize_text(anchor.get_text(" ", strip=True))
        haystack = f"{text} {href}".lower()
        if "/verein/" not in href and "/startseite/verein/" not in href:
            continue
        score = 0
        if "transfermarkt" in href:
            score += 2
        if folded_name and folded_name in haystack:
            score += 4
        if score <= 0:
            continue
        if score > (4 if best_href else -1):
            best_href = href
            best_text = text
    if best_href.startswith("/"):
        best_href = "https://www.transfermarkt.com" + best_href
    return best_href, best_text


def _parse_transfermarkt_team_page(html: str, team_name: str) -> tuple[float, str]:
    text = BeautifulSoup(html or "", "html.parser").get_text(" ", strip=True)
    candidates = []
    for marker in ("Total market value", "Total:", "Squad value", "Squad worth", "Cumulative market value"):
        idx = text.lower().find(marker.lower())
        if idx >= 0:
            candidates.append(text[max(0, idx - 120): idx + 220])
    candidates.append(text[:1200])
    for snippet in candidates:
        value = extract_market_value_eur_m(snippet)
        if value > 0:
            return value, normalize_text(snippet)[:220]
    return 0.0, ""


def fetch_team_market_value_via_playwright(team_name: str) -> TeamMarketValue:
    result = TeamMarketValue(team_name=team_name)
    browser = PlaywrightCliBrowser(PlaywrightCliSettings.from_env())
    try:
        return fetch_team_market_value_with_browser(browser, team_name)
    except Exception as exc:  # noqa: BLE001
        result.error = f"playwright market value failed: {exc}"
        return result
    finally:
        browser.close()


def fetch_team_market_value_with_browser(
    browser: PlaywrightCliBrowser,
    team_name: str,
) -> TeamMarketValue:
    result = TeamMarketValue(team_name=team_name)
    try:
        last_error = ""
        for query_name in _name_candidates(team_name):
            browser.goto(_tm_search_url(query_name))
            search_html = browser.eval("document.documentElement.outerHTML")
            team_url, label = _extract_transfermarkt_candidate(str(search_html or ""), query_name)
            if not team_url:
                last_error = f"Transfermarkt team page not found for {query_name}"
                continue
            browser.goto(team_url)
            page_html = browser.eval("document.documentElement.outerHTML")
            value, _snippet = _parse_transfermarkt_team_page(str(page_html or ""), query_name)
            if value <= 0:
                last_error = f"Transfermarkt value not parsed for {query_name}"
                continue
            result.value_eur_m = value
            result.source_url = team_url
            result.source_label = "Transfermarkt"
            result.summary = format_market_value_summary(team_name, value, result.source_label)
            return result
        raise RuntimeError(last_error or "Transfermarkt value not parsed")
    except Exception as exc:  # noqa: BLE001
        result.error = f"playwright market value failed: {exc}"
        return result


def _anysearch_command() -> list[str]:
    configured = Path("C:/Users/15696/.codex/skills/anysearch/runtime.conf")
    if configured.exists():
        for line in configured.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("Command:"):
                command = line.split(":", 1)[1].strip()
                if command:
                    return command.split()
    return ANYSEARCH_COMMAND[:]


def _run_anysearch_search(query: str) -> str:
    command = _anysearch_command() + ["search", query, "--max_results", "5"]
    completed = subprocess.run(
        command,
        capture_output=True,
        timeout=int(os.environ.get("FOOTBALL_ANYSEARCH_TIMEOUT", "30")),
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = (completed.stdout or "").strip()
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or output or f"exit {completed.returncode}").strip())
    return output


def _run_anysearch_extract(url: str) -> str:
    command = _anysearch_command() + ["extract", url]
    completed = subprocess.run(
        command,
        capture_output=True,
        timeout=int(os.environ.get("FOOTBALL_ANYSEARCH_EXTRACT_TIMEOUT", os.environ.get("FOOTBALL_ANYSEARCH_TIMEOUT", "30"))),
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = (completed.stdout or "").strip()
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or output or f"exit {completed.returncode}").strip())
    return output


def _team_tokens(team_name: str) -> list[str]:
    aliases = _name_candidates(team_name)
    try:
        from source_supplement_client import _resolve_team_en_name  # type: ignore

        resolved = _resolve_team_en_name(team_name)
        if resolved:
            aliases.append(resolved)
    except Exception:  # noqa: BLE001
        pass
    try:
        from team_name_aliases import resolve_team_aliases

        aliases.extend(resolve_team_aliases(team_name))
    except Exception:  # noqa: BLE001
        pass
    tokens: list[str] = []
    stopwords = {"fc", "cf", "sc", "afc", "club", "team", "national", "football"}
    for alias in aliases:
        for token in re.findall(r"[a-z0-9]+", alias.lower()):
            if len(token) >= 3 and token not in stopwords and token not in tokens:
                tokens.append(token)
    return tokens[:5]


def _search_item_mentions_team(team_name: str, title: str, url: str, snippet: str) -> bool:
    haystack = f"{title} {url} {snippet}".lower()
    tokens = _team_tokens(team_name)
    if not tokens:
        return True
    return any(token in haystack for token in tokens)


def _is_generic_competition_page(title: str, url: str, snippet: str) -> bool:
    haystack = f"{title} {url} {snippet}".lower()
    bad_tokens = (
        "participating teams",
        "teilnehmer",
        "competition",
        "world cup",
        "league table",
        "standings",
        "wettbewerb",
        "pokalwettbewerb",
        "wertvollstespieler",
        "marktwertetop",
        "spieler-statistik",
        "most valuable players",
    )
    return any(token in haystack for token in bad_tokens)


def _is_squad_market_value_page(title: str, url: str, snippet: str) -> bool:
    haystack = f"{title} {url} {snippet}".lower()
    if "transfermarkt" not in haystack:
        return False
    return any(
        token in haystack
        for token in (
            "/kader/",
            "/erweiterterkader/",
            "/startseite/",
            "/marktwertanalyse/",
            "detailed squad",
            "extended squad",
            "club profile",
            "squad worth",
            "squad value",
            "squad valuations",
            "market value analysis",
            "current squad with market values",
            "total:",
        )
    )


def _market_value_candidate_priority(title: str, url: str, snippet: str) -> int:
    haystack = f"{title} {url} {snippet}".lower()
    if "/kader/" in haystack or "/erweiterterkader/" in haystack:
        return 40
    if "/startseite/" in haystack or "club profile" in haystack:
        return 35
    if "/marktwertanalyse/" in haystack or "market value analysis" in haystack:
        return 30
    if "squad worth" in haystack or "squad value" in haystack or "squad valuation" in haystack:
        return 20
    return 10


def _has_total_market_value(text: str) -> bool:
    return bool(
        re.search(r"\bTotal\s*(?:market\s*value)?\s*:[^€\u20ac$£0-9]*[€\u20ac$£]?\s*[0-9]", text or "", re.I)
        or re.search(r"(?:squad|club|team)\s+(?:worth|value|valuation)[^0-9€£$]{0,80}[€\u20ac$£]?\s*[0-9]", text or "", re.I)
        or re.search(r"(?:cumulative|total)\s+(?:market\s+)?value[^0-9€£$]{0,80}[€\u20ac$£]?\s*[0-9]", text or "", re.I)
    )


def _iter_search_items(output: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        items: list[dict[str, Any]] = []
        for block in re.split(r"\n###\s+\d+\.\s+", output):
            block = block.strip()
            if not block or block.startswith("## Search Results"):
                continue
            lines = block.splitlines()
            title = normalize_text(lines[0] if lines else "")
            url_match = re.search(r"- \*\*URL\*\*:\s*(\S+)", block)
            url = url_match.group(1) if url_match else ""
            snippet = normalize_text(re.sub(r"- \*\*URL\*\*:\s*\S+", "", "\n".join(lines[1:])))
            items.append({"title": title, "url": url, "snippet": snippet})
        return items or [{"title": "", "url": "", "snippet": output}]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("results", "data", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [payload]
    return []


def fetch_team_market_value_via_anysearch(team_name: str, league: str = "") -> TeamMarketValue:
    result = TeamMarketValue(team_name=team_name)
    candidates = _name_candidates(team_name)
    query_name = candidates[1] if len(candidates) > 1 else candidates[0]
    query = f"{query_name} {league} Transfermarkt squad market value".strip()
    try:
        output = _run_anysearch_search(query)
        best_value = 0.0
        best_url = ""
        best_label = "AnySearch"
        best_snippet = ""
        best_priority = -1
        best_partial_value = 0.0
        best_partial_url = ""
        best_partial_snippet = ""
        best_partial_priority = -1
        for item in _iter_search_items(output):
            title = normalize_text(str(item.get("title", "") or item.get("name", "") or ""))
            url = normalize_text(str(item.get("url", "") or item.get("link", "") or ""))
            snippet = normalize_text(
                str(item.get("snippet", "") or item.get("description", "") or item.get("content", "") or "")
            )
            if _is_generic_competition_page(title, url, snippet):
                continue
            if not _is_squad_market_value_page(title, url, snippet):
                continue
            if not _search_item_mentions_team(team_name, title, url, snippet):
                continue
            priority = _market_value_candidate_priority(title, url, snippet)
            text = " ".join([title, snippet, url])
            value = extract_market_value_eur_m(text)
            if value <= 0 and url:
                try:
                    extracted = _run_anysearch_extract(url)
                    value = extract_market_value_eur_m(extracted)
                    if value <= 0:
                        partial_value = extract_partial_squad_value_eur_m(extracted)
                        if partial_value > 0 and (
                            priority > best_partial_priority
                            or (priority == best_partial_priority and partial_value > best_partial_value)
                        ):
                            best_partial_value = partial_value
                            best_partial_url = url
                            best_partial_snippet = normalize_text(extracted)[:180]
                            best_partial_priority = priority
                    elif not snippet:
                        snippet = normalize_text(extracted)[:180]
                except Exception as exc:  # noqa: BLE001
                    if not result.error:
                        result.error = f"anysearch extract note: {exc}"
            elif value <= 0 and _has_total_market_value(text):
                value = extract_market_value_eur_m(text)
            partial_from_snippet = extract_partial_squad_value_eur_m(text)
            if value <= 0 and partial_from_snippet > 0 and (
                priority > best_partial_priority
                or (priority == best_partial_priority and partial_from_snippet > best_partial_value)
            ):
                best_partial_value = partial_from_snippet
                best_partial_url = url
                best_partial_snippet = snippet or title
                best_partial_priority = priority
            if value > 0 and (
                priority > best_priority
                or (priority == best_priority and value > best_value)
            ):
                best_value = value
                best_url = url
                best_snippet = snippet or title
                best_priority = priority
        if best_value <= 0 and best_partial_value > 0:
            best_value = best_partial_value
            best_url = best_partial_url
            best_snippet = best_partial_snippet
            best_label = "AnySearch partial Transfermarkt"
        if best_value <= 0:
            raise RuntimeError("AnySearch value not parsed")
        result.value_eur_m = best_value
        result.source_url = best_url or "anysearch"
        result.source_label = best_label
        result.summary = format_market_value_summary(team_name, best_value, best_label)
        if best_snippet:
            result.summary += f"; snippet: {best_snippet[:180]}"
        return result
    except Exception as exc:  # noqa: BLE001
        result.error = f"anysearch market value failed: {exc}"
        return result


def fetch_match_market_values(home_team: str, away_team: str, league: str = "") -> dict[str, Any]:
    home = fetch_team_market_value_via_playwright(home_team)
    away = fetch_team_market_value_via_playwright(away_team)

    if home.value_eur_m <= 0:
        fallback = fetch_team_market_value_via_anysearch(home_team, league)
        if fallback.value_eur_m > 0:
            home = fallback
        elif fallback.error:
            home.error = "; ".join([item for item in (home.error, fallback.error) if item])
    if away.value_eur_m <= 0:
        fallback = fetch_team_market_value_via_anysearch(away_team, league)
        if fallback.value_eur_m > 0:
            away = fallback
        elif fallback.error:
            away.error = "; ".join([item for item in (away.error, fallback.error) if item])

    sources = []
    for item in (home, away):
        if item.source_url:
            sources.append(f"{item.source_label or 'MarketValue'}: {item.source_url}")

    return {
        "home_value_eur_m": home.value_eur_m,
        "away_value_eur_m": away.value_eur_m,
        "market_value_summary": build_market_value_summary(home, away),
        "sources": sources,
        "errors": [item.error for item in (home, away) if item.error],
        "success": bool(home.value_eur_m > 0 or away.value_eur_m > 0),
    }


__all__ = [
    "extract_market_value_eur_m",
    "fetch_match_market_values",
    "fetch_team_market_value_via_anysearch",
    "fetch_team_market_value_via_playwright",
    "fetch_team_market_value_with_browser",
]
