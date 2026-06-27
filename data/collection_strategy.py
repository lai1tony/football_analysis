from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote_plus

from playwright_cli_client import PlaywrightCliBrowser, PlaywrightCliSettings
from source_market_value_client import fetch_match_market_values
from source_supplement_client import supplement_match_data


ANYSEARCH_COMMAND = [
    "node",
    "C:/Users/15696/.codex/skills/anysearch/scripts/anysearch_cli.js",
]

DIMENSION_FIELDS: dict[str, tuple[str, ...]] = {
    "dim1_strength": (
        "elo_home",
        "elo_away",
        "market_value_summary",
        "recent_form_home",
        "recent_form_away",
        "home_away_form",
    ),
    "dim2_dynamics": (
        "head_to_head_summary",
        "injury_or_lineup_notes",
        "motivation_or_schedule_notes",
    ),
    "dim3_market": (
        "european_odds_movement_summary",
        "betting_heat_summary",
    ),
}

FIELD_QUERY_TEMPLATES: dict[str, tuple[str, ...]] = {
    "elo_home": ("{home_team} {league} standings ranking points form",),
    "elo_away": ("{away_team} {league} standings ranking points form",),
    "recent_form_home": ("{home_team} recent form last matches {league}",),
    "recent_form_away": ("{away_team} recent form last matches {league}",),
    "home_away_form": (
        "{home_team} home form {away_team} away form {league}",
        "{home_team} vs {away_team} home away form",
    ),
    "head_to_head_summary": (
        "{home_team} vs {away_team} head to head recent matches",
        "{home_team} {away_team} h2h results",
    ),
    "injury_or_lineup_notes": (
        "{home_team} vs {away_team} injuries lineup team news",
        "{home_team} {away_team} predicted lineups injuries",
    ),
    "motivation_or_schedule_notes": (
        "{home_team} vs {away_team} match preview motivation schedule",
        "{home_team} {away_team} fixture congestion preview",
    ),
    "european_odds_movement_summary": (
        "{home_team} vs {away_team} odds movement opening current odds",
        "{home_team} {away_team} betting odds movement",
    ),
    "betting_heat_summary": (
        "{home_team} vs {away_team} betting trends public money",
        "{home_team} {away_team} betting heat popularity",
    ),
}

LOW_CONFIDENCE_MARKERS = (
    "not found",
    "failed",
    "fallback",
    "collection note",
    "unavailable",
    "placeholder",
)

SEARCH_NOISE_PHRASES = (
    "跳至内容",
    "辅助功能反馈",
    "隐私政策",
    "使用条款",
    "skip to content",
    "accessibility feedback",
    "privacy policy",
    "terms of use",
    "terms and conditions",
    "english",
    "rewards",
)

SEARCH_LINKS_JS = (
    "JSON.stringify("
    "[...document.querySelectorAll('li.b_algo h2 a[href], li.b_algo .b_title a[href], li.b_algo a[href]')]"
    ".slice(0, 20)"
    ".map(a => ({ text: (a.textContent || '').trim(), href: a.href }))"
    ")"
)

QUALITY_SUMMARY_PREFIXES = tuple(f"{dimension}." for dimension in DIMENSION_FIELDS)


@dataclass
class StrategyEvent:
    dimension: str
    field: str
    stage: str
    source: str
    status: str
    quality: float
    detail: str = ""

    def format(self) -> str:
        detail = f"; detail={self.detail}" if self.detail else ""
        return (
            f"{self.dimension}.{self.field}: status={self.status}; "
            f"stage={self.stage}; source={self.source}; quality={self.quality:.2f}"
            f"{detail}"
        )


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


def split_lines(value: str) -> list[str]:
    return [line for line in unique_non_empty((value or "").splitlines()) if line]


def _is_search_noise_summary(title: str, snippet: str) -> bool:
    text = normalize_text(f"{title} {snippet}")
    if not text:
        return True
    folded = text.casefold()
    if not any(phrase in folded for phrase in SEARCH_NOISE_PHRASES):
        return False
    useful_text = folded
    for phrase in SEARCH_NOISE_PHRASES:
        useful_text = useful_text.replace(phrase, "")
    useful_text = re.sub(r"[\W_]+", " ", useful_text).strip()
    return not useful_text


def _field_value(analysis: Mapping[str, Any], field: str) -> str:
    return normalize_text(str(analysis.get(field, "") or ""))


def _has_usable_value(analysis: Mapping[str, Any], field: str) -> bool:
    value = _field_value(analysis, field)
    if not value:
        return False
    folded = value.lower()
    return not any(marker in folded for marker in LOW_CONFIDENCE_MARKERS)


def _dimension_for_field(field: str) -> str:
    for dimension, fields in DIMENSION_FIELDS.items():
        if field in fields:
            return dimension
    return "other"


def _quality_for_existing(field: str, analysis: Mapping[str, Any]) -> float:
    value = _field_value(analysis, field)
    if not value:
        return 0.0
    folded_sources = _field_value(analysis, "collected_sources").lower()
    folded_links = _field_value(analysis, "media_source_links").lower()
    if field == "market_value_summary":
        if "transfermarkt" in folded_sources or "transfermarkt" in folded_links:
            return 0.86
        if "anysearch" in folded_sources or "anysearch" in value.lower():
            return 0.62
        return 0.72
    if field in {"european_odds_movement_summary", "betting_heat_summary"}:
        return 0.86 if "500" in folded_sources or "500.com" in folded_links else 0.64
    if field == "injury_or_lineup_notes":
        if "flashscore" in folded_sources or "flashscore" in folded_links:
            return 0.82
        return 0.62
    if field in {"elo_home", "elo_away"}:
        return 0.78 if "jifen" in folded_links or "500" in folded_sources else 0.62
    return 0.76 if "500" in folded_sources or "500.com" in folded_links else 0.60


def _anysearch_command() -> list[str]:
    configured = Path("C:/Users/15696/.codex/skills/anysearch/runtime.conf")
    if configured.exists():
        for line in configured.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("Command:"):
                command = line.split(":", 1)[1].strip()
                if command:
                    return command.split()
    return ANYSEARCH_COMMAND[:]


def _run_anysearch_search(query: str, max_results: int = 5) -> str:
    command = _anysearch_command() + ["search", query, "--max_results", str(max_results)]
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


def _iter_search_items(output: str) -> list[dict[str, str]]:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        items: list[dict[str, str]] = []
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
        return items
    if isinstance(payload, list):
        return [
            {
                "title": normalize_text(str(item.get("title", "") or item.get("name", ""))),
                "url": normalize_text(str(item.get("url", "") or item.get("link", ""))),
                "snippet": normalize_text(
                    str(item.get("snippet", "") or item.get("description", "") or item.get("content", ""))
                ),
            }
            for item in payload
            if isinstance(item, dict)
        ]
    if isinstance(payload, dict):
        for key in ("results", "data", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return _iter_search_items(json.dumps(value))
    return []


def _format_search_summary(items: list[dict[str, str]], provider: str) -> tuple[str, list[str]]:
    lines: list[str] = []
    sources: list[str] = []
    for item in items:
        title = normalize_text(item.get("title", ""))
        snippet = normalize_text(item.get("snippet", ""))
        url = normalize_text(item.get("url", ""))
        if _is_search_noise_summary(title, snippet):
            continue
        summary = title
        if snippet:
            summary = f"{summary}: {snippet[:180]}" if summary else snippet[:180]
        lines.append(summary[:260])
        if url:
            sources.append(f"{provider}: {url}")
        if len(lines) >= 3:
            break
    return "\n".join(lines), sources


def _build_queries(field: str, match: Mapping[str, Any]) -> list[str]:
    context = {
        "home_team": str(match.get("home_team", "") or ""),
        "away_team": str(match.get("away_team", "") or ""),
        "league": str(match.get("league", "") or ""),
        "match_time": str(match.get("match_time", "") or ""),
    }
    templates = FIELD_QUERY_TEMPLATES.get(field, ())
    return unique_non_empty([template.format(**context).strip() for template in templates])


def _playwright_search_field(field: str, match: Mapping[str, Any]) -> tuple[str, list[str], str]:
    if os.environ.get("FOOTBALL_SKIP_PLAYWRIGHT_FIELD_FALLBACK", "").strip() == "1":
        raise RuntimeError("FOOTBALL_SKIP_PLAYWRIGHT_FIELD_FALLBACK=1")
    queries = _build_queries(field, match)
    if not queries:
        return "", [], ""
    browser = PlaywrightCliBrowser(PlaywrightCliSettings.from_env())
    try:
        for query in queries[:2]:
            browser.goto("https://www.bing.com/search?q=" + quote_plus(query))
            title = browser.title()
            body = browser.body_text()[:3000].lower()
            if "one last step" in body or "solve this puzzle" in body:
                continue
            raw = browser.eval(SEARCH_LINKS_JS) or []
            if isinstance(raw, str):
                raw = json.loads(raw or "[]")
            items = [
                {
                    "title": normalize_text(str(item.get("text", ""))),
                    "url": normalize_text(str(item.get("href", ""))),
                    "snippet": "",
                }
                for item in raw
                if isinstance(item, dict) and normalize_text(str(item.get("href", "")))
            ]
            summary, sources = _format_search_summary(items, "PlaywrightSearch")
            if summary:
                return summary, sources, f"{query}; page={title}"
    finally:
        browser.close()
    return "", [], ""


def _anysearch_field(field: str, match: Mapping[str, Any]) -> tuple[str, list[str], str]:
    queries = _build_queries(field, match)
    if not queries:
        return "", [], ""
    last_error = ""
    for query in queries[:2]:
        try:
            output = _run_anysearch_search(query)
            summary, sources = _format_search_summary(_iter_search_items(output), "AnySearch")
            if summary:
                return summary, sources, query
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
    if last_error:
        raise RuntimeError(last_error)
    return "", [], ""


def _append_sources(analysis: dict[str, Any], sources: list[str]) -> None:
    if not sources:
        return
    links = split_lines(str(analysis.get("media_source_links", "") or ""))
    labels = split_lines(str(analysis.get("collected_sources", "") or ""))
    for source in sources:
        if ":" in source:
            label, url = source.split(":", 1)
            label = normalize_text(label)
            url = normalize_text(url)
            if url:
                links.append(url)
            if label:
                labels.append(label)
        else:
            labels.append(source)
    analysis["media_source_links"] = "\n".join(unique_non_empty(links))
    analysis["collected_sources"] = "\n".join(unique_non_empty(labels))


def _append_remarks(analysis: dict[str, Any], remarks: list[str]) -> None:
    if not remarks:
        return
    existing = split_lines(str(analysis.get("remarks", "") or ""))
    analysis["remarks"] = "\n".join(unique_non_empty(existing + remarks))


def _write_quality_summary(analysis: dict[str, Any], events: list[StrategyEvent]) -> None:
    existing = [
        line
        for line in split_lines(str(analysis.get("collection_quality_summary", "") or ""))
        if not line.startswith("strategy:")
        and not line.startswith(QUALITY_SUMMARY_PREFIXES)
    ]
    lines = ["strategy: primary -> playwright-cli -> anysearch"]
    lines.extend(event.format() for event in events)
    analysis["collection_quality_summary"] = "\n".join(unique_non_empty(existing + lines))


def _apply_market_value(match: Mapping[str, Any], analysis: dict[str, Any]) -> tuple[list[str], list[str]]:
    sources: list[str] = []
    errors: list[str] = []
    if _has_usable_value(analysis, "market_value_summary"):
        return sources, errors
    result = fetch_match_market_values(
        str(match.get("home_team", "") or ""),
        str(match.get("away_team", "") or ""),
        str(match.get("league", "") or ""),
    )
    if result.get("market_value_summary"):
        analysis["market_value_summary"] = result["market_value_summary"]
    sources.extend(str(item) for item in result.get("sources", []) or [])
    errors.extend(f"MarketValue: {item}" for item in result.get("errors", []) or [] if item)
    return sources, errors


def _apply_playwright_supplement(
    match: Mapping[str, Any],
    analysis: dict[str, Any],
    missing_fields: list[str],
) -> tuple[list[str], list[str]]:
    if not missing_fields:
        return [], []
    supplementable = {
        field
        for field in missing_fields
        if _dimension_for_field(field) in {"dim1_strength", "dim2_dynamics"}
    }
    if not supplementable:
        return [], []
    result = supplement_match_data(
        home_team=str(match.get("home_team", "") or ""),
        away_team=str(match.get("away_team", "") or ""),
        league=str(match.get("league", "") or ""),
        match_time=str(match.get("match_time", "") or ""),
        missing_dimensions=None,
    )
    for field in (
        "elo_home",
        "elo_away",
        "market_value_summary",
        "head_to_head_summary",
        "recent_form_home",
        "recent_form_away",
        "injury_or_lineup_notes",
    ):
        if result.get(field) and not _has_usable_value(analysis, field):
            analysis[field] = result[field]
    sources = [str(item) for item in result.get("sources", []) or []]
    errors = [f"PlaywrightSupplement: {item}" for item in result.get("errors", []) or [] if item]
    return sources, errors


def apply_unified_collection_strategy(
    match: Mapping[str, Any],
    analysis: dict[str, Any],
    *,
    required_fields: list[str],
    missing_fields: list[str] | None = None,
    allow_fallback: bool = True,
) -> dict[str, Any]:
    allow_fallback = allow_fallback and os.environ.get(
        "FOOTBALL_SKIP_EXTERNAL_SUPPLEMENT", ""
    ).strip().lower() not in {"1", "true", "yes", "on"}
    events: list[StrategyEvent] = []
    sources: list[str] = []
    remarks: list[str] = []

    for field in required_fields:
        if _has_usable_value(analysis, field):
            events.append(
                StrategyEvent(
                    dimension=_dimension_for_field(field),
                    field=field,
                    stage="primary",
                    source="500.com/existing",
                    status="success",
                    quality=_quality_for_existing(field, analysis),
                )
            )

    if allow_fallback:
        market_sources, market_errors = _apply_market_value(match, analysis)
        sources.extend(market_sources)
        remarks.extend(market_errors)

    current_missing = [
        field
        for field in (missing_fields or required_fields)
        if not _has_usable_value(analysis, field)
    ]
    if allow_fallback:
        playwright_sources, playwright_errors = _apply_playwright_supplement(
            match,
            analysis,
            current_missing,
        )
        sources.extend(playwright_sources)
        remarks.extend(playwright_errors)

    for field in required_fields:
        if _has_usable_value(analysis, field) and not any(event.field == field for event in events):
            events.append(
                StrategyEvent(
                    dimension=_dimension_for_field(field),
                    field=field,
                    stage="playwright-cli",
                    source="external supplement",
                    status="success",
                    quality=_quality_for_existing(field, analysis),
                )
            )

    for field in required_fields:
        if _has_usable_value(analysis, field):
            continue
        if not allow_fallback:
            events.append(
                StrategyEvent(
                    dimension=_dimension_for_field(field),
                    field=field,
                    stage="primary",
                    source="existing",
                    status="missing",
                    quality=0.0,
                )
            )
            continue
        try:
            summary, field_sources, query = _playwright_search_field(field, match)
            if summary:
                analysis[field] = summary
                sources.extend(field_sources)
                events.append(
                    StrategyEvent(
                        dimension=_dimension_for_field(field),
                        field=field,
                        stage="playwright-cli",
                        source="search",
                        status="success",
                        quality=0.52,
                        detail=query[:160],
                    )
                )
                continue
        except Exception as exc:  # noqa: BLE001
            remarks.append(f"UnifiedStrategy Playwright {field}: {exc}")

        try:
            summary, field_sources, query = _anysearch_field(field, match)
            if summary:
                analysis[field] = summary
                sources.extend(field_sources)
                events.append(
                    StrategyEvent(
                        dimension=_dimension_for_field(field),
                        field=field,
                        stage="anysearch",
                        source="search",
                        status="success",
                        quality=0.45,
                        detail=query[:160],
                    )
                )
                continue
        except Exception as exc:  # noqa: BLE001
            remarks.append(f"UnifiedStrategy AnySearch {field}: {exc}")

        events.append(
            StrategyEvent(
                dimension=_dimension_for_field(field),
                field=field,
                stage="anysearch",
                source="search",
                status="missing",
                quality=0.0,
            )
        )

    _append_sources(analysis, sources)
    _append_remarks(analysis, remarks)
    _write_quality_summary(analysis, events)
    return analysis


__all__ = [
    "DIMENSION_FIELDS",
    "StrategyEvent",
    "apply_unified_collection_strategy",
]
