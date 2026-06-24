from __future__ import annotations

import argparse
from datetime import datetime
from typing import Any

from collection_repository import get_match_analysis, init_db, list_issues, list_matches_by_issue, save_analysis
from collection_service import split_lines, unique_non_empty
from source_market_value_client import (
    TeamMarketValue,
    build_market_value_summary,
    fetch_team_market_value_via_anysearch,
    fetch_team_market_value_with_browser,
)
from playwright_cli_client import PlaywrightCliBrowser, PlaywrightCliSettings


ANALYSIS_FIELDS = (
    "match_id",
    "collected_at",
    "elo_home",
    "elo_away",
    "market_value_summary",
    "recent_form_home",
    "recent_form_away",
    "home_away_form",
    "head_to_head_summary",
    "injury_or_lineup_notes",
    "motivation_or_schedule_notes",
    "european_odds_movement_summary",
    "betting_heat_summary",
    "media_source_links",
    "collected_sources",
    "remarks",
)


def _row_dict(row: Any) -> dict[str, Any]:
    return {field: row[field] if field in row.keys() else "" for field in ANALYSIS_FIELDS}


def _merge_sources(analysis: dict[str, Any], sources: list[str]) -> None:
    links = split_lines(str(analysis.get("media_source_links", "") or ""))
    labels = split_lines(str(analysis.get("collected_sources", "") or ""))
    for src in sources:
        if ":" not in src:
            continue
        label, url = src.split(":", 1)
        label = label.strip()
        url = url.strip()
        if url:
            links.append(url)
        if label:
            labels.append(label)
    analysis["media_source_links"] = "\n".join(unique_non_empty(links))
    analysis["collected_sources"] = "\n".join(unique_non_empty(labels))


def _append_remarks(analysis: dict[str, Any], remarks: list[str]) -> None:
    existing = split_lines(str(analysis.get("remarks", "") or ""))
    analysis["remarks"] = "\n".join(unique_non_empty(existing + [item for item in remarks if item]))


def backfill_market_values(*, issue: str = "", limit: int = 0, dry_run: bool = False) -> dict[str, int]:
    init_db()
    issues = [issue] if issue else list_issues()
    scanned = updated = skipped = failed = 0
    team_cache: dict[tuple[str, str], TeamMarketValue] = {}
    browser: PlaywrightCliBrowser | None = None

    def fetch_team(team_name: str, league: str) -> TeamMarketValue:
        nonlocal browser
        key = (team_name, league)
        if key in team_cache:
            return team_cache[key]
        if browser is None:
            browser = PlaywrightCliBrowser(PlaywrightCliSettings.from_env())
        value = fetch_team_market_value_with_browser(browser, team_name)
        if value.value_eur_m <= 0:
            fallback = fetch_team_market_value_via_anysearch(team_name, league)
            if fallback.value_eur_m > 0:
                value = fallback
            elif fallback.error:
                value.error = "; ".join([item for item in (value.error, fallback.error) if item])
        team_cache[key] = value
        return value

    try:
        for issue_text in issues:
            for match in list_matches_by_issue(issue_text):
                if limit and scanned >= limit:
                    return {
                        "scanned": scanned,
                        "updated": updated,
                        "skipped": skipped,
                        "failed": failed,
                    }
                analysis_row = get_match_analysis(match["match_id"])
                if analysis_row is None or not str(analysis_row["collected_at"] or "").strip():
                    skipped += 1
                    continue
                existing_summary = str(analysis_row["market_value_summary"] or "").strip()
                if existing_summary and "squad market value" in existing_summary:
                    skipped += 1
                    continue

                scanned += 1
                label = f"{match['issue']} {match['home_team']} vs {match['away_team']}"
                print(f"[{scanned}] market value backfill: {label}", flush=True)
                try:
                    league = str(match["league"] or "")
                    home_value = fetch_team(str(match["home_team"] or ""), league)
                    away_value = fetch_team(str(match["away_team"] or ""), league)
                    summary = build_market_value_summary(home_value, away_value).strip()
                    if not summary:
                        failed += 1
                        print(f"  miss: {'; '.join([home_value.error, away_value.error])}", flush=True)
                        continue

                    analysis = _row_dict(analysis_row)
                    analysis["market_value_summary"] = summary
                    sources = []
                    for item in (home_value, away_value):
                        if item.source_url:
                            sources.append(f"{item.source_label or 'MarketValue'}: {item.source_url}")
                    _merge_sources(analysis, sources)
                    remarks = [f"MarketValue backfill completed at {datetime.now():%Y-%m-%d %H:%M:%S}"]
                    remarks.extend(f"MarketValue: {err}" for err in (home_value.error, away_value.error) if err)
                    _append_remarks(analysis, remarks)
                    if not dry_run:
                        save_analysis(analysis)
                    updated += 1
                    print("  updated", flush=True)
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    print(f"  failed: {exc}", flush=True)
    finally:
        if browser is not None:
            browser.close()

    return {
        "scanned": scanned,
        "updated": updated,
        "skipped": skipped,
        "failed": failed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill team squad market values into analyses.")
    parser.add_argument("--issue", default="", help="Only process one issue.")
    parser.add_argument("--limit", type=int, default=0, help="Max missing rows to process.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(backfill_market_values(issue=args.issue, limit=args.limit, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
