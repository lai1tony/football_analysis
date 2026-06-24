from __future__ import annotations

import argparse
from datetime import datetime

from collection_repository import (
    get_match_analysis,
    init_db,
    list_issues,
    list_matches_by_issue,
    save_analysis,
    serialize_match,
)
from collection_strategy import DIMENSION_FIELDS, apply_unified_collection_strategy


QUALITY_SUMMARY_PREFIXES = tuple(f"{dimension}." for dimension in DIMENSION_FIELDS)


def has_complete_quality_summary(value: str) -> bool:
    lines = [line.strip() for line in str(value or "").splitlines() if line.strip()]
    if not any(line.startswith("strategy:") for line in lines):
        return False
    seen_fields = {
        line.split(":", 1)[0].split(".", 1)[1]
        for line in lines
        if line.startswith(QUALITY_SUMMARY_PREFIXES) and "." in line.split(":", 1)[0]
    }
    required_fields = {field for fields in DIMENSION_FIELDS.values() for field in fields}
    return required_fields.issubset(seen_fields)


def backfill_issue(issue: str, *, dry_run: bool = False, force: bool = False) -> dict[str, int]:
    required_fields = [field for fields in DIMENSION_FIELDS.values() for field in fields]
    updated = 0
    skipped = 0
    for match in list_matches_by_issue(issue):
        row = get_match_analysis(match["match_id"])
        if row is None:
            skipped += 1
            continue
        analysis = dict(row)
        if not analysis.get("collected_at"):
            skipped += 1
            continue
        before = str(analysis.get("collection_quality_summary", "") or "")
        if before and not force and has_complete_quality_summary(before):
            skipped += 1
            continue
        apply_unified_collection_strategy(
            serialize_match(match),
            analysis,
            required_fields=required_fields,
            missing_fields=[],
            allow_fallback=False,
        )
        if analysis.get("collection_quality_summary") == before:
            skipped += 1
            continue
        analysis["collected_at"] = analysis.get("collected_at") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        updated += 1
        if not dry_run:
            save_analysis(analysis)
    return {"updated": updated, "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill unified collection quality summaries.")
    parser.add_argument("--issue", default="", help="Issue to backfill. Defaults to all retained issues.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rewrite existing quality summaries instead of only filling incomplete rows.",
    )
    args = parser.parse_args()

    init_db()
    issues = [args.issue] if args.issue else list_issues()
    total_updated = 0
    total_skipped = 0
    for issue in issues:
        stats = backfill_issue(str(issue), dry_run=args.dry_run, force=args.force)
        total_updated += stats["updated"]
        total_skipped += stats["skipped"]
        print(f"{issue}: updated={stats['updated']} skipped={stats['skipped']}")
    print(f"total: updated={total_updated} skipped={total_skipped}")


if __name__ == "__main__":
    main()
