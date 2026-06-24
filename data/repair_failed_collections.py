#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Repair failed collections for a given issue.

Re-runs collection only for matches that have:
  - placeholder values (e.g., "未在公开来源中命中预计首发")
  - empty required fields
  - collection_status == 'failed'

Usage:
    python data/repair_failed_collections.py [--issue 26079] [--dry-run]
"""

import argparse
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from collection_repository import (
    COLLECTION_FAILURE_PREFIX,
    get_collection_stats,
    init_db,
    list_matches_by_issue,
    get_match_analysis,
)
from collection_service import (
    COLLECTION_FAILURE_PREFIX,
    MISSING_DIMENSION_PREFIX,
    COLLECTION_PLACEHOLDER_TEXTS,
    collect_match,
    get_collection_failure_reason,
    get_missing_required_dimensions,
    has_required_dimension_payload,
    format_missing_required_dimensions,
    REQUIRED_DIMENSION_TITLES,
    FIELD_GROUPS,
    FIELD_LABELS,
)


def required_dimension_fields():
    fields = []
    for title, group_fields in FIELD_GROUPS:
        if title not in REQUIRED_DIMENSION_TITLES:
            continue
        for field in group_fields:
            fields.append((title, field, FIELD_LABELS.get(field, field)))
    return fields


def is_failed_analysis(analysis_row) -> bool:
    """Check if an analysis row represents a failed/incomplete collection."""
    if analysis_row is None:
        return True
    if not (analysis_row["collected_at"] or "").strip():
        return True
    # Check explicit failure status (collection_status may not exist in old schemas)
    try:
        status = (analysis_row["collection_status"] or "").strip()
        if status == "failed":
            return True
    except (KeyError, IndexError):
        pass
    # Check failure remarks (covers both COLLECTION_FAILURE_PREFIX and MISSING_DIMENSION_PREFIX)
    remarks = analysis_row["remarks"] or ""
    if remarks.startswith(COLLECTION_FAILURE_PREFIX):
        return True
    # Check for empty required fields or placeholder values
    for _title, field, _label in required_dimension_fields():
        val = (analysis_row[field] or "").strip()
        if not val:
            return True
        if any(tok in val for tok in COLLECTION_PLACEHOLDER_TEXTS):
            return True
    return False


def repair_issue(issue: str, dry_run: bool = False) -> dict:
    """Repair all failed matches in an issue."""
    init_db()

    match_rows = list_matches_by_issue(issue)
    total = len(match_rows)

    failed_match_ids = []
    for row in match_rows:
        analysis = get_match_analysis(row["match_id"])
        if is_failed_analysis(analysis):
            failed_match_ids.append(row["match_id"])

    print(f"Issue {issue}: {total} matches total, {len(failed_match_ids)} need repair")

    if dry_run:
        for mid in failed_match_ids:
            analysis = get_match_analysis(mid)
            reason = get_collection_failure_reason(analysis) if analysis else "No analysis"
            print(f"  Would repair: {mid} - {reason}")
        return {"total": total, "failed": len(failed_match_ids), "repaired": 0, "still_failed": 0}

    repaired = 0
    still_failed = 0
    errors = []

    for i, mid in enumerate(failed_match_ids, 1):
        print(f"  [{i}/{len(failed_match_ids)}] Repairing match_id={mid}...")
        try:
            result = collect_match(mid)
            failure_reason = get_collection_failure_reason(result)
            if failure_reason:
                still_failed += 1
                print(f"    STILL FAILED: {failure_reason}")
                errors.append(f"{mid}: {failure_reason}")
            else:
                repaired += 1
                print(f"    REPAIRED successfully")
        except Exception as exc:
            still_failed += 1
            print(f"    ERROR: {exc}")
            errors.append(f"{mid}: {exc}")
        # Small delay between matches to be gentle on external sites
        if i < len(failed_match_ids):
            time.sleep(1)

    print(f"\nRepair complete: {repaired} repaired, {still_failed} still failed")
    if errors:
        print("Errors:")
        for err in errors:
            print(f"  - {err}")

    return {
        "total": total,
        "failed": len(failed_match_ids),
        "repaired": repaired,
        "still_failed": still_failed,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description="Repair failed collections")
    parser.add_argument("--issue", default="26079", help="Issue number to repair")
    parser.add_argument("--dry-run", action="store_true", help="Only show what would be repaired")
    args = parser.parse_args()

    result = repair_issue(args.issue, dry_run=args.dry_run)

    stats = get_collection_stats(args.issue)
    print(f"\nCollection stats for {args.issue}:")
    print(f"  Total matches: {stats['total_matches']}")
    print(f"  Success: {stats['success_analyses']}")
    print(f"  Failed: {stats['failed_analyses']}")


if __name__ == "__main__":
    main()
