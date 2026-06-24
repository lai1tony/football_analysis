import argparse
import sqlite3

from collection_repository import get_connection
from collection_service import collect_match


PLACEHOLDER_ELO = "页面未直接展示原始Elo值"


def list_match_ids_needing_elo_refresh(limit: int | None = None) -> list[str]:
    conn = get_connection()
    try:
        query = """
            SELECT match_id
            FROM analyses
            WHERE elo_home = ?
               OR elo_away = ?
               OR TRIM(COALESCE(elo_home, '')) = ''
               OR TRIM(COALESCE(elo_away, '')) = ''
            ORDER BY collected_at DESC, match_id DESC
        """
        params: tuple[object, ...] = (PLACEHOLDER_ELO, PLACEHOLDER_ELO)
        if limit is not None:
            query += " LIMIT ?"
            params = (PLACEHOLDER_ELO, PLACEHOLDER_ELO, limit)
        rows = conn.execute(query, params).fetchall()
        return [row["match_id"] for row in rows]
    finally:
        conn.close()


def repair_elo_snapshots(match_ids: list[str]) -> tuple[int, int]:
    repaired = 0
    failed = 0

    for index, match_id in enumerate(match_ids, 1):
        try:
            analysis = collect_match(match_id)
            print(
                f"[{index}/{len(match_ids)}] repaired {match_id}: "
                f"{analysis['elo_home']} | {analysis['elo_away']}"
            )
            repaired += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[{index}/{len(match_ids)}] failed {match_id}: {exc}")
            failed += 1

    return repaired, failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill stale dimension-one Elo snapshots in analyses."
    )
    parser.add_argument("--match-id", help="Repair a single match_id only.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Repair at most N stale rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.match_id:
        match_ids = [args.match_id]
    else:
        match_ids = list_match_ids_needing_elo_refresh(limit=args.limit)

    if not match_ids:
        print("No stale Elo snapshots found.")
        return

    repaired, failed = repair_elo_snapshots(match_ids)
    print(f"Finished. repaired={repaired} failed={failed}")


if __name__ == "__main__":
    main()
