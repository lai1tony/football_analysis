from collector_store import (
    collect_all_matches,
    get_collection_stats,
    init_db,
    sync_matches,
)


def main() -> None:
    init_db()
    sync_result = sync_matches(return_details=True)
    matches = sync_result["matches"]
    print(f"[1/3] 已同步对赛: {len(matches)} 场")
    print(f"      {sync_result['status_message']}")

    collection_result = collect_all_matches(return_details=True)
    print(f"[2/3] 已执行采集: {collection_result['total_matches']} 场")
    print(f"      {collection_result['status_message']}")
    for entry in collection_result.get("failed_matches", [])[:10]:
        print(f"      - {entry['match_label']}: {entry['reason']}")

    stats = get_collection_stats()
    print("[3/3] 采集统计")
    print(f"  - 总场次: {stats['total_matches']}")
    print(f"  - 已采集记录: {stats['total_analyses']}")
    print(f"  - 成功场次: {stats['success_analyses']}")
    print(f"  - 失败场次: {stats['failed_analyses']}")


if __name__ == "__main__":
    main()
