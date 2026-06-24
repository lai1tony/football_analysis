import os

from source_500_client import fetch_current_matches


def main() -> None:
    os.environ["FOOTBALL_SCRAPER_BACKEND"] = "playwright-cli"
    os.environ.setdefault("PLAYWRIGHT_CLI_HEADED", "1")

    matches = fetch_current_matches()
    print(f"backend=playwright-cli rows={len(matches)}")
    for index, match in enumerate(matches, 1):
        print(
            index,
            match["league"],
            match["match_time"],
            match["home_team"],
            "vs",
            match["away_team"],
            "match_id=",
            match["match_id"],
        )


if __name__ == "__main__":
    main()
