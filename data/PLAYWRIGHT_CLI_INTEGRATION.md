# Playwright CLI Integration

This project supports `playwright-cli` as an optional browser-backed scraper and as the primary external supplement path inside the unified collection strategy.

## Current behavior

- Default backend is still `requests`.
- Browser scraping is opt-in via environment variables.
- For `https://trade.500.com/sfc/`, direct headless Playwright currently lands on a `403 Forbidden` page.
- The same page works with `playwright-cli` in headed mode.
- Per-field supplemental collection now follows `primary/existing -> playwright-cli -> anysearch`.
- Team squad market value collection uses Transfermarkt through Playwright first, then AnySearch when Playwright does not find a usable value.

That means Playwright CLI should be treated as a supplemental scraper for this repo, not the default 500.com list/detail fetcher.

## Files

- [playwright_cli_client.py](/D:/football_analysis/data/playwright_cli_client.py)
- [source_500_client.py](/D:/football_analysis/data/source_500_client.py)
- [collection_strategy.py](/D:/football_analysis/data/collection_strategy.py)
- [source_market_value_client.py](/D:/football_analysis/data/source_market_value_client.py)
- [list_current_matches_playwright.py](/D:/football_analysis/data/list_current_matches_playwright.py)

## Environment variables

- `FOOTBALL_SCRAPER_BACKEND=requests|playwright-cli`
- `PLAYWRIGHT_CLI_HEADED=1|0`
- `PLAYWRIGHT_CLI_BIN=<path to playwright-cli>`
- `PLAYWRIGHT_CLI_WAIT_MS=800`
- `PLAYWRIGHT_CLI_TIMEOUT_MS=120000`
- `PLAYWRIGHT_CLI_SESSION_PREFIX=football-analysis`

Recommended settings for 500.com:

```powershell
$env:FOOTBALL_SCRAPER_BACKEND = "playwright-cli"
$env:PLAYWRIGHT_CLI_HEADED = "1"
```

## Quick probe

```powershell
& "D:\football_analysis\data\.myenv\Scripts\python.exe" data\list_current_matches_playwright.py
```

## Full pipeline with browser backend

```powershell
$env:FOOTBALL_SCRAPER_BACKEND = "playwright-cli"
$env:PLAYWRIGHT_CLI_HEADED = "1"
& "D:\football_analysis\data\.myenv\Scripts\python.exe" data\run_full_pipeline.py
```

## Notes

- Headed mode will open a visible browser window.
- Browser-backed collection is slower than the current `requests` path.
- Keep `requests` as the default path for routine batch collection unless the target page needs browser execution or anti-bot mitigation.
