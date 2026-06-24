# Agent Notes

## Current Reality
- The root `README.md` has been updated to describe the current Python + Flask + SQLite prototype.
- The executable code in this repo is the Python data and prediction module under `data/`.
- Root-level frontend artifacts such as `package.json`, `.next/`, or Playwright folders may exist as residue from older work, but they are not the current application entry point.
- There is no verified repo-wide test, lint, formatter, or typecheck config in this checkout.

## Entry Points
- Web review UI: `python data/app.py`
- Batch collection pipeline: `python data/run_full_pipeline.py`
- Quick scraper probes: `python data/list_current_matches.py` and `python data/collect_current_first_match.py`
- Do not switch these to `python -m ...` without refactoring imports first; the files use local absolute imports like `from collector_store import ...` and `from collection_service import ...`.

## Architecture Boundaries
- `data/source_500_client.py`: only external scraping and HTML fetch/decode logic.
- `data/collection_repository.py`: SQLite schema and persistence. The DB path is hard-coded to `data/football_data.db` via `Path(__file__).resolve().parent`.
- `data/collection_service.py`: orchestration and HTML-to-fields parsing.
- `data/collection_strategy.py`: unified per-dimension fallback policy. It records field-level source stage and quality in `analyses.collection_quality_summary`.
- `data/source_market_value_client.py`: team squad market value collection, using Transfermarkt via Playwright first and AnySearch extraction of Transfermarkt pages as fallback.
- `data/source_lineup_client.py`: injury and expected-lineup supplemental collection. Flashscore is primary; RotoWire is a fallback for missing lineup/injury text, not a market-value source.
- `data/source_supplement_client.py` and `data/source_understat_client.py`: public-source supplemental collection for missing strength, form, H2H, lineup, injury, and xG-style signals.
- `data/collector_store.py` is a compatibility facade only. New code should import from `source_500_client.py`, `collection_repository.py`, or `collection_service.py` directly.

## Runtime Quirks
- Scraping targets live `500.com` pages: `https://trade.500.com/sfc/` plus `odds.500.com` detail pages.
- `source_500_client.fetch_html()` decodes responses as `gb18030` and retries 3 times with backoff. Keep that behavior unless you have a concrete decoding bug to fix.
- The HTTP session sets `trust_env = False`, so proxy-related behavior from the shell environment is intentionally ignored.
- Field collection follows the strategy `primary/existing -> playwright-cli -> anysearch` for the supported analysis dimensions. Missing external values should be recorded with source/quality notes, not silently invented.
- The Flask UI auto-runs `sync_matches()` on `GET /` when the database has no matches yet.
- Failed collections are persisted into `analyses.remarks` with the prefix `采集失败：`; collection stats rely on that exact prefix.
- `coverage_draw_rescue` is the current target batch production strategy. `predict_match()` applies it to unsettled latest runs from the same issue after a single run is saved, and `predict_issue()` applies it again after the full issue is predicted. It writes final `recommendation`, `recommended_outcome`, `suggested_stake_pct`, and `effective_*` fields with `effective_action_source='target_batch_strategy'`.
- Target batch strategy calculations must read the original single-match model baseline from `algo_recommendation`, `algo_recommended_outcome`, and `algo_suggested_stake_pct` when the current run was already written by `target_batch_strategy`; otherwise repeated application will feed on its own prior output.
- Post-kickoff predictions must not replace the pre-match canonical baseline. `save_prediction_run()` preserves existing runs when the new run is created after `matches.match_time`, and `apply_target_batch_strategy_to_issue()` skips runs that already have `feedback_logs`. Existing local rows generated before this guard may still reflect older behavior unless restored from backup.
- Historical learning replay backfill lives in `data/replay_backfill.py` and the UI route `POST /learning/replay-backfill`. It derives earlier missing issues, syncs matches, collects each match in a child process with timeout, predicts, settles, and marks feedback with `roi_source='replay_backfill'`; reruns resume already collected rows, and partial issues are not valid complete learning-loop samples.
- As of the 2026-06-20 local SQLite snapshot, learning profile #43 is active. It is a `handicap_bucket_table` target strategy for Asian handicap recommendations, trained/evaluated on real settled historical rows with target hit rate >= 70% and action share >= 60%. The full-history current-strategy replay has 1245 samples, 751 actions, 60.32% action share, and 72.17% hit rate. The homepage "historical feedback" figures are saved prediction feedback from mixed older profiles and must not be read as the active strategy replay.

## Data / Output Gotchas
- `export_review_workbook.py` is not driven by SQLite. It reads the CSV file `D:\football_analysis\data\football_match_collection.csv` and writes `D:\football_analysis\data\football_match_collection_review.xlsx`.
- Because those export paths are hard-coded Windows absolute paths, changing the repo location will break the export script unless you update it.

## Working Rules For This Repo
- Prefer verifying behavior from the Python scripts and code under `data/` over prose docs.
- If docs conflict with code, trust the code. In particular, the current repo state supports the Python data and prediction prototype under `data/`, not any stale frontend residue.
- Avoid reading values from `.env` or architecture docs that include credentials; use local environment files only when the task truly requires them.
- Public GitHub pushes must keep only source/docs/scripts plus sanitized templates. Do not commit real `.env`, SQLite databases, logs, virtual environments, build outputs, installers, or old history containing secrets.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **football_analysis** (3934 symbols, 6508 relationships, 200 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/football_analysis/context` | Codebase overview, check index freshness |
| `gitnexus://repo/football_analysis/clusters` | All functional areas |
| `gitnexus://repo/football_analysis/processes` | All execution flows |
| `gitnexus://repo/football_analysis/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
