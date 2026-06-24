"""Understat team-level xG client.

抓 https://understat.com/league/<LEAGUE>/<SEASON> 页面，从客户端渲染后的
``<div id="league-chemp">`` 表格里解析每支球队整赛季的 xG/xGA。

设计选择：
- 抓 *team* 维度，不抓单场。原因：team-level xG 已经对整赛季做了累积，
  足以表达"两队进攻防守强度"；match-level xG 抓取量大 5-10x 而对
  forecast 增益边际很小。
- **强制走 playwright-cli**。understat 在 2025 年改成了客户端渲染：
  requests 拿到的 HTML 里 ``<div id="league-chemp">`` 永远是空的，
  必须 JS 跑完才填数据。
- 缓存到 data/.cache/understat/<league>_<season>.json，TTL 12h。
- 队名映射放在 team_name_aliases.py，本文件不做匹配。

使用：
    from source_understat_client import (
        load_league_xg,
        get_team_xg_metrics,
        UNDERSTAT_LEAGUES,
    )
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup


BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / ".cache" / "understat"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL = timedelta(hours=12)  # team xG 一天最多更新一次，缓存 12 小时

UNDERSTAT_LEAGUES = {
    "EPL": {"slug": "EPL", "label": "英超"},
    "La_liga": {"slug": "La_liga", "label": "西甲"},
    "Serie_A": {"slug": "Serie_A", "label": "意甲"},
    "Bundesliga": {"slug": "Bundesliga", "label": "德甲"},
    "Ligue_1": {"slug": "Ligue_1", "label": "法甲"},
    "RFPL": {"slug": "RFPL", "label": "俄超"},
}


def _current_season() -> str:
    """欧洲赛季约定：2025-26 赛季 -> '2025'（前一年）。"""

    now = datetime.now()
    year = now.year
    # 7 月之后归本年赛季；之前归上年赛季。
    return str(year if now.month >= 7 else year - 1)


def _cache_path(league: str, season: str) -> Path:
    return CACHE_DIR / f"{league}_{season}.json"


def _read_cache(league: str, season: str) -> dict[str, Any] | None:
    path = _cache_path(league, season)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    fetched_at_text = str(payload.get("fetched_at", "") or "")
    try:
        fetched_at = datetime.strptime(fetched_at_text, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    if datetime.now() - fetched_at > CACHE_TTL:
        return None
    return payload


def _write_cache(league: str, season: str, payload: dict[str, Any]) -> None:
    path = _cache_path(league, season)
    payload["fetched_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _fetch_league_html(league: str, season: str) -> str:
    """通过 playwright-cli 渲染抓 understat 联赛页面。

    understat 在 2025 改版后必须走 playwright，否则 #league-chemp 表格
    没有 JS 渲染、抓不到任何数据。我们仍然保留 requests 作为快速探测，
    但只用于检测站点本身是否还活着——遇到非 200 / 404 等再抛错。
    """

    url = f"https://understat.com/league/{league}/{season}"
    last_error: Exception | None = None
    # playwright 单次开销大约 5-10s，做最多 2 次尝试避免偶发卡住。
    for attempt in range(2):
        try:
            return _fetch_via_playwright(url)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < 1:
                time.sleep(2.0)
    raise RuntimeError(f"understat playwright 抓取失败 {url}: {last_error}")


def _fetch_via_playwright(url: str) -> str:
    """通过本地 playwright-cli 渲染抓 understat。

    与 source_500_client 共享 playwright_cli_client 模块；只要项目已经为
    500.com 配置过 playwright-cli（PLAYWRIGHT_CLI_BIN 等），这里直接可用。
    """

    try:
        from playwright_cli_client import fetch_html_via_playwright_cli
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"playwright-cli 不可用：{exc}")
    return fetch_html_via_playwright_cli(url)


_TEAM_LINK_RE = re.compile(r"team/([^/]+)/(\d+)")


def _to_float(text: str) -> float:
    """把 '72.35' 或 '72.35+4.35' 之类的 cell 文本转成 float。"""

    if not text:
        return 0.0
    match = re.match(r"\s*(-?\d+(?:\.\d+)?)", text)
    if not match:
        return 0.0
    try:
        return float(match.group(1))
    except ValueError:
        return 0.0


def _to_int(text: str) -> int:
    if not text:
        return 0
    match = re.match(r"\s*(-?\d+)", text)
    if not match:
        return 0
    try:
        return int(match.group(1))
    except ValueError:
        return 0


def _parse_understat_table(html: str) -> dict[str, dict[str, Any]]:
    """从客户端渲染后的 #league-chemp 表格提取 team-level metrics。

    表头顺序（understat 2025 现行）：
      0: №      1: Team    2: M    3: W   4: D   5: L
      6: G      7: GA      8: PTS  9: xG  10: xGA 11: xPTS
    某些球队 xG/xGA 单元格内带有 ``<sup>`` 偏差标记，先文字化再 _to_float
    会自动去掉。

    Returns: {team_title_lower: aggregated_metrics}
    """

    soup = BeautifulSoup(html, "html.parser")
    container = soup.find("div", id="league-chemp")
    if container is None:
        return {}
    table = container.find("table")
    if table is None:
        return {}
    tbody = table.find("tbody")
    if tbody is None:
        return {}

    teams: dict[str, dict[str, Any]] = {}
    for tr in tbody.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) < 12:
            continue

        # team name from anchor href: team/Arsenal/2025 → Arsenal
        team_anchor = cells[1].find("a")
        if team_anchor is None:
            continue
        title_raw = team_anchor.get_text(strip=True)
        href_match = _TEAM_LINK_RE.search(team_anchor.get("href", ""))
        url_slug = href_match.group(1) if href_match else title_raw
        if not title_raw:
            continue

        matches = _to_int(cells[2].get_text(strip=True))
        if matches <= 0:
            continue
        wins = _to_int(cells[3].get_text(strip=True))
        draws = _to_int(cells[4].get_text(strip=True))
        loses = _to_int(cells[5].get_text(strip=True))
        goals_for = _to_int(cells[6].get_text(strip=True))
        goals_against = _to_int(cells[7].get_text(strip=True))
        points = _to_int(cells[8].get_text(strip=True))
        xg_total = _to_float(cells[9].get_text(strip=True))
        xga_total = _to_float(cells[10].get_text(strip=True))
        xpts_total = _to_float(cells[11].get_text(strip=True))

        metrics = {
            "team_id": url_slug,
            "title": title_raw,
            "matches": matches,
            "wins": wins,
            "draws": draws,
            "loses": loses,
            "goals_for": goals_for,
            "goals_against": goals_against,
            "points": points,
            "xg_total": round(xg_total, 4),
            "xga_total": round(xga_total, 4),
            "xpts_total": round(xpts_total, 4),
            "xg_per_game": round(xg_total / matches, 4),
            "xga_per_game": round(xga_total / matches, 4),
            "xpts_per_game": round(xpts_total / matches, 4),
            "win_rate": round(wins / matches, 4),
            # PPDA / deep-completion are no longer rendered by default in
            # the league summary table; leave them as 0 so feature_engine
            # callers that read these keys still get safe defaults.
            "ppda_for": 0.0,
            "ppda_against": 0.0,
            "deep_per_game": 0.0,
            "deep_allowed_per_game": 0.0,
        }
        teams[title_raw.lower()] = metrics

    return teams


def load_league_xg(league: str, season: str | None = None, *, force_refresh: bool = False) -> dict[str, Any]:
    """返回单个联赛的 team xG 字典：{team_title_lower: metrics}。

    缓存优先，过期或 force_refresh 时回源 understat。
    """

    season = str(season or _current_season())
    if league not in UNDERSTAT_LEAGUES:
        raise RuntimeError(f"不支持的 understat 联赛: {league}")

    if not force_refresh:
        cached = _read_cache(league, season)
        if cached is not None:
            return cached

    html = _fetch_league_html(league, season)
    teams = _parse_understat_table(html)
    if not teams:
        # 把抓到的 HTML 写到 cache 目录便于调试
        debug_path = CACHE_DIR / f"{league}_{season}.empty.html"
        try:
            debug_path.write_text(html, encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
        raise RuntimeError(
            f"understat 解析空：{league} 赛季 {season} 表格未渲染或结构改变。"
            f"原始 HTML 已写到 {debug_path}"
        )

    payload = {
        "league": league,
        "league_label": UNDERSTAT_LEAGUES[league]["label"],
        "season": season,
        "teams": teams,
    }
    _write_cache(league, season, payload)
    return payload


def load_all_leagues(season: str | None = None, *, force_refresh: bool = False) -> dict[str, dict[str, Any]]:
    """一次性把六个联赛全部抓回来。"""

    season = str(season or _current_season())
    out: dict[str, dict[str, Any]] = {}
    for league in UNDERSTAT_LEAGUES:
        try:
            out[league] = load_league_xg(league, season, force_refresh=force_refresh)
        except Exception as exc:  # noqa: BLE001
            out[league] = {"league": league, "season": season, "teams": {}, "error": str(exc)}
    return out


def lookup_team_in_league(
    league_payload: dict[str, Any],
    candidate_titles: list[str],
) -> dict[str, Any] | None:
    """在已加载的联赛字典中按候选名称（含别名）查找球队。"""

    teams: dict[str, Any] = league_payload.get("teams") or {}
    if not teams:
        return None
    for title in candidate_titles:
        norm = (title or "").strip().lower()
        if not norm:
            continue
        if norm in teams:
            return teams[norm]
        # 模糊：前缀匹配（"Real Madrid" vs "Real Madrid CF"）
        for stored_title, metrics in teams.items():
            if stored_title.startswith(norm) or norm.startswith(stored_title):
                return metrics
    return None


__all__ = [
    "UNDERSTAT_LEAGUES",
    "load_all_leagues",
    "load_league_xg",
    "lookup_team_in_league",
]
