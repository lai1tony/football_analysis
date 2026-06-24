from __future__ import annotations

import json
import math
import re
from datetime import datetime, timedelta
from typing import Any, Mapping


OUTCOMES = ("home", "draw", "away")
# Cached lookup of understat league xG payload, keyed by league code +
# season. We do not refresh per match — populated once per process via
# build_xg_lookup() / extract_xg_metrics().
_XG_LEAGUE_CACHE: dict[str, dict[str, Any]] = {}


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def infer_match_datetime(match_time: str, *, now: datetime | None = None) -> datetime | None:
    cleaned = normalize_text(match_time)
    if not cleaned:
        return None

    current = now or datetime.now()
    for pattern in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%m-%d %H:%M",
        "%y-%m-%d",
        "%Y-%m-%d",
    ):
        try:
            parsed = datetime.strptime(cleaned, pattern)
        except ValueError:
            continue
        if pattern == "%m-%d %H:%M":
            candidate = parsed.replace(year=current.year)
            if candidate - current > timedelta(days=200):
                candidate = candidate.replace(year=current.year - 1)
            elif current - candidate > timedelta(days=200):
                candidate = candidate.replace(year=current.year + 1)
            return candidate
        if pattern == "%y-%m-%d":
            return parsed
        return parsed
    return None


def poisson_probability(lmbda: float, goals: int) -> float:
    return math.exp(-lmbda) * (lmbda**goals) / math.factorial(goals)


def normalize_probs(home: float, draw: float, away: float) -> dict[str, float]:
    values = [max(home, 0.001), max(draw, 0.001), max(away, 0.001)]
    total = sum(values)
    return {
        "home": values[0] / total,
        "draw": values[1] / total,
        "away": values[2] / total,
    }


def market_implied_probs(odds: Mapping[str, float]) -> dict[str, float]:
    home = 1 / odds["home"] if odds["home"] > 0 else 0.33
    draw = 1 / odds["draw"] if odds["draw"] > 0 else 0.33
    away = 1 / odds["away"] if odds["away"] > 0 else 0.33
    return normalize_probs(home, draw, away)


def extract_market_odds(text: str, fallback_row: Mapping[str, Any]) -> dict[str, float]:
    match = re.search(
        r"(?:初赔|列表均赔)\s*(\d+\.\d+)\/(\d+\.\d+)\/(\d+\.\d+)(?:\s*->\s*即时\s*(\d+\.\d+)\/(\d+\.\d+)\/(\d+\.\d+))?",
        text or "",
    )
    if match:
        groups = match.groups()
        if groups[3] and groups[4] and groups[5]:
            return {
                "home": float(groups[3]),
                "draw": float(groups[4]),
                "away": float(groups[5]),
            }
        return {
            "home": float(groups[0]),
            "draw": float(groups[1]),
            "away": float(groups[2]),
        }
    return {
        "home": safe_float(fallback_row.get("list_odds_win")),
        "draw": safe_float(fallback_row.get("list_odds_draw")),
        "away": safe_float(fallback_row.get("list_odds_loss")),
    }


def _handicap_line_to_float(value: str) -> float:
    text = normalize_text(str(value or ""))
    if not text:
        return 0.0
    numeric = re.search(r"[+-]?\d+(?:\.\d+)?", text)
    if numeric:
        return safe_float(numeric.group(0))
    sign = -1.0 if text.startswith("-") or "受" in text else 1.0
    normalized = text.lstrip("+-")
    mapping = {
        "平手": 0.0,
        "平": 0.0,
        "平/半": 0.25,
        "平半": 0.25,
        "半球": 0.5,
        "半": 0.5,
        "半/一": 0.75,
        "半一": 0.75,
        "一球": 1.0,
        "一": 1.0,
        "一/球半": 1.25,
        "一/一半": 1.25,
        "一球/球半": 1.25,
        "球半": 1.5,
        "球半/两": 1.75,
        "球半/两球": 1.75,
        "两球": 2.0,
        "两": 2.0,
        "两/两半": 2.25,
        "两球/两球半": 2.25,
        "两半": 2.5,
        "两球半": 2.5,
        "两半/三": 2.75,
        "三球": 3.0,
        "三": 3.0,
    }
    for key, line in sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True):
        if key in normalized:
            return sign * line
    return 0.0


def extract_asian_handicap(summary: str) -> dict[str, float]:
    empty = {
        "initial_home_odds": 0.0,
        "initial_line": 0.0,
        "initial_away_odds": 0.0,
        "current_home_odds": 0.0,
        "current_line": 0.0,
        "current_away_odds": 0.0,
    }
    text = normalize_text(summary)
    match = re.search(
        r"初盘\s*([0-9.]+)\s*/\s*([^/]*?)\s*/\s*([0-9.]+)\s*->\s*即时\s*([0-9.]+)\s*/\s*([^/]*?)\s*/\s*([0-9.]+)",
        text,
    )
    if not match:
        return empty
    init_home, init_line, init_away, cur_home, cur_line, cur_away = match.groups()
    return {
        "initial_home_odds": safe_float(init_home),
        "initial_line": _handicap_line_to_float(init_line),
        "initial_away_odds": safe_float(init_away),
        "current_home_odds": safe_float(cur_home),
        "current_line": _handicap_line_to_float(cur_line),
        "current_away_odds": safe_float(cur_away),
    }


def extract_record_metrics(text: str) -> dict[str, float | int]:
    empty = {
        "matches": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "goals_for": 0,
        "goals_against": 0,
        "win_rate": 0.0,
        "handicap_rate": 0.0,
        "over_rate": 0.0,
        "points_per_game": 0.0,
        "goal_diff_per_game": 0.0,
        "goals_for_per_game": 0.0,
        "goals_against_per_game": 0.0,
    }
    if not text:
        return empty

    match = re.search(
        r"近(\d+)场\s*(\d+)胜(\d+)平(\d+)负[，,]\s*进(\d+)球失(\d+)球[，,]\s*胜率(\d+)%[，,]\s*赢盘率(\d+)%[，,]\s*大球率(\d+)%",
        normalize_text(text),
    )
    if not match:
        return empty

    matches, wins, draws, losses, goals_for, goals_against, win_rate, handicap_rate, over_rate = [
        int(item) for item in match.groups()
    ]
    games = max(matches, 1)
    return {
        "matches": matches,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "win_rate": win_rate / 100.0,
        "handicap_rate": handicap_rate / 100.0,
        "over_rate": over_rate / 100.0,
        "points_per_game": (wins * 3 + draws) / games,
        "goal_diff_per_game": (goals_for - goals_against) / games,
        "goals_for_per_game": goals_for / games,
        "goals_against_per_game": goals_against / games,
    }


def extract_home_away_metrics(text: str) -> dict[str, float | int]:
    empty = {
        "home_wins": 0,
        "home_draws": 0,
        "home_losses": 0,
        "home_gf": 0,
        "home_ga": 0,
        "home_ppg": 0.0,
        "home_goal_diff_pg": 0.0,
        "away_wins": 0,
        "away_draws": 0,
        "away_losses": 0,
        "away_gf": 0,
        "away_ga": 0,
        "away_ppg": 0.0,
        "away_goal_diff_pg": 0.0,
    }
    if not text:
        return empty

    match = re.search(
        r"主场\s*(\d+)胜(\d+)平(\d+)负[，,]\s*进(\d+)失(\d+)\s*\|\s*客场\s*(\d+)胜(\d+)平(\d+)负[，,]\s*进(\d+)失(\d+)",
        normalize_text(text),
    )
    if not match:
        return empty

    values = [int(item) for item in match.groups()]
    home_matches = max(sum(values[0:3]), 1)
    away_matches = max(sum(values[5:8]), 1)
    return {
        "home_wins": values[0],
        "home_draws": values[1],
        "home_losses": values[2],
        "home_gf": values[3],
        "home_ga": values[4],
        "home_ppg": (values[0] * 3 + values[1]) / home_matches,
        "home_goal_diff_pg": (values[3] - values[4]) / home_matches,
        "away_wins": values[5],
        "away_draws": values[6],
        "away_losses": values[7],
        "away_gf": values[8],
        "away_ga": values[9],
        "away_ppg": (values[5] * 3 + values[6]) / away_matches,
        "away_goal_diff_pg": (values[8] - values[9]) / away_matches,
    }


def extract_h2h_metrics(text: str) -> dict[str, float | int]:
    empty = {
        "matches": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "goals_for": 0,
        "goals_against": 0,
        "edge": 0.0,
    }
    if not text:
        return empty

    match = re.search(
        r"近(\d+)次交锋\s*(\d+)胜(\d+)平(\d+)负[，,]\s*进(\d+)球失(\d+)球",
        normalize_text(text),
    )
    if not match:
        return empty

    matches, wins, draws, losses, goals_for, goals_against = [
        int(item) for item in match.groups()
    ]
    return {
        "matches": matches,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "edge": (wins - losses) / max(matches, 1),
    }


def extract_strength_metrics(text: str) -> dict[str, float | int]:
    empty = {
        "rank": 0,
        "team_count": 0,
        "points": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "goals_for": 0,
        "goals_against": 0,
        "goal_diff": 0,
        "rating": 1500.0,
    }
    if not text:
        return empty

    match = re.search(
        r"第(\d+)(?:/(\d+))?[，,]\s*(\d+)分[，,]\s*(\d+)胜(\d+)平(\d+)负[，,]\s*进(\d+)[，,]\s*失(\d+)[，,]\s*净胜(-?\d+)",
        normalize_text(text),
    )
    if not match:
        return empty

    rank, team_count, points, wins, draws, losses, goals_for, goals_against, goal_diff = match.groups()
    rank_i = int(rank)
    team_count_i = int(team_count or 0)
    points_i = int(points)
    goal_diff_i = int(goal_diff)
    wins_i = int(wins)
    draws_i = int(draws)
    losses_i = int(losses)
    goals_for_i = int(goals_for)
    goals_against_i = int(goals_against)

    rank_factor = 0.0
    if team_count_i > 1:
        rank_factor = (0.5 - ((rank_i - 1) / (team_count_i - 1))) * 180.0

    rating = 1450.0 + rank_factor + points_i * 2.4 + goal_diff_i * 3.0 + wins_i * 1.5
    rating = clamp(rating, 1325.0, 1875.0)

    return {
        "rank": rank_i,
        "team_count": team_count_i,
        "points": points_i,
        "wins": wins_i,
        "draws": draws_i,
        "losses": losses_i,
        "goals_for": goals_for_i,
        "goals_against": goals_against_i,
        "goal_diff": goal_diff_i,
        "rating": rating,
    }


def extract_market_value_metrics(text: str) -> dict[str, float | int]:
    empty = {
        "home_value_eur_m": 0.0,
        "away_value_eur_m": 0.0,
        "gap_eur_m": 0.0,
        "ratio": 0.0,
        "log_gap": 0.0,
        "coverage": 0,
    }
    if not text:
        return empty

    def _to_eur_m(amount: str, unit: str) -> float:
        try:
            number = float(str(amount).replace(",", "."))
        except (TypeError, ValueError):
            return 0.0
        unit_text = str(unit or "").lower()
        if unit_text in {"bn", "b", "billion"}:
            return number * 1000.0
        if unit_text in {"k", "th", "thousand"}:
            return number / 1000.0
        return number

    values = []
    value_pattern = re.compile(
        r"(?:EUR|€|\u20ac)\s*([0-9]+(?:[.,][0-9]+)?)\s*(bn|b|billion|m|mil|million|k|th|thousand)?",
        re.I,
    )
    for line in str(text).splitlines():
        value = 0.0
        match = value_pattern.search(line)
        if match:
            value = _to_eur_m(match.group(1), match.group(2) or "m")
        else:
            alt = re.search(
                r"([0-9]+(?:[.,][0-9]+)?)\s*(bn|b|billion|m|mil|million|k|th|thousand)\s*(?:EUR|€|\u20ac)",
                line,
                re.I,
            )
            if alt:
                value = _to_eur_m(alt.group(1), alt.group(2))
        if value > 0:
            values.append(value)
        if len(values) >= 2:
            break

    if len(values) < 2:
        return empty
    home_value, away_value = values[0], values[1]
    ratio = home_value / away_value if away_value > 0 else 0.0
    log_gap = math.log1p(home_value) - math.log1p(away_value)
    return {
        "home_value_eur_m": round(home_value, 3),
        "away_value_eur_m": round(away_value, 3),
        "gap_eur_m": round(home_value - away_value, 3),
        "ratio": round(ratio, 4),
        "log_gap": round(log_gap, 6),
        "coverage": 1,
    }


_LINEUP_PLACEHOLDER_TOKENS = (
    "未在公开来源中命中预计首发",
    "未在公开来源中命中明确伤停",
    "外部公开来源补采未命中",
    "伤停/阵容补采失败",
    "页面含预计阵容模块",
    "当前页面未列出明确伤停名单",
)


def _lineup_text_has_real_signal(text: str) -> bool:
    """Decide whether the textual lineup notes carry usable signal.

    Many records fall back to placeholder strings when the external lineup
    scrape (Flashscore / Soccerway / LiveScore) misses the match. The old
    code happily fed those placeholders through, ending up at the
    "neutral" availability default of 0.92 — i.e. it pretended every team
    had a healthy 92% squad even when we had no data at all. That is
    noise, not signal. We mark such rows as data_available=False so that
    quant/ml models can opt to *ignore* the lineup channel rather than
    average over a fake 0.92.
    """

    cleaned = normalize_text(text)
    if not cleaned:
        return False
    # If the text only contains placeholder phrases, treat as missing.
    if any(token in cleaned for token in _LINEUP_PLACEHOLDER_TOKENS):
        # The placeholder may co-exist with real "主队：..." block; only
        # treat as missing if there is no concrete lineup line.
        if "主队：" not in cleaned and "客队：" not in cleaned and "关键伤停：" not in cleaned:
            return False
    return True


def _parse_lineup_items(text: str) -> list[str]:
    if not text or "未在公开来源中命中预计首发" in text:
        return []
    return [normalize_text(item) for item in text.split("/") if normalize_text(item)]


def _parse_injury_items(text: str) -> list[str]:
    if not text or "未在公开来源中命中明确伤停" in text:
        return []
    return [normalize_text(item) for item in re.split(r"[；;]", text) if normalize_text(item)]


def extract_lineup_text_metrics(text: str) -> dict[str, float | int]:
    empty = {
        "home_starters": 0,
        "away_starters": 0,
        "home_absent_count": 0,
        "away_absent_count": 0,
        "home_doubtful_count": 0,
        "away_doubtful_count": 0,
        "home_absence_impact": 0.0,
        "away_absence_impact": 0.0,
        "home_availability": 0.92,
        "away_availability": 0.92,
        "data_available": 0,
    }
    if not _lineup_text_has_real_signal(text):
        return empty

    home_line = ""
    away_line = ""
    injury_line = ""
    for line in text.splitlines():
        clean_line = normalize_text(line)
        if clean_line.startswith("主队："):
            home_line = clean_line.split("：", 1)[1]
        elif clean_line.startswith("客队："):
            away_line = clean_line.split("：", 1)[1]
        elif clean_line.startswith("关键伤停："):
            injury_line = clean_line.split("：", 1)[1]

    starters_home = _parse_lineup_items(home_line)
    starters_away = _parse_lineup_items(away_line)

    home_injury_text = ""
    away_injury_text = ""
    injury_match = re.search(r"主队(.*?)(?:；|;)\s*客队(.*)", injury_line)
    if injury_match:
        home_injury_text = injury_match.group(1)
        away_injury_text = injury_match.group(2)

    home_absent_items = _parse_injury_items(home_injury_text)
    away_absent_items = _parse_injury_items(away_injury_text)
    home_doubtful_count = sum(1 for item in home_absent_items if "成疑" in item)
    away_doubtful_count = sum(1 for item in away_absent_items if "成疑" in item)
    home_absent_count = sum(1 for item in home_absent_items if "成疑" not in item)
    away_absent_count = sum(1 for item in away_absent_items if "成疑" not in item)

    home_absence_impact = clamp(
        home_absent_count * 0.06 + home_doubtful_count * 0.03 + max(0, 11 - len(starters_home)) * 0.015,
        0.0,
        0.36,
    )
    away_absence_impact = clamp(
        away_absent_count * 0.06 + away_doubtful_count * 0.03 + max(0, 11 - len(starters_away)) * 0.015,
        0.0,
        0.36,
    )

    # Mark as available only if at least one of the two teams has real
    # lineup or injury content. A row that only matched the regex header
    # but produced no real items is still effectively empty.
    has_real_payload = bool(
        starters_home or starters_away or home_absent_items or away_absent_items
    )

    return {
        "home_starters": len(starters_home),
        "away_starters": len(starters_away),
        "home_absent_count": home_absent_count,
        "away_absent_count": away_absent_count,
        "home_doubtful_count": home_doubtful_count,
        "away_doubtful_count": away_doubtful_count,
        "home_absence_impact": home_absence_impact,
        "away_absence_impact": away_absence_impact,
        "home_availability": clamp(1.0 - home_absence_impact, 0.58, 1.0),
        "away_availability": clamp(1.0 - away_absence_impact, 0.58, 1.0),
        "data_available": 1 if has_real_payload else 0,
    }


def build_lineup_metrics(lineup_data: Mapping[str, Any] | None, text: str = "") -> dict[str, float | int]:
    if lineup_data:
        home_starters = safe_int(lineup_data.get("home_starters"), len(lineup_data.get("home_lineup", []) or []))
        away_starters = safe_int(lineup_data.get("away_starters"), len(lineup_data.get("away_lineup", []) or []))
        home_absent_count = safe_int(
            lineup_data.get("home_absent_count"),
            sum(1 for item in (lineup_data.get("home_missing") or []) if str(item.get("status", "out")) != "doubtful"),
        )
        away_absent_count = safe_int(
            lineup_data.get("away_absent_count"),
            sum(1 for item in (lineup_data.get("away_missing") or []) if str(item.get("status", "out")) != "doubtful"),
        )
        home_doubtful_count = safe_int(
            lineup_data.get("home_doubtful_count"),
            sum(1 for item in (lineup_data.get("home_missing") or []) if str(item.get("status", "")) == "doubtful"),
        )
        away_doubtful_count = safe_int(
            lineup_data.get("away_doubtful_count"),
            sum(1 for item in (lineup_data.get("away_missing") or []) if str(item.get("status", "")) == "doubtful"),
        )
        metrics = {
            "home_starters": home_starters,
            "away_starters": away_starters,
            "home_absent_count": home_absent_count,
            "away_absent_count": away_absent_count,
            "home_doubtful_count": home_doubtful_count,
            "away_doubtful_count": away_doubtful_count,
            "home_absence_impact": safe_float(lineup_data.get("home_absence_impact"), 0.0),
            "away_absence_impact": safe_float(lineup_data.get("away_absence_impact"), 0.0),
        }
        if metrics["home_absence_impact"] <= 0:
            metrics["home_absence_impact"] = clamp(
                metrics["home_absent_count"] * 0.06
                + metrics["home_doubtful_count"] * 0.03
                + max(0, 11 - metrics["home_starters"]) * 0.015,
                0.0,
                0.36,
            )
        if metrics["away_absence_impact"] <= 0:
            metrics["away_absence_impact"] = clamp(
                metrics["away_absent_count"] * 0.06
                + metrics["away_doubtful_count"] * 0.03
                + max(0, 11 - metrics["away_starters"]) * 0.015,
                0.0,
                0.36,
            )
        metrics["home_availability"] = clamp(
            1.0 - metrics["home_absence_impact"], 0.58, 1.0
        )
        metrics["away_availability"] = clamp(
            1.0 - metrics["away_absence_impact"], 0.58, 1.0
        )
        # When supplemental scrape supplied at least one starter or injury,
        # treat lineup data as available even if the text channel was
        # placeholder-only.
        metrics["data_available"] = 1 if (
            home_starters > 0 or away_starters > 0
            or home_absent_count > 0 or away_absent_count > 0
            or home_doubtful_count > 0 or away_doubtful_count > 0
        ) else 0
        return metrics
    return extract_lineup_text_metrics(text)
    return extract_lineup_text_metrics(text)


def _extract_recent_dates_from_block(html: str, block_id: str) -> list[datetime]:
    match_time_regex = re.compile(rf'<div id="{re.escape(block_id)}".*?</table>', re.S)
    block_match = match_time_regex.search(html)
    if not block_match:
        return []

    block_html = block_match.group(0)
    dates = []
    for date_text in re.findall(r"<td>(\d{2}-\d{2}-\d{2})</td>", block_html):
        parsed = infer_match_datetime(date_text)
        if parsed is not None:
            dates.append(parsed)
    return dates


def extract_schedule_metrics(match_time: str, shuju_html: str) -> dict[str, float | int]:
    empty = {
        "home_rest_days": 0.0,
        "away_rest_days": 0.0,
        "home_load_14": 0,
        "away_load_14": 0,
        "rest_advantage": 0.0,
        "schedule_gap": 0.0,
    }
    if not shuju_html:
        return empty

    match_dt = infer_match_datetime(match_time)
    if match_dt is None:
        return empty

    home_dates = _extract_recent_dates_from_block(shuju_html, "team_zhanji1_1")
    away_dates = _extract_recent_dates_from_block(shuju_html, "team_zhanji1_0")

    def calc_rest_and_load(dates: list[datetime]) -> tuple[float, int]:
        prior_dates = [item for item in dates if item < match_dt]
        if not prior_dates:
            return 0.0, 0
        latest = max(prior_dates)
        load = sum(1 for item in prior_dates if 0 <= (match_dt - item).days <= 14)
        return float((match_dt - latest).days), load

    home_rest, home_load = calc_rest_and_load(home_dates)
    away_rest, away_load = calc_rest_and_load(away_dates)
    return {
        "home_rest_days": home_rest,
        "away_rest_days": away_rest,
        "home_load_14": home_load,
        "away_load_14": away_load,
        "rest_advantage": home_rest - away_rest,
        "schedule_gap": away_load - home_load,
    }


def extract_xg_metrics(
    home_team: str,
    away_team: str,
    league_label: str,
    *,
    season: str | None = None,
) -> dict[str, Any]:
    """Pull team-level xG for both clubs from understat.

    Returns a dict with per-game xG for/against for each team, plus a
    coverage flag. When the league is not supported by understat, or
    either team cannot be matched in the loaded league snapshot, returns
    coverage=0 with all-zero metrics so quant_model can fall back.

    The understat fetch is cached on disk (12h TTL) and in-process
    (_XG_LEAGUE_CACHE), so calling this in a tight loop over 14 matches
    only triggers at most one network round-trip per league.
    """

    empty = {
        "coverage": 0,
        "league": "",
        "season": "",
        "home_xg_per_game": 0.0,
        "away_xg_per_game": 0.0,
        "home_xga_per_game": 0.0,
        "away_xga_per_game": 0.0,
        "home_matches": 0,
        "away_matches": 0,
        "home_title": "",
        "away_title": "",
    }
    try:
        # Local imports to avoid hard dependency at module load time
        # (e.g. unit tests that don't hit the network).
        from source_understat_client import (
            UNDERSTAT_LEAGUES,
            load_league_xg,
            lookup_team_in_league,
        )
        from team_name_aliases import (
            league_codes_for_500_label,
            resolve_team_aliases,
        )
    except Exception:  # noqa: BLE001
        return empty

    candidate_codes = league_codes_for_500_label(league_label)
    if not candidate_codes:
        return empty

    home_aliases = resolve_team_aliases(home_team)
    away_aliases = resolve_team_aliases(away_team)
    if not home_aliases or not away_aliases:
        return empty

    for league_code in candidate_codes:
        if league_code not in UNDERSTAT_LEAGUES:
            continue
        cache_key = f"{league_code}:{season or ''}"
        payload = _XG_LEAGUE_CACHE.get(cache_key)
        if payload is None:
            try:
                payload = load_league_xg(league_code, season)
            except Exception:  # noqa: BLE001
                payload = {"teams": {}, "league": league_code}
            _XG_LEAGUE_CACHE[cache_key] = payload
        teams = (payload or {}).get("teams") or {}
        if not teams:
            continue
        home_metrics = lookup_team_in_league(payload, home_aliases)
        away_metrics = lookup_team_in_league(payload, away_aliases)
        if not home_metrics or not away_metrics:
            continue
        return {
            "coverage": 1,
            "league": league_code,
            "season": str(payload.get("season", "") or ""),
            "home_xg_per_game": safe_float(home_metrics.get("xg_per_game")),
            "away_xg_per_game": safe_float(away_metrics.get("xg_per_game")),
            "home_xga_per_game": safe_float(home_metrics.get("xga_per_game")),
            "away_xga_per_game": safe_float(away_metrics.get("xga_per_game")),
            "home_matches": safe_int(home_metrics.get("matches")),
            "away_matches": safe_int(away_metrics.get("matches")),
            "home_title": str(home_metrics.get("title", "") or ""),
            "away_title": str(away_metrics.get("title", "") or ""),
        }
    return empty


def build_feature_snapshot(
    row: Mapping[str, Any],
    *,
    shuju_html: str = "",
    lineup_data: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    recent_home = extract_record_metrics(str(row.get("recent_form_home", "") or ""))
    recent_away = extract_record_metrics(str(row.get("recent_form_away", "") or ""))
    split = extract_home_away_metrics(str(row.get("home_away_form", "") or ""))
    h2h = extract_h2h_metrics(str(row.get("head_to_head_summary", "") or ""))
    strength_home = extract_strength_metrics(str(row.get("elo_home", "") or ""))
    strength_away = extract_strength_metrics(str(row.get("elo_away", "") or ""))
    market_value = extract_market_value_metrics(str(row.get("market_value_summary", "") or ""))
    lineup = build_lineup_metrics(lineup_data, str(row.get("injury_or_lineup_notes", "") or ""))
    schedule = extract_schedule_metrics(str(row.get("match_time", "") or ""), shuju_html)
    market_odds = extract_market_odds(
        str(row.get("european_odds_movement_summary", "") or ""),
        row,
    )
    market_probs = market_implied_probs(market_odds)
    asian_handicap = extract_asian_handicap(str(row.get("asian_handicap_summary", "") or ""))
    xg_metrics = extract_xg_metrics(
        str(row.get("home_team", "") or ""),
        str(row.get("away_team", "") or ""),
        str(row.get("league", "") or ""),
    )

    snapshot = {
        "match_id": str(row.get("match_id", "") or ""),
        "issue": str(row.get("issue", "") or ""),
        "snapshot_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "home_rating": round(safe_float(strength_home["rating"]), 3),
        "away_rating": round(safe_float(strength_away["rating"]), 3),
        "recent_home_ppg": round(safe_float(recent_home["points_per_game"]), 4),
        "recent_away_ppg": round(safe_float(recent_away["points_per_game"]), 4),
        "recent_home_gf_pg": round(safe_float(recent_home["goals_for_per_game"]), 4),
        "recent_away_gf_pg": round(safe_float(recent_away["goals_for_per_game"]), 4),
        "recent_home_ga_pg": round(safe_float(recent_home["goals_against_per_game"]), 4),
        "recent_away_ga_pg": round(safe_float(recent_away["goals_against_per_game"]), 4),
        "home_split_ppg": round(safe_float(split["home_ppg"]), 4),
        "away_split_ppg": round(safe_float(split["away_ppg"]), 4),
        "home_absent_count": safe_int(lineup["home_absent_count"]),
        "away_absent_count": safe_int(lineup["away_absent_count"]),
        "home_doubtful_count": safe_int(lineup["home_doubtful_count"]),
        "away_doubtful_count": safe_int(lineup["away_doubtful_count"]),
        "home_absence_impact": round(safe_float(lineup["home_absence_impact"]), 4),
        "away_absence_impact": round(safe_float(lineup["away_absence_impact"]), 4),
        "lineup_home_availability": round(safe_float(lineup["home_availability"]), 4),
        "lineup_away_availability": round(safe_float(lineup["away_availability"]), 4),
        "rest_days_home": round(safe_float(schedule["home_rest_days"]), 2),
        "rest_days_away": round(safe_float(schedule["away_rest_days"]), 2),
        "schedule_load_home": safe_int(schedule["home_load_14"]),
        "schedule_load_away": safe_int(schedule["away_load_14"]),
        "h2h_edge": round(safe_float(h2h["edge"]), 4),
        "market_home_prob": round(market_probs["home"], 6),
        "market_draw_prob": round(market_probs["draw"], 6),
        "market_away_prob": round(market_probs["away"], 6),
    }
    snapshot["feature_payload"] = json.dumps(
        {
            "recent_home": recent_home,
            "recent_away": recent_away,
            "split": split,
            "h2h": h2h,
            "strength_home": strength_home,
            "strength_away": strength_away,
            "market_value": market_value,
            "lineup": lineup,
            "lineup_raw": dict(lineup_data or {}),
            "schedule": schedule,
            "market_odds": market_odds,
            "market_probs": market_probs,
            "asian_handicap": asian_handicap,
            "xg": xg_metrics,
        },
        ensure_ascii=False,
    )
    return snapshot


def _load_feature_payload(snapshot: Mapping[str, Any] | None) -> dict[str, Any]:
    if not snapshot:
        return {}
    payload_text = str(snapshot.get("feature_payload", "") or "")
    if not payload_text:
        return {}
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _expected_ppg_from_rank(rank: int, team_count: int) -> float:
    """Approximate the PPG a team at this league rank would produce on average.

    Treats each league as a linear gradient: rank 1 ~ 2.4 PPG, last place
    ~ 0.7 PPG. With 18-20 team leagues this maps roughly to 75 / 32 points
    per season at the extremes, which matches recent EPL/La Liga data.

    Used so that "PPG 2.0 by a 5th-place team" is recognised as roughly
    par for that team's class, while "PPG 2.0 by a 17th-place team" is
    a strong over-performance signal worth lifting.
    """

    if rank <= 0 or team_count <= 1:
        return 1.4
    rank_norm = (rank - 1) / (team_count - 1)  # 0 = top, 1 = bottom
    return clamp(2.4 - rank_norm * 1.7, 0.6, 2.5)


def _expected_goal_diff_pg_from_rank(rank: int, team_count: int) -> float:
    """Same idea for goal difference per game.

    Rank 1 ~ +1.5 GDpG, mid-table ~ 0, last place ~ -1.2 GDpG.
    """

    if rank <= 0 or team_count <= 1:
        return 0.0
    rank_norm = (rank - 1) / (team_count - 1)
    return clamp(1.5 - rank_norm * 2.7, -1.4, 1.6)


def opponent_adjusted_form(
    record: Mapping[str, Any],
    strength: Mapping[str, Any],
) -> dict[str, float]:
    """Form metrics relative to what the team's league rank would predict.

    Returns residuals (actual - expected); positive = team is over-
    performing its league position, negative = under-performing. These
    residuals carry more signal than raw PPG because they cancel league-
    quality bias: in胜负彩 we mix EPL / Bundesliga / lower-tier matches
    in the same coupon, and absolute PPG is comparable across leagues
    only after normalising by rank.
    """

    rank = safe_int(strength.get("rank"))
    team_count = safe_int(strength.get("team_count"))
    expected_ppg = _expected_ppg_from_rank(rank, team_count)
    expected_gd_pg = _expected_goal_diff_pg_from_rank(rank, team_count)
    actual_ppg = safe_float(record.get("points_per_game"))
    actual_gd_pg = safe_float(record.get("goal_diff_per_game"))
    return {
        "expected_ppg": expected_ppg,
        "expected_goal_diff_pg": expected_gd_pg,
        "ppg_residual": actual_ppg - expected_ppg,
        "goal_diff_residual": actual_gd_pg - expected_gd_pg,
    }


def build_match_features(
    row: Mapping[str, Any],
    snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _load_feature_payload(snapshot)
    recent_home = payload.get("recent_home") or extract_record_metrics(
        str(row.get("recent_form_home", "") or "")
    )
    recent_away = payload.get("recent_away") or extract_record_metrics(
        str(row.get("recent_form_away", "") or "")
    )
    split = payload.get("split") or extract_home_away_metrics(
        str(row.get("home_away_form", "") or "")
    )
    h2h = payload.get("h2h") or extract_h2h_metrics(
        str(row.get("head_to_head_summary", "") or "")
    )
    market_odds = payload.get("market_odds") or extract_market_odds(
        str(row.get("european_odds_movement_summary", "") or ""),
        row,
    )
    market_probs = payload.get("market_probs") or market_implied_probs(market_odds)
    asian_handicap = payload.get("asian_handicap") or extract_asian_handicap(
        str(row.get("asian_handicap_summary", "") or "")
    )
    lineup = payload.get("lineup") or build_lineup_metrics(
        None,
        str(row.get("injury_or_lineup_notes", "") or ""),
    )
    lineup_raw = payload.get("lineup_raw") or {}
    schedule = payload.get("schedule") or {
        "home_rest_days": safe_float(snapshot.get("rest_days_home")) if snapshot else 0.0,
        "away_rest_days": safe_float(snapshot.get("rest_days_away")) if snapshot else 0.0,
        "home_load_14": safe_int(snapshot.get("schedule_load_home")) if snapshot else 0,
        "away_load_14": safe_int(snapshot.get("schedule_load_away")) if snapshot else 0,
        "rest_advantage": (
            safe_float(snapshot.get("rest_days_home")) - safe_float(snapshot.get("rest_days_away"))
            if snapshot
            else 0.0
        ),
        "schedule_gap": (
            safe_float(snapshot.get("schedule_load_away")) - safe_float(snapshot.get("schedule_load_home"))
            if snapshot
            else 0.0
        ),
    }

    home_rating = safe_float(snapshot.get("home_rating")) if snapshot else extract_strength_metrics(
        str(row.get("elo_home", "") or "")
    )["rating"]
    away_rating = safe_float(snapshot.get("away_rating")) if snapshot else extract_strength_metrics(
        str(row.get("elo_away", "") or "")
    )["rating"]

    # Opponent-strength-adjusted form. We pull rank / team_count from the
    # cached feature payload when available, otherwise re-derive from the
    # text. This lets quant/ml models read residuals (actual - expected)
    # which carry more signal than raw PPG: a 17th-place team with PPG
    # 2.0 is a much stronger over-performance than a 5th-place team with
    # the same PPG, but raw PPG treats them identically.
    strength_home = payload.get("strength_home") or extract_strength_metrics(
        str(row.get("elo_home", "") or "")
    )
    strength_away = payload.get("strength_away") or extract_strength_metrics(
        str(row.get("elo_away", "") or "")
    )
    market_value = payload.get("market_value") or extract_market_value_metrics(
        str(row.get("market_value_summary", "") or "")
    )
    home_form_adj = opponent_adjusted_form(recent_home, strength_home)
    away_form_adj = opponent_adjusted_form(recent_away, strength_away)
    market_value_rating_gap = 0.0
    if isinstance(market_value, Mapping) and safe_int(market_value.get("coverage")):
        market_value_rating_gap = clamp(
            safe_float(market_value.get("log_gap")) * 85.0,
            -190.0,
            190.0,
        )

    home_strength = (
        home_rating
        + safe_float(recent_home["points_per_game"]) * 24.0
        + safe_float(split["home_ppg"]) * 18.0
        + safe_float(recent_home["goal_diff_per_game"]) * 22.0
        + max(market_value_rating_gap, 0.0) * 0.45
    )
    away_strength = (
        away_rating
        + safe_float(recent_away["points_per_game"]) * 24.0
        + safe_float(split["away_ppg"]) * 18.0
        + safe_float(recent_away["goal_diff_per_game"]) * 22.0
        + max(-market_value_rating_gap, 0.0) * 0.45
    )

    motivation_text = str(row.get("motivation_or_schedule_notes", "") or "")
    motivation_signal = 0.15 if "参考最近交锋" in motivation_text else 0.0

    # xG: prefer cached payload, otherwise fetch live (with disk cache).
    xg_metrics = payload.get("xg") if isinstance(payload, dict) else None
    if not isinstance(xg_metrics, dict) or not xg_metrics:
        xg_metrics = extract_xg_metrics(
            str(row.get("home_team", "") or ""),
            str(row.get("away_team", "") or ""),
            str(row.get("league", "") or ""),
        )

    return {
        "recent_home": recent_home,
        "recent_away": recent_away,
        "split": split,
        "h2h": h2h,
        "market_odds": market_odds,
        "market_probs": market_probs,
        "asian_handicap": asian_handicap,
        "home_rating": home_rating,
        "away_rating": away_rating,
        "rating_gap": home_rating - away_rating,
        "market_value": market_value,
        "market_value_rating_gap": market_value_rating_gap,
        "home_strength": home_strength,
        "away_strength": away_strength,
        "strength_gap": home_strength - away_strength,
        "h2h_edge": safe_float(h2h.get("edge")),
        "lineup": lineup,
        "lineup_raw": lineup_raw,
        "schedule": schedule,
        "motivation_signal": motivation_signal,
        "market_bias_anchor": market_probs["home"] - market_probs["away"],
        "feature_snapshot_id": safe_int(snapshot.get("snapshot_id")) if snapshot else 0,
        "home_form_adj": home_form_adj,
        "away_form_adj": away_form_adj,
        "form_residual_gap": home_form_adj["ppg_residual"] - away_form_adj["ppg_residual"],
        "goal_diff_residual_gap": home_form_adj["goal_diff_residual"] - away_form_adj["goal_diff_residual"],
        "xg": xg_metrics,
    }


def probability_vector_for_outcome(
    probs: Mapping[str, float],
    actual_result: str,
) -> tuple[float, float, float]:
    if actual_result == "home":
        return (1.0, 0.0, 0.0)
    if actual_result == "draw":
        return (0.0, 1.0, 0.0)
    return (0.0, 0.0, 1.0)


__all__ = [
    "OUTCOMES",
    "build_feature_snapshot",
    "build_match_features",
    "clamp",
    "extract_home_away_metrics",
    "extract_h2h_metrics",
    "extract_market_odds",
    "extract_asian_handicap",
    "extract_market_value_metrics",
    "extract_record_metrics",
    "extract_schedule_metrics",
    "extract_strength_metrics",
    "infer_match_datetime",
    "market_implied_probs",
    "normalize_probs",
    "normalize_text",
    "poisson_probability",
    "probability_vector_for_outcome",
    "safe_float",
    "safe_int",
]
