import os
import re
import sqlite3
import statistics
from datetime import datetime
from typing import Any, Mapping
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from collection_repository import (
    DEFAULT_ISSUE_RETENTION_COUNT,
    get_collection_stats,
    get_match,
    get_match_analysis,
    init_db,
    list_issues,
    list_matches,
    list_matches_by_issue,
    prune_to_recent_issues,
    save_analysis,
    save_feature_snapshot,
    save_failed_analysis,
    serialize_match,
    upsert_matches,
)
from collection_strategy import DIMENSION_FIELDS, apply_unified_collection_strategy
from feature_engine import build_feature_snapshot
from source_500_client import fetch_current_matches, fetch_html, fetch_issue_matches
from source_lineup_client import build_failure_notes, supplement_injury_or_lineup_notes
from source_supplement_client import supplement_match_data


LIANSAI_BASE_URL = "https://liansai.500.com"
_JIFEN_CANDIDATES_CACHE: dict[str, list[tuple[str, str]]] = {}
_JIFEN_TABLE_CACHE: dict[str, dict[str, dict[str, str]]] = {}
_TEAM_LEAGUE_CACHE: dict[str, dict[str, str]] = {}


def _env_flag(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


FIELD_GROUPS = [
    (
        "基础信息",
        [
            "match_id",
            "issue",
            "league",
            "match_no",
            "match_time",
            "home_team",
            "away_team",
            "source_match_url",
            "shuju_url",
            "ouzhi_url",
            "touzhu_url",
            "yazhi_url",
            "sync_time",
            "collected_at",
        ],
    ),
    (
        "维度一：基础实力",
        [
            "elo_home",
            "elo_away",
            "market_value_summary",
            "recent_form_home",
            "recent_form_away",
            "home_away_form",
        ],
    ),
    (
        "维度二：近期动态",
        [
            "head_to_head_summary",
            "injury_or_lineup_notes",
            "motivation_or_schedule_notes",
        ],
    ),
    (
        "维度三：市场数据",
        [
            "european_odds_movement_summary",
            "asian_handicap_summary",
            "betting_heat_summary",
        ],
    ),
    (
        "来源与备注",
        ["media_source_links", "collected_sources", "collection_quality_summary", "remarks"],
    ),
]

FIELD_LABELS = {
    "match_id": "match_id",
    "issue": "期号",
    "league": "赛事",
    "match_no": "序号",
    "match_time": "比赛时间",
    "home_team": "主队",
    "away_team": "客队",
    "source_match_url": "胜负彩页",
    "shuju_url": "数据分析页",
    "ouzhi_url": "欧指页",
    "touzhu_url": "投注页",
    "yazhi_url": "亚盘页",
    "sync_time": "列表同步时间",
    "collected_at": "采集时间",
    "elo_home": "主队 Elo/实力代理",
    "elo_away": "客队 Elo/实力代理",
    "market_value_summary": "球队球员身价",
    "recent_form_home": "主队近期状态",
    "recent_form_away": "客队近期状态",
    "home_away_form": "主客场表现",
    "head_to_head_summary": "交锋记录",
    "injury_or_lineup_notes": "伤停/阵容",
    "motivation_or_schedule_notes": "战意/赛程",
    "european_odds_movement_summary": "欧赔变化",
    "asian_handicap_summary": "让球亚盘变化",
    "betting_heat_summary": "投注热度",
    "media_source_links": "来源链接",
    "collected_sources": "采集来源",
    "collection_quality_summary": "采集策略与质量",
    "remarks": "备注",
}

REQUIRED_DIMENSION_TITLES = (
    "维度一：基础实力",
    "维度二：近期动态",
    "维度三：市场数据",
)
COLLECTION_FAILURE_PREFIX = "采集失败："
MISSING_DIMENSION_PREFIX = f"{COLLECTION_FAILURE_PREFIX}缺少采集维度："
COLLECTION_PLACEHOLDER_TEXTS = (
    "未在公开来源中命中预计首发",
    "外部公开来源补采未命中",
    "伤停/阵容补采失败",
)
# 这些值表示"真实无数据"而非"采集失败"，不应被当作缺失维度
LEGITIMATE_EMPTY_TEXTS = (
    "双方暂无交战历史",
    "当前页面未列出明确伤停名单",
    "页面含预计阵容模块，可结合临场信息再人工复核伤停",
)
OPTIONAL_DIMENSION_FIELDS = {"elo_home", "elo_away", "market_value_summary"}
BATCH_FAST_MODE_RETRY_FIELDS = ("elo_home", "elo_away", "market_value_summary")


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def unique_non_empty(values: list[str]) -> list[str]:
    seen = set()
    results = []
    for value in values:
        text = normalize_text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        results.append(text)
    return results


def unique_link_items(items: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen = set()
    results = []
    for label, url in items:
        clean_url = normalize_text(url)
        if not clean_url or clean_url in seen:
            continue
        seen.add(clean_url)
        results.append((normalize_text(label), clean_url))
    return results


def split_lines(value: str) -> list[str]:
    return [line for line in unique_non_empty((value or "").splitlines()) if line]


def _has_field(data: Mapping[str, Any], field: str) -> bool:
    try:
        keys = data.keys()
    except Exception:  # noqa: BLE001
        try:
            data[field]
            return True
        except Exception:  # noqa: BLE001
            return False
    return field in keys


def _field_value(data: Mapping[str, Any] | None, field: str, default: Any = "") -> Any:
    if data is None:
        return default
    try:
        value = data[field]
    except Exception:  # noqa: BLE001
        return default
    return default if value is None else value


def required_dimension_fields() -> list[tuple[str, str, str]]:
    fields: list[tuple[str, str, str]] = []
    for title, group_fields in FIELD_GROUPS:
        if title not in REQUIRED_DIMENSION_TITLES:
            continue
        for field in group_fields:
            fields.append((title, field, FIELD_LABELS.get(field, field)))
    return fields


def get_missing_required_dimensions(data: Mapping[str, Any] | None) -> list[str]:
    if data is None:
        return []

    missing: list[str] = []
    for title, field, label in required_dimension_fields():
        if field in OPTIONAL_DIMENSION_FIELDS:
            continue
        value = normalize_text(str(_field_value(data, field, "") or ""))
        if not value:
            missing.append(f"{title}/{label}")
        elif any(token in value for token in COLLECTION_PLACEHOLDER_TEXTS):
            missing.append(f"{title}/{label}")
        elif any(value == token for token in LEGITIMATE_EMPTY_TEXTS):
            # 合法的空数据（如"双方暂无交战历史"），不算缺失
            pass
    return missing


def _is_only_elo_missing(missing: list[str]) -> bool:
    """Check if the only missing fields are elo-related (strength proxy)."""
    for item in missing:
        # Missing items are formatted as "维度一：基础实力/主队 Elo/实力代理"
        if "Elo" not in item and "elo" not in item and "实力代理" not in item:
            return False
    return bool(missing)


def has_required_dimension_payload(data: Mapping[str, Any] | None) -> bool:
    if data is None:
        return False
    return any(_has_field(data, field) for _title, field, _label in required_dimension_fields())


def summarize_issue_entries(entries: list[dict[str, Any]], limit: int = 5) -> str:
    parts = []
    for entry in entries[:limit]:
        label = str(entry.get("match_label") or entry.get("match_id") or "").strip()
        reason = str(entry.get("reason") or "").strip()
        if label and reason:
            parts.append(f"{label}（{reason}）")
        elif label:
            parts.append(label)
        elif reason:
            parts.append(reason)
    if len(entries) > limit:
        parts.append(f"等 {len(entries)} 场")
    return "；".join(parts)


def format_missing_required_dimensions(missing: list[str], limit: int = 8) -> str:
    if not missing:
        return ""
    visible = missing[:limit]
    suffix = f" 等 {len(missing)} 项" if len(missing) > limit else ""
    return "、".join(visible) + suffix


def missing_dimension_fields(missing: list[str]) -> list[str]:
    labels_to_fields = {label: field for _title, field, label in required_dimension_fields()}
    fields: list[str] = []
    for item in missing:
        label = item.rsplit("/", 1)[-1]
        field = labels_to_fields.get(label)
        if field:
            fields.append(field)
    return fields


def get_collection_failure_reason(data: Mapping[str, Any] | None) -> str:
    if data is None:
        return "未找到对赛采集记录"
    if not normalize_text(str(_field_value(data, "collected_at", "") or "")):
        return "当前场次未采集，请先采集"

    has_payload = has_required_dimension_payload(data)
    missing = get_missing_required_dimensions(data) if has_payload else []
    remarks = str(_field_value(data, "remarks", "") or "")
    for line in split_lines(remarks):
        if line.startswith(MISSING_DIMENSION_PREFIX):
            if missing:
                return f"缺少采集维度：{format_missing_required_dimensions(missing)}"
            if not has_payload:
                return line
            continue
        if line.startswith(COLLECTION_FAILURE_PREFIX):
            return line

    status = normalize_text(str(_field_value(data, "collection_status", "") or ""))
    if status == "success":
        return ""

    if has_payload:
        if missing:
            return f"缺少采集维度：{format_missing_required_dimensions(missing)}"
        return ""

    if status == "failed":
        return "采集维度不完整，请重新采集"
    return ""


def make_absolute_liansai_url(url: str) -> str:
    if not url:
        return ""
    return urljoin(f"{LIANSAI_BASE_URL}/", url)


def classify_source_link(url: str) -> tuple[str, str, str]:
    parsed = urlparse(url or "")
    host = parsed.netloc.lower().lstrip("www.")
    path = parsed.path.lower()

    if "trade.500.com" in host and "/sfc/" in path:
        return ("match-list", "对赛列表", "500胜负彩页")
    if "odds.500.com" in host and "shuju-" in path:
        return ("dim12-data", "维度一/二 · 基础数据 / 交锋 / 赛程", "500数据分析页")
    if "odds.500.com" in host and "ouzhi-" in path:
        return ("dim3-odds", "维度三 · 欧赔变化", "500百家欧赔页")
    if "odds.500.com" in host and "touzhu-" in path:
        return ("dim3-heat", "维度三 · 投注热度", "500投注分析页")
    if "liansai.500.com" in host and "jifen-" in path:
        return ("dim1-strength", "维度一 · 实力代理 / 联赛积分", "500联赛积分页")
    if "flashscore" in host:
        return ("dim2-lineup", "维度二 · 伤停 / 阵容", "Flashscore")
    if "soccerway.com" in host:
        return ("dim2-lineup", "维度二 · 伤停 / 阵容", "Soccerway")
    if "livescore.com" in host:
        return ("dim2-lineup", "维度二 · 伤停 / 阵容", "LiveScore")
    domain_label = host or "外部来源"
    return ("other", "补充来源", domain_label)


def build_source_notes_summary(data: dict) -> dict[str, list[dict] | list[str]]:
    source_urls = unique_non_empty(
        [data.get("source_match_url", "")]
        + split_lines(data.get("media_source_links", ""))
    )
    grouped: dict[str, dict[str, object]] = {}
    group_order = [
        "match-list",
        "dim1-strength",
        "dim12-data",
        "dim2-lineup",
        "dim3-odds",
        "dim3-heat",
        "other",
    ]

    for url in source_urls:
        key, title, label = classify_source_link(url)
        if key not in grouped:
            grouped[key] = {"title": title, "links": []}
        links = grouped[key]["links"]
        if isinstance(links, list) and not any(item["url"] == url for item in links):
            links.append({"label": label, "url": url})

    ordered_groups = [grouped[key] for key in group_order if key in grouped]
    remarks = [
        line
        for line in split_lines(data.get("remarks", ""))
        if not line.startswith("精简展示模式：")
    ]
    return {
        "groups": ordered_groups,
        "remarks": remarks,
    }


def find_jifen_links(html: str, fallback_label: str = "") -> list[tuple[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href", "")
        if "jifen-" not in href:
            continue
        url = make_absolute_liansai_url(href)
        if "liansai.500.com" not in url:
            continue
        label = normalize_text(anchor.get_text(" ", strip=True)) or fallback_label
        links.append((label or "赛程积分榜", url))
    return unique_link_items(links)


def extract_team_ids_and_competition_id(shuju_html: str) -> tuple[list[str], str]:
    team_ids = unique_non_empty(
        re.findall(r"liansai\.500\.com/team/(\d+)/?", shuju_html)
    )
    competition_ids = unique_non_empty(
        re.findall(r"liansai\.500\.com/zuqiu-(\d+)/", shuju_html)
    )
    return team_ids[:2], (competition_ids[0] if competition_ids else "")


def get_competition_jifen_candidates(competition_id: str) -> list[tuple[str, str]]:
    if competition_id in _JIFEN_CANDIDATES_CACHE:
        return _JIFEN_CANDIDATES_CACHE[competition_id]

    competition_url = f"{LIANSAI_BASE_URL}/zuqiu-{competition_id}/"
    competition_html = fetch_html(competition_url)
    homepage_links = find_jifen_links(competition_html, fallback_label="赛程积分榜")
    default_candidate = homepage_links[0] if homepage_links else ("赛程积分榜", "")

    stage_links: list[tuple[str, str]] = []
    if default_candidate[1]:
        default_html = fetch_html(default_candidate[1])
        stage_links = find_jifen_links(default_html)

    preferred = []
    preferred.extend(
        [item for item in stage_links if "联赛" in item[0] or "总积分" in item[0]]
    )
    if default_candidate[1]:
        preferred.append(default_candidate)
    preferred.extend(
        [
            item
            for item in stage_links
            if item[1] != default_candidate[1]
            and "联赛" not in item[0]
            and "总积分" not in item[0]
        ]
    )
    preferred.extend(homepage_links[1:])

    candidates = unique_link_items(preferred or homepage_links)
    _JIFEN_CANDIDATES_CACHE[competition_id] = candidates
    return candidates


def is_cup_competition(label: str) -> bool:
    text = normalize_text(label)
    if not text:
        return False
    keywords = (
        "杯",
        "欧冠",
        "欧联",
        "欧协",
        "亚冠",
        "世俱杯",
        "友谊",
        "超级杯",
        "资格赛",
        "附加赛",
    )
    return any(keyword in text for keyword in keywords)


def get_team_league_context(team_id: str) -> dict[str, str]:
    if team_id in _TEAM_LEAGUE_CACHE:
        return _TEAM_LEAGUE_CACHE[team_id]

    team_url = f"{LIANSAI_BASE_URL}/team/{team_id}/"
    html = fetch_html(team_url)
    soup = BeautifulSoup(html, "html.parser")

    competition_labels: dict[str, str] = {}
    direct_jifen_links: list[tuple[str, str, str]] = []
    league_competitions: list[tuple[str, str]] = []

    for anchor in soup.find_all("a", href=True):
        href = make_absolute_liansai_url(anchor.get("href", ""))
        label = normalize_text(anchor.get_text(" ", strip=True))

        competition_match = re.search(r"/zuqiu-(\d+)/?$", href)
        if competition_match and label:
            competition_id = competition_match.group(1)
            competition_labels.setdefault(competition_id, label)
            if not is_cup_competition(label):
                league_competitions.append((competition_id, label))

        jifen_match = re.search(r"/zuqiu-(\d+)/jifen-\d+/", href)
        if jifen_match:
            competition_id = jifen_match.group(1)
            direct_jifen_links.append(
                (
                    competition_id,
                    competition_labels.get(competition_id, ""),
                    href,
                )
            )

    direct_jifen_links = [
        item
        for item in direct_jifen_links
        if item[1] and not is_cup_competition(item[1])
    ] or direct_jifen_links

    result = {
        "team_url": team_url,
        "competition_id": "",
        "competition_label": "",
        "jifen_url": "",
    }
    if direct_jifen_links:
        competition_id, competition_label, jifen_url = direct_jifen_links[0]
        result.update(
            {
                "competition_id": competition_id,
                "competition_label": competition_label,
                "jifen_url": jifen_url,
            }
        )
    elif league_competitions:
        competition_id, competition_label = league_competitions[0]
        result.update(
            {
                "competition_id": competition_id,
                "competition_label": competition_label,
            }
        )

    _TEAM_LEAGUE_CACHE[team_id] = result
    return result


def normalize_header(text: str) -> str:
    return re.sub(r"[\s:：()（）]+", "", text or "")


def parse_integer(text: str) -> str:
    match = re.search(r"[+-]?\d+", text or "")
    if not match:
        return ""
    return str(int(match.group(0)))


def parse_goals_pair(text: str) -> tuple[str, str]:
    match = re.search(r"(\d+)\s*/\s*(\d+)", text or "")
    if not match:
        return "", ""
    return match.group(1), match.group(2)


def map_header_key(header: str) -> str:
    cleaned = normalize_header(header)
    if not cleaned:
        return ""
    if "积分" in cleaned or cleaned in {"分"}:
        return "points"
    if "净" in cleaned:
        return "goal_diff"
    if "进" in cleaned and "失" in cleaned:
        return "goals_pair"
    if "排名" in cleaned or "名次" in cleaned or cleaned in {"名", "排"}:
        return "rank"
    if cleaned in {"赛", "场", "场次", "比赛"} or "场次" in cleaned:
        return "played"
    if cleaned == "胜":
        return "wins"
    if cleaned == "平":
        return "draws"
    if cleaned == "负":
        return "losses"
    if cleaned in {"进", "进球"}:
        return "goals_for"
    if cleaned in {"失", "失球"}:
        return "goals_against"
    return ""


def parse_jifen_row(headers: list[str], values: list[str]) -> dict[str, str]:
    row: dict[str, str] = {}
    for header, value in zip(headers, values):
        key = map_header_key(header)
        if not key:
            continue
        if key == "goals_pair":
            goals_for, goals_against = parse_goals_pair(value)
            if goals_for:
                row["goals_for"] = goals_for
            if goals_against:
                row["goals_against"] = goals_against
            continue
        parsed = parse_integer(value)
        if parsed:
            row[key] = parsed
    return row


def parse_jifen_table(html: str) -> dict[str, dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    rows_by_team: dict[str, dict[str, str]] = {}

    for table in soup.find_all("table"):
        classes = table.get("class") or []
        if isinstance(classes, str):
            classes = [classes]
        if not any("jifen" in cls for cls in classes):
            continue

        headers: list[str] = []
        for tr in table.find_all("tr"):
            header_cells = tr.find_all("th")
            if not header_cells:
                continue
            candidate_headers = [
                normalize_text(cell.get_text(" ", strip=True)) for cell in header_cells
            ]
            if any(map_header_key(text) for text in candidate_headers):
                headers = candidate_headers
                break

        table_rows: list[tuple[str, dict[str, str]]] = []
        for tr in table.find_all("tr"):
            anchor = tr.find("a", href=re.compile(r"/team/\d+"))
            if anchor is None:
                continue
            href = anchor.get("href", "")
            match = re.search(r"/team/(\d+)/?", href)
            if not match:
                continue
            cells = tr.find_all("td")
            if not cells:
                continue

            team_id = match.group(1)
            values = [normalize_text(cell.get_text(" ", strip=True)) for cell in cells]
            parsed_row = parse_jifen_row(headers, values)
            parsed_row["team_name"] = normalize_text(anchor.get_text(" ", strip=True))
            table_rows.append((team_id, parsed_row))

        team_count = str(len(table_rows)) if table_rows else ""
        for team_id, parsed_row in table_rows:
            parsed_row["team_count"] = team_count
            rows_by_team[team_id] = parsed_row

    return rows_by_team


def load_jifen_table(url: str) -> dict[str, dict[str, str]]:
    if url not in _JIFEN_TABLE_CACHE:
        _JIFEN_TABLE_CACHE[url] = parse_jifen_table(fetch_html(url))
    return _JIFEN_TABLE_CACHE[url]


def find_team_row_from_context(team_id: str, context: dict[str, str]) -> dict[str, str]:
    jifen_url = context.get("jifen_url", "")
    if jifen_url:
        row = load_jifen_table(jifen_url).get(team_id)
        if row is not None:
            return {
                "competition_label": context.get("competition_label", ""),
                "stage_label": "联赛积分榜",
                "source_url": jifen_url,
                "row": row,
            }

    competition_id = context.get("competition_id", "")
    if not competition_id:
        return {}

    for stage_label, url in get_competition_jifen_candidates(competition_id):
        row = load_jifen_table(url).get(team_id)
        if row is not None:
            return {
                "competition_label": context.get("competition_label", ""),
                "stage_label": stage_label or "联赛积分榜",
                "source_url": url,
                "row": row,
            }
    return {}


def format_strength_proxy(
    match: sqlite3.Row,
    stage_label: str,
    row: dict[str, str] | None,
    competition_label: str = "",
) -> str:
    if not row:
        return ""

    title = normalize_text(competition_label) or normalize_text(match["league"])
    label = normalize_text(stage_label)
    if label and label not in title:
        title = f"{title} {label}".strip()

    segments = []
    rank = row.get("rank", "")
    team_count = row.get("team_count", "")
    if rank:
        segments.append(f"第{rank}/{team_count}" if team_count else f"第{rank}")
    if row.get("points"):
        segments.append(f"{row['points']}分")

    record = []
    if row.get("wins"):
        record.append(f"{row['wins']}胜")
    if row.get("draws"):
        record.append(f"{row['draws']}平")
    if row.get("losses"):
        record.append(f"{row['losses']}负")
    if record:
        segments.append("".join(record))

    goal_parts = []
    if row.get("goals_for"):
        goal_parts.append(f"进{row['goals_for']}")
    if row.get("goals_against"):
        goal_parts.append(f"失{row['goals_against']}")
    if row.get("goal_diff"):
        goal_parts.append(f"净胜{row['goal_diff']}")
    if goal_parts:
        segments.append("，".join(goal_parts))

    if not segments:
        return title
    return f"{title} {'，'.join(segments)}"


def extract_strength_snapshots(match: sqlite3.Row, shuju_html: str) -> dict[str, str]:
    team_ids, competition_id = extract_team_ids_and_competition_id(shuju_html)
    if len(team_ids) < 2:
        return {
            "elo_home": "",
            "elo_away": "",
            "source_url": "",
            "source_label": "",
            "note": "数据页未解析出两队的联赛 team id，无法定位积分榜行。",
        }
    if not competition_id:
        return {
            "elo_home": "",
            "elo_away": "",
            "source_url": "",
            "source_label": "",
            "note": "数据页未解析出赛事 id，无法定位联赛积分页。",
        }

    home_team_id, away_team_id = team_ids[0], team_ids[1]
    candidates = get_competition_jifen_candidates(competition_id)
    if not candidates:
        return {
            "elo_home": "",
            "elo_away": "",
            "source_url": "",
            "source_label": "",
            "note": f"未找到赛事 {competition_id} 的积分页入口。",
        }

    best_partial: dict[str, str] | None = None
    for stage_label, url in candidates:
        rows = load_jifen_table(url)
        home_row = rows.get(home_team_id)
        away_row = rows.get(away_team_id)
        found_count = int(home_row is not None) + int(away_row is not None)
        if not found_count:
            continue

        snapshot = {
            "elo_home": format_strength_proxy(match, stage_label, home_row),
            "elo_away": format_strength_proxy(match, stage_label, away_row),
            "source_url": url,
            "source_label": stage_label or "赛程积分榜",
            "note": "",
            "found_count": str(found_count),
        }
        if found_count == 2:
            return snapshot
        if best_partial is None or int(snapshot["found_count"]) > int(
            best_partial["found_count"]
        ):
            best_partial = snapshot

    # 尝试回退到球队所属联赛积分页（处理跨联赛/附加赛等情况）
    home_context = get_team_league_context(home_team_id)
    away_context = get_team_league_context(away_team_id)
    home_snapshot = find_team_row_from_context(home_team_id, home_context)
    away_snapshot = find_team_row_from_context(away_team_id, away_context)

    # 合并 best_partial 和 fallback 结果，优先使用有数据的部分
    merged_home = ""
    merged_away = ""
    merged_sources: list[str] = []
    merged_labels: list[str] = []

    # 从 best_partial 取已有的
    if best_partial is not None:
        merged_home = best_partial.get("elo_home", "")
        merged_away = best_partial.get("elo_away", "")
        if best_partial.get("source_url"):
            merged_sources.append(best_partial["source_url"])
        if best_partial.get("source_label"):
            merged_labels.append(best_partial["source_label"])

    # 从 fallback 补充缺失的
    if home_snapshot:
        fb_home = format_strength_proxy(
            match,
            home_snapshot.get("stage_label", ""),
            home_snapshot.get("row"),
            home_snapshot.get("competition_label", ""),
        )
        if fb_home and not merged_home:
            merged_home = fb_home
        if home_snapshot.get("source_url"):
            merged_sources.append(home_snapshot["source_url"])
        if home_snapshot.get("competition_label"):
            merged_labels.append(home_snapshot["competition_label"])

    if away_snapshot:
        fb_away = format_strength_proxy(
            match,
            away_snapshot.get("stage_label", ""),
            away_snapshot.get("row"),
            away_snapshot.get("competition_label", ""),
        )
        if fb_away and not merged_away:
            merged_away = fb_away
        if away_snapshot.get("source_url"):
            merged_sources.append(away_snapshot["source_url"])
        if away_snapshot.get("competition_label"):
            merged_labels.append(away_snapshot["competition_label"])

    if merged_home or merged_away:
        parts: list[str] = []
        if best_partial is not None:
            parts.append("积分页已匹配到部分球队，另一侧未在同页积分表中出现。")
        if (home_snapshot or away_snapshot) and best_partial is not None:
            parts.append("已通过球队所属联赛积分页回退补全。")
        elif home_snapshot or away_snapshot:
            parts.append("赛事积分页未命中，已回退到球队所属联赛积分页。")
        return {
            "elo_home": merged_home,
            "elo_away": merged_away,
            "source_url": "\n".join(unique_non_empty(merged_sources)),
            "source_label": "球队联赛积分回退" if (home_snapshot or away_snapshot) and best_partial is None else (
                "混合来源（赛事积分页 + 球队联赛回退）" if (home_snapshot or away_snapshot) and best_partial is not None
                else "; ".join(unique_non_empty(merged_labels)) or "赛程积分榜"
            ),
            "note": " ".join(parts),
        }

    if best_partial is not None:
        best_partial["note"] = "积分页已匹配到部分球队，另一侧未在同页积分表中出现。"
        return best_partial

    return {
        "elo_home": "",
        "elo_away": "",
        "source_url": "",
        "source_label": "",
        "note": "已访问联赛积分页，但未匹配到两队在积分表中的行。",
    }


def _build_sync_retention_message(match_count: int, retention: dict[str, object]) -> str:
    kept_issues = [str(issue) for issue in (retention.get("kept_issues") or []) if str(issue).strip()]
    kept_text = "、".join(kept_issues) if kept_issues else "-"
    deleted_issue_count = int(retention.get("deleted_issue_count", 0) or 0)
    deleted_total = int(retention.get("deleted_total", 0) or 0)
    if deleted_issue_count > 0 or deleted_total > 0:
        return (
            f"已同步当前对赛 {match_count} 场，并自动仅保留最近 {DEFAULT_ISSUE_RETENTION_COUNT} 期：{kept_text}。"
            f"已清理旧数据：删除 {deleted_issue_count} 期、{retention.get('deleted_matches', 0)} 场比赛、"
            f"{retention.get('deleted_analyses', 0)} 条采集、{retention.get('deleted_feature_snapshots', 0)} 条特征快照、"
            f"{retention.get('deleted_prediction_runs', 0)} 条预测、{retention.get('deleted_feedback_logs', 0)} 条反馈。"
        )
    return f"已同步当前对赛 {match_count} 场，系统当前仅保留最近 {DEFAULT_ISSUE_RETENTION_COUNT} 期：{kept_text}。"


def _build_issue_sync_retention_message(
    issue: str,
    match_count: int,
    retention: dict[str, object],
) -> tuple[str, str]:
    issue_text = str(issue or "").strip()
    kept_issues = [str(item) for item in (retention.get("kept_issues") or []) if str(item).strip()]
    kept_text = "、".join(kept_issues) if kept_issues else "-"
    retained = issue_text in kept_issues
    deleted_issue_count = int(retention.get("deleted_issue_count", 0) or 0)
    deleted_total = int(retention.get("deleted_total", 0) or 0)

    if not retained:
        return (
            f"已抓取期号 {issue_text} 对赛 {match_count} 场，但系统仅保留最近 "
            f"{DEFAULT_ISSUE_RETENTION_COUNT} 期：{kept_text}，该期未进入保留窗口。",
            "warning",
        )
    if deleted_issue_count > 0 or deleted_total > 0:
        return (
            f"已补入期号 {issue_text} 对赛 {match_count} 场，并自动仅保留最近 "
            f"{DEFAULT_ISSUE_RETENTION_COUNT} 期：{kept_text}。"
            f"已清理旧数据：删除 {deleted_issue_count} 期、{retention.get('deleted_matches', 0)} 场比赛、"
            f"{retention.get('deleted_analyses', 0)} 条采集、{retention.get('deleted_feature_snapshots', 0)} 条特征快照、"
            f"{retention.get('deleted_prediction_runs', 0)} 条预测、{retention.get('deleted_feedback_logs', 0)} 条反馈。",
            "success",
        )
    return (
        f"已补入期号 {issue_text} 对赛 {match_count} 场，系统当前仅保留最近 "
        f"{DEFAULT_ISSUE_RETENTION_COUNT} 期：{kept_text}。",
        "success",
    )


def sync_matches(return_details: bool = False) -> list[dict] | dict[str, object]:
    matches = fetch_current_matches()
    upsert_matches(matches)
    retention = prune_to_recent_issues(DEFAULT_ISSUE_RETENTION_COUNT)
    if return_details:
        status_message = _build_sync_retention_message(len(matches), retention)
        return {
            "matches": matches,
            "retention": retention,
            "status_message": status_message,
            "status_level": "success",
        }
    return matches


def sync_issue_matches(issue: str, return_details: bool = False) -> list[dict] | dict[str, object]:
    issue_text = str(issue or "").strip()
    matches = fetch_issue_matches(issue_text)
    upsert_matches(matches)
    retention = prune_to_recent_issues(DEFAULT_ISSUE_RETENTION_COUNT)
    if return_details:
        status_message, status_level = _build_issue_sync_retention_message(
            issue_text,
            len(matches),
            retention,
        )
        return {
            "matches": matches,
            "issue": issue_text,
            "retention": retention,
            "status_message": status_message,
            "status_level": status_level,
        }
    return matches


def _section_text(node) -> str:
    if node is None:
        return ""
    return normalize_text(node.get_text(" ", strip=True).replace("\xa0", " "))


def _extract_record_summary(section) -> str:
    if section is None:
        return ""

    record = section.find("p", class_="record_msg")
    if record is None:
        return ""

    text = _section_text(record)
    match = re.search(
        (
            r"近\s*(\d+)场.*?"
            r"(\d+)胜\s*(\d+)平\s*(\d+)负.*?"
            r"进\s*(\d+)球\s*失\s*(\d+)球.*?"
            r"胜率\s*(\d+%).*?"
            r"赢盘率\s*(\d+%).*?"
            r"大球率\s*(\d+%)"
        ),
        text,
    )
    if not match:
        return ""

    match_count, wins, draws, losses, goals_for, goals_against, win_rate, cover_rate, over_rate = match.groups()
    return (
        f"近{match_count}场 {wins}胜{draws}平{losses}负，"
        f"进{goals_for}球失{goals_against}球，"
        f"胜率{win_rate}，赢盘率{cover_rate}，大球率{over_rate}"
    )


def extract_recent_forms(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    return (
        _extract_record_summary(soup.find(id="team_zhanji1_1")),
        _extract_record_summary(soup.find(id="team_zhanji1_0")),
    )


def extract_home_away_form(html: str) -> str:
    blocks = re.findall(
        r'<p><strong>(.*?)</strong>近10场战绩.*?<span class="ying">(\d+)胜</span><span class="ping">(\d+)平</span><span class="shu">(\d+)负</span>.*?近?<span class="ying">(\d+)球</span>失<span class="shu">(\d+)球</span>',
        html,
        re.S,
    )
    if not blocks:
        return ""

    home = None
    away = None
    for block in blocks:
        label = normalize_text(block[0])
        if "主场" in label and home is None:
            home = block
        if "客场" in label and away is None:
            away = block

    if home is None and len(blocks) >= 2:
        home = blocks[-2]
    if away is None and len(blocks) >= 1:
        away = blocks[-1]
    if home is None or away is None:
        return ""

    return (
        f"主场 {home[1]}胜{home[2]}平{home[3]}负，进{home[4]}失{home[5]}"
        f" | 客场 {away[1]}胜{away[2]}平{away[3]}负，进{away[4]}失{away[5]}"
    )


def extract_h2h(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    section = soup.find(id="team_jiaozhan")
    if section is None:
        return ""

    # 检查页面是否明确显示"双方暂无交战历史"
    section_text = _section_text(section)
    if "暂无交战" in section_text or "暂无交锋" in section_text:
        return "双方暂无交战历史"

    summary = ""
    summary_text = _section_text(section.find("span", class_="his_info"))
    summary_match = re.search(
        (
            r"双方近\s*(\d+)\s*次交战.*?"
            r"(\d+)胜\s*(\d+)平\s*(\d+)负.*?"
            r"进\s*(\d+)球.*?失\s*(\d+)球"
        ),
        summary_text,
    )
    if summary_match:
        count, wins, draws, losses, goals_for, goals_against = summary_match.groups()
        summary = f"近{count}次交锋 {wins}胜{draws}平{losses}负，进{goals_for}球失{goals_against}球"

    rows = []
    table = section.find("table", class_="pub_table")
    if table is not None:
        for tr in table.find_all("tr"):
            classes = tr.get("class") or []
            if "bmatch" in classes:
                continue

            cells = tr.find_all("td")
            if len(cells) < 5:
                continue

            match_date = normalize_text(cells[1].get_text(" ", strip=True))
            matchup = cells[2]
            left_node = matchup.find("span", class_=re.compile(r"\bdz-l\b"))
            right_node = matchup.find("span", class_=re.compile(r"\bdz-r\b"))
            score_node = matchup.find("em")
            if left_node is None or right_node is None or score_node is None:
                continue

            left = normalize_text(left_node.get_text(" ", strip=True))
            right = normalize_text(right_node.get_text(" ", strip=True))
            score = normalize_text(score_node.get_text(" ", strip=True)).replace(" ", "")
            result_text = normalize_text(cells[4].get_text(" ", strip=True))

            left = re.sub(r"\[\d+\]", "", left).strip()
            right = re.sub(r"\[\d+\]", "", right).strip()
            if not match_date or not left or not right or score.upper() == "VS" or result_text == "-":
                continue

            rows.append(f"{match_date} {left} {score} {right}")
            if len(rows) == 3:
                break

    detail = "；".join(rows)
    if summary and detail:
        return f"{summary}；{detail}"
    return summary or detail


def extract_lineup_notes(html: str) -> str:
    if "预计阵容" in html:
        return "页面含预计阵容模块，可结合临场信息再人工复核伤停。"
    return "当前页面未列出明确伤停名单。"


def extract_motivation(match: sqlite3.Row, h2h_summary: str) -> str:
    base = f"{match['league']} {match['home_team']} vs {match['away_team']}"
    if h2h_summary:
        latest = h2h_summary.split("；")[-1]
        return f"{base}；可参考最近交锋：{latest}"
    return base


def _format_odds_value(value: float) -> str:
    return f"{value:.2f}"


def _parse_float_values(text: str) -> list[float]:
    values: list[float] = []
    for raw in re.findall(r"(?<![\d.])(\d+\.\d+)(?![\d.])", text):
        try:
            value = float(raw)
        except ValueError:
            continue
        if 1.01 <= value <= 30:
            values.append(value)
    return values


def extract_company_odds_medians(html: str) -> tuple[str, str, str, str, str, str]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="datatb")
    if table is None:
        return "", "", "", "", "", ""

    initial_rows: list[tuple[float, float, float]] = []
    current_rows: list[tuple[float, float, float]] = []
    for row in table.find_all("tr", recursive=False):
        row_classes = set(row.get("class") or [])
        if not ({"tr1", "tr2"} & row_classes):
            continue
        cells = [
            " ".join(cell.get_text(" ", strip=True).split())
            for cell in row.find_all(["td", "th"], recursive=False)
        ]
        if len(cells) < 3:
            continue
        values = _parse_float_values(cells[2])
        if len(values) < 6:
            continue
        initial_rows.append((values[0], values[1], values[2]))
        current_rows.append((values[3], values[4], values[5]))

    if not initial_rows or not current_rows:
        return "", "", "", "", "", ""

    initial_columns = list(zip(*initial_rows))
    current_columns = list(zip(*current_rows))
    initial = [_format_odds_value(statistics.median(column)) for column in initial_columns]
    current = [_format_odds_value(statistics.median(column)) for column in current_columns]
    return (*initial, *current)


def extract_average_odds(html: str) -> tuple[str, str, str, str, str, str]:
    average_row = re.search(
        r"平均欧赔.*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+)",
        html,
        re.S,
    )
    if average_row:
        return average_row.groups()
    return extract_company_odds_medians(html)


def build_odds_summary(match: sqlite3.Row, ouzhi_html: str) -> str:
    init_win, init_draw, init_loss, now_win, now_draw, now_loss = extract_average_odds(
        ouzhi_html
    )
    if all([init_win, init_draw, init_loss, now_win, now_draw, now_loss]):
        deltas = (
            float(now_win) - float(init_win),
            float(now_draw) - float(init_draw),
            float(now_loss) - float(init_loss),
        )
        return (
            f"初赔 {init_win}/{init_draw}/{init_loss} -> "
            f"即时 {now_win}/{now_draw}/{now_loss}；"
            f"变化 {deltas[0]:+.2f}/{deltas[1]:+.2f}/{deltas[2]:+.2f}"
        )
    if match["list_odds_win"]:
        return (
            "列表均赔 "
            f"{match['list_odds_win']}/{match['list_odds_draw']}/{match['list_odds_loss']}"
        )
    return ""


def build_asian_handicap_summary(yazhi_html: str) -> str:
    soup = BeautifulSoup(yazhi_html or "", "html.parser")
    average_cell = soup.find(string=re.compile(r"平均值"))
    if not average_cell:
        return ""
    average_row = average_cell.find_parent("tr")
    if average_row is None:
        return ""
    nested_rows = average_row.select("table.pl_table_data tr")
    if len(nested_rows) < 2:
        return ""

    def _cells(row) -> list[str]:
        return [normalize_text(cell.get_text(" ", strip=True)) for cell in row.find_all("td")]

    current = _cells(nested_rows[0])
    initial = _cells(nested_rows[1])
    if len(current) < 3 or len(initial) < 3:
        return ""
    return (
        f"亚盘平均 初盘 {initial[0]}/{initial[1]}/{initial[2]} -> "
        f"即时 {current[0]}/{current[1]}/{current[2]}"
    )


def build_heat_summary(match: sqlite3.Row, touzhu_html: str) -> str:
    prices = re.findall(r"成交价.*?(\d+\.\d+)", touzhu_html, re.S)
    if match["list_heat_win"]:
        summary = (
            f"投注比 胜{match['list_heat_win']}% "
            f"平{match['list_heat_draw']}% 负{match['list_heat_loss']}%"
        )
        if len(prices) >= 3:
            return f"{summary}；成交价 {prices[0]}/{prices[1]}/{prices[2]}"
        return summary
    return ""


def _emit_progress(progress_callback, **payload) -> None:
    if progress_callback is None:
        return
    progress_callback(**payload)


def _supplement_missing_dimensions(
    match: sqlite3.Row,
    analysis: dict,
    missing_dimensions: list[str],
) -> None:
    """通过免费公开资源补采缺失维度，直接修改 analysis 字典。"""
    if _env_flag("FOOTBALL_SKIP_EXTERNAL_SUPPLEMENT"):
        existing_remarks = split_lines(str(_field_value(analysis, "remarks", "") or ""))
        existing_remarks.append(
            "外部公开源补采跳过：FOOTBALL_SKIP_EXTERNAL_SUPPLEMENT=1"
        )
        analysis["remarks"] = "\n".join(unique_non_empty(existing_remarks))
        return

    home_team = match["home_team"]
    away_team = match["away_team"]
    league = match["league"]
    match_time = match["match_time"] if "match_time" in match.keys() else ""

    supplement = supplement_match_data(
        home_team=home_team,
        away_team=away_team,
        league=league,
        match_time=match_time,
        missing_dimensions=missing_dimensions,
    )

    # 合并补采结果到 analysis
    if supplement.get("elo_home") and not _field_value(analysis, "elo_home"):
        analysis["elo_home"] = supplement["elo_home"]
    if supplement.get("elo_away") and not _field_value(analysis, "elo_away"):
        analysis["elo_away"] = supplement["elo_away"]
    if supplement.get("market_value_summary") and not _field_value(analysis, "market_value_summary"):
        analysis["market_value_summary"] = supplement["market_value_summary"]
    if supplement.get("head_to_head_summary") and not _field_value(analysis, "head_to_head_summary"):
        analysis["head_to_head_summary"] = supplement["head_to_head_summary"]
        # 更新战意/赛程（依赖交锋记录）
        analysis["motivation_or_schedule_notes"] = extract_motivation(
            match, supplement["head_to_head_summary"]
        )
    if supplement.get("recent_form_home") and not _field_value(analysis, "recent_form_home"):
        analysis["recent_form_home"] = supplement["recent_form_home"]
    if supplement.get("recent_form_away") and not _field_value(analysis, "recent_form_away"):
        analysis["recent_form_away"] = supplement["recent_form_away"]
    if supplement.get("injury_or_lineup_notes"):
        existing = normalize_text(str(_field_value(analysis, "injury_or_lineup_notes", "") or ""))
        placeholder_tokens = COLLECTION_PLACEHOLDER_TEXTS
        if not existing or any(tok in existing for tok in placeholder_tokens):
            analysis["injury_or_lineup_notes"] = supplement["injury_or_lineup_notes"]

    # 更新来源链接
    if supplement.get("sources"):
        existing_links = unique_non_empty(
            split_lines(str(_field_value(analysis, "media_source_links", "") or ""))
        )
        existing_sources = unique_non_empty(
            split_lines(str(_field_value(analysis, "collected_sources", "") or ""))
        )
        for src in supplement["sources"]:
            if ":" in src:
                label, url = src.split(":", 1)
                url = normalize_text(url)
                label = normalize_text(label)
                if url and url not in existing_links:
                    existing_links.append(url)
                    existing_sources.append(label)
        analysis["media_source_links"] = "\n".join(existing_links)
        analysis["collected_sources"] = "\n".join(existing_sources)

    # 记录补采备注
    if supplement.get("errors"):
        existing_remarks = split_lines(str(_field_value(analysis, "remarks", "") or ""))
        for err in supplement["errors"]:
            existing_remarks.append(f"补充采集备注：{err}")
        analysis["remarks"] = "\n".join(unique_non_empty(existing_remarks))


def collect_match(match_id: str, progress_callback=None, *, fast_mode: bool = False) -> dict:
    match = get_match(match_id)
    if match is None:
        raise RuntimeError(f"未找到 match_id={match_id} 的对赛。")

    match_label = f"{match['home_team']} vs {match['away_team']}"
    _emit_progress(
        progress_callback,
        current_item_label=match_label,
        current_step="准备采集",
        message=f"准备采集：{match_label}",
    )

    _emit_progress(
        progress_callback,
        current_step="抓取 500 页面",
        message=f"正在抓取 500 基础页面：{match_label}",
    )
    shuju_html = fetch_html(match["shuju_url"])
    ouzhi_html = fetch_html(match["ouzhi_url"])
    touzhu_html = fetch_html(match["touzhu_url"])
    yazhi_url = match["yazhi_url"] if "yazhi_url" in match.keys() and match["yazhi_url"] else f"https://odds.500.com/fenxi/yazhi-{match_id}.shtml"
    yazhi_html = fetch_html(yazhi_url)

    _emit_progress(
        progress_callback,
        current_step="解析基础数据",
        message=f"正在解析维度一/三基础数据：{match_label}",
    )
    recent_home, recent_away = extract_recent_forms(shuju_html)
    h2h_summary = extract_h2h(shuju_html)
    lineup_notes = extract_lineup_notes(shuju_html)

    lineup_structured = {}
    try:
        strength_snapshot = extract_strength_snapshots(match, shuju_html)
    except Exception as exc:  # noqa: BLE001
        strength_snapshot = {
            "elo_home": "",
            "elo_away": "",
            "source_url": "",
            "source_label": "",
            "note": f"Elo 替代指标采集失败：{exc}",
        }

    media_links = [match["shuju_url"], match["ouzhi_url"], match["touzhu_url"], yazhi_url]
    if strength_snapshot["source_url"]:
        media_links.extend(strength_snapshot["source_url"].splitlines())

    collected_sources = ["500胜负彩页", "500数据分析页", "500百家欧赔页", "500投注分析页", "500亚盘页"]
    if strength_snapshot["source_url"]:
        collected_sources.append("500联赛积分页")

    remarks = ["精简展示模式：优先保留关键数据值和对比，弱化长文本描述。"]
    if strength_snapshot["note"]:
        remarks.append(strength_snapshot["note"])

    existing_analysis = get_match_analysis(match_id)
    market_value_summary = (
        str(existing_analysis["market_value_summary"] or "")
        if existing_analysis is not None and "market_value_summary" in existing_analysis.keys()
        else ""
    )

    skip_lineup_supplement = fast_mode or _env_flag("FOOTBALL_SKIP_LINEUP_SUPPLEMENT")
    if fast_mode:
        remarks.append("批量快速采集：跳过伤停/阵容外部慢补采，若维度缺失将自动补采。")
    else:
        try:
            _emit_progress(
                progress_callback,
                current_step="补采伤停/阵容",
                message=f"正在补采维度二伤停/阵容：{match_label}",
            )
            if skip_lineup_supplement:
                raise RuntimeError("FOOTBALL_SKIP_LINEUP_SUPPLEMENT=1")
            lineup_supplement = supplement_injury_or_lineup_notes(match, shuju_html)
            # Only use supplement notes if they contain actual lineup/injury data (not placeholders)
            _supplement_notes = lineup_supplement.notes or ""
            _placeholder_tokens = COLLECTION_PLACEHOLDER_TEXTS
            _has_placeholder = any(tok in _supplement_notes for tok in _placeholder_tokens)
            if _supplement_notes and not _has_placeholder:
                lineup_notes = _supplement_notes
            lineup_structured = dict(lineup_supplement.structured_data or {})
            media_links.extend(lineup_supplement.source_links)
            collected_sources.extend(lineup_supplement.source_labels)
            remarks.extend(lineup_supplement.remarks)
        except Exception as exc:  # noqa: BLE001
            # Keep primary source lineup_notes, just log the supplement failure
            remarks.append(f"伤停/阵容补采失败：{exc}")

    _emit_progress(
        progress_callback,
        current_step="写入结果",
        message=f"正在保存采集结果：{match_label}",
    )
    analysis = {
        "match_id": match["match_id"],
        "collected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "elo_home": strength_snapshot["elo_home"],
        "elo_away": strength_snapshot["elo_away"],
        "market_value_summary": market_value_summary,
        "recent_form_home": recent_home,
        "recent_form_away": recent_away,
        "home_away_form": extract_home_away_form(shuju_html),
        "head_to_head_summary": h2h_summary,
        "injury_or_lineup_notes": lineup_notes,
        "motivation_or_schedule_notes": extract_motivation(match, h2h_summary),
        "european_odds_movement_summary": build_odds_summary(match, ouzhi_html),
        "asian_handicap_summary": build_asian_handicap_summary(yazhi_html),
        "betting_heat_summary": build_heat_summary(match, touzhu_html),
        "media_source_links": "\n".join(unique_non_empty(media_links)),
        "collected_sources": "\n".join(unique_non_empty(collected_sources)),
        "collection_quality_summary": "",
        "remarks": "\n".join(unique_non_empty(remarks)),
    }
    _emit_progress(
        progress_callback,
        current_step="统一采集策略",
        message=(
            f"正在按 500 主源执行快速采集完整性检查：{match_label}"
            if fast_mode
            else f"正在按主源 -> playwright-cli -> anysearch 统一补齐采集维度：{match_label}"
        ),
    )
    required_strategy_fields = [
        field
        for fields in DIMENSION_FIELDS.values()
        for field in fields
    ]
    apply_unified_collection_strategy(
        serialize_match(match),
        analysis,
        required_fields=required_strategy_fields,
        allow_fallback=not fast_mode,
    )
    snapshot = None
    try:
        snapshot = build_feature_snapshot(
            {
                **serialize_match(match),
                **analysis,
            },
            shuju_html=shuju_html,
            lineup_data=lineup_structured or None,
        )
    except Exception as exc:  # noqa: BLE001
        analysis["remarks"] = "\n".join(
            unique_non_empty(
                split_lines(analysis["remarks"])
                + [f"特征快照构建失败：{exc}"]
            )
        )
    missing_required_dimensions = get_missing_required_dimensions(analysis)

    collection_status = "success"
    status_level = "success"
    status_message = f"已完成采集：{match_label}"

    # ── 补充采集：当主源缺失维度时，通过免费公开资源补采 ──
    if missing_required_dimensions and fast_mode:
        collection_status = "failed"
        status_level = "warning"
        missing_text = format_missing_required_dimensions(missing_required_dimensions)
        failure_remark = f"{MISSING_DIMENSION_PREFIX}{missing_text}"
        analysis["remarks"] = "\n".join(
            unique_non_empty(
                [
                    failure_remark,
                    "批量快速采集：暂未调用外部公开源补采，进入自动补采流程。",
                ]
                + [
                    line
                    for line in split_lines(analysis["remarks"])
                    if not line.startswith(COLLECTION_FAILURE_PREFIX)
                    and not line.startswith(MISSING_DIMENSION_PREFIX)
                ]
            )
        )
        status_message = f"快速采集仍缺少维度：{match_label}，缺少采集维度：{missing_text}"
    elif missing_required_dimensions:
        _emit_progress(
            progress_callback,
            current_step="补充采集",
            message=f"正在通过外部公开来源补采缺失维度：{match_label}",
        )
        _supplement_missing_dimensions(match, analysis, missing_required_dimensions)
        apply_unified_collection_strategy(
            serialize_match(match),
            analysis,
            required_fields=required_strategy_fields,
            missing_fields=missing_dimension_fields(missing_required_dimensions),
        )
        # 补采后重新检查
        missing_after_supplement = get_missing_required_dimensions(analysis)
        if not missing_after_supplement:
            # 补采成功，全部填满
            missing_text = format_missing_required_dimensions(missing_required_dimensions)
            analysis["remarks"] = "\n".join(
                unique_non_empty(
                    [f"外部公开来源补采完成，已补全维度：{missing_text}"]
                    + [
                        line
                        for line in split_lines(analysis['remarks'])
                        if not line.startswith(COLLECTION_FAILURE_PREFIX)
                        and not line.startswith(MISSING_DIMENSION_PREFIX)
                    ]
                )
            )
            status_message = f"已完成采集（含外部补采）：{match_label}"
        elif _is_only_elo_missing(missing_after_supplement):
            # 只剩 elo 缺失，可接受
            status_level = 'success'
            missing_text = format_missing_required_dimensions(missing_after_supplement)
            analysis["remarks"] = "\n".join(
                unique_non_empty(
                    [f"采集完成，部分实力代理数据不可用：{missing_text}"]
                    + [
                        line
                        for line in split_lines(analysis['remarks'])
                        if not line.startswith(COLLECTION_FAILURE_PREFIX)
                        and not line.startswith(MISSING_DIMENSION_PREFIX)
                    ]
                )
            )
            status_message = f"采集完成（部分实力代理不可用）：{match_label}"
        elif len(missing_after_supplement) < len(missing_required_dimensions):
            # 补采部分成功
            collection_status = "success"
            status_level = "success"
            still_missing = format_missing_required_dimensions(missing_after_supplement)
            analysis["remarks"] = "\n".join(
                unique_non_empty(
                    [f"外部公开来源补采部分成功，仍缺少维度：{still_missing}"]
                    + [
                        line
                        for line in split_lines(analysis['remarks'])
                        if not line.startswith(COLLECTION_FAILURE_PREFIX)
                        and not line.startswith(MISSING_DIMENSION_PREFIX)
                    ]
                )
            )
            status_message = f"采集完成（部分维度仍缺失）：{match_label}"
        else:
            # 补采完全未命中，标记失败
            collection_status = "failed"
            status_level = "warning"
            missing_text = format_missing_required_dimensions(missing_after_supplement)
            failure_remark = f"{MISSING_DIMENSION_PREFIX}{missing_text}"
            analysis["remarks"] = "\n".join(
                unique_non_empty(
                    [failure_remark]
                    + [
                        line
                        for line in split_lines(analysis['remarks'])
                        if not line.startswith(COLLECTION_FAILURE_PREFIX)
                        and not line.startswith(MISSING_DIMENSION_PREFIX)
                    ]
                )
            )
            status_message = f"采集异常：{match_label}，缺少采集维度：{missing_text}"
    save_analysis(analysis)
    if snapshot is not None:
        save_feature_snapshot(snapshot)
    _emit_progress(
        progress_callback,
        current_step="当前对赛完成" if status_level == "success" else "当前对赛异常",
        message=status_message,
        level=status_level,
    )
    result = dict(analysis)
    result.update(
        {
            "match_label": match_label,
            "collection_status": collection_status,
            "missing_required_dimensions": missing_required_dimensions,
            "status_message": status_message,
            "status_level": status_level,
            "task_message": status_message,
        }
    )
    return result


def collect_match_with_auto_retry(
    match_id: str,
    *,
    max_retries: int = 2,
    progress_callback=None,
    collector=None,
    first_attempt_collector=None,
    context_label: str = "自动补采",
) -> dict:
    attempts = max(int(max_retries or 0), 0) + 1
    last_result: dict | None = None
    last_reason = ""
    for attempt in range(1, attempts + 1):
        if attempt > 1:
            _emit_progress(
                progress_callback,
                current_step=f"{context_label}第 {attempt - 1} 次",
                message=f"采集仍缺失，正在执行{context_label}第 {attempt - 1}/{attempts - 1} 次：{match_id}",
                level="warning",
            )
        try:
            active_collector = (
                first_attempt_collector
                if attempt == 1 and first_attempt_collector is not None
                else collector
            )
            if active_collector is None:
                result = collect_match(match_id, progress_callback=progress_callback)
            else:
                result = active_collector(match_id)
        except Exception as exc:  # noqa: BLE001
            last_reason = f"{COLLECTION_FAILURE_PREFIX}{exc}"
            result = {
                "match_id": match_id,
                "collection_status": "failed",
                "remarks": last_reason,
                "reason": last_reason,
                "status_level": "warning",
                "status_message": last_reason,
            }
        last_result = dict(result or {})
        last_reason = get_collection_failure_reason(last_result)
        if not last_reason:
            retry_count = attempt - 1
            last_result["auto_retry_count"] = retry_count
            if retry_count:
                last_result["status_message"] = f"自动补采成功（已重试 {retry_count} 次）：{last_result.get('match_label') or match_id}"
                last_result["task_message"] = last_result["status_message"]
            return last_result

    final_result = last_result or {"match_id": match_id}
    final_result["collection_status"] = "failed"
    final_result["status_level"] = "warning"
    final_result["auto_retry_count"] = max(attempts - 1, 0)
    final_result["final_failure_reason"] = last_reason
    final_result["status_message"] = f"自动补采 {max(attempts - 1, 0)} 次仍失败：{last_reason}"
    final_result["task_message"] = final_result["status_message"]
    return final_result


def _collect_match_fast_then_retry_if_incomplete(
    match_id: str,
    progress_callback=None,
) -> dict:
    result = collect_match(match_id, progress_callback=progress_callback, fast_mode=True)
    missing_fields = [
        FIELD_LABELS.get(field, field)
        for field in BATCH_FAST_MODE_RETRY_FIELDS
        if not normalize_text(str(_field_value(result, field, "") or ""))
    ]
    if not missing_fields or get_collection_failure_reason(result):
        return result

    missing_text = "、".join(missing_fields)
    retry_result = dict(result)
    retry_result["collection_status"] = "failed"
    retry_result["remarks"] = "\n".join(
        unique_non_empty(
            [
                f"{COLLECTION_FAILURE_PREFIX}批量快速采集缺少补采维度：{missing_text}",
                *[
                    line
                    for line in split_lines(str(_field_value(result, "remarks", "") or ""))
                    if not line.startswith(COLLECTION_FAILURE_PREFIX)
                    and not line.startswith(MISSING_DIMENSION_PREFIX)
                ],
            ]
        )
    )
    retry_result["status_level"] = "warning"
    retry_result["status_message"] = f"快速采集仍缺少补采维度：{missing_text}"
    retry_result["task_message"] = retry_result["status_message"]
    retry_result["missing_required_dimensions"] = list(
        result.get("missing_required_dimensions", [])
    ) + missing_fields
    return retry_result


def collect_all_matches(
    issue: str | None = None,
    progress_callback=None,
    return_details: bool = False,
) -> list[dict] | dict[str, Any]:
    match_rows = list_matches_by_issue(issue)
    results = []
    failed_matches: list[dict[str, Any]] = []
    total_matches = len(match_rows)

    _emit_progress(
        progress_callback,
        current_step="准备批量采集",
        total_items=total_matches,
        completed_items=0,
        current_item_index=0,
        message=f"准备采集 {total_matches} 场对赛",
    )

    for index, row in enumerate(match_rows, start=1):
        match_id = row["match_id"]
        match_label = f"{row['home_team']} vs {row['away_team']}"

        _emit_progress(
            progress_callback,
            total_items=total_matches,
            completed_items=index - 1,
            current_item_index=index,
            current_item_label=match_label,
            current_step="进入当前场次",
            message=f"正在采集第 {index}/{total_matches} 场：{match_label}",
        )

        def _match_progress(**payload):
            merged_payload = {
                "total_items": total_matches,
                "completed_items": index - 1,
                "current_item_index": index,
                "current_item_label": match_label,
            }
            merged_payload.update(payload)
            _emit_progress(progress_callback, **merged_payload)

        try:
            match_result = collect_match_with_auto_retry(
                match_id,
                progress_callback=_match_progress,
                first_attempt_collector=lambda item: _collect_match_fast_then_retry_if_incomplete(
                    item,
                    progress_callback=_match_progress,
                ),
                context_label="自动补采",
            )
            results.append(match_result)
            failure_reason = get_collection_failure_reason(match_result)
            if failure_reason:
                failed_entry = {
                    "match_id": match_id,
                    "match_label": match_label,
                    "issue": _field_value(row, "issue", ""),
                    "status": "failed",
                    "reason": failure_reason,
                    "auto_retry_count": int(match_result.get("auto_retry_count", 0) or 0),
                    "missing_required_dimensions": match_result.get(
                        "missing_required_dimensions",
                        [],
                    ),
                }
                failed_matches.append(failed_entry)
                _emit_progress(
                    progress_callback,
                    total_items=total_matches,
                    completed_items=index,
                    current_item_index=index,
                    current_item_label=match_label,
                    current_step="当前场次异常",
                    message=f"第 {index}/{total_matches} 场采集异常：{match_label}，{failure_reason}",
                    level="warning",
                )
                continue
            _emit_progress(
                progress_callback,
                total_items=total_matches,
                completed_items=index,
                current_item_index=index,
                current_item_label=match_label,
                current_step="当前场次完成",
                message=f"已完成第 {index}/{total_matches} 场：{match_label}",
            )
        except Exception as exc:  # noqa: BLE001
            failed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            remarks = f"{COLLECTION_FAILURE_PREFIX}{exc}"
            save_failed_analysis(match_id, remarks, failed_at)
            failed_entry = {
                "match_id": match_id,
                "match_label": match_label,
                "issue": _field_value(row, "issue", ""),
                "status": "failed",
                "reason": remarks,
                "missing_required_dimensions": [],
            }
            failed_matches.append(failed_entry)
            results.append(
                {
                    "match_id": match_id,
                    "match_label": match_label,
                    "collection_status": "failed",
                    "remarks": remarks,
                    "status_level": "warning",
                    "status_message": remarks,
                    "task_message": remarks,
                }
            )

            _emit_progress(
                progress_callback,
                total_items=total_matches,
                completed_items=index,
                current_item_index=index,
                current_item_label=match_label,
                current_step="当前场次失败",
                message=f"第 {index}/{total_matches} 场采集失败：{match_label}",
                level="warning",
            )

    failed_count = len(failed_matches)
    collected_count = max(total_matches - failed_count, 0)
    if total_matches == 0:
        task_message = "当前没有可采集的对赛。"
        status_level = "warning"
    else:
        task_message = (
            f"批量采集完成：共 {total_matches} 场，成功 {collected_count} 场，异常 {failed_count} 场。"
        )
        if failed_matches:
            task_message += f"异常场次：{summarize_issue_entries(failed_matches)}。"
        status_level = "success" if failed_count == 0 else "warning"

    _emit_progress(
        progress_callback,
        total_items=total_matches,
        completed_items=total_matches,
        current_item_index=total_matches,
        current_item_label=match_rows[-1]["home_team"] + " vs " + match_rows[-1]["away_team"]
        if match_rows
        else "",
        current_step="批量采集完成",
        message=task_message,
        level=status_level,
    )
    if return_details:
        return {
            "results": results,
            "total_matches": total_matches,
            "collected_count": collected_count,
            "failed_count": failed_count,
            "failed_matches": failed_matches,
            "task_message": task_message,
            "status_message": task_message,
            "status_level": status_level,
        }
    return results


def build_sections(row: sqlite3.Row | None) -> list[dict]:
    if row is None:
        return []

    data = dict(row)
    sections = []
    section_variants = {
        "维度一：基础实力": "comparison",
        "维度二：近期动态": "insight-cards",
        "维度三：市场数据": "market-cards",
        "来源与备注": "source-notes",
    }

    for title, fields in FIELD_GROUPS:
        if title == "基础信息":
            continue
        items = []
        for field in fields:
            items.append(
                {
                    "field": field,
                    "label": FIELD_LABELS.get(field, field),
                    "value": data.get(field, "") or "",
                }
            )
        sections.append(
            {
                "title": title,
                "variant": section_variants.get(title, "kv-table"),
                "items": items,
                "source_notes_summary": build_source_notes_summary(data)
                if title == "来源与备注"
                else None,
            }
        )
    return sections


__all__ = [
    "FIELD_GROUPS",
    "FIELD_LABELS",
    "build_sections",
    "collect_all_matches",
    "collect_match",
    "collect_match_with_auto_retry",
    "format_missing_required_dimensions",
    "get_collection_stats",
    "get_collection_failure_reason",
    "get_match_analysis",
    "get_missing_required_dimensions",
    "init_db",
    "list_issues",
    "list_matches",
    "list_matches_by_issue",
    "normalize_text",
    "summarize_issue_entries",
    "serialize_match",
    "sync_issue_matches",
    "sync_matches",
]
