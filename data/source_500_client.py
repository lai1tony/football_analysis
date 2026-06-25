import os
import re
from datetime import datetime
from time import sleep
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from playwright_cli_client import fetch_html_via_playwright_cli


SOURCE_URL = "https://trade.500.com/sfc/"
ISSUE_SOURCE_URL_TEMPLATE = "https://trade.500.com/sfc/?expect={issue}"
RESULT_SOURCE_URL_TEMPLATE = "https://trade.500.com/rj/?expect={issue}"
LIVE_SELECTABLE_URL = "https://live.500.com/2h1.php"
SFC_RESULTS_INDEX_URL = "https://kaijiang.500.com/sfc.shtml"
RESULT_PAGE_BASE_URL = "https://odds.500.com/fenxi/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

HTTP = requests.Session()
HTTP.trust_env = False


def resolve_scraper_backend(backend: str | None = None) -> str:
    selected = (backend or os.getenv("FOOTBALL_SCRAPER_BACKEND", "requests")).strip()
    selected = selected.lower() or "requests"
    if selected not in {"requests", "playwright-cli"}:
        raise RuntimeError(
            "Unsupported FOOTBALL_SCRAPER_BACKEND. Use 'requests' or "
            f"'playwright-cli', got: {selected}"
        )
    return selected


def fetch_html_via_requests(url: str) -> str:
    attempts = 3
    last_error = None
    for idx in range(attempts):
        try:
            response = HTTP.get(url, headers=HEADERS, timeout=30)
            if response.status_code == 429:
                raise requests.HTTPError("429 Too Many Requests", response=response)
            response.raise_for_status()
            return response.content.decode("gb18030", errors="ignore")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if idx < attempts - 1:
                sleep(1.2 * (idx + 1))
            else:
                raise
    raise RuntimeError(f"抓取失败: {url}; {last_error}")


def fetch_html(url: str, backend: str | None = None) -> str:
    selected_backend = resolve_scraper_backend(backend)
    if selected_backend == "requests":
        return fetch_html_via_requests(url)
    return fetch_html_via_playwright_cli(url)


def extract_issue(html: str) -> str:
    selected = re.search(
        r"<option\b(?=[^>]*\bselected\b)(?=[^>]*\bvalue=[\"'](\d+)[\"'])[^>]*>",
        html,
    )
    if selected:
        return selected.group(1)
    fallback = re.search(r"expect=(\d+)", html)
    return fallback.group(1) if fallback else ""


def split_data_attr(value: str) -> tuple[str, str, str]:
    parts = (value or "").split(",")
    padded = (parts + ["", "", ""])[:3]
    return tuple(item.strip() for item in padded)


def issue_source_url(issue: str) -> str:
    issue_text = str(issue or "").strip()
    if not issue_text:
        raise RuntimeError("issue 不能为空")
    if not issue_text.isdigit():
        raise RuntimeError("issue 必须是数字期号")
    return ISSUE_SOURCE_URL_TEMPLATE.format(issue=issue_text)


def parse_match_list_html(
    html: str,
    *,
    requested_issue: str = "",
    source_url: str = "",
) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", {"id": "vsTable"})
    if table is None:
        raise RuntimeError("未找到对阵表")

    issue = extract_issue(html)
    expected_issue = str(requested_issue or "").strip()
    if expected_issue:
        if not issue:
            raise RuntimeError(f"未能确认期号 {expected_issue} 的对阵表")
        if issue != expected_issue:
            raise RuntimeError(f"返回期号 {issue} 与请求期号 {expected_issue} 不一致")

    sync_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    matches = []
    for row in table.find_all("tr", class_="bet-tb-tr"):
        data_td = row.find("td", class_="td-data")
        data_link = data_td.find("a", href=True) if data_td else None
        href = data_link["href"] if data_link else ""
        match_obj = re.search(r"shuju-(\d+)\.shtml", href)
        if not match_obj:
            continue

        team_td = row.find("td", class_="td-team")
        home = (
            team_td.find("a", class_="team-l").get_text(strip=True) if team_td else ""
        )
        away = (
            team_td.find("a", class_="team-r").get_text(strip=True) if team_td else ""
        )
        odds_win, odds_draw, odds_loss = split_data_attr(row.get("data-bjpl", ""))
        heat_win, heat_draw, heat_loss = split_data_attr(row.get("data-pjgl", ""))
        match_id = match_obj.group(1)

        matches.append(
            {
                "match_id": match_id,
                "issue": issue,
                "league": row.find("td", class_="td-evt").get_text(strip=True),
                "match_no": row.find("td", class_="td-no").get_text(strip=True),
                "match_time": row.find("td", class_="td-endtime").get_text(strip=True),
                "home_team": home,
                "away_team": away,
                "source_match_url": source_url or SOURCE_URL
                if not issue
                else issue_source_url(issue),
                "shuju_url": f"https://odds.500.com/fenxi/shuju-{match_id}.shtml",
                "ouzhi_url": f"https://odds.500.com/fenxi/ouzhi-{match_id}.shtml?ctype=2",
                "touzhu_url": f"https://odds.500.com/fenxi/touzhu-{match_id}.shtml",
                "yazhi_url": f"https://odds.500.com/fenxi/yazhi-{match_id}.shtml",
                "list_odds_win": odds_win,
                "list_odds_draw": odds_draw,
                "list_odds_loss": odds_loss,
                "list_heat_win": heat_win,
                "list_heat_draw": heat_draw,
                "list_heat_loss": heat_loss,
                "sync_time": sync_time,
            }
        )
    return matches


def fetch_current_matches() -> list[dict]:
    html = fetch_html(SOURCE_URL)
    return parse_match_list_html(html, source_url=SOURCE_URL)


def fetch_issue_matches(issue: str) -> list[dict]:
    url = issue_source_url(issue)
    html = fetch_html(url)
    return parse_match_list_html(html, requested_issue=str(issue or "").strip(), source_url=url)


def _normalize_live_match_time(value: str) -> str:
    text = re.sub(r"\s+", " ", (value or "").replace("\xa0", " ")).strip()
    if not text:
        return ""
    if re.search(r"\d{4}-\d{1,2}-\d{1,2}", text):
        return text
    if re.search(r"^\d{1,2}-\d{1,2}\b", text):
        return f"{datetime.now().year}-{text}"
    return text


def _live_team_name(cell) -> str:
    if cell is None:
        return ""
    links = cell.find_all("a")
    if links:
        return links[-1].get_text(strip=True)
    text = cell.get_text(" ", strip=True)
    return re.sub(r"\[[^\]]+\]", "", text).strip()


def parse_live_selectable_matches_html(
    html: str,
    *,
    issue: str = "",
    source_url: str = LIVE_SELECTABLE_URL,
) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    sync_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    issue_text = str(issue or "").strip()
    matches: list[dict] = []
    seen: set[str] = set()

    for row in soup.find_all("tr"):
        match_id = str(row.get("fid") or "").strip()
        if not match_id:
            checkbox = row.find("input", attrs={"name": re.compile(r"check_id")})
            match_id = str(checkbox.get("value") or "").strip() if checkbox else ""
        if not match_id or match_id in seen:
            continue

        shuju_link = row.find("a", href=re.compile(r"shuju-\d+\.shtml"))
        shuju_href = str(shuju_link.get("href") or "") if shuju_link else ""
        shuju_match = re.search(r"shuju-(\d+)\.shtml", shuju_href)
        if shuju_match:
            match_id = shuju_match.group(1)
        if not shuju_match and not re.fullmatch(r"\d+", match_id):
            continue

        cells = row.find_all("td")
        if len(cells) < 8:
            continue
        league = cells[1].get_text(" ", strip=True) if len(cells) > 1 else ""
        match_time = _normalize_live_match_time(cells[3].get_text(" ", strip=True) if len(cells) > 3 else "")
        home = _live_team_name(cells[5] if len(cells) > 5 else None)
        away = _live_team_name(cells[7] if len(cells) > 7 else None)
        if not home or not away:
            continue

        match_no = str(9000 + len(matches) + 1)
        seen.add(match_id)
        matches.append(
            {
                "match_id": match_id,
                "issue": issue_text,
                "league": league,
                "match_no": match_no,
                "match_time": match_time,
                "home_team": home,
                "away_team": away,
                "source_match_url": source_url,
                "shuju_url": f"https://odds.500.com/fenxi/shuju-{match_id}.shtml",
                "ouzhi_url": f"https://odds.500.com/fenxi/ouzhi-{match_id}.shtml?ctype=2",
                "touzhu_url": f"https://odds.500.com/fenxi/touzhu-{match_id}.shtml",
                "yazhi_url": f"https://odds.500.com/fenxi/yazhi-{match_id}.shtml",
                "list_odds_win": "",
                "list_odds_draw": "",
                "list_odds_loss": "",
                "list_heat_win": "",
                "list_heat_draw": "",
                "list_heat_loss": "",
                "sync_time": sync_time,
            }
        )
    return matches


def fetch_live_selectable_matches(issue: str = "") -> list[dict]:
    html = fetch_html(LIVE_SELECTABLE_URL)
    return parse_live_selectable_matches_html(
        html,
        issue=str(issue or "").strip(),
        source_url=LIVE_SELECTABLE_URL,
    )


def parse_sfc_issue_sequence_html(html: str) -> list[str]:
    issues = {
        issue
        for issue in re.findall(r"/shtml/sfc/(\d{5})\.shtml", html or "")
        if issue.isdigit()
    }
    return sorted(issues, key=int)


def fetch_sfc_issue_sequence() -> list[str]:
    html = fetch_html(SFC_RESULTS_INDEX_URL)
    return parse_sfc_issue_sequence_html(html)


def issue_result_source_url(issue: str) -> str:
    issue_text = str(issue or "").strip()
    if not issue_text:
        raise RuntimeError("issue 不能为空")
    return RESULT_SOURCE_URL_TEMPLATE.format(issue=issue_text)


def _parse_result_score(score_text: str) -> tuple[str, str]:
    match = re.search(r"(\d+)\s*[:\-]\s*(\d+)", score_text or "")
    if not match:
        return "", ""

    home_score = int(match.group(1))
    away_score = int(match.group(2))
    if home_score > away_score:
        actual_result = "home"
    elif home_score < away_score:
        actual_result = "away"
    else:
        actual_result = "draw"
    return f"{home_score}-{away_score}", actual_result


def _extract_result_page_url(row) -> str:
    data_td = row.find("td", class_="td-data")
    if data_td is None:
        return ""

    for link in data_td.find_all("a", href=True):
        href = str(link["href"] or "").strip()
        if re.search(r"shuju-(\d+)\.shtml", href):
            if href.startswith("http://") or href.startswith("https://"):
                return href
            if href.startswith("//"):
                return f"https:{href}"
            return urljoin(RESULT_PAGE_BASE_URL, href)
    return ""


def _parse_result_score_from_shuju_html(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    score_node = soup.select_one("p.odds_hd_bf strong")
    score_text = score_node.get_text(" ", strip=True) if score_node else ""
    return _parse_result_score(score_text)


def _fetch_result_from_shuju_url(url: str) -> tuple[str, str]:
    if not str(url or "").strip():
        return "", ""
    html = fetch_html(url)
    return _parse_result_score_from_shuju_html(html)


def fetch_result_from_match_url(url: str) -> dict:
    url_text = str(url or "").strip()
    if not url_text:
        return {}
    match_obj = re.search(r"shuju-(\d+)\.shtml", url_text)
    actual_score, actual_result = _fetch_result_from_shuju_url(url_text)
    if not actual_score or not actual_result:
        return {}
    return {
        "match_id": match_obj.group(1) if match_obj else "",
        "issue": "",
        "home_team": "",
        "away_team": "",
        "actual_score": actual_score,
        "actual_result": actual_result,
        "result_status": "settled",
        "result_source_url": url_text,
    }


def parse_issue_results_html(html: str, *, issue: str = "") -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    table = (
        soup.find("table", id="vsTable")
        or soup.find("table", class_="bet-tb-dg")
        or soup.find("table", class_="bet-tb")
    )
    if table is None:
        raise RuntimeError("未找到赛果对阵表")

    resolved_issue = str(issue or "").strip() or extract_issue(html)
    results: list[dict] = []
    for row in table.find_all("tr", class_="bet-tb-tr"):
        result_page_url = _extract_result_page_url(row)
        match_obj = re.search(r"shuju-(\d+)\.shtml", result_page_url)
        if not match_obj:
            continue

        team_td = row.find("td", class_="td-team")
        if team_td is None:
            continue
        score_node = team_td.find("i", class_="team-vs")
        score_text = score_node.get_text(" ", strip=True) if score_node else ""
        actual_score, actual_result = _parse_result_score(score_text)
        result_source_url = issue_result_source_url(resolved_issue)
        if (not actual_score or not actual_result) and result_page_url:
            try:
                actual_score, actual_result = _fetch_result_from_shuju_url(result_page_url)
            except Exception:  # noqa: BLE001
                actual_score, actual_result = "", ""
            if actual_score and actual_result:
                result_source_url = result_page_url
        if not actual_score or not actual_result:
            continue

        home_node = team_td.find("a", class_="team-l")
        away_node = team_td.find("a", class_="team-r")
        results.append(
            {
                "match_id": match_obj.group(1),
                "issue": resolved_issue,
                "home_team": home_node.get_text(strip=True) if home_node else "",
                "away_team": away_node.get_text(strip=True) if away_node else "",
                "actual_score": actual_score,
                "actual_result": actual_result,
                "result_status": "settled",
                "result_source_url": result_source_url,
            }
        )
    return results


def fetch_issue_results(issue: str) -> list[dict]:
    issue_text = str(issue or "").strip()
    html = fetch_html(issue_result_source_url(issue_text))
    return parse_issue_results_html(html, issue=issue_text)
