import re
import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0"}
HTTP = requests.Session()
HTTP.trust_env = False

r = HTTP.get("https://trade.500.com/sfc/", headers=HEADERS, timeout=30)
r.encoding = "gb2312"
soup = BeautifulSoup(r.text, "html.parser")
table = soup.find("table", {"id": "vsTable"})
rows = table.find_all("tr", class_="bet-tb-tr") if table else []
print(f"rows={len(rows)}")
for i, row in enumerate(rows, 1):
    evt = (
        row.find("td", class_="td-evt").get_text(strip=True)
        if row.find("td", class_="td-evt")
        else ""
    )
    tm = (
        row.find("td", class_="td-endtime").get_text(strip=True)
        if row.find("td", class_="td-endtime")
        else ""
    )
    team_td = row.find("td", class_="td-team")
    home = (
        team_td.find("a", class_="team-l").get_text(strip=True)
        if team_td and team_td.find("a", class_="team-l")
        else ""
    )
    away = (
        team_td.find("a", class_="team-r").get_text(strip=True)
        if team_td and team_td.find("a", class_="team-r")
        else ""
    )
    href = ""
    data_td = row.find("td", class_="td-data")
    if data_td:
        a = data_td.find("a", href=True)
        if a:
            href = a["href"]
    m = re.search(r"shuju-(\d+)\.shtml", href)
    match_id = m.group(1) if m else ""
    print(i, evt, tm, home, "vs", away, "match_id=", match_id)
