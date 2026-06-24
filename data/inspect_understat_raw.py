"""一次性诊断：搞清楚 understat 抓不到 teamsData 的根因。

用法：
    python data/inspect_understat_raw.py            # 用 requests
    python data/inspect_understat_raw.py --playwright  # 用 playwright-cli

不写任何缓存，直接打到 understat。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import requests


URL = "https://understat.com/league/EPL"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}

CHALLENGE_MARKERS = (
    ("Cloudflare", "cloudflare"),
    ("Just a moment", "just a moment"),
    ("Checking your browser", "checking your browser"),
    ("Attention Required", "attention required"),
    ("captcha", "captcha"),
    ("ddos", "ddos"),
    ("bot detection", "bot detection"),
    ("Access denied", "access denied"),
    ("403 Forbidden", "403 forbidden"),
    ("404 Not Found", "404 not found"),
    ("502 Bad Gateway", "502 bad gateway"),
    ("nginx error", "nginx error"),
    ("teamsData (我们要的)", "teamsdata"),
    ("datesData (单场维度)", "datesdata"),
    ("statisticsData", "statisticsdata"),
)


def section(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def diagnose_html(html: str, source: str) -> None:
    section(f"[{source}] HTML 诊断")
    print(f"  长度: {len(html):,} 字符")

    body_lower = html.lower()
    print()
    print("  关键标记扫描:")
    for label, needle in CHALLENGE_MARKERS:
        present = needle in body_lower
        mark = "✓" if present else " "
        print(f"    {mark} {label}")

    # 第一段
    print()
    print("  HTML 头 1500 字符:")
    print("  ----------------------------------------")
    head = html[:1500]
    for line in head.splitlines()[:30]:
        print(f"    {line[:140]}")
    print("  ----------------------------------------")

    # title 标签
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
    if title_match:
        print(f"  <title>: {title_match.group(1).strip()[:160]}")

    # h1 标签
    h1_match = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.I | re.S)
    if h1_match:
        print(f"  <h1>: {h1_match.group(1).strip()[:160]}")

    # script 数量
    script_count = len(re.findall(r"<script[\s>]", html, re.I))
    print(f"  <script> 标签数: {script_count}")

    # JSON.parse 出现次数（understat 的 var teamsData 是用 JSON.parse 包的）
    parse_count = len(re.findall(r"JSON\.parse\(", html))
    print(f"  JSON.parse() 出现次数: {parse_count}")

    # 我们的正则
    teams_match = re.search(
        r"var\s+teamsData\s*=\s*JSON\.parse\(\s*'([^']+)'\s*\)",
        html,
    )
    print(f"  teamsData 正则命中: {'是' if teams_match else '否'}")
    if not teams_match:
        # 试更宽松：var teamsData = ... 任意内容
        loose = re.search(r"var\s+teamsData\s*=", html)
        if loose:
            start = loose.start()
            print(f"  → 但找到了 'var teamsData =' 文本（位置 {start}）")
            print(f"    后续 200 字符: {html[start:start+200]!r}")

    # 写盘以便人工详细看
    out_path = Path(__file__).resolve().parent / f".cache/understat_raw_{source}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"  完整 HTML 已写到: {out_path}")


def fetch_via_requests() -> str:
    print(f"  开始 requests GET {URL}")
    s = requests.Session()
    s.trust_env = False
    response = s.get(URL, headers=HEADERS, timeout=20)
    print(f"  HTTP {response.status_code}, content-type={response.headers.get('content-type', '?')}")
    print(f"  server={response.headers.get('server', '?')}")
    print(f"  cf-ray={response.headers.get('cf-ray', '?')}（如有 cf-ray 说明是 Cloudflare）")
    response.raise_for_status()
    return response.text


def fetch_via_playwright() -> str:
    try:
        from playwright_cli_client import fetch_html_via_playwright_cli
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"playwright-cli 不可用：{exc}")
    print(f"  开始 playwright-cli GET {URL}")
    return fetch_html_via_playwright_cli(URL)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--playwright", action="store_true")
    parser.add_argument("--both", action="store_true")
    args = parser.parse_args()

    sources: list[tuple[str, callable]] = []
    if args.playwright:
        sources.append(("playwright", fetch_via_playwright))
    elif args.both:
        sources.append(("requests", fetch_via_requests))
        sources.append(("playwright", fetch_via_playwright))
    else:
        sources.append(("requests", fetch_via_requests))

    for name, fetcher in sources:
        section(f"=== 抓取通道: {name} ===")
        try:
            html = fetcher()
            diagnose_html(html, name)
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ 失败: {type(exc).__name__}: {exc}")
            continue

    return 0


if __name__ == "__main__":
    sys.exit(main())
