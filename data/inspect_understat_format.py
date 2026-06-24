"""二次诊断：扫已经下载的 HTML，找到 understat 新格式里的数据藏在哪里。

依赖：先跑过 inspect_understat_raw.py，
     ".cache/understat_raw_requests.html" 已存在。

不抓站，只读本地缓存。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
HTML_PATHS = [
    HERE / ".cache" / "understat_raw_requests.html",
    HERE / ".cache" / "understat_raw_playwright.html",
]


def section(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def scan(path: Path) -> None:
    section(f"扫描: {path.name}")
    if not path.exists():
        print(f"  ✗ 文件不存在")
        return
    html = path.read_text(encoding="utf-8")
    print(f"  长度: {len(html):,}")

    # 1) 所有 var XXXX = 的赋值
    print()
    print("  全部 'var XXX = ...' 赋值（最多 20 个）:")
    for m in list(re.finditer(r"var\s+(\w+)\s*=\s*(.{0,60})", html))[:20]:
        name = m.group(1)
        snippet = m.group(2).strip().replace("\n", " ")
        print(f"    var {name:20s} = {snippet[:80]}")

    # 2) 全部 JSON.parse(...) 的位置和参数前 80 字符
    print()
    print("  全部 JSON.parse(...) 调用:")
    for idx, m in enumerate(re.finditer(r"JSON\.parse\(\s*(['\"])([^'\"]{0,2000})\1", html)):
        ctx_start = max(m.start() - 100, 0)
        ctx_lines = html[ctx_start:m.start()].strip().splitlines()
        context = ctx_lines[-1] if ctx_lines else ""
        encoded_preview = m.group(2)[:80]
        print(f"    [{idx}] 上文: ...{context[-60:]}")
        print(f"        参数前 80: {encoded_preview!r}")

    # 3) 全部 \.parse 类似的（possibly 改成了不同包装）
    print()
    print("  其他可疑数据 sink（最多 8 项）:")
    suspects = [
        (r"window\.\w+\s*=\s*(\{|\[)", "window 全局赋值"),
        (r"data-teams\b", "data-teams 属性"),
        (r"data-stats\b", "data-stats 属性"),
        (r"<script\s+id=\"__NEXT_DATA__", "Next.js __NEXT_DATA__"),
        (r"window\.__INITIAL_STATE__", "Redux 初始 state"),
        (r"new Highcharts", "Highcharts 数据"),
        (r"<table[^>]*id=\"league-chemp", "联赛积分表标签"),
        (r"<table[^>]*class=\"\w*chemp\w*\"", "积分表 class"),
    ]
    for pattern, label in suspects:
        if re.search(pattern, html):
            print(f"    ✓ {label}: 命中 {pattern!r}")

    # 4) 检查是不是改成 HTML 表格 + AJAX
    print()
    table_count = len(re.findall(r"<table[^>]*>", html))
    print(f"  <table> 标签数: {table_count}")
    if table_count:
        for idx, m in enumerate(list(re.finditer(r"<table([^>]*)>", html))[:5]):
            print(f"    [{idx}] {m.group(1).strip()[:120]}")

    # 5) 找所有 fetch/ajax 端点
    print()
    print("  AJAX 端点候选:")
    for m in list(re.finditer(r"['\"]/(?:api|league|team|match|stats)[^'\"\s]*['\"]", html))[:10]:
        endpoint = m.group(0).strip("'\"")
        print(f"    {endpoint[:100]}")


def main() -> int:
    for path in HTML_PATHS:
        scan(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
