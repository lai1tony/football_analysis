from __future__ import annotations

import importlib.util
import shutil
import sqlite3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(__file__).resolve().parent / "football_data.db"


def status(label: str, ok: bool, detail: str = "") -> None:
    marker = "OK" if ok else "FAIL"
    suffix = f" - {detail}" if detail else ""
    print(f"{marker:4} {label}{suffix}")


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> int:
    print("Football analysis runtime check")
    print(f"root: {ROOT}")
    print(f"python: {sys.executable}")
    print(f"version: {sys.version.split()[0]}")
    print()

    required_modules = ("flask", "requests", "bs4")
    all_ok = True
    for name in required_modules:
        ok = module_available(name)
        all_ok = all_ok and ok
        status(f"python module {name}", ok)

    print()
    status("database exists", DB_PATH.exists(), str(DB_PATH))
    if DB_PATH.exists():
        try:
            with sqlite3.connect(DB_PATH) as conn:
                matches = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
                analyses = conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
            status("database readable", True, f"matches={matches}, analyses={analyses}")
        except Exception as exc:  # noqa: BLE001
            all_ok = False
            status("database readable", False, str(exc))

    print()
    for command in ("playwright-cli", "node"):
        resolved = shutil.which(command)
        status(f"command {command}", bool(resolved), resolved or "not found")

    if not all_ok:
        print()
        print("Use an explicit Python executable with project dependencies installed.")
        print(r"Example: C:\Users\15696\AppData\Local\Programs\Python\Python312\python.exe data\check_runtime.py")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
