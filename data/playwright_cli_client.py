import atexit
import json
import os
import shutil
import subprocess
import threading
import uuid
from dataclasses import dataclass


def _read_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _read_int_env(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got: {value}") from exc


def _resolve_cli_bin() -> str:
    cli_bin = os.getenv("PLAYWRIGHT_CLI_BIN", "").strip()
    if cli_bin:
        return cli_bin

    candidates = ["playwright-cli"]
    if os.name == "nt":
        candidates = ["playwright-cli.cmd", "playwright-cli.exe", "playwright-cli"]

    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved

    raise RuntimeError(
        "playwright-cli command not found. Install @playwright/cli or set "
        "PLAYWRIGHT_CLI_BIN to the executable path."
    )


@dataclass
class PlaywrightCliSettings:
    cli_bin: str
    headed: bool
    wait_ms: int
    timeout_ms: int
    session_name: str

    @classmethod
    def from_env(cls) -> "PlaywrightCliSettings":
        session_prefix = (
            os.getenv("PLAYWRIGHT_CLI_SESSION_PREFIX", "").strip()
            or "football-analysis"
        )
        return cls(
            cli_bin=_resolve_cli_bin(),
            headed=_read_bool_env("PLAYWRIGHT_CLI_HEADED", True),
            wait_ms=_read_int_env("PLAYWRIGHT_CLI_WAIT_MS", 800),
            timeout_ms=_read_int_env("PLAYWRIGHT_CLI_TIMEOUT_MS", 120000),
            session_name=f"{session_prefix}-{os.getpid()}-{uuid.uuid4().hex[:8]}",
        )


class PlaywrightCliBrowser:
    def __init__(self, settings: PlaywrightCliSettings) -> None:
        self._settings = settings
        self._is_open = False

    def _run(self, *args: str, raw: bool = False) -> str:
        command = [self._settings.cli_bin, f"-s={self._settings.session_name}"]
        if raw:
            command.append("--raw")
        command.extend(args)

        completed = subprocess.run(
            command,
            capture_output=True,
            timeout=max(1, self._settings.timeout_ms // 1000),
            check=False,
        )
        stdout = completed.stdout.decode("utf-8", errors="replace").strip()
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()

        if completed.returncode != 0:
            details = stderr or stdout or f"exit code {completed.returncode}"
            raise RuntimeError(
                f"playwright-cli failed for session {self._settings.session_name}: "
                f"{details}"
            )
        return stdout

    def _open(self, url: str) -> None:
        args = ["open", url]
        if self._settings.headed:
            args.append("--headed")
        self._run(*args)
        self._is_open = True

    def _navigate(self, url: str) -> None:
        if not self._is_open:
            self._open(url)
            return

        try:
            self._run("goto", url)
        except RuntimeError as exc:
            if "is not open" not in str(exc):
                raise
            self._is_open = False
            self._open(url)

    def _wait_for_page(self) -> None:
        wait_code = (
            "async page => { "
            "await page.waitForLoadState('domcontentloaded'); "
            f"await page.waitForTimeout({self._settings.wait_ms}); "
            "}"
        )
        self._run("run-code", wait_code)

    def _eval(self, expression: str):
        value = self._run("eval", expression, raw=True)
        if not value:
            return ""
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    def open(self, url: str) -> None:
        self._open(url)
        self._wait_for_page()

    def goto(self, url: str) -> None:
        self._navigate(url)
        self._wait_for_page()

    def wait_for_page(self) -> None:
        self._wait_for_page()

    def eval(self, expression: str):
        return self._eval(expression)

    def title(self) -> str:
        return str(self._eval("document.title") or "")

    def page_url(self) -> str:
        return str(self._eval("window.location.href") or "")

    def body_text(self) -> str:
        value = self._eval("document.body.innerText")
        if isinstance(value, str):
            return value
        return str(value or "")

    def fetch_html(self, url: str) -> str:
        self.goto(url)

        title = self.title()
        html = self._eval("document.documentElement.outerHTML")
        if not isinstance(html, str):
            raise RuntimeError(
                "playwright-cli returned a non-string HTML payload for "
                f"{url}: {type(html).__name__}"
            )

        if title == "403 Forbidden" and not self._settings.headed:
            raise RuntimeError(
                "playwright-cli received a 403 page. trade.500.com currently "
                "requires headed browser mode; set PLAYWRIGHT_CLI_HEADED=1."
            )
        return html

    def close(self) -> None:
        if not self._is_open:
            return
        try:
            self._run("close")
        except RuntimeError:
            pass
        finally:
            self._is_open = False


_BROWSER_LOCK = threading.Lock()
_BROWSER: PlaywrightCliBrowser | None = None


def fetch_html_via_playwright_cli(url: str) -> str:
    global _BROWSER
    with _BROWSER_LOCK:
        if _BROWSER is None:
            _BROWSER = PlaywrightCliBrowser(PlaywrightCliSettings.from_env())
        return _BROWSER.fetch_html(url)


def close_playwright_cli_browser() -> None:
    global _BROWSER
    with _BROWSER_LOCK:
        if _BROWSER is None:
            return
        _BROWSER.close()
        _BROWSER = None


atexit.register(close_playwright_cli_browser)
