#!/usr/bin/env python3
"""
setup.py — Jarvis Viral Video Editor — Onboarding Script
---------------------------------------------------------
Run once before first use:

    python setup.py

What it does
------------
1. Checks Python version (>= 3.10 required).
2. Checks for ffmpeg / ffprobe system binaries.
3. Checks for required Python packages.
4. Prompts for environment variables and writes (or updates) a .env file.
5. Installs the Playwright Chromium browser if playwright is available.
6. Creates required project directories.
7. Prints a colour-coded summary and a final usage hint.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Rich — optional; graceful fallback to plain print
# ---------------------------------------------------------------------------

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    _console = Console()

    def _ok(msg: str)   -> None: _console.print(f"  [bold green]✓[/bold green]  {msg}")
    def _warn(msg: str) -> None: _console.print(f"  [bold yellow]![/bold yellow]  {msg}")
    def _err(msg: str)  -> None: _console.print(f"  [bold red]✗[/bold red]  {msg}")
    def _info(msg: str) -> None: _console.print(f"  [dim]→[/dim]  {msg}")

    def _header(msg: str) -> None:
        _console.print(f"\n[bold white]{msg}[/bold white]")

    _HAS_RICH = True

except ImportError:
    _HAS_RICH  = False
    _console   = None  # type: ignore[assignment]

    def _ok(msg: str)     -> None: print(f"  [OK]   {msg}")   # type: ignore[misc]
    def _warn(msg: str)   -> None: print(f"  [WARN] {msg}")   # type: ignore[misc]
    def _err(msg: str)    -> None: print(f"  [ERR]  {msg}")   # type: ignore[misc]
    def _info(msg: str)   -> None: print(f"         {msg}")   # type: ignore[misc]
    def _header(msg: str) -> None: print(f"\n{msg}")          # type: ignore[misc]


# ---------------------------------------------------------------------------
# Project root (directory containing this file)
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent.resolve()

# ---------------------------------------------------------------------------
# Required project directories (relative to project root)
# ---------------------------------------------------------------------------

_REQUIRED_DIRS: list[str] = [
    "output/clips",
    "data/cookies",
    "data/taste_profiles",
    "data/cache",
]

# ---------------------------------------------------------------------------
# Required Python packages: (import_name, pip_install_name)
# ---------------------------------------------------------------------------

_REQUIRED_PACKAGES: list[tuple[str, str]] = [
    ("openai",      "openai"),
    ("whisper",     "openai-whisper"),
    ("cv2",         "opencv-python"),
    ("librosa",     "librosa"),
    ("playwright",  "playwright"),
    ("rich",        "rich"),
    ("dotenv",      "python-dotenv"),
]

# ---------------------------------------------------------------------------
# Environment variable definitions
# ---------------------------------------------------------------------------

_ENV_VARS: list[dict] = [
    {
        "key":      "OPENROUTER_API_KEY",
        "label":    "OpenRouter API key",
        "required": True,
        "prompt":   "OpenRouter API key (https://openrouter.ai/keys): ",
        "secret":   True,
    },
    {
        "key":      "CAPCUT_EMAIL",
        "label":    "CapCut email",
        "required": False,
        "prompt":   "CapCut account email (optional — press Enter to skip): ",
        "secret":   False,
    },
    {
        "key":      "CAPCUT_PASSWORD",
        "label":    "CapCut password",
        "required": False,
        "prompt":   "CapCut account password (optional — press Enter to skip): ",
        "secret":   True,
    },
    {
        "key":      "PIXABAY_API_KEY",
        "label":    "Pixabay API key",
        "required": False,
        "prompt":   "Pixabay API key for music matching (https://pixabay.com/api/docs/, optional): ",
        "secret":   False,
    },
    {
        "key":      "HUGGINGFACE_TOKEN",
        "label":    "HuggingFace token",
        "required": False,
        "prompt":   "HuggingFace token for speaker diarisation (optional): ",
        "secret":   True,
    },
]


# ---------------------------------------------------------------------------
# Check functions — each returns (ok: bool, message: str)
# ---------------------------------------------------------------------------

def check_python_version() -> tuple[bool, str]:
    """Verify Python >= 3.10."""
    major, minor, micro = sys.version_info[:3]
    version_str = f"{major}.{minor}.{micro}"
    if (major, minor) >= (3, 10):
        return True, f"Python {version_str}"
    return False, f"Python {version_str} — 3.10+ required"


def check_ffmpeg() -> tuple[bool, str]:
    """Check that ffmpeg is on PATH and executable."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            first_line = (result.stdout or "").splitlines()[0]
            # Trim to a readable length
            return True, first_line[:70]
        return False, "ffmpeg returned a non-zero exit code"
    except FileNotFoundError:
        return (
            False,
            "ffmpeg not found — install from https://ffmpeg.org/download.html  "
            "or: sudo apt install ffmpeg / brew install ffmpeg",
        )
    except Exception as exc:
        return False, f"ffmpeg check error: {exc}"


def check_ffprobe() -> tuple[bool, str]:
    """Check that ffprobe is on PATH (usually bundled with ffmpeg)."""
    try:
        result = subprocess.run(
            ["ffprobe", "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            first_line = (result.stdout or "").splitlines()[0]
            return True, first_line[:70]
        return False, "ffprobe returned a non-zero exit code"
    except FileNotFoundError:
        return False, "ffprobe not found (usually bundled with ffmpeg)"
    except Exception as exc:
        return False, f"ffprobe check error: {exc}"


def check_package(import_name: str, pip_name: str) -> tuple[bool, str]:
    """Try to import *import_name* and report its version."""
    try:
        mod     = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "?")
        return True, f"{pip_name} {version}"
    except ImportError:
        return False, f"{pip_name} not installed — run: pip install {pip_name}"


# ---------------------------------------------------------------------------
# .env loading and writing
# ---------------------------------------------------------------------------

def _load_env_file() -> dict[str, str]:
    """
    Parse the project-root .env file and return a dict of key → value.
    Also includes any matching values already set in the process environment.
    """
    env_path = _ROOT / ".env"
    values: dict[str, str] = {}

    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            values[key.strip()] = val.strip().strip('"').strip("'")

    # Process environment takes precedence / fills gaps
    for var in _ENV_VARS:
        k = var["key"]
        if k not in values and os.environ.get(k):
            values[k] = os.environ[k]

    return values


def _missing_required_keys(current: dict[str, str]) -> list[str]:
    return [
        v["key"]
        for v in _ENV_VARS
        if v["required"] and not current.get(v["key"])
    ]


def _prompt_value(prompt_text: str, secret: bool) -> str:
    """Prompt for a value, hiding input if *secret* is True."""
    try:
        if secret:
            import getpass
            return getpass.getpass(f"  {prompt_text}").strip()
        return input(f"  {prompt_text}").strip()
    except (KeyboardInterrupt, EOFError):
        print()
        return ""


def prompt_for_env_vars(existing: dict[str, str]) -> dict[str, str]:
    """
    Interactively prompt for any env var that is not already set.
    Returns the merged dict (old values + newly entered values).
    """
    collected: dict[str, str] = dict(existing)
    print()

    for var in _ENV_VARS:
        key    = var["key"]
        secret = var["secret"]

        if collected.get(key):
            display = "****" if secret else collected[key][:40]
            _ok(f"{key} already set  ({display})")
            continue

        value = _prompt_value(var["prompt"], secret)

        if value:
            collected[key] = value
            _ok(f"{key} saved.")
        elif var["required"]:
            _err(f"{key} is required — Jarvis cannot start without it.")
        else:
            _warn(f"{key} skipped (optional).")

    return collected


def write_env_file(values: dict[str, str]) -> None:
    """
    Write all env vars to ``<project_root>/.env``.
    Empty/unset optional values are commented out.
    """
    env_path = _ROOT / ".env"
    lines = [
        "# Jarvis configuration — generated by setup.py",
        "# Edit this file or re-run setup.py to update values.",
        "",
    ]

    for var in _ENV_VARS:
        key   = var["key"]
        label = var["label"]
        value = values.get(key, "")
        lines.append(f"# {label}")
        if value:
            lines.append(f'{key}="{value}"')
        else:
            lines.append(f"# {key}=")
        lines.append("")

    env_path.write_text("\n".join(lines), encoding="utf-8")
    _ok(f".env written → {env_path}")


# ---------------------------------------------------------------------------
# Playwright
# ---------------------------------------------------------------------------

def install_playwright_chromium() -> tuple[bool, str]:
    """Run ``playwright install chromium`` if playwright is importable."""
    try:
        importlib.import_module("playwright")
    except ImportError:
        return False, "playwright not installed — skipping browser install"

    _info("Running: playwright install chromium …")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            timeout=300,
        )
        if result.returncode == 0:
            return True, "Playwright Chromium installed successfully"
        return False, f"playwright install exited with code {result.returncode}"
    except Exception as exc:
        return False, f"playwright install failed: {exc}"


# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------

def create_directories() -> list[tuple[bool, str]]:
    """Create all required project directories, return (ok, rel_path) pairs."""
    results: list[tuple[bool, str]] = []
    for rel_path in _REQUIRED_DIRS:
        target = _ROOT / rel_path
        try:
            target.mkdir(parents=True, exist_ok=True)
            results.append((True, str(target.relative_to(_ROOT))))
        except Exception as exc:
            results.append((False, f"{rel_path}: {exc}"))
    return results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary(rows: list[tuple[bool, str, str]]) -> None:
    """Print a colour-coded summary of all checks."""
    if _HAS_RICH:
        table = Table(
            title="Jarvis Setup Summary",
            show_header=True,
            header_style="bold magenta",
            border_style="dim",
        )
        table.add_column("Status",   width=8,  justify="center")
        table.add_column("Category", width=22)
        table.add_column("Detail")

        for ok_flag, category, message in rows:
            if ok_flag:
                status_col  = "[bold green] OK [/bold green]"
                message_col = f"[green]{message}[/green]"
            else:
                status_col  = "[bold red]FAIL[/bold red]"
                message_col = f"[red]{message}[/red]"
            table.add_row(status_col, category, message_col)

        _console.print()
        _console.print(table)
    else:
        print()
        print("=" * 68)
        print("  Jarvis Setup Summary")
        print("=" * 68)
        for ok_flag, category, message in rows:
            flag = " OK " if ok_flag else "FAIL"
            print(f"  [{flag}]  {category:<22} {message}")
        print("=" * 68)


# ---------------------------------------------------------------------------
# Main setup runner
# ---------------------------------------------------------------------------

def run_setup() -> None:
    """
    Full onboarding flow:
      1. Python version check
      2. System binary checks (ffmpeg, ffprobe)
      3. Python package checks
      4. .env / environment variable prompts
      5. Playwright Chromium install
      6. Project directory creation
      7. Colour-coded summary + final usage message
    """
    summary: list[tuple[bool, str, str]] = []
    hard_errors = False

    # ------------------------------------------------------------------
    _header("Jarvis — Viral Video Editor  ·  Setup")
    _header("━" * 50)

    # ---- 1. Python version ------------------------------------------------
    _header("[ 1 / 6 ]  Python version")
    ok_flag, msg = check_python_version()
    (ok_flag and _ok or _err)(msg)
    summary.append((ok_flag, "Python", msg))
    if not ok_flag:
        hard_errors = True

    # ---- 2. System tools --------------------------------------------------
    _header("[ 2 / 6 ]  System tools  (ffmpeg, ffprobe)")
    for check_fn, label in [(check_ffmpeg, "ffmpeg"), (check_ffprobe, "ffprobe")]:
        ok_flag, msg = check_fn()
        (ok_flag and _ok or _err)(msg)
        summary.append((ok_flag, label, msg))
        if not ok_flag:
            hard_errors = True

    # ---- 3. Python packages -----------------------------------------------
    _header("[ 3 / 6 ]  Python packages")
    for import_name, pip_name in _REQUIRED_PACKAGES:
        ok_flag, msg = check_package(import_name, pip_name)
        # Missing packages are warnings, not hard errors (user may install later)
        (ok_flag and _ok or _warn)(msg)
        summary.append((ok_flag, "package", msg))

    if not _HAS_RICH:
        _warn("rich not installed — output will be plain text")
        _info("Install it with: pip install rich")

    # ---- 4. Environment variables -----------------------------------------
    _header("[ 4 / 6 ]  Environment variables  (.env)")
    current_values = _load_env_file()
    missing        = _missing_required_keys(current_values)

    if missing:
        _warn(f"Missing required key(s): {', '.join(missing)}")
        _info("Please enter the values below. Press Ctrl+C to skip (Jarvis may not work).")
        all_values = prompt_for_env_vars(current_values)
    else:
        _ok("All required env vars are already set.")
        # Still offer to fill in optional vars that are unset
        unset_optional = [
            v["key"] for v in _ENV_VARS
            if not v["required"] and not current_values.get(v["key"])
        ]
        if unset_optional:
            _info(
                f"Optional vars not set: {', '.join(unset_optional)}"
                " — run setup.py again to configure them."
            )
        all_values = current_values

    write_env_file(all_values)

    for var in _ENV_VARS:
        key      = var["key"]
        required = var["required"]
        is_set   = bool(all_values.get(key))

        if is_set:
            summary.append((True, "env var", f"{key} — set"))
        elif required:
            summary.append((False, "env var", f"{key} — MISSING (required)"))
            hard_errors = True
        else:
            summary.append((True, "env var", f"{key} — not set (optional)"))

    # ---- 5. Playwright browser --------------------------------------------
    _header("[ 5 / 6 ]  Playwright browser")
    ok_flag, msg = install_playwright_chromium()
    (ok_flag and _ok or _warn)(msg)
    summary.append((ok_flag, "playwright", msg))

    # ---- 6. Project directories -------------------------------------------
    _header("[ 6 / 6 ]  Project directories")
    dir_results = create_directories()
    for ok_flag, rel_path in dir_results:
        (ok_flag and _ok or _err)(rel_path)
        summary.append((ok_flag, "directory", rel_path))
        if not ok_flag:
            hard_errors = True

    # ---- Summary ----------------------------------------------------------
    _print_summary(summary)

    # ---- Final message ----------------------------------------------------
    if hard_errors:
        if _HAS_RICH:
            _console.print(
                Panel(
                    "[bold red]Setup completed with errors.[/bold red]\n\n"
                    "Fix the items marked [bold red]FAIL[/bold red] above, "
                    "then re-run [bold]python setup.py[/bold].",
                    border_style="red",
                    padding=(1, 4),
                )
            )
        else:
            print()
            print("  Setup completed with errors.")
            print("  Fix the FAIL items above, then re-run: python setup.py")
            print()
    else:
        if _HAS_RICH:
            _console.print(
                Panel(
                    "[bold green]Setup complete![/bold green]\n\n"
                    "Run:  [bold cyan]python main.py --opus your_video.mp4[/bold cyan]",
                    border_style="green",
                    padding=(1, 4),
                )
            )
        else:
            print()
            print("  Setup complete!")
            print("  Run: python main.py --opus your_video.mp4")
            print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_setup()
