#!/usr/bin/env python3
"""
Jarvis AI Video Editor — Setup Script
======================================
Checks all dependencies, prompts for API keys, and prepares the environment.

Run once before first use:
    python setup.py
"""

import os
import subprocess
import sys

# Try rich for colored output, fall back to plain print
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    console = Console()
    def ok(msg):   console.print(f"  [bold green]✓[/bold green]  {msg}")
    def warn(msg): console.print(f"  [bold yellow]![/bold yellow]  {msg}")
    def err(msg):  console.print(f"  [bold red]✗[/bold red]  {msg}")
    def info(msg): console.print(f"  [dim]→[/dim]  {msg}")
    _RICH = True
except ImportError:
    def ok(msg):   print(f"  [OK]   {msg}")
    def warn(msg): print(f"  [WARN] {msg}")
    def err(msg):  print(f"  [ERR]  {msg}")
    def info(msg): print(f"  [...]  {msg}")
    _RICH = False


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_python_version() -> tuple[bool, str]:
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 10):
        return False, f"Python {major}.{minor} — need 3.10+"
    return True, f"Python {major}.{minor}"


def check_ffmpeg() -> tuple[bool, str]:
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        line = result.stdout.splitlines()[0] if result.stdout else ""
        version = line.split("version")[1].strip().split()[0] if "version" in line else "unknown"
        return True, f"ffmpeg {version}"
    except FileNotFoundError:
        return False, "ffmpeg not found — install: sudo apt install ffmpeg  OR  brew install ffmpeg"


def check_ffprobe() -> tuple[bool, str]:
    try:
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=True)
        return True, "ffprobe available"
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False, "ffprobe not found (usually installed with ffmpeg)"


def check_package(package: str, import_name: str = None) -> tuple[bool, str]:
    name = import_name or package
    try:
        __import__(name)
        return True, f"{package} installed"
    except ImportError:
        return False, f"{package} not installed — run: pip install {package}"


def check_env_file() -> tuple[bool, str]:
    if os.path.exists(".env"):
        return True, ".env file exists"
    return False, ".env file missing — will create"


def check_openrouter_key() -> tuple[bool, str]:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if key and key != "your_key_here":
        return True, "OPENROUTER_API_KEY set"
    # Try reading from .env
    if os.path.exists(".env"):
        with open(".env") as f:
            for line in f:
                if line.startswith("OPENROUTER_API_KEY="):
                    val = line.split("=", 1)[1].strip()
                    if val and val != "your_key_here":
                        return True, "OPENROUTER_API_KEY set in .env"
    return False, "OPENROUTER_API_KEY not set (required)"


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def prompt(label: str, default: str = "", secret: bool = False) -> str:
    suffix = f" [{default}]" if default and not secret else ""
    try:
        if secret:
            import getpass
            val = getpass.getpass(f"  {label}{suffix}: ").strip()
        else:
            val = input(f"  {label}{suffix}: ").strip()
    except (KeyboardInterrupt, EOFError):
        return default
    return val if val else default


def read_env() -> dict:
    env = {}
    if os.path.exists(".env"):
        with open(".env") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip()
    return env


def write_env(env: dict):
    lines = []
    # Preserve comments from template if it exists
    if os.path.exists(".env.example"):
        with open(".env.example") as f:
            template = f.read()
    else:
        template = ""

    for k, v in env.items():
        lines.append(f"{k}={v}")

    with open(".env", "w") as f:
        if template:
            # Write template with values filled in
            result = template
            for k, v in env.items():
                import re
                result = re.sub(rf"^{k}=.*$", f"{k}={v}", result, flags=re.MULTILINE)
            f.write(result)
        else:
            f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------

REQUIRED_DIRS = [
    "output/clips",
    "data/cookies",
    "data/taste_profiles",
    "data/cache",
    "data/uploads",
]


def create_directories():
    for d in REQUIRED_DIRS:
        os.makedirs(d, exist_ok=True)
    ok(f"Created {len(REQUIRED_DIRS)} required directories")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_setup():
    print()
    if _RICH:
        from rich.text import Text
        t = Text("  Jarvis AI Video Editor — Setup", style="bold blue")
        console.print(Panel(t, border_style="blue"))
    else:
        print("=" * 50)
        print("  Jarvis AI Video Editor — Setup")
        print("=" * 50)
    print()

    # ---- System checks ----
    print("Checking system requirements...")
    print()

    all_ok = True

    checks = [
        check_python_version(),
        check_ffmpeg(),
        check_ffprobe(),
    ]
    for passed, msg in checks:
        (ok if passed else err)(msg)
        if not passed:
            all_ok = False

    print()
    print("Checking Python packages...")
    print()

    packages = [
        ("openai", None),
        ("rich", None),
        ("dotenv", "dotenv"),
        ("cv2", "cv2"),
        ("librosa", None),
        ("whisper", "whisper"),
        ("numpy", "numpy"),
        ("playwright", "playwright"),
        ("fastapi", None),
        ("uvicorn", None),
        ("jinja2", "jinja2"),
    ]

    missing_packages = []
    for pkg, imp in packages:
        passed, msg = check_package(pkg, imp)
        (ok if passed else warn)(msg)
        if not passed:
            missing_packages.append(pkg)

    if missing_packages:
        print()
        info(f"Install missing packages: pip install {' '.join(missing_packages)}")
        info("Or install everything: pip install -r requirements.txt")

    # ---- API keys ----
    print()
    print("Configuring API keys...")
    print()

    env = read_env()
    env_changed = False

    # OpenRouter (required)
    key_ok, _ = check_openrouter_key()
    if not key_ok:
        warn("OPENROUTER_API_KEY is required for AI features")
        info("Get your key at: https://openrouter.ai/keys")
        key = prompt("OPENROUTER_API_KEY", secret=True)
        if key:
            env["OPENROUTER_API_KEY"] = key
            env_changed = True
            ok("OPENROUTER_API_KEY saved")
        else:
            err("OPENROUTER_API_KEY not set — AI features will not work")
            all_ok = False
    else:
        ok("OPENROUTER_API_KEY already set")

    # CapCut (optional)
    if not env.get("CAPCUT_EMAIL"):
        info("CapCut email/password (optional — only needed for CapCut mode)")
        email = prompt("CAPCUT_EMAIL (Enter to skip)", "")
        if email:
            env["CAPCUT_EMAIL"] = email
            password = prompt("CAPCUT_PASSWORD", secret=True)
            if password:
                env["CAPCUT_PASSWORD"] = password
            env_changed = True
    else:
        ok("CapCut credentials already set")

    # Pixabay (optional)
    if not env.get("PIXABAY_API_KEY"):
        info("Pixabay API key (optional — enables auto music matching)")
        info("Free key at: https://pixabay.com/api/docs/")
        key = prompt("PIXABAY_API_KEY (Enter to skip)", "")
        if key:
            env["PIXABAY_API_KEY"] = key
            env_changed = True

    # HuggingFace (optional)
    if not env.get("HUGGINGFACE_TOKEN"):
        info("HuggingFace token (optional — enables pro speaker diarization)")
        key = prompt("HUGGINGFACE_TOKEN (Enter to skip)", "")
        if key:
            env["HUGGINGFACE_TOKEN"] = key
            env_changed = True

    if env_changed:
        write_env(env)
        ok(".env file updated")

    # ---- Playwright ----
    print()
    try:
        import playwright
        info("Installing Playwright browsers...")
        result = subprocess.run(
            ["playwright", "install", "chromium"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            ok("Playwright chromium installed")
        else:
            warn(f"Playwright install: {result.stderr[:100]}")
    except ImportError:
        warn("Playwright not installed — CapCut mode unavailable")

    # ---- Directories ----
    print()
    create_directories()

    # ---- Summary ----
    print()
    if all_ok:
        if _RICH:
            console.print(Panel(
                "[bold green]Setup complete![/bold green]\n\n"
                "Run: [bold]python main.py --opus your_video.mp4[/bold]\n"
                "Web UI: [bold]python -m uvicorn jarvis.web.app:app --reload[/bold]",
                border_style="green"
            ))
        else:
            print("Setup complete!")
            print("Run: python main.py --opus your_video.mp4")
            print("Web UI: python -m uvicorn jarvis.web.app:app --reload")
    else:
        if _RICH:
            console.print(Panel(
                "[bold yellow]Setup complete with warnings.[/bold yellow]\n"
                "Fix the items marked [red]✗[/red] above before running Jarvis.",
                border_style="yellow"
            ))
        else:
            print("Setup complete with warnings. Fix errors above before running.")
    print()


if __name__ == "__main__":
    run_setup()
