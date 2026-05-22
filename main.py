"""
Jarvis AI Video Editor
======================
Fully autonomous, viral-quality video editing powered by:
  • Gemini vision (watches and understands your clips)
  • Beat detection (every cut lands on a beat)
  • Viral templates (proven structures for Reels, TikTok, Shorts)
  • Real-time trend awareness
  • Personal taste learning
  • Vision-based CapCut control (no hardcoded selectors, self-healing)

Usage:
  # Director Mode — fully autonomous
  python main.py --director clip1.mp4 clip2.mp4 --platform instagram --style cinematic

  # Assistant Mode — give commands in plain English
  python main.py

  # Rate last edit to train Jarvis to your taste
  python main.py --rate
"""

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, FloatPrompt
from rich.table import Table
from rich.text import Text

from jarvis.config import cfg
from jarvis.browser import BrowserManager
from jarvis.vision_controller import VisionController
from jarvis.director import JarvisDirector
from jarvis.ai_brain import JarvisAssistant
from jarvis.taste_learner import TasteLearner
from jarvis.adapters import get_adapter, ADAPTERS

load_dotenv()
console = Console()


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

def print_banner():
    banner = Text()
    banner.append("  ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗\n", style="bold blue")
    banner.append("  ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝\n", style="bold blue")
    banner.append("  ██║███████║██████╔╝██║   ██║██║███████╗\n", style="bold cyan")
    banner.append("  ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║\n", style="bold cyan")
    banner.append("  ██║██║  ██║██║  ██║ ╚████╔╝ ██║███████║\n", style="bold white")
    banner.append("  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝\n", style="bold white")
    banner.append("\n  AI Video Editor  •  Viral-Quality  •  Fully Autonomous\n", style="dim")
    console.print(Panel(banner, border_style="blue", padding=(0, 2)))


# ---------------------------------------------------------------------------
# Director Mode
# ---------------------------------------------------------------------------

async def run_director(args, controller: VisionController):
    console.print("\n[bold green]Director Mode[/bold green] — Jarvis will edit your video autonomously.\n")

    for p in args.director:
        if not os.path.exists(p):
            console.print(f"[red]File not found:[/red] {p}")
            sys.exit(1)

    director = JarvisDirector(controller=controller, console=console)

    plan = await director.direct(
        media_paths=args.director,
        project_name=args.project,
        platform=args.platform,
        style_hint=args.style,
        music_path=args.music or None,
    )

    # Print edit summary
    summary = Table(title="Edit Summary", border_style="green")
    summary.add_column("Setting", style="bold")
    summary.add_column("Value")
    summary.add_row("Project", plan.project_name)
    summary.add_row("Template", plan.template_name)
    summary.add_row("Aspect ratio", plan.aspect_ratio)
    summary.add_row("Clips edited", str(len(plan.clips)))
    summary.add_row("Beat cuts", str(len(plan.beat_cut_timestamps)))
    summary.add_row("Text layers", str(len(plan.text_layers)))
    summary.add_row("Auto captions", "Yes" if plan.use_auto_captions else "No")
    summary.add_row("Music", plan.music_query or "None")
    summary.add_row("Color grade", plan.color_grade)
    summary.add_row("Export", f"{plan.export_resolution} @ {plan.export_fps}fps")
    console.print(summary)

    console.print(f"\n[italic dim]{plan.director_notes}[/italic dim]\n")

    # Taste learning — ask for rating
    if cfg.enable_taste_learning:
        console.print("[dim]Rate this edit to help Jarvis learn your taste (1=bad, 5=perfect, Enter to skip):[/dim]")
        try:
            rating_str = Prompt.ask("Rating", default="")
            if rating_str.strip():
                rating = float(rating_str)
                taste = TasteLearner()
                taste.rate_video(rating)
                console.print(f"[green]Thanks! Jarvis will remember your preferences.[/green]")
        except (ValueError, KeyboardInterrupt):
            pass

    console.print("\n[bold green]Your video is ready in CapCut![/bold green]")


# ---------------------------------------------------------------------------
# Assistant Mode
# ---------------------------------------------------------------------------

async def run_assistant(controller: VisionController):
    console.print("\n[bold cyan]Assistant Mode[/bold cyan] — Tell Jarvis what to do.\n")
    console.print("[dim]Type your editing command. Special commands: quit / reset / status[/dim]\n")

    jarvis = JarvisAssistant(controller=controller, console=console)

    while True:
        try:
            user_input = Prompt.ask("[bold white]You[/bold white]")
        except (KeyboardInterrupt, EOFError):
            break

        cmd = user_input.strip().lower()

        if not cmd:
            continue
        if cmd in ("quit", "exit", "bye"):
            console.print("[dim]Jarvis signing off.[/dim]")
            break
        if cmd == "reset":
            jarvis.reset()
            console.print("[dim]Conversation cleared.[/dim]")
            continue
        if cmd == "status":
            info = await controller.get_timeline_info()
            console.print(info)
            continue

        try:
            reply = await jarvis.chat(user_input)
            console.print(f"\n[bold cyan]Jarvis:[/bold cyan] {reply}\n")
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")


# ---------------------------------------------------------------------------
# Rating mode
# ---------------------------------------------------------------------------

def run_rating():
    taste = TasteLearner()
    console.print(f"\n[bold]Your taste profile:[/bold] {taste.get_profile_summary()}\n")
    console.print("[dim]Videos edited:[/dim]", taste.profile.videos_edited)
    if taste.profile.preferred_filters:
        console.print("[dim]Preferred filters:[/dim]", ", ".join(taste.profile.preferred_filters))
    if taste.profile.preferred_transitions:
        console.print("[dim]Preferred transitions:[/dim]", ", ".join(taste.profile.preferred_transitions))
    if taste.profile.disliked:
        console.print("[dim]Dislikes:[/dim]", ", ".join(taste.profile.disliked))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    print_banner()

    parser = argparse.ArgumentParser(description="Jarvis AI Video Editor")
    parser.add_argument("--director", nargs="+", metavar="FILE",
                        help="Director Mode: paths to video clips.")
    parser.add_argument("--platform", default="instagram",
                        choices=["instagram", "tiktok", "youtube", "universal"],
                        help="Target platform for viral optimization.")
    parser.add_argument("--style", default="cinematic",
                        help="Visual style (cinematic, vlog, hype, music_video, documentary).")
    parser.add_argument("--project", default="Jarvis Edit", help="CapCut project name.")
    parser.add_argument("--music", metavar="FILE", help="Custom music file to use.")
    parser.add_argument("--headless", action="store_true", help="Run browser in background.")
    parser.add_argument("--rate", action="store_true", help="View/update your taste profile.")
    available_sw = [k for k, v in ADAPTERS.items() if v is not None] + ["capcut"]
    parser.add_argument("--software", default="capcut",
                        choices=available_sw,
                        help=f"Editing software to use. Options: {', '.join(available_sw)}")
    args = parser.parse_args()

    # Rating mode — no browser needed
    if args.rate:
        run_rating()
        return

    # Validate config
    try:
        cfg.validate()
    except ValueError as e:
        console.print(f"[red]Config error:[/red] {e}")
        sys.exit(1)

    # Override headless from arg
    if args.headless:
        cfg.headless = True

    software = args.software.lower()

    # ----------------------------------------------------------------
    # Non-CapCut adapters — no browser needed
    # ----------------------------------------------------------------
    if software != "capcut":
        console.print(f"[dim]Starting {software} adapter...[/dim]")
        try:
            adapter = get_adapter(software)
            await adapter.start()
        except Exception as e:
            console.print(f"[red]Could not start {software}:[/red] {e}")
            sys.exit(1)

        console.print(f"[green]{software.title()} ready.[/green]\n")

        try:
            if args.director:
                await run_director(args, adapter)
            else:
                console.print(f"[yellow]Assistant Mode is optimised for CapCut.\n"
                              f"For {software}, use Director Mode: --director your_clips[/yellow]")
        finally:
            await adapter.stop()
        return

    # ----------------------------------------------------------------
    # CapCut — browser-based
    # ----------------------------------------------------------------
    console.print("[dim]Starting stealth browser...[/dim]")
    browser = BrowserManager(headless=cfg.headless)
    page = await browser.start()

    console.print("[dim]Connecting to CapCut...[/dim]")
    logged_in = await browser.ensure_logged_in(
        email=cfg.capcut_email,
        password=cfg.capcut_password,
    )
    if not logged_in:
        console.print("[red]Could not log in to CapCut.[/red]")
        await browser.stop()
        sys.exit(1)

    console.print("[green]Connected to CapCut.[/green]\n")
    controller = VisionController(page)

    try:
        if args.director:
            await run_director(args, controller)
        else:
            await run_assistant(controller)
    finally:
        await browser.save_session()
        await browser.stop()


if __name__ == "__main__":
    asyncio.run(main())
