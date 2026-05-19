"""
Jarvis AI Video Editor
======================
Autonomous CapCut video editing powered by Claude AI.

Usage:
  # Director Mode (fully autonomous — Jarvis edits by itself)
  python main.py --director clip1.mp4 clip2.mp4 --style cinematic

  # Assistant Mode (you give commands in plain English)
  python main.py

  # Voice input (requires pyaudio)
  python main.py --voice
"""

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

import anthropic

from jarvis.ai_brain import JarvisAssistant
from jarvis.capcut_controller import CapCutController
from jarvis.director import JarvisDirector

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
    banner.append("\n  AI Video Editor  •  Powered by Claude + CapCut\n", style="dim")
    console.print(Panel(banner, border_style="blue", padding=(0, 2)))


# ---------------------------------------------------------------------------
# Director Mode
# ---------------------------------------------------------------------------

async def run_director(
    media_paths: list[str],
    style: str,
    project_name: str,
    capcut: CapCutController,
    client: anthropic.Anthropic,
):
    console.print(f"\n[bold green]Director Mode[/bold green] — Jarvis will edit [bold]{len(media_paths)}[/bold] clip(s) autonomously.\n")

    for p in media_paths:
        if not os.path.exists(p):
            console.print(f"[red]File not found:[/red] {p}")
            sys.exit(1)

    director = JarvisDirector(client=client, capcut=capcut, console=console)
    plan = await director.direct(
        media_paths=media_paths,
        style_hint=style,
        project_name=project_name,
    )

    console.print("\n[bold green]Edit Complete![/bold green]")
    console.print(Panel(
        f"[bold]Project:[/bold] {plan.project_name}\n"
        f"[bold]Clips edited:[/bold] {len(plan.clips)}\n"
        f"[bold]Transitions:[/bold] {len(plan.transitions)}\n"
        f"[bold]Text overlays:[/bold] {len(plan.text_overlays)}\n"
        f"[bold]Music:[/bold] {plan.music_query or 'none'}\n"
        f"[bold]Export:[/bold] {plan.export_resolution} @ {plan.export_fps}fps\n\n"
        f"[italic]{plan.director_notes}[/italic]",
        title="Edit Summary",
        border_style="green",
    ))


# ---------------------------------------------------------------------------
# Assistant Mode
# ---------------------------------------------------------------------------

async def run_assistant(
    use_voice: bool,
    capcut: CapCutController,
    client: anthropic.Anthropic,
):
    console.print("\n[bold cyan]Assistant Mode[/bold cyan] — Tell Jarvis what to do.\n")
    console.print("[dim]Commands: 'quit' to exit, 'reset' to clear history, 'voice' to toggle mic[/dim]\n")

    jarvis = JarvisAssistant(client=client, capcut=capcut, console=console)

    voice_active = use_voice
    if voice_active:
        console.print("[yellow]Voice input active. Speak your command or type it.[/yellow]\n")

    while True:
        # Get input
        if voice_active:
            from jarvis.voice_input import listen_once
            console.print("[dim]Listening...[/dim]", end="")
            spoken = listen_once(timeout=8)
            if spoken:
                console.print(f"\r[bold]You (voice):[/bold] {spoken}")
                user_input = spoken
            else:
                console.print("\r[dim]No speech detected — type instead:[/dim] ", end="")
                user_input = Prompt.ask("")
        else:
            user_input = Prompt.ask("[bold white]You[/bold white]")

        if not user_input.strip():
            continue

        cmd = user_input.strip().lower()
        if cmd in ("quit", "exit", "bye"):
            console.print("[dim]Jarvis signing off.[/dim]")
            break
        if cmd == "reset":
            jarvis.reset()
            console.print("[dim]Conversation history cleared.[/dim]")
            continue
        if cmd == "voice":
            voice_active = not voice_active
            state = "enabled" if voice_active else "disabled"
            console.print(f"[dim]Voice input {state}.[/dim]")
            continue

        reply = await jarvis.chat(user_input)
        console.print(f"\n[bold cyan]Jarvis:[/bold cyan] {reply}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    print_banner()

    parser = argparse.ArgumentParser(description="Jarvis AI Video Editor")
    parser.add_argument(
        "--director", nargs="+", metavar="FILE",
        help="Director Mode: paths to video clips to edit autonomously.",
    )
    parser.add_argument(
        "--style", default="cinematic",
        help="Visual style for Director Mode (e.g. cinematic, vlog, music_video, energetic).",
    )
    parser.add_argument(
        "--project", default="Jarvis Edit",
        help="CapCut project name.",
    )
    parser.add_argument(
        "--voice", action="store_true",
        help="Enable voice input in Assistant Mode.",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run browser in headless mode (no visible window).",
    )
    args = parser.parse_args()

    # Validate API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[red]Error:[/red] ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # Launch CapCut browser
    headless = args.headless or os.getenv("HEADLESS", "false").lower() == "true"
    capcut = CapCutController(headless=headless)

    console.print("[dim]Starting browser and opening CapCut...[/dim]")
    await capcut.start()

    # Log in if credentials are provided
    email = os.getenv("CAPCUT_EMAIL")
    password = os.getenv("CAPCUT_PASSWORD")
    if email and password:
        console.print("[dim]Logging in to CapCut...[/dim]")
        ok = await capcut.login(email, password)
        if ok:
            console.print("[green]Logged in successfully.[/green]")
        else:
            console.print("[yellow]Auto-login failed. Please log in manually in the browser window.[/yellow]")
            if not headless:
                input("Press Enter once you have logged in...")
    else:
        console.print("[yellow]No CapCut credentials in .env — please log in manually in the browser window.[/yellow]")
        if not headless:
            input("Press Enter once you have logged in...")

    try:
        if args.director:
            await run_director(
                media_paths=args.director,
                style=args.style,
                project_name=args.project,
                capcut=capcut,
                client=client,
            )
        else:
            await run_assistant(
                use_voice=args.voice,
                capcut=capcut,
                client=client,
            )
    finally:
        await capcut.stop()


if __name__ == "__main__":
    asyncio.run(main())
