"""
Jarvis adapter registry.

Maps --software argument to the correct adapter class.
Each adapter implements BaseAdapter so the Director works identically
regardless of which editing software is underneath.
"""

from .base import BaseAdapter
from .ffmpeg_adapter import FFmpegAdapter
from .davinci_adapter import DaVinciAdapter
from .premiere_adapter import PremiereAdapter

import sys

# Final Cut Pro is Mac-only — only import on macOS
if sys.platform == "darwin":
    from .finalcut_adapter import FinalCutAdapter
else:
    FinalCutAdapter = None

# CapCut uses VisionController directly — not a BaseAdapter subclass
# It's handled separately in main.py


ADAPTERS: dict[str, type] = {
    "ffmpeg":   FFmpegAdapter,
    "davinci":  DaVinciAdapter,
    "premiere": PremiereAdapter,
    "capcut":   None,   # Handled via VisionController + BrowserManager
}

if FinalCutAdapter:
    ADAPTERS["finalcut"] = FinalCutAdapter


def get_adapter(software: str) -> BaseAdapter:
    """Instantiate the right adapter for the given software name."""
    software = software.lower().strip()

    if software == "capcut":
        raise ValueError("CapCut uses VisionController — use main.py's browser flow instead.")

    cls = ADAPTERS.get(software)
    if cls is None:
        available = [k for k, v in ADAPTERS.items() if v is not None]
        raise ValueError(
            f"Unknown software: '{software}'. "
            f"Available: {', '.join(available)}"
        )
    return cls()
