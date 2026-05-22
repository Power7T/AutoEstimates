"""
Jarvis adapter base class.
Every software adapter implements this interface so the Director
can drive any editing app with the same edit plan.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseAdapter(ABC):
    """Abstract interface that every software adapter must implement."""

    name: str = "base"
    platform: str = "unknown"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    async def start(self) -> None:
        """Launch / connect to the editing software."""

    @abstractmethod
    async def stop(self) -> None:
        """Disconnect / close the editing software."""

    # ------------------------------------------------------------------
    # Project
    # ------------------------------------------------------------------

    @abstractmethod
    async def create_project(self, name: str) -> dict:
        """Create a new project."""

    @abstractmethod
    async def import_media(self, file_path: str) -> dict:
        """Import a media file into the project."""

    # ------------------------------------------------------------------
    # Timeline edits
    # ------------------------------------------------------------------

    @abstractmethod
    async def trim_clip(self, clip_index: int, start: float, end: float) -> dict:
        """Trim clip to new in/out points (seconds)."""

    @abstractmethod
    async def add_text(self, text: str, position: str, start: float, duration: float, **kwargs) -> dict:
        """Add a text overlay."""

    @abstractmethod
    async def add_transition(self, clip_index: int, transition_type: str, duration: float) -> dict:
        """Add a transition between clips."""

    @abstractmethod
    async def add_music(self, query_or_path: str, volume: float) -> dict:
        """Add background music."""

    @abstractmethod
    async def apply_filter(self, filter_name: str, intensity: float) -> dict:
        """Apply a color filter."""

    @abstractmethod
    async def adjust_speed(self, clip_index: int, speed_multiplier: float) -> dict:
        """Change clip playback speed."""

    @abstractmethod
    async def adjust_volume(self, clip_index: int, volume: float) -> dict:
        """Adjust clip volume."""

    @abstractmethod
    async def export_video(self, resolution: str, fps: int, output_path: str) -> dict:
        """Export the final video."""

    # ------------------------------------------------------------------
    # Tool dispatcher (routes edit plan actions to methods above)
    # ------------------------------------------------------------------

    async def execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> dict:
        dispatch = {
            "open_project":   lambda i: self.create_project(i.get("project_name", "Jarvis Edit")),
            "import_media":   lambda i: self.import_media(i["file_path"]),
            "trim_clip":      lambda i: self.trim_clip(i.get("clip_index", 0), i["start_seconds"], i["end_seconds"]),
            "add_text":       lambda i: self.add_text(i["text"], i.get("position", "bottom"), i.get("start_seconds", 0), i.get("duration_seconds", 3)),
            "add_transition": lambda i: self.add_transition(i.get("clip_index", 0), i.get("transition_type", "fade"), i.get("duration_seconds", 0.5)),
            "add_music":      lambda i: self.add_music(i["query_or_path"], i.get("volume", 0.5)),
            "apply_filter":   lambda i: self.apply_filter(i["filter_name"], i.get("intensity", 0.7)),
            "apply_lut":      lambda i: self.apply_filter(i["lut_style"], 1.0),
            "adjust_speed":   lambda i: self.adjust_speed(i.get("clip_index", 0), i["speed_multiplier"]),
            "adjust_volume":  lambda i: self.adjust_volume(i.get("clip_index", 0), i["volume"]),
            "export_video":   lambda i: self.export_video(i.get("resolution", "1080p"), i.get("fps", 30), i.get("output_path", "output.mp4")),
        }
        handler = dispatch.get(tool_name)
        if not handler:
            return {"status": "skipped", "reason": f"{self.name} does not support tool: {tool_name}"}
        try:
            return await handler(tool_input)
        except Exception as e:
            return {"status": "error", "message": str(e)}
