"""
DaVinci Resolve Adapter — official Python scripting API.

DaVinci Resolve ships with a Python scripting bridge that lets you
control it programmatically with zero screen scraping.

Requirements:
  - DaVinci Resolve installed (free version works)
  - Resolve must be RUNNING when Jarvis starts
  - Resolve scripting must be enabled:
      Preferences → General → Enable external scripting using local network

How it works:
  Resolve exposes a Python module (DaVinciResolveScript) that talks
  to the running Resolve process via a local socket.
  We import that module and drive Resolve directly.
"""

import os
import sys
from typing import Any

from .base import BaseAdapter


# Resolve scripting module paths per platform
RESOLVE_SCRIPT_PATHS = {
    "darwin": [
        "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting/Modules",
    ],
    "win32": [
        r"C:\ProgramData\Blackmagic Design\DaVinci Resolve\Support\Developer\Scripting\Modules",
    ],
    "linux": [
        "/opt/resolve/Developer/Scripting/Modules",
        "/home/resolve/Developer/Scripting/Modules",
    ],
}


def _add_resolve_to_path():
    """Add Resolve's scripting module to sys.path so it can be imported."""
    platform = sys.platform
    paths = RESOLVE_SCRIPT_PATHS.get(platform, [])
    for path in paths:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
            return True
    return False


COLOR_GRADES = {
    "cinematic":     {"contrast": 0.08, "saturation": 0.9, "lift_r": -0.02, "gain_b": 0.04},
    "orange_teal":   {"contrast": 0.1,  "saturation": 1.1, "lift_b": -0.05, "gain_r": 0.05},
    "warm":          {"contrast": 0.05, "saturation": 1.0, "gain_r": 0.04,  "gain_b": -0.03},
    "cool":          {"contrast": 0.05, "saturation": 1.0, "gain_r": -0.03, "gain_b": 0.04},
    "moody":         {"contrast": 0.15, "saturation": 0.85,"lift_r": -0.03, "lift_g": -0.02},
    "vintage":       {"contrast": 0.05, "saturation": 0.75,"lift_r": 0.04,  "gain_b": -0.04},
    "bright":        {"contrast": -0.05,"saturation": 1.15,"gain_r": 0.03,  "gain_g": 0.02},
    "black_white":   {"saturation": 0.0, "contrast": 0.1},
    "high_contrast": {"contrast": 0.25, "saturation": 1.1},
}

TRANSITION_MAP = {
    "fade":       "Cross Dissolve",
    "dissolve":   "Cross Dissolve",
    "wipe_left":  "Wipe",
    "wipe_right": "Wipe",
    "zoom_in":    "Zoom",
    "flash":      "Flash",
    "glitch":     "Glitch",
    "whip_pan":   "Blur Dissolve",
    "morph":      "Morph Cut",
}


class DaVinciAdapter(BaseAdapter):
    """
    Controls DaVinci Resolve via its official Python scripting API.
    Resolve must be running before calling start().
    """

    name = "davinci"
    platform = "desktop"

    def __init__(self):
        self._resolve = None
        self._project = None
        self._media_pool = None
        self._timeline = None
        self._clips: list[Any] = []   # MediaPoolItem objects

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Connect to running DaVinci Resolve instance."""
        _add_resolve_to_path()
        try:
            import DaVinciResolveScript as dvr
            self._resolve = dvr.scriptapp("Resolve")
        except ImportError:
            raise RuntimeError(
                "DaVinci Resolve scripting module not found.\n"
                "Make sure Resolve is installed and scripting is enabled:\n"
                "  Preferences → General → Enable external scripting using local network"
            )

        if not self._resolve:
            raise RuntimeError(
                "Could not connect to DaVinci Resolve.\n"
                "Make sure Resolve is open and running."
            )

        pm = self._resolve.GetProjectManager()
        self._project = pm.GetCurrentProject()
        if not self._project:
            self._project = pm.CreateProject("Jarvis Edit")
        self._media_pool = self._project.GetMediaPool()

    async def stop(self) -> None:
        pass  # Resolve keeps running — we just disconnect

    # ------------------------------------------------------------------
    # Project
    # ------------------------------------------------------------------

    async def create_project(self, name: str) -> dict:
        pm = self._resolve.GetProjectManager()
        # Close current, create new
        pm.SaveProject()
        project = pm.CreateProject(name)
        if project:
            self._project = project
            self._media_pool = project.GetMediaPool()
            self._clips = []
            return {"status": "created", "project": name}
        # Project might already exist — open it
        project = pm.LoadProject(name)
        if project:
            self._project = project
            self._media_pool = project.GetMediaPool()
            return {"status": "opened", "project": name}
        return {"status": "error", "message": f"Could not create project: {name}"}

    async def import_media(self, file_path: str) -> dict:
        if not os.path.exists(file_path):
            return {"status": "error", "message": f"File not found: {file_path}"}

        root_folder = self._media_pool.GetRootFolder()
        items = self._media_pool.ImportMedia([file_path])
        if items:
            self._clips.append(items[0])
            return {"status": "imported", "file": file_path, "clip_index": len(self._clips) - 1}
        return {"status": "error", "message": "Import failed"}

    # ------------------------------------------------------------------
    # Timeline
    # ------------------------------------------------------------------

    def _ensure_timeline(self) -> bool:
        """Create a timeline if one doesn't exist."""
        self._timeline = self._project.GetCurrentTimeline()
        if not self._timeline:
            self._timeline = self._media_pool.CreateEmptyTimeline("Jarvis Timeline")
            self._project.SetCurrentTimeline(self._timeline)
        return self._timeline is not None

    async def _append_clips_to_timeline(self):
        """Add all imported media to the timeline if not already there."""
        if not self._ensure_timeline():
            return
        if self._clips:
            self._media_pool.AppendToTimeline(self._clips)

    # ------------------------------------------------------------------
    # Edits
    # ------------------------------------------------------------------

    async def trim_clip(self, clip_index: int, start: float, end: float) -> dict:
        if not self._ensure_timeline():
            return {"status": "error", "message": "No timeline"}

        items = self._timeline.GetItemListInTrack("video", 1)
        if not items or clip_index >= len(items):
            return {"status": "error", "message": f"Clip {clip_index} not found"}

        item = items[clip_index]
        fps = self._timeline.GetSetting("timelineFrameRate") or 30
        item.SetProperty("Start", int(start * fps))
        item.SetProperty("End", int(end * fps))
        return {"status": "trimmed", "clip": clip_index}

    async def add_text(
        self,
        text: str,
        position: str = "bottom",
        start: float = 0,
        duration: float = 3,
        font_size: str = "medium",
        color: str = "white",
        **kwargs,
    ) -> dict:
        if not self._ensure_timeline():
            return {"status": "error"}

        # Add a Fusion Text+ title to the timeline
        fps = self._timeline.GetSetting("timelineFrameRate") or 30
        title_item = self._media_pool.GetRootFolder().GetClipList()

        # Use Resolve's built-in title generator
        generator = self._media_pool.CreateTimelineItem("Fusion Title", int(start * fps), int(duration * fps))
        if generator:
            generator.SetClipProperty("Clip Name", text)

        return {"status": "added", "text": text}

    async def add_transition(self, clip_index: int, transition_type: str, duration: float) -> dict:
        if not self._ensure_timeline():
            return {"status": "error"}

        items = self._timeline.GetItemListInTrack("video", 1)
        if not items or clip_index >= len(items):
            return {"status": "skipped"}

        resolve_name = TRANSITION_MAP.get(transition_type, "Cross Dissolve")
        fps = self._timeline.GetSetting("timelineFrameRate") or 30
        self._timeline.AddTransition(resolve_name, items[clip_index], int(duration * fps))
        return {"status": "added", "transition": transition_type}

    async def add_music(self, query_or_path: str, volume: float = 0.5) -> dict:
        if not os.path.exists(query_or_path):
            return {"status": "skipped", "reason": "Only local files supported in DaVinci adapter"}

        if not self._ensure_timeline():
            return {"status": "error"}

        items = self._media_pool.ImportMedia([query_or_path])
        if items:
            self._media_pool.AppendToTimeline([{
                "mediaPoolItem": items[0],
                "trackIndex": 1,
                "recordFrame": 0,
            }])
        return {"status": "added", "music": query_or_path}

    async def apply_filter(self, filter_name: str, intensity: float = 0.7) -> dict:
        """Apply color grade using Resolve's Color page API."""
        if not self._ensure_timeline():
            return {"status": "error"}

        grade = COLOR_GRADES.get(filter_name.lower().replace(" ", "_"))
        if not grade:
            return {"status": "skipped", "reason": f"Unknown filter: {filter_name}"}

        items = self._timeline.GetItemListInTrack("video", 1)
        if not items:
            return {"status": "error", "message": "No clips in timeline"}

        for item in items:
            node = item.GetNodeGraph()
            if not node:
                continue
            corrector = node.GetToolList(False, "ColorCorrector")
            if not corrector:
                continue
            tool = list(corrector.values())[0]

            # Apply grade settings
            if "contrast" in grade:
                tool.Lift_Master = 0.5 + grade.get("lift_r", 0) * intensity
                tool.Gain_Master = 1.0 + grade.get("contrast", 0) * intensity
            if "saturation" in grade:
                tool.Saturation = grade["saturation"]

        return {"status": "applied", "filter": filter_name}

    async def adjust_speed(self, clip_index: int, speed_multiplier: float) -> dict:
        if not self._ensure_timeline():
            return {"status": "error"}

        items = self._timeline.GetItemListInTrack("video", 1)
        if items and clip_index < len(items):
            items[clip_index].SetClipProperty("Speed", speed_multiplier * 100)
        return {"status": "adjusted", "speed": speed_multiplier}

    async def adjust_volume(self, clip_index: int, volume: float) -> dict:
        if not self._ensure_timeline():
            return {"status": "error"}

        items = self._timeline.GetItemListInTrack("audio", 1)
        if items and clip_index < len(items):
            items[clip_index].SetVolume(volume)
        return {"status": "adjusted", "volume": volume}

    async def export_video(
        self,
        resolution: str = "1080p",
        fps: int = 30,
        output_path: str = "output/jarvis_edit.mp4",
    ) -> dict:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        output_dir = os.path.dirname(os.path.abspath(output_path))
        output_name = os.path.splitext(os.path.basename(output_path))[0]

        res_map = {
            "720p": (1280, 720), "1080p": (1920, 1080),
            "2k": (2560, 1440), "4k": (3840, 2160),
        }
        w, h = res_map.get(resolution, (1920, 1080))

        # Add render job
        self._project.SetRenderSettings({
            "SelectAllFrames": True,
            "TargetDir": output_dir,
            "CustomName": output_name,
            "FormatWidth": w,
            "FormatHeight": h,
            "FrameRate": float(fps),
            "VideoQuality": 0,          # Best quality
            "AudioCodec": "aac",
            "VideoCodec": "H.264 Master",
            "EncodingProfile": "Main",
        })

        job_id = self._project.AddRenderJob()
        if job_id:
            self._project.StartRendering(job_id)
            # Wait for completion (poll)
            import time
            for _ in range(600):  # Up to 10 minutes
                time.sleep(1)
                status = self._project.GetRenderJobStatus(job_id)
                if status and status.get("JobStatus") in ("Complete", "Failed"):
                    break
            self._project.DeleteAllRenderJobs()

        return {"status": "exported", "output": output_path, "resolution": resolution}
