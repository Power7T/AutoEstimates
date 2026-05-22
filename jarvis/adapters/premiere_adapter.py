"""
Adobe Premiere Pro Adapter.

Supports two control methods depending on what's available:

Method A — UXP Plugin (recommended, modern):
  Jarvis runs a local HTTP server.
  A UXP plugin installed in Premiere connects to it and executes commands.
  This is Adobe's official modern extension system.

Method B — COM Automation (Windows only, legacy):
  On Windows, Premiere Pro exposes a COM interface.
  No plugin needed — Jarvis talks directly via pywin32.

Method C — ExtendScript via CEP (legacy, all platforms):
  Sends JavaScript to Premiere's scripting engine via a temp file.
  Works on Mac and Windows. Slower but reliable.

This adapter uses Method A first, falls back to C.
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .base import BaseAdapter

# UXP plugin communicates with this local server
UXP_HOST = "127.0.0.1"
UXP_PORT = 8765

# ExtendScript snippets for each operation
EXTENDSCRIPT = {
    "import_media": """
var file = new File("{path}");
app.project.importFiles([file.fsName], true, app.project.rootItem, false);
""",
    "trim_clip": """
var seq = app.project.activeSequence;
var track = seq.videoTracks[0];
var clip = track.clips[{index}];
clip.inPoint = new Time({start}, 1);
clip.outPoint = new Time({end}, 1);
""",
    "add_text": """
var seq = app.project.activeSequence;
var gfx = seq.videoTracks[0].overrides;
// Add a Premiere motion graphics title
app.project.activeSequence.videoTracks[1].insertClip(
    app.project.importFiles(['{title_path}'], true, app.project.rootItem, false)[0],
    new Time({start}, 1)
);
""",
    "adjust_speed": """
var seq = app.project.activeSequence;
var clip = seq.videoTracks[0].clips[{index}];
clip.speed = {speed};
""",
    "adjust_volume": """
var seq = app.project.activeSequence;
var clip = seq.audioTracks[0].clips[{index}];
clip.volume.setValue({volume} * 100);
""",
    "export": """
var encoder = app.encoder;
var seq = app.project.activeSequence;
encoder.encodeSequence(
    seq,
    '{output_path}',
    '{preset}',
    app.encoder.ENCODE_IN_TO_OUT,
    true
);
""",
    "apply_lumetri": """
var seq = app.project.activeSequence;
var track = seq.videoTracks[0];
for (var i = 0; i < track.clips.numItems; i++) {{
    var clip = track.clips[i];
    var effect = clip.components.addItem('Lumetri Color');
    if (effect) {{
        effect.properties.getParamForName('Temperature').setValue({temperature});
        effect.properties.getParamForName('Contrast').setValue({contrast});
        effect.properties.getParamForName('Saturation').setValue({saturation});
    }}
}}
""",
}

FILTER_LUMETRI = {
    "cinematic":     {"temperature": -5,  "contrast": 8,  "saturation": 90},
    "warm":          {"temperature": 15,  "contrast": 5,  "saturation": 100},
    "cool":          {"temperature": -15, "contrast": 5,  "saturation": 100},
    "moody":         {"temperature": -5,  "contrast": 15, "saturation": 80},
    "vintage":       {"temperature": 8,   "contrast": 5,  "saturation": 75},
    "bright":        {"temperature": 5,   "contrast": -5, "saturation": 110},
    "black_white":   {"temperature": 0,   "contrast": 10, "saturation": 0},
    "high_contrast": {"temperature": 0,   "contrast": 25, "saturation": 110},
    "orange_teal":   {"temperature": 10,  "contrast": 10, "saturation": 115},
}


class PremiereAdapter(BaseAdapter):
    """
    Controls Adobe Premiere Pro.
    Uses UXP plugin server if available, falls back to ExtendScript.
    """

    name = "premiere"
    platform = "desktop"

    def __init__(self):
        self._method: str = "extendscript"
        self._reader = None
        self._writer = None
        self._clips: list[str] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        # Try UXP plugin connection first
        try:
            self._reader, self._writer = await asyncio.open_connection(UXP_HOST, UXP_PORT)
            self._method = "uxp"
            print("[Premiere] Connected via UXP plugin")
            return
        except (ConnectionRefusedError, OSError):
            pass

        # Check if Premiere is running for ExtendScript fallback
        if self._is_premiere_running():
            self._method = "extendscript"
            print("[Premiere] Using ExtendScript fallback")
        else:
            raise RuntimeError(
                "Adobe Premiere Pro is not running.\n"
                "Open Premiere Pro and either:\n"
                "  A) Install the Jarvis UXP plugin, or\n"
                "  B) Just open a project (ExtendScript will be used automatically)"
            )

    async def stop(self) -> None:
        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Project
    # ------------------------------------------------------------------

    async def create_project(self, name: str) -> dict:
        # Premiere projects are typically opened via the UI.
        # We work with whatever project is currently open.
        return {"status": "using_current_project", "note": "Open your Premiere project manually first"}

    async def import_media(self, file_path: str) -> dict:
        if not os.path.exists(file_path):
            return {"status": "error", "message": f"File not found: {file_path}"}

        script = EXTENDSCRIPT["import_media"].format(path=file_path.replace("\\", "/"))
        result = await self._execute(script)
        self._clips.append(file_path)
        return {"status": "imported", "file": file_path}

    # ------------------------------------------------------------------
    # Edits
    # ------------------------------------------------------------------

    async def trim_clip(self, clip_index: int, start: float, end: float) -> dict:
        script = EXTENDSCRIPT["trim_clip"].format(
            index=clip_index, start=int(start * 254016000), end=int(end * 254016000)
        )
        await self._execute(script)
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
        # Generate a simple MOGRT/graphic via ExtendScript
        size_map = {"small": 36, "medium": 52, "large": 72}
        y_map = {"top": 0.1, "center": 0.5, "bottom": 0.85}
        script = f"""
var seq = app.project.activeSequence;
var textItem = seq.videoTracks[1].insertClip(
    app.project.createMGT("Basic Title"),
    new Time({int(start * 254016000)}, 1)
);
if (textItem) {{
    textItem.setField("Text/Source Text", "{text.replace(chr(34), chr(39))}");
}}
"""
        await self._execute(script)
        return {"status": "added", "text": text}

    async def add_transition(self, clip_index: int, transition_type: str, duration: float) -> dict:
        premiere_transitions = {
            "fade": "Cross Dissolve", "dissolve": "Cross Dissolve",
            "wipe_left": "Wipe", "flash": "Film Dissolve", "glitch": "Morph Cut",
        }
        tr = premiere_transitions.get(transition_type, "Cross Dissolve")
        script = f"""
var seq = app.project.activeSequence;
var clip = seq.videoTracks[0].clips[{clip_index}];
if (clip) {{
    seq.videoTracks[0].insertTransition(
        qe.project.getVideoTransitionByName("{tr}"),
        {clip_index}, {int(duration * 254016000)}
    );
}}
"""
        await self._execute(script)
        return {"status": "added", "transition": transition_type}

    async def add_music(self, query_or_path: str, volume: float = 0.5) -> dict:
        if not os.path.exists(query_or_path):
            return {"status": "skipped", "reason": "Only local audio files supported"}
        script = EXTENDSCRIPT["import_media"].format(path=query_or_path.replace("\\", "/"))
        await self._execute(script)
        return {"status": "added", "music": query_or_path}

    async def apply_filter(self, filter_name: str, intensity: float = 0.7) -> dict:
        grade = FILTER_LUMETRI.get(filter_name.lower().replace(" ", "_"))
        if not grade:
            return {"status": "skipped"}
        script = EXTENDSCRIPT["apply_lumetri"].format(
            temperature=grade["temperature"] * intensity,
            contrast=grade["contrast"] * intensity,
            saturation=grade["saturation"],
        )
        await self._execute(script)
        return {"status": "applied", "filter": filter_name}

    async def adjust_speed(self, clip_index: int, speed_multiplier: float) -> dict:
        script = EXTENDSCRIPT["adjust_speed"].format(index=clip_index, speed=speed_multiplier * 100)
        await self._execute(script)
        return {"status": "adjusted", "speed": speed_multiplier}

    async def adjust_volume(self, clip_index: int, volume: float) -> dict:
        script = EXTENDSCRIPT["adjust_volume"].format(index=clip_index, volume=volume)
        await self._execute(script)
        return {"status": "adjusted", "volume": volume}

    async def export_video(
        self,
        resolution: str = "1080p",
        fps: int = 30,
        output_path: str = "output/jarvis_edit.mp4",
    ) -> dict:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        preset_map = {
            "720p": "H.264 720p",
            "1080p": "H.264 1080p",
            "2k": "H.264 2K",
            "4k": "H.264 4K Ultra HD",
        }
        preset = preset_map.get(resolution, "H.264 1080p")
        script = EXTENDSCRIPT["export"].format(
            output_path=output_path.replace("\\", "/"),
            preset=preset,
        )
        await self._execute(script)
        return {"status": "exporting", "output": output_path}

    # ------------------------------------------------------------------
    # Execution engine
    # ------------------------------------------------------------------

    async def _execute(self, script: str) -> str:
        """Execute script in Premiere via UXP or ExtendScript."""
        if self._method == "uxp":
            return await self._execute_uxp(script)
        return await self._execute_extendscript(script)

    async def _execute_uxp(self, script: str) -> str:
        """Send script to UXP plugin over TCP."""
        payload = json.dumps({"script": script}) + "\n"
        self._writer.write(payload.encode())
        await self._writer.drain()
        line = await self._reader.readline()
        return line.decode().strip()

    async def _execute_extendscript(self, script: str) -> str:
        """Run ExtendScript by writing to a temp file and invoking via osascript/doScript."""
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsx", delete=False)
        tmp.write(script)
        tmp.close()

        try:
            if sys.platform == "darwin":
                cmd = [
                    "osascript", "-e",
                    f'tell application "Adobe Premiere Pro" to do script (read POSIX file "{tmp.name}")',
                ]
            elif sys.platform == "win32":
                cmd = ["cscript", "//nologo", tmp.name]
            else:
                return ""  # Linux: no ExtendScript host without Adobe

            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await proc.communicate()
            return stdout.decode().strip()
        finally:
            os.unlink(tmp.name)

    @staticmethod
    def _is_premiere_running() -> bool:
        """Check if Premiere Pro process is running."""
        try:
            if sys.platform == "darwin":
                result = subprocess.run(
                    ["pgrep", "-x", "Adobe Premiere Pro"], capture_output=True
                )
                return result.returncode == 0
            elif sys.platform == "win32":
                result = subprocess.run(
                    ["tasklist", "/FI", "IMAGENAME eq Adobe Premiere Pro.exe"],
                    capture_output=True, text=True,
                )
                return "Adobe Premiere Pro.exe" in result.stdout
        except Exception:
            pass
        return False
