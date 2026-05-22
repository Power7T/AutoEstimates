"""
FFmpeg Adapter — pure CLI video editing. No GUI, no software needed.
Runs on any machine, server, or cloud instance with ffmpeg installed.

This is the most reliable adapter:
  - No UI to scrape or click
  - No API to authenticate
  - No software to install beyond ffmpeg
  - 10x faster than any GUI-based adapter
  - Works headlessly on servers
"""

import asyncio
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .base import BaseAdapter

# LUT presets — colour grades as ffmpeg filter strings
LUT_FILTERS = {
    "cinematic":     "curves=r='0/0 0.2/0.18 0.8/0.78 1/1':g='0/0 0.2/0.19 0.8/0.77 1/1':b='0/0 0.2/0.22 0.8/0.82 1/1',unsharp=3:3:0.5",
    "orange_teal":   "colorchannelmixer=rr=1.1:gb=0.15:br=-0.1:bb=0.9:gr=0.05",
    "warm":          "colortemperature=temperature=7500",
    "cool":          "colortemperature=temperature=5000",
    "moody":         "curves=r='0/0 0.5/0.45 1/0.9':g='0/0 0.5/0.48 1/0.92':b='0/0.05 0.5/0.5 1/1',vignette=PI/4",
    "vintage":       "curves=r='0/0.05 1/0.95':g='0/0.02 1/0.92':b='0/0.1 1/0.85',noise=alls=8:allf=t",
    "bright":        "eq=brightness=0.06:contrast=1.05:saturation=1.1",
    "black_white":   "hue=s=0,curves=all='0/0 0.5/0.55 1/1'",
    "high_contrast": "eq=contrast=1.3:brightness=-0.05:saturation=1.2",
}

TRANSITION_FILTERS = {
    "fade":       "fade=t=in:st=0:d={dur}",
    "dissolve":   "xfade=transition=dissolve:duration={dur}:offset={offset}",
    "wipe_left":  "xfade=transition=wipeleft:duration={dur}:offset={offset}",
    "wipe_right": "xfade=transition=wiperight:duration={dur}:offset={offset}",
    "zoom_in":    "xfade=transition=zoomin:duration={dur}:offset={offset}",
    "flash":      "xfade=transition=fade:duration={dur}:offset={offset}",
    "glitch":     "xfade=transition=pixelize:duration={dur}:offset={offset}",
}


class FFmpegAdapter(BaseAdapter):
    """
    Edits video by building and executing ffmpeg commands.
    No GUI. Runs anywhere. Fully headless.
    """

    name = "ffmpeg"
    platform = "any"

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # State built up during editing
        self._clips: list[dict] = []         # {path, start, end, speed, volume}
        self._text_layers: list[dict] = []
        self._filters: list[str] = []
        self._music_path: str | None = None
        self._music_volume: float = 0.4
        self._transitions: list[dict] = []
        self._project_name: str = "jarvis_edit"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg not found. Install it: https://ffmpeg.org/download.html")

    async def stop(self) -> None:
        pass  # Nothing to close

    # ------------------------------------------------------------------
    # Project
    # ------------------------------------------------------------------

    async def create_project(self, name: str) -> dict:
        self._project_name = name.replace(" ", "_").lower()
        self._clips = []
        self._text_layers = []
        self._filters = []
        self._music_path = None
        self._transitions = []
        return {"status": "created", "project": name}

    async def import_media(self, file_path: str) -> dict:
        if not os.path.exists(file_path):
            return {"status": "error", "message": f"File not found: {file_path}"}
        self._clips.append({
            "path": file_path,
            "index": len(self._clips),
            "start": 0.0,
            "end": None,
            "speed": 1.0,
            "volume": 1.0,
        })
        return {"status": "imported", "file": file_path, "clip_index": len(self._clips) - 1}

    # ------------------------------------------------------------------
    # Timeline edits (queue up, applied at export)
    # ------------------------------------------------------------------

    async def trim_clip(self, clip_index: int, start: float, end: float) -> dict:
        if clip_index < len(self._clips):
            self._clips[clip_index]["start"] = start
            self._clips[clip_index]["end"] = end
        return {"status": "trimmed", "clip": clip_index, "start": start, "end": end}

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
        size_map = {"small": 36, "medium": 52, "large": 72}
        px = size_map.get(font_size, 52)
        y_map = {"top": "h*0.08", "center": "(h-text_h)/2", "bottom": "h*0.85"}
        y = y_map.get(position, "h*0.85")
        self._text_layers.append({
            "text": text.replace("'", "\\'"),
            "size": px,
            "color": color,
            "y": y,
            "start": start,
            "end": start + duration,
        })
        return {"status": "queued", "text": text}

    async def add_transition(self, clip_index: int, transition_type: str, duration: float) -> dict:
        self._transitions.append({
            "after_clip": clip_index,
            "type": transition_type,
            "duration": duration,
        })
        return {"status": "queued", "transition": transition_type}

    async def add_music(self, query_or_path: str, volume: float = 0.4) -> dict:
        # Only local files work for FFmpeg — skip library search
        if os.path.exists(query_or_path):
            self._music_path = query_or_path
            self._music_volume = volume
            return {"status": "added", "music": query_or_path}
        return {"status": "skipped", "reason": "FFmpeg adapter only supports local music files"}

    async def apply_filter(self, filter_name: str, intensity: float = 1.0) -> dict:
        filt = LUT_FILTERS.get(filter_name.lower().replace(" ", "_"))
        if filt:
            self._filters.append(filt)
        return {"status": "queued", "filter": filter_name}

    async def apply_lut(self, lut_style: str, intensity: float = 1.0) -> dict:
        return await self.apply_filter(lut_style, intensity)

    async def adjust_speed(self, clip_index: int, speed_multiplier: float) -> dict:
        if clip_index < len(self._clips):
            self._clips[clip_index]["speed"] = speed_multiplier
        return {"status": "queued", "speed": speed_multiplier}

    async def adjust_volume(self, clip_index: int, volume: float) -> dict:
        if clip_index < len(self._clips):
            self._clips[clip_index]["volume"] = volume
        return {"status": "queued", "volume": volume}

    # ------------------------------------------------------------------
    # Export — this is where ffmpeg actually runs
    # ------------------------------------------------------------------

    async def export_video(
        self,
        resolution: str = "1080p",
        fps: int = 30,
        output_path: str | None = None,
    ) -> dict:
        if not self._clips:
            return {"status": "error", "message": "No clips imported"}

        output_path = output_path or str(self.output_dir / f"{self._project_name}.mp4")
        res_map = {"480p": "854:480", "720p": "1280:720", "1080p": "1920:1080", "2k": "2560:1440", "4k": "3840:2160"}
        scale = res_map.get(resolution, "1920:1080")

        # Step 1: Process each clip individually (trim + speed + volume)
        processed: list[str] = []
        for clip in self._clips:
            out = str(self.output_dir / f"_clip_{clip['index']}.mp4")
            cmd = ["ffmpeg", "-y"]

            # Trim
            if clip["start"] > 0:
                cmd += ["-ss", str(clip["start"])]
            cmd += ["-i", clip["path"]]
            if clip["end"] is not None:
                cmd += ["-t", str(clip["end"] - clip["start"])]

            # Video filters: scale + speed + fps
            vf_parts = [f"scale={scale}:force_original_aspect_ratio=decrease,pad={scale}:(ow-iw)/2:(oh-ih)/2"]
            if clip["speed"] != 1.0:
                vf_parts.append(f"setpts={1/clip['speed']}*PTS")
            vf_parts.append(f"fps={fps}")

            # Audio filters: speed + volume
            af_parts = []
            if clip["speed"] != 1.0:
                af_parts.append(f"atempo={min(max(clip['speed'], 0.5), 2.0)}")
            if clip["volume"] != 1.0:
                af_parts.append(f"volume={clip['volume']}")

            cmd += ["-vf", ",".join(vf_parts)]
            if af_parts:
                cmd += ["-af", ",".join(af_parts)]
            cmd += ["-c:v", "libx264", "-c:a", "aac", "-preset", "fast", out]

            await self._run(cmd)
            processed.append(out)

        # Step 2: Concatenate clips
        concat_out = str(self.output_dir / "_concat.mp4")
        if len(processed) == 1:
            concat_out = processed[0]
        else:
            list_file = str(self.output_dir / "_concat_list.txt")
            with open(list_file, "w") as f:
                for p in processed:
                    f.write(f"file '{p}'\n")
            await self._run([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", list_file, "-c", "copy", concat_out,
            ])

        # Step 3: Apply colour filters + text overlays
        vf_chain = []
        if self._filters:
            vf_chain.extend(self._filters)

        for layer in self._text_layers:
            escaped = layer["text"].replace(":", "\\:")
            vf_chain.append(
                f"drawtext=text='{escaped}'"
                f":fontsize={layer['size']}"
                f":fontcolor={layer['color']}"
                f":x=(w-text_w)/2:y={layer['y']}"
                f":enable='between(t,{layer['start']},{layer['end']})'"
                f":shadowcolor=black:shadowx=2:shadowy=2"
            )

        graded_out = str(self.output_dir / "_graded.mp4")
        if vf_chain:
            await self._run([
                "ffmpeg", "-y", "-i", concat_out,
                "-vf", ",".join(vf_chain),
                "-c:v", "libx264", "-c:a", "copy", "-preset", "fast",
                graded_out,
            ])
        else:
            graded_out = concat_out

        # Step 4: Mix in background music
        final_source = graded_out
        if self._music_path and os.path.exists(self._music_path):
            music_out = str(self.output_dir / "_with_music.mp4")
            await self._run([
                "ffmpeg", "-y",
                "-i", final_source,
                "-i", self._music_path,
                "-filter_complex",
                f"[1:a]volume={self._music_volume},afade=t=in:d=1,afade=t=out:st=0:d=1[music];"
                f"[0:a][music]amix=inputs=2:duration=first[outa]",
                "-map", "0:v", "-map", "[outa]",
                "-c:v", "copy", "-c:a", "aac",
                music_out,
            ])
            final_source = music_out

        # Step 5: Final output
        await self._run(["ffmpeg", "-y", "-i", final_source, "-c", "copy", output_path])

        # Cleanup temp files
        for f in self.output_dir.glob("_*.mp4"):
            try: f.unlink()
            except Exception: pass
        for f in self.output_dir.glob("_*.txt"):
            try: f.unlink()
            except Exception: pass

        size_mb = round(os.path.getsize(output_path) / 1024 / 1024, 1)
        return {"status": "exported", "output": output_path, "size_mb": size_mb}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _run(self, cmd: list[str]) -> None:
        """Run an ffmpeg command asynchronously."""
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{stderr.decode()[-500:]}")
