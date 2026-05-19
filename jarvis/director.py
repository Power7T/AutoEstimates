"""
Jarvis Director Mode — autonomous video editing.

Pipeline:
  1. Extract frames + metadata from every clip
  2. Send to Claude vision for scene-by-scene analysis
  3. Claude acts as director and produces a full EditPlan
  4. Execute the plan in CapCut via CapCutController
"""

import json
from dataclasses import dataclass, field
from typing import Any

import anthropic

from .frame_extractor import ClipAnalysisData, extract_multi_clip_data
from .tools import TOOLS


# ---------------------------------------------------------------------------
# Edit plan data model
# ---------------------------------------------------------------------------

@dataclass
class TextOverlay:
    text: str
    position: str = "bottom"
    start_seconds: float = 0
    duration_seconds: float = 3
    font_size: str = "medium"
    color: str = "white"


@dataclass
class TransitionPlan:
    after_clip_index: int
    transition_type: str = "fade"
    duration_seconds: float = 0.5


@dataclass
class ClipEdit:
    source_path: str
    clip_index: int
    trim_start: float = 0.0
    trim_end: float | None = None      # None = keep original end
    speed_multiplier: float = 1.0
    volume: float = 1.0
    filter_name: str | None = None
    filter_intensity: float = 0.7


@dataclass
class EditPlan:
    project_name: str
    aspect_ratio: str = "16:9"
    clips: list[ClipEdit] = field(default_factory=list)
    transitions: list[TransitionPlan] = field(default_factory=list)
    text_overlays: list[TextOverlay] = field(default_factory=list)
    music_query: str | None = None
    music_volume: float = 0.4
    export_resolution: str = "1080p"
    export_fps: int = 30
    director_notes: str = ""          # Claude's creative reasoning


# ---------------------------------------------------------------------------
# Director
# ---------------------------------------------------------------------------

DIRECTOR_SYSTEM_PROMPT = """You are Jarvis, an elite AI video director.
You will be given metadata and sample frames from one or more raw video clips.
Your job is to produce a complete, high-quality edit plan as a structured JSON object.

Style guidelines you must always apply:
- Keep only the best moments; cut any shaky, blurry, or boring footage
- Use smooth transitions; prefer dissolve/fade for calm content, glitch/flash for energetic content
- Add concise, impactful text overlays at key moments
- Choose music that matches the overall mood you detect in the footage
- Apply a consistent color grade / filter across all clips for a polished look
- Export at 1080p 30fps unless the source is clearly 4K-worthy

Respond ONLY with a JSON object matching this exact schema — no commentary, no markdown fences:
{
  "project_name": "string",
  "aspect_ratio": "16:9 | 9:16 | 1:1",
  "director_notes": "one paragraph explaining your creative choices",
  "clips": [
    {
      "source_path": "string",
      "clip_index": 0,
      "trim_start": 0.0,
      "trim_end": null,
      "speed_multiplier": 1.0,
      "volume": 1.0,
      "filter_name": "cinematic | vintage | bright | moody | black_white | warm | cool | null",
      "filter_intensity": 0.7
    }
  ],
  "transitions": [
    {"after_clip_index": 0, "transition_type": "fade", "duration_seconds": 0.5}
  ],
  "text_overlays": [
    {
      "text": "string",
      "position": "top | center | bottom",
      "start_seconds": 0.0,
      "duration_seconds": 3.0,
      "font_size": "small | medium | large",
      "color": "white"
    }
  ],
  "music_query": "string or null",
  "music_volume": 0.4,
  "export_resolution": "1080p",
  "export_fps": 30
}"""


class JarvisDirector:
    """Autonomous video editor: analyzes footage, plans and executes the edit."""

    def __init__(self, client: anthropic.Anthropic, capcut, console=None):
        self.client = client
        self.capcut = capcut
        self.console = console

    def _log(self, msg: str):
        if self.console:
            self.console.print(f"[cyan]Jarvis ▸[/cyan] {msg}")
        else:
            print(f"Jarvis ▸ {msg}")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def direct(
        self,
        media_paths: list[str],
        style_hint: str = "cinematic",
        project_name: str = "Jarvis Edit",
    ) -> EditPlan:
        """
        Full autonomous pipeline:
          analyze → plan → execute in CapCut → return EditPlan
        """
        self._log(f"Analyzing {len(media_paths)} clip(s)...")
        clips_data = extract_multi_clip_data(media_paths, frames_per_clip=5)

        if not clips_data:
            raise ValueError("No valid video clips found at the provided paths.")

        self._log("Sending footage to Claude for analysis and edit planning...")
        plan = await self._plan_edit(clips_data, style_hint, project_name)

        self._log(f"Edit plan ready. Director's note: {plan.director_notes[:120]}...")
        self._log("Executing edit in CapCut...")
        await self._execute_plan(plan)

        self._log("All done! Your video is ready.")
        return plan

    # ------------------------------------------------------------------
    # Step 1: Analyze footage and produce an EditPlan via Claude vision
    # ------------------------------------------------------------------

    async def _plan_edit(
        self,
        clips_data: list[ClipAnalysisData],
        style_hint: str,
        project_name: str,
    ) -> EditPlan:
        messages = self._build_analysis_messages(clips_data, style_hint, project_name)

        response = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=DIRECTOR_SYSTEM_PROMPT,
            messages=messages,
        )

        raw = response.content[0].text.strip()
        # Strip accidental markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        return self._parse_plan(data)

    def _build_analysis_messages(
        self,
        clips_data: list[ClipAnalysisData],
        style_hint: str,
        project_name: str,
    ) -> list[dict]:
        """Build the multimodal message that sends frames + metadata to Claude."""
        content: list[dict] = []

        content.append({
            "type": "text",
            "text": (
                f"Project name: {project_name}\n"
                f"Style hint: {style_hint}\n"
                f"Number of clips: {len(clips_data)}\n\n"
                "Below you will find metadata and sample frames for each clip. "
                "Analyze all footage carefully and produce the best possible edit plan."
            ),
        })

        for i, clip in enumerate(clips_data):
            content.append({
                "type": "text",
                "text": (
                    f"\n--- Clip {i} ---\n"
                    f"Path: {clip.path}\n"
                    f"Duration: {clip.duration_seconds:.1f}s  FPS: {clip.fps}  "
                    f"Resolution: {clip.width}x{clip.height}\n"
                    f"Estimated scenes: {clip.estimated_scenes}  "
                    f"Has audio: {clip.has_audio}\n"
                    f"Sample frames ({len(clip.sample_frames)}):"
                ),
            })
            for b64 in clip.sample_frames:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64,
                    },
                })

        return [{"role": "user", "content": content}]

    # ------------------------------------------------------------------
    # Step 2: Execute the EditPlan in CapCut
    # ------------------------------------------------------------------

    async def _execute_plan(self, plan: EditPlan):
        """Drive CapCut via the controller to carry out every step of the plan."""
        # Open / create project
        await self.capcut.execute_tool("open_project", {
            "project_name": plan.project_name,
            "create_new": True,
        })

        # Import and configure each clip
        for clip_edit in plan.clips:
            self._log(f"Importing clip {clip_edit.clip_index}: {clip_edit.source_path}")
            await self.capcut.execute_tool("import_media", {"file_path": clip_edit.source_path})

            if clip_edit.trim_end is not None:
                await self.capcut.execute_tool("trim_clip", {
                    "clip_index": clip_edit.clip_index,
                    "start_seconds": clip_edit.trim_start,
                    "end_seconds": clip_edit.trim_end,
                })

            if clip_edit.speed_multiplier != 1.0:
                await self.capcut.execute_tool("adjust_speed", {
                    "clip_index": clip_edit.clip_index,
                    "speed_multiplier": clip_edit.speed_multiplier,
                })

            if clip_edit.volume != 1.0:
                await self.capcut.execute_tool("adjust_volume", {
                    "clip_index": clip_edit.clip_index,
                    "volume": clip_edit.volume,
                })

            if clip_edit.filter_name:
                await self.capcut.execute_tool("apply_filter", {
                    "filter_name": clip_edit.filter_name,
                    "intensity": clip_edit.filter_intensity,
                    "apply_to": "clip",
                    "clip_index": clip_edit.clip_index,
                })

        # Transitions
        for t in plan.transitions:
            self._log(f"Adding {t.transition_type} transition after clip {t.after_clip_index}")
            await self.capcut.execute_tool("add_transition", {
                "clip_index": t.after_clip_index,
                "transition_type": t.transition_type,
                "duration_seconds": t.duration_seconds,
            })

        # Text overlays
        for overlay in plan.text_overlays:
            self._log(f"Adding text: \"{overlay.text}\"")
            await self.capcut.execute_tool("add_text", {
                "text": overlay.text,
                "position": overlay.position,
                "start_seconds": overlay.start_seconds,
                "duration_seconds": overlay.duration_seconds,
                "font_size": overlay.font_size,
                "color": overlay.color,
            })

        # Music
        if plan.music_query:
            self._log(f"Adding music: \"{plan.music_query}\"")
            await self.capcut.execute_tool("add_music", {
                "query_or_path": plan.music_query,
                "source": "library",
                "volume": plan.music_volume,
                "fade_in": True,
                "fade_out": True,
            })

        # Aspect ratio
        await self.capcut.execute_tool("crop_video", {"aspect_ratio": plan.aspect_ratio})

        # Export
        self._log(f"Exporting at {plan.export_resolution} {plan.export_fps}fps...")
        await self.capcut.execute_tool("export_video", {
            "resolution": plan.export_resolution,
            "fps": plan.export_fps,
            "format": "mp4",
        })

    # ------------------------------------------------------------------
    # JSON → EditPlan
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_plan(data: dict[str, Any]) -> EditPlan:
        clips = [
            ClipEdit(
                source_path=c["source_path"],
                clip_index=c.get("clip_index", i),
                trim_start=c.get("trim_start", 0.0),
                trim_end=c.get("trim_end"),
                speed_multiplier=c.get("speed_multiplier", 1.0),
                volume=c.get("volume", 1.0),
                filter_name=c.get("filter_name"),
                filter_intensity=c.get("filter_intensity", 0.7),
            )
            for i, c in enumerate(data.get("clips", []))
        ]
        transitions = [
            TransitionPlan(
                after_clip_index=t["after_clip_index"],
                transition_type=t.get("transition_type", "fade"),
                duration_seconds=t.get("duration_seconds", 0.5),
            )
            for t in data.get("transitions", [])
        ]
        overlays = [
            TextOverlay(
                text=o["text"],
                position=o.get("position", "bottom"),
                start_seconds=o.get("start_seconds", 0),
                duration_seconds=o.get("duration_seconds", 3),
                font_size=o.get("font_size", "medium"),
                color=o.get("color", "white"),
            )
            for o in data.get("text_overlays", [])
        ]
        return EditPlan(
            project_name=data.get("project_name", "Jarvis Edit"),
            aspect_ratio=data.get("aspect_ratio", "16:9"),
            clips=clips,
            transitions=transitions,
            text_overlays=overlays,
            music_query=data.get("music_query"),
            music_volume=data.get("music_volume", 0.4),
            export_resolution=data.get("export_resolution", "1080p"),
            export_fps=data.get("export_fps", 30),
            director_notes=data.get("director_notes", ""),
        )
