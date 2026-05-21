"""
Jarvis Director — fully autonomous video editing pipeline.

Pipeline:
  1. Analyze every clip with Gemini vision + Whisper (understand what's REALLY there)
  2. Analyze audio: beat detection, energy map, drops
  3. Fetch current trends
  4. Load user taste profile
  5. Select best viral template for this content
  6. Claude plans the complete edit using all context
  7. Execute every step in CapCut via vision-based controller
  8. Ask user to rate → update taste profile
"""

import json
from dataclasses import dataclass, field

from .config import cfg
from .openrouter import router
from .video_analyzer import VideoAnalyzer, ClipUnderstanding
from .audio_analyzer import analyze_audio, select_beat_cut_points, AudioAnalysis
from .templates import EditingTemplate, select_template, TEMPLATES
from .trends import get_current_trends
from .taste_learner import TasteLearner
from .vision_controller import VisionController


# ---------------------------------------------------------------------------
# Edit plan
# ---------------------------------------------------------------------------

@dataclass
class ClipEdit:
    source_path: str
    clip_index: int
    trim_start: float = 0.0
    trim_end: float | None = None
    speed_multiplier: float = 1.0
    speed_ramp: str | None = None       # "drop" | "slow_mo" | None
    volume: float = 1.0
    filter_name: str | None = None
    filter_intensity: float = 0.7
    effects: list[str] = field(default_factory=list)
    stabilize: bool = False
    auto_enhance: bool = False


@dataclass
class TextLayer:
    text: str
    position: str = "bottom"
    start_seconds: float = 0
    duration_seconds: float = 3
    font_size: str = "medium"
    color: str = "white"
    animated: bool = False
    style: str = "clean"


@dataclass
class EditPlan:
    project_name: str
    aspect_ratio: str
    template_name: str
    clips: list[ClipEdit]
    beat_cut_timestamps: list[float]
    text_layers: list[TextLayer]
    use_auto_captions: bool
    music_query: str | None
    music_volume: float
    color_grade: str
    add_vignette: bool
    add_film_grain: bool
    add_light_leak: bool
    export_resolution: str
    export_fps: int
    director_notes: str


DIRECTOR_SYSTEM = """You are Jarvis, an elite AI video director with years of experience
creating viral content for Instagram Reels, TikTok, and YouTube Shorts.

You have been given:
- Deep analysis of every video clip (what's in them, mood, quality, best moments)
- Audio analysis (beats, energy, drops, mood)
- Current viral trends
- The user's personal taste profile
- A proven editing template structure

Your job: produce the PERFECT edit plan as a JSON object.

Rules:
- Put the most attention-grabbing moment FIRST (hook engineering)
- Only keep the best moments from each clip (quality > quantity)
- Sync every cut to a beat timestamp when possible
- Apply speed ramps at music drop points
- Use color grade that matches the mood of the content
- Add text overlays that add context or emotion — not just captions
- Choose music that fits the emotional arc you're creating
- Apply the template structure strictly
- Factor in the user's taste profile preferences

Return ONLY valid JSON. No explanation. No markdown."""

DIRECTOR_PROMPT = """Create the complete edit plan for this video project.

=== CLIP ANALYSIS ===
{clips_analysis}

=== AUDIO ANALYSIS ===
Tempo: {tempo_bpm} BPM
Mood: {audio_mood}
Beat timestamps (first 20): {beat_timestamps}
Drop timestamps: {drop_timestamps}
Energy drops at: {high_energy}

=== CURRENT TRENDS ===
{trends}

=== USER TASTE PROFILE ===
{taste_profile}

=== TEMPLATE TO FOLLOW ===
{template}

Project name: {project_name}
Platform: {platform}
Style hint: {style_hint}

Return JSON matching exactly this schema:
{{
  "project_name": "string",
  "aspect_ratio": "9:16|16:9|1:1",
  "template_name": "string",
  "director_notes": "paragraph explaining creative choices",
  "clips": [
    {{
      "source_path": "string",
      "clip_index": 0,
      "trim_start": 0.0,
      "trim_end": null,
      "speed_multiplier": 1.0,
      "speed_ramp": "drop|slow_mo|null",
      "volume": 1.0,
      "filter_name": "cinematic|warm|cool|moody|vintage|bright|black_white|null",
      "filter_intensity": 0.7,
      "effects": [],
      "stabilize": false,
      "auto_enhance": false
    }}
  ],
  "beat_cut_timestamps": [0.0],
  "text_layers": [
    {{
      "text": "string",
      "position": "top|center|bottom",
      "start_seconds": 0.0,
      "duration_seconds": 3.0,
      "font_size": "small|medium|large",
      "color": "white",
      "animated": false,
      "style": "clean|bold|kinetic"
    }}
  ],
  "use_auto_captions": false,
  "music_query": "string or null",
  "music_volume": 0.4,
  "color_grade": "orange_teal|warm|cool|moody|high_contrast|vintage|bright",
  "add_vignette": true,
  "add_film_grain": false,
  "add_light_leak": false,
  "export_resolution": "1080p",
  "export_fps": 30
}}"""


class JarvisDirector:
    """Fully autonomous video editor — analyzes, plans, and executes."""

    def __init__(self, controller: VisionController, console=None):
        self.controller = controller
        self.console = console
        self.video_analyzer = VideoAnalyzer()
        self.taste = TasteLearner()

    def _log(self, msg: str, style: str = "cyan"):
        if self.console:
            self.console.print(f"[{style}]Jarvis ▸[/{style}] {msg}")
        else:
            print(f"Jarvis ▸ {msg}")

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    async def direct(
        self,
        media_paths: list[str],
        project_name: str = "Jarvis Edit",
        platform: str = "instagram",
        style_hint: str = "cinematic",
        music_path: str | None = None,
    ) -> EditPlan:

        # --- Step 1: Deep clip analysis ---
        self._log(f"Watching and understanding {len(media_paths)} clip(s)...")
        clip_analyses = self.video_analyzer.analyze_all(media_paths)
        if not clip_analyses:
            raise ValueError("No valid clips could be analyzed.")

        # --- Step 2: Audio analysis ---
        self._log("Detecting beats, energy, and music drops...")
        audio_path = music_path or media_paths[0]
        try:
            audio = analyze_audio(audio_path)
        except Exception as e:
            self._log(f"Audio analysis skipped: {e}", "yellow")
            audio = None

        # --- Step 3: Trends ---
        trends = {}
        if cfg.enable_trends:
            self._log("Checking current viral trends...")
            try:
                trends = get_current_trends()
            except Exception:
                pass

        # --- Step 4: Select template ---
        dominant_mood = max(
            clip_analyses, key=lambda c: c.energy_level
        ).mood if clip_analyses else "neutral"
        total_duration = sum(c.duration_seconds for c in clip_analyses)
        template = select_template(dominant_mood, platform, total_duration, has_speech=any(c.transcript for c in clip_analyses))

        # --- Step 5: AI director plans the edit ---
        self._log("Claude is planning the perfect edit...")
        plan = self._plan_edit(
            clip_analyses=clip_analyses,
            audio=audio,
            trends=trends,
            template=template,
            project_name=project_name,
            platform=platform,
            style_hint=style_hint,
        )

        self._log(f"Edit planned. Director's note: {plan.director_notes[:100]}...")

        # --- Step 6: Execute in CapCut ---
        self._log("Executing edit in CapCut...")
        await self._execute(plan)

        return plan

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def _plan_edit(
        self,
        clip_analyses: list[ClipUnderstanding],
        audio: AudioAnalysis | None,
        trends: dict,
        template: EditingTemplate,
        project_name: str,
        platform: str,
        style_hint: str,
    ) -> EditPlan:

        # Build clips summary for the prompt
        clips_text = "\n\n".join(
            f"Clip {i} — {c.path}\n"
            f"  Duration: {c.duration_seconds}s | Mood: {c.mood} | Energy: {c.energy_level:.1f}\n"
            f"  Scene type: {c.scene_type} | Quality: {c.quality_score:.1f}\n"
            f"  Content: {c.content_description}\n"
            f"  Best moments: {[(m.timestamp, m.reason) for m in c.key_moments[:3]]}\n"
            f"  Suggested filter: {c.suggested_filter} | Suggested effect: {c.suggested_effect}\n"
            f"  Transcript: {c.transcript[:200] if c.transcript else 'none'}"
            for i, c in enumerate(clip_analyses)
        )

        beat_timestamps = (audio.beat_timestamps[:20] if audio else [])
        drop_timestamps = (audio.drop_timestamps[:5] if audio else [])
        high_energy_times = (
            [i for i, e in enumerate(audio.energy_curve) if e > 0.8][:10]
            if audio else []
        )

        prompt = DIRECTOR_PROMPT.format(
            clips_analysis=clips_text,
            tempo_bpm=audio.tempo_bpm if audio else "unknown",
            audio_mood=audio.mood if audio else "neutral",
            beat_timestamps=beat_timestamps,
            drop_timestamps=drop_timestamps,
            high_energy=high_energy_times,
            trends=json.dumps(trends, indent=2)[:1500],
            taste_profile=self.taste.get_profile_summary(),
            template=json.dumps({
                "name": template.name,
                "segments": [
                    {"name": s.name, "energy": s.energy, "cut_frequency": s.cut_frequency,
                     "effects": s.effects, "notes": s.notes}
                    for s in template.segments
                ],
                "music_mood": template.music_mood,
                "color_grade": template.color_grade,
            }, indent=2),
            project_name=project_name,
            platform=platform,
            style_hint=style_hint,
        )

        data = router.complete_json(
            prompt=prompt,
            system=DIRECTOR_SYSTEM,
            model=cfg.models.director,
            max_tokens=4096,
        )

        return self._parse_plan(data)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def _execute(self, plan: EditPlan):
        # 1. Create project
        self._log("Creating CapCut project...")
        await self.controller.execute_tool("open_project", {"project_name": plan.project_name})

        # 2. Set aspect ratio
        await self.controller.execute_tool("crop_video", {"aspect_ratio": plan.aspect_ratio})

        # 3. Import and configure each clip
        for clip in plan.clips:
            self._log(f"Importing clip {clip.clip_index}: {clip.source_path.split('/')[-1]}")
            await self.controller.execute_tool("import_media", {"file_path": clip.source_path})

            if clip.trim_end is not None:
                await self.controller.execute_tool("trim_clip", {
                    "clip_index": clip.clip_index,
                    "start_seconds": clip.trim_start,
                    "end_seconds": clip.trim_end,
                })

            if clip.stabilize:
                await self.controller.execute_tool("stabilize_clip", {"clip_index": clip.clip_index})

            if clip.auto_enhance:
                await self.controller.execute_tool("auto_enhance", {"clip_index": clip.clip_index})

            if clip.speed_ramp:
                await self.controller.execute_tool("apply_speed_ramp", {
                    "clip_index": clip.clip_index,
                    "ramp_type": clip.speed_ramp,
                })
            elif clip.speed_multiplier != 1.0:
                await self.controller.execute_tool("adjust_speed", {
                    "clip_index": clip.clip_index,
                    "speed_multiplier": clip.speed_multiplier,
                })

            if clip.volume != 1.0:
                await self.controller.execute_tool("adjust_volume", {
                    "clip_index": clip.clip_index,
                    "volume": clip.volume,
                })

            if clip.filter_name:
                await self.controller.execute_tool("apply_filter", {
                    "filter_name": clip.filter_name,
                    "intensity": clip.filter_intensity,
                })

            for effect in clip.effects:
                if effect == "ken_burns":
                    await self.controller.execute_tool("apply_ken_burns", {"clip_index": clip.clip_index})
                elif effect not in ("none", ""):
                    await self.controller.execute_tool("add_effect", {"effect_name": effect})

        # 4. Beat-synced cuts
        if plan.beat_cut_timestamps and cfg.enable_beat_sync:
            self._log(f"Syncing {len(plan.beat_cut_timestamps)} cuts to the beat...")
            await self.controller.execute_tool("add_beat_sync_cuts", {
                "beat_timestamps": plan.beat_cut_timestamps[:15],  # Cap at 15 cuts
            })

        # 5. Color grade / vignette / grain / light leak
        if plan.color_grade:
            self._log(f"Applying {plan.color_grade} color grade...")
            await self.controller.execute_tool("apply_lut", {"lut_style": plan.color_grade})

        if plan.add_vignette:
            await self.controller.execute_tool("add_vignette", {"intensity": 0.4})

        if plan.add_film_grain:
            await self.controller.execute_tool("add_film_grain", {"intensity": 0.25})

        if plan.add_light_leak:
            await self.controller.execute_tool("add_light_leak", {})

        # 6. Music
        if plan.music_query:
            self._log(f"Adding music: {plan.music_query}")
            await self.controller.execute_tool("add_music", {
                "query_or_path": plan.music_query,
                "volume": plan.music_volume,
            })
            await self.controller.execute_tool("duck_audio", {"duck_level": 0.3})

        # 7. Text overlays
        for layer in plan.text_layers:
            self._log(f"Adding text: \"{layer.text}\"")
            await self.controller.execute_tool("add_text", {
                "text": layer.text,
                "position": layer.position,
                "start_seconds": layer.start_seconds,
                "duration_seconds": layer.duration_seconds,
                "font_size": layer.font_size,
                "color": layer.color,
                "animated": layer.animated,
                "style": layer.style,
            })

        # 8. Auto captions
        if plan.use_auto_captions:
            self._log("Generating auto-captions...")
            await self.controller.execute_tool("add_auto_captions", {})

        # 9. Export
        self._log(f"Exporting at {plan.export_resolution} {plan.export_fps}fps...")
        await self.controller.execute_tool("export_video", {
            "resolution": plan.export_resolution,
            "fps": plan.export_fps,
            "format": "mp4",
        })

    # ------------------------------------------------------------------
    # Plan parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_plan(data: dict) -> EditPlan:
        clips = [
            ClipEdit(
                source_path=c["source_path"],
                clip_index=c.get("clip_index", i),
                trim_start=c.get("trim_start", 0.0),
                trim_end=c.get("trim_end"),
                speed_multiplier=c.get("speed_multiplier", 1.0),
                speed_ramp=c.get("speed_ramp"),
                volume=c.get("volume", 1.0),
                filter_name=c.get("filter_name"),
                filter_intensity=c.get("filter_intensity", 0.7),
                effects=c.get("effects", []),
                stabilize=c.get("stabilize", False),
                auto_enhance=c.get("auto_enhance", False),
            )
            for i, c in enumerate(data.get("clips", []))
        ]
        text_layers = [
            TextLayer(
                text=t["text"],
                position=t.get("position", "bottom"),
                start_seconds=t.get("start_seconds", 0),
                duration_seconds=t.get("duration_seconds", 3),
                font_size=t.get("font_size", "medium"),
                color=t.get("color", "white"),
                animated=t.get("animated", False),
                style=t.get("style", "clean"),
            )
            for t in data.get("text_layers", [])
        ]
        return EditPlan(
            project_name=data.get("project_name", "Jarvis Edit"),
            aspect_ratio=data.get("aspect_ratio", "9:16"),
            template_name=data.get("template_name", ""),
            clips=clips,
            beat_cut_timestamps=data.get("beat_cut_timestamps", []),
            text_layers=text_layers,
            use_auto_captions=data.get("use_auto_captions", False),
            music_query=data.get("music_query"),
            music_volume=data.get("music_volume", 0.4),
            color_grade=data.get("color_grade", "cinematic"),
            add_vignette=data.get("add_vignette", True),
            add_film_grain=data.get("add_film_grain", False),
            add_light_leak=data.get("add_light_leak", False),
            export_resolution=data.get("export_resolution", "1080p"),
            export_fps=data.get("export_fps", 30),
            director_notes=data.get("director_notes", ""),
        )
