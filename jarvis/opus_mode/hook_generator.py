"""
Viral hook text-overlay generator.

Takes a ViralMoment and generates punchy on-screen hook text for the first
3 seconds of a clip — the same pattern that stops the scroll.

Usage:
    from jarvis.opus_mode.hook_generator import generate_hook, burn_hook_to_clip
    from jarvis.opus_mode.moment_finder import ViralMoment

    hook = generate_hook(moment)
    out  = burn_hook_to_clip("clip.mp4", hook, "clip_hooked.mp4")
"""

import logging
import shlex
import subprocess
from dataclasses import dataclass
from typing import Literal

from ..openrouter import router
from ..config import cfg
from .moment_finder import ViralMoment

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style → visual hints passed to FFmpeg drawtext
# ---------------------------------------------------------------------------

StyleName = Literal["bold_white", "yellow_box", "minimal"]

_STYLE_FFMPEG: dict[str, dict] = {
    "bold_white": {
        "fontcolor": "white",
        "bordercolor": "black",
        "borderw": 4,
        "box": 0,
        "boxcolor": "black@0.0",
    },
    "yellow_box": {
        "fontcolor": "black",
        "bordercolor": "black",
        "borderw": 0,
        "box": 1,
        "boxcolor": "yellow@0.92",
    },
    "minimal": {
        "fontcolor": "white",
        "bordercolor": "black@0.5",
        "borderw": 2,
        "box": 0,
        "boxcolor": "black@0.0",
    },
}

# Emotion → default style mapping used when the AI doesn't override
_EMOTION_STYLE: dict[str, StyleName] = {
    "shock":        "bold_white",
    "excitement":   "bold_white",
    "humor":        "yellow_box",
    "inspiration":  "minimal",
    "relatability": "minimal",
}

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class HookOverlay:
    text: str
    style: StyleName = "bold_white"
    duration_seconds: float = 3.0
    font_size: int = 72
    position: Literal["top", "center", "bottom"] = "top"


# ---------------------------------------------------------------------------
# System / prompt
# ---------------------------------------------------------------------------

_HOOK_SYSTEM = """You are a viral short-form video editor who specialises in
stop-the-scroll hook overlays. You know every pattern that makes people freeze
mid-swipe and watch a clip through to the end.

Rules for great hook text:
- 3 to 6 words MAXIMUM (shorter = more powerful)
- Create curiosity, tension, shock, or FOMO
- Must feel urgent — like you NEED to see what happens next
- Avoid clickbait clichés like "You won't believe…"
- Match the emotional register of the clip

Example hooks by emotion:
  shock:       "Wait for the twist..."  |  "This destroyed everything"  |  "Nobody prepared for this"
  humor:       "It gets worse 😭"       |  "He actually said this"      |  "Plot twist incoming"
  inspiration: "This changed my life"   |  "One sentence. Life altered." | "The advice nobody gives"
  excitement:  "It's finally happening" |  "Watch until the end"        |  "This broke records"
  relatability:"We've all been here"    |  "Say it louder"              |  "This is painfully true"

Style guidance:
  shock / excitement  →  bold_white  (big white text, thick black border)
  humor               →  yellow_box  (black text on yellow box)
  inspiration         →  minimal     (clean white text, subtle shadow)

Respond ONLY with a valid JSON object. No explanation outside the JSON."""

_HOOK_PROMPT = """Generate a viral hook overlay for this video clip.

CLIP TITLE:      {title}
HOOK SENTENCE:   {hook}
EMOTION:         {emotion}
CLIP TYPE:       {clip_type}
VIRALITY SCORE:  {virality_score}/100
TRANSCRIPT:
{transcript_excerpt}

Return a JSON object with exactly these keys:
{{
  "text":             "<3-6 word hook>",
  "style":            "<bold_white | yellow_box | minimal>",
  "duration_seconds": <float, typically 2.5-4.0>,
  "font_size":        <int, 60-96>,
  "position":        "<top | center | bottom>"
}}"""


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------


def generate_hook(moment: ViralMoment) -> HookOverlay:
    """Ask the director model for the best hook overlay for *moment*.

    Falls back to a rule-based hook if the AI call fails.
    """
    prompt = _HOOK_PROMPT.format(
        title=moment.title,
        hook=moment.hook,
        emotion=moment.emotion,
        clip_type=moment.clip_type,
        virality_score=moment.virality_score,
        transcript_excerpt=moment.transcript_excerpt,
    )

    try:
        data = router.complete_json(
            prompt=prompt,
            system=_HOOK_SYSTEM,
            model=cfg.models.director,
            max_tokens=256,
        )
    except Exception as exc:
        log.warning("Hook AI call failed (%s) — using rule-based fallback", exc)
        return _fallback_hook(moment)

    # Validate / coerce the returned fields
    text = str(data.get("text", moment.hook[:40])).strip()
    if not text:
        text = moment.hook[:40]

    raw_style = data.get("style", _EMOTION_STYLE.get(moment.emotion, "bold_white"))
    style: StyleName = raw_style if raw_style in _STYLE_FFMPEG else "bold_white"

    try:
        duration = float(data.get("duration_seconds", 3.0))
        duration = max(1.0, min(duration, 6.0))
    except (TypeError, ValueError):
        duration = 3.0

    try:
        font_size = int(data.get("font_size", 72))
        font_size = max(40, min(font_size, 120))
    except (TypeError, ValueError):
        font_size = 72

    raw_pos = data.get("position", "top")
    position = raw_pos if raw_pos in ("top", "center", "bottom") else "top"

    return HookOverlay(
        text=text,
        style=style,
        duration_seconds=duration,
        font_size=font_size,
        position=position,
    )


def _fallback_hook(moment: ViralMoment) -> HookOverlay:
    """Rule-based fallback used when the AI call fails."""
    _templates: dict[str, str] = {
        "shock":        "Wait for the twist...",
        "excitement":   "This changes everything",
        "humor":        "It gets worse...",
        "inspiration":  "This changed my life",
        "relatability": "We've all been here",
    }
    text = _templates.get(moment.emotion, moment.hook[:40])
    style: StyleName = _EMOTION_STYLE.get(moment.emotion, "bold_white")
    return HookOverlay(text=text, style=style, duration_seconds=3.0, font_size=72, position="top")


# ---------------------------------------------------------------------------
# FFmpeg burn-in
# ---------------------------------------------------------------------------


def burn_hook_to_clip(video_path: str, hook: HookOverlay, output_path: str) -> str:
    """Burn the hook text overlay into the first N seconds of *video_path*.

    The text fades in over 0.3 s, holds, then fades out over 0.3 s.
    Requires FFmpeg with libfreetype support (drawtext filter).

    Returns *output_path*.
    """
    style = _STYLE_FFMPEG.get(hook.style, _STYLE_FFMPEG["bold_white"])
    duration = hook.duration_seconds
    fade_dur = 0.3

    # Vertical position expression
    _y_expr: dict[str, str] = {
        "top":    "h*0.08",
        "center": "(h-text_h)/2",
        "bottom": "h*0.82",
    }
    y_expr = _y_expr.get(hook.position, "h*0.08")

    # Fade-in / fade-out alpha expression.
    # Ramp up over fade_dur, hold at 1, ramp down over fade_dur.
    alpha_expr = (
        f"if(lt(t,{fade_dur}),"
        f"t/{fade_dur},"
        f"if(lt(t,{duration - fade_dur}),"
        f"1,"
        f"if(lt(t,{duration}),"
        f"({duration}-t)/{fade_dur},"
        f"0)))"
    )

    # Escape text for FFmpeg drawtext (backslash, single-quote, colon are special)
    safe_text = (
        hook.text
        .replace("\\", "\\\\")
        .replace("'",  "\\'")
        .replace(":",  "\\:")
    )

    # Preferred font path — falls back to empty (FFmpeg default) on failure
    font_path = "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf"

    def _build_drawtext(include_font: bool) -> str:
        parts = [
            f"text='{safe_text}'",
            f"fontsize={hook.font_size}",
            f"fontcolor={style['fontcolor']}",
            f"bordercolor={style['bordercolor']}",
            f"borderw={style['borderw']}",
            f"x=(w-text_w)/2",
            f"y={y_expr}",
            f"alpha='{alpha_expr}'",
            f"enable='between(t,0,{duration})'",
        ]
        if include_font:
            parts.insert(1, f"fontfile={font_path}")
        if style["box"]:
            parts += [
                "box=1",
                f"boxcolor={style['boxcolor']}",
                "boxborderw=12",
            ]
        else:
            # Drop-shadow for non-box styles
            parts += [
                "shadowcolor=black@0.6",
                "shadowx=3",
                "shadowy=3",
            ]
        return ":".join(parts)

    def _run(drawtext_filter: str) -> subprocess.CompletedProcess:
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"drawtext={drawtext_filter}",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-c:a", "copy",
            output_path,
        ]
        log.debug("burn_hook_to_clip: %s", shlex.join(cmd))
        return subprocess.run(cmd, capture_output=True, text=True)

    # First attempt: explicit Arial Bold font
    result = _run(_build_drawtext(include_font=True))
    if result.returncode != 0:
        log.warning(
            "FFmpeg drawtext failed with Arial Bold — retrying with default font.\n%s",
            result.stderr[-800:],
        )
        # Second attempt: let FFmpeg choose the system default font
        result2 = _run(_build_drawtext(include_font=False))
        if result2.returncode != 0:
            raise RuntimeError(
                f"FFmpeg hook burn-in failed:\n{result2.stderr[-1200:]}"
            )

    log.info("Hook burned into %s", output_path)
    return output_path
