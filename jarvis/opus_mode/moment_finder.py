"""
AI-powered viral moment detection.

Reads the full transcript of a long video and finds the
10 most viral-worthy clips — the same way OpusClip does it,
but powered by Claude for deeper understanding.

For each moment it finds:
  - Exact start/end timestamps
  - The hook sentence (what makes you stop scrolling)
  - Why it will perform
  - Virality score 0-100
  - Suggested caption style
"""

import json
from dataclasses import dataclass, field

from ..openrouter import router
from ..config import cfg
from .transcriber import Transcript


@dataclass
class ViralMoment:
    start: float                  # seconds into original video
    end: float                    # seconds
    duration: float               # end - start
    hook: str                     # the attention-grabbing opener sentence
    title: str                    # short clip title
    virality_score: float         # 0-100
    virality_reason: str          # why this will perform
    emotion: str                  # excitement | shock | humor | inspiration | relatability
    clip_type: str                # quote | story | fact | moment | argument | revelation
    suggested_caption_style: str  # bold | clean | kinetic
    transcript_excerpt: str       # the actual words spoken


MOMENT_FINDER_SYSTEM = """You are an expert viral content strategist who has studied
millions of short-form videos and knows exactly what makes people stop scrolling.

You will be given a full transcript of a long video.
Your job is to find the best viral clip moments — the same way OpusClip does,
but with deeper creative judgment.

What makes a moment viral:
1. HOOK — First sentence makes you NEED to know what comes next
2. EMOTION — Makes you feel something (shock, laugh, inspired, seen)
3. QUOTABLE — Something people want to screenshot or share
4. COMPLETE — Has a clear beginning, middle, end within 30-90 seconds
5. UNIVERSAL — Anyone can relate, not just fans of the creator
6. SURPRISE — A reveal, twist, or unexpected perspective

Return ONLY valid JSON. No explanation."""

MOMENT_FINDER_PROMPT = """Find the {n_clips} best viral moments in this transcript.

VIDEO DURATION: {duration:.0f} seconds ({duration_min:.1f} minutes)
LANGUAGE: {language}

FULL TRANSCRIPT:
{transcript}

For each moment return the exact timestamps from the transcript.
Clips should be 30-90 seconds long. Never overlap clips.
Find moments that work as STANDALONE content — no context needed.

Return JSON:
{{
  "moments": [
    {{
      "start": seconds_float,
      "end": seconds_float,
      "title": "short punchy title (max 6 words)",
      "hook": "the exact first sentence that makes someone stop scrolling",
      "virality_score": 0-100,
      "virality_reason": "specific explanation of why this will perform",
      "emotion": "excitement|shock|humor|inspiration|relatability|curiosity",
      "clip_type": "quote|story|fact|moment|argument|revelation",
      "suggested_caption_style": "bold|clean|kinetic",
      "transcript_excerpt": "the actual words spoken in this clip"
    }}
  ]
}}

Sort by virality_score descending. Return exactly {n_clips} moments."""


def find_viral_moments(
    transcript: Transcript,
    n_clips: int = 10,
    min_duration: float = 25.0,
    max_duration: float = 90.0,
) -> list[ViralMoment]:
    """
    Find the top N viral moments in a transcript.
    Uses Claude to understand context, emotion, and viral potential.
    """
    # Build transcript text with timestamps
    transcript_text = _format_transcript(transcript, max_chars=12000)

    prompt = MOMENT_FINDER_PROMPT.format(
        n_clips=n_clips,
        duration=transcript.duration,
        duration_min=transcript.duration / 60,
        language=transcript.language,
        transcript=transcript_text,
    )

    data = router.complete_json(
        prompt=prompt,
        system=MOMENT_FINDER_SYSTEM,
        model=cfg.models.director,
        max_tokens=4096,
    )

    moments = []
    for m in data.get("moments", []):
        start = float(m.get("start", 0))
        end = float(m.get("end", start + 60))

        # Enforce duration limits
        duration = end - start
        if duration < min_duration:
            end = start + min_duration
        if duration > max_duration:
            end = start + max_duration

        # Clamp to video duration
        end = min(end, transcript.duration)
        if start >= end:
            continue

        moments.append(ViralMoment(
            start=round(start, 2),
            end=round(end, 2),
            duration=round(end - start, 2),
            hook=m.get("hook", ""),
            title=m.get("title", f"Clip {len(moments)+1}"),
            virality_score=float(m.get("virality_score", 50)),
            virality_reason=m.get("virality_reason", ""),
            emotion=m.get("emotion", "inspiration"),
            clip_type=m.get("clip_type", "moment"),
            suggested_caption_style=m.get("suggested_caption_style", "bold"),
            transcript_excerpt=m.get("transcript_excerpt", ""),
        ))

    # Sort by virality score
    moments.sort(key=lambda x: x.virality_score, reverse=True)
    return moments[:n_clips]


def _format_transcript(transcript: Transcript, max_chars: int = 12000) -> str:
    """Format transcript with timestamps for the prompt."""
    lines = []
    for seg in transcript.segments:
        ts = f"[{_fmt_time(seg.start)} → {_fmt_time(seg.end)}]"
        lines.append(f"{ts} {seg.text}")
    full = "\n".join(lines)
    if len(full) > max_chars:
        # Keep first + last portions
        half = max_chars // 2
        full = full[:half] + "\n...[middle trimmed]...\n" + full[-half:]
    return full


def _fmt_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"
