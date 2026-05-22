"""
Word-by-word animated captions — OpusClip style.

Every word appears EXACTLY when it is spoken.
The active word is highlighted (bright/colored).
Previous words stay visible but dimmer.
One line at a time (3-4 words max).

Implementation: ASS subtitle format via FFmpeg.
ASS supports karaoke-style word highlighting natively.
This is the same format used by professional subtitle tools.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path

from .transcriber import Word, Transcript


@dataclass
class CaptionStyle:
    font_name: str = "Arial"
    font_size: int = 72              # Large, easy to read
    primary_color: str = "&H00FFFFFF"   # White (ASS format: &HAABBGGRR)
    highlight_color: str = "&H0000FFFF" # Yellow highlight for active word
    outline_color: str = "&H00000000"   # Black outline
    shadow_color: str = "&H80000000"    # Semi-transparent shadow
    bold: bool = True
    outline_width: float = 3.0
    shadow_depth: float = 2.0
    margin_bottom: int = 150            # pixels from bottom
    words_per_line: int = 4             # max words shown at once
    uppercase: bool = True             # ALL CAPS like OpusClip


STYLES = {
    "bold": CaptionStyle(
        font_size=76, bold=True, uppercase=True,
        highlight_color="&H0000FFFF",   # yellow
    ),
    "clean": CaptionStyle(
        font_size=64, bold=False, uppercase=False,
        highlight_color="&H000080FF",   # orange
    ),
    "kinetic": CaptionStyle(
        font_size=80, bold=True, uppercase=True,
        highlight_color="&H000000FF",   # red
    ),
}


def generate_caption_file(
    transcript: Transcript,
    output_path: str,
    style_name: str = "bold",
    clip_start: float = 0.0,
    clip_end: float | None = None,
) -> str:
    """
    Generate an ASS subtitle file for a clip.

    clip_start/clip_end: timestamps in the original video.
    All times are offset so clip_start becomes 0:00:00 in the output.

    Returns path to the generated .ass file.
    """
    style = STYLES.get(style_name, STYLES["bold"])
    clip_end = clip_end or transcript.duration

    # Filter words within the clip
    clip_words = [
        w for w in transcript.words
        if w.start >= clip_start and w.end <= clip_end + 0.5
    ]

    if not clip_words:
        return ""

    # Offset times so clip starts at 0
    offset_words = [
        Word(
            text=w.text.upper() if style.uppercase else w.text,
            start=round(w.start - clip_start, 3),
            end=round(w.end - clip_start, 3),
            confidence=w.confidence,
        )
        for w in clip_words
        if w.start >= clip_start
    ]

    ass_content = _build_ass(offset_words, style)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(ass_content, encoding="utf-8")
    return output_path


def _build_ass(words: list[Word], style: CaptionStyle) -> str:
    """Build complete ASS file content."""
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes
YCbCr Matrix: None

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{style.font_name},{style.font_size},{style.primary_color},{style.highlight_color},{style.outline_color},{style.shadow_color},{1 if style.bold else 0},0,0,0,100,100,0,0,1,{style.outline_width},{style.shadow_depth},2,30,30,{style.margin_bottom},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = _build_events(words, style)
    return header + events


def _build_events(words: list[Word], style: CaptionStyle) -> str:
    """Build ASS dialogue events with karaoke-style word highlighting."""
    if not words:
        return ""

    lines = []
    # Group words into lines of N words each
    groups = _group_words(words, style.words_per_line)

    for group in groups:
        if not group:
            continue

        group_start = group[0].start
        group_end = group[-1].end

        # Build karaoke text: {\\k<centiseconds>}word
        # \\k = karaoke — word changes to SecondaryColour (highlight) during its duration
        karaoke_text = ""
        for word in group:
            duration_cs = max(1, int((word.end - word.start) * 100))  # centiseconds
            # Clean the word text for ASS format
            clean = _clean_text(word.text)
            karaoke_text += f"{{\\k{duration_cs}}}{clean} "

        karaoke_text = karaoke_text.rstrip()

        lines.append(
            f"Dialogue: 0,{_tc(group_start)},{_tc(group_end)},"
            f"Default,,0,0,0,,{{\\K0}}{karaoke_text}"
        )

    return "\n".join(lines) + "\n"


def _group_words(words: list[Word], words_per_line: int) -> list[list[Word]]:
    """Group words into lines, breaking on natural pauses or word count."""
    groups = []
    current = []

    for i, word in enumerate(words):
        current.append(word)

        # Break conditions
        is_last = i == len(words) - 1
        at_limit = len(current) >= words_per_line
        long_pause = (
            i + 1 < len(words) and
            words[i + 1].start - word.end > 0.4  # 400ms pause = new line
        )
        ends_sentence = word.text.rstrip().endswith((".", "!", "?", ","))

        if is_last or at_limit or (long_pause and len(current) >= 2) or (ends_sentence and len(current) >= 2):
            groups.append(current)
            current = []

    if current:
        groups.append(current)

    return groups


def _tc(seconds: float) -> str:
    """Convert seconds to ASS timecode H:MM:SS.CC"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _clean_text(text: str) -> str:
    """Escape special ASS characters."""
    text = text.strip()
    text = text.replace("{", "").replace("}", "")
    text = text.replace("\\", "\\\\")
    return text


def ffmpeg_caption_filter(ass_file: str) -> str:
    """Return the FFmpeg filter string to burn captions into video."""
    # Escape path for FFmpeg on different platforms
    safe_path = ass_file.replace("\\", "/").replace(":", "\\:")
    return f"ass={safe_path}"
