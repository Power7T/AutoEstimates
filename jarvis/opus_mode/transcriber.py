"""
Full video transcription with word-level timestamps.
Uses Whisper to get the exact second every word is spoken.
This powers both moment detection and word-by-word captions.
"""

import os
import subprocess
import tempfile
from dataclasses import dataclass, field


@dataclass
class Word:
    text: str
    start: float      # seconds
    end: float        # seconds
    confidence: float = 1.0


@dataclass
class Segment:
    text: str
    start: float
    end: float
    words: list[Word] = field(default_factory=list)
    speaker: str = "Speaker 1"   # filled by speaker detector


@dataclass
class Transcript:
    full_text: str
    segments: list[Segment]
    words: list[Word]
    duration: float
    language: str = "en"


def transcribe(video_path: str, model_size: str = "base") -> Transcript:
    """
    Transcribe a video file with word-level timestamps.
    Returns a Transcript with every word's exact start/end time.
    """
    try:
        import whisper
    except ImportError:
        raise ImportError("openai-whisper required: pip install openai-whisper")

    audio_path = _extract_audio(video_path)

    try:
        model = whisper.load_model(model_size)
        result = model.transcribe(
            audio_path,
            word_timestamps=True,
            verbose=False,
            condition_on_previous_text=True,
            temperature=0.0,
        )
    finally:
        if audio_path != video_path and os.path.exists(audio_path):
            os.unlink(audio_path)

    words: list[Word] = []
    segments: list[Segment] = []

    for seg in result.get("segments", []):
        seg_words = []
        for w in seg.get("words", []):
            word = Word(
                text=w["word"].strip(),
                start=round(w["start"], 3),
                end=round(w["end"], 3),
                confidence=round(w.get("probability", 1.0), 3),
            )
            seg_words.append(word)
            words.append(word)

        segments.append(Segment(
            text=seg["text"].strip(),
            start=round(seg["start"], 3),
            end=round(seg["end"], 3),
            words=seg_words,
        ))

    return Transcript(
        full_text=result.get("text", "").strip(),
        segments=segments,
        words=words,
        duration=segments[-1].end if segments else 0.0,
        language=result.get("language", "en"),
    )


def _extract_audio(video_path: str) -> str:
    ext = os.path.splitext(video_path)[1].lower()
    if ext in {".mp3", ".wav", ".flac", ".m4a"}:
        return video_path
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn",
         "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", tmp.name],
        capture_output=True, check=True,
    )
    return tmp.name
