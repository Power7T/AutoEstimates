"""
Audio analysis — beat detection, energy mapping, mood detection.
Uses librosa for all audio intelligence.
"""

import os
import subprocess
import tempfile
from dataclasses import dataclass, field

import numpy as np


@dataclass
class AudioAnalysis:
    path: str
    duration_seconds: float
    tempo_bpm: float
    beat_timestamps: list[float]          # Exact seconds of every beat
    downbeat_timestamps: list[float]      # Every 4th beat (strong beats)
    energy_curve: list[float]             # Normalized energy per second
    drop_timestamps: list[float]          # High-energy peaks (music drops)
    mood: str                             # detected mood
    has_speech: bool
    silence_regions: list[tuple[float, float]]  # (start, end) of silent sections


def analyze_audio(video_or_audio_path: str) -> AudioAnalysis:
    """Extract full audio intelligence from a video or audio file."""
    try:
        import librosa
    except ImportError:
        raise ImportError("librosa is required: pip install librosa soundfile")

    audio_path = _extract_audio(video_or_audio_path)

    try:
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)

        # Beat tracking
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
        beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
        tempo_bpm = float(tempo) if not isinstance(tempo, np.ndarray) else float(tempo[0])

        # Downbeats (every 4th beat = strong beat on "1")
        downbeats = beat_times[::4]

        # Energy curve (RMS energy per second)
        hop_length = sr  # 1-second windows
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        energy_norm = (rms / (rms.max() + 1e-8)).tolist()

        # Drop detection (energy spikes above 80th percentile)
        threshold = np.percentile(rms, 80)
        drop_frames = np.where(rms > threshold)[0]
        drop_times = librosa.frames_to_time(drop_frames, hop_length=hop_length, sr=sr).tolist()
        # Cluster nearby drops (within 2 seconds = same drop)
        drop_timestamps = _cluster_timestamps(drop_times, gap=2.0)

        # Silence detection
        silence_regions = _detect_silence(y, sr)

        # Mood estimation from spectral features
        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        mood = _estimate_mood(tempo_bpm, spectral_centroid, float(np.mean(rms)))

        # Speech detection (rough heuristic via zero-crossing rate)
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        has_speech = zcr > 0.08

        return AudioAnalysis(
            path=audio_path,
            duration_seconds=round(duration, 2),
            tempo_bpm=round(tempo_bpm, 1),
            beat_timestamps=[round(t, 3) for t in beat_times],
            downbeat_timestamps=[round(t, 3) for t in downbeats],
            energy_curve=[round(e, 3) for e in energy_norm],
            drop_timestamps=[round(t, 3) for t in drop_timestamps],
            mood=mood,
            has_speech=has_speech,
            silence_regions=silence_regions,
        )
    finally:
        # Cleanup temp file if we created one
        if audio_path != video_or_audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)


def select_beat_cut_points(
    analysis: AudioAnalysis,
    clip_durations: list[float],
    max_cuts: int = 20,
) -> list[float]:
    """
    Select the best beat timestamps to use as cut points
    given a list of clip durations and a maximum number of cuts.
    """
    total = sum(clip_durations)
    # Filter beats to those within the video duration
    beats = [t for t in analysis.beat_timestamps if t < total]

    # Prefer downbeats and drop points
    priority = set(analysis.downbeat_timestamps + analysis.drop_timestamps)
    priority_beats = [b for b in beats if any(abs(b - p) < 0.1 for p in priority)]

    # Fill up to max_cuts with regular beats if needed
    all_cuts = sorted(set(priority_beats + beats[:max_cuts]))
    return all_cuts[:max_cuts]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_audio(video_path: str) -> str:
    """Extract audio track from a video file to a temp WAV file."""
    ext = os.path.splitext(video_path)[1].lower()
    if ext in {".mp3", ".wav", ".flac", ".ogg", ".m4a"}:
        return video_path  # Already audio

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
             "-ar", "22050", "-ac", "1", tmp.name],
            capture_output=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # ffmpeg not available — return original path, librosa will handle it
        os.unlink(tmp.name)
        return video_path
    return tmp.name


def _cluster_timestamps(timestamps: list[float], gap: float = 2.0) -> list[float]:
    """Merge timestamps that are within `gap` seconds of each other."""
    if not timestamps:
        return []
    clusters = [[timestamps[0]]]
    for t in timestamps[1:]:
        if t - clusters[-1][-1] < gap:
            clusters[-1].append(t)
        else:
            clusters.append([t])
    return [round(sum(c) / len(c), 3) for c in clusters]


def _detect_silence(y, sr, threshold_db: float = -40.0) -> list[tuple[float, float]]:
    """Return list of (start, end) silence regions in seconds."""
    try:
        import librosa
        intervals = librosa.effects.split(y, top_db=abs(threshold_db))
        # Invert: find gaps between non-silent intervals
        silence = []
        prev_end = 0.0
        for start, end in librosa.frames_to_time(intervals, sr=sr):
            if start > prev_end + 0.5:
                silence.append((round(prev_end, 2), round(float(start), 2)))
            prev_end = float(end)
        return silence
    except Exception:
        return []


def _estimate_mood(tempo: float, spectral_centroid: float, rms_energy: float) -> str:
    """Rough mood label from audio features."""
    if tempo > 140 and rms_energy > 0.05:
        return "energetic"
    if tempo > 120 and spectral_centroid > 3000:
        return "upbeat"
    if tempo < 80 and rms_energy < 0.03:
        return "calm"
    if tempo < 90 and spectral_centroid < 2000:
        return "melancholic"
    if tempo > 100:
        return "happy"
    return "neutral"
