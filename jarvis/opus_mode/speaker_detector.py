"""
Multi-speaker detection for podcast-style content.

Detects when speaker changes occur and provides timestamps
so the pipeline can:
  - Switch which speaker is shown on screen
  - Create split-screen when two speakers talk simultaneously
  - Add name labels per speaker

Implementation:
  - Uses audio energy + simple diarization heuristics
  - Optional: pyannote.audio for pro-grade diarization (if installed)
  - Falls back to single-speaker if diarization unavailable
"""

from dataclasses import dataclass, field


@dataclass
class Speaker:
    id: str             # "Speaker_1", "Speaker_2", ...
    segments: list[tuple[float, float]] = field(default_factory=list)  # (start, end) pairs
    total_duration: float = 0.0
    name: str = ""      # filled in later if detected / provided


@dataclass
class DiarizationResult:
    speakers: list[Speaker]
    segments: list[tuple[float, float, str]]  # (start, end, speaker_id) timeline
    speaker_count: int
    is_multi_speaker: bool


def detect_speakers(
    video_path: str,
    min_segment_duration: float = 2.0,
) -> DiarizationResult:
    """
    Detect and separate speakers in a video.

    Tries pyannote.audio first (best quality), then simple energy-based
    approach, then falls back to single-speaker.
    """
    try:
        return _pyannote_diarize(video_path, min_segment_duration)
    except ImportError:
        pass
    except Exception as e:
        print(f"[SpeakerDetector] pyannote failed: {e}, falling back to energy-based")

    try:
        return _energy_diarize(video_path, min_segment_duration)
    except Exception as e:
        print(f"[SpeakerDetector] Energy diarize failed: {e}, using single speaker")

    return _single_speaker_result(video_path)


def _pyannote_diarize(video_path: str, min_segment_duration: float) -> DiarizationResult:
    """Use pyannote.audio for speaker diarization."""
    from pyannote.audio import Pipeline
    import os

    token = os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise ImportError("HUGGINGFACE_TOKEN not set — skipping pyannote")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token,
    )

    audio_path = _extract_audio_temp(video_path)
    try:
        diarization = pipeline(audio_path)
    finally:
        import os as _os
        if audio_path != video_path and _os.path.exists(audio_path):
            _os.unlink(audio_path)

    # Parse pyannote output
    speakers: dict[str, Speaker] = {}
    timeline: list[tuple[float, float, str]] = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start = round(turn.start, 3)
        end = round(turn.end, 3)
        if end - start < min_segment_duration:
            continue

        if speaker not in speakers:
            speakers[speaker] = Speaker(id=speaker)
        speakers[speaker].segments.append((start, end))
        speakers[speaker].total_duration += end - start
        timeline.append((start, end, speaker))

    timeline.sort(key=lambda x: x[0])
    speaker_list = sorted(speakers.values(), key=lambda s: s.total_duration, reverse=True)

    return DiarizationResult(
        speakers=speaker_list,
        segments=timeline,
        speaker_count=len(speaker_list),
        is_multi_speaker=len(speaker_list) > 1,
    )


def _energy_diarize(video_path: str, min_segment_duration: float) -> DiarizationResult:
    """
    Simple energy-based speaker change detection using librosa.
    Not true diarization — detects pauses and high-energy switches
    to estimate speaker turns.
    """
    import librosa
    import numpy as np

    audio_path = _extract_audio_temp(video_path)
    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
    finally:
        import os
        if audio_path != video_path and os.path.exists(audio_path):
            os.unlink(audio_path)

    # Compute RMS energy in short frames
    frame_len = int(sr * 0.05)  # 50ms frames
    hop_len = frame_len // 2
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
    times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop_len)

    # Find silences (RMS below threshold)
    threshold = np.percentile(rms, 20) * 2
    is_silent = rms < threshold

    # Find speech segments separated by silences > 0.5s
    segments = []
    in_speech = False
    seg_start = 0.0

    for i, (t, silent) in enumerate(zip(times, is_silent)):
        if not silent and not in_speech:
            seg_start = t
            in_speech = True
        elif silent and in_speech:
            if t - seg_start >= min_segment_duration:
                segments.append((seg_start, t))
            in_speech = False

    if in_speech and times[-1] - seg_start >= min_segment_duration:
        segments.append((seg_start, float(times[-1])))

    # Assign alternating speakers (naive but better than nothing)
    # Real diarization needs ML — this is a structural approximation
    sp1 = Speaker(id="Speaker_1")
    sp2 = Speaker(id="Speaker_2")
    timeline = []

    for i, (start, end) in enumerate(segments):
        speaker = sp1 if i % 2 == 0 else sp2
        speaker.segments.append((start, end))
        speaker.total_duration += end - start
        timeline.append((start, end, speaker.id))

    speaker_list = [sp1, sp2] if sp2.total_duration > 0 else [sp1]
    is_multi = len(speaker_list) > 1 and sp2.total_duration > sp1.total_duration * 0.2

    return DiarizationResult(
        speakers=speaker_list,
        segments=timeline,
        speaker_count=len(speaker_list),
        is_multi_speaker=is_multi,
    )


def _single_speaker_result(video_path: str) -> DiarizationResult:
    """Fallback: treat entire video as one speaker."""
    import subprocess, json

    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", video_path],
            capture_output=True, text=True, check=True,
        )
        data = json.loads(result.stdout)
        duration = float(data["format"]["duration"])
    except Exception:
        duration = 60.0

    sp = Speaker(id="Speaker_1", segments=[(0.0, duration)], total_duration=duration)
    return DiarizationResult(
        speakers=[sp],
        segments=[(0.0, duration, "Speaker_1")],
        speaker_count=1,
        is_multi_speaker=False,
    )


def get_dominant_speaker(
    diarization: DiarizationResult,
    clip_start: float,
    clip_end: float,
) -> str:
    """Return the speaker ID who talks most during a clip window."""
    durations: dict[str, float] = {}
    for seg_start, seg_end, speaker_id in diarization.segments:
        overlap_start = max(seg_start, clip_start)
        overlap_end = min(seg_end, clip_end)
        if overlap_end > overlap_start:
            durations[speaker_id] = durations.get(speaker_id, 0) + (overlap_end - overlap_start)

    if not durations:
        return "Speaker_1"
    return max(durations, key=durations.get)


def _extract_audio_temp(video_path: str) -> str:
    """Extract audio to a temp WAV file. Returns video_path if already audio."""
    import os, subprocess, tempfile

    ext = os.path.splitext(video_path)[1].lower()
    if ext in {".wav", ".mp3", ".flac", ".m4a"}:
        return video_path

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn",
         "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", tmp.name],
        capture_output=True, check=True,
    )
    return tmp.name
