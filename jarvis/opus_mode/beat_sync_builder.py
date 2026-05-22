"""
beat_sync_builder.py

Produces a beat-synced montage from multiple video clips and a music file.

Workflow
--------
1. Analyse the music track with ``jarvis.audio_analyzer.AudioAnalyzer`` to
   obtain beat timestamps, tempo, and energy curve.
2. Divide the beat grid into windows of *beats_per_clip* beats each and assign
   one source clip per window.
3. For each window, calculate the speed adjustment needed so the clip exactly
   fills the beat window, then trim, scale, and speed-adjust the clip via
   FFmpeg.
4. Concatenate all processed clips and mix in the music track, dropping the
   original clip audio.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field

from jarvis.audio_analyzer import analyze_audio, select_beat_cut_points  # noqa: F401

logger = logging.getLogger(__name__)

# Maximum source material used per beat window: we draw from at most this
# multiple of the window duration so a clip is never stretched beyond usability.
_MAX_SOURCE_RATIO: float = 2.0

# Playback speed limits applied to each clip.
_MIN_SPEED: float = 0.5
_MAX_SPEED: float = 2.0


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BeatSyncClip:
    """Describes how one source clip maps to a beat interval in the timeline."""

    source_path: str
    source_start: float   # where inside the source clip we start (seconds)
    source_end: float     # where inside the source clip we end (seconds)
    beat_start: float     # position of this clip in the final timeline (seconds)
    beat_end: float       # end position in the final timeline (seconds)
    speed: float = 1.0   # playback speed factor (1.0 = normal, 1.2 = 20% faster)


@dataclass
class BeatSyncPlan:
    """Full assignment plan produced before rendering."""

    clips: list[BeatSyncClip] = field(default_factory=list)
    total_duration: float = 0.0
    bpm: float = 0.0
    beat_count: int = 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_beat_sync_montage(
    clip_paths: list[str],
    music_path: str,
    output_path: str,
    beats_per_clip: int = 2,
    target_width: int = 1080,
    target_height: int = 1920,
) -> BeatSyncPlan:
    """Create a beat-synced montage from *clip_paths* set to *music_path*.

    Parameters
    ----------
    clip_paths:
        Ordered list of source video files.  At least one path is required.
    music_path:
        Audio or video file whose audio track drives the beat grid.
    output_path:
        Destination for the rendered montage MP4.
    beats_per_clip:
        Number of beat intervals each clip should span in the output.
        Higher values produce longer, less frenetic cuts.
    target_width:
        Output frame width in pixels (default 1080 for 9:16 portrait).
    target_height:
        Output frame height in pixels (default 1920 for 9:16 portrait).

    Returns
    -------
    BeatSyncPlan
        The assignment plan used for rendering.

    Raises
    ------
    ValueError
        If *clip_paths* is empty or *beats_per_clip* is less than 1.
    FileNotFoundError
        If *music_path* or any clip path does not exist.
    RuntimeError
        If beat planning produces no clips or FFmpeg fails.
    """
    if not clip_paths:
        raise ValueError("clip_paths must not be empty.")
    if beats_per_clip < 1:
        raise ValueError("beats_per_clip must be at least 1.")
    if not os.path.isfile(music_path):
        raise FileNotFoundError(f"Music file not found: {music_path}")
    for cp in clip_paths:
        if not os.path.isfile(cp):
            raise FileNotFoundError(f"Clip not found: {cp}")

    logger.info("Analysing music: %s", music_path)
    analysis = analyze_audio(music_path)
    logger.info(
        "BPM=%.1f, %d beats detected, mood=%s",
        analysis.tempo_bpm,
        len(analysis.beat_timestamps),
        analysis.mood,
    )

    plan = _plan_clip_assignments(clip_paths, analysis.beat_timestamps, beats_per_clip)
    plan.bpm = analysis.tempo_bpm
    plan.beat_count = len(analysis.beat_timestamps)

    if not plan.clips:
        raise RuntimeError(
            "Beat planning produced no clips — check that the music track has "
            "detectable beat timestamps."
        )

    _render_montage(plan, music_path, output_path, target_width, target_height)
    logger.info(
        "Beat-sync montage complete: %d clips, %.2fs total → %s",
        len(plan.clips),
        plan.total_duration,
        output_path,
    )
    return plan


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def _plan_clip_assignments(
    clip_paths: list[str],
    beat_timestamps: list[float],
    beats_per_clip: int,
) -> BeatSyncPlan:
    """Map each source clip to a beat-aligned time window.

    The beat grid is divided into non-overlapping windows of *beats_per_clip*
    beats each.  Clips are consumed in order; excess windows are discarded so
    that the montage spans exactly ``len(clip_paths)`` windows (or fewer if
    the music doesn't contain enough beats).

    Parameters
    ----------
    clip_paths:
        Source video file paths.
    beat_timestamps:
        Beat positions in seconds, as returned by AudioAnalyzer.
    beats_per_clip:
        Number of beats per window.

    Returns
    -------
    BeatSyncPlan
        Plan with speed/trim metadata per clip.  `bpm` and `beat_count` are
        filled in by the caller.
    """
    if not beat_timestamps:
        raise ValueError("No beat timestamps available for planning.")

    beats = sorted(beat_timestamps)
    n_clips = len(clip_paths)

    # Build windows: each window spans beats[i] → beats[i + beats_per_clip]
    windows: list[tuple[float, float]] = []
    i = 0
    while i < len(beats) and len(windows) < n_clips:
        w_start = beats[i]
        next_idx = i + beats_per_clip
        if next_idx < len(beats):
            w_end = beats[next_idx]
        else:
            # Estimate end of last window from the average beat period.
            if len(beats) >= 2:
                avg_period = (beats[-1] - beats[0]) / (len(beats) - 1)
            else:
                avg_period = 60.0 / 120.0  # 120 BPM fallback
            w_end = beats[-1] + avg_period * beats_per_clip
        windows.append((w_start, w_end))
        i += beats_per_clip

    if not windows:
        # Absolute fallback: evenly divide the beat span.
        span_start = beats[0] if beats else 0.0
        span_end = beats[-1] if len(beats) > 1 else span_start + 4.0
        clip_dur = (span_end - span_start) / n_clips
        windows = [
            (span_start + k * clip_dur, span_start + (k + 1) * clip_dur)
            for k in range(n_clips)
        ]

    assigned: list[BeatSyncClip] = []

    for idx, (window_start, window_end) in enumerate(windows):
        source_path = clip_paths[idx]
        beat_duration = window_end - window_start

        if beat_duration <= 0:
            logger.warning(
                "Window %d has zero or negative duration (%.3f – %.3f); skipping.",
                idx, window_start, window_end,
            )
            continue

        source_duration = _probe_duration(source_path)

        # Decide how much of the source clip to use.
        # We cap at _MAX_SOURCE_RATIO × beat_duration to avoid extreme slow-downs.
        source_use = min(source_duration, beat_duration * _MAX_SOURCE_RATIO)

        # Speed = source_use / beat_duration
        # speed > 1 → clip plays faster (more source material fits the window)
        # speed < 1 → clip plays slower (less source material stretched to fill)
        raw_speed = source_use / beat_duration
        speed = max(_MIN_SPEED, min(_MAX_SPEED, raw_speed))

        # Recompute actual source end based on clamped speed
        actual_source_end = min(source_duration, beat_duration * speed)

        assigned.append(BeatSyncClip(
            source_path=source_path,
            source_start=0.0,
            source_end=round(actual_source_end, 3),
            beat_start=round(window_start, 3),
            beat_end=round(window_end, 3),
            speed=round(speed, 4),
        ))

    total_duration = assigned[-1].beat_end if assigned else 0.0

    return BeatSyncPlan(
        clips=assigned,
        total_duration=round(total_duration, 3),
        bpm=0.0,      # filled in by caller
        beat_count=0, # filled in by caller
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _render_montage(
    plan: BeatSyncPlan,
    music_path: str,
    output_path: str,
    target_width: int,
    target_height: int,
) -> str:
    """Render the BeatSyncPlan to a final MP4.

    Strategy
    --------
    1. For each clip: trim to its assigned source window, scale to
       ``target_width × target_height`` (with black-bar padding), and apply
       speed adjustment via ``setpts`` (video) and ``atempo`` (audio).
    2. Concatenate the processed segments via the FFmpeg concat demuxer.
    3. Drop the original clip audio and mix in the music track, trimmed to
       the total montage duration.

    Parameters
    ----------
    plan:
        Beat assignment plan.
    music_path:
        Music file to mix into the final output.
    output_path:
        Destination MP4 path.
    target_width / target_height:
        Output frame dimensions.

    Returns
    -------
    str
        The *output_path* on success.

    Raises
    ------
    RuntimeError
        If any FFmpeg invocation fails.
    """
    tmp_dir = tempfile.mkdtemp(prefix="beat_sync_")
    segment_paths: list[str] = []

    try:
        # ------------------------------------------------------------------ #
        # Step 1: render each clip segment                                    #
        # ------------------------------------------------------------------ #
        for idx, clip in enumerate(plan.clips):
            seg_path = os.path.join(tmp_dir, f"seg_{idx:04d}.mp4")
            _render_clip_segment(clip, seg_path, target_width, target_height)
            segment_paths.append(seg_path)

        if not segment_paths:
            raise RuntimeError("No segments were rendered — cannot build montage.")

        # ------------------------------------------------------------------ #
        # Step 2: write concat list                                           #
        # ------------------------------------------------------------------ #
        concat_list = os.path.join(tmp_dir, "concat.txt")
        with open(concat_list, "w", encoding="utf-8") as fh:
            for sp in segment_paths:
                fh.write(f"file '{sp}'\n")

        # ------------------------------------------------------------------ #
        # Step 3: concatenate clip segments (video only)                     #
        # ------------------------------------------------------------------ #
        merged_video = os.path.join(tmp_dir, "merged_video.mp4")
        cmd_concat = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list,
            "-an",                  # strip clip audio; music added next
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            merged_video,
        ]
        logger.debug("Concatenating %d clip segment(s) into merged video", len(segment_paths))
        result = subprocess.run(cmd_concat, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg concat failed (exit {result.returncode}):\n"
                f"{result.stderr[-2000:]}"
            )

        # ------------------------------------------------------------------ #
        # Step 4: mix in music track, output final MP4                       #
        # ------------------------------------------------------------------ #
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cmd_mix = [
            "ffmpeg", "-y",
            "-i", merged_video,
            "-i", music_path,
            "-map", "0:v:0",        # video from merged clips
            "-map", "1:a:0",        # audio from music track
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-t", str(plan.total_duration),
            "-shortest",            # stop at whichever stream ends first
            "-movflags", "+faststart",
            output_path,
        ]
        logger.debug("Mixing music → final output: %s", output_path)
        result = subprocess.run(cmd_mix, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg music mix failed (exit {result.returncode}):\n"
                f"{result.stderr[-2000:]}"
            )

    finally:
        # Best-effort cleanup of all temp files
        for sp in segment_paths:
            try:
                os.remove(sp)
            except OSError:
                pass
        for name in ("concat.txt", "merged_video.mp4"):
            try:
                os.remove(os.path.join(tmp_dir, name))
            except OSError:
                pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass

    return output_path


def _render_clip_segment(
    clip: BeatSyncClip,
    output_path: str,
    target_width: int,
    target_height: int,
) -> None:
    """Trim, scale, and speed-adjust a single :class:`BeatSyncClip`.

    Video speed is achieved by manipulating PTS (presentation timestamps):
    ``setpts = (1/speed) * PTS``.  Audio speed uses the ``atempo`` filter,
    chained for speeds outside ``[0.5, 2.0]``.

    Parameters
    ----------
    clip:
        The clip assignment to render.
    output_path:
        Destination file path for this segment.
    target_width / target_height:
        Output frame size.

    Raises
    ------
    RuntimeError
        If FFmpeg exits with a non-zero return code.
    """
    beat_duration = clip.beat_end - clip.beat_start
    pts_factor = 1.0 / clip.speed  # setpts factor: speed=2 → 0.5× timestamps

    # Scale to target size, padding with black bars to preserve aspect ratio.
    scale_filter = (
        f"scale={target_width}:{target_height}"
        f":force_original_aspect_ratio=decrease,"
        f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:black"
    )
    vf = f"{scale_filter},setpts={pts_factor:.6f}*PTS"
    af = _build_atempo_chain(clip.speed)

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(clip.source_start),
        "-to", str(clip.source_end),
        "-i", clip.source_path,
        "-vf", vf,
        "-af", af,
        "-t", str(beat_duration),   # hard-trim to exact beat window length
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "128k",
        "-r", "30",                 # normalise frame rate across all clips
        output_path,
    ]
    logger.debug(
        "Rendering clip: %s | speed=%.3f | src=%.2f–%.2f s | beat_dur=%.2f s",
        os.path.basename(clip.source_path),
        clip.speed,
        clip.source_start,
        clip.source_end,
        beat_duration,
    )
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg clip render failed for '{clip.source_path}' "
            f"(exit {result.returncode}):\n{result.stderr[-2000:]}"
        )


def _build_atempo_chain(speed: float) -> str:
    """Build an ``atempo`` filter chain that handles arbitrary speed values.

    FFmpeg's ``atempo`` filter only accepts values in ``[0.5, 2.0]`` per
    stage, so speeds outside that range require chaining multiple stages.

    Parameters
    ----------
    speed:
        Desired playback speed factor (e.g. 1.5 = 50% faster).

    Returns
    -------
    str
        A comma-separated FFmpeg audio filter string, e.g.
        ``"atempo=2.0,atempo=1.25"`` for speed=2.5.
    """
    if abs(speed - 1.0) < 1e-6:
        return "anull"

    filters: list[str] = []
    remaining = speed

    # Chain atempo=2.0 stages for large speed-ups
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0

    # Chain atempo=0.5 stages for large slow-downs
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining *= 2.0

    # Final fractional stage
    if abs(remaining - 1.0) > 1e-6:
        filters.append(f"atempo={remaining:.6f}")

    return ",".join(filters) if filters else "anull"


# ---------------------------------------------------------------------------
# FFmpeg probe helper
# ---------------------------------------------------------------------------

def _probe_duration(video_path: str) -> float:
    """Return the duration of a media file in seconds via ``ffprobe``.

    Falls back to 5.0 seconds if probing fails (e.g. file is malformed or
    ffprobe is not installed).
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
        if duration <= 0:
            raise ValueError(f"Non-positive duration: {duration}")
        return duration
    except (subprocess.CalledProcessError, ValueError, OSError) as exc:
        logger.warning(
            "Could not probe duration for '%s': %s — defaulting to 5.0 s",
            video_path,
            exc,
        )
        return 5.0
