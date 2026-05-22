"""
Platform-specific video export.

Each platform has strict specs for resolution, bitrate, duration, and audio.
This module ensures every exported clip is compliant and optimised.

Loudness normalisation uses the EBU R128 standard (I=-14 LUFS, TP=-1, LRA=11)
which is what TikTok, Instagram, and YouTube all target internally.
Two-pass loudnorm is used so the correction is linear and artifact-free.

Usage:
    from jarvis.opus_mode.platform_exporter import (
        export_for_platform,
        export_all_platforms,
        validate_for_platform,
        PLATFORM_SPECS,
    )

    out = export_for_platform("clip.mp4", "clip_ig.mp4", platform="instagram")
    issues = validate_for_platform("clip_ig.mp4", "instagram")
"""

import json
import logging
import os
import shlex
import subprocess
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Platform spec dataclass
# ---------------------------------------------------------------------------


@dataclass
class PlatformSpec:
    width: int
    height: int
    max_duration: float      # seconds
    fps: int
    video_bitrate: str
    audio_bitrate: str
    max_file_mb: int
    codec: str
    profile: str
    audio_codec: str
    pixel_format: str = "yuv420p"
    loudnorm_target_lufs: float = -14.0
    loudnorm_true_peak: float = -1.0
    loudnorm_lra: float = 11.0


# ---------------------------------------------------------------------------
# Platform specs
# ---------------------------------------------------------------------------

PLATFORM_SPECS: dict[str, PlatformSpec] = {
    "tiktok": PlatformSpec(
        width=1080, height=1920, max_duration=180, fps=30,
        video_bitrate="2500k", audio_bitrate="128k", max_file_mb=287,
        codec="libx264", profile="high", audio_codec="aac",
    ),
    "instagram": PlatformSpec(
        width=1080, height=1920, max_duration=90, fps=30,
        video_bitrate="3500k", audio_bitrate="192k", max_file_mb=100,
        codec="libx264", profile="high", audio_codec="aac",
    ),
    "youtube_shorts": PlatformSpec(
        width=1080, height=1920, max_duration=60, fps=60,
        video_bitrate="8000k", audio_bitrate="192k", max_file_mb=256,
        codec="libx264", profile="high", audio_codec="aac",
    ),
    "twitter": PlatformSpec(
        width=1080, height=1920, max_duration=140, fps=30,
        video_bitrate="2000k", audio_bitrate="128k", max_file_mb=512,
        codec="libx264", profile="main", audio_codec="aac",
    ),
    "linkedin": PlatformSpec(
        width=1080, height=1920, max_duration=600, fps=30,
        video_bitrate="2000k", audio_bitrate="128k", max_file_mb=200,
        codec="libx264", profile="main", audio_codec="aac",
    ),
}


# ---------------------------------------------------------------------------
# Core export function
# ---------------------------------------------------------------------------


def export_for_platform(
    input_path: str,
    output_path: str,
    platform: str = "instagram",
    loudnorm: bool = True,
) -> str:
    """Re-encode a clip to meet *platform*'s exact technical specs.

    Steps:
      1. Probe source duration; trim to spec.max_duration if needed.
      2. Scale to spec resolution (letterbox/pillarbox to preserve AR).
      3. Two-pass loudness normalisation to EBU R128 (I=-14, TP=-1, LRA=11).
      4. Encode with correct codec / profile / bitrate.
      5. Write moov atom at the start for instant streaming (faststart).

    Parameters
    ----------
    input_path  : source video file
    output_path : destination path (created if parent dirs missing)
    platform    : one of the keys in PLATFORM_SPECS
    loudnorm    : apply EBU R128 loudness normalisation (recommended True)

    Returns
    -------
    output_path
    """
    spec = PLATFORM_SPECS.get(platform)
    if spec is None:
        raise ValueError(
            f"Unknown platform '{platform}'. "
            f"Valid options: {sorted(PLATFORM_SPECS.keys())}"
        )

    parent = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(parent, exist_ok=True)

    # Probe source duration to decide whether to trim
    src_duration = _get_duration(input_path)
    duration_args: list[str] = []
    if src_duration > spec.max_duration:
        log.info(
            "Trimming %s from %.1fs to %.1fs for %s",
            input_path, src_duration, spec.max_duration, platform,
        )
        duration_args = ["-t", str(spec.max_duration)]

    # Video filter chain:
    #   1. scale to fit within spec dimensions (keep AR)
    #   2. pad to exact spec dimensions (black bars)
    #   3. set output fps
    vf = (
        f"scale={spec.width}:{spec.height}:force_original_aspect_ratio=decrease,"
        f"pad={spec.width}:{spec.height}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"fps={spec.fps}"
    )

    # Audio filter: two-pass loudnorm or passthrough
    if loudnorm:
        af = _build_loudnorm_filter(
            input_path,
            target_lufs=spec.loudnorm_target_lufs,
            true_peak=spec.loudnorm_true_peak,
            lra=spec.loudnorm_lra,
        )
    else:
        af = "aresample=44100"

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        *duration_args,
        "-vf", vf,
        "-af", af,
        "-c:v", spec.codec,
        "-profile:v", spec.profile,
        "-b:v", spec.video_bitrate,
        "-c:a", spec.audio_codec,
        "-b:a", spec.audio_bitrate,
        "-pix_fmt", spec.pixel_format,
        "-movflags", "+faststart",
        output_path,
    ]

    log.debug("export_for_platform [%s]: %s", platform, shlex.join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg export failed for platform '{platform}':\n{result.stderr[-800:]}"
        )

    out_mb = os.path.getsize(output_path) / 1_000_000
    log.info(
        "Exported for %s → %s (%.1f MB)",
        platform, output_path, out_mb,
    )
    return output_path


# ---------------------------------------------------------------------------
# Batch export
# ---------------------------------------------------------------------------


def export_all_platforms(
    input_path: str,
    output_dir: str,
    platforms: Optional[list[str]] = None,
) -> dict[str, str]:
    """Export a clip to every platform (or a specified subset).

    Parameters
    ----------
    input_path  : source video file
    output_dir  : directory where platform-specific files are saved
    platforms   : list of platform names; defaults to all in PLATFORM_SPECS

    Returns
    -------
    dict mapping platform name → output file path (only successful exports)
    """
    if platforms is None:
        platforms = list(PLATFORM_SPECS.keys())

    unknown = [p for p in platforms if p not in PLATFORM_SPECS]
    if unknown:
        raise ValueError(f"Unknown platform(s): {unknown}")

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    results: dict[str, str] = {}
    for platform in platforms:
        out_path = os.path.join(output_dir, f"{base_name}_{platform}.mp4")
        try:
            export_for_platform(input_path, out_path, platform=platform)
            results[platform] = out_path
        except Exception as exc:
            log.error("export_all_platforms: %s failed — %s", platform, exc)

    return results


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_for_platform(video_path: str, platform: str) -> list[str]:
    """Check whether *video_path* meets *platform*'s technical requirements.

    Parameters
    ----------
    video_path : path to the video file to check
    platform   : platform key (e.g. "instagram")

    Returns
    -------
    List of human-readable issue strings.  An empty list means the file is
    fully compliant.
    """
    spec = PLATFORM_SPECS.get(platform)
    if spec is None:
        return [f"Unknown platform '{platform}'. Valid: {sorted(PLATFORM_SPECS.keys())}"]

    issues: list[str] = []

    # --- Probe with ffprobe ---
    try:
        probe_result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_streams", "-show_format",
                video_path,
            ],
            capture_output=True, text=True, check=True,
        )
        probe_data = json.loads(probe_result.stdout)
    except FileNotFoundError:
        return ["ffprobe not found — install FFmpeg to enable validation."]
    except subprocess.CalledProcessError as exc:
        return [f"ffprobe failed: {exc.stderr[-400:]}"]
    except json.JSONDecodeError:
        return ["ffprobe returned unparseable output."]

    fmt = probe_data.get("format", {})
    streams = probe_data.get("streams", [])

    # --- Duration ---
    try:
        duration = float(fmt.get("duration", 0))
    except (TypeError, ValueError):
        duration = 0.0
    if duration > spec.max_duration:
        issues.append(
            f"Duration {duration:.1f}s exceeds platform maximum {spec.max_duration:.0f}s."
        )

    # --- Video stream ---
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    if not video_streams:
        issues.append("No video stream found.")
    else:
        vs = video_streams[0]
        w = vs.get("width", 0)
        h = vs.get("height", 0)
        if w != spec.width or h != spec.height:
            issues.append(
                f"Resolution {w}x{h} does not match required {spec.width}x{spec.height}."
            )

        codec_name = vs.get("codec_name", "")
        if codec_name not in ("h264", "hevc", "av1"):
            issues.append(
                f"Video codec '{codec_name}' may not be supported "
                f"(recommended: h264 / libx264)."
            )

        # Check aspect ratio (9:16 portrait)
        if w > 0 and h > 0:
            ar = w / h
            target_ar = spec.width / spec.height  # e.g. 1080/1920 = 0.5625
            if abs(ar - target_ar) > 0.05:
                issues.append(
                    f"Aspect ratio {ar:.3f} ({w}:{h}) is not 9:16 portrait "
                    f"(expected ~{target_ar:.3f})."
                )

        # Frame rate check
        r_frame_rate = vs.get("r_frame_rate", "0/1")
        try:
            num, den = map(int, r_frame_rate.split("/"))
            actual_fps = num / den if den else 0
            if actual_fps > spec.fps + 1:
                issues.append(
                    f"Frame rate {actual_fps:.1f} fps exceeds platform maximum {spec.fps} fps."
                )
        except (ValueError, ZeroDivisionError):
            pass

    # --- Audio stream ---
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    if not audio_streams:
        issues.append("No audio stream found.")
    else:
        a_codec = audio_streams[0].get("codec_name", "")
        if a_codec not in ("aac", "mp3", "opus"):
            issues.append(
                f"Audio codec '{a_codec}' may not be supported "
                f"(recommended: aac)."
            )

    # --- File size ---
    try:
        file_mb = os.path.getsize(video_path) / 1_000_000
        if file_mb > spec.max_file_mb:
            issues.append(
                f"File size {file_mb:.1f} MB exceeds platform maximum {spec.max_file_mb} MB."
            )
    except OSError:
        issues.append("Could not determine file size.")

    return issues


# ---------------------------------------------------------------------------
# Loudnorm helpers (two-pass EBU R128)
# ---------------------------------------------------------------------------


def _build_loudnorm_filter(
    input_path: str,
    target_lufs: float = -14.0,
    true_peak: float = -1.0,
    lra: float = 11.0,
) -> str:
    """Build a two-pass EBU R128 loudnorm audio filter string.

    Pass 1: measure the actual loudness of the source.
    Pass 2: the returned filter string feeds measured values back so FFmpeg
            can apply a linear (not dynamic) correction — this avoids the
            pumping artefacts of single-pass loudnorm.
    """
    # Pass 1 — measure only (output to /dev/null)
    measure_filter = (
        f"loudnorm=I={target_lufs}:TP={true_peak}:LRA={lra}:print_format=json"
    )
    cmd_measure = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-af", measure_filter,
        "-f", "null", "-",
    ]
    log.debug("loudnorm pass-1: %s", shlex.join(cmd_measure))
    result = subprocess.run(cmd_measure, capture_output=True, text=True)

    # loudnorm writes its JSON block to stderr
    measured = _parse_loudnorm_json(result.stderr)
    if measured:
        log.debug("loudnorm measured: %s", measured)
        return (
            f"loudnorm=I={target_lufs}:TP={true_peak}:LRA={lra}"
            f":measured_I={measured['input_i']}"
            f":measured_TP={measured['input_tp']}"
            f":measured_LRA={measured['input_lra']}"
            f":measured_thresh={measured['input_thresh']}"
            f":offset={measured['target_offset']}"
            f":linear=true:print_format=none"
        )

    # Fallback to single-pass if measurement failed (e.g. silent track)
    log.warning(
        "loudnorm pass-1 measurement failed — falling back to single-pass.\n%s",
        result.stderr[-400:],
    )
    return f"loudnorm=I={target_lufs}:TP={true_peak}:LRA={lra}"


def _parse_loudnorm_json(stderr: str) -> Optional[dict]:
    """Extract the JSON statistics block that loudnorm prints to stderr."""
    try:
        start = stderr.rfind("{")
        end = stderr.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(stderr[start:end])
    except (json.JSONDecodeError, ValueError):
        pass
    return None


# ---------------------------------------------------------------------------
# FFprobe utility
# ---------------------------------------------------------------------------


def _get_duration(video_path: str) -> float:
    """Return video duration in seconds using ffprobe, or 0.0 on failure."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                video_path,
            ],
            capture_output=True, text=True, check=True,
        )
        return float(json.loads(result.stdout)["format"]["duration"])
    except Exception as exc:
        log.warning("_get_duration failed for %s: %s", video_path, exc)
        return 0.0
