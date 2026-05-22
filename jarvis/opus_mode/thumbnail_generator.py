"""
Viral thumbnail generator.

Extracts candidate frames from a video clip, scores each one for thumbnail
quality (face presence, sharpness, brightness, motion blur), and saves the
winner as a JPEG.  Optionally burns title text onto the thumbnail.

Requires:
  - FFmpeg (always)
  - OpenCV (optional, enables face detection and more accurate scoring)

Usage:
    from jarvis.opus_mode.thumbnail_generator import generate_thumbnail

    thumb = generate_thumbnail(
        video_path="clip.mp4",
        output_path="thumb.jpg",
        clip_start=0.0,
        clip_end=30.0,
        add_text="This changed everything",
    )
    print(thumb.path, thumb.score, thumb.has_face)
"""

import logging
import math
import os
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional OpenCV import
# ---------------------------------------------------------------------------

try:
    import cv2
    import numpy as np
    _CV2_AVAILABLE = True
    log.debug("OpenCV available — using native frame extraction and scoring.")
except ImportError:
    cv2 = None       # type: ignore[assignment]
    np = None        # type: ignore[assignment]
    _CV2_AVAILABLE = False
    log.debug("OpenCV not available — falling back to FFmpeg JPEG extraction.")

# Haar cascade path (bundled with OpenCV)
_HAAR_CASCADE_PATH: Optional[str] = None
if _CV2_AVAILABLE:
    _data_dir = getattr(cv2, "data", None)
    if _data_dir and hasattr(_data_dir, "haarcascades"):
        _candidate = os.path.join(
            _data_dir.haarcascades, "haarcascade_frontalface_default.xml"
        )
        if os.path.exists(_candidate):
            _HAAR_CASCADE_PATH = _candidate

_face_cascade = None  # lazy-loaded


def _get_face_cascade():
    global _face_cascade
    if _face_cascade is None and _HAAR_CASCADE_PATH and _CV2_AVAILABLE:
        _face_cascade = cv2.CascadeClassifier(_HAAR_CASCADE_PATH)
    return _face_cascade


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class Thumbnail:
    path: str          # path to saved thumbnail JPEG
    timestamp: float   # seconds into the clip where this frame lives
    score: float       # quality score 0-100
    has_face: bool


# ---------------------------------------------------------------------------
# Frame scoring
# ---------------------------------------------------------------------------


def _score_frame(frame_bgr, prev_frame=None) -> float:
    """Score a BGR numpy frame for thumbnail quality (0-100).

    Criteria
    --------
    face_bonus   : +35 if at least one face detected by Haar cascade
    sharpness    : 0-40 based on Laplacian variance (log-scaled)
    brightness   : 0-20 based on how close mean luminance is to 128/255
    motion_blur  : -15 penalty if this frame and the previous are too similar
                   (low inter-frame difference suggests camera motion / pan blur)
    """
    if not _CV2_AVAILABLE or frame_bgr is None:
        return 50.0  # neutral score when cv2 unavailable

    score = 0.0

    # --- Face detection (+35) ---
    has_face = False
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    cascade = _get_face_cascade()
    if cascade is not None:
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
        )
        if len(faces) > 0:
            has_face = True
            score += 35.0

    # --- Sharpness via Laplacian variance (0-40) ---
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # Log curve: 0→0, 500→40
    sharpness_score = min(40.0, 40.0 * math.log1p(laplacian_var) / math.log1p(500))
    score += sharpness_score

    # --- Brightness (0-20) — mid-range luminance preferred ---
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mean_v = float(hsv[:, :, 2].mean())  # 0-255
    # Peak at 128; falls off linearly toward 0 and 255
    brightness_score = 20.0 * (1.0 - abs(mean_v - 128.0) / 128.0)
    score += brightness_score

    # --- Motion-blur / static penalty (-15) ---
    # Very low difference between adjacent frames → camera motion or static shot
    if prev_frame is not None:
        diff = float(cv2.absdiff(frame_bgr, prev_frame).mean())
        if diff < 5.0:
            score -= 15.0

    return max(0.0, min(100.0, score))


def _has_face_simple(frame_bgr) -> bool:
    """Return True if a face is detected in the frame (requires OpenCV)."""
    if not _CV2_AVAILABLE:
        return False
    cascade = _get_face_cascade()
    if cascade is None:
        return False
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
    )
    return len(faces) > 0


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------


def _extract_frames_cv2(
    video_path: str,
    start: float,
    end: Optional[float],
    sample_fps: float = 2.0,
) -> "list[tuple[float, object]]":
    """Extract frames using OpenCV VideoCapture.

    Returns a list of (timestamp_seconds, bgr_ndarray) tuples.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_duration = total_frames / native_fps if native_fps > 0 else 0.0

    clip_end = end if (end is not None and 0 < end <= clip_duration) else clip_duration

    # Step size in native frames between each sampled frame
    frame_step = max(1, int(native_fps / sample_fps))

    start_frame = max(0, int(start * native_fps))
    end_frame = min(total_frames - 1, int(clip_end * native_fps))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    results: list[tuple[float, object]] = []
    frame_idx = start_frame

    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = frame_idx / native_fps
        results.append((timestamp, frame))
        frame_idx += frame_step
        if frame_idx <= end_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    cap.release()
    return results


def _extract_frames_ffmpeg(
    video_path: str,
    start: float,
    end: Optional[float],
    sample_fps: float = 2.0,
    tmp_dir: Optional[str] = None,
) -> "list[tuple[float, str]]":
    """Extract frames via FFmpeg, writing JPEG files into *tmp_dir*.

    Returns a list of (timestamp_seconds, jpeg_path) tuples.
    """
    work_dir = tmp_dir or tempfile.gettempdir()
    output_pattern = os.path.join(work_dir, "frame_%04d.jpg")

    cmd: list[str] = ["ffmpeg", "-y"]
    if start > 0:
        cmd += ["-ss", str(start)]
    cmd += ["-i", video_path]
    if end is not None and end > start:
        cmd += ["-t", str(end - start)]
    cmd += [
        "-vf", f"fps={sample_fps}",
        "-q:v", "2",
        output_pattern,
    ]

    log.debug("_extract_frames_ffmpeg: %s", shlex.join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg frame extraction failed:\n{result.stderr[-800:]}")

    frame_files = sorted(
        f for f in os.listdir(work_dir)
        if f.startswith("frame_") and f.endswith(".jpg")
    )

    frames: list[tuple[float, str]] = []
    for i, fname in enumerate(frame_files):
        ts = start + i / sample_fps
        frames.append((ts, os.path.join(work_dir, fname)))
    return frames


# ---------------------------------------------------------------------------
# Best-frame pickers
# ---------------------------------------------------------------------------


def _pick_best_cv2(
    frames: "list[tuple[float, object]]",
) -> "tuple[float, object, float, bool]":
    """Score all frames and return (best_ts, best_frame, best_score, has_face)."""
    best_ts = 0.0
    best_frame = None
    best_score = -1.0
    best_has_face = False

    prev_frame = None
    for ts, frame in frames:
        score = _score_frame(frame, prev_frame)
        if score > best_score:
            best_score = score
            best_ts = ts
            best_frame = frame
            best_has_face = _has_face_simple(frame)
        prev_frame = frame

    return best_ts, best_frame, max(0.0, best_score), best_has_face


def _pick_best_ffmpeg(
    frames: "list[tuple[float, str]]",
) -> "tuple[float, str, float, bool]":
    """Score JPEG files by file size as a proxy for sharpness/detail.

    When cv2 is unavailable, JPEG file size correlates reasonably well with
    image detail (larger = more DCT coefficients = sharper frame).
    """
    best_ts = frames[0][0]
    best_path = frames[0][1]
    best_score = -1.0

    for ts, path in frames:
        try:
            size_kb = os.path.getsize(path) / 1024.0
        except OSError:
            continue
        # Normalise file size to 0-100 (typical range 20-200 KB for HD frames)
        score = min(100.0, size_kb / 2.0)
        if score > best_score:
            best_score = score
            best_ts = ts
            best_path = path

    return best_ts, best_path, max(0.0, best_score), False


# ---------------------------------------------------------------------------
# Frame save helper
# ---------------------------------------------------------------------------


def _save_frame_cv2(frame, output_path: str) -> None:
    """Write a cv2 BGR frame to *output_path* as JPEG (quality 95)."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    success, buf = cv2.imencode(".jpg", frame, encode_params)
    if not success:
        raise RuntimeError("cv2.imencode failed when saving thumbnail.")
    with open(output_path, "wb") as fh:
        fh.write(buf.tobytes())


# ---------------------------------------------------------------------------
# Text burn-in
# ---------------------------------------------------------------------------


def _burn_text_on_thumbnail(
    image_path: str,
    text: str,
    output_path: str,
    font_size: int = 64,
) -> str:
    """Overlay *text* onto a JPEG image using FFmpeg drawtext.

    Text is centred horizontally, placed in the bottom quarter of the image,
    rendered in bold white with a thick black border and drop shadow.
    Returns *output_path* (may equal *image_path*).
    """
    safe_text = (
        text.replace("\\", "\\\\")
            .replace("'",  "\\'")
            .replace(":",  "\\:")
    )

    drawtext = (
        f"text='{safe_text}'"
        f":fontsize={font_size}"
        ":fontcolor=white"
        ":bordercolor=black"
        ":borderw=4"
        ":shadowcolor=black@0.7"
        ":shadowx=2:shadowy=2"
        ":x=(w-text_w)/2"
        ":y=h*0.78"
    )

    tmp_out = output_path + ".tmp.jpg"
    cmd = [
        "ffmpeg", "-y",
        "-i", image_path,
        "-vf", f"drawtext={drawtext}",
        "-q:v", "2",
        tmp_out,
    ]
    log.debug("_burn_text_on_thumbnail: %s", shlex.join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.warning(
            "Thumbnail text burn failed — keeping plain frame.\n%s",
            result.stderr[-600:],
        )
        return output_path  # return un-annotated path rather than crashing

    os.replace(tmp_out, output_path)
    return output_path


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------


def generate_thumbnail(
    video_path: str,
    output_path: str,
    clip_start: float = 0.0,
    clip_end: Optional[float] = None,
    add_text: Optional[str] = None,
) -> Thumbnail:
    """Extract the best frame from *video_path* and save it as a JPEG thumbnail.

    Parameters
    ----------
    video_path  : source video file
    output_path : where to save the final JPEG
    clip_start  : start of the region to sample (seconds, default 0)
    clip_end    : end of the region to sample (seconds); None = full clip
    add_text    : optional title text to burn onto the thumbnail via FFmpeg

    Returns
    -------
    Thumbnail dataclass with path, timestamp, score, has_face
    """
    sample_fps = 2.0  # 2 frames per second gives good coverage without overhead

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if _CV2_AVAILABLE:
        raw_frames = _extract_frames_cv2(video_path, clip_start, clip_end, sample_fps)
        if not raw_frames:
            raise RuntimeError("No frames extracted from video via OpenCV.")
        best_ts, best_frame, best_score, best_has_face = _pick_best_cv2(raw_frames)
        _save_frame_cv2(best_frame, output_path)
    else:
        with tempfile.TemporaryDirectory(prefix="thumb_frames_") as tmp_dir:
            raw_frames_ffmpeg = _extract_frames_ffmpeg(
                video_path, clip_start, clip_end, sample_fps, tmp_dir=tmp_dir
            )
            if not raw_frames_ffmpeg:
                raise RuntimeError("No frames extracted from video via FFmpeg.")
            best_ts, best_jpeg, best_score, best_has_face = _pick_best_ffmpeg(
                raw_frames_ffmpeg
            )
            shutil.copy2(best_jpeg, output_path)

    log.info(
        "Best thumbnail frame: t=%.2fs  score=%.1f  face=%s  → %s",
        best_ts, best_score, best_has_face, output_path,
    )

    if add_text:
        output_path = _burn_text_on_thumbnail(output_path, add_text, output_path)

    return Thumbnail(
        path=output_path,
        timestamp=best_ts,
        score=best_score,
        has_face=best_has_face,
    )
