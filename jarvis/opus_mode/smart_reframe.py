"""
Smart reframe: automatically crops landscape video to 9:16 (vertical).

Strategy:
  1. Sample frames throughout the clip
  2. Detect faces / dominant subjects in each frame
  3. Build a smooth crop trajectory (no jitter)
  4. Output FFmpeg zoompan filter string for the clip

Falls back to center crop when no face is detected.
"""

import os
import subprocess
import tempfile
from dataclasses import dataclass

try:
    import cv2
    import numpy as np
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


@dataclass
class CropTrajectory:
    """Describes how to crop a video to 9:16."""
    x_positions: list[float]   # normalized 0-1 center x per keyframe
    y_positions: list[float]   # normalized 0-1 center y per keyframe
    sample_interval: float     # seconds between keyframes
    source_width: int
    source_height: int
    target_width: int          # output width
    target_height: int         # output height

    @property
    def crop_width(self) -> int:
        """Width of crop window in source pixels."""
        # For 9:16 from landscape: use full height, crop width accordingly
        return int(self.source_height * 9 / 16)

    @property
    def crop_height(self) -> int:
        return self.source_height


def analyze_reframe(
    video_path: str,
    clip_start: float,
    clip_end: float,
    sample_every_n_seconds: float = 0.5,
    target_width: int = 1080,
    target_height: int = 1920,
) -> CropTrajectory:
    """
    Analyze a clip and return a smooth crop trajectory for 9:16 reframe.
    """
    if not _CV2_AVAILABLE:
        return _center_trajectory(video_path, clip_start, clip_end, sample_every_n_seconds, target_width, target_height)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Load face detector
    face_cascade = _get_face_cascade()

    x_positions = []
    y_positions = []

    t = clip_start
    while t <= clip_end:
        frame_idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        cx, cy = _detect_subject_center(frame, face_cascade, src_w, src_h)
        x_positions.append(cx / src_w)
        y_positions.append(cy / src_h)
        t += sample_every_n_seconds

    cap.release()

    if not x_positions:
        x_positions = [0.5]
        y_positions = [0.5]

    # Smooth trajectory to eliminate jitter
    x_positions = _smooth(x_positions, window=5)
    y_positions = _smooth(y_positions, window=5)

    return CropTrajectory(
        x_positions=x_positions,
        y_positions=y_positions,
        sample_interval=sample_every_n_seconds,
        source_width=src_w,
        source_height=src_h,
        target_width=target_width,
        target_height=target_height,
    )


def build_ffmpeg_crop_filter(
    trajectory: CropTrajectory,
    clip_start: float = 0.0,
) -> str:
    """
    Build an FFmpeg filtergraph string that applies the crop trajectory.

    For simple cases (single position), returns a static crop.
    For animated trajectories, uses crop with expressions.
    """
    src_w = trajectory.source_width
    src_h = trajectory.source_height
    cw = trajectory.crop_width
    ch = trajectory.crop_height

    if len(trajectory.x_positions) <= 1:
        # Static crop
        cx = trajectory.x_positions[0] if trajectory.x_positions else 0.5
        x = int(max(0, min(cx * src_w - cw // 2, src_w - cw)))
        y = 0
        scale = f"scale={trajectory.target_width}:{trajectory.target_height}"
        return f"crop={cw}:{ch}:{x}:{y},{scale}"

    # Build piecewise linear crop using FFmpeg 'crop' with if/between expressions
    # For each time interval, interpolate x position
    # This uses FFmpeg's 'crop' filter with dynamic x computed via 'if' chains
    interval = trajectory.sample_interval

    # Build x expression: interpolate between keyframes
    # x = lerp(x0, x1, (t - t0) / interval) for each segment
    x_expr = _build_lerp_expr(
        trajectory.x_positions, interval, src_w, cw, clip_start, clamp_lo=0, clamp_hi=src_w - cw
    )

    scale = f"scale={trajectory.target_width}:{trajectory.target_height}"
    return f"crop={cw}:{ch}:{x_expr}:0,{scale}"


def _build_lerp_expr(positions: list[float], interval: float, src_dim: int, crop_dim: int,
                     start_offset: float, clamp_lo: int, clamp_hi: int) -> str:
    """Build FFmpeg expression for piecewise linear interpolation of crop position."""
    if len(positions) == 1:
        x = int(positions[0] * src_dim - crop_dim // 2)
        return str(max(clamp_lo, min(x, clamp_hi)))

    # Convert positions to pixel centers
    px = [int(p * src_dim - crop_dim // 2) for p in positions]
    px = [max(clamp_lo, min(v, clamp_hi)) for v in px]

    # Build nested if expression: if(between(t,t0,t1), lerp, if(...))
    # t in FFmpeg crop filter is `t` (current time in seconds from start of clip)
    parts = []
    for i in range(len(px) - 1):
        t0 = start_offset + i * interval
        t1 = start_offset + (i + 1) * interval
        x0, x1 = px[i], px[i + 1]
        # lerp: x0 + (x1-x0) * (t - t0) / interval
        if x0 == x1:
            lerp = str(x0)
        else:
            lerp = f"{x0}+({x1-x0})*((t-{t0:.3f})/{interval:.3f})"
        parts.append((t0, t1, lerp))

    # Build if chain
    expr = str(px[-1])  # default: last position
    for t0, t1, lerp in reversed(parts):
        expr = f"if(between(t\\,{t0:.3f}\\,{t1:.3f})\\,{lerp}\\,{expr})"

    return expr


def _detect_subject_center(frame, face_cascade, src_w: int, src_h: int) -> tuple[int, int]:
    """Return pixel (x, y) of the detected subject center. Falls back to frame center."""
    if face_cascade is None:
        return src_w // 2, src_h // 2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) == 0:
        return src_w // 2, src_h // 2

    # Use the largest face
    areas = [(w * h, x, y, w, h) for (x, y, w, h) in faces]
    areas.sort(reverse=True)
    _, x, y, w, h = areas[0]

    cx = x + w // 2
    cy = y + h // 2
    return cx, cy


def _get_face_cascade():
    """Load OpenCV face cascade. Returns None if unavailable."""
    if not _CV2_AVAILABLE:
        return None
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if os.path.exists(cascade_path):
        return cv2.CascadeClassifier(cascade_path)
    return None


def _smooth(values: list[float], window: int = 5) -> list[float]:
    """Apply simple moving average smoothing."""
    if len(values) <= window:
        avg = sum(values) / len(values)
        return [avg] * len(values)

    result = []
    half = window // 2
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        result.append(sum(values[lo:hi]) / (hi - lo))
    return result


def _center_trajectory(video_path, clip_start, clip_end, sample_interval,
                       target_width, target_height) -> CropTrajectory:
    """Fallback: center crop trajectory without OpenCV."""
    # Try to get video dimensions via ffprobe
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", video_path],
            capture_output=True, text=True, check=True,
        )
        import json
        data = json.loads(result.stdout)
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                src_w = int(stream["width"])
                src_h = int(stream["height"])
                break
        else:
            src_w, src_h = 1920, 1080
    except Exception:
        src_w, src_h = 1920, 1080

    n = max(1, int((clip_end - clip_start) / sample_interval))
    return CropTrajectory(
        x_positions=[0.5] * n,
        y_positions=[0.5] * n,
        sample_interval=sample_interval,
        source_width=src_w,
        source_height=src_h,
        target_width=target_width,
        target_height=target_height,
    )


def reframe_clip_ffmpeg(
    video_path: str,
    output_path: str,
    clip_start: float,
    clip_end: float,
    target_width: int = 1080,
    target_height: int = 1920,
) -> str:
    """
    Crop and reframe a clip to 9:16 using face-tracking.
    Returns output_path.
    """
    trajectory = analyze_reframe(video_path, clip_start, clip_end, target_width=target_width, target_height=target_height)
    crop_filter = build_ffmpeg_crop_filter(trajectory, clip_start=0.0)

    duration = clip_end - clip_start
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    subprocess.run([
        "ffmpeg", "-y",
        "-ss", str(clip_start),
        "-t", str(duration),
        "-i", video_path,
        "-vf", crop_filter,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "192k",
        output_path,
    ], capture_output=True, check=True)

    return output_path
