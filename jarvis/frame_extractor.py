"""Extract representative frames from video clips for Claude vision analysis."""

import base64
import os
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ClipAnalysisData:
    path: str
    duration_seconds: float
    fps: float
    width: int
    height: int
    frame_count: int
    # Base64-encoded JPEG frames sampled across the clip
    sample_frames: list[str]
    # Audio presence heuristic (non-silent ratio)
    has_audio: bool
    # Estimated scene count via histogram diff
    estimated_scenes: int


def extract_clip_data(video_path: str, max_frames: int = 6) -> ClipAnalysisData:
    """
    Open a video file, sample frames across its duration, and return metadata
    plus base64-encoded frames for Claude vision.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps

    # Sample frames evenly across the clip, skipping first/last 5%
    start = int(total_frames * 0.05)
    end = int(total_frames * 0.95)
    sample_indices = np.linspace(start, end, min(max_frames, total_frames), dtype=int)

    sample_frames: list[str] = []
    prev_hist = None
    scene_changes = 0

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue

        # Encode frame as small JPEG for efficient API transmission
        frame_resized = _resize_for_api(frame)
        _, buf = cv2.imencode(".jpg", frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
        sample_frames.append(base64.b64encode(buf).decode())

        # Rough scene-change detection via histogram comparison
        hist = _compute_histogram(frame)
        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if diff > 0.4:
                scene_changes += 1
        prev_hist = hist

    cap.release()

    return ClipAnalysisData(
        path=video_path,
        duration_seconds=round(duration, 2),
        fps=round(fps, 2),
        width=width,
        height=height,
        frame_count=total_frames,
        sample_frames=sample_frames,
        has_audio=_probe_audio(video_path),
        estimated_scenes=max(1, scene_changes),
    )


def extract_multi_clip_data(paths: list[str], frames_per_clip: int = 4) -> list[ClipAnalysisData]:
    """Extract data for a list of video files."""
    results = []
    for path in paths:
        try:
            results.append(extract_clip_data(path, max_frames=frames_per_clip))
        except Exception as e:
            print(f"[FrameExtractor] Skipping {path}: {e}")
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resize_for_api(frame: np.ndarray, max_dim: int = 512) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(max_dim / w, max_dim / h, 1.0)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return frame


def _compute_histogram(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
    cv2.normalize(hist, hist)
    return hist


def _probe_audio(video_path: str) -> bool:
    """Return True if the file likely has an audio stream (heuristic: most mp4/mov do)."""
    ext = os.path.splitext(video_path)[1].lower()
    return ext in {".mp4", ".mov", ".mkv", ".avi", ".webm"}
