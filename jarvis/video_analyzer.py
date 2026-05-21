"""
Deep video understanding using Gemini vision + Whisper transcription.

For each clip this module produces:
  - Full content description (what's actually happening)
  - Emotional moment timestamps
  - Best/worst moments
  - Motion energy map
  - Spoken dialogue (Whisper)
  - Visual quality score
  - Recommended in/out points
"""

import base64
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field

import cv2
import numpy as np

from .openrouter import router
from .config import cfg


@dataclass
class MomentScore:
    timestamp: float
    score: float          # 0.0–1.0 (1.0 = most compelling)
    reason: str


@dataclass
class ClipUnderstanding:
    path: str
    duration_seconds: float
    fps: float
    resolution: tuple[int, int]
    # What's actually happening in this clip
    content_description: str
    # Overall mood/feel
    mood: str
    # Scene type
    scene_type: str        # action, dialogue, landscape, closeup, broll, etc.
    # Energy level 0.0–1.0
    energy_level: float
    # Best moment to use (recommended trim points)
    best_start: float
    best_end: float
    # Top emotional/visual moments
    key_moments: list[MomentScore]
    # Spoken words (from Whisper)
    transcript: str
    # Motion scores per second
    motion_scores: list[float]
    # Visual quality 0.0–1.0
    quality_score: float
    # Suggested effects
    suggested_filter: str
    suggested_effect: str


VIDEO_ANALYSIS_SYSTEM = """You are an expert video editor and cinematographer analyzing footage.
Study the provided frames carefully and understand what is TRULY happening — not just objects,
but the story, emotion, energy, and cinematic quality of each moment.
Return only valid JSON, no markdown, no explanation."""

VIDEO_ANALYSIS_PROMPT = """Analyze these {n_frames} frames sampled from a {duration:.1f}-second video clip.
Transcript of spoken audio: "{transcript}"

Determine:
1. What is actually happening in this clip?
2. What is the dominant mood/emotion?
3. What type of scene is it? (action/dialogue/landscape/closeup/broll/transition)
4. Energy level from 0.0 to 1.0
5. Best in/out points to keep (trim bad parts)
6. Top 3 most compelling moments with timestamps and why
7. Visual quality score 0.0–1.0
8. Best filter to apply (cinematic/warm/cool/moody/vintage/bright/black_white)
9. Best visual effect (none/ken_burns/zoom_in/light_leak/film_grain)

Return JSON:
{{
  "content_description": "detailed description of what's happening",
  "mood": "energetic|happy|calm|melancholic|tense|romantic|neutral",
  "scene_type": "action|dialogue|landscape|closeup|broll|transition",
  "energy_level": 0.0-1.0,
  "best_start": seconds_from_start,
  "best_end": seconds_from_start,
  "key_moments": [
    {{"timestamp": seconds, "score": 0.0-1.0, "reason": "why this moment matters"}}
  ],
  "quality_score": 0.0-1.0,
  "suggested_filter": "filter_name",
  "suggested_effect": "effect_name"
}}"""


class VideoAnalyzer:
    """Watches and understands video clips using Gemini vision + Whisper."""

    def __init__(self):
        self._whisper_model = None

    def analyze(self, video_path: str, frames_to_sample: int = 8) -> ClipUnderstanding:
        """Full analysis of a single video clip."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps
        cap.release()

        # 1. Sample frames across the clip
        frames_b64, timestamps = self._sample_frames(video_path, frames_to_sample)

        # 2. Compute motion scores (optical flow)
        motion_scores = self._compute_motion_scores(video_path)

        # 3. Transcribe audio with Whisper
        transcript = self._transcribe(video_path)

        # 4. Send to Gemini for deep understanding
        prompt = VIDEO_ANALYSIS_PROMPT.format(
            n_frames=len(frames_b64),
            duration=duration,
            transcript=transcript or "(no speech detected)",
        )
        raw = router.vision(
            prompt=prompt,
            images_b64=frames_b64,
            model=cfg.models.vision,
            max_tokens=1500,
            system=VIDEO_ANALYSIS_SYSTEM,
        )

        data = self._parse_json(raw)

        # 5. Extract key_moments with timestamps mapped to actual video time
        key_moments = []
        for i, m in enumerate(data.get("key_moments", [])):
            ts = float(m.get("timestamp", timestamps[min(i, len(timestamps)-1)]))
            key_moments.append(MomentScore(
                timestamp=round(ts, 2),
                score=float(m.get("score", 0.5)),
                reason=m.get("reason", ""),
            ))

        return ClipUnderstanding(
            path=video_path,
            duration_seconds=round(duration, 2),
            fps=round(fps, 2),
            resolution=(width, height),
            content_description=data.get("content_description", ""),
            mood=data.get("mood", "neutral"),
            scene_type=data.get("scene_type", "broll"),
            energy_level=float(data.get("energy_level", 0.5)),
            best_start=float(data.get("best_start", 0.0)),
            best_end=float(data.get("best_end", duration)),
            key_moments=key_moments,
            transcript=transcript,
            motion_scores=motion_scores,
            quality_score=float(data.get("quality_score", 0.7)),
            suggested_filter=data.get("suggested_filter", "cinematic"),
            suggested_effect=data.get("suggested_effect", "none"),
        )

    def analyze_all(self, video_paths: list[str]) -> list[ClipUnderstanding]:
        """Analyze multiple clips."""
        results = []
        for path in video_paths:
            try:
                results.append(self.analyze(path))
            except Exception as e:
                print(f"[VideoAnalyzer] Skipping {path}: {e}")
        return results

    def find_hook_moment(self, clips: list[ClipUnderstanding]) -> tuple[str, float]:
        """
        Find the single most attention-grabbing moment across all clips.
        Returns (clip_path, timestamp).
        """
        best_clip = None
        best_ts = 0.0
        best_score = 0.0

        for clip in clips:
            for moment in clip.key_moments:
                combined = moment.score * 0.7 + clip.energy_level * 0.3
                if combined > best_score:
                    best_score = combined
                    best_clip = clip.path
                    best_ts = moment.timestamp

        return best_clip or clips[0].path, best_ts

    # ------------------------------------------------------------------
    # Frame sampling
    # ------------------------------------------------------------------

    def _sample_frames(self, video_path: str, n: int) -> tuple[list[str], list[float]]:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Skip first and last 5% to avoid black frames
        start = int(total * 0.05)
        end = int(total * 0.95)
        indices = np.linspace(start, end, min(n, total), dtype=int)

        frames_b64 = []
        timestamps = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            # Resize to max 720px for efficiency
            h, w = frame.shape[:2]
            scale = min(720 / w, 720 / h, 1.0)
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frames_b64.append(base64.b64encode(buf).decode())
            timestamps.append(round(idx / fps, 2))

        cap.release()
        return frames_b64, timestamps

    # ------------------------------------------------------------------
    # Motion scoring via optical flow
    # ------------------------------------------------------------------

    def _compute_motion_scores(self, video_path: str, sample_fps: float = 2.0) -> list[float]:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        skip = max(1, int(fps / sample_fps))

        scores = []
        prev_gray = None
        frame_n = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_n % skip == 0:
                gray = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                    scores.append(float(np.mean(magnitude)))
                prev_gray = gray
            frame_n += 1

        cap.release()

        if not scores:
            return []

        max_score = max(scores) + 1e-8
        return [round(s / max_score, 3) for s in scores]

    # ------------------------------------------------------------------
    # Whisper transcription
    # ------------------------------------------------------------------

    def _transcribe(self, video_path: str) -> str:
        try:
            import whisper
        except ImportError:
            return ""

        try:
            if self._whisper_model is None:
                self._whisper_model = whisper.load_model(cfg.whisper_model)

            # Extract audio to temp file
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
                 "-ar", "16000", "-ac", "1", tmp.name],
                capture_output=True,
            )
            if result.returncode != 0:
                return ""

            out = self._whisper_model.transcribe(tmp.name, fp16=False)
            os.unlink(tmp.name)
            return out.get("text", "").strip()
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # JSON parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(raw: str) -> dict:
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            return {}
