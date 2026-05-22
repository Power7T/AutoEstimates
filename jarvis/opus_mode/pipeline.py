"""
OpusClip-level pipeline — turns a long video into viral short clips.

Full flow:
  1. Transcribe (Whisper, word-level timestamps)
  2. Find viral moments (Claude)
  3. Score each moment (6-dimension virality score)
  4. Smart reframe to 9:16 (face tracking)
  5. Burn word-by-word animated captions (ASS format)
  6. Export final clips ranked by score

Output: folder of MP4 clips, each named by rank and title.
"""

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .transcriber import transcribe, Transcript
from .moment_finder import find_viral_moments, ViralMoment
from .virality_scorer import score_all, ViralityReport
from .caption_animator import generate_caption_file, ffmpeg_caption_filter
from .smart_reframe import analyze_reframe, build_ffmpeg_crop_filter
from .speaker_detector import detect_speakers


@dataclass
class ProcessedClip:
    rank: int
    moment: ViralMoment
    report: ViralityReport
    output_path: str
    duration: float


@dataclass
class PipelineResult:
    input_video: str
    clips: list[ProcessedClip] = field(default_factory=list)
    transcript: Transcript | None = None
    total_clips_found: int = 0
    output_dir: str = ""


def run_pipeline(
    video_path: str,
    output_dir: str = "output/clips",
    n_clips: int = 10,
    top_n_export: int = 5,
    style: str = "bold",
    target_width: int = 1080,
    target_height: int = 1920,
    whisper_model: str = "base",
    enable_reframe: bool = True,
    enable_captions: bool = True,
    min_score: float = 0.0,
    on_progress: callable = None,
) -> PipelineResult:
    """
    End-to-end pipeline: video → viral clips.

    Args:
        video_path: Path to source video
        output_dir: Where to save clips
        n_clips: How many moments to find (more = better selection)
        top_n_export: How many to actually render (top N by score)
        style: Caption style — "bold" | "clean" | "kinetic"
        target_width/height: Output resolution (default 1080x1920 = 9:16)
        whisper_model: "tiny" | "base" | "small" | "medium" | "large"
        enable_reframe: Whether to apply 9:16 smart reframe
        enable_captions: Whether to burn in word-by-word captions
        min_score: Skip clips below this virality score
        on_progress: Optional callback(step: str, pct: float)
    """
    result = PipelineResult(input_video=video_path, output_dir=output_dir)
    os.makedirs(output_dir, exist_ok=True)

    def _progress(step: str, pct: float):
        print(f"[Pipeline] {step} ({pct:.0f}%)")
        if on_progress:
            on_progress(step, pct)

    # Step 1: Transcribe
    _progress("Transcribing video...", 5)
    transcript = transcribe(video_path, model_size=whisper_model)
    result.transcript = transcript
    _progress(f"Transcript ready — {len(transcript.words)} words, {transcript.duration:.0f}s", 20)

    # Step 2: Find viral moments
    _progress("Finding viral moments...", 25)
    moments = find_viral_moments(transcript, n_clips=n_clips)
    result.total_clips_found = len(moments)
    _progress(f"Found {len(moments)} moments", 40)

    # Step 3: Score moments
    _progress("Scoring virality...", 42)
    scored = score_all(moments)
    _progress(f"Scored {len(scored)} moments", 55)

    # Filter and take top N
    scored = [(m, r) for m, r in scored if r.overall_score >= min_score]
    scored = scored[:top_n_export]

    # Step 4-6: Render each clip
    total = len(scored)
    for idx, (moment, report) in enumerate(scored):
        rank = idx + 1
        pct = 55 + (idx / max(total, 1)) * 40

        safe_title = _safe_filename(moment.title)
        clip_filename = f"{rank:02d}_{safe_title}_{report.overall_score:.0f}.mp4"
        clip_path = os.path.join(output_dir, clip_filename)

        _progress(f"Rendering clip {rank}/{total}: {moment.title}", pct)

        try:
            _render_clip(
                video_path=video_path,
                moment=moment,
                transcript=transcript,
                output_path=clip_path,
                style=style,
                target_width=target_width,
                target_height=target_height,
                enable_reframe=enable_reframe,
                enable_captions=enable_captions,
                output_dir=output_dir,
            )

            result.clips.append(ProcessedClip(
                rank=rank,
                moment=moment,
                report=report,
                output_path=clip_path,
                duration=moment.duration,
            ))
        except Exception as e:
            print(f"[Pipeline] Failed to render clip {rank}: {e}")

    _progress("Done!", 100)
    return result


def _render_clip(
    video_path: str,
    moment: ViralMoment,
    transcript: Transcript,
    output_path: str,
    style: str,
    target_width: int,
    target_height: int,
    enable_reframe: bool,
    enable_captions: bool,
    output_dir: str,
):
    """Render a single clip with reframe and captions."""
    duration = moment.end - moment.start

    # Build filter chain
    filters = []

    if enable_reframe:
        trajectory = analyze_reframe(
            video_path, moment.start, moment.end,
            target_width=target_width, target_height=target_height,
        )
        crop_filter = build_ffmpeg_crop_filter(trajectory, clip_start=0.0)
        filters.append(crop_filter)
    else:
        # Just scale to target
        filters.append(f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease")
        filters.append(f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2")

    # Generate caption file
    ass_file = None
    if enable_captions and transcript.words:
        ass_file = os.path.join(output_dir, f"_caps_{moment.start:.0f}.ass")
        result = generate_caption_file(
            transcript=transcript,
            output_path=ass_file,
            style_name=style,
            clip_start=moment.start,
            clip_end=moment.end,
        )
        if result:
            caption_filter = ffmpeg_caption_filter(ass_file)
            filters.append(caption_filter)

    vf = ",".join(filters) if filters else "null"

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(moment.start),
        "-t", str(duration),
        "-i", video_path,
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr[-500:]}")

    # Clean up temp ASS file
    if ass_file and os.path.exists(ass_file):
        os.unlink(ass_file)


def _safe_filename(title: str) -> str:
    """Convert title to safe filename."""
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", title)
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe[:40]


def print_results(result: PipelineResult):
    """Print a formatted summary of pipeline results."""
    print(f"\n{'='*60}")
    print(f"OPUS PIPELINE COMPLETE")
    print(f"Source: {result.input_video}")
    print(f"Clips found: {result.total_clips_found} → Rendered: {len(result.clips)}")
    print(f"{'='*60}")

    for clip in result.clips:
        m = clip.moment
        r = clip.report
        print(f"\n#{clip.rank} [{r.grade}] {m.title}")
        print(f"   Score: {r.overall_score:.1f}/100  |  {r.predicted_views_range} views")
        print(f"   Platform: {r.best_platform}  |  {clip.duration:.0f}s")
        print(f"   Hook: {m.hook[:70]}")
        print(f"   Verdict: {r.verdict}")
        print(f"   File: {clip.output_path}")
        if r.improvements:
            print(f"   To improve: {r.improvements[0]}")

    print(f"\n{'='*60}\n")
