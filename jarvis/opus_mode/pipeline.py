"""
OpusClip-level pipeline — turns a long video into viral short clips.

Full flow:
  1. Transcribe (Whisper, word-level timestamps)
  2. Remove silences / filler words
  3. Find viral moments (Claude)
  4. Score each moment (6-dimension virality score)
  5. Smart reframe to 9:16 (face tracking)
  6. Burn word-by-word animated captions (ASS format)
  7. Generate hook text overlay (first 3s)
  8. Generate thumbnail per clip
  9. Export with platform-specific specs
  10. Optionally translate subtitles

Checkpointing: saves progress to {output_dir}/.checkpoint.json after each
heavy step so a crashed run resumes from where it left off.

Output: folder of MP4 clips, each named by rank and title.
"""

import json
import os
import re
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .transcriber import transcribe, Transcript
from .moment_finder import find_viral_moments, ViralMoment
from .virality_scorer import score_all, ViralityReport
from .caption_animator import generate_caption_file, ffmpeg_caption_filter
from .smart_reframe import analyze_reframe, build_ffmpeg_crop_filter
from .speaker_detector import detect_speakers

try:
    from .silence_remover import remove_silences, get_keep_intervals
    _SILENCE_REMOVER = True
except ImportError:
    _SILENCE_REMOVER = False

try:
    from .hook_generator import generate_hook, burn_hook_to_clip
    _HOOK_GEN = True
except ImportError:
    _HOOK_GEN = False

try:
    from .thumbnail_generator import generate_thumbnail
    _THUMB_GEN = True
except ImportError:
    _THUMB_GEN = False

try:
    from .platform_exporter import export_for_platform, PLATFORM_SPECS
    _PLATFORM_EXPORT = True
except ImportError:
    _PLATFORM_EXPORT = False

try:
    from .subtitle_translator import translate_subtitles
    _TRANSLATOR = True
except ImportError:
    _TRANSLATOR = False


@dataclass
class ProcessedClip:
    rank: int
    moment: ViralMoment
    report: ViralityReport
    output_path: str
    duration: float
    thumbnail_path: str = ""
    translated_paths: dict = field(default_factory=dict)  # lang -> path


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
    enable_silence_removal: bool = True,
    enable_hooks: bool = True,
    enable_thumbnails: bool = True,
    platform: str = "instagram",
    translate_to: list[str] = None,
    min_score: float = 0.0,
    resume: bool = True,
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
        enable_silence_removal: Remove pauses and filler words
        enable_hooks: Burn hook text overlay on first 3 seconds
        enable_thumbnails: Generate thumbnail per clip
        platform: Target platform for export specs
        translate_to: List of languages to translate subtitles into
        min_score: Skip clips below this virality score
        resume: Resume from checkpoint if available
        on_progress: Optional callback(step: str, pct: float)
    """
    result = PipelineResult(input_video=video_path, output_dir=output_dir)
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_path = os.path.join(output_dir, ".checkpoint.json")
    checkpoint = _load_checkpoint(checkpoint_path) if resume else {}

    def _progress(step: str, pct: float):
        print(f"[Pipeline] {step} ({pct:.0f}%)")
        if on_progress:
            on_progress(step, pct)

    # Step 1: Transcribe
    if "transcript" in checkpoint:
        _progress("Resuming from checkpoint — transcript ready", 20)
        transcript = _deserialize_transcript(checkpoint["transcript"])
    else:
        _progress("Transcribing video...", 5)
        transcript = transcribe(video_path, model_size=whisper_model)
        checkpoint["transcript"] = _serialize_transcript(transcript)
        _save_checkpoint(checkpoint_path, checkpoint)
        _progress(f"Transcript ready — {len(transcript.words)} words, {transcript.duration:.0f}s", 20)

    result.transcript = transcript

    # Step 2: Remove silences/fillers (optional, applied before moment finding)
    if enable_silence_removal and _SILENCE_REMOVER and "keep_intervals" not in checkpoint:
        _progress("Removing silences and filler words...", 22)
        try:
            keep_intervals = get_keep_intervals(transcript, gap_threshold=0.4, remove_fillers=True)
            checkpoint["keep_intervals"] = keep_intervals
            _save_checkpoint(checkpoint_path, checkpoint)
            words_before = len(transcript.words)
            # Trim words to keep intervals for moment finding
            transcript = _filter_transcript_to_intervals(transcript, keep_intervals)
            _progress(f"Silence removed — {words_before} → {len(transcript.words)} words", 24)
        except Exception as e:
            _progress(f"Silence removal skipped: {e}", 24)

    # Step 3: Find viral moments
    if "moments" in checkpoint:
        _progress("Resuming from checkpoint — moments ready", 40)
        moments = [_deserialize_moment(m) for m in checkpoint["moments"]]
    else:
        _progress("Finding viral moments...", 25)
        moments = find_viral_moments(transcript, n_clips=n_clips)
        result.total_clips_found = len(moments)
        checkpoint["moments"] = [_serialize_moment(m) for m in moments]
        _save_checkpoint(checkpoint_path, checkpoint)
        _progress(f"Found {len(moments)} moments", 40)

    result.total_clips_found = len(moments)

    # Step 4: Score moments
    if "scored" in checkpoint:
        _progress("Resuming from checkpoint — scores ready", 55)
        scored_raw = checkpoint["scored"]
        scored = [
            (_deserialize_moment(s["moment"]), _deserialize_report(s["report"]))
            for s in scored_raw
        ]
    else:
        _progress("Scoring virality...", 42)
        scored = score_all(moments)
        checkpoint["scored"] = [
            {"moment": _serialize_moment(m), "report": _serialize_report(r)}
            for m, r in scored
        ]
        _save_checkpoint(checkpoint_path, checkpoint)
        _progress(f"Scored {len(scored)} moments", 55)

    # Filter and take top N
    scored = [(m, r) for m, r in scored if r.overall_score >= min_score]
    scored = scored[:top_n_export]

    # Step 5-9: Render each clip
    total = len(scored)
    rendered_ids = set(checkpoint.get("rendered_clips", []))

    for idx, (moment, report) in enumerate(scored):
        rank = idx + 1
        pct = 55 + (idx / max(total, 1)) * 40

        safe_title = _safe_filename(moment.title)
        clip_filename = f"{rank:02d}_{safe_title}_{report.overall_score:.0f}.mp4"
        clip_path = os.path.join(output_dir, clip_filename)
        clip_id = f"clip_{rank}"

        if clip_id in rendered_ids and os.path.exists(clip_path):
            _progress(f"Skipping clip {rank}/{total} (already rendered)", pct)
        else:
            _progress(f"Rendering clip {rank}/{total}: {moment.title}", pct)
            try:
                _render_clip(
                    video_path=video_path,
                    moment=moment,
                    transcript=result.transcript,
                    output_path=clip_path,
                    style=style,
                    target_width=target_width,
                    target_height=target_height,
                    enable_reframe=enable_reframe,
                    enable_captions=enable_captions,
                    output_dir=output_dir,
                )
                rendered_ids.add(clip_id)
                checkpoint["rendered_clips"] = list(rendered_ids)
                _save_checkpoint(checkpoint_path, checkpoint)
            except Exception as e:
                print(f"[Pipeline] Failed to render clip {rank}: {e}")
                continue

        # Hook overlay
        if enable_hooks and _HOOK_GEN and os.path.exists(clip_path):
            try:
                hooked_path = clip_path.replace(".mp4", "_hooked.mp4")
                hook = generate_hook(moment)
                burn_hook_to_clip(clip_path, hook, hooked_path)
                if os.path.exists(hooked_path):
                    os.replace(hooked_path, clip_path)
            except Exception as e:
                print(f"[Pipeline] Hook generation skipped: {e}")

        # Platform-specific export
        if _PLATFORM_EXPORT and platform != "generic" and os.path.exists(clip_path):
            try:
                platform_path = clip_path.replace(".mp4", f"_{platform}.mp4")
                export_for_platform(clip_path, platform_path, platform=platform)
                if os.path.exists(platform_path):
                    os.replace(platform_path, clip_path)
            except Exception as e:
                print(f"[Pipeline] Platform export skipped: {e}")

        # Thumbnail
        thumb_path = ""
        if enable_thumbnails and _THUMB_GEN and os.path.exists(clip_path):
            try:
                thumb_path = clip_path.replace(".mp4", "_thumb.jpg")
                generate_thumbnail(clip_path, thumb_path, add_text=moment.title)
            except Exception as e:
                print(f"[Pipeline] Thumbnail skipped: {e}")

        # Subtitle translation
        translated_paths = {}
        if translate_to and _TRANSLATOR:
            for lang in translate_to:
                try:
                    ass_src = os.path.join(output_dir, f"_caps_{moment.start:.0f}.ass")
                    if os.path.exists(ass_src):
                        lang_ass = ass_src.replace(".ass", f"_{lang}.ass")
                        translate_subtitles(ass_src, lang, lang_ass)
                        translated_paths[lang] = lang_ass
                except Exception as e:
                    print(f"[Pipeline] Translation to {lang} skipped: {e}")

        result.clips.append(ProcessedClip(
            rank=rank,
            moment=moment,
            report=report,
            output_path=clip_path,
            duration=moment.duration,
            thumbnail_path=thumb_path,
            translated_paths=translated_paths,
        ))

    # Clean up checkpoint on success
    if os.path.exists(checkpoint_path):
        os.unlink(checkpoint_path)

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
    filters = []

    if enable_reframe:
        trajectory = analyze_reframe(
            video_path, moment.start, moment.end,
            target_width=target_width, target_height=target_height,
        )
        crop_filter = build_ffmpeg_crop_filter(trajectory, clip_start=0.0)
        filters.append(crop_filter)
    else:
        filters.append(f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease")
        filters.append(f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2")

    ass_file = None
    if enable_captions and transcript and transcript.words:
        ass_file = os.path.join(output_dir, f"_caps_{moment.start:.0f}.ass")
        result = generate_caption_file(
            transcript=transcript,
            output_path=ass_file,
            style_name=style,
            clip_start=moment.start,
            clip_end=moment.end,
        )
        if result:
            filters.append(ffmpeg_caption_filter(ass_file))

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

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {proc.stderr[-500:]}")

    if ass_file and os.path.exists(ass_file):
        os.unlink(ass_file)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(path: str, data: dict):
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def _load_checkpoint(path: str) -> dict:
    if os.path.exists(path):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _serialize_transcript(t: Transcript) -> dict:
    return {
        "full_text": t.full_text,
        "duration": t.duration,
        "language": t.language,
        "words": [{"text": w.text, "start": w.start, "end": w.end, "confidence": w.confidence} for w in t.words],
        "segments": [
            {"text": s.text, "start": s.start, "end": s.end,
             "words": [{"text": w.text, "start": w.start, "end": w.end, "confidence": w.confidence} for w in s.words]}
            for s in t.segments
        ],
    }


def _deserialize_transcript(d: dict) -> Transcript:
    from .transcriber import Word, Segment
    words = [Word(**w) for w in d["words"]]
    segments = [
        Segment(text=s["text"], start=s["start"], end=s["end"],
                words=[Word(**w) for w in s.get("words", [])])
        for s in d["segments"]
    ]
    return Transcript(full_text=d["full_text"], segments=segments, words=words,
                      duration=d["duration"], language=d.get("language", "en"))


def _serialize_moment(m: ViralMoment) -> dict:
    return {k: v for k, v in m.__dict__.items()}


def _deserialize_moment(d: dict) -> ViralMoment:
    return ViralMoment(**d)


def _serialize_report(r: ViralityReport) -> dict:
    return {k: v for k, v in r.__dict__.items()}


def _deserialize_report(d: dict) -> ViralityReport:
    return ViralityReport(**d)


def _filter_transcript_to_intervals(transcript: Transcript, keep_intervals: list) -> Transcript:
    """Filter transcript words to only those within keep intervals."""
    from .transcriber import Word, Segment, Transcript as T

    def in_keep(start: float, end: float) -> bool:
        for ks, ke in keep_intervals:
            if start >= ks and end <= ke + 0.05:
                return True
        return False

    filtered_words = [w for w in transcript.words if in_keep(w.start, w.end)]
    filtered_segments = []
    for seg in transcript.segments:
        seg_words = [w for w in seg.words if in_keep(w.start, w.end)]
        if seg_words:
            filtered_segments.append(Segment(
                text=" ".join(w.text for w in seg_words),
                start=seg_words[0].start,
                end=seg_words[-1].end,
                words=seg_words,
            ))

    return T(
        full_text=" ".join(w.text for w in filtered_words),
        segments=filtered_segments,
        words=filtered_words,
        duration=transcript.duration,
        language=transcript.language,
    )


def _safe_filename(title: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", title)
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe[:40]


def print_results(result: PipelineResult):
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
        if clip.thumbnail_path:
            print(f"   Thumb: {clip.thumbnail_path}")
        if clip.translated_paths:
            print(f"   Translations: {', '.join(clip.translated_paths.keys())}")
        if r.improvements:
            print(f"   To improve: {r.improvements[0]}")

    print(f"\n{'='*60}\n")
