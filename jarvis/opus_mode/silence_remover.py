"""
silence_remover.py

Removes silences and filler words from video clips to tighten pacing.
Uses the FFmpeg concat-demuxer approach (per-segment trim + concat) for
simple, reliable segment splicing.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field

from jarvis.opus_mode.transcriber import Transcript, Word  # noqa: F401

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Filler word vocabulary
# ---------------------------------------------------------------------------

# Always removed regardless of confidence when surrounded by silence.
_FILLER_SINGLE: set[str] = {
    "um", "uh", "hmm", "hm",
}

# Context-dependent fillers: removed only when confidence is low or paused.
_FILLER_CONTEXT: set[str] = {
    "like", "basically", "literally", "actually", "so", "right", "okay", "ok",
}

# Multi-word fillers detected by matching a span of consecutive words.
_FILLER_MULTI: set[str] = {
    "you know",
    "i mean",
}

# Confidence below which a context filler is unconditionally removed.
_LOW_CONFIDENCE: float = 0.75

# Half of gap_threshold: if a word has this much silence on either side it is
# treated as filler-eligible regardless of confidence.
_PAUSE_FILLER_RATIO: float = 0.5


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SilenceReport:
    """Summary of what was removed during silence/filler processing."""

    segments_removed: int = 0
    time_saved_seconds: float = 0.0
    filler_words_removed: int = 0
    keep_intervals: list[tuple[float, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def remove_silences(
    video_path: str,
    transcript: Transcript,
    output_path: str,
    gap_threshold: float = 0.4,
    remove_fillers: bool = True,
    min_keep_duration: float = 0.1,
) -> tuple[str, SilenceReport]:
    """Remove silences and optionally filler words from *video_path*.

    Parameters
    ----------
    video_path:
        Path to the source video file.
    transcript:
        Transcript produced by transcriber.transcribe() — must contain
        word-level timing (Word.start / Word.end / Word.text).
    output_path:
        Destination path for the cleaned video.
    gap_threshold:
        Pauses longer than this many seconds between consecutive words are cut.
        Default is 0.4 s.
    remove_fillers:
        When True, standalone filler words/phrases are also removed.
    min_keep_duration:
        Intervals shorter than this (seconds) are discarded to avoid single-frame
        or near-invisible clips.

    Returns
    -------
    tuple[str, SilenceReport]
        The output path and a report describing what was removed.

    Raises
    ------
    FileNotFoundError
        If *video_path* does not exist.
    RuntimeError
        If FFmpeg exits with a non-zero return code.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not transcript.words:
        logger.warning("Empty transcript — copying source unchanged.")
        _ffmpeg_copy(video_path, output_path)
        return output_path, SilenceReport()

    keep_intervals, filler_count = get_keep_intervals(
        transcript, gap_threshold, remove_fillers
    )

    # Discard micro-clips that would cause concat artefacts.
    keep_intervals = [
        (s, e) for s, e in keep_intervals if e - s >= min_keep_duration
    ]

    if not keep_intervals:
        logger.warning("No keep intervals found after filtering — copying source unchanged.")
        _ffmpeg_copy(video_path, output_path)
        return output_path, SilenceReport(filler_words_removed=filler_count)

    original_duration = transcript.duration
    kept_duration = sum(e - s for s, e in keep_intervals)
    time_saved = max(0.0, original_duration - kept_duration)
    segments_removed = _count_removed_segments(keep_intervals, original_duration)

    report = SilenceReport(
        segments_removed=segments_removed,
        time_saved_seconds=round(time_saved, 3),
        filler_words_removed=filler_count,
        keep_intervals=keep_intervals,
    )

    try:
        _build_ffmpeg_silence_filter(video_path, keep_intervals, output_path)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"FFmpeg failed while removing silences: {exc.stderr}"
        ) from exc

    logger.info(
        "Silence removal complete: %d segment(s) removed, %.2fs saved, "
        "%d filler word(s) cut.",
        report.segments_removed,
        report.time_saved_seconds,
        report.filler_words_removed,
    )
    return output_path, report


def get_keep_intervals(
    transcript: Transcript,
    gap_threshold: float,
    remove_fillers: bool,
) -> tuple[list[tuple[float, float]], int]:
    """Compute the list of ``(start, end)`` intervals to keep.

    Algorithm
    ---------
    1. Walk through every word and flag filler candidates for removal.
    2. Merge consecutive non-removed words into intervals, splitting whenever
       the gap between two words exceeds *gap_threshold*.
    3. Add small edge padding (50 ms) so cuts don't clip phonemes.
    4. Leading/trailing silence is implicitly excluded because intervals only
       span the spoken words.

    Parameters
    ----------
    transcript:
        Word-level transcript.
    gap_threshold:
        Maximum silence (seconds) between consecutive words before a cut is made.
    remove_fillers:
        Whether to apply filler-word removal logic.

    Returns
    -------
    tuple[list[tuple[float, float]], int]
        Keep intervals and the number of filler words identified for removal.
    """
    words: list[Word] = transcript.words
    if not words:
        return [], 0

    filler_count = 0
    # keep_flags[i] == False means the word is removed entirely.
    keep_flags: list[bool] = [True] * len(words)

    if remove_fillers:
        n = len(words)
        i = 0
        while i < n:
            word = words[i]
            normalized = word.text.strip().lower().rstrip(".,!?;:")
            confidence = getattr(word, "confidence", 1.0)

            # Check for multi-word filler spanning i..i+k
            matched_multi = False
            for phrase in _FILLER_MULTI:
                phrase_tokens = phrase.split()
                span = len(phrase_tokens)
                if i + span <= n:
                    candidate = " ".join(
                        words[j].text.strip().lower().rstrip(".,!?;:")
                        for j in range(i, i + span)
                    )
                    if candidate == phrase:
                        # Mark all words in the span for removal
                        for j in range(i, i + span):
                            keep_flags[j] = False
                        filler_count += span
                        i += span
                        matched_multi = True
                        break

            if matched_multi:
                continue

            # Single-word filler check
            if normalized in _FILLER_SINGLE:
                keep_flags[i] = False
                filler_count += 1
            elif normalized in _FILLER_CONTEXT:
                prev_end = words[i - 1].end if i > 0 else 0.0
                next_start = words[i + 1].start if i < n - 1 else transcript.duration
                gap_before = word.start - prev_end
                gap_after = next_start - word.end
                pause_threshold = gap_threshold * _PAUSE_FILLER_RATIO

                if (
                    confidence < _LOW_CONFIDENCE
                    or gap_before >= pause_threshold
                    or gap_after >= pause_threshold
                    or i == 0
                    or i == n - 1
                ):
                    keep_flags[i] = False
                    filler_count += 1

            i += 1

    # Build intervals by merging runs of kept words separated by gaps ≤ gap_threshold
    intervals: list[tuple[float, float]] = []
    current_start: float | None = None
    current_end: float | None = None

    for i, (word, keep) in enumerate(zip(words, keep_flags)):
        if not keep:
            if current_start is not None:
                intervals.append((current_start, current_end))  # type: ignore[arg-type]
                current_start = None
                current_end = None
            continue

        if current_start is None:
            # Start a new interval; add a small leading pad so the cut doesn't
            # land on the first phoneme.
            current_start = max(0.0, word.start - 0.05)
            current_end = word.end
        else:
            gap = word.start - current_end  # type: ignore[operator]
            if gap <= gap_threshold:
                current_end = word.end
            else:
                intervals.append((current_start, current_end))  # type: ignore[arg-type]
                current_start = max(0.0, word.start - 0.05)
                current_end = word.end

    if current_start is not None:
        intervals.append((current_start, current_end))  # type: ignore[arg-type]

    # Add trailing padding (50 ms) so the last phoneme isn't clipped.
    padded: list[tuple[float, float]] = [
        (s, min(transcript.duration, e + 0.05)) for s, e in intervals
    ]

    return padded, filler_count


def _build_ffmpeg_silence_filter(
    video_path: str,
    keep_intervals: list[tuple[float, float]],
    output_path: str,
) -> str:
    """Trim *video_path* to *keep_intervals* and write *output_path* via FFmpeg.

    Implementation
    --------------
    For each keep interval, a separate segment clip is produced with
    ``-ss``/``-to`` seek arguments.  All segments are then concatenated with
    the FFmpeg concat demuxer.  This is simpler and more reliable than a
    complex filtergraph, especially for long files.

    Parameters
    ----------
    video_path:
        Source video file.
    keep_intervals:
        Ordered list of ``(start, end)`` pairs in seconds.
    output_path:
        Destination MP4 path.

    Returns
    -------
    str
        The *output_path* on success.

    Raises
    ------
    subprocess.CalledProcessError
        If any FFmpeg invocation exits with a non-zero return code.
    """
    tmp_dir = tempfile.mkdtemp(prefix="silence_remover_")
    segment_paths: list[str] = []

    try:
        # ------------------------------------------------------------------ #
        # Step 1: extract each keep interval as an independent clip           #
        # ------------------------------------------------------------------ #
        for idx, (start, end) in enumerate(keep_intervals):
            seg_path = os.path.join(tmp_dir, f"seg_{idx:04d}.mp4")
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-to", str(end),
                "-i", video_path,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "18",
                "-c:a", "aac",
                "-b:a", "192k",
                "-avoid_negative_ts", "make_zero",
                seg_path,
            ]
            logger.debug("Extracting segment %d: %.3f – %.3f s", idx, start, end)
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise subprocess.CalledProcessError(
                    result.returncode, cmd, result.stdout, result.stderr
                )
            segment_paths.append(seg_path)

        # ------------------------------------------------------------------ #
        # Step 2: write the concat demuxer list                               #
        # ------------------------------------------------------------------ #
        concat_list_path = os.path.join(tmp_dir, "concat.txt")
        with open(concat_list_path, "w", encoding="utf-8") as fh:
            for sp in segment_paths:
                fh.write(f"file '{sp}'\n")

        # ------------------------------------------------------------------ #
        # Step 3: concatenate all segments into the final output              #
        # ------------------------------------------------------------------ #
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cmd_concat = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list_path,
            "-c", "copy",
            output_path,
        ]
        logger.debug(
            "Concatenating %d segments → %s", len(segment_paths), output_path
        )
        result = subprocess.run(cmd_concat, capture_output=True, text=True)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, cmd_concat, result.stdout, result.stderr
            )

    finally:
        # Best-effort cleanup of temp files
        for sp in segment_paths:
            try:
                os.remove(sp)
            except OSError:
                pass
        for name in ("concat.txt",):
            try:
                os.remove(os.path.join(tmp_dir, name))
            except OSError:
                pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass

    return output_path


def _is_filler(word_text: str, word_confidence: float = 1.0) -> bool:
    """Return True if *word_text* is a filler candidate.

    Hard fillers (``um``, ``uh``, ``hmm``) are always flagged.
    Context fillers (``like``, ``so``, ``actually``, …) are flagged only when
    *word_confidence* is below :data:`_LOW_CONFIDENCE`.
    """
    normalized = word_text.strip().lower().rstrip(".,!?;:")

    if normalized in _FILLER_SINGLE:
        return True

    if normalized in _FILLER_CONTEXT and word_confidence < _LOW_CONFIDENCE:
        return True

    return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _count_removed_segments(
    keep_intervals: list[tuple[float, float]],
    total_duration: float,
) -> int:
    """Count the number of distinct removed segments.

    Removed segments = leading silence + gaps between kept intervals +
    trailing silence.
    """
    if not keep_intervals:
        return 1  # entire video removed

    count = 0
    if keep_intervals[0][0] > 0.05:
        count += 1
    for i in range(len(keep_intervals) - 1):
        gap = keep_intervals[i + 1][0] - keep_intervals[i][1]
        if gap > 0.001:
            count += 1
    if total_duration - keep_intervals[-1][1] > 0.05:
        count += 1
    return count


def _ffmpeg_copy(src: str, dst: str) -> None:
    """Stream-copy *src* to *dst* without re-encoding."""
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-c", "copy", dst],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg copy failed (exit {result.returncode}):\n{result.stderr[-1000:]}"
        )
