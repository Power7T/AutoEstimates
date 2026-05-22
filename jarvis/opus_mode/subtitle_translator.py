"""
subtitle_translator.py
----------------------
Translates ASS subtitle files to other languages for broader audience reach.

Workflow
--------
1. Read the source .ass file.
2. Parse every ``Dialogue:`` line and extract visible text (override tags stripped).
3. Translate visible text in batches of 50 lines via the AI router.
4. Inject translated text back into each line, preserving all ASS override tags
   (``{\\k...}``, ``{\\K...}``, ``{\\an...}``, etc.) in their original positions.
5. Write the reassembled file and return its path.

Supported target languages (examples):
    "Spanish", "Portuguese", "Hindi", "French", "German", "Japanese",
    "Korean", "Indonesian", "Arabic", "Italian", "Turkish", ...

Usage
-----
    from jarvis.opus_mode.subtitle_translator import translate_subtitles

    out = translate_subtitles(
        ass_path="/path/to/video.ass",
        target_language="Spanish",
    )
    print(f"Translated file: {out}")
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from ..openrouter import router

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

# Matches a complete Dialogue line:
#   Group 1 — everything up to and including the final ",,"
#             e.g. "Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,"
#   Group 2 — the raw text field  (visible words interleaved with {tag} blocks)
_DIALOGUE_RE = re.compile(
    r"^(Dialogue:[^\n]*,,)"   # prefix, including the last ",,"
    r"(.*)$",                  # raw text field
    re.MULTILINE,
)

# Matches any single ASS override tag block, e.g. {\k30} {\an8} {\K45}
_TAG_RE = re.compile(r"\{[^}]*\}")

# Number of subtitle lines per AI translation call
_BATCH_SIZE = 50


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def translate_subtitles(
    ass_path: str,
    target_language: str,
    output_path: Optional[str] = None,
) -> str:
    """
    Translate an ASS subtitle file to *target_language*.

    Parameters
    ----------
    ass_path:
        Absolute path to the source ``.ass`` file.
    target_language:
        Human-readable language name, e.g. ``"Spanish"``, ``"Hindi"``.
    output_path:
        Where to write the translated file. Defaults to
        ``<original_stem>.<lang_slug>.ass`` in the same directory as the source.

    Returns
    -------
    str
        Absolute path to the written translated file.

    Raises
    ------
    FileNotFoundError
        If *ass_path* does not exist.
    ValueError
        If the file contains no parseable Dialogue lines.
    """
    source = Path(ass_path)
    if not source.exists():
        raise FileNotFoundError(f"ASS file not found: {ass_path}")

    original_content = source.read_text(encoding="utf-8", errors="replace")

    # 1. Extract visible text from every Dialogue line
    indexed_texts: list[tuple[int, str]] = _extract_dialogue_text(original_content)
    if not indexed_texts:
        raise ValueError(
            f"No translatable Dialogue lines found in: {ass_path}"
        )

    logger.info(
        "Translating %d subtitle lines to %s (batch size %d).",
        len(indexed_texts),
        target_language,
        _BATCH_SIZE,
    )

    # 2. Translate in batches
    all_indices = [idx  for idx, _    in indexed_texts]
    all_texts   = [text for _,   text in indexed_texts]
    translations: dict[int, str] = {}

    for batch_start in range(0, len(all_texts), _BATCH_SIZE):
        batch_texts   = all_texts  [batch_start: batch_start + _BATCH_SIZE]
        batch_indices = all_indices[batch_start: batch_start + _BATCH_SIZE]

        translated = _translate_batch(batch_texts, target_language)
        for line_idx, translated_text in zip(batch_indices, translated):
            translations[line_idx] = translated_text

        logger.debug(
            "Translated lines %d–%d.",
            batch_start,
            batch_start + len(batch_texts) - 1,
        )

    # 3. Reassemble with translated text + original tags intact
    translated_content = _reassemble_ass(original_content, translations)

    # 4. Write output file
    if output_path is None:
        lang_slug   = target_language.lower().replace(" ", "_")
        out_name    = f"{source.stem}.{lang_slug}.ass"
        output_path = str(source.parent / out_name)

    Path(output_path).write_text(translated_content, encoding="utf-8")
    logger.info("Translated subtitle written to: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Step 1 — Extract visible text
# ---------------------------------------------------------------------------

def _extract_dialogue_text(ass_content: str) -> list[tuple[int, str]]:
    """
    Return ``(match_index, visible_text)`` pairs for every Dialogue line.

    *match_index* is the zero-based index of the regex match within the full
    file — used later to map translations back to the correct line.

    *visible_text* has all ``{...}`` override tags stripped and is trimmed of
    leading/trailing whitespace. Lines that are blank after stripping are
    excluded (they contain only positioning tags with no visible words).
    """
    results: list[tuple[int, str]] = []
    for match_idx, m in enumerate(_DIALOGUE_RE.finditer(ass_content)):
        raw_text = m.group(2)
        visible  = _strip_tags(raw_text).strip()
        if visible:
            results.append((match_idx, visible))
    return results


def _strip_tags(text: str) -> str:
    """Remove all ``{...}`` blocks from *text*, leaving only visible words."""
    return _TAG_RE.sub("", text)


# ---------------------------------------------------------------------------
# Step 2 — AI batch translation
# ---------------------------------------------------------------------------

def _translate_batch(texts: list[str], target_language: str) -> list[str]:
    """
    Translate a list of visible subtitle strings to *target_language* using AI.

    Parameters
    ----------
    texts:
        Visible subtitle strings with override tags already removed.
    target_language:
        Target language name.

    Returns
    -------
    list[str]
        Translated strings in the same order as the input.
        Falls back to the original strings if the AI call fails.
    """
    if not texts:
        return []

    # Map string index → original text for the JSON payload
    numbered: dict[str, str] = {str(i): t for i, t in enumerate(texts)}

    system_prompt = (
        f"You are a professional subtitle translator specialising in short-form video.\n"
        f"Translate each subtitle line to {target_language}.\n\n"
        "Rules:\n"
        "- Preserve the meaning, tone, and natural spoken feel of every line.\n"
        "- Keep translations concise — subtitles must be readable quickly.\n"
        "- Do NOT add explanatory notes, parenthetical remarks, or ellipses.\n"
        "- Do NOT merge or split lines — one input line = one output line.\n"
        "- Return a JSON object where every key is the original integer index "
        "  (as a string) and every value is the translated string.\n"
        "- Include ALL indices from the input, even for very short text."
    )

    user_prompt = (
        f"Translate the following subtitle lines to {target_language}.\n\n"
        f"Input JSON:\n"
        f"{json.dumps(numbered, ensure_ascii=False, indent=2)}\n\n"
        "Return only a valid JSON object with translated values. "
        "Do not include markdown fences or any extra text."
    )

    schema = {
        "type": "object",
        "additionalProperties": {"type": "string"},
    }

    try:
        result: dict = router.complete_json(
            system=system_prompt,
            user=user_prompt,
            schema=schema,
        )
        # Rebuild list in original order; fall back to source text for missing keys
        return [result.get(str(i), texts[i]) for i in range(len(texts))]

    except Exception as exc:
        logger.warning(
            "Translation batch failed (%s) — keeping original text for this batch.", exc
        )
        return list(texts)


# ---------------------------------------------------------------------------
# Step 3 — Reassemble ASS with tags preserved
# ---------------------------------------------------------------------------

def _reassemble_ass(
    original_content: str,
    translations: dict[int, str],
) -> str:
    """
    Rebuild the full ASS content, replacing visible text in each Dialogue line
    with its translation while keeping all override tags intact.

    Processes matches in reverse order so that string offsets remain valid
    as the content string is modified.
    """
    matches = list(_DIALOGUE_RE.finditer(original_content))
    result  = original_content

    for match_idx in range(len(matches) - 1, -1, -1):
        if match_idx not in translations:
            continue

        m                = matches[match_idx]
        prefix           = m.group(1)      # "Dialogue: ...,,"
        raw_text         = m.group(2)      # original text field with tags
        translated_vis   = translations[match_idx]

        new_text_field   = _inject_tags(raw_text, translated_vis)
        new_line         = prefix + new_text_field

        start, end = m.span()
        result = result[:start] + new_line + result[end:]

    return result


def _inject_tags(original_raw: str, translated_visible: str) -> str:
    """
    Reconstruct a dialogue text field by interleaving the original ASS override
    tags around the translated visible text.

    Algorithm
    ---------
    1. Tokenise *original_raw* into alternating ``("tag", "{...}")`` and
       ``("text", "...")`` tokens, preserving their order.
    2. Determine how many source words belong to each text token.
    3. Distribute the translated words proportionally across those same slots.
    4. Re-interleave tag tokens and translated text chunks.

    This faithfully preserves ``{\\k}`` karaoke timing, ``{\\an}`` alignment,
    and all other positional/style tags. Lines that are tags-only are returned
    unchanged.
    """
    # ---- Tokenise -------------------------------------------------------
    tokens: list[tuple[str, str]] = []   # ("tag" | "text", value)
    cursor = 0
    for tag_match in _TAG_RE.finditer(original_raw):
        start, end = tag_match.span()
        if start > cursor:
            tokens.append(("text", original_raw[cursor:start]))
        tokens.append(("tag", tag_match.group(0)))
        cursor = end
    if cursor < len(original_raw):
        tokens.append(("text", original_raw[cursor:]))

    # ---- Guard: nothing visible -----------------------------------------
    text_chunks = [v for kind, v in tokens if kind == "text"]
    if not any(chunk.strip() for chunk in text_chunks):
        return original_raw

    # ---- Proportional word distribution ---------------------------------
    translated_words = translated_visible.split()
    word_counts      = [len(chunk.split()) for chunk in text_chunks]
    total_src_words  = sum(word_counts) or 1

    translated_chunks: list[str] = []
    word_cursor = 0
    for i, src_count in enumerate(word_counts):
        is_last = (i == len(word_counts) - 1)
        if is_last:
            chunk_words = translated_words[word_cursor:]
        else:
            share       = round(src_count / total_src_words * len(translated_words))
            chunk_words = translated_words[word_cursor: word_cursor + share]
            word_cursor += share

        # Mirror the leading/trailing whitespace pattern of the original chunk
        orig           = text_chunks[i]
        leading_space  = " " if orig and orig[0] == " " else ""
        trailing_space = " " if orig and orig[-1] == " " else ""
        translated_chunks.append(
            leading_space + " ".join(chunk_words) + trailing_space
        )

    # ---- Reassemble -----------------------------------------------------
    text_iter    = iter(translated_chunks)
    result_parts: list[str] = []
    for kind, value in tokens:
        if kind == "tag":
            result_parts.append(value)
        else:
            result_parts.append(next(text_iter, ""))

    return "".join(result_parts)
