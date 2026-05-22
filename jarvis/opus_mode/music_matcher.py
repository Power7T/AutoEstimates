"""
music_matcher.py
----------------
Matches a video moment's mood and energy to a royalty-free music track.

Sources
-------
1. Local directory scan  — ranks audio files by filename keyword scoring.
2. Pixabay Music API     — free search at https://pixabay.com/api/
                          Requires PIXABAY_API_KEY env var.
                          If no key is found, a manual search query is embedded
                          in the returned MusicMatch.url so the user can find
                          tracks manually on the Pixabay website.

Usage
-----
    from jarvis.opus_mode.music_matcher import match_music, MusicMatch

    match: MusicMatch = match_music(
        moment_emotion="excitement",
        moment_energy="high",
        music_dir="/path/to/local/music",   # optional
    )
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..openrouter import router
from ..config import cfg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MOOD_TO_SEARCH_QUERY: dict[str, str] = {
    "excitement":   "energetic upbeat electronic",
    "shock":        "dramatic tension cinematic",
    "humor":        "fun quirky comedy",
    "inspiration":  "uplifting motivational",
    "relatability": "chill acoustic emotional",
    "curiosity":    "mysterious ambient",
}

# Keywords checked against local audio filenames
_MOOD_KEYWORDS: dict[str, list[str]] = {
    "excitement":   ["hype", "upbeat", "energy", "pump", "epic", "fast", "intense",
                     "exciting", "power", "drive", "electric", "adrenaline"],
    "shock":        ["dramatic", "tension", "suspense", "cinematic", "dark", "impact",
                     "sudden", "stinger", "jolt", "danger", "tense"],
    "humor":        ["funny", "quirky", "comedy", "silly", "fun", "playful", "wacky",
                     "cartoon", "light", "laugh", "goofy"],
    "inspiration":  ["inspire", "motivate", "uplifting", "rise", "triumph", "hope",
                     "soar", "achieve", "success", "winning", "overcome"],
    "relatability": ["chill", "acoustic", "emotional", "soft", "gentle", "calm",
                     "vibe", "cozy", "warm", "lofi", "lo-fi", "sad", "heartfelt"],
    "curiosity":    ["mysterious", "ambient", "wonder", "space", "explore", "strange",
                     "unknown", "ethereal", "discovery", "dreamy"],
}

_SYNONYM_MAP: dict[str, str] = {
    "funny":        "humor",
    "laugh":        "humor",
    "hilarious":    "humor",
    "energy":       "excitement",
    "exciting":     "excitement",
    "excited":      "excitement",
    "energetic":    "excitement",
    "hype":         "excitement",
    "motivated":    "inspiration",
    "motivational": "inspiration",
    "inspiring":    "inspiration",
    "scary":        "shock",
    "surprising":   "shock",
    "surprised":    "shock",
    "tense":        "shock",
    "dramatic":     "shock",
    "wonder":       "curiosity",
    "mysterious":   "curiosity",
    "chill":        "relatability",
    "sad":          "relatability",
    "emotional":    "relatability",
    "relatable":    "relatability",
}

_AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a"}
)

_PIXABAY_API_URL = "https://pixabay.com/api/"
_BATCH_LIMIT = 20   # max candidates forwarded to the AI


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class MusicMatch:
    title: str
    artist: str
    mood: str
    bpm_estimate: float
    why_it_fits: str
    url: str = ""
    local_path: str = ""
    royalty_free: bool = True


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def match_music(
    moment_emotion: str,
    moment_energy: str,
    music_dir: Optional[str] = None,
) -> MusicMatch:
    """
    Return the best royalty-free MusicMatch for the given emotion + energy level.

    Parameters
    ----------
    moment_emotion:
        Dominant emotion key (e.g. "excitement", "humor") or free-form description.
    moment_energy:
        Energy descriptor: "low" | "medium" | "high" | free-form text.
    music_dir:
        Optional path to a local directory of audio files. Scanned when provided.

    Returns
    -------
    MusicMatch
        Best matching track with full metadata.
    """
    emotion_key = _normalise_emotion(moment_emotion)
    search_query = MOOD_TO_SEARCH_QUERY.get(
        emotion_key,
        f"{moment_emotion} {moment_energy} music",
    )
    candidates: list[dict] = []

    # 1. Local directory scan
    if music_dir:
        local_candidates = _scan_local_dir(music_dir, emotion_key)
        candidates.extend(local_candidates)
        logger.debug("Local candidates found: %d", len(local_candidates))

    # 2. Pixabay API
    pixabay_candidates = _search_pixabay(search_query)
    candidates.extend(pixabay_candidates)
    logger.debug("Pixabay candidates found: %d", len(pixabay_candidates))

    # 3. If nothing found, create a manual-search placeholder and let AI reason
    if not candidates:
        logger.info(
            "No candidates found. Set PIXABAY_API_KEY or provide music_dir.\n"
            "Manual Pixabay search: https://pixabay.com/music/?q=%s",
            urllib.parse.quote_plus(search_query),
        )
        candidates.append(_make_manual_candidate(search_query, emotion_key))

    return _ai_pick_best(candidates, moment_emotion, moment_energy)


# ---------------------------------------------------------------------------
# Pixabay API
# ---------------------------------------------------------------------------

def _search_pixabay(query: str) -> list[dict]:
    """
    Search Pixabay's free music API and return normalised candidate dicts.

    Returns an empty list when the API key is absent or the request fails.

    Each returned dict has keys:
        source, title, artist, url, preview_url, tags, duration, local_path
    """
    api_key = (
        os.environ.get("PIXABAY_API_KEY", "").strip()
        or cfg.get("pixabay_api_key", "")
    )
    if not api_key:
        logger.info(
            "PIXABAY_API_KEY not set — skipping Pixabay search for query: %s", query
        )
        return []

    params = urllib.parse.urlencode(
        {
            "key":      api_key,
            "q":        query,
            "category": "music",
            "per_page": 15,
        }
    )
    request_url = f"{_PIXABAY_API_URL}?{params}"

    try:
        with urllib.request.urlopen(request_url, timeout=12) as resp:
            data: dict = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, OSError) as exc:
        logger.warning("Pixabay request failed: %s", exc)
        return []

    candidates: list[dict] = []
    for hit in data.get("hits", []):
        raw_tags = hit.get("tags", "")
        # Use first tag as title when no dedicated title field is present
        title = (
            hit.get("title")
            or (raw_tags.split(",")[0].strip().title() if raw_tags else "Unknown Track")
        )
        candidates.append(
            {
                "source":      "pixabay",
                "title":       title,
                "artist":      hit.get("user", "Unknown Artist"),
                "url":         hit.get("pageURL", ""),
                "preview_url": hit.get("previewURL", ""),
                "tags":        raw_tags,
                "duration":    hit.get("duration", 0),
                "local_path":  "",
            }
        )

    return candidates


# ---------------------------------------------------------------------------
# Local file scoring + scanning
# ---------------------------------------------------------------------------

def _score_local_file(path: str, target_mood: str) -> float:
    """
    Score a local audio file against *target_mood* based on filename keywords.

    Returns a float in [0.0, 1.0] — higher means a better mood match.

    Scoring rules
    -------------
    - Each matching keyword from the target-mood list adds 1 point.
    - The mood key itself appearing verbatim in the filename adds a 0.3 bonus.
    - A BPM hint in the filename (e.g. "120bpm") adds a 0.2 bonus.
    - The final score is normalised by the number of mood keywords and clamped
      to [0.0, 1.0].
    """
    filename = Path(path).stem.lower()
    filename = re.sub(r"[-_.]", " ", filename)
    tokens = set(filename.split())

    mood_key = _normalise_emotion(target_mood)
    keywords = _MOOD_KEYWORDS.get(mood_key, [])
    if not keywords:
        return 0.0

    hits = sum(
        1
        for kw in keywords
        if kw in tokens or any(kw in tok for tok in tokens)
    )
    score = hits / len(keywords)

    # Bonus: mood key present verbatim in the filename
    if mood_key in filename:
        score = min(1.0, score + 0.3)

    # Bonus: explicit BPM hint (e.g. "120bpm", "140 bpm")
    if re.search(r"\d{2,3}\s*bpm", filename):
        score = min(1.0, score + 0.2)

    return round(score, 4)


def _scan_local_dir(music_dir: str, emotion_key: str) -> list[dict]:
    """
    Walk *music_dir* recursively, score each audio file, and return the top 10
    as normalised candidate dicts sorted by score descending.
    """
    root = Path(music_dir)
    if not root.is_dir():
        logger.warning("music_dir is not a directory: %s", music_dir)
        return []

    scored: list[tuple[float, Path]] = []
    for p in root.rglob("*"):
        if p.suffix.lower() in _AUDIO_EXTENSIONS:
            score = _score_local_file(str(p), emotion_key)
            scored.append((score, p))

    scored.sort(key=lambda t: t[0], reverse=True)

    results: list[dict] = []
    for score, p in scored[:10]:
        results.append(
            {
                "source":      "local",
                "title":       p.stem.replace("-", " ").replace("_", " ").title(),
                "artist":      "Local Library",
                "url":         "",
                "preview_url": "",
                "tags":        emotion_key,
                "duration":    0,
                "local_path":  str(p),
                "score":       score,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manual_candidate(query: str, emotion_key: str) -> dict:
    """
    Placeholder candidate used when no real tracks are available.
    The url field points the user to a Pixabay manual search.
    """
    return {
        "source":      "manual_search",
        "title":       f'Search Pixabay: "{query}"',
        "artist":      "— set PIXABAY_API_KEY or search manually —",
        "url":         (
            "https://pixabay.com/music/search/?q="
            + urllib.parse.quote_plus(query)
        ),
        "preview_url": "",
        "tags":        emotion_key,
        "duration":    0,
        "local_path":  "",
        "score":       0.0,
    }


def _normalise_emotion(emotion: str) -> str:
    """
    Map a free-form emotion string to one of the canonical MOOD_TO_SEARCH_QUERY
    keys. Falls back to ``"excitement"`` when no match is found.
    """
    lower = emotion.lower().strip()

    if lower in MOOD_TO_SEARCH_QUERY:
        return lower

    if lower in _SYNONYM_MAP:
        return _SYNONYM_MAP[lower]

    for synonym, key in _SYNONYM_MAP.items():
        if synonym in lower:
            return key
    for key in MOOD_TO_SEARCH_QUERY:
        if key in lower:
            return key

    logger.debug("Unknown emotion '%s'; defaulting to 'excitement'.", emotion)
    return "excitement"


def _energy_to_bpm(energy: str) -> float:
    """Convert a coarse energy string to a reasonable BPM estimate."""
    lower = energy.lower()
    if any(w in lower for w in ("high", "intense", "fast", "hyper", "extreme")):
        return 140.0
    if any(w in lower for w in ("low", "slow", "calm", "chill", "relaxed")):
        return 75.0
    return 110.0  # medium


# ---------------------------------------------------------------------------
# AI selection
# ---------------------------------------------------------------------------

def _ai_pick_best(
    candidates: list[dict],
    emotion: str,
    energy: str,
) -> MusicMatch:
    """
    Ask the AI to select the best track from *candidates* given the video
    moment's *emotion* and *energy*, then return a fully populated MusicMatch.
    """
    trimmed = candidates[:_BATCH_LIMIT]

    # Compact representation — avoids sending large blobs to the AI
    compact = [
        {
            "index":      i,
            "title":      c.get("title", ""),
            "artist":     c.get("artist", ""),
            "tags":       c.get("tags", ""),
            "source":     c.get("source", ""),
            "has_url":    bool(c.get("url") or c.get("preview_url")),
            "has_local":  bool(c.get("local_path")),
            "duration_s": c.get("duration", 0),
        }
        for i, c in enumerate(trimmed)
    ]

    system_prompt = (
        "You are a music supervisor specialising in short-form viral video content. "
        "Your job is to select the single royalty-free track that best amplifies the "
        "emotional impact of a video moment. Consider energy level, mood alignment, "
        "tempo, and how well the track's tags match the required vibe."
    )

    user_prompt = (
        f"Video moment details:\n"
        f"  Dominant emotion : {emotion}\n"
        f"  Energy level     : {energy}\n\n"
        f"Candidate tracks (JSON):\n"
        f"{json.dumps(compact, indent=2)}\n\n"
        "Pick the SINGLE best track. "
        "Return a JSON object with these exact keys:\n"
        "  index         (integer — the candidate's index from the list above)\n"
        "  bpm_estimate  (float   — your estimate based on title/tags/energy)\n"
        "  why_it_fits   (string  — 1-2 sentences explaining the choice)\n"
    )

    schema = {
        "type": "object",
        "properties": {
            "index":        {"type": "integer"},
            "bpm_estimate": {"type": "number"},
            "why_it_fits":  {"type": "string"},
        },
        "required": ["index", "bpm_estimate", "why_it_fits"],
    }

    try:
        result: dict = router.complete_json(
            system=system_prompt,
            user=user_prompt,
            schema=schema,
            model=cfg.get("model", "openai/gpt-4o-mini"),
        )
    except Exception as exc:
        logger.warning("AI music pick failed (%s); using deterministic fallback.", exc)
        result = _fallback_result(trimmed, emotion, energy)

    idx = max(0, min(int(result.get("index", 0)), len(trimmed) - 1))
    chosen = trimmed[idx]

    return MusicMatch(
        title=chosen.get("title", "Unknown Track"),
        artist=chosen.get("artist", "Unknown Artist"),
        mood=_normalise_emotion(emotion),
        bpm_estimate=float(result.get("bpm_estimate", _energy_to_bpm(energy))),
        why_it_fits=result.get("why_it_fits", "Best available match for this mood."),
        url=chosen.get("url") or chosen.get("preview_url", ""),
        local_path=chosen.get("local_path", ""),
        royalty_free=True,
    )


def _fallback_result(candidates: list[dict], emotion: str, energy: str) -> dict:
    """
    Deterministic fallback used when the AI call fails.
    Prefers local files over API results; falls back to the first candidate.
    """
    local = [c for c in candidates if c.get("source") == "local"]
    chosen = (local or candidates)[0]
    return {
        "index":        candidates.index(chosen),
        "bpm_estimate": _energy_to_bpm(energy),
        "why_it_fits": (
            f"Deterministic fallback: best available match for '{emotion}' mood."
        ),
    }
