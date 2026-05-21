"""
Real-time trend awareness — scrapes and caches what's working right now.
Checks trending sounds, effects, color grades, and editing styles.
Cached daily so it doesn't hammer external sites.
"""

import json
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from .openrouter import router
from .config import cfg

CACHE_FILE = Path("data/trends/daily_trends.json")
CACHE_TTL_SECONDS = 86400  # 24 hours


def get_current_trends() -> dict:
    """
    Return current trending editing elements.
    Cached for 24 hours to avoid excessive scraping.
    """
    if _cache_is_fresh():
        return json.loads(CACHE_FILE.read_text())

    trends = _fetch_trends()
    _save_cache(trends)
    return trends


def get_trending_sounds(platform: str = "tiktok") -> list[str]:
    trends = get_current_trends()
    return trends.get("sounds", {}).get(platform, _default_sounds())


def get_trending_effects() -> list[str]:
    trends = get_current_trends()
    return trends.get("effects", _default_effects())


def get_trending_color_grade() -> str:
    trends = get_current_trends()
    grades = trends.get("color_grades", [])
    return grades[0] if grades else "cinematic"


def get_trending_caption_style() -> str:
    trends = get_current_trends()
    return trends.get("caption_style", "bold")


# ---------------------------------------------------------------------------
# Fetching and parsing
# ---------------------------------------------------------------------------

def _fetch_trends() -> dict:
    """Fetch trend data from public sources and synthesize with AI."""
    raw_data = {}

    # Try to get trending info from public sources
    sources = [
        ("https://www.tiktok.com/trending", "TikTok trending page"),
        ("https://creators.tiktok.com/creator-portal/en-us/inspiration/trending-discoveries/", "TikTok creator trends"),
    ]

    for url, label in sources:
        try:
            resp = requests.get(url, timeout=8, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
            if resp.ok:
                soup = BeautifulSoup(resp.text, "html.parser")
                raw_data[label] = soup.get_text()[:3000]
        except Exception:
            continue

    # Ask AI to synthesize trends from what we gathered + its training knowledge
    prompt = f"""Based on current social media trends (as of your knowledge cutoff) and this data:
{json.dumps(raw_data, indent=2)[:2000]}

What are the current top trending video editing styles, sounds, effects, and color grades
on TikTok, Instagram Reels, and YouTube Shorts?

Return JSON:
{{
  "sounds": {{
    "tiktok": ["sound1", "sound2", "sound3"],
    "instagram": ["sound1", "sound2", "sound3"]
  }},
  "effects": ["effect1", "effect2", "effect3", "effect4", "effect5"],
  "color_grades": ["grade1", "grade2", "grade3"],
  "caption_style": "bold|clean|kinetic|subtitle",
  "editing_styles": ["style1", "style2"],
  "trending_transitions": ["transition1", "transition2", "transition3"],
  "hook_formats": ["hook_type1", "hook_type2"],
  "summary": "2-sentence summary of current trends"
}}"""

    try:
        result = router.complete_json(
            prompt=prompt,
            model=cfg.models.planner,
        )
        return result
    except Exception:
        return _default_trends()


def _cache_is_fresh() -> bool:
    if not CACHE_FILE.exists():
        return False
    age = time.time() - CACHE_FILE.stat().st_mtime
    return age < CACHE_TTL_SECONDS


def _save_cache(data: dict):
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Fallback defaults
# ---------------------------------------------------------------------------

def _default_trends() -> dict:
    return {
        "sounds": {
            "tiktok": ["trending hip hop beat", "viral phonk", "emotional piano"],
            "instagram": ["cinematic orchestral", "chill lofi", "upbeat pop"],
        },
        "effects": ["speed_ramp", "glitch", "light_leak", "film_grain", "zoom_punch"],
        "color_grades": ["orange_teal", "moody_dark", "warm_vintage"],
        "caption_style": "bold",
        "editing_styles": ["beat_sync_cuts", "cinematic_slow_mo"],
        "trending_transitions": ["whip_pan", "zoom_transition", "glitch_cut"],
        "hook_formats": ["shocking_visual", "bold_text_question"],
        "summary": "Fast cuts, beat sync, and cinematic color grades dominate current trends.",
    }


def _default_sounds() -> list[str]:
    return ["trending beat", "viral sound", "cinematic orchestral"]


def _default_effects() -> list[str]:
    return ["speed_ramp", "glitch", "light_leak", "film_grain"]
