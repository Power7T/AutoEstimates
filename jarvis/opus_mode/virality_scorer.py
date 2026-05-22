"""
Virality scorer — rates every clip 0-100 across 6 dimensions
and gives actionable improvement suggestions.

Dimensions (matching what actually drives performance):
  1. Hook strength     — will this stop the scroll?
  2. Emotional impact  — does it make you feel something?
  3. Quotability       — would someone screenshot this?
  4. Pacing            — is the energy right for the platform?
  5. Completeness      — does it feel satisfying on its own?
  6. Trend alignment   — does it match what's working right now?
"""

from dataclasses import dataclass
from ..openrouter import router
from ..config import cfg
from .moment_finder import ViralMoment


@dataclass
class ViralityReport:
    overall_score: float            # 0-100
    hook_score: float
    emotion_score: float
    quotability_score: float
    pacing_score: float
    completeness_score: float
    trend_score: float
    grade: str                      # S / A / B / C / D
    verdict: str                    # one-line summary
    improvements: list[str]         # actionable things to fix
    predicted_views_range: str      # "10K-50K", "100K-500K", etc.
    best_platform: str              # tiktok | instagram | youtube_shorts


SCORER_SYSTEM = """You are a viral content analyst who has studied the performance
metrics of 10 million short-form videos across TikTok, Instagram, and YouTube Shorts.
You can accurately predict whether a clip will go viral based on specific criteria.
Return only valid JSON."""

SCORER_PROMPT = """Score this video clip for viral potential across 6 dimensions.

CLIP TITLE: {title}
DURATION: {duration:.0f} seconds
EMOTION: {emotion}
CLIP TYPE: {clip_type}
HOOK SENTENCE: {hook}
TRANSCRIPT EXCERPT:
{excerpt}

Score each dimension 0-100 based on these strict criteria:

hook_score: Will this specific hook make someone stop mid-scroll?
  90-100: Irresistible — creates immediate NEED to keep watching
  70-89:  Strong — most people will pause
  50-69:  Decent — some will stop, most won't
  0-49:   Weak — most will scroll past

emotion_score: Does this make you feel something strong?
  90-100: Hits hard — laugh out loud, gasp, tears, anger
  70-89:  Clear emotion — noticeable feeling
  50-69:  Mild feeling
  0-49:   Flat — no real emotional response

quotability_score: Would someone screenshot or share this quote/moment?
  90-100: Screenshot-worthy — memorable phrase, surprising stat, perfect insight
  70-89:  Shareable
  50-69:  Somewhat memorable
  0-49:   Forgettable

pacing_score: Is the energy and information density right?
  90-100: Perfect — every second matters, no dead air
  70-89:  Good flow
  50-69:  Some slow parts
  0-49:   Too slow or too rushed

completeness_score: Does it feel satisfying as a standalone clip?
  90-100: Perfect standalone — setup, payoff, no missing context
  70-89:  Works well alone
  50-69:  Needs some context
  0-49:   Confusing without watching more

trend_score: Does this match formats currently going viral?
  Score based on: controversy, storytelling, surprising facts, how-to, personal story

Return JSON:
{{
  "hook_score": 0-100,
  "emotion_score": 0-100,
  "quotability_score": 0-100,
  "pacing_score": 0-100,
  "completeness_score": 0-100,
  "trend_score": 0-100,
  "grade": "S|A|B|C|D",
  "verdict": "one punchy sentence about this clip's potential",
  "improvements": ["specific thing 1 to improve", "specific thing 2", "specific thing 3"],
  "predicted_views_range": "e.g. 10K-50K or 500K-2M",
  "best_platform": "tiktok|instagram|youtube_shorts"
}}"""


def score_moment(moment: ViralMoment, trends: dict | None = None) -> ViralityReport:
    """Score a single viral moment across all dimensions."""
    data = router.complete_json(
        prompt=SCORER_PROMPT.format(
            title=moment.title,
            duration=moment.duration,
            emotion=moment.emotion,
            clip_type=moment.clip_type,
            hook=moment.hook,
            excerpt=moment.transcript_excerpt[:800],
        ),
        system=SCORER_SYSTEM,
        model=cfg.models.planner,
        max_tokens=800,
    )

    h = float(data.get("hook_score", 50))
    e = float(data.get("emotion_score", 50))
    q = float(data.get("quotability_score", 50))
    p = float(data.get("pacing_score", 50))
    c = float(data.get("completeness_score", 50))
    t = float(data.get("trend_score", 50))

    # Weighted average — hook and emotion matter most
    overall = (h * 0.25 + e * 0.20 + q * 0.15 + p * 0.15 + c * 0.15 + t * 0.10)

    return ViralityReport(
        overall_score=round(overall, 1),
        hook_score=h,
        emotion_score=e,
        quotability_score=q,
        pacing_score=p,
        completeness_score=c,
        trend_score=t,
        grade=data.get("grade", _grade(overall)),
        verdict=data.get("verdict", ""),
        improvements=data.get("improvements", []),
        predicted_views_range=data.get("predicted_views_range", "Unknown"),
        best_platform=data.get("best_platform", "instagram"),
    )


def score_all(moments: list[ViralMoment]) -> list[tuple[ViralMoment, ViralityReport]]:
    """Score all moments and return sorted by overall score."""
    scored = []
    for moment in moments:
        try:
            report = score_moment(moment)
            scored.append((moment, report))
        except Exception as e:
            print(f"[Scorer] Failed to score '{moment.title}': {e}")
    scored.sort(key=lambda x: x[1].overall_score, reverse=True)
    return scored


def _grade(score: float) -> str:
    if score >= 85: return "S"
    if score >= 75: return "A"
    if score >= 60: return "B"
    if score >= 45: return "C"
    return "D"
