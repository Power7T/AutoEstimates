"""
Personal taste learning — Jarvis learns your editing preferences over time.

After each edit, the user rates decisions (1-5).
After a few videos, Jarvis edits exactly like you would.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

PROFILES_DIR = Path("data/taste_profiles")


@dataclass
class EditDecision:
    decision_type: str   # filter | transition | music | speed | effect | color_grade
    value: str           # What was chosen (e.g. "cinematic", "fade", "lofi hip hop")
    context: str         # Why it was chosen
    rating: float = -1   # -1 = not yet rated, 1.0–5.0 = user rating


@dataclass
class TasteProfile:
    user_id: str
    videos_edited: int = 0
    decisions: list[EditDecision] = field(default_factory=list)
    # Aggregated preferences (updated after each rating)
    preferred_filters: list[str] = field(default_factory=list)
    preferred_transitions: list[str] = field(default_factory=list)
    preferred_music_moods: list[str] = field(default_factory=list)
    preferred_effects: list[str] = field(default_factory=list)
    preferred_pace: str = "medium"          # slow | medium | fast
    preferred_color_grade: str = "cinematic"
    preferred_caption_style: str = "clean"
    disliked: list[str] = field(default_factory=list)


class TasteLearner:
    """Tracks editing decisions and user ratings to build a personal taste profile."""

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.profile = self._load_profile()

    # ------------------------------------------------------------------
    # Recording decisions
    # ------------------------------------------------------------------

    def record_decision(self, decision_type: str, value: str, context: str = "") -> str:
        """Record an editing decision made by Jarvis. Returns decision ID."""
        decision = EditDecision(
            decision_type=decision_type,
            value=value,
            context=context,
        )
        self.profile.decisions.append(decision)
        decision_id = f"{len(self.profile.decisions) - 1}"
        self._save_profile()
        return decision_id

    def rate_decision(self, decision_id: str, rating: float):
        """User rates a decision 1.0–5.0."""
        idx = int(decision_id)
        if 0 <= idx < len(self.profile.decisions):
            self.profile.decisions[idx].rating = max(1.0, min(5.0, rating))
            self._update_preferences()
            self._save_profile()

    def rate_video(self, overall_rating: float):
        """Rate the entire video edit 1-5. Applies to all unrated decisions of this video."""
        for d in self.profile.decisions:
            if d.rating == -1:
                d.rating = overall_rating
        self.profile.videos_edited += 1
        self._update_preferences()
        self._save_profile()

    # ------------------------------------------------------------------
    # Reading preferences
    # ------------------------------------------------------------------

    def get_preferred_filter(self, fallback: str = "cinematic") -> str:
        return self.profile.preferred_filters[0] if self.profile.preferred_filters else fallback

    def get_preferred_transition(self, fallback: str = "fade") -> str:
        return self.profile.preferred_transitions[0] if self.profile.preferred_transitions else fallback

    def get_preferred_music_mood(self, fallback: str = "cinematic") -> str:
        return self.profile.preferred_music_moods[0] if self.profile.preferred_music_moods else fallback

    def get_preferred_pace(self) -> str:
        return self.profile.preferred_pace

    def get_preferred_color_grade(self) -> str:
        return self.profile.preferred_color_grade

    def get_profile_summary(self) -> str:
        if self.profile.videos_edited < 2:
            return "No taste profile yet — rate more videos to personalize Jarvis."
        return (
            f"Videos edited: {self.profile.videos_edited} | "
            f"Preferred filter: {self.profile.preferred_color_grade} | "
            f"Preferred pace: {self.profile.preferred_pace} | "
            f"Preferred transition: {self.profile.preferred_transitions[:2]} | "
            f"Preferred music: {self.profile.preferred_music_moods[:2]}"
        )

    def is_trained(self) -> bool:
        """Returns True if enough data exists to meaningfully personalize edits."""
        return self.profile.videos_edited >= 3

    # ------------------------------------------------------------------
    # Preference aggregation
    # ------------------------------------------------------------------

    def _update_preferences(self):
        """Recalculate preferred choices from all rated decisions."""
        rated = [d for d in self.profile.decisions if d.rating >= 1]
        if not rated:
            return

        # Group by decision type
        by_type: dict[str, list[EditDecision]] = {}
        for d in rated:
            by_type.setdefault(d.decision_type, []).append(d)

        def top_choices(decisions: list[EditDecision], n: int = 3) -> list[str]:
            """Return top-rated values, weighted by rating."""
            scores: dict[str, list[float]] = {}
            for d in decisions:
                scores.setdefault(d.value, []).append(d.rating)
            avg = {v: sum(r) / len(r) for v, r in scores.items()}
            return [v for v, _ in sorted(avg.items(), key=lambda x: -x[1])][:n]

        def disliked(decisions: list[EditDecision]) -> list[str]:
            return [d.value for d in decisions if d.rating <= 2.0]

        if "filter" in by_type:
            self.profile.preferred_filters = top_choices(by_type["filter"])
        if "color_grade" in by_type:
            self.profile.preferred_color_grade = top_choices(by_type["color_grade"], 1)[0] if top_choices(by_type["color_grade"], 1) else "cinematic"
        if "transition" in by_type:
            self.profile.preferred_transitions = top_choices(by_type["transition"])
        if "music" in by_type:
            self.profile.preferred_music_moods = top_choices(by_type["music"])
        if "effect" in by_type:
            self.profile.preferred_effects = top_choices(by_type["effect"])
        if "pace" in by_type:
            paces = top_choices(by_type["pace"], 1)
            self.profile.preferred_pace = paces[0] if paces else "medium"

        # Collect dislikes across all types
        all_disliked = []
        for decisions in by_type.values():
            all_disliked.extend(disliked(decisions))
        self.profile.disliked = list(set(all_disliked))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_profile(self) -> TasteProfile:
        path = PROFILES_DIR / f"{self.user_id}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text())
                decisions = [EditDecision(**d) for d in data.pop("decisions", [])]
                profile = TasteProfile(**data)
                profile.decisions = decisions
                return profile
            except Exception:
                pass
        return TasteProfile(user_id=self.user_id)

    def _save_profile(self):
        PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        path = PROFILES_DIR / f"{self.user_id}.json"
        data = {
            "user_id": self.profile.user_id,
            "videos_edited": self.profile.videos_edited,
            "decisions": [vars(d) for d in self.profile.decisions],
            "preferred_filters": self.profile.preferred_filters,
            "preferred_transitions": self.profile.preferred_transitions,
            "preferred_music_moods": self.profile.preferred_music_moods,
            "preferred_effects": self.profile.preferred_effects,
            "preferred_pace": self.profile.preferred_pace,
            "preferred_color_grade": self.profile.preferred_color_grade,
            "preferred_caption_style": self.profile.preferred_caption_style,
            "disliked": self.profile.disliked,
        }
        path.write_text(json.dumps(data, indent=2))
