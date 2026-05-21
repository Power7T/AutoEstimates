"""
Viral editing templates — proven structures for different platforms and styles.

Each template defines the pacing, energy arc, and editing decisions
that make content perform well on a specific platform.
"""

from dataclasses import dataclass, field


@dataclass
class TemplateSegment:
    name: str                    # e.g. "hook", "build", "drop", "outro"
    duration_ratio: float        # Fraction of total video length
    energy: str                  # low | medium | high | peak
    cut_frequency: str           # slow | medium | fast | very_fast
    effects: list[str]           # Suggested effects for this segment
    notes: str                   # Editorial guidance


@dataclass
class EditingTemplate:
    name: str
    platform: str                # instagram | tiktok | youtube_shorts | universal
    style: str                   # cinematic | vlog | music_video | documentary | hype
    ideal_duration_seconds: tuple[int, int]  # (min, max)
    aspect_ratio: str
    segments: list[TemplateSegment]
    music_mood: str
    color_grade: str
    caption_style: str           # clean | bold | kinetic | subtitle
    description: str


TEMPLATES: dict[str, EditingTemplate] = {

    "instagram_reel_cinematic": EditingTemplate(
        name="Instagram Reel — Cinematic",
        platform="instagram",
        style="cinematic",
        ideal_duration_seconds=(25, 45),
        aspect_ratio="9:16",
        segments=[
            TemplateSegment(
                name="hook",
                duration_ratio=0.10,
                energy="peak",
                cut_frequency="fast",
                effects=["zoom_in", "light_leak"],
                notes="Best single moment from the entire video. No context needed. Just impact.",
            ),
            TemplateSegment(
                name="build",
                duration_ratio=0.45,
                energy="medium",
                cut_frequency="medium",
                effects=["ken_burns", "cinematic_filter"],
                notes="Tell the story. Slow builds. Let shots breathe. One cut per 2-3 seconds.",
            ),
            TemplateSegment(
                name="drop",
                duration_ratio=0.30,
                energy="peak",
                cut_frequency="very_fast",
                effects=["speed_ramp", "beat_sync", "glitch"],
                notes="Music drop hits. Speed ramp into it. Fast cuts every beat. High energy.",
            ),
            TemplateSegment(
                name="outro",
                duration_ratio=0.15,
                energy="low",
                cut_frequency="slow",
                effects=["fade_out", "vignette"],
                notes="Satisfying resolution. Wide shot. Let it breathe.",
            ),
        ],
        music_mood="cinematic epic or emotional",
        color_grade="orange_teal",
        caption_style="clean",
        description="Cinematic storytelling reel optimized for Instagram. Orange/teal grade. Epic music.",
    ),

    "tiktok_hype": EditingTemplate(
        name="TikTok Hype",
        platform="tiktok",
        style="hype",
        ideal_duration_seconds=(15, 30),
        aspect_ratio="9:16",
        segments=[
            TemplateSegment(
                name="hook",
                duration_ratio=0.12,
                energy="peak",
                cut_frequency="very_fast",
                effects=["zoom_punch", "flash"],
                notes="First 2 seconds must be insane. Most energetic frame. No intro.",
            ),
            TemplateSegment(
                name="main",
                duration_ratio=0.65,
                energy="high",
                cut_frequency="fast",
                effects=["beat_sync", "transitions", "speed_ramp"],
                notes="Every cut on a beat. Keep energy constant. No slow moments.",
            ),
            TemplateSegment(
                name="payoff",
                duration_ratio=0.15,
                energy="peak",
                cut_frequency="very_fast",
                effects=["glitch", "flash", "freeze_frame"],
                notes="Climax moment. Freeze frame on the best shot.",
            ),
            TemplateSegment(
                name="cta",
                duration_ratio=0.08,
                energy="medium",
                cut_frequency="slow",
                effects=["text_overlay"],
                notes="Simple text CTA. Follow for more / Link in bio.",
            ),
        ],
        music_mood="trending hype or trap beat",
        color_grade="high_contrast",
        caption_style="bold",
        description="Maximum energy TikTok content. Beat-synced cuts. Bold captions. Trending sound.",
    ),

    "youtube_short_storytelling": EditingTemplate(
        name="YouTube Short — Storytelling",
        platform="youtube_shorts",
        style="documentary",
        ideal_duration_seconds=(45, 58),
        aspect_ratio="9:16",
        segments=[
            TemplateSegment(
                name="hook",
                duration_ratio=0.06,
                energy="high",
                cut_frequency="fast",
                effects=["text_hook", "zoom_in"],
                notes="Bold text question or shocking statement. 3 seconds max.",
            ),
            TemplateSegment(
                name="context",
                duration_ratio=0.20,
                energy="medium",
                cut_frequency="medium",
                effects=["auto_captions", "ken_burns"],
                notes="Set up the story. Who, what, where. Auto-captions essential.",
            ),
            TemplateSegment(
                name="conflict",
                duration_ratio=0.35,
                energy="high",
                cut_frequency="medium",
                effects=["auto_captions", "zoom_in", "tension_music"],
                notes="Build tension. The problem or challenge. Keep viewer curious.",
            ),
            TemplateSegment(
                name="resolution",
                duration_ratio=0.25,
                energy="peak",
                cut_frequency="fast",
                effects=["auto_captions", "reveal_transition"],
                notes="The payoff. Answer the hook question. Satisfying conclusion.",
            ),
            TemplateSegment(
                name="cta",
                duration_ratio=0.14,
                energy="medium",
                cut_frequency="slow",
                effects=["text_overlay", "subscribe_animation"],
                notes="Subscribe / follow prompt. Keep it short.",
            ),
        ],
        music_mood="background atmospheric or tension building",
        color_grade="warm",
        caption_style="subtitle",
        description="Story-driven YouTube Short with auto-captions. Hook → Tension → Payoff structure.",
    ),

    "travel_vlog": EditingTemplate(
        name="Travel Vlog",
        platform="universal",
        style="vlog",
        ideal_duration_seconds=(30, 60),
        aspect_ratio="16:9",
        segments=[
            TemplateSegment(
                name="destination_reveal",
                duration_ratio=0.15,
                energy="high",
                cut_frequency="fast",
                effects=["ken_burns", "whip_pan", "title_text"],
                notes="Reveal the destination with wide drone or landscape shot + location title.",
            ),
            TemplateSegment(
                name="exploration",
                duration_ratio=0.55,
                energy="medium",
                cut_frequency="medium",
                effects=["warm_filter", "ken_burns", "transitions"],
                notes="Show the place. Mix wide shots with close details. Warm color grade.",
            ),
            TemplateSegment(
                name="highlight_moment",
                duration_ratio=0.20,
                energy="peak",
                cut_frequency="fast",
                effects=["speed_ramp", "beat_sync", "light_leak"],
                notes="The best moment of the trip. Slow mo into it, speed ramp out.",
            ),
            TemplateSegment(
                name="sign_off",
                duration_ratio=0.10,
                energy="low",
                cut_frequency="slow",
                effects=["sunset_filter", "fade_out"],
                notes="Peaceful closing shot. Golden hour if available.",
            ),
        ],
        music_mood="uplifting acoustic or chill indie",
        color_grade="warm",
        caption_style="clean",
        description="Warm, aspirational travel vlog. Wide shots, warm grade, uplifting music.",
    ),

    "product_showcase": EditingTemplate(
        name="Product Showcase",
        platform="instagram",
        style="cinematic",
        ideal_duration_seconds=(20, 40),
        aspect_ratio="9:16",
        segments=[
            TemplateSegment(
                name="hook",
                duration_ratio=0.10,
                energy="peak",
                cut_frequency="fast",
                effects=["zoom_in", "flash"],
                notes="Product reveal. Don't show full product yet. Tease it.",
            ),
            TemplateSegment(
                name="problem",
                duration_ratio=0.20,
                energy="medium",
                cut_frequency="medium",
                effects=["auto_captions", "moody_filter"],
                notes="Pain point the product solves. Relatable situation.",
            ),
            TemplateSegment(
                name="solution",
                duration_ratio=0.45,
                energy="high",
                cut_frequency="fast",
                effects=["clean_filter", "product_zoom", "feature_text"],
                notes="Show the product solving the problem. Key features as text overlays.",
            ),
            TemplateSegment(
                name="proof",
                duration_ratio=0.15,
                energy="medium",
                cut_frequency="slow",
                effects=["testimonial_style", "warm_filter"],
                notes="Results or social proof moment.",
            ),
            TemplateSegment(
                name="cta",
                duration_ratio=0.10,
                energy="high",
                cut_frequency="medium",
                effects=["bold_text", "price_reveal"],
                notes="Clear CTA with offer. Link in bio / Shop now.",
            ),
        ],
        music_mood="modern upbeat or brand-appropriate",
        color_grade="bright",
        caption_style="bold",
        description="High-converting product showcase. Problem → Solution → Proof → CTA.",
    ),

    "music_video": EditingTemplate(
        name="Music Video",
        platform="universal",
        style="music_video",
        ideal_duration_seconds=(60, 180),
        aspect_ratio="16:9",
        segments=[
            TemplateSegment(
                name="intro",
                duration_ratio=0.10,
                energy="low",
                cut_frequency="slow",
                effects=["film_grain", "vignette", "slow_mo"],
                notes="Atmospheric intro. Establish mood before beat drops.",
            ),
            TemplateSegment(
                name="verse_1",
                duration_ratio=0.25,
                energy="medium",
                cut_frequency="medium",
                effects=["beat_sync", "cinematic_filter"],
                notes="Tell the story. Cuts on lyrics. Medium energy.",
            ),
            TemplateSegment(
                name="chorus",
                duration_ratio=0.20,
                energy="peak",
                cut_frequency="very_fast",
                effects=["beat_sync", "speed_ramp", "light_leak", "glitch"],
                notes="Maximum energy. Every beat = cut. Speed ramps. Visual effects.",
            ),
            TemplateSegment(
                name="verse_2",
                duration_ratio=0.20,
                energy="medium",
                cut_frequency="medium",
                effects=["beat_sync", "color_shift"],
                notes="New visual palette. Build anticipation for next chorus.",
            ),
            TemplateSegment(
                name="final_chorus",
                duration_ratio=0.20,
                energy="peak",
                cut_frequency="very_fast",
                effects=["beat_sync", "triple_speed_ramp", "flash", "glitch"],
                notes="Biggest energy moment. More effects than first chorus.",
            ),
            TemplateSegment(
                name="outro",
                duration_ratio=0.05,
                energy="low",
                cut_frequency="slow",
                effects=["fade_to_black"],
                notes="Quick, satisfying fade out.",
            ),
        ],
        music_mood="matches the actual music track",
        color_grade="moody",
        caption_style="kinetic",
        description="Full music video structure. Beat-synced cuts. Cinematic color. Chorus explosions.",
    ),
}


def select_template(
    clips_mood: str,
    platform: str,
    total_duration: float,
    has_speech: bool,
) -> EditingTemplate:
    """
    Automatically pick the best template based on content analysis.
    """
    mood_to_style = {
        "energetic": "tiktok_hype",
        "happy": "travel_vlog",
        "calm": "travel_vlog",
        "melancholic": "youtube_short_storytelling",
        "tense": "youtube_short_storytelling",
        "romantic": "instagram_reel_cinematic",
        "neutral": "instagram_reel_cinematic",
    }

    # Platform overrides
    if platform == "tiktok":
        return TEMPLATES["tiktok_hype"]
    if platform == "youtube":
        return TEMPLATES["youtube_short_storytelling"]

    # Duration-based selection
    if total_duration > 90:
        return TEMPLATES["music_video"]

    # Mood-based selection
    template_key = mood_to_style.get(clips_mood, "instagram_reel_cinematic")
    return TEMPLATES[template_key]
