"""Central configuration loaded from environment variables."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Models:
    # Cheap + fast vision model (watches video frames)
    vision: str = "google/gemini-flash-1.5"
    # Good reasoning, cheap (edit planning)
    planner: str = "meta-llama/llama-3.1-70b-instruct"
    # Best quality (director-level creative decisions)
    director: str = "anthropic/claude-sonnet-4-6"


@dataclass
class Config:
    openrouter_api_key: str = ""
    capcut_email: str = ""
    capcut_password: str = ""
    headless: bool = True
    whisper_model: str = "base"
    enable_trends: bool = True
    enable_taste_learning: bool = True
    enable_beat_sync: bool = True
    models: Models = None

    def __post_init__(self):
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.capcut_email = os.getenv("CAPCUT_EMAIL", "")
        self.capcut_password = os.getenv("CAPCUT_PASSWORD", "")
        self.headless = os.getenv("HEADLESS", "true").lower() == "true"
        self.whisper_model = os.getenv("WHISPER_MODEL", "base")
        self.enable_trends = os.getenv("ENABLE_TRENDS", "true").lower() == "true"
        self.enable_taste_learning = os.getenv("ENABLE_TASTE_LEARNING", "true").lower() == "true"
        self.enable_beat_sync = os.getenv("ENABLE_BEAT_SYNC", "true").lower() == "true"
        self.models = Models(
            vision=os.getenv("VISION_MODEL", "google/gemini-flash-1.5"),
            planner=os.getenv("PLANNER_MODEL", "meta-llama/llama-3.1-70b-instruct"),
            director=os.getenv("DIRECTOR_MODEL", "anthropic/claude-sonnet-4-6"),
        )

    def validate(self):
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is not set. Add it to your .env file.")


cfg = Config()
