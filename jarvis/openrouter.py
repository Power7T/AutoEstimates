"""
OpenRouter client — routes tasks to the cheapest capable model.

Usage:
    from jarvis.openrouter import router

    text = await router.complete(prompt, model=cfg.models.planner)
    text = await router.vision(prompt, images_b64, model=cfg.models.vision)
"""

import asyncio
import base64
import json
from pathlib import Path
from typing import Any

from openai import OpenAI

from .config import cfg

OPENROUTER_BASE = "https://openrouter.ai/api/v1"


class OpenRouterClient:
    def __init__(self):
        self._client = OpenAI(
            api_key=cfg.openrouter_api_key,
            base_url=OPENROUTER_BASE,
        )

    # ------------------------------------------------------------------
    # Text completion
    # ------------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> str:
        model = model or cfg.models.planner
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_tokens}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content.strip()

    # ------------------------------------------------------------------
    # Vision (image + text)
    # ------------------------------------------------------------------

    def vision(
        self,
        prompt: str,
        images_b64: list[str],
        model: str | None = None,
        max_tokens: int = 1024,
        system: str = "",
    ) -> str:
        model = model or cfg.models.vision
        content: list[dict] = []

        if system:
            content.append({"type": "text", "text": system})

        content.append({"type": "text", "text": prompt})

        for b64 in images_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

        resp = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()

    # ------------------------------------------------------------------
    # Structured JSON output
    # ------------------------------------------------------------------

    def complete_json(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> dict:
        raw = self.complete(
            prompt=prompt,
            system=system,
            model=model,
            max_tokens=max_tokens,
            json_mode=True,
        )
        # Strip markdown fences if present
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)

    # ------------------------------------------------------------------
    # Vision JSON output
    # ------------------------------------------------------------------

    def vision_json(
        self,
        prompt: str,
        images_b64: list[str],
        model: str | None = None,
        max_tokens: int = 512,
        system: str = "",
    ) -> dict:
        full_prompt = prompt + "\n\nRespond ONLY with a valid JSON object. No explanation."
        raw = self.vision(full_prompt, images_b64, model=model, max_tokens=max_tokens, system=system)
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}


router = OpenRouterClient()
