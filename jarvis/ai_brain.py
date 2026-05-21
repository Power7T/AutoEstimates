"""
Jarvis Assistant Mode — natural language command execution via OpenRouter + tools.
"""

import json
from typing import Any

from openai import OpenAI

from .config import cfg
from .vision_controller import VisionController

CAPCUT_URL = "https://www.capcut.com"

ASSISTANT_SYSTEM = """You are Jarvis, an expert AI video editor controlling CapCut's web app.
You execute video editing commands through a set of tools.

Guidelines:
- Confirm what you're about to do before doing it
- Chain multiple tool calls when a request implies several steps
- After executing, describe what was done clearly
- Use creative judgment when instructions are vague (e.g. "make it cinematic" = cinematic filter + vignette + film grain)
- If something is ambiguous, ask ONE clarifying question
- Always verify the user is happy with the result"""

TOOLS = [
    {"type": "function", "function": {
        "name": "open_project",
        "description": "Open or create a CapCut project by name.",
        "parameters": {"type": "object", "properties": {
            "project_name": {"type": "string"},
            "create_new": {"type": "boolean", "default": False},
        }, "required": ["project_name"]},
    }},
    {"type": "function", "function": {
        "name": "import_media",
        "description": "Import a video or image file into the current project.",
        "parameters": {"type": "object", "properties": {
            "file_path": {"type": "string"},
        }, "required": ["file_path"]},
    }},
    {"type": "function", "function": {
        "name": "trim_clip",
        "description": "Trim a clip to new in/out points.",
        "parameters": {"type": "object", "properties": {
            "clip_index": {"type": "integer", "default": 0},
            "start_seconds": {"type": "number"},
            "end_seconds": {"type": "number"},
        }, "required": ["start_seconds", "end_seconds"]},
    }},
    {"type": "function", "function": {
        "name": "add_text",
        "description": "Add a text overlay at a position and time range.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string"},
            "position": {"type": "string", "enum": ["top", "center", "bottom"], "default": "bottom"},
            "start_seconds": {"type": "number", "default": 0},
            "duration_seconds": {"type": "number", "default": 3},
            "font_size": {"type": "string", "enum": ["small", "medium", "large"], "default": "medium"},
            "color": {"type": "string", "default": "white"},
            "animated": {"type": "boolean", "default": False},
            "style": {"type": "string", "enum": ["clean", "bold", "kinetic"], "default": "clean"},
        }, "required": ["text"]},
    }},
    {"type": "function", "function": {
        "name": "add_auto_captions",
        "description": "Auto-generate captions/subtitles from speech in the video.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "add_transition",
        "description": "Add a transition between clips.",
        "parameters": {"type": "object", "properties": {
            "clip_index": {"type": "integer", "default": 0},
            "transition_type": {"type": "string", "enum": [
                "fade", "dissolve", "wipe_left", "wipe_right", "zoom_in", "zoom_out",
                "flash", "glitch", "whip_pan", "morph", "light_leak", "none",
            ], "default": "fade"},
            "duration_seconds": {"type": "number", "default": 0.5},
        }, "required": ["transition_type"]},
    }},
    {"type": "function", "function": {
        "name": "add_music",
        "description": "Add background music from CapCut library.",
        "parameters": {"type": "object", "properties": {
            "query_or_path": {"type": "string"},
            "volume": {"type": "number", "default": 0.5},
        }, "required": ["query_or_path"]},
    }},
    {"type": "function", "function": {
        "name": "apply_filter",
        "description": "Apply a color filter to the video.",
        "parameters": {"type": "object", "properties": {
            "filter_name": {"type": "string"},
            "intensity": {"type": "number", "default": 0.7},
        }, "required": ["filter_name"]},
    }},
    {"type": "function", "function": {
        "name": "apply_lut",
        "description": "Apply a cinematic LUT color grade (orange_teal, warm, cool, moody, high_contrast, vintage, bright).",
        "parameters": {"type": "object", "properties": {
            "lut_style": {"type": "string"},
        }, "required": ["lut_style"]},
    }},
    {"type": "function", "function": {
        "name": "adjust_speed",
        "description": "Change clip playback speed (0.5 = slow mo, 2.0 = double speed).",
        "parameters": {"type": "object", "properties": {
            "clip_index": {"type": "integer", "default": 0},
            "speed_multiplier": {"type": "number"},
        }, "required": ["speed_multiplier"]},
    }},
    {"type": "function", "function": {
        "name": "apply_speed_ramp",
        "description": "Apply cinematic speed ramp (slow then fast at music drop).",
        "parameters": {"type": "object", "properties": {
            "clip_index": {"type": "integer", "default": 0},
            "ramp_type": {"type": "string", "enum": ["drop", "slow_mo"], "default": "drop"},
        }, "required": []},
    }},
    {"type": "function", "function": {
        "name": "adjust_volume",
        "description": "Adjust clip audio volume (0.0 = mute, 1.0 = normal).",
        "parameters": {"type": "object", "properties": {
            "clip_index": {"type": "integer", "default": 0},
            "volume": {"type": "number"},
        }, "required": ["volume"]},
    }},
    {"type": "function", "function": {
        "name": "add_sticker",
        "description": "Add an animated sticker overlay.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
        }, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "add_effect",
        "description": "Add a visual effect (glitch, light_leak, particles, chromatic_aberration, etc.).",
        "parameters": {"type": "object", "properties": {
            "effect_name": {"type": "string"},
        }, "required": ["effect_name"]},
    }},
    {"type": "function", "function": {
        "name": "apply_ken_burns",
        "description": "Apply Ken Burns zoom/pan animation to a clip.",
        "parameters": {"type": "object", "properties": {
            "clip_index": {"type": "integer", "default": 0},
            "direction": {"type": "string", "enum": ["zoom_in", "zoom_out", "pan_left", "pan_right"], "default": "zoom_in"},
        }, "required": []},
    }},
    {"type": "function", "function": {
        "name": "remove_background",
        "description": "AI background removal for a clip.",
        "parameters": {"type": "object", "properties": {
            "clip_index": {"type": "integer", "default": 0},
        }, "required": []},
    }},
    {"type": "function", "function": {
        "name": "stabilize_clip",
        "description": "Stabilize shaky footage.",
        "parameters": {"type": "object", "properties": {
            "clip_index": {"type": "integer", "default": 0},
        }, "required": []},
    }},
    {"type": "function", "function": {
        "name": "auto_enhance",
        "description": "AI auto-enhance brightness, contrast, and color of a clip.",
        "parameters": {"type": "object", "properties": {
            "clip_index": {"type": "integer", "default": 0},
        }, "required": []},
    }},
    {"type": "function", "function": {
        "name": "crop_video",
        "description": "Crop to a specific aspect ratio (9:16, 16:9, 1:1, 4:3).",
        "parameters": {"type": "object", "properties": {
            "aspect_ratio": {"type": "string", "enum": ["9:16", "16:9", "1:1", "4:3", "3:4"]},
        }, "required": ["aspect_ratio"]},
    }},
    {"type": "function", "function": {
        "name": "add_vignette",
        "description": "Add vignette darkening around the edges.",
        "parameters": {"type": "object", "properties": {
            "intensity": {"type": "number", "default": 0.5},
        }, "required": []},
    }},
    {"type": "function", "function": {
        "name": "add_film_grain",
        "description": "Add cinematic film grain texture.",
        "parameters": {"type": "object", "properties": {
            "intensity": {"type": "number", "default": 0.3},
        }, "required": []},
    }},
    {"type": "function", "function": {
        "name": "add_light_leak",
        "description": "Add a light leak / lens flare overlay for cinematic feel.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "reverse_clip",
        "description": "Reverse a clip to play backwards.",
        "parameters": {"type": "object", "properties": {
            "clip_index": {"type": "integer", "default": 0},
        }, "required": []},
    }},
    {"type": "function", "function": {
        "name": "freeze_frame",
        "description": "Freeze a specific frame for a duration.",
        "parameters": {"type": "object", "properties": {
            "clip_index": {"type": "integer", "default": 0},
            "at_seconds": {"type": "number", "default": 0},
            "duration": {"type": "number", "default": 1.0},
        }, "required": []},
    }},
    {"type": "function", "function": {
        "name": "mirror_clip",
        "description": "Mirror/flip a clip horizontally or vertically.",
        "parameters": {"type": "object", "properties": {
            "clip_index": {"type": "integer", "default": 0},
            "direction": {"type": "string", "enum": ["horizontal", "vertical"], "default": "horizontal"},
        }, "required": []},
    }},
    {"type": "function", "function": {
        "name": "reduce_noise",
        "description": "Reduce background audio noise from a clip.",
        "parameters": {"type": "object", "properties": {
            "clip_index": {"type": "integer", "default": 0},
        }, "required": []},
    }},
    {"type": "function", "function": {
        "name": "duck_audio",
        "description": "Auto-duck clip audio under background music.",
        "parameters": {"type": "object", "properties": {
            "duck_level": {"type": "number", "default": 0.3},
        }, "required": []},
    }},
    {"type": "function", "function": {
        "name": "split_screen",
        "description": "Apply split screen layout.",
        "parameters": {"type": "object", "properties": {
            "layout": {"type": "string", "enum": ["side_by_side", "top_bottom", "quad"], "default": "side_by_side"},
        }, "required": []},
    }},
    {"type": "function", "function": {
        "name": "export_video",
        "description": "Export the final video.",
        "parameters": {"type": "object", "properties": {
            "resolution": {"type": "string", "enum": ["720p", "1080p", "2k", "4k"], "default": "1080p"},
            "fps": {"type": "integer", "enum": [24, 30, 60], "default": 30},
            "format": {"type": "string", "enum": ["mp4", "mov"], "default": "mp4"},
        }, "required": []},
    }},
    {"type": "function", "function": {
        "name": "undo",
        "description": "Undo recent actions.",
        "parameters": {"type": "object", "properties": {
            "steps": {"type": "integer", "default": 1},
        }, "required": []},
    }},
    {"type": "function", "function": {
        "name": "get_timeline_info",
        "description": "Get information about the current timeline state.",
        "parameters": {"type": "object", "properties": {}},
    }},
]


class JarvisAssistant:
    """Claude-powered natural language video editing assistant."""

    def __init__(self, controller: VisionController, console=None):
        self.controller = controller
        self.console = console
        self._client = OpenAI(
            api_key=cfg.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self._history: list[dict] = []

    def _log(self, msg: str):
        if self.console:
            self.console.print(f"[cyan]Jarvis ▸[/cyan] {msg}")
        else:
            print(f"Jarvis ▸ {msg}")

    async def chat(self, user_message: str) -> str:
        self._history.append({"role": "user", "content": user_message})

        while True:
            resp = self._client.chat.completions.create(
                model=cfg.models.director,
                max_tokens=2048,
                messages=[{"role": "system", "content": ASSISTANT_SYSTEM}] + self._history,
                tools=TOOLS,
                tool_choice="auto",
            )

            choice = resp.choices[0]
            message = choice.message

            # Append assistant turn to history
            self._history.append({"role": "assistant", "content": message.content or "", "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in (message.tool_calls or [])
            ]})

            if not message.tool_calls or choice.finish_reason == "stop":
                return message.content or ""

            # Execute tool calls
            for tc in message.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)
                self._log(f"Executing: {name}({self._fmt(args)})")
                result = await self.controller.execute_tool(name, args)
                self._history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                })

    def reset(self):
        self._history = []

    @staticmethod
    def _fmt(args: dict) -> str:
        parts = []
        for k, v in args.items():
            sv = str(v)
            if len(sv) > 30:
                sv = sv[:27] + "..."
            parts.append(f"{k}={sv!r}")
        return ", ".join(parts)
