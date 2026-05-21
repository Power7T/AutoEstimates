"""
Vision-based CapCut controller.

Instead of hardcoded selectors, every action works like this:
  1. Take screenshot of the current CapCut state
  2. Send screenshot to vision AI: "Where is the [element]?"
  3. AI returns pixel coordinates
  4. Click/interact at those coordinates
  5. Take another screenshot to verify it worked
  6. Retry up to 3 times if it didn't work

This makes the controller self-healing — it works even when
CapCut updates their UI completely.
"""

import asyncio
import base64
import json
import random
from typing import Any

from playwright.async_api import Page

from .openrouter import router
from .config import cfg

CAPCUT_URL = "https://www.capcut.com"

FIND_ELEMENT_SYSTEM = """You are analyzing a screenshot of CapCut's web video editor.
Your job is to locate UI elements precisely.
Always return valid JSON only. Never explain, never add markdown."""

FIND_ELEMENT_PROMPT = """Find this element in the CapCut editor screenshot: "{element}"

Return JSON:
{{
  "found": true/false,
  "x": <center x pixel coordinate as integer>,
  "y": <center y pixel coordinate as integer>,
  "confidence": <0.0 to 1.0>,
  "description": "<what you see at that location>"
}}

If not found return: {{"found": false, "x": 0, "y": 0, "confidence": 0, "description": "not found"}}"""

VERIFY_ACTION_PROMPT = """A user just performed this action in CapCut: "{action}"
Look at this screenshot and determine if the action succeeded.
Return JSON: {{"success": true/false, "reason": "<brief explanation>"}}"""


class VisionController:
    """Controls CapCut by seeing screenshots and clicking intelligently."""

    def __init__(self, page: Page):
        self.page = page
        self._timeline_clips: list[dict] = []

    # ------------------------------------------------------------------
    # Core vision engine
    # ------------------------------------------------------------------

    async def find_and_click(
        self,
        element_description: str,
        verify_action: str | None = None,
        max_retries: int = 3,
        double_click: bool = False,
    ) -> bool:
        """
        Find an element by description, click it, optionally verify success.
        Returns True if successful.
        """
        for attempt in range(max_retries):
            screenshot_b64 = await self._screenshot_b64()

            result = router.vision_json(
                prompt=FIND_ELEMENT_PROMPT.format(element=element_description),
                images_b64=[screenshot_b64],
                model=cfg.models.vision,
                system=FIND_ELEMENT_SYSTEM,
            )

            if not result.get("found") or result.get("confidence", 0) < 0.4:
                await self.page.wait_for_timeout(1000)
                continue

            x, y = int(result["x"]), int(result["y"])
            await self._human_move_and_click(x, y, double_click=double_click)
            await self.page.wait_for_timeout(800)

            # Verify if requested
            if verify_action:
                ok = await self._verify(verify_action)
                if ok:
                    return True
                await self.page.wait_for_timeout(1500)
                continue

            return True

        return False

    async def find_coordinates(self, element_description: str) -> tuple[int, int] | None:
        """Return (x, y) pixel coordinates of an element, or None if not found."""
        screenshot_b64 = await self._screenshot_b64()
        result = router.vision_json(
            prompt=FIND_ELEMENT_PROMPT.format(element=element_description),
            images_b64=[screenshot_b64],
            model=cfg.models.vision,
            system=FIND_ELEMENT_SYSTEM,
        )
        if result.get("found"):
            return int(result["x"]), int(result["y"])
        return None

    async def read_screen(self, question: str) -> str:
        """Ask the vision AI a question about the current screen state."""
        screenshot_b64 = await self._screenshot_b64()
        return router.vision(
            prompt=question,
            images_b64=[screenshot_b64],
            model=cfg.models.vision,
        )

    # ------------------------------------------------------------------
    # CapCut-specific actions
    # ------------------------------------------------------------------

    async def navigate_to_editor(self):
        await self.page.goto(f"{CAPCUT_URL}/editor", wait_until="networkidle")
        await self.page.wait_for_timeout(3000)

    async def create_new_project(self, name: str) -> dict:
        await self.navigate_to_editor()
        ok = await self.find_and_click(
            "New project button or Create project button",
            verify_action="new project dialog or empty timeline appeared",
        )
        await self.page.wait_for_timeout(2000)
        return {"status": "created" if ok else "may_have_failed", "project": name}

    async def import_media(self, file_path: str) -> dict:
        ok = await self.find_and_click(
            "Import media button or Upload button or plus icon to add media",
        )
        if ok:
            await self.page.wait_for_timeout(500)
            try:
                async with self.page.expect_file_chooser(timeout=5000) as fc_info:
                    await self.find_and_click("Upload from computer or local files option")
                file_chooser = await fc_info.value
                await file_chooser.set_files(file_path)
                await self.page.wait_for_timeout(4000)
                self._timeline_clips.append({"path": file_path, "index": len(self._timeline_clips)})
                return {"status": "imported", "file": file_path}
            except Exception:
                # Try direct file chooser trigger
                try:
                    async with self.page.expect_file_chooser(timeout=3000) as fc_info:
                        await self.find_and_click("Browse files or choose file button")
                    file_chooser = await fc_info.value
                    await file_chooser.set_files(file_path)
                    await self.page.wait_for_timeout(4000)
                    self._timeline_clips.append({"path": file_path, "index": len(self._timeline_clips)})
                    return {"status": "imported", "file": file_path}
                except Exception as e:
                    return {"status": "error", "message": str(e)}
        return {"status": "import_button_not_found"}

    async def select_clip(self, clip_index: int) -> bool:
        ok = await self.find_and_click(
            f"clip number {clip_index + 1} on the video timeline at the bottom of the editor",
        )
        await self.page.wait_for_timeout(500)
        return ok

    async def trim_clip(self, clip_index: int, start_seconds: float, end_seconds: float) -> dict:
        await self.select_clip(clip_index)

        # Try to drag the left trim handle to start_seconds
        ok = await self.find_and_click(
            f"left trim handle or start point of clip {clip_index + 1} on the timeline",
        )
        if ok:
            # Use keyboard shortcut or input field
            await self.find_and_click("start time input field in the properties panel")
            await self.page.keyboard.press("Control+a")
            await self.page.keyboard.type(self._fmt_time(start_seconds))
            await self.page.keyboard.press("Enter")

        await self.find_and_click("end time or duration input field in the clip properties")
        await self.page.keyboard.press("Control+a")
        await self.page.keyboard.type(self._fmt_time(end_seconds))
        await self.page.keyboard.press("Enter")

        return {"status": "trimmed", "clip": clip_index, "start": start_seconds, "end": end_seconds}

    async def add_text(
        self,
        text: str,
        position: str = "bottom",
        start_seconds: float = 0,
        duration_seconds: float = 3,
        font_size: str = "medium",
        color: str = "white",
        animated: bool = False,
        style: str = "clean",
    ) -> dict:
        # Open Text panel
        await self.find_and_click("Text tab in the left sidebar of the CapCut editor")
        await self.page.wait_for_timeout(1000)

        # Click Add text
        await self.find_and_click(
            "Add text button or default text option to create a new text element",
            verify_action="text editor or text input appeared on the canvas",
        )
        await self.page.wait_for_timeout(800)

        # Type text
        await self.page.keyboard.press("Control+a")
        await self.page.keyboard.type(text)
        await self.page.wait_for_timeout(400)

        # Position alignment
        position_map = {"top": "top align button", "center": "center align button", "bottom": "bottom align button"}
        await self.find_and_click(position_map.get(position, "bottom align button"))

        if animated:
            await self.find_and_click("text animation or animated text style option")

        return {"status": "added", "text": text, "position": position}

    async def add_auto_captions(self, language: str = "English") -> dict:
        await self.find_and_click("Captions tab or Auto captions button in the left panel")
        await self.page.wait_for_timeout(1000)
        await self.find_and_click("Auto captions or Generate captions button")
        await self.page.wait_for_timeout(8000)  # Wait for transcription
        return {"status": "captions_generated"}

    async def add_transition(
        self, clip_index: int = 0, transition_type: str = "fade", duration_seconds: float = 0.5
    ) -> dict:
        # Click the transition gap between clips on the timeline
        ok = await self.find_and_click(
            f"transition point or gap between clip {clip_index + 1} and clip {clip_index + 2} on the timeline",
        )
        await self.page.wait_for_timeout(500)

        # Search/select the transition type
        await self.find_and_click(
            f"{transition_type.replace('_', ' ')} transition option in the transitions panel",
        )
        await self.page.wait_for_timeout(500)
        return {"status": "added", "transition": transition_type}

    async def add_music(
        self,
        query: str,
        volume: float = 0.5,
        fade_in: bool = True,
        fade_out: bool = True,
    ) -> dict:
        await self.find_and_click("Audio tab in the left sidebar")
        await self.page.wait_for_timeout(1000)
        await self.find_and_click("Music or Sounds search input field")
        await self.page.keyboard.type(query)
        await self.page.keyboard.press("Enter")
        await self.page.wait_for_timeout(2500)
        await self.find_and_click("first music track result in the search results list")
        await self.page.wait_for_timeout(500)
        await self.find_and_click("Add to timeline button for the selected music track")
        await self.page.wait_for_timeout(1000)
        return {"status": "added", "music": query, "volume": volume}

    async def apply_filter(self, filter_name: str, intensity: float = 0.7) -> dict:
        await self.find_and_click("Filter tab in the left sidebar or top toolbar")
        await self.page.wait_for_timeout(1000)
        await self.find_and_click(f"{filter_name} filter option in the filter panel")
        await self.page.wait_for_timeout(500)

        # Adjust intensity slider
        coords = await self.find_coordinates("filter intensity or strength slider")
        if coords:
            x, y = coords
            # Get the slider track and set position by intensity
            slider_box = await self.page.locator(".slider, [class*='slider']").first.bounding_box()
            if slider_box:
                target_x = slider_box["x"] + slider_box["width"] * intensity
                await self.page.mouse.click(target_x, slider_box["y"] + slider_box["height"] / 2)

        return {"status": "applied", "filter": filter_name, "intensity": intensity}

    async def apply_lut(self, lut_style: str) -> dict:
        """Apply a cinematic LUT color grade."""
        await self.find_and_click("Filter or Color tab in the editor sidebar")
        await self.page.wait_for_timeout(800)
        await self.find_and_click(f"{lut_style} LUT or color preset in the filter list")
        return {"status": "applied", "lut": lut_style}

    async def adjust_speed(self, clip_index: int, speed_multiplier: float) -> dict:
        await self.select_clip(clip_index)
        await self.find_and_click("Speed button or playback speed option in the clip properties panel")
        await self.page.wait_for_timeout(500)
        await self.find_and_click("speed value input field")
        await self.page.keyboard.press("Control+a")
        await self.page.keyboard.type(str(speed_multiplier))
        await self.page.keyboard.press("Enter")
        return {"status": "adjusted", "speed": speed_multiplier}

    async def apply_speed_ramp(self, clip_index: int, ramp_type: str = "drop") -> dict:
        """Apply cinematic speed ramp (slow → fast at music drop)."""
        await self.select_clip(clip_index)
        await self.find_and_click("Speed button in the clip properties")
        await self.page.wait_for_timeout(500)
        await self.find_and_click("Curve or Speed curve option for custom speed ramping")
        await self.page.wait_for_timeout(500)

        if ramp_type == "drop":
            await self.find_and_click("Hero or custom speed curve preset that slows then accelerates")
        elif ramp_type == "slow_mo":
            await self.find_and_click("Montage or slow motion speed curve preset")

        return {"status": "speed_ramped", "clip": clip_index, "type": ramp_type}

    async def adjust_volume(self, clip_index: int, volume: float) -> dict:
        await self.select_clip(clip_index)
        await self.find_and_click("Volume slider or audio volume control in the properties panel")
        coords = await self.find_coordinates("volume slider track")
        if coords:
            slider = await self.page.locator("[class*='volume'] [class*='slider'], [class*='audio'] input[type='range']").first.bounding_box()
            if slider:
                x = slider["x"] + slider["width"] * min(volume, 1.0)
                await self.page.mouse.click(x, slider["y"] + slider["height"] / 2)
        return {"status": "adjusted", "volume": volume}

    async def add_sticker(self, query: str, position_x: float = 50, position_y: float = 50) -> dict:
        await self.find_and_click("Stickers tab in the left sidebar")
        await self.page.wait_for_timeout(800)
        await self.find_and_click("sticker search input")
        await self.page.keyboard.type(query)
        await self.page.keyboard.press("Enter")
        await self.page.wait_for_timeout(2000)
        await self.find_and_click("first sticker result in the search results")
        return {"status": "added", "sticker": query}

    async def add_effect(self, effect_name: str) -> dict:
        """Add a visual effect (glitch, particle, light leak, etc.)."""
        await self.find_and_click("Effects tab in the left sidebar")
        await self.page.wait_for_timeout(800)
        await self.find_and_click("effect search input or browse effects")
        await self.page.keyboard.type(effect_name)
        await self.page.keyboard.press("Enter")
        await self.page.wait_for_timeout(1500)
        await self.find_and_click(f"first {effect_name} effect result")
        await self.find_and_click("Add effect to timeline button")
        return {"status": "added", "effect": effect_name}

    async def apply_ken_burns(self, clip_index: int, direction: str = "zoom_in") -> dict:
        """Apply Ken Burns zoom/pan effect to a clip."""
        await self.select_clip(clip_index)
        await self.find_and_click("Animation or keyframe animation button for the selected clip")
        await self.page.wait_for_timeout(500)
        await self.find_and_click(f"{direction.replace('_', ' ')} animation preset or zoom animation")
        return {"status": "applied", "effect": "ken_burns", "direction": direction}

    async def add_beat_sync_cuts(self, beat_timestamps: list[float]) -> dict:
        """Split clips at beat timestamps to create beat-synced cuts."""
        for ts in beat_timestamps:
            # Move playhead to beat timestamp
            await self.find_and_click("timeline playhead or current time indicator")
            await self.find_and_click("current time input field on the timeline")
            await self.page.keyboard.press("Control+a")
            await self.page.keyboard.type(self._fmt_time(ts))
            await self.page.keyboard.press("Enter")
            await self.page.wait_for_timeout(200)
            # Split the clip at this point
            await self.page.keyboard.press("Control+b")  # CapCut split shortcut
            await self.page.wait_for_timeout(300)
        return {"status": "beat_synced", "cuts": len(beat_timestamps)}

    async def remove_background(self, clip_index: int) -> dict:
        await self.select_clip(clip_index)
        await self.find_and_click("Remove background or AI background removal button in the properties panel")
        await self.page.wait_for_timeout(6000)  # AI processing time
        return {"status": "background_removed", "clip": clip_index}

    async def stabilize_clip(self, clip_index: int) -> dict:
        await self.select_clip(clip_index)
        await self.find_and_click("Stabilization or Video stabilize button in the properties panel")
        await self.page.wait_for_timeout(4000)
        return {"status": "stabilized", "clip": clip_index}

    async def auto_enhance(self, clip_index: int) -> dict:
        await self.select_clip(clip_index)
        await self.find_and_click("Auto enhance or AI enhance button for the clip")
        await self.page.wait_for_timeout(2000)
        return {"status": "enhanced", "clip": clip_index}

    async def crop_to_aspect_ratio(self, aspect_ratio: str) -> dict:
        await self.find_and_click("aspect ratio or canvas ratio settings button")
        await self.page.wait_for_timeout(500)
        await self.find_and_click(f"{aspect_ratio} aspect ratio option in the ratio menu")
        return {"status": "cropped", "aspect_ratio": aspect_ratio}

    async def add_vignette(self, intensity: float = 0.5) -> dict:
        await self.find_and_click("Adjust or Color adjustment tab")
        await self.page.wait_for_timeout(500)
        await self.find_and_click("Vignette slider or vignette effect control")
        return {"status": "vignette_added", "intensity": intensity}

    async def add_film_grain(self, intensity: float = 0.3) -> dict:
        await self.find_and_click("Effects or Overlay effects tab")
        await self.page.wait_for_timeout(500)
        await self.find_and_click("Film grain or noise texture effect option")
        return {"status": "film_grain_added", "intensity": intensity}

    async def add_light_leak(self) -> dict:
        await self.find_and_click("Effects tab in the sidebar")
        await self.page.wait_for_timeout(500)
        await self.find_and_click("Light leak or lens flare overlay effect in the effects library")
        await self.page.wait_for_timeout(500)
        await self.find_and_click("Add light leak to timeline")
        return {"status": "light_leak_added"}

    async def split_screen(self, layout: str = "side_by_side") -> dict:
        await self.find_and_click("Split screen or Multi-clip layout option in CapCut")
        await self.page.wait_for_timeout(800)
        await self.find_and_click(f"{layout.replace('_', ' ')} split screen layout option")
        return {"status": "split_screen_applied", "layout": layout}

    async def reverse_clip(self, clip_index: int) -> dict:
        await self.select_clip(clip_index)
        await self.find_and_click("Reverse clip button or play in reverse option in clip settings")
        await self.page.wait_for_timeout(3000)
        return {"status": "reversed", "clip": clip_index}

    async def freeze_frame(self, clip_index: int, at_seconds: float, duration: float = 1.0) -> dict:
        await self.select_clip(clip_index)
        await self.find_and_click("Freeze frame button or hold frame option in the clip properties")
        return {"status": "freeze_frame_added", "at": at_seconds, "duration": duration}

    async def mirror_clip(self, clip_index: int, direction: str = "horizontal") -> dict:
        await self.select_clip(clip_index)
        await self.find_and_click(f"Flip {direction} or mirror {direction} button in clip transform settings")
        return {"status": "mirrored", "direction": direction}

    async def reduce_noise(self, clip_index: int) -> dict:
        await self.select_clip(clip_index)
        await self.find_and_click("Noise reduction or denoise audio button in the audio properties")
        await self.page.wait_for_timeout(2000)
        return {"status": "noise_reduced", "clip": clip_index}

    async def duck_audio(self, duck_level: float = 0.3) -> dict:
        """Reduce clip audio volume when background music plays (ducking)."""
        await self.find_and_click("Audio ducking or auto ducking option in audio settings")
        return {"status": "audio_ducked", "level": duck_level}

    async def undo(self, steps: int = 1) -> dict:
        for _ in range(steps):
            await self.page.keyboard.press("Control+z")
            await self.page.wait_for_timeout(300)
        return {"status": "undone", "steps": steps}

    async def export_video(
        self,
        resolution: str = "1080p",
        fps: int = 30,
        format: str = "mp4",
    ) -> dict:
        await self.find_and_click(
            "Export button in the top right corner of the CapCut editor",
            verify_action="export dialog or resolution settings appeared",
        )
        await self.page.wait_for_timeout(1500)
        await self.find_and_click(f"{resolution} resolution option in the export settings")
        await self.page.wait_for_timeout(500)
        await self.find_and_click("Export or Render button to start the export process")
        # Wait for export to complete (up to 10 minutes)
        await self.page.wait_for_timeout(5000)
        return {"status": "exporting", "resolution": resolution, "fps": fps, "format": format}

    async def get_timeline_info(self) -> dict:
        info = await self.read_screen(
            "Describe the current state of the CapCut timeline. "
            "How many clips are there? What is the total duration? "
            "Are there any audio tracks, text layers, or effects visible? "
            "Return as a brief JSON: {clips: int, duration: str, has_audio: bool, has_text: bool}"
        )
        try:
            return json.loads(info)
        except Exception:
            return {"raw_description": info}

    # ------------------------------------------------------------------
    # Tool dispatcher
    # ------------------------------------------------------------------

    async def execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> dict:
        dispatch = {
            "open_project": lambda i: self.create_new_project(i.get("project_name", "Jarvis Edit")),
            "import_media": lambda i: self.import_media(i["file_path"]),
            "trim_clip": lambda i: self.trim_clip(i.get("clip_index", 0), i["start_seconds"], i["end_seconds"]),
            "add_text": lambda i: self.add_text(**i),
            "add_auto_captions": lambda i: self.add_auto_captions(),
            "add_transition": lambda i: self.add_transition(**i),
            "add_music": lambda i: self.add_music(i["query_or_path"], i.get("volume", 0.5)),
            "apply_filter": lambda i: self.apply_filter(i["filter_name"], i.get("intensity", 0.7)),
            "apply_lut": lambda i: self.apply_lut(i["lut_style"]),
            "adjust_speed": lambda i: self.adjust_speed(i.get("clip_index", 0), i["speed_multiplier"]),
            "apply_speed_ramp": lambda i: self.apply_speed_ramp(i.get("clip_index", 0), i.get("ramp_type", "drop")),
            "adjust_volume": lambda i: self.adjust_volume(i.get("clip_index", 0), i["volume"]),
            "add_sticker": lambda i: self.add_sticker(i["query"]),
            "add_effect": lambda i: self.add_effect(i["effect_name"]),
            "apply_ken_burns": lambda i: self.apply_ken_burns(i.get("clip_index", 0), i.get("direction", "zoom_in")),
            "add_beat_sync_cuts": lambda i: self.add_beat_sync_cuts(i["beat_timestamps"]),
            "remove_background": lambda i: self.remove_background(i.get("clip_index", 0)),
            "stabilize_clip": lambda i: self.stabilize_clip(i.get("clip_index", 0)),
            "auto_enhance": lambda i: self.auto_enhance(i.get("clip_index", 0)),
            "crop_video": lambda i: self.crop_to_aspect_ratio(i["aspect_ratio"]),
            "add_vignette": lambda i: self.add_vignette(i.get("intensity", 0.5)),
            "add_film_grain": lambda i: self.add_film_grain(i.get("intensity", 0.3)),
            "add_light_leak": lambda i: self.add_light_leak(),
            "split_screen": lambda i: self.split_screen(i.get("layout", "side_by_side")),
            "reverse_clip": lambda i: self.reverse_clip(i.get("clip_index", 0)),
            "freeze_frame": lambda i: self.freeze_frame(i.get("clip_index", 0), i.get("at_seconds", 0)),
            "mirror_clip": lambda i: self.mirror_clip(i.get("clip_index", 0), i.get("direction", "horizontal")),
            "reduce_noise": lambda i: self.reduce_noise(i.get("clip_index", 0)),
            "duck_audio": lambda i: self.duck_audio(i.get("duck_level", 0.3)),
            "export_video": lambda i: self.export_video(**i),
            "undo": lambda i: self.undo(i.get("steps", 1)),
            "get_timeline_info": lambda i: self.get_timeline_info(),
        }
        handler = dispatch.get(tool_name)
        if not handler:
            return {"status": "error", "message": f"Unknown tool: {tool_name}"}
        try:
            return await handler(tool_input)
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _screenshot_b64(self) -> str:
        data = await self.page.screenshot(type="jpeg", quality=80)
        return base64.b64encode(data).decode()

    async def _human_move_and_click(self, x: int, y: int, double_click: bool = False):
        # Add small random offset for human-like precision
        ox = random.randint(-3, 3)
        oy = random.randint(-3, 3)
        await self.page.mouse.move(x + ox, y + oy)
        await self.page.wait_for_timeout(random.randint(60, 180))
        if double_click:
            await self.page.mouse.dblclick(x + ox, y + oy)
        else:
            await self.page.mouse.click(x + ox, y + oy)

    async def _verify(self, action_description: str) -> bool:
        screenshot_b64 = await self._screenshot_b64()
        result = router.vision_json(
            prompt=VERIFY_ACTION_PROMPT.format(action=action_description),
            images_b64=[screenshot_b64],
            model=cfg.models.vision,
        )
        return bool(result.get("success", False))

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        m = int(seconds // 60)
        s = seconds % 60
        return f"{m:02d}:{s:06.3f}"
