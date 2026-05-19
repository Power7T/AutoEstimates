"""CapCut web app controller using Playwright browser automation."""

import asyncio
import os
from typing import Any

from playwright.async_api import async_playwright, Browser, BrowserContext, Page


CAPCUT_URL = "https://www.capcut.com"


class CapCutController:
    """Automates CapCut's web editor at capcut.com."""

    def __init__(self, headless: bool = False):
        self.headless = headless
        self._playwright = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self.page: Page | None = None
        self._timeline_clips: list[dict] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        """Launch the browser and navigate to CapCut."""
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=["--start-maximized"],
        )
        self._context = await self._browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        self.page = await self._context.new_page()
        await self.page.goto(CAPCUT_URL, wait_until="networkidle")

    async def stop(self):
        """Close the browser."""
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    # ------------------------------------------------------------------
    # Login
    # ------------------------------------------------------------------

    async def login(self, email: str, password: str) -> bool:
        """Log into CapCut with email and password."""
        try:
            await self.page.goto(f"{CAPCUT_URL}/login", wait_until="networkidle")
            await self.page.wait_for_timeout(2000)

            # Click email login option if available
            email_btn = self.page.locator("text=Log in with email").first
            if await email_btn.is_visible():
                await email_btn.click()
                await self.page.wait_for_timeout(1000)

            await self.page.fill('input[type="email"], input[name="email"]', email)
            await self.page.fill('input[type="password"], input[name="password"]', password)
            await self.page.click('button[type="submit"]')
            await self.page.wait_for_timeout(3000)

            # Verify login succeeded
            return await self._is_logged_in()
        except Exception as e:
            print(f"[CapCut] Login error: {e}")
            return False

    async def _is_logged_in(self) -> bool:
        """Check if the user is currently logged in."""
        try:
            # Look for user avatar or dashboard element
            return await self.page.locator('[data-testid="user-avatar"], .user-avatar').is_visible(timeout=3000)
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Project management
    # ------------------------------------------------------------------

    async def open_project(self, project_name: str, create_new: bool = False) -> dict:
        """Open or create a CapCut project."""
        await self.page.goto(f"{CAPCUT_URL}/editor", wait_until="networkidle")
        await self.page.wait_for_timeout(2000)

        if create_new:
            # Click "New Project" button
            new_btn = self.page.locator('button:has-text("New project"), [data-testid="new-project"]').first
            if await new_btn.is_visible():
                await new_btn.click()
                await self.page.wait_for_timeout(2000)
            return {"status": "created", "project": project_name}

        # Search for existing project
        search = self.page.locator('input[placeholder*="search" i], input[placeholder*="project" i]').first
        if await search.is_visible():
            await search.fill(project_name)
            await self.page.wait_for_timeout(1000)

        project_card = self.page.locator(f'text="{project_name}"').first
        if await project_card.is_visible():
            await project_card.click()
            await self.page.wait_for_timeout(3000)
            return {"status": "opened", "project": project_name}

        return {"status": "not_found", "project": project_name}

    # ------------------------------------------------------------------
    # Media import
    # ------------------------------------------------------------------

    async def import_media(self, file_path: str) -> dict:
        """Import a media file into the project."""
        if not os.path.exists(file_path):
            return {"status": "error", "message": f"File not found: {file_path}"}

        # Click the "Import" or "+" button
        import_btn = self.page.locator(
            '[data-testid="import-btn"], button:has-text("Import"), .import-button'
        ).first
        if await import_btn.is_visible():
            async with self.page.expect_file_chooser() as fc_info:
                await import_btn.click()
            file_chooser = await fc_info.value
            await file_chooser.set_files(file_path)
            await self.page.wait_for_timeout(3000)

        self._timeline_clips.append({"path": file_path, "index": len(self._timeline_clips)})
        return {"status": "imported", "file": file_path}

    # ------------------------------------------------------------------
    # Timeline operations
    # ------------------------------------------------------------------

    async def trim_clip(self, clip_index: int, start_seconds: float, end_seconds: float) -> dict:
        """Trim a clip to a new in/out point."""
        await self._select_clip(clip_index)
        await self.page.wait_for_timeout(500)

        # Use keyboard shortcut to open split/trim panel, or drag handles
        # Try the trim input fields that CapCut shows in the right panel
        start_input = self.page.locator('input[data-testid="start-time"], input[placeholder*="start" i]').first
        end_input = self.page.locator('input[data-testid="end-time"], input[placeholder*="end" i]').first

        if await start_input.is_visible():
            await start_input.fill(self._format_time(start_seconds))
            await start_input.press("Enter")

        if await end_input.is_visible():
            await end_input.fill(self._format_time(end_seconds))
            await end_input.press("Enter")

        await self.page.wait_for_timeout(500)
        return {"status": "trimmed", "clip": clip_index, "start": start_seconds, "end": end_seconds}

    async def add_text(
        self,
        text: str,
        position: str = "bottom",
        start_seconds: float = 0,
        duration_seconds: float = 3,
        font_size: str = "medium",
        color: str = "white",
    ) -> dict:
        """Add a text overlay to the video."""
        # Click the Text tab in the left toolbar
        text_tab = self.page.locator('[data-testid="text-tab"], button:has-text("Text"), .text-tab').first
        if await text_tab.is_visible():
            await text_tab.click()
            await self.page.wait_for_timeout(1000)

        # Click "Add text" or the default text option
        add_text_btn = self.page.locator('button:has-text("Add text"), [data-testid="add-text"]').first
        if await add_text_btn.is_visible():
            await add_text_btn.click()
            await self.page.wait_for_timeout(1000)

        # Type the text into the canvas text editor
        text_editor = self.page.locator('textarea[data-testid="text-editor"], .text-input-area').first
        if await text_editor.is_visible():
            await text_editor.fill(text)

        # Set position using alignment buttons
        position_map = {"top": "top", "center": "center", "bottom": "bottom"}
        pos_btn = self.page.locator(f'[data-testid="align-{position_map[position]}"]').first
        if await pos_btn.is_visible():
            await pos_btn.click()

        await self.page.wait_for_timeout(500)
        return {
            "status": "added",
            "text": text,
            "position": position,
            "start": start_seconds,
            "duration": duration_seconds,
        }

    async def add_transition(
        self, clip_index: int = 0, transition_type: str = "fade", duration_seconds: float = 0.5
    ) -> dict:
        """Add a transition between clips."""
        # Click the Transitions tab
        transition_tab = self.page.locator(
            '[data-testid="transition-tab"], button:has-text("Transition")'
        ).first
        if await transition_tab.is_visible():
            await transition_tab.click()
            await self.page.wait_for_timeout(1000)

        # Click the transition point between clips on the timeline
        # CapCut shows a "+" between clips on the timeline
        transition_point = self.page.locator(f'.timeline-transition-point >> nth={clip_index}').first
        if await transition_point.is_visible():
            await transition_point.click()
            await self.page.wait_for_timeout(500)

        # Select the transition type
        transition_item = self.page.locator(f'[data-testid="transition-{transition_type}"], text="{transition_type.replace("_", " ").title()}"').first
        if await transition_item.is_visible():
            await transition_item.click()
            await self.page.wait_for_timeout(500)

        return {"status": "added", "transition": transition_type, "clip": clip_index, "duration": duration_seconds}

    async def add_music(
        self,
        query_or_path: str,
        source: str = "library",
        volume: float = 0.5,
        fade_in: bool = True,
        fade_out: bool = True,
    ) -> dict:
        """Add background music."""
        # Click the Audio tab
        audio_tab = self.page.locator('[data-testid="audio-tab"], button:has-text("Audio")').first
        if await audio_tab.is_visible():
            await audio_tab.click()
            await self.page.wait_for_timeout(1000)

        if source == "library":
            search_input = self.page.locator('input[placeholder*="search" i]').first
            if await search_input.is_visible():
                await search_input.fill(query_or_path)
                await self.page.keyboard.press("Enter")
                await self.page.wait_for_timeout(2000)

            first_track = self.page.locator('.music-item, [data-testid="music-track"]').first
            if await first_track.is_visible():
                await first_track.click()
                await self.page.wait_for_timeout(500)

                add_btn = self.page.locator('button:has-text("Add"), [data-testid="add-music"]').first
                if await add_btn.is_visible():
                    await add_btn.click()
                    await self.page.wait_for_timeout(1000)
        else:
            # Local file
            upload_btn = self.page.locator('button:has-text("Upload"), [data-testid="upload-audio"]').first
            if await upload_btn.is_visible():
                async with self.page.expect_file_chooser() as fc_info:
                    await upload_btn.click()
                file_chooser = await fc_info.value
                await file_chooser.set_files(query_or_path)
                await self.page.wait_for_timeout(2000)

        return {"status": "added", "music": query_or_path, "volume": volume}

    async def apply_filter(
        self,
        filter_name: str,
        intensity: float = 0.7,
        apply_to: str = "all",
        clip_index: int = 0,
    ) -> dict:
        """Apply a visual filter."""
        # Click the Filter tab
        filter_tab = self.page.locator('[data-testid="filter-tab"], button:has-text("Filter")').first
        if await filter_tab.is_visible():
            await filter_tab.click()
            await self.page.wait_for_timeout(1000)

        # Find and click the filter
        filter_item = self.page.locator(f'text="{filter_name.title()}", [data-filter-name="{filter_name}"]').first
        if await filter_item.is_visible():
            await filter_item.click()
            await self.page.wait_for_timeout(500)

            # Adjust intensity slider
            slider = self.page.locator('.filter-intensity-slider, [data-testid="intensity-slider"]').first
            if await slider.is_visible():
                await self._set_slider(slider, intensity)

        return {"status": "applied", "filter": filter_name, "intensity": intensity}

    async def adjust_speed(self, clip_index: int = 0, speed_multiplier: float = 1.0) -> dict:
        """Adjust clip playback speed."""
        await self._select_clip(clip_index)
        await self.page.wait_for_timeout(500)

        speed_btn = self.page.locator('[data-testid="speed-btn"], button:has-text("Speed")').first
        if await speed_btn.is_visible():
            await speed_btn.click()
            await self.page.wait_for_timeout(500)

            speed_input = self.page.locator('input[data-testid="speed-input"]').first
            if await speed_input.is_visible():
                await speed_input.fill(str(speed_multiplier))
                await speed_input.press("Enter")

        return {"status": "adjusted", "clip": clip_index, "speed": speed_multiplier}

    async def adjust_volume(self, clip_index: int = 0, volume: float = 1.0) -> dict:
        """Adjust clip audio volume."""
        await self._select_clip(clip_index)
        await self.page.wait_for_timeout(500)

        volume_slider = self.page.locator('[data-testid="volume-slider"], .volume-control').first
        if await volume_slider.is_visible():
            await self._set_slider(volume_slider, volume)

        return {"status": "adjusted", "clip": clip_index, "volume": volume}

    async def add_sticker(
        self,
        query: str,
        position_x: float = 50,
        position_y: float = 50,
        start_seconds: float = 0,
        duration_seconds: float = 2,
    ) -> dict:
        """Add an animated sticker overlay."""
        sticker_tab = self.page.locator('[data-testid="sticker-tab"], button:has-text("Sticker")').first
        if await sticker_tab.is_visible():
            await sticker_tab.click()
            await self.page.wait_for_timeout(1000)

        search_input = self.page.locator('input[placeholder*="search" i]').first
        if await search_input.is_visible():
            await search_input.fill(query)
            await self.page.keyboard.press("Enter")
            await self.page.wait_for_timeout(2000)

        first_sticker = self.page.locator('.sticker-item, [data-testid="sticker"]').first
        if await first_sticker.is_visible():
            await first_sticker.click()
            await self.page.wait_for_timeout(500)

        return {"status": "added", "sticker": query, "position": (position_x, position_y)}

    async def crop_video(self, aspect_ratio: str) -> dict:
        """Crop video to a specific aspect ratio."""
        ratio_btn = self.page.locator(f'[data-testid="ratio-{aspect_ratio.replace(":", "x")}"], text="{aspect_ratio}"').first
        if await ratio_btn.is_visible():
            await ratio_btn.click()
            await self.page.wait_for_timeout(500)

        return {"status": "cropped", "aspect_ratio": aspect_ratio}

    async def export_video(self, resolution: str = "1080p", fps: int = 30, format: str = "mp4") -> dict:
        """Export the final video."""
        export_btn = self.page.locator('button:has-text("Export"), [data-testid="export-btn"]').first
        if await export_btn.is_visible():
            await export_btn.click()
            await self.page.wait_for_timeout(1000)

        # Set resolution
        res_option = self.page.locator(f'text="{resolution}", [data-resolution="{resolution}"]').first
        if await res_option.is_visible():
            await res_option.click()
            await self.page.wait_for_timeout(300)

        # Confirm export
        confirm_btn = self.page.locator('button:has-text("Export"), button:has-text("Render")').last
        if await confirm_btn.is_visible():
            await confirm_btn.click()
            # Wait for export to complete (up to 5 minutes)
            await self.page.wait_for_selector(
                'text="Export complete", text="Download", [data-testid="export-done"]',
                timeout=300_000,
            )

        return {"status": "exported", "resolution": resolution, "fps": fps, "format": format}

    async def undo(self, steps: int = 1) -> dict:
        """Undo recent actions."""
        for _ in range(steps):
            await self.page.keyboard.press("Control+z")
            await self.page.wait_for_timeout(300)
        return {"status": "undone", "steps": steps}

    async def get_timeline_info(self) -> dict:
        """Return info about the current timeline state."""
        clips = []
        clip_elements = self.page.locator('.timeline-clip, [data-testid="timeline-clip"]')
        count = await clip_elements.count()
        for i in range(count):
            clip = clip_elements.nth(i)
            label = await clip.get_attribute("data-label") or f"Clip {i+1}"
            clips.append({"index": i, "label": label})

        duration_el = self.page.locator('[data-testid="total-duration"], .timeline-duration').first
        total_duration = "unknown"
        if await duration_el.is_visible():
            total_duration = await duration_el.inner_text()

        return {"clips": clips, "total_duration": total_duration, "clip_count": count}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _select_clip(self, clip_index: int):
        """Click on a clip in the timeline to select it."""
        clips = self.page.locator('.timeline-clip, [data-testid="timeline-clip"]')
        count = await clips.count()
        if clip_index < count:
            await clips.nth(clip_index).click()

    async def _set_slider(self, slider, value: float):
        """Set a range slider to a fractional value (0.0–1.0)."""
        box = await slider.bounding_box()
        if box:
            x = box["x"] + box["width"] * value
            y = box["y"] + box["height"] / 2
            await self.page.mouse.click(x, y)

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Convert seconds to MM:SS.mmm string."""
        m = int(seconds // 60)
        s = seconds % 60
        return f"{m:02d}:{s:06.3f}"

    # ------------------------------------------------------------------
    # Tool dispatcher — called by the AI brain
    # ------------------------------------------------------------------

    async def execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> dict:
        """Route a tool call from Jarvis to the right method."""
        dispatch = {
            "open_project": self.open_project,
            "import_media": self.import_media,
            "trim_clip": self.trim_clip,
            "add_text": self.add_text,
            "add_transition": self.add_transition,
            "add_music": self.add_music,
            "apply_filter": self.apply_filter,
            "adjust_speed": self.adjust_speed,
            "adjust_volume": self.adjust_volume,
            "add_sticker": self.add_sticker,
            "crop_video": self.crop_video,
            "export_video": self.export_video,
            "undo": self.undo,
            "get_timeline_info": self.get_timeline_info,
        }
        handler = dispatch.get(tool_name)
        if handler is None:
            return {"status": "error", "message": f"Unknown tool: {tool_name}"}
        try:
            return await handler(**tool_input)
        except Exception as e:
            return {"status": "error", "message": str(e)}
