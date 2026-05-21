"""
Stealth browser manager with cookie persistence.

First run: opens visible browser so user can log in manually → saves cookies.
Subsequent runs: loads saved cookies → already logged in.
"""

import json
import os
from pathlib import Path

from playwright.async_api import async_playwright, Browser, BrowserContext, Page, Playwright

COOKIES_PATH = Path("data/cookies/capcut_session.json")
CAPCUT_URL = "https://www.capcut.com"


class BrowserManager:
    def __init__(self, headless: bool = True):
        self.headless = headless
        self._pw: Playwright | None = None
        self._browser: Browser | None = None
        self.context: BrowserContext | None = None
        self.page: Page | None = None

    async def start(self) -> Page:
        self._pw = await async_playwright().start()

        # Use new headless mode — harder for bot detection to identify
        self._browser = await self._pw.chromium.launch(
            headless=self.headless,
            args=[
                "--headless=new" if self.headless else "",
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--window-size=1920,1080",
                "--ignore-certificate-errors",
                "--ignore-ssl-errors",
                "--disable-web-security",
            ],
        )

        self.context = await self._browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            locale="en-US",
            timezone_id="America/New_York",
            ignore_https_errors=True,
        )

        # Apply stealth scripts to mask automation fingerprints
        await self._apply_stealth(self.context)

        # Load saved cookies if available
        if COOKIES_PATH.exists():
            cookies = json.loads(COOKIES_PATH.read_text())
            await self.context.add_cookies(cookies)

        self.page = await self.context.new_page()
        return self.page

    async def ensure_logged_in(self, email: str = "", password: str = "") -> bool:
        """
        Check login status. If not logged in:
          1. Try auto-login with credentials
          2. Fall back to manual login (opens visible browser)
        Returns True when logged in.
        """
        await self.page.goto(CAPCUT_URL, wait_until="networkidle")
        await self.page.wait_for_timeout(2000)

        if await self._is_logged_in():
            return True

        # Try auto-login if credentials provided
        if email and password:
            success = await self._auto_login(email, password)
            if success:
                await self._save_cookies()
                return True

        # Fall back: open visible browser for manual login
        print("\n[Jarvis] Please log in to CapCut in the browser window.")
        print("[Jarvis] Press Enter here once you are logged in.")

        # Make browser visible for manual login
        if self.headless:
            # Re-launch in headed mode
            await self._browser.close()
            self._browser = await self._pw.chromium.launch(headless=False)
            self.context = await self._browser.new_context(viewport={"width": 1920, "height": 1080})
            await self._apply_stealth(self.context)
            self.page = await self.context.new_page()
            await self.page.goto(CAPCUT_URL, wait_until="networkidle")

        input()
        await self._save_cookies()
        return await self._is_logged_in()

    async def save_session(self):
        """Save current cookies to disk."""
        await self._save_cookies()

    async def stop(self):
        if self._browser:
            await self._browser.close()
        if self._pw:
            await self._pw.stop()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _is_logged_in(self) -> bool:
        try:
            # CapCut shows avatar / user menu when logged in
            logged_in = await self.page.evaluate("""
                () => {
                    return (
                        document.cookie.includes('passport') ||
                        document.cookie.includes('sid_tt') ||
                        !!document.querySelector('[class*="avatar"], [class*="userInfo"]')
                    );
                }
            """)
            return bool(logged_in)
        except Exception:
            return False

    async def _auto_login(self, email: str, password: str) -> bool:
        try:
            await self.page.goto(f"{CAPCUT_URL}/login", wait_until="networkidle")
            await self._human_delay(1000, 2000)

            # Try email login tab
            for selector in ['text="Log in with email"', '[data-e2e="email-tab"]']:
                try:
                    btn = self.page.locator(selector).first
                    if await btn.is_visible(timeout=2000):
                        await self._human_click(btn)
                        break
                except Exception:
                    continue

            await self._human_delay(500, 1000)

            # Fill credentials
            for selector in ['input[type="email"]', 'input[name="email"]', 'input[placeholder*="email" i]']:
                try:
                    field = self.page.locator(selector).first
                    if await field.is_visible(timeout=2000):
                        await self._human_type(field, email)
                        break
                except Exception:
                    continue

            for selector in ['input[type="password"]', 'input[name="password"]']:
                try:
                    field = self.page.locator(selector).first
                    if await field.is_visible(timeout=2000):
                        await self._human_type(field, password)
                        break
                except Exception:
                    continue

            await self._human_delay(500, 1000)
            await self.page.keyboard.press("Enter")
            await self.page.wait_for_timeout(4000)

            return await self._is_logged_in()
        except Exception as e:
            print(f"[Browser] Auto-login error: {e}")
            return False

    async def _save_cookies(self):
        COOKIES_PATH.parent.mkdir(parents=True, exist_ok=True)
        cookies = await self.context.cookies()
        COOKIES_PATH.write_text(json.dumps(cookies, indent=2))

    async def _human_click(self, locator):
        """Click with a small random offset to look human."""
        import random
        box = await locator.bounding_box()
        if box:
            x = box["x"] + box["width"] * random.uniform(0.3, 0.7)
            y = box["y"] + box["height"] * random.uniform(0.3, 0.7)
            await self.page.mouse.move(x, y)
            await self.page.wait_for_timeout(random.randint(80, 200))
            await self.page.mouse.click(x, y)
        else:
            await locator.click()

    async def _human_type(self, locator, text: str):
        """Type text character by character with random delays."""
        import random
        await locator.click()
        await self.page.wait_for_timeout(random.randint(100, 300))
        for char in text:
            await self.page.keyboard.type(char)
            await self.page.wait_for_timeout(random.randint(40, 120))

    async def _human_delay(self, min_ms: int, max_ms: int):
        import random
        await self.page.wait_for_timeout(random.randint(min_ms, max_ms))

    @staticmethod
    async def _apply_stealth(context: BrowserContext):
        """Inject JS to mask automation signals."""
        await context.add_init_script("""
            // Mask webdriver flag
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });

            // Mask automation-related chrome flags
            window.chrome = { runtime: {} };

            // Spoof plugins
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5],
            });

            // Spoof languages
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en'],
            });

            // Spoof permissions
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) =>
                parameters.name === 'notifications'
                    ? Promise.resolve({ state: Notification.permission })
                    : originalQuery(parameters);
        """)
