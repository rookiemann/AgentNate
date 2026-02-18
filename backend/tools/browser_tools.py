"""
Browser Tools - Playwright-based browser automation.
"""

from typing import Dict, Any, List, Optional
import logging
import uuid
import asyncio
from pathlib import Path

logger = logging.getLogger("tools.browser")


TOOL_DEFINITIONS = [
    {
        "name": "browser_open",
        "description": "Open a URL in the automated browser. Creates a new browser session if needed.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to navigate to"
                },
                "wait_for": {
                    "type": "string",
                    "description": "Wait condition: 'load' (default), 'domcontentloaded', 'networkidle'",
                    "enum": ["load", "domcontentloaded", "networkidle"],
                    "default": "load"
                }
            },
            "required": ["url"]
        }
    },
    {
        "name": "browser_screenshot",
        "description": "Take a screenshot of the current page. Returns the path to the saved image.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to save screenshot (optional, auto-generated if not provided)"
                },
                "full_page": {
                    "type": "boolean",
                    "description": "Capture full page scroll (default false)",
                    "default": False
                },
                "selector": {
                    "type": "string",
                    "description": "CSS selector to screenshot specific element (optional)"
                }
            },
            "required": []
        }
    },
    {
        "name": "browser_click",
        "description": "Click an element on the page using CSS selector.",
        "parameters": {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector for the element to click"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in milliseconds (default 30000)",
                    "default": 30000
                }
            },
            "required": ["selector"]
        }
    },
    {
        "name": "browser_type",
        "description": "Type text into an input field.",
        "parameters": {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector for the input field"
                },
                "text": {
                    "type": "string",
                    "description": "Text to type"
                },
                "clear_first": {
                    "type": "boolean",
                    "description": "Clear existing text before typing (default true)",
                    "default": True
                },
                "press_enter": {
                    "type": "boolean",
                    "description": "Press Enter after typing (default false)",
                    "default": False
                }
            },
            "required": ["selector", "text"]
        }
    },
    {
        "name": "browser_extract",
        "description": "Extract data from page elements using CSS selectors.",
        "parameters": {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector for elements to extract"
                },
                "attribute": {
                    "type": "string",
                    "description": "Attribute to extract: 'textContent' (default), 'innerHTML', 'href', 'src', or any attribute name",
                    "default": "textContent"
                },
                "multiple": {
                    "type": "boolean",
                    "description": "Extract from all matching elements (default true)",
                    "default": True
                }
            },
            "required": ["selector"]
        }
    },
    {
        "name": "browser_get_text",
        "description": "Get all visible text content from the current page.",
        "parameters": {
            "type": "object",
            "properties": {
                "max_length": {
                    "type": "integer",
                    "description": "Maximum text length to return (default 50000)",
                    "default": 50000
                }
            },
            "required": []
        }
    },
    {
        "name": "browser_scroll",
        "description": "Scroll the page up or down.",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "description": "Scroll direction: 'down', 'up', 'top', 'bottom'",
                    "enum": ["down", "up", "top", "bottom"],
                    "default": "down"
                },
                "amount": {
                    "type": "integer",
                    "description": "Pixels to scroll (for 'up'/'down'). Default 500.",
                    "default": 500
                }
            },
            "required": []
        }
    },
    {
        "name": "browser_close",
        "description": "Close the browser session and free resources.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
]


class BrowserTools:
    """Tools for browser automation using Playwright."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize browser tools.

        Args:
            config: Optional configuration dict with keys:
                - headless: bool (default True)
                - timeout_ms: int (default 30000)
                - screenshot_dir: str (default 'screenshots')
        """
        self.config = config or {}
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._lock = asyncio.Lock()

    async def _ensure_browser(self):
        """Ensure browser is started, creating one if needed."""
        async with self._lock:
            if self._page is not None:
                return self._page

            try:
                from playwright.async_api import async_playwright
            except ImportError:
                raise ImportError(
                    "playwright not installed. Run: pip install playwright && playwright install chromium"
                )

            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=self.config.get("headless", True)
            )
            self._context = await self._browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            )
            self._page = await self._context.new_page()
            self._page.set_default_timeout(self.config.get("timeout_ms", 30000))

            return self._page

    async def browser_open(
        self,
        url: str,
        wait_for: str = "load"
    ) -> Dict[str, Any]:
        """
        Open a URL in the browser.

        Args:
            url: URL to navigate to
            wait_for: Wait condition ('load', 'domcontentloaded', 'networkidle')

        Returns:
            Dict with navigation result
        """
        try:
            page = await self._ensure_browser()
            await page.goto(url, wait_until=wait_for)

            return {
                "success": True,
                "url": page.url,
                "title": await page.title()
            }

        except ImportError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"browser_open error: {e}")
            return {"success": False, "error": str(e)}

    async def browser_screenshot(
        self,
        path: Optional[str] = None,
        full_page: bool = False,
        selector: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Take a screenshot of the page or element.

        Args:
            path: Path to save screenshot (auto-generated if not provided)
            full_page: Capture full scrollable page
            selector: CSS selector for specific element

        Returns:
            Dict with screenshot path
        """
        try:
            page = await self._ensure_browser()

            # Generate path if not provided
            if not path:
                screenshot_dir = Path(self.config.get("screenshot_dir", "screenshots"))
                screenshot_dir.mkdir(parents=True, exist_ok=True)
                path = str(screenshot_dir / f"screenshot_{uuid.uuid4().hex[:8]}.png")

            if selector:
                element = await page.query_selector(selector)
                if element:
                    await element.screenshot(path=path)
                else:
                    return {"success": False, "error": f"Element not found: {selector}"}
            else:
                await page.screenshot(path=path, full_page=full_page)

            return {
                "success": True,
                "path": path,
                "url": page.url
            }

        except Exception as e:
            logger.error(f"browser_screenshot error: {e}")
            return {"success": False, "error": str(e)}

    async def browser_click(
        self,
        selector: str,
        timeout: int = 30000
    ) -> Dict[str, Any]:
        """
        Click an element.

        Args:
            selector: CSS selector for element
            timeout: Timeout in milliseconds

        Returns:
            Dict with click result
        """
        try:
            page = await self._ensure_browser()
            await page.click(selector, timeout=timeout)

            return {
                "success": True,
                "clicked": selector,
                "url": page.url
            }

        except Exception as e:
            logger.error(f"browser_click error: {e}")
            return {"success": False, "error": str(e)}

    async def browser_type(
        self,
        selector: str,
        text: str,
        clear_first: bool = True,
        press_enter: bool = False
    ) -> Dict[str, Any]:
        """
        Type text into an input field.

        Args:
            selector: CSS selector for input
            text: Text to type
            clear_first: Clear existing text first
            press_enter: Press Enter after typing

        Returns:
            Dict with type result
        """
        try:
            page = await self._ensure_browser()

            if clear_first:
                await page.fill(selector, text)
            else:
                await page.type(selector, text)

            if press_enter:
                await page.press(selector, "Enter")

            return {
                "success": True,
                "selector": selector,
                "typed_length": len(text)
            }

        except Exception as e:
            logger.error(f"browser_type error: {e}")
            return {"success": False, "error": str(e)}

    async def browser_extract(
        self,
        selector: str,
        attribute: str = "textContent",
        multiple: bool = True
    ) -> Dict[str, Any]:
        """
        Extract data from page elements.

        Args:
            selector: CSS selector for elements
            attribute: Attribute to extract
            multiple: Extract from all matching elements

        Returns:
            Dict with extracted data
        """
        try:
            page = await self._ensure_browser()

            if multiple:
                elements = await page.query_selector_all(selector)
                results = []
                for el in elements:
                    if attribute == "textContent":
                        value = await el.text_content()
                    elif attribute == "innerHTML":
                        value = await el.inner_html()
                    else:
                        value = await el.get_attribute(attribute)
                    results.append(value)
            else:
                element = await page.query_selector(selector)
                if element:
                    if attribute == "textContent":
                        results = [await element.text_content()]
                    elif attribute == "innerHTML":
                        results = [await element.inner_html()]
                    else:
                        results = [await element.get_attribute(attribute)]
                else:
                    results = []

            return {
                "success": True,
                "selector": selector,
                "attribute": attribute,
                "count": len(results),
                "data": results
            }

        except Exception as e:
            logger.error(f"browser_extract error: {e}")
            return {"success": False, "error": str(e)}

    async def browser_get_text(
        self,
        max_length: int = 50000
    ) -> Dict[str, Any]:
        """
        Get all visible text from the page.

        Args:
            max_length: Maximum text length to return

        Returns:
            Dict with page text
        """
        try:
            page = await self._ensure_browser()

            # Get text content from body
            text = await page.evaluate("""
                () => {
                    const body = document.body;
                    // Remove script and style content
                    const scripts = body.querySelectorAll('script, style, noscript');
                    scripts.forEach(s => s.remove());
                    return body.innerText;
                }
            """)

            if len(text) > max_length:
                text = text[:max_length] + "\n\n[Content truncated...]"

            return {
                "success": True,
                "url": page.url,
                "title": await page.title(),
                "length": len(text),
                "text": text
            }

        except Exception as e:
            logger.error(f"browser_get_text error: {e}")
            return {"success": False, "error": str(e)}

    async def browser_scroll(
        self,
        direction: str = "down",
        amount: int = 500
    ) -> Dict[str, Any]:
        """
        Scroll the page.

        Args:
            direction: 'down', 'up', 'top', 'bottom'
            amount: Pixels to scroll (for up/down)

        Returns:
            Dict with scroll result
        """
        try:
            page = await self._ensure_browser()

            if direction == "down":
                await page.evaluate(f"window.scrollBy(0, {amount})")
            elif direction == "up":
                await page.evaluate(f"window.scrollBy(0, -{amount})")
            elif direction == "top":
                await page.evaluate("window.scrollTo(0, 0)")
            elif direction == "bottom":
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")

            # Get current scroll position
            scroll_y = await page.evaluate("window.scrollY")

            return {
                "success": True,
                "direction": direction,
                "scroll_position": scroll_y
            }

        except Exception as e:
            logger.error(f"browser_scroll error: {e}")
            return {"success": False, "error": str(e)}

    async def browser_close(self) -> Dict[str, Any]:
        """
        Close the browser and clean up resources.

        Returns:
            Dict with close result
        """
        try:
            async with self._lock:
                if self._browser:
                    await self._browser.close()
                if self._playwright:
                    await self._playwright.stop()

                self._page = None
                self._context = None
                self._browser = None
                self._playwright = None

            return {"success": True, "message": "Browser closed"}

        except Exception as e:
            logger.error(f"browser_close error: {e}")
            return {"success": False, "error": str(e)}

    async def cleanup(self):
        """Cleanup method for graceful shutdown."""
        await self.browser_close()
