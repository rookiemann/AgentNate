"""
Web Tools - Multi-engine web search and URL fetching.

Search engines supported:
- Google Custom Search (requires API key + cx)
- Serper.dev (requires API key)
- DuckDuckGo (no key needed, always available)

Key rotation: when multiple keys are configured per engine,
they rotate round-robin on each call.
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger("tools.web")

_SEARCH_PARAMS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Search query"
        },
        "num_results": {
            "type": "integer",
            "description": "Number of results (default 5, max 20)",
            "default": 5
        },
    },
    "required": ["query"]
}

TOOL_DEFINITIONS = [
    {
        "name": "web_search",
        "description": "Search the web using the default configured engine (Google, Serper, or DuckDuckGo). Use this for general searches.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results (default 5, max 20)",
                    "default": 5
                },
                "region": {
                    "type": "string",
                    "description": "Region for results (e.g., 'us-en'). Only used with DuckDuckGo.",
                    "default": "wt-wt"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "google_search",
        "description": "Search specifically using Google Custom Search API. Higher quality results. Requires API key in settings.",
        "parameters": _SEARCH_PARAMS
    },
    {
        "name": "serper_search",
        "description": "Search specifically using Serper.dev Google Search API. Fast, structured results. Requires API key in settings.",
        "parameters": _SEARCH_PARAMS
    },
    {
        "name": "duckduckgo_search",
        "description": "Search specifically using DuckDuckGo. No API key needed. Good for privacy-focused searches.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results (default 5, max 20)",
                    "default": 5
                },
                "region": {
                    "type": "string",
                    "description": "Region code (e.g., 'us-en', 'uk-en'). Default 'wt-wt' (worldwide).",
                    "default": "wt-wt"
                },
            },
            "required": ["query"]
        }
    },
    {
        "name": "fetch_url",
        "description": "Fetch a URL and return its content as text or markdown. Uses browser-like headers to bypass basic anti-bot. Useful for reading web pages.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch"
                },
                "extract_text": {
                    "type": "boolean",
                    "description": "Extract main text content (default true). If false, returns raw HTML.",
                    "default": True
                },
                "timeout": {
                    "type": "integer",
                    "description": "Request timeout in seconds (default 30)",
                    "default": 30
                },
                "max_length": {
                    "type": "integer",
                    "description": "Maximum content length to return (default 50000 chars)",
                    "default": 50000
                }
            },
            "required": ["url"]
        }
    },
]


class WebTools:
    """Tools for web search and URL fetching with multi-engine support."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        # Key rotation indices (in-memory, reset on restart)
        self._key_indices: Dict[str, int] = {}

    def _get_search_config(self) -> Dict[str, Any]:
        """Get search config, handling both old and new formats."""
        return self.config.get("search", self.config)

    def _rotate_key(self, engine: str, keys: List[Dict]) -> Optional[Dict]:
        """Pick next key via round-robin rotation. Returns the key dict."""
        if not keys:
            return None
        idx = self._key_indices.get(engine, 0) % len(keys)
        self._key_indices[engine] = idx + 1
        return keys[idx]

    # ======================== Public tools ========================

    async def web_search(
        self,
        query: str,
        num_results: int = 5,
        region: str = "wt-wt"
    ) -> Dict[str, Any]:
        """Search using the default configured engine."""
        num_results = min(max(num_results, 1), 20)
        search_config = self._get_search_config()
        default = search_config.get("default_engine", "duckduckgo")

        # Try the default engine first
        if default == "google":
            result = await self.google_search(query, num_results)
            if result.get("success"):
                return result
            logger.warning(f"Default Google search failed, falling back: {result.get('error')}")
        elif default == "serper":
            result = await self.serper_search(query, num_results)
            if result.get("success"):
                return result
            logger.warning(f"Default Serper search failed, falling back: {result.get('error')}")

        # Fallback to DuckDuckGo
        return await self.duckduckgo_search(query, num_results, region)

    async def google_search(
        self,
        query: str,
        num_results: int = 5
    ) -> Dict[str, Any]:
        """Search using Google Custom Search API with key rotation."""
        num_results = min(max(num_results, 1), 20)
        search_config = self._get_search_config()
        google_config = search_config.get("google", {})

        if not google_config.get("enabled"):
            return {"success": False, "error": "Google Search not enabled in settings"}

        keys = google_config.get("keys", [])
        # Backwards compat: single key format
        if not keys and google_config.get("api_key"):
            keys = [{"api_key": google_config["api_key"], "cx": google_config.get("cx", "")}]

        if not keys:
            return {"success": False, "error": "No Google Search API keys configured"}

        # Try keys with rotation (try all before giving up)
        last_error = ""
        for _ in range(len(keys)):
            key_info = self._rotate_key("google", keys)
            if not key_info or not key_info.get("api_key") or not key_info.get("cx"):
                continue

            result = await self._google_search_with_key(
                query, num_results, key_info["api_key"], key_info["cx"]
            )
            if result.get("success"):
                result["key_label"] = key_info.get("label", "")
                return result
            last_error = result.get("error", "Unknown error")
            logger.warning(f"Google key '{key_info.get('label', '?')}' failed: {last_error}")

        return {"success": False, "error": f"All Google keys failed. Last error: {last_error}"}

    async def serper_search(
        self,
        query: str,
        num_results: int = 5
    ) -> Dict[str, Any]:
        """Search using Serper.dev API with key rotation."""
        num_results = min(max(num_results, 1), 20)
        search_config = self._get_search_config()
        serper_config = search_config.get("serper", {})

        if not serper_config.get("enabled"):
            return {"success": False, "error": "Serper not enabled in settings"}

        keys = serper_config.get("keys", [])
        if not keys:
            return {"success": False, "error": "No Serper API keys configured"}

        last_error = ""
        for _ in range(len(keys)):
            key_info = self._rotate_key("serper", keys)
            if not key_info or not key_info.get("api_key"):
                continue

            result = await self._serper_search_with_key(
                query, num_results, key_info["api_key"]
            )
            if result.get("success"):
                result["key_label"] = key_info.get("label", "")
                return result
            last_error = result.get("error", "Unknown error")
            logger.warning(f"Serper key '{key_info.get('label', '?')}' failed: {last_error}")

        return {"success": False, "error": f"All Serper keys failed. Last error: {last_error}"}

    async def duckduckgo_search(
        self,
        query: str,
        num_results: int = 5,
        region: str = "wt-wt"
    ) -> Dict[str, Any]:
        """Search using DuckDuckGo (no API key needed)."""
        num_results = min(max(num_results, 1), 20)

        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS
        except ImportError:
            return {"success": False, "error": "ddgs not installed. Run: pip install ddgs"}

        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, region=region, max_results=num_results))

            formatted = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", r.get("link", "")),
                    "snippet": r.get("body", r.get("snippet", ""))
                }
                for r in results
            ]

            return {
                "success": True,
                "query": query,
                "engine": "duckduckgo",
                "count": len(formatted),
                "results": formatted
            }
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return {"success": False, "error": str(e)}

    # ======================== Internal engines ========================

    async def _google_search_with_key(
        self, query: str, num_results: int, api_key: str, cx: str
    ) -> Dict[str, Any]:
        """Execute a Google Custom Search API call with a specific key."""
        try:
            import httpx

            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    "https://www.googleapis.com/customsearch/v1",
                    params={
                        "key": api_key,
                        "cx": cx,
                        "q": query,
                        "num": min(num_results, 10),
                    },
                )

                if resp.status_code == 429:
                    return {"success": False, "error": "Rate limited (429)"}
                if resp.status_code == 403:
                    return {"success": False, "error": "Forbidden (403) - check API key/quota"}
                if resp.status_code != 200:
                    return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}

                data = resp.json()
                items = data.get("items", [])

                return {
                    "success": True,
                    "query": query,
                    "engine": "google",
                    "count": len(items),
                    "results": [
                        {
                            "title": item.get("title", ""),
                            "url": item.get("link", ""),
                            "snippet": item.get("snippet", ""),
                        }
                        for item in items
                    ],
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _serper_search_with_key(
        self, query: str, num_results: int, api_key: str
    ) -> Dict[str, Any]:
        """Execute a Serper.dev search with a specific key."""
        try:
            import httpx

            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    "https://google.serper.dev/search",
                    headers={
                        "X-API-KEY": api_key,
                        "Content-Type": "application/json",
                    },
                    json={"q": query, "num": min(num_results, 20)},
                )

                if resp.status_code == 429:
                    return {"success": False, "error": "Rate limited (429)"}
                if resp.status_code == 403:
                    return {"success": False, "error": "Invalid API key (403)"}
                if resp.status_code != 200:
                    return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}

                data = resp.json()
                organic = data.get("organic", [])

                results = [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("link", ""),
                        "snippet": r.get("snippet", ""),
                    }
                    for r in organic[:num_results]
                ]

                # Include knowledge graph if present
                kg = data.get("knowledgeGraph")
                answer_box = data.get("answerBox")

                result = {
                    "success": True,
                    "query": query,
                    "engine": "serper",
                    "count": len(results),
                    "results": results,
                }

                if kg:
                    result["knowledge_graph"] = {
                        "title": kg.get("title"),
                        "type": kg.get("type"),
                        "description": kg.get("description"),
                    }
                if answer_box:
                    result["answer_box"] = answer_box.get("answer") or answer_box.get("snippet")

                return result

        except Exception as e:
            return {"success": False, "error": str(e)}

    # ======================== Validation ========================

    async def validate_key(self, engine: str, api_key: str, cx: str = "") -> Dict[str, Any]:
        """Test a search API key with a simple query. Returns success/failure."""
        test_query = "test"

        if engine == "google":
            if not cx:
                return {"valid": False, "error": "Google requires both API key and Search Engine ID (cx)"}
            result = await self._google_search_with_key(test_query, 1, api_key, cx)
        elif engine == "serper":
            result = await self._serper_search_with_key(test_query, 1, api_key)
        else:
            return {"valid": False, "error": f"Unknown engine: {engine}"}

        if result.get("success"):
            return {"valid": True, "results_returned": result.get("count", 0)}
        else:
            return {"valid": False, "error": result.get("error", "Unknown error")}

    # ======================== URL fetching ========================

    async def fetch_url(
        self,
        url: str,
        extract_text: bool = True,
        timeout: int = 30,
        max_length: int = 50000
    ) -> Dict[str, Any]:
        """
        Fetch a URL and return its content.

        Args:
            url: URL to fetch
            extract_text: If True, extract main text content. If False, return raw HTML.
            timeout: Request timeout in seconds
            max_length: Maximum content length to return

        Returns:
            Dict with page content
        """
        try:
            import httpx
        except ImportError:
            return {
                "success": False,
                "error": "httpx not installed. Run: pip install httpx"
            }

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }

            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            html = response.text

            if extract_text and "text/html" in content_type:
                content = self._extract_text(html)
            else:
                content = html

            # Truncate if needed
            if len(content) > max_length:
                content = content[:max_length] + "\n\n[Content truncated...]"

            return {
                "success": True,
                "url": str(response.url),
                "status_code": response.status_code,
                "content_type": content_type,
                "length": len(content),
                "content": content
            }

        except httpx.TimeoutException:
            return {"success": False, "error": f"Request timed out after {timeout}s"}
        except httpx.HTTPStatusError as e:
            return {"success": False, "error": f"HTTP error {e.response.status_code}"}
        except Exception as e:
            logger.error(f"fetch_url error: {e}")
            return {"success": False, "error": str(e)}

    def _extract_text(self, html: str) -> str:
        """
        Extract main text content from HTML.

        Tries trafilatura first (better quality), falls back to BeautifulSoup.
        """
        # Try trafilatura first (best for article extraction)
        try:
            import trafilatura
            text = trafilatura.extract(html, include_links=True, include_formatting=True)
            if text:
                return text
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"trafilatura extraction failed: {e}")

        # Fall back to BeautifulSoup
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
                element.decompose()

            # Get text
            text = soup.get_text(separator="\n", strip=True)

            # Clean up multiple newlines
            import re
            text = re.sub(r'\n{3,}', '\n\n', text)

            return text

        except ImportError:
            return html  # Return raw HTML if no extraction lib available
        except Exception as e:
            logger.debug(f"BeautifulSoup extraction failed: {e}")
            return html
