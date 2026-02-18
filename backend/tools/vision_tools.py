"""
Vision Tools - Image analysis using vision-capable models.

These tools leverage vision-capable LLMs (like LLaVA, Qwen-VL, etc.) to analyze
images, screenshots, and visual content.
"""

from typing import Dict, Any, List, Optional, TYPE_CHECKING
import logging
import base64
import mimetypes
from pathlib import Path

if TYPE_CHECKING:
    from orchestrator.orchestrator import Orchestrator
    from backend.tools.browser_tools import BrowserTools

logger = logging.getLogger("tools.vision")


TOOL_DEFINITIONS = [
    {
        "name": "analyze_image",
        "description": "Analyze an image using a vision-capable AI model. Can describe content, extract information, answer questions about the image.",
        "parameters": {
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": "Image source: file path, URL, or base64 data URI"
                },
                "prompt": {
                    "type": "string",
                    "description": "What to analyze or question to answer about the image",
                    "default": "Describe this image in detail."
                },
                "instance_id": {
                    "type": "string",
                    "description": "Specific vision model instance to use (optional, uses default if not specified)"
                }
            },
            "required": ["image"]
        }
    },
    {
        "name": "analyze_screenshot",
        "description": "Take a screenshot of the current browser page and analyze it with vision AI. Useful for understanding page content, finding elements, or debugging.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "What to analyze or look for in the screenshot",
                    "default": "Describe what you see on this webpage. List any important elements, buttons, forms, or content."
                },
                "full_page": {
                    "type": "boolean",
                    "description": "Capture full scrollable page (default false)",
                    "default": False
                },
                "instance_id": {
                    "type": "string",
                    "description": "Specific vision model instance to use (optional)"
                }
            },
            "required": []
        }
    },
    {
        "name": "extract_text_from_image",
        "description": "Extract text/OCR from an image using vision AI. Good for screenshots, documents, signs, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": "Image source: file path, URL, or base64 data URI"
                },
                "format": {
                    "type": "string",
                    "description": "Output format: 'plain' (just text), 'structured' (preserve layout), 'markdown' (formatted)",
                    "enum": ["plain", "structured", "markdown"],
                    "default": "plain"
                },
                "instance_id": {
                    "type": "string",
                    "description": "Specific vision model instance to use (optional)"
                }
            },
            "required": ["image"]
        }
    },
    {
        "name": "describe_ui",
        "description": "Analyze a UI screenshot and identify interactive elements. Useful for browser automation - helps find buttons, links, forms, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": "Screenshot image: file path, URL, or base64"
                },
                "element_type": {
                    "type": "string",
                    "description": "Focus on specific element types (optional)",
                    "enum": ["all", "buttons", "links", "forms", "navigation", "text"],
                    "default": "all"
                },
                "instance_id": {
                    "type": "string",
                    "description": "Specific vision model instance to use (optional)"
                }
            },
            "required": ["image"]
        }
    },
    {
        "name": "compare_images",
        "description": "Compare two images and describe the differences. Useful for visual regression testing, before/after comparisons.",
        "parameters": {
            "type": "object",
            "properties": {
                "image1": {
                    "type": "string",
                    "description": "First image (before): file path, URL, or base64"
                },
                "image2": {
                    "type": "string",
                    "description": "Second image (after): file path, URL, or base64"
                },
                "focus": {
                    "type": "string",
                    "description": "What to focus on when comparing",
                    "default": "Describe all visible differences between these two images."
                },
                "instance_id": {
                    "type": "string",
                    "description": "Specific vision model instance to use (optional)"
                }
            },
            "required": ["image1", "image2"]
        }
    },
    {
        "name": "find_element",
        "description": "Find a specific element in a screenshot by description. Returns location hints for browser automation.",
        "parameters": {
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": "Screenshot image: file path, URL, or base64"
                },
                "element_description": {
                    "type": "string",
                    "description": "Description of the element to find (e.g., 'the blue Login button', 'search input field')"
                },
                "instance_id": {
                    "type": "string",
                    "description": "Specific vision model instance to use (optional)"
                }
            },
            "required": ["image", "element_description"]
        }
    },
]


class VisionTools:
    """Tools for image analysis using vision-capable models."""

    def __init__(
        self,
        orchestrator: "Orchestrator",
        browser_tools: Optional["BrowserTools"] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize vision tools.

        Args:
            orchestrator: The model orchestrator for running inference
            browser_tools: Optional browser tools instance for screenshot integration
            config: Optional configuration dict with keys:
                - default_instance_id: str - Default vision model instance
                - max_tokens: int - Max response tokens
                - temperature: float - Model temperature
        """
        self.orchestrator = orchestrator
        self.browser_tools = browser_tools
        self.config = config or {}
        self.default_instance_id = self.config.get("default_instance_id")
        self.max_tokens = self.config.get("max_tokens", 1024)
        self.temperature = self.config.get("temperature", 0.3)

    def _prepare_image(self, image: str) -> str:
        """
        Prepare image for the model.

        Converts file paths to base64 data URIs if needed.

        Args:
            image: File path, URL, or base64 data URI

        Returns:
            Image in a format ready for the model
        """
        # Already a data URI or URL
        if image.startswith(("data:", "http://", "https://")):
            return image

        # File path - convert to base64
        path = Path(image)
        if path.exists():
            mime_type, _ = mimetypes.guess_type(str(path))
            if not mime_type:
                mime_type = "image/png"

            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")

            return f"data:{mime_type};base64,{b64}"

        # Assume it's a path that the model can handle directly
        return image

    async def _get_instance_id(self, instance_id: Optional[str] = None) -> Optional[str]:
        """
        Get a valid vision model instance ID.

        Args:
            instance_id: Explicitly requested instance ID

        Returns:
            Instance ID or None if no vision model available
        """
        if instance_id:
            return instance_id

        if self.default_instance_id:
            return self.default_instance_id

        # Try to find any loaded model (ideally we'd check for vision capability)
        instances = self.orchestrator.get_loaded_instances()
        if instances:
            # Return first available instance
            # TODO: In future, could check model metadata for vision capability
            return instances[0].id

        return None

    async def _run_vision_inference(
        self,
        prompt: str,
        images: List[str],
        instance_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run vision inference with the given prompt and images.

        Args:
            prompt: Text prompt
            images: List of image sources
            instance_id: Optional specific instance to use

        Returns:
            Dict with success status and response
        """
        from providers.base import ChatMessage, InferenceRequest

        inst_id = await self._get_instance_id(instance_id)
        if not inst_id:
            return {
                "success": False,
                "error": "No vision-capable model loaded. Load a vision model first."
            }

        try:
            # Prepare images
            prepared_images = [self._prepare_image(img) for img in images]

            # Create message with images
            message = ChatMessage(
                role="user",
                content=prompt,
                images=prepared_images
            )

            # Create request
            request = InferenceRequest(
                messages=[message],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            # Run inference (collect streaming response)
            full_response = ""
            async for response in self.orchestrator.chat(inst_id, request):
                if response.error:
                    return {"success": False, "error": response.error}
                if response.text:
                    full_response += response.text

            return {
                "success": True,
                "response": full_response.strip(),
                "instance_id": inst_id
            }

        except Exception as e:
            logger.error(f"Vision inference error: {e}")
            return {"success": False, "error": str(e)}

    async def analyze_image(
        self,
        image: str,
        prompt: str = "Describe this image in detail.",
        instance_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze an image with a vision model.

        Args:
            image: Image source (file path, URL, or base64)
            prompt: Analysis prompt
            instance_id: Optional specific model instance

        Returns:
            Dict with analysis result
        """
        result = await self._run_vision_inference(
            prompt=prompt,
            images=[image],
            instance_id=instance_id
        )

        if result["success"]:
            return {
                "success": True,
                "image": image[:100] + "..." if len(image) > 100 else image,
                "analysis": result["response"],
                "instance_id": result["instance_id"]
            }
        return result

    async def analyze_screenshot(
        self,
        prompt: str = "Describe what you see on this webpage. List any important elements, buttons, forms, or content.",
        full_page: bool = False,
        instance_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Take a screenshot and analyze it.

        Args:
            prompt: Analysis prompt
            full_page: Capture full page
            instance_id: Optional specific model instance

        Returns:
            Dict with screenshot path and analysis
        """
        if not self.browser_tools:
            return {
                "success": False,
                "error": "Browser tools not available. Open a page with browser_open first."
            }

        # Take screenshot
        screenshot_result = await self.browser_tools.browser_screenshot(
            full_page=full_page
        )

        if not screenshot_result.get("success"):
            return screenshot_result

        screenshot_path = screenshot_result["path"]

        # Analyze it
        result = await self._run_vision_inference(
            prompt=prompt,
            images=[screenshot_path],
            instance_id=instance_id
        )

        if result["success"]:
            return {
                "success": True,
                "screenshot_path": screenshot_path,
                "url": screenshot_result.get("url"),
                "analysis": result["response"],
                "instance_id": result["instance_id"]
            }
        return result

    async def extract_text_from_image(
        self,
        image: str,
        format: str = "plain",
        instance_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract text (OCR) from an image.

        Args:
            image: Image source
            format: Output format (plain, structured, markdown)
            instance_id: Optional specific model instance

        Returns:
            Dict with extracted text
        """
        format_prompts = {
            "plain": "Extract all text from this image. Return only the text content, nothing else.",
            "structured": "Extract all text from this image, preserving the layout and structure as much as possible. Use spacing and line breaks to maintain the original positioning.",
            "markdown": "Extract all text from this image and format it as markdown. Use headers, lists, and formatting as appropriate based on the visual hierarchy."
        }

        prompt = format_prompts.get(format, format_prompts["plain"])

        result = await self._run_vision_inference(
            prompt=prompt,
            images=[image],
            instance_id=instance_id
        )

        if result["success"]:
            return {
                "success": True,
                "format": format,
                "text": result["response"],
                "instance_id": result["instance_id"]
            }
        return result

    async def describe_ui(
        self,
        image: str,
        element_type: str = "all",
        instance_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze UI elements in a screenshot.

        Args:
            image: Screenshot image
            element_type: Type of elements to focus on
            instance_id: Optional specific model instance

        Returns:
            Dict with UI element descriptions
        """
        type_prompts = {
            "all": """Analyze this UI screenshot and list all interactive elements you can see.
For each element, describe:
1. What type of element it is (button, link, input, etc.)
2. What text or label it has
3. Its approximate location (top-left, center, bottom-right, etc.)
4. Any CSS selectors that might help identify it (class names, IDs visible)

Format as a numbered list.""",

            "buttons": "List all buttons in this screenshot. For each button, describe its text/label, color, and location.",

            "links": "List all clickable links in this screenshot. Include the link text and approximate location.",

            "forms": "Identify all form elements in this screenshot: input fields, dropdowns, checkboxes, submit buttons. Describe each with its label and location.",

            "navigation": "Identify the navigation elements in this screenshot: menus, nav bars, breadcrumbs, tabs. Describe the structure and options available.",

            "text": "Identify the main text content areas in this screenshot. Describe headers, paragraphs, and text hierarchy."
        }

        prompt = type_prompts.get(element_type, type_prompts["all"])

        result = await self._run_vision_inference(
            prompt=prompt,
            images=[image],
            instance_id=instance_id
        )

        if result["success"]:
            return {
                "success": True,
                "element_type": element_type,
                "elements": result["response"],
                "instance_id": result["instance_id"]
            }
        return result

    async def compare_images(
        self,
        image1: str,
        image2: str,
        focus: str = "Describe all visible differences between these two images.",
        instance_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare two images and describe differences.

        Args:
            image1: First image (before)
            image2: Second image (after)
            focus: What to focus on when comparing
            instance_id: Optional specific model instance

        Returns:
            Dict with comparison results
        """
        prompt = f"""I'm showing you two images for comparison.

IMAGE 1 (First/Before):
[First image attached]

IMAGE 2 (Second/After):
[Second image attached]

{focus}

Be specific about:
- What changed or is different
- What stayed the same
- The significance of any changes"""

        result = await self._run_vision_inference(
            prompt=prompt,
            images=[image1, image2],
            instance_id=instance_id
        )

        if result["success"]:
            return {
                "success": True,
                "comparison": result["response"],
                "instance_id": result["instance_id"]
            }
        return result

    async def find_element(
        self,
        image: str,
        element_description: str,
        instance_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Find a specific element in a screenshot.

        Args:
            image: Screenshot image
            element_description: Description of element to find
            instance_id: Optional specific model instance

        Returns:
            Dict with element location information
        """
        prompt = f"""Look at this screenshot and find the following element:
"{element_description}"

If you find it, describe:
1. FOUND: Yes/No
2. LOCATION: Where it is on the screen (e.g., "top-right corner", "center of page", "in the navigation bar")
3. VISUAL DESCRIPTION: What it looks like (color, size, surrounding elements)
4. SELECTOR HINTS: Any text, IDs, or class names visible that could help select it programmatically
5. CONFIDENCE: How certain you are (high/medium/low)

If multiple matches exist, describe each one."""

        result = await self._run_vision_inference(
            prompt=prompt,
            images=[image],
            instance_id=instance_id
        )

        if result["success"]:
            return {
                "success": True,
                "element_description": element_description,
                "result": result["response"],
                "instance_id": result["instance_id"]
            }
        return result
