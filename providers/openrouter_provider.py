"""
OpenRouter Provider - Cloud API with 100+ models.

Requires API key from https://openrouter.ai
"""

import asyncio
import json
from typing import AsyncIterator, Optional, List, Dict, Any
import time

try:
    import aiohttp
except ImportError:
    aiohttp = None

from .base import (
    BaseProvider,
    ProviderType,
    ModelInstance,
    ModelStatus,
    InferenceRequest,
    InferenceResponse,
)


class OpenRouterProvider(BaseProvider):
    """
    Provider for OpenRouter API.

    OpenRouter provides access to 100+ models through a unified API.
    Requires an API key.

    Rate Limits:
    - Free models: 20 req/min, 50-1000 req/day depending on credits purchased
    - Paid models: $1 balance = 1 RPS, up to 500 RPS max
    - ~50 concurrent requests typical cap
    """

    BASE_URL = "https://openrouter.ai/api/v1"

    # Retry configuration for rate limits
    MAX_RETRIES = 5
    BASE_RETRY_DELAY = 1.0  # seconds
    MAX_RETRY_DELAY = 30.0  # seconds

    def __init__(
        self,
        api_key: str = "",
        site_url: str = "https://agentnate.local",
        app_name: str = "AgentNate",
    ):
        super().__init__(ProviderType.OPENROUTER)
        self.api_key = api_key
        self.site_url = site_url
        self.app_name = app_name
        self._session: Optional[aiohttp.ClientSession] = None
        self._models_cache: List[Dict] = []
        self._cache_time: float = 0

    def _get_retry_delay(self, attempt: int, retry_after: Optional[str] = None) -> float:
        """Calculate retry delay with exponential backoff."""
        if retry_after:
            try:
                # Retry-After can be seconds or HTTP date
                return min(float(retry_after), self.MAX_RETRY_DELAY)
            except ValueError:
                pass
        # Exponential backoff: 1, 2, 4, 8, 16... capped at MAX_RETRY_DELAY
        delay = self.BASE_RETRY_DELAY * (2 ** attempt)
        return min(delay, self.MAX_RETRY_DELAY)

    def set_api_key(self, key: str):
        """Update API key."""
        self.api_key = key

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session with auth headers."""
        if aiohttp is None:
            raise ImportError("aiohttp is required for OpenRouter provider")

        if self._session is None or self._session.closed:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": self.site_url,
                "X-Title": self.app_name,
                "Content-Type": "application/json",
            }
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session

    async def load_model(
        self,
        model_identifier: str,
        **kwargs
    ) -> ModelInstance:
        """
        Create an instance for an OpenRouter model.

        OpenRouter doesn't require loading - models are available on-demand.
        This creates a reference instance for tracking.

        Args:
            model_identifier: Model ID (e.g., "openai/gpt-4", "anthropic/claude-3")
        """
        import logging
        logger = logging.getLogger(__name__)

        logger.info(f"[OpenRouter DEBUG] load_model called with: {model_identifier}")
        logger.info(f"[OpenRouter DEBUG] kwargs: {kwargs}")

        if not self.api_key:
            logger.error("[OpenRouter DEBUG] No API key configured!")
            raise ValueError("OpenRouter API key not configured")

        # Get model info if available
        model_info = None
        has_vision = False
        logger.info("[OpenRouter DEBUG] Fetching models list...")
        models = await self.list_models()
        logger.info(f"[OpenRouter DEBUG] Got {len(models)} models from list_models")

        for m in models:
            if m.get("id") == model_identifier:
                model_info = m
                has_vision = m.get("has_vision", False)
                logger.info(f"[OpenRouter DEBUG] Found model info: context_length={m.get('context_length')}, has_vision={has_vision}")
                break

        # If not found in cache, try to detect from model name
        if model_info is None:
            logger.warning(f"[OpenRouter DEBUG] Model {model_identifier} not found in list, using fallback detection")
            has_vision = self._detect_vision({"id": model_identifier})

        display_name = model_identifier.split("/")[-1]
        context_length = 4096  # Default
        if model_info:
            display_name = model_info.get("name", display_name)
            # Get context length from model info (returned by list_models)
            raw_context = model_info.get("context_length")
            context_length = raw_context or 4096
            logger.info(f"[OpenRouter DEBUG] Raw context_length from model_info: {raw_context}, using: {context_length}")

        logger.info(f"[OpenRouter DEBUG] Creating ModelInstance: display_name={display_name}, context_length={context_length}")

        instance = ModelInstance(
            provider_type=self.provider_type,
            model_identifier=model_identifier,
            display_name=f"OpenRouter: {display_name}",
            status=ModelStatus.READY,
            context_length=context_length,
            metadata={
                "openrouter_model": model_identifier,
                "model_info": model_info,
                "has_vision": has_vision,
            }
        )

        self.instances[instance.id] = instance
        logger.info(f"[OpenRouter DEBUG] Instance created: id={instance.id}, context_length={instance.context_length}")
        return instance

    async def unload_model(self, instance_id: str) -> bool:
        """Remove instance tracking."""
        if instance_id in self.instances:
            del self.instances[instance_id]
            return True
        return False

    async def chat(
        self,
        instance_id: str,
        request: InferenceRequest
    ) -> AsyncIterator[InferenceResponse]:
        """Stream via OpenRouter SSE with retry logic for rate limits."""
        import logging
        logger = logging.getLogger(__name__)

        instance = self.instances.get(instance_id)
        if not instance:
            yield InferenceResponse(
                request_id=request.request_id,
                error="Instance not found"
            )
            return

        if not self.api_key:
            yield InferenceResponse(
                request_id=request.request_id,
                error="API key not configured"
            )
            return

        session = await self._get_session()
        start_time = time.time()

        payload = {
            "model": instance.model_identifier,
            "messages": request.to_messages_dict(),
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
            "repetition_penalty": request.repeat_penalty,
            "stream": True,
        }

        if request.stop:
            payload["stop"] = request.stop

        # Retry loop for rate limit handling
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                async with session.post(
                    f"{self.BASE_URL}/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as response:
                    # Handle rate limits with retry
                    if response.status == 429:
                        retry_after = response.headers.get("Retry-After")
                        delay = self._get_retry_delay(attempt, retry_after)
                        logger.warning(
                            f"[OpenRouter] Rate limited (429), attempt {attempt + 1}/{self.MAX_RETRIES}, "
                            f"retrying in {delay:.1f}s"
                        )
                        if attempt < self.MAX_RETRIES - 1:
                            await asyncio.sleep(delay)
                            continue
                        else:
                            error_text = await response.text()
                            yield InferenceResponse(
                                request_id=request.request_id,
                                error=f"Rate limit exceeded after {self.MAX_RETRIES} retries: {error_text}"
                            )
                            return

                    # Handle other errors (no retry)
                    if response.status != 200:
                        error_text = await response.text()
                        yield InferenceResponse(
                            request_id=request.request_id,
                            error=f"OpenRouter error {response.status}: {error_text}"
                        )
                        return

                    # Success - process stream
                    async for line in response.content:
                        line = line.decode().strip()

                        if not line:
                            continue

                        if line.startswith("data: "):
                            data = line[6:]

                            if data == "[DONE]":
                                yield InferenceResponse(
                                    request_id=request.request_id,
                                    done=True,
                                    total_time=time.time() - start_time
                                )
                                return  # Success, exit retry loop

                            try:
                                chunk = json.loads(data)

                                # Check for errors in response
                                if "error" in chunk:
                                    yield InferenceResponse(
                                        request_id=request.request_id,
                                        error=chunk["error"].get("message", str(chunk["error"]))
                                    )
                                    return

                                choices = chunk.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        yield InferenceResponse(
                                            request_id=request.request_id,
                                            text=content
                                        )

                                    # Check for finish reason
                                    if choices[0].get("finish_reason"):
                                        usage = chunk.get("usage", {})
                                        yield InferenceResponse(
                                            request_id=request.request_id,
                                            done=True,
                                            total_time=time.time() - start_time,
                                            usage=usage
                                        )
                                        return  # Success, exit retry loop

                            except json.JSONDecodeError:
                                pass

                    # If we got here without returning, stream completed
                    return

            except aiohttp.ClientError as e:
                last_error = f"Connection error: {str(e)}"
                if attempt < self.MAX_RETRIES - 1:
                    delay = self._get_retry_delay(attempt)
                    logger.warning(f"[OpenRouter] Connection error, retrying in {delay:.1f}s: {e}")
                    await asyncio.sleep(delay)
                    continue
            except asyncio.TimeoutError:
                last_error = "Request timed out"
                if attempt < self.MAX_RETRIES - 1:
                    delay = self._get_retry_delay(attempt)
                    logger.warning(f"[OpenRouter] Timeout, retrying in {delay:.1f}s")
                    await asyncio.sleep(delay)
                    continue
            except asyncio.CancelledError:
                logger.info(f"[OpenRouter] Generation cancelled for request {request.request_id}")
                yield InferenceResponse(
                    request_id=request.request_id,
                    error="Generation cancelled"
                )
                return

        # All retries exhausted
        yield InferenceResponse(
            request_id=request.request_id,
            error=last_error or "Request failed after retries"
        )

    async def get_credits(self) -> Dict[str, Any]:
        """
        Get current credit/usage info from OpenRouter.

        Returns dict with usage info. For prepaid accounts includes balance.
        For pay-as-you-go accounts shows usage totals.
        """
        if not self.api_key:
            return {"error": "No API key"}

        session = await self._get_session()

        try:
            async with session.get(
                "https://openrouter.ai/api/v1/auth/key",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    data = result.get("data", {})

                    return {
                        "label": data.get("label"),
                        "usage_total": data.get("usage"),  # Total usage in $
                        "usage_monthly": data.get("usage_monthly"),
                        "limit": data.get("limit"),  # None for pay-as-you-go
                        "limit_remaining": data.get("limit_remaining"),
                        "is_free_tier": data.get("is_free_tier", False),
                    }
                return {"error": f"HTTP {response.status}"}
        except Exception as e:
            return {"error": str(e)}

    async def list_free_models(self) -> List[Dict[str, Any]]:
        """List only free models (no credit usage)."""
        models = await self.list_models()
        free = []

        for m in models:
            model_id = m.get("id", "")
            pricing = m.get("pricing", {})

            # Check for :free suffix or zero pricing
            prompt_price = float(pricing.get("prompt") or 0)
            completion_price = float(pricing.get("completion") or 0)

            if ":free" in model_id or (prompt_price == 0 and completion_price == 0):
                free.append(m)

        return free

    def _detect_vision(self, model_data: Dict) -> bool:
        """
        Detect if model supports vision/image input.

        OpenRouter provides architecture.input_modalities which includes
        "image" for vision-capable models.
        """
        arch = model_data.get("architecture", {})
        input_mods = arch.get("input_modalities", [])

        # Check if image is in input modalities
        if "image" in input_mods:
            return True

        # Fallback: check modality string for image
        modality = arch.get("modality", "")
        if "image" in modality.lower():
            return True

        # Fallback: check model name patterns
        model_id = model_data.get("id", "").lower()
        vision_patterns = [
            "vision", "llava", "gpt-4o", "gpt-4-turbo", "claude-3",
            "gemini-pro-vision", "gemini-1.5", "gemini-2", "pixtral",
            "qwen-vl", "qwen2-vl", "cogvlm", "internvl", "molmo"
        ]
        return any(p in model_id for p in vision_patterns)

    async def list_models(self) -> List[Dict[str, Any]]:
        """
        Fetch available models from OpenRouter.

        Caches results for 5 minutes.
        """
        import logging
        logger = logging.getLogger(__name__)

        # Check cache
        if self._models_cache and (time.time() - self._cache_time) < 300:
            logger.debug(f"[OpenRouter DEBUG] Returning {len(self._models_cache)} models from cache")
            return self._models_cache

        if not self.api_key:
            logger.warning("[OpenRouter DEBUG] list_models: No API key, returning empty list")
            return []

        session = await self._get_session()

        try:
            logger.info("[OpenRouter DEBUG] Fetching models from OpenRouter API...")
            async with session.get(
                f"{self.BASE_URL}/models",
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                logger.info(f"[OpenRouter DEBUG] API response status: {response.status}")
                if response.status == 200:
                    data = await response.json()
                    models = data.get("data", [])
                    logger.info(f"[OpenRouter DEBUG] Got {len(models)} models from API")

                    self._models_cache = [
                        {
                            "id": m.get("id"),
                            "name": m.get("name", m.get("id")),
                            "provider": "openrouter",
                            "context_length": m.get("context_length"),
                            "pricing": m.get("pricing"),
                            "description": m.get("description"),
                            "has_vision": self._detect_vision(m),
                            "input_modalities": m.get("architecture", {}).get("input_modalities", []),
                        }
                        for m in models
                    ]
                    self._cache_time = time.time()

                    # Log a few sample models with context lengths
                    for m in self._models_cache[:3]:
                        logger.info(f"[OpenRouter DEBUG] Sample model: {m.get('id')} - context_length={m.get('context_length')}")

                    return self._models_cache
                else:
                    error_text = await response.text()
                    logger.error(f"[OpenRouter DEBUG] API error {response.status}: {error_text[:200]}")

        except Exception as e:
            logger.error(f"[OpenRouter DEBUG] list_models exception: {e}")

        return self._models_cache or []

    async def health_check(self) -> Dict[str, Any]:
        """Check OpenRouter connectivity."""
        if not self.api_key:
            return {
                "provider": "openrouter",
                "status": "no_api_key",
                "connected": False,
            }

        try:
            session = await self._get_session()

            async with session.get(
                f"{self.BASE_URL}/models",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                return {
                    "provider": "openrouter",
                    "status": "healthy" if response.status == 200 else "error",
                    "connected": response.status == 200,
                    "model_count": len(self._models_cache),
                }

        except Exception as e:
            return {
                "provider": "openrouter",
                "status": "offline",
                "error": str(e),
                "connected": False,
            }

    async def get_status(self, instance_id: str) -> ModelStatus:
        """Get instance status."""
        if instance_id not in self.instances:
            return ModelStatus.UNLOADED
        return ModelStatus.READY

    async def close(self):
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
        await super().close()
