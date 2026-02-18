"""
Ollama Provider - Local Ollama API at localhost:11434.

Ollama manages model downloads and loading internally.
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


class OllamaProvider(BaseProvider):
    """
    Provider for Ollama API.

    Ollama manages model downloads and loading. Models are loaded
    on first request and can be kept warm with keep_alive.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        keep_alive: str = "5m",
    ):
        super().__init__(ProviderType.OLLAMA)
        self.base_url = base_url.rstrip("/")
        self.keep_alive = keep_alive
        self._session: Optional[aiohttp.ClientSession] = None

    def _is_vision_model(self, model_name: str) -> bool:
        """Detect if model supports vision based on name patterns."""
        vision_patterns = [
            'vision', 'llava', 'bakllava', 'moondream', 'cogvlm',
            'minicpm-v', 'llama-3.2-vision', 'qwen-vl', 'qwen2-vl',
            'internvl', 'molmo', 'pixtral', 'gemma-3',  # gemma-3 has vision
        ]
        model_lower = model_name.lower()
        return any(p in model_lower for p in vision_patterns)

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if aiohttp is None:
            raise ImportError("aiohttp is required for Ollama provider")

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def load_model(
        self,
        model_identifier: str,
        pre_warm: bool = True,
        **kwargs
    ) -> ModelInstance:
        """
        Load/pre-warm an Ollama model.

        Ollama loads models on first request. This optionally
        pre-warms the model with an empty prompt.

        Args:
            model_identifier: Model name (e.g., "llama3:8b", "codellama")
            pre_warm: If True, warm up the model immediately
            **kwargs:
                context_length/n_ctx: Context window size (passed as num_ctx at inference)
        """
        # Accept both context_length and n_ctx (llama.cpp style)
        context_length = kwargs.get("context_length") or kwargs.get("n_ctx")

        # Detect vision capability
        has_vision = self._is_vision_model(model_identifier)

        instance = ModelInstance(
            provider_type=self.provider_type,
            model_identifier=model_identifier,
            display_name=f"Ollama: {model_identifier}",
            status=ModelStatus.LOADING,
            context_length=context_length,
            metadata={
                "num_ctx": context_length,  # Ollama's parameter name
                "has_vision": has_vision,
            } if context_length else {"has_vision": has_vision},
        )
        self.instances[instance.id] = instance

        if pre_warm:
            # Pre-warm the model with empty generate
            session = await self._get_session()

            try:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model_identifier,
                        "prompt": "",
                        "keep_alive": self.keep_alive,
                    },
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status == 200:
                        instance.status = ModelStatus.READY
                    else:
                        error_text = await response.text()
                        instance.status = ModelStatus.ERROR
                        instance.metadata["error"] = error_text

            except Exception as e:
                instance.status = ModelStatus.ERROR
                instance.metadata["error"] = str(e)
        else:
            instance.status = ModelStatus.READY

        return instance

    async def unload_model(self, instance_id: str) -> bool:
        """
        Unload a model from Ollama.

        Sets keep_alive to 0 to unload immediately.
        """
        instance = self.instances.get(instance_id)
        if instance:
            session = await self._get_session()

            try:
                # Tell Ollama to unload
                await session.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": instance.model_identifier,
                        "prompt": "",
                        "keep_alive": "0",
                    },
                    timeout=aiohttp.ClientTimeout(total=10)
                )
            except Exception:
                pass

            del self.instances[instance_id]
            return True

        return False

    async def chat(
        self,
        instance_id: str,
        request: InferenceRequest
    ) -> AsyncIterator[InferenceResponse]:
        """Stream via Ollama /api/chat."""
        instance = self.instances.get(instance_id)
        if not instance:
            yield InferenceResponse(
                request_id=request.request_id,
                error="Instance not found"
            )
            return

        session = await self._get_session()
        start_time = time.time()

        # Build options dict
        options = {
            "num_predict": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "repeat_penalty": request.repeat_penalty,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
        }

        # Add context length if specified at load time
        num_ctx = instance.metadata.get("num_ctx") if instance.metadata else None
        if num_ctx:
            options["num_ctx"] = num_ctx

        payload = {
            "model": instance.model_identifier,
            "messages": request.to_messages_dict(),
            "stream": True,
            "keep_alive": self.keep_alive,
            "options": options,
        }

        if request.stop:
            payload["options"]["stop"] = request.stop

        # Add mirostat if enabled
        if request.mirostat > 0:
            payload["options"]["mirostat"] = request.mirostat
            payload["options"]["mirostat_tau"] = request.mirostat_tau
            payload["options"]["mirostat_eta"] = request.mirostat_eta

        try:
            async with session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    yield InferenceResponse(
                        request_id=request.request_id,
                        error=f"Ollama error {response.status}: {error_text}"
                    )
                    return

                async for line in response.content:
                    line = line.decode().strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)

                        if data.get("done"):
                            # Final response with stats
                            yield InferenceResponse(
                                request_id=request.request_id,
                                done=True,
                                total_time=time.time() - start_time,
                                usage={
                                    "prompt_tokens": data.get("prompt_eval_count", 0),
                                    "completion_tokens": data.get("eval_count", 0),
                                    "total_tokens": (
                                        data.get("prompt_eval_count", 0) +
                                        data.get("eval_count", 0)
                                    ),
                                }
                            )
                            break

                        # Streaming token
                        message = data.get("message", {})
                        content = message.get("content", "")
                        if content:
                            yield InferenceResponse(
                                request_id=request.request_id,
                                text=content
                            )

                    except json.JSONDecodeError:
                        pass

        except asyncio.CancelledError:
            import logging
            logging.getLogger(__name__).info(f"[Ollama] Generation cancelled for request {request.request_id}")
            yield InferenceResponse(
                request_id=request.request_id,
                error="Generation cancelled"
            )
            return
        except aiohttp.ClientError as e:
            yield InferenceResponse(
                request_id=request.request_id,
                error=f"Connection error: {str(e)}"
            )
        except asyncio.TimeoutError:
            yield InferenceResponse(
                request_id=request.request_id,
                error="Request timed out"
            )

    def _estimate_context_length(self, model_name: str, model_details: Optional[Dict] = None) -> int:
        """
        Estimate context length from model name patterns or details.

        Ollama models have context length in their modelfile parameters,
        but we can also infer from common naming patterns.
        """
        # Check model details if available
        if model_details:
            # Ollama returns parameters like "num_ctx 4096" in modelfile
            params = model_details.get("parameters", "")
            if params:
                for line in params.split("\n"):
                    if "num_ctx" in line:
                        try:
                            return int(line.split()[-1])
                        except (ValueError, IndexError):
                            pass

            # Also check model_info for context_length
            model_info = model_details.get("model_info", {})
            for key in model_info:
                if "context_length" in key.lower():
                    try:
                        return int(model_info[key])
                    except (ValueError, TypeError):
                        pass

        # Infer from model name patterns
        name_lower = model_name.lower()

        # Explicit context sizes in name
        if "128k" in name_lower:
            return 131072
        elif "64k" in name_lower:
            return 65536
        elif "32k" in name_lower:
            return 32768
        elif "16k" in name_lower:
            return 16384
        elif "8k" in name_lower:
            return 8192

        # Model family defaults
        if "llama3" in name_lower or "llama-3" in name_lower:
            return 8192  # Llama 3 default
        elif "llama2" in name_lower or "llama-2" in name_lower:
            return 4096  # Llama 2 default
        elif "mistral" in name_lower:
            return 32768  # Mistral default
        elif "mixtral" in name_lower:
            return 32768
        elif "qwen" in name_lower:
            return 32768  # Qwen default
        elif "phi" in name_lower:
            return 16384  # Phi default
        elif "gemma" in name_lower:
            return 8192  # Gemma default
        elif "deepseek" in name_lower:
            return 32768
        elif "codellama" in name_lower:
            return 16384
        elif "command-r" in name_lower:
            return 128000

        # Safe default
        return 4096

    async def _get_model_context_length(self, model_name: str) -> Optional[int]:
        """Get exact context length from model info via /api/show."""
        try:
            details = await self.show_model(model_name)
            if "error" not in details:
                # Check model_info for context_length keys
                model_info = details.get("model_info", {})
                for key, value in model_info.items():
                    if "context_length" in key.lower():
                        try:
                            return int(value)
                        except (ValueError, TypeError):
                            pass

                # Check parameters in modelfile
                params = details.get("parameters", "")
                if params:
                    for line in params.split("\n"):
                        if "num_ctx" in line.lower():
                            try:
                                return int(line.split()[-1])
                            except (ValueError, IndexError):
                                pass
        except Exception:
            pass
        return None

    async def list_models(self) -> List[Dict[str, Any]]:
        """List downloaded Ollama models (fast - no per-model API calls)."""
        session = await self._get_session()

        try:
            async with session.get(
                f"{self.base_url}/api/tags",
                timeout=aiohttp.ClientTimeout(total=3)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return [
                        {
                            "id": m["name"],
                            "name": m["name"],
                            "provider": "ollama",
                            "size_bytes": m.get("size"),
                            "modified_at": m.get("modified_at"),
                            "digest": m.get("digest"),
                            "has_vision": self._is_vision_model(m["name"]),
                            # Context length is fetched on selection (lazy loading)
                            "context_length": None,
                        }
                        for m in data.get("models", [])
                    ]

        except Exception:
            pass

        return []

    async def get_model_context_length(self, model_name: str) -> int:
        """
        Get exact context length for a single model (called on selection).

        This makes ONE API call to /api/show for the selected model.
        Returns exact value if found, otherwise falls back to estimation.
        """
        exact = await self._get_model_context_length(model_name)
        if exact is not None:
            return exact
        return self._estimate_context_length(model_name)

    async def pull_model(self, model_name: str) -> AsyncIterator[Dict[str, Any]]:
        """
        Pull/download a model from Ollama registry.

        Yields progress updates with status, completed, total bytes.
        """
        session = await self._get_session()

        try:
            async with session.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name, "stream": True},
                timeout=aiohttp.ClientTimeout(total=None)  # No timeout for downloads
            ) as response:
                if response.status != 200:
                    yield {"error": f"HTTP {response.status}"}
                    return

                async for line in response.content:
                    line = line.decode().strip()
                    if line:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            yield {"error": str(e)}

    async def delete_model(self, model_name: str) -> Dict[str, Any]:
        """Delete a model from Ollama."""
        session = await self._get_session()

        try:
            async with session.delete(
                f"{self.base_url}/api/delete",
                json={"name": model_name},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    return {"success": True, "model": model_name}
                else:
                    error = await response.text()
                    return {"success": False, "error": error}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def show_model(self, model_name: str) -> Dict[str, Any]:
        """
        Get detailed info about a model.

        Returns modelfile, parameters, template, license, etc.
        """
        session = await self._get_session()

        try:
            async with session.post(
                f"{self.base_url}/api/show",
                json={"name": model_name},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"HTTP {response.status}"}
        except Exception as e:
            return {"error": str(e)}

    async def list_running(self) -> List[Dict[str, Any]]:
        """
        List currently running/loaded models in Ollama.

        Returns models that are loaded in memory.
        """
        session = await self._get_session()

        try:
            async with session.get(
                f"{self.base_url}/api/ps",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("models", [])
                return []
        except Exception:
            return []

    async def copy_model(self, source: str, destination: str) -> Dict[str, Any]:
        """
        Copy a model to create a new variant.

        Useful for creating custom model names or versions.
        """
        session = await self._get_session()

        try:
            async with session.post(
                f"{self.base_url}/api/copy",
                json={"source": source, "destination": destination},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    return {"success": True, "source": source, "destination": destination}
                else:
                    error = await response.text()
                    return {"success": False, "error": error}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def generate(
        self,
        instance_id: str,
        prompt: str,
        system: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[InferenceResponse]:
        """
        Raw text generation (non-chat) via /api/generate.

        Useful for completion-style tasks.
        """
        instance = self.instances.get(instance_id)
        if not instance:
            yield InferenceResponse(
                request_id=kwargs.get("request_id", ""),
                error="Instance not found"
            )
            return

        session = await self._get_session()
        start_time = time.time()

        payload = {
            "model": instance.model_identifier,
            "prompt": prompt,
            "stream": True,
            "keep_alive": self.keep_alive,
            "options": {
                "num_predict": kwargs.get("max_tokens", 256),
                "temperature": kwargs.get("temperature", 0.7),
            }
        }

        if system:
            payload["system"] = system

        try:
            async with session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    yield InferenceResponse(
                        request_id=kwargs.get("request_id", ""),
                        error=f"Ollama error {response.status}: {error_text}"
                    )
                    return

                async for line in response.content:
                    line = line.decode().strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)

                        if data.get("done"):
                            yield InferenceResponse(
                                request_id=kwargs.get("request_id", ""),
                                done=True,
                                total_time=time.time() - start_time,
                            )
                            break

                        text = data.get("response", "")
                        if text:
                            yield InferenceResponse(
                                request_id=kwargs.get("request_id", ""),
                                text=text
                            )

                    except json.JSONDecodeError:
                        pass

        except Exception as e:
            yield InferenceResponse(
                request_id=kwargs.get("request_id", ""),
                error=str(e)
            )

    async def health_check(self) -> Dict[str, Any]:
        """Check Ollama connectivity and status."""
        # Quick async socket pre-check — Windows doesn't send RST for closed
        # ports, so aiohttp would wait ~2s for TCP timeout. This probe fails
        # fast (300ms) when Ollama isn't running, without blocking the event loop.
        try:
            from urllib.parse import urlparse
            parsed = urlparse(self.base_url)
            host = parsed.hostname or "127.0.0.1"
            port = parsed.port or 11434
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=0.3
            )
            writer.close()
            await writer.wait_closed()
        except Exception:
            return {
                "provider": "ollama",
                "status": "offline",
                "error": f"Cannot connect to {self.base_url}",
            }

        try:
            session = await self._get_session()

            # Get downloaded models
            async with session.get(
                f"{self.base_url}/api/tags",
                timeout=aiohttp.ClientTimeout(total=3, connect=1)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get("models", [])

                    # Also check running models
                    running = await self.list_running()

                    return {
                        "provider": "ollama",
                        "status": "healthy",
                        "model_count": len(models),
                        "running_count": len(running),
                        "running_models": [m.get("name") for m in running],
                        "base_url": self.base_url,
                    }
                else:
                    return {
                        "provider": "ollama",
                        "status": "error",
                        "error": f"HTTP {response.status}",
                    }

        except aiohttp.ClientError as e:
            return {
                "provider": "ollama",
                "status": "offline",
                "error": str(e),
            }
        except Exception as e:
            return {
                "provider": "ollama",
                "status": "error",
                "error": str(e),
            }

    async def get_status(self, instance_id: str) -> ModelStatus:
        """Get instance status."""
        instance = self.instances.get(instance_id)
        if not instance:
            return ModelStatus.UNLOADED
        return instance.status

    async def close(self):
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
        await super().close()
