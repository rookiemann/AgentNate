"""
LM Studio Provider - Full v4 SDK integration.

LM Studio v4 Features:
- GPU isolation (load models on specific GPUs)
- Multiple model instances simultaneously
- Parallel inference / continuous batching on single model
- JIT loading via OpenAI-compatible API
"""

import asyncio
import json
import logging
from typing import AsyncIterator, Optional, List, Dict, Any
import time

try:
    import aiohttp
except ImportError:
    aiohttp = None

try:
    import lmstudio
    from lmstudio import LlmLoadModelConfig
    from lmstudio._sdk_models import GpuSetting
    HAS_SDK = True
except ImportError:
    lmstudio = None
    LlmLoadModelConfig = None
    GpuSetting = None
    HAS_SDK = False

from .base import (
    BaseProvider,
    ProviderType,
    ModelInstance,
    ModelStatus,
    InferenceRequest,
    InferenceResponse,
)

# Import GPU detection utility
try:
    from gpu_utils import get_available_gpus
    HAS_GPU_UTILS = True
except ImportError:
    HAS_GPU_UTILS = False
    def get_available_gpus():
        return ["CPU"]  # Fallback


class LMStudioProvider(BaseProvider):
    """
    Provider for LM Studio v4 with full SDK integration.

    Features:
    - Load models on specific GPUs via SDK
    - Multiple model instances (same or different models)
    - Parallel inference on single model (continuous batching)
    - OpenAI-compatible API for streaming chat
    - JIT loading (request any model, LM Studio loads it)
    - Concurrent inference support (no request blocking)
    """

    def __init__(self, base_url: str = "http://localhost:1234/v1", settings=None):
        super().__init__(ProviderType.LM_STUDIO)
        self.base_url = base_url.rstrip("/")
        self.settings = settings
        self._session: Optional[aiohttp.ClientSession] = None
        self._sdk_client = None
        self._sdk_api_host: Optional[str] = None
        self._sdk_handles: Dict[str, Any] = {}  # instance_id -> SDK model handle
        # Track concurrent requests per instance (for metrics, not blocking)
        self._active_requests: Dict[str, int] = {}  # instance_id -> count
        self._logger = logging.getLogger(__name__)

    # ==================== Vision Detection ====================

    def _is_vision_model(self, model_id: str) -> bool:
        """Detect if model supports vision based on name patterns."""
        vision_patterns = [
            'vision', 'llava', 'bakllava', 'moondream', 'cogvlm',
            'minicpm-v', 'qwen-vl', 'internvl', 'yi-vl', 'deepseek-vl',
            'llava-v', 'obsidian', 'pixtral', 'molmo'
        ]
        model_lower = model_id.lower()
        return any(p in model_lower for p in vision_patterns)

    # ==================== SDK Connection ====================

    def _get_sdk_client_sync(self):
        """Get cached SDK client (sync, for use after async init)."""
        return self._sdk_client

    async def _get_sdk_client_async(self):
        """Get or create SDK client (async - runs blocking calls in executor)."""
        if not HAS_SDK:
            return None

        if self._sdk_client is None:
            try:
                import logging
                logger = logging.getLogger(__name__)
                start = time.time()

                loop = asyncio.get_event_loop()

                # Find SDK API port in executor - this is slow (port scanning)
                self._sdk_api_host = await loop.run_in_executor(
                    None,
                    lmstudio.Client.find_default_local_api_host
                )
                elapsed = time.time() - start
                logger.info(f"[LMStudio] SDK host lookup took {elapsed:.2f}s: {self._sdk_api_host}")

                if self._sdk_api_host:
                    # Create client in executor too (may do network calls)
                    self._sdk_client = await loop.run_in_executor(
                        None,
                        lambda: lmstudio.Client(api_host=self._sdk_api_host)
                    )
                    # Clean up stale agentnate-* models from previous server sessions
                    await self._cleanup_stale_sdk_models()
            except Exception as e:
                print(f"[LMStudio] SDK client error: {e}")
                return None

        return self._sdk_client

    async def _ensure_sdk_ready(self, timeout: float = 10.0) -> bool:
        """Wait for SDK to be ready."""
        if not HAS_SDK:
            return False

        client = await self._get_sdk_client_async()
        if client is None:
            return False

        loop = asyncio.get_event_loop()
        start = time.time()

        while time.time() - start < timeout:
            try:
                await loop.run_in_executor(
                    None,
                    lambda: list(client.llm.list_loaded())
                )
                return True
            except Exception as e:
                if "not yet resolved" in str(e):
                    await asyncio.sleep(0.5)
                    continue
                return False

        return False

    async def _cleanup_stale_sdk_models(self):
        """Unload agentnate-* models from previous server sessions."""
        client = self._sdk_client
        if not client:
            return

        try:
            loop = asyncio.get_event_loop()
            loaded = await loop.run_in_executor(
                None,
                lambda: list(client.llm.list_loaded())
            )

            stale = [m for m in loaded if m.identifier.startswith("agentnate-")]
            if stale:
                print(f"[LMStudio] Found {len(stale)} stale agentnate model(s), cleaning up...")
                for m in stale:
                    try:
                        await loop.run_in_executor(None, m.unload)
                        print(f"[LMStudio] Unloaded stale model: {m.identifier}")
                    except Exception as e:
                        print(f"[LMStudio] Failed to unload {m.identifier}: {e}")
        except Exception as e:
            print(f"[LMStudio] Stale model cleanup error: {e}")

    # ==================== HTTP Session ====================

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if aiohttp is None:
            raise ImportError("aiohttp is required for LM Studio provider")

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    # ==================== Model Loading ====================

    async def load_model(
        self,
        model_identifier: str = "current",
        **kwargs
    ) -> ModelInstance:
        """
        Load a model in LM Studio.

        Args:
            model_identifier: Model path/key to load, or "current" for default
            **kwargs:
                gpu_index: int - Specific GPU to use (0, 1, etc.)
                disabled_gpus: List[int] - GPUs to exclude
                context_length: int - Context window size
                instance_id: str - Custom instance identifier
                use_sdk: bool - Force SDK loading (default: True if available)

        Returns:
            ModelInstance tracking the loaded model
        """
        gpu_index = kwargs.get("gpu_index")
        disabled_gpus = kwargs.get("disabled_gpus")
        # Accept both context_length and n_ctx (llama.cpp style)
        context_length = kwargs.get("context_length") or kwargs.get("n_ctx") or 4096
        custom_instance_id = kwargs.get("instance_id")
        use_sdk = kwargs.get("use_sdk", True)
        force_single_gpu = kwargs.get("force_single_gpu", True)

        # Special case: "current" means use whatever's loaded or default
        if model_identifier == "current":
            return await self._load_current_model()

        # Enforce deterministic single-GPU loading by default.
        # If caller didn't specify a GPU, use configured default GPU (fallback: GPU 0).
        if force_single_gpu and gpu_index is None:
            default_gpu = None
            if self.settings and hasattr(self.settings, "get"):
                default_gpu = self.settings.get("providers.lm_studio.default_gpu_index")
                if default_gpu is None:
                    default_gpu = self.settings.get("providers.lm_studio.default_gpu")
            if default_gpu is None:
                detected_gpus = get_available_gpus()
                num_gpus = len([g for g in detected_gpus if str(g).startswith("GPU")])
                if num_gpus > 0:
                    default_gpu = 1  # GPU 1 = RTX 3090 (primary inference GPU)
            if default_gpu is not None:
                try:
                    gpu_index = int(default_gpu)
                    self._logger.info(f"[LMStudio] Auto-selected GPU {gpu_index} for single-GPU load")
                except (TypeError, ValueError):
                    self._logger.warning(f"[LMStudio] Invalid default GPU setting: {default_gpu!r}")
                    gpu_index = None

        # Use SDK whenever available so GPU/context settings are consistently applied.
        if use_sdk and HAS_SDK:
            return await self._load_via_sdk(
                model_identifier,
                gpu_index=gpu_index,
                disabled_gpus=disabled_gpus,
                context_length=context_length,
                custom_instance_id=custom_instance_id,
            )

        # Fallback path without SDK.
        # Note: JIT loading cannot guarantee single-GPU pinning.
        if force_single_gpu:
            self._logger.warning("[LMStudio] SDK unavailable; falling back to JIT load without strict GPU pinning")

        # For LM Studio v4, we can use model ID directly (JIT loading)
        models = await self.list_models()
        model_exists = any(m["id"] == model_identifier for m in models)

        if not model_exists:
            # Try partial match
            matches = [m for m in models if model_identifier.lower() in m["id"].lower()]
            if matches:
                model_identifier = matches[0]["id"]

        instance = ModelInstance(
            provider_type=self.provider_type,
            model_identifier=model_identifier,
            display_name=f"LM Studio: {model_identifier}",
            status=ModelStatus.READY,
            gpu_index=gpu_index,  # Preserve GPU index even for JIT loading
            context_length=context_length,
            metadata={
                "lm_studio_model": model_identifier,
                "jit_load": True,
                "gpu_index": gpu_index,
                "has_vision": self._is_vision_model(model_identifier),
            }
        )

        self.instances[instance.id] = instance
        return instance

    async def _load_via_sdk(
        self,
        model_path: str,
        gpu_index: Optional[int] = None,
        disabled_gpus: Optional[List[int]] = None,
        context_length: int = 4096,
        custom_instance_id: Optional[str] = None,
    ) -> ModelInstance:
        """Load model via SDK with GPU control."""
        if not await self._ensure_sdk_ready(timeout=10):
            raise ConnectionError("LM Studio SDK not available")

        client = self._get_sdk_client_sync()
        loop = asyncio.get_event_loop()

        # Build GPU config for single-GPU loading
        gpu_config = None
        if gpu_index is not None or disabled_gpus:
            gpu_kwargs = {}
            if gpu_index is not None:
                # Dynamically detect all GPUs
                detected_gpus = get_available_gpus()
                # Count actual GPUs (exclude "CPU" which is first in the list)
                num_gpus = len([g for g in detected_gpus if g.startswith("GPU")])

                # Validate gpu_index is valid
                if num_gpus == 0:
                    print(f"[LMStudio] WARNING: No GPUs detected, ignoring gpu_index={gpu_index}")
                    gpu_index = None  # Fall back to default behavior
                elif gpu_index >= num_gpus:
                    raise ValueError(
                        f"GPU {gpu_index} requested but only {num_gpus} GPU(s) available (0-{num_gpus-1})"
                    )
                else:
                    gpu_kwargs["main_gpu"] = gpu_index
                    # IMPORTANT: To force single-GPU loading, we must disable all other GPUs
                    # Otherwise LM Studio will split the model across available GPUs
                    if disabled_gpus is None:
                        disabled_gpus = [i for i in range(num_gpus) if i != gpu_index]
                        print(f"[LMStudio] Detected {num_gpus} GPU(s), disabling: {disabled_gpus}")
                    # Set ratio to 1.0 to use 100% of the main GPU
                    gpu_kwargs["ratio"] = 1.0

            if disabled_gpus:
                gpu_kwargs["disabled_gpus"] = disabled_gpus

            if gpu_kwargs:  # Only create config if we have valid settings
                print(f"[LMStudio] GPU config: main_gpu={gpu_index}, disabled_gpus={disabled_gpus}, ratio={gpu_kwargs.get('ratio')}")
                gpu_config = GpuSetting(**gpu_kwargs)
                print(f"[LMStudio] GpuSetting created: {gpu_config}")

        # Build load config
        load_config = LlmLoadModelConfig(
            gpu=gpu_config,
            context_length=context_length,
        )

        print(f"[LMStudio] Load config: context_length={context_length}, gpu={gpu_config}")

        # Generate instance identifier
        instance_identifier = custom_instance_id or f"agentnate-{int(time.time())}"

        print(f"[LMStudio] Loading model via SDK: {model_path}")
        print(f"[LMStudio] Instance identifier: {instance_identifier}")

        # Load via SDK
        handle = await loop.run_in_executor(
            None,
            lambda: client.llm.load_new_instance(
                model_path,
                instance_identifier,
                config=load_config,
                ttl=3600,  # 1 hour
            )
        )
        print(f"[LMStudio] Model loaded successfully via SDK")

        # Create instance
        instance = ModelInstance(
            provider_type=self.provider_type,
            model_identifier=model_path,
            display_name=f"LM Studio: {model_path.split('/')[-1]}",
            status=ModelStatus.READY,
            gpu_index=gpu_index,
            context_length=context_length,
            metadata={
                "lm_studio_model": model_path,
                "sdk_loaded": True,
                "sdk_identifier": instance_identifier,
                "gpu_index": gpu_index,
                "disabled_gpus": disabled_gpus,
                "has_vision": self._is_vision_model(model_path),
            }
        )

        self.instances[instance.id] = instance
        self._sdk_handles[instance.id] = handle

        return instance

    async def _load_current_model(self) -> ModelInstance:
        """Create instance for currently loaded model."""
        health = await self.health_check()

        if health.get("status") != "healthy":
            raise ConnectionError("LM Studio not reachable")

        if not health.get("loaded"):
            raise RuntimeError("No model loaded in LM Studio")

        model_name = health.get("model", "lm-studio-model")

        instance = ModelInstance(
            provider_type=self.provider_type,
            model_identifier=model_name,
            display_name=f"LM Studio: {model_name}",
            status=ModelStatus.READY,
            metadata={
                "lm_studio_model": model_name,
                "sdk_loaded": False,
            }
        )

        self.instances[instance.id] = instance
        return instance

    async def unload_model(self, instance_id: str) -> bool:
        """Unload a model instance from LM Studio."""
        instance = self.instances.get(instance_id)
        if not instance:
            return False

        unloaded_from_lm = False

        # If SDK-loaded, unload via stored handle
        handle = self._sdk_handles.pop(instance_id, None)
        if handle is not None:
            try:
                loop = asyncio.get_event_loop()
                await asyncio.wait_for(
                    loop.run_in_executor(None, handle.unload),
                    timeout=15.0,
                )
                unloaded_from_lm = True
                print(f"[LMStudio] Unloaded via SDK handle: {instance.model_identifier}")
            except asyncio.TimeoutError:
                print(f"[LMStudio] SDK handle unload timed out")
            except Exception as e:
                print(f"[LMStudio] SDK handle unload error: {e}")

        # Fallback: if no handle or handle unload failed, try finding model by name via SDK
        if not unloaded_from_lm:
            try:
                client = await self._get_sdk_client_async()
                if client:
                    loop = asyncio.get_event_loop()
                    loaded = await asyncio.wait_for(
                        loop.run_in_executor(None, lambda: list(client.llm.list_loaded())),
                        timeout=10.0,
                    )
                    # Match by identifier (agentnate-*) or by model path substring
                    model_name = instance.model_identifier or ""
                    sdk_id = (instance.metadata or {}).get("sdk_identifier", "")
                    for m in loaded:
                        ident = getattr(m, "identifier", "") or ""
                        m_path = getattr(m, "path", "") or ""
                        if (sdk_id and ident == sdk_id) or \
                           (ident.startswith("agentnate-")) or \
                           (model_name and model_name.lower() in ident.lower()) or \
                           (model_name and model_name.lower() in m_path.lower()):
                            try:
                                await asyncio.wait_for(
                                    loop.run_in_executor(None, m.unload),
                                    timeout=15.0,
                                )
                                unloaded_from_lm = True
                                print(f"[LMStudio] Unloaded via SDK search: {ident}")
                            except Exception as e:
                                print(f"[LMStudio] SDK search unload failed for {ident}: {e}")
                            break
            except Exception as e:
                print(f"[LMStudio] Fallback SDK unload error: {e}")

        if not unloaded_from_lm:
            print(f"[LMStudio] Warning: could not unload from LM Studio (SDK unavailable or model not found)")

        del self.instances[instance_id]
        return True

    # ==================== Chat / Inference ====================

    async def chat(
        self,
        instance_id: str,
        request: InferenceRequest
    ) -> AsyncIterator[InferenceResponse]:
        """
        Stream chat via OpenAI-compatible API or SDK.

        Supports concurrent inference - multiple requests can run
        simultaneously on the same model (LM Studio v4 continuous batching).
        """
        instance = self.instances.get(instance_id)
        if not instance:
            yield InferenceResponse(
                request_id=request.request_id,
                error="Instance not found"
            )
            return

        # Track concurrent requests (for metrics, not blocking)
        self._active_requests[instance_id] = self._active_requests.get(instance_id, 0) + 1
        active_count = self._active_requests[instance_id]

        # Check if request has images (vision/multimodal)
        has_images = any(msg.images for msg in request.messages if msg.images)

        try:
            # Always use OpenAI-compatible HTTP API for chat.
            # SDK is for load/unload only - it provides GPU selection control.
            # HTTP API supports LM Studio's native continuous batching for concurrency.
            async for resp in self._chat_via_api(instance, request):
                yield resp
        finally:
            # Decrement concurrent request count
            self._active_requests[instance_id] = max(0, self._active_requests.get(instance_id, 1) - 1)

    async def _chat_via_sdk(
        self,
        handle: Any,
        request: InferenceRequest
    ) -> AsyncIterator[InferenceResponse]:
        """
        Chat using SDK handle with streaming support.

        Uses SDK's streaming API when available for better responsiveness.
        Falls back to blocking respond() if streaming not supported.
        """
        start_time = time.time()

        try:
            loop = asyncio.get_event_loop()
            messages = request.to_messages_dict()

            # SDK respond() accepts:
            # 1. String prompt (simple)
            # 2. Dict with 'messages' key (multi-turn)
            if len(messages) == 1 and messages[0].get("role") == "user":
                prompt = messages[0].get("content", "")
            else:
                prompt = {"messages": messages}

            config = {
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
            }

            # Try streaming first (SDK v0.1.0+)
            try:
                # Check if handle supports streaming
                if hasattr(handle, 'respond_stream') or hasattr(handle, 'complete_stream'):
                    # Use streaming API
                    stream_method = getattr(handle, 'respond_stream', None) or getattr(handle, 'complete_stream', None)

                    def _stream():
                        for chunk in stream_method(prompt, config=config):
                            yield chunk

                    # Run in executor to avoid blocking
                    import queue
                    q = queue.Queue()

                    def _producer():
                        try:
                            for chunk in stream_method(prompt, config=config):
                                q.put(("chunk", chunk))
                            q.put(("done", None))
                        except Exception as e:
                            q.put(("error", str(e)))

                    import threading
                    thread = threading.Thread(target=_producer, daemon=True)
                    thread.start()

                    while True:
                        try:
                            msg_type, data = await loop.run_in_executor(None, lambda: q.get(timeout=0.1))
                            if msg_type == "chunk":
                                text = data.content if hasattr(data, 'content') else str(data)
                                if text:
                                    yield InferenceResponse(
                                        request_id=request.request_id,
                                        text=text,
                                    )
                            elif msg_type == "done":
                                break
                            elif msg_type == "error":
                                yield InferenceResponse(
                                    request_id=request.request_id,
                                    error=f"SDK stream error: {data}"
                                )
                                return
                        except:
                            # Queue timeout, check if thread is still alive
                            if not thread.is_alive():
                                break

                    yield InferenceResponse(
                        request_id=request.request_id,
                        done=True,
                        total_time=time.time() - start_time
                    )
                    return

            except (AttributeError, TypeError):
                pass  # Streaming not available, fall back to blocking

            # Fallback: blocking respond() call
            result = await loop.run_in_executor(
                None,
                lambda: handle.respond(prompt, config=config)
            )

            content = result.content if hasattr(result, 'content') else str(result)

            yield InferenceResponse(
                request_id=request.request_id,
                text=content,
            )

            yield InferenceResponse(
                request_id=request.request_id,
                done=True,
                total_time=time.time() - start_time
            )

        except Exception as e:
            yield InferenceResponse(
                request_id=request.request_id,
                error=f"SDK error: {str(e)}"
            )

    async def _chat_via_api(
        self,
        instance: ModelInstance,
        request: InferenceRequest
    ) -> AsyncIterator[InferenceResponse]:
        """Chat using OpenAI-compatible API (streaming)."""
        session = await self._get_session()
        start_time = time.time()
        debug_logs = self._logger.isEnabledFor(logging.DEBUG)

        model_id = instance.model_identifier

        messages_dict = request.to_messages_dict()
        if debug_logs:
            self._logger.debug(f"[LMStudio] _chat_via_api: {len(messages_dict)} messages, model={model_id}")
            for i, msg in enumerate(messages_dict):
                content = msg.get("content")
                if isinstance(content, list):
                    self._logger.debug(f"[LMStudio]   Message {i}: role={msg.get('role')}, content_parts={len(content)}")
                    for j, part in enumerate(content):
                        self._logger.debug(f"[LMStudio]     Part {j}: type={part.get('type')}")
                else:
                    clen = len(str(content)) if content else 0
                    self._logger.debug(f"[LMStudio]   Message {i}: role={msg.get('role')}, content_len={clen}")

        payload = {
            "model": model_id,
            "messages": messages_dict,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
            "stream": True,
        }

        if request.stop:
            payload["stop"] = request.stop

        url = f"{self.base_url}/chat/completions"
        if debug_logs:
            self._logger.debug(f"[LMStudio] POST {url}")
            self._logger.debug(f"[LMStudio] Payload model={model_id}, max_tokens={request.max_tokens}, temp={request.temperature}")

        try:
            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                if debug_logs:
                    self._logger.debug(f"[LMStudio] Response status: {response.status}")

                if response.status != 200:
                    error_text = await response.text()
                    self._logger.warning(f"[LMStudio] Error response: {error_text}")
                    yield InferenceResponse(
                        request_id=request.request_id,
                        error=f"LM Studio error {response.status}: {error_text}"
                    )
                    return

                chunk_count = 0
                async for line in response.content:
                    line = line.decode().strip()

                    if not line:
                        continue

                    if line.startswith("data: "):
                        data = line[6:]

                        if data == "[DONE]":
                            if debug_logs:
                                self._logger.debug(f"[LMStudio] Stream done after {chunk_count} chunks, time={time.time() - start_time:.2f}s")
                            yield InferenceResponse(
                                request_id=request.request_id,
                                done=True,
                                total_time=time.time() - start_time
                            )
                            break

                        try:
                            chunk = json.loads(data)
                            choices = chunk.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    chunk_count += 1
                                    yield InferenceResponse(
                                        request_id=request.request_id,
                                        text=content
                                    )
                        except json.JSONDecodeError as e:
                            if debug_logs:
                                self._logger.debug(f"[LMStudio] JSON decode error: {e}, data: {data[:100]}")

        except asyncio.CancelledError:
            if debug_logs:
                self._logger.debug(f"[LMStudio] Generation cancelled for request {request.request_id}")
            yield InferenceResponse(
                request_id=request.request_id,
                error="Generation cancelled"
            )
            return
        except aiohttp.ClientError as e:
            self._logger.warning(f"[LMStudio] ClientError: {e}")
            yield InferenceResponse(
                request_id=request.request_id,
                error=f"Connection error: {str(e)}"
            )
        except asyncio.TimeoutError:
            self._logger.warning(f"[LMStudio] Timeout after {time.time() - start_time:.2f}s")
            yield InferenceResponse(
                request_id=request.request_id,
                error="Request timed out"
            )
        except Exception as e:
            self._logger.error(f"[LMStudio] Unexpected error: {type(e).__name__}: {e}")
            yield InferenceResponse(
                request_id=request.request_id,
                error=f"Unexpected error: {str(e)}"
            )

    # ==================== Model Discovery ====================

    def _estimate_context_length(self, model_id: str) -> int:
        """
        Estimate context length from model name/path patterns.

        LM Studio models are typically GGUF files with naming conventions
        that indicate context length and model family.
        """
        model_lower = model_id.lower()

        # Explicit context sizes in name
        if "128k" in model_lower:
            return 131072
        elif "64k" in model_lower:
            return 65536
        elif "32k" in model_lower:
            return 32768
        elif "16k" in model_lower:
            return 16384
        elif "8k" in model_lower:
            return 8192

        # Model family defaults
        if "llama-3" in model_lower or "llama3" in model_lower:
            return 8192  # Llama 3 default
        elif "llama-2" in model_lower or "llama2" in model_lower:
            return 4096  # Llama 2 default
        elif "mistral" in model_lower:
            return 32768  # Mistral default
        elif "mixtral" in model_lower:
            return 32768
        elif "qwen" in model_lower:
            return 32768  # Qwen default
        elif "phi" in model_lower:
            return 16384  # Phi default
        elif "gemma" in model_lower:
            return 8192  # Gemma default
        elif "deepseek" in model_lower:
            return 32768
        elif "codellama" in model_lower:
            return 16384
        elif "command-r" in model_lower:
            return 128000
        elif "claude" in model_lower:
            return 200000  # Claude models (if proxied)
        elif "gpt-4" in model_lower:
            return 128000  # GPT-4 variants

        # Safe default
        return 4096

    async def list_models(self) -> List[Dict[str, Any]]:
        """List models from LM Studio API."""
        import logging
        logger = logging.getLogger(__name__)

        start = time.time()
        session = await self._get_session()

        # Try native LM Studio API first (/api/v0/models) - returns max_context_length
        try:
            # Native API is at port 1234 base, path /api/v0/models
            native_url = self.base_url.replace("/v1", "") + "/api/v0/models"
            async with session.get(
                native_url,
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    # Native API returns list directly or in "data" key
                    models_list = data if isinstance(data, list) else data.get("data", [])

                    if models_list:
                        elapsed = time.time() - start
                        logger.debug(f"[LMStudio] list_models via /api/v0 took {elapsed:.2f}s, got {len(models_list)} models")

                        return [
                            {
                                "id": m.get("id", m.get("path", "unknown")),
                                "name": m.get("id", m.get("path", "LM Studio Model")),
                                "provider": "lm_studio",
                                "has_vision": self._is_vision_model(m.get("id", m.get("path", ""))),
                                # Native API returns max_context_length; null means fetch on selection
                                "context_length": m.get("max_context_length"),
                                "quantization": m.get("quantization"),
                                "state": m.get("state"),  # "loaded" or "not-loaded"
                            }
                            for m in models_list
                        ]
        except Exception as e:
            logger.debug(f"[LMStudio] Native API /api/v0/models failed: {e}")

        # Fallback to OpenAI-compatible API (/v1/models) - no context length
        try:
            async with session.get(
                f"{self.base_url}/models",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                elapsed = time.time() - start
                logger.debug(f"[LMStudio] list_models via /v1 took {elapsed:.2f}s")

                if response.status == 200:
                    data = await response.json()
                    models = data.get("data", [])
                    return [
                        {
                            "id": m.get("id", "unknown"),
                            "name": m.get("id", "LM Studio Model"),
                            "provider": "lm_studio",
                            "has_vision": self._is_vision_model(m.get("id", "")),
                            # OpenAI API doesn't return context length; fetch on selection
                            "context_length": None,
                        }
                        for m in models
                    ]
        except asyncio.TimeoutError:
            logger.warning(f"[LMStudio] list_models timed out after {time.time() - start:.2f}s")
        except Exception as e:
            logger.debug(f"[LMStudio] list_models error: {e}")

        return []

    async def list_downloaded_models(self) -> List[Dict[str, Any]]:
        """List all downloaded models via native API or SDK."""
        import logging
        logger = logging.getLogger(__name__)

        # Try native REST API first - returns max_context_length
        try:
            session = await self._get_session()
            native_url = self.base_url.replace("/v1", "") + "/api/v0/models"

            async with session.get(
                native_url,
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    models_list = data if isinstance(data, list) else data.get("data", [])

                    if models_list:
                        logger.debug(f"[LMStudio] list_downloaded via /api/v0: {len(models_list)} models")
                        return [
                            {
                                "id": m.get("id", m.get("path", "unknown")),
                                "name": m.get("id", m.get("path", "")).split("/")[-1] if "/" in m.get("id", m.get("path", "")) else m.get("id", m.get("path", "")),
                                "path": m.get("path", m.get("id", "")),
                                "provider": "lm_studio",
                                "has_vision": self._is_vision_model(m.get("id", m.get("path", ""))),
                                # Native API returns max_context_length; null means fetch on selection
                                "context_length": m.get("max_context_length"),
                                "quantization": m.get("quantization"),
                                "state": m.get("state"),
                            }
                            for m in models_list
                        ]
        except Exception as e:
            logger.debug(f"[LMStudio] Native API for downloaded models failed: {e}")

        # Fallback to SDK
        if not await self._ensure_sdk_ready(timeout=5):
            return []

        client = self._get_sdk_client_sync()
        if not client:
            return []

        try:
            loop = asyncio.get_event_loop()
            models = await loop.run_in_executor(
                None,
                lambda: list(client.llm.list_downloaded())
            )

            return [
                {
                    "id": m.path,
                    "name": m.path.split("/")[-1] if "/" in m.path else m.path,
                    "path": m.path,
                    "provider": "lm_studio",
                    "has_vision": self._is_vision_model(m.path),
                    # SDK doesn't return context length; fetch on selection
                    "context_length": None,
                }
                for m in models
            ]
        except Exception as e:
            print(f"[LMStudio] Error listing downloaded: {e}")
            return []

    async def list_loaded_instances(self) -> List[Dict[str, Any]]:
        """List currently loaded model instances via SDK."""
        if not await self._ensure_sdk_ready(timeout=5):
            return []

        client = self._get_sdk_client_sync()
        if not client:
            return []

        try:
            loop = asyncio.get_event_loop()
            loaded = await loop.run_in_executor(
                None,
                lambda: list(client.llm.list_loaded())
            )

            return [
                {
                    "identifier": m.identifier,
                    "provider": "lm_studio",
                }
                for m in loaded
            ]
        except Exception as e:
            print(f"[LMStudio] Error listing loaded: {e}")
            return []

    # ==================== Health Check ====================

    async def health_check(self) -> Dict[str, Any]:
        """Check LM Studio connectivity and capabilities."""
        # Count total active concurrent requests
        total_active = sum(self._active_requests.values())

        result = {
            "provider": "lm_studio",
            "sdk_available": HAS_SDK,
            "base_url": self.base_url,
            "concurrent_requests": total_active,
            "active_per_instance": dict(self._active_requests),
        }

        # Check API connectivity (short connect timeout for fast offline detection)
        try:
            session = await self._get_session()

            async with session.get(
                f"{self.base_url}/models",
                timeout=aiohttp.ClientTimeout(total=3, connect=1)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get("data", [])
                    result.update({
                        "status": "healthy",
                        "loaded": len(models) > 0,
                        "model": models[0]["id"] if models else None,
                        "model_count": len(models),
                    })
                else:
                    result.update({
                        "status": "error",
                        "error": f"HTTP {response.status}",
                        "loaded": False,
                    })

        except Exception as e:
            result.update({
                "status": "offline",
                "error": str(e),
                "loaded": False,
            })

        # Check SDK - use cached client if available, don't scan for host on every health check
        if HAS_SDK:
            if self._sdk_api_host:
                result["sdk_api_host"] = self._sdk_api_host
                result["sdk_connected"] = True
            else:
                # Only try to find host if not already cached (expensive operation)
                result["sdk_connected"] = False
                result["sdk_api_host"] = None

        return result

    async def get_status(self, instance_id: str) -> ModelStatus:
        """Get instance status."""
        instance = self.instances.get(instance_id)
        if not instance:
            return ModelStatus.UNLOADED

        if instance_id in self._sdk_handles:
            return ModelStatus.READY

        health = await self.health_check()
        if health.get("loaded"):
            return ModelStatus.READY
        return ModelStatus.ERROR

    async def close(self):
        """Close connections."""
        # Unload SDK instances
        for instance_id in list(self._sdk_handles.keys()):
            await self.unload_model(instance_id)

        if self._session and not self._session.closed:
            await self._session.close()

        if self._sdk_client:
            try:
                self._sdk_client.close()
            except:
                pass
            self._sdk_client = None

        await super().close()
