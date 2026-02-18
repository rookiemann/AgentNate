"""
Model Orchestrator - Central management for all LLM providers.

Features:
- Multiple providers (llama.cpp, LM Studio, OpenRouter, Ollama, vLLM)
- Multiple instances per provider
- JIT/lazy loading
- Priority request queue
- Slot management for pre-warmed instances
- Health monitoring
"""

import asyncio
import sys
import os
from typing import Dict, List, Optional, AsyncIterator, Callable, Any
import time

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.signals import Signal
from providers.base import (
    BaseProvider,
    ProviderType,
    ModelInstance,
    ModelStatus,
    InferenceRequest,
    InferenceResponse,
)
from settings.settings_manager import SettingsManager
from .request_queue import RequestQueue, RequestStatus


class ModelOrchestrator:
    """
    Central orchestrator managing all providers and model instances.

    Supports:
    - Multiple providers (llama.cpp, LM Studio, OpenRouter, Ollama)
    - Multiple instances per provider
    - JIT/lazy loading
    - Priority queue for requests
    - Slot management for pre-warmed instances
    - Health monitoring
    - Concurrent model loading (no queue bottleneck)
    - Pending state tracking for immediate UI feedback
    - Load cancellation support
    """

    def __init__(self, settings: SettingsManager):
        """
        Initialize orchestrator.

        Args:
            settings: SettingsManager instance
        """
        self.settings = settings

        # Providers (initialized lazily)
        self.providers: Dict[ProviderType, BaseProvider] = {}

        # All instances across all providers
        self.instances: Dict[str, ModelInstance] = {}

        # Pending loads - tracked separately for immediate UI feedback
        # Key: pending_{instance_id}, Value: ModelInstance with LOADING status
        self._pending_loads: Dict[str, ModelInstance] = {}

        # Active load tasks - for cancellation support
        self._load_tasks: Dict[str, asyncio.Task] = {}

        # Request queue (for inference, not loading)
        max_concurrent = settings.get("orchestrator.max_concurrent_inferences", 4)
        self.request_queue = RequestQueue(max_concurrent=max_concurrent)

        # Slots for pre-warmed instances
        self._slots: Dict[str, str] = {}  # slot_name -> instance_id

        # Background tasks
        self._health_task: Optional[asyncio.Task] = None
        self._queue_task: Optional[asyncio.Task] = None
        self._running = False

        # Health check cache (TTL-based to avoid redundant provider polls)
        self._health_cache: Optional[Dict[str, Dict[str, Any]]] = None
        self._health_cache_time: float = 0
        self._health_cache_ttl: float = 5.0  # seconds

        # Signals
        self.on_instance_loaded = Signal()       # (instance_id, display_name, provider)
        self.on_instance_unloaded = Signal()     # (instance_id,)
        self.on_instance_status_changed = Signal()  # (instance_id, new_status)
        self.on_instance_load_started = Signal()  # (instance_id, display_name, provider) - NEW
        self.on_instance_load_failed = Signal()   # (instance_id, error_msg) - NEW
        self.on_health_update = Signal()         # (health_dict,)
        self.on_stream_token = Signal()          # (request_id, token)
        self.on_stream_complete = Signal()       # (request_id,)
        self.on_stream_error = Signal()          # (request_id, error)
        self.on_queue_update = Signal()          # (queue_length, processing_count)

    async def start(self):
        """Start the orchestrator and background tasks."""
        if self._running:
            return

        self._running = True

        # Initialize providers
        await self._init_providers()

        # Start request queue
        await self.request_queue.start()

        # Start background tasks
        health_interval = self.settings.get("orchestrator.health_check_interval", 30)
        self._health_task = asyncio.create_task(self._health_monitor(health_interval))
        self._queue_task = asyncio.create_task(self._process_queue())

    async def stop(self):
        """Stop the orchestrator and clean up."""
        self._running = False

        # Cancel background tasks
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        if self._queue_task:
            self._queue_task.cancel()
            try:
                await self._queue_task
            except asyncio.CancelledError:
                pass

        # Stop request queue
        await self.request_queue.stop()

        # Unload all instances
        for instance_id in list(self.instances.keys()):
            await self.unload_model(instance_id)

        # Close providers
        for provider in self.providers.values():
            await provider.close()

    async def _init_providers(self):
        """Initialize enabled providers."""
        # Import providers lazily to avoid circular imports
        from providers.llama_cpp_provider import LlamaCppProvider
        from providers.lm_studio_provider import LMStudioProvider
        from providers.openrouter_provider import OpenRouterProvider
        from providers.ollama_provider import OllamaProvider

        # LlamaCpp
        if self.settings.get("providers.llama_cpp.enabled", True):
            models_dir = self.settings.get("providers.llama_cpp.models_directory", "")
            self.providers[ProviderType.LLAMA_CPP] = LlamaCppProvider(
                models_directory=models_dir,
                settings=self.settings,
            )

        # LM Studio
        if self.settings.get("providers.lm_studio.enabled", True):
            base_url = self.settings.get("providers.lm_studio.base_url", "http://localhost:1234/v1")
            self.providers[ProviderType.LM_STUDIO] = LMStudioProvider(
                base_url=base_url,
                settings=self.settings,
            )

        # OpenRouter
        if self.settings.get("providers.openrouter.enabled", False):
            api_key = self.settings.get("providers.openrouter.api_key", "")
            self.providers[ProviderType.OPENROUTER] = OpenRouterProvider(api_key=api_key)

        # Ollama
        if self.settings.get("providers.ollama.enabled", True):
            base_url = self.settings.get("providers.ollama.base_url", "http://localhost:11434")
            self.providers[ProviderType.OLLAMA] = OllamaProvider(base_url=base_url)

        # vLLM
        if self.settings.get("providers.vllm.enabled", False):
            from providers.vllm_provider import VLLMProvider
            env_path = self.settings.get("providers.vllm.env_path", "envs/vllm")
            port_start = self.settings.get("providers.vllm.default_port_range_start", 8100)
            models_dir = self.settings.get("providers.vllm.models_directory", "")
            self.providers[ProviderType.VLLM] = VLLMProvider(
                env_path=env_path,
                port_start=port_start,
                settings=self.settings,
                models_directory=models_dir,
            )


    # ==================== Model Management ====================

    async def load_model(
        self,
        provider_type: ProviderType,
        model_identifier: str,
        slot_name: Optional[str] = None,
        **kwargs
    ) -> ModelInstance:
        """
        Load a model on the specified provider.

        Supports concurrent loading - multiple models can load simultaneously.
        Emits on_instance_load_started immediately for UI feedback.

        Args:
            provider_type: Which provider to use
            model_identifier: Path for local, model ID for API
            slot_name: Optional slot name for quick access
            **kwargs: Provider-specific options

        Returns:
            ModelInstance with status READY on success

        Raises:
            ValueError: If provider not enabled
            Exception: On load failure
        """
        provider = self.providers.get(provider_type)
        if not provider:
            raise ValueError(f"Provider {provider_type.value} not enabled")

        # Create pending instance for immediate UI feedback
        import uuid
        import os
        pending_id = str(uuid.uuid4())
        display_name = os.path.basename(model_identifier) if os.path.exists(model_identifier) else model_identifier

        pending_instance = ModelInstance(
            provider_type=provider_type,
            model_identifier=model_identifier,
            display_name=display_name,
            status=ModelStatus.LOADING,
            metadata={"pending": True}
        )
        pending_instance.id = pending_id  # Pre-assign ID

        # Track pending load
        pending_key = f"pending_{pending_id}"
        self._pending_loads[pending_key] = pending_instance

        # Emit load started signal for immediate UI feedback
        self.on_instance_load_started.emit(
            pending_id,
            display_name,
            provider_type.value
        )

        try:
            # Load the model (this can run concurrently with other loads)
            instance = await provider.load_model(model_identifier, **kwargs)

            # Remove from pending
            self._pending_loads.pop(pending_key, None)

            # Track in our registry
            self.instances[instance.id] = instance

            # Assign to slot if requested
            if slot_name:
                self._slots[slot_name] = instance.id
                instance.slot_name = slot_name

            # Emit loaded signal
            self.on_instance_loaded.emit(
                instance.id,
                instance.display_name,
                provider_type.value
            )

            return instance

        except Exception as e:
            # Remove from pending on failure
            self._pending_loads.pop(pending_key, None)

            # Emit failure signal
            self.on_instance_load_failed.emit(pending_id, str(e))

            raise

    async def load_model_jit(
        self,
        provider_type: ProviderType,
        model_identifier: str,
        **kwargs
    ) -> str:
        """
        Get or create an instance for the model (Just-In-Time loading).

        If already loaded, returns existing instance ID.
        Otherwise loads a new instance.

        Args:
            provider_type: Which provider to use
            model_identifier: Path for local, model ID for API
            **kwargs: Provider-specific options

        Returns:
            Instance ID (existing or new)
        """
        if not self.settings.get("orchestrator.jit_loading_enabled", True):
            # JIT disabled, always load new
            instance = await self.load_model(provider_type, model_identifier, **kwargs)
            return instance.id

        # Check if already loaded
        provider = self.providers.get(provider_type)
        if provider:
            existing = provider.find_instance_by_model(model_identifier)
            if existing:
                existing.touch()
                return existing.id

        # Load new instance
        instance = await self.load_model(provider_type, model_identifier, **kwargs)
        return instance.id

    async def unload_model(self, instance_id: str) -> bool:
        """
        Unload a specific model instance.

        Args:
            instance_id: UUID of instance to unload

        Returns:
            True if unloaded, False if not found
        """
        instance = self.instances.get(instance_id)
        if not instance:
            return False

        provider = self.providers.get(instance.provider_type)
        if provider:
            await provider.unload_model(instance_id)

        # Remove from slots
        for slot_name, slot_id in list(self._slots.items()):
            if slot_id == instance_id:
                del self._slots[slot_name]

        # Remove from registry
        del self.instances[instance_id]

        # Emit signal
        self.on_instance_unloaded.emit(instance_id)

        return True

    def get_slot_instance(self, slot_name: str) -> Optional[ModelInstance]:
        """Get instance by slot name."""
        instance_id = self._slots.get(slot_name)
        if instance_id:
            return self.instances.get(instance_id)
        return None

    async def load_model_async(
        self,
        provider_type: ProviderType,
        model_identifier: str,
        slot_name: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Start loading a model asynchronously (non-blocking).

        Returns immediately with pending instance ID. Use get_instance()
        to check status, or listen for on_instance_loaded/on_instance_load_failed.

        Args:
            provider_type: Which provider to use
            model_identifier: Path for local, model ID for API
            slot_name: Optional slot name for quick access
            **kwargs: Provider-specific options

        Returns:
            Pending instance ID (can be used to cancel or track)
        """
        import uuid
        import os

        pending_id = str(uuid.uuid4())
        display_name = os.path.basename(model_identifier) if os.path.exists(model_identifier) else model_identifier

        # Create pending instance for immediate UI feedback
        pending_instance = ModelInstance(
            provider_type=provider_type,
            model_identifier=model_identifier,
            display_name=display_name,
            status=ModelStatus.LOADING,
            metadata={"pending": True, "slot_name": slot_name}
        )
        pending_instance.id = pending_id

        # Track pending load
        pending_key = f"pending_{pending_id}"
        self._pending_loads[pending_key] = pending_instance

        # Emit load started signal
        self.on_instance_load_started.emit(
            pending_id,
            display_name,
            provider_type.value
        )

        # Create background task for actual loading
        async def _do_load():
            try:
                provider = self.providers.get(provider_type)
                if not provider:
                    raise ValueError(f"Provider {provider_type.value} not enabled")

                instance = await provider.load_model(model_identifier, **kwargs)

                # Remove from pending
                self._pending_loads.pop(pending_key, None)
                self._load_tasks.pop(pending_id, None)

                # Track in registry
                self.instances[instance.id] = instance

                # Assign to slot if requested
                if slot_name:
                    self._slots[slot_name] = instance.id
                    instance.slot_name = slot_name

                # Emit loaded signal
                self.on_instance_loaded.emit(
                    instance.id,
                    instance.display_name,
                    provider_type.value
                )

            except asyncio.CancelledError:
                # Load was cancelled
                self._pending_loads.pop(pending_key, None)
                self._load_tasks.pop(pending_id, None)
                self.on_instance_load_failed.emit(pending_id, "Load cancelled")
                raise

            except Exception as e:
                # Load failed
                self._pending_loads.pop(pending_key, None)
                self._load_tasks.pop(pending_id, None)
                self.on_instance_load_failed.emit(pending_id, str(e))

        # Start the load task
        task = asyncio.create_task(_do_load())
        self._load_tasks[pending_id] = task

        return pending_id

    async def cancel_load(self, pending_id: str) -> bool:
        """
        Cancel a pending model load.

        Args:
            pending_id: ID returned from load_model_async

        Returns:
            True if cancelled, False if not found or already completed
        """
        task = self._load_tasks.get(pending_id)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return True

        # Also try to cancel via provider if it supports it
        pending_key = f"pending_{pending_id}"
        pending_instance = self._pending_loads.get(pending_key)
        if pending_instance:
            provider = self.providers.get(pending_instance.provider_type)
            if provider and hasattr(provider, 'cancel_load'):
                await provider.cancel_load(pending_id)
            self._pending_loads.pop(pending_key, None)
            self.on_instance_load_failed.emit(pending_id, "Load cancelled")
            return True

        return False

    def get_pending_loads(self) -> List[ModelInstance]:
        """Get all currently loading (pending) instances."""
        return list(self._pending_loads.values())

    # ==================== Inference ====================

    async def chat(
        self,
        instance_id: str,
        request: InferenceRequest,
        callback: Optional[Callable[[InferenceResponse], None]] = None
    ) -> AsyncIterator[InferenceResponse]:
        """
        Execute chat inference on a specific instance.

        Args:
            instance_id: UUID of model instance
            request: InferenceRequest with messages and parameters
            callback: Optional callback for each response

        Yields:
            InferenceResponse objects (streaming tokens, then final with done=True)
        """
        print(f"[Orchestrator] chat() called: instance_id={instance_id}, request_id={request.request_id}")

        instance = self.instances.get(instance_id)
        if not instance:
            print(f"[Orchestrator] ERROR: Instance not found: {instance_id}")
            print(f"[Orchestrator] Available instances: {list(self.instances.keys())}")
            error_response = InferenceResponse(
                request_id=request.request_id,
                error="Instance not found"
            )
            if callback:
                callback(error_response)
            self.on_stream_error.emit(request.request_id, "Instance not found")
            yield error_response
            return

        print(f"[Orchestrator] Found instance: model={instance.model_identifier}, provider={instance.provider_type.value}")

        provider = self.providers.get(instance.provider_type)
        if not provider:
            print(f"[Orchestrator] ERROR: Provider not available: {instance.provider_type.value}")
            error_response = InferenceResponse(
                request_id=request.request_id,
                error="Provider not available"
            )
            if callback:
                callback(error_response)
            self.on_stream_error.emit(request.request_id, "Provider not available")
            yield error_response
            return

        # Update instance status
        instance.status = ModelStatus.BUSY
        self.on_instance_status_changed.emit(instance_id, ModelStatus.BUSY.value)

        start_time = time.time()
        first_token_time = None
        response_count = 0

        print(f"[Orchestrator] Calling provider.chat()...")
        try:
            async for response in provider.chat(instance_id, request):
                response_count += 1
                # Track timing
                if response.text and first_token_time is None:
                    first_token_time = time.time()
                    response.time_to_first_token = first_token_time - start_time
                    print(f"[Orchestrator] First token received in {response.time_to_first_token:.2f}s")

                # Callback
                if callback:
                    callback(response)

                # Emit signals
                if response.text:
                    self.on_stream_token.emit(request.request_id, response.text)
                if response.done:
                    response.total_time = time.time() - start_time
                    print(f"[Orchestrator] Chat done: {response_count} responses, {response.total_time:.2f}s")
                    self.on_stream_complete.emit(request.request_id)
                if response.error:
                    print(f"[Orchestrator] Chat error: {response.error}")
                    self.on_stream_error.emit(request.request_id, response.error)

                yield response

        except Exception as e:
            print(f"[Orchestrator] Exception in chat: {type(e).__name__}: {e}")
            raise
        finally:
            # Restore instance status
            instance.status = ModelStatus.READY
            instance.touch()
            self.on_instance_status_changed.emit(instance_id, ModelStatus.READY.value)
            print(f"[Orchestrator] Chat completed, instance status restored to READY")

    async def queue_chat(
        self,
        instance_id: str,
        request: InferenceRequest,
        priority: int = 0
    ) -> asyncio.Future:
        """
        Queue a chat request for processing.

        Args:
            instance_id: UUID of model instance
            request: InferenceRequest
            priority: Higher = more important

        Returns:
            Future that resolves with list of InferenceResponse
        """
        request_data = {
            "messages": request.to_messages_dict(),
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "repeat_penalty": request.repeat_penalty,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
            "mirostat": request.mirostat,
            "mirostat_tau": request.mirostat_tau,
            "mirostat_eta": request.mirostat_eta,
            "typical_p": request.typical_p,
            "tfs_z": request.tfs_z,
            "stop": request.stop,
        }

        future = await self.request_queue.enqueue(
            request_id=request.request_id,
            instance_id=instance_id,
            request_data=request_data,
            priority=priority,
        )

        # Emit queue update
        self.on_queue_update.emit(
            self.request_queue.get_queue_length(),
            self.request_queue.get_processing_count()
        )

        return future

    async def _process_queue(self):
        """Background task to process queued requests."""
        while self._running:
            try:
                # Get next request
                queued = await self.request_queue.dequeue()
                if queued is None:
                    continue

                # Process it
                try:
                    results = []
                    request = InferenceRequest(
                        request_id=queued.request_id,
                        **queued.request_data
                    )

                    async for response in self.chat(queued.instance_id, request):
                        results.append(response)

                    await self.request_queue.complete(queued.request_id, result=results)

                except Exception as e:
                    await self.request_queue.complete(
                        queued.request_id,
                        error=str(e)
                    )

                # Emit queue update
                self.on_queue_update.emit(
                    self.request_queue.get_queue_length(),
                    self.request_queue.get_processing_count()
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Queue processor error: {e}")
                await asyncio.sleep(1)

    # ==================== Model Discovery ====================

    async def list_all_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        List models from all providers (in parallel for speed).

        Returns:
            Dict mapping provider name to list of model info
        """
        import time as _time

        async def fetch_provider_models(provider_type, provider):
            start = _time.time()
            try:
                models = await provider.list_models()
                elapsed = _time.time() - start
                print(f"[Orchestrator] {provider_type.value}.list_models() took {elapsed:.2f}s, found {len(models)} models")
                return provider_type.value, models
            except Exception as e:
                elapsed = _time.time() - start
                print(f"[Orchestrator] {provider_type.value}.list_models() FAILED after {elapsed:.2f}s: {e}")
                return provider_type.value, []

        # Run all providers in parallel
        overall_start = _time.time()
        tasks = [
            fetch_provider_models(pt, p)
            for pt, p in self.providers.items()
        ]
        results = await asyncio.gather(*tasks)
        print(f"[Orchestrator] list_all_models total: {_time.time() - overall_start:.2f}s")

        return dict(results)

    def get_loaded_instances(self) -> List[ModelInstance]:
        """Get all currently loaded instances."""
        loaded = []
        for instance in self.instances.values():
            status = getattr(instance, "status", None)
            status_value = getattr(status, "value", str(status)).lower() if status is not None else ""
            if status in (ModelStatus.READY, ModelStatus.BUSY, ModelStatus.LOADING) or status_value in {"ready", "busy", "loading"}:
                loaded.append(instance)
        return loaded

    def get_instance(self, instance_id: str) -> Optional[ModelInstance]:
        """Get instance by ID."""
        return self.instances.get(instance_id)

    # ==================== Health Monitoring ====================

    async def _health_monitor(self, interval: int):
        """Background task to monitor provider health."""
        while self._running:
            try:
                health = await self.check_all_health(force=True)
                self.on_health_update.emit(health)
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Health monitor error: {e}")
                await asyncio.sleep(5)

    async def check_all_health(self, force: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Check health of all providers (in parallel for speed).
        Results are cached with a short TTL to avoid redundant provider polls
        when multiple API requests arrive in quick succession.

        Args:
            force: If True, bypass cache and do a fresh poll

        Returns:
            Dict mapping provider name to health info
        """
        # Return cached result if fresh enough
        now = time.monotonic()
        if not force and self._health_cache is not None and (now - self._health_cache_time) < self._health_cache_ttl:
            return self._health_cache

        async def check_provider(provider_type, provider):
            try:
                health = await provider.health_check()
                health["loaded_count"] = len(provider.get_ready_instances())
                return provider_type.value, health
            except Exception as e:
                return provider_type.value, {
                    "provider": provider_type.value,
                    "status": "error",
                    "error": str(e),
                    "loaded_count": 0,
                }

        # Run all health checks in parallel
        tasks = [
            check_provider(pt, p)
            for pt, p in self.providers.items()
        ]
        results = await asyncio.gather(*tasks)

        result = dict(results)
        self._health_cache = result
        self._health_cache_time = time.monotonic()
        return result

    async def check_provider_health(self, provider_type: ProviderType) -> Dict[str, Any]:
        """Check health of a specific provider."""
        provider = self.providers.get(provider_type)
        if not provider:
            return {"status": "not_enabled"}
        return await provider.health_check()

    # ==================== Provider Management ====================

    def get_provider(self, provider_type: ProviderType) -> Optional[BaseProvider]:
        """Get a provider by type."""
        return self.providers.get(provider_type)

    def get_enabled_providers(self) -> List[ProviderType]:
        """Get list of enabled provider types."""
        return list(self.providers.keys())

    async def reload_provider(self, provider_type: ProviderType):
        """Reload a provider with current settings."""
        # Unload all instances from this provider
        for instance_id, instance in list(self.instances.items()):
            if instance.provider_type == provider_type:
                await self.unload_model(instance_id)

        # Close old provider
        if provider_type in self.providers:
            await self.providers[provider_type].close()
            del self.providers[provider_type]

        # Re-initialize
        await self._init_providers()
