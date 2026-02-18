"""
Base classes and data structures for LLM providers.

All providers implement BaseProvider to ensure uniform interface
across different backends (local, API, cloud).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional, List, Dict, Any, Callable
from enum import Enum
import uuid
import time


class ProviderType(Enum):
    """Supported LLM provider types."""
    LLAMA_CPP = "llama_cpp"
    LM_STUDIO = "lm_studio"
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    VLLM = "vllm"


class ModelStatus(Enum):
    """Model instance lifecycle states."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"


@dataclass
class ModelInstance:
    """Represents a loaded model instance."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    provider_type: ProviderType = ProviderType.LLAMA_CPP
    model_identifier: str = ""  # Path for local, model ID for API
    status: ModelStatus = ModelStatus.UNLOADED
    display_name: str = ""
    context_length: int = 4096
    metadata: Dict[str, Any] = field(default_factory=dict)
    # For slots/warm instances
    slot_name: Optional[str] = None
    # Tracking
    last_used: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    request_count: int = 0
    # GPU assignment
    gpu_index: Optional[int] = None  # None = auto, -1 = CPU, 0+ = specific GPU

    def touch(self):
        """Update last_used timestamp."""
        self.last_used = time.time()
        self.request_count += 1


@dataclass
class ChatMessage:
    """A single message in a chat conversation."""
    role: str  # "system", "user", "assistant"
    content: str
    # Optional for vision models
    images: Optional[List[str]] = None  # List of image paths or URLs

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for API calls."""
        if self.images:
            # Vision format - OpenAI-compatible multimodal content
            content_parts = [{"type": "text", "text": self.content}]
            for img in self.images:
                if img.startswith("data:"):
                    # Base64 data URI (from frontend upload)
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": img}
                    })
                elif img.startswith(("http://", "https://")):
                    # Remote URL
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": img}
                    })
                else:
                    # Local file path
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"file:///{img.replace(chr(92), '/')}"}
                    })
            return {"role": self.role, "content": content_parts}
        return {"role": self.role, "content": self.content}


@dataclass
class InferenceRequest:
    """Standardized inference request across all providers."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[ChatMessage] = field(default_factory=list)

    # Generation parameters
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40

    # Penalty parameters
    repeat_penalty: float = 1.1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    # Mirostat parameters
    mirostat: int = 0  # 0=off, 1=v1, 2=v2
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1

    # Additional sampling
    typical_p: float = 1.0
    tfs_z: float = 1.0

    # Queue priority (higher = more important)
    priority: int = 0

    # Speculative decoding (llama.cpp only)
    draft_model_id: Optional[str] = None

    # Stop sequences
    stop: Optional[List[str]] = None

    # Callback for streaming (alternative to async iteration)
    stream_callback: Optional[Callable[[str], None]] = None

    def to_messages_dict(self) -> List[Dict[str, Any]]:
        """Convert messages to list of dicts for API calls."""
        return [msg.to_dict() for msg in self.messages]


@dataclass
class InferenceResponse:
    """Response from inference, used for both streaming and final results."""
    request_id: str
    text: str = ""
    done: bool = False
    error: Optional[str] = None

    # Token usage (populated on completion)
    usage: Dict[str, int] = field(default_factory=dict)
    # prompt_tokens, completion_tokens, total_tokens

    # Timing info
    time_to_first_token: Optional[float] = None
    total_time: Optional[float] = None
    tokens_per_second: Optional[float] = None


class BaseProvider(ABC):
    """
    Abstract base class for all LLM providers.

    Providers manage model instances and handle inference requests.
    All methods are async to support non-blocking operations.
    """

    def __init__(self, provider_type: ProviderType):
        self.provider_type = provider_type
        self.instances: Dict[str, ModelInstance] = {}
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    @abstractmethod
    async def load_model(
        self,
        model_identifier: str,
        **kwargs
    ) -> ModelInstance:
        """
        Load a model and return instance info.

        Args:
            model_identifier: Path for local models, model ID for APIs
            **kwargs: Provider-specific options (n_ctx, n_gpu_layers, etc.)

        Returns:
            ModelInstance with status READY on success

        Raises:
            Exception on failure
        """
        pass

    @abstractmethod
    async def unload_model(self, instance_id: str) -> bool:
        """
        Unload a specific model instance.

        Args:
            instance_id: UUID of the instance to unload

        Returns:
            True if successfully unloaded, False if not found
        """
        pass

    @abstractmethod
    async def chat(
        self,
        instance_id: str,
        request: InferenceRequest
    ) -> AsyncIterator[InferenceResponse]:
        """
        Stream chat completions.

        Args:
            instance_id: UUID of the model instance
            request: InferenceRequest with messages and parameters

        Yields:
            InferenceResponse objects with streaming tokens
            Final response has done=True
        """
        pass

    @abstractmethod
    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models for this provider.

        Returns:
            List of model info dicts with at least:
            - id: Model identifier
            - name: Display name
            - provider: Provider type string
        """
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Check provider health/connectivity.

        Returns:
            Dict with at least:
            - provider: Provider type string
            - status: "healthy", "offline", "error", etc.
        """
        pass

    @abstractmethod
    async def get_status(self, instance_id: str) -> ModelStatus:
        """
        Get status of a specific instance.

        Args:
            instance_id: UUID of the instance

        Returns:
            ModelStatus enum value
        """
        pass

    def get_instance(self, instance_id: str) -> Optional[ModelInstance]:
        """Get instance by ID if it exists."""
        return self.instances.get(instance_id)

    def get_all_instances(self) -> List[ModelInstance]:
        """Get all instances for this provider."""
        return list(self.instances.values())

    def get_ready_instances(self) -> List[ModelInstance]:
        """Get all instances in READY state."""
        return [i for i in self.instances.values() if i.status == ModelStatus.READY]

    def find_instance_by_model(self, model_identifier: str) -> Optional[ModelInstance]:
        """Find an existing instance for a model identifier."""
        for instance in self.instances.values():
            if instance.model_identifier == model_identifier:
                if instance.status in (ModelStatus.READY, ModelStatus.BUSY):
                    return instance
        return None

    async def close(self):
        """Clean up provider resources. Override if needed."""
        # Unload all instances
        for instance_id in list(self.instances.keys()):
            await self.unload_model(instance_id)
