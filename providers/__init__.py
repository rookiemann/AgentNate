"""
AgentNate Providers Package

Unified interface for multiple LLM backends:
- LlamaCpp: Local llama.cpp inference via subprocess
- LMStudio: OpenAI-compatible API at localhost:1234
- OpenRouter: Cloud API with 100+ models
- Ollama: Local Ollama API at localhost:11434
- vLLM: High-throughput local inference with PagedAttention
"""

from .base import (
    ProviderType,
    ModelStatus,
    ModelInstance,
    ChatMessage,
    InferenceRequest,
    InferenceResponse,
    BaseProvider,
)

__all__ = [
    "ProviderType",
    "ModelStatus",
    "ModelInstance",
    "ChatMessage",
    "InferenceRequest",
    "InferenceResponse",
    "BaseProvider",
]
