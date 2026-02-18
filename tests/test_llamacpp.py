"""Test llama.cpp provider with GPU detection and pool."""
import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import asyncio
from providers.llama_cpp_provider import (
    LlamaCppProvider,
    get_available_gpus,
    get_device_options,
)
from providers.base import InferenceRequest, ChatMessage

async def test_llamacpp():
    print("=== llama.cpp Provider Test ===")

    # GPU Detection
    print("\n--- GPU Detection ---")
    gpus = get_available_gpus()

    if gpus:
        print(f"Found {len(gpus)} GPU(s):")
        for gpu in gpus:
            print(f"  {gpu['display']}")
    else:
        print("No NVIDIA GPUs found (will use CPU)")

    # Device options
    print("\n--- Available Devices ---")
    devices = get_device_options()
    for dev in devices:
        print(f"  [{dev['index']:2d}] {dev['display']}")

    # Provider health check
    print("\n--- Provider Health ---")

    # Set models directory if you have one
    models_dir = os.environ.get("LLAMA_MODELS_DIR", "E:\\LL STUDIO")

    provider = LlamaCppProvider(models_directory=models_dir)
    health = await provider.health_check()

    print(f"Status: {health['status']}")
    print(f"Models directory: {health['models_directory']}")
    print(f"GPU count: {health['gpu_count']}")
    print(f"CUDA available: {health['has_cuda']}")
    print(f"Loaded workers: {health['loaded_count']}")

    # List models
    print("\n--- Available Models ---")
    models = await provider.list_models()

    if models:
        print(f"Found {len(models)} .gguf models:")
        for m in models[:10]:
            size_gb = m.get("size_bytes", 0) / (1024**3)
            vision = " [Vision]" if m.get("is_vision") else ""
            print(f"  - {m['name']} ({size_gb:.1f} GB){vision}")
        if len(models) > 10:
            print(f"  ... and {len(models) - 10} more")
    else:
        print(f"No models found in {models_dir}")
        print("\nTo test loading, set LLAMA_MODELS_DIR or place .gguf files in the directory.")

    # Example pool creation (commented out - requires models)
    print("\n--- Pool Example (conceptual) ---")
    print("""
    # Create a pool across all GPUs:
    pool, instances = await provider.create_pool(
        model_path="path/to/model.gguf",
        gpu_indices=None,  # Auto-detect all GPUs
        n_ctx=4096,
        n_gpu_layers=99
    )

    # Chat with automatic load balancing:
    async for response in provider.chat_pooled("path/to/model.gguf", request):
        print(response.text, end="")

    # Or load on specific devices:
    pool, instances = await provider.create_pool(
        model_path="path/to/model.gguf",
        gpu_indices=[0, 1],  # GPU 0 and 1 only
    )

    # CPU-only:
    pool, instances = await provider.create_pool(
        model_path="path/to/model.gguf",
        gpu_indices=[-1],  # CPU
        n_gpu_layers=0
    )
    """)

    await provider.close()
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    asyncio.run(test_llamacpp())

