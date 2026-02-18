"""Test llama.cpp multi-GPU pool with isolated model loading."""
import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import asyncio
from providers.llama_cpp_provider import (
    LlamaCppProvider,
    get_available_gpus,
)
from providers.base import InferenceRequest, ChatMessage

# Models directory
MODELS_DIR = "E:\\LL STUDIO"

async def test_multi_gpu_pool():
    print("=== llama.cpp Multi-GPU Pool Test ===")
    print("Goal: Load one small model per GPU, verify isolation\n")

    # Detect GPUs
    gpus = get_available_gpus()
    print(f"Detected {len(gpus)} GPU(s):")
    for gpu in gpus:
        print(f"  [{gpu['index']}] {gpu['name']} ({gpu['memory_mb']} MB)")

    if len(gpus) < 2:
        print("\nNeed at least 2 GPUs for this test!")
        return

    # Initialize provider
    provider = LlamaCppProvider(models_directory=MODELS_DIR)

    # Find 2 small models (under 4GB)
    print("\n--- Finding Small Models ---")
    all_models = await provider.list_models()

    small_models = [
        m for m in all_models
        if m.get("size_bytes", 0) < 4 * 1024**3  # Under 4GB
        and "mmproj" not in m["name"].lower()  # Skip vision projectors
        and not m.get("is_vision")  # Skip vision models (need projector)
        and "embed" not in m["name"].lower()  # Skip embedding models
        and "nomic" not in m["name"].lower()  # Skip nomic embedding
        and "bitnet" not in m["name"].lower()  # Skip bitnet (special format)
        and "i2_s" not in m["name"].lower()  # Skip bitnet quantization
        and ("instruct" in m["name"].lower() or "chat" in m["name"].lower()
             or "it" in m["name"].lower())  # Prefer instruction-tuned
    ]

    # Sort by size
    small_models.sort(key=lambda m: m.get("size_bytes", 0))

    print(f"Found {len(small_models)} small models (< 4GB):")
    for m in small_models[:5]:
        size_gb = m.get("size_bytes", 0) / (1024**3)
        print(f"  - {m['name']} ({size_gb:.2f} GB)")

    if len(small_models) < 2:
        print("Need at least 2 small models!")
        await provider.close()
        return

    # Select 2 different small models
    model1 = small_models[0]
    model2 = small_models[1] if len(small_models) > 1 else small_models[0]

    print(f"\nSelected models:")
    print(f"  GPU 0: {model1['name']}")
    print(f"  GPU 1: {model2['name']}")

    # ========== Load Model 1 on GPU 0 ==========
    print(f"\n{'='*50}")
    print(f"LOADING MODEL 1 ON GPU 0 (RTX 3060)")
    print(f"Model: {model1['name']}")
    print(f"{'='*50}")

    try:
        instance1 = await provider.load_model(
            model_identifier=model1["path"],
            n_ctx=2048,  # Small context for testing
            n_gpu_layers=99,  # Full GPU offload
            gpu_index=0,  # GPU 0 only
        )
        print(f"[OK] Instance 1 created: {instance1.id[:8]}...")
        print(f"  Status: {instance1.status}")
        print(f"  GPU: {instance1.gpu_index}")
    except Exception as e:
        print(f"[FAIL] Failed to load model 1: {e}")
        await provider.close()
        return

    # Brief pause to let GPU settle
    await asyncio.sleep(2)

    # ========== Load Model 2 on GPU 1 ==========
    print(f"\n{'='*50}")
    print(f"LOADING MODEL 2 ON GPU 1 (RTX 3090)")
    print(f"Model: {model2['name']}")
    print(f"{'='*50}")

    try:
        instance2 = await provider.load_model(
            model_identifier=model2["path"],
            n_ctx=2048,
            n_gpu_layers=99,
            gpu_index=1,  # GPU 1 only
        )
        print(f"[OK] Instance 2 created: {instance2.id[:8]}...")
        print(f"  Status: {instance2.status}")
        print(f"  GPU: {instance2.gpu_index}")
    except Exception as e:
        print(f"[FAIL] Failed to load model 2: {e}")
        # Continue to test model 1

    # ========== Check Health ==========
    print(f"\n{'='*50}")
    print("PROVIDER STATUS")
    print(f"{'='*50}")

    health = await provider.health_check()
    print(f"Loaded workers: {health['loaded_count']}")
    print(f"Workers:")
    for wid, info in health.get("workers", {}).items():
        print(f"  - {wid[:8]}...: GPU={info['gpu']}, loaded={info['loaded']}, busy={info['busy']}")

    # ========== Test Chat with Model 1 ==========
    print(f"\n{'='*50}")
    print("CHAT TEST: MODEL 1 (GPU 0)")
    print(f"{'='*50}")

    request1 = InferenceRequest(
        messages=[
            ChatMessage(role="user", content="Say 'Hello from GPU zero' in exactly those 4 words.")
        ],
        max_tokens=30,
        temperature=0.3,
    )

    print("Response: ", end="", flush=True)
    async for response in provider.chat(instance1.id, request1):
        if response.text:
            print(response.text, end="", flush=True)
        if response.done:
            print(f"\n[Done - {response.total_time:.2f}s]")
        if response.error:
            print(f"\n[Error: {response.error}]")

    # ========== Test Chat with Model 2 ==========
    if 'instance2' in dir():
        print(f"\n{'='*50}")
        print("CHAT TEST: MODEL 2 (GPU 1)")
        print(f"{'='*50}")

        request2 = InferenceRequest(
            messages=[
                ChatMessage(role="user", content="Say 'Hello from GPU one' in exactly those 4 words.")
            ],
            max_tokens=30,
            temperature=0.3,
        )

        print("Response: ", end="", flush=True)
        async for response in provider.chat(instance2.id, request2):
            if response.text:
                print(response.text, end="", flush=True)
            if response.done:
                print(f"\n[Done - {response.total_time:.2f}s]")
            if response.error:
                print(f"\n[Error: {response.error}]")

    # ========== Parallel Chat Test ==========
    print(f"\n{'='*50}")
    print("PARALLEL CHAT TEST (Both GPUs)")
    print(f"{'='*50}")

    async def chat_task(instance_id, gpu_num, provider):
        request = InferenceRequest(
            messages=[
                ChatMessage(role="user", content=f"Count from 1 to 5, each number on a new line.")
            ],
            max_tokens=50,
            temperature=0.1,
        )
        result = []
        async for response in provider.chat(instance_id, request):
            if response.text:
                result.append(response.text)
            if response.done:
                return f"GPU {gpu_num}", "".join(result), response.total_time
            if response.error:
                return f"GPU {gpu_num}", f"ERROR: {response.error}", 0
        return f"GPU {gpu_num}", "".join(result), 0

    # Run both chats in parallel
    if 'instance2' in dir():
        print("Running parallel inference on both GPUs...")
        tasks = [
            chat_task(instance1.id, 0, provider),
            chat_task(instance2.id, 1, provider),
        ]
        results = await asyncio.gather(*tasks)

        for gpu, text, time_taken in results:
            print(f"\n{gpu} ({time_taken:.2f}s):")
            print(f"  {text.strip()[:100]}...")

    # ========== Cleanup ==========
    print(f"\n{'='*50}")
    print("CLEANUP")
    print(f"{'='*50}")

    await provider.close()
    print("[OK] All workers closed")

    print("\n=== Test Complete ===")

if __name__ == "__main__":
    asyncio.run(test_multi_gpu_pool())

