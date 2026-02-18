"""Stress test llama.cpp with larger models on each GPU."""
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

MODELS_DIR = "E:\\LL STUDIO"

async def test_large_models():
    print("=== llama.cpp Large Model Stress Test ===")
    print("Goal: Load larger models to stress GPU memory\n")

    # Detect GPUs
    gpus = get_available_gpus()
    print(f"Available GPUs:")
    for gpu in gpus:
        print(f"  [{gpu['index']}] {gpu['name']} ({gpu['memory_mb']/1024:.1f} GB)")

    # Initialize provider
    provider = LlamaCppProvider(models_directory=MODELS_DIR)

    # Find larger models (4-10GB range for good stress test)
    print("\n--- Finding Larger Models (4-10GB) ---")
    all_models = await provider.list_models()

    large_models = [
        m for m in all_models
        if 4 * 1024**3 < m.get("size_bytes", 0) < 10 * 1024**3  # 4-10GB
        and "mmproj" not in m["name"].lower()
        and not m.get("is_vision")
        and "embed" not in m["name"].lower()
    ]

    # Sort by size descending
    large_models.sort(key=lambda m: m.get("size_bytes", 0), reverse=True)

    print(f"Found {len(large_models)} models in 4-10GB range:")
    for m in large_models[:8]:
        size_gb = m.get("size_bytes", 0) / (1024**3)
        print(f"  - {m['name']} ({size_gb:.2f} GB)")

    if len(large_models) < 2:
        print("Need at least 2 large models for stress test!")
        # Fall back to any models over 3GB
        large_models = [
            m for m in all_models
            if m.get("size_bytes", 0) > 3 * 1024**3
            and "mmproj" not in m["name"].lower()
            and not m.get("is_vision")
            and "embed" not in m["name"].lower()
        ]
        large_models.sort(key=lambda m: m.get("size_bytes", 0), reverse=True)
        print(f"\nFallback - found {len(large_models)} models over 3GB")

    if len(large_models) < 2:
        print("Not enough large models available!")
        await provider.close()
        return

    # Select models - pick ones likely to be good chat models
    # Prefer models with instruct/chat in name
    instruct_models = [m for m in large_models if
                       "instruct" in m["name"].lower() or
                       "chat" in m["name"].lower()]

    if len(instruct_models) >= 2:
        model1 = instruct_models[0]
        model2 = instruct_models[1]
    else:
        model1 = large_models[0]
        model2 = large_models[1] if len(large_models) > 1 else large_models[0]

    size1 = model1.get("size_bytes", 0) / (1024**3)
    size2 = model2.get("size_bytes", 0) / (1024**3)

    print(f"\nSelected for stress test:")
    print(f"  GPU 0 (12GB): {model1['name']} ({size1:.2f} GB)")
    print(f"  GPU 1 (24GB): {model2['name']} ({size2:.2f} GB)")

    # ========== Load Model 1 on GPU 0 ==========
    print(f"\n{'='*60}")
    print(f"LOADING ON GPU 0 (RTX 3060 - 12GB)")
    print(f"Model: {model1['name']} ({size1:.2f} GB)")
    print(f"{'='*60}")

    try:
        instance1 = await provider.load_model(
            model_identifier=model1["path"],
            n_ctx=4096,
            n_gpu_layers=99,
            gpu_index=0,
        )
        print(f"[OK] Instance 1 loaded: {instance1.id[:8]}...")
        print(f"     Status: {instance1.status}")
    except Exception as e:
        print(f"[FAIL] Failed to load: {e}")
        await provider.close()
        return

    # Wait for GPU memory to settle
    print("Waiting for GPU memory to settle...")
    await asyncio.sleep(3)

    # ========== Load Model 2 on GPU 1 ==========
    print(f"\n{'='*60}")
    print(f"LOADING ON GPU 1 (RTX 3090 - 24GB)")
    print(f"Model: {model2['name']} ({size2:.2f} GB)")
    print(f"{'='*60}")

    try:
        instance2 = await provider.load_model(
            model_identifier=model2["path"],
            n_ctx=4096,
            n_gpu_layers=99,
            gpu_index=1,
        )
        print(f"[OK] Instance 2 loaded: {instance2.id[:8]}...")
        print(f"     Status: {instance2.status}")
    except Exception as e:
        print(f"[FAIL] Failed to load: {e}")
        # Continue with just model 1
        instance2 = None

    # ========== Health Check ==========
    print(f"\n{'='*60}")
    print("PROVIDER STATUS")
    print(f"{'='*60}")

    health = await provider.health_check()
    print(f"Loaded workers: {health['loaded_count']}")
    for wid, info in health.get("workers", {}).items():
        print(f"  Worker {wid[:8]}: GPU={info['gpu']}, loaded={info['loaded']}")

    # ========== Chat Test GPU 0 ==========
    print(f"\n{'='*60}")
    print("CHAT TEST: GPU 0")
    print(f"{'='*60}")

    request1 = InferenceRequest(
        messages=[
            ChatMessage(role="system", content="You are a helpful assistant. Be concise."),
            ChatMessage(role="user", content="Explain quantum computing in 2 sentences.")
        ],
        max_tokens=100,
        temperature=0.7,
    )

    print("Prompt: Explain quantum computing in 2 sentences.")
    print("Response: ", end="", flush=True)

    start = asyncio.get_event_loop().time()
    tokens = 0
    async for response in provider.chat(instance1.id, request1):
        if response.text:
            print(response.text, end="", flush=True)
            tokens += 1
        if response.done:
            elapsed = asyncio.get_event_loop().time() - start
            print(f"\n[Done - {response.total_time:.2f}s, ~{tokens} tokens, {tokens/elapsed:.1f} t/s]")
        if response.error:
            print(f"\n[Error: {response.error}]")

    # ========== Chat Test GPU 1 ==========
    if instance2:
        print(f"\n{'='*60}")
        print("CHAT TEST: GPU 1")
        print(f"{'='*60}")

        request2 = InferenceRequest(
            messages=[
                ChatMessage(role="system", content="You are a helpful assistant. Be concise."),
                ChatMessage(role="user", content="What is the meaning of life? Answer philosophically in 2 sentences.")
            ],
            max_tokens=100,
            temperature=0.7,
        )

        print("Prompt: What is the meaning of life?")
        print("Response: ", end="", flush=True)

        start = asyncio.get_event_loop().time()
        tokens = 0
        async for response in provider.chat(instance2.id, request2):
            if response.text:
                print(response.text, end="", flush=True)
                tokens += 1
            if response.done:
                elapsed = asyncio.get_event_loop().time() - start
                print(f"\n[Done - {response.total_time:.2f}s, ~{tokens} tokens, {tokens/elapsed:.1f} t/s]")
            if response.error:
                print(f"\n[Error: {response.error}]")

    # ========== Parallel Generation Test ==========
    if instance2:
        print(f"\n{'='*60}")
        print("PARALLEL GENERATION TEST")
        print("Running same prompt on both GPUs simultaneously...")
        print(f"{'='*60}")

        async def generate_story(inst_id, gpu_num, provider):
            request = InferenceRequest(
                messages=[
                    ChatMessage(role="user",
                               content="Write a haiku about artificial intelligence.")
                ],
                max_tokens=50,
                temperature=0.8,
            )
            result = []
            total_time = 0
            async for response in provider.chat(inst_id, request):
                if response.text:
                    result.append(response.text)
                if response.done:
                    total_time = response.total_time
                if response.error:
                    return gpu_num, f"ERROR: {response.error}", 0
            return gpu_num, "".join(result), total_time

        import time
        parallel_start = time.time()

        results = await asyncio.gather(
            generate_story(instance1.id, 0, provider),
            generate_story(instance2.id, 1, provider),
        )

        parallel_total = time.time() - parallel_start

        print(f"\nParallel execution completed in {parallel_total:.2f}s total\n")

        for gpu_num, text, gen_time in results:
            print(f"GPU {gpu_num} ({gen_time:.2f}s):")
            print(f"  {text.strip()}")
            print()

    # ========== Cleanup ==========
    print(f"{'='*60}")
    print("CLEANUP")
    print(f"{'='*60}")

    await provider.close()
    print("[OK] All workers closed")
    print("\n=== Stress Test Complete ===")

if __name__ == "__main__":
    asyncio.run(test_large_models())

