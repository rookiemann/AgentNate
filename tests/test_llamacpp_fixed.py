"""Test the fixed llama.cpp multi-GPU implementation."""
import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import asyncio
import time
import logging

# Enable logging to see worker communication
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

from providers.llama_cpp_provider import (
    LlamaCppProvider,
    get_available_gpus,
    estimate_model_vram,
)
from providers.base import InferenceRequest, ChatMessage

MODELS_DIR = "E:\\LL STUDIO"


async def test_fixed_implementation():
    print("=" * 70)
    print("LLAMA.CPP FIXED IMPLEMENTATION TEST")
    print("=" * 70)

    # ========== GPU Detection ==========
    print("\n--- GPU Detection ---")
    gpus = get_available_gpus()
    for gpu in gpus:
        print(f"  [{gpu['index']}] {gpu['name']}: {gpu['memory_free_mb']} MB free / {gpu['memory_total_mb']} MB")

    if len(gpus) < 2:
        print("WARNING: Need 2 GPUs for full test!")

    # ========== Initialize Provider ==========
    print("\n--- Initialize Provider ---")
    provider = LlamaCppProvider(models_directory=MODELS_DIR)

    # ========== Find Test Models ==========
    print("\n--- Find Small Models ---")
    models = await provider.list_models()

    # Filter for small chat models
    small = [
        m for m in models
        if m.get("size_bytes", 0) < 2 * 1024**3
        and "mmproj" not in m["name"].lower()
        and "embed" not in m["name"].lower()
        and "nomic" not in m["name"].lower()
        and "bitnet" not in m["name"].lower()
        and ("instruct" in m["name"].lower() or "chat" in m["name"].lower() or "it" in m["name"].lower())
    ]
    small.sort(key=lambda m: m.get("size_bytes", 0))

    if len(small) < 2:
        print("Need at least 2 small models!")
        return

    model_a, model_b = small[0], small[1]
    print(f"  Model A: {model_a['name']} ({model_a['size_bytes']/1024**3:.2f} GB)")
    print(f"  Model B: {model_b['name']} ({model_b['size_bytes']/1024**3:.2f} GB)")

    # Show VRAM estimates
    for m in [model_a, model_b]:
        est = estimate_model_vram(m['path'], 2048)
        print(f"    Estimated VRAM for {m['name']}: {est} MB")

    # ========== Test 1: Load Model A on GPU 0 ==========
    print("\n" + "=" * 70)
    print("TEST 1: Load Model A on GPU 0")
    print("=" * 70)

    start = time.time()
    inst_a = await provider.load_model(
        model_a["path"],
        n_ctx=2048,
        n_gpu_layers=99,
        gpu_index=0
    )
    print(f"[OK] Loaded in {time.time()-start:.1f}s")
    print(f"     Instance: {inst_a.id[:8]}...")
    print(f"     Status: {inst_a.status}")
    print(f"     GPU: {inst_a.gpu_index}")

    # ========== Test 2: Load Model B on GPU 1 ==========
    print("\n" + "=" * 70)
    print("TEST 2: Load Model B on GPU 1")
    print("=" * 70)

    start = time.time()
    inst_b = await provider.load_model(
        model_b["path"],
        n_ctx=2048,
        n_gpu_layers=99,
        gpu_index=1
    )
    print(f"[OK] Loaded in {time.time()-start:.1f}s")
    print(f"     Instance: {inst_b.id[:8]}...")
    print(f"     Status: {inst_b.status}")
    print(f"     GPU: {inst_b.gpu_index}")

    # ========== Test 3: Health Check ==========
    print("\n" + "=" * 70)
    print("TEST 3: Health Check")
    print("=" * 70)

    health = await provider.health_check()
    print(f"  Status: {health['status']}")
    print(f"  Workers: {health['loaded_count']}")
    print(f"  Healthy: {health['healthy_count']}")
    for wid, info in health['workers'].items():
        print(f"    {wid[:8]}: GPU={info['gpu']}, model={info['model']}, busy={info['busy']}")

    # ========== Test 4: Sequential Inference ==========
    print("\n" + "=" * 70)
    print("TEST 4: Sequential Inference")
    print("=" * 70)

    for inst, name in [(inst_a, "GPU 0"), (inst_b, "GPU 1")]:
        req = InferenceRequest(
            messages=[ChatMessage(role="user", content="Say 'Hello' and nothing else.")],
            max_tokens=20,
            temperature=0.1,
        )

        result = []
        t = 0
        async for resp in provider.chat(inst.id, req):
            if resp.text:
                result.append(resp.text)
            if resp.done:
                t = resp.total_time
            if resp.error:
                print(f"  {name}: ERROR - {resp.error}")
                break

        if result:
            print(f"  {name}: {''.join(result)[:50]} ({t:.2f}s)")

    # ========== Test 5: Parallel Inference ==========
    print("\n" + "=" * 70)
    print("TEST 5: Parallel Inference (both GPUs simultaneously)")
    print("=" * 70)

    async def chat_task(inst_id, gpu_name):
        req = InferenceRequest(
            messages=[ChatMessage(role="user", content="Count from 1 to 3, one per line.")],
            max_tokens=30,
            temperature=0.1,
        )
        result = []
        total_time = 0
        error = None
        async for resp in provider.chat(inst_id, req):
            if resp.text:
                result.append(resp.text)
            if resp.done:
                total_time = resp.total_time
            if resp.error:
                error = resp.error
                break
        return gpu_name, "".join(result), total_time, error

    print("  Starting parallel requests...")
    start = time.time()
    results = await asyncio.gather(
        chat_task(inst_a.id, "GPU 0"),
        chat_task(inst_b.id, "GPU 1"),
    )
    wall_time = time.time() - start

    print(f"\n  Results (wall time: {wall_time:.2f}s):")
    for name, text, t, err in results:
        if err:
            print(f"    {name}: ERROR - {err}")
        else:
            print(f"    {name} ({t:.2f}s): {text.strip()[:60]}")

    # Check for parallelism
    times = [r[2] for r in results if not r[3]]
    if len(times) == 2 and wall_time < sum(times) * 0.8:
        print(f"\n  [PARALLEL] Requests ran in parallel! (wall < sum)")
    elif len(times) == 2:
        print(f"\n  [SEQUENTIAL] Requests may have been sequential")

    # ========== Test 6: Load Second Model on Same GPU ==========
    print("\n" + "=" * 70)
    print("TEST 6: Load Second Model on GPU 0 (multi-model per GPU)")
    print("=" * 70)

    # Pick a different small model
    model_c = small[2] if len(small) > 2 else small[0]
    print(f"  Loading: {model_c['name']}")

    start = time.time()
    inst_c = await provider.load_model(
        model_c["path"],
        n_ctx=2048,
        n_gpu_layers=99,
        gpu_index=0  # Same GPU as model A
    )
    print(f"[OK] Loaded in {time.time()-start:.1f}s")
    print(f"     Instance: {inst_c.id[:8]}...")

    # Check GPU info
    gpu_info = await provider.get_gpu_info()
    for gpu in gpu_info:
        if gpu['models_loaded']:
            print(f"  GPU {gpu['index']}: {len(gpu['models_loaded'])} models, {gpu['effective_free_mb']} MB remaining")
            for m in gpu['models_loaded']:
                print(f"    - {m['model']}")

    # ========== Test 7: Unload Model A ==========
    print("\n" + "=" * 70)
    print("TEST 7: Unload Model A from GPU 0")
    print("=" * 70)

    await provider.unload_model(inst_a.id)
    health = await provider.health_check()
    print(f"  Remaining workers: {health['loaded_count']}")
    for wid, info in health['workers'].items():
        print(f"    {wid[:8]}: GPU={info['gpu']}, model={info['model']}")

    # ========== Test 8: Create Pool ==========
    print("\n" + "=" * 70)
    print("TEST 8: Create Pool (same model on both GPUs)")
    print("=" * 70)

    # First unload remaining models
    await provider.unload_model(inst_b.id)
    await provider.unload_model(inst_c.id)

    pool, pool_instances = await provider.create_pool(
        model_a["path"],
        gpu_indices=[0, 1],
        n_ctx=2048,
    )
    print(f"  Pool size: {pool.size}")
    print(f"  Available workers: {pool.available_count}")

    # ========== Test 9: Pooled Chat ==========
    print("\n" + "=" * 70)
    print("TEST 9: Pooled Chat with Load Balancing")
    print("=" * 70)

    for i in range(3):
        req = InferenceRequest(
            messages=[ChatMessage(role="user", content=f"Say 'Request {i+1}' exactly.")],
            max_tokens=20,
            temperature=0.1,
        )

        result = []
        async for resp in provider.chat_pooled(model_a["path"], req):
            if resp.text:
                result.append(resp.text)
            if resp.done:
                print(f"  Request {i+1}: {''.join(result)[:40]}")

    # ========== Test 10: Parallel Pooled Chat ==========
    print("\n" + "=" * 70)
    print("TEST 10: Parallel Pooled Chat")
    print("=" * 70)

    async def pooled_chat(req_num):
        req = InferenceRequest(
            messages=[ChatMessage(role="user", content=f"Say '{req_num}'")],
            max_tokens=10,
            temperature=0.1,
        )
        result = []
        t = 0
        async for resp in provider.chat_pooled(model_a["path"], req):
            if resp.text:
                result.append(resp.text)
            if resp.done:
                t = resp.total_time
        return req_num, "".join(result), t

    start = time.time()
    results = await asyncio.gather(
        pooled_chat("Alpha"),
        pooled_chat("Beta"),
    )
    wall_time = time.time() - start

    for num, text, t in results:
        print(f"  {num}: {text[:30]} ({t:.2f}s)")
    print(f"  Wall time: {wall_time:.2f}s")

    # ========== Cleanup ==========
    print("\n" + "=" * 70)
    print("CLEANUP")
    print("=" * 70)

    await provider.close()
    print("[OK] All workers closed")

    # Verify cleanup
    gpus = get_available_gpus()
    for gpu in gpus:
        print(f"  GPU {gpu['index']}: {gpu['memory_free_mb']} MB free")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_fixed_implementation())

