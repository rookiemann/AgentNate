"""Comprehensive test: Multi-model loading, unloading, parallelism on both llama.cpp and LM Studio."""
import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import asyncio
import time
import threading

print("=" * 70)
print("COMPREHENSIVE MODEL LOADING/UNLOADING/PARALLELISM TEST")
print("=" * 70)

# ============================================================
# PART 1: LLAMA.CPP SUBPROCESS ISOLATION
# ============================================================

async def test_llamacpp():
    from providers.llama_cpp_provider import LlamaCppProvider, get_available_gpus
    from providers.base import InferenceRequest, ChatMessage

    print("\n" + "=" * 70)
    print("PART 1: LLAMA.CPP - SUBPROCESS ISOLATION")
    print("=" * 70)

    gpus = get_available_gpus()
    print(f"\nDetected GPUs: {len(gpus)}")
    for g in gpus:
        print(f"  [{g['index']}] {g['name']} ({g['memory_mb']} MB)")

    provider = LlamaCppProvider(models_directory="E:\\LL STUDIO")
    models = await provider.list_models()

    # Find small models for testing
    small = [m for m in models
             if m.get("size_bytes", 0) < 2 * 1024**3
             and "mmproj" not in m["name"].lower()
             and "embed" not in m["name"].lower()
             and ("instruct" in m["name"].lower() or "chat" in m["name"].lower())]
    small.sort(key=lambda m: m.get("size_bytes", 0))

    if len(small) < 2:
        print("Need at least 2 small models!")
        return

    model1, model2 = small[0], small[1]
    print(f"\nTest models:")
    print(f"  Model A: {model1['name']} ({model1['size_bytes']/1024**3:.2f} GB)")
    print(f"  Model B: {model2['name']} ({model2['size_bytes']/1024**3:.2f} GB)")

    # Test 1: Load on GPU 0
    print("\n--- Test 1: Load model A on GPU 0 ---")
    inst_a = await provider.load_model(model1["path"], n_ctx=2048, gpu_index=0)
    print(f"[OK] Loaded: {inst_a.id[:8]}... on GPU 0")

    # Test 2: Load on GPU 1
    print("\n--- Test 2: Load model B on GPU 1 ---")
    inst_b = await provider.load_model(model2["path"], n_ctx=2048, gpu_index=1)
    print(f"[OK] Loaded: {inst_b.id[:8]}... on GPU 1")

    # Test 3: Sequential inference
    print("\n--- Test 3: Sequential inference ---")
    req = InferenceRequest(
        messages=[ChatMessage(role="user", content="Say 'test' once.")],
        max_tokens=10, temperature=0.1
    )

    for inst, name in [(inst_a, "GPU 0"), (inst_b, "GPU 1")]:
        result = []
        async for r in provider.chat(inst.id, req):
            if r.text:
                result.append(r.text)
            if r.done:
                print(f"  {name}: {''.join(result)[:30]} ({r.total_time:.2f}s)")

    # Test 4: Parallel inference
    print("\n--- Test 4: Parallel inference (both GPUs) ---")
    async def infer(inst_id, gpu_name):
        req = InferenceRequest(
            messages=[ChatMessage(role="user", content="Count 1 to 3.")],
            max_tokens=20, temperature=0.1
        )
        result = []
        t = 0
        async for r in provider.chat(inst_id, req):
            if r.text:
                result.append(r.text)
            if r.done:
                t = r.total_time
        return gpu_name, "".join(result), t

    start = time.time()
    results = await asyncio.gather(
        infer(inst_a.id, "GPU 0"),
        infer(inst_b.id, "GPU 1"),
    )
    total = time.time() - start

    for name, text, t in results:
        print(f"  {name} ({t:.2f}s): {text[:40]}")
    print(f"  Total wall-clock: {total:.2f}s")

    # Test 5: Unload model A
    print("\n--- Test 5: Unload model A ---")
    await provider.unload_model(inst_a.id)
    health = await provider.health_check()
    print(f"  Remaining workers: {health['loaded_count']}")

    # Test 6: Load multiple instances of SAME model on different GPUs
    print("\n--- Test 6: Same model on both GPUs (pool) ---")
    pool, instances = await provider.create_pool(
        model1["path"],
        gpu_indices=[0, 1],
        n_ctx=2048
    )
    print(f"  Pool created with {pool.size} workers")
    print(f"  Available: {pool.available_count}")

    # Test 7: Pooled chat
    print("\n--- Test 7: Pooled inference ---")
    req = InferenceRequest(
        messages=[ChatMessage(role="user", content="Hello!")],
        max_tokens=10, temperature=0.1
    )
    result = []
    async for r in provider.chat_pooled(model1["path"], req):
        if r.text:
            result.append(r.text)
        if r.done:
            print(f"  Response: {''.join(result)[:40]}")

    # Cleanup
    print("\n--- Cleanup ---")
    await provider.close()
    print("[OK] All llama.cpp workers closed")


# ============================================================
# PART 2: LM STUDIO v4 SDK
# ============================================================

def test_lmstudio():
    import lmstudio
    from lmstudio import LlmLoadModelConfig
    from lmstudio._sdk_models import GpuSetting

    print("\n" + "=" * 70)
    print("PART 2: LM STUDIO v4 - SDK FEATURES")
    print("=" * 70)

    api_host = lmstudio.Client.find_default_local_api_host()
    print(f"\nSDK API: {api_host}")
    client = lmstudio.Client(api_host=api_host)
    time.sleep(1)

    # Clean slate
    for m in list(client.llm.list_loaded()):
        client.llm.unload(m.identifier)

    model_path = "lmstudio-community/Phi-4-mini-instruct-GGUF/Phi-4-mini-instruct-Q8_0.gguf"
    print(f"Test model: {model_path.split('/')[-1]}")

    # Test 1: Load on GPU 0
    print("\n--- Test 1: Load on GPU 0 ---")
    cfg0 = LlmLoadModelConfig(
        gpu=GpuSetting(main_gpu=0, disabled_gpus=[1]),
        context_length=2048
    )
    m0 = client.llm.load_new_instance(model_path, "gpu0-test", config=cfg0, ttl=120)
    print(f"[OK] Loaded: {m0.identifier}")

    # Test 2: Load on GPU 1
    print("\n--- Test 2: Load on GPU 1 ---")
    cfg1 = LlmLoadModelConfig(
        gpu=GpuSetting(main_gpu=1, disabled_gpus=[0]),
        context_length=2048
    )
    m1 = client.llm.load_new_instance(model_path, "gpu1-test", config=cfg1, ttl=120)
    print(f"[OK] Loaded: {m1.identifier}")

    # Test 3: List loaded instances
    print("\n--- Test 3: List loaded instances ---")
    loaded = list(client.llm.list_loaded())
    for m in loaded:
        print(f"  - {m.identifier}")

    # Test 4: Sequential inference
    print("\n--- Test 4: Sequential inference ---")
    r0 = m0.respond("Say 'GPU zero' exactly.", config={"max_tokens": 10})
    print(f"  GPU 0: {r0.content[:40]}")
    r1 = m1.respond("Say 'GPU one' exactly.", config={"max_tokens": 10})
    print(f"  GPU 1: {r1.content[:40]}")

    # Test 5: Parallel inference (different GPUs)
    print("\n--- Test 5: Parallel inference (both GPUs) ---")
    results = {}
    timings = {}

    def infer(model, key, prompt):
        start = time.time()
        r = model.respond(prompt, config={"max_tokens": 20})
        results[key] = r.content
        timings[key] = time.time() - start

    threads = [
        threading.Thread(target=infer, args=(m0, "GPU0", "Count 1 to 3.")),
        threading.Thread(target=infer, args=(m1, "GPU1", "Count 4 to 6.")),
    ]

    start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total = time.time() - start

    for k in ["GPU0", "GPU1"]:
        print(f"  {k} ({timings[k]:.2f}s): {results[k][:40]}")
    print(f"  Total wall-clock: {total:.2f}s")

    # Test 6: Unload GPU 0 instance
    print("\n--- Test 6: Unload GPU 0 instance ---")
    m0.unload()
    loaded = list(client.llm.list_loaded())
    print(f"  Remaining: {len(loaded)} instance(s)")

    # Test 7: Parallel requests to SINGLE model (continuous batching)
    print("\n--- Test 7: Continuous batching on single model ---")
    results = {}
    timings = {}

    threads = []
    for i in range(3):
        t = threading.Thread(
            target=infer,
            args=(m1, f"Req{i}", f"What is {i+1}+1?")
        )
        threads.append(t)

    start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total = time.time() - start

    individual_sum = sum(timings.values())
    for k in ["Req0", "Req1", "Req2"]:
        print(f"  {k} ({timings[k]:.2f}s): {results[k][:40]}")
    print(f"  Total: {total:.2f}s (sum: {individual_sum:.2f}s)")

    if total < individual_sum * 0.7:
        print("  [PARALLEL] Continuous batching confirmed!")
    else:
        print("  [SEQUENTIAL] Requests processed sequentially")

    # Cleanup
    print("\n--- Cleanup ---")
    m1.unload()
    client.close()
    print("[OK] LM Studio cleanup complete")


# ============================================================
# RUN ALL TESTS
# ============================================================

if __name__ == "__main__":
    # Run llama.cpp async tests
    asyncio.run(test_llamacpp())

    # Run LM Studio sync tests
    test_lmstudio()

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)

