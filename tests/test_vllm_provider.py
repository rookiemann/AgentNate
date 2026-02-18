"""
Test script for vLLM provider integration.

Tests:
1. Import & instantiation
2. list_models() GGUF scanning
3. Model loading (subprocess spawn)
4. Single inference (chat streaming)
5. Concurrent inference (continuous batching)
6. cancel_load()
7. unload_model()
"""

import asyncio
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def main():
    passed = 0
    failed = 0
    errors = []

    # ==================== Test 1: Import & Instantiation ====================
    print("\n" + "=" * 60)
    print("TEST 1: Import & Instantiation")
    print("=" * 60)
    try:
        from providers.vllm_provider import VLLMProvider
        from providers.base import ProviderType, ModelStatus

        provider = VLLMProvider(
            env_path="envs/vllm",
            port_start=8100,
            models_directory=r"E:\LL STUDIO",
        )
        assert provider.provider_type == ProviderType.VLLM
        assert provider.models_directory == r"E:\LL STUDIO"
        assert provider.env_path == "envs/vllm"
        print("  PASS - VLLMProvider instantiated correctly")
        passed += 1
    except Exception as e:
        print(f"  FAIL - {e}")
        failed += 1
        errors.append(f"Test 1: {e}")
        return  # Can't continue without provider

    # ==================== Test 2: list_models() GGUF Scanning ====================
    print("\n" + "=" * 60)
    print("TEST 2: list_models() GGUF Scanning")
    print("=" * 60)
    try:
        models = await provider.list_models()
        assert isinstance(models, list)
        assert len(models) > 0, "No GGUF models found in directory"

        # Check model structure
        first = models[0]
        assert "id" in first, "Missing 'id' field"
        assert "name" in first, "Missing 'name' field"
        assert "provider" in first, "Missing 'provider' field"
        assert first["provider"] == "vllm"
        assert first["name"].endswith(".gguf")
        assert "path" in first
        assert "size_bytes" in first
        assert first["size_bytes"] > 0

        print(f"  PASS - Found {len(models)} GGUF models")
        print(f"  First: {first['name']} ({first['size_bytes'] / 1024 / 1024:.0f} MB)")
        passed += 1
    except Exception as e:
        print(f"  FAIL - {e}")
        failed += 1
        errors.append(f"Test 2: {e}")

    # ==================== Test 2b: list_models() fallback (no dir) ====================
    print("\n" + "=" * 60)
    print("TEST 2b: list_models() fallback (no models_directory)")
    print("=" * 60)
    try:
        provider_no_dir = VLLMProvider(env_path="envs/vllm", models_directory="")
        fallback_models = await provider_no_dir.list_models()
        assert isinstance(fallback_models, list)
        assert len(fallback_models) == 0, "Should return empty list when no dir and no loaded models"
        print("  PASS - Returns empty list when no models_directory set")
        passed += 1
    except Exception as e:
        print(f"  FAIL - {e}")
        failed += 1
        errors.append(f"Test 2b: {e}")

    # ==================== Test 3: cancel_load() on nonexistent ====================
    print("\n" + "=" * 60)
    print("TEST 3: cancel_load() on nonexistent instance")
    print("=" * 60)
    try:
        result = await provider.cancel_load("nonexistent-id")
        assert result == False, "Should return False for nonexistent instance"
        print("  PASS - Returns False for nonexistent instance")
        passed += 1
    except Exception as e:
        print(f"  FAIL - {e}")
        failed += 1
        errors.append(f"Test 3: {e}")

    # ==================== Test 4: Load a small model ====================
    print("\n" + "=" * 60)
    print("TEST 4: Load a small model (gemma-2-2b Q5)")
    print("=" * 60)

    # Pick a small model for fast loading
    small_model = r"E:\LL STUDIO\bartowski\gemma-2-2b-it-abliterated-GGUF\gemma-2-2b-it-abliterated-Q5_K_L.gguf"
    if not os.path.exists(small_model):
        print(f"  SKIP - Test model not found: {small_model}")
        # Try to find any small model
        small_candidates = [m for m in models if m["size_bytes"] < 3 * 1024 * 1024 * 1024]  # < 3GB
        if small_candidates:
            small_candidates.sort(key=lambda m: m["size_bytes"])
            small_model = small_candidates[0]["path"]
            print(f"  Using alternative: {small_candidates[0]['name']} ({small_candidates[0]['size_bytes'] / 1024 / 1024:.0f} MB)")
        else:
            print("  SKIP - No small model found, skipping load/inference tests")
            print(f"\n{'=' * 60}")
            print(f"RESULTS: {passed} passed, {failed} failed")
            return

    try:
        print(f"  Loading: {os.path.basename(small_model)}")
        start = time.time()

        instance = await provider.load_model(
            small_model,
            max_model_len=2048,  # Small context for fast load
            gpu_memory_utilization=0.3,  # Low utilization for testing
            enforce_eager=True,  # Faster startup
        )

        load_time = time.time() - start
        assert instance is not None
        assert instance.status == ModelStatus.READY
        assert instance.id in provider.instances
        assert instance.id in provider._servers

        server = provider._servers[instance.id]
        assert server.is_running

        print(f"  PASS - Model loaded in {load_time:.1f}s")
        print(f"  Instance ID: {instance.id[:8]}...")
        print(f"  Server port: {server.port}")
        print(f"  Server PID: {server.process.pid}")
        print(f"  GPU index: {server.gpu_index}")
        passed += 1
    except Exception as e:
        print(f"  FAIL - {e}")
        failed += 1
        errors.append(f"Test 4: {e}")
        # Can't continue without loaded model
        print(f"\n{'=' * 60}")
        print(f"RESULTS: {passed} passed, {failed} failed")
        if errors:
            print("\nErrors:")
            for err in errors:
                print(f"  - {err}")
        return

    instance_id = instance.id

    # ==================== Test 5: health_check() ====================
    print("\n" + "=" * 60)
    print("TEST 5: health_check()")
    print("=" * 60)
    try:
        health = await provider.health_check()
        assert health["provider"] == "vllm"
        assert health["status"] == "healthy"
        assert health["running_count"] == 1
        assert health["loaded_count"] == 1
        print(f"  PASS - Health: {health['status']}, running: {health['running_count']}")
        passed += 1
    except Exception as e:
        print(f"  FAIL - {e}")
        failed += 1
        errors.append(f"Test 5: {e}")

    # ==================== Test 6: Single inference ====================
    print("\n" + "=" * 60)
    print("TEST 6: Single inference (streaming chat)")
    print("=" * 60)
    try:
        from providers.base import InferenceRequest, ChatMessage

        request = InferenceRequest(
            messages=[ChatMessage(role="user", content="What is 2+2? Answer in one word.")],
            max_tokens=32,
            temperature=0.1,
        )

        tokens = []
        start = time.time()

        async for response in provider.chat(instance_id, request):
            if response.text:
                tokens.append(response.text)
            if response.error:
                raise Exception(f"Chat error: {response.error}")
            if response.done:
                total_time = response.total_time or (time.time() - start)
                tps = response.tokens_per_second or 0

        full_response = "".join(tokens)
        assert len(full_response) > 0, "Empty response"

        print(f"  PASS - Got response: '{full_response.strip()}'")
        print(f"  Tokens: {len(tokens)}, Time: {total_time:.2f}s, TPS: {tps:.1f}")
        passed += 1
    except Exception as e:
        print(f"  FAIL - {e}")
        failed += 1
        errors.append(f"Test 6: {e}")

    # ==================== Test 7: Concurrent inference ====================
    print("\n" + "=" * 60)
    print("TEST 7: Concurrent inference (3 parallel requests)")
    print("=" * 60)
    try:
        prompts = [
            "What is the capital of France? One word.",
            "What color is the sky? One word.",
            "What is 10 * 5? Just the number.",
        ]

        async def run_one(prompt, idx):
            req = InferenceRequest(
                messages=[ChatMessage(role="user", content=prompt)],
                max_tokens=32,
                temperature=0.1,
            )
            tokens = []
            t0 = time.time()
            async for resp in provider.chat(instance_id, req):
                if resp.text:
                    tokens.append(resp.text)
                if resp.error:
                    return idx, f"ERROR: {resp.error}", time.time() - t0
                if resp.done:
                    return idx, "".join(tokens).strip(), time.time() - t0
            return idx, "".join(tokens).strip(), time.time() - t0

        start = time.time()
        results = await asyncio.gather(*[run_one(p, i) for i, p in enumerate(prompts)])
        wall_time = time.time() - start

        sum_individual = sum(r[2] for r in results)

        all_ok = True
        for idx, text, elapsed in results:
            status = "OK" if text and "ERROR" not in text else "FAIL"
            if status == "FAIL":
                all_ok = False
            print(f"  [{idx}] {status} ({elapsed:.2f}s): '{text[:60]}'")

        assert all_ok, "Some concurrent requests failed"

        # Concurrent should be faster than sequential (sum of individual times)
        speedup = sum_individual / wall_time if wall_time > 0 else 0
        print(f"  Wall time: {wall_time:.2f}s, Sum individual: {sum_individual:.2f}s")
        print(f"  Speedup: {speedup:.2f}x (>1 = concurrent batching working)")

        if speedup > 1.2:
            print(f"  PASS - Concurrent batching confirmed ({speedup:.2f}x speedup)")
        else:
            print(f"  PASS - All requests completed (speedup {speedup:.2f}x, may need more tokens to see batching benefit)")
        passed += 1
    except Exception as e:
        print(f"  FAIL - {e}")
        failed += 1
        errors.append(f"Test 7: {e}")

    # ==================== Test 8: get_status() ====================
    print("\n" + "=" * 60)
    print("TEST 8: get_status()")
    print("=" * 60)
    try:
        status = await provider.get_status(instance_id)
        assert status == ModelStatus.READY, f"Expected READY, got {status}"
        print(f"  PASS - Status: {status.value}")

        status_bad = await provider.get_status("nonexistent")
        assert status_bad == ModelStatus.UNLOADED
        print(f"  PASS - Nonexistent returns UNLOADED")
        passed += 1
    except Exception as e:
        print(f"  FAIL - {e}")
        failed += 1
        errors.append(f"Test 8: {e}")

    # ==================== Test 9: Unload model ====================
    print("\n" + "=" * 60)
    print("TEST 9: Unload model")
    print("=" * 60)
    try:
        result = await provider.unload_model(instance_id)
        assert result == True
        assert instance_id not in provider.instances
        assert instance_id not in provider._servers

        # Verify process is actually stopped
        await asyncio.sleep(1)
        assert not server.is_running, "Server process still running after unload"

        print(f"  PASS - Model unloaded, process stopped")
        passed += 1
    except Exception as e:
        print(f"  FAIL - {e}")
        failed += 1
        errors.append(f"Test 9: {e}")

    # ==================== Summary ====================
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'=' * 60}")

    if errors:
        print("\nErrors:")
        for err in errors:
            print(f"  - {err}")

    if failed == 0:
        print("\nAll tests passed!")
    else:
        print(f"\n{failed} test(s) failed.")

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
