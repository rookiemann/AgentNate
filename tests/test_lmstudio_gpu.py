"""Test LM Studio v4 GPU isolation via SDK."""
import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import lmstudio
from lmstudio import LlmLoadModelConfig
from lmstudio._sdk_models import GpuSetting

import time

def wait_for_connection(client, timeout=30):
    """Wait for SDK to connect to LM Studio."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            list(client.llm.list_loaded())
            return True
        except Exception as e:
            if "not yet resolved" in str(e):
                time.sleep(1)
                continue
            raise
    return False

def test_gpu_isolation():
    print("=" * 60)
    print("LM STUDIO v4 GPU ISOLATION TEST")
    print("=" * 60)

    print("\n--- Connecting to LM Studio ---")

    # Find SDK port
    api_host = lmstudio.Client.find_default_local_api_host()
    print(f"SDK API host: {api_host}")

    if not api_host:
        print("[FAIL] LM Studio SDK API not found. Make sure LM Studio is running.")
        return

    client = lmstudio.Client(api_host=api_host)

    if not wait_for_connection(client, timeout=15):
        print("[FAIL] Could not connect to LM Studio SDK")
        return

    print("[OK] Connected")

    # List available models
    print("\n--- Available Models ---")
    downloaded = client.llm.list_downloaded()
    print(f"Found {len(list(downloaded))} models")

    # Find a small model to test with
    downloaded = list(client.llm.list_downloaded())
    small_models = [m for m in downloaded if 'tiny' in m.path.lower()
                    or 'small' in m.path.lower()
                    or '1b' in m.path.lower()
                    or '2b' in m.path.lower()
                    or '3b' in m.path.lower()]

    if not small_models:
        # Just use any model
        small_models = downloaded[:2]

    print(f"Using models for test:")
    for i, m in enumerate(small_models[:2]):
        print(f"  [{i}] {m.path}")

    if len(small_models) < 1:
        print("No models available!")
        return

    test_model = small_models[0].path
    print(f"\nTest model: {test_model}")

    # First, unload any existing models
    print("\n--- Unloading existing models ---")
    loaded = list(client.llm.list_loaded())
    for m in loaded:
        print(f"  Unloading: {m.identifier}")
        try:
            client.llm.unload(m.identifier)
        except Exception as e:
            print(f"    Error: {e}")

    # TEST 1: Load on GPU 0 only
    print("\n" + "=" * 60)
    print("TEST 1: Load model on GPU 0 only (RTX 3060)")
    print("=" * 60)

    try:
        gpu_config = GpuSetting(
            main_gpu=0,
            disabled_gpus=[1],  # Disable GPU 1
            split_strategy='favorMainGpu'
        )

        load_config = LlmLoadModelConfig(
            gpu=gpu_config,
            context_length=2048,
            flash_attention=True,
        )

        print(f"Loading with config: main_gpu=0, disabled_gpus=[1]")

        model1 = client.llm.load_new_instance(
            test_model,
            instance_identifier="gpu0-test",
            config=load_config,
            ttl=300,  # 5 min timeout
        )

        print(f"[OK] Loaded: {model1.identifier}")

        # Quick test
        print("Testing inference...")
        response = model1.respond("Say 'GPU zero' in 2 words.", config={"max_tokens": 20})
        result = "".join(response)
        print(f"Response: {result[:100]}")

        # Unload
        print("Unloading...")
        model1.unload()
        print("[OK] Unloaded")

    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()

    # TEST 2: Load on GPU 1 only
    print("\n" + "=" * 60)
    print("TEST 2: Load model on GPU 1 only (RTX 3090)")
    print("=" * 60)

    try:
        gpu_config = GpuSetting(
            main_gpu=1,
            disabled_gpus=[0],  # Disable GPU 0
            split_strategy='favorMainGpu'
        )

        load_config = LlmLoadModelConfig(
            gpu=gpu_config,
            context_length=2048,
            flash_attention=True,
        )

        print(f"Loading with config: main_gpu=1, disabled_gpus=[0]")

        model2 = client.llm.load_new_instance(
            test_model,
            instance_identifier="gpu1-test",
            config=load_config,
            ttl=300,
        )

        print(f"[OK] Loaded: {model2.identifier}")

        # Quick test
        print("Testing inference...")
        response = model2.respond("Say 'GPU one' in 2 words.", config={"max_tokens": 20})
        result = "".join(response)
        print(f"Response: {result[:100]}")

        # Unload
        print("Unloading...")
        model2.unload()
        print("[OK] Unloaded")

    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()

    # TEST 3: Try to load BOTH simultaneously on different GPUs
    print("\n" + "=" * 60)
    print("TEST 3: Load TWO instances on DIFFERENT GPUs simultaneously")
    print("=" * 60)

    try:
        # Load on GPU 0
        gpu_config_0 = GpuSetting(main_gpu=0, disabled_gpus=[1])
        config_0 = LlmLoadModelConfig(gpu=gpu_config_0, context_length=2048)

        print("Loading instance A on GPU 0...")
        model_a = client.llm.load_new_instance(
            test_model,
            instance_identifier="parallel-gpu0",
            config=config_0,
            ttl=300,
        )
        print(f"[OK] Instance A: {model_a.identifier}")

        # Load on GPU 1
        gpu_config_1 = GpuSetting(main_gpu=1, disabled_gpus=[0])
        config_1 = LlmLoadModelConfig(gpu=gpu_config_1, context_length=2048)

        print("Loading instance B on GPU 1...")
        model_b = client.llm.load_new_instance(
            test_model,
            instance_identifier="parallel-gpu1",
            config=config_1,
            ttl=300,
        )
        print(f"[OK] Instance B: {model_b.identifier}")

        # List loaded models
        print("\n--- Currently Loaded ---")
        loaded = list(client.llm.list_loaded())
        for m in loaded:
            print(f"  - {m.identifier}")

        # Test both
        print("\nTesting parallel responses...")
        resp_a = "".join(model_a.respond("Say 'Alpha'", config={"max_tokens": 10}))
        resp_b = "".join(model_b.respond("Say 'Beta'", config={"max_tokens": 10}))

        print(f"GPU 0 response: {resp_a[:50]}")
        print(f"GPU 1 response: {resp_b[:50]}")

        # Cleanup
        print("\nCleaning up...")
        model_a.unload()
        model_b.unload()
        print("[OK] Both unloaded")

    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()

    client.close()
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    test_gpu_isolation()

