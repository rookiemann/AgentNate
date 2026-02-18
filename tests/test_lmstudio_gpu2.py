"""Test LM Studio v4 GPU isolation via SDK - using small models."""
import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import time
import lmstudio
from lmstudio import LlmLoadModelConfig
from lmstudio._sdk_models import GpuSetting

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

    # Connect
    print("\n--- Connecting to LM Studio ---")
    api_host = lmstudio.Client.find_default_local_api_host()
    print(f"SDK API host: {api_host}")

    if not api_host:
        print("[FAIL] LM Studio SDK API not found")
        return

    client = lmstudio.Client(api_host=api_host)

    if not wait_for_connection(client, timeout=15):
        print("[FAIL] Could not connect")
        return

    print("[OK] Connected")

    # Find small models (under 5GB for safety on 12GB GPU)
    print("\n--- Finding Small Models ---")
    downloaded = list(client.llm.list_downloaded())

    # Look for tinyllama, phi-3-mini, qwen-1.5b, gemma-2b, etc.
    small_keywords = ['tiny', 'mini', '1b', '2b', '3b', '1.5b', 'small']

    small_models = []
    for m in downloaded:
        path_lower = m.path.lower()
        for kw in small_keywords:
            if kw in path_lower and 'embed' not in path_lower:
                small_models.append(m)
                break

    if not small_models:
        print("No small models found. Available models:")
        for m in downloaded[:10]:
            print(f"  - {m.path}")
        print("\nUsing first available non-embedding model...")
        small_models = [m for m in downloaded if 'embed' not in m.path.lower()][:2]

    print(f"Found {len(small_models)} small models:")
    for m in small_models[:5]:
        print(f"  - {m.path}")

    if not small_models:
        print("No suitable models found!")
        return

    test_model = small_models[0].path
    print(f"\nUsing: {test_model}")

    # Unload existing
    print("\n--- Unloading existing models ---")
    loaded = list(client.llm.list_loaded())
    for m in loaded:
        try:
            print(f"  Unloading: {m.identifier}")
            client.llm.unload(m.identifier)
        except:
            pass

    # TEST 1: Load on GPU 0 only (RTX 3060 - 12GB)
    print("\n" + "=" * 60)
    print("TEST 1: Load on GPU 0 only (RTX 3060 - 12GB)")
    print("=" * 60)

    model1 = None
    try:
        gpu_config = GpuSetting(
            main_gpu=0,
            disabled_gpus=[1],
        )
        load_config = LlmLoadModelConfig(
            gpu=gpu_config,
            context_length=2048,
        )

        print(f"Config: main_gpu=0, disabled_gpus=[1]")
        model1 = client.llm.load_new_instance(
            test_model,
            instance_identifier="test-gpu0",
            config=load_config,
            ttl=300,
        )
        print(f"[OK] Loaded: {model1.identifier}")

        # Test inference
        print("Testing...")
        response_gen = model1.respond("Say 'Hello from GPU zero'", config={"max_tokens": 20})
        response_text = ""
        for chunk in response_gen:
            response_text += chunk
        print(f"Response: {response_text[:80]}")

    except Exception as e:
        print(f"[FAIL] {e}")

    # TEST 2: Load on GPU 1 only (RTX 3090 - 24GB)
    print("\n" + "=" * 60)
    print("TEST 2: Load on GPU 1 only (RTX 3090 - 24GB)")
    print("=" * 60)

    model2 = None
    try:
        gpu_config = GpuSetting(
            main_gpu=1,
            disabled_gpus=[0],
        )
        load_config = LlmLoadModelConfig(
            gpu=gpu_config,
            context_length=2048,
        )

        print(f"Config: main_gpu=1, disabled_gpus=[0]")
        model2 = client.llm.load_new_instance(
            test_model,
            instance_identifier="test-gpu1",
            config=load_config,
            ttl=300,
        )
        print(f"[OK] Loaded: {model2.identifier}")

        # Test inference
        print("Testing...")
        response_gen = model2.respond("Say 'Hello from GPU one'", config={"max_tokens": 20})
        response_text = ""
        for chunk in response_gen:
            response_text += chunk
        print(f"Response: {response_text[:80]}")

    except Exception as e:
        print(f"[FAIL] {e}")

    # Check what's loaded
    print("\n" + "=" * 60)
    print("LOADED INSTANCES")
    print("=" * 60)
    loaded = list(client.llm.list_loaded())
    print(f"Total loaded: {len(loaded)}")
    for m in loaded:
        print(f"  - {m.identifier}")

    # TEST 3: Both at once (if both loaded)
    if model1 and model2:
        print("\n" + "=" * 60)
        print("TEST 3: Both models loaded - parallel test")
        print("=" * 60)

        try:
            print("GPU 0 says: ", end="", flush=True)
            for chunk in model1.respond("Count 1 2 3", config={"max_tokens": 15}):
                print(chunk, end="", flush=True)
            print()

            print("GPU 1 says: ", end="", flush=True)
            for chunk in model2.respond("Count A B C", config={"max_tokens": 15}):
                print(chunk, end="", flush=True)
            print()

            print("\n[OK] Both GPUs responding independently!")

        except Exception as e:
            print(f"[FAIL] {e}")

    # Cleanup
    print("\n--- Cleanup ---")
    try:
        if model1:
            model1.unload()
            print("[OK] GPU 0 model unloaded")
    except:
        pass
    try:
        if model2:
            model2.unload()
            print("[OK] GPU 1 model unloaded")
    except:
        pass

    client.close()
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    test_gpu_isolation()

