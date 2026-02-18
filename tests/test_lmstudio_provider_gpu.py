"""Test updated LM Studio provider with GPU isolation."""
import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import asyncio
from providers.lm_studio_provider import LMStudioProvider, HAS_SDK
from providers.base import InferenceRequest, ChatMessage

MODEL_PATH = "lmstudio-community/Phi-4-mini-instruct-GGUF/Phi-4-mini-instruct-Q8_0.gguf"

async def test_provider_gpu():
    print("=" * 60)
    print("LM STUDIO PROVIDER - GPU ISOLATION TEST")
    print("=" * 60)
    print(f"SDK Available: {HAS_SDK}")

    provider = LMStudioProvider()

    # Health check
    health = await provider.health_check()
    print(f"\nHealth: {health.get('status')}")
    print(f"SDK Connected: {health.get('sdk_connected')}")
    print(f"SDK Host: {health.get('sdk_api_host')}")

    if health.get("status") != "healthy":
        print("LM Studio not running!")
        await provider.close()
        return

    # Load on GPU 0
    print("\n" + "-" * 60)
    print("LOAD MODEL ON GPU 0")
    print("-" * 60)

    try:
        instance0 = await provider.load_model(
            MODEL_PATH,
            gpu_index=0,
            disabled_gpus=[1],
            context_length=2048,
            instance_id="provider-gpu0",
        )
        print(f"[OK] Loaded: {instance0.display_name}")
        print(f"     Instance ID: {instance0.id[:12]}...")
        print(f"     GPU: {instance0.metadata.get('gpu_index')}")
    except Exception as e:
        print(f"[FAIL] {e}")
        instance0 = None

    # Load on GPU 1
    print("\n" + "-" * 60)
    print("LOAD MODEL ON GPU 1")
    print("-" * 60)

    try:
        instance1 = await provider.load_model(
            MODEL_PATH,
            gpu_index=1,
            disabled_gpus=[0],
            context_length=2048,
            instance_id="provider-gpu1",
        )
        print(f"[OK] Loaded: {instance1.display_name}")
        print(f"     Instance ID: {instance1.id[:12]}...")
        print(f"     GPU: {instance1.metadata.get('gpu_index')}")
    except Exception as e:
        print(f"[FAIL] {e}")
        instance1 = None

    # Check loaded instances
    print("\n" + "-" * 60)
    print("LOADED INSTANCES (SDK)")
    print("-" * 60)
    sdk_loaded = await provider.list_loaded_instances()
    for m in sdk_loaded:
        print(f"  - {m['identifier']}")

    # Test chat on both
    if instance0:
        print("\n" + "-" * 60)
        print("CHAT TEST: GPU 0")
        print("-" * 60)

        request = InferenceRequest(
            messages=[ChatMessage(role="user", content="Say 'GPU zero works'")],
            max_tokens=20,
            temperature=0.3,
        )

        print("Response: ", end="", flush=True)
        async for resp in provider.chat(instance0.id, request):
            if resp.text:
                print(resp.text, end="", flush=True)
            if resp.done:
                print(f" [{resp.total_time:.2f}s]")
            if resp.error:
                print(f" [Error: {resp.error}]")

    if instance1:
        print("\n" + "-" * 60)
        print("CHAT TEST: GPU 1")
        print("-" * 60)

        request = InferenceRequest(
            messages=[ChatMessage(role="user", content="Say 'GPU one works'")],
            max_tokens=20,
            temperature=0.3,
        )

        print("Response: ", end="", flush=True)
        async for resp in provider.chat(instance1.id, request):
            if resp.text:
                print(resp.text, end="", flush=True)
            if resp.done:
                print(f" [{resp.total_time:.2f}s]")
            if resp.error:
                print(f" [Error: {resp.error}]")

    # Cleanup
    print("\n" + "-" * 60)
    print("CLEANUP")
    print("-" * 60)
    await provider.close()
    print("[OK] Provider closed")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_provider_gpu())

