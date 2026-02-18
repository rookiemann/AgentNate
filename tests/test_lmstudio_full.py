"""Full test of LM Studio v4 integration."""
import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import asyncio
from providers.lm_studio_provider import LMStudioProvider, HAS_SDK
from providers.base import InferenceRequest, ChatMessage

async def test_full_integration():
    print("=== LM Studio v4 Full Integration Test ===")
    print(f"SDK Available: {HAS_SDK}")

    lm = LMStudioProvider()

    # Health check
    health = await lm.health_check()
    print(f"\nStatus: {health.get('status')}")

    if health.get("status") != "healthy":
        print("LM Studio not running!")
        await lm.close()
        return

    # List available models
    models = await lm.list_models()
    chat_models = [m for m in models if "embed" not in m["id"].lower()]
    print(f"Chat models available: {len(chat_models)}")

    # Test 1: Load specific model (phi-4)
    print("\n--- Test 1: Load phi-4 ---")
    try:
        instance1 = await lm.load_model("phi-4")
        print(f"Instance ID: {instance1.id}")
        print(f"Model: {instance1.model_identifier}")
        print(f"JIT Load: {instance1.metadata.get('jit_load')}")
    except Exception as e:
        print(f"Failed: {e}")
        await lm.close()
        return

    # Test 2: Chat with phi-4
    print("\n--- Test 2: Chat with phi-4 ---")
    request = InferenceRequest(
        messages=[
            ChatMessage(role="user", content="Say hello in exactly 5 words.")
        ],
        max_tokens=50,
        temperature=0.7,
    )

    print("Response: ", end="", flush=True)
    async for response in lm.chat(instance1.id, request):
        if response.text:
            print(response.text, end="", flush=True)
        if response.done:
            print(f"\n[Done - {response.total_time:.2f}s]")
        if response.error:
            print(f"\n[Error: {response.error}]")

    # Test 3: Load a second model (granite)
    print("\n--- Test 3: Load granite-3.3-8b-instruct ---")
    try:
        instance2 = await lm.load_model("granite-3.3-8b-instruct")
        print(f"Instance ID: {instance2.id}")
        print(f"Model: {instance2.model_identifier}")
    except Exception as e:
        print(f"Failed: {e}")
        await lm.close()
        return

    # Test 4: Chat with granite
    print("\n--- Test 4: Chat with granite ---")
    request2 = InferenceRequest(
        messages=[
            ChatMessage(role="user", content="What is 2+2? Answer briefly.")
        ],
        max_tokens=50,
        temperature=0.3,
    )

    print("Response: ", end="", flush=True)
    async for response in lm.chat(instance2.id, request2):
        if response.text:
            print(response.text, end="", flush=True)
        if response.done:
            print(f"\n[Done - {response.total_time:.2f}s]")
        if response.error:
            print(f"\n[Error: {response.error}]")

    # Test 5: Both instances are available
    print("\n--- Test 5: Multiple instances ---")
    print(f"Active instances: {len(lm.instances)}")
    for iid, inst in lm.instances.items():
        print(f"  - {inst.model_identifier} ({iid[:8]}...)")

    # Cleanup
    await lm.close()
    print("\n=== All Tests Complete ===")

if __name__ == "__main__":
    asyncio.run(test_full_integration())

