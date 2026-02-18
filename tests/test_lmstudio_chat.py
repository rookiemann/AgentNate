"""Test chat generation with LM Studio."""
import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import asyncio
from providers.lm_studio_provider import LMStudioProvider
from providers.base import InferenceRequest, ChatMessage

async def test_lmstudio_chat():
    print("=== Testing LM Studio Chat ===")

    lm = LMStudioProvider()

    # Check health first
    health = await lm.health_check()
    print(f"Health: {health}")

    if not health.get("loaded"):
        print("No model loaded in LM Studio!")
        await lm.close()
        return

    model_name = health.get("model", "unknown")
    print(f"Model loaded: {model_name}")

    # Note: phi-4 is a chat model, so we don't need to check for embedding models

    # Create instance - use phi-4 instead of current (which might be embedding model)
    instance = await lm.load_model("phi-4")
    print(f"Instance created: {instance.id}")
    print(f"Model: {instance.model_identifier}")

    # Create a simple request
    request = InferenceRequest(
        messages=[
            ChatMessage(role="user", content="Say hello in exactly 5 words.")
        ],
        max_tokens=50,
        temperature=0.7,
    )

    print("Generating response...")

    async for response in lm.chat(instance.id, request):
        if response.text:
            print(response.text, end="", flush=True)
        if response.done:
            print()
            print(f"\n[Done - Total time: {response.total_time:.2f}s]")
        if response.error:
            print(f"\n[Error: {response.error}]")

    await lm.close()
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    asyncio.run(test_lmstudio_chat())

