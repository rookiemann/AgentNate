"""Test loading a specific model in LM Studio and chatting."""
import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import asyncio
from providers.lm_studio_provider import LMStudioProvider
from providers.base import InferenceRequest, ChatMessage

async def test_load_and_chat():
    print("=== Testing LM Studio Model Loading ===")

    lm = LMStudioProvider()

    # Check health first
    health = await lm.health_check()
    print(f"Status: {health.get('status')}")
    print(f"Models in library: {health.get('loaded_count', 0)}")

    if health.get("status") != "healthy":
        print("LM Studio not running!")
        await lm.close()
        return

    # Test loading a specific model via API
    # Using granite-3.3-8b-instruct which should be a good chat model
    model_to_load = "granite-3.3-8b-instruct"
    print(f"\n--- Loading model: {model_to_load} ---")

    try:
        instance = await lm.load_model(model_to_load)
        print(f"Instance created: {instance.id}")
        print(f"Display name: {instance.display_name}")
        print(f"API loaded: {instance.metadata.get('api_loaded')}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("\nTrying with current model instead...")
        try:
            instance = await lm.load_model("current")
            print(f"Using current model: {instance.display_name}")
        except Exception as e2:
            print(f"Could not get current model: {e2}")
            await lm.close()
            return

    # Check if it's an embedding model
    if "embed" in instance.model_identifier.lower():
        print("\n⚠️  Warning: Current model is an embedding model, not a chat model.")
        print("Please load a chat model in LM Studio (phi-4, granite-3.3-8b-instruct, etc.)")
        await lm.close()
        return

    # Test chat
    print("\n--- Testing Chat ---")
    request = InferenceRequest(
        messages=[
            ChatMessage(role="user", content="Say hello in exactly 5 words.")
        ],
        max_tokens=50,
        temperature=0.7,
    )

    print("Generating response...")
    full_response = ""

    async for response in lm.chat(instance.id, request):
        if response.text:
            print(response.text, end="", flush=True)
            full_response += response.text
        if response.done:
            print()
            print(f"\n[Done - Total time: {response.total_time:.2f}s]")
        if response.error:
            print(f"\n[Error: {response.error}]")

    await lm.close()
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    asyncio.run(test_load_and_chat())

