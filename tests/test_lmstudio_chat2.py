"""Test chat with LM Studio using a specific model ID."""
import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import asyncio
from providers.lm_studio_provider import LMStudioProvider
from providers.base import InferenceRequest, ChatMessage

async def test_chat_with_model():
    print("=== Testing LM Studio Chat ===")

    lm = LMStudioProvider()

    # Check health
    health = await lm.health_check()
    print(f"Status: {health.get('status')}")
    print(f"SDK available: {health.get('sdk_available')}")

    if health.get("status") != "healthy":
        print("LM Studio not running!")
        await lm.close()
        return

    # List available chat models (excluding embedding models)
    print("\n--- Available Chat Models ---")
    models = await lm.list_models()
    chat_models = [m for m in models if "embed" not in m["id"].lower()]

    for m in chat_models[:10]:
        print(f"  - {m['id']}")
    if len(chat_models) > 10:
        print(f"  ... and {len(chat_models) - 10} more")

    if not chat_models:
        print("No chat models found in LM Studio!")
        await lm.close()
        return

    # Create instance for current model
    instance = await lm.load_model("current")
    print(f"\nUsing model: {instance.model_identifier}")

    # Skip if it's an embedding model
    if "embed" in instance.model_identifier.lower():
        print("\nCurrent model is an embedding model.")
        print("LM Studio v4 lists all models but you need to load a chat model in the GUI.")
        print("\nSuggested chat models from your library:")
        for m in chat_models[:5]:
            print(f"  - {m['id']}")
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
    asyncio.run(test_chat_with_model())

