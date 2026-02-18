"""Test OpenRouter with FREE models only - no credit usage."""
import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import asyncio
from providers.openrouter_provider import OpenRouterProvider
from providers.base import InferenceRequest, ChatMessage

API_KEY = "your-api-key-here"

async def test_openrouter_safe():
    print("=== OpenRouter Safe Test (Free Models Only) ===")

    orr = OpenRouterProvider(api_key=API_KEY)

    # Health check (free - just API connectivity)
    print("\n--- Health Check ---")
    health = await orr.health_check()
    print(f"Status: {health.get('status')}")
    print(f"Connected: {health.get('connected')}")

    if not health.get("connected"):
        print("Failed to connect to OpenRouter!")
        await orr.close()
        return

    # List models and find free ones
    print("\n--- Finding Free Models ---")
    models = await orr.list_models()
    print(f"Total models available: {len(models)}")

    # Find free models (pricing = $0 or :free suffix)
    free_models = []
    for m in models:
        model_id = m.get("id", "")
        pricing = m.get("pricing", {})
        prompt_price = float(pricing.get("prompt", "1") or "1")
        completion_price = float(pricing.get("completion", "1") or "1")

        if ":free" in model_id or (prompt_price == 0 and completion_price == 0):
            free_models.append(m)

    print(f"Free models found: {len(free_models)}")
    print("\nFree models:")
    for m in free_models[:10]:
        print(f"  - {m['id']}")
    if len(free_models) > 10:
        print(f"  ... and {len(free_models) - 10} more")

    if not free_models:
        print("No free models available! Skipping chat test.")
        await orr.close()
        return

    # Test with first free model
    test_model = free_models[0]["id"]
    print(f"\n--- Testing Chat with: {test_model} ---")

    instance = await orr.load_model(test_model)
    print(f"Instance ID: {instance.id}")

    # Minimal request - very few tokens
    request = InferenceRequest(
        messages=[
            ChatMessage(role="user", content="Say hi in 3 words.")
        ],
        max_tokens=20,  # Very limited
        temperature=0.5,
    )

    print("Response: ", end="", flush=True)
    async for response in orr.chat(instance.id, request):
        if response.text:
            print(response.text, end="", flush=True)
        if response.done:
            print(f"\n[Done - {response.total_time:.2f}s]")
            if response.usage:
                print(f"Usage: {response.usage}")
        if response.error:
            print(f"\n[Error: {response.error}]")

    await orr.close()
    print("\n=== Test Complete (No credits used!) ===")

if __name__ == "__main__":
    asyncio.run(test_openrouter_safe())

