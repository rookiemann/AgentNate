"""Test chat generation with OpenRouter (using free model)."""
import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import asyncio
from providers.openrouter_provider import OpenRouterProvider
from providers.base import InferenceRequest, ChatMessage

async def test_openrouter_chat():
    print("=== Testing OpenRouter Chat (Free Model) ===")

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY not set; skipping OpenRouter live chat test.")
        return
    orr = OpenRouterProvider(api_key=api_key)

    # Use a free model
    model_id = "arcee-ai/trinity-large-preview:free"
    print(f"Loading model: {model_id}")

    instance = await orr.load_model(model_id)
    print(f"Instance created: {instance.id}")

    # Create a simple request
    request = InferenceRequest(
        messages=[
            ChatMessage(role="user", content="Say hello in exactly 5 words.")
        ],
        max_tokens=50,
        temperature=0.7,
    )

    print("Generating response...")
    full_response = ""

    async for response in orr.chat(instance.id, request):
        if response.text:
            print(response.text, end="", flush=True)
            full_response += response.text
        if response.done:
            print()
            print(f"\n[Done - Total time: {response.total_time:.2f}s]")
        if response.error:
            print(f"\n[Error: {response.error}]")

    await orr.close()
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    asyncio.run(test_openrouter_chat())

