"""Quick test of LM Studio and OpenRouter providers."""
import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import asyncio

async def test_providers():
    # Test LM Studio
    print("=== Testing LM Studio ===")
    from providers.lm_studio_provider import LMStudioProvider
    lm = LMStudioProvider()
    health = await lm.health_check()
    print(f"Status: {health}")

    if health.get("loaded"):
        print("LM Studio has a model loaded!")
        models = await lm.list_models()
        print(f"Models: {models}")
    else:
        print("No model loaded or LM Studio not running")

    await lm.close()

    # Test OpenRouter
    print()
    print("=== Testing OpenRouter ===")
    from providers.openrouter_provider import OpenRouterProvider
    api_key = "your-api-key-here"
    orr = OpenRouterProvider(api_key=api_key)
    health = await orr.health_check()
    print(f"Status: {health}")

    if health.get("connected"):
        print("OpenRouter connected!")
        models = await orr.list_models()
        print(f"Found {len(models)} models available")
        print("First 5 models:")
        for m in models[:5]:
            print(f"  - {m['id']}")
    else:
        print("OpenRouter not connected")

    await orr.close()

    print()
    print("=== Tests Complete ===")

if __name__ == "__main__":
    asyncio.run(test_providers())

