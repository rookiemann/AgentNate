"""Test LM Studio SDK integration."""
import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import asyncio
from providers.lm_studio_provider import LMStudioProvider, HAS_SDK

async def test_lmstudio_sdk():
    print("=== Testing LM Studio SDK Integration ===")
    print(f"SDK Available: {HAS_SDK}")

    lm = LMStudioProvider()

    # Check health (includes SDK status)
    print("\n--- Health Check ---")
    health = await lm.health_check()
    for key, value in health.items():
        print(f"  {key}: {value}")

    if health.get("status") != "healthy":
        print("\nLM Studio not reachable. Make sure it's running.")
        await lm.close()
        return

    # List available models in library (via SDK)
    if HAS_SDK:
        print("\n--- Available Models (SDK Library) ---")
        available = await lm.list_available_models()
        if available:
            print(f"Found {len(available)} models in library:")
            for m in available[:10]:  # Show first 10
                print(f"  - {m['name']} ({m['id']})")
            if len(available) > 10:
                print(f"  ... and {len(available) - 10} more")
        else:
            print("No models found or SDK error")

        # List loaded models
        print("\n--- Currently Loaded Models ---")
        loaded = await lm.list_loaded_models()
        if loaded:
            for m in loaded:
                print(f"  - {m['name']} ({m['id']})")
        else:
            print("No models currently loaded")

    # Test with current model (whatever is loaded in GUI)
    print("\n--- Testing Current Model ---")
    try:
        instance = await lm.load_model("current")
        print(f"Instance created: {instance.id}")
        print(f"Model: {instance.display_name}")
        print(f"SDK loaded: {instance.metadata.get('sdk_loaded')}")
    except Exception as e:
        print(f"Could not connect to current model: {e}")
        print("\nTo test, load a chat model in LM Studio GUI first.")

    await lm.close()
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    asyncio.run(test_lmstudio_sdk())

