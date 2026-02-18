#!/usr/bin/env python3
"""Test which provider is slow."""
import sys
import os
import asyncio
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_providers():
    from backend.server import app

    # Get orchestrator from app state (need to trigger startup)
    from orchestrator.orchestrator import Orchestrator
    from providers.base import ProviderType

    # Load settings
    settings_path = os.path.join(os.path.dirname(__file__), "settings.json")
    settings = {}
    if os.path.exists(settings_path):
        import json
        with open(settings_path) as f:
            settings = json.load(f)

    # Create orchestrator
    orch = Orchestrator(settings)
    await orch.start()

    print("\n=== Testing each provider's list_models() ===\n")

    for provider_type, provider in orch.providers.items():
        print(f"Testing {provider_type.value}...", end=" ", flush=True)
        start = time.time()
        try:
            models = await asyncio.wait_for(provider.list_models(), timeout=60)
            elapsed = time.time() - start
            print(f"{elapsed:.2f}s - found {len(models)} models")
        except asyncio.TimeoutError:
            print(f"TIMEOUT after 60s")
        except Exception as e:
            elapsed = time.time() - start
            print(f"ERROR after {elapsed:.2f}s: {e}")

    print("\n=== Now testing all in parallel ===\n")
    start = time.time()
    results = await orch.list_all_models()
    elapsed = time.time() - start
    print(f"Total parallel time: {elapsed:.2f}s")
    for name, models in results.items():
        print(f"  {name}: {len(models)} models")

    await orch.stop()

if __name__ == "__main__":
    asyncio.run(test_providers())
