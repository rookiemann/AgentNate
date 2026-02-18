"""Test Ollama provider (requires Ollama to be running)."""
import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import asyncio
from providers.ollama_provider import OllamaProvider
from providers.base import InferenceRequest, ChatMessage

async def test_ollama():
    print("=== Ollama Provider Test ===")

    ollama = OllamaProvider()

    # Health check
    print("\n--- Health Check ---")
    health = await ollama.health_check()
    print(f"Status: {health.get('status')}")

    if health.get("status") == "offline":
        print("Ollama not running!")
        print("\nTo start Ollama:")
        print("  1. Install from https://ollama.ai")
        print("  2. Run: ollama serve")
        print("  3. Pull a model: ollama pull llama3.2:3b")
        await ollama.close()
        return

    print(f"Downloaded models: {health.get('model_count', 0)}")
    print(f"Running models: {health.get('running_count', 0)}")
    if health.get("running_models"):
        print(f"  Currently loaded: {', '.join(health['running_models'])}")

    # List models
    print("\n--- Downloaded Models ---")
    models = await ollama.list_models()
    if models:
        for m in models:
            size_gb = (m.get("size_bytes", 0) or 0) / (1024**3)
            print(f"  - {m['name']} ({size_gb:.1f} GB)")
    else:
        print("No models downloaded. Run: ollama pull llama3.2:3b")
        await ollama.close()
        return

    # Load first model
    model_name = models[0]["name"]
    print(f"\n--- Loading Model: {model_name} ---")
    instance = await ollama.load_model(model_name, pre_warm=True)
    print(f"Instance ID: {instance.id}")
    print(f"Status: {instance.status}")

    if instance.status.value == "error":
        print(f"Error: {instance.metadata.get('error')}")
        await ollama.close()
        return

    # Chat test
    print("\n--- Chat Test ---")
    request = InferenceRequest(
        messages=[
            ChatMessage(role="user", content="Say hello in exactly 5 words.")
        ],
        max_tokens=50,
        temperature=0.7,
    )

    print("Response: ", end="", flush=True)
    async for response in ollama.chat(instance.id, request):
        if response.text:
            print(response.text, end="", flush=True)
        if response.done:
            print(f"\n[Done - {response.total_time:.2f}s]")
            if response.usage:
                print(f"Tokens: {response.usage.get('total_tokens', 'N/A')}")
        if response.error:
            print(f"\n[Error: {response.error}]")

    # Show model info
    print(f"\n--- Model Info: {model_name} ---")
    info = await ollama.show_model(model_name)
    if "error" not in info:
        params = info.get("parameters", "")
        if params:
            # Show first few lines of params
            lines = params.split('\n')[:5]
            print("Parameters:")
            for line in lines:
                print(f"  {line}")

    # Cleanup
    await ollama.close()
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    asyncio.run(test_ollama())

