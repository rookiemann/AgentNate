"""Test the model orchestrator with LM Studio provider."""
import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import asyncio
from settings.settings_manager import SettingsManager
from orchestrator.orchestrator import ModelOrchestrator
from providers.base import ProviderType, InferenceRequest, ChatMessage

MODEL_PATH = "lmstudio-community/Phi-4-mini-instruct-GGUF/Phi-4-mini-instruct-Q8_0.gguf"

async def test_orchestrator():
    print("=" * 60)
    print("ORCHESTRATOR INTEGRATION TEST")
    print("=" * 60)

    # Initialize
    settings = SettingsManager(settings_dir=os.path.dirname(__file__))
    orchestrator = ModelOrchestrator(settings)

    # Start orchestrator
    print("\n[1] Starting orchestrator...")
    await orchestrator.start()
    print("    [OK] Started")

    # Check providers
    print("\n[2] Enabled providers:")
    for p in orchestrator.get_enabled_providers():
        print(f"    - {p.value}")

    # Health check
    print("\n[3] Health check...")
    health = await orchestrator.check_all_health()
    for provider, status in health.items():
        print(f"    {provider}: {status.get('status', 'unknown')}")

    # Load model via LM Studio with GPU 0
    print("\n[4] Loading model on GPU 0...")
    try:
        instance = await orchestrator.load_model(
            ProviderType.LM_STUDIO,
            MODEL_PATH,
            gpu_index=0,
            disabled_gpus=[1],
            context_length=2048,
            instance_id="test-gpu0",
        )
        print(f"    [OK] Loaded: {instance.display_name}")
        print(f"    Instance ID: {instance.id}")
    except Exception as e:
        print(f"    [FAIL] {e}")
        await orchestrator.stop()
        return

    # List loaded instances
    print("\n[5] Loaded instances:")
    for inst in orchestrator.get_loaded_instances():
        print(f"    - {inst.display_name} ({inst.status.value})")

    # Chat test
    print("\n[6] Chat test...")
    request = InferenceRequest(
        messages=[ChatMessage(role="user", content="Say 'Orchestrator works!' exactly.")],
        max_tokens=20,
        temperature=0.3,
    )

    print("    Response: ", end="", flush=True)
    async for resp in orchestrator.chat(instance.id, request):
        if resp.text:
            print(resp.text, end="", flush=True)
        if resp.done:
            print(f" [{resp.total_time:.2f}s]")
        if resp.error:
            print(f" [Error: {resp.error}]")

    # JIT loading test (should return same instance)
    print("\n[7] JIT loading test (same model)...")
    jit_id = await orchestrator.load_model_jit(
        ProviderType.LM_STUDIO,
        MODEL_PATH,
    )
    if jit_id == instance.id:
        print(f"    [OK] JIT returned existing instance")
    else:
        print(f"    [INFO] JIT created new instance: {jit_id}")

    # Cleanup
    print("\n[8] Stopping orchestrator...")
    await orchestrator.stop()
    print("    [OK] Stopped")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_orchestrator())

