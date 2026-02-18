"""Test the Meta Agent and Tool System."""
import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import asyncio
import json

async def test_tools():
    """Test the tool system directly."""
    print("=" * 60)
    print("META AGENT TOOL SYSTEM TEST")
    print("=" * 60)

    from settings.settings_manager import SettingsManager
    from orchestrator.orchestrator import ModelOrchestrator
    from backend.n8n_manager import N8nManager
    from backend.tools import ToolRouter

    # Initialize
    print("\n--- Initializing ---")
    settings = SettingsManager(settings_dir=".")
    orchestrator = ModelOrchestrator(settings)
    await orchestrator.start()

    n8n_manager = N8nManager(
        base_port=5678,
        n8n_path=os.path.join(".", "node_modules", "n8n", "bin", "n8n")
    )

    tool_router = ToolRouter(orchestrator, n8n_manager, settings)
    print("[OK] Initialized")

    # Test 1: List available tools
    print("\n--- Test 1: Tool List ---")
    tools = tool_router.get_tool_list()
    print(f"Available tools: {len(tools)}")

    # Test 2: Get GPU status
    print("\n--- Test 2: GPU Status ---")
    result = await tool_router.execute("get_gpu_status", {})
    if result["success"]:
        for gpu in result.get("gpus", []):
            print(f"  GPU {gpu['index']}: {gpu['name']}")
            print(f"    Memory: {gpu['memory_used_mb']}/{gpu['memory_total_mb']} MB")
    else:
        print(f"  Error: {result.get('error')}")

    # Test 3: List available models
    print("\n--- Test 3: List Available Models ---")
    result = await tool_router.execute("list_available_models", {"provider": "llama_cpp"})
    if result["success"]:
        print(f"  Found {result['count']} models")
        for m in result["models"][:5]:
            print(f"    - {m['name']} ({m['size_gb']} GB)")
    else:
        print(f"  Error: {result.get('error')}")

    # Test 4: Load a model
    print("\n--- Test 4: Load Model ---")
    result = await tool_router.execute("load_model", {
        "model_name": "tinyllama",
        "gpu_index": 0,
        "provider": "llama_cpp"
    })
    if result["success"]:
        instance_id = result["instance_id"]
        print(f"  Loaded: {result['model']}")
        print(f"  Instance ID: {instance_id}")
    else:
        print(f"  Error: {result.get('error')}")
        instance_id = None

    # Test 5: List loaded models
    print("\n--- Test 5: List Loaded Models ---")
    result = await tool_router.execute("list_loaded_models", {})
    if result["success"]:
        print(f"  Loaded: {result['count']} models")
        for m in result["instances"]:
            print(f"    - {m['model']} on GPU {m['gpu']}")
    else:
        print(f"  Error: {result.get('error')}")

    # Test 6: Create quick workflow
    print("\n--- Test 6: Create Quick Workflow ---")
    result = await tool_router.execute("create_quick_workflow", {
        "template": "webhook_llm",
        "name": "Test Chat Workflow",
        "webhook_path": "test-chat"
    })
    if result["success"]:
        workflow = result["workflow"]
        print(f"  Created: {workflow['name']}")
        print(f"  Nodes: {len(workflow['nodes'])}")
    else:
        print(f"  Error: {result.get('error')}")

    # Test 7: Parse and execute from text
    print("\n--- Test 7: Parse Tool Call from Text ---")
    test_text = '''
I'll check the GPU status for you.

```json
{"tool": "get_gpu_status", "arguments": {}}
```
'''
    result = await tool_router.parse_and_execute(test_text)
    if result:
        print(f"  Parsed and executed: get_gpu_status")
        print(f"  Success: {result.get('success')}")
    else:
        print("  No tool call found in text")

    # Test 8: Unload model
    if instance_id:
        print("\n--- Test 8: Unload Model ---")
        result = await tool_router.execute("unload_model", {"instance_id": instance_id})
        print(f"  Result: {result.get('message') or result.get('error')}")

    # Cleanup
    print("\n--- Cleanup ---")
    await orchestrator.stop()
    print("[OK] Done")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_tools())

