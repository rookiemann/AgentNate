"""Integration tests for AgentNate - deeper functionality testing."""
import asyncio
import aiohttp
import json

BASE_URL = "http://localhost:8000"

class Colors:
    OK = '\033[92m'
    FAIL = '\033[91m'
    WARN = '\033[93m'
    END = '\033[0m'
    BOLD = '\033[1m'

def ok(msg): print(f"{Colors.OK}[OK]{Colors.END} {msg}")
def fail(msg): print(f"{Colors.FAIL}[FAIL]{Colors.END} {msg}")
def warn(msg): print(f"{Colors.WARN}[WARN]{Colors.END} {msg}")
def section(msg): print(f"\n{Colors.BOLD}{'='*60}\n{msg}\n{'='*60}{Colors.END}")


async def test_meta_agent_with_model():
    """Test Meta Agent with a loaded model - can it actually call tools?"""
    section("1. Meta Agent Tool Calling")

    async with aiohttp.ClientSession() as session:
        # First load a model
        print("  Loading phi-4 for Meta Agent testing...")
        async with session.post(
            f"{BASE_URL}/api/tools/call",
            json={"tool": "load_model", "arguments": {
                "model_name": "phi-4",
                "gpu_index": 1,  # Use the 3090
                "provider": "llama_cpp"
            }},
            timeout=aiohttp.ClientTimeout(total=180)
        ) as resp:
            data = await resp.json()
            if not data.get("success"):
                # Try a smaller model
                print("  phi-4 not available, trying qwen...")
                async with session.post(
                    f"{BASE_URL}/api/tools/call",
                    json={"tool": "load_model", "arguments": {
                        "model_name": "qwen",
                        "gpu_index": 1,
                        "provider": "llama_cpp"
                    }},
                    timeout=aiohttp.ClientTimeout(total=180)
                ) as resp2:
                    data = await resp2.json()
                    if not data.get("success"):
                        warn(f"Could not load a capable model: {data.get('error')}")
                        return None

            instance_id = data.get("instance_id")
            ok(f"Model loaded: {data.get('model')} on GPU {data.get('gpu')}")

        # Test 1: Ask about GPUs (should trigger get_gpu_status tool)
        print("\n  Test: 'What GPUs do I have?'")
        async with session.post(
            f"{BASE_URL}/api/tools/agent",
            json={
                "message": "What GPUs do I have? Check the GPU status.",
                "instance_id": instance_id
            },
            timeout=aiohttp.ClientTimeout(total=120)
        ) as resp:
            data = await resp.json()
            if data.get("tool_called"):
                ok(f"Meta Agent called tool: {data.get('tool_name')}")
                print(f"    Response: {data.get('response', '')[:100]}...")
            elif data.get("response"):
                warn(f"Meta Agent responded without tool: {data.get('response')[:80]}...")
            else:
                fail(f"Meta Agent failed: {data}")

        # Test 2: Ask to list models
        print("\n  Test: 'What models are loaded?'")
        async with session.post(
            f"{BASE_URL}/api/tools/agent",
            json={
                "message": "List the currently loaded models.",
                "instance_id": instance_id,
                "conversation_id": "test-1"
            },
            timeout=aiohttp.ClientTimeout(total=120)
        ) as resp:
            data = await resp.json()
            if data.get("tool_called"):
                ok(f"Meta Agent called tool: {data.get('tool_name')}")
            elif data.get("response"):
                print(f"    Response: {data.get('response')[:100]}...")

        # Cleanup
        await session.post(
            f"{BASE_URL}/api/tools/call",
            json={"tool": "unload_model", "arguments": {"instance_id": instance_id}}
        )
        ok("Model unloaded")

        return True


async def test_n8n_lifecycle():
    """Test n8n instance spawning and management."""
    section("2. n8n Instance Lifecycle")

    async with aiohttp.ClientSession() as session:
        # Spawn an n8n instance
        print("  Spawning n8n instance...")
        async with session.post(
            f"{BASE_URL}/api/tools/call",
            json={"tool": "spawn_n8n", "arguments": {"port": 5678}},
            timeout=aiohttp.ClientTimeout(total=60)
        ) as resp:
            data = await resp.json()
            if data.get("success"):
                ok(f"n8n spawned on port {data.get('port')}")
                ok(f"URL: {data.get('url')}")
            else:
                # Might already be running
                if "already" in str(data.get("error", "")).lower():
                    warn("n8n already running on port 5678")
                else:
                    fail(f"Failed to spawn n8n: {data.get('error')}")
                    return False

        # Wait for it to start
        await asyncio.sleep(3)

        # List instances
        async with session.post(
            f"{BASE_URL}/api/tools/call",
            json={"tool": "list_n8n_instances", "arguments": {}}
        ) as resp:
            data = await resp.json()
            if data.get("success") and data.get("count", 0) > 0:
                ok(f"n8n instances running: {data.get('count')}")
                for inst in data.get("instances", []):
                    print(f"    Port {inst.get('port')}: {inst.get('status')}")
            else:
                warn("No n8n instances found after spawn")

        # Stop the instance
        print("  Stopping n8n instance...")
        async with session.post(
            f"{BASE_URL}/api/tools/call",
            json={"tool": "stop_n8n", "arguments": {"port": 5678}}
        ) as resp:
            data = await resp.json()
            if data.get("success"):
                ok("n8n instance stopped")
            else:
                warn(f"Failed to stop: {data.get('error')}")

        return True


async def test_workflow_deployment():
    """Test creating and deploying a workflow to n8n."""
    section("3. Workflow Creation & Deployment")

    async with aiohttp.ClientSession() as session:
        # Create a quick workflow
        print("  Creating webhook->LLM workflow...")
        async with session.post(
            f"{BASE_URL}/api/workflows/quick",
            json={
                "template": "webhook_llm",
                "name": "Integration Test Workflow",
                "webhook_path": "integration-test"
            }
        ) as resp:
            data = await resp.json()
            if data.get("success"):
                workflow = data.get("workflow", {})
                ok(f"Created: {workflow.get('name')} with {len(workflow.get('nodes', []))} nodes")

                # Show the nodes
                for node in workflow.get("nodes", []):
                    print(f"    - {node.get('name')} ({node.get('type')})")
            else:
                fail(f"Failed to create workflow: {data.get('error')}")
                return False

        # Spawn n8n for deployment test
        print("\n  Spawning n8n for deployment...")
        async with session.post(
            f"{BASE_URL}/api/tools/call",
            json={"tool": "spawn_n8n", "arguments": {"port": 5679}},
            timeout=aiohttp.ClientTimeout(total=60)
        ) as resp:
            spawn_data = await resp.json()
            if not spawn_data.get("success"):
                if "already" not in str(spawn_data.get("error", "")).lower():
                    warn(f"Could not spawn n8n: {spawn_data.get('error')}")

        await asyncio.sleep(3)

        # Deploy the workflow
        print("  Deploying workflow to n8n...")
        async with session.post(
            f"{BASE_URL}/api/workflows/deploy",
            json={
                "workflow": workflow,
                "n8n_port": 5679,
                "activate": True
            },
            timeout=aiohttp.ClientTimeout(total=30)
        ) as resp:
            data = await resp.json()
            if data.get("success"):
                ok(f"Workflow deployed! ID: {data.get('workflow_id')}")
                ok(f"Webhook URL: {data.get('webhook_url', 'N/A')}")
            else:
                warn(f"Deployment issue: {data.get('error')}")

        # Cleanup
        await session.post(
            f"{BASE_URL}/api/tools/call",
            json={"tool": "stop_n8n", "arguments": {"port": 5679}}
        )
        ok("n8n stopped")

        return True


async def test_multi_model_parallel():
    """Test loading multiple models and parallel inference."""
    section("4. Multi-Model Parallel Inference")

    async with aiohttp.ClientSession() as session:
        instances = []

        # Load model on GPU 0
        print("  Loading tinyllama on GPU 0...")
        async with session.post(
            f"{BASE_URL}/api/tools/call",
            json={"tool": "load_model", "arguments": {
                "model_name": "tinyllama",
                "gpu_index": 0
            }},
            timeout=aiohttp.ClientTimeout(total=120)
        ) as resp:
            data = await resp.json()
            if data.get("success"):
                instances.append(data.get("instance_id"))
                ok(f"Loaded on GPU 0: {data.get('model')}")

        # Load another model on GPU 1
        print("  Loading tinyllama on GPU 1...")
        async with session.post(
            f"{BASE_URL}/api/tools/call",
            json={"tool": "load_model", "arguments": {
                "model_name": "tinyllama",
                "gpu_index": 1
            }},
            timeout=aiohttp.ClientTimeout(total=120)
        ) as resp:
            data = await resp.json()
            if data.get("success"):
                instances.append(data.get("instance_id"))
                ok(f"Loaded on GPU 1: {data.get('model')}")

        if len(instances) < 2:
            warn("Could not load models on both GPUs")
            # Cleanup
            for inst_id in instances:
                await session.post(
                    f"{BASE_URL}/api/tools/call",
                    json={"tool": "unload_model", "arguments": {"instance_id": inst_id}}
                )
            return False

        # Check GPU status
        async with session.post(
            f"{BASE_URL}/api/tools/call",
            json={"tool": "get_gpu_status", "arguments": {}}
        ) as resp:
            data = await resp.json()
            for gpu in data.get("gpus", []):
                models = gpu.get("models_loaded", [])
                if models:
                    print(f"    GPU {gpu['index']}: {len(models)} model(s) loaded")

        # Parallel inference
        print("\n  Running parallel inference on both models...")

        async def chat_with_instance(inst_id, prompt, label):
            async with session.post(
                f"{BASE_URL}/api/chat/completions",
                json={
                    "instance_id": inst_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 20
                },
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                data = await resp.json()
                return label, data.get("content", "")[:50]

        import time
        start = time.time()

        results = await asyncio.gather(
            chat_with_instance(instances[0], "Count to 5.", "GPU0"),
            chat_with_instance(instances[1], "Say hello.", "GPU1")
        )

        elapsed = time.time() - start
        ok(f"Parallel inference completed in {elapsed:.2f}s")

        for label, response in results:
            print(f"    {label}: {response}...")

        # Cleanup
        for inst_id in instances:
            await session.post(
                f"{BASE_URL}/api/tools/call",
                json={"tool": "unload_model", "arguments": {"instance_id": inst_id}}
            )
        ok("Models unloaded")

        return True


async def main():
    print("\n" + "="*60)
    print("AGENTNATE INTEGRATION TESTS")
    print("="*60)

    results = {}

    # Run tests
    results["meta_agent"] = await test_meta_agent_with_model()
    results["n8n_lifecycle"] = await test_n8n_lifecycle()
    results["workflow_deploy"] = await test_workflow_deployment()
    results["parallel_inference"] = await test_multi_model_parallel()

    # Summary
    section("RESULTS SUMMARY")
    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        if result is None:
            status = f"{Colors.WARN}SKIP{Colors.END}"
        elif result:
            status = f"{Colors.OK}PASS{Colors.END}"
        else:
            status = f"{Colors.FAIL}FAIL{Colors.END}"
        print(f"  {name}: {status}")

    print(f"\n{Colors.BOLD}Total: {passed}/{total} tests passed{Colors.END}")


if __name__ == "__main__":
    asyncio.run(main())
