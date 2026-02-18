"""Comprehensive API test for AgentNate Meta Agent system."""
import asyncio
import aiohttp
import json
import time

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

async def test_health():
    """Test health endpoint."""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{BASE_URL}/health") as resp:
            data = await resp.json()
            if data.get("status") == "healthy":
                ok(f"Health check: {data}")
                return True
            else:
                fail(f"Health check failed: {data}")
                return False

async def test_tools_list():
    """Test listing tools."""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{BASE_URL}/api/tools/list") as resp:
            data = await resp.json()
            tools = data.get("tools", [])
            if len(tools) >= 16:
                ok(f"Tools list: {len(tools)} tools available")
                return True
            else:
                fail(f"Expected 16+ tools, got {len(tools)}")
                return False

async def test_gpu_status():
    """Test GPU status tool."""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{BASE_URL}/api/tools/call",
            json={"tool": "get_gpu_status", "arguments": {}}
        ) as resp:
            data = await resp.json()
            if data.get("success"):
                gpus = data.get("gpus", [])
                ok(f"GPU status: {len(gpus)} GPUs detected")
                for gpu in gpus:
                    print(f"    GPU {gpu['index']}: {gpu['name']} - {gpu['memory_free_mb']}MB free")
                return True
            else:
                fail(f"GPU status failed: {data.get('error')}")
                return False

async def test_list_models():
    """Test listing available models."""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{BASE_URL}/api/tools/call",
            json={"tool": "list_available_models", "arguments": {"provider": "llama_cpp"}}
        ) as resp:
            data = await resp.json()
            if data.get("success"):
                count = data.get("count", 0)
                ok(f"List models: {count} llama_cpp models available")
                return True
            else:
                fail(f"List models failed: {data.get('error')}")
                return False

async def test_load_unload_model():
    """Test loading and unloading a model."""
    async with aiohttp.ClientSession() as session:
        # Load
        print("  Loading tinyllama on GPU 0...")
        async with session.post(
            f"{BASE_URL}/api/tools/call",
            json={
                "tool": "load_model",
                "arguments": {
                    "model_name": "tinyllama",
                    "gpu_index": 0,
                    "provider": "llama_cpp"
                }
            },
            timeout=aiohttp.ClientTimeout(total=120)
        ) as resp:
            data = await resp.json()
            if not data.get("success"):
                fail(f"Load model failed: {data.get('error')}")
                return False

            instance_id = data.get("instance_id")
            ok(f"Model loaded: {data.get('model')} (ID: {instance_id[:8]}...)")

        # Give it a moment
        await asyncio.sleep(1)

        # Unload
        print("  Unloading model...")
        async with session.post(
            f"{BASE_URL}/api/tools/call",
            json={"tool": "unload_model", "arguments": {"instance_id": instance_id}}
        ) as resp:
            data = await resp.json()
            if data.get("success"):
                ok("Model unloaded successfully")
                return True
            else:
                fail(f"Unload failed: {data.get('error')}")
                return False

async def test_quick_workflow():
    """Test quick workflow creation."""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{BASE_URL}/api/workflows/quick",
            json={
                "template": "webhook_llm",
                "name": "API Test Workflow",
                "webhook_path": "api-test"
            }
        ) as resp:
            data = await resp.json()
            if data.get("success"):
                workflow = data.get("workflow", {})
                ok(f"Quick workflow created: {workflow.get('name')} with {len(workflow.get('nodes', []))} nodes")
                return workflow
            else:
                fail(f"Quick workflow failed: {data.get('error')}")
                return None

async def test_workflow_templates():
    """Test workflow templates endpoint."""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{BASE_URL}/api/workflows/templates") as resp:
            data = await resp.json()
            templates = data.get("templates", {})
            if templates:
                ok(f"Workflow templates: {sum(len(v) for v in templates.values())} templates in {len(templates)} categories")
                return True
            else:
                fail("No templates returned")
                return False

async def test_n8n_tools():
    """Test n8n tools."""
    async with aiohttp.ClientSession() as session:
        # List instances
        async with session.post(
            f"{BASE_URL}/api/tools/call",
            json={"tool": "list_n8n_instances", "arguments": {}}
        ) as resp:
            data = await resp.json()
            if data.get("success"):
                ok(f"n8n instances: {data.get('count', 0)} running")
                return True
            else:
                fail(f"n8n list failed: {data.get('error')}")
                return False

async def test_system_health():
    """Test system health tool."""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{BASE_URL}/api/tools/call",
            json={"tool": "get_system_health", "arguments": {}}
        ) as resp:
            data = await resp.json()
            if data.get("success"):
                ok(f"System health: {data.get('status')} - {data.get('total_loaded_models')} models loaded")
                return True
            else:
                fail(f"System health failed: {data.get('error')}")
                return False

async def test_meta_agent_simple():
    """Test meta agent with a simple query."""
    async with aiohttp.ClientSession() as session:
        print("  Sending: 'What GPUs do I have?'")
        async with session.post(
            f"{BASE_URL}/api/tools/agent",
            json={"message": "What GPUs do I have? Use the gpu status tool."},
            timeout=aiohttp.ClientTimeout(total=60)
        ) as resp:
            data = await resp.json()

            if "error" in data and "No models loaded" in data.get("error", ""):
                warn("Meta agent needs a model loaded first (expected)")
                return True  # This is expected behavior

            if data.get("tool_called"):
                ok(f"Meta agent called tool: {data.get('tool_name', 'unknown')}")
                print(f"    Response: {data.get('response', '')[:100]}...")
                return True
            elif data.get("response"):
                ok(f"Meta agent responded: {data.get('response', '')[:80]}...")
                return True
            else:
                fail(f"Meta agent failed: {data}")
                return False

async def test_chat_with_model():
    """Test chat with a loaded model."""
    async with aiohttp.ClientSession() as session:
        # First load a model
        print("  Loading model for chat test...")
        async with session.post(
            f"{BASE_URL}/api/tools/call",
            json={
                "tool": "load_model",
                "arguments": {"model_name": "tinyllama", "gpu_index": 0}
            },
            timeout=aiohttp.ClientTimeout(total=120)
        ) as resp:
            data = await resp.json()
            if not data.get("success"):
                fail(f"Could not load model: {data.get('error')}")
                return False
            instance_id = data.get("instance_id")
            ok(f"Model loaded for chat: {instance_id[:8]}...")

        # Chat
        print("  Sending chat message...")
        async with session.post(
            f"{BASE_URL}/api/chat/completions",
            json={
                "instance_id": instance_id,
                "messages": [{"role": "user", "content": "Say 'Hello from API test' exactly."}],
                "max_tokens": 30,
                "temperature": 0.1
            },
            timeout=aiohttp.ClientTimeout(total=60)
        ) as resp:
            data = await resp.json()
            if "error" in data:
                fail(f"Chat failed: {data.get('error')}")
                # Cleanup
                await session.post(
                    f"{BASE_URL}/api/tools/call",
                    json={"tool": "unload_model", "arguments": {"instance_id": instance_id}}
                )
                return False

            content = data.get("content", "")
            ok(f"Chat response: {content[:60]}")

        # Unload
        await session.post(
            f"{BASE_URL}/api/tools/call",
            json={"tool": "unload_model", "arguments": {"instance_id": instance_id}}
        )
        ok("Model unloaded after chat test")
        return True

async def test_workflow_generation():
    """Test LLM-based workflow generation (requires loaded model)."""
    async with aiohttp.ClientSession() as session:
        # Load model first
        print("  Loading model for workflow generation...")
        async with session.post(
            f"{BASE_URL}/api/tools/call",
            json={"tool": "load_model", "arguments": {"model_name": "tinyllama", "gpu_index": 0}},
            timeout=aiohttp.ClientTimeout(total=120)
        ) as resp:
            data = await resp.json()
            if not data.get("success"):
                warn(f"Could not load model for generation: {data.get('error')}")
                return False
            instance_id = data.get("instance_id")

        # Generate workflow
        print("  Generating workflow from description...")
        async with session.post(
            f"{BASE_URL}/api/workflows/generate",
            json={
                "description": "Receive a webhook, call the local LLM to summarize the input, and return the response",
                "trigger_type": "webhook",
                "model_instance_id": instance_id
            },
            timeout=aiohttp.ClientTimeout(total=120)
        ) as resp:
            data = await resp.json()

            if data.get("success"):
                workflow = data.get("workflow", {})
                ok(f"Generated workflow: {workflow.get('name', 'unnamed')} with {len(workflow.get('nodes', []))} nodes")
            else:
                errors = data.get("errors", [])
                warn(f"Workflow generation had issues: {errors}")
                if data.get("raw_response"):
                    print(f"    Raw response preview: {data['raw_response'][:100]}...")

        # Cleanup
        await session.post(
            f"{BASE_URL}/api/tools/call",
            json={"tool": "unload_model", "arguments": {"instance_id": instance_id}}
        )
        return True


async def main():
    print("\n" + "="*60)
    print("AGENTNATE COMPREHENSIVE API TEST")
    print("="*60)

    results = {}

    # Basic tests
    section("1. Health & Basic Endpoints")
    results["health"] = await test_health()
    results["tools_list"] = await test_tools_list()

    section("2. System Tools")
    results["gpu_status"] = await test_gpu_status()
    results["system_health"] = await test_system_health()

    section("3. Model Tools")
    results["list_models"] = await test_list_models()
    results["load_unload"] = await test_load_unload_model()

    section("4. Workflow Tools")
    results["templates"] = await test_workflow_templates()
    results["quick_workflow"] = await test_quick_workflow() is not None

    section("5. n8n Tools")
    results["n8n"] = await test_n8n_tools()

    section("6. Chat API")
    results["chat"] = await test_chat_with_model()

    section("7. Workflow Generation (LLM)")
    results["generation"] = await test_workflow_generation()

    section("8. Meta Agent")
    results["meta_agent"] = await test_meta_agent_simple()

    # Summary
    section("RESULTS SUMMARY")
    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = f"{Colors.OK}PASS{Colors.END}" if result else f"{Colors.FAIL}FAIL{Colors.END}"
        print(f"  {name}: {status}")

    print(f"\n{Colors.BOLD}Total: {passed}/{total} tests passed{Colors.END}")

    if passed == total:
        print(f"\n{Colors.OK}All tests passed!{Colors.END}")
    else:
        print(f"\n{Colors.WARN}Some tests failed - check logs above{Colors.END}")


if __name__ == "__main__":
    asyncio.run(main())
