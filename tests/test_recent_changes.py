"""
Comprehensive test suite for recent AgentNate changes.
Tests: ComfyUI tools, personas, agent loop, abort endpoint, tool routing, frontend.
Run with: python\\python.exe test_recent_changes.py
"""

import sys
import os
import json
import time
import asyncio
import re
import traceback

# Portable - use project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

API = "http://localhost:8000/api"

# Track results
results = {"passed": 0, "failed": 0, "errors": []}
ALL_TESTS = []

def test(name):
    """Decorator for test functions - collects into ALL_TESTS list."""
    def decorator(func):
        async def wrapper():
            try:
                result = await func()
                if result is True or result is None:
                    results["passed"] += 1
                    print(f"  PASS  {name}")
                else:
                    results["failed"] += 1
                    results["errors"].append(f"{name}: returned {result}")
                    print(f"  FAIL  {name}: {result}")
            except Exception as e:
                results["failed"] += 1
                msg = f"{name}: {type(e).__name__}: {e}"
                results["errors"].append(msg)
                print(f"  FAIL  {msg}")
        wrapper.__test_name__ = name
        ALL_TESTS.append(wrapper)
        return wrapper
    return decorator


# ==================== HTTP helpers ====================

async def http_get(path):
    import httpx
    async with httpx.AsyncClient(timeout=10) as c:
        return await c.get(f"{API}{path}")

async def http_post(path, data=None):
    import httpx
    async with httpx.AsyncClient(timeout=10) as c:
        return await c.post(f"{API}{path}", json=data)

async def http_delete(path):
    import httpx
    async with httpx.AsyncClient(timeout=10) as c:
        return await c.delete(f"{API}{path}")

async def http_get_raw(url):
    import httpx
    async with httpx.AsyncClient(timeout=10) as c:
        return await c.get(url)


# ==================== 1. Health & System ====================

@test("1.1 Health endpoint returns healthy")
async def _():
    r = await http_get_raw("http://localhost:8000/health")
    d = r.json()
    assert r.status_code == 200, f"status {r.status_code}"
    assert d["status"] == "healthy"
    assert d["orchestrator"] is True
    assert d["n8n_manager"] is True

@test("1.2 System GPU endpoint")
async def _():
    r = await http_get("/system/gpu")
    assert r.status_code == 200, f"status {r.status_code}"
    d = r.json()
    assert "gpus" in d or isinstance(d, list)

@test("1.3 System queue endpoint")
async def _():
    r = await http_get("/system/queue")
    assert r.status_code == 200, f"status {r.status_code}"

@test("1.4 System models summary")
async def _():
    r = await http_get("/system/models/summary")
    assert r.status_code == 200

@test("1.5 Root serves index.html")
async def _():
    r = await http_get_raw("http://localhost:8000/")
    assert r.status_code == 200
    assert "AgentNate" in r.text

@test("1.6 Favicon SVG served")
async def _():
    r = await http_get_raw("http://localhost:8000/favicon.svg")
    assert r.status_code == 200
    assert "svg" in r.headers.get("content-type", "")


# ==================== 2. Static File Serving ====================

@test("2.1 JS files served with correct MIME type")
async def _():
    r = await http_get_raw("http://localhost:8000/static/js/app.js")
    assert r.status_code == 200
    ct = r.headers.get("content-type", "")
    assert "javascript" in ct, f"wrong content-type: {ct}"

@test("2.2 CSS files served with no-cache")
async def _():
    r = await http_get_raw("http://localhost:8000/static/styles.css")
    assert r.status_code == 200
    cc = r.headers.get("cache-control", "")
    assert "no-cache" in cc, f"missing no-cache: {cc}"

@test("2.3 JS files served with no-cache headers")
async def _():
    r = await http_get_raw("http://localhost:8000/static/js/state.js")
    assert r.status_code == 200
    cc = r.headers.get("cache-control", "")
    assert "no-cache" in cc, f"missing no-cache: {cc}"

@test("2.4 Non-existent static file returns 404")
async def _():
    r = await http_get_raw("http://localhost:8000/static/nonexistent_file_xyz.js")
    assert r.status_code in (404, 307), f"expected 404, got {r.status_code}"


# ==================== 3. Models & Providers ====================

@test("3.1 Loaded models endpoint")
async def _():
    r = await http_get("/models/loaded")
    assert r.status_code == 200

@test("3.2 Models list endpoint")
async def _():
    r = await http_get("/models/list")
    assert r.status_code == 200

@test("3.3 Providers health endpoint")
async def _():
    r = await http_get("/models/health/all")
    assert r.status_code == 200
    d = r.json()
    assert "llama_cpp" in d

@test("3.4 Providers health keys")
async def _():
    r = await http_get("/models/health/all")
    assert r.status_code == 200
    d = r.json()
    expected_providers = {"llama_cpp", "vllm", "lm_studio", "ollama", "openrouter"}
    assert expected_providers.issubset(set(d.keys())), f"missing providers: {expected_providers - set(d.keys())}"

@test("3.5 Pending loads endpoint")
async def _():
    r = await http_get("/models/pending")
    assert r.status_code == 200


# ==================== 4. Personas ====================

@test("4.1 Personas endpoint returns list")
async def _():
    r = await http_get("/tools/personas")
    assert r.status_code == 200
    d = r.json()
    assert "personas" in d
    assert len(d["personas"]) > 0

@test("4.2 power_agent persona exists")
async def _():
    r = await http_get("/tools/personas")
    d = r.json()
    ids = [p["id"] for p in d["personas"]]
    assert "power_agent" in ids, f"power_agent missing, got: {ids}"

@test("4.3 image_creator persona exists (new)")
async def _():
    r = await http_get("/tools/personas")
    d = r.json()
    ids = [p["id"] for p in d["personas"]]
    assert "image_creator" in ids, f"image_creator missing, got: {ids}"

@test("4.4 image_creator has comfyui tools")
async def _():
    r = await http_get("/tools/personas")
    d = r.json()
    persona = next((p for p in d["personas"] if p["id"] == "image_creator"), None)
    assert persona is not None, "image_creator not found"
    assert "comfyui" in persona.get("tools", []), f"tools: {persona.get('tools')}"

@test("4.5 All expected personas present")
async def _():
    r = await http_get("/tools/personas")
    d = r.json()
    ids = {p["id"] for p in d["personas"]}
    expected = {"power_agent", "system_agent", "researcher", "coder", "workflow_builder",
                "vision_agent", "data_analyst", "image_creator"}
    missing = expected - ids
    assert not missing, f"missing personas: {missing}"

@test("4.6 Individual persona endpoint")
async def _():
    r = await http_get("/tools/personas/power_agent")
    assert r.status_code == 200
    d = r.json()
    assert d.get("id") == "power_agent" or d.get("persona", {}).get("id") == "power_agent"

@test("4.7 image_creator persona has description")
async def _():
    r = await http_get("/tools/personas")
    d = r.json()
    persona = next((p for p in d["personas"] if p["id"] == "image_creator"), None)
    assert persona is not None
    desc = persona.get("description", "")
    assert len(desc) > 10, f"description too short: '{desc}'"


# ==================== 5. Tool System ====================

@test("5.1 Tools list endpoint returns tools")
async def _():
    r = await http_get("/tools/list")
    assert r.status_code == 200
    d = r.json()
    assert "tools" in d or "categories" in d

@test("5.2 Exactly 88 tools registered")
async def _():
    from backend.tools import AVAILABLE_TOOLS
    assert len(AVAILABLE_TOOLS) == 88, f"expected 88, got {len(AVAILABLE_TOOLS)}"

@test("5.3 Exactly 9 ComfyUI tools in AVAILABLE_TOOLS")
async def _():
    from backend.tools import AVAILABLE_TOOLS
    comfyui = [t for t in AVAILABLE_TOOLS if t["name"].startswith("comfyui_")]
    assert len(comfyui) == 9, f"expected 9, got {len(comfyui)}: {[t['name'] for t in comfyui]}"

@test("5.4 ComfyUI tools have correct names")
async def _():
    from backend.tools import AVAILABLE_TOOLS
    comfyui_names = {t["name"] for t in AVAILABLE_TOOLS if t["name"].startswith("comfyui_")}
    expected = {"comfyui_status", "comfyui_install", "comfyui_start_api", "comfyui_stop_api",
                "comfyui_list_instances", "comfyui_add_instance", "comfyui_start_instance",
                "comfyui_stop_instance", "comfyui_list_models"}
    assert comfyui_names == expected, f"mismatch: got {comfyui_names}"

@test("5.5 comfyui tool group in TOOL_GROUPS")
async def _():
    from backend.personas import TOOL_GROUPS
    assert "comfyui" in TOOL_GROUPS, f"comfyui not in TOOL_GROUPS: {list(TOOL_GROUPS.keys())}"
    assert len(TOOL_GROUPS["comfyui"]) == 9

@test("5.6 All tool categories present in CATEGORY_INFO")
async def _():
    from backend.tools.tool_router import CATEGORY_INFO, CATEGORY_ORDER
    expected_cats = ["system", "model", "workflow", "n8n", "comfyui", "web", "files",
                     "code", "communication", "data", "utility", "vision", "codebase"]
    for cat in expected_cats:
        assert cat in CATEGORY_INFO, f"category '{cat}' missing from CATEGORY_INFO"
        assert cat in CATEGORY_ORDER, f"category '{cat}' missing from CATEGORY_ORDER"

@test("5.7 ComfyUI category positioned after n8n")
async def _():
    from backend.tools.tool_router import CATEGORY_ORDER
    n8n_idx = CATEGORY_ORDER.index("n8n")
    comfyui_idx = CATEGORY_ORDER.index("comfyui")
    assert comfyui_idx == n8n_idx + 1, f"comfyui at {comfyui_idx}, n8n at {n8n_idx}"

@test("5.8 Every tool has name, description, and parameters")
async def _():
    from backend.tools import AVAILABLE_TOOLS
    for tool in AVAILABLE_TOOLS:
        assert "name" in tool, f"tool missing 'name': {tool}"
        assert "description" in tool, f"tool {tool['name']} missing 'description'"
        assert "parameters" in tool, f"tool {tool['name']} missing 'parameters'"

@test("5.9 Tool info endpoint for comfyui_status")
async def _():
    r = await http_get("/tools/info/comfyui_status")
    assert r.status_code == 200


# ==================== 6. Tool Router Wiring ====================

@test("6.1 ToolRouter instantiates with comfyui_manager")
async def _():
    from backend.tools.tool_router import ToolRouter
    from settings.settings_manager import SettingsManager
    settings = SettingsManager(settings_dir=BASE_DIR)
    router = ToolRouter(orchestrator=None, n8n_manager=None, settings=settings, comfyui_manager=None)
    assert hasattr(router, "comfyui_tools")
    assert router.comfyui_tools is not None

@test("6.2 ToolRouter has all 9 ComfyUI routes")
async def _():
    from backend.tools.tool_router import ToolRouter
    from settings.settings_manager import SettingsManager
    settings = SettingsManager(settings_dir=BASE_DIR)
    router = ToolRouter(orchestrator=None, n8n_manager=None, settings=settings, comfyui_manager=None)
    comfyui_routes = [k for k in router._routes if k.startswith("comfyui_")]
    assert len(comfyui_routes) == 9, f"expected 9 routes, got {len(comfyui_routes)}: {comfyui_routes}"

@test("6.3 ToolRouter.execute() handles comfyui_status")
async def _():
    from backend.tools.tool_router import ToolRouter
    from settings.settings_manager import SettingsManager
    settings = SettingsManager(settings_dir=BASE_DIR)
    router = ToolRouter(orchestrator=None, n8n_manager=None, settings=settings, comfyui_manager=None)
    result = await router.execute("comfyui_status", {})
    assert isinstance(result, dict)
    assert "success" in result

@test("6.4 ToolRouter.execute() handles comfyui_list_models")
async def _():
    from backend.tools.tool_router import ToolRouter
    from settings.settings_manager import SettingsManager
    settings = SettingsManager(settings_dir=BASE_DIR)
    router = ToolRouter(orchestrator=None, n8n_manager=None, settings=settings, comfyui_manager=None)
    result = await router.execute("comfyui_list_models", {})
    assert isinstance(result, dict)

@test("6.5 ToolRouter.execute() handles comfyui_add_instance with args")
async def _():
    from backend.tools.tool_router import ToolRouter
    from settings.settings_manager import SettingsManager
    settings = SettingsManager(settings_dir=BASE_DIR)
    router = ToolRouter(orchestrator=None, n8n_manager=None, settings=settings, comfyui_manager=None)
    result = await router.execute("comfyui_add_instance", {"gpu_device": 0, "vram_mode": "normal"})
    assert isinstance(result, dict)

@test("6.6 ToolRouter.execute() returns error for unknown tool")
async def _():
    from backend.tools.tool_router import ToolRouter
    from settings.settings_manager import SettingsManager
    settings = SettingsManager(settings_dir=BASE_DIR)
    router = ToolRouter(orchestrator=None, n8n_manager=None, settings=settings, comfyui_manager=None)
    result = await router.execute("nonexistent_xyz", {})
    assert isinstance(result, dict)
    has_error = "error" in str(result).lower() or result.get("success") is False
    assert has_error, f"expected error, got: {result}"


# ==================== 7. ComfyUI Tools Direct Tests ====================

@test("7.1 ComfyUITools class has all 9 methods")
async def _():
    from backend.tools.comfyui_tools import ComfyUITools
    tools = ComfyUITools(comfyui_manager=None)
    methods = ["comfyui_status", "comfyui_install", "comfyui_start_api", "comfyui_stop_api",
               "comfyui_list_instances", "comfyui_add_instance", "comfyui_start_instance",
               "comfyui_stop_instance", "comfyui_list_models"]
    for m in methods:
        assert hasattr(tools, m), f"missing method: {m}"
        assert callable(getattr(tools, m)), f"not callable: {m}"

@test("7.2 comfyui_status graceful fallback (no manager)")
async def _():
    from backend.tools.comfyui_tools import ComfyUITools
    tools = ComfyUITools(comfyui_manager=None)
    result = await tools.comfyui_status()
    assert isinstance(result, dict)
    assert result["success"] is False
    assert "not configured" in result.get("error", "").lower()

@test("7.3 comfyui_start_api graceful fallback")
async def _():
    from backend.tools.comfyui_tools import ComfyUITools
    tools = ComfyUITools(comfyui_manager=None)
    result = await tools.comfyui_start_api()
    assert result["success"] is False

@test("7.4 comfyui_stop_api graceful fallback")
async def _():
    from backend.tools.comfyui_tools import ComfyUITools
    tools = ComfyUITools(comfyui_manager=None)
    result = await tools.comfyui_stop_api()
    assert result["success"] is False

@test("7.5 comfyui_list_instances graceful fallback")
async def _():
    from backend.tools.comfyui_tools import ComfyUITools
    tools = ComfyUITools(comfyui_manager=None)
    result = await tools.comfyui_list_instances()
    assert result["success"] is False

@test("7.6 comfyui_list_models graceful fallback")
async def _():
    from backend.tools.comfyui_tools import ComfyUITools
    tools = ComfyUITools(comfyui_manager=None)
    result = await tools.comfyui_list_models()
    assert result["success"] is False

@test("7.7 comfyui_install graceful fallback")
async def _():
    from backend.tools.comfyui_tools import ComfyUITools
    tools = ComfyUITools(comfyui_manager=None)
    result = await tools.comfyui_install()
    assert result["success"] is False

@test("7.8 comfyui_add_instance with kwargs")
async def _():
    from backend.tools.comfyui_tools import ComfyUITools
    tools = ComfyUITools(comfyui_manager=None)
    result = await tools.comfyui_add_instance(gpu_device=0, port=8188, vram_mode="normal")
    assert result["success"] is False

@test("7.9 comfyui_start_instance with instance_id")
async def _():
    from backend.tools.comfyui_tools import ComfyUITools
    tools = ComfyUITools(comfyui_manager=None)
    result = await tools.comfyui_start_instance(instance_id="test-123")
    assert result["success"] is False


# ==================== 8. Tool Execution via API ====================

@test("8.1 Execute comfyui_status via /tools/call")
async def _():
    r = await http_post("/tools/call", {"tool": "comfyui_status", "arguments": {}})
    assert r.status_code == 200
    d = r.json()
    assert "result" in d or "success" in d, f"unexpected response keys: {list(d.keys())}"

@test("8.2 Execute get_gpu_status via /tools/call")
async def _():
    r = await http_post("/tools/call", {"tool": "get_gpu_status", "arguments": {}})
    assert r.status_code == 200

@test("8.3 Execute list_loaded_models via /tools/call")
async def _():
    r = await http_post("/tools/call", {"tool": "list_loaded_models", "arguments": {}})
    assert r.status_code == 200

@test("8.4 Unknown tool returns error via /tools/call")
async def _():
    r = await http_post("/tools/call", {"tool": "nonexistent_tool_xyz", "arguments": {}})
    d = r.json()
    result = d.get("result", d)
    has_error = "error" in str(result).lower() or "not found" in str(result).lower() or "unknown" in str(result).lower()
    assert has_error, f"expected error for unknown tool, got: {d}"

@test("8.5 Execute comfyui_list_models via /tools/call")
async def _():
    r = await http_post("/tools/call", {"tool": "comfyui_list_models", "arguments": {}})
    assert r.status_code == 200

@test("8.6 Execute comfyui_list_instances via /tools/call")
async def _():
    r = await http_post("/tools/call", {"tool": "comfyui_list_instances", "arguments": {}})
    assert r.status_code == 200


# ==================== 9. Agent Abort Endpoint ====================

@test("9.1 Agent abort endpoint exists and responds")
async def _():
    r = await http_post("/tools/agent/abort", {"abort_id": "test_nonexistent_123"})
    assert r.status_code == 200
    d = r.json()
    assert "status" in d or "ok" in d or "message" in d

@test("9.2 Agent abort handles missing abort_id gracefully")
async def _():
    import httpx
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.post(f"{API}/tools/agent/abort", json={})
    assert r.status_code in (200, 400, 422), f"unexpected status: {r.status_code}"


# ==================== 10. Agent Endpoints ====================

@test("10.1 Agent non-streaming endpoint exists")
async def _():
    r = await http_post("/tools/agent", {
        "message": "test",
        "instance_id": "nonexistent",
        "persona_id": "power_agent"
    })
    # Should return something (error about no model is fine)
    assert r.status_code in (200, 400, 422, 500)

@test("10.2 Agent stream endpoint exists")
async def _():
    r = await http_post("/tools/agent/stream", {
        "message": "hello",
        "instance_id": None,
        "persona_id": "power_agent",
        "autonomous": False,
        "max_tool_calls": 5
    })
    assert r.status_code in (200, 400, 422, 500)

@test("10.3 Agent debug endpoint")
async def _():
    r = await http_get("/tools/agent/debug")
    assert r.status_code == 200


# ==================== 11. Settings ====================

@test("11.1 Settings GET endpoint")
async def _():
    r = await http_get("/settings")
    assert r.status_code == 200

@test("11.2 Settings returns dict")
async def _():
    r = await http_get("/settings")
    d = r.json()
    assert isinstance(d, dict)


# ==================== 12. n8n Endpoints ====================

@test("12.1 n8n main status endpoint")
async def _():
    r = await http_get("/n8n/main/status")
    assert r.status_code == 200

@test("12.2 n8n workers list endpoint")
async def _():
    r = await http_get("/n8n/workers")
    assert r.status_code == 200

@test("12.3 n8n workflows endpoint")
async def _():
    r = await http_get("/n8n/workflows")
    assert r.status_code == 200


# ==================== 13. ComfyUI Backend Routes ====================

@test("13.1 ComfyUI status route")
async def _():
    r = await http_get("/comfyui/status")
    assert r.status_code == 200
    d = r.json()
    assert "module_downloaded" in d or "api_running" in d or isinstance(d, dict)

@test("13.2 ComfyUI instances route (503 if API not running)")
async def _():
    r = await http_get("/comfyui/instances")
    assert r.status_code in (200, 503), f"unexpected status: {r.status_code}"

@test("13.3 ComfyUI GPUs route (503 if API not running)")
async def _():
    r = await http_get("/comfyui/gpus")
    assert r.status_code in (200, 503), f"unexpected status: {r.status_code}"

@test("13.4 ComfyUI models local route (503 if API not running)")
async def _():
    r = await http_get("/comfyui/models/local")
    assert r.status_code in (200, 503), f"unexpected status: {r.status_code}"

@test("13.5 ComfyUI nodes installed route (503 if API not running)")
async def _():
    r = await http_get("/comfyui/nodes/installed")
    assert r.status_code in (200, 503), f"unexpected status: {r.status_code}"


# ==================== 14. Frontend JS Validation ====================

@test("14.1 agent.js has all required functions")
async def _():
    r = await http_get_raw("http://localhost:8000/static/js/agent.js")
    assert r.status_code == 200
    content = r.text
    required = ["toggleAgentMode", "sendAgentModeMessage", "addToolCallMessage",
                "showAgentWelcome", "fillAgentPrompt", "stopAgent",
                "TOOL_ICONS", "TOOL_FRIENDLY_NAMES", "getToolIcon",
                "getToolFriendlyName", "summarizeArgs", "scrollChatToBottom"]
    for fn in required:
        assert fn in content, f"missing: {fn}"

@test("14.2 state.js has new agent state fields")
async def _():
    r = await http_get_raw("http://localhost:8000/static/js/state.js")
    assert r.status_code == 200
    content = r.text
    assert "agentWelcomeShown" in content
    assert "agentAbortId" in content

@test("14.3 app.js exports fillAgentPrompt and stopAgent")
async def _():
    r = await http_get_raw("http://localhost:8000/static/js/app.js")
    assert r.status_code == 200
    content = r.text
    assert "fillAgentPrompt" in content
    assert "stopAgent" in content

@test("14.4 styles.css has all new agent styles")
async def _():
    r = await http_get_raw("http://localhost:8000/static/styles.css")
    assert r.status_code == 200
    content = r.text
    required_classes = [".agent-welcome", ".agent-quick-actions", ".tool-call-card",
                        ".agent-stop-btn", ".agent-status-message", ".agent-welcome-capabilities",
                        ".agent-capability", ".agent-welcome-tip"]
    for cls in required_classes:
        assert cls in content, f"missing CSS: {cls}"

@test("14.5 index.html has required structure")
async def _():
    r = await http_get_raw("http://localhost:8000/")
    assert r.status_code == 200
    content = r.text
    assert 'type="module"' in content
    assert 'js/app.js' in content
    assert 'agent-mode-toggle' in content
    assert 'autonomous-toggle' in content
    assert 'persona-select' in content


# ==================== 15. HTML/JS Onclick Consistency ====================

@test("15.1 All onclick handlers map to window exports")
async def _():
    r = await http_get_raw("http://localhost:8000/")
    html = r.text
    r2 = await http_get_raw("http://localhost:8000/static/js/app.js")
    app_js = r2.text

    onclick_pattern = re.compile(r'onclick="(\w+)\(')
    html_funcs = set(onclick_pattern.findall(html))

    window_section = app_js[app_js.find("Object.assign(window"):]
    window_pattern = re.compile(r'^\s+(\w+)[,:]', re.MULTILINE)
    window_funcs = set(window_pattern.findall(window_section))

    known_aliases = {"selectCategory"}
    missing = html_funcs - window_funcs - known_aliases

    assert not missing, f"onclick functions not exported to window: {missing}"


# ==================== 16. Debug Log Endpoints ====================

@test("16.1 Debug log POST endpoint")
async def _():
    import httpx
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.post("http://localhost:8000/api/debug/log",
                         json={"action": "test", "detail": "automated test"})
    assert r.status_code == 200
    d = r.json()
    assert d.get("ok") is True

@test("16.2 Debug log DELETE endpoint")
async def _():
    import httpx
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.delete("http://localhost:8000/api/debug/log")
    assert r.status_code == 200


# ==================== 17. Conversations ====================

@test("17.1 Conversations list endpoint")
async def _():
    r = await http_get("/tools/conversations")
    assert r.status_code == 200

@test("17.2 Marketplace categories endpoint")
async def _():
    r = await http_get("/marketplace/categories")
    assert r.status_code == 200


# ==================== 18. Source Code Validation ====================

@test("18.1 SYSTEM_CHANGING_TOOLS includes ComfyUI tools")
async def _():
    source = open(os.path.join(BASE_DIR, "backend", "routes", "tools.py")).read()
    assert "comfyui_start_api" in source
    assert "comfyui_stop_api" in source
    assert "comfyui_add_instance" in source
    assert "SYSTEM_CHANGING_TOOLS" in source

@test("18.2 Agent max_tool_calls default is 10")
async def _():
    source = open(os.path.join(BASE_DIR, "backend", "routes", "tools.py")).read()
    assert "max_tool_calls: int = 10" in source

@test("18.3 abort_id field exists in AgentChatRequest")
async def _():
    source = open(os.path.join(BASE_DIR, "backend", "routes", "tools.py")).read()
    assert "abort_id" in source

@test("18.4 abort_signals initialized in server.py")
async def _():
    source = open(os.path.join(BASE_DIR, "backend", "server.py")).read()
    assert "abort_signals" in source

@test("18.5 ComfyUITools imported in __init__.py")
async def _():
    source = open(os.path.join(BASE_DIR, "backend", "tools", "__init__.py")).read()
    assert "ComfyUITools" in source
    assert "comfyui_tools" in source


# ==================== 19. Persona Manager Direct ====================

@test("19.1 PersonaManager loads all personas")
async def _():
    from backend.personas import PersonaManager
    from pathlib import Path
    pm = PersonaManager(config_dir=Path(BASE_DIR))
    all_personas = pm.list_all()
    assert len(all_personas) >= 8, f"expected >= 8 personas, got {len(all_personas)}"

@test("19.2 PersonaManager has image_creator")
async def _():
    from backend.personas import PersonaManager
    from pathlib import Path
    pm = PersonaManager(config_dir=Path(BASE_DIR))
    all_personas = pm.list_all()
    ids = [p.id for p in all_personas]
    assert "image_creator" in ids

@test("19.3 image_creator persona has correct tool groups")
async def _():
    from backend.personas import PersonaManager
    from pathlib import Path
    pm = PersonaManager(config_dir=Path(BASE_DIR))
    all_personas = pm.list_all()
    ic = next(p for p in all_personas if p.id == "image_creator")
    assert "comfyui" in ic.tools
    assert "web" in ic.tools
    assert "files" in ic.tools


# ==================== 20. Tools Prompt Generation ====================

@test("20.1 get_tools_for_prompt includes ComfyUI tools")
async def _():
    from backend.tools import get_tools_for_prompt
    prompt = get_tools_for_prompt()
    assert "comfyui_status" in prompt
    assert "comfyui_install" in prompt

@test("20.2 get_tools_for_prompt returns non-empty string")
async def _():
    from backend.tools import get_tools_for_prompt
    prompt = get_tools_for_prompt()
    assert isinstance(prompt, str)
    assert len(prompt) > 100

@test("20.3 Condensed prompt includes categories")
async def _():
    from backend.tools import get_tools_for_prompt
    prompt = get_tools_for_prompt(condensed=True)
    assert "ComfyUI" in prompt or "comfyui" in prompt.lower()


# ==================== Run All ====================

async def main():
    print("\n" + "=" * 60)
    print("  AgentNate Comprehensive Test Suite")
    print("  Testing: ComfyUI tools, personas, agent loop,")
    print("  abort, tool routing, API routes, frontend")
    print("=" * 60 + "\n")

    print(f"Running {len(ALL_TESTS)} tests...\n")

    for t in ALL_TESTS:
        await t()

    print("\n" + "=" * 60)
    total = results["passed"] + results["failed"]
    pct = (results["passed"] / total * 100) if total > 0 else 0
    print(f"  Results: {results['passed']}/{total} passed ({pct:.0f}%)")
    if results["failed"] > 0:
        print(f"  {results['failed']} FAILED")
    else:
        print(f"  ALL TESTS PASSED")
    print("=" * 60)

    if results["errors"]:
        print("\nFailures:")
        for e in results["errors"]:
            print(f"  - {e}")

    print()
    return results["failed"] == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
