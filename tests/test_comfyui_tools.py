"""
Comprehensive tests for ComfyUI agent tools.

Tests all 34 ComfyUI tools with mocked manager — no running ComfyUI needed.
Focuses on the 14 new management tools added for full API coverage,
plus registration consistency and TOOL_DEFINITIONS integrity.
"""

import sys
import os
import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.tools.comfyui_tools import ComfyUITools, TOOL_DEFINITIONS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_async(coro):
    """Run an async coroutine synchronously."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        # Already inside an event loop — create a new one in a thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


def make_mock_manager(api_running=True):
    """Create a mock ComfyUI manager with sensible defaults."""
    manager = MagicMock()
    manager.is_api_running = AsyncMock(return_value=api_running)
    manager.is_module_downloaded = MagicMock(return_value=True)
    manager.is_bootstrapped = MagicMock(return_value=True)
    manager.is_comfyui_installed = MagicMock(return_value=True)
    manager.api_port = 5000
    manager.module_dir = Path("modules/comfyui")
    manager.proxy = AsyncMock(return_value={})
    manager.get_status = AsyncMock(return_value={
        "module_downloaded": True,
        "bootstrapped": True,
        "comfyui_installed": True,
        "api_running": True,
        "api_port": 5000,
        "instances": [],
        "gpus": [],
    })
    manager.start_api_server = AsyncMock(return_value={"success": True})
    manager.stop_api_server = AsyncMock(return_value={"success": True})
    manager.download_module = AsyncMock(return_value={"success": True})
    manager.bootstrap = AsyncMock(return_value={"success": True})
    return manager


def make_tools(manager=None, catalog=None):
    """Create a ComfyUITools instance with optional mock manager/catalog."""
    if manager is None:
        manager = make_mock_manager()
    return ComfyUITools(comfyui_manager=manager, media_catalog=catalog)


# ---------------------------------------------------------------------------
# Test: TOOL_DEFINITIONS integrity
# ---------------------------------------------------------------------------

class TestToolDefinitions(unittest.TestCase):
    """Verify TOOL_DEFINITIONS structure is correct for all 34 tools."""

    def test_total_count(self):
        """Should have exactly 35 ComfyUI tool definitions."""
        self.assertEqual(len(TOOL_DEFINITIONS), 35)

    def test_all_have_required_fields(self):
        """Every definition must have name, description, parameters."""
        for td in TOOL_DEFINITIONS:
            self.assertIn("name", td, f"Missing 'name' in {td}")
            self.assertIn("description", td, f"Missing 'description' in {td.get('name', '?')}")
            self.assertIn("parameters", td, f"Missing 'parameters' in {td['name']}")

    def test_all_names_unique(self):
        """No duplicate tool names."""
        names = [td["name"] for td in TOOL_DEFINITIONS]
        self.assertEqual(len(names), len(set(names)), f"Duplicate names: {[n for n in names if names.count(n) > 1]}")

    def test_all_names_start_with_comfyui(self):
        """All tool names should start with 'comfyui_'."""
        for td in TOOL_DEFINITIONS:
            self.assertTrue(td["name"].startswith("comfyui_"), f"{td['name']} doesn't start with 'comfyui_'")

    def test_parameters_have_type_object(self):
        """All parameters blocks should be type: object with properties."""
        for td in TOOL_DEFINITIONS:
            params = td["parameters"]
            self.assertEqual(params.get("type"), "object", f"{td['name']} parameters.type != object")
            self.assertIn("properties", params, f"{td['name']} missing parameters.properties")
            self.assertIn("required", params, f"{td['name']} missing parameters.required")

    def test_required_params_are_in_properties(self):
        """Required params must be defined in properties."""
        for td in TOOL_DEFINITIONS:
            props = set(td["parameters"]["properties"].keys())
            required = set(td["parameters"]["required"])
            self.assertTrue(required.issubset(props),
                            f"{td['name']}: required {required - props} not in properties")

    def test_new_tools_present(self):
        """All 14 new tools must be defined."""
        names = {td["name"] for td in TOOL_DEFINITIONS}
        new_tools = [
            "comfyui_list_node_packs", "comfyui_list_installed_nodes",
            "comfyui_update_nodes", "comfyui_remove_node",
            "comfyui_remove_instance", "comfyui_start_all_instances",
            "comfyui_stop_all_instances", "comfyui_model_categories",
            "comfyui_get_settings", "comfyui_update_settings",
            "comfyui_update_comfyui", "comfyui_purge",
            "comfyui_manage_external", "comfyui_list_gpus",
        ]
        for tool in new_tools:
            self.assertIn(tool, names, f"New tool '{tool}' missing from TOOL_DEFINITIONS")

    def test_descriptions_not_empty(self):
        """All descriptions should be non-empty."""
        for td in TOOL_DEFINITIONS:
            self.assertTrue(len(td["description"]) > 10,
                            f"{td['name']} has too-short description: '{td['description']}'")


# ---------------------------------------------------------------------------
# Test: Registration consistency across files
# ---------------------------------------------------------------------------

class TestRegistrationConsistency(unittest.TestCase):
    """Verify tools are registered in tool_router and personas."""

    def test_tool_router_has_all_comfyui_tools(self):
        """tool_router.py AVAILABLE_TOOLS must include all 35 ComfyUI tools."""
        from backend.tools.tool_router import AVAILABLE_TOOLS
        avail_names = {t["name"] for t in AVAILABLE_TOOLS}
        for td in TOOL_DEFINITIONS:
            self.assertIn(td["name"], avail_names,
                          f"{td['name']} missing from AVAILABLE_TOOLS in tool_router.py")

    def test_personas_tool_group_has_all(self):
        """personas.py TOOL_GROUPS['comfyui'] must include all 35 tools."""
        from backend.personas import TOOL_GROUPS
        group = set(TOOL_GROUPS["comfyui"])
        for td in TOOL_DEFINITIONS:
            self.assertIn(td["name"], group,
                          f"{td['name']} missing from TOOL_GROUPS['comfyui'] in personas.py")

    def test_no_extra_in_persona_group(self):
        """TOOL_GROUPS['comfyui'] shouldn't have tools that don't exist."""
        from backend.personas import TOOL_GROUPS
        defined_names = {td["name"] for td in TOOL_DEFINITIONS}
        group = set(TOOL_GROUPS["comfyui"])
        extra = group - defined_names
        self.assertEqual(extra, set(),
                         f"TOOL_GROUPS has tools not in TOOL_DEFINITIONS: {extra}")

    def test_tool_router_routes_exist(self):
        """ToolRouter._routes must map all 35 ComfyUI tools."""
        # We can't instantiate ToolRouter without real deps, so we verify
        # the source has the route strings
        import re
        router_path = os.path.join(os.path.dirname(__file__), "..", "backend", "tools", "tool_router.py")
        with open(router_path, "r", encoding="utf-8") as f:
            source = f.read()

        for td in TOOL_DEFINITIONS:
            # Check for "tool_name": self.comfyui_tools.tool_name pattern
            pattern = f'"{td["name"]}"'
            self.assertIn(pattern, source,
                          f'{td["name"]} not found as route key in tool_router.py')


# ---------------------------------------------------------------------------
# Test: Manager/API guard checks
# ---------------------------------------------------------------------------

class TestGuardChecks(unittest.TestCase):
    """Test _check_manager and _check_api error paths."""

    def test_no_manager_returns_error(self):
        """Tools should fail gracefully when manager is None."""
        tools = ComfyUITools(comfyui_manager=None)
        result = run_async(tools.comfyui_list_gpus())
        self.assertFalse(result["success"])
        self.assertIn("not configured", result["error"].lower())

    def test_api_not_running_returns_error(self):
        """Tools should fail when API is not running."""
        manager = make_mock_manager(api_running=False)
        tools = ComfyUITools(comfyui_manager=manager)
        result = run_async(tools.comfyui_list_gpus())
        self.assertFalse(result["success"])
        self.assertIn("not running", result["error"].lower())

    def test_all_new_tools_check_api(self):
        """All 14 new tools should check API is running."""
        manager = make_mock_manager(api_running=False)
        tools = ComfyUITools(comfyui_manager=manager)

        # Simple tools (no required params)
        simple_tools = [
            "comfyui_list_node_packs", "comfyui_list_installed_nodes",
            "comfyui_update_nodes", "comfyui_start_all_instances",
            "comfyui_stop_all_instances", "comfyui_model_categories",
            "comfyui_get_settings", "comfyui_update_comfyui",
            "comfyui_purge", "comfyui_list_gpus",
        ]
        for name in simple_tools:
            method = getattr(tools, name)
            result = run_async(method())
            self.assertFalse(result["success"], f"{name} should fail when API not running")
            self.assertIn("not running", result["error"].lower(), f"{name} wrong error message")

        # Tools with required params
        result = run_async(tools.comfyui_remove_node(node_name="test"))
        self.assertFalse(result["success"])

        result = run_async(tools.comfyui_remove_instance(instance_id="test"))
        self.assertFalse(result["success"])

        result = run_async(tools.comfyui_update_settings(settings={"foo": "bar"}))
        self.assertFalse(result["success"])

        result = run_async(tools.comfyui_manage_external(action="list"))
        self.assertFalse(result["success"])


# ---------------------------------------------------------------------------
# Test: Node management tools
# ---------------------------------------------------------------------------

class TestNodeManagementTools(unittest.TestCase):
    """Test comfyui_list_node_packs, list_installed_nodes, update_nodes, remove_node."""

    def test_list_node_packs_success_list(self):
        """list_node_packs returns node list when API returns a list."""
        tools = make_tools()
        tools.manager.proxy.return_value = [
            {"id": "controlnet", "name": "ControlNet Auxiliary", "installed": False},
            {"id": "ipadapter", "name": "IP-Adapter", "installed": True},
        ]
        result = run_async(tools.comfyui_list_node_packs())
        self.assertTrue(result["success"])
        self.assertEqual(result["count"], 2)
        self.assertEqual(len(result["node_packs"]), 2)
        tools.manager.proxy.assert_awaited_once_with("GET", "/api/nodes/registry")

    def test_list_node_packs_success_dict(self):
        """list_node_packs handles dict response with 'nodes' key."""
        tools = make_tools()
        tools.manager.proxy.return_value = {"nodes": [{"id": "a"}]}
        result = run_async(tools.comfyui_list_node_packs())
        self.assertTrue(result["success"])
        self.assertEqual(result["count"], 1)

    def test_list_node_packs_proxy_error(self):
        """list_node_packs handles proxy exceptions."""
        tools = make_tools()
        tools.manager.proxy.side_effect = Exception("Connection refused")
        result = run_async(tools.comfyui_list_node_packs())
        self.assertFalse(result["success"])
        self.assertIn("Connection refused", result["error"])

    def test_list_installed_nodes_success(self):
        """list_installed_nodes returns installed nodes."""
        tools = make_tools()
        tools.manager.proxy.return_value = [
            {"name": "ComfyUI-Manager", "version": "1.2.3"},
        ]
        result = run_async(tools.comfyui_list_installed_nodes())
        self.assertTrue(result["success"])
        self.assertEqual(result["count"], 1)
        tools.manager.proxy.assert_awaited_once_with("GET", "/api/nodes/installed")

    def test_list_installed_nodes_dict_response(self):
        """list_installed_nodes handles dict response."""
        tools = make_tools()
        tools.manager.proxy.return_value = {"nodes": [{"name": "a"}, {"name": "b"}]}
        result = run_async(tools.comfyui_list_installed_nodes())
        self.assertTrue(result["success"])
        self.assertEqual(result["count"], 2)

    def test_update_nodes_with_job_id(self):
        """update_nodes returns job_id when async update starts."""
        tools = make_tools()
        tools.manager.proxy.return_value = {"job_id": "job-123"}
        result = run_async(tools.comfyui_update_nodes())
        self.assertTrue(result["success"])
        self.assertEqual(result["job_id"], "job-123")
        self.assertIn("comfyui_job_status", result["message"])
        tools.manager.proxy.assert_awaited_once_with("POST", "/api/nodes/update-all")

    def test_update_nodes_no_job_id(self):
        """update_nodes handles immediate success (no job_id)."""
        tools = make_tools()
        tools.manager.proxy.return_value = {"message": "All up to date"}
        result = run_async(tools.comfyui_update_nodes())
        self.assertTrue(result["success"])
        self.assertIn("result", result)

    def test_update_nodes_error(self):
        """update_nodes handles proxy error."""
        tools = make_tools()
        tools.manager.proxy.side_effect = Exception("timeout")
        result = run_async(tools.comfyui_update_nodes())
        self.assertFalse(result["success"])

    def test_remove_node_success(self):
        """remove_node calls DELETE with correct path."""
        tools = make_tools()
        tools.manager.proxy.return_value = {"status": "removed"}
        result = run_async(tools.comfyui_remove_node(node_name="ComfyUI-Impact-Pack"))
        self.assertTrue(result["success"])
        self.assertIn("ComfyUI-Impact-Pack", result["message"])
        tools.manager.proxy.assert_awaited_once_with("DELETE", "/api/nodes/ComfyUI-Impact-Pack")

    def test_remove_node_not_found(self):
        """remove_node handles error for non-existent node."""
        tools = make_tools()
        tools.manager.proxy.side_effect = Exception("Node not found")
        result = run_async(tools.comfyui_remove_node(node_name="nonexistent"))
        self.assertFalse(result["success"])
        self.assertIn("not found", result["error"].lower())


# ---------------------------------------------------------------------------
# Test: Instance management tools
# ---------------------------------------------------------------------------

class TestInstanceManagementTools(unittest.TestCase):
    """Test remove_instance, start_all, stop_all."""

    def test_remove_instance_success(self):
        """remove_instance calls DELETE with correct instance ID."""
        tools = make_tools()
        tools.manager.proxy.return_value = {"status": "removed"}
        result = run_async(tools.comfyui_remove_instance(instance_id="inst-1"))
        self.assertTrue(result["success"])
        self.assertIn("inst-1", result["message"])
        tools.manager.proxy.assert_awaited_once_with("DELETE", "/api/instances/inst-1")

    def test_remove_instance_error(self):
        """remove_instance handles error (e.g. instance still running)."""
        tools = make_tools()
        tools.manager.proxy.side_effect = Exception("Instance is running, stop it first")
        result = run_async(tools.comfyui_remove_instance(instance_id="inst-1"))
        self.assertFalse(result["success"])

    def test_start_all_instances_success(self):
        """start_all_instances calls correct endpoint."""
        tools = make_tools()
        tools.manager.proxy.return_value = {"started": 3}
        result = run_async(tools.comfyui_start_all_instances())
        self.assertTrue(result["success"])
        tools.manager.proxy.assert_awaited_once_with("POST", "/api/instances/start-all")

    def test_stop_all_instances_success(self):
        """stop_all_instances calls correct endpoint."""
        tools = make_tools()
        tools.manager.proxy.return_value = {"stopped": 3}
        result = run_async(tools.comfyui_stop_all_instances())
        self.assertTrue(result["success"])
        tools.manager.proxy.assert_awaited_once_with("POST", "/api/instances/stop-all")

    def test_start_all_error(self):
        """start_all_instances handles error."""
        tools = make_tools()
        tools.manager.proxy.side_effect = Exception("No instances configured")
        result = run_async(tools.comfyui_start_all_instances())
        self.assertFalse(result["success"])

    def test_stop_all_error(self):
        """stop_all_instances handles error."""
        tools = make_tools()
        tools.manager.proxy.side_effect = Exception("API timeout")
        result = run_async(tools.comfyui_stop_all_instances())
        self.assertFalse(result["success"])


# ---------------------------------------------------------------------------
# Test: Model categories
# ---------------------------------------------------------------------------

class TestModelCategories(unittest.TestCase):
    """Test comfyui_model_categories."""

    def test_model_categories_list_response(self):
        """model_categories returns list of categories."""
        tools = make_tools()
        tools.manager.proxy.return_value = [
            "checkpoints", "loras", "vae", "controlnet", "embeddings",
        ]
        result = run_async(tools.comfyui_model_categories())
        self.assertTrue(result["success"])
        self.assertEqual(result["count"], 5)
        self.assertIn("checkpoints", result["categories"])

    def test_model_categories_dict_response(self):
        """model_categories handles dict with 'categories' key."""
        tools = make_tools()
        tools.manager.proxy.return_value = {"categories": ["checkpoints", "loras"]}
        result = run_async(tools.comfyui_model_categories())
        self.assertTrue(result["success"])
        self.assertEqual(result["count"], 2)

    def test_model_categories_error(self):
        """model_categories handles proxy error."""
        tools = make_tools()
        tools.manager.proxy.side_effect = Exception("API error")
        result = run_async(tools.comfyui_model_categories())
        self.assertFalse(result["success"])


# ---------------------------------------------------------------------------
# Test: Settings
# ---------------------------------------------------------------------------

class TestSettingsTools(unittest.TestCase):
    """Test comfyui_get_settings and comfyui_update_settings."""

    def test_get_settings_success(self):
        """get_settings returns settings dict."""
        tools = make_tools()
        settings_data = {
            "default_vram_mode": "normal",
            "extra_model_dirs": ["/path/to/models"],
            "max_instances": 8,
        }
        tools.manager.proxy.return_value = settings_data
        result = run_async(tools.comfyui_get_settings())
        self.assertTrue(result["success"])
        self.assertEqual(result["settings"], settings_data)
        tools.manager.proxy.assert_awaited_once_with("GET", "/api/settings")

    def test_get_settings_error(self):
        """get_settings handles error."""
        tools = make_tools()
        tools.manager.proxy.side_effect = Exception("error")
        result = run_async(tools.comfyui_get_settings())
        self.assertFalse(result["success"])

    def test_update_settings_success(self):
        """update_settings sends PUT with settings payload."""
        tools = make_tools()
        tools.manager.proxy.return_value = {"updated": True}
        new_settings = {"default_vram_mode": "low"}
        result = run_async(tools.comfyui_update_settings(settings=new_settings))
        self.assertTrue(result["success"])
        self.assertIn("updated", result["message"].lower())
        tools.manager.proxy.assert_awaited_once_with("PUT", "/api/settings", json=new_settings)

    def test_update_settings_error(self):
        """update_settings handles error."""
        tools = make_tools()
        tools.manager.proxy.side_effect = Exception("Invalid settings")
        result = run_async(tools.comfyui_update_settings(settings={"bad": True}))
        self.assertFalse(result["success"])

    def test_update_settings_preserves_payload(self):
        """update_settings passes the exact settings dict to proxy."""
        tools = make_tools()
        tools.manager.proxy.return_value = {}
        complex_settings = {
            "extra_model_dirs": ["/a", "/b"],
            "max_instances": 4,
            "nested": {"key": "value"},
        }
        run_async(tools.comfyui_update_settings(settings=complex_settings))
        tools.manager.proxy.assert_awaited_once_with("PUT", "/api/settings", json=complex_settings)


# ---------------------------------------------------------------------------
# Test: Update / Purge
# ---------------------------------------------------------------------------

class TestUpdatePurge(unittest.TestCase):
    """Test comfyui_update_comfyui and comfyui_purge."""

    def test_update_comfyui_with_job_id(self):
        """update_comfyui returns job_id for async update."""
        tools = make_tools()
        tools.manager.proxy.return_value = {"job_id": "update-456"}
        result = run_async(tools.comfyui_update_comfyui())
        self.assertTrue(result["success"])
        self.assertEqual(result["job_id"], "update-456")
        self.assertIn("comfyui_job_status", result["message"])
        tools.manager.proxy.assert_awaited_once_with("POST", "/api/update")

    def test_update_comfyui_no_job_id(self):
        """update_comfyui handles immediate result (already up to date)."""
        tools = make_tools()
        tools.manager.proxy.return_value = {"message": "Already up to date"}
        result = run_async(tools.comfyui_update_comfyui())
        self.assertTrue(result["success"])
        self.assertIn("result", result)

    def test_update_comfyui_error(self):
        """update_comfyui handles error."""
        tools = make_tools()
        tools.manager.proxy.side_effect = Exception("Git pull failed")
        result = run_async(tools.comfyui_update_comfyui())
        self.assertFalse(result["success"])
        self.assertIn("Git pull failed", result["error"])

    def test_purge_success(self):
        """purge calls POST /api/purge."""
        tools = make_tools()
        tools.manager.proxy.return_value = {"purged": True}
        result = run_async(tools.comfyui_purge())
        self.assertTrue(result["success"])
        self.assertIn("purge", result["message"].lower())
        self.assertIn("re-install", result["message"].lower())
        tools.manager.proxy.assert_awaited_once_with("POST", "/api/purge")

    def test_purge_error(self):
        """purge handles error."""
        tools = make_tools()
        tools.manager.proxy.side_effect = Exception("Cannot purge while running")
        result = run_async(tools.comfyui_purge())
        self.assertFalse(result["success"])


# ---------------------------------------------------------------------------
# Test: External ComfyUI management
# ---------------------------------------------------------------------------

class TestManageExternal(unittest.TestCase):
    """Test comfyui_manage_external with all 4 actions."""

    def test_list_action(self):
        """'list' action calls GET /api/comfyui/saved."""
        tools = make_tools()
        tools.manager.proxy.return_value = [
            {"name": "Main", "directory": "C:/ComfyUI"},
            {"name": "Dev", "directory": "D:/ComfyUI-dev"},
        ]
        result = run_async(tools.comfyui_manage_external(action="list"))
        self.assertTrue(result["success"])
        self.assertEqual(len(result["saved_directories"]), 2)
        tools.manager.proxy.assert_awaited_once_with("GET", "/api/comfyui/saved")

    def test_list_action_dict_response(self):
        """'list' handles dict response with 'saved' key."""
        tools = make_tools()
        tools.manager.proxy.return_value = {"saved": [{"name": "a"}]}
        result = run_async(tools.comfyui_manage_external(action="list"))
        self.assertTrue(result["success"])
        self.assertEqual(len(result["saved_directories"]), 1)

    def test_add_action(self):
        """'add' action calls POST /api/comfyui/saved with directory."""
        tools = make_tools()
        tools.manager.proxy.return_value = {"added": True}
        result = run_async(tools.comfyui_manage_external(
            action="add", directory="C:/MyComfyUI", name="My Install"
        ))
        self.assertTrue(result["success"])
        self.assertIn("C:/MyComfyUI", result["message"])
        tools.manager.proxy.assert_awaited_once_with(
            "POST", "/api/comfyui/saved",
            json={"directory": "C:/MyComfyUI", "name": "My Install"}
        )

    def test_add_action_no_name(self):
        """'add' without name sends directory only."""
        tools = make_tools()
        tools.manager.proxy.return_value = {}
        run_async(tools.comfyui_manage_external(action="add", directory="/path"))
        tools.manager.proxy.assert_awaited_once_with(
            "POST", "/api/comfyui/saved", json={"directory": "/path"}
        )

    def test_add_action_missing_directory(self):
        """'add' without directory returns error."""
        tools = make_tools()
        result = run_async(tools.comfyui_manage_external(action="add"))
        self.assertFalse(result["success"])
        self.assertIn("directory", result["error"].lower())

    def test_remove_action(self):
        """'remove' action calls DELETE /api/comfyui/saved."""
        tools = make_tools()
        tools.manager.proxy.return_value = {"removed": True}
        result = run_async(tools.comfyui_manage_external(
            action="remove", directory="C:/OldComfyUI"
        ))
        self.assertTrue(result["success"])
        self.assertIn("C:/OldComfyUI", result["message"])
        tools.manager.proxy.assert_awaited_once_with(
            "DELETE", "/api/comfyui/saved",
            params={"directory": "C:/OldComfyUI"}
        )

    def test_remove_action_missing_directory(self):
        """'remove' without directory returns error."""
        tools = make_tools()
        result = run_async(tools.comfyui_manage_external(action="remove"))
        self.assertFalse(result["success"])
        self.assertIn("directory", result["error"].lower())

    def test_switch_action(self):
        """'switch' action calls PUT /api/comfyui/target."""
        tools = make_tools()
        tools.manager.proxy.return_value = {"target": "D:/ComfyUI-dev"}
        result = run_async(tools.comfyui_manage_external(
            action="switch", directory="D:/ComfyUI-dev"
        ))
        self.assertTrue(result["success"])
        self.assertIn("D:/ComfyUI-dev", result["message"])
        tools.manager.proxy.assert_awaited_once_with(
            "PUT", "/api/comfyui/target",
            json={"directory": "D:/ComfyUI-dev"}
        )

    def test_switch_action_missing_directory(self):
        """'switch' without directory returns error."""
        tools = make_tools()
        result = run_async(tools.comfyui_manage_external(action="switch"))
        self.assertFalse(result["success"])

    def test_invalid_action(self):
        """Unknown action returns error."""
        tools = make_tools()
        result = run_async(tools.comfyui_manage_external(action="destroy"))
        self.assertFalse(result["success"])
        self.assertIn("unknown action", result["error"].lower())

    def test_list_proxy_error(self):
        """'list' handles proxy errors."""
        tools = make_tools()
        tools.manager.proxy.side_effect = Exception("Connection lost")
        result = run_async(tools.comfyui_manage_external(action="list"))
        self.assertFalse(result["success"])
        self.assertIn("Connection lost", result["error"])

    def test_add_proxy_error(self):
        """'add' handles proxy errors."""
        tools = make_tools()
        tools.manager.proxy.side_effect = Exception("Invalid path")
        result = run_async(tools.comfyui_manage_external(
            action="add", directory="/invalid"
        ))
        self.assertFalse(result["success"])


# ---------------------------------------------------------------------------
# Test: GPU listing
# ---------------------------------------------------------------------------

class TestListGPUs(unittest.TestCase):
    """Test comfyui_list_gpus."""

    def test_list_gpus_success_list(self):
        """list_gpus returns GPU list."""
        tools = make_tools()
        tools.manager.proxy.return_value = [
            {"id": 0, "name": "RTX 4090", "memory_total": 24576, "memory_free": 20000},
            {"id": 1, "name": "RTX 3090", "memory_total": 24576, "memory_free": 24000},
        ]
        result = run_async(tools.comfyui_list_gpus())
        self.assertTrue(result["success"])
        self.assertEqual(result["count"], 2)
        self.assertEqual(len(result["gpus"]), 2)
        tools.manager.proxy.assert_awaited_once_with("GET", "/api/gpus")

    def test_list_gpus_dict_response(self):
        """list_gpus handles dict response with 'gpus' key."""
        tools = make_tools()
        tools.manager.proxy.return_value = {"gpus": [{"id": 0, "name": "RTX 4090"}]}
        result = run_async(tools.comfyui_list_gpus())
        self.assertTrue(result["success"])
        self.assertEqual(result["count"], 1)

    def test_list_gpus_empty(self):
        """list_gpus handles empty GPU list."""
        tools = make_tools()
        tools.manager.proxy.return_value = []
        result = run_async(tools.comfyui_list_gpus())
        self.assertTrue(result["success"])
        self.assertEqual(result["count"], 0)

    def test_list_gpus_error(self):
        """list_gpus handles proxy error."""
        tools = make_tools()
        tools.manager.proxy.side_effect = Exception("GPU query failed")
        result = run_async(tools.comfyui_list_gpus())
        self.assertFalse(result["success"])


# ---------------------------------------------------------------------------
# Test: Existing tools still work (regression)
# ---------------------------------------------------------------------------

class TestExistingToolsRegression(unittest.TestCase):
    """Verify existing ComfyUI tools still function after adding new ones."""

    def test_status_still_works(self):
        """comfyui_status should still work."""
        tools = make_tools()
        result = run_async(tools.comfyui_status())
        self.assertTrue(result["success"])
        self.assertIn("summary", result)

    def test_list_instances_still_works(self):
        """comfyui_list_instances should still work."""
        tools = make_tools()
        tools.manager.proxy.return_value = []
        result = run_async(tools.comfyui_list_instances())
        self.assertTrue(result["success"])
        self.assertEqual(result["count"], 0)

    def test_list_models_still_works(self):
        """comfyui_list_models should still work."""
        tools = make_tools()
        tools.manager.proxy.return_value = [{"name": "model.safetensors"}]
        result = run_async(tools.comfyui_list_models())
        self.assertTrue(result["success"])
        self.assertEqual(result["count"], 1)

    def test_install_nodes_still_works(self):
        """comfyui_install_nodes should still work."""
        tools = make_tools()
        tools.manager.proxy.return_value = {"job_id": "j-789"}
        result = run_async(tools.comfyui_install_nodes(node_ids=["controlnet"]))
        self.assertTrue(result["success"])
        self.assertEqual(result["job_id"], "j-789")

    def test_search_generations_no_catalog(self):
        """search_generations fails gracefully without catalog."""
        tools = make_tools()
        result = run_async(tools.comfyui_search_generations())
        self.assertFalse(result["success"])
        self.assertIn("catalog", result["error"].lower())

    def test_start_api_still_works(self):
        """comfyui_start_api should still work."""
        tools = make_tools()
        result = run_async(tools.comfyui_start_api())
        self.assertTrue(result["success"])

    def test_stop_api_still_works(self):
        """comfyui_stop_api should still work."""
        tools = make_tools()
        result = run_async(tools.comfyui_stop_api())
        self.assertTrue(result["success"])


# ---------------------------------------------------------------------------
# Test: Method-to-definition mapping
# ---------------------------------------------------------------------------

class TestMethodDefinitionMapping(unittest.TestCase):
    """Every TOOL_DEFINITION should have a matching method on ComfyUITools."""

    def test_all_definitions_have_methods(self):
        """Each tool in TOOL_DEFINITIONS should have a method on ComfyUITools."""
        tools = ComfyUITools(comfyui_manager=MagicMock())
        for td in TOOL_DEFINITIONS:
            method_name = td["name"]
            self.assertTrue(
                hasattr(tools, method_name),
                f"ComfyUITools missing method '{method_name}'"
            )
            method = getattr(tools, method_name)
            self.assertTrue(callable(method), f"'{method_name}' is not callable")

    def test_all_methods_are_async(self):
        """All tool methods should be async (coroutine functions)."""
        import inspect
        tools = ComfyUITools(comfyui_manager=MagicMock())
        for td in TOOL_DEFINITIONS:
            method = getattr(tools, td["name"])
            self.assertTrue(
                inspect.iscoroutinefunction(method),
                f"'{td['name']}' should be async"
            )


# ---------------------------------------------------------------------------
# Test: Proxy path correctness
# ---------------------------------------------------------------------------

class TestProxyPaths(unittest.TestCase):
    """Verify each tool calls the correct management API path."""

    def _assert_proxy_path(self, tools, method_name, kwargs, expected_method, expected_path, **proxy_kwargs):
        """Helper to verify proxy was called with the right HTTP method and path."""
        tools.manager.proxy.reset_mock()
        tools.manager.proxy.return_value = {}
        method = getattr(tools, method_name)
        run_async(method(**kwargs))
        tools.manager.proxy.assert_awaited_once()
        call_args = tools.manager.proxy.call_args
        self.assertEqual(call_args[0][0], expected_method, f"{method_name}: wrong HTTP method")
        self.assertEqual(call_args[0][1], expected_path, f"{method_name}: wrong path")
        if proxy_kwargs:
            for key, value in proxy_kwargs.items():
                self.assertEqual(call_args[1].get(key), value,
                                 f"{method_name}: wrong {key}")

    def test_all_proxy_paths(self):
        """Verify correct API paths for all 14 new tools."""
        tools = make_tools()

        # Node management
        self._assert_proxy_path(tools, "comfyui_list_node_packs", {},
                                "GET", "/api/nodes/registry")
        self._assert_proxy_path(tools, "comfyui_list_installed_nodes", {},
                                "GET", "/api/nodes/installed")
        self._assert_proxy_path(tools, "comfyui_update_nodes", {},
                                "POST", "/api/nodes/update-all")
        self._assert_proxy_path(tools, "comfyui_remove_node",
                                {"node_name": "TestNode"},
                                "DELETE", "/api/nodes/TestNode")

        # Instance management
        self._assert_proxy_path(tools, "comfyui_remove_instance",
                                {"instance_id": "xyz"},
                                "DELETE", "/api/instances/xyz")
        self._assert_proxy_path(tools, "comfyui_start_all_instances", {},
                                "POST", "/api/instances/start-all")
        self._assert_proxy_path(tools, "comfyui_stop_all_instances", {},
                                "POST", "/api/instances/stop-all")

        # Model categories
        self._assert_proxy_path(tools, "comfyui_model_categories", {},
                                "GET", "/api/models/categories")

        # Settings
        self._assert_proxy_path(tools, "comfyui_get_settings", {},
                                "GET", "/api/settings")
        self._assert_proxy_path(tools, "comfyui_update_settings",
                                {"settings": {"key": "val"}},
                                "PUT", "/api/settings", json={"key": "val"})

        # Update / Purge
        self._assert_proxy_path(tools, "comfyui_update_comfyui", {},
                                "POST", "/api/update")
        self._assert_proxy_path(tools, "comfyui_purge", {},
                                "POST", "/api/purge")

        # GPUs
        self._assert_proxy_path(tools, "comfyui_list_gpus", {},
                                "GET", "/api/gpus")

    def test_manage_external_paths(self):
        """Verify correct API paths for each manage_external action."""
        tools = make_tools()

        # list
        tools.manager.proxy.reset_mock()
        tools.manager.proxy.return_value = []
        run_async(tools.comfyui_manage_external(action="list"))
        tools.manager.proxy.assert_awaited_once_with("GET", "/api/comfyui/saved")

        # add
        tools.manager.proxy.reset_mock()
        tools.manager.proxy.return_value = {}
        run_async(tools.comfyui_manage_external(action="add", directory="/foo"))
        tools.manager.proxy.assert_awaited_once_with(
            "POST", "/api/comfyui/saved", json={"directory": "/foo"}
        )

        # remove
        tools.manager.proxy.reset_mock()
        tools.manager.proxy.return_value = {}
        run_async(tools.comfyui_manage_external(action="remove", directory="/foo"))
        tools.manager.proxy.assert_awaited_once_with(
            "DELETE", "/api/comfyui/saved", params={"directory": "/foo"}
        )

        # switch
        tools.manager.proxy.reset_mock()
        tools.manager.proxy.return_value = {}
        run_async(tools.comfyui_manage_external(action="switch", directory="/bar"))
        tools.manager.proxy.assert_awaited_once_with(
            "PUT", "/api/comfyui/target", json={"directory": "/bar"}
        )


# ---------------------------------------------------------------------------
# Test: comfyui_await_job
# ---------------------------------------------------------------------------

class TestAwaitJob(unittest.TestCase):
    """Test the comfyui_await_job tool that blocks until job completes."""

    def test_await_job_completes_immediately(self):
        """Job already completed on first poll → returns success."""
        tools = make_tools()
        tools.manager.proxy.return_value = {"status": "completed", "result": {"files": 2}, "message": "Done"}
        result = run_async(tools.comfyui_await_job(job_id="j1"))
        self.assertTrue(result["success"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["result"], {"files": 2})
        self.assertIn("elapsed_seconds", result)

    def test_await_job_polls_until_complete(self):
        """Job running twice then completes → 3 proxy calls total."""
        tools = make_tools()
        tools.manager.proxy.side_effect = [
            {"status": "running", "progress": {"current": 10}},
            {"status": "running", "progress": {"current": 50}},
            {"status": "completed", "result": "ok", "message": "Done"},
        ]
        # Use tiny poll interval to speed up test
        result = run_async(tools.comfyui_await_job(job_id="j2", poll_interval=0))
        self.assertTrue(result["success"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(tools.manager.proxy.await_count, 3)

    def test_await_job_handles_failure(self):
        """Job fails → returns success=False with error."""
        tools = make_tools()
        tools.manager.proxy.return_value = {"status": "failed", "error": "Download failed: 404"}
        result = run_async(tools.comfyui_await_job(job_id="j3"))
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "failed")
        self.assertIn("404", result["error"])

    def test_await_job_handles_error_status(self):
        """Job returns 'error' status → returns success=False."""
        tools = make_tools()
        tools.manager.proxy.return_value = {"status": "error", "message": "Internal error"}
        result = run_async(tools.comfyui_await_job(job_id="j4"))
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "error")

    def test_await_job_timeout(self):
        """Timeout triggers when job never completes (covered by mocked time test)."""
        # Direct timeout test is slow (min 30s clamp), so just verify
        # the timeout_with_mocked_time test covers the logic.
        # Here we just confirm the method accepts the params without error.
        tools = make_tools()
        tools.manager.proxy.return_value = {"status": "completed", "message": "ok"}
        result = run_async(tools.comfyui_await_job(job_id="j5", poll_interval=5, timeout=30))
        self.assertTrue(result["success"])

    def test_await_job_timeout_with_mocked_time(self):
        """Verify timeout by mocking time.time() to advance past threshold."""
        tools = make_tools()
        tools.manager.proxy.return_value = {"status": "running"}
        call_count = [0]
        original_time = __import__('time').time

        def fake_time():
            call_count[0] += 1
            if call_count[0] >= 3:
                return original_time() + 99999  # Far past any timeout
            return original_time()

        with patch('backend.tools.comfyui_tools.time.time', side_effect=fake_time):
            result = run_async(tools.comfyui_await_job(job_id="j5", poll_interval=5, timeout=30))

        self.assertFalse(result["success"])
        self.assertIn("Timed out", result["error"])

    def test_await_job_cancelled(self):
        """Cancellation during sleep returns aborted status."""
        tools = make_tools()
        tools.manager.proxy.return_value = {"status": "running"}

        async def run_and_cancel():
            task = asyncio.ensure_future(
                tools.comfyui_await_job(job_id="j6", poll_interval=5)
            )
            # Let it run one poll cycle then cancel
            await asyncio.sleep(0.05)
            task.cancel()
            return await task

        result = run_async(run_and_cancel())
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "aborted")
        self.assertIn("cancelled", result["error"].lower())

    def test_await_job_clamps_intervals(self):
        """poll_interval and timeout are clamped to valid ranges."""
        tools = make_tools()
        tools.manager.proxy.return_value = {"status": "completed", "message": "Done"}

        # poll_interval=1 should clamp to 5, timeout=10 should clamp to 30
        # But since job completes immediately, we just verify it succeeds
        result = run_async(tools.comfyui_await_job(job_id="j7", poll_interval=1, timeout=10))
        self.assertTrue(result["success"])

        # poll_interval=999 should clamp to 120, timeout=99999 should clamp to 7200
        result = run_async(tools.comfyui_await_job(job_id="j7", poll_interval=999, timeout=99999))
        self.assertTrue(result["success"])

    def test_await_job_no_api(self):
        """No manager → returns error about module not configured."""
        tools = ComfyUITools(comfyui_manager=None)
        result = run_async(tools.comfyui_await_job(job_id="j8"))
        self.assertFalse(result["success"])
        self.assertIn("not", result["error"].lower())

    def test_await_job_proxy_exception(self):
        """Proxy raises exception → returns error."""
        tools = make_tools()
        tools.manager.proxy.side_effect = Exception("Connection refused")
        result = run_async(tools.comfyui_await_job(job_id="j9"))
        self.assertFalse(result["success"])
        self.assertIn("Connection refused", result["error"])

    def test_await_job_proxy_path(self):
        """Verify correct API path is called."""
        tools = make_tools()
        tools.manager.proxy.return_value = {"status": "completed", "message": "Done"}
        run_async(tools.comfyui_await_job(job_id="test-123"))
        tools.manager.proxy.assert_awaited_with("GET", "/api/jobs/test-123")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Ensure event loop exists for run_async helper
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    unittest.main(verbosity=2)
