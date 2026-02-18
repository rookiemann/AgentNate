"""
Tests for flash_workflow in workflow_tools.py.

Tests the deploy → activate → trigger → collect → delete lifecycle,
error paths, and cleanup (finally block).
"""

import sys
import os
import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.tools.workflow_tools import WorkflowTools


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
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


def make_mock_manager():
    """Create a mock n8n manager."""
    manager = MagicMock()
    manager.instances = {5678: MagicMock()}
    return manager


def make_tools(manager=None):
    """Create WorkflowTools with mock dependencies."""
    if manager is None:
        manager = make_mock_manager()
    orchestrator = MagicMock()
    return WorkflowTools(orchestrator, manager)


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_WEBHOOK_WORKFLOW = {
    "name": "Flash Test",
    "nodes": [
        {
            "name": "Webhook",
            "type": "n8n-nodes-base.webhook",
            "parameters": {"path": "flash-test", "httpMethod": "POST",
                           "responseMode": "responseNode"},
        },
        {
            "name": "Code",
            "type": "n8n-nodes-base.code",
            "parameters": {"jsCode": "return [{json: {ok: true}}]"},
        },
        {
            "name": "Respond",
            "type": "n8n-nodes-base.respondToWebhook",
            "parameters": {"respondWith": "json"},
        },
    ],
    "connections": {},
    "meta": {"webhook_path": "flash-test"},
}

SAMPLE_NO_WEBHOOK_WORKFLOW = {
    "name": "No Webhook",
    "nodes": [
        {
            "name": "Manual",
            "type": "n8n-nodes-base.manualTrigger",
            "parameters": {},
        },
    ],
    "connections": {},
}


# ---------------------------------------------------------------------------
# Test: flash_workflow
# ---------------------------------------------------------------------------

class TestFlashWorkflow(unittest.TestCase):
    """Tests for flash_workflow tool."""

    def test_n8n_not_running(self):
        manager = MagicMock()
        manager.instances = {}  # no instances
        tools = make_tools(manager)

        result = run_async(tools.flash_workflow(SAMPLE_WEBHOOK_WORKFLOW))

        self.assertFalse(result["success"])
        self.assertIn("not running", result["error"].lower())

    def test_no_webhook_trigger_fails(self):
        tools = make_tools()

        result = run_async(tools.flash_workflow(SAMPLE_NO_WEBHOOK_WORKFLOW))

        self.assertFalse(result["success"])
        self.assertIn("webhook", result["error"].lower())

    def test_webhook_path_from_node(self):
        """Verify webhook path is extracted from node parameters."""
        tools = make_tools()

        # Mock deploy and trigger to capture the webhook path used
        webhook_path_used = []

        async def mock_deploy(wf_json, n8n_port=5678, activate=False):
            return {"success": True, "workflow_id": "wf-123"}

        async def mock_trigger(path, data, method, test_mode, port):
            webhook_path_used.append(path)
            return {"success": True, "response": {"ok": True}}

        async def mock_delete(wf_id, port=5678):
            return {"success": True}

        tools.deploy_workflow = mock_deploy
        tools.trigger_webhook = mock_trigger
        tools.delete_workflow = mock_delete

        result = run_async(tools.flash_workflow(SAMPLE_WEBHOOK_WORKFLOW))

        self.assertTrue(result["success"])
        self.assertEqual(webhook_path_used[0], "flash-test")

    def test_webhook_path_from_meta_fallback(self):
        """If no webhook node has path, fall back to meta.webhook_path."""
        tools = make_tools()

        wf = {
            "name": "Meta Path",
            "nodes": [
                {
                    "name": "Webhook",
                    "type": "n8n-nodes-base.webhook",
                    "parameters": {"httpMethod": "POST"},  # no "path" key
                },
            ],
            "connections": {},
            "meta": {"webhook_path": "meta-path"},
        }

        async def mock_deploy(wf_json, n8n_port=5678, activate=False):
            return {"success": True, "workflow_id": "wf-meta"}

        async def mock_trigger(path, data, method, test_mode, port):
            return {"success": True, "response": {"result": "from meta"}}

        async def mock_delete(wf_id, port=5678):
            return {"success": True}

        tools.deploy_workflow = mock_deploy
        tools.trigger_webhook = mock_trigger
        tools.delete_workflow = mock_delete

        result = run_async(tools.flash_workflow(wf))

        self.assertTrue(result["success"])

    def test_deploy_failure_returns_error(self):
        tools = make_tools()

        async def mock_deploy(wf_json, n8n_port=5678, activate=False):
            return {"success": False, "error": "n8n API error 500"}

        tools.deploy_workflow = mock_deploy

        result = run_async(tools.flash_workflow(SAMPLE_WEBHOOK_WORKFLOW))

        self.assertFalse(result["success"])
        self.assertIn("Deploy failed", result["error"])

    def test_direct_webhook_response(self):
        """Test the happy path: webhook returns direct response."""
        tools = make_tools()

        async def mock_deploy(wf_json, n8n_port=5678, activate=False):
            return {"success": True, "workflow_id": "wf-direct"}

        async def mock_trigger(path, data, method, test_mode, port):
            return {"success": True, "response": {"answer": 42}}

        async def mock_delete(wf_id, port=5678):
            return {"success": True}

        tools.deploy_workflow = mock_deploy
        tools.trigger_webhook = mock_trigger
        tools.delete_workflow = mock_delete

        result = run_async(tools.flash_workflow(SAMPLE_WEBHOOK_WORKFLOW))

        self.assertTrue(result["success"])
        self.assertEqual(result["result"], {"answer": 42})
        self.assertEqual(result["mode"], "webhook_response")
        self.assertEqual(result["workflow_id"], "wf-direct")

    def test_cleanup_always_runs(self):
        """Verify delete_workflow is called even on error."""
        tools = make_tools()
        delete_called = []

        async def mock_deploy(wf_json, n8n_port=5678, activate=False):
            return {"success": True, "workflow_id": "wf-cleanup"}

        async def mock_trigger(path, data, method, test_mode, port):
            raise RuntimeError("trigger exploded")

        async def mock_delete(wf_id, port=5678):
            delete_called.append(wf_id)
            return {"success": True}

        tools.deploy_workflow = mock_deploy
        tools.trigger_webhook = mock_trigger
        tools.delete_workflow = mock_delete

        result = run_async(tools.flash_workflow(SAMPLE_WEBHOOK_WORKFLOW))

        # Should fail (trigger raised), but cleanup should still run
        self.assertFalse(result["success"])
        self.assertEqual(delete_called, ["wf-cleanup"])

    def test_cleanup_not_called_if_no_deploy(self):
        """If deploy fails (no workflow_id), cleanup shouldn't try to delete."""
        tools = make_tools()
        delete_called = []

        async def mock_deploy(wf_json, n8n_port=5678, activate=False):
            return {"success": False, "error": "deploy failed"}

        async def mock_delete(wf_id, port=5678):
            delete_called.append(wf_id)
            return {"success": True}

        tools.deploy_workflow = mock_deploy
        tools.delete_workflow = mock_delete

        result = run_async(tools.flash_workflow(SAMPLE_WEBHOOK_WORKFLOW))

        self.assertFalse(result["success"])
        self.assertEqual(delete_called, [])

    def test_webhook_data_passed_through(self):
        """Verify webhook_data is passed to trigger_webhook."""
        tools = make_tools()
        trigger_data_received = []

        async def mock_deploy(wf_json, n8n_port=5678, activate=False):
            return {"success": True, "workflow_id": "wf-data"}

        async def mock_trigger(path, data, method, test_mode, port):
            trigger_data_received.append(data)
            return {"success": True, "response": {"ok": True}}

        async def mock_delete(wf_id, port=5678):
            return {"success": True}

        tools.deploy_workflow = mock_deploy
        tools.trigger_webhook = mock_trigger
        tools.delete_workflow = mock_delete

        run_async(tools.flash_workflow(
            SAMPLE_WEBHOOK_WORKFLOW,
            webhook_data={"task": "write hello world"},
        ))

        self.assertEqual(trigger_data_received[0], {"task": "write hello world"})

    def test_webhook_404_returns_not_found_error(self):
        """When webhook returns 404, flash should return clear error."""
        tools = make_tools()

        async def mock_deploy(wf_json, n8n_port=5678, activate=False):
            return {"success": True, "workflow_id": "wf-404"}

        async def mock_trigger(path, data, method, test_mode, port):
            return {
                "success": False,
                "status_code": 404,
                "error": "Webhook 'flash-test' not found.",
            }

        async def mock_delete(wf_id, port=5678):
            return {"success": True}

        tools.deploy_workflow = mock_deploy
        tools.trigger_webhook = mock_trigger
        tools.delete_workflow = mock_delete

        result = run_async(tools.flash_workflow(SAMPLE_WEBHOOK_WORKFLOW))

        self.assertFalse(result["success"])
        self.assertIn("not found", result["error"].lower())

    def test_string_json_input(self):
        """flash_workflow should accept string JSON input."""
        tools = make_tools()
        import json

        async def mock_deploy(wf_json, n8n_port=5678, activate=False):
            return {"success": True, "workflow_id": "wf-str"}

        async def mock_trigger(path, data, method, test_mode, port):
            return {"success": True, "response": {"ok": True}}

        async def mock_delete(wf_id, port=5678):
            return {"success": True}

        tools.deploy_workflow = mock_deploy
        tools.trigger_webhook = mock_trigger
        tools.delete_workflow = mock_delete

        result = run_async(tools.flash_workflow(
            json.dumps(SAMPLE_WEBHOOK_WORKFLOW)
        ))

        self.assertTrue(result["success"])

    def test_invalid_json_string_fails(self):
        tools = make_tools()

        result = run_async(tools.flash_workflow("not json {{{"))

        self.assertFalse(result["success"])
        self.assertIn("Invalid JSON", result["error"])

    def test_execution_poll_fallback(self):
        """When webhook doesn't return direct result, should poll executions."""
        tools = make_tools()
        poll_count = [0]

        async def mock_deploy(wf_json, n8n_port=5678, activate=False):
            return {"success": True, "workflow_id": "wf-poll"}

        async def mock_trigger(path, data, method, test_mode, port):
            return {
                "success": False,
                "status_code": 200,
                "error": "No response body",
            }

        async def mock_list_executions(workflow_id=None, limit=1, n8n_port=5678):
            poll_count[0] += 1
            if poll_count[0] < 2:
                return {"success": True, "executions": []}
            return {
                "success": True,
                "executions": [{
                    "id": "exec-123",
                    "finished": True,
                    "status": "success",
                }],
            }

        async def mock_get_execution(exec_id, n8n_port=5678):
            return {
                "success": True,
                "node_results": {"Code": {"output": [{"json": {"ok": True}}]}},
            }

        async def mock_delete(wf_id, port=5678):
            return {"success": True}

        tools.deploy_workflow = mock_deploy
        tools.trigger_webhook = mock_trigger
        tools.list_executions = mock_list_executions
        tools.get_execution_result = mock_get_execution
        tools.delete_workflow = mock_delete

        result = run_async(tools.flash_workflow(
            SAMPLE_WEBHOOK_WORKFLOW, timeout=30
        ))

        self.assertTrue(result["success"])
        self.assertEqual(result["mode"], "execution_poll")
        self.assertEqual(result["execution_id"], "exec-123")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
