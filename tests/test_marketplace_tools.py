"""
Tests for marketplace agent tools.

Tests all 4 marketplace tools with mocked dependencies — no running server needed.
"""

import sys
import os
import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.tools.marketplace_tools import MarketplaceTools, TOOL_DEFINITIONS


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
    """Create MarketplaceTools with mock dependencies."""
    if manager is None:
        manager = make_mock_manager()
    orchestrator = MagicMock()
    return MarketplaceTools(orchestrator, manager)


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_SEARCH_RESULTS = {
    "results": [
        {
            "id": "1234",
            "name": "Slack RSS Feed",
            "description": "Post RSS items to Slack channel automatically",
            "category": "Communication",
            "complexity": "low",
            "trigger_type": "schedule",
            "node_count": 4,
            "integrations": ["Slack", "RSS"],
            "totalViews": 5000,
        },
        {
            "id": "5678",
            "name": "Discord Bot Webhook",
            "description": "A" * 250,  # long description for truncation test
            "category": "Communication",
            "complexity": "medium",
            "trigger_type": "webhook",
            "node_count": 6,
            "integrations": ["Discord"],
            "totalViews": 3200,
        },
    ]
}

SAMPLE_WORKFLOW_RESPONSE = {
    "workflow": {
        "json": {
            "name": "Test Workflow",
            "nodes": [
                {
                    "name": "Webhook",
                    "type": "n8n-nodes-base.webhook",
                    "parameters": {"path": "test-hook"},
                    "credentials": {},
                },
                {
                    "name": "Send Slack",
                    "type": "n8n-nodes-base.slack",
                    "parameters": {
                        "channel": {"__rl": True, "mode": "name", "value": ""},
                        "text": "YOUR_MESSAGE_HERE",
                    },
                    "credentials": {
                        "slackApi": {"id": "", "name": ""},
                    },
                },
                {
                    "name": "HTTP Request",
                    "type": "n8n-nodes-base.httpRequest",
                    "parameters": {
                        "url": "https://example.com/api",
                        "method": "POST",
                    },
                    "credentials": {
                        "httpHeaderAuth": {"id": "42", "name": "My Auth"},
                    },
                },
            ],
            "connections": {},
        },
        "metadata": {
            "id": "1234",
            "name": "Test Workflow",
            "description": "A test workflow",
            "category": "Communication",
            "complexity": "low",
            "trigger_type": "webhook",
            "node_count": 3,
            "integrations": ["Slack"],
        },
    }
}

SAMPLE_WORKFLOW_JSON = SAMPLE_WORKFLOW_RESPONSE["workflow"]["json"]


# ---------------------------------------------------------------------------
# Test: TOOL_DEFINITIONS
# ---------------------------------------------------------------------------

class TestToolDefinitions(unittest.TestCase):
    """Verify TOOL_DEFINITIONS structure."""

    def test_total_count(self):
        self.assertEqual(len(TOOL_DEFINITIONS), 4)

    def test_all_names_unique(self):
        names = [td["name"] for td in TOOL_DEFINITIONS]
        self.assertEqual(len(names), len(set(names)))

    def test_expected_tools_present(self):
        names = {td["name"] for td in TOOL_DEFINITIONS}
        expected = {"search_marketplace", "get_marketplace_workflow",
                    "inspect_workflow", "configure_workflow"}
        self.assertEqual(names, expected)

    def test_all_have_required_fields(self):
        for td in TOOL_DEFINITIONS:
            self.assertIn("name", td)
            self.assertIn("description", td)
            self.assertIn("parameters", td)
            self.assertIn("type", td["parameters"])
            self.assertEqual(td["parameters"]["type"], "object")


# ---------------------------------------------------------------------------
# Test: search_marketplace
# ---------------------------------------------------------------------------

class TestSearchMarketplace(unittest.TestCase):
    """Tests for search_marketplace tool."""

    @patch("aiohttp.ClientSession")
    def test_search_success(self, mock_session_cls):
        tools = make_tools()

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=SAMPLE_SEARCH_RESULTS)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = mock_session

        result = run_async(tools.search_marketplace("slack rss"))

        self.assertTrue(result["success"])
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["query"], "slack rss")
        self.assertEqual(result["workflows"][0]["id"], "1234")
        self.assertEqual(result["workflows"][0]["name"], "Slack RSS Feed")
        self.assertIn("hint", result)

    @patch("aiohttp.ClientSession")
    def test_search_truncates_long_description(self, mock_session_cls):
        tools = make_tools()

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=SAMPLE_SEARCH_RESULTS)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = mock_session

        result = run_async(tools.search_marketplace("discord"))

        # Second result has 250-char description, should be truncated to 200
        desc = result["workflows"][1]["description"]
        self.assertLessEqual(len(desc), 200)
        self.assertTrue(desc.endswith("..."))

    @patch("aiohttp.ClientSession")
    def test_search_api_error(self, mock_session_cls):
        tools = make_tools()

        mock_resp = AsyncMock()
        mock_resp.status = 500
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = mock_session

        result = run_async(tools.search_marketplace("test"))

        self.assertFalse(result["success"])
        self.assertIn("500", result["error"])

    @patch("aiohttp.ClientSession")
    def test_search_empty_results(self, mock_session_cls):
        tools = make_tools()

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"results": []})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = mock_session

        result = run_async(tools.search_marketplace("nonexistent"))

        self.assertTrue(result["success"])
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["workflows"], [])

    @patch("aiohttp.ClientSession")
    def test_search_connection_error(self, mock_session_cls):
        tools = make_tools()
        mock_session_cls.side_effect = Exception("Connection refused")

        result = run_async(tools.search_marketplace("test"))

        self.assertFalse(result["success"])
        self.assertIn("Connection refused", result["error"])


# ---------------------------------------------------------------------------
# Test: get_marketplace_workflow
# ---------------------------------------------------------------------------

class TestGetMarketplaceWorkflow(unittest.TestCase):
    """Tests for get_marketplace_workflow tool."""

    @patch("aiohttp.ClientSession")
    def test_get_success(self, mock_session_cls):
        tools = make_tools()

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=SAMPLE_WORKFLOW_RESPONSE)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = mock_session

        result = run_async(tools.get_marketplace_workflow("1234"))

        self.assertTrue(result["success"])
        self.assertIn("workflow_json", result)
        self.assertIn("metadata", result)
        self.assertEqual(result["metadata"]["id"], "1234")
        self.assertEqual(len(result["workflow_json"]["nodes"]), 3)
        self.assertIn("hint", result)

    @patch("aiohttp.ClientSession")
    def test_get_not_found(self, mock_session_cls):
        tools = make_tools()

        mock_resp = AsyncMock()
        mock_resp.status = 404
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = mock_session

        result = run_async(tools.get_marketplace_workflow("9999"))

        self.assertFalse(result["success"])
        self.assertIn("not found", result["error"])

    @patch("aiohttp.ClientSession")
    def test_get_empty_nodes(self, mock_session_cls):
        tools = make_tools()

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "workflow": {"json": {"nodes": []}, "metadata": {}}
        })
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = mock_session

        result = run_async(tools.get_marketplace_workflow("empty"))

        self.assertFalse(result["success"])
        self.assertIn("no nodes", result["error"].lower())


# ---------------------------------------------------------------------------
# Test: inspect_workflow
# ---------------------------------------------------------------------------

class TestInspectWorkflow(unittest.TestCase):
    """Tests for inspect_workflow tool."""

    def test_inspect_identifies_missing_credentials(self):
        tools = make_tools()
        wf = deepcopy(SAMPLE_WORKFLOW_JSON)

        result = run_async(tools.inspect_workflow(wf))

        self.assertTrue(result["success"])
        creds = result["credentials"]
        # slackApi has empty id → should be "missing"
        slack_creds = [c for c in creds if c["credential_type"] == "slackApi"]
        self.assertEqual(len(slack_creds), 1)
        self.assertEqual(slack_creds[0]["status"], "missing")
        self.assertEqual(slack_creds[0]["node_name"], "Send Slack")

    def test_inspect_identifies_preconfigured_credentials(self):
        tools = make_tools()
        wf = deepcopy(SAMPLE_WORKFLOW_JSON)

        result = run_async(tools.inspect_workflow(wf))

        # httpHeaderAuth has id="42" → should be "pre-configured"
        http_creds = [c for c in result["credentials"]
                      if c["credential_type"] == "httpHeaderAuth"]
        self.assertEqual(len(http_creds), 1)
        self.assertEqual(http_creds[0]["status"], "pre-configured")

    def test_inspect_finds_empty_rl_placeholder(self):
        tools = make_tools()
        wf = deepcopy(SAMPLE_WORKFLOW_JSON)

        result = run_async(tools.inspect_workflow(wf))

        placeholders = result["placeholders"]
        rl_placeholders = [p for p in placeholders if p["placeholder_type"] == "empty_rl"]
        self.assertGreaterEqual(len(rl_placeholders), 1)
        # The Slack channel has __rl with empty value
        slack_rl = [p for p in rl_placeholders if p["node_name"] == "Send Slack"]
        self.assertEqual(len(slack_rl), 1)

    def test_inspect_finds_placeholder_text(self):
        tools = make_tools()
        wf = deepcopy(SAMPLE_WORKFLOW_JSON)

        result = run_async(tools.inspect_workflow(wf))

        placeholders = result["placeholders"]
        text_placeholders = [p for p in placeholders
                             if p["placeholder_type"] == "placeholder_text"]
        # "YOUR_MESSAGE_HERE" matches YOUR_ pattern
        your_placeholders = [p for p in text_placeholders
                             if "YOUR_" in p.get("current_value", "")]
        self.assertGreaterEqual(len(your_placeholders), 1)

    def test_inspect_finds_example_com_placeholder(self):
        tools = make_tools()
        wf = deepcopy(SAMPLE_WORKFLOW_JSON)

        result = run_async(tools.inspect_workflow(wf))

        placeholders = result["placeholders"]
        example_placeholders = [p for p in placeholders
                                if "example.com" in p.get("current_value", "")]
        self.assertGreaterEqual(len(example_placeholders), 1)

    def test_inspect_summary_fields(self):
        tools = make_tools()
        wf = deepcopy(SAMPLE_WORKFLOW_JSON)

        result = run_async(tools.inspect_workflow(wf))

        summary = result["summary"]
        self.assertEqual(summary["name"], "Test Workflow")
        self.assertEqual(summary["node_count"], 3)
        self.assertEqual(summary["trigger_type"], "webhook")
        self.assertIn("slackApi", summary["credential_types_needed"])
        self.assertGreaterEqual(summary["credentials_missing"], 1)
        self.assertGreaterEqual(summary["placeholders_found"], 1)
        self.assertFalse(summary["ready_to_deploy"])  # has missing creds + placeholders

    def test_inspect_ready_to_deploy_when_all_configured(self):
        tools = make_tools()
        # Workflow with no missing creds and no placeholders
        wf = {
            "name": "Clean Workflow",
            "nodes": [
                {
                    "name": "Webhook",
                    "type": "n8n-nodes-base.webhook",
                    "parameters": {"path": "clean"},
                    "credentials": {},
                },
                {
                    "name": "Code",
                    "type": "n8n-nodes-base.code",
                    "parameters": {"jsCode": "return [{json: {ok: true}}]"},
                    "credentials": {},
                },
            ],
        }

        result = run_async(tools.inspect_workflow(wf))

        self.assertTrue(result["success"])
        self.assertTrue(result["summary"]["ready_to_deploy"])

    def test_inspect_empty_workflow(self):
        tools = make_tools()

        result = run_async(tools.inspect_workflow({"nodes": []}))

        self.assertFalse(result["success"])
        self.assertIn("no nodes", result["error"].lower())

    def test_inspect_detects_schedule_trigger(self):
        tools = make_tools()
        wf = {
            "name": "Scheduled",
            "nodes": [
                {
                    "name": "Schedule",
                    "type": "n8n-nodes-base.scheduleTrigger",
                    "parameters": {"rule": {"interval": [{"field": "hours", "hoursInterval": 1}]}},
                    "credentials": {},
                },
            ],
        }

        result = run_async(tools.inspect_workflow(wf))

        self.assertTrue(result["success"])
        self.assertEqual(result["summary"]["trigger_type"], "schedule")

    def test_inspect_recursive_param_scanning(self):
        """Test that nested params and arrays are scanned for placeholders."""
        tools = make_tools()
        wf = {
            "name": "Nested",
            "nodes": [
                {
                    "name": "Node1",
                    "type": "n8n-nodes-base.code",
                    "parameters": {
                        "nested": {
                            "deep": {
                                "value": "REPLACE_THIS"
                            }
                        },
                        "list": [
                            {"item": "TODO fix this"}
                        ],
                    },
                    "credentials": {},
                },
            ],
        }

        result = run_async(tools.inspect_workflow(wf))

        placeholders = result["placeholders"]
        self.assertGreaterEqual(len(placeholders), 2)
        paths = [p["param_path"] for p in placeholders]
        self.assertTrue(any("nested.deep.value" in p for p in paths))
        self.assertTrue(any("list[0].item" in p for p in paths))


# ---------------------------------------------------------------------------
# Test: configure_workflow
# ---------------------------------------------------------------------------

class TestConfigureWorkflow(unittest.TestCase):
    """Tests for configure_workflow tool."""

    def test_configure_fills_credential_ids(self):
        tools = make_tools()
        wf = deepcopy(SAMPLE_WORKFLOW_JSON)

        result = run_async(tools.configure_workflow(
            wf,
            credential_map={"slackApi": "99"},
        ))

        self.assertTrue(result["success"])
        # Verify the slackApi credential ID was set
        configured_wf = result["workflow_json"]
        slack_node = [n for n in configured_wf["nodes"]
                      if n["name"] == "Send Slack"][0]
        self.assertEqual(slack_node["credentials"]["slackApi"]["id"], "99")
        self.assertGreaterEqual(result["changes_count"], 1)

    def test_configure_fills_param_overrides(self):
        tools = make_tools()
        wf = deepcopy(SAMPLE_WORKFLOW_JSON)

        result = run_async(tools.configure_workflow(
            wf,
            param_overrides={
                "HTTP Request": {"url": "https://myapi.com/endpoint"},
            },
        ))

        self.assertTrue(result["success"])
        configured_wf = result["workflow_json"]
        http_node = [n for n in configured_wf["nodes"]
                     if n["name"] == "HTTP Request"][0]
        self.assertEqual(http_node["parameters"]["url"], "https://myapi.com/endpoint")

    def test_configure_handles_rl_objects(self):
        tools = make_tools()
        wf = deepcopy(SAMPLE_WORKFLOW_JSON)

        result = run_async(tools.configure_workflow(
            wf,
            param_overrides={
                "Send Slack": {"channel": "#alerts"},
            },
        ))

        self.assertTrue(result["success"])
        configured_wf = result["workflow_json"]
        slack_node = [n for n in configured_wf["nodes"]
                      if n["name"] == "Send Slack"][0]
        # Should update the __rl value field, not replace the whole object
        channel = slack_node["parameters"]["channel"]
        self.assertTrue(channel.get("__rl"))
        self.assertEqual(channel["value"], "#alerts")

    def test_configure_does_not_mutate_original(self):
        tools = make_tools()
        wf = deepcopy(SAMPLE_WORKFLOW_JSON)
        original_slack_id = wf["nodes"][1]["credentials"]["slackApi"]["id"]

        run_async(tools.configure_workflow(
            wf,
            credential_map={"slackApi": "999"},
        ))

        # Original should be unchanged
        self.assertEqual(wf["nodes"][1]["credentials"]["slackApi"]["id"], original_slack_id)

    def test_configure_empty_workflow_fails(self):
        tools = make_tools()

        result = run_async(tools.configure_workflow({"nodes": []}))

        self.assertFalse(result["success"])
        self.assertIn("no nodes", result["error"].lower())

    def test_configure_tracks_all_changes(self):
        tools = make_tools()
        wf = deepcopy(SAMPLE_WORKFLOW_JSON)

        result = run_async(tools.configure_workflow(
            wf,
            credential_map={"slackApi": "10", "httpHeaderAuth": "20"},
            param_overrides={
                "HTTP Request": {"url": "https://new.com"},
                "Send Slack": {"text": "Hello World"},
            },
        ))

        self.assertTrue(result["success"])
        changes = result["changes_made"]
        # Should have: 2 credential changes + 2 param changes = 4
        self.assertGreaterEqual(len(changes), 4)
        change_types = [c["change_type"] for c in changes]
        self.assertIn("credential", change_types)
        self.assertIn("param", change_types)

    def test_configure_no_changes(self):
        tools = make_tools()
        wf = deepcopy(SAMPLE_WORKFLOW_JSON)

        result = run_async(tools.configure_workflow(wf))

        self.assertTrue(result["success"])
        self.assertEqual(result["changes_count"], 0)

    def test_configure_nonexistent_node_override_ignored(self):
        tools = make_tools()
        wf = deepcopy(SAMPLE_WORKFLOW_JSON)

        result = run_async(tools.configure_workflow(
            wf,
            param_overrides={"Nonexistent Node": {"key": "value"}},
        ))

        self.assertTrue(result["success"])
        self.assertEqual(result["changes_count"], 0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
