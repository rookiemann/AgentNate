import os
import sys
import unittest
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.tools.agent_tools import AgentTools, SubAgentState
from backend.tools.tool_router import ToolRouter


class MockOrchestrator:
    def get_loaded_instances(self):
        return []

    def get_instance(self, _instance_id):
        return None

    async def get_all_providers_health(self):
        return {}


class MockN8n:
    instances = {}
    main = None

    async def get_main_info(self):
        return None


class MockSettings:
    tools = {}

    def get(self, _key, default=None):
        return default


class TestToolRouterNormalization(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.router = ToolRouter(MockOrchestrator(), MockN8n(), MockSettings())
        self.router.set_agent_context(
            instance_id="inst-parent",
            conversation_store=object(),
            persona_manager=object(),
            parent_conv_id="conv-parent-1",
        )

    async def test_spawn_agent_normalizes_and_filters_args(self):
        async def fake_spawn(task, persona_id="general_assistant", max_tool_calls=8):
            return {
                "task": task,
                "persona_id": persona_id,
                "max_tool_calls": max_tool_calls,
            }

        self.router._routes["spawn_agent"] = fake_spawn

        result = await self.router.execute(
            "spawn_agent",
            {
                "task": "hello",
                "persona_id": None,
                "max_tool_calls": False,
                "junk": "drop-me",
            },
        )

        self.assertEqual(result["task"], "hello")
        self.assertEqual(result["persona_id"], "general_assistant")
        self.assertEqual(result["max_tool_calls"], 8)

    async def test_check_agents_wrapper_ignores_unknown_kwargs(self):
        async def fake_check_agents(_parent_conv_id=None):
            return {"success": True, "parent": _parent_conv_id}

        self.router.agent_tools.check_agents = fake_check_agents

        result = await self.router.execute("check_agents", {"unexpected": 1})
        self.assertTrue(result["success"])
        self.assertEqual(result["parent"], "conv-parent-1")

    async def test_get_agent_result_wrapper_scopes_parent(self):
        async def fake_get_agent_result(agent_id, _parent_conv_id=None):
            return {"success": True, "agent_id": agent_id, "parent": _parent_conv_id}

        self.router.agent_tools.get_agent_result = fake_get_agent_result

        result = await self.router.execute(
            "get_agent_result",
            {"agent_id": "sub-abc123", "unexpected": "drop"},
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["agent_id"], "sub-abc123")
        self.assertEqual(result["parent"], "conv-parent-1")

    async def test_get_agent_result_alias_and_missing_id(self):
        async def fake_get_agent_result(agent_id, _parent_conv_id=None):
            return {"success": True, "agent_id": agent_id, "parent": _parent_conv_id}

        self.router.agent_tools.get_agent_result = fake_get_agent_result

        aliased = await self.router.execute("get_agent_result", {"id": "sub-alias-1"})
        self.assertTrue(aliased["success"])
        self.assertEqual(aliased["agent_id"], "sub-alias-1")

        missing = await self.router.execute("get_agent_result", {"foo": "bar"})
        self.assertFalse(missing["success"])
        self.assertIn("agent_id is required", missing["error"])

    async def test_comfy_local_first_blocks_web_until_local_probe(self):
        self.router.set_agent_context(
            instance_id="inst-parent",
            conversation_store=object(),
            persona_manager=object(),
            parent_conv_id="conv-parent-1",
            persona_id="image_creator",
        )

        async def fake_web_search(query):
            return {"success": True, "query": query}

        async def fake_list_templates():
            return {"success": True, "templates": []}

        self.router._routes["web_search"] = fake_web_search
        self.router._routes["comfyui_list_templates"] = fake_list_templates

        blocked = await self.router.execute("web_search", {"query": "ltx comfy"})
        self.assertFalse(blocked["success"])
        self.assertIn("Local-first policy", blocked["error"])

        local_probe = await self.router.execute("comfyui_list_templates", {})
        self.assertTrue(local_probe["success"])

        allowed = await self.router.execute("web_search", {"query": "ltx comfy"})
        self.assertTrue(allowed["success"])
        self.assertEqual(allowed["query"], "ltx comfy")

    async def test_non_comfy_persona_not_blocked_by_local_first(self):
        self.router.set_agent_context(
            instance_id="inst-parent",
            conversation_store=object(),
            persona_manager=object(),
            parent_conv_id="conv-parent-1",
            persona_id="general_assistant",
        )

        async def fake_web_search(query):
            return {"success": True, "query": query}

        self.router._routes["web_search"] = fake_web_search
        allowed = await self.router.execute("web_search", {"query": "test"})
        self.assertTrue(allowed["success"])


class TestAgentToolsIsolation(unittest.TestCase):
    def setUp(self):
        self.tools = AgentTools(MockOrchestrator(), MockN8n(), MockSettings())
        # Isolate class-level shared state for this test module.
        self.tools._agents.clear()
        while not self.tools._event_queue.empty():
            try:
                self.tools._event_queue.get_nowait()
            except Exception:
                break

    def test_check_agents_scoped_to_parent(self):
        a1 = SubAgentState("a1", "one", "task", "general_assistant", "c1", "m1", "p1")
        a2 = SubAgentState("a2", "two", "task", "general_assistant", "c2", "m1", "p2")
        a1.status = "completed"
        a2.status = "running"
        self.tools._agents.update({"a1": a1, "a2": a2})

        result = asyncio.run(self.tools.check_agents(_parent_conv_id="p1"))
        self.assertTrue(result["success"])
        self.assertEqual(len(result["agents"]), 1)
        self.assertEqual(result["agents"][0]["agent_id"], "a1")
        self.assertEqual(result["summary"]["completed"], 1)
        self.assertEqual(result["summary"]["running"], 0)

    def test_get_agent_result_respects_parent_scope(self):
        a1 = SubAgentState("a1", "one", "task", "general_assistant", "c1", "m1", "p1")
        a1.status = "completed"
        a1.result = {"response": "ok", "tool_calls": [], "tool_call_count": 0, "conversation_id": "c1"}
        self.tools._agents["a1"] = a1

        wrong_parent = asyncio.run(self.tools.get_agent_result("a1", _parent_conv_id="p2"))
        self.assertFalse(wrong_parent["success"])
        self.assertEqual(wrong_parent["available"], [])

        right_parent = asyncio.run(self.tools.get_agent_result("a1", _parent_conv_id="p1"))
        self.assertTrue(right_parent["success"])
        self.assertEqual(right_parent["agent_id"], "a1")

    def test_drain_events_scoped(self):
        self.tools._event_queue.put_nowait({"parent_conv_id": "p1", "event": "started"})
        self.tools._event_queue.put_nowait({"parent_conv_id": "p2", "event": "started"})

        p1_events = self.tools.drain_events(parent_conv_id="p1")
        self.assertEqual(len(p1_events), 1)
        self.assertEqual(p1_events[0]["parent_conv_id"], "p1")

        p2_events = self.tools.drain_events(parent_conv_id="p2")
        self.assertEqual(len(p2_events), 1)
        self.assertEqual(p2_events[0]["parent_conv_id"], "p2")

    def test_abort_all_scoped(self):
        a1 = SubAgentState("a1", "one", "task", "general_assistant", "c1", "m1", "p1")
        a2 = SubAgentState("a2", "two", "task", "general_assistant", "c2", "m1", "p2")
        a1.status = "running"
        a2.status = "running"
        self.tools._agents.update({"a1": a1, "a2": a2})

        self.tools.abort_all(parent_conv_id="p1")
        self.assertTrue(a1.should_abort())
        self.assertFalse(a2.should_abort())

    def test_supervise_workers_nudges_stale_running_agents(self):
        a1 = SubAgentState("a1", "one", "task", "general_assistant", "c1", "m1", "p1")
        a1.status = "running"
        # Make it look stale/stuck.
        a1.started_at -= 120
        a1.last_progress_at -= 120
        self.tools._agents["a1"] = a1

        result = asyncio.run(
            self.tools.supervise_workers(
                parent_conv_id="p1",
                stuck_after_s=30,
                nudge_after_s=30,
                max_auto_nudges=2,
            )
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["running_workers"], 1)
        self.assertEqual(len(result["nudged_workers"]), 1)
        self.assertEqual(a1.auto_nudges, 1)
        self.assertGreaterEqual(len(a1.drain_guidance()), 1)


if __name__ == "__main__":
    unittest.main()
