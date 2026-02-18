"""
Tests for workflow_bridge.py — n8n workflow pattern generators.

Tests all 4 patterns (swarm, pipeline, multi_coder, image_pipeline),
entry point, helper functions, and generated workflow structure.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.workflow_bridge import (
    generate_workflow,
    generate_swarm_workflow,
    generate_pipeline_workflow,
    generate_multi_coder_workflow,
    generate_image_pipeline_workflow,
    PATTERNS,
    _pos,
    _resolve_code_js,
    _llm_call_js,
    _merge_results_js,
    _wrap_workflow,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_node(wf, name=None, type_contains=None):
    """Find a node by name or type substring."""
    for node in wf.get("nodes", []):
        if name and node.get("name") == name:
            return node
        if type_contains and type_contains in node.get("type", ""):
            return node
    return None


def _find_nodes(wf, type_contains):
    """Find all nodes containing a type substring."""
    return [n for n in wf.get("nodes", [])
            if type_contains in n.get("type", "")]


def _get_connections_from(wf, node_name):
    """Get the list of target nodes connected from a source node."""
    conns = wf.get("connections", {}).get(node_name, {}).get("main", [[]])
    targets = []
    for output_list in conns:
        for conn in output_list:
            targets.append(conn.get("node"))
    return targets


# ---------------------------------------------------------------------------
# Test: Entry Point
# ---------------------------------------------------------------------------

class TestGenerateWorkflow(unittest.TestCase):
    """Tests for the generate_workflow entry point."""

    def test_known_patterns(self):
        self.assertIn("swarm", PATTERNS)
        self.assertIn("pipeline", PATTERNS)
        self.assertIn("multi_coder", PATTERNS)
        self.assertIn("image_pipeline", PATTERNS)
        self.assertEqual(len(PATTERNS), 4)

    def test_unknown_pattern_raises(self):
        with self.assertRaises(ValueError) as ctx:
            generate_workflow("nonexistent", {})
        self.assertIn("nonexistent", str(ctx.exception))
        self.assertIn("Available", str(ctx.exception))

    def test_delegates_to_correct_generator(self):
        # swarm with at least one persona should work
        wf = generate_workflow("swarm", {
            "personas": ["coder"],
            "webhook_path": "test-entry-swarm",
        })
        self.assertEqual(wf["meta"]["webhook_path"], "test-entry-swarm")


# ---------------------------------------------------------------------------
# Test: Helper Functions
# ---------------------------------------------------------------------------

class TestHelperFunctions(unittest.TestCase):
    """Tests for utility functions."""

    def test_pos_origin(self):
        self.assertEqual(_pos(0, 0), [250, 200])

    def test_pos_offset(self):
        self.assertEqual(_pos(2, 3), [250 + 2 * 250, 200 + 3 * 150])

    def test_resolve_code_js_single_persona(self):
        js = _resolve_code_js(["coder"])
        self.assertIn("coder", js)
        self.assertIn("/api/routing/resolve/coder", js)
        self.assertIn("Promise.all", js)

    def test_resolve_code_js_multiple_personas(self):
        js = _resolve_code_js(["coder", "researcher", "analyst"])
        self.assertIn("/api/routing/resolve/coder", js)
        self.assertIn("/api/routing/resolve/researcher", js)
        self.assertIn("/api/routing/resolve/analyst", js)

    def test_llm_call_js_contains_persona(self):
        js = _llm_call_js("coder", "You are a coder.", "data.input.task", 2048)
        self.assertIn('"coder"', js)
        self.assertIn("You are a coder.", js)
        self.assertIn("/api/chat/completions", js)
        self.assertIn("coder_response", js)  # default output_field

    def test_llm_call_js_custom_output_field(self):
        js = _llm_call_js("coder", "prompt", "data.x", 1024, "my_output")
        self.assertIn("my_output", js)
        self.assertNotIn("coder_response", js)

    def test_llm_call_js_escapes_backticks(self):
        js = _llm_call_js("x", "Prompt with `backtick`", "data.x", 1024)
        self.assertIn("\\`", js)

    def test_merge_results_js(self):
        js = _merge_results_js(["coder", "researcher"])
        self.assertIn("coder_response", js)
        self.assertIn("researcher_response", js)
        self.assertIn("items[0]", js)
        self.assertIn("items[1]", js)

    def test_wrap_workflow(self):
        wf = _wrap_workflow("Test", [{"name": "N1"}], {"N1": {}}, "test-path")
        self.assertEqual(wf["name"], "Test")
        self.assertFalse(wf["active"])
        self.assertEqual(wf["nodes"], [{"name": "N1"}])
        self.assertEqual(wf["meta"]["webhook_path"], "test-path")
        self.assertEqual(wf["meta"]["generator"], "agentnate-workflow-bridge")


# ---------------------------------------------------------------------------
# Test: Swarm Pattern
# ---------------------------------------------------------------------------

class TestSwarmWorkflow(unittest.TestCase):
    """Tests for generate_swarm_workflow."""

    def test_requires_personas(self):
        with self.assertRaises(ValueError):
            generate_swarm_workflow({"personas": []})

    def test_single_persona_string(self):
        wf = generate_swarm_workflow({"personas": ["coder"]})
        self.assertEqual(wf["name"], "Routing Swarm")
        # Should have: Webhook, Resolve Routes, LLM Coder, Send Response
        self.assertEqual(len(wf["nodes"]), 4)

    def test_single_persona_no_merge(self):
        wf = generate_swarm_workflow({"personas": ["coder"]})
        # With single persona, no Merge Results node
        merge = _find_node(wf, name="Merge Results")
        self.assertIsNone(merge)

    def test_multi_persona_has_merge(self):
        wf = generate_swarm_workflow({
            "personas": ["coder", "researcher", "analyst"],
        })
        merge = _find_node(wf, name="Merge Results")
        self.assertIsNotNone(merge)
        # Should have: Webhook, Resolve, 3 LLM nodes, Merge, Respond = 7
        self.assertEqual(len(wf["nodes"]), 7)

    def test_webhook_node_present(self):
        wf = generate_swarm_workflow({"personas": ["coder"]})
        webhook = _find_node(wf, type_contains="webhook")
        self.assertIsNotNone(webhook)
        self.assertEqual(webhook["parameters"]["httpMethod"], "POST")
        self.assertEqual(webhook["parameters"]["responseMode"], "responseNode")

    def test_resolve_node_present(self):
        wf = generate_swarm_workflow({"personas": ["coder", "researcher"]})
        resolve = _find_node(wf, name="Resolve Routes")
        self.assertIsNotNone(resolve)
        self.assertIn("/api/routing/resolve/coder", resolve["parameters"]["jsCode"])
        self.assertIn("/api/routing/resolve/researcher", resolve["parameters"]["jsCode"])

    def test_connections_webhook_to_resolve(self):
        wf = generate_swarm_workflow({"personas": ["coder"]})
        targets = _get_connections_from(wf, "Webhook")
        self.assertIn("Resolve Routes", targets)

    def test_connections_resolve_to_personas(self):
        wf = generate_swarm_workflow({"personas": ["coder", "researcher"]})
        targets = _get_connections_from(wf, "Resolve Routes")
        self.assertIn("LLM Coder", targets)
        self.assertIn("LLM Researcher", targets)

    def test_custom_name_and_path(self):
        wf = generate_swarm_workflow({
            "personas": ["coder"],
            "name": "My Swarm",
            "webhook_path": "my-swarm",
        })
        self.assertEqual(wf["name"], "My Swarm")
        self.assertEqual(wf["meta"]["webhook_path"], "my-swarm")

    def test_persona_dicts(self):
        wf = generate_swarm_workflow({
            "personas": [
                {"id": "coder", "system_prompt": "Write code.", "max_tokens": 4096},
                {"id": "writer", "system_prompt": "Write prose."},
            ],
        })
        llm_coder = _find_node(wf, name="LLM Coder")
        self.assertIn("Write code.", llm_coder["parameters"]["jsCode"])
        llm_writer = _find_node(wf, name="LLM Writer")
        self.assertIn("Write prose.", llm_writer["parameters"]["jsCode"])

    def test_respond_node_present(self):
        wf = generate_swarm_workflow({"personas": ["coder"]})
        respond = _find_node(wf, type_contains="respondToWebhook")
        self.assertIsNotNone(respond)

    def test_all_nodes_have_ids(self):
        wf = generate_swarm_workflow({"personas": ["a", "b", "c"]})
        for node in wf["nodes"]:
            self.assertIn("id", node)
            self.assertTrue(len(node["id"]) > 0)


# ---------------------------------------------------------------------------
# Test: Pipeline Pattern
# ---------------------------------------------------------------------------

class TestPipelineWorkflow(unittest.TestCase):
    """Tests for generate_pipeline_workflow."""

    def test_requires_stages(self):
        with self.assertRaises(ValueError):
            generate_pipeline_workflow({"stages": []})

    def test_single_stage(self):
        wf = generate_pipeline_workflow({
            "stages": [
                {"persona_id": "coder", "system_prompt": "Code it.", "output_field": "code"},
            ],
        })
        # Webhook, Resolve, Stage 1, Respond = 4
        self.assertEqual(len(wf["nodes"]), 4)

    def test_multi_stage_sequential(self):
        wf = generate_pipeline_workflow({
            "stages": [
                {"persona_id": "researcher", "system_prompt": "Research.", "output_field": "research"},
                {"persona_id": "writer", "system_prompt": "Write.", "output_field": "article"},
                {"persona_id": "editor", "system_prompt": "Edit.", "output_field": "final"},
            ],
        })
        # Webhook, Resolve, Stage1, Stage2, Stage3, Respond = 6
        self.assertEqual(len(wf["nodes"]), 6)

    def test_stages_chained_sequentially(self):
        wf = generate_pipeline_workflow({
            "stages": [
                {"persona_id": "a", "system_prompt": "A", "output_field": "out_a"},
                {"persona_id": "b", "system_prompt": "B", "output_field": "out_b"},
            ],
        })
        # Resolve → Stage 1
        targets_resolve = _get_connections_from(wf, "Resolve Routes")
        self.assertIn("Stage 1: A", targets_resolve)

        # Stage 1 → Stage 2
        targets_stage1 = _get_connections_from(wf, "Stage 1: A")
        self.assertIn("Stage 2: B", targets_stage1)

        # Stage 2 → Respond
        targets_stage2 = _get_connections_from(wf, "Stage 2: B")
        self.assertIn("Send Response", targets_stage2)

    def test_second_stage_uses_previous_output(self):
        wf = generate_pipeline_workflow({
            "stages": [
                {"persona_id": "a", "system_prompt": "A", "output_field": "out_a"},
                {"persona_id": "b", "system_prompt": "B", "output_field": "out_b"},
            ],
        })
        stage2 = _find_node(wf, name="Stage 2: B")
        js = stage2["parameters"]["jsCode"]
        # Second stage should reference first stage's output
        self.assertIn("out_a", js)

    def test_deduplicates_persona_ids_for_resolve(self):
        wf = generate_pipeline_workflow({
            "stages": [
                {"persona_id": "coder", "system_prompt": "A", "output_field": "a"},
                {"persona_id": "coder", "system_prompt": "B", "output_field": "b"},
            ],
        })
        resolve = _find_node(wf, name="Resolve Routes")
        js = resolve["parameters"]["jsCode"]
        # Should only resolve "coder" once, not twice
        count = js.count("/api/routing/resolve/coder")
        self.assertEqual(count, 1)


# ---------------------------------------------------------------------------
# Test: Multi-Coder Pattern
# ---------------------------------------------------------------------------

class TestMultiCoderWorkflow(unittest.TestCase):
    """Tests for generate_multi_coder_workflow."""

    def test_default_config(self):
        wf = generate_multi_coder_workflow({})
        # Webhook, Resolve, 3 Coders, Collect, Reviewer, Respond = 8
        self.assertEqual(len(wf["nodes"]), 8)

    def test_custom_coder_count(self):
        wf = generate_multi_coder_workflow({"coder_count": 5})
        # Webhook, Resolve, 5 Coders, Collect, Reviewer, Respond = 10
        self.assertEqual(len(wf["nodes"]), 10)

    def test_coder_nodes_parallel(self):
        wf = generate_multi_coder_workflow({"coder_count": 3})
        targets = _get_connections_from(wf, "Resolve Routes")
        self.assertIn("Coder 1", targets)
        self.assertIn("Coder 2", targets)
        self.assertIn("Coder 3", targets)

    def test_coders_connect_to_merge(self):
        wf = generate_multi_coder_workflow({"coder_count": 2})
        t1 = _get_connections_from(wf, "Coder 1")
        t2 = _get_connections_from(wf, "Coder 2")
        self.assertIn("Collect Solutions", t1)
        self.assertIn("Collect Solutions", t2)

    def test_merge_connects_to_reviewer(self):
        wf = generate_multi_coder_workflow({})
        targets = _get_connections_from(wf, "Collect Solutions")
        self.assertIn("Reviewer", targets)

    def test_reviewer_connects_to_respond(self):
        wf = generate_multi_coder_workflow({})
        targets = _get_connections_from(wf, "Reviewer")
        self.assertIn("Send Response", targets)

    def test_custom_system_prompts(self):
        wf = generate_multi_coder_workflow({
            "coder_system_prompt": "Write Rust code only.",
            "reviewer_system_prompt": "Review Rust solutions.",
        })
        coder1 = _find_node(wf, name="Coder 1")
        reviewer = _find_node(wf, name="Reviewer")
        self.assertIn("Write Rust code only.", coder1["parameters"]["jsCode"])
        self.assertIn("Review Rust solutions.", reviewer["parameters"]["jsCode"])

    def test_default_webhook_path(self):
        wf = generate_multi_coder_workflow({})
        self.assertEqual(wf["meta"]["webhook_path"], "multi-coder")


# ---------------------------------------------------------------------------
# Test: Image Pipeline Pattern
# ---------------------------------------------------------------------------

class TestImagePipelineWorkflow(unittest.TestCase):
    """Tests for generate_image_pipeline_workflow."""

    def test_basic_structure(self):
        wf = generate_image_pipeline_workflow({})
        # Webhook, Resolve, LLM, Prepare, Generate, Wait, Poll, Respond = 8
        self.assertEqual(len(wf["nodes"]), 8)

    def test_has_comfyui_generate_request(self):
        wf = generate_image_pipeline_workflow({})
        http_nodes = _find_nodes(wf, "httpRequest")
        self.assertGreaterEqual(len(http_nodes), 1)
        gen_node = http_nodes[0]
        self.assertIn("/api/comfyui/generate", gen_node["parameters"]["url"])

    def test_has_wait_node(self):
        wf = generate_image_pipeline_workflow({})
        wait = _find_node(wf, type_contains="wait")
        self.assertIsNotNone(wait)

    def test_has_poll_result_code_node(self):
        wf = generate_image_pipeline_workflow({})
        poll = _find_node(wf, name="Poll Result")
        self.assertIsNotNone(poll)
        self.assertIn("/api/comfyui/result/", poll["parameters"]["jsCode"])

    def test_custom_instance_and_checkpoint(self):
        wf = generate_image_pipeline_workflow({
            "instance_id": "comfy-abc",
            "checkpoint": "sd_xl_base_1.0.safetensors",
        })
        prep = _find_node(wf, name="Prepare ComfyUI Request")
        js = prep["parameters"]["jsCode"]
        self.assertIn("comfy-abc", js)
        self.assertIn("sd_xl_base_1.0.safetensors", js)

    def test_default_webhook_path(self):
        wf = generate_image_pipeline_workflow({})
        self.assertEqual(wf["meta"]["webhook_path"], "image-pipeline")

    def test_sequential_connections(self):
        wf = generate_image_pipeline_workflow({})
        # Webhook → Resolve
        self.assertIn("Resolve Routes", _get_connections_from(wf, "Webhook"))
        # Resolve → Generate SD Prompt
        self.assertIn("Generate SD Prompt", _get_connections_from(wf, "Resolve Routes"))
        # ... → Prepare → Generate → Wait → Poll → Respond
        self.assertIn("Prepare ComfyUI Request", _get_connections_from(wf, "Generate SD Prompt"))
        self.assertIn("Generate Image", _get_connections_from(wf, "Prepare ComfyUI Request"))
        self.assertIn("Wait for Generation", _get_connections_from(wf, "Generate Image"))
        self.assertIn("Poll Result", _get_connections_from(wf, "Wait for Generation"))
        self.assertIn("Send Response", _get_connections_from(wf, "Poll Result"))


# ---------------------------------------------------------------------------
# Test: Common Workflow Properties
# ---------------------------------------------------------------------------

class TestCommonWorkflowProperties(unittest.TestCase):
    """Tests that apply to all generated workflows."""

    def _all_workflows(self):
        """Generate one workflow per pattern for common checks."""
        return [
            generate_swarm_workflow({"personas": ["coder", "researcher"]}),
            generate_pipeline_workflow({
                "stages": [{"persona_id": "a", "system_prompt": "A", "output_field": "x"}],
            }),
            generate_multi_coder_workflow({}),
            generate_image_pipeline_workflow({}),
        ]

    def test_all_start_inactive(self):
        for wf in self._all_workflows():
            self.assertFalse(wf["active"], f"{wf['name']} should start inactive")

    def test_all_have_webhook_trigger(self):
        for wf in self._all_workflows():
            webhook = _find_node(wf, type_contains="webhook")
            self.assertIsNotNone(webhook, f"{wf['name']} missing webhook trigger")

    def test_all_have_resolve_routes(self):
        for wf in self._all_workflows():
            resolve = _find_node(wf, name="Resolve Routes")
            self.assertIsNotNone(resolve, f"{wf['name']} missing Resolve Routes")

    def test_all_have_respond_node(self):
        for wf in self._all_workflows():
            respond = _find_node(wf, type_contains="respondToWebhook")
            self.assertIsNotNone(respond, f"{wf['name']} missing Respond node")

    def test_all_have_meta(self):
        for wf in self._all_workflows():
            self.assertIn("meta", wf)
            self.assertIn("webhook_path", wf["meta"])
            self.assertEqual(wf["meta"]["generator"], "agentnate-workflow-bridge")

    def test_all_nodes_have_unique_ids(self):
        for wf in self._all_workflows():
            ids = [n["id"] for n in wf["nodes"]]
            self.assertEqual(len(ids), len(set(ids)),
                             f"{wf['name']} has duplicate node IDs")

    def test_all_have_execution_order_v1(self):
        for wf in self._all_workflows():
            self.assertEqual(wf["settings"]["executionOrder"], "v1")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
