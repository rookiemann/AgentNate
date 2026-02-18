"""
Tests for RoutingPresetManager — CRUD operations, resolve(), recommend().

Uses a temporary directory for preset storage to avoid touching real data.
"""

import sys
import os
import json
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.routing_presets import (
    RoutingPresetManager,
    _sanitize_filename,
    _CODING_KEYWORDS,
    _VISION_KEYWORDS,
)
from providers.base import ProviderType, ModelStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_settings(overrides=None):
    """Create a mock settings manager."""
    store = overrides or {}
    settings = MagicMock()
    settings.get = MagicMock(side_effect=lambda k, default=None: store.get(k, default))
    settings.set = MagicMock(side_effect=lambda k, v: store.__setitem__(k, v))
    return settings


def make_mock_instance(provider="llama_cpp", model_id="deepseek-coder-1.3b",
                       display_name="DeepSeek Coder 1.3B", status="ready",
                       instance_id=None):
    """Create a mock model instance using real ProviderType/ModelStatus enums."""
    inst = MagicMock()
    inst.id = instance_id or f"inst-{model_id[:8]}"
    inst.provider_type = ProviderType(provider)
    inst.model_identifier = model_id
    inst.display_name = display_name
    inst.status = ModelStatus(status)
    return inst


class RoutingPresetTestBase(unittest.TestCase):
    """Base class that sets up a temp directory for preset storage."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self._orig_presets_dir = None
        # Patch PRESETS_DIR to use temp directory
        import backend.routing_presets as rp_module
        self._orig_presets_dir = rp_module.PRESETS_DIR
        rp_module.PRESETS_DIR = self.temp_dir

        self.settings = make_mock_settings()
        self.mgr = RoutingPresetManager(self.settings)

    def tearDown(self):
        import backend.routing_presets as rp_module
        rp_module.PRESETS_DIR = self._orig_presets_dir
        shutil.rmtree(self.temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test: _sanitize_filename
# ---------------------------------------------------------------------------

class TestSanitizeFilename(unittest.TestCase):
    """Tests for _sanitize_filename helper."""

    def test_spaces_to_underscores(self):
        self.assertEqual(_sanitize_filename("My Preset Name"), "my_preset_name")

    def test_special_chars_removed(self):
        self.assertEqual(_sanitize_filename("test@#$%!"), "test")

    def test_truncation_at_100(self):
        result = _sanitize_filename("x" * 200)
        self.assertLessEqual(len(result), 100)

    def test_empty_returns_preset(self):
        self.assertEqual(_sanitize_filename("!!!"), "preset")

    def test_hyphens_preserved(self):
        self.assertIn("-", _sanitize_filename("my-preset"))


# ---------------------------------------------------------------------------
# Test: CRUD Operations
# ---------------------------------------------------------------------------

class TestPresetCRUD(RoutingPresetTestBase):
    """Tests for save, get, list, delete operations."""

    def test_save_preset(self):
        routes = {"coder": {"provider": "llama_cpp", "model_match": "deepseek"}}
        preset = self.mgr.save_preset("Test Routing", routes, "My description")

        self.assertIn("id", preset)
        self.assertTrue(preset["id"].startswith("rp-"))
        self.assertEqual(preset["name"], "Test Routing")
        self.assertEqual(preset["routes"], routes)
        self.assertEqual(preset["description"], "My description")

    def test_save_creates_file(self):
        self.mgr.save_preset("File Test", {"a": {}})
        files = os.listdir(self.temp_dir)
        self.assertEqual(len(files), 1)
        self.assertTrue(files[0].endswith(".json"))

    def test_get_preset_by_id(self):
        preset = self.mgr.save_preset("Findable", {"x": {}})
        found = self.mgr.get_preset(preset["id"])
        self.assertIsNotNone(found)
        self.assertEqual(found["name"], "Findable")

    def test_get_preset_not_found(self):
        result = self.mgr.get_preset("rp-nonexistent")
        self.assertIsNone(result)

    def test_list_presets_empty(self):
        presets = self.mgr.list_presets()
        self.assertEqual(presets, [])

    def test_list_presets_multiple(self):
        self.mgr.save_preset("Alpha", {"a": {}})
        self.mgr.save_preset("Beta", {"b": {}})
        self.mgr.save_preset("Charlie", {"c": {}})

        presets = self.mgr.list_presets()
        self.assertEqual(len(presets), 3)
        # Should be sorted alphabetically
        names = [p["name"] for p in presets]
        self.assertEqual(names, ["Alpha", "Beta", "Charlie"])

    def test_delete_preset(self):
        preset = self.mgr.save_preset("Deletable", {"x": {}})
        result = self.mgr.delete_preset(preset["id"])
        self.assertTrue(result)
        self.assertIsNone(self.mgr.get_preset(preset["id"]))

    def test_delete_nonexistent_returns_false(self):
        result = self.mgr.delete_preset("rp-fake")
        self.assertFalse(result)

    def test_delete_active_preset_disables_routing(self):
        settings_store = {
            "agent.routing_enabled": True,
        }
        settings = make_mock_settings(settings_store)
        mgr = RoutingPresetManager(settings)

        preset = mgr.save_preset("Active One", {"x": {}})
        settings_store["agent.active_routing_preset_id"] = preset["id"]

        mgr.delete_preset(preset["id"])
        # Should have called settings.set to disable routing
        settings.set.assert_any_call("agent.routing_enabled", False)
        settings.set.assert_any_call("agent.active_routing_preset_id", None)

    def test_save_overwrites_same_name(self):
        self.mgr.save_preset("Same Name", {"v": 1})
        self.mgr.save_preset("Same Name", {"v": 2})
        files = os.listdir(self.temp_dir)
        # Both write to same filename but second overwrites
        self.assertEqual(len(files), 1)

    def test_ignores_non_json_files(self):
        # Place a non-JSON file in the directory
        with open(os.path.join(self.temp_dir, "readme.txt"), "w") as f:
            f.write("ignore me")
        presets = self.mgr.list_presets()
        self.assertEqual(presets, [])

    def test_ignores_corrupt_json(self):
        with open(os.path.join(self.temp_dir, "bad.json"), "w") as f:
            f.write("not valid json{{{")
        presets = self.mgr.list_presets()
        self.assertEqual(presets, [])


# ---------------------------------------------------------------------------
# Test: resolve()
# ---------------------------------------------------------------------------

class TestResolve(RoutingPresetTestBase):
    """Tests for resolve() — matching persona to loaded model instance."""

    def _save_and_resolve(self, routes, persona, instances):
        preset = self.mgr.save_preset("Test", routes)
        return self.mgr.resolve(preset["id"], persona, instances)

    def test_resolve_exact_match(self):
        instances = [
            make_mock_instance("llama_cpp", "deepseek-coder-1.3b", "DeepSeek Coder"),
        ]
        routes = {"coder": {"provider": "llama_cpp", "model_match": "deepseek"}}
        result = self._save_and_resolve(routes, "coder", instances)
        self.assertIsNotNone(result)

    def test_resolve_no_matching_provider(self):
        instances = [
            make_mock_instance("ollama", "llama3", "Llama 3"),
        ]
        routes = {"coder": {"provider": "llama_cpp", "model_match": "llama"}}
        result = self._save_and_resolve(routes, "coder", instances)
        self.assertIsNone(result)

    def test_resolve_no_matching_model_name(self):
        instances = [
            make_mock_instance("llama_cpp", "mistral-7b", "Mistral 7B"),
        ]
        routes = {"coder": {"provider": "llama_cpp", "model_match": "deepseek"}}
        result = self._save_and_resolve(routes, "coder", instances)
        self.assertIsNone(result)

    def test_resolve_unknown_persona(self):
        instances = [
            make_mock_instance("llama_cpp", "deepseek", "DeepSeek"),
        ]
        routes = {"coder": {"provider": "llama_cpp", "model_match": "deepseek"}}
        result = self._save_and_resolve(routes, "unknown_persona", instances)
        self.assertIsNone(result)

    def test_resolve_nonexistent_preset(self):
        result = self.mgr.resolve("rp-fake", "coder", [])
        self.assertIsNone(result)

    def test_resolve_case_insensitive(self):
        instances = [
            make_mock_instance("llama_cpp", "DeepSeek-Coder-V2", "DeepSeek Coder V2"),
        ]
        routes = {"coder": {"provider": "llama_cpp", "model_match": "deepseek"}}
        result = self._save_and_resolve(routes, "coder", instances)
        self.assertIsNotNone(result)

    def test_resolve_skips_loading_models(self):
        instances = [
            make_mock_instance("llama_cpp", "deepseek", "DeepSeek", status="loading"),
        ]
        routes = {"coder": {"provider": "llama_cpp", "model_match": "deepseek"}}
        result = self._save_and_resolve(routes, "coder", instances)
        self.assertIsNone(result)

    def test_resolve_empty_route_fields(self):
        instances = [make_mock_instance()]
        routes = {"coder": {"provider": "", "model_match": ""}}
        result = self._save_and_resolve(routes, "coder", instances)
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Test: recommend()
# ---------------------------------------------------------------------------

class TestRecommend(RoutingPresetTestBase):
    """Tests for recommend() — heuristic routing suggestions."""

    def test_recommend_empty_models(self):
        result = self.mgr.recommend([])
        self.assertFalse(result["success"])
        self.assertIn("No models loaded", result["error"])

    def test_recommend_single_model(self):
        instances = [make_mock_instance("llama_cpp", "llama-3-8b", "Llama 3 8B")]
        result = self.mgr.recommend(instances)
        self.assertTrue(result["success"])
        self.assertIn("recommended_routes", result)
        routes = result["recommended_routes"]
        # Should have routes for standard personas
        self.assertIn("coder", routes)
        self.assertIn("researcher", routes)

    def test_recommend_coding_model_preferred_for_coder(self):
        instances = [
            make_mock_instance("llama_cpp", "deepseek-coder-33b", "DeepSeek Coder 33B"),
            make_mock_instance("openrouter", "claude-3-sonnet", "Claude 3 Sonnet"),
        ]
        result = self.mgr.recommend(instances)
        self.assertTrue(result["success"])
        coder_route = result["recommended_routes"].get("coder", {})
        # DeepSeek Coder should be preferred for coder persona
        self.assertIn("deepseek", coder_route.get("model_match", "").lower())

    def test_recommend_includes_loaded_models(self):
        instances = [make_mock_instance("llama_cpp", "model-a", "Model A")]
        result = self.mgr.recommend(instances)
        self.assertIn("loaded_models", result)
        self.assertEqual(len(result["loaded_models"]), 1)

    def test_recommend_has_summary(self):
        instances = [make_mock_instance()]
        result = self.mgr.recommend(instances)
        self.assertIn("summary", result)
        self.assertIn("note", result)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
