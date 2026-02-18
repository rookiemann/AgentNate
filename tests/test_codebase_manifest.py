"""
Tests for codebase manifest — ManifestGenerator, ManifestCache, query logic.

Uses a temporary directory with synthetic Python files to avoid coupling
to the real codebase structure.
"""

import sys
import os
import json
import shutil
import tempfile
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.codebase_manifest import ManifestGenerator, ManifestCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_file(base_dir, rel_path, content):
    """Write a file at rel_path under base_dir, creating dirs as needed."""
    full = os.path.join(base_dir, rel_path.replace("/", os.sep))
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)


def build_sample_project(root):
    """Create a minimal fake project tree for testing."""
    # A simple module
    _write_file(root, "core/__init__.py", '"""Core module."""\n')
    _write_file(root, "core/engine.py", '''\
"""The main engine."""

import os
from typing import Dict, Optional

MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30

class Engine:
    """Runs things."""

    def __init__(self, config: Dict):
        self.config = config

    async def start(self, port: int = 8000) -> bool:
        """Start the engine."""
        pass

    def _private_method(self):
        pass

def helper_function(x: int, y: str = "default") -> Optional[str]:
    """A helper."""
    return None

async def async_helper() -> None:
    pass
''')

    # A route file
    _write_file(root, "backend/__init__.py", "")
    _write_file(root, "backend/routes/__init__.py", "")
    _write_file(root, "backend/routes/items.py", '''\
"""Item routes."""
from fastapi import APIRouter
router = APIRouter()

@router.get("/")
async def list_items():
    return []

@router.post("/")
async def create_item():
    return {}

@router.delete("/{item_id}")
async def delete_item(item_id: str):
    return {"ok": True}
''')

    # A tools file
    _write_file(root, "backend/tools/__init__.py", "")
    _write_file(root, "backend/tools/sample_tools.py", '''\
"""Sample tools."""

TOOL_DEFINITIONS = [
    {
        "name": "do_thing",
        "description": "Does a thing.",
        "parameters": {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "The input"}
            },
            "required": ["input"]
        }
    },
    {
        "name": "do_other",
        "description": "Does another thing.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
]
''')

    # A server file with include_router
    _write_file(root, "backend/server.py", '''\
"""Server."""
from fastapi import FastAPI
app = FastAPI()
from backend.routes import items
app.include_router(items.router, prefix="/api/items", tags=["Items"])
''')


# ---------------------------------------------------------------------------
# Test: ManifestGenerator
# ---------------------------------------------------------------------------

class TestManifestGenerator(unittest.TestCase):
    """Tests for ManifestGenerator file/AST scanning."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        build_sample_project(self.temp_dir)
        self.gen = ManifestGenerator(project_root=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_generate_returns_all_sections(self):
        manifest = self.gen.generate()
        for key in ("files", "endpoints", "tools", "stats"):
            self.assertIn(key, manifest, f"Missing section: {key}")

    def test_files_scanned(self):
        manifest = self.gen.generate()
        paths = [f["path"] for f in manifest["files"]]
        self.assertTrue(any("core/engine.py" in p for p in paths))

    def test_file_classes_extracted(self):
        manifest = self.gen.generate()
        engine_file = next(
            (f for f in manifest["files"] if "engine.py" in f["path"]), None
        )
        self.assertIsNotNone(engine_file)
        class_names = [c["name"] for c in engine_file.get("classes", [])]
        self.assertIn("Engine", class_names)

    def test_file_functions_extracted(self):
        manifest = self.gen.generate()
        engine_file = next(
            (f for f in manifest["files"] if "engine.py" in f["path"]), None
        )
        func_names = [fn["name"] for fn in engine_file.get("functions", [])]
        self.assertIn("helper_function", func_names)
        self.assertIn("async_helper", func_names)

    def test_async_function_flagged(self):
        manifest = self.gen.generate()
        engine_file = next(
            (f for f in manifest["files"] if "engine.py" in f["path"]), None
        )
        async_fn = next(
            (fn for fn in engine_file.get("functions", [])
             if fn["name"] == "async_helper"), None
        )
        self.assertIsNotNone(async_fn)
        self.assertTrue(async_fn.get("is_async"))

    def test_function_args_extracted(self):
        manifest = self.gen.generate()
        engine_file = next(
            (f for f in manifest["files"] if "engine.py" in f["path"]), None
        )
        helper = next(
            (fn for fn in engine_file.get("functions", [])
             if fn["name"] == "helper_function"), None
        )
        self.assertIsNotNone(helper)
        self.assertIn("x: int", helper.get("args", []))

    def test_class_methods_extracted(self):
        manifest = self.gen.generate()
        engine_file = next(
            (f for f in manifest["files"] if "engine.py" in f["path"]), None
        )
        engine_cls = next(
            (c for c in engine_file.get("classes", []) if c["name"] == "Engine"),
            None
        )
        self.assertIsNotNone(engine_cls)
        method_names = [m["name"] for m in engine_cls.get("methods", [])]
        self.assertIn("__init__", method_names)
        self.assertIn("start", method_names)
        self.assertIn("_private_method", method_names)

    def test_constants_extracted(self):
        manifest = self.gen.generate()
        engine_file = next(
            (f for f in manifest["files"] if "engine.py" in f["path"]), None
        )
        self.assertIn("MAX_RETRIES", engine_file.get("constants", []))
        self.assertIn("DEFAULT_TIMEOUT", engine_file.get("constants", []))

    def test_docstring_extracted(self):
        manifest = self.gen.generate()
        engine_file = next(
            (f for f in manifest["files"] if "engine.py" in f["path"]), None
        )
        self.assertEqual(engine_file.get("docstring"), "The main engine.")

    def test_endpoints_extracted(self):
        manifest = self.gen.generate()
        endpoints = manifest["endpoints"]
        methods = [e["method"] for e in endpoints]
        self.assertIn("GET", methods)
        self.assertIn("POST", methods)
        self.assertIn("DELETE", methods)

    def test_endpoint_path(self):
        manifest = self.gen.generate()
        delete_ep = next(
            (e for e in manifest["endpoints"] if e["method"] == "DELETE"), None
        )
        self.assertIsNotNone(delete_ep)
        self.assertEqual(delete_ep["path"], "/{item_id}")

    def test_tools_extracted(self):
        manifest = self.gen.generate()
        tool_cats = manifest["tools"]
        self.assertTrue(len(tool_cats) >= 1)
        sample_cat = next(
            (c for c in tool_cats if c["category"] == "sample"), None
        )
        self.assertIsNotNone(sample_cat)
        tool_names = [t["name"] for t in sample_cat["tools"]]
        self.assertIn("do_thing", tool_names)
        self.assertIn("do_other", tool_names)

    def test_tool_parameters_extracted(self):
        manifest = self.gen.generate()
        sample_cat = next(
            (c for c in manifest["tools"] if c["category"] == "sample"), None
        )
        do_thing = next(
            (t for t in sample_cat["tools"] if t["name"] == "do_thing"), None
        )
        self.assertIn("parameters", do_thing)
        self.assertIn("properties", do_thing["parameters"])

    def test_route_prefixes_extracted(self):
        manifest = self.gen.generate()
        prefixes = manifest["route_prefixes"]
        self.assertTrue(len(prefixes) >= 1)
        self.assertEqual(prefixes[0]["prefix"], "/api/items")
        self.assertEqual(prefixes[0]["tag"], "Items")

    def test_stats_complete(self):
        manifest = self.gen.generate()
        stats = manifest["stats"]
        self.assertGreater(stats["total_files"], 0)
        self.assertGreater(stats["total_lines"], 0)
        self.assertGreater(stats["generated_at"], 0)

    def test_excludes_pycache(self):
        """__pycache__ dirs should be skipped."""
        os.makedirs(os.path.join(self.temp_dir, "__pycache__"), exist_ok=True)
        _write_file(self.temp_dir, "__pycache__/cached.py", "x = 1\n")
        manifest = self.gen.generate()
        paths = [f["path"] for f in manifest["files"]]
        self.assertFalse(any("__pycache__" in p for p in paths))

    def test_syntax_error_file_still_included(self):
        """Files with syntax errors should appear with an error flag."""
        _write_file(self.temp_dir, "broken.py", "def bad(\n")
        manifest = self.gen.generate()
        broken = next(
            (f for f in manifest["files"] if "broken.py" in f["path"]), None
        )
        self.assertIsNotNone(broken)
        self.assertEqual(broken.get("error"), "syntax_error")


# ---------------------------------------------------------------------------
# Test: ManifestCache
# ---------------------------------------------------------------------------

class TestManifestCache(unittest.TestCase):
    """Tests for ManifestCache persistence and staleness."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.manifest_path = os.path.join(self.temp_dir, "manifest.json")
        build_sample_project(self.temp_dir)
        self.cache = ManifestCache(
            project_root=self.temp_dir,
            manifest_path=self.manifest_path,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_returns_none_initially(self):
        self.assertIsNone(self.cache.get())

    def test_refresh_creates_manifest(self):
        manifest = self.cache.refresh()
        self.assertIn("stats", manifest)
        self.assertGreater(manifest["stats"]["total_files"], 0)

    def test_refresh_saves_to_disk(self):
        self.cache.refresh()
        self.assertTrue(os.path.exists(self.manifest_path))
        with open(self.manifest_path, "r") as f:
            data = json.load(f)
        self.assertIn("stats", data)

    def test_get_loads_from_disk(self):
        self.cache.refresh()
        # New cache instance (no memory)
        cache2 = ManifestCache(
            project_root=self.temp_dir,
            manifest_path=self.manifest_path,
        )
        manifest = cache2.get()
        self.assertIsNotNone(manifest)
        self.assertIn("stats", manifest)

    def test_get_or_refresh_generates_if_missing(self):
        manifest = self.cache.get_or_refresh()
        self.assertIn("stats", manifest)

    def test_is_stale_true_when_no_manifest(self):
        self.assertTrue(self.cache.is_stale())

    def test_is_stale_false_after_refresh(self):
        manifest = self.cache.refresh()
        # Immediately after refresh, nothing has changed
        self.assertFalse(self.cache.is_stale(manifest))

    def test_is_stale_true_after_file_modification(self):
        manifest = self.cache.refresh()
        # Ensure the modification time is clearly after generated_at
        time.sleep(0.1)
        _write_file(self.temp_dir, "new_file.py", "x = 1\n")
        self.assertTrue(self.cache.is_stale(manifest))

    def test_query_files(self):
        self.cache.refresh()
        result = self.cache.query("files")
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_query_files_with_filter(self):
        self.cache.refresh()
        result = self.cache.query("files", "path", "engine")
        self.assertTrue(all("engine" in f["path"] for f in result))

    def test_query_unknown_section(self):
        self.cache.refresh()
        result = self.cache.query("nonexistent")
        self.assertIn("error", result)

    def test_query_stats(self):
        self.cache.refresh()
        result = self.cache.query("stats")
        self.assertIn("total_files", result)

    def test_build_summary(self):
        self.cache.refresh()
        summary = self.cache.build_summary()
        self.assertIn("stats", summary)
        self.assertIn("files_per_directory", summary)
        self.assertIn("tools_per_category", summary)
        self.assertIn("endpoints_per_route", summary)

    def test_build_summary_directory_counts(self):
        self.cache.refresh()
        summary = self.cache.build_summary()
        dirs = summary["files_per_directory"]
        self.assertIn("core", dirs)
        self.assertIn("backend", dirs)

    def test_corrupt_disk_file_handled(self):
        """Corrupt JSON on disk should not crash — returns None."""
        with open(self.manifest_path, "w") as f:
            f.write("not json {{{")
        result = self.cache.get()
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Test: ManifestGenerator on real codebase (integration)
# ---------------------------------------------------------------------------

class TestRealCodebase(unittest.TestCase):
    """Integration tests against the actual AgentNate codebase."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.gen = ManifestGenerator(project_root=cls.project_root)
        cls.manifest = cls.gen.generate()

    def test_real_files_found(self):
        paths = [f["path"] for f in self.manifest["files"]]
        self.assertTrue(any("backend/server.py" in p for p in paths))
        self.assertTrue(any("orchestrator/orchestrator.py" in p for p in paths))

    def test_real_tools_found(self):
        tool_cats = self.manifest["tools"]
        all_tools = []
        for cat in tool_cats:
            all_tools.extend(t["name"] for t in cat.get("tools", []))
        self.assertIn("web_search", all_tools)
        self.assertIn("deploy_workflow", all_tools)
        self.assertIn("generate_manifest", all_tools)
        self.assertIn("query_codebase", all_tools)

    def test_real_endpoints_found(self):
        endpoints = self.manifest["endpoints"]
        paths = [e["path"] for e in endpoints]
        # Should find at least some standard endpoints
        self.assertTrue(len(endpoints) > 10)

    def test_real_personas_found(self):
        personas = self.manifest["personas"]
        ids = [p["id"] for p in personas]
        self.assertIn("system_agent", ids)
        self.assertIn("codebase_guide", ids)

    def test_real_providers_found(self):
        providers = self.manifest["providers"]
        values = [p["value"] for p in providers]
        self.assertIn("llama_cpp", values)
        self.assertIn("vllm", values)

    def test_real_tool_groups_found(self):
        groups = self.manifest["tool_groups"]
        self.assertIn("codebase", groups)
        self.assertIn("generate_manifest", groups["codebase"])
        self.assertIn("query_codebase", groups["codebase"])

    def test_real_route_prefixes_found(self):
        prefixes = self.manifest["route_prefixes"]
        prefix_strs = [p["prefix"] for p in prefixes]
        self.assertIn("/api/models", prefix_strs)
        self.assertIn("/api/chat", prefix_strs)

    def test_real_stats_reasonable(self):
        stats = self.manifest["stats"]
        self.assertGreater(stats["total_files"], 30)
        self.assertGreater(stats["total_tools"], 50)
        self.assertGreater(stats["total_endpoints"], 20)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
