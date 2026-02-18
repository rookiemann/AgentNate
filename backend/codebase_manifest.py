"""
Codebase Manifest — Pre-generated index of files, endpoints, tools, personas, providers.

ManifestGenerator walks the codebase once and builds a JSON manifest.
ManifestCache manages disk persistence and staleness detection.
Storage: data/codebase_manifest.json
"""

import ast
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("codebase_manifest")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MANIFEST_PATH = os.path.join(_PROJECT_ROOT, "data", "codebase_manifest.json")

# Directories to skip during file scanning
_EXCLUDE_DIRS = {
    "__pycache__", ".git", "node_modules", ".n8n-instances",
    "python", "_archive", "xxx", ".claude", "venv", ".venv",
    "envs", "vllm-source", "modules",
}


# ---------------------------------------------------------------------------
# ManifestGenerator
# ---------------------------------------------------------------------------

class ManifestGenerator:
    """Walks the codebase and builds a structured manifest."""

    def __init__(self, project_root: str = _PROJECT_ROOT):
        self.root = Path(project_root)

    def generate(self) -> Dict[str, Any]:
        """Generate the full codebase manifest."""
        files = self._scan_files()
        endpoints = self._extract_endpoints()
        tools = self._extract_tools()
        personas = self._extract_personas()
        providers = self._extract_providers()
        tool_groups = self._extract_tool_groups()
        route_prefixes = self._extract_route_prefixes()

        total_lines = sum(f.get("lines", 0) for f in files)
        total_classes = sum(len(f.get("classes", [])) for f in files)
        total_functions = sum(len(f.get("functions", [])) for f in files)
        total_tools = sum(len(cat.get("tools", [])) for cat in tools)

        stats = {
            "total_files": len(files),
            "total_lines": total_lines,
            "total_classes": total_classes,
            "total_functions": total_functions,
            "total_endpoints": len(endpoints),
            "total_tools": total_tools,
            "total_personas": len(personas),
            "generated_at": time.time(),
        }

        return {
            "files": files,
            "endpoints": endpoints,
            "tools": tools,
            "personas": personas,
            "providers": providers,
            "tool_groups": tool_groups,
            "route_prefixes": route_prefixes,
            "stats": stats,
        }

    # -- File scanning -------------------------------------------------------

    def _scan_files(self) -> List[Dict]:
        """Walk project tree and analyze all Python files."""
        results = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            # Prune excluded directories
            dirnames[:] = [
                d for d in dirnames
                if d not in _EXCLUDE_DIRS and not d.startswith(".")
            ]
            for fname in sorted(filenames):
                if not fname.endswith(".py"):
                    continue
                full_path = Path(dirpath) / fname
                rel_path = str(full_path.relative_to(self.root)).replace("\\", "/")
                info = self._analyze_file(full_path, rel_path)
                if info:
                    results.append(info)
        return results

    def _analyze_file(self, full_path: Path, rel_path: str) -> Optional[Dict]:
        """AST-parse a Python file and extract structure."""
        try:
            content = full_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return None

        info: Dict[str, Any] = {
            "path": rel_path,
            "lines": content.count("\n") + 1,
        }

        try:
            tree = ast.parse(content)
        except SyntaxError:
            info["error"] = "syntax_error"
            return info

        # Module docstring
        if (tree.body
                and isinstance(tree.body[0], ast.Expr)
                and isinstance(tree.body[0].value, ast.Constant)
                and isinstance(tree.body[0].value.value, str)):
            doc = tree.body[0].value.value.strip()
            info["docstring"] = doc.split("\n")[0][:200]

        classes = []
        functions = []
        constants = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(self._extract_class(node))
            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                functions.append(self._extract_function(node))
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        constants.append(target.id)

        if classes:
            info["classes"] = classes
        if functions:
            info["functions"] = functions
        if constants:
            info["constants"] = constants

        return info

    def _extract_class(self, node: ast.ClassDef) -> Dict:
        """Extract class info with methods."""
        cls: Dict[str, Any] = {"name": node.name}

        # Docstring
        if (node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)):
            cls["docstring"] = node.body[0].value.value.strip().split("\n")[0][:150]

        # Base classes
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(ast.dump(base))
        if bases:
            cls["bases"] = bases

        # Methods
        methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(self._extract_function(item))
        if methods:
            cls["methods"] = methods

        return cls

    def _extract_function(self, node) -> Dict:
        """Extract function/method info with signature."""
        func: Dict[str, Any] = {"name": node.name}

        if isinstance(node, ast.AsyncFunctionDef):
            func["is_async"] = True

        # Arguments
        args = self._extract_args(node.args)
        if args:
            func["args"] = args

        # Return annotation
        if node.returns:
            func["returns"] = self._annotation_str(node.returns)

        return func

    def _extract_args(self, args: ast.arguments) -> List[str]:
        """Extract function argument names with annotations."""
        result = []
        for arg in args.args:
            if arg.arg == "self":
                continue
            if arg.annotation:
                result.append(f"{arg.arg}: {self._annotation_str(arg.annotation)}")
            else:
                result.append(arg.arg)
        return result

    def _annotation_str(self, node) -> str:
        """Convert an AST annotation node to a readable string."""
        if isinstance(node, ast.Constant):
            return repr(node.value)
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{self._annotation_str(node.value)}.{node.attr}"
        if isinstance(node, ast.Subscript):
            base = self._annotation_str(node.value)
            slc = self._annotation_str(node.slice)
            return f"{base}[{slc}]"
        if isinstance(node, ast.Tuple):
            elts = ", ".join(self._annotation_str(e) for e in node.elts)
            return elts
        return "..."

    # -- Endpoint extraction -------------------------------------------------

    def _extract_endpoints(self) -> List[Dict]:
        """Extract API endpoints from backend/routes/*.py."""
        endpoints = []
        routes_dir = self.root / "backend" / "routes"
        if not routes_dir.exists():
            return endpoints

        for py_file in sorted(routes_dir.glob("*.py")):
            if py_file.name == "__init__.py":
                continue
            try:
                content = py_file.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            rel = str(py_file.relative_to(self.root)).replace("\\", "/")

            # Match @router.method("path") followed by async def handler_name
            pattern = r'@router\.(get|post|put|delete|patch)\s*\(\s*["\']([^"\']+)["\']'
            func_pattern = r'@router\.\w+\([^)]*\)\s*\nasync\s+def\s+(\w+)'

            methods = re.findall(pattern, content, re.IGNORECASE)
            funcs = re.findall(func_pattern, content)

            for i, (method, path) in enumerate(methods):
                ep = {
                    "method": method.upper(),
                    "path": path,
                    "route_file": rel,
                }
                if i < len(funcs):
                    ep["handler"] = funcs[i]
                endpoints.append(ep)

        return endpoints

    # -- Tool extraction -----------------------------------------------------

    def _extract_tools(self) -> List[Dict]:
        """Extract TOOL_DEFINITIONS from backend/tools/*_tools.py files."""
        categories = []
        tools_dir = self.root / "backend" / "tools"
        if not tools_dir.exists():
            return categories

        for py_file in sorted(tools_dir.glob("*_tools.py")):
            if py_file.name == "__init__.py":
                continue
            try:
                content = py_file.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            tools = self._parse_tool_definitions(content)
            if tools:
                category = py_file.stem.replace("_tools", "")
                categories.append({
                    "category": category,
                    "file": str(py_file.relative_to(self.root)).replace("\\", "/"),
                    "tools": tools,
                })

        return categories

    def _parse_tool_definitions(self, content: str) -> List[Dict]:
        """Parse TOOL_DEFINITIONS list from a tools file using AST."""
        tools = []
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return tools

        for node in ast.iter_child_nodes(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "TOOL_DEFINITIONS":
                    if isinstance(node.value, ast.List):
                        for item in node.value.elts:
                            tool = self._eval_tool_dict(item)
                            if tool and tool.get("name"):
                                tools.append(tool)
        return tools

    def _eval_tool_dict(self, node) -> Optional[Dict]:
        """Safely evaluate a tool definition dict from AST."""
        if not isinstance(node, ast.Dict):
            return None
        try:
            # Use ast.literal_eval on the source slice is unreliable,
            # so manually extract key-value pairs for simple constants
            result = {}
            for key, value in zip(node.keys, node.values):
                if not isinstance(key, ast.Constant):
                    continue
                k = key.value
                if k == "name" and isinstance(value, ast.Constant):
                    result["name"] = value.value
                elif k == "description" and isinstance(value, ast.Constant):
                    result["description"] = value.value
                elif k == "parameters" and isinstance(value, ast.Dict):
                    result["parameters"] = self._eval_simple_dict(value)
            return result
        except Exception:
            return None

    def _eval_simple_dict(self, node: ast.Dict) -> Dict:
        """Recursively evaluate a simple dict (constants, lists, dicts only)."""
        result = {}
        for key, value in zip(node.keys, node.values):
            if not isinstance(key, ast.Constant):
                continue
            result[key.value] = self._eval_simple_value(value)
        return result

    def _eval_simple_value(self, node) -> Any:
        """Evaluate a simple AST value (constant, list, dict)."""
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.List):
            return [self._eval_simple_value(e) for e in node.elts]
        if isinstance(node, ast.Dict):
            return self._eval_simple_dict(node)
        if isinstance(node, ast.Name):
            # Handle True, False, None
            if node.id == "True":
                return True
            if node.id == "False":
                return False
            if node.id == "None":
                return None
            return node.id  # Enum references etc.
        return None

    # -- Persona extraction --------------------------------------------------

    def _extract_personas(self) -> List[Dict]:
        """Extract persona definitions from backend/personas.py."""
        try:
            from backend.personas import PREDEFINED_PERSONAS
            return [
                {
                    "id": p.id,
                    "name": p.name,
                    "description": p.description,
                    "tools": p.tools,
                    "temperature": p.temperature,
                    "include_system_state": p.include_system_state,
                }
                for p in PREDEFINED_PERSONAS
            ]
        except Exception as e:
            logger.warning(f"Failed to extract personas: {e}")
            return []

    # -- Provider extraction -------------------------------------------------

    def _extract_providers(self) -> List[Dict]:
        """Extract provider types from providers/base.py."""
        try:
            from providers.base import ProviderType
            return [{"name": p.name, "value": p.value} for p in ProviderType]
        except Exception as e:
            logger.warning(f"Failed to extract providers: {e}")
            return []

    # -- Tool group extraction -----------------------------------------------

    def _extract_tool_groups(self) -> Dict[str, List[str]]:
        """Extract tool groups from backend/personas.py."""
        try:
            from backend.personas import TOOL_GROUPS
            return {
                k: v for k, v in TOOL_GROUPS.items()
                if v is not None
            }
        except Exception as e:
            logger.warning(f"Failed to extract tool groups: {e}")
            return {}

    # -- Route prefix extraction ---------------------------------------------

    def _extract_route_prefixes(self) -> List[Dict]:
        """Extract router mount points from backend/server.py."""
        server_path = self.root / "backend" / "server.py"
        if not server_path.exists():
            return []

        try:
            content = server_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return []

        # Match: app.include_router(xxx.router, prefix="/api/yyy", tags=["ZZZ"])
        pattern = r'app\.include_router\(\s*(\w+)\.router\s*,\s*prefix\s*=\s*["\']([^"\']+)["\']'
        tag_pattern = r'tags\s*=\s*\[\s*["\']([^"\']+)["\']\s*\]'

        results = []
        for match in re.finditer(pattern, content):
            module = match.group(1)
            prefix = match.group(2)
            # Look for tags in the same line region
            rest = content[match.end():match.end() + 100]
            tag_match = re.search(tag_pattern, rest)
            tag = tag_match.group(1) if tag_match else module
            results.append({"prefix": prefix, "module": module, "tag": tag})

        return results


# ---------------------------------------------------------------------------
# ManifestCache
# ---------------------------------------------------------------------------

class ManifestCache:
    """Manages disk persistence and staleness for the codebase manifest."""

    def __init__(self, project_root: str = _PROJECT_ROOT,
                 manifest_path: str = _MANIFEST_PATH):
        self.project_root = project_root
        self.manifest_path = manifest_path
        self._cache: Optional[Dict] = None

        # Ensure data directory exists
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    def get(self) -> Optional[Dict]:
        """Return cached manifest (memory first, then disk)."""
        if self._cache:
            return self._cache
        return self._load_from_disk()

    def refresh(self) -> Dict:
        """Force regenerate the manifest and save to disk."""
        generator = ManifestGenerator(self.project_root)
        manifest = generator.generate()
        self._cache = manifest
        self._save_to_disk(manifest)
        logger.info(
            f"Manifest regenerated: {manifest['stats']['total_files']} files, "
            f"{manifest['stats']['total_tools']} tools, "
            f"{manifest['stats']['total_endpoints']} endpoints"
        )
        return manifest

    def get_or_refresh(self) -> Dict:
        """Return cached manifest if fresh, regenerate if stale or missing."""
        cached = self.get()
        if cached and not self.is_stale(cached):
            return cached
        return self.refresh()

    def is_stale(self, manifest: Optional[Dict] = None) -> bool:
        """Check if any .py file has been modified since the manifest was generated."""
        if manifest is None:
            manifest = self.get()
        if not manifest:
            return True

        generated_at = manifest.get("stats", {}).get("generated_at", 0)
        if generated_at == 0:
            return True

        # Check key directories for recent modifications
        for dirpath, dirnames, filenames in os.walk(self.project_root):
            dirnames[:] = [
                d for d in dirnames
                if d not in _EXCLUDE_DIRS and not d.startswith(".")
            ]
            for fname in filenames:
                if not fname.endswith(".py"):
                    continue
                try:
                    mtime = os.path.getmtime(os.path.join(dirpath, fname))
                    if mtime > generated_at:
                        return True
                except OSError:
                    continue
        return False

    def query(self, section: str, filter_field: Optional[str] = None,
              filter_value: Optional[str] = None) -> Any:
        """
        Query a section of the manifest with optional filtering.

        Args:
            section: Manifest key to query (files, endpoints, tools, etc.)
            filter_field: Field name to filter on
            filter_value: Value to match (case-insensitive substring)

        Returns:
            Filtered section data
        """
        manifest = self.get_or_refresh()
        data = manifest.get(section)
        if data is None:
            return {"error": f"Unknown section: {section}"}

        if not filter_field or not filter_value:
            return data

        filter_lower = filter_value.lower()

        # Handle list of dicts (files, endpoints, personas, providers)
        if isinstance(data, list):
            return [
                item for item in data
                if isinstance(item, dict)
                and filter_lower in str(item.get(filter_field, "")).lower()
            ]

        # Handle tools (list of category dicts containing tool lists)
        if section == "tools" and isinstance(data, list):
            results = []
            for cat in data:
                matching = [
                    t for t in cat.get("tools", [])
                    if isinstance(t, dict)
                    and filter_lower in str(t.get(filter_field, "")).lower()
                ]
                if matching:
                    results.append({
                        "category": cat.get("category"),
                        "tools": matching,
                    })
            return results

        # Handle dict (tool_groups, stats)
        if isinstance(data, dict):
            return {
                k: v for k, v in data.items()
                if filter_lower in str(k).lower() or filter_lower in str(v).lower()
            }

        return data

    def build_summary(self) -> Dict[str, Any]:
        """Build a high-level summary of the codebase."""
        manifest = self.get_or_refresh()

        # Files per directory
        dir_counts: Dict[str, int] = {}
        for f in manifest.get("files", []):
            parts = f["path"].split("/")
            top_dir = parts[0] if len(parts) > 1 else "(root)"
            dir_counts[top_dir] = dir_counts.get(top_dir, 0) + 1

        # Tools per category
        tool_counts: Dict[str, int] = {}
        for cat in manifest.get("tools", []):
            tool_counts[cat["category"]] = len(cat.get("tools", []))

        # Endpoints per route file
        ep_counts: Dict[str, int] = {}
        for ep in manifest.get("endpoints", []):
            rf = ep.get("route_file", "unknown")
            ep_counts[rf] = ep_counts.get(rf, 0) + 1

        return {
            "stats": manifest.get("stats", {}),
            "files_per_directory": dir_counts,
            "tools_per_category": tool_counts,
            "endpoints_per_route": ep_counts,
            "personas": [
                {"id": p["id"], "name": p["name"]}
                for p in manifest.get("personas", [])
            ],
            "providers": [p["value"] for p in manifest.get("providers", [])],
        }

    def _load_from_disk(self) -> Optional[Dict]:
        """Load manifest from disk cache."""
        if not os.path.exists(self.manifest_path):
            return None
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                self._cache = json.load(f)
            return self._cache
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load manifest from disk: {e}")
            return None

    def _save_to_disk(self, manifest: Dict) -> None:
        """Save manifest to disk."""
        try:
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, default=str)
        except OSError as e:
            logger.error(f"Failed to save manifest to disk: {e}")
