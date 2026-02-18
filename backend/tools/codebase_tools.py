"""
Codebase Tools - Explore and understand the AgentNate codebase.

These tools help users understand the codebase structure, find features,
and learn how different components work together.
"""

from typing import Dict, Any, List, Optional
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger("tools.codebase")
AGENTNATE_BASE = os.getenv("AGENTNATE_BASE_URL", "http://127.0.0.1:8000")


TOOL_DEFINITIONS = [
    {
        "name": "scan_codebase",
        "description": "Scan the codebase and return the directory structure with file descriptions. Good starting point for understanding the project.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Subdirectory to scan (default: entire project)",
                    "default": "."
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum directory depth to scan (default 3)",
                    "default": 3
                },
                "include_descriptions": {
                    "type": "boolean",
                    "description": "Include brief file descriptions from docstrings (default true)",
                    "default": True
                }
            },
            "required": []
        }
    },
    {
        "name": "explain_file",
        "description": "Read a source file and explain what it does, including its classes, functions, and purpose.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to explain"
                },
                "detail_level": {
                    "type": "string",
                    "description": "Level of detail: 'summary' (quick overview), 'detailed' (full explanation), 'api' (public interfaces only)",
                    "enum": ["summary", "detailed", "api"],
                    "default": "summary"
                }
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "find_feature",
        "description": "Search for where a feature or functionality is implemented in the codebase.",
        "parameters": {
            "type": "object",
            "properties": {
                "feature": {
                    "type": "string",
                    "description": "Feature to find (e.g., 'model loading', 'chat endpoint', 'GPU detection')"
                },
                "search_type": {
                    "type": "string",
                    "description": "Search approach: 'smart' (AI-guided), 'grep' (text search), 'both'",
                    "enum": ["smart", "grep", "both"],
                    "default": "both"
                }
            },
            "required": ["feature"]
        }
    },
    {
        "name": "get_architecture",
        "description": "Get a high-level overview of the AgentNate architecture, components, and how they interact.",
        "parameters": {
            "type": "object",
            "properties": {
                "component": {
                    "type": "string",
                    "description": "Specific component to focus on (optional). Options: 'all', 'backend', 'orchestrator', 'providers', 'tools', 'api'",
                    "enum": ["all", "backend", "orchestrator", "providers", "tools", "api"],
                    "default": "all"
                }
            },
            "required": []
        }
    },
    {
        "name": "list_api_endpoints",
        "description": "List all API endpoints with their methods, paths, and descriptions.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Filter by category (optional)",
                    "enum": ["all", "chat", "models", "system", "workflows", "tools", "settings"],
                    "default": "all"
                }
            },
            "required": []
        }
    },
    {
        "name": "list_tools",
        "description": "List all available Meta Agent tools with their descriptions and parameters.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Filter by tool category (optional)",
                    "default": "all"
                },
                "verbose": {
                    "type": "boolean",
                    "description": "Include full parameter details (default false)",
                    "default": False
                }
            },
            "required": []
        }
    },
    {
        "name": "explain_concept",
        "description": "Explain a concept or term used in AgentNate (e.g., 'orchestrator', 'provider', 'persona', 'instance').",
        "parameters": {
            "type": "object",
            "properties": {
                "concept": {
                    "type": "string",
                    "description": "The concept to explain"
                }
            },
            "required": ["concept"]
        }
    },
    {
        "name": "get_capabilities",
        "description": "Get a friendly overview of what AgentNate can do. Perfect for new users who want to understand the system's capabilities organized by use case.",
        "parameters": {
            "type": "object",
            "properties": {
                "focus": {
                    "type": "string",
                    "description": "Optional area to focus on",
                    "enum": ["all", "ai_models", "automation", "research", "coding", "communication", "system"],
                    "default": "all"
                }
            },
            "required": []
        }
    },
    {
        "name": "get_quick_start",
        "description": "Get a step-by-step quick start guide for new users. Shows how to get up and running with AgentNate.",
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "What the user wants to accomplish (optional - provides personalized steps)",
                    "default": "general"
                }
            },
            "required": []
        }
    },
    {
        "name": "generate_manifest",
        "description": "Generate or refresh the codebase manifest — a structured index of all files, classes, endpoints, tools, personas, and providers in AgentNate. Returns stats and a summary.",
        "parameters": {
            "type": "object",
            "properties": {
                "force": {
                    "type": "boolean",
                    "description": "Force regeneration even if manifest is fresh (default: false)",
                    "default": False
                }
            },
            "required": []
        }
    },
    {
        "name": "query_codebase",
        "description": "Query the codebase manifest for structured information. Sections: 'files', 'endpoints', 'tools', 'personas', 'providers', 'tool_groups', 'route_prefixes', 'stats'. Optionally filter by a field value.",
        "parameters": {
            "type": "object",
            "properties": {
                "section": {
                    "type": "string",
                    "description": "Manifest section to query",
                    "enum": ["files", "endpoints", "tools", "personas", "providers", "tool_groups", "route_prefixes", "stats", "summary"]
                },
                "filter_field": {
                    "type": "string",
                    "description": "Field to filter on (e.g. 'name', 'path', 'method')"
                },
                "filter_value": {
                    "type": "string",
                    "description": "Value to match (case-insensitive substring)"
                },
                "detail": {
                    "type": "string",
                    "description": "Detail level: 'brief' (names only) or 'full' (all fields)",
                    "enum": ["brief", "full"],
                    "default": "brief"
                }
            },
            "required": ["section"]
        }
    },
]




# Concept explanations
CONCEPTS = {
    "orchestrator": "The central coordinator that manages all model instances and routes inference requests. It tracks which models are loaded on which GPUs and handles the lifecycle of model instances.",

    "provider": "An integration with an LLM backend. AgentNate supports multiple providers: llama_cpp (direct GGUF loading), lm_studio (LM Studio API), ollama (Ollama API), and openrouter (cloud API). Each provider handles model loading and inference differently.",

    "instance": "A loaded model. When you load a model, it creates an instance with a unique ID. You can have multiple instances of different models (or even the same model) loaded simultaneously on different GPUs.",

    "persona": "A personality configuration for the Meta Agent. Defines the system prompt, available tools, temperature, and behavior. Examples: System Agent (full control), Research Agent (web focused), Code Agent (execution focused).",

    "tool": "A function the Meta Agent can call to perform actions. Tools are grouped into categories (web, files, code, etc.) and personas can have access to different tool sets.",

    "meta agent": "The AI assistant that can use tools to help users. It receives a system prompt with available tools, analyzes user requests, and calls appropriate tools to accomplish tasks.",

    "tool group": "A named collection of related tools. Examples: 'web' (web_search, fetch_url, browser tools), 'code' (run_python, run_shell), 'files' (read_file, write_file). Personas reference tool groups for easy configuration.",

    "n8n": "An open-source workflow automation tool. AgentNate can spawn n8n instances and create workflows programmatically. The Meta Agent can design and deploy automation workflows.",

    "gguf": "A file format for quantized LLM models. Used by llama.cpp. AgentNate can load GGUF files directly using the llama_cpp provider.",

    "inference": "The process of generating text from an LLM. When you send a chat message, the model performs inference to generate a response.",

    "streaming": "Sending the model's response token-by-token as it's generated, rather than waiting for the complete response. Provides faster perceived response time.",

    "context length": "The maximum number of tokens a model can process in a single request (prompt + response). Larger context = more conversation history but more VRAM needed.",

    "gpu index": "Which GPU to use for a model. 0 = first GPU, 1 = second GPU, -1 = CPU only. AgentNate supports loading different models on different GPUs.",

    "vision model": "An LLM that can process images as well as text. Examples: LLaVA, Qwen-VL. AgentNate supports vision through the ChatMessage.images field.",
}


class CodebaseTools:
    """Tools for exploring and understanding the AgentNate codebase."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize codebase tools.

        Args:
            config: Optional configuration with:
                - project_root: str - Root directory of the project
        """
        self.config = config or {}
        root = self.config.get("project_root")
        self.project_root = Path(root) if root else Path(__file__).parent.parent.parent

        # Directories to exclude from scanning
        self.exclude_dirs = {
            "__pycache__", ".git", "node_modules", ".n8n-instances",
            "python", "_archive", "xxx", ".claude", "venv", ".venv"
        }

        # File extensions to include
        self.include_extensions = {".py", ".ts", ".tsx", ".js", ".jsx", ".json", ".md"}

    async def scan_codebase(
        self,
        path: str = ".",
        max_depth: int = 3,
        include_descriptions: bool = True
    ) -> Dict[str, Any]:
        """
        Scan the codebase structure.

        Args:
            path: Subdirectory to scan
            max_depth: Maximum depth
            include_descriptions: Include file descriptions

        Returns:
            Dict with codebase structure
        """
        try:
            scan_path = self.project_root / path

            if not scan_path.exists():
                return {"success": False, "error": f"Path not found: {path}"}

            structure = self._scan_directory(scan_path, max_depth, include_descriptions)

            return {
                "success": True,
                "root": str(scan_path),
                "structure": structure
            }

        except Exception as e:
            logger.error(f"scan_codebase error: {e}")
            return {"success": False, "error": str(e)}

    def _scan_directory(
        self,
        path: Path,
        max_depth: int,
        include_descriptions: bool,
        current_depth: int = 0
    ) -> Dict[str, Any]:
        """Recursively scan a directory."""
        if current_depth >= max_depth:
            return {"type": "directory", "truncated": True}

        result = {"type": "directory", "children": {}}

        try:
            for item in sorted(path.iterdir()):
                if item.name.startswith(".") and item.name not in [".env.example"]:
                    continue
                if item.name in self.exclude_dirs:
                    continue

                if item.is_dir():
                    result["children"][item.name + "/"] = self._scan_directory(
                        item, max_depth, include_descriptions, current_depth + 1
                    )
                elif item.suffix in self.include_extensions:
                    file_info = {"type": "file", "size": item.stat().st_size}

                    if include_descriptions and item.suffix == ".py":
                        desc = self._get_file_description(item)
                        if desc:
                            file_info["description"] = desc

                    result["children"][item.name] = file_info

        except PermissionError:
            result["error"] = "Permission denied"

        return result

    def _get_file_description(self, path: Path) -> Optional[str]:
        """Extract module docstring from a Python file."""
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            # Look for module docstring
            match = re.match(r'^[\s]*["\'][\"\'][\"\'](.+?)["\'][\"\'][\"\']', content, re.DOTALL)
            if match:
                desc = match.group(1).strip().split("\n")[0]
                return desc[:100]

            # Try single-line docstring
            match = re.match(r'^[\s]*["\'](.+?)["\']', content)
            if match:
                return match.group(1)[:100]

        except Exception:
            pass
        return None

    async def explain_file(
        self,
        file_path: str,
        detail_level: str = "summary"
    ) -> Dict[str, Any]:
        """
        Read and explain a source file.

        Args:
            file_path: Path to file
            detail_level: Level of detail

        Returns:
            Dict with file explanation
        """
        try:
            full_path = self.project_root / file_path

            if not full_path.exists():
                return {"success": False, "error": f"File not found: {file_path}"}

            content = full_path.read_text(encoding="utf-8", errors="ignore")

            # Extract structure based on file type
            if full_path.suffix == ".py":
                structure = self._analyze_python_file(content, detail_level)
            else:
                structure = {"content_preview": content[:2000]}

            return {
                "success": True,
                "file": file_path,
                "size": len(content),
                "lines": content.count("\n") + 1,
                "detail_level": detail_level,
                "analysis": structure,
                "content": content if detail_level == "detailed" else None
            }

        except Exception as e:
            logger.error(f"explain_file error: {e}")
            return {"success": False, "error": str(e)}

    def _analyze_python_file(self, content: str, detail_level: str) -> Dict[str, Any]:
        """Analyze a Python file structure."""
        import ast

        result = {
            "docstring": None,
            "imports": [],
            "classes": [],
            "functions": [],
            "constants": []
        }

        try:
            tree = ast.parse(content)

            # Module docstring
            if (tree.body and isinstance(tree.body[0], ast.Expr) and
                isinstance(tree.body[0].value, ast.Constant)):
                result["docstring"] = tree.body[0].value.value

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        result["imports"].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        result["imports"].append(f"from {node.module}")

                elif isinstance(node, ast.ClassDef):
                    class_info = {"name": node.name, "methods": []}
                    if detail_level != "summary":
                        # Get docstring
                        if (node.body and isinstance(node.body[0], ast.Expr) and
                            isinstance(node.body[0].value, ast.Constant)):
                            class_info["docstring"] = node.body[0].value.value

                        # Get methods
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef):
                                method_info = {"name": item.name}
                                if detail_level == "api" and item.name.startswith("_"):
                                    continue
                                class_info["methods"].append(method_info)

                    result["classes"].append(class_info)

                elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                    # Top-level function
                    if detail_level == "api" and node.name.startswith("_"):
                        continue
                    func_info = {"name": node.name}
                    result["functions"].append(func_info)

                elif isinstance(node, ast.Assign) and node.col_offset == 0:
                    # Top-level constant
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id.isupper():
                            result["constants"].append(target.id)

        except SyntaxError:
            result["error"] = "Could not parse Python file"

        return result

    async def find_feature(
        self,
        feature: str,
        search_type: str = "both"
    ) -> Dict[str, Any]:
        """
        Find where a feature is implemented.

        Args:
            feature: Feature description
            search_type: Search approach

        Returns:
            Dict with search results
        """
        results = {
            "success": True,
            "feature": feature,
            "matches": []
        }

        # Feature to search term mapping
        feature_patterns = {
            "model loading": ["load_model", "LoadModel", "model.*load"],
            "chat": ["chat", "completions", "inference"],
            "gpu": ["gpu", "cuda", "nvidia", "vram"],
            "tool": ["tool", "TOOL_DEFINITIONS", "execute"],
            "persona": ["persona", "Persona", "system_prompt"],
            "n8n": ["n8n", "workflow", "spawn"],
            "streaming": ["stream", "yield", "async for"],
            "vision": ["vision", "image", "multimodal"],
            "browser": ["browser", "playwright", "screenshot"],
        }

        # Find relevant search terms
        search_terms = []
        feature_lower = feature.lower()
        for key, patterns in feature_patterns.items():
            if key in feature_lower:
                search_terms.extend(patterns)

        if not search_terms:
            # Use the feature description directly
            search_terms = feature.split()[:3]

        # Search files
        for term in search_terms[:5]:  # Limit search terms
            matches = self._grep_codebase(term)
            for match in matches[:10]:  # Limit matches per term
                if match not in results["matches"]:
                    results["matches"].append(match)

        # Sort by relevance (number of matches in file)
        results["matches"] = results["matches"][:20]
        results["search_terms"] = search_terms

        return results

    def _grep_codebase(self, pattern: str) -> List[Dict]:
        """Search for pattern in codebase."""
        matches = []

        for root, dirs, files in os.walk(self.project_root):
            # Filter directories
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]

            for file in files:
                if not any(file.endswith(ext) for ext in [".py", ".ts", ".tsx", ".js"]):
                    continue

                file_path = Path(root) / file
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    if re.search(pattern, content, re.IGNORECASE):
                        rel_path = file_path.relative_to(self.project_root)
                        # Find line numbers
                        line_matches = []
                        for i, line in enumerate(content.split("\n"), 1):
                            if re.search(pattern, line, re.IGNORECASE):
                                line_matches.append({"line": i, "content": line.strip()[:100]})
                                if len(line_matches) >= 3:
                                    break

                        matches.append({
                            "file": str(rel_path),
                            "matches": line_matches
                        })

                except Exception:
                    continue

        return matches

    async def get_architecture(
        self,
        component: str = "all"
    ) -> Dict[str, Any]:
        """
        Dynamically generate architecture overview by scanning the codebase.

        Args:
            component: Component to focus on (all, backend, orchestrator, providers, tools, api)

        Returns:
            Dict with architecture info generated from actual code
        """
        try:
            overview_parts = []

            if component in ("all", "overview"):
                overview_parts.append(self._generate_project_overview())

            if component in ("all", "backend"):
                overview_parts.append(self._generate_component_overview("backend"))

            if component in ("all", "orchestrator"):
                overview_parts.append(self._generate_component_overview("orchestrator"))

            if component in ("all", "providers"):
                overview_parts.append(self._generate_component_overview("providers"))

            if component in ("all", "tools"):
                overview_parts.append(self._generate_tools_overview())

            if component in ("all", "api"):
                overview_parts.append(await self._generate_api_overview())

            return {
                "success": True,
                "component": component,
                "overview": "\n\n".join(overview_parts)
            }

        except Exception as e:
            logger.error(f"get_architecture error: {e}")
            return {"success": False, "error": str(e)}

    def _generate_project_overview(self) -> str:
        """Generate high-level project overview from directory structure."""
        lines = ["# AgentNate Project Overview", ""]
        lines.append("*Dynamically generated from codebase*\n")

        # Scan top-level directories
        dirs_found = []
        for item in sorted(self.project_root.iterdir()):
            if item.is_dir() and item.name not in self.exclude_dirs and not item.name.startswith("."):
                py_files = list(item.glob("*.py"))
                if py_files or item.name in ("backend", "orchestrator", "providers", "core", "settings"):
                    desc = self._get_dir_description(item)
                    dirs_found.append((item.name, desc, len(py_files)))

        lines.append("## Project Structure\n")
        for name, desc, file_count in dirs_found:
            lines.append(f"- **{name}/** - {desc} ({file_count} Python files)")

        # Key files in root
        root_py = [f.name for f in self.project_root.glob("*.py") if not f.name.startswith("test_")]
        if root_py:
            lines.append(f"\n**Root files:** {', '.join(root_py)}")

        return "\n".join(lines)

    def _get_dir_description(self, dir_path: Path) -> str:
        """Get description for a directory from __init__.py or first file."""
        init_file = dir_path / "__init__.py"
        if init_file.exists():
            desc = self._get_file_description(init_file)
            if desc:
                return desc

        # Try first .py file
        for py_file in sorted(dir_path.glob("*.py")):
            if py_file.name != "__init__.py":
                desc = self._get_file_description(py_file)
                if desc:
                    return desc
                break

        # Default descriptions based on common names
        defaults = {
            "backend": "FastAPI server, routes, and tools",
            "orchestrator": "Model management and inference routing",
            "providers": "LLM provider integrations",
            "core": "Shared utilities and base classes",
            "settings": "Configuration management",
            "routes": "API endpoint handlers",
            "tools": "Meta Agent tool implementations",
        }
        return defaults.get(dir_path.name, "Project component")

    def _generate_component_overview(self, component: str) -> str:
        """Generate overview for a specific component by scanning its files."""
        comp_path = self.project_root / component
        if not comp_path.exists():
            return f"# {component.title()}\n\n*Directory not found*"

        lines = [f"# {component.title()} Component (`{component}/`)", ""]

        # Get component description
        desc = self._get_dir_description(comp_path)
        lines.append(f"{desc}\n")

        lines.append("## Files\n")

        # Scan all Python files
        for py_file in sorted(comp_path.glob("*.py")):
            if py_file.name == "__init__.py":
                continue

            file_info = self._analyze_file_quick(py_file)
            lines.append(f"### {py_file.name}")
            if file_info.get("docstring"):
                lines.append(f"{file_info['docstring']}\n")

            if file_info.get("classes"):
                lines.append("**Classes:**")
                for cls in file_info["classes"][:10]:  # Limit
                    lines.append(f"- `{cls['name']}` - {cls.get('docstring', '')[:80]}")

            if file_info.get("functions"):
                funcs = [f for f in file_info["functions"] if not f["name"].startswith("_")]
                if funcs:
                    lines.append("**Functions:**")
                    for func in funcs[:10]:
                        lines.append(f"- `{func['name']}()`")

            lines.append("")

        # Scan subdirectories
        for subdir in sorted(comp_path.iterdir()):
            if subdir.is_dir() and subdir.name not in self.exclude_dirs:
                py_count = len(list(subdir.glob("*.py")))
                if py_count > 0:
                    lines.append(f"### {subdir.name}/ ({py_count} files)")
                    sub_desc = self._get_dir_description(subdir)
                    lines.append(f"{sub_desc}\n")

        return "\n".join(lines)

    def _analyze_file_quick(self, file_path: Path) -> Dict[str, Any]:
        """Quick analysis of a Python file."""
        import ast

        result = {"docstring": None, "classes": [], "functions": []}

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(content)

            # Module docstring
            if (tree.body and isinstance(tree.body[0], ast.Expr) and
                isinstance(tree.body[0].value, ast.Constant)):
                doc = tree.body[0].value.value
                result["docstring"] = doc.strip().split("\n")[0][:100]

            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef):
                    cls_info = {"name": node.name, "docstring": ""}
                    # Get class docstring
                    if (node.body and isinstance(node.body[0], ast.Expr) and
                        isinstance(node.body[0].value, ast.Constant)):
                        cls_info["docstring"] = node.body[0].value.value.strip().split("\n")[0]
                    result["classes"].append(cls_info)

                elif isinstance(node, ast.FunctionDef):
                    result["functions"].append({"name": node.name})

        except Exception:
            pass

        return result

    def _generate_tools_overview(self) -> str:
        """Generate tools overview by scanning the tools directory."""
        tools_path = self.project_root / "backend" / "tools"
        if not tools_path.exists():
            return "# Tools\n\n*Tools directory not found*"

        lines = ["# Tools (`backend/tools/`)", ""]
        lines.append("Meta Agent tools organized by category.\n")

        tool_files = []
        for py_file in sorted(tools_path.glob("*_tools.py")):
            if py_file.name in ("__init__.py",):
                continue

            # Extract TOOL_DEFINITIONS from file
            tools = self._extract_tool_definitions(py_file)
            if tools:
                tool_files.append((py_file.name, tools))

        for filename, tools in tool_files:
            category = filename.replace("_tools.py", "").replace("_", " ").title()
            lines.append(f"## {category} Tools (`{filename}`)")
            for tool in tools:
                lines.append(f"- `{tool['name']}` - {tool.get('description', '')[:80]}")
            lines.append("")

        return "\n".join(lines)

    def _extract_tool_definitions(self, file_path: Path) -> List[Dict]:
        """Extract TOOL_DEFINITIONS from a tools file."""
        tools = []
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")

            # Find TOOL_DEFINITIONS list
            import ast
            tree = ast.parse(content)

            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == "TOOL_DEFINITIONS":
                            # Evaluate the list
                            if isinstance(node.value, ast.List):
                                for item in node.value.elts:
                                    if isinstance(item, ast.Dict):
                                        tool = {}
                                        for key, value in zip(item.keys, item.values):
                                            if isinstance(key, ast.Constant) and key.value in ("name", "description"):
                                                if isinstance(value, ast.Constant):
                                                    tool[key.value] = value.value
                                        if tool.get("name"):
                                            tools.append(tool)
        except Exception:
            pass

        return tools

    async def _generate_api_overview(self) -> str:
        """Generate API overview by scanning route files."""
        routes_path = self.project_root / "backend" / "routes"
        if not routes_path.exists():
            return "# API Endpoints\n\n*Routes directory not found*"

        lines = ["# API Endpoints", ""]
        lines.append("*Extracted from route files*\n")

        for py_file in sorted(routes_path.glob("*.py")):
            if py_file.name == "__init__.py":
                continue

            endpoints = self._extract_endpoints(py_file)
            if endpoints:
                category = py_file.stem.title()
                lines.append(f"## {category} (`/api/{py_file.stem}/`)\n")
                for ep in endpoints:
                    lines.append(f"- `{ep['method']} {ep['path']}` - {ep.get('name', '')}")
                lines.append("")

        return "\n".join(lines)

    def _extract_endpoints(self, file_path: Path) -> List[Dict]:
        """Extract API endpoints from a route file."""
        endpoints = []
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")

            # Find @router.get/post/etc decorators
            import re
            pattern = r'@router\.(get|post|put|delete|patch)\s*\(\s*["\']([^"\']+)["\']'
            matches = re.findall(pattern, content, re.IGNORECASE)

            # Also find function names
            func_pattern = r'@router\.\w+\([^)]*\)\s*\nasync\s+def\s+(\w+)'
            func_matches = re.findall(func_pattern, content)

            for i, (method, path) in enumerate(matches):
                ep = {"method": method.upper(), "path": path}
                if i < len(func_matches):
                    ep["name"] = func_matches[i].replace("_", " ")
                endpoints.append(ep)

        except Exception:
            pass

        return endpoints

    async def list_api_endpoints(
        self,
        category: str = "all"
    ) -> Dict[str, Any]:
        """
        Dynamically list API endpoints by scanning route files.

        Args:
            category: Filter category

        Returns:
            Dict with endpoints
        """
        try:
            overview = await self._generate_api_overview()

            return {
                "success": True,
                "category": category,
                "endpoints": overview
            }

        except Exception as e:
            logger.error(f"list_api_endpoints error: {e}")
            return {"success": False, "error": str(e)}

    async def list_tools(
        self,
        category: str = "all",
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        List available Meta Agent tools.

        Args:
            category: Filter category
            verbose: Include full details

        Returns:
            Dict with tool list
        """
        from backend.tools import AVAILABLE_TOOLS
        from backend.personas import TOOL_GROUPS

        tools_by_category = {}

        # Build category mapping
        tool_to_category = {}
        for cat, tools in TOOL_GROUPS.items():
            if tools:  # Skip meta groups (None)
                for tool in tools:
                    tool_to_category[tool] = cat

        # Organize tools
        for tool in AVAILABLE_TOOLS:
            cat = tool_to_category.get(tool["name"], "other")
            if category != "all" and cat != category:
                continue

            if cat not in tools_by_category:
                tools_by_category[cat] = []

            tool_info = {
                "name": tool["name"],
                "description": tool["description"]
            }

            if verbose:
                tool_info["parameters"] = tool.get("parameters", {})

            tools_by_category[cat].append(tool_info)

        return {
            "success": True,
            "category": category,
            "total": sum(len(t) for t in tools_by_category.values()),
            "tools": tools_by_category
        }

    async def explain_concept(
        self,
        concept: str
    ) -> Dict[str, Any]:
        """
        Explain a concept.

        Args:
            concept: Concept to explain

        Returns:
            Dict with explanation
        """
        concept_lower = concept.lower().replace(" ", "_").replace("-", "_")

        # Direct match
        if concept_lower in CONCEPTS:
            return {
                "success": True,
                "concept": concept,
                "explanation": CONCEPTS[concept_lower]
            }

        # Fuzzy match
        for key, value in CONCEPTS.items():
            if concept_lower in key or key in concept_lower:
                return {
                    "success": True,
                    "concept": concept,
                    "matched": key,
                    "explanation": value
                }

        # Not found
        return {
            "success": True,
            "concept": concept,
            "explanation": f"No specific explanation for '{concept}'. Try: {', '.join(CONCEPTS.keys())}",
            "available_concepts": list(CONCEPTS.keys())
        }

    async def get_capabilities(
        self,
        focus: str = "all"
    ) -> Dict[str, Any]:
        """
        Get a friendly overview of AgentNate capabilities for new users.

        Args:
            focus: Area to focus on (all, ai_models, automation, research, coding, communication, system)

        Returns:
            Dict with capabilities overview
        """
        capabilities = {
            "ai_models": {
                "title": "AI Model Management",
                "description": "Load and run AI models locally or via cloud APIs",
                "what_you_can_do": [
                    "Load GGUF models directly (llama.cpp)",
                    "Connect to LM Studio for model management",
                    "Use Ollama models",
                    "Access cloud models via OpenRouter",
                    "Run multiple models simultaneously on different GPUs",
                    "Stream responses in real-time"
                ],
                "example_commands": [
                    "Load a model: 'Load llama3 on GPU 0'",
                    "Check status: 'What models are loaded?'",
                    "Switch models: 'Unload the current model and load mistral'"
                ]
            },
            "automation": {
                "title": "Workflow Automation",
                "description": "Create and manage automated workflows with n8n integration",
                "what_you_can_do": [
                    "Generate n8n workflows from natural language descriptions",
                    "Spawn multiple n8n instances for different projects",
                    "Create automated data pipelines",
                    "Schedule recurring tasks",
                    "Connect APIs and services together"
                ],
                "example_commands": [
                    "'Create a workflow that monitors RSS feeds and sends Discord notifications'",
                    "'Spawn a new n8n instance for my email automation'",
                    "'Show my active workflows'"
                ]
            },
            "research": {
                "title": "Web Research & Data Gathering",
                "description": "Search the web, browse sites, and extract information",
                "what_you_can_do": [
                    "Search the web with web_search",
                    "Fetch and parse web pages",
                    "Control a browser (click, type, screenshot)",
                    "Extract text and data from websites",
                    "Analyze images and screenshots with vision AI"
                ],
                "example_commands": [
                    "'Search for recent news about AI regulation'",
                    "'Open example.com and take a screenshot'",
                    "'Extract all links from this page'"
                ]
            },
            "coding": {
                "title": "Code Execution & Development",
                "description": "Write and execute code in multiple languages",
                "what_you_can_do": [
                    "Execute Python code",
                    "Run JavaScript/Node.js",
                    "Execute shell commands",
                    "Run PowerShell scripts",
                    "Read and write code files",
                    "Search codebases"
                ],
                "example_commands": [
                    "'Run this Python script to process the CSV'",
                    "'List all files in the project directory'",
                    "'Find where authentication is implemented'"
                ]
            },
            "communication": {
                "title": "Messaging & Notifications",
                "description": "Send messages across multiple platforms",
                "what_you_can_do": [
                    "Send Discord messages/embeds",
                    "Post to Slack channels",
                    "Send emails",
                    "Send Telegram messages",
                    "Trigger webhooks"
                ],
                "example_commands": [
                    "'Send a Discord message to #general saying the build passed'",
                    "'Email the report to team@example.com'",
                    "'Post this update to Slack'"
                ]
            },
            "system": {
                "title": "System Management",
                "description": "Monitor and manage the AgentNate system",
                "what_you_can_do": [
                    "Check GPU status and memory usage",
                    "Monitor system health",
                    "View loaded models and their status",
                    "Get provider connectivity status",
                    "Manage n8n instances"
                ],
                "example_commands": [
                    "'Show GPU status'",
                    "'What's the system health?'",
                    "'Which providers are available?'"
                ]
            }
        }

        if focus != "all" and focus in capabilities:
            return {
                "success": True,
                "focus": focus,
                "capability": capabilities[focus]
            }

        # Return all capabilities with a welcome message
        return {
            "success": True,
            "welcome": "Welcome to AgentNate!",
            "summary": "AgentNate is a local AI orchestration platform that lets you run AI models, automate workflows, and use 70+ tools to accomplish tasks.",
            "capabilities": capabilities,
            "tip": "Select a persona from the sidebar to get specialized assistance. The 'Power Agent' has access to all tools, while specialized agents like 'Research Agent' or 'Code Agent' are optimized for specific tasks."
        }

    async def get_quick_start(
        self,
        goal: str = "general"
    ) -> Dict[str, Any]:
        """
        Get a step-by-step quick start guide.

        Args:
            goal: What the user wants to accomplish

        Returns:
            Dict with quick start guide
        """
        # Common first steps
        docs_url = f"{AGENTNATE_BASE}/docs"
        common_steps = [
            {
                "step": 1,
                "title": "Start the Server",
                "description": "Run AgentNate using the portable Python environment",
                "command": "E:\\AgentNate\\python\\python.exe run.py",
                "note": f"The server starts at {AGENTNATE_BASE}"
            },
            {
                "step": 2,
                "title": "Load a Model",
                "description": "Load an AI model to start chatting",
                "options": [
                    "Use LM Studio: Start LM Studio, load a model, then select 'lm_studio' provider in AgentNate",
                    "Use Ollama: Start Ollama, pull a model (ollama pull llama3), select 'ollama' provider",
                    "Use OpenRouter: Add your API key in settings.json, select 'openrouter' provider"
                ]
            },
            {
                "step": 3,
                "title": "Choose a Persona",
                "description": "Select an agent persona that matches your task",
                "personas": {
                    "Power Agent": "Full access to all 70+ tools",
                    "Research Agent": "Web search and data gathering",
                    "Code Agent": "Write and execute code",
                    "Codebase Guide": "Learn about AgentNate's code",
                    "General Assistant": "Simple chat without tools"
                }
            }
        ]

        # Goal-specific steps
        goal_steps = {
            "general": [
                {
                    "step": 4,
                    "title": "Start Chatting",
                    "description": "Send a message to the AI and it will use its tools to help you",
                    "examples": [
                        "'What can you do?' - Get an overview of capabilities",
                        "'Search for...' - Research a topic",
                        "'Create a workflow that...' - Automate a task"
                    ]
                }
            ],
            "research": [
                {
                    "step": 4,
                    "title": "Select Research Agent",
                    "description": "Choose the 'Research Agent' persona for optimized web research"
                },
                {
                    "step": 5,
                    "title": "Start Researching",
                    "description": "Ask questions and the agent will search the web and gather information",
                    "examples": [
                        "'Research the latest developments in quantum computing'",
                        "'Find and summarize the top 5 articles about...'",
                        "'Compare these products: X vs Y'"
                    ]
                }
            ],
            "automation": [
                {
                    "step": 4,
                    "title": "Select Workflow Builder or Automator",
                    "description": "Choose a persona specialized in automation"
                },
                {
                    "step": 5,
                    "title": "Describe Your Workflow",
                    "description": "Tell the agent what you want to automate in plain language",
                    "examples": [
                        "'Create a workflow that monitors my email and sends important ones to Slack'",
                        "'I want to automatically backup my database every night'",
                        "'Build a pipeline that processes uploaded CSVs'"
                    ]
                },
                {
                    "step": 6,
                    "title": "Deploy to n8n",
                    "description": "The agent will generate an n8n workflow and help you deploy it"
                }
            ],
            "coding": [
                {
                    "step": 4,
                    "title": "Select Code Agent",
                    "description": "Choose the 'Code Agent' persona for code execution"
                },
                {
                    "step": 5,
                    "title": "Write and Execute Code",
                    "description": "The agent can write, run, and debug code for you",
                    "examples": [
                        "'Write a Python script that processes this CSV file'",
                        "'Debug this code: [paste code]'",
                        "'Create a shell script that...'",
                        "'Run this JavaScript: console.log(2+2)'"
                    ]
                }
            ]
        }

        # Match goal to specific guide
        goal_lower = goal.lower()
        matched_goal = "general"
        for key in goal_steps.keys():
            if key in goal_lower:
                matched_goal = key
                break

        return {
            "success": True,
            "title": "AgentNate Quick Start Guide",
            "goal": matched_goal if matched_goal != "general" else "Getting Started",
            "steps": common_steps + goal_steps.get(matched_goal, goal_steps["general"]),
            "next_steps": [
                "Explore different personas to find the best fit for your tasks",
                "Try the Codebase Guide persona to learn more about how AgentNate works",
                f"Check the API docs at {docs_url} for direct API access"
            ],
            "resources": {
                "api_docs": docs_url,
                "readme": "README.md",
                "architecture": "CODEBASE_OVERVIEW.md"
            }
        }

    async def generate_manifest(self, force: bool = False) -> Dict[str, Any]:
        """
        Generate or refresh the codebase manifest.

        Args:
            force: Force regeneration even if manifest is fresh

        Returns:
            Dict with stats and summary
        """
        try:
            from backend.codebase_manifest import ManifestCache
            cache = ManifestCache(project_root=str(self.project_root))

            if force:
                manifest = cache.refresh()
            else:
                manifest = cache.get_or_refresh()

            summary = cache.build_summary()

            return {
                "success": True,
                "stats": manifest["stats"],
                "summary": summary,
            }
        except Exception as e:
            logger.error(f"generate_manifest error: {e}")
            return {"success": False, "error": str(e)}

    async def query_codebase(
        self,
        section: str,
        filter_field: Optional[str] = None,
        filter_value: Optional[str] = None,
        detail: str = "brief"
    ) -> Dict[str, Any]:
        """
        Query the codebase manifest for structured information.

        Args:
            section: Manifest section to query
            filter_field: Field to filter on
            filter_value: Value to match (case-insensitive substring)
            detail: 'brief' or 'full'

        Returns:
            Dict with query results
        """
        try:
            from backend.codebase_manifest import ManifestCache
            cache = ManifestCache(project_root=str(self.project_root))

            # Special "summary" section
            if section == "summary":
                summary = cache.build_summary()
                return {"success": True, "section": "summary", "data": summary}

            data = cache.query(section, filter_field, filter_value)

            if isinstance(data, dict) and "error" in data:
                return {"success": False, "error": data["error"]}

            # Apply brief/full detail level
            if detail == "brief":
                data = self._abbreviate(section, data)

            return {
                "success": True,
                "section": section,
                "filter": {"field": filter_field, "value": filter_value}
                         if filter_field else None,
                "count": len(data) if isinstance(data, list) else None,
                "data": data,
            }
        except Exception as e:
            logger.error(f"query_codebase error: {e}")
            return {"success": False, "error": str(e)}

    def _abbreviate(self, section: str, data: Any) -> Any:
        """Strip verbose fields for brief mode."""
        if section == "files" and isinstance(data, list):
            return [
                {
                    "path": f.get("path"),
                    "lines": f.get("lines"),
                    "docstring": f.get("docstring"),
                    "classes": [c.get("name") for c in f.get("classes", [])],
                    "functions": [fn.get("name") for fn in f.get("functions", [])],
                }
                for f in data
            ]
        if section == "tools" and isinstance(data, list):
            result = []
            for cat in data:
                tools = cat.get("tools", []) if isinstance(cat, dict) else []
                result.append({
                    "category": cat.get("category") if isinstance(cat, dict) else None,
                    "tools": [
                        {"name": t.get("name"), "description": t.get("description", "")[:80]}
                        for t in tools if isinstance(t, dict)
                    ],
                })
            return result
        if section == "endpoints" and isinstance(data, list):
            return [
                {"method": e.get("method"), "path": e.get("path"), "handler": e.get("handler")}
                for e in data
            ]
        return data
