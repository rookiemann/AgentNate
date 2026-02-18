"""
Workflow Tools - Generate and manage n8n workflows.
"""

from typing import Dict, Any, Optional, List
import json as json_module
import logging
import aiohttp
import os

logger = logging.getLogger("tools.workflow")
AGENTNATE_BASE = os.getenv("AGENTNATE_BASE_URL", "http://127.0.0.1:8000")

# Add file handler for debug logging
_debug_log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "workflow_debug.log")
_file_handler = logging.FileHandler(_debug_log_path, mode='a', encoding='utf-8')
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(_file_handler)
logger.setLevel(logging.DEBUG)


TOOL_DEFINITIONS = [
    {
        "name": "describe_node",
        "description": """Get parameter details for node types BEFORE building a workflow. Call this first to learn what params each node needs.

Returns: category, all parameters with types/defaults/required/options, and usage notes.

Use node_types: ["all"] to see every available node type grouped by category.""",
        "parameters": {
            "type": "object",
            "properties": {
                "node_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Node type names to describe (e.g. ['postgres', 'http_request']). Use ['all'] for complete list."
                }
            },
            "required": ["node_types"]
        }
    },
    {
        "name": "list_credentials",
        "description": "List credentials configured in the n8n instance. Call before building workflows that need external services (databases, cloud APIs, etc). Returns credential id, name, and type.",
        "parameters": {
            "type": "object",
            "properties": {
                "n8n_port": {
                    "type": "integer",
                    "description": "n8n instance port (default: 5678)"
                }
            },
            "required": []
        }
    },
    {
        "name": "build_workflow",
        "description": """Build an n8n workflow from node specs. Set deploy=true to auto-deploy to n8n (recommended).

IMPORTANT: Put params at TOP LEVEL of node spec (not nested in "parameters"):
  CORRECT: {"type": "http_request", "url": "https://example.com"}
  WRONG:   {"type": "http_request", "parameters": {"url": "..."}}

For branching workflows (if/switch/merge), provide connections list:
  connections: [{"from": "IF", "to": "True Path", "output": 0}, {"from": "IF", "to": "False Path", "output": 1}]

Set deploy=true to build AND deploy in one step. Set activate=true to also activate it.""",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name for the workflow"
                },
                "nodes": {
                    "type": "array",
                    "description": "List of node specs. Each node needs 'type' and optional params. Use describe_node to learn available params.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "description": "Node type (e.g., manual_trigger, http_request, postgres)"
                            }
                        },
                        "required": ["type"]
                    }
                },
                "connections": {
                    "type": "array",
                    "description": "Custom connections for branching workflows. Each: {from, to, output (default 0), input (default 0)}. If omitted, nodes connect linearly.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "from": {"type": "string", "description": "Source node name"},
                            "to": {"type": "string", "description": "Target node name"},
                            "output": {"type": "integer", "description": "Source output index (default 0)"},
                            "input": {"type": "integer", "description": "Target input index (default 0)"}
                        },
                        "required": ["from", "to"]
                    }
                },
                "deploy": {
                    "type": "boolean",
                    "description": "Auto-deploy to n8n after building (default: true). Eliminates need for separate deploy_workflow call."
                },
                "activate": {
                    "type": "boolean",
                    "description": "Activate the workflow after deploying (default: false). Only used when deploy=true."
                },
                "n8n_port": {
                    "type": "integer",
                    "description": "n8n instance port for deployment (default: 5678). Only used when deploy=true."
                }
            },
            "required": ["name", "nodes"]
        }
    },
    {
        "name": "deploy_workflow",
        "description": "Deploy a workflow to an n8n instance",
        "parameters": {
            "type": "object",
            "properties": {
                "workflow_json": {
                    "type": "object",
                    "description": "The workflow JSON to deploy"
                },
                "n8n_port": {
                    "type": "integer",
                    "description": "n8n instance port (default: 5678)"
                },
                "activate": {
                    "type": "boolean",
                    "description": "Whether to activate the workflow (default: false)"
                }
            },
            "required": ["workflow_json"]
        }
    },
    {
        "name": "list_workflows",
        "description": "List workflows in an n8n instance",
        "parameters": {
            "type": "object",
            "properties": {
                "n8n_port": {
                    "type": "integer",
                    "description": "n8n instance port (default: 5678)"
                }
            },
            "required": []
        }
    },
    {
        "name": "delete_workflow",
        "description": "Delete a specific workflow from n8n by ID",
        "parameters": {
            "type": "object",
            "properties": {
                "workflow_id": {
                    "type": "string",
                    "description": "The workflow ID to delete"
                },
                "n8n_port": {
                    "type": "integer",
                    "description": "n8n instance port (default: 5678)"
                }
            },
            "required": ["workflow_id"]
        }
    },
    {
        "name": "delete_all_workflows",
        "description": "Delete ALL workflows from an n8n instance. Use with caution!",
        "parameters": {
            "type": "object",
            "properties": {
                "n8n_port": {
                    "type": "integer",
                    "description": "n8n instance port (default: 5678)"
                },
                "confirm": {
                    "type": "boolean",
                    "description": "Must be true to confirm deletion"
                }
            },
            "required": ["confirm"]
        }
    },

    # ---- Credential Management ----
    {
        "name": "describe_credential_types",
        "description": "Discover what credential types are available in n8n and what data fields each type requires. Call this BEFORE create_credential to learn the schema. Like describe_node but for credentials.",
        "parameters": {
            "type": "object",
            "properties": {
                "filter": {
                    "type": "string",
                    "description": "Optional substring filter on type name (e.g. 'openai', 'http', 'postgres')"
                },
                "n8n_port": {
                    "type": "integer",
                    "description": "n8n instance port (default: 5678)"
                }
            },
            "required": []
        }
    },
    {
        "name": "create_credential",
        "description": "Create a new credential in n8n. Use describe_credential_types first to learn what data fields the type requires. n8n encrypts credential data at rest.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Display name for the credential (e.g. 'My OpenAI Key')"
                },
                "credential_type": {
                    "type": "string",
                    "description": "Credential type ID from describe_credential_types (e.g. 'openAiApi', 'httpHeaderAuth', 'postgresApi')"
                },
                "data": {
                    "type": "object",
                    "description": "Credential data fields matching the type schema (e.g. {\"apiKey\": \"sk-...\"} for openAiApi)"
                },
                "n8n_port": {
                    "type": "integer",
                    "description": "n8n instance port (default: 5678)"
                }
            },
            "required": ["name", "credential_type", "data"]
        }
    },
    {
        "name": "update_credential",
        "description": "Update an existing credential's name or data. Use list_credentials to find the credential ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "credential_id": {
                    "type": "string",
                    "description": "Credential ID to update"
                },
                "name": {
                    "type": "string",
                    "description": "New display name (optional)"
                },
                "data": {
                    "type": "object",
                    "description": "Updated credential data fields (optional)"
                },
                "n8n_port": {
                    "type": "integer",
                    "description": "n8n instance port (default: 5678)"
                }
            },
            "required": ["credential_id"]
        }
    },
    {
        "name": "delete_credential",
        "description": "Delete a credential from n8n by ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "credential_id": {
                    "type": "string",
                    "description": "Credential ID to delete"
                },
                "n8n_port": {
                    "type": "integer",
                    "description": "n8n instance port (default: 5678)"
                }
            },
            "required": ["credential_id"]
        }
    },

    # ---- Execution Monitoring ----
    {
        "name": "list_executions",
        "description": "List recent workflow executions with optional filtering. Use to check what ran, succeeded, or failed.",
        "parameters": {
            "type": "object",
            "properties": {
                "workflow_id": {
                    "type": "string",
                    "description": "Filter by workflow ID (optional)"
                },
                "status": {
                    "type": "string",
                    "description": "Filter by status: success, error, crashed, waiting (optional)",
                    "enum": ["success", "error", "crashed", "waiting"]
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default: 10)"
                },
                "n8n_port": {
                    "type": "integer",
                    "description": "n8n instance port (default: 5678)"
                }
            },
            "required": []
        }
    },
    {
        "name": "get_execution_result",
        "description": "Get the full result of a specific workflow execution including output data, errors, and node-level details. Use list_executions to find execution IDs.",
        "parameters": {
            "type": "object",
            "properties": {
                "execution_id": {
                    "type": "string",
                    "description": "Execution ID to inspect"
                },
                "n8n_port": {
                    "type": "integer",
                    "description": "n8n instance port (default: 5678)"
                }
            },
            "required": ["execution_id"]
        }
    },

    # ---- Workflow Lifecycle ----
    {
        "name": "update_workflow",
        "description": "Update an existing workflow's definition (nodes, connections, settings). Preserves workflow ID and history. Use list_workflows to find the ID, then modify the JSON.",
        "parameters": {
            "type": "object",
            "properties": {
                "workflow_id": {
                    "type": "string",
                    "description": "Workflow ID to update"
                },
                "workflow_json": {
                    "type": "object",
                    "description": "Updated workflow JSON (full replacement)"
                },
                "n8n_port": {
                    "type": "integer",
                    "description": "n8n instance port (default: 5678)"
                }
            },
            "required": ["workflow_id", "workflow_json"]
        }
    },
    {
        "name": "activate_workflow",
        "description": "Activate a workflow to enable its triggers (webhooks, schedules, etc).",
        "parameters": {
            "type": "object",
            "properties": {
                "workflow_id": {
                    "type": "string",
                    "description": "Workflow ID to activate"
                },
                "n8n_port": {
                    "type": "integer",
                    "description": "n8n instance port (default: 5678)"
                }
            },
            "required": ["workflow_id"]
        }
    },
    {
        "name": "deactivate_workflow",
        "description": "Deactivate a workflow to disable its triggers without deleting it.",
        "parameters": {
            "type": "object",
            "properties": {
                "workflow_id": {
                    "type": "string",
                    "description": "Workflow ID to deactivate"
                },
                "n8n_port": {
                    "type": "integer",
                    "description": "n8n instance port (default: 5678)"
                }
            },
            "required": ["workflow_id"]
        }
    },

    # ---- Integration ----
    {
        "name": "trigger_webhook",
        "description": "Trigger a webhook-based workflow by sending data to its webhook URL. The workflow must be active with a webhook trigger node.",
        "parameters": {
            "type": "object",
            "properties": {
                "webhook_path": {
                    "type": "string",
                    "description": "Webhook path (e.g. 'my-webhook' for /webhook/my-webhook)"
                },
                "data": {
                    "type": "object",
                    "description": "JSON data to send as the webhook payload (optional)"
                },
                "method": {
                    "type": "string",
                    "description": "HTTP method: GET or POST (default: POST)",
                    "enum": ["GET", "POST"],
                    "default": "POST"
                },
                "test_mode": {
                    "type": "boolean",
                    "description": "Use test webhook URL (/webhook-test/) instead of production (default: false)"
                },
                "n8n_port": {
                    "type": "integer",
                    "description": "n8n instance port (default: 5678)"
                }
            },
            "required": ["webhook_path"]
        }
    },
    {
        "name": "set_variable",
        "description": "Create or update an n8n variable. Variables are shared key-value config accessible by all workflows via $vars.key.",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Variable name (e.g. 'api_base_url', 'default_model')"
                },
                "value": {
                    "type": "string",
                    "description": "Variable value"
                },
                "n8n_port": {
                    "type": "integer",
                    "description": "n8n instance port (default: 5678)"
                }
            },
            "required": ["key", "value"]
        }
    },
    {
        "name": "list_variables",
        "description": "List all n8n variables. Variables are shared config accessible by workflows via $vars.key.",
        "parameters": {
            "type": "object",
            "properties": {
                "n8n_port": {
                    "type": "integer",
                    "description": "n8n instance port (default: 5678)"
                }
            },
            "required": []
        }
    },

    # ---- Flash Workflow ----
    {
        "name": "flash_workflow",
        "description": "Deploy a workflow, trigger it, collect results, and delete it — all in one step. Ideal for one-shot automation that doesn't need to persist. Provide the workflow JSON (from generate_preset_workflow or build_workflow) and optional webhook data.",
        "parameters": {
            "type": "object",
            "properties": {
                "workflow_json": {
                    "type": "object",
                    "description": "Complete n8n workflow JSON (from generate_preset_workflow or build_workflow)"
                },
                "webhook_data": {
                    "type": "object",
                    "description": "JSON data to send to the workflow's webhook trigger (optional)"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Max seconds to wait for results (default: 120)"
                },
                "n8n_port": {
                    "type": "integer",
                    "description": "n8n instance port (default: 5678)"
                }
            },
            "required": ["workflow_json"]
        }
    },
]


class WorkflowTools:
    """Tools for workflow generation and management."""

    def __init__(self, orchestrator, n8n_manager):
        self.orchestrator = orchestrator
        self.n8n_manager = n8n_manager

    async def _n8n_request(self, method: str, path: str, n8n_port: int = 5678,
                           json_data=None, params=None) -> tuple:
        """Shared helper for n8n REST API calls. Returns (status, data_or_text)."""
        from backend.routes.n8n import _get_or_create_auth
        auth_cookie = await _get_or_create_auth(n8n_port)

        headers = {"Content-Type": "application/json"}
        if auth_cookie:
            headers["Cookie"] = f"n8n-auth={auth_cookie}"

        url = f"http://127.0.0.1:{n8n_port}{path}"

        async with aiohttp.ClientSession() as session:
            req_method = getattr(session, method.lower())
            kwargs = {"headers": headers}
            if json_data is not None:
                kwargs["json"] = json_data
            if params is not None:
                kwargs["params"] = params

            async with req_method(url, **kwargs) as resp:
                try:
                    data = await resp.json()
                except Exception:
                    data = await resp.text()
                return resp.status, data

    def _check_n8n(self, n8n_port: int) -> Optional[Dict[str, Any]]:
        """Return error dict if n8n not running, None if OK."""
        import socket
        # Check spawned instances dict first (fast path)
        if n8n_port in self.n8n_manager.instances:
            return None
        # Fallback: direct port connectivity check (covers main admin started via N8nQueueManager)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2)
                s.connect(('127.0.0.1', n8n_port))
                return None
        except (OSError, ConnectionRefusedError):
            pass
        return {"success": False, "error": f"n8n not running on port {n8n_port}. Start n8n first."}

    async def describe_node(self, node_types: list) -> Dict[str, Any]:
        """Get parameter schemas for node types."""
        from backend.workflow_templates import get_node_params, get_all_node_params, NODE_REGISTRY

        try:
            if not node_types:
                return {"success": False, "error": "Provide at least one node type, or ['all'] for all types."}

            if node_types == ["all"] or node_types == ["*"]:
                all_params = get_all_node_params()
                return {
                    "success": True,
                    "total_types": len(NODE_REGISTRY),
                    "categories": all_params
                }

            results = {}
            not_found = []
            for nt in node_types:
                info = get_node_params(nt)
                if info:
                    results[nt] = info
                else:
                    not_found.append(nt)

            response = {"success": True, "nodes": results}
            if not_found:
                available = sorted(NODE_REGISTRY.keys())
                response["not_found"] = not_found
                response["hint"] = f"Unknown types: {not_found}. Available: {available}"

            return response

        except Exception as e:
            logger.error(f"describe_node error: {e}")
            return {"success": False, "error": str(e)}

    async def list_credentials(self, n8n_port: int = 5678) -> Dict[str, Any]:
        """List credentials configured in n8n."""
        try:
            err = self._check_n8n(n8n_port)
            if err:
                return err

            from backend.routes.n8n import _get_or_create_auth
            auth_cookie = await _get_or_create_auth(n8n_port)

            headers = {}
            if auth_cookie:
                headers["Cookie"] = f"n8n-auth={auth_cookie}"

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://127.0.0.1:{n8n_port}/rest/credentials",
                    headers=headers
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        creds = data.get("data", data) if isinstance(data, dict) else data
                        if not isinstance(creds, list):
                            creds = []

                        return {
                            "success": True,
                            "count": len(creds),
                            "credentials": [
                                {
                                    "id": c.get("id"),
                                    "name": c.get("name"),
                                    "type": c.get("type"),
                                }
                                for c in creds
                            ]
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"n8n API error {resp.status}"
                        }

        except Exception as e:
            logger.error(f"list_credentials error: {e}")
            return {"success": False, "error": str(e)}

    async def build_workflow(
        self,
        name: str,
        nodes: list,
        connections: list = None,
        deploy: bool = True,
        activate: bool = False,
        n8n_port: int = 5678
    ) -> Dict[str, Any]:
        """Build a workflow from node specifications using templates.

        Args:
            name: Workflow name
            nodes: List of node specs, each with "type" and optional params
            connections: Optional custom connections for branching workflows
            deploy: Auto-deploy to n8n after building (default True)
            activate: Activate after deploying (default False)
            n8n_port: n8n port for deployment (default 5678)
        """
        from backend.workflow_templates import build_workflow_from_nodes, get_node_types
        from backend.workflow_generator import validate_workflow

        try:
            if not nodes:
                return {
                    "success": False,
                    "error": "No nodes provided. Need at least one node.",
                    "available_types": get_node_types()
                }

            # Convert n8n-format connections dict to list format if needed
            # n8n format: {"NodeA": {"main": [[{"node": "NodeB", ...}]]}}
            # list format: [{"from": "NodeA", "to": "NodeB", "output": 0}]
            if connections and isinstance(connections, dict):
                conn_list = []
                for source, outputs in connections.items():
                    if isinstance(outputs, dict) and "main" in outputs:
                        for out_idx, targets in enumerate(outputs["main"]):
                            if isinstance(targets, list):
                                for target in targets:
                                    if isinstance(target, dict):
                                        conn_list.append({
                                            "from": source,
                                            "to": target.get("node", ""),
                                            "output": out_idx,
                                            "input": target.get("index", 0),
                                        })
                connections = conn_list if conn_list else None

            # Build the workflow
            workflow = build_workflow_from_nodes(
                name, nodes,
                connection_mode="linear" if not connections else "custom",
                custom_connections=connections
            )

            logger.info(f"[BUILD_WORKFLOW] Built workflow '{name}' with {len(nodes)} nodes")
            logger.info(f"[BUILD_WORKFLOW] Node types: {[n.get('type') for n in nodes]}")
            if connections:
                logger.info(f"[BUILD_WORKFLOW] Custom connections: {len(connections)}")

            # Validate before returning
            is_valid, errors = validate_workflow(workflow)

            if errors:
                msg = f"Built '{name}' with {len(workflow['nodes'])} nodes. VALIDATION ISSUES: {'; '.join(errors)}. Fix and rebuild, or deploy anyway."
            else:
                msg = f"Built '{name}' with {len(workflow['nodes'])} nodes. Valid."

            result = {
                "success": True,
                "workflow": workflow,
                "valid": is_valid,
                "validation_errors": errors if errors else None,
                "message": msg,
                "nodes": [n.get("name") for n in workflow["nodes"]]
            }

            # Auto-deploy if requested (default)
            if deploy:
                deploy_result = await self.deploy_workflow(
                    workflow_json=workflow,
                    n8n_port=n8n_port,
                    activate=activate
                )
                result["deployed"] = deploy_result.get("success", False)
                if deploy_result.get("success"):
                    result["workflow_id"] = deploy_result.get("workflow_id")
                    result["workflow_url"] = deploy_result.get("url")
                    activated_str = " and activated" if activate and deploy_result.get("activated") else ""
                    result["message"] += f" Deployed{activated_str} to n8n (port {n8n_port}). ID: {deploy_result.get('workflow_id')}"
                    if deploy_result.get("webhook_url"):
                        result["webhook_url"] = deploy_result["webhook_url"]
                        result["message"] += f". Webhook: {deploy_result['webhook_url']}"
                else:
                    result["deploy_error"] = deploy_result.get("error", "Unknown deploy error")
                    result["message"] += f" Deploy FAILED: {deploy_result.get('error')}. Use deploy_workflow manually."

            return result

        except ValueError as e:
            return {
                "success": False,
                "error": str(e),
                "available_types": get_node_types()
            }
        except Exception as e:
            logger.error(f"build_workflow error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e),
                "hint": "Put params at TOP LEVEL of each node (not nested). "
                        "Example: {\"type\": \"webhook\"}, {\"type\": \"http_request\", \"url\": \"...\", \"method\": \"POST\"}, {\"type\": \"respond_webhook\"}. "
                        "Connections should be: [{\"from\": \"Webhook\", \"to\": \"HTTP Request\"}]"
            }

    async def deploy_workflow(
        self,
        workflow_json: Dict,
        n8n_port: int = 5678,
        activate: bool = False
    ) -> Dict[str, Any]:
        """Deploy a workflow to n8n."""
        import json as json_module
        from backend.workflow_generator import fix_workflow

        try:
            # Handle string input - parse as JSON
            if isinstance(workflow_json, str):
                try:
                    workflow_json = json_module.loads(workflow_json)
                except json_module.JSONDecodeError as e:
                    return {"success": False, "error": f"Invalid JSON: {e}"}

            # Always run fix_workflow to ensure required fields
            workflow_json = fix_workflow(workflow_json)

            # Check if n8n is running
            err = self._check_n8n(n8n_port)
            if err:
                return err

            # DEBUG: Log the workflow being deployed
            logger.info("=" * 60)
            logger.info("[DEPLOY DEBUG] === Deploying Workflow ===")
            logger.info(f"[DEPLOY DEBUG] Workflow name: {workflow_json.get('name')}")
            logger.info(f"[DEPLOY DEBUG] Active: {workflow_json.get('active')}")

            # Log nodes
            nodes = workflow_json.get("nodes", [])
            logger.info(f"[DEPLOY DEBUG] Nodes ({len(nodes)}):")
            for i, node in enumerate(nodes):
                logger.info(f"  [{i}] name='{node.get('name')}' type='{node.get('type')}' pos={node.get('position')}")

            # Log connections
            connections = workflow_json.get("connections", {})
            logger.info(f"[DEPLOY DEBUG] Connections:")
            if not isinstance(connections, dict):
                logger.warning(f"[DEPLOY DEBUG] Connections is {type(connections).__name__}, expected dict")
                connections = {}
            for source, outputs in connections.items():
                if isinstance(outputs, dict) and "main" in outputs:
                    for conn_list in outputs["main"]:
                        for conn in conn_list:
                            if isinstance(conn, dict):
                                target = conn.get("node")
                                logger.info(f"  '{source}' -> '{target}'")

            # Log full JSON for debugging
            logger.info(f"[DEPLOY DEBUG] Full workflow JSON:")
            logger.info(json_module.dumps(workflow_json, indent=2))
            logger.info("=" * 60)

            # Get auth cookie
            from backend.routes.n8n import _get_or_create_auth
            auth_cookie = await _get_or_create_auth(n8n_port)

            headers = {"Content-Type": "application/json"}
            if auth_cookie:
                headers["Cookie"] = f"n8n-auth={auth_cookie}"

            # Create workflow via n8n API
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://127.0.0.1:{n8n_port}/rest/workflows",
                    json=workflow_json,
                    headers=headers
                ) as resp:
                    response_text = await resp.text()
                    logger.info(f"[DEPLOY DEBUG] n8n response status: {resp.status}")
                    logger.info(f"[DEPLOY DEBUG] n8n response body: {response_text[:500]}")

                    if resp.status in (200, 201):
                        import json as _json
                        data = _json.loads(response_text)
                        if not isinstance(data, dict):
                            return {"success": False, "error": f"Unexpected deploy response type: {type(data).__name__}"}
                        workflow_id = data.get("id") or data.get("data", {}).get("id")

                        # Activate if requested
                        activated = False
                        if activate and workflow_id:
                            act_resp = await session.patch(
                                f"http://127.0.0.1:{n8n_port}/rest/workflows/{workflow_id}",
                                json={"active": True},
                                headers=headers
                            )
                            activated = act_resp.status in (200, 201)

                        # Check for webhook URL
                        webhook_url = None
                        for n in nodes:
                            if n.get("type") in ("n8n-nodes-base.webhook", "@n8n/n8n-nodes-base.webhook"):
                                webhook_path = n.get("parameters", {}).get("path", "")
                                if webhook_path:
                                    webhook_url = f"http://127.0.0.1:{n8n_port}/webhook/{webhook_path}"
                                break

                        result = {
                            "success": True,
                            "workflow_id": workflow_id,
                            "activated": activated,
                            "message": f"Workflow deployed to n8n:{n8n_port}",
                            "url": f"{AGENTNATE_BASE}/api/n8n/{n8n_port}/proxy/workflow/{workflow_id}",
                            "debug": {
                                "nodes": [n.get("name") for n in nodes],
                                "connections": list(connections.keys())
                            }
                        }
                        if webhook_url:
                            result["webhook_url"] = webhook_url
                        return result
                    else:
                        return {
                            "success": False,
                            "error": f"n8n API error {resp.status}: {response_text[:200]}"
                        }

        except Exception as e:
            logger.error(f"deploy_workflow error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e)}

    async def list_workflows(self, n8n_port: int = 5678) -> Dict[str, Any]:
        """List workflows in n8n."""
        try:
            err = self._check_n8n(n8n_port)
            if err:
                return err

            from backend.routes.n8n import _get_or_create_auth
            auth_cookie = await _get_or_create_auth(n8n_port)

            headers = {}
            if auth_cookie:
                headers["Cookie"] = f"n8n-auth={auth_cookie}"

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://127.0.0.1:{n8n_port}/rest/workflows",
                    headers=headers
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if not isinstance(data, dict):
                            return {"success": False, "error": f"Unexpected response type: {type(data).__name__}"}
                        workflows = data.get("data", [])

                        return {
                            "success": True,
                            "count": len(workflows),
                            "workflows": [
                                {
                                    "id": w.get("id"),
                                    "name": w.get("name"),
                                    "active": w.get("active", False)
                                }
                                for w in workflows
                            ]
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"n8n API error {resp.status}"
                        }

        except Exception as e:
            logger.error(f"list_workflows error: {e}")
            return {"success": False, "error": str(e)}

    async def delete_workflow(
        self,
        workflow_id: str,
        n8n_port: int = 5678
    ) -> Dict[str, Any]:
        """Delete a specific workflow from n8n."""
        try:
            logger.info(f"[DELETE DEBUG] Starting delete for workflow {workflow_id}")

            err = self._check_n8n(n8n_port)
            if err:
                return err

            from backend.routes.n8n import _get_or_create_auth
            auth_cookie = await _get_or_create_auth(n8n_port)

            headers = {"Content-Type": "application/json"}
            if auth_cookie:
                headers["Cookie"] = f"n8n-auth={auth_cookie}"

            async with aiohttp.ClientSession() as session:
                workflow_url = f"http://127.0.0.1:{n8n_port}/rest/workflows/{workflow_id}"

                # Step 1: Deactivate the workflow
                logger.info(f"[DELETE DEBUG] Deactivating workflow {workflow_id}")
                async with session.patch(
                    workflow_url,
                    json={"active": False},
                    headers=headers
                ) as deactivate_resp:
                    deactivate_body = await deactivate_resp.text()
                    logger.info(f"[DELETE DEBUG] Deactivate {workflow_id}: status={deactivate_resp.status}, body={deactivate_body[:200]}")

                # Step 2: Archive the workflow (required before deletion)
                # n8n uses a dedicated POST endpoint for archiving, not PATCH
                logger.info(f"[DELETE DEBUG] Archiving workflow {workflow_id}")
                archive_url = f"http://127.0.0.1:{n8n_port}/rest/workflows/{workflow_id}/archive"
                async with session.post(
                    archive_url,
                    headers=headers
                ) as archive_resp:
                    archive_body = await archive_resp.text()
                    logger.info(f"[DELETE DEBUG] Archive {workflow_id}: status={archive_resp.status}, body={archive_body[:200]}")

                # Step 3: Delete the workflow
                logger.info(f"[DELETE DEBUG] Deleting workflow {workflow_id}")
                async with session.delete(
                    f"http://127.0.0.1:{n8n_port}/rest/workflows/{workflow_id}",
                    headers=headers
                ) as resp:
                    resp_body = await resp.text()
                    logger.info(f"[DELETE DEBUG] Delete {workflow_id}: status={resp.status}, body={resp_body[:200]}")

                    if resp.status in (200, 204):
                        return {
                            "success": True,
                            "message": f"Workflow {workflow_id} deleted"
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"n8n API error {resp.status}: {resp_body[:200]}"
                        }

        except Exception as e:
            logger.error(f"delete_workflow error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e)}

    async def delete_all_workflows(
        self,
        confirm: bool = False,
        n8n_port: int = 5678
    ) -> Dict[str, Any]:
        """Delete ALL workflows from n8n. Requires confirm=True."""
        if not confirm:
            return {
                "success": False,
                "error": "Must set confirm=True to delete all workflows"
            }

        try:
            # First list all workflows
            list_result = await self.list_workflows(n8n_port)
            if not list_result.get("success"):
                return list_result

            workflows = list_result.get("workflows", [])
            if not workflows:
                return {
                    "success": True,
                    "message": "No workflows to delete",
                    "deleted_count": 0
                }

            # Delete each workflow
            deleted = []
            failed = []
            for wf in workflows:
                wf_id = wf.get("id")
                result = await self.delete_workflow(wf_id, n8n_port)
                if result.get("success"):
                    deleted.append(wf_id)
                else:
                    failed.append({"id": wf_id, "error": result.get("error")})

            return {
                "success": len(failed) == 0,
                "deleted_count": len(deleted),
                "deleted": deleted,
                "failed": failed if failed else None,
                "message": f"Deleted {len(deleted)} workflows" + (f", {len(failed)} failed" if failed else "")
            }

        except Exception as e:
            logger.error(f"delete_all_workflows error: {e}")
            return {"success": False, "error": str(e)}

    # ========================================================================
    # Credential Management
    # ========================================================================

    async def describe_credential_types(
        self, filter: str = None, n8n_port: int = 5678
    ) -> Dict[str, Any]:
        """Discover available credential types and their required fields."""
        try:
            err = self._check_n8n(n8n_port)
            if err:
                return err

            status, data = await self._n8n_request("GET", "/rest/credential-types", n8n_port)
            if status != 200:
                return {"success": False, "error": f"n8n API error {status}"}

            # data should be a list of credential type objects
            types_list = data if isinstance(data, list) else data.get("data", [])

            # Apply filter if provided
            if filter:
                filter_lower = filter.lower()
                types_list = [
                    t for t in types_list
                    if filter_lower in t.get("name", "").lower()
                    or filter_lower in t.get("displayName", "").lower()
                ]

            # Extract useful info from each type
            results = []
            for t in types_list:
                props = t.get("properties", [])
                fields = []
                for p in props:
                    field = {
                        "name": p.get("name"),
                        "displayName": p.get("displayName", p.get("name")),
                        "type": p.get("type", "string"),
                        "required": p.get("required", False),
                    }
                    if "default" in p:
                        field["default"] = p["default"]
                    if "options" in p:
                        field["options"] = [
                            o.get("value", o.get("name")) for o in p["options"]
                            if isinstance(o, dict)
                        ]
                    fields.append(field)

                results.append({
                    "type": t.get("name"),
                    "displayName": t.get("displayName", t.get("name")),
                    "fields": fields,
                })

            return {
                "success": True,
                "count": len(results),
                "credential_types": results,
            }

        except Exception as e:
            logger.error(f"describe_credential_types error: {e}")
            return {"success": False, "error": str(e)}

    async def create_credential(
        self, name: str, credential_type: str, data: Dict[str, Any],
        n8n_port: int = 5678
    ) -> Dict[str, Any]:
        """Create a new credential in n8n."""
        try:
            err = self._check_n8n(n8n_port)
            if err:
                return err

            payload = {
                "name": name,
                "type": credential_type,
                "data": data,
            }

            status, resp_data = await self._n8n_request(
                "POST", "/rest/credentials", n8n_port, json_data=payload
            )

            if status in (200, 201):
                cred = resp_data if isinstance(resp_data, dict) else {}
                return {
                    "success": True,
                    "credential_id": cred.get("id"),
                    "name": cred.get("name", name),
                    "type": cred.get("type", credential_type),
                    "message": f"Credential '{name}' ({credential_type}) created successfully",
                }
            else:
                err_msg = resp_data if isinstance(resp_data, str) else json_module.dumps(resp_data)
                return {"success": False, "error": f"n8n API error {status}: {err_msg[:300]}"}

        except Exception as e:
            logger.error(f"create_credential error: {e}")
            return {"success": False, "error": str(e)}

    async def update_credential(
        self, credential_id: str, name: str = None, data: Dict[str, Any] = None,
        n8n_port: int = 5678
    ) -> Dict[str, Any]:
        """Update an existing credential's name or data."""
        try:
            err = self._check_n8n(n8n_port)
            if err:
                return err

            if not name and not data:
                return {"success": False, "error": "Provide at least name or data to update"}

            payload = {}
            if name is not None:
                payload["name"] = name
            if data is not None:
                payload["data"] = data

            status, resp_data = await self._n8n_request(
                "PATCH", f"/rest/credentials/{credential_id}", n8n_port, json_data=payload
            )

            if status == 200:
                cred = resp_data if isinstance(resp_data, dict) else {}
                return {
                    "success": True,
                    "credential_id": credential_id,
                    "name": cred.get("name"),
                    "type": cred.get("type"),
                    "message": f"Credential {credential_id} updated",
                }
            elif status == 404:
                return {"success": False, "error": f"Credential {credential_id} not found"}
            else:
                err_msg = resp_data if isinstance(resp_data, str) else json_module.dumps(resp_data)
                return {"success": False, "error": f"n8n API error {status}: {err_msg[:300]}"}

        except Exception as e:
            logger.error(f"update_credential error: {e}")
            return {"success": False, "error": str(e)}

    async def delete_credential(
        self, credential_id: str, n8n_port: int = 5678
    ) -> Dict[str, Any]:
        """Delete a credential from n8n."""
        try:
            err = self._check_n8n(n8n_port)
            if err:
                return err

            status, resp_data = await self._n8n_request(
                "DELETE", f"/rest/credentials/{credential_id}", n8n_port
            )

            if status in (200, 204):
                return {
                    "success": True,
                    "message": f"Credential {credential_id} deleted",
                }
            elif status == 404:
                return {"success": False, "error": f"Credential {credential_id} not found"}
            else:
                err_msg = resp_data if isinstance(resp_data, str) else json_module.dumps(resp_data)
                return {"success": False, "error": f"n8n API error {status}: {err_msg[:300]}"}

        except Exception as e:
            logger.error(f"delete_credential error: {e}")
            return {"success": False, "error": str(e)}

    # ========================================================================
    # Execution Monitoring
    # ========================================================================

    async def list_executions(
        self, workflow_id: str = None, status: str = None,
        limit: int = 10, n8n_port: int = 5678
    ) -> Dict[str, Any]:
        """List recent workflow executions with optional filtering."""
        try:
            err = self._check_n8n(n8n_port)
            if err:
                return err

            params = {"limit": str(limit)}
            if workflow_id:
                params["workflowId"] = workflow_id
            if status:
                params["status"] = status

            resp_status, data = await self._n8n_request(
                "GET", "/rest/executions", n8n_port, params=params
            )

            if resp_status != 200:
                return {"success": False, "error": f"n8n API error {resp_status}"}

            executions = data.get("data", data) if isinstance(data, dict) else data
            if not isinstance(executions, list):
                executions = []

            results = []
            for ex in executions[:limit]:
                results.append({
                    "id": ex.get("id"),
                    "status": ex.get("status"),
                    "finished": ex.get("finished"),
                    "startedAt": ex.get("startedAt"),
                    "stoppedAt": ex.get("stoppedAt"),
                    "workflowId": ex.get("workflowId"),
                    "workflowName": ex.get("workflowName", ex.get("workflowData", {}).get("name", "")),
                    "mode": ex.get("mode"),
                })

            return {
                "success": True,
                "count": len(results),
                "executions": results,
            }

        except Exception as e:
            logger.error(f"list_executions error: {e}")
            return {"success": False, "error": str(e)}

    async def get_execution_result(
        self, execution_id: str, n8n_port: int = 5678
    ) -> Dict[str, Any]:
        """Get the full result of a specific execution."""
        try:
            err = self._check_n8n(n8n_port)
            if err:
                return err

            resp_status, data = await self._n8n_request(
                "GET", f"/rest/executions/{execution_id}", n8n_port
            )

            if resp_status == 404:
                return {"success": False, "error": f"Execution {execution_id} not found"}
            if resp_status != 200:
                return {"success": False, "error": f"n8n API error {resp_status}"}

            if not isinstance(data, dict):
                return {"success": False, "error": "Unexpected response format"}

            # Extract key info
            result = {
                "success": True,
                "execution_id": data.get("id"),
                "status": data.get("status"),
                "finished": data.get("finished"),
                "startedAt": data.get("startedAt"),
                "stoppedAt": data.get("stoppedAt"),
                "mode": data.get("mode"),
            }

            # Extract result data (node outputs)
            result_data = data.get("data", {}).get("resultData", {})
            run_data = result_data.get("runData", {})

            # Summarize node results
            node_results = {}
            for node_name, runs in run_data.items():
                if isinstance(runs, list) and runs:
                    last_run = runs[-1]
                    node_info = {
                        "status": last_run.get("executionStatus", "unknown"),
                    }
                    # Get output data
                    output = last_run.get("data", {}).get("main", [[]])
                    if output and isinstance(output, list):
                        flat_items = []
                        for branch in output:
                            if isinstance(branch, list):
                                for item in branch:
                                    if isinstance(item, dict):
                                        flat_items.append(item.get("json", item))
                        if flat_items:
                            node_info["output_count"] = len(flat_items)
                            # Include output data, truncated
                            output_str = json_module.dumps(flat_items, default=str)
                            if len(output_str) > 5000:
                                node_info["output"] = json_module.loads(output_str[:5000] + "...")
                                node_info["truncated"] = True
                            else:
                                node_info["output"] = flat_items

                    # Include error if present
                    if last_run.get("error"):
                        node_info["error"] = str(last_run["error"])[:500]

                    node_results[node_name] = node_info

            result["node_results"] = node_results

            # Include top-level error if execution failed
            last_error = result_data.get("error")
            if last_error:
                result["error"] = str(last_error)[:1000]

            return result

        except Exception as e:
            logger.error(f"get_execution_result error: {e}")
            return {"success": False, "error": str(e)}

    # ========================================================================
    # Workflow Lifecycle
    # ========================================================================

    async def update_workflow(
        self, workflow_id: str, workflow_json: Dict[str, Any],
        n8n_port: int = 5678
    ) -> Dict[str, Any]:
        """Update an existing workflow's definition."""
        try:
            err = self._check_n8n(n8n_port)
            if err:
                return err

            # Handle string input
            if isinstance(workflow_json, str):
                try:
                    workflow_json = json_module.loads(workflow_json)
                except json_module.JSONDecodeError as e:
                    return {"success": False, "error": f"Invalid JSON: {e}"}

            from backend.workflow_generator import fix_workflow
            workflow_json = fix_workflow(workflow_json)

            status, data = await self._n8n_request(
                "PUT", f"/rest/workflows/{workflow_id}", n8n_port,
                json_data=workflow_json
            )

            if status == 200:
                wf = data if isinstance(data, dict) else {}
                return {
                    "success": True,
                    "workflow_id": workflow_id,
                    "name": wf.get("name"),
                    "active": wf.get("active", False),
                    "message": f"Workflow {workflow_id} updated",
                    "nodes": [n.get("name") for n in wf.get("nodes", [])],
                }
            elif status == 404:
                return {"success": False, "error": f"Workflow {workflow_id} not found"}
            else:
                err_msg = data if isinstance(data, str) else json_module.dumps(data)
                return {"success": False, "error": f"n8n API error {status}: {err_msg[:300]}"}

        except Exception as e:
            logger.error(f"update_workflow error: {e}")
            return {"success": False, "error": str(e)}

    async def activate_workflow(
        self, workflow_id: str, n8n_port: int = 5678
    ) -> Dict[str, Any]:
        """Activate a workflow to enable its triggers."""
        try:
            err = self._check_n8n(n8n_port)
            if err:
                return err

            status, data = await self._n8n_request(
                "PATCH", f"/rest/workflows/{workflow_id}", n8n_port,
                json_data={"active": True}
            )

            if status == 200:
                return {
                    "success": True,
                    "workflow_id": workflow_id,
                    "active": True,
                    "message": f"Workflow {workflow_id} activated",
                }
            elif status == 404:
                return {"success": False, "error": f"Workflow {workflow_id} not found"}
            else:
                err_msg = data if isinstance(data, str) else json_module.dumps(data)
                return {"success": False, "error": f"n8n API error {status}: {err_msg[:300]}"}

        except Exception as e:
            logger.error(f"activate_workflow error: {e}")
            return {"success": False, "error": str(e)}

    async def deactivate_workflow(
        self, workflow_id: str, n8n_port: int = 5678
    ) -> Dict[str, Any]:
        """Deactivate a workflow to disable its triggers."""
        try:
            err = self._check_n8n(n8n_port)
            if err:
                return err

            status, data = await self._n8n_request(
                "PATCH", f"/rest/workflows/{workflow_id}", n8n_port,
                json_data={"active": False}
            )

            if status == 200:
                return {
                    "success": True,
                    "workflow_id": workflow_id,
                    "active": False,
                    "message": f"Workflow {workflow_id} deactivated",
                }
            elif status == 404:
                return {"success": False, "error": f"Workflow {workflow_id} not found"}
            else:
                err_msg = data if isinstance(data, str) else json_module.dumps(data)
                return {"success": False, "error": f"n8n API error {status}: {err_msg[:300]}"}

        except Exception as e:
            logger.error(f"deactivate_workflow error: {e}")
            return {"success": False, "error": str(e)}

    # ========================================================================
    # Integration
    # ========================================================================

    async def trigger_webhook(
        self, webhook_path: str, data: Dict[str, Any] = None,
        method: str = "POST", test_mode: bool = False,
        n8n_port: int = 5678
    ) -> Dict[str, Any]:
        """Trigger a webhook-based workflow."""
        try:
            err = self._check_n8n(n8n_port)
            if err:
                return err

            # Build webhook URL
            prefix = "webhook-test" if test_mode else "webhook"
            webhook_path = webhook_path.lstrip("/")
            url = f"http://127.0.0.1:{n8n_port}/{prefix}/{webhook_path}"

            async with aiohttp.ClientSession() as session:
                if method.upper() == "GET":
                    async with session.get(url, params=data) as resp:
                        resp_status = resp.status
                        try:
                            resp_data = await resp.json()
                        except Exception:
                            resp_data = await resp.text()
                else:
                    async with session.post(
                        url, json=data or {},
                        headers={"Content-Type": "application/json"}
                    ) as resp:
                        resp_status = resp.status
                        try:
                            resp_data = await resp.json()
                        except Exception:
                            resp_data = await resp.text()

            if resp_status in (200, 201):
                return {
                    "success": True,
                    "status_code": resp_status,
                    "response": resp_data,
                    "message": f"Webhook {webhook_path} triggered successfully",
                }
            elif resp_status == 404:
                return {
                    "success": False,
                    "error": f"Webhook '{webhook_path}' not found. Is the workflow active? Try test_mode=true for inactive workflows.",
                }
            else:
                return {
                    "success": False,
                    "error": f"Webhook returned {resp_status}",
                    "response": resp_data if isinstance(resp_data, str) else json_module.dumps(resp_data)[:500],
                }

        except Exception as e:
            logger.error(f"trigger_webhook error: {e}")
            return {"success": False, "error": str(e)}

    async def set_variable(
        self, key: str, value: str, n8n_port: int = 5678
    ) -> Dict[str, Any]:
        """Create or update an n8n variable."""
        try:
            err = self._check_n8n(n8n_port)
            if err:
                return err

            # First try to list existing variables to check if key exists
            list_status, list_data = await self._n8n_request(
                "GET", "/rest/variables", n8n_port
            )

            existing_id = None
            if list_status == 200:
                variables = list_data if isinstance(list_data, list) else list_data.get("data", [])
                for v in variables:
                    if isinstance(v, dict) and v.get("key") == key:
                        existing_id = v.get("id")
                        break

            if existing_id:
                # Update existing variable
                status, data = await self._n8n_request(
                    "PATCH", f"/rest/variables/{existing_id}", n8n_port,
                    json_data={"key": key, "value": value}
                )
                action = "updated"
            else:
                # Create new variable
                status, data = await self._n8n_request(
                    "POST", "/rest/variables", n8n_port,
                    json_data={"key": key, "value": value}
                )
                action = "created"

            if status in (200, 201):
                return {
                    "success": True,
                    "key": key,
                    "value": value,
                    "action": action,
                    "message": f"Variable '{key}' {action}. Access in workflows via $vars.{key}",
                }
            else:
                err_msg = data if isinstance(data, str) else json_module.dumps(data)
                return {"success": False, "error": f"n8n API error {status}: {err_msg[:300]}"}

        except Exception as e:
            logger.error(f"set_variable error: {e}")
            return {"success": False, "error": str(e)}

    async def list_variables(self, n8n_port: int = 5678) -> Dict[str, Any]:
        """List all n8n variables."""
        try:
            err = self._check_n8n(n8n_port)
            if err:
                return err

            status, data = await self._n8n_request(
                "GET", "/rest/variables", n8n_port
            )

            if status != 200:
                return {"success": False, "error": f"n8n API error {status}"}

            variables = data if isinstance(data, list) else data.get("data", [])
            if not isinstance(variables, list):
                variables = []

            return {
                "success": True,
                "count": len(variables),
                "variables": [
                    {
                        "id": v.get("id"),
                        "key": v.get("key"),
                        "value": v.get("value"),
                    }
                    for v in variables
                    if isinstance(v, dict)
                ],
            }

        except Exception as e:
            logger.error(f"list_variables error: {e}")
            return {"success": False, "error": str(e)}

    # ========================================================================
    # Flash Workflow
    # ========================================================================

    async def flash_workflow(
        self,
        workflow_json: Dict[str, Any],
        webhook_data: Dict[str, Any] = None,
        timeout: int = 120,
        n8n_port: int = 5678,
    ) -> Dict[str, Any]:
        """
        Deploy → activate → trigger → collect results → delete.

        One-shot workflow execution for ephemeral automation.
        """
        import asyncio

        err = self._check_n8n(n8n_port)
        if err:
            return err

        # Handle string input
        if isinstance(workflow_json, str):
            try:
                workflow_json = json_module.loads(workflow_json)
            except json_module.JSONDecodeError as e:
                return {"success": False, "error": f"Invalid JSON: {e}"}

        workflow_id = None
        try:
            # --- Step 1: Extract webhook path from workflow nodes ---
            webhook_path = None
            for node in workflow_json.get("nodes", []):
                if "webhook" in node.get("type", "").lower():
                    params = node.get("parameters", {})
                    if "path" in params:
                        webhook_path = params["path"]
                        break
                    # Also check top-level (template format before building)
                    if "path" in node:
                        webhook_path = node["path"]
                        break

            # Also check meta
            if not webhook_path:
                webhook_path = workflow_json.get("meta", {}).get("webhook_path")

            if not webhook_path:
                return {
                    "success": False,
                    "error": "Workflow has no webhook trigger node with a 'path' parameter. "
                             "flash_workflow requires a webhook-triggered workflow.",
                }

            # --- Step 2: Deploy + activate ---
            deploy_result = await self.deploy_workflow(
                workflow_json, n8n_port, activate=True
            )
            if not deploy_result.get("success"):
                return {
                    "success": False,
                    "error": f"Deploy failed: {deploy_result.get('error')}",
                }
            workflow_id = deploy_result.get("workflow_id")
            logger.info(f"[FLASH] Deployed workflow {workflow_id}, activating...")

            # --- Step 3: Wait for webhook registration ---
            await asyncio.sleep(1.5)

            # --- Step 4: Trigger webhook ---
            logger.info(f"[FLASH] Triggering webhook /{webhook_path}")
            trigger_result = await self.trigger_webhook(
                webhook_path, webhook_data or {}, "POST", False, n8n_port
            )

            if trigger_result.get("success"):
                # Webhook responded directly (responseNode mode)
                logger.info(f"[FLASH] Got direct webhook response")
                return {
                    "success": True,
                    "result": trigger_result.get("response"),
                    "workflow_id": workflow_id,
                    "mode": "webhook_response",
                }

            # Webhook might return non-200 if it's async (lastNode/onReceived mode)
            # or if there was an error. Check if we should poll executions.
            if trigger_result.get("status_code") in (404,):
                return {
                    "success": False,
                    "error": f"Webhook not found. The workflow may not have activated in time. "
                             f"Details: {trigger_result.get('error')}",
                    "workflow_id": workflow_id,
                }

            # --- Step 5: Poll executions for result ---
            logger.info(f"[FLASH] Webhook didn't return direct result, polling executions...")
            deadline = asyncio.get_event_loop().time() + timeout
            last_error = None

            while asyncio.get_event_loop().time() < deadline:
                await asyncio.sleep(3)

                exec_result = await self.list_executions(
                    workflow_id=workflow_id, limit=1, n8n_port=n8n_port
                )
                if not exec_result.get("success"):
                    continue

                executions = exec_result.get("executions", [])
                if not executions:
                    continue

                latest = executions[0]
                if latest.get("finished"):
                    exec_id = latest.get("id")
                    detail = await self.get_execution_result(exec_id, n8n_port)
                    return {
                        "success": latest.get("status") == "success",
                        "result": detail.get("node_results"),
                        "execution_id": exec_id,
                        "status": latest.get("status"),
                        "workflow_id": workflow_id,
                        "mode": "execution_poll",
                    }
                last_error = f"Execution still running after polling"

            return {
                "success": False,
                "error": f"Timed out after {timeout}s waiting for workflow execution",
                "workflow_id": workflow_id,
            }

        except Exception as e:
            logger.error(f"flash_workflow error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e), "workflow_id": workflow_id}

        finally:
            # --- Cleanup: Always delete the workflow ---
            if workflow_id:
                try:
                    logger.info(f"[FLASH] Cleaning up workflow {workflow_id}")
                    await self.delete_workflow(workflow_id, n8n_port)
                except Exception as cleanup_err:
                    logger.warning(f"[FLASH] Cleanup failed for {workflow_id}: {cleanup_err}")
