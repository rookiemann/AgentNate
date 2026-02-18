"""
Marketplace Tools - Search, fetch, inspect, and configure n8n marketplace workflows.

Bridges the meta agent to the n8n workflow marketplace, enabling autonomous
discovery and deployment of pre-built automation templates.
"""

import copy
import logging
import os
from typing import Dict, Any, List, Optional

import aiohttp

logger = logging.getLogger("tools.marketplace")
AGENTNATE_BASE = os.getenv("AGENTNATE_BASE_URL", "http://127.0.0.1:8000")

# Map of n8n node types to their required credential types.
# Used when marketplace workflows have credentials stripped (empty {} dict).
_NODE_TYPE_CREDENTIALS = {
    # Messaging
    "n8n-nodes-base.slack": ["slackOAuth2Api"],
    "n8n-nodes-base.telegram": ["telegramApi"],
    "n8n-nodes-base.discord": ["discordApi"],
    "n8n-nodes-base.emailSend": ["smtp"],
    "n8n-nodes-base.microsoftTeams": ["microsoftTeamsOAuth2Api"],
    "n8n-nodes-base.twilio": ["twilioApi"],
    "n8n-nodes-base.matrix": ["matrixApi"],
    # Triggers
    "n8n-nodes-base.slackTrigger": ["slackOAuth2Api"],
    "n8n-nodes-base.telegramTrigger": ["telegramApi"],
    "n8n-nodes-base.discordTrigger": ["discordApi"],
    "n8n-nodes-base.emailReadImap": ["imap"],
    "n8n-nodes-base.shopifyTrigger": ["shopifyApi"],
    "n8n-nodes-base.githubTrigger": ["githubApi"],
    "n8n-nodes-base.gitlabTrigger": ["gitlabApi"],
    # Cloud / Storage
    "n8n-nodes-base.googleSheets": ["googleSheetsOAuth2Api"],
    "n8n-nodes-base.googleDrive": ["googleDriveOAuth2Api"],
    "n8n-nodes-base.googleCalendar": ["googleCalendarOAuth2Api"],
    "n8n-nodes-base.googleGmail": ["gmailOAuth2"],
    "n8n-nodes-base.dropbox": ["dropboxOAuth2Api"],
    "n8n-nodes-base.oneDrive": ["microsoftOneDriveOAuth2Api"],
    "n8n-nodes-base.notion": ["notionApi"],
    "n8n-nodes-base.airtable": ["airtableTokenApi"],
    "n8n-nodes-base.awsS3": ["aws"],
    # Database
    "n8n-nodes-base.mySql": ["mySql"],
    "n8n-nodes-base.postgres": ["postgres"],
    "n8n-nodes-base.mongoDb": ["mongoDb"],
    "n8n-nodes-base.redis": ["redis"],
    # Dev tools
    "n8n-nodes-base.github": ["githubApi"],
    "n8n-nodes-base.gitlab": ["gitlabApi"],
    "n8n-nodes-base.jira": ["jiraSoftwareCloudApi"],
    # AI / LLM (LangChain nodes)
    "@n8n/n8n-nodes-langchain.lmChatOpenAi": ["openAiApi"],
    "@n8n/n8n-nodes-langchain.lmChatAnthropic": ["anthropicApi"],
    "@n8n/n8n-nodes-langchain.lmChatGoogleGemini": ["googleGeminiApi"],
    "@n8n/n8n-nodes-langchain.lmChatOllama": ["ollamaApi"],
    "@n8n/n8n-nodes-langchain.lmChatGroq": ["groqApi"],
    # Utilities
    "n8n-nodes-base.ftp": ["ftp"],
    "n8n-nodes-base.ssh": ["sshPassword"],
    "n8n-nodes-base.hubspot": ["hubspotApi"],
    "n8n-nodes-base.salesforce": ["salesforceOAuth2Api"],
    "n8n-nodes-base.stripe": ["stripeApi"],
    "n8n-nodes-base.shopify": ["shopifyApi"],
}

# Common placeholder patterns in marketplace workflow parameters
_PLACEHOLDER_PATTERNS = [
    "YOUR_", "REPLACE_", "TODO", "CHANGEME", "INSERT_",
    "example.com", "example.org",
    "sk-", "xoxb-", "xoxp-",
    "API_KEY", "api_key", "apiKey",
    "<your", "[your",
]

def _sanitize_surrogates(obj):
    """Recursively replace lone surrogate characters in strings.

    Marketplace workflow JSON sometimes contains broken emoji sequences
    (e.g. \\udc8d) that cause UnicodeEncodeError when serialized to JSON/UTF-8.
    """
    if isinstance(obj, str):
        return obj.encode("utf-8", errors="replace").decode("utf-8")
    if isinstance(obj, dict):
        return {k: _sanitize_surrogates(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_surrogates(item) for item in obj]
    return obj


TOOL_DEFINITIONS = [
    {
        "name": "search_marketplace",
        "description": "Search the n8n workflow marketplace for automation templates. Returns workflow summaries with ID, name, description, category, complexity, integrations, and popularity. Use get_marketplace_workflow to fetch the full workflow for a result.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords (e.g. 'slack notification', 'rss to email', 'google sheets')"
                },
                "category": {
                    "type": "string",
                    "description": "Filter by category (e.g. 'Discord', 'Google Sheets'). Optional."
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default: 10)"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_marketplace_workflow",
        "description": "Fetch a complete workflow from the n8n marketplace by ID. Returns the full workflow JSON and metadata (description, integrations, complexity). Use inspect_workflow next to see what credentials and params need filling before deployment.",
        "parameters": {
            "type": "object",
            "properties": {
                "workflow_id": {
                    "type": "string",
                    "description": "Marketplace workflow ID (from search_marketplace results)"
                }
            },
            "required": ["workflow_id"]
        }
    },
    {
        "name": "inspect_workflow",
        "description": "Analyze a workflow to identify required credentials, placeholder parameters, and configuration needs before deployment. Works on marketplace workflows, imported workflows, or any n8n workflow JSON. Cross-references with configured n8n credentials to show what's available vs missing.",
        "parameters": {
            "type": "object",
            "properties": {
                "workflow_json": {
                    "type": "object",
                    "description": "The workflow JSON to inspect"
                },
                "n8n_port": {
                    "type": "integer",
                    "description": "n8n port to check existing credentials against (default: 5678)"
                }
            },
            "required": ["workflow_json"]
        }
    },
    {
        "name": "configure_workflow",
        "description": "Configure a workflow by filling in credential IDs and parameter values. Use after inspect_workflow to fill identified gaps. Returns the configured workflow JSON ready for deploy_workflow or flash_workflow.",
        "parameters": {
            "type": "object",
            "properties": {
                "workflow_json": {
                    "type": "object",
                    "description": "The workflow JSON to configure"
                },
                "credential_map": {
                    "type": "object",
                    "description": "Map credential types to credential IDs: {\"slackWebhookApi\": \"5\", \"openAiApi\": \"12\"}"
                },
                "param_overrides": {
                    "type": "object",
                    "description": "Map node names to parameter overrides: {\"Send Slack\": {\"channel\": \"#alerts\"}, \"HTTP Request\": {\"url\": \"https://myapi.com\"}}"
                }
            },
            "required": ["workflow_json"]
        }
    },
]


class MarketplaceTools:
    """Tools for searching, fetching, inspecting, and configuring marketplace workflows."""

    def __init__(self, orchestrator, n8n_manager):
        self.orchestrator = orchestrator
        self.n8n_manager = n8n_manager

    # ------------------------------------------------------------------
    # Tool 1: search_marketplace
    # ------------------------------------------------------------------

    async def search_marketplace(
        self, query: str, category: str = "", limit: int = 10
    ) -> Dict[str, Any]:
        """Search the n8n marketplace for workflow templates."""
        try:
            params = {"q": query, "limit": str(min(limit, 50))}
            if category:
                params["category"] = category

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{AGENTNATE_BASE}/api/marketplace/search",
                    params=params,
                ) as resp:
                    if resp.status != 200:
                        return {"success": False, "error": f"Marketplace API error {resp.status}"}
                    data = await resp.json()

            results = data.get("results", [])
            # Simplify and truncate descriptions
            workflows = []
            for wf in results[:limit]:
                desc = wf.get("description", "") or ""
                if len(desc) > 200:
                    desc = desc[:197] + "..."
                workflows.append({
                    "id": wf.get("id"),
                    "name": wf.get("name"),
                    "description": desc,
                    "category": wf.get("category"),
                    "complexity": wf.get("complexity"),
                    "trigger_type": wf.get("trigger_type"),
                    "node_count": wf.get("node_count"),
                    "integrations": wf.get("integrations", []),
                    "views": wf.get("totalViews", 0),
                })

            return {
                "success": True,
                "query": query,
                "count": len(workflows),
                "workflows": workflows,
                "hint": "Use get_marketplace_workflow(workflow_id) to fetch the full workflow for any result.",
            }

        except Exception as e:
            logger.error(f"search_marketplace error: {e}")
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Tool 2: get_marketplace_workflow
    # ------------------------------------------------------------------

    async def get_marketplace_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Fetch a complete workflow from the marketplace."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{AGENTNATE_BASE}/api/marketplace/workflow/{workflow_id}",
                ) as resp:
                    if resp.status == 404:
                        return {"success": False, "error": f"Workflow {workflow_id} not found in marketplace"}
                    if resp.status != 200:
                        return {"success": False, "error": f"Marketplace API error {resp.status}"}
                    data = await resp.json()

            workflow_data = data.get("workflow", {})
            workflow_json = workflow_data.get("json", {})
            metadata = workflow_data.get("metadata", {})

            if not workflow_json or not workflow_json.get("nodes"):
                return {"success": False, "error": "Workflow has no nodes"}

            return {
                "success": True,
                "workflow_json": workflow_json,
                "metadata": {
                    "id": metadata.get("id"),
                    "name": metadata.get("name"),
                    "description": metadata.get("description", ""),
                    "category": metadata.get("category"),
                    "complexity": metadata.get("complexity"),
                    "trigger_type": metadata.get("trigger_type"),
                    "node_count": metadata.get("node_count"),
                    "integrations": metadata.get("integrations", []),
                },
                "hint": "Use inspect_workflow(workflow_json) to see what credentials and params need filling before deployment.",
            }

        except Exception as e:
            logger.error(f"get_marketplace_workflow error: {e}")
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Tool 3: inspect_workflow
    # ------------------------------------------------------------------

    async def inspect_workflow(
        self, workflow_json: Dict[str, Any], n8n_port: int = 5678
    ) -> Dict[str, Any]:
        """Analyze a workflow for required credentials and placeholder params."""
        try:
            workflow_json = _sanitize_surrogates(workflow_json)
            nodes = workflow_json.get("nodes", [])
            if not nodes:
                return {"success": False, "error": "Workflow has no nodes"}

            # --- Fetch existing credentials from n8n ---
            existing_creds = {}  # {type: [{id, name}]}
            existing_cred_ids = set()  # All local credential IDs
            n8n_running = False
            try:
                # Check if n8n is running (supports both legacy and queue manager)
                instance_found = False
                if hasattr(self.n8n_manager, 'instances'):
                    instance_found = n8n_port in self.n8n_manager.instances
                if not instance_found and hasattr(self.n8n_manager, 'main'):
                    if self.n8n_manager.main and self.n8n_manager.main.port == n8n_port:
                        instance_found = getattr(self.n8n_manager.main, 'is_running', False)
                        if callable(instance_found):
                            instance_found = instance_found()

                if instance_found:
                    from backend.routes.n8n import _get_or_create_auth
                    auth_cookie = await _get_or_create_auth(n8n_port)
                    headers = {}
                    if auth_cookie:
                        headers["Cookie"] = f"n8n-auth={auth_cookie}"
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            f"http://127.0.0.1:{n8n_port}/rest/credentials",
                            headers=headers,
                        ) as resp:
                            if resp.status == 200:
                                n8n_running = True
                                cred_data = await resp.json()
                                creds_list = cred_data.get("data", cred_data) if isinstance(cred_data, dict) else cred_data
                                if isinstance(creds_list, list):
                                    for c in creds_list:
                                        ctype = c.get("type", "")
                                        cid = c.get("id")
                                        if ctype not in existing_creds:
                                            existing_creds[ctype] = []
                                        existing_creds[ctype].append({
                                            "id": cid,
                                            "name": c.get("name"),
                                        })
                                        if cid:
                                            existing_cred_ids.add(str(cid))
            except Exception as e:
                logger.warning(f"Could not fetch existing credentials: {e}")

            # --- Analyze nodes ---
            credential_needs = []
            placeholders = []
            node_summary = []
            trigger_type = "manual"

            for node in nodes:
                node_name = node.get("name", "Unknown")
                node_type = node.get("type", "")

                # Detect trigger type
                if "webhook" in node_type.lower():
                    trigger_type = "webhook"
                elif "schedule" in node_type.lower() or "cron" in node_type.lower():
                    trigger_type = "schedule"
                elif "emailTrigger" in node_type or "email_trigger" in node_type.lower():
                    trigger_type = "email"

                # Extract short type name
                short_type = node_type.split(".")[-1] if "." in node_type else node_type
                node_summary.append({"name": node_name, "type": short_type})

                # --- Check credentials ---
                creds = node.get("credentials", {})
                cred_types_found = set()

                for cred_type, cred_ref in creds.items():
                    if not isinstance(cred_ref, dict):
                        continue
                    cred_types_found.add(cred_type)
                    cred_id = cred_ref.get("id", "")

                    if cred_id and n8n_running:
                        # n8n is running — verify the embedded ID exists locally
                        if str(cred_id) in existing_cred_ids:
                            status = "pre-configured"
                            available = []
                        elif cred_type in existing_creds:
                            # ID doesn't match but user has credentials of this type
                            status = "available"
                            available = existing_creds[cred_type]
                        else:
                            # ID doesn't match and no local credentials of this type
                            status = "missing"
                            available = []
                    elif cred_id and not n8n_running:
                        # n8n not running — embedded IDs are from marketplace author,
                        # treat as needing setup since we can't verify
                        status = "pre-configured"
                        available = []
                    elif cred_type in existing_creds:
                        status = "available"
                        available = existing_creds[cred_type]
                    else:
                        status = "missing"
                        available = []

                    credential_needs.append({
                        "node_name": node_name,
                        "credential_type": cred_type,
                        "status": status,
                        "available_credentials": available,
                    })

                # Fallback: if credentials dict is empty but node type is known
                # to require credentials (common with marketplace-stripped workflows),
                # flag the expected credential types as missing/available.
                if not creds and node_type in _NODE_TYPE_CREDENTIALS:
                    for expected_cred_type in _NODE_TYPE_CREDENTIALS[node_type]:
                        if expected_cred_type in cred_types_found:
                            continue  # Already handled above
                        if expected_cred_type in existing_creds:
                            status = "available"
                            available = existing_creds[expected_cred_type]
                        else:
                            status = "missing"
                            available = []
                        credential_needs.append({
                            "node_name": node_name,
                            "credential_type": expected_cred_type,
                            "status": status,
                            "available_credentials": available,
                        })

                # --- Check parameters for placeholders ---
                params = node.get("parameters", {})
                self._scan_params(node_name, params, "", placeholders)

            # --- Build summary ---
            cred_types_needed = list(set(c["credential_type"] for c in credential_needs))
            missing_count = sum(1 for c in credential_needs if c["status"] == "missing")
            needs_setup_count = sum(1 for c in credential_needs if c["status"] == "needs_setup")
            available_count = sum(1 for c in credential_needs if c["status"] == "available")
            ready = missing_count == 0 and needs_setup_count == 0 and len(placeholders) == 0

            return {
                "success": True,
                "summary": {
                    "name": workflow_json.get("name", "Unnamed"),
                    "node_count": len(nodes),
                    "trigger_type": trigger_type,
                    "nodes": node_summary,
                    "credential_types_needed": cred_types_needed,
                    "credentials_missing": missing_count,
                    "credentials_needs_setup": needs_setup_count,
                    "credentials_available": available_count,
                    "credentials_pre_configured": sum(1 for c in credential_needs if c["status"] == "pre-configured"),
                    "placeholders_found": len(placeholders),
                    "ready_to_deploy": ready,
                },
                "credentials": credential_needs,
                "placeholders": placeholders,
                "hint": (
                    "Ready to deploy! Use deploy_workflow or flash_workflow."
                    if ready else
                    "Use configure_workflow to fill in missing credentials and placeholders. "
                    "Use create_credential for any missing credential types. "
                    "Use ask_user to get API keys or service details."
                ),
            }

        except Exception as e:
            logger.error(f"inspect_workflow error: {e}")
            return {"success": False, "error": str(e)}

    def _scan_params(
        self, node_name: str, params: Any, path: str,
        placeholders: List[Dict],
    ) -> None:
        """Recursively scan parameters for placeholders."""
        if isinstance(params, dict):
            # Check for __rl objects with empty values
            if params.get("__rl") and not params.get("value"):
                placeholders.append({
                    "node_name": node_name,
                    "param_path": path or "(root)",
                    "current_value": "",
                    "placeholder_type": "empty_rl",
                })
                return

            for key, value in params.items():
                if key in ("__rl", "id", "typeVersion", "position"):
                    continue
                child_path = f"{path}.{key}" if path else key
                self._scan_params(node_name, value, child_path, placeholders)

        elif isinstance(params, str):
            if not params:
                return
            # Check for placeholder patterns
            for pattern in _PLACEHOLDER_PATTERNS:
                if pattern in params:
                    placeholders.append({
                        "node_name": node_name,
                        "param_path": path,
                        "current_value": params[:100],
                        "placeholder_type": "placeholder_text",
                    })
                    return

        elif isinstance(params, list):
            for i, item in enumerate(params):
                self._scan_params(node_name, item, f"{path}[{i}]", placeholders)

    # ------------------------------------------------------------------
    # Tool 4: configure_workflow
    # ------------------------------------------------------------------

    async def configure_workflow(
        self,
        workflow_json: Dict[str, Any],
        credential_map: Dict[str, str] = None,
        param_overrides: Dict[str, Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Configure a workflow by filling in credential IDs and parameter values."""
        try:
            wf = copy.deepcopy(_sanitize_surrogates(workflow_json))
            changes = []

            nodes = wf.get("nodes", [])
            if not nodes:
                return {"success": False, "error": "Workflow has no nodes"}

            # --- Apply credential_map ---
            if credential_map:
                for node in nodes:
                    node_name = node.get("name", "Unknown")
                    node_type = node.get("type", "")
                    creds = node.get("credentials", {})
                    matched_types = set()

                    for cred_type, cred_ref in creds.items():
                        if not isinstance(cred_ref, dict):
                            continue
                        if cred_type in credential_map:
                            matched_types.add(cred_type)
                            old_id = cred_ref.get("id", "")
                            new_id = str(credential_map[cred_type])
                            cred_ref["id"] = new_id
                            changes.append({
                                "node": node_name,
                                "change_type": "credential",
                                "detail": f"Set {cred_type} credential ID: '{old_id}' -> '{new_id}'",
                            })

                    # Fallback: if credentials dict is empty but node type is
                    # known to require credentials, inject them from the map.
                    if node_type in _NODE_TYPE_CREDENTIALS:
                        for expected_type in _NODE_TYPE_CREDENTIALS[node_type]:
                            if expected_type in matched_types:
                                continue
                            if expected_type in credential_map:
                                new_id = str(credential_map[expected_type])
                                if "credentials" not in node:
                                    node["credentials"] = {}
                                node["credentials"][expected_type] = {
                                    "id": new_id,
                                    "name": expected_type,
                                }
                                changes.append({
                                    "node": node_name,
                                    "change_type": "credential_added",
                                    "detail": f"Added {expected_type} credential ID: '{new_id}'",
                                })

            # --- Apply param_overrides ---
            if param_overrides:
                for node in nodes:
                    node_name = node.get("name", "Unknown")
                    if node_name not in param_overrides:
                        continue

                    overrides = param_overrides[node_name]
                    params = node.get("parameters", {})

                    for key, value in overrides.items():
                        old_value = params.get(key)

                        # Handle __rl objects: update the value field
                        if isinstance(old_value, dict) and old_value.get("__rl"):
                            old_inner = old_value.get("value", "")
                            old_value["value"] = value
                            changes.append({
                                "node": node_name,
                                "change_type": "param_rl",
                                "detail": f"Set {key}.value: '{old_inner}' -> '{value}'",
                            })
                        else:
                            params[key] = value
                            old_str = str(old_value)[:50] if old_value is not None else "(empty)"
                            new_str = str(value)[:50]
                            changes.append({
                                "node": node_name,
                                "change_type": "param",
                                "detail": f"Set {key}: '{old_str}' -> '{new_str}'",
                            })

            return {
                "success": True,
                "workflow_json": wf,
                "changes_made": changes,
                "changes_count": len(changes),
                "hint": "Use deploy_workflow to deploy, or flash_workflow for one-shot execution.",
            }

        except Exception as e:
            logger.error(f"configure_workflow error: {e}")
            return {"success": False, "error": str(e)}

