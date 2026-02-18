"""
Workflow Generator

Uses LLM to generate n8n workflows from natural language descriptions.
"""

import json
import re
import logging
import os
from typing import Dict, Any, Optional, List
import uuid

from .workflow_templates import (
    build_workflow, node_id, get_node_types
)

logger = logging.getLogger("workflow_generator")

# Add file handler for debug logging
_debug_log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "workflow_debug.log")
_file_handler = logging.FileHandler(_debug_log_path, mode='a', encoding='utf-8')
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(_file_handler)
logger.setLevel(logging.DEBUG)


SYSTEM_PROMPT = '''You are an n8n workflow generator. Your job is to create n8n workflow JSON from natural language descriptions.

{templates}

## Response Format
You MUST respond with valid JSON only. The workflow structure is:
```json
{{
  "name": "Workflow Name",
  "active": true,
  "settings": {{
    "executionOrder": "v1"
  }},
  "nodes": [
    {{
      "id": "unique-uuid",
      "name": "Node Display Name",
      "type": "n8n-nodes-base.nodeType",
      "typeVersion": 1,
      "position": [x, y],
      "parameters": {{}}
    }}
  ],
  "connections": {{
    "Source Node Name": {{
      "main": [[{{"node": "Target Node Name", "type": "main", "index": 0}}]]
    }}
  }}
}}
```

## Important Rules
1. Node positions should flow left to right: trigger at [250,300], processing at [450,300], [650,300], etc.
2. Use $json.fieldName to reference data from previous nodes
3. For LLM calls, ALWAYS use http://localhost:8000/api/chat/completions (NEVER use host.docker.internal or other Docker URLs - this is a portable app)
4. Generate unique UUIDs for each node id
5. Connection keys use the node "name" field, not the "id"
6. Respond with ONLY the JSON, no markdown code blocks, no explanation
7. REQUIRED: Always include "active": true in the workflow root object
8. REQUIRED: Always include "settings": {{"executionOrder": "v1"}} in the workflow root object

## Code Node Limitations
The n8n Code node runs in a SANDBOX - it does NOT have access to Node.js modules like fs, path, http, etc.
- NEVER use require() in Code nodes
- NEVER use fs.writeFileSync or fs.readFileSync
- For file operations, use the readWriteFile node instead

## Writing Files to Disk
To save data to a file:
1. Use HTTP Request with responseFormat: "file" to get binary data
2. Connect to readWriteFile node with operation: "write"

Example - Fetch webpage and save as HTML file:
```json
{{
  "type": "n8n-nodes-base.httpRequest",
  "parameters": {{
    "method": "GET",
    "url": "https://example.com",
    "options": {{"response": {{"response": {{"responseFormat": "file"}}}}}}
  }}
}}
```
Then connect to:
```json
{{
  "type": "n8n-nodes-base.readWriteFile",
  "parameters": {{
    "operation": "write",
    "fileName": "E:\\\\path\\\\to\\\\file.html",
    "dataPropertyName": "data"
  }}
}}
```

## Example LLM Node
```json
{{
  "id": "abc-123",
  "name": "Process with AI",
  "type": "n8n-nodes-base.httpRequest",
  "typeVersion": 4,
  "position": [650, 300],
  "parameters": {{
    "method": "POST",
    "url": "http://localhost:8000/api/chat/completions",
    "sendHeaders": true,
    "headerParameters": {{
      "parameters": [{{"name": "Content-Type", "value": "application/json"}}]
    }},
    "sendBody": true,
    "specifyBody": "json",
    "jsonBody": "={{ \\"instance_id\\": \\"default\\", \\"messages\\": [{{\\"role\\": \\"user\\", \\"content\\": $json.input}}], \\"max_tokens\\": 500 }}"
  }}
}}
```
'''


def create_generation_prompt(user_description: str, trigger_type: str = "webhook") -> str:
    """Create the full prompt for workflow generation."""
    node_types = get_node_types()
    templates = "# Available Node Types\n\n"
    for type_name, desc in node_types.items():
        templates += f"- **{type_name}**: {desc}\n"
    system = SYSTEM_PROMPT.format(templates=templates)

    user_prompt = f"""Create an n8n workflow that does the following:

{user_description}

Use trigger type: {trigger_type}

Remember:
- Respond with ONLY valid JSON
- Include all necessary nodes
- Connect nodes in the correct order
- Use realistic parameter values
"""
    return system, user_prompt


def extract_json_from_response(response: str) -> Optional[Dict]:
    """Extract JSON from LLM response, handling markdown code blocks."""
    # Try direct parse first
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try to extract from markdown code block
    patterns = [
        r'```json\s*([\s\S]*?)\s*```',
        r'```\s*([\s\S]*?)\s*```',
        r'\{[\s\S]*\}'
    ]

    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            try:
                json_str = match.group(1) if '```' in pattern else match.group(0)
                return json.loads(json_str)
            except (json.JSONDecodeError, IndexError):
                continue

    return None


def validate_workflow(workflow: Dict) -> tuple[bool, List[str]]:
    """Validate a workflow structure. Returns (is_valid, errors)."""
    errors = []

    if not isinstance(workflow, dict):
        return False, ["Workflow must be a dictionary"]

    if "nodes" not in workflow:
        errors.append("Missing 'nodes' array")
    elif not isinstance(workflow["nodes"], list):
        errors.append("'nodes' must be an array")
    elif len(workflow["nodes"]) == 0:
        errors.append("Workflow has no nodes")

    if "connections" not in workflow:
        errors.append("Missing 'connections' object")
    elif not isinstance(workflow["connections"], dict):
        errors.append("'connections' must be an object")

    if "name" not in workflow:
        errors.append("Missing 'name' field")

    # Validate nodes have required fields
    node_names = set()
    if "nodes" in workflow and isinstance(workflow["nodes"], list):
        for i, node in enumerate(workflow["nodes"]):
            if not isinstance(node, dict):
                errors.append(f"Node {i} is not a dictionary")
                continue
            if "type" not in node:
                errors.append(f"Node {i} missing 'type'")
            if "name" not in node:
                errors.append(f"Node {i} missing 'name'")
            else:
                node_names.add(node["name"])
            if "position" not in node:
                errors.append(f"Node {i} missing 'position'")

    # Validate connections reference existing node names
    conn_errors = validate_connections(workflow, node_names)
    errors.extend(conn_errors)

    return len(errors) == 0, errors


def validate_connections(workflow: Dict, node_names: set = None) -> List[str]:
    """
    Validate that connections reference existing node names.

    Returns list of error messages.
    """
    errors = []
    connections = workflow.get("connections", {})

    if node_names is None:
        node_names = {n.get("name") for n in workflow.get("nodes", []) if n.get("name")}

    logger.debug(f"[WORKFLOW DEBUG] Node names in workflow: {node_names}")
    logger.debug(f"[WORKFLOW DEBUG] Connection sources: {list(connections.keys())}")

    for source_name, outputs in connections.items():
        # Check if source node exists
        if source_name not in node_names:
            errors.append(f"Connection source '{source_name}' not found in nodes")
            logger.warning(f"[WORKFLOW DEBUG] Connection source '{source_name}' NOT FOUND in nodes: {node_names}")

        # Check targets
        if isinstance(outputs, dict) and "main" in outputs:
            for output_idx, output_connections in enumerate(outputs["main"]):
                if isinstance(output_connections, list):
                    for conn in output_connections:
                        if isinstance(conn, dict):
                            target_name = conn.get("node")
                            if target_name and target_name not in node_names:
                                errors.append(f"Connection target '{target_name}' not found in nodes")
                                logger.warning(f"[WORKFLOW DEBUG] Connection target '{target_name}' NOT FOUND in nodes: {node_names}")

    return errors


def fix_workflow(workflow: Dict) -> Dict:
    """Attempt to fix common issues in generated workflows."""
    logger.info("[WORKFLOW DEBUG] === Starting fix_workflow ===")

    # Ensure required fields
    if "name" not in workflow:
        workflow["name"] = "Generated Workflow"

    if "nodes" not in workflow:
        workflow["nodes"] = []

    if "connections" not in workflow:
        workflow["connections"] = {}

    # Ensure settings is a proper dict with executionOrder
    if "settings" not in workflow or not isinstance(workflow.get("settings"), dict):
        workflow["settings"] = {
            "executionOrder": "v1",
            "saveManualExecutions": True
        }
    elif "executionOrder" not in workflow["settings"]:
        workflow["settings"]["executionOrder"] = "v1"

    # n8n requires the 'active' field - default to True so workflows are ready to use
    if "active" not in workflow:
        workflow["active"] = True

    # Log nodes before fixing
    logger.info(f"[WORKFLOW DEBUG] Nodes BEFORE fix ({len(workflow.get('nodes', []))} nodes):")
    for i, node in enumerate(workflow.get("nodes", [])):
        logger.info(f"  [{i}] name='{node.get('name')}' type='{node.get('type')}' id='{node.get('id', 'NO ID')}'")

    # Build short-name → full n8n type mapping for type normalization
    try:
        from backend.workflow_templates import get_short_to_n8n_type_map
        type_map = get_short_to_n8n_type_map()
    except Exception:
        type_map = {}

    # Ensure each node has required fields
    for i, node in enumerate(workflow.get("nodes", [])):
        if "id" not in node:
            node["id"] = node_id()
        if "position" not in node:
            node["position"] = [250 + (i * 200), 300]
        if "typeVersion" not in node:
            node["typeVersion"] = 1
        if "parameters" not in node:
            node["parameters"] = {}

        # Normalize short type names to full n8n type strings
        node_type = node.get("type", "")
        if node_type and not node_type.startswith("n8n-nodes-base.") and not node_type.startswith("@n8n/"):
            if node_type in type_map:
                node["type"] = type_map[node_type]
                logger.info(f"[WORKFLOW DEBUG] Normalized node type: '{node_type}' -> '{node['type']}'")
            elif node_type.replace("_", "").lower() in {k.replace("_", "").lower(): k for k in type_map}:
                # Fuzzy match (ignore underscores and case)
                fuzzy = {k.replace("_", "").lower(): k for k in type_map}
                matched_key = fuzzy[node_type.replace("_", "").lower()]
                node["type"] = type_map[matched_key]
                logger.info(f"[WORKFLOW DEBUG] Fuzzy-matched node type: '{node_type}' -> '{node['type']}')")

    # Add workflow-level IDs if missing
    if "id" not in workflow:
        workflow["id"] = node_id()
    if "versionId" not in workflow:
        workflow["versionId"] = node_id()

    # Fix connections
    workflow = fix_connections(workflow)

    # Log final state
    logger.info(f"[WORKFLOW DEBUG] Nodes AFTER fix ({len(workflow.get('nodes', []))} nodes):")
    for i, node in enumerate(workflow.get("nodes", [])):
        logger.info(f"  [{i}] name='{node.get('name')}' type='{node.get('type')}'")

    logger.info(f"[WORKFLOW DEBUG] Connections AFTER fix:")
    for source, outputs in workflow.get("connections", {}).items():
        if isinstance(outputs, dict) and "main" in outputs:
            for conn_list in outputs["main"]:
                for conn in conn_list:
                    target = conn.get("node") if isinstance(conn, dict) else conn
                    logger.info(f"  '{source}' -> '{target}'")

    return workflow


def fix_connections(workflow: Dict) -> Dict:
    """
    Attempt to fix connection issues in the workflow.

    Common problems:
    - Connection source/target names don't match node names exactly
    - Using node ID instead of node name
    """
    nodes = workflow.get("nodes", [])
    connections = workflow.get("connections", {})

    if not nodes or not connections:
        return workflow

    # Build lookup maps
    node_names = {n.get("name") for n in nodes if n.get("name")}
    node_id_to_name = {n.get("id"): n.get("name") for n in nodes if n.get("id") and n.get("name")}
    node_name_lower = {n.lower(): n for n in node_names}

    logger.info(f"[WORKFLOW DEBUG] fix_connections - node_names: {node_names}")
    logger.info(f"[WORKFLOW DEBUG] fix_connections - node_id_to_name: {node_id_to_name}")

    fixed_connections = {}

    for source_name, outputs in connections.items():
        # Try to fix source name
        fixed_source = source_name
        if source_name not in node_names:
            # Try case-insensitive match
            if source_name.lower() in node_name_lower:
                fixed_source = node_name_lower[source_name.lower()]
                logger.info(f"[WORKFLOW DEBUG] Fixed source '{source_name}' -> '{fixed_source}' (case)")
            # Try ID lookup
            elif source_name in node_id_to_name:
                fixed_source = node_id_to_name[source_name]
                logger.info(f"[WORKFLOW DEBUG] Fixed source '{source_name}' -> '{fixed_source}' (id->name)")
            else:
                logger.warning(f"[WORKFLOW DEBUG] Could not fix source '{source_name}'")

        # Process outputs and fix targets
        if isinstance(outputs, dict) and "main" in outputs:
            fixed_main = []
            for output_connections in outputs["main"]:
                fixed_output = []
                if isinstance(output_connections, list):
                    for conn in output_connections:
                        if isinstance(conn, dict):
                            target_name = conn.get("node", "")
                            fixed_target = target_name

                            if target_name not in node_names:
                                # Try case-insensitive match
                                if target_name.lower() in node_name_lower:
                                    fixed_target = node_name_lower[target_name.lower()]
                                    logger.info(f"[WORKFLOW DEBUG] Fixed target '{target_name}' -> '{fixed_target}' (case)")
                                # Try ID lookup
                                elif target_name in node_id_to_name:
                                    fixed_target = node_id_to_name[target_name]
                                    logger.info(f"[WORKFLOW DEBUG] Fixed target '{target_name}' -> '{fixed_target}' (id->name)")
                                else:
                                    logger.warning(f"[WORKFLOW DEBUG] Could not fix target '{target_name}'")

                            fixed_conn = conn.copy()
                            fixed_conn["node"] = fixed_target
                            fixed_output.append(fixed_conn)
                        else:
                            fixed_output.append(conn)
                fixed_main.append(fixed_output)

            fixed_connections[fixed_source] = {"main": fixed_main}
        else:
            fixed_connections[fixed_source] = outputs

    workflow["connections"] = fixed_connections
    return workflow


async def generate_workflow(
    orchestrator,
    description: str,
    trigger_type: str = "webhook",
    model_instance_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a workflow using the LLM.

    Args:
        orchestrator: ModelOrchestrator instance
        description: Natural language description of the workflow
        trigger_type: Type of trigger (webhook, schedule, manual)
        model_instance_id: Specific model to use (None for default)

    Returns:
        Dict with 'success', 'workflow', and 'errors' keys
    """
    from providers.base import InferenceRequest, ChatMessage

    # Get a model to use
    if model_instance_id is None:
        # Find any loaded model
        instances = orchestrator.get_loaded_instances()
        if not instances:
            return {
                "success": False,
                "workflow": None,
                "errors": ["No models loaded. Load a model first."]
            }
        model_instance_id = instances[0].id

    # Create the prompt
    system_prompt, user_prompt = create_generation_prompt(description, trigger_type)

    # Create inference request
    request = InferenceRequest(
        messages=[
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt)
        ],
        max_tokens=2000,
        temperature=0.3,  # Lower temperature for more consistent JSON
    )

    # Generate
    logger.info("=" * 60)
    logger.info(f"[GENERATE DEBUG] === Generating Workflow ===")
    logger.info(f"[GENERATE DEBUG] Description: {description}")
    logger.info(f"[GENERATE DEBUG] Trigger type: {trigger_type}")
    logger.info(f"[GENERATE DEBUG] Using model: {model_instance_id}")

    full_response = ""
    try:
        async for response in orchestrator.chat(model_instance_id, request):
            full_response += response.text
    except Exception as e:
        logger.error(f"[GENERATE DEBUG] LLM error: {e}")
        return {
            "success": False,
            "workflow": None,
            "errors": [f"LLM error: {str(e)}"]
        }

    logger.info(f"[GENERATE DEBUG] Raw LLM response ({len(full_response)} chars):")
    logger.info(full_response[:2000])
    if len(full_response) > 2000:
        logger.info(f"... (truncated, {len(full_response) - 2000} more chars)")

    # Parse JSON
    workflow = extract_json_from_response(full_response)
    if workflow is None:
        logger.error(f"[GENERATE DEBUG] Failed to parse JSON from response")
        return {
            "success": False,
            "workflow": None,
            "errors": ["Failed to parse JSON from LLM response"],
            "raw_response": full_response[:500]
        }

    logger.info(f"[GENERATE DEBUG] Parsed workflow - name: {workflow.get('name')}")
    logger.info(f"[GENERATE DEBUG] Parsed nodes: {[n.get('name') for n in workflow.get('nodes', [])]}")
    logger.info(f"[GENERATE DEBUG] Parsed connections: {list(workflow.get('connections', {}).keys())}")

    # Validate
    is_valid, errors = validate_workflow(workflow)
    logger.info(f"[GENERATE DEBUG] Initial validation: valid={is_valid}, errors={errors}")

    # Always run fix_workflow to ensure required fields (like settings.executionOrder) are present
    logger.info(f"[GENERATE DEBUG] Running fix_workflow to ensure all required fields...")
    workflow = fix_workflow(workflow)
    is_valid, errors = validate_workflow(workflow)
    logger.info(f"[GENERATE DEBUG] After fix: valid={is_valid}, errors={errors}")

    logger.info("=" * 60)

    return {
        "success": is_valid,
        "workflow": workflow,
        "errors": errors if not is_valid else []
    }


# ============================================================
# Quick workflow builders (no LLM needed)
# ============================================================

def quick_webhook_llm_workflow(
    name: str = "Webhook to LLM",
    webhook_path: str = "chat",
    input_field: str = "message",
    max_tokens: int = 1024
) -> Dict[str, Any]:
    """Build a webhook -> LLM -> respond workflow without LLM generation."""

    webhook_id = node_id()
    llm_id = node_id()
    respond_id = node_id()

    nodes = [
        {
            "id": webhook_id,
            "name": "Webhook",
            "type": "n8n-nodes-base.webhook",
            "typeVersion": 1,
            "position": [250, 300],
            "webhookId": node_id(),
            "parameters": {
                "path": webhook_path,
                "httpMethod": "POST",
                "responseMode": "responseNode"
            }
        },
        {
            "id": llm_id,
            "name": "Local LLM",
            "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4,
            "position": [500, 300],
            "parameters": {
                "method": "POST",
                "url": "http://localhost:8000/api/chat/completions",
                "sendHeaders": True,
                "headerParameters": {
                    "parameters": [
                        {"name": "Content-Type", "value": "application/json"}
                    ]
                },
                "sendBody": True,
                "specifyBody": "json",
                "jsonBody": f'''={{
  "instance_id": "{{{{ $env.DEFAULT_MODEL_ID }}}}",
  "messages": [{{"role": "user", "content": "{{{{ $json.body.{input_field} }}}}"}}],
  "max_tokens": {max_tokens},
  "stream": false
}}'''
            }
        },
        {
            "id": respond_id,
            "name": "Respond",
            "type": "n8n-nodes-base.respondToWebhook",
            "typeVersion": 1,
            "position": [750, 300],
            "parameters": {
                "respondWith": "json",
                "responseBody": "={{ $json }}"
            }
        }
    ]

    connections = {
        "Webhook": {
            "main": [[{"node": "Local LLM", "type": "main", "index": 0}]]
        },
        "Local LLM": {
            "main": [[{"node": "Respond", "type": "main", "index": 0}]]
        }
    }

    return build_workflow(name, nodes, connections)


def quick_schedule_summary_workflow(
    name: str = "Scheduled Summary",
    cron: str = "0 9 * * *",
    source_url: str = "https://api.example.com/data"
) -> Dict[str, Any]:
    """Build a schedule -> fetch -> summarize workflow."""

    schedule_id = node_id()
    fetch_id = node_id()
    summarize_id = node_id()
    notify_id = node_id()

    nodes = [
        {
            "id": schedule_id,
            "name": "Schedule",
            "type": "n8n-nodes-base.scheduleTrigger",
            "typeVersion": 1,
            "position": [250, 300],
            "parameters": {
                "rule": {
                    "interval": [{"field": "cronExpression", "expression": cron}]
                }
            }
        },
        {
            "id": fetch_id,
            "name": "Fetch Data",
            "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4,
            "position": [450, 300],
            "parameters": {
                "method": "GET",
                "url": source_url
            }
        },
        {
            "id": summarize_id,
            "name": "Summarize",
            "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4,
            "position": [650, 300],
            "parameters": {
                "method": "POST",
                "url": "http://localhost:8000/api/chat/completions",
                "sendHeaders": True,
                "headerParameters": {
                    "parameters": [
                        {"name": "Content-Type", "value": "application/json"}
                    ]
                },
                "sendBody": True,
                "specifyBody": "json",
                "jsonBody": '''={
  "instance_id": "{{ $env.DEFAULT_MODEL_ID }}",
  "messages": [
    {"role": "system", "content": "Summarize the following data concisely."},
    {"role": "user", "content": "{{ JSON.stringify($json) }}"}
  ],
  "max_tokens": 500
}'''
            }
        },
        {
            "id": notify_id,
            "name": "Log Result",
            "type": "n8n-nodes-base.set",
            "typeVersion": 3,
            "position": [850, 300],
            "parameters": {
                "mode": "manual",
                "assignments": {
                    "assignments": [
                        {
                            "id": node_id(),
                            "name": "summary",
                            "value": "={{ $json.content }}",
                            "type": "string"
                        }
                    ]
                }
            }
        }
    ]

    connections = {
        "Schedule": {
            "main": [[{"node": "Fetch Data", "type": "main", "index": 0}]]
        },
        "Fetch Data": {
            "main": [[{"node": "Summarize", "type": "main", "index": 0}]]
        },
        "Summarize": {
            "main": [[{"node": "Log Result", "type": "main", "index": 0}]]
        }
    }

    return build_workflow(name, nodes, connections)


def create_quick_workflow(
    template: str,
    name: Optional[str] = None,
    webhook_path: Optional[str] = None,
    config: Optional[Dict] = None
) -> Optional[Dict[str, Any]]:
    """
    Create a workflow from a template ID.

    Supports:
    - Recipe templates: webhook_llm_respond, scheduled_summary, discord_ai_bot, etc.
    - Single node templates: webhook, schedule, local_llm_chat, etc.
    """
    config = config or {}

    # Recipe templates (multi-node workflows)
    if template in ("webhook_llm", "webhook_llm_respond"):
        return quick_webhook_llm_workflow(
            name=name or "Webhook + AI Response",
            webhook_path=webhook_path or config.get("webhook_path", "ai-chat")
        )

    if template in ("schedule_summary", "scheduled_summary"):
        return quick_schedule_summary_workflow(
            name=name or "Daily AI Summary",
            cron=config.get("cron", "0 9 * * *")
        )

    if template == "discord_ai_bot":
        return _create_discord_ai_bot(
            name=name or "Discord AI Bot",
            webhook_path=webhook_path or config.get("webhook_path", "discord-bot")
        )

    if template == "sentiment_classifier":
        return _create_sentiment_classifier(
            name=name or "Sentiment Classifier",
            webhook_path=webhook_path or config.get("webhook_path", "classify"),
            categories=config.get("categories", "positive, negative, neutral")
        )

    if template == "email_summarizer":
        return _create_email_summarizer(
            name=name or "Email Summarizer"
        )

    if template == "json_extractor":
        return _create_json_extractor(
            name=name or "Data Extractor",
            webhook_path=webhook_path or config.get("webhook_path", "extract"),
            schema=config.get("schema", "name, email, phone")
        )

    # Single node templates
    if template in TRIGGERS:
        return _create_single_trigger_workflow(template, name)

    if template in AI_NODES:
        return _create_single_ai_workflow(template, name)

    if template in ACTION_NODES:
        return _create_single_action_workflow(template, name)

    if template in DATA_NODES:
        return _create_single_data_workflow(template, name)

    return None


def _create_discord_ai_bot(name: str, webhook_path: str) -> Dict[str, Any]:
    """Webhook -> LLM -> Discord workflow."""
    nodes = [
        {
            "id": node_id(),
            "name": "Discord Webhook",
            "type": "n8n-nodes-base.webhook",
            "typeVersion": 1,
            "position": [250, 300],
            "webhookId": node_id(),
            "parameters": {
                "path": webhook_path,
                "httpMethod": "POST",
                "responseMode": "onReceived"
            }
        },
        {
            "id": node_id(),
            "name": "AI Response",
            "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4,
            "position": [500, 300],
            "parameters": {
                "method": "POST",
                "url": "http://localhost:8000/api/chat/completions",
                "sendHeaders": True,
                "headerParameters": {
                    "parameters": [{"name": "Content-Type", "value": "application/json"}]
                },
                "sendBody": True,
                "specifyBody": "json",
                "jsonBody": '''={
  "instance_id": "{{ $env.DEFAULT_MODEL_ID }}",
  "messages": [{"role": "user", "content": "{{ $json.body.content }}"}],
  "max_tokens": 500
}'''
            }
        },
        {
            "id": node_id(),
            "name": "Send to Discord",
            "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4,
            "position": [750, 300],
            "parameters": {
                "method": "POST",
                "url": "={{ $env.DISCORD_WEBHOOK }}",
                "sendHeaders": True,
                "headerParameters": {
                    "parameters": [{"name": "Content-Type", "value": "application/json"}]
                },
                "sendBody": True,
                "specifyBody": "json",
                "jsonBody": '={"content": "{{ $json.content }}"}'
            }
        }
    ]

    connections = {
        "Discord Webhook": {"main": [[{"node": "AI Response", "type": "main", "index": 0}]]},
        "AI Response": {"main": [[{"node": "Send to Discord", "type": "main", "index": 0}]]}
    }

    return build_workflow(name, nodes, connections)


def _create_sentiment_classifier(name: str, webhook_path: str, categories: str) -> Dict[str, Any]:
    """Webhook -> Classify -> Conditional branching."""
    nodes = [
        {
            "id": node_id(),
            "name": "Webhook",
            "type": "n8n-nodes-base.webhook",
            "typeVersion": 1,
            "position": [250, 300],
            "webhookId": node_id(),
            "parameters": {
                "path": webhook_path,
                "httpMethod": "POST",
                "responseMode": "responseNode"
            }
        },
        {
            "id": node_id(),
            "name": "Classify Sentiment",
            "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4,
            "position": [500, 300],
            "parameters": {
                "method": "POST",
                "url": "http://localhost:8000/api/chat/completions",
                "sendHeaders": True,
                "headerParameters": {
                    "parameters": [{"name": "Content-Type", "value": "application/json"}]
                },
                "sendBody": True,
                "specifyBody": "json",
                "jsonBody": f'''={{
  "instance_id": "{{{{ $env.DEFAULT_MODEL_ID }}}}",
  "messages": [
    {{"role": "system", "content": "Classify the sentiment into: {categories}. Reply with only the category."}},
    {{"role": "user", "content": "{{{{ $json.body.text }}}}"}}
  ],
  "max_tokens": 20
}}'''
            }
        },
        {
            "id": node_id(),
            "name": "Respond",
            "type": "n8n-nodes-base.respondToWebhook",
            "typeVersion": 1,
            "position": [750, 300],
            "parameters": {
                "respondWith": "json",
                "responseBody": '={"sentiment": "{{ $json.content }}", "text": "{{ $json.body.text }}"}'
            }
        }
    ]

    connections = {
        "Webhook": {"main": [[{"node": "Classify Sentiment", "type": "main", "index": 0}]]},
        "Classify Sentiment": {"main": [[{"node": "Respond", "type": "main", "index": 0}]]}
    }

    return build_workflow(name, nodes, connections)


def _create_email_summarizer(name: str) -> Dict[str, Any]:
    """Email trigger -> Summarize -> Slack notification."""
    nodes = [
        {
            "id": node_id(),
            "name": "Email Trigger",
            "type": "n8n-nodes-base.emailReadImap",
            "typeVersion": 2,
            "position": [250, 300],
            "parameters": {
                "mailbox": "INBOX",
                "options": {}
            }
        },
        {
            "id": node_id(),
            "name": "Summarize Email",
            "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4,
            "position": [500, 300],
            "parameters": {
                "method": "POST",
                "url": "http://localhost:8000/api/chat/completions",
                "sendHeaders": True,
                "headerParameters": {
                    "parameters": [{"name": "Content-Type", "value": "application/json"}]
                },
                "sendBody": True,
                "specifyBody": "json",
                "jsonBody": '''={
  "instance_id": "{{ $env.DEFAULT_MODEL_ID }}",
  "messages": [
    {"role": "system", "content": "Summarize this email in 2-3 sentences."},
    {"role": "user", "content": "Subject: {{ $json.subject }}\n\n{{ $json.text }}"}
  ],
  "max_tokens": 200
}'''
            }
        },
        {
            "id": node_id(),
            "name": "Send to Slack",
            "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4,
            "position": [750, 300],
            "parameters": {
                "method": "POST",
                "url": "={{ $env.SLACK_WEBHOOK }}",
                "sendHeaders": True,
                "headerParameters": {
                    "parameters": [{"name": "Content-Type", "value": "application/json"}]
                },
                "sendBody": True,
                "specifyBody": "json",
                "jsonBody": '={"text": "*New Email Summary*\\n{{ $json.content }}"}'
            }
        }
    ]

    connections = {
        "Email Trigger": {"main": [[{"node": "Summarize Email", "type": "main", "index": 0}]]},
        "Summarize Email": {"main": [[{"node": "Send to Slack", "type": "main", "index": 0}]]}
    }

    return build_workflow(name, nodes, connections)


def _create_json_extractor(name: str, webhook_path: str, schema: str) -> Dict[str, Any]:
    """Webhook -> Extract JSON -> Respond."""
    nodes = [
        {
            "id": node_id(),
            "name": "Webhook",
            "type": "n8n-nodes-base.webhook",
            "typeVersion": 1,
            "position": [250, 300],
            "webhookId": node_id(),
            "parameters": {
                "path": webhook_path,
                "httpMethod": "POST",
                "responseMode": "responseNode"
            }
        },
        {
            "id": node_id(),
            "name": "Extract Data",
            "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4,
            "position": [500, 300],
            "parameters": {
                "method": "POST",
                "url": "http://localhost:8000/api/chat/completions",
                "sendHeaders": True,
                "headerParameters": {
                    "parameters": [{"name": "Content-Type", "value": "application/json"}]
                },
                "sendBody": True,
                "specifyBody": "json",
                "jsonBody": f'''={{
  "instance_id": "{{{{ $env.DEFAULT_MODEL_ID }}}}",
  "messages": [
    {{"role": "system", "content": "Extract these fields: {schema}. Return ONLY valid JSON."}},
    {{"role": "user", "content": "{{{{ $json.body.text }}}}"}}
  ],
  "max_tokens": 500
}}'''
            }
        },
        {
            "id": node_id(),
            "name": "Parse JSON",
            "type": "n8n-nodes-base.code",
            "typeVersion": 2,
            "position": [750, 300],
            "parameters": {
                "mode": "runOnceForEachItem",
                "jsCode": '''try {
  const extracted = JSON.parse($json.content);
  return { json: extracted };
} catch (e) {
  return { json: { error: "Failed to parse", raw: $json.content } };
}'''
            }
        },
        {
            "id": node_id(),
            "name": "Respond",
            "type": "n8n-nodes-base.respondToWebhook",
            "typeVersion": 1,
            "position": [1000, 300],
            "parameters": {
                "respondWith": "json",
                "responseBody": "={{ $json }}"
            }
        }
    ]

    connections = {
        "Webhook": {"main": [[{"node": "Extract Data", "type": "main", "index": 0}]]},
        "Extract Data": {"main": [[{"node": "Parse JSON", "type": "main", "index": 0}]]},
        "Parse JSON": {"main": [[{"node": "Respond", "type": "main", "index": 0}]]}
    }

    return build_workflow(name, nodes, connections)


def _create_single_trigger_workflow(template: str, name: Optional[str]) -> Dict[str, Any]:
    """Create a workflow with a single trigger node."""
    trigger_info = TRIGGERS[template]
    trigger_node = trigger_info["template"]()
    trigger_node["name"] = name or trigger_node.get("name", template.replace("_", " ").title())

    return build_workflow(
        name or f"{template.replace('_', ' ').title()} Workflow",
        [trigger_node],
        {}
    )


def _create_single_ai_workflow(template: str, name: Optional[str]) -> Dict[str, Any]:
    """Create a workflow with webhook + AI node + respond."""
    ai_info = AI_NODES[template]

    webhook = TRIGGERS["webhook"]["template"](path="ai-" + template.replace("_", "-"))
    webhook["name"] = "Webhook"

    ai_node = ai_info["template"]()
    ai_node["name"] = name or template.replace("_", " ").title()
    ai_node["position"] = [500, 300]

    respond = ACTION_NODES["respond_webhook"]["template"]()
    respond["name"] = "Respond"
    respond["position"] = [750, 300]

    connections = {
        "Webhook": {"main": [[{"node": ai_node["name"], "type": "main", "index": 0}]]},
        ai_node["name"]: {"main": [[{"node": "Respond", "type": "main", "index": 0}]]}
    }

    return build_workflow(
        name or f"{template.replace('_', ' ').title()} Workflow",
        [webhook, ai_node, respond],
        connections
    )


def _create_single_action_workflow(template: str, name: Optional[str]) -> Dict[str, Any]:
    """Create a workflow with manual trigger + action node."""
    trigger = TRIGGERS["manual"]["template"]()
    trigger["name"] = "Manual Trigger"

    action_info = ACTION_NODES[template]
    action_node = action_info["template"]()
    action_node["name"] = name or template.replace("_", " ").title()
    action_node["position"] = [450, 300]

    connections = {
        "Manual Trigger": {"main": [[{"node": action_node["name"], "type": "main", "index": 0}]]}
    }

    return build_workflow(
        name or f"{template.replace('_', ' ').title()} Workflow",
        [trigger, action_node],
        connections
    )


def _create_single_data_workflow(template: str, name: Optional[str]) -> Dict[str, Any]:
    """Create a workflow with manual trigger + data node."""
    trigger = TRIGGERS["manual"]["template"]()
    trigger["name"] = "Manual Trigger"

    data_info = DATA_NODES[template]
    data_node = data_info["template"]()
    data_node["name"] = name or template.replace("_", " ").title()
    data_node["position"] = [450, 300]

    connections = {
        "Manual Trigger": {"main": [[{"node": data_node["name"], "type": "main", "index": 0}]]}
    }

    return build_workflow(
        name or f"{template.replace('_', ' ').title()} Workflow",
        [trigger, data_node],
        connections
    )
