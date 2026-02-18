"""
Workflow Routes

Endpoints for workflow generation and management.
"""

import json
import logging
import os
from typing import Optional, Dict, Any
from fastapi import APIRouter, Request
from pydantic import BaseModel

logger = logging.getLogger("workflows")
router = APIRouter()
AGENTNATE_BASE = os.getenv("AGENTNATE_BASE_URL", "http://127.0.0.1:8000")


class GenerateRequest(BaseModel):
    description: str
    trigger_type: str = "webhook"
    model_instance_id: Optional[str] = None


class QuickWorkflowRequest(BaseModel):
    template: str  # webhook_llm, schedule_summary, discord_ai_bot, etc.
    name: Optional[str] = None
    webhook_path: Optional[str] = "chat"
    category: Optional[str] = None
    config: Optional[dict] = None


class BuildWorkflowRequest(BaseModel):
    """Request for the unified workflow builder."""
    name: str
    nodes: list  # List of node specs with "type" and optional params


class DeployRequest(BaseModel):
    workflow: dict
    n8n_port: int = 5678
    activate: bool = False


@router.post("/generate")
async def generate_workflow(request: Request, body: GenerateRequest):
    """
    Generate a workflow from natural language description.

    Uses the LLM to create n8n workflow JSON.
    """
    from backend.workflow_generator import generate_workflow

    orchestrator = request.app.state.orchestrator

    result = await generate_workflow(
        orchestrator,
        body.description,
        body.trigger_type,
        body.model_instance_id
    )

    return result


@router.post("/quick")
async def create_quick_workflow(request: Request, body: QuickWorkflowRequest):
    """
    Create a workflow from a pre-built template (no LLM needed).
    """
    from backend.workflow_generator import create_quick_workflow as gen_quick_workflow

    try:
        workflow = gen_quick_workflow(
            template=body.template,
            name=body.name,
            webhook_path=body.webhook_path,
            config=body.config
        )

        if workflow is None:
            return {
                "success": False,
                "error": f"Unknown template: {body.template}",
                "available": [
                    "webhook_llm_respond", "scheduled_summary", "discord_ai_bot",
                    "sentiment_classifier", "email_summarizer", "json_extractor",
                    "webhook", "schedule", "manual", "local_llm_chat", "classify",
                    "extract_json", "discord_webhook", "slack_webhook"
                ]
            }

        return {
            "success": True,
            "workflow": workflow
        }

    except Exception as e:
        logger.error(f"Quick workflow error: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@router.post("/deploy")
async def deploy_workflow(request: Request, body: DeployRequest):
    """
    Deploy a workflow to an n8n instance.
    """
    import aiohttp
    from backend.routes.n8n import _get_or_create_auth

    n8n_manager = request.app.state.n8n_manager
    queue_manager = getattr(request.app.state, 'n8n_queue_manager', None)
    port = body.n8n_port

    # Check both legacy manager and queue manager for the instance
    instance_found = port in n8n_manager.instances
    if not instance_found and queue_manager:
        if queue_manager.main and queue_manager.main.port == port:
            instance_found = True
        elif port in queue_manager.workers:
            instance_found = True

    if not instance_found:
        return {
            "success": False,
            "error": f"n8n not running on port {port}"
        }

    try:
        auth_cookie = await _get_or_create_auth(port)
        headers = {"Content-Type": "application/json"}
        if auth_cookie:
            headers["Cookie"] = f"n8n-auth={auth_cookie}"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://127.0.0.1:{port}/rest/workflows",
                json=body.workflow,
                headers=headers
            ) as resp:
                if resp.status in (200, 201):
                    data = await resp.json()
                    workflow_id = data.get("id") or data.get("data", {}).get("id")

                    if body.activate and workflow_id:
                        await session.patch(
                            f"http://127.0.0.1:{port}/rest/workflows/{workflow_id}",
                            json={"active": True},
                            headers=headers
                        )

                    return {
                        "success": True,
                        "workflow_id": workflow_id,
                        "url": f"{AGENTNATE_BASE}/api/n8n/{port}/proxy/workflow/{workflow_id}"
                    }
                else:
                    error_text = await resp.text()
                    return {
                        "success": False,
                        "error": f"n8n error {resp.status}: {error_text[:200]}"
                    }

    except Exception as e:
        logger.error(f"Deploy error: {e}")
        return {"success": False, "error": str(e)}


@router.post("/build")
async def build_workflow(body: BuildWorkflowRequest):
    """
    Build a workflow from node specifications using templates.

    This is the unified workflow builder that generates valid n8n JSON
    from simple node specs.
    """
    from backend.workflow_templates import build_workflow_from_nodes, get_node_types

    try:
        workflow = build_workflow_from_nodes(body.name, body.nodes, connection_mode="linear")

        return {
            "success": True,
            "workflow": workflow,
            "nodes": [n.get("name") for n in workflow["nodes"]]
        }

    except ValueError as e:
        return {
            "success": False,
            "error": str(e),
            "available_types": get_node_types()
        }
    except Exception as e:
        logger.error(f"Build workflow error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/node-types")
async def list_node_types():
    """List all available node types for building workflows."""
    from backend.workflow_templates import get_node_types

    return {
        "node_types": get_node_types(),
        "usage": "Pass these as 'type' in the nodes array when calling POST /build"
    }


class ImportWorkflowRequest(BaseModel):
    workflow_json: dict


class ImportDeployRequest(BaseModel):
    workflow_json: dict
    mode: str = "once"
    loop_target: Optional[int] = None


@router.post("/import")
async def import_workflow(body: ImportWorkflowRequest):
    """
    Validate and fix an imported workflow JSON.

    Performs structural validation (no LLM needed):
    - Valid dict with required n8n fields (nodes, connections)
    - Each node has type, name, position
    - Connections reference valid node names
    - Applies fix_workflow() to add missing IDs, meta fields

    Returns validation results, fixed JSON, and a summary.
    """
    from backend.workflow_generator import validate_workflow, fix_workflow

    workflow = body.workflow_json
    warnings = []

    # Step 1: Validate structure
    is_valid, errors = validate_workflow(workflow)

    # Step 2: Try to fix regardless (fix_workflow adds missing fields)
    try:
        import copy
        fixed = copy.deepcopy(workflow)
        fixed = fix_workflow(fixed)

        # Re-validate after fix
        is_valid_after, remaining_errors = validate_workflow(fixed)

        # Track what was fixed
        if not is_valid and is_valid_after:
            warnings.append("Some issues were auto-fixed (missing IDs, positions, settings)")

        # Build summary
        nodes = fixed.get("nodes", [])
        node_types = [n.get("type", "unknown") for n in nodes]

        # Detect trigger type
        trigger_type = "manual"
        for nt in node_types:
            if "webhook" in nt.lower():
                trigger_type = "webhook"
                break
            elif "cron" in nt.lower() or "schedule" in nt.lower():
                trigger_type = "schedule"
                break
            elif "trigger" in nt.lower():
                trigger_type = nt.split(".")[-1] if "." in nt else "trigger"
                break

        # Extract integrations (unique service names from node types)
        integrations = []
        for nt in node_types:
            parts = nt.replace("n8n-nodes-base.", "").split(".")
            service = parts[0] if parts else nt
            if service not in integrations and service not in ("manualTrigger", "noOp", "set", "code", "function"):
                integrations.append(service)

        summary = {
            "name": fixed.get("name", "Unnamed Workflow"),
            "node_count": len(nodes),
            "trigger_type": trigger_type,
            "integrations": integrations,
        }

        return {
            "valid": is_valid_after,
            "errors": remaining_errors,
            "warnings": warnings,
            "fixed_json": fixed,
            "summary": summary,
        }

    except Exception as e:
        logger.error(f"Import validation error: {e}")
        return {
            "valid": False,
            "errors": errors + [f"Fix failed: {str(e)}"],
            "warnings": warnings,
            "fixed_json": None,
            "summary": None,
        }


@router.post("/import/deploy")
async def import_deploy(request: Request, body: ImportDeployRequest):
    """
    Deploy an imported workflow.

    Uses the same pipeline as deploy-and-run:
    1. Auto-starts Main Admin if needed
    2. Deploys workflow to Main Admin
    3. Spawns isolated worker

    The workflow_json should be the fixed_json from the /import endpoint.
    """
    import httpx
    import asyncio
    from backend.routes.n8n import _get_or_create_auth

    queue_manager = getattr(request.app.state, 'n8n_queue_manager', None)
    if not queue_manager:
        return {"success": False, "error": "Queue manager not initialized"}

    try:
        # 1. Ensure main admin is running
        main_status = queue_manager.get_main_status()
        if not main_status.get("running"):
            logger.info("Auto-starting main admin for import deploy")
            await queue_manager.start_main()
            await asyncio.sleep(2)

        # 2. Deploy to main admin
        main_port = queue_manager.main_port
        auth_cookie = await _get_or_create_auth(main_port)

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"http://127.0.0.1:{main_port}/rest/workflows",
                json=body.workflow_json,
                headers={"Cookie": f"n8n-auth={auth_cookie}"} if auth_cookie else {},
            )

            if resp.status_code not in (200, 201):
                return {
                    "success": False,
                    "error": f"Failed to deploy: {resp.status_code} - {resp.text[:200]}",
                }

            workflow_data = resp.json()
            workflow_id = workflow_data.get("data", {}).get("id") or workflow_data.get("id")

            if not workflow_id:
                return {"success": False, "error": "Deployed but no workflow ID returned"}

        # 3. Spawn worker
        worker = await queue_manager.spawn_worker(
            workflow_id=workflow_id,
            mode=body.mode,
            loop_target=body.loop_target,
        )

        return {
            "success": True,
            "workflow_id": workflow_id,
            "worker": worker.to_dict(),
        }

    except Exception as e:
        logger.error(f"Import deploy error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/templates")
async def list_templates():
    """List available workflow templates with details."""
    from backend.workflow_templates import get_all_templates

    raw_templates = get_all_templates()

    # Convert to list format with metadata
    templates = {}
    for category, items in raw_templates.items():
        templates[category] = [
            {
                "type": name,
                "name": name.replace("_", " ").title(),
                "description": info.get("description", "")
            }
            for name, info in items.items()
        ]

    return {"templates": templates}


@router.get("/templates/{category}")
async def get_template_category(category: str):
    """Get templates in a category."""
    from backend.workflow_templates import get_all_templates

    templates = get_all_templates()

    if category not in templates:
        return {
            "error": f"Unknown category: {category}",
            "available": list(templates.keys())
        }

    return {
        "category": category,
        "templates": {
            name: info.get("description", "")
            for name, info in templates[category].items()
        }
    }


# ==================== Marketplace Inspection ====================

class InspectRequest(BaseModel):
    workflow_json: dict
    n8n_port: int = 5678


class ConfigureRequest(BaseModel):
    workflow_json: dict
    credential_map: Optional[Dict[str, str]] = None
    param_overrides: Optional[Dict[str, Dict[str, Any]]] = None


@router.post("/inspect")
async def inspect_workflow(request: Request, body: InspectRequest):
    """
    Inspect a workflow for missing credentials and placeholder values.

    Returns a structured report of what needs to be configured before deployment.
    Used by the marketplace UI to show a configuration panel.
    """
    from backend.tools.marketplace_tools import MarketplaceTools

    orchestrator = request.app.state.orchestrator
    n8n_manager = getattr(request.app.state, 'n8n_queue_manager', None) or \
                  getattr(request.app.state, 'n8n_manager', None)

    tools = MarketplaceTools(orchestrator, n8n_manager)
    result = await tools.inspect_workflow(
        workflow_json=body.workflow_json,
        n8n_port=body.n8n_port,
    )
    return result


@router.post("/configure")
async def configure_workflow(request: Request, body: ConfigureRequest):
    """
    Patch a workflow with credential IDs and parameter overrides.

    Returns the modified workflow JSON ready for deployment.
    Used after the user fills in the inspection panel form.
    """
    from backend.tools.marketplace_tools import MarketplaceTools

    orchestrator = request.app.state.orchestrator
    n8n_manager = getattr(request.app.state, 'n8n_queue_manager', None) or \
                  getattr(request.app.state, 'n8n_manager', None)

    tools = MarketplaceTools(orchestrator, n8n_manager)
    result = await tools.configure_workflow(
        workflow_json=body.workflow_json,
        credential_map=body.credential_map or {},
        param_overrides=body.param_overrides or {},
    )
    return result
