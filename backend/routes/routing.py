"""
Model Routing Preset Routes

REST API for managing routing presets that map persona IDs to specific
provider/model combinations. Key endpoint: /resolve/{persona_id} enables
n8n workflows to route tasks to the optimal loaded model.
"""

import logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from backend.routing_presets import RoutingPresetManager

logger = logging.getLogger("routing")
router = APIRouter()


class SavePresetBody(BaseModel):
    name: str
    routes: Dict[str, Any]
    description: str = ""


class ActivateBody(BaseModel):
    enabled: bool
    preset_id: Optional[str] = None


def _get_manager(request: Request) -> RoutingPresetManager:
    settings = request.app.state.settings
    return RoutingPresetManager(settings)


@router.get("/presets")
async def list_presets(request: Request):
    """List all saved routing presets."""
    mgr = _get_manager(request)
    return {"presets": mgr.list_presets()}


@router.get("/presets/{preset_id}")
async def get_preset(preset_id: str, request: Request):
    """Get a specific routing preset."""
    mgr = _get_manager(request)
    preset = mgr.get_preset(preset_id)
    if not preset:
        raise HTTPException(404, f"Routing preset not found: {preset_id}")
    return preset


@router.post("/presets")
async def save_preset(body: SavePresetBody, request: Request):
    """Save a new routing preset."""
    mgr = _get_manager(request)
    preset = mgr.save_preset(body.name, body.routes, body.description)
    return {"success": True, "preset": preset}


@router.delete("/presets/{preset_id}")
async def delete_preset(preset_id: str, request: Request):
    """Delete a routing preset."""
    mgr = _get_manager(request)
    if mgr.delete_preset(preset_id):
        return {"success": True}
    raise HTTPException(404, f"Routing preset not found: {preset_id}")


@router.post("/activate")
async def activate_routing(body: ActivateBody, request: Request):
    """Enable or disable model routing."""
    settings = request.app.state.settings
    if body.enabled:
        if not body.preset_id:
            raise HTTPException(400, "preset_id required when enabling routing")
        mgr = _get_manager(request)
        preset = mgr.get_preset(body.preset_id)
        if not preset:
            raise HTTPException(404, f"Routing preset not found: {body.preset_id}")
        settings.set("agent.routing_enabled", True)
        settings.set("agent.active_routing_preset_id", body.preset_id)
        return {"success": True, "routing_enabled": True, "preset": preset["name"]}
    else:
        settings.set("agent.routing_enabled", False)
        settings.set("agent.active_routing_preset_id", None)
        return {"success": True, "routing_enabled": False}


@router.get("/status")
async def routing_status(request: Request):
    """Get current routing state and resolved mappings."""
    settings = request.app.state.settings
    enabled = settings.get("agent.routing_enabled", False)
    preset_id = settings.get("agent.active_routing_preset_id")

    result = {
        "routing_enabled": enabled,
        "active_preset_id": preset_id,
        "active_preset_name": None,
        "routes": {},
    }

    if enabled and preset_id:
        mgr = _get_manager(request)
        preset = mgr.get_preset(preset_id)
        if preset:
            result["active_preset_name"] = preset.get("name")
            result["routes"] = preset.get("routes", {})

    return result


@router.get("/resolve/{persona_id}")
async def resolve_persona(persona_id: str, request: Request):
    """
    Resolve a persona to a loaded model instance_id.
    Key endpoint for n8n workflow integration.

    Returns: {instance_id, provider, model, resolution}
    """
    settings = request.app.state.settings
    orchestrator = request.app.state.orchestrator

    enabled = settings.get("agent.routing_enabled", False)
    if not enabled:
        return {
            "instance_id": None,
            "resolution": "disabled",
            "message": "Model routing is not enabled.",
        }

    preset_id = settings.get("agent.active_routing_preset_id")
    if not preset_id:
        return {
            "instance_id": None,
            "resolution": "no_preset",
            "message": "No active routing preset.",
        }

    mgr = _get_manager(request)
    loaded = orchestrator.get_loaded_instances()
    instance_id = mgr.resolve(preset_id, persona_id, loaded)

    if instance_id:
        instance = orchestrator.get_instance(instance_id)
        return {
            "instance_id": instance_id,
            "provider": instance.provider_type.value if instance else None,
            "model": instance.display_name or instance.model_identifier if instance else None,
            "resolution": "routed",
        }

    return {
        "instance_id": None,
        "resolution": "no_match",
        "message": f"No loaded model matches the route for persona '{persona_id}'.",
    }


class WorkflowGenBody(BaseModel):
    pattern: str
    config: Dict[str, Any] = {}


@router.post("/presets/{preset_id}/workflow")
async def generate_preset_workflow(preset_id: str, body: WorkflowGenBody, request: Request):
    """Generate an n8n workflow JSON from a routing preset + pattern."""
    mgr = _get_manager(request)
    preset = mgr.get_preset(preset_id)
    if not preset:
        raise HTTPException(404, f"Routing preset not found: {preset_id}")

    config = dict(body.config)
    config["preset_id"] = preset_id
    config.setdefault("name", f"{preset['name']} - {body.pattern}")

    from backend.workflow_bridge import generate_workflow, PATTERNS
    if body.pattern not in PATTERNS:
        raise HTTPException(400, f"Unknown pattern: {body.pattern}. Available: {list(PATTERNS.keys())}")

    try:
        workflow = generate_workflow(body.pattern, config)
        return {"success": True, "workflow": workflow, "pattern": body.pattern}
    except ValueError as e:
        raise HTTPException(400, str(e))
