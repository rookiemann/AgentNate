"""
ComfyUI Module Routes

Provides API endpoints for ComfyUI lifecycle management.
Routes either handle operations directly (download, bootstrap, API server)
or proxy to the ComfyUI management API on port 5000.
"""

import logging
from fastapi import APIRouter, Request, HTTPException

logger = logging.getLogger("ComfyUI.Routes")

router = APIRouter()


def _get_manager(request: Request):
    """Get ComfyUI manager from app state."""
    manager = getattr(request.app.state, 'comfyui_manager', None)
    if not manager:
        raise HTTPException(status_code=500, detail="ComfyUI manager not initialized")
    return manager


async def _ensure_api(manager):
    """Ensure the management API is running, raise if not."""
    if not await manager.is_api_running():
        raise HTTPException(status_code=503, detail="ComfyUI API server not running. Start it first.")


# ======================== Status ========================

@router.get("/status")
async def get_status(request: Request):
    """Get combined ComfyUI module status."""
    manager = _get_manager(request)
    return await manager.get_status()


# ======================== Module Lifecycle ========================

@router.post("/module/download")
async def download_module(request: Request):
    """Clone the portable installer from GitHub."""
    manager = _get_manager(request)
    return await manager.download_module()


@router.post("/module/bootstrap")
async def bootstrap_module(request: Request):
    """Run bootstrap: download Python, Git, FFmpeg."""
    manager = _get_manager(request)
    return await manager.bootstrap()


@router.post("/module/update")
async def update_module(request: Request):
    """Update the portable installer via git pull."""
    manager = _get_manager(request)
    return await manager.update_module()


# ======================== API Server ========================

@router.post("/api/start")
async def start_api_server(request: Request):
    """Start the ComfyUI management API server."""
    manager = _get_manager(request)
    return await manager.start_api_server()


@router.post("/api/stop")
async def stop_api_server(request: Request):
    """Stop the ComfyUI management API server."""
    manager = _get_manager(request)
    return await manager.stop_api_server()


# ======================== Installation (proxy) ========================

@router.post("/install")
async def install_comfyui(request: Request):
    """Trigger full ComfyUI installation (returns job_id)."""
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("POST", "/api/install")


@router.post("/update")
async def update_comfyui(request: Request):
    """Update ComfyUI via git pull (returns job_id)."""
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("POST", "/api/update")


@router.post("/purge")
async def purge_comfyui(request: Request):
    """Purge ComfyUI installation (keeps models + Python)."""
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("POST", "/api/purge")


# ======================== Instances (proxy) ========================

@router.get("/instances")
async def list_instances(request: Request):
    """List all ComfyUI instances."""
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("GET", "/api/instances")


@router.post("/instances")
async def add_instance(request: Request):
    """Add a new ComfyUI instance."""
    manager = _get_manager(request)
    await _ensure_api(manager)
    body = await request.json()
    return await manager.proxy("POST", "/api/instances", json=body)


@router.delete("/instances/{instance_id}")
async def remove_instance(request: Request, instance_id: str):
    """Remove a ComfyUI instance."""
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("DELETE", f"/api/instances/{instance_id}")


@router.post("/instances/{instance_id}/start")
async def start_instance(request: Request, instance_id: str):
    """Start a specific ComfyUI instance."""
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("POST", f"/api/instances/{instance_id}/start")


@router.post("/instances/{instance_id}/stop")
async def stop_instance(request: Request, instance_id: str):
    """Stop a specific ComfyUI instance."""
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("POST", f"/api/instances/{instance_id}/stop")


@router.post("/instances/start-all")
async def start_all_instances(request: Request):
    """Start all ComfyUI instances."""
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("POST", "/api/instances/start-all")


@router.post("/instances/stop-all")
async def stop_all_instances(request: Request):
    """Stop all ComfyUI instances."""
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("POST", "/api/instances/stop-all")


# ======================== GPUs (proxy) ========================

@router.get("/gpus")
async def list_gpus(request: Request):
    """List available GPUs."""
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("GET", "/api/gpus")


# ======================== Models (proxy) ========================

@router.get("/models/registry")
async def models_registry(request: Request):
    """List models from the curated registry."""
    manager = _get_manager(request)
    await _ensure_api(manager)
    category = request.query_params.get("category")
    params = {"category": category} if category else None
    return await manager.proxy("GET", "/api/models/registry", params=params)


@router.get("/models/local")
async def models_local(request: Request):
    """Scan locally installed models."""
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("GET", "/api/models/local")


@router.get("/models/categories")
async def models_categories(request: Request):
    """List model categories."""
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("GET", "/api/models/categories")


@router.post("/models/download")
async def download_models(request: Request):
    """Download models by ID (returns job_id)."""
    manager = _get_manager(request)
    await _ensure_api(manager)
    body = await request.json()
    return await manager.proxy("POST", "/api/models/download", json=body)


@router.get("/models/search")
async def search_models(request: Request):
    """Search HuggingFace for models."""
    manager = _get_manager(request)
    await _ensure_api(manager)
    q = request.query_params.get("q", "")
    return await manager.proxy("GET", "/api/models/search", params={"q": q})


# ======================== Custom Nodes (proxy) ========================

@router.get("/nodes/registry")
async def nodes_registry(request: Request):
    """List curated custom nodes with install status."""
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("GET", "/api/nodes/registry")


@router.get("/nodes/installed")
async def nodes_installed(request: Request):
    """List installed custom nodes."""
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("GET", "/api/nodes/installed")


@router.post("/nodes/install")
async def install_nodes(request: Request):
    """Install custom nodes by ID (returns job_id)."""
    manager = _get_manager(request)
    await _ensure_api(manager)
    body = await request.json()
    return await manager.proxy("POST", "/api/nodes/install", json=body)


@router.post("/nodes/update-all")
async def update_all_nodes(request: Request):
    """Update all installed custom nodes."""
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("POST", "/api/nodes/update-all")


@router.delete("/nodes/{node_name}")
async def remove_node(request: Request, node_name: str):
    """Remove a custom node."""
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("DELETE", f"/api/nodes/{node_name}")


# ======================== Jobs (proxy) ========================

@router.get("/jobs/{job_id}")
async def get_job(request: Request, job_id: str):
    """Get async job status and progress."""
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("GET", f"/api/jobs/{job_id}")


# ======================== External ComfyUI (proxy) ========================

@router.get("/external")
async def list_external(request: Request):
    """List saved external ComfyUI directories."""
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("GET", "/api/comfyui/saved")


@router.post("/external")
async def add_external(request: Request):
    """Add an external ComfyUI directory."""
    manager = _get_manager(request)
    await _ensure_api(manager)
    body = await request.json()
    return await manager.proxy("POST", "/api/comfyui/saved", json=body)


@router.delete("/external")
async def remove_external(request: Request):
    """Remove an external ComfyUI directory."""
    manager = _get_manager(request)
    await _ensure_api(manager)
    body = await request.json()
    return await manager.proxy("DELETE", "/api/comfyui/saved", params=body)


@router.put("/target")
async def set_target(request: Request):
    """Switch active ComfyUI target directory."""
    manager = _get_manager(request)
    await _ensure_api(manager)
    body = await request.json()
    return await manager.proxy("PUT", "/api/comfyui/target", json=body)


# ======================== Settings (proxy) ========================

@router.get("/settings")
async def get_settings(request: Request):
    """Get ComfyUI module settings."""
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("GET", "/api/settings")


@router.put("/settings")
async def update_settings(request: Request):
    """Update ComfyUI module settings."""
    manager = _get_manager(request)
    await _ensure_api(manager)
    body = await request.json()
    return await manager.proxy("PUT", "/api/settings", json=body)


# ==================== Simple Generation Endpoints (for n8n) ====================

from pydantic import BaseModel
from typing import Optional


class SimpleGenerateBody(BaseModel):
    instance_id: str
    prompt: str
    checkpoint: str
    negative_prompt: str = "blurry, low quality, distorted"
    seed: int = -1
    width: Optional[int] = None
    height: Optional[int] = None
    steps: Optional[int] = None
    cfg: Optional[float] = None
    sampler_name: Optional[str] = None
    scheduler: Optional[str] = None


@router.post("/generate")
async def simple_generate(body: SimpleGenerateBody, request: Request):
    """
    Simple image generation endpoint for n8n workflows.
    Builds a txt2img workflow, queues it on the specified instance.
    Returns prompt_id for polling via /result endpoint.
    """
    import httpx
    from backend.comfyui_utils import detect_model_defaults, build_txt2img_workflow

    manager = _get_manager(request)
    await _ensure_api(manager)

    # Auto-detect defaults from checkpoint name
    defaults = detect_model_defaults(body.checkpoint)
    width = body.width or defaults["width"]
    height = body.height or defaults["height"]
    steps = body.steps or defaults["steps"]
    cfg = body.cfg if body.cfg is not None else defaults["cfg"]
    sampler = body.sampler_name or defaults["sampler_name"]
    scheduler = body.scheduler or defaults["scheduler"]

    workflow = build_txt2img_workflow(
        body.checkpoint, body.prompt, body.negative_prompt,
        width, height, steps, cfg, body.seed, sampler, scheduler,
    )

    # Resolve instance to port
    try:
        instances_data = await manager.proxy("GET", "/api/instances")
        instances = instances_data if isinstance(instances_data, list) else instances_data.get("instances", [])
        port = None
        for inst in instances:
            if str(inst.get("id")) == str(body.instance_id) or str(inst.get("instance_id")) == str(body.instance_id):
                port = inst.get("port")
                break
        if not port:
            raise HTTPException(404, f"ComfyUI instance '{body.instance_id}' not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to resolve instance: {e}")

    # Queue the prompt
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"http://127.0.0.1:{port}/prompt",
                json={"prompt": workflow},
            )
            if resp.status_code >= 400:
                raise HTTPException(resp.status_code, f"ComfyUI error: {resp.text[:500]}")
            result = resp.json()
    except httpx.ConnectError:
        raise HTTPException(503, f"Cannot connect to ComfyUI instance on port {port}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to queue prompt: {e}")

    return {
        "success": True,
        "prompt_id": result.get("prompt_id"),
        "instance_id": body.instance_id,
        "port": port,
    }


@router.get("/result/{instance_id}/{prompt_id}")
async def get_generation_result(instance_id: str, prompt_id: str, request: Request):
    """
    Poll for image generation result.
    Returns status + image URLs when complete.
    """
    import httpx

    manager = _get_manager(request)
    await _ensure_api(manager)

    # Resolve instance to port
    try:
        instances_data = await manager.proxy("GET", "/api/instances")
        instances = instances_data if isinstance(instances_data, list) else instances_data.get("instances", [])
        port = None
        for inst in instances:
            if str(inst.get("id")) == str(instance_id) or str(inst.get("instance_id")) == str(instance_id):
                port = inst.get("port")
                break
        if not port:
            raise HTTPException(404, f"ComfyUI instance '{instance_id}' not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to resolve instance: {e}")

    # Check history
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"http://127.0.0.1:{port}/history/{prompt_id}")
            if resp.status_code >= 400:
                raise HTTPException(resp.status_code, f"ComfyUI error: {resp.text[:500]}")
            history = resp.json()
    except httpx.ConnectError:
        raise HTTPException(503, f"Cannot connect to ComfyUI instance on port {port}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to check history: {e}")

    if prompt_id not in history:
        return {"success": True, "status": "running", "prompt_id": prompt_id}

    entry = history[prompt_id]
    status_info = entry.get("status", {})

    if status_info.get("status_str") == "error":
        return {
            "success": False,
            "status": "failed",
            "prompt_id": prompt_id,
            "error": str(status_info.get("messages", [])),
        }

    # Extract output images
    outputs = entry.get("outputs", {})
    images = []
    for node_id, node_output in outputs.items():
        for img in node_output.get("images", []):
            images.append({
                "filename": img["filename"],
                "subfolder": img.get("subfolder", ""),
                "type": img.get("type", "output"),
                "view_url": f"http://127.0.0.1:{port}/view?filename={img['filename']}&subfolder={img.get('subfolder', '')}&type={img.get('type', 'output')}",
            })

    return {
        "success": True,
        "status": "completed",
        "prompt_id": prompt_id,
        "images": images,
        "image_count": len(images),
    }
