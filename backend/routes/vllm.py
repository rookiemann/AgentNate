"""
vLLM Module Routes

Provides API endpoints for vLLM module lifecycle management:
- Status checking
- Single-step install (download + bootstrap) with progress tracking
"""

import asyncio
import logging
from fastapi import APIRouter, Request, HTTPException

logger = logging.getLogger("vLLM.Routes")

router = APIRouter()


def _get_manager(request: Request):
    """Get vLLM manager from app state."""
    manager = getattr(request.app.state, 'vllm_manager', None)
    if not manager:
        raise HTTPException(status_code=500, detail="vLLM manager not initialized")
    return manager


# ======================== Status ========================

@router.get("/status")
async def get_status(request: Request):
    """Get vLLM module status."""
    manager = _get_manager(request)
    return manager.get_status()


# ======================== Install ========================

@router.post("/install")
async def install(request: Request):
    """Start vLLM installation (download + bootstrap) as a background task."""
    manager = _get_manager(request)

    # Don't await — run in background so the endpoint returns immediately
    asyncio.create_task(manager.install())

    return {"success": True, "message": "Installation started"}


@router.get("/install-progress")
async def install_progress(request: Request):
    """Get current install progress."""
    manager = _get_manager(request)
    return manager.get_install_progress()
