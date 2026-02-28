"""
vLLM Module Routes

Provides API endpoints for vLLM module lifecycle management:
- Status checking
- Module download from GitHub releases
- Bootstrap (install.bat)
- Download progress tracking
"""

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


# ======================== Module Lifecycle ========================

@router.post("/module/download")
async def download_module(request: Request):
    """Download vLLM release from GitHub."""
    manager = _get_manager(request)
    return await manager.download_module()


@router.post("/module/bootstrap")
async def bootstrap_module(request: Request):
    """Run install.bat to set up Python + vLLM."""
    manager = _get_manager(request)
    return await manager.bootstrap()


# ======================== Progress ========================

@router.get("/module/download-progress")
async def download_progress(request: Request):
    """Get current download progress."""
    manager = _get_manager(request)
    return manager.get_download_progress()
