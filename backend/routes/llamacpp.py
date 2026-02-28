"""
llama.cpp Routes

API endpoints for llama-cpp-python wheel installation:
- Status checking
- One-click install (download wheel + pip install)
- Install progress tracking
"""

import logging
from fastapi import APIRouter, Request, HTTPException

logger = logging.getLogger("LlamaCpp.Routes")

router = APIRouter()


def _get_manager(request: Request):
    """Get llama.cpp manager from app state."""
    manager = getattr(request.app.state, 'llamacpp_manager', None)
    if not manager:
        raise HTTPException(status_code=500, detail="llama.cpp manager not initialized")
    return manager


@router.get("/status")
async def get_status(request: Request):
    """Get llama-cpp-python installation status."""
    manager = _get_manager(request)
    return manager.get_status()


@router.post("/install")
async def install(request: Request):
    """Download CUDA wheel and pip install llama-cpp-python."""
    manager = _get_manager(request)
    return await manager.install_wheel()


@router.get("/install-progress")
async def install_progress(request: Request):
    """Get current install progress."""
    manager = _get_manager(request)
    return manager.get_install_progress()
