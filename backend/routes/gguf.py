"""
GGUF Model Search & Download Routes

Provides REST API for searching HuggingFace for GGUF models
and downloading them to the local models directory.
"""

import logging
from fastapi import APIRouter, Request, HTTPException, Query
from pydantic import BaseModel
from typing import Optional

logger = logging.getLogger("GGUF.Routes")

router = APIRouter()


def _get_downloader(request: Request):
    dl = getattr(request.app.state, 'gguf_downloader', None)
    if not dl:
        raise HTTPException(status_code=500, detail="GGUF downloader not initialized")
    return dl


# ======================== Search ========================

@router.get("/search")
async def search_models(
    request: Request,
    query: str = Query(..., min_length=1),
    author: Optional[str] = Query(None),
    sort: str = Query("downloads"),
    limit: int = Query(20, ge=1, le=50),
):
    """Search HuggingFace for GGUF model repositories."""
    dl = _get_downloader(request)
    return await dl.search(query, author=author, sort=sort, limit=limit)


@router.get("/files/{repo_owner}/{repo_name}")
async def list_files(repo_owner: str, repo_name: str, request: Request):
    """List GGUF files in a HuggingFace repository."""
    dl = _get_downloader(request)
    repo_id = f"{repo_owner}/{repo_name}"
    return await dl.get_model_files(repo_id)


# ======================== Downloads ========================

class DownloadRequest(BaseModel):
    repo_id: str
    filename: str


@router.post("/download")
async def start_download(body: DownloadRequest, request: Request):
    """Start downloading a GGUF model file."""
    dl = _get_downloader(request)
    return await dl.start_download(body.repo_id, body.filename)


@router.get("/downloads")
async def list_downloads(request: Request):
    """List all active and recent downloads."""
    dl = _get_downloader(request)
    return dl.list_downloads()


@router.get("/downloads/{download_id}")
async def download_status(download_id: str, request: Request):
    """Get status of a specific download."""
    dl = _get_downloader(request)
    return dl.get_download_status(download_id)


@router.delete("/downloads/{download_id}")
async def cancel_download(download_id: str, request: Request):
    """Cancel an active download."""
    dl = _get_downloader(request)
    return await dl.cancel_download(download_id)


# ======================== Directory ========================

@router.get("/directory")
async def get_directory(request: Request):
    """Get current models directory info."""
    dl = _get_downloader(request)
    return dl.get_directory_info()


@router.post("/directory/ensure")
async def ensure_directory(request: Request):
    """Ensure a models directory exists, creating the default if needed."""
    dl = _get_downloader(request)
    return await dl.ensure_models_directory()
