"""
Media routes — Image proxy, generation catalog, gallery API, input pipeline.
"""

import os
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel

logger = logging.getLogger("routes.media")

router = APIRouter()


# ================================================================
# Request models
# ================================================================

class GenerationUpdate(BaseModel):
    title: Optional[str] = None
    tags: Optional[str] = None
    rating: Optional[int] = None
    favorite: Optional[int] = None
    notes: Optional[str] = None


class InputFromGeneration(BaseModel):
    generation_id: str
    file_index: int = 0
    instance_id: str = ""


class InputFromUrl(BaseModel):
    url: str
    instance_id: str = ""


class BatchIds(BaseModel):
    ids: list


class BatchTag(BaseModel):
    ids: list
    tags: str


# ================================================================
# Helpers
# ================================================================

def _get_catalog(request: Request):
    catalog = getattr(request.app.state, "media_catalog", None)
    if not catalog:
        raise HTTPException(status_code=503, detail="Media catalog not initialized")
    return catalog


def _get_output_dir(request: Request) -> str:
    """Resolve ComfyUI output directory."""
    catalog = _get_catalog(request)
    if catalog.output_dir and os.path.isdir(catalog.output_dir):
        return catalog.output_dir
    # Fallback: try standard location
    base = Path(__file__).parent.parent.parent / "modules" / "comfyui" / "comfyui" / "output"
    return str(base)


def _get_input_dir(request: Request) -> str:
    """Resolve ComfyUI input directory."""
    output_dir = _get_output_dir(request)
    # input/ is sibling to output/
    return os.path.join(os.path.dirname(output_dir), "input")


# ================================================================
# Image proxy — serves files directly from disk
# ================================================================

@router.get("/images/{filename}")
async def serve_image(filename: str, request: Request,
                      subfolder: str = "", type: str = "output"):
    """Serve a ComfyUI image directly from disk. Works even if ComfyUI is stopped."""
    output_dir = _get_output_dir(request)

    # Security: prevent path traversal
    if ".." in filename or filename.startswith("/") or filename.startswith("\\"):
        raise HTTPException(status_code=400, detail="Invalid filename")
    if ".." in subfolder:
        raise HTTPException(status_code=400, detail="Invalid subfolder")

    # Resolve path
    if type == "input":
        base_dir = _get_input_dir(request)
    else:
        base_dir = output_dir

    if subfolder:
        filepath = os.path.join(base_dir, subfolder, filename)
    else:
        filepath = os.path.join(base_dir, filename)

    if not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    # Determine media type
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "png"
    media_types = {
        # Images
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
        "gif": "image/gif",
        "bmp": "image/bmp",
        "tiff": "image/tiff",
        "tif": "image/tiff",
        # Video
        "mp4": "video/mp4",
        "webm": "video/webm",
        "mkv": "video/x-matroska",
        "avi": "video/x-msvideo",
        "mov": "video/quicktime",
        # Audio
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "flac": "audio/flac",
        "ogg": "audio/ogg",
        "m4a": "audio/mp4",
        "aac": "audio/aac",
    }
    media_type = media_types.get(ext, "application/octet-stream")

    return FileResponse(
        filepath,
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=3600"},
    )


# ================================================================
# Generation catalog CRUD
# ================================================================

@router.get("/media/generations")
async def list_generations(
    request: Request,
    query: str = None,
    checkpoint: str = None,
    tags: str = None,
    favorite: Optional[bool] = None,
    min_rating: Optional[int] = None,
    date_from: str = None,
    date_to: str = None,
    sort: str = "newest",
    limit: int = 50,
    offset: int = 0,
):
    catalog = _get_catalog(request)
    results, total = catalog.search_generations(
        query=query, checkpoint=checkpoint, tags=tags,
        favorite=favorite, min_rating=min_rating,
        date_from=date_from, date_to=date_to,
        sort=sort, limit=limit, offset=offset,
    )
    return {"generations": results, "total": total}


@router.get("/media/generations/{generation_id}")
async def get_generation(generation_id: str, request: Request):
    catalog = _get_catalog(request)
    gen = catalog.get_generation(generation_id)
    if not gen:
        raise HTTPException(status_code=404, detail="Generation not found")

    lineage = catalog.get_lineage(generation_id)
    return {"generation": gen, "lineage": lineage}


@router.patch("/media/generations/{generation_id}")
async def update_generation(generation_id: str, body: GenerationUpdate, request: Request):
    catalog = _get_catalog(request)
    updates = {k: v for k, v in body.dict().items() if v is not None}
    if not updates:
        return {"success": True, "message": "No changes"}
    ok = catalog.update_generation(generation_id, **updates)
    if not ok:
        raise HTTPException(status_code=404, detail="Generation not found")
    return {"success": True}


@router.delete("/media/generations/{generation_id}")
async def delete_generation(generation_id: str, request: Request):
    catalog = _get_catalog(request)
    ok = catalog.delete_generation(generation_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Generation not found")
    return {"success": True}


# ================================================================
# Statistics
# ================================================================

@router.get("/media/stats")
async def get_stats(request: Request):
    catalog = _get_catalog(request)
    return catalog.get_stats()


# ================================================================
# Retroactive scan
# ================================================================

@router.post("/media/scan")
async def scan_output(request: Request):
    catalog = _get_catalog(request)
    output_dir = _get_output_dir(request)
    return catalog.scan_output_directory(output_dir)


@router.get("/media/orphans")
async def list_orphans(request: Request, limit: int = 100, offset: int = 0):
    catalog = _get_catalog(request)
    files = catalog.get_orphan_files(limit=limit, offset=offset)
    return {"files": files, "total": len(files)}


# ================================================================
# Input pipeline
# ================================================================

@router.post("/media/input/from-generation")
async def input_from_generation(body: InputFromGeneration, request: Request):
    catalog = _get_catalog(request)
    input_dir = _get_input_dir(request)
    filename = catalog.copy_to_input(body.generation_id, body.file_index, input_dir)
    if not filename:
        raise HTTPException(status_code=404, detail="Generation file not found or missing on disk")
    return {"success": True, "filename": filename, "input_dir": input_dir}


@router.post("/media/input/from-url")
async def input_from_url(body: InputFromUrl, request: Request):
    catalog = _get_catalog(request)
    input_dir = _get_input_dir(request)
    filename = catalog.download_to_input(body.url, input_dir)
    if not filename:
        raise HTTPException(status_code=500, detail="Failed to download image")
    return {"success": True, "filename": filename}


@router.post("/media/input/upload")
async def upload_input_image(request: Request, file: UploadFile = File(...)):
    input_dir = _get_input_dir(request)
    os.makedirs(input_dir, exist_ok=True)

    # Security: sanitize filename
    safe_name = os.path.basename(file.filename or "upload.png")
    if ".." in safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename")

    dest = os.path.join(input_dir, safe_name)
    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)

    return {"success": True, "filename": safe_name, "size": len(content)}


# ================================================================
# Batch operations
# ================================================================

@router.post("/media/generations/batch-delete")
async def batch_delete(body: BatchIds, request: Request):
    catalog = _get_catalog(request)
    count = 0
    for gen_id in body.ids:
        if catalog.delete_generation(gen_id):
            count += 1
    return {"success": True, "deleted": count}


@router.post("/media/generations/batch-tag")
async def batch_tag(body: BatchTag, request: Request):
    catalog = _get_catalog(request)
    count = 0
    for gen_id in body.ids:
        gen = catalog.get_generation(gen_id)
        if gen:
            existing = gen.get("tags", "")
            merged = set(t.strip() for t in existing.split(",") if t.strip())
            merged.update(t.strip() for t in body.tags.split(",") if t.strip())
            catalog.update_generation(gen_id, tags=",".join(sorted(merged)))
            count += 1
    return {"success": True, "tagged": count}
