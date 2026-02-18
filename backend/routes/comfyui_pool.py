"""
ComfyUI Pool REST Endpoints

Provides pool-level generation dispatch across multiple ComfyUI instances.
"""

from typing import Optional, List

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

router = APIRouter()


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #

class PoolSubmitBody(BaseModel):
    checkpoint: str
    prompt: str
    negative_prompt: str = "blurry, low quality, distorted"
    width: Optional[int] = None
    height: Optional[int] = None
    steps: Optional[int] = None
    cfg: Optional[float] = None
    seed: int = -1
    sampler_name: Optional[str] = None
    scheduler: Optional[str] = None
    instance_id: Optional[str] = None


class BatchJobSpec(BaseModel):
    prompt: str
    negative_prompt: str = "blurry, low quality, distorted"
    seed: int = -1
    width: Optional[int] = None
    height: Optional[int] = None


class PoolBatchBody(BaseModel):
    checkpoint: str
    jobs: List[BatchJobSpec]
    steps: Optional[int] = None
    cfg: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    sampler_name: Optional[str] = None
    scheduler: Optional[str] = None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _get_pool(request: Request):
    pool = getattr(request.app.state, "comfyui_pool", None)
    if not pool:
        raise HTTPException(503, "ComfyUI pool not initialized")
    return pool


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@router.post("/submit")
async def pool_submit(body: PoolSubmitBody, request: Request):
    """Submit a single generation job to the pool (auto-selects best instance)."""
    pool = _get_pool(request)
    return await pool.submit_job(
        checkpoint=body.checkpoint,
        prompt=body.prompt,
        negative_prompt=body.negative_prompt,
        width=body.width,
        height=body.height,
        steps=body.steps,
        cfg=body.cfg,
        seed=body.seed,
        sampler_name=body.sampler_name,
        scheduler=body.scheduler,
        instance_id=body.instance_id,
    )


@router.post("/batch")
async def pool_batch(body: PoolBatchBody, request: Request):
    """Submit N jobs distributed across pool instances."""
    pool = _get_pool(request)
    jobs = [j.model_dump() for j in body.jobs]
    return await pool.submit_batch(
        checkpoint=body.checkpoint,
        jobs=jobs,
        steps=body.steps,
        cfg=body.cfg,
        width=body.width,
        height=body.height,
        sampler_name=body.sampler_name,
        scheduler=body.scheduler,
    )


@router.get("/status")
async def pool_status(request: Request):
    """Pool overview: per-instance metrics, loaded models, active jobs."""
    pool = _get_pool(request)
    return await pool.get_pool_status()


@router.get("/job/{job_id}")
async def pool_job_status(job_id: str, request: Request):
    """Track a specific pool job."""
    pool = _get_pool(request)
    result = await pool.get_job_status(job_id)
    if result is None:
        raise HTTPException(404, f"Pool job '{job_id}' not found")
    return result


@router.get("/batch/{batch_id}")
async def pool_batch_status(batch_id: str, request: Request):
    """Track a batch of pool jobs (aggregated results)."""
    pool = _get_pool(request)
    result = await pool.get_batch_status(batch_id)
    if result is None:
        raise HTTPException(404, f"Batch '{batch_id}' not found")
    return result
