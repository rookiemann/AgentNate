"""
TTS Module Routes

Provides API endpoints for TTS server lifecycle management.
Routes either handle operations directly (download, bootstrap, API server)
or proxy to the TTS API gateway on port 8100.
"""

import json
import logging
from pathlib import Path
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import FileResponse

logger = logging.getLogger("TTS.Routes")

router = APIRouter()


def _get_manager(request: Request):
    """Get TTS manager from app state."""
    manager = getattr(request.app.state, 'tts_manager', None)
    if not manager:
        raise HTTPException(status_code=500, detail="TTS manager not initialized")
    return manager


async def _ensure_api(manager):
    """Ensure the TTS API is running, raise if not."""
    if not await manager.is_api_running():
        raise HTTPException(status_code=503, detail="TTS API server not running. Start it first.")


# ======================== Status ========================

@router.get("/status")
async def get_status(request: Request):
    """Get combined TTS module status."""
    manager = _get_manager(request)
    return await manager.get_status()


# ======================== Module Lifecycle ========================

@router.post("/module/download")
async def download_module(request: Request):
    """Clone the portable TTS server from GitHub."""
    manager = _get_manager(request)
    return await manager.download_module()


@router.post("/module/bootstrap")
async def bootstrap_module(request: Request):
    """Run bootstrap: download Python, Git, FFmpeg, create venvs."""
    manager = _get_manager(request)
    return await manager.bootstrap()


@router.post("/module/update")
async def update_module(request: Request):
    """Update the TTS server via git pull."""
    manager = _get_manager(request)
    return await manager.update_module()


# ======================== API Server ========================

@router.post("/server/start")
async def start_api_server(request: Request):
    """Start the TTS API gateway server."""
    manager = _get_manager(request)
    return await manager.start_api_server()


@router.post("/server/stop")
async def stop_api_server(request: Request):
    """Stop the TTS API gateway server."""
    manager = _get_manager(request)
    return await manager.stop_api_server()


# ======================== Model Management (local) ========================

@router.get("/model-info")
async def get_model_info(request: Request):
    """Get info about all TTS models: env installed, weights downloaded."""
    manager = _get_manager(request)
    return {"models": manager.get_model_info()}


@router.post("/environments/{env_name}/install")
async def install_env(env_name: str, request: Request):
    """Install a model's virtual environment (venv + pip packages)."""
    manager = _get_manager(request)
    return await manager.install_model_env(env_name)


@router.post("/model-weights/{model_id}/download")
async def download_weights(model_id: str, request: Request):
    """Download model weights from HuggingFace."""
    manager = _get_manager(request)
    return await manager.download_model_weights(model_id)


# ======================== Models (proxy) ========================

@router.get("/models")
async def list_models(request: Request):
    """List available TTS models."""
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("GET", "/api/models")


@router.get("/models/status")
async def models_status(request: Request):
    """Get per-model worker status."""
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("GET", "/api/models/status")


@router.post("/models/{model}/load")
async def load_model(model: str, request: Request):
    """Load a TTS model (auto-spawn worker)."""
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("POST", f"/api/models/{model}/load")


@router.post("/models/{model}/unload")
async def unload_model(model: str, request: Request):
    """Unload a TTS model (kill workers)."""
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("POST", f"/api/models/{model}/unload")


@router.post("/models/{model}/scale")
async def scale_model(model: str, request: Request):
    """Scale model workers up or down."""
    manager = _get_manager(request)
    await _ensure_api(manager)
    body = await request.json()
    return await manager.proxy("POST", f"/api/models/{model}/scale", json=body)


# ======================== Devices (proxy) ========================

@router.get("/devices")
async def list_devices(request: Request):
    """List available GPUs/devices."""
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("GET", "/api/devices")


# ======================== Workers (proxy) ========================

@router.get("/workers")
async def list_workers(request: Request):
    """List all TTS workers."""
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("GET", "/api/workers")


@router.post("/workers/spawn")
async def spawn_worker(request: Request):
    """Spawn a new TTS worker for a model."""
    manager = _get_manager(request)
    await _ensure_api(manager)
    body = await request.json()
    return await manager.proxy("POST", "/api/workers/spawn", json=body)


@router.delete("/workers/{worker_id}")
async def kill_worker(worker_id: str, request: Request):
    """Kill a specific TTS worker."""
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("DELETE", f"/api/workers/{worker_id}")


# ======================== TTS Inference (proxy) ========================

@router.post("/generate/{model}")
async def generate_tts(model: str, request: Request):
    """Generate speech from text using a TTS model."""
    manager = _get_manager(request)
    await _ensure_api(manager)
    body = await request.json()
    return await manager.proxy("POST", f"/api/tts/{model}", json=body)


@router.get("/voices/{model}")
async def get_voices(model: str, request: Request):
    """Get available voices for a TTS model."""
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("GET", f"/api/tts/{model}/voices")


@router.post("/generate/{model}/cancel")
async def cancel_generation(model: str, request: Request):
    """Cancel running TTS generation for a model."""
    manager = _get_manager(request)
    await _ensure_api(manager)
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    return await manager.proxy("POST", f"/api/tts/{model}/cancel", json=body)


# ======================== Jobs (proxy) ========================

@router.get("/jobs")
async def list_jobs(request: Request):
    """List all TTS jobs."""
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("GET", "/api/jobs")


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, request: Request):
    """Get details of a specific TTS job."""
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("GET", f"/api/jobs/{job_id}")


@router.post("/jobs/{job_id}/recover")
async def recover_job(job_id: str, request: Request):
    """Recover/resume an interrupted TTS job."""
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("POST", f"/api/jobs/{job_id}/recover")


# ======================== Whisper (proxy) ========================

@router.get("/whisper")
async def whisper_status(request: Request):
    """Get Whisper verification status."""
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("GET", "/api/whisper")


@router.post("/whisper/{size}/load")
async def load_whisper(size: str, request: Request):
    """Load a Whisper model for verification."""
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("POST", f"/api/whisper/{size}/load")


@router.post("/whisper/{size}/unload")
async def unload_whisper(size: str, request: Request):
    """Unload a Whisper model."""
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("POST", f"/api/whisper/{size}/unload")


# ======================== Library (local) ========================

@router.get("/library")
async def list_library(request: Request):
    """List completed TTS jobs with audio files for the library view."""
    manager = _get_manager(request)
    jobs_dir = manager.module_dir / "output" / "jobs"

    if not jobs_dir.is_dir():
        return {"items": []}

    items = []
    for job_dir in sorted(jobs_dir.iterdir(), reverse=True):
        if not job_dir.is_dir():
            continue
        job_json = job_dir / "job.json"
        if not job_json.is_file():
            continue

        try:
            with open(job_json, "r", encoding="utf-8") as f:
                job = json.load(f)
            if job.get("status") != "completed":
                continue

            # Find the final audio file
            final_file = job.get("final_file")
            audio_path = None
            if final_file:
                candidate = job_dir / final_file
                if candidate.is_file():
                    audio_path = str(candidate)

            if not audio_path:
                # Search for any final audio file
                for ext in ("wav", "mp3", "ogg", "flac", "m4a"):
                    for f in job_dir.glob(f"*_final.{ext}"):
                        audio_path = str(f)
                        break
                    if audio_path:
                        break

            if not audio_path:
                continue

            items.append({
                "job_id": job.get("job_id", job_dir.name),
                "model": job.get("model", "unknown"),
                "timestamp": job.get("timestamp", ""),
                "text": (job.get("input_text", "") or "")[:200],
                "duration_sec": job.get("total_duration_sec"),
                "sample_rate": job.get("sample_rate"),
                "format": job.get("output_format", "wav"),
                "chunks": job.get("total_chunks", 0),
                "voice": job.get("parameters", {}).get("voice", ""),
                "has_audio": True,
            })
        except Exception:
            continue

    return {"items": items}


@router.get("/library/{job_id}/audio")
async def get_library_audio(job_id: str, request: Request):
    """Serve an audio file from a completed TTS job."""
    manager = _get_manager(request)
    job_dir = manager.module_dir / "output" / "jobs" / job_id

    if not job_dir.is_dir():
        raise HTTPException(status_code=404, detail="Job not found")

    # Find the final audio file
    job_json = job_dir / "job.json"
    if job_json.is_file():
        try:
            with open(job_json, "r", encoding="utf-8") as f:
                job = json.load(f)
            final_file = job.get("final_file")
            if final_file:
                candidate = job_dir / final_file
                if candidate.is_file():
                    media_type = {
                        ".wav": "audio/wav",
                        ".mp3": "audio/mpeg",
                        ".ogg": "audio/ogg",
                        ".flac": "audio/flac",
                        ".m4a": "audio/mp4",
                    }.get(candidate.suffix.lower(), "audio/wav")
                    return FileResponse(str(candidate), media_type=media_type)
        except Exception:
            pass

    # Fallback: search for any final audio
    for ext in ("wav", "mp3", "ogg", "flac", "m4a"):
        for f in job_dir.glob(f"*_final.{ext}"):
            media_type = {
                "wav": "audio/wav", "mp3": "audio/mpeg",
                "ogg": "audio/ogg", "flac": "audio/flac", "m4a": "audio/mp4",
            }.get(ext, "audio/wav")
            return FileResponse(str(f), media_type=media_type)

    raise HTTPException(status_code=404, detail="Audio file not found")


@router.delete("/library/{job_id}")
async def delete_library_item(job_id: str, request: Request):
    """Delete a TTS job and its files."""
    manager = _get_manager(request)
    job_dir = manager.module_dir / "output" / "jobs" / job_id

    if not job_dir.is_dir():
        raise HTTPException(status_code=404, detail="Job not found")

    import shutil
    try:
        shutil.rmtree(str(job_dir))
        return {"success": True, "message": f"Job {job_id} deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete job: {e}")
