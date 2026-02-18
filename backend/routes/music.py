"""
Music Module Routes

Provides API endpoints for Music server lifecycle management.
Routes either handle operations directly (download, bootstrap, API server)
or proxy to the Music API gateway on port 9150.
"""

import json
import logging
from pathlib import Path
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import FileResponse, Response

logger = logging.getLogger("Music.Routes")

router = APIRouter()


def _get_manager(request: Request):
    manager = getattr(request.app.state, 'music_manager', None)
    if not manager:
        raise HTTPException(status_code=500, detail="Music manager not initialized")
    return manager


async def _ensure_api(manager):
    if not await manager.is_api_running():
        raise HTTPException(status_code=503, detail="Music API server not running. Start it first.")


# ======================== Status ========================

@router.get("/status")
async def get_status(request: Request):
    manager = _get_manager(request)
    return await manager.get_status()


# ======================== Module Lifecycle ========================

@router.post("/module/download")
async def download_module(request: Request):
    manager = _get_manager(request)
    return await manager.download_module()


@router.post("/module/bootstrap")
async def bootstrap_module(request: Request):
    manager = _get_manager(request)
    return await manager.bootstrap()


@router.post("/module/update")
async def update_module(request: Request):
    manager = _get_manager(request)
    return await manager.update_module()


# ======================== API Server ========================

@router.post("/server/start")
async def start_api_server(request: Request):
    manager = _get_manager(request)
    return await manager.start_api_server()


@router.post("/server/stop")
async def stop_api_server(request: Request):
    manager = _get_manager(request)
    return await manager.stop_api_server()


# ======================== Models (proxy) ========================

@router.get("/models")
async def list_models(request: Request):
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("GET", "/api/models")


@router.get("/models/status")
async def models_status(request: Request):
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("GET", "/api/models/status")


@router.get("/models/{model}/params")
async def model_params(model: str, request: Request):
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("GET", f"/api/models/{model}/params")


@router.get("/models/{model}/presets")
async def model_presets(model: str, request: Request):
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("GET", f"/api/models/{model}/presets")


@router.get("/models/{model}/display")
async def model_display(model: str, request: Request):
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("GET", f"/api/models/{model}/display")


@router.post("/models/{model}/load")
async def load_model(model: str, request: Request):
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("POST", f"/api/models/{model}/load")


@router.post("/models/{model}/unload")
async def unload_model(model: str, request: Request):
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("POST", f"/api/models/{model}/unload")


@router.post("/models/{model}/scale")
async def scale_model(model: str, request: Request):
    manager = _get_manager(request)
    await _ensure_api(manager)
    body = await request.json()
    return await manager.proxy("POST", f"/api/models/{model}/scale", json=body)


# ======================== Devices (proxy) ========================

@router.get("/devices")
async def list_devices(request: Request):
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("GET", "/api/devices")


# ======================== Workers (proxy) ========================

@router.get("/workers")
async def list_workers(request: Request):
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("GET", "/api/workers")


@router.post("/workers/spawn")
async def spawn_worker(request: Request):
    manager = _get_manager(request)
    await _ensure_api(manager)
    body = await request.json()
    return await manager.proxy("POST", "/api/workers/spawn", json=body)


@router.delete("/workers/{worker_id}")
async def kill_worker(worker_id: str, request: Request):
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("DELETE", f"/api/workers/{worker_id}")


@router.post("/workers/kill-all")
async def kill_all_workers(request: Request):
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("POST", "/api/workers/kill-all")


@router.get("/workers/{worker_id}/logs")
async def worker_logs(worker_id: str, request: Request):
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("GET", f"/api/workers/{worker_id}/logs")


# ======================== Music Generation (proxy) ========================

@router.post("/generate/{model}")
async def generate_music(model: str, request: Request):
    manager = _get_manager(request)
    await _ensure_api(manager)
    body = await request.json()
    return await manager.proxy("POST", f"/api/music/{model}", json=body)


# ======================== Install Management (proxy) ========================

@router.get("/install/status")
async def install_status(request: Request):
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("GET", "/api/install/status")


@router.get("/install/status/{model_id}")
async def install_model_status(model_id: str, request: Request):
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("GET", f"/api/install/status/{model_id}")


@router.post("/install/{model_id}")
async def install_model(model_id: str, request: Request):
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("POST", f"/api/install/{model_id}")


@router.post("/install/{model_id}/download")
async def download_weights(model_id: str, request: Request):
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("POST", f"/api/install/{model_id}/download")


@router.delete("/install/{model_id}")
async def uninstall_model(model_id: str, request: Request):
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("DELETE", f"/api/install/{model_id}")


@router.post("/install/{model_id}/cancel")
async def cancel_install(model_id: str, request: Request):
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("POST", f"/api/install/{model_id}/cancel")


@router.get("/install/jobs")
async def install_jobs(request: Request):
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("GET", "/api/install/jobs")


@router.get("/install/jobs/{job_id}")
async def install_job_status(job_id: str, request: Request):
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("GET", f"/api/install/jobs/{job_id}")


@router.get("/install/jobs/{job_id}/logs")
async def install_job_logs(job_id: str, request: Request):
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("GET", f"/api/install/jobs/{job_id}/logs")


# ======================== CLAP Scorer (proxy) ========================

@router.get("/clap/status")
async def clap_status(request: Request):
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("GET", "/api/clap/status")


@router.post("/clap/start")
async def clap_start(request: Request):
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("POST", "/api/clap/start")


@router.post("/clap/stop")
async def clap_stop(request: Request):
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("POST", "/api/clap/stop")


@router.post("/clap/score")
async def clap_score(request: Request):
    manager = _get_manager(request)
    await _ensure_api(manager)
    body = await request.json()
    return await manager.proxy("POST", "/api/clap/score", json=body)


# ======================== Output Library (proxy) ========================

@router.get("/outputs")
async def list_outputs(request: Request):
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("GET", "/api/outputs")


@router.get("/outputs/{entry_id}")
async def get_output(entry_id: str, request: Request):
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("GET", f"/api/outputs/{entry_id}")


@router.get("/outputs/{entry_id}/audio")
async def get_output_audio(entry_id: str, request: Request):
    """Serve audio file from the music server's output directory."""
    manager = _get_manager(request)
    gen_dir = manager.module_dir / "output" / "generations"

    if not gen_dir.is_dir():
        raise HTTPException(status_code=404, detail="Generations directory not found")

    # Find audio file matching the entry_id
    for f in gen_dir.iterdir():
        if f.stem.endswith(entry_id) and f.suffix in ('.wav', '.mp3', '.ogg', '.flac', '.m4a'):
            media_type = {
                ".wav": "audio/wav", ".mp3": "audio/mpeg",
                ".ogg": "audio/ogg", ".flac": "audio/flac", ".m4a": "audio/mp4",
            }.get(f.suffix.lower(), "audio/wav")
            return FileResponse(str(f), media_type=media_type)

    # Fallback: proxy to the music server
    await _ensure_api(manager)
    return await manager.proxy("GET", f"/api/outputs/{entry_id}/audio")


@router.delete("/outputs/{entry_id}")
async def delete_output(entry_id: str, request: Request):
    manager = _get_manager(request)
    await _ensure_api(manager)
    return await manager.proxy("DELETE", f"/api/outputs/{entry_id}")


@router.delete("/outputs/batch")
async def delete_outputs_batch(request: Request):
    manager = _get_manager(request)
    await _ensure_api(manager)
    body = await request.json()
    return await manager.proxy("DELETE", "/api/outputs/batch", json=body)
