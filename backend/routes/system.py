"""
System Routes - GPU monitoring, system stats, shutdown.
"""

import subprocess
import sys
import os
import signal
import asyncio
import time
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Request
import logging

router = APIRouter()
logger = logging.getLogger("routes.system")

# Flag to track shutdown state
_shutdown_in_progress = False

# GPU info cache to reduce nvidia-smi calls
_gpu_cache = {
    "data": None,
    "timestamp": 0,
    "driver_version": None,
    "cuda_version": None,
}
_GPU_CACHE_TTL = 1.0  # 1 second cache


def _safe_int(value: str, default: int = 0) -> int:
    """Safely parse int from nvidia-smi output."""
    if not value or value in ['[N/A]', 'N/A', '', '[Not Supported]']:
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


def _safe_float(value: str, default: Optional[float] = None) -> Optional[float]:
    """Safely parse float from nvidia-smi output."""
    if not value or value in ['[N/A]', 'N/A', '', '[Not Supported]']:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _get_driver_info() -> tuple:
    """Get NVIDIA driver and CUDA version (cached)."""
    global _gpu_cache

    if _gpu_cache["driver_version"] is not None:
        return _gpu_cache["driver_version"], _gpu_cache["cuda_version"]

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        if result.returncode == 0 and result.stdout.strip():
            _gpu_cache["driver_version"] = result.stdout.strip().split('\n')[0]

        # Get CUDA version from nvidia-smi header
        result2 = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        if result2.returncode == 0:
            for line in result2.stdout.split('\n'):
                if 'CUDA Version' in line:
                    parts = line.split('CUDA Version:')
                    if len(parts) > 1:
                        _gpu_cache["cuda_version"] = parts[1].strip().split()[0]
                    break
    except Exception as e:
        logger.warning(f"Failed to get driver info: {e}")

    return _gpu_cache["driver_version"], _gpu_cache["cuda_version"]


@router.get("/gpu")
async def get_gpu_stats(request: Request):
    """Get detailed GPU statistics for dashboard."""
    global _gpu_cache

    orchestrator = request.app.state.orchestrator

    try:
        # Check cache first
        now = time.time()
        if _gpu_cache["data"] and (now - _gpu_cache["timestamp"]) < _GPU_CACHE_TTL:
            # Update models_loaded from orchestrator (real-time) but use cached GPU stats
            cached = _gpu_cache["data"].copy()
            cached["gpus"] = _update_models_on_gpus(cached["gpus"], orchestrator, request.app.state)
            return cached

        # Get nvidia-smi info with more fields
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,power.limit,fan.speed,pstate",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )

        gpus = []
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 6:
                        gpu_idx = _safe_int(parts[0])
                        mem_total = _safe_int(parts[2])
                        mem_used = _safe_int(parts[3])
                        mem_percent = (mem_used / mem_total * 100) if mem_total > 0 else 0

                        gpus.append({
                            "index": gpu_idx,
                            "name": parts[1] if len(parts) > 1 else "Unknown",
                            "memory_total_mb": mem_total,
                            "memory_used_mb": mem_used,
                            "memory_free_mb": _safe_int(parts[4]) if len(parts) > 4 else 0,
                            "memory_percent": round(mem_percent, 1),
                            "utilization_percent": _safe_int(parts[5]) if len(parts) > 5 else 0,
                            "temperature_c": _safe_int(parts[6]) if len(parts) > 6 else None,
                            "power_draw_w": _safe_float(parts[7]) if len(parts) > 7 else None,
                            "power_limit_w": _safe_float(parts[8]) if len(parts) > 8 else None,
                            "fan_speed_percent": _safe_int(parts[9]) if len(parts) > 9 else None,
                            "pstate": parts[10] if len(parts) > 10 and parts[10] not in ['[N/A]', 'N/A', ''] else None,
                            "models_loaded": []
                        })

        # Update with loaded models
        gpus = _update_models_on_gpus(gpus, orchestrator, request.app.state)

        # Get driver info
        driver_version, cuda_version = _get_driver_info()

        response = {
            "success": True,
            "gpu_count": len(gpus),
            "gpus": gpus,
            "driver_version": driver_version,
            "cuda_version": cuda_version,
        }

        # Cache the response
        _gpu_cache["data"] = response
        _gpu_cache["timestamp"] = now

        return response

    except FileNotFoundError:
        return {
            "success": True,
            "gpu_count": 0,
            "gpus": [],
            "note": "nvidia-smi not found - no NVIDIA GPUs or drivers not installed"
        }
    except Exception as e:
        logger.error(f"get_gpu_stats error: {e}")
        return {"success": False, "error": str(e)}


def _update_models_on_gpus(gpus: List[Dict], orchestrator, app_state=None) -> List[Dict]:
    """Update GPU list with currently loaded models."""
    from providers.base import ProviderType

    # Create a copy to avoid modifying cached data
    gpus = [dict(gpu) for gpu in gpus]

    for gpu in gpus:
        gpu_idx = gpu["index"]
        models_on_gpu = []

        # Check llama.cpp workers
        if hasattr(orchestrator, 'providers') and ProviderType.LLAMA_CPP in orchestrator.providers:
            provider = orchestrator.providers[ProviderType.LLAMA_CPP]
            for inst_id, worker in provider._workers.items():
                if worker.gpu_index == gpu_idx:
                    inst = provider.instances.get(inst_id)
                    if inst:
                        models_on_gpu.append({
                            "instance_id": inst_id[:8],
                            "model": inst.display_name,
                            "provider": "llama_cpp",
                            "busy": worker.is_busy
                        })

        # Check LM Studio instances
        # LM Studio doesn't always report gpu_index — default to GPU 0 when null
        if hasattr(orchestrator, 'providers') and ProviderType.LM_STUDIO in orchestrator.providers:
            lm_provider = orchestrator.providers[ProviderType.LM_STUDIO]
            for inst_id, inst in lm_provider.instances.items():
                effective_gpu = inst.gpu_index if inst.gpu_index is not None else 0
                if effective_gpu == gpu_idx:
                    models_on_gpu.append({
                        "instance_id": inst_id[:8],
                        "model": inst.display_name,
                        "provider": "lm_studio",
                        "busy": inst.status.value == "busy"
                    })

        # Check vLLM instances
        if hasattr(orchestrator, 'providers') and ProviderType.VLLM in orchestrator.providers:
            vllm_provider = orchestrator.providers[ProviderType.VLLM]
            for inst_id, inst in vllm_provider.instances.items():
                if getattr(inst, 'gpu_index', None) == gpu_idx:
                    models_on_gpu.append({
                        "instance_id": inst_id[:8],
                        "model": getattr(inst, 'display_name', str(inst.model_id)[:40]),
                        "provider": "vllm",
                        "busy": getattr(inst, 'status', None) and inst.status.value == "busy"
                    })

        gpu["models_loaded"] = models_on_gpu

    # Add ComfyUI loaded checkpoints from pool metrics
    try:
        comfyui_pool = getattr(app_state, 'comfyui_pool', None) if app_state else None
        if comfyui_pool and hasattr(comfyui_pool, '_instance_metrics'):
            for iid, metrics in comfyui_pool._instance_metrics.items():
                loaded_ckpt = getattr(metrics, 'loaded_checkpoint', None)
                if loaded_ckpt:
                    comfy_gpu = getattr(metrics, 'gpu_index', 0) or 0
                    for gpu in gpus:
                        if gpu["index"] == comfy_gpu:
                            name = loaded_ckpt if len(loaded_ckpt) <= 40 else loaded_ckpt[:37] + "..."
                            gpu["models_loaded"].append({
                                "instance_id": f"comfy-{iid}",
                                "model": name,
                                "provider": "comfyui",
                                "busy": getattr(metrics, 'queue_size', 0) > 0
                            })
    except Exception:
        pass  # ComfyUI pool not available

    return gpus


@router.get("/gpu/history")
async def get_gpu_history(request: Request, samples: int = 60):
    """Get GPU history from in-memory cache (for charts)."""
    # Return cached history if available
    history = getattr(request.app.state, 'gpu_history', None)
    if history:
        return {"success": True, "history": history[-samples:]}
    return {"success": True, "history": []}


@router.get("/queue")
async def get_queue_stats(request: Request):
    """Get request queue statistics."""
    orchestrator = request.app.state.orchestrator

    return {
        "success": True,
        "pending": orchestrator.request_queue.get_queue_length(),
        "processing": orchestrator.request_queue.get_processing_count()
    }


@router.get("/models/summary")
async def get_models_summary(request: Request):
    """Get summary of loaded models across all providers."""
    orchestrator = request.app.state.orchestrator

    models = []
    for provider in orchestrator.providers.values():
        for inst_id, inst in provider.instances.items():
            models.append({
                "id": inst_id[:8],
                "name": inst.display_name,
                "provider": inst.provider_type.value,
                "gpu_index": inst.gpu_index,
                "status": inst.status.value,
                "requests": inst.request_count
            })

    return {
        "success": True,
        "count": len(models),
        "models": models
    }


@router.post("/shutdown")
async def shutdown_system(request: Request):
    """
    Aggressively shut down the entire AgentNate system.

    This will:
    1. Unload all loaded models (kill worker processes)
    2. Stop all n8n instances
    3. Kill any orphaned processes
    4. Trigger server shutdown
    """
    global _shutdown_in_progress

    if _shutdown_in_progress:
        return {"success": False, "error": "Shutdown already in progress"}

    _shutdown_in_progress = True
    logger.info("=== AGGRESSIVE SHUTDOWN INITIATED ===")

    orchestrator = request.app.state.orchestrator
    n8n_manager = request.app.state.n8n_manager
    n8n_queue_manager = request.app.state.n8n_queue_manager
    registry = getattr(request.app.state, 'process_registry', None)

    results = {
        "models_unloaded": 0,
        "n8n_stopped": 0,
        "queue_workers_stopped": 0,
        "processes_killed": 0,
        "errors": []
    }

    # 1. Cancel all pending model loads
    try:
        if hasattr(orchestrator, '_load_tasks'):
            for task_id, task in list(orchestrator._load_tasks.items()):
                logger.info(f"Cancelling pending load: {task_id}")
                task.cancel()
            orchestrator._load_tasks.clear()
        if hasattr(orchestrator, '_pending_loads'):
            orchestrator._pending_loads.clear()
    except Exception as e:
        results["errors"].append(f"Cancel pending loads: {e}")
        logger.error(f"Error cancelling pending loads: {e}")

    # 2. Unload all models (this should kill worker processes)
    try:
        instance_ids = list(orchestrator.instances.keys())
        logger.info(f"Unloading {len(instance_ids)} models...")
        for instance_id in instance_ids:
            try:
                await orchestrator.unload_model(instance_id)
                results["models_unloaded"] += 1
                logger.info(f"Unloaded model: {instance_id}")
            except Exception as e:
                results["errors"].append(f"Unload {instance_id}: {e}")
                logger.error(f"Error unloading {instance_id}: {e}")
    except Exception as e:
        results["errors"].append(f"Model unload loop: {e}")
        logger.error(f"Error in model unload loop: {e}")

    # 3. Force-stop any llama.cpp workers that might be lingering
    try:
        from providers.base import ProviderType
        if ProviderType.LLAMA_CPP in orchestrator.providers:
            provider = orchestrator.providers[ProviderType.LLAMA_CPP]

            # Kill any workers in _workers dict
            if hasattr(provider, '_workers'):
                for worker_id, worker in list(provider._workers.items()):
                    try:
                        logger.info(f"Force stopping worker: {worker_id}")
                        await worker.stop()
                        results["processes_killed"] += 1
                    except Exception as e:
                        logger.error(f"Error stopping worker {worker_id}: {e}")
                provider._workers.clear()

            # Kill any workers in _loading_workers dict
            if hasattr(provider, '_loading_workers'):
                for worker_id, worker in list(provider._loading_workers.items()):
                    try:
                        logger.info(f"Force stopping loading worker: {worker_id}")
                        await worker.stop()
                        results["processes_killed"] += 1
                    except Exception as e:
                        logger.error(f"Error stopping loading worker {worker_id}: {e}")
                provider._loading_workers.clear()
    except Exception as e:
        results["errors"].append(f"Force stop workers: {e}")
        logger.error(f"Error force stopping workers: {e}")

    # 4. Stop all n8n instances
    try:
        ports = list(n8n_manager.instances.keys())
        logger.info(f"Stopping {len(ports)} n8n instances...")
        for port in ports:
            try:
                await n8n_manager.stop(port)
                results["n8n_stopped"] += 1
                logger.info(f"Stopped n8n on port {port}")
            except Exception as e:
                results["errors"].append(f"Stop n8n {port}: {e}")
                logger.error(f"Error stopping n8n {port}: {e}")
    except Exception as e:
        results["errors"].append(f"n8n stop loop: {e}")
        logger.error(f"Error in n8n stop loop: {e}")

    # 5. Stop queue manager (workers + main)
    try:
        if n8n_queue_manager:
            worker_count = len(n8n_queue_manager.workers)
            logger.info(f"Stopping queue manager ({worker_count} workers + main)...")
            await n8n_queue_manager.shutdown()
            results["queue_workers_stopped"] = worker_count
            logger.info("Queue manager shutdown complete")
    except Exception as e:
        results["errors"].append(f"Queue manager shutdown: {e}")
        logger.error(f"Error shutting down queue manager: {e}")

    # 6. Stop ComfyUI instances and API server
    try:
        comfyui_manager = getattr(request.app.state, 'comfyui_manager', None)
        if comfyui_manager:
            await comfyui_manager.shutdown()
            results["comfyui_stopped"] = True
            logger.info("ComfyUI shutdown complete")
    except Exception as e:
        results["errors"].append(f"ComfyUI shutdown: {e}")
        logger.error(f"Error shutting down ComfyUI: {e}")

    # 7. Safety-net: force-unload remaining LM Studio models via SDK
    try:
        from providers.base import ProviderType
        if ProviderType.LM_STUDIO in orchestrator.providers:
            lm_provider = orchestrator.providers[ProviderType.LM_STUDIO]
            sdk_unloaded = await _force_unload_lm_studio(lm_provider)
            results["lm_studio_sdk_unloaded"] = sdk_unloaded
            if sdk_unloaded > 0:
                logger.info(f"LM Studio SDK safety net unloaded {sdk_unloaded} model(s)")
    except Exception as e:
        results["errors"].append(f"LM Studio SDK safety net: {e}")
        logger.error(f"Error in LM Studio SDK safety net: {e}")

    # 8. Kill any orphaned processes (llama-server, zombie servers, ComfyUI)
    orphans_killed = await _kill_orphaned_processes()
    results["processes_killed"] += orphans_killed

    # 9. Registry safety net: kill anything still registered
    if registry:
        try:
            reg_killed = registry.kill_all_registered()
            results["processes_killed"] += reg_killed
            if reg_killed > 0:
                logger.info(f"Registry safety net killed {reg_killed} remaining process(es)")
        except Exception as e:
            results["errors"].append(f"Registry cleanup: {e}")
            logger.error(f"Error in registry cleanup: {e}")

    logger.info(f"=== SHUTDOWN COMPLETE: {results} ===")

    # 10. Schedule server exit after response is sent
    asyncio.create_task(_delayed_exit())

    return {
        "success": True,
        "message": "Shutdown initiated - server will exit shortly",
        "results": results
    }


async def _force_unload_lm_studio(lm_provider) -> int:
    """Force-unload ALL agentnate-* models from LM Studio via SDK.

    This is a safety net — even if the per-instance unload failed or timed out,
    this will catch any remaining models loaded by AgentNate.
    """
    unloaded = 0
    try:
        client = await lm_provider._get_sdk_client_async()
        if not client:
            logger.info("LM Studio SDK not available, skipping force-unload")
            return 0

        loop = asyncio.get_event_loop()
        loaded = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: list(client.llm.list_loaded())),
            timeout=10.0,
        )

        for m in loaded:
            ident = getattr(m, "identifier", "") or ""
            if ident.startswith("agentnate-"):
                try:
                    await asyncio.wait_for(
                        loop.run_in_executor(None, m.unload),
                        timeout=15.0,
                    )
                    unloaded += 1
                    logger.info(f"Force-unloaded LM Studio model: {ident}")
                except asyncio.TimeoutError:
                    logger.warning(f"Force-unload timed out for {ident}")
                except Exception as e:
                    logger.warning(f"Force-unload failed for {ident}: {e}")
    except asyncio.TimeoutError:
        logger.warning("LM Studio SDK list_loaded timed out")
    except Exception as e:
        logger.warning(f"LM Studio force-unload error: {e}")

    return unloaded


async def _kill_orphaned_processes():
    """Kill orphaned processes: llama-server, zombie AgentNate servers, ComfyUI API."""
    killed = 0
    my_pid = os.getpid()

    if sys.platform == "win32":
        # Kill llama-server.exe processes
        killed += _win_kill_by_image("llama-server.exe")

        # Kill zombie run.py server processes (not us)
        killed += _win_kill_zombie_servers(my_pid)

        # Kill orphaned ComfyUI API server processes
        killed += _win_kill_by_cmdline("installer_app.py", exclude_pid=my_pid)
    else:
        # Unix: use pkill
        for pattern in ["llama-server", "installer_app.py"]:
            try:
                subprocess.run(["pkill", "-f", pattern], capture_output=True)
                killed += 1
            except Exception:
                pass

    return killed


def _win_kill_by_image(image_name: str) -> int:
    """Kill all processes matching an image name on Windows."""
    killed = 0
    try:
        result = subprocess.run(
            ["taskkill", "/F", "/IM", image_name],
            capture_output=True, text=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if "SUCCESS" in result.stdout:
            killed = result.stdout.count("SUCCESS")
            logger.info(f"Killed {killed} orphaned {image_name} process(es)")
    except Exception as e:
        logger.warning(f"Error killing {image_name}: {e}")
    return killed


def _win_kill_zombie_servers(my_pid: int) -> int:
    """Kill zombie run.py --mode server processes (not our own PID)."""
    killed = 0
    try:
        result = subprocess.run(
            ["wmic", "process", "where",
             "name='python.exe' and commandline like '%run.py%--mode server%'",
             "get", "ProcessId", "/format:csv"],
            capture_output=True, text=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("Node"):
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                try:
                    pid = int(parts[-1].strip())
                    if pid != my_pid:
                        logger.info(f"Killing zombie server PID {pid}")
                        subprocess.run(
                            ["wmic", "process", "where", f"ProcessId={pid}",
                             "call", "terminate"],
                            capture_output=True, timeout=5,
                            creationflags=subprocess.CREATE_NO_WINDOW,
                        )
                        killed += 1
                except (ValueError, subprocess.TimeoutExpired):
                    pass
    except Exception as e:
        logger.warning(f"Error killing zombie servers: {e}")
    return killed


def _win_kill_by_cmdline(pattern: str, exclude_pid: int = 0) -> int:
    """Kill processes whose command line matches a pattern on Windows."""
    killed = 0
    try:
        result = subprocess.run(
            ["wmic", "process", "where",
             f"commandline like '%{pattern}%'",
             "get", "ProcessId", "/format:csv"],
            capture_output=True, text=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("Node"):
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                try:
                    pid = int(parts[-1].strip())
                    if pid != exclude_pid:
                        logger.info(f"Killing orphaned process PID {pid} ({pattern})")
                        subprocess.run(
                            ["wmic", "process", "where", f"ProcessId={pid}",
                             "call", "terminate"],
                            capture_output=True, timeout=5,
                            creationflags=subprocess.CREATE_NO_WINDOW,
                        )
                        killed += 1
                except (ValueError, subprocess.TimeoutExpired):
                    pass
    except Exception as e:
        logger.warning(f"Error killing processes matching '{pattern}': {e}")
    return killed


async def _delayed_exit():
    """Exit the server after a short delay to allow response to be sent."""
    await asyncio.sleep(3)
    # Clear registry before hard exit (os._exit skips atexit handlers)
    try:
        from backend.server import process_registry
        if process_registry:
            process_registry.clear()
    except Exception:
        pass
    logger.info("Server exiting...")
    os._exit(0)


@router.get("/shutdown/status")
async def shutdown_status():
    """Check if shutdown is in progress."""
    return {"shutdown_in_progress": _shutdown_in_progress}
