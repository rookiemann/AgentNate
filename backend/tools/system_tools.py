"""
System Tools - GPU status, health checks, system info.
"""

from typing import Dict, Any, Optional
import logging
import os
import subprocess
import sys

from providers.base import ProviderType
from .suggestions import SuggestionEngine

logger = logging.getLogger("tools.system")
AGENTNATE_BASE = os.getenv("AGENTNATE_BASE_URL", "http://127.0.0.1:8000")

# Map string names to ProviderType enum
PROVIDER_TYPE_MAP = {
    "llama_cpp": ProviderType.LLAMA_CPP,
    "lm_studio": ProviderType.LM_STUDIO,
    "ollama": ProviderType.OLLAMA,
    "openrouter": ProviderType.OPENROUTER,
}


TOOL_DEFINITIONS = [
    {
        "name": "get_gpu_status",
        "description": "Get current GPU status including memory usage and loaded models",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_system_health",
        "description": "Get overall system health including all providers",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_provider_status",
        "description": "Get status of a specific provider",
        "parameters": {
            "type": "object",
            "properties": {
                "provider": {
                    "type": "string",
                    "description": "Provider name: llama_cpp, lm_studio, ollama, openrouter",
                    "enum": ["llama_cpp", "lm_studio", "ollama", "openrouter"]
                }
            },
            "required": ["provider"]
        }
    },
    {
        "name": "get_full_status",
        "description": "Get complete system overview in one call: loaded models, GPU status, n8n instances, and queue status",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "quick_setup",
        "description": "Quick setup: load a model and optionally spawn n8n in one command",
        "parameters": {
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "description": "Model name to load (e.g., 'phi-4', 'llama-3')"
                },
                "gpu_index": {
                    "type": "integer",
                    "description": "GPU to load on (default: auto-select)"
                },
                "spawn_n8n": {
                    "type": "boolean",
                    "description": "Whether to also spawn an n8n instance (default: true)"
                }
            },
            "required": ["model_name"]
        }
    },
    {
        "name": "suggest_actions",
        "description": "Get contextual suggestions for what to do next based on current system state",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]


class SystemTools:
    """Tools for system monitoring and health."""

    def __init__(self, orchestrator, settings, n8n_manager=None, model_tools=None):
        self.orchestrator = orchestrator
        self.settings = settings
        self.n8n_manager = n8n_manager
        self.model_tools = model_tools
        self.suggestion_engine = SuggestionEngine()

    async def get_gpu_status(self) -> Dict[str, Any]:
        """Get GPU status with memory info."""
        try:
            loaded_instances = self.orchestrator.get_loaded_instances() or []
            # llama.cpp exposes worker-level busy state; use it when available.
            llama_worker_busy = {}
            if ProviderType.LLAMA_CPP in self.orchestrator.providers:
                provider = self.orchestrator.providers[ProviderType.LLAMA_CPP]
                workers = getattr(provider, "_workers", {}) or {}
                for inst_id, worker in workers.items():
                    llama_worker_busy[inst_id] = bool(getattr(worker, "is_busy", False))

            # Get nvidia-smi info
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
                 "--format=csv,noheader,nounits"],
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
                            gpu_idx = int(parts[0])

                            # Find models on this GPU (all providers, not just llama.cpp)
                            models_on_gpu = []
                            for inst in loaded_instances:
                                if getattr(inst, "gpu_index", None) != gpu_idx:
                                    continue
                                status = getattr(getattr(inst, "status", None), "value", "").lower()
                                models_on_gpu.append({
                                    "instance_id": inst.id,
                                    "model": inst.display_name or inst.model_identifier,
                                    "provider": inst.provider_type.value,
                                    "busy": llama_worker_busy.get(inst.id, status == "busy"),
                                })

                            gpus.append({
                                "index": gpu_idx,
                                "name": parts[1],
                                "memory_total_mb": int(parts[2]),
                                "memory_used_mb": int(parts[3]),
                                "memory_free_mb": int(parts[4]),
                                "utilization_percent": int(parts[5]),
                                "models_loaded": models_on_gpu
                            })

            return {
                "success": True,
                "gpu_count": len(gpus),
                "gpus": gpus
            }

        except FileNotFoundError:
            return {
                "success": True,
                "gpu_count": 0,
                "gpus": [],
                "note": "nvidia-smi not found - no NVIDIA GPUs or drivers not installed"
            }
        except Exception as e:
            logger.error(f"get_gpu_status error: {e}")
            return {"success": False, "error": str(e)}

    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health."""
        try:
            health = await self.orchestrator.check_all_health()

            # Count loaded models
            total_loaded = 0
            for provider in self.orchestrator.providers.values():
                total_loaded += len(provider.instances)

            return {
                "success": True,
                "status": "healthy",
                "providers": health.get("providers", {}),
                "total_loaded_models": total_loaded,
                "queue_status": {
                    "pending": self.orchestrator.request_queue.get_queue_length(),
                    "processing": self.orchestrator.request_queue.get_processing_count()
                }
            }

        except Exception as e:
            logger.error(f"get_system_health error: {e}")
            return {"success": False, "error": str(e)}

    async def get_provider_status(self, provider: str) -> Dict[str, Any]:
        """Get status of a specific provider."""
        try:
            provider_type = PROVIDER_TYPE_MAP.get(provider)
            if not provider_type or provider_type not in self.orchestrator.providers:
                return {
                    "success": False,
                    "error": f"Provider '{provider}' not found or not enabled"
                }

            prov = self.orchestrator.providers[provider_type]
            health = await prov.health_check()

            return {
                "success": True,
                "provider": provider,
                "enabled": prov.enabled,
                "health": health
            }

        except Exception as e:
            logger.error(f"get_provider_status error: {e}")
            return {"success": False, "error": str(e)}

    async def get_system_snapshot(self) -> Dict[str, Any]:
        """
        Get a complete snapshot of system state for dynamic prompts.

        This aggregates all state into a single dict that can be used
        for building dynamic system prompts and generating suggestions.
        """
        snapshot = {
            "models": [],
            "gpus": [],
            "n8n_instances": [],
            "queue": {"pending": 0, "processing": 0}
        }

        try:
            # Get loaded models
            instances = self.orchestrator.get_loaded_instances()
            for inst in instances:
                snapshot["models"].append({
                    "instance_id": inst.id,
                    "model": inst.display_name or inst.model_identifier,
                    "provider": inst.provider_type.value,
                    "gpu": inst.gpu_index,
                    "status": inst.status.value,
                    "context_length": inst.context_length
                })

            # Get GPU status
            gpu_result = await self.get_gpu_status()
            if gpu_result.get("success"):
                snapshot["gpus"] = gpu_result.get("gpus", [])

            # Get n8n instances (handle both N8nQueueManager and legacy N8nManager)
            if self.n8n_manager:
                if hasattr(self.n8n_manager, 'main') and hasattr(self.n8n_manager, 'workers'):
                    # N8nQueueManager
                    if self.n8n_manager.main and self.n8n_manager.main.is_running:
                        snapshot["n8n_instances"].append({
                            "port": self.n8n_manager.main.port,
                            "running": True,
                            "url": f"http://localhost:{self.n8n_manager.main.port}"
                        })
                    for port, worker in self.n8n_manager.workers.items():
                        snapshot["n8n_instances"].append({
                            "port": port,
                            "running": worker.is_running,
                            "url": f"http://localhost:{port}",
                            "workflow": worker.workflow_name if hasattr(worker, 'workflow_name') else None
                        })
                elif hasattr(self.n8n_manager, 'instances'):
                    # Legacy N8nManager
                    for port, inst in self.n8n_manager.instances.items():
                        is_running = inst.is_running if inst else False
                        snapshot["n8n_instances"].append({
                            "port": port,
                            "running": is_running,
                            "url": f"http://localhost:{port}"
                        })

            # Get queue status
            snapshot["queue"] = {
                "pending": self.orchestrator.request_queue.get_queue_length(),
                "processing": self.orchestrator.request_queue.get_processing_count()
            }

        except Exception as e:
            logger.error(f"get_system_snapshot error: {e}")

        return snapshot

    async def get_full_status(self) -> Dict[str, Any]:
        """Get complete system overview in one call."""
        try:
            snapshot = await self.get_system_snapshot()

            return {
                "success": True,
                "models": {
                    "count": len(snapshot["models"]),
                    "instances": snapshot["models"]
                },
                "gpus": {
                    "count": len(snapshot["gpus"]),
                    "devices": snapshot["gpus"]
                },
                "n8n": {
                    "count": len([i for i in snapshot["n8n_instances"] if i.get("running")]),
                    "instances": snapshot["n8n_instances"]
                },
                "queue": snapshot["queue"]
            }

        except Exception as e:
            logger.error(f"get_full_status error: {e}")
            return {"success": False, "error": str(e)}

    async def quick_setup(
        self,
        model_name: str,
        gpu_index: Optional[int] = None,
        spawn_n8n: bool = True
    ) -> Dict[str, Any]:
        """Quick setup: load a model and optionally spawn n8n."""
        results = {
            "success": True,
            "model": None,
            "n8n": None,
            "messages": []
        }

        try:
            # Load the model
            if self.model_tools:
                model_result = await self.model_tools.load_model(
                    model_name=model_name,
                    gpu_index=gpu_index
                )
                results["model"] = model_result

                if model_result.get("success"):
                    results["messages"].append(
                        f"Model '{model_result.get('model')}' loaded on GPU {model_result.get('gpu')}"
                    )
                else:
                    results["success"] = False
                    results["messages"].append(f"Failed to load model: {model_result.get('error')}")
            else:
                results["success"] = False
                results["messages"].append("Model tools not available")

            # Spawn n8n if requested and model succeeded
            if spawn_n8n and self.n8n_manager and results["success"]:
                try:
                    instance = await self.n8n_manager.spawn()
                    results["n8n"] = {
                        "success": True,
                        "port": instance.port,
                        "url": f"{AGENTNATE_BASE}/api/n8n/{instance.port}/proxy/"
                    }
                    results["messages"].append(f"n8n started on port {instance.port}")
                except Exception as e:
                    results["n8n"] = {"success": False, "error": str(e)}
                    results["messages"].append(f"Failed to spawn n8n: {e}")

            return results

        except Exception as e:
            logger.error(f"quick_setup error: {e}")
            return {"success": False, "error": str(e)}

    async def suggest_actions(self) -> Dict[str, Any]:
        """Get contextual suggestions for what to do next."""
        try:
            snapshot = await self.get_system_snapshot()
            suggestions = self.suggestion_engine.generate_suggestions(snapshot)

            return {
                "success": True,
                "count": len(suggestions),
                "suggestions": suggestions
            }

        except Exception as e:
            logger.error(f"suggest_actions error: {e}")
            return {"success": False, "error": str(e)}
