"""
Music Tools - Music generation module tools.

Provides tools for the Meta Agent to control the Music server:
status, start/stop, model management, music generation, output library.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger("tools.music")


TOOL_DEFINITIONS = [
    {
        "name": "music_status",
        "description": "Get Music module status: installed, bootstrapped, API server running. Call this first to understand what's available.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "music_start_server",
        "description": "Start the Music API gateway server (required before loading models or generating music). Takes ~30-60s on first start.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "music_stop_server",
        "description": "Stop the Music API gateway server and all running workers",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "music_list_models",
        "description": "List available music generation models with status (loaded/unloaded, worker count). Requires API server running. Models: ace_step (ACE-Step v1.5, lyrics+tags), ace_step_v1 (ACE-Step v1), heartmula (HeartMuLa 3B, prompt→song), diffrythm (DiffRhythm, lyrics+melody), yue (YuE, lyrics→song), musicgen (MusicGen, text→music), riffusion (Riffusion, text→music), stable_audio (Stable Audio Open, text→music)",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "music_list_workers",
        "description": "List running Music workers (each worker serves one model). Shows worker ID, model, device, status.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "music_load_model",
        "description": "Load a music model by spawning a worker for it. The model must be installed first (use music_install_status to check). VRAM varies: musicgen ~4GB, riffusion ~4GB, stable_audio ~4GB, ace_step ~8GB, heartmula ~12GB.",
        "parameters": {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "description": "Model ID: ace_step, ace_step_v1, heartmula, diffrythm, yue, musicgen, riffusion, stable_audio"
                },
                "device": {
                    "type": "string",
                    "description": "Device to load on (e.g. 'cuda:0', 'cuda:1', 'cpu'). Default: auto-select."
                }
            },
            "required": ["model"]
        }
    },
    {
        "name": "music_unload_model",
        "description": "Unload a music model (kills its workers, frees VRAM)",
        "parameters": {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "description": "Model ID to unload"
                }
            },
            "required": ["model"]
        }
    },
    {
        "name": "music_generate",
        "description": "Generate music from a text prompt. Model must be loaded first. Returns generation info. For ACE-Step, use lyrics with [verse]/[chorus] tags. For MusicGen/Riffusion/Stable Audio, use descriptive text prompts.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Text prompt or lyrics for music generation. ACE-Step supports structured lyrics with tags. Others take descriptive prompts like 'upbeat electronic dance music with synths'."
                },
                "model": {
                    "type": "string",
                    "description": "Model ID: ace_step, ace_step_v1, heartmula, diffrythm, yue, musicgen, riffusion, stable_audio"
                },
                "duration": {
                    "type": "number",
                    "description": "Duration in seconds (model-specific limits apply). Default varies by model."
                },
                "seed": {
                    "type": "integer",
                    "description": "Random seed for reproducibility. Default: random."
                }
            },
            "required": ["prompt", "model"]
        }
    },
    {
        "name": "music_get_presets",
        "description": "Get parameter presets for a specific music model. Shows recommended settings, available parameters, and their ranges.",
        "parameters": {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "description": "Model ID to get presets for"
                }
            },
            "required": ["model"]
        }
    },
    {
        "name": "music_install_model",
        "description": "Install a music model (venv + weights). Triggers the music server's install pipeline. Takes 5-30 minutes depending on model size and download speed.",
        "parameters": {
            "type": "object",
            "properties": {
                "model_id": {
                    "type": "string",
                    "description": "Model ID to install: ace_step, ace_step_v1, heartmula, diffrythm, yue, musicgen, riffusion, stable_audio"
                }
            },
            "required": ["model_id"]
        }
    },
    {
        "name": "music_install_status",
        "description": "Get installation status for all music models: which are installed, downloading, or need setup.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "music_list_outputs",
        "description": "List generated music in the output library. Shows entry ID, model, prompt, duration, format.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
]


class MusicTools:
    """Handler class for Music tools."""

    def __init__(self, music_manager):
        self.manager = music_manager

    async def _check_manager(self) -> Dict[str, Any]:
        """Return error dict if manager is unavailable."""
        if not self.manager:
            return {"success": False, "error": "Music module not initialized. Is the Music manager configured?"}
        return None

    async def _check_api(self) -> Dict[str, Any]:
        """Return error dict if API server is not running."""
        err = await self._check_manager()
        if err:
            return err
        if not await self.manager.is_api_running():
            return {"success": False, "error": "Music API server not running. Use music_start_server first."}
        return None

    # ---- Status & Lifecycle ----

    async def music_status(self, **kwargs) -> Dict[str, Any]:
        """Get Music module status."""
        err = await self._check_manager()
        if err:
            return err
        try:
            status = await self.manager.get_status()
            return {"success": True, **status}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def music_start_server(self, **kwargs) -> Dict[str, Any]:
        """Start the Music API gateway server."""
        err = await self._check_manager()
        if err:
            return err
        try:
            result = await self.manager.start_api_server()
            return {"success": True, **result, "hint": "Music API server started. Use music_list_models to see available models, or music_load_model to load one."}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def music_stop_server(self, **kwargs) -> Dict[str, Any]:
        """Stop the Music API gateway server."""
        err = await self._check_manager()
        if err:
            return err
        try:
            result = await self.manager.stop_api_server()
            return {"success": True, **result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ---- Models & Workers ----

    async def music_list_models(self, **kwargs) -> Dict[str, Any]:
        """List available music models with status."""
        err = await self._check_api()
        if err:
            return err
        try:
            result = await self.manager.proxy("GET", "/api/models")
            return {"success": True, **result} if isinstance(result, dict) else {"success": True, "models": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def music_list_workers(self, **kwargs) -> Dict[str, Any]:
        """List running Music workers."""
        err = await self._check_api()
        if err:
            return err
        try:
            result = await self.manager.proxy("GET", "/api/workers")
            return {"success": True, **result} if isinstance(result, dict) else {"success": True, "workers": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def music_load_model(self, **kwargs) -> Dict[str, Any]:
        """Load a music model (spawn worker)."""
        err = await self._check_api()
        if err:
            return err
        model = kwargs.get("model")
        if not model:
            return {"success": False, "error": "Missing required parameter: model"}
        try:
            result = await self.manager.proxy("POST", f"/api/models/{model}/load")
            return {"success": True, **result, "hint": f"Model '{model}' loaded. Use music_generate with model='{model}' to generate music."} if isinstance(result, dict) else {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def music_unload_model(self, **kwargs) -> Dict[str, Any]:
        """Unload a music model (kill workers)."""
        err = await self._check_api()
        if err:
            return err
        model = kwargs.get("model")
        if not model:
            return {"success": False, "error": "Missing required parameter: model"}
        try:
            result = await self.manager.proxy("POST", f"/api/models/{model}/unload")
            return {"success": True, **result} if isinstance(result, dict) else {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ---- Generation ----

    async def music_generate(self, **kwargs) -> Dict[str, Any]:
        """Generate music from a text prompt."""
        err = await self._check_api()
        if err:
            return err
        prompt = kwargs.get("prompt")
        model = kwargs.get("model")
        if not prompt or not model:
            return {"success": False, "error": "Missing required parameters: prompt, model"}

        body = {"prompt": prompt}
        if kwargs.get("duration"):
            body["duration"] = kwargs["duration"]
        if kwargs.get("seed") is not None:
            body["seed"] = kwargs["seed"]

        try:
            result = await self.manager.proxy("POST", f"/api/music/{model}", json=body)
            return {"success": True, **result} if isinstance(result, dict) else {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def music_get_presets(self, **kwargs) -> Dict[str, Any]:
        """Get parameter presets for a model."""
        err = await self._check_api()
        if err:
            return err
        model = kwargs.get("model")
        if not model:
            return {"success": False, "error": "Missing required parameter: model"}
        try:
            result = await self.manager.proxy("GET", f"/api/models/{model}/presets")
            return {"success": True, **result} if isinstance(result, dict) else {"success": True, "presets": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ---- Install Management ----

    async def music_install_model(self, **kwargs) -> Dict[str, Any]:
        """Install a music model (venv + weights)."""
        err = await self._check_api()
        if err:
            return err
        model_id = kwargs.get("model_id")
        if not model_id:
            return {"success": False, "error": "Missing required parameter: model_id"}
        try:
            result = await self.manager.proxy("POST", f"/api/install/{model_id}")
            return {"success": True, **result} if isinstance(result, dict) else {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def music_install_status(self, **kwargs) -> Dict[str, Any]:
        """Get installation status for all models."""
        err = await self._check_api()
        if err:
            return err
        try:
            result = await self.manager.proxy("GET", "/api/install/status")
            return {"success": True, **result} if isinstance(result, dict) else {"success": True, "status": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def music_list_outputs(self, **kwargs) -> Dict[str, Any]:
        """List generated music in the output library."""
        err = await self._check_api()
        if err:
            return err
        try:
            result = await self.manager.proxy("GET", "/api/outputs")
            return {"success": True, **result} if isinstance(result, dict) else {"success": True, "outputs": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
