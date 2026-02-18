"""
TTS Tools - Text-to-Speech generation module tools.

Provides tools for the Meta Agent to control the TTS server:
status, start/stop, model management, voice listing, speech generation.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger("tools.tts")


TOOL_DEFINITIONS = [
    {
        "name": "tts_status",
        "description": "Get TTS module status: installed, bootstrapped, API server running, available models. Call this first to understand what's available.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "tts_start_server",
        "description": "Start the TTS API gateway server (required before loading models or generating speech). Takes ~30-60s on first start.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "tts_stop_server",
        "description": "Stop the TTS API gateway server and all running workers",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "tts_list_models",
        "description": "List available TTS models with their status (loaded/unloaded, worker count). Requires API server running. Models: kokoro (82M, fast, 54 voices), xtts (500M, multilingual voice cloning), dia (1.6B, dialogue with [S1]/[S2] tags), bark (1B, expressive - laughter/music), fish (500M, fast voice cloning), chatterbox (500M, emotion control), f5 (300M, diffusion cloning), qwen (7B, multimodal), vibevoice (1.5B, speaker-conditioned), higgs (3B, CPU supported)",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "tts_list_workers",
        "description": "List running TTS workers (each worker serves one model). Shows worker ID, model, device, status.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "tts_load_model",
        "description": "Load a TTS model by spawning a worker for it. The model's venv must be installed first (use tts_get_model_info to check). VRAM usage: kokoro ~1GB, fish ~1GB, xtts ~3GB, f5 ~2GB, chatterbox ~2GB, dia ~6GB, bark ~5GB.",
        "parameters": {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "description": "Model ID: kokoro, xtts, dia, bark, fish, chatterbox, f5, qwen, vibevoice, higgs"
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
        "name": "tts_unload_model",
        "description": "Unload a TTS model (kills its workers, frees VRAM)",
        "parameters": {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "description": "Model ID to unload: kokoro, xtts, dia, bark, fish, chatterbox, f5, qwen, vibevoice, higgs"
                }
            },
            "required": ["model"]
        }
    },
    {
        "name": "tts_generate",
        "description": "Generate speech from text. Model must be loaded first (use tts_load_model). Returns job info with audio details. For dialogue with Dia, use [S1] and [S2] tags. For Bark, use special tokens like [laughter], [music].",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to convert to speech. For Dia dialogue: '[S1] Hello! [S2] Hi there!'. Max ~500 chars for best quality."
                },
                "model": {
                    "type": "string",
                    "description": "Model ID: kokoro, xtts, dia, bark, fish, chatterbox, f5, qwen, vibevoice, higgs"
                },
                "voice": {
                    "type": "string",
                    "description": "Voice name (model-specific). Use tts_list_voices to see options. Default: model's default voice."
                },
                "output_format": {
                    "type": "string",
                    "description": "Output format: wav, mp3, ogg, flac (default: wav)"
                }
            },
            "required": ["text", "model"]
        }
    },
    {
        "name": "tts_list_voices",
        "description": "List available voices for a specific TTS model. Kokoro has 54 built-in voices, XTTS has 58.",
        "parameters": {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "description": "Model ID: kokoro, xtts, dia, bark, fish, chatterbox, f5, qwen, vibevoice, higgs"
                }
            },
            "required": ["model"]
        }
    },
    {
        "name": "tts_get_model_info",
        "description": "Get local install status for all TTS models: which venvs are installed, which weights are downloaded. Use this to know what needs to be set up before loading a model.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "tts_install_env",
        "description": "Install a model's virtual environment (downloads Python packages). Some models share envs: unified_env (kokoro+fish+dia), coqui_env (xtts+bark). Takes 2-10 minutes.",
        "parameters": {
            "type": "object",
            "properties": {
                "env_name": {
                    "type": "string",
                    "description": "Environment name: unified_env, coqui_env, chatterbox_env, f5tts_env, qwen3_env, vibevoice_env, higgs_env"
                }
            },
            "required": ["env_name"]
        }
    },
    {
        "name": "tts_download_weights",
        "description": "Download model weights from HuggingFace. Required for models with explicit weight repos (kokoro, xtts, dia, bark, fish, chatterbox, f5). Takes 1-15 minutes depending on size.",
        "parameters": {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "description": "Model ID whose weights to download: kokoro, xtts, dia, bark, fish, chatterbox, f5"
                }
            },
            "required": ["model"]
        }
    },
]


class TTSTools:
    """Handler class for TTS tools."""

    def __init__(self, tts_manager):
        self.manager = tts_manager

    async def _check_manager(self) -> Dict[str, Any]:
        """Return error dict if manager is unavailable."""
        if not self.manager:
            return {"success": False, "error": "TTS module not initialized. Is the TTS manager configured?"}
        return None

    async def _check_api(self) -> Dict[str, Any]:
        """Return error dict if API server is not running."""
        err = await self._check_manager()
        if err:
            return err
        if not await self.manager.is_api_running():
            return {"success": False, "error": "TTS API server not running. Use tts_start_server first."}
        return None

    # ---- Status & Lifecycle ----

    async def tts_status(self, **kwargs) -> Dict[str, Any]:
        """Get TTS module status."""
        err = await self._check_manager()
        if err:
            return err
        try:
            status = await self.manager.get_status()
            return {"success": True, **status}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def tts_start_server(self, **kwargs) -> Dict[str, Any]:
        """Start the TTS API gateway server."""
        err = await self._check_manager()
        if err:
            return err
        try:
            result = await self.manager.start_api_server()
            return {"success": True, **result, "hint": "TTS API server started. Use tts_list_models to see available models, or tts_load_model to load one."}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def tts_stop_server(self, **kwargs) -> Dict[str, Any]:
        """Stop the TTS API gateway server."""
        err = await self._check_manager()
        if err:
            return err
        try:
            result = await self.manager.stop_api_server()
            return {"success": True, **result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ---- Models & Workers ----

    async def tts_list_models(self, **kwargs) -> Dict[str, Any]:
        """List available TTS models with status."""
        err = await self._check_api()
        if err:
            return err
        try:
            result = await self.manager.proxy("GET", "/api/models")
            # Enrich with local install status
            local_info = self.manager.get_model_info()
            local_map = {m["id"]: m for m in local_info}
            if isinstance(result, dict) and "models" in result:
                for model in result["models"]:
                    mid = model.get("id", model.get("name", ""))
                    if mid in local_map:
                        model["env_installed"] = local_map[mid]["env_installed"]
                        model["weights_downloaded"] = local_map[mid]["weights_downloaded"]
            return {"success": True, **result} if isinstance(result, dict) else {"success": True, "models": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def tts_list_workers(self, **kwargs) -> Dict[str, Any]:
        """List running TTS workers."""
        err = await self._check_api()
        if err:
            return err
        try:
            result = await self.manager.proxy("GET", "/api/workers")
            return {"success": True, **result} if isinstance(result, dict) else {"success": True, "workers": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def tts_load_model(self, **kwargs) -> Dict[str, Any]:
        """Load a TTS model (spawn worker)."""
        err = await self._check_api()
        if err:
            return err
        model = kwargs.get("model")
        if not model:
            return {"success": False, "error": "Missing required parameter: model"}
        try:
            result = await self.manager.proxy("POST", f"/api/models/{model}/load")
            return {"success": True, **result, "hint": f"Model '{model}' loaded. Use tts_generate with model='{model}' to generate speech."} if isinstance(result, dict) else {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def tts_unload_model(self, **kwargs) -> Dict[str, Any]:
        """Unload a TTS model (kill workers)."""
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

    async def tts_generate(self, **kwargs) -> Dict[str, Any]:
        """Generate speech from text."""
        err = await self._check_api()
        if err:
            return err
        text = kwargs.get("text")
        model = kwargs.get("model")
        if not text or not model:
            return {"success": False, "error": "Missing required parameters: text, model"}

        body = {"text": text}
        if kwargs.get("voice"):
            body["voice"] = kwargs["voice"]
        if kwargs.get("output_format"):
            body["output_format"] = kwargs["output_format"]

        try:
            result = await self.manager.proxy("POST", f"/api/tts/{model}", json=body)
            return {"success": True, **result} if isinstance(result, dict) else {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def tts_list_voices(self, **kwargs) -> Dict[str, Any]:
        """List available voices for a model."""
        err = await self._check_api()
        if err:
            return err
        model = kwargs.get("model")
        if not model:
            return {"success": False, "error": "Missing required parameter: model"}
        try:
            result = await self.manager.proxy("GET", f"/api/tts/{model}/voices")
            return {"success": True, **result} if isinstance(result, dict) else {"success": True, "voices": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ---- Install Management ----

    async def tts_get_model_info(self, **kwargs) -> Dict[str, Any]:
        """Get local install status for all models."""
        err = await self._check_manager()
        if err:
            return err
        try:
            models = self.manager.get_model_info()
            return {"success": True, "models": models}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def tts_install_env(self, **kwargs) -> Dict[str, Any]:
        """Install a model's virtual environment."""
        err = await self._check_manager()
        if err:
            return err
        env_name = kwargs.get("env_name")
        if not env_name:
            return {"success": False, "error": "Missing required parameter: env_name"}
        try:
            result = await self.manager.install_model_env(env_name)
            return {"success": True, **result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def tts_download_weights(self, **kwargs) -> Dict[str, Any]:
        """Download model weights from HuggingFace."""
        err = await self._check_manager()
        if err:
            return err
        model = kwargs.get("model")
        if not model:
            return {"success": False, "error": "Missing required parameter: model"}
        try:
            result = await self.manager.download_model_weights(model)
            return {"success": True, **result}
        except Exception as e:
            return {"success": False, "error": str(e)}
