"""
Model Tools - Load, unload, and manage LLM models.
"""

from typing import Dict, Any, List, Optional
import logging

from providers.base import ProviderType

logger = logging.getLogger("tools.model")


TOOL_DEFINITIONS = [
    {
        "name": "list_available_models",
        "description": "List all available models that can be loaded (from all providers)",
        "parameters": {
            "type": "object",
            "properties": {
                "provider": {
                    "type": "string",
                    "description": "Filter by provider: llama_cpp, lm_studio, ollama, openrouter. Leave empty for all.",
                    "enum": ["llama_cpp", "lm_studio", "ollama", "openrouter", ""]
                }
            },
            "required": []
        }
    },
    {
        "name": "list_loaded_models",
        "description": "List currently loaded model instances",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "load_model",
        "description": "Load a model onto a GPU. Returns the instance ID for chat.",
        "parameters": {
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "description": "Model name or partial name to match (e.g., 'phi-4', 'llama-3')"
                },
                "gpu_index": {
                    "type": "integer",
                    "description": "GPU to load on: 0, 1, etc. Use -1 for CPU. Default is auto-select."
                },
                "provider": {
                    "type": "string",
                    "description": "Provider to use: llama_cpp, lm_studio. Default is llama_cpp.",
                    "enum": ["llama_cpp", "lm_studio"]
                },
                "context_length": {
                    "type": "integer",
                    "description": "Context length (default 4096)"
                }
            },
            "required": ["model_name"]
        }
    },
    {
        "name": "unload_model",
        "description": "Unload a model instance to free GPU memory",
        "parameters": {
            "type": "object",
            "properties": {
                "instance_id": {
                    "type": "string",
                    "description": "The instance ID to unload (from list_loaded_models)"
                }
            },
            "required": ["instance_id"]
        }
    },
    {
        "name": "get_model_status",
        "description": "Get detailed status of a loaded model",
        "parameters": {
            "type": "object",
            "properties": {
                "instance_id": {
                    "type": "string",
                    "description": "The instance ID to check"
                }
            },
            "required": ["instance_id"]
        }
    },
    {
        "name": "list_model_presets",
        "description": "List saved model load presets (pre-configured model+provider+GPU combinations). Use load_from_preset to load a model from a preset.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "load_from_preset",
        "description": "Load a model using a saved preset configuration (provider, model path, context length, GPU index are all pre-configured). Returns instance_id.",
        "parameters": {
            "type": "object",
            "properties": {
                "preset_name": {
                    "type": "string",
                    "description": "Name of the model load preset (from list_model_presets)"
                }
            },
            "required": ["preset_name"]
        }
    },
    {
        "name": "save_model_preset",
        "description": "Save a model loading preset for later use by users or agents. Useful for repeatable model+GPU assignments.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Preset name (e.g., 'LMStudio Coder GPU1')"
                },
                "provider": {
                    "type": "string",
                    "description": "Provider for this preset",
                    "enum": ["llama_cpp", "lm_studio", "ollama", "vllm", "openrouter"]
                },
                "model_id": {
                    "type": "string",
                    "description": "Exact model identifier/path to load"
                },
                "context_length": {
                    "type": "integer",
                    "description": "Context length for loading (default: 4096)"
                },
                "gpu_index": {
                    "type": "integer",
                    "description": "GPU index for single-GPU loading (0, 1, ...)."
                },
                "gpu_layers": {
                    "type": "integer",
                    "description": "llama.cpp GPU layers setting (default: -1)"
                },
            },
            "required": ["name", "provider", "model_id"]
        }
    },
]


class ModelTools:
    """Tools for model management."""

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

    async def list_available_models(self, provider: str = "") -> Dict[str, Any]:
        """List available models."""
        try:
            models_by_provider = await self.orchestrator.list_all_models()

            # Flatten the dict into a list
            all_models = []
            for prov, model_list in models_by_provider.items():
                if provider and prov != provider:
                    continue
                all_models.extend(model_list)

            # Simplify output
            models = []
            for m in all_models[:50]:  # Limit to 50
                models.append({
                    "name": m.get("name", m.get("id")),
                    "provider": m.get("provider"),
                    "size_gb": round(m.get("size_bytes", 0) / (1024**3), 2) if m.get("size_bytes") else None
                })

            return {
                "success": True,
                "count": len(models),
                "models": models
            }
        except Exception as e:
            logger.error(f"list_available_models error: {e}")
            return {"success": False, "error": str(e)}

    async def list_loaded_models(self) -> Dict[str, Any]:
        """List loaded model instances."""
        try:
            instances = self.orchestrator.get_loaded_instances()

            loaded = []
            for inst in instances:
                prov = inst.provider_type.value
                loaded.append({
                    "instance_id": inst.id,
                    "model": inst.display_name or inst.model_identifier,
                    "provider": prov,
                    "gpu": inst.gpu_index,
                    "status": inst.status.value,
                    "context_length": inst.context_length,
                    "locality": "local" if prov in ("lm_studio", "llama_cpp", "vllm", "ollama") else "cloud",
                    "cost": "free" if prov != "openrouter" else "paid",
                })

            return {
                "success": True,
                "count": len(loaded),
                "instances": loaded
            }
        except Exception as e:
            logger.error(f"list_loaded_models error: {e}")
            return {"success": False, "error": str(e)}

    async def load_model(
        self,
        model_name: str,
        gpu_index: Optional[int] = None,
        provider: str = "llama_cpp",
        context_length: int = 4096
    ) -> Dict[str, Any]:
        """Load a model."""
        try:
            # Find matching model
            models_by_provider = await self.orchestrator.list_all_models()

            # Get models for requested provider
            provider_models = models_by_provider.get(provider, [])

            # Find match
            match = None
            model_name_lower = model_name.lower()
            for m in provider_models:
                name = (m.get("name") or m.get("id") or "").lower()
                if model_name_lower in name:
                    match = m
                    break

            if not match:
                return {
                    "success": False,
                    "error": f"No model matching '{model_name}' found in {provider}"
                }

            # Load the model
            model_path = match.get("path") or match.get("id")

            # Map string to ProviderType
            provider_type_map = {
                "llama_cpp": ProviderType.LLAMA_CPP,
                "lm_studio": ProviderType.LM_STUDIO,
                "ollama": ProviderType.OLLAMA,
            }

            ptype = provider_type_map.get(provider)
            if not ptype or ptype not in self.orchestrator.providers:
                return {"success": False, "error": f"Unknown or unavailable provider: {provider}"}

            # Use orchestrator.load_model to ensure instance is registered
            if provider == "llama_cpp":
                instance = await self.orchestrator.load_model(
                    provider_type=ptype,
                    model_identifier=model_path,
                    n_ctx=context_length,
                    gpu_index=gpu_index
                )
            elif provider == "lm_studio":
                instance = await self.orchestrator.load_model(
                    provider_type=ptype,
                    model_identifier=model_path,
                    gpu_index=gpu_index,
                    context_length=context_length
                )
            else:
                return {"success": False, "error": f"Provider {provider} not supported for loading"}

            return {
                "success": True,
                "instance_id": instance.id,
                "model": instance.display_name,
                "gpu": instance.gpu_index,
                "message": f"Model loaded successfully. Use instance_id '{instance.id}' for chat."
            }

        except Exception as e:
            logger.error(f"load_model error: {e}")
            return {"success": False, "error": str(e)}

    async def unload_model(self, instance_id: str) -> Dict[str, Any]:
        """Unload a model instance."""
        try:
            # Use orchestrator.unload_model to properly clean up
            success = await self.orchestrator.unload_model(instance_id)
            if success:
                return {
                    "success": True,
                    "message": f"Model {instance_id} unloaded successfully"
                }
            else:
                return {
                    "success": False,
                    "error": f"Instance {instance_id} not found"
                }

        except Exception as e:
            logger.error(f"unload_model error: {e}")
            return {"success": False, "error": str(e)}

    async def get_model_status(self, instance_id: str) -> Dict[str, Any]:
        """Get model instance status."""
        try:
            for provider in self.orchestrator.providers.values():
                instance = provider.instances.get(instance_id)
                if instance:
                    status = await provider.get_status(instance_id)
                    return {
                        "success": True,
                        "instance_id": instance_id,
                        "model": instance.display_name,
                        "status": status.value,
                        "gpu": instance.gpu_index,
                        "context_length": instance.context_length,
                        "request_count": instance.request_count
                    }

            return {
                "success": False,
                "error": f"Instance {instance_id} not found"
            }

        except Exception as e:
            logger.error(f"get_model_status error: {e}")
            return {"success": False, "error": str(e)}

    async def list_model_presets(self) -> Dict[str, Any]:
        """List saved model load presets from data/presets/."""
        import os, json
        presets_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "presets",
        )
        presets = []
        if os.path.isdir(presets_dir):
            for fname in sorted(os.listdir(presets_dir)):
                if fname.endswith(".json"):
                    try:
                        with open(os.path.join(presets_dir, fname), 'r', encoding='utf-8') as f:
                            p = json.load(f)
                            presets.append({
                                "name": p.get("name", fname.replace(".json", "")),
                                "provider": p.get("provider"),
                                "model_name": p.get("modelName"),
                                "context_length": p.get("contextLength"),
                                "gpu_index": p.get("gpuIndex"),
                            })
                    except Exception:
                        continue
        return {"success": True, "presets": presets, "count": len(presets)}

    async def load_from_preset(self, preset_name: str) -> Dict[str, Any]:
        """Load a model using a saved preset configuration."""
        import os, json
        presets_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "presets",
        )

        if not os.path.isdir(presets_dir):
            return {"success": False, "error": "No presets directory found."}

        # Find matching preset (case-insensitive name or filename match)
        preset = None
        name_lower = preset_name.lower()
        for fname in os.listdir(presets_dir):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(presets_dir, fname), 'r', encoding='utf-8') as f:
                        p = json.load(f)
                        if (p.get("name", "").lower() == name_lower
                                or fname.replace(".json", "").lower() == name_lower):
                            preset = p
                            break
                except Exception:
                    continue

        if not preset:
            return {
                "success": False,
                "error": f"Preset '{preset_name}' not found. Use list_model_presets to see available presets.",
            }

        model_id = preset.get("modelId") or preset.get("modelName")
        if not model_id:
            return {"success": False, "error": f"Preset '{preset_name}' has no model path configured."}

        # Enforce explicit GPU for LM Studio presets unless caller already saved one.
        # This keeps LM Studio loads deterministic and single-GPU pinned.
        gpu_index = preset.get("gpuIndex")
        if gpu_index is None and preset.get("provider") == "lm_studio":
            gpu_index = 0

        return await self.load_model(
            model_name=model_id,
            gpu_index=gpu_index,
            provider=preset.get("provider", "llama_cpp"),
            context_length=preset.get("contextLength", 4096),
        )

    async def save_model_preset(
        self,
        name: str,
        provider: str,
        model_id: str,
        context_length: int = 4096,
        gpu_index: Optional[int] = None,
        gpu_layers: int = -1,
    ) -> Dict[str, Any]:
        """Save a model preset to data/presets for later quick loading."""
        import os
        import re
        import json
        import time

        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        presets_dir = os.path.join(root, "data", "presets")
        os.makedirs(presets_dir, exist_ok=True)

        safe_name = re.sub(r"[^\w\-]", "", name.replace(" ", "_")).lower()[:100] or "preset"
        preset = {
            "id": f"preset-{int(time.time() * 1000)}",
            "name": name,
            "provider": provider,
            "modelId": model_id,
            "modelName": model_id,
            "contextLength": int(context_length or 4096),
            "gpuLayers": int(gpu_layers if gpu_layers is not None else -1),
            "gpuIndex": int(gpu_index) if gpu_index is not None else (0 if provider == "lm_studio" else None),
            "createdAt": int(time.time() * 1000),
        }

        path = os.path.join(presets_dir, f"{safe_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(preset, f, indent=2)

        return {
            "success": True,
            "preset": preset,
            "path": path,
            "message": f"Saved preset '{name}'. Use load_from_preset('{name}') to load it.",
        }
