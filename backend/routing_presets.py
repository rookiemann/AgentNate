"""
Routing Preset Manager

Manages model routing presets that map persona IDs to specific provider/model
combinations. Enables sub-agent swarm patterns where different personas
(coder, researcher, etc.) route to different loaded models.

Storage: Individual JSON files in data/routing_presets/
"""

import os
import re
import json
import time
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger("routing_presets")

# Project root (two levels up from this file)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRESETS_DIR = os.path.join(_PROJECT_ROOT, "data", "routing_presets")

# Heuristic keywords for recommend()
_CODING_KEYWORDS = ["deepseek", "coder", "code", "starcoder", "codellama", "codestral", "qwen2.5-coder"]
_VISION_KEYWORDS = ["llava", "vision", "bakllava", "pixtral", "moondream"]
_GENERAL_KEYWORDS = ["claude", "gpt", "sonnet", "opus", "gemini", "llama", "qwen", "mistral", "phi", "command"]


def _sanitize_filename(name: str) -> str:
    """Sanitize preset name for use as filename."""
    name = name.replace(" ", "_")
    name = re.sub(r'[^\w\-]', '', name)
    name = name.lower()[:100]
    return name or "preset"


class RoutingPresetManager:
    """Manages routing presets stored as JSON files."""

    def __init__(self, settings):
        self.settings = settings
        os.makedirs(PRESETS_DIR, exist_ok=True)

    def list_presets(self) -> List[Dict]:
        """List all saved routing presets."""
        presets = []
        try:
            for filename in os.listdir(PRESETS_DIR):
                if filename.endswith(".json"):
                    filepath = os.path.join(PRESETS_DIR, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            presets.append(json.load(f))
                    except (json.JSONDecodeError, IOError) as e:
                        logger.warning(f"Failed to load routing preset {filename}: {e}")
        except OSError as e:
            logger.error(f"Failed to list routing presets: {e}")
        presets.sort(key=lambda p: p.get("name", "").lower())
        return presets

    def get_preset(self, preset_id: str) -> Optional[Dict]:
        """Load a single preset by ID."""
        try:
            for filename in os.listdir(PRESETS_DIR):
                if filename.endswith(".json"):
                    filepath = os.path.join(PRESETS_DIR, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            preset = json.load(f)
                            if preset.get("id") == preset_id:
                                return preset
                    except (json.JSONDecodeError, IOError):
                        continue
        except OSError:
            pass
        return None

    def save_preset(self, name: str, routes: Dict, description: str = "") -> Dict:
        """Save a routing preset to disk. Returns the preset dict."""
        preset_id = f"rp-{int(time.time() * 1000)}"
        preset = {
            "id": preset_id,
            "name": name,
            "description": description,
            "created_at": int(time.time() * 1000),
            "routes": routes,
        }
        filepath = os.path.join(PRESETS_DIR, f"{_sanitize_filename(name)}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(preset, f, indent=2)
        logger.info(f"Saved routing preset '{name}' ({preset_id}) to {filepath}")
        return preset

    def delete_preset(self, preset_id: str) -> bool:
        """Delete a routing preset by ID."""
        try:
            for filename in os.listdir(PRESETS_DIR):
                if filename.endswith(".json"):
                    filepath = os.path.join(PRESETS_DIR, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            preset = json.load(f)
                        # Close file before delete (Windows file locking)
                        if preset.get("id") == preset_id:
                            os.remove(filepath)
                            logger.info(f"Deleted routing preset '{preset.get('name')}' ({preset_id})")
                            # Clear active if this was the active preset
                            if self.settings.get("agent.active_routing_preset_id") == preset_id:
                                self.settings.set("agent.routing_enabled", False)
                                self.settings.set("agent.active_routing_preset_id", None)
                            return True
                    except (json.JSONDecodeError, IOError):
                        continue
        except OSError as e:
            logger.error(f"Failed to delete routing preset: {e}")
        return False

    def resolve(self, preset_id: str, persona_id: str,
                loaded_instances: list) -> Optional[str]:
        """
        Resolve persona_id to a loaded instance_id using the preset.

        Supports two resolution modes:
        1. Direct instance_id — route has "instance_id" field, matched against loaded instances
        2. Pattern match — route has "provider" + "model_match", matched by provider type and name substring

        Returns instance_id or None if no match found.
        """
        preset = self.get_preset(preset_id)
        if not preset:
            return None

        routes_data = preset.get("routes", {})

        # Handle both dict format {"persona_id": {route}} and list format [{persona, model, ...}]
        route = None
        if isinstance(routes_data, dict):
            route = routes_data.get(persona_id)
        elif isinstance(routes_data, list):
            # List format: find matching persona entry
            for entry in routes_data:
                if isinstance(entry, dict):
                    entry_persona = entry.get("persona_id") or entry.get("persona") or ""
                    if entry_persona == persona_id or entry_persona in persona_id or persona_id in entry_persona:
                        route = entry
                        break
            # If no persona match, use first entry as default
            if not route and routes_data and isinstance(routes_data[0], dict):
                route = routes_data[0]

        if not route:
            return None

        from providers.base import ModelStatus

        # Mode 1: Direct instance_id match
        direct_id = route.get("instance_id")
        if direct_id:
            for instance in loaded_instances:
                if instance.id == direct_id and instance.status in (ModelStatus.READY, ModelStatus.BUSY):
                    logger.info(
                        f"[routing] Resolved {persona_id} → {instance.display_name or instance.model_identifier} "
                        f"(direct, {instance.id[:8]}...)"
                    )
                    return instance.id
            # Direct ID specified but not loaded/ready
            return None

        # Mode 2: Provider + model_match pattern
        target_provider = route.get("provider", "").lower()
        target_match = (route.get("model_match") or route.get("model_pattern") or "").lower()

        # Mode 3: Direct model name match (8B models store "model" key)
        if not target_provider and not target_match:
            model_name = (route.get("model") or "").lower()
            if model_name:
                for instance in loaded_instances:
                    if instance.status not in (ModelStatus.READY, ModelStatus.BUSY):
                        continue
                    if (model_name in instance.model_identifier.lower()
                            or model_name in (instance.display_name or "").lower()):
                        logger.info(
                            f"[routing] Resolved {persona_id} → {instance.display_name or instance.model_identifier} "
                            f"(model name match, {instance.id[:8]}...)"
                        )
                        return instance.id
            return None

        if not target_provider or not target_match:
            return None

        for instance in loaded_instances:
            provider_match = instance.provider_type.value.lower() == target_provider
            if not provider_match:
                continue

            name_match = (
                target_match in instance.model_identifier.lower()
                or target_match in (instance.display_name or "").lower()
            )
            if not name_match:
                continue

            if instance.status in (ModelStatus.READY, ModelStatus.BUSY):
                logger.info(
                    f"[routing] Resolved {persona_id} → {instance.display_name or instance.model_identifier} "
                    f"({instance.provider_type.value}, {instance.id[:8]}...)"
                )
                return instance.id

        return None

    def recommend(self, loaded_instances: list) -> Dict[str, Any]:
        """
        Analyze loaded models and recommend persona→model routing.

        Uses heuristic keyword matching on model names to determine
        which model is best suited for each persona.
        """
        if not loaded_instances:
            return {
                "success": False,
                "error": "No models loaded. Load at least 2 models to get a routing recommendation.",
            }

        # Classify each loaded instance
        coding_models = []
        vision_models = []
        general_models = []

        for inst in loaded_instances:
            name_lower = (inst.model_identifier + " " + inst.display_name).lower()

            if any(kw in name_lower for kw in _CODING_KEYWORDS):
                coding_models.append(inst)
            elif any(kw in name_lower for kw in _VISION_KEYWORDS):
                vision_models.append(inst)

            # All models are candidates for general use
            general_models.append(inst)

        # Prefer cloud models for orchestration (they're smarter)
        cloud_providers = {"openrouter", "lm_studio"}
        local_providers = {"llama_cpp", "vllm", "ollama"}

        cloud_models = [m for m in general_models if m.provider_type.value in cloud_providers]
        local_models = [m for m in general_models if m.provider_type.value in local_providers]

        # Pick best orchestrator: cloud if available, else largest local
        orchestrator_model = (cloud_models or general_models)[0] if general_models else None

        # Pick best coder: coding-specific if available, else local general, else orchestrator
        coder_model = (coding_models or local_models or general_models)[0] if general_models else None

        # Pick best researcher: same as orchestrator (needs reasoning)
        researcher_model = orchestrator_model

        # Pick best vision: vision-specific if available, else orchestrator
        vision_model = (vision_models or [orchestrator_model])[0] if orchestrator_model else None

        # Build route suggestions
        routes = {}
        model_descriptions = []

        def _make_route(persona_id, inst, role_desc):
            if inst:
                routes[persona_id] = {
                    "provider": inst.provider_type.value,
                    "model_match": inst.display_name or inst.model_identifier.split("/")[-1].split("\\")[-1],
                    "label": f"{inst.display_name or inst.model_identifier} ({inst.provider_type.value})",
                }
                model_descriptions.append(f"  {persona_id} → {routes[persona_id]['label']} ({role_desc})")

        _make_route("coder", coder_model, "code generation")
        _make_route("code_assistant", coder_model, "code review")
        _make_route("researcher", researcher_model, "web research")
        _make_route("data_analyst", coder_model, "data analysis")
        _make_route("image_creator", orchestrator_model, "image generation orchestration")
        _make_route("workflow_builder", orchestrator_model, "workflow automation")
        _make_route("vision_agent", vision_model, "image analysis")

        return {
            "success": True,
            "recommended_routes": routes,
            "summary": "\n".join(model_descriptions),
            "loaded_models": [
                {
                    "id": inst.id,
                    "provider": inst.provider_type.value,
                    "name": inst.display_name or inst.model_identifier,
                    "status": inst.status.value,
                }
                for inst in loaded_instances
            ],
            "note": "Use save_routing_preset to save this recommendation. "
                    "Personas not listed will use the parent agent's model.",
        }
