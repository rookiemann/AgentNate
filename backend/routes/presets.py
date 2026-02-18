"""
Model Presets Routes

REST API for saving/loading model load configurations to disk.
Presets are stored as individual JSON files in data/presets/
"""

import os
import re
import json
import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()

# Presets directory (relative to project root)
PRESETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "presets")


def _ensure_presets_dir():
    """Create presets directory if it doesn't exist."""
    os.makedirs(PRESETS_DIR, exist_ok=True)


def _sanitize_filename(name: str) -> str:
    """
    Sanitize preset name for use as filename.

    - Replaces spaces with underscores
    - Removes special characters except alphanumeric, underscore, hyphen
    - Limits length to 100 characters
    - Lowercases everything
    """
    # Replace spaces with underscores
    name = name.replace(" ", "_")
    # Remove anything that's not alphanumeric, underscore, or hyphen
    name = re.sub(r'[^\w\-]', '', name)
    # Lowercase and limit length
    name = name.lower()[:100]
    # Ensure not empty
    if not name:
        name = "preset"
    return name


def _get_preset_path(name: str) -> str:
    """Get full path to a preset file."""
    safe_name = _sanitize_filename(name)
    return os.path.join(PRESETS_DIR, f"{safe_name}.json")


class SavePresetRequest(BaseModel):
    """Request to save a preset - accepts any preset structure from frontend."""
    preset: Dict[str, Any]


@router.get("/list")
async def list_presets() -> List[Dict[str, Any]]:
    """
    List all saved presets.

    Returns list of preset data objects.
    """
    _ensure_presets_dir()
    presets = []

    try:
        for filename in os.listdir(PRESETS_DIR):
            if filename.endswith(".json"):
                filepath = os.path.join(PRESETS_DIR, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        preset = json.load(f)
                        presets.append(preset)
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"Failed to load preset {filename}: {e}")
                    continue
    except OSError as e:
        logger.error(f"Failed to list presets: {e}")

    # Sort by name
    presets.sort(key=lambda p: p.get("name", "").lower())
    return presets


@router.post("/save")
async def save_preset(body: SavePresetRequest) -> Dict[str, Any]:
    """
    Save a preset to disk.

    Filename is derived from preset name (sanitized).
    """
    _ensure_presets_dir()

    preset = body.preset
    filepath = _get_preset_path(preset.get("name", "unnamed"))

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(preset, f, indent=2)

        logger.info(f"Saved preset '{preset['name']}' to {filepath}")
        return {"success": True, "path": filepath, "preset": preset}

    except IOError as e:
        logger.error(f"Failed to save preset: {e}")
        return {"success": False, "error": str(e)}


@router.delete("/{preset_id}")
async def delete_preset(preset_id: str) -> Dict[str, Any]:
    """
    Delete a preset by ID.

    Searches for preset file by ID and removes it.
    """
    _ensure_presets_dir()

    # Find preset file by ID
    try:
        for filename in os.listdir(PRESETS_DIR):
            if filename.endswith(".json"):
                filepath = os.path.join(PRESETS_DIR, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        preset = json.load(f)
                        if preset.get("id") == preset_id:
                            os.remove(filepath)
                            logger.info(f"Deleted preset '{preset.get('name')}' ({filepath})")
                            return {"success": True, "deleted": preset_id}
                except (json.JSONDecodeError, IOError):
                    continue

        return {"success": False, "error": f"Preset not found: {preset_id}"}

    except OSError as e:
        logger.error(f"Failed to delete preset: {e}")
        return {"success": False, "error": str(e)}


@router.get("/{preset_id}")
async def get_preset(preset_id: str) -> Dict[str, Any]:
    """
    Get a specific preset by ID.
    """
    _ensure_presets_dir()

    try:
        for filename in os.listdir(PRESETS_DIR):
            if filename.endswith(".json"):
                filepath = os.path.join(PRESETS_DIR, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        preset = json.load(f)
                        if preset.get("id") == preset_id:
                            return {"success": True, "preset": preset}
                except (json.JSONDecodeError, IOError):
                    continue

        return {"success": False, "error": f"Preset not found: {preset_id}"}

    except OSError as e:
        logger.error(f"Failed to get preset: {e}")
        return {"success": False, "error": str(e)}
