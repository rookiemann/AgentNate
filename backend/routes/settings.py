"""
Settings Routes - API endpoints for application settings.
"""

from typing import Dict, Any, Optional
from fastapi import APIRouter, Request
from pydantic import BaseModel
import logging

router = APIRouter()
logger = logging.getLogger("routes.settings")


class SettingsUpdateRequest(BaseModel):
    """Request body for updating settings."""
    key: str
    value: Any


class SectionUpdateRequest(BaseModel):
    """Request body for updating a settings section."""
    section: str
    values: Dict[str, Any]


@router.get("")
async def get_all_settings(request: Request):
    """Get all application settings."""
    settings = request.app.state.settings
    return {
        "success": True,
        "settings": settings.get_all()
    }


@router.get("/{section}")
async def get_settings_section(request: Request, section: str):
    """Get a specific settings section."""
    settings = request.app.state.settings

    value = settings.get_section(section)
    if value is not None:
        return {
            "success": True,
            "section": section,
            "settings": value
        }
    return {
        "success": False,
        "error": f"Section '{section}' not found"
    }


@router.post("/update")
async def update_setting(request: Request, body: SettingsUpdateRequest):
    """Update a single setting by key."""
    settings = request.app.state.settings

    try:
        old_value = settings.get(body.key)
        settings.set(body.key, body.value)
        logger.info(f"Setting updated: {body.key} = {body.value}")
        return {
            "success": True,
            "key": body.key,
            "old_value": old_value,
            "new_value": body.value
        }
    except Exception as e:
        logger.error(f"Failed to update setting {body.key}: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@router.post("/section")
async def update_settings_section(request: Request, body: SectionUpdateRequest):
    """Update multiple settings in a section."""
    settings = request.app.state.settings

    try:
        settings.set_section(body.section, body.values)
        logger.info(f"Section updated: {body.section}")
        return {
            "success": True,
            "section": body.section,
            "updated_keys": list(body.values.keys())
        }
    except Exception as e:
        logger.error(f"Failed to update section {body.section}: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@router.post("/reset")
async def reset_settings(request: Request, section: Optional[str] = None):
    """Reset settings to defaults."""
    settings = request.app.state.settings

    try:
        settings.reset_to_defaults(section)
        if section:
            logger.info(f"Settings section reset: {section}")
            return {
                "success": True,
                "message": f"Section '{section}' reset to defaults"
            }
        else:
            logger.info("All settings reset to defaults")
            return {
                "success": True,
                "message": "All settings reset to defaults"
            }
    except Exception as e:
        logger.error(f"Failed to reset settings: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@router.get("/path/info")
async def get_settings_path(request: Request):
    """Get the settings file path."""
    settings = request.app.state.settings
    return {
        "success": True,
        "settings_path": str(settings.settings_path),
        "settings_dir": str(settings.settings_dir)
    }


class ValidateSearchKeyRequest(BaseModel):
    engine: str  # "google" or "serper"
    api_key: str
    cx: str = ""  # Only needed for Google


@router.post("/validate-search-key")
async def validate_search_key(body: ValidateSearchKeyRequest):
    """
    Test a search API key with a simple query.
    Returns whether the key is valid and working.
    """
    from backend.tools.web_tools import WebTools

    tools = WebTools({})
    result = await tools.validate_key(body.engine, body.api_key, body.cx)
    return result
