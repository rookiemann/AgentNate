"""
Settings Manager with JSON persistence.

Supports nested key access via dot notation (e.g., "providers.openrouter.api_key")
and automatic persistence to %APPDATA%/AgentNate/settings.json (Windows)
or ~/.config/AgentNate/settings.json (Linux/macOS).
"""

import json
import os
import sys
from typing import Any, Optional, Dict
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.signals import Signal


class SettingsManager:
    """
    Manages application settings with JSON persistence.
    Supports nested keys via dot notation (e.g., "providers.openrouter.api_key")
    """

    DEFAULT_SETTINGS = {
        "providers": {
            "llama_cpp": {
                "enabled": True,
                "models_directory": r"E:\LL STUDIO",
                "default_n_ctx": 4096,
                "default_n_gpu_layers": 99,
                "default_n_parallel": 1,
                "use_mmap": True,
                "flash_attn": True,
                "use_mlock": True,
            },
            "lm_studio": {
                "enabled": True,
                "base_url": "http://localhost:1234/v1",
                # Optional fallback GPU index for LM Studio loads when no gpu_index is provided.
                # Null means auto-select first detected GPU.
                "default_gpu_index": None,
            },
            "openrouter": {
                "enabled": False,
                "api_key": "",
                "default_model": "openrouter/auto",
                "site_url": "https://agentnate.local",
                "app_name": "AgentNate",
            },
            "ollama": {
                "enabled": True,
                "base_url": "http://localhost:11434",
                "keep_alive": "5m",
            },
            "vllm": {
                "enabled": False,
                "env_path": "envs/vllm",
                "models_directory": r"E:\LL STUDIO",
                "default_port_range_start": 8100,
                "default_max_model_len": None,  # Auto-detect from model
                "default_gpu_memory_utilization": 0.6,
                "default_tensor_parallel_size": 1,
                "enforce_eager": True,  # CUDA graphs require Triton (not on Windows)
                "load_timeout": 600,  # 10 minutes for large models
            },
        },
        "services": {
            "search": {
                "default_engine": "duckduckgo",  # google, serper, duckduckgo
                "google": {
                    "enabled": False,
                    "keys": [],  # [{api_key, cx, label}]
                },
                "serper": {
                    "enabled": False,
                    "keys": [],  # [{api_key, label}]
                },
                "duckduckgo": {
                    "enabled": True,
                },
            },
        },
        "inference": {
            "default_max_tokens": 1024,
            "default_temperature": 0.7,
            "default_top_p": 0.95,
            "default_top_k": 40,
            "default_repeat_penalty": 1.1,
            "default_presence_penalty": 0.0,
            "default_frequency_penalty": 0.0,
            "default_mirostat": 0,
            "default_mirostat_tau": 5.0,
            "default_mirostat_eta": 0.1,
            "default_typical_p": 1.0,
            "default_tfs_z": 1.0,
        },
        "orchestrator": {
            "max_concurrent_inferences": 4,
            "health_check_interval": 30,
            "request_timeout": 300,
            "jit_loading_enabled": True,
            "auto_unload_idle_minutes": 0,  # 0 = disabled
        },
        "ui": {
            "theme": "dark",
            "auto_scroll": True,
            "show_timestamps": True,
            "code_highlighting": True,
            "window_width": 1400,
            "window_height": 900,
            "auto_load_preset": None,
        },
        "chat": {
            "system_prompt": "",
            "save_history": True,
            "history_directory": "",  # Empty = use default
            "max_history_messages": 100,
        },
        "agent": {
            "max_sub_agents": 4,
            "sub_agent_timeout": 300,
            "routing_enabled": False,
            "active_routing_preset_id": None,
            "pin_head_to_openrouter": False,
            "delegate_all": True,
            "tool_race_enabled": True,
            "tool_race_candidates": 3,
            "batch_workers": 3,
        },
    }

    def __init__(self, app_name: str = "AgentNate", settings_dir: Optional[str] = None):
        """
        Initialize settings manager.

        Args:
            app_name: Application name for settings directory
            settings_dir: Override settings directory (useful for portable mode)
        """
        self.app_name = app_name
        self._settings: Dict[str, Any] = {}
        self._settings_dir = settings_dir
        self._settings_path = self._get_settings_path()

        # Signals
        self.on_settings_changed = Signal()

        # Load settings
        self._load()

    def _get_settings_path(self) -> Path:
        """Get platform-appropriate settings directory."""
        if self._settings_dir:
            # Use provided directory (portable mode)
            settings_dir = Path(self._settings_dir)
        elif os.name == "nt":  # Windows
            base = os.environ.get("APPDATA", os.path.expanduser("~"))
            settings_dir = Path(base) / self.app_name
        else:  # macOS/Linux
            base = os.path.expanduser("~/.config")
            settings_dir = Path(base) / self.app_name

        settings_dir.mkdir(parents=True, exist_ok=True)
        return settings_dir / "settings.json"

    def _load(self):
        """Load settings from file, merging with defaults."""
        # Start with deep copy of defaults
        self._settings = self._deep_copy(self.DEFAULT_SETTINGS)

        if self._settings_path.exists():
            try:
                with open(self._settings_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    self._deep_merge(self._settings, loaded)
            except Exception as e:
                print(f"Warning: Failed to load settings: {e}")

    def _save(self):
        """Persist settings to disk."""
        try:
            with open(self._settings_path, "w", encoding="utf-8") as f:
                json.dump(self._settings, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Failed to save settings: {e}")

    def _deep_copy(self, obj: Any) -> Any:
        """Create a deep copy of nested dicts/lists."""
        if isinstance(obj, dict):
            return {k: self._deep_copy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_copy(item) for item in obj]
        return obj

    def _deep_merge(self, base: dict, override: dict):
        """Recursively merge override into base."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get setting value using dot notation.

        Example:
            get("providers.openrouter.api_key")
            get("inference.default_temperature", 0.7)
        """
        keys = key.split(".")
        value = self._settings

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any, save: bool = True):
        """
        Set setting value using dot notation.

        Example:
            set("providers.openrouter.api_key", "sk-...")
            set("inference.default_temperature", 0.8)
        """
        keys = key.split(".")
        target = self._settings

        # Navigate/create nested structure
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]

        # Set the value
        old_value = target.get(keys[-1])
        target[keys[-1]] = value

        if save:
            self._save()

        # Emit signal if value changed
        if old_value != value:
            self.on_settings_changed.emit(key, value)

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get an entire settings section.

        Example:
            get_section("providers.llama_cpp")
        """
        value = self.get(section, {})
        return value if isinstance(value, dict) else {}

    def set_section(self, section: str, values: Dict[str, Any], save: bool = True):
        """
        Set multiple values in a section.

        Example:
            set_section("inference", {"default_temperature": 0.8, "default_top_p": 0.9})
        """
        for key, value in values.items():
            self.set(f"{section}.{key}", value, save=False)

        if save:
            self._save()

    def get_all(self) -> Dict[str, Any]:
        """Get complete settings dictionary (deep copy)."""
        return self._deep_copy(self._settings)

    def reset_to_defaults(self, section: Optional[str] = None):
        """
        Reset settings to defaults.

        Args:
            section: If provided, only reset that section. Otherwise reset all.
        """
        if section:
            # Reset specific section
            default_value = self._navigate_defaults(section)
            if default_value is not None:
                self.set(section, self._deep_copy(default_value))
        else:
            # Reset all
            self._settings = self._deep_copy(self.DEFAULT_SETTINGS)
            self._save()
            self.on_settings_changed.emit("*", None)

    def _navigate_defaults(self, key: str) -> Any:
        """Navigate to a key in DEFAULT_SETTINGS."""
        keys = key.split(".")
        value = self.DEFAULT_SETTINGS

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return None

        return value

    @property
    def settings_path(self) -> Path:
        """Get the settings file path."""
        return self._settings_path

    @property
    def settings_dir(self) -> Path:
        """Get the settings directory path."""
        return self._settings_path.parent

    def export_settings(self, path: str):
        """Export settings to a file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._settings, f, indent=2, ensure_ascii=False)

    def import_settings(self, path: str, merge: bool = True):
        """
        Import settings from a file.

        Args:
            path: Path to settings file
            merge: If True, merge with existing. If False, replace entirely.
        """
        with open(path, "r", encoding="utf-8") as f:
            imported = json.load(f)

        if merge:
            self._deep_merge(self._settings, imported)
        else:
            self._settings = self._deep_copy(self.DEFAULT_SETTINGS)
            self._deep_merge(self._settings, imported)

        self._save()
        self.on_settings_changed.emit("*", None)
