"""
AgentNate Tool System

Tools that the Meta Agent can call to control the system.
"""

from .tool_router import ToolRouter, AVAILABLE_TOOLS, get_tools_for_prompt
from .model_tools import ModelTools
from .workflow_tools import WorkflowTools
from .system_tools import SystemTools
from .n8n_tools import N8nTools
from .suggestions import SuggestionEngine, format_suggestions_for_prompt

# New super tools
from .web_tools import WebTools
from .browser_tools import BrowserTools
from .file_tools import FileTools
from .code_tools import CodeTools
from .communication_tools import CommunicationTools
from .data_tools import DataTools
from .utility_tools import UtilityTools
from .vision_tools import VisionTools
from .codebase_tools import CodebaseTools
from .comfyui_tools import ComfyUITools

__all__ = [
    # Core
    'ToolRouter',
    'AVAILABLE_TOOLS',
    'get_tools_for_prompt',

    # Existing tools
    'ModelTools',
    'WorkflowTools',
    'SystemTools',
    'N8nTools',
    'SuggestionEngine',
    'format_suggestions_for_prompt',

    # New super tools
    'WebTools',
    'BrowserTools',
    'FileTools',
    'CodeTools',
    'CommunicationTools',
    'DataTools',
    'UtilityTools',
    'VisionTools',
    'CodebaseTools',
    'ComfyUITools',
]
