"""
Tool Router - Routes tool calls to appropriate handlers.

This is the central dispatcher for all Meta Agent tool calls.
"""

from typing import Dict, Any, List, Optional, Callable, TYPE_CHECKING
import json
import logging
import inspect

from .model_tools import ModelTools, TOOL_DEFINITIONS as MODEL_TOOLS
from .workflow_tools import WorkflowTools, TOOL_DEFINITIONS as WORKFLOW_TOOLS
from .system_tools import SystemTools, TOOL_DEFINITIONS as SYSTEM_TOOLS
from .n8n_tools import N8nTools, TOOL_DEFINITIONS as N8N_TOOLS
from .suggestions import format_suggestions_for_prompt

# New super tools
from .web_tools import WebTools, TOOL_DEFINITIONS as WEB_TOOLS
from .browser_tools import BrowserTools, TOOL_DEFINITIONS as BROWSER_TOOLS
from .file_tools import FileTools, TOOL_DEFINITIONS as FILE_TOOLS
from .code_tools import CodeTools, TOOL_DEFINITIONS as CODE_TOOLS
from .communication_tools import CommunicationTools, TOOL_DEFINITIONS as COMMUNICATION_TOOLS
from .data_tools import DataTools, TOOL_DEFINITIONS as DATA_TOOLS
from .utility_tools import UtilityTools, TOOL_DEFINITIONS as UTILITY_TOOLS
from .vision_tools import VisionTools, TOOL_DEFINITIONS as VISION_TOOLS
from .codebase_tools import CodebaseTools, TOOL_DEFINITIONS as CODEBASE_TOOLS
from .comfyui_tools import ComfyUITools, TOOL_DEFINITIONS as COMFYUI_TOOLS
from .agent_tools import AgentTools, TOOL_DEFINITIONS as AGENT_TOOLS
from .marketplace_tools import MarketplaceTools, TOOL_DEFINITIONS as MARKETPLACE_TOOLS
from .tts_tools import TTSTools, TOOL_DEFINITIONS as TTS_TOOLS
from .music_tools import MusicTools, TOOL_DEFINITIONS as MUSIC_TOOLS
from .gguf_tools import GGUFTools, TOOL_DEFINITIONS as GGUF_TOOLS

if TYPE_CHECKING:
    from backend.personas import Persona

logger = logging.getLogger("tool_router")


# Combine all tool definitions
AVAILABLE_TOOLS = (
    MODEL_TOOLS + WORKFLOW_TOOLS + SYSTEM_TOOLS + N8N_TOOLS +
    WEB_TOOLS + BROWSER_TOOLS + FILE_TOOLS + CODE_TOOLS +
    COMMUNICATION_TOOLS + DATA_TOOLS + UTILITY_TOOLS + VISION_TOOLS +
    CODEBASE_TOOLS + COMFYUI_TOOLS + AGENT_TOOLS + MARKETPLACE_TOOLS +
    TTS_TOOLS + MUSIC_TOOLS + GGUF_TOOLS
)


# Category display names and order
CATEGORY_INFO: Dict[str, str] = {
    "model": "Model Management",
    "system": "System Status",
    "workflow": "Workflow Automation",
    "n8n": "n8n Instances",
    "web": "Web & Browser",
    "files": "File Operations",
    "code": "Code Execution",
    "communication": "Communication",
    "data": "Data & APIs",
    "utility": "Utilities",
    "vision": "Vision & Images",
    "codebase": "Codebase Guide",
    "comfyui": "ComfyUI Image Generation",
    "comfyui_pool": "ComfyUI Instance Pool",
    "agents": "Sub-Agent Spawning",
    "marketplace": "n8n Marketplace",
    "tts": "Text-to-Speech",
    "music": "Music Generation",
    "gguf": "GGUF Model Downloads",
}

# Order for displaying categories
CATEGORY_ORDER = [
    "model", "gguf", "system", "workflow", "marketplace", "n8n", "comfyui", "comfyui_pool",
    "tts", "music",
    "web", "files", "code", "communication",
    "data", "utility", "vision", "codebase", "agents"
]


def _get_tool_category_map() -> Dict[str, str]:
    """Build a map from tool name to category."""
    from backend.personas import TOOL_GROUPS

    tool_to_category = {}
    for category, tools in TOOL_GROUPS.items():
        if tools is not None:  # Skip meta groups like "all", "safe", "power"
            for tool_name in tools:
                tool_to_category[tool_name] = category
    return tool_to_category


def _format_tool_condensed(tool: Dict) -> str:
    """
    Format a single tool in condensed format.

    Example output: "- web_search(query*, num_results=5): Search web via DuckDuckGo"
    """
    name = tool["name"]

    # Build parameter signature
    params = []
    properties = tool.get("parameters", {}).get("properties", {})
    required = tool.get("parameters", {}).get("required", [])

    for param_name, param_info in properties.items():
        is_required = param_name in required
        default = param_info.get("default")

        if is_required:
            params.append(f"{param_name}*")
        elif default is not None:
            # Show default value (truncate if too long)
            default_str = str(default)
            if len(default_str) > 15:
                default_str = default_str[:12] + "..."
            params.append(f"{param_name}={default_str}")
        else:
            params.append(param_name)

    param_str = ", ".join(params) if params else ""

    # Truncate description to first sentence or ~60 chars
    desc = tool.get("description", "")
    # Take first sentence
    if ". " in desc:
        desc = desc.split(". ")[0]
    # Truncate if still too long
    if len(desc) > 80:
        desc = desc[:77] + "..."

    return f"  - {name}({param_str}): {desc}"


def _format_tools_by_category(tools: List[Dict]) -> str:
    """Format tools grouped by category with headers."""
    tool_to_category = _get_tool_category_map()

    # Group tools by category
    categorized: Dict[str, List[Dict]] = {}
    uncategorized: List[Dict] = []

    for tool in tools:
        category = tool_to_category.get(tool["name"])
        if category:
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(tool)
        else:
            uncategorized.append(tool)

    # Build output in category order
    lines = []
    for category in CATEGORY_ORDER:
        if category in categorized:
            display_name = CATEGORY_INFO.get(category, category.title())
            lines.append(f"## {display_name}")
            for tool in categorized[category]:
                lines.append(_format_tool_condensed(tool))
            lines.append("")  # Blank line between categories

    # Add any uncategorized tools at the end
    if uncategorized:
        lines.append("## Other")
        for tool in uncategorized:
            lines.append(_format_tool_condensed(tool))
        lines.append("")

    return "\n".join(lines)


def get_tools_for_prompt(condensed: bool = True) -> str:
    """
    Get tool definitions formatted for the system prompt.

    Args:
        condensed: If True, use compact format with categories (~5k chars).
                   If False, use verbose format (~17k chars).
    """
    if condensed:
        tools_text = "# Available Tools\n\n"
        tools_text += "Call tools with: `{\"tool\": \"name\", \"arguments\": {...}}`\n"
        tools_text += "Parameters marked with * are required.\n\n"
        tools_text += _format_tools_by_category(list(AVAILABLE_TOOLS))
        return tools_text

    # Verbose format (original)
    tools_text = "# Available Tools\n\n"
    tools_text += "You can call these tools by responding with a JSON object in this format:\n"
    tools_text += '```json\n{"tool": "tool_name", "arguments": {...}}\n```\n\n'

    for tool in AVAILABLE_TOOLS:
        tools_text += f"## {tool['name']}\n"
        tools_text += f"{tool['description']}\n"

        if tool.get("parameters", {}).get("properties"):
            tools_text += "**Parameters:**\n"
            for param, info in tool["parameters"]["properties"].items():
                required = param in tool["parameters"].get("required", [])
                req_str = " (required)" if required else ""
                tools_text += f"- `{param}`{req_str}: {info.get('description', '')}\n"

        tools_text += "\n"

    return tools_text


def get_openai_tools_format() -> List[Dict]:
    """Get tools in OpenAI function calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool.get("parameters", {"type": "object", "properties": {}})
            }
        }
        for tool in AVAILABLE_TOOLS
    ]


class ToolRouter:
    """Routes tool calls to appropriate handlers."""

    def __init__(self, orchestrator, n8n_manager, settings, comfyui_manager=None, media_catalog=None, comfyui_pool=None, tts_manager=None, music_manager=None, gguf_downloader=None):
        self.orchestrator = orchestrator
        self.n8n_manager = n8n_manager
        self.settings = settings
        self.comfyui_manager = comfyui_manager

        # Get tool configurations from settings
        tools_config = getattr(settings, 'tools', {}) if settings else {}

        # Initialize tool handlers
        self.model_tools = ModelTools(orchestrator)
        self.workflow_tools = WorkflowTools(orchestrator, n8n_manager)
        self.n8n_tools = N8nTools(n8n_manager)
        # SystemTools needs references to other tools for quick_setup
        self.system_tools = SystemTools(
            orchestrator,
            settings,
            n8n_manager=n8n_manager,
            model_tools=self.model_tools
        )

        # Build web tools config from settings (inject search engine configs)
        web_config = tools_config.get("web_tools", {})
        if settings and hasattr(settings, 'get'):
            search_config = settings.get("services.search")
            if search_config:
                web_config["search"] = search_config

        # Initialize new super tools
        self.web_tools = WebTools(web_config)
        self.browser_tools = BrowserTools(tools_config.get("browser_tools", {}))
        self.file_tools = FileTools(tools_config.get("file_tools", {}))
        self.code_tools = CodeTools(tools_config.get("code_tools", {}))
        self.communication_tools = CommunicationTools(tools_config.get("communication", {}))
        self.data_tools = DataTools(tools_config.get("data_tools", {}))
        self.utility_tools = UtilityTools(tools_config.get("utility_tools", {}))

        # Vision tools need orchestrator and browser_tools for screenshot integration
        self.vision_tools = VisionTools(
            orchestrator=orchestrator,
            browser_tools=self.browser_tools,
            config=tools_config.get("vision_tools", {})
        )

        # Codebase tools for exploring the project
        self.codebase_tools = CodebaseTools(tools_config.get("codebase_tools", {}))

        # ComfyUI tools for image generation module
        self.comfyui_tools = ComfyUITools(comfyui_manager, media_catalog, pool=comfyui_pool)

        # Agent tools for sub-agent spawning
        self.agent_tools = AgentTools(orchestrator, n8n_manager, settings, comfyui_manager, comfyui_pool=comfyui_pool)

        # Marketplace tools for workflow discovery
        self.marketplace_tools = MarketplaceTools(orchestrator, n8n_manager)

        # TTS tools for text-to-speech module
        self.tts_tools = TTSTools(tts_manager)

        # Music tools for music generation module
        self.music_tools = MusicTools(music_manager)

        # GGUF download tools
        self.gguf_tools = GGUFTools(gguf_downloader)

        # Context injected at request time for agent tools
        self._agent_context = {}
        # Optional request-scoped tool allowlist (None = unrestricted)
        self._allowed_tools: Optional[set] = None
        # Request/session-scoped state for policy checks.
        self._session_tool_history: List[str] = []
        self._session_persona_id: Optional[str] = None

        # Build routing table
        self._routes: Dict[str, Callable] = {
            # Model tools
            "list_available_models": self.model_tools.list_available_models,
            "list_loaded_models": self.model_tools.list_loaded_models,
            "load_model": self.model_tools.load_model,
            "unload_model": self.model_tools.unload_model,
            "get_model_status": self.model_tools.get_model_status,

            # Workflow tools
            "describe_node": self.workflow_tools.describe_node,
            "list_credentials": self.workflow_tools.list_credentials,
            "build_workflow": self.workflow_tools.build_workflow,
            "deploy_workflow": self.workflow_tools.deploy_workflow,
            "list_workflows": self.workflow_tools.list_workflows,
            "delete_workflow": self.workflow_tools.delete_workflow,
            "delete_all_workflows": self.workflow_tools.delete_all_workflows,
            "describe_credential_types": self.workflow_tools.describe_credential_types,
            "create_credential": self.workflow_tools.create_credential,
            "update_credential": self.workflow_tools.update_credential,
            "delete_credential": self.workflow_tools.delete_credential,
            "list_executions": self.workflow_tools.list_executions,
            "get_execution_result": self.workflow_tools.get_execution_result,
            "update_workflow": self.workflow_tools.update_workflow,
            "activate_workflow": self.workflow_tools.activate_workflow,
            "deactivate_workflow": self.workflow_tools.deactivate_workflow,
            "trigger_webhook": self.workflow_tools.trigger_webhook,
            "set_variable": self.workflow_tools.set_variable,
            "list_variables": self.workflow_tools.list_variables,
            "flash_workflow": self.workflow_tools.flash_workflow,

            # System tools
            "get_gpu_status": self.system_tools.get_gpu_status,
            "get_system_health": self.system_tools.get_system_health,
            "get_provider_status": self.system_tools.get_provider_status,
            "get_full_status": self.system_tools.get_full_status,
            "quick_setup": self.system_tools.quick_setup,
            "suggest_actions": self.system_tools.suggest_actions,

            # n8n tools
            "spawn_n8n": self.n8n_tools.spawn_n8n,
            "stop_n8n": self.n8n_tools.stop_n8n,
            "list_n8n_instances": self.n8n_tools.list_n8n_instances,
            "get_n8n_status": self.n8n_tools.get_n8n_status,

            # Web tools
            "web_search": self.web_tools.web_search,
            "google_search": self.web_tools.google_search,
            "serper_search": self.web_tools.serper_search,
            "duckduckgo_search": self.web_tools.duckduckgo_search,
            "fetch_url": self.web_tools.fetch_url,

            # Browser tools
            "browser_open": self.browser_tools.browser_open,
            "browser_screenshot": self.browser_tools.browser_screenshot,
            "browser_click": self.browser_tools.browser_click,
            "browser_type": self.browser_tools.browser_type,
            "browser_extract": self.browser_tools.browser_extract,
            "browser_get_text": self.browser_tools.browser_get_text,
            "browser_scroll": self.browser_tools.browser_scroll,
            "browser_close": self.browser_tools.browser_close,

            # File tools
            "read_file": self.file_tools.read_file,
            "write_file": self.file_tools.write_file,
            "list_directory": self.file_tools.list_directory,
            "search_files": self.file_tools.search_files,
            "search_content": self.file_tools.search_content,
            "file_info": self.file_tools.file_info,
            "delete_file": self.file_tools.delete_file,
            "move_file": self.file_tools.move_file,
            "copy_file": self.file_tools.copy_file,

            # Code tools
            "run_python": self.code_tools.run_python,
            "run_javascript": self.code_tools.run_javascript,
            "run_shell": self.code_tools.run_shell,
            "run_powershell": self.code_tools.run_powershell,

            # Communication tools
            "send_discord": self.communication_tools.send_discord,
            "send_slack": self.communication_tools.send_slack,
            "send_email": self.communication_tools.send_email,
            "send_telegram": self.communication_tools.send_telegram,
            "send_webhook": self.communication_tools.send_webhook,

            # Data tools
            "http_request": self.data_tools.http_request,
            "parse_json": self.data_tools.parse_json,
            "parse_html": self.data_tools.parse_html,
            "convert_data": self.data_tools.convert_data,
            "database_query": self.data_tools.database_query,

            # Utility tools
            "get_datetime": self.utility_tools.get_datetime,
            "calculate": self.utility_tools.calculate,
            "generate_uuid": self.utility_tools.generate_uuid,
            "encode_decode": self.utility_tools.encode_decode,
            "hash_text": self.utility_tools.hash_text,
            "regex_match": self.utility_tools.regex_match,
            "text_transform": self.utility_tools.text_transform,
            "random_string": self.utility_tools.random_string,

            # Vision tools
            "analyze_image": self.vision_tools.analyze_image,
            "analyze_screenshot": self.vision_tools.analyze_screenshot,
            "extract_text_from_image": self.vision_tools.extract_text_from_image,
            "describe_ui": self.vision_tools.describe_ui,
            "compare_images": self.vision_tools.compare_images,
            "find_element": self.vision_tools.find_element,

            # Codebase tools
            "scan_codebase": self.codebase_tools.scan_codebase,
            "explain_file": self.codebase_tools.explain_file,
            "find_feature": self.codebase_tools.find_feature,
            "get_architecture": self.codebase_tools.get_architecture,
            "list_api_endpoints": self.codebase_tools.list_api_endpoints,
            "list_tools": self.codebase_tools.list_tools,
            "explain_concept": self.codebase_tools.explain_concept,
            "get_capabilities": self.codebase_tools.get_capabilities,
            "get_quick_start": self.codebase_tools.get_quick_start,
            "generate_manifest": self.codebase_tools.generate_manifest,
            "query_codebase": self.codebase_tools.query_codebase,

            # ComfyUI tools
            "comfyui_status": self.comfyui_tools.comfyui_status,
            "comfyui_install": self.comfyui_tools.comfyui_install,
            "comfyui_start_api": self.comfyui_tools.comfyui_start_api,
            "comfyui_stop_api": self.comfyui_tools.comfyui_stop_api,
            "comfyui_list_instances": self.comfyui_tools.comfyui_list_instances,
            "comfyui_add_instance": self.comfyui_tools.comfyui_add_instance,
            "comfyui_start_instance": self.comfyui_tools.comfyui_start_instance,
            "comfyui_stop_instance": self.comfyui_tools.comfyui_stop_instance,
            "comfyui_list_models": self.comfyui_tools.comfyui_list_models,
            "comfyui_search_models": self.comfyui_tools.comfyui_search_models,
            "comfyui_download_model": self.comfyui_tools.comfyui_download_model,
            "comfyui_job_status": self.comfyui_tools.comfyui_job_status,
            "comfyui_await_job": self.comfyui_tools.comfyui_await_job,
            "comfyui_generate_image": self.comfyui_tools.comfyui_generate_image,
            "comfyui_get_result": self.comfyui_tools.comfyui_get_result,
            "comfyui_await_result": self.comfyui_tools.comfyui_await_result,
            "comfyui_install_nodes": self.comfyui_tools.comfyui_install_nodes,
            "comfyui_describe_nodes": self.comfyui_tools.comfyui_describe_nodes,
            "comfyui_build_workflow": self.comfyui_tools.comfyui_build_workflow,
            "comfyui_execute_workflow": self.comfyui_tools.comfyui_execute_workflow,
            "comfyui_search_generations": self.comfyui_tools.comfyui_search_generations,
            "comfyui_prepare_input": self.comfyui_tools.comfyui_prepare_input,
            "comfyui_list_node_packs": self.comfyui_tools.comfyui_list_node_packs,
            "comfyui_list_installed_nodes": self.comfyui_tools.comfyui_list_installed_nodes,
            "comfyui_update_nodes": self.comfyui_tools.comfyui_update_nodes,
            "comfyui_remove_node": self.comfyui_tools.comfyui_remove_node,
            "comfyui_remove_instance": self.comfyui_tools.comfyui_remove_instance,
            "comfyui_start_all_instances": self.comfyui_tools.comfyui_start_all_instances,
            "comfyui_stop_all_instances": self.comfyui_tools.comfyui_stop_all_instances,
            "comfyui_model_categories": self.comfyui_tools.comfyui_model_categories,
            "comfyui_get_settings": self.comfyui_tools.comfyui_get_settings,
            "comfyui_update_settings": self.comfyui_tools.comfyui_update_settings,
            "comfyui_update_comfyui": self.comfyui_tools.comfyui_update_comfyui,
            "comfyui_purge": self.comfyui_tools.comfyui_purge,
            "comfyui_manage_external": self.comfyui_tools.comfyui_manage_external,
            "comfyui_list_gpus": self.comfyui_tools.comfyui_list_gpus,
            "comfyui_list_templates": self.comfyui_tools.comfyui_list_templates,
            "comfyui_analyze_workflow": self.comfyui_tools.comfyui_analyze_workflow,

            # ComfyUI Pool tools
            "comfyui_pool_status": self.comfyui_tools.comfyui_pool_status,
            "comfyui_pool_generate": self.comfyui_tools.comfyui_pool_generate,
            "comfyui_pool_batch": self.comfyui_tools.comfyui_pool_batch,
            "comfyui_pool_results": self.comfyui_tools.comfyui_pool_results,

            # Agent tools (sub-agent spawning, memory, user interaction)
            "spawn_agent": self._spawn_agent_wrapper,
            # super_spawn removed — racing now at tool-call level (race_executor.py)
            "batch_spawn_agents": self._batch_spawn_wrapper,
            "check_agents": self._check_agents_wrapper,
            "get_agent_result": self._get_agent_result_wrapper,
            "remember": self.agent_tools.remember,
            "recall": self.agent_tools.recall,
            "ask_user": self.agent_tools.ask_user,
            "list_routing_presets": self.agent_tools.list_routing_presets,
            "save_routing_preset": self.agent_tools.save_routing_preset,
            "activate_routing": self.agent_tools.activate_routing,
            "recommend_routing": self.agent_tools.recommend_routing,
            "provision_models": self.agent_tools.provision_models,
            "generate_preset_workflow": self.agent_tools.generate_preset_workflow,
            "list_model_presets": self.model_tools.list_model_presets,
            "load_from_preset": self.model_tools.load_from_preset,
            "save_model_preset": self.model_tools.save_model_preset,

            # Marketplace tools
            "search_marketplace": self.marketplace_tools.search_marketplace,
            "get_marketplace_workflow": self.marketplace_tools.get_marketplace_workflow,
            "inspect_workflow": self.marketplace_tools.inspect_workflow,
            "configure_workflow": self.marketplace_tools.configure_workflow,

            # TTS tools
            "tts_status": self.tts_tools.tts_status,
            "tts_start_server": self.tts_tools.tts_start_server,
            "tts_stop_server": self.tts_tools.tts_stop_server,
            "tts_list_models": self.tts_tools.tts_list_models,
            "tts_list_workers": self.tts_tools.tts_list_workers,
            "tts_load_model": self.tts_tools.tts_load_model,
            "tts_unload_model": self.tts_tools.tts_unload_model,
            "tts_generate": self.tts_tools.tts_generate,
            "tts_list_voices": self.tts_tools.tts_list_voices,
            "tts_get_model_info": self.tts_tools.tts_get_model_info,
            "tts_install_env": self.tts_tools.tts_install_env,
            "tts_download_weights": self.tts_tools.tts_download_weights,

            # Music tools
            "music_status": self.music_tools.music_status,
            "music_start_server": self.music_tools.music_start_server,
            "music_stop_server": self.music_tools.music_stop_server,
            "music_list_models": self.music_tools.music_list_models,
            "music_list_workers": self.music_tools.music_list_workers,
            "music_load_model": self.music_tools.music_load_model,
            "music_unload_model": self.music_tools.music_unload_model,
            "music_generate": self.music_tools.music_generate,
            "music_get_presets": self.music_tools.music_get_presets,
            "music_install_model": self.music_tools.music_install_model,
            "music_install_status": self.music_tools.music_install_status,
            "music_list_outputs": self.music_tools.music_list_outputs,

            # GGUF tools
            "gguf_search": self.gguf_tools.gguf_search,
            "gguf_list_files": self.gguf_tools.gguf_list_files,
            "gguf_download": self.gguf_tools.gguf_download,
            "gguf_download_status": self.gguf_tools.gguf_download_status,
            "gguf_cancel_download": self.gguf_tools.gguf_cancel_download,
        }

    def set_agent_context(self, instance_id: str, conversation_store, persona_manager,
                          parent_conv_id: str = None, persona_id: str = None):
        """Set request-scoped context needed by agent tools (spawn_agent)."""
        self._agent_context = {
            "_instance_id": instance_id,
            "_conversation_store": conversation_store,
            "_persona_manager": persona_manager,
            "_parent_conv_id": parent_conv_id,
        }
        self._session_persona_id = persona_id
        self._session_tool_history = []

    def set_routing_preset_override(self, preset_id: str):
        """Override the global routing preset for this session (per-panel routing)."""
        self._routing_preset_override = preset_id
        # Also inject into agent context so sub-agents inherit it
        if hasattr(self, '_agent_context'):
            self._agent_context["_routing_preset_id"] = preset_id

    def set_allowed_tools(self, tool_names: Optional[List[str]]):
        """Set request-scoped tool allowlist for execution enforcement."""
        if tool_names is None:
            self._allowed_tools = None
        else:
            self._allowed_tools = set(tool_names)

    async def _spawn_agent_wrapper(self, **kwargs):
        """Wrapper that injects request context into spawn_agent."""
        kwargs.update(self._agent_context)
        # Pass routing override if set
        if hasattr(self, '_routing_preset_override') and self._routing_preset_override:
            kwargs.setdefault("_routing_preset_id", self._routing_preset_override)
        return await self.agent_tools.spawn_agent(**kwargs)

    async def _batch_spawn_wrapper(self, **kwargs):
        """Wrapper that injects request context into batch_spawn_agents."""
        kwargs.update(self._agent_context)
        if hasattr(self, '_routing_preset_override') and self._routing_preset_override:
            kwargs.setdefault("_routing_preset_id", self._routing_preset_override)
        return await self.agent_tools.batch_spawn_agents(**kwargs)

    async def _check_agents_wrapper(self, **kwargs):
        """Wrapper that scopes check_agents to current parent conversation."""
        parent_conv_id = kwargs.get("_parent_conv_id", self._agent_context.get("_parent_conv_id"))
        return await self.agent_tools.check_agents(_parent_conv_id=parent_conv_id)

    async def _get_agent_result_wrapper(self, **kwargs):
        """Wrapper that scopes get_agent_result to current parent conversation."""
        parent_conv_id = kwargs.get("_parent_conv_id", self._agent_context.get("_parent_conv_id"))
        agent_id = kwargs.get("agent_id")
        if not agent_id:
            return {
                "success": False,
                "error": "agent_id is required",
            }
        return await self.agent_tools.get_agent_result(agent_id=agent_id, _parent_conv_id=parent_conv_id)

    async def cleanup(self):
        """Cleanup resources (browser, etc.) on shutdown."""
        try:
            await self.browser_tools.cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up browser tools: {e}")

    async def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call."""
        # Alias common weaker-model tool names to real tools.
        tool_aliases = {
            "comfyui_workflow": "comfyui_build_workflow",
            "comfyui_poll_workflow_progress": "comfyui_get_result",
            "comfyui_poll_result": "comfyui_get_result",
            "comfyui_get_generation_result": "comfyui_get_result",
        }
        if tool_name not in self._routes and tool_name in tool_aliases:
            mapped = tool_aliases[tool_name]
            if mapped in self._routes:
                tool_name = mapped

        if tool_name not in self._routes:
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}",
                "available_tools": list(self._routes.keys())
            }

        # Enforce persona/request tool restrictions server-side.
        if self._allowed_tools is not None and tool_name not in self._allowed_tools:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' is not allowed for this persona/session."
            }

        # Comfy local-first policy for Comfy-focused personas.
        # Keep web tools available, but only after local Comfy references are checked.
        web_tools = {"web_search", "fetch_url", "scrape_page"}
        comfy_personas = {"image_creator", "ai_creative"}
        comfy_local_prereq_tools = {
            "comfyui_list_templates",
            "comfyui_analyze_workflow",
            "comfyui_describe_nodes",
            "comfyui_list_models",
            "comfyui_list_installed_nodes",
            "comfyui_list_node_packs",
            "comfyui_build_workflow",
        }
        if tool_name in web_tools and self._session_persona_id in comfy_personas:
            has_local_probe = any(
                t in comfy_local_prereq_tools for t in self._session_tool_history
            )
            if not has_local_probe:
                return {
                    "success": False,
                    "error": (
                        "Local-first policy: check local Comfy references before web. "
                        "Call comfyui_list_templates and/or comfyui_analyze_workflow first. "
                        "Use web tools only as fallback when local references are insufficient."
                    )
                }

        try:
            handler = self._routes[tool_name]
            normalized_args = self._normalize_tool_arguments(tool_name, arguments or {})
            safe_args = self._filter_handler_arguments(handler, normalized_args)
            result = await handler(**safe_args)
            self._session_tool_history.append(tool_name)
            return result
        except TypeError as e:
            # Likely wrong arguments
            return {
                "success": False,
                "error": f"Invalid arguments for {tool_name}: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Tool execution error: {tool_name} - {e}")
            return {
                "success": False,
                "error": f"Tool execution failed: {str(e)}"
            }

    def _normalize_tool_arguments(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Best-effort argument normalization for weaker models."""
        args = dict(arguments or {})

        if tool_name == "spawn_agent":
            if args.get("persona_id") is None:
                args.pop("persona_id", None)
            mtc = args.get("max_tool_calls")
            if isinstance(mtc, bool) or (mtc is not None and not isinstance(mtc, int)):
                args.pop("max_tool_calls", None)

        if tool_name == "batch_spawn_agents":
            agents = args.get("agents")
            if isinstance(agents, list):
                cleaned = []
                for spec in agents:
                    if isinstance(spec, dict):
                        spec = dict(spec)
                        if spec.get("persona_id") is None:
                            spec.pop("persona_id", None)
                        mtc = spec.get("max_tool_calls")
                        if isinstance(mtc, bool) or (mtc is not None and not isinstance(mtc, int)):
                            spec.pop("max_tool_calls", None)
                    cleaned.append(spec)
                args["agents"] = cleaned

        if tool_name == "get_agent_result":
            if not args.get("agent_id"):
                alias = args.get("id") or args.get("sub_agent_id")
                if alias:
                    args["agent_id"] = alias

        if tool_name == "load_from_preset":
            if not args.get("preset_name"):
                alias = args.get("name") or args.get("preset")
                if alias:
                    args["preset_name"] = alias

        return args

    def _filter_handler_arguments(self, handler: Callable, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Drop unknown kwargs for handlers that don't accept **kwargs.
        This avoids brittle failures when models hallucinate extra keys.
        """
        sig = inspect.signature(handler)
        params = sig.parameters
        accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if accepts_var_kw:
            return arguments

        allowed = {
            name for name, p in params.items()
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        return {k: v for k, v in arguments.items() if k in allowed}

    @staticmethod
    def _strip_json_comments(s: str) -> str:
        """Strip JS-style // comments from JSON strings (8B models add these).

        Only strips // comments that appear OUTSIDE of JSON string values.
        This avoids corrupting URLs like http://... inside strings.
        """
        result = []
        in_string = False
        escape_next = False
        i = 0
        while i < len(s):
            ch = s[i]
            if escape_next:
                result.append(ch)
                escape_next = False
                i += 1
                continue
            if in_string:
                if ch == '\\':
                    escape_next = True
                    result.append(ch)
                elif ch == '"':
                    in_string = False
                    result.append(ch)
                else:
                    result.append(ch)
            else:
                if ch == '"':
                    in_string = True
                    result.append(ch)
                elif ch == '/' and i + 1 < len(s) and s[i + 1] == '/':
                    # Skip to end of line (JS-style comment)
                    while i < len(s) and s[i] != '\n':
                        i += 1
                    continue  # don't increment i again
                else:
                    result.append(ch)
            i += 1
        return ''.join(result)

    def parse_tool_call(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Parse an LLM response for tool calls WITHOUT executing.

        Returns None if no tool call found, otherwise returns dict with:
        - tool: the tool name
        - arguments: the arguments parsed
        - known: True if tool exists in routes, False if unknown
        """
        import re

        unknown_tool_attempt = None

        # Strategy 1: Look for explicit tool call markers
        patterns = [
            r'<tool_call>\s*(\{.*?\})\s*</tool_call>',
            r'```json\s*(\{.*?\})\s*```',
        ]

        for pattern in patterns:
            match = re.search(pattern, response_text, re.DOTALL)
            if match:
                raw = match.group(1)
                for attempt_text in [raw, self._strip_json_comments(raw)]:
                    try:
                        tool_call = json.loads(attempt_text)
                        tool_name = tool_call.get("tool") or tool_call.get("name")
                        if tool_name and tool_name in self._routes:
                            logger.info(f"[parse] Strategy 1 matched tool: {tool_name}")
                            return {"tool": tool_name, "arguments": tool_call.get("arguments", {}), "known": True}
                        elif tool_name:
                            unknown_tool_attempt = {"tool": tool_name, "arguments": tool_call.get("arguments", {})}
                        break
                    except json.JSONDecodeError:
                        continue

        # Strategy 2: Find JSON objects that look like tool calls
        tool_pattern = r'\{\s*"(?:tool|name)"\s*:\s*"([^"]+)"'
        for match in re.finditer(tool_pattern, response_text):
            start_pos = match.start()
            json_str = self._extract_balanced_json(response_text[start_pos:])
            if json_str:
                for attempt_text in [json_str, self._strip_json_comments(json_str)]:
                    try:
                        tool_call = json.loads(attempt_text)
                        tool_name = tool_call.get("tool") or tool_call.get("name")
                        if tool_name and tool_name in self._routes:
                            logger.info(f"[parse] Strategy 2 matched tool: {tool_name}")
                            return {"tool": tool_name, "arguments": tool_call.get("arguments", {}), "known": True}
                        elif tool_name and not unknown_tool_attempt:
                            unknown_tool_attempt = {"tool": tool_name, "arguments": tool_call.get("arguments", {})}
                        break
                    except json.JSONDecodeError as e:
                        logger.debug(f"[parse] Strategy 2 JSONDecodeError at pos {start_pos}: {e}")
                        continue
            else:
                logger.debug(f"[parse] Strategy 2 balanced JSON extraction failed at pos {start_pos}, text starts: {response_text[start_pos:start_pos+80]!r}")

        # Strategy 3: Try json.loads on entire response as last resort
        if not unknown_tool_attempt:
            stripped = response_text.strip()
            if stripped.startswith('{') and stripped.endswith('}'):
                try:
                    tool_call = json.loads(stripped)
                    tool_name = tool_call.get("tool") or tool_call.get("name")
                    if tool_name and tool_name in self._routes:
                        logger.info(f"[parse] Strategy 3 (full text) matched tool: {tool_name}")
                        return {"tool": tool_name, "arguments": tool_call.get("arguments", {}), "known": True}
                    elif tool_name:
                        unknown_tool_attempt = {"tool": tool_name, "arguments": tool_call.get("arguments", {})}
                except json.JSONDecodeError:
                    pass

        if unknown_tool_attempt:
            return {**unknown_tool_attempt, "known": False}

        logger.debug(f"[parse] No tool call found in {len(response_text)} char response")
        return None

    async def parse_and_execute(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Parse an LLM response for tool calls and execute them.

        Returns None if no tool call found, otherwise returns dict with:
        - tool: the tool name that was called
        - arguments: the arguments passed
        - result: the tool execution result
        """
        parsed = self.parse_tool_call(response_text)
        if not parsed:
            return None

        tool_name = parsed["tool"]
        arguments = parsed["arguments"]

        if parsed.get("known", True):
            result = await self.execute(tool_name, arguments)
            return {"tool": tool_name, "arguments": arguments, "result": result}
        else:
            # Unknown tool — return error with suggestions
            logger.warning(f"Agent tried unknown tool: '{tool_name}'")
            suggestions = self._suggest_similar_tools(tool_name)
            error_msg = f"Unknown tool: '{tool_name}'. This tool does not exist."
            if suggestions:
                error_msg += f" Did you mean: {', '.join(suggestions)}?"
            return {
                "tool": tool_name,
                "arguments": arguments,
                "result": {"success": False, "error": error_msg}
            }

    def _suggest_similar_tools(self, attempted_name: str, max_suggestions: int = 5) -> List[str]:
        """Find tools with similar names to suggest alternatives."""
        attempted_lower = attempted_name.lower()
        # Split into parts for prefix/suffix matching
        parts = attempted_lower.replace("_", " ").split()

        scored = []
        for name in self._routes:
            name_lower = name.lower()
            score = 0
            # Exact prefix match (e.g. "comfyui_" tools)
            prefix = attempted_lower.split("_")[0] + "_"
            if name_lower.startswith(prefix):
                score += 3
            # Word overlap
            name_parts = name_lower.replace("_", " ").split()
            overlap = len(set(parts) & set(name_parts))
            score += overlap * 2
            # Substring match
            for part in parts:
                if len(part) >= 3 and part in name_lower:
                    score += 1
            if score > 0:
                scored.append((name, score))

        scored.sort(key=lambda x: -x[1])
        return [name for name, _ in scored[:max_suggestions]]

    def _extract_balanced_json(self, text: str) -> Optional[str]:
        """Extract a balanced JSON object from the start of text."""
        if not text or text[0] != '{':
            return None

        depth = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(text):
            if escape_next:
                escape_next = False
                continue

            if char == '\\' and in_string:
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if not in_string:
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        return text[:i + 1]

        return None

    def get_tool_list(self) -> List[str]:
        """Get list of available tool names."""
        return list(self._routes.keys())

    def get_tool_info(self, tool_name: str) -> Optional[Dict]:
        """Get info about a specific tool."""
        for tool in AVAILABLE_TOOLS:
            if tool["name"] == tool_name:
                return tool
        return None

    async def build_dynamic_prompt(self, persona: Optional["Persona"] = None) -> str:
        """
        Build a dynamic system prompt with current system state.

        This gives the Meta Agent awareness of what's currently loaded,
        GPU status, n8n instances, and provides proactive suggestions.

        Args:
            persona: Optional persona to customize the prompt. If None or if
                    persona.include_system_state is True, full state is included.
        """
        # If persona exists and doesn't want system state, return empty
        if persona and not persona.include_system_state:
            return ""

        # Get current system snapshot
        snapshot = await self.system_tools.get_system_snapshot()

        return self._format_system_state(snapshot, persona)

    def _format_system_state(self, snapshot: Dict[str, Any],
                             persona: Optional["Persona"] = None) -> str:
        """Format the system state snapshot as a prompt section."""
        state_lines = ["# Current System State\n"]

        # Models section
        models = snapshot.get("models", [])
        if models:
            state_lines.append(f"## Loaded Models ({len(models)})")
            for m in models:
                status_icon = "✅" if m.get("status") == "ready" else "⏳"
                provider = m.get('provider', 'unknown')
                locality = 'local' if provider in ('lm_studio', 'llama_cpp', 'vllm', 'ollama') else 'cloud'
                ctx = m.get('context_length', 0)
                state_lines.append(
                    f"- {m.get('model')} (id: {m.get('instance_id')}, "
                    f"{provider}, {locality}, GPU {m.get('gpu')}, ctx: {ctx}, "
                    f"{m.get('status')}) {status_icon}"
                )
        else:
            state_lines.append("## Loaded Models (0)")
            state_lines.append("- *No models loaded*")
        state_lines.append("")

        # GPUs section
        gpus = snapshot.get("gpus", [])
        if gpus:
            state_lines.append(f"## GPUs ({len(gpus)})")
            for g in gpus:
                total = g.get("memory_total_mb", 0)
                free = g.get("memory_free_mb", 0)
                used_pct = int((1 - free / total) * 100) if total > 0 else 0
                state_lines.append(
                    f"- GPU {g.get('index')}: {g.get('name')} - "
                    f"{free}MB free ({used_pct}% used)"
                )
        else:
            state_lines.append("## GPUs")
            state_lines.append("- *No NVIDIA GPUs detected*")
        state_lines.append("")

        # n8n section
        n8n_instances = snapshot.get("n8n_instances", [])
        running_n8n = [i for i in n8n_instances if i.get("running")]
        if running_n8n:
            state_lines.append(f"## n8n Instances ({len(running_n8n)})")
            for n in running_n8n:
                state_lines.append(f"- Port {n.get('port')}: running")
        else:
            state_lines.append("## n8n Instances (0)")
            state_lines.append("- *No n8n instances running*")
        state_lines.append("")

        # Queue section
        queue = snapshot.get("queue", {})
        state_lines.append("## Request Queue")
        state_lines.append(f"- Pending: {queue.get('pending', 0)}, Processing: {queue.get('processing', 0)}")
        state_lines.append("")

        # Generate suggestions only if persona has system tools
        include_suggestions = True
        if persona and persona.tools:
            # Only include suggestions if persona has system or all tools
            has_system_tools = "all" in persona.tools or "system" in persona.tools
            include_suggestions = has_system_tools

        if include_suggestions:
            suggestions = self.system_tools.suggestion_engine.generate_suggestions(snapshot)
            suggestions_text = format_suggestions_for_prompt(suggestions)
            if suggestions_text:
                state_lines.append(suggestions_text)
                state_lines.append("")

        return "\n".join(state_lines)

    def get_tools_prompt_for_categories(self, persona: "Persona",
                                        categories: List[str]) -> str:
        """
        Get tool definitions filtered to only the specified categories.

        Falls back to full persona tool list if filtering yields nothing.
        """
        from backend.personas import TOOL_GROUPS

        # Collect tool names from selected categories
        allowed_tools = set()
        for cat in categories:
            group_tools = TOOL_GROUPS.get(cat)
            if group_tools is not None:
                allowed_tools.update(group_tools)

        if not allowed_tools:
            logger.warning(f"Category filter yielded no tools for {categories}, falling back to full list")
            return self.get_tools_prompt_for_persona(persona)

        # Filter AVAILABLE_TOOLS
        filtered = [t for t in AVAILABLE_TOOLS if t["name"] in allowed_tools]
        if not filtered:
            return self.get_tools_prompt_for_persona(persona)

        tools_text = "# Available Tools\n\n"
        tools_text += "Call tools with: `{\"tool\": \"name\", \"arguments\": {...}}`\n"
        tools_text += "Parameters marked with * are required.\n\n"
        tools_text += _format_tools_by_category(filtered)

        logger.info(f"Filtered tools to {len(filtered)} from categories {categories}")
        return tools_text

    def get_tools_prompt_for_persona(self, persona: "Persona") -> str:
        """
        Get tool definitions formatted for the system prompt, filtered by persona.

        Args:
            persona: The persona whose tools to include

        Returns:
            Formatted tool definitions string, or empty string if no tools
        """
        if not persona.tools:
            logger.debug(f"Persona {persona.id} has no tools configured")
            return ""

        # Get the list of allowed tool names
        all_tool_names = list(self._routes.keys())
        logger.debug(f"All available tools: {all_tool_names}")

        # Import here to avoid circular import
        from backend.personas import TOOL_GROUPS

        # Resolve tool groups manually (we don't have PersonaManager instance here)
        if "all" in persona.tools:
            allowed_tools = set(all_tool_names)
            logger.debug(f"Persona {persona.id} has 'all' tools: {len(allowed_tools)} tools")
        else:
            allowed_tools = set()
            for tool_ref in persona.tools:
                if tool_ref in TOOL_GROUPS:
                    group_tools = TOOL_GROUPS[tool_ref]
                    if group_tools is not None:
                        allowed_tools.update(group_tools)
                elif tool_ref in all_tool_names:
                    allowed_tools.add(tool_ref)
            logger.debug(f"Persona {persona.id} resolved tools: {allowed_tools}")

        if not allowed_tools:
            logger.warning(f"No allowed tools found for persona {persona.id}")
            return ""

        # Filter AVAILABLE_TOOLS
        filtered_tools = [t for t in AVAILABLE_TOOLS if t["name"] in allowed_tools]
        logger.debug(f"Filtered to {len(filtered_tools)} tool definitions")

        if not filtered_tools:
            logger.warning(f"No tool definitions matched for persona {persona.id}")
            return ""

        # Use condensed format with categories (saves ~60% tokens)
        tools_text = "# Available Tools\n\n"
        tools_text += "Call tools with: `{\"tool\": \"name\", \"arguments\": {...}}`\n"
        tools_text += "Parameters marked with * are required.\n\n"
        tools_text += _format_tools_by_category(filtered_tools)

        return tools_text
