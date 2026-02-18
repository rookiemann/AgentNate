"""
Tool Routes

API endpoints for the Meta Agent tool system.
"""

import asyncio
import json
import logging
import re as _re
import time
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("tools")
router = APIRouter()


class ToolCallRequest(BaseModel):
    tool: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class AgentChatRequest(BaseModel):
    """Request for agent chat with tool calling."""
    message: str
    instance_id: Optional[str] = None
    conversation_id: Optional[str] = None
    persona_id: Optional[str] = "auto"
    additional_instructions: Optional[str] = None  # User's custom instructions layered on top of persona
    max_tool_calls: int = 25
    autonomous: bool = False  # If True, keep executing tools until done or max reached
    params: Optional[Dict[str, Any]] = None  # Inference params (max_tokens, temperature, etc.)
    abort_id: Optional[str] = None  # ID to check for abort signals
    routing_preset_id: Optional[str] = None  # Per-panel routing preset override


# Safety limits for autonomous tool loops.
MIN_AGENT_TOOL_CALLS = 1
MAX_AGENT_TOOL_CALLS = 50


def _clamp_max_tool_calls(value: Optional[int]) -> int:
    """Clamp requested autonomous tool-call budget to safe server limits."""
    requested = value if value is not None else 25
    return max(MIN_AGENT_TOOL_CALLS, min(int(requested), MAX_AGENT_TOOL_CALLS))


# Limited direct-execution tools for head quick checks.
def _is_comfy_intent(message: str) -> bool:
    t = (message or "").lower()
    return any(k in t for k in ("comfy", "image", "sdxl", "stable diffusion", "workflow json", "prompt graph"))


def _is_n8n_intent(message: str) -> bool:
    t = (message or "").lower()
    return any(k in t for k in ("n8n", "workflow", "webhook", "automation", "node graph"))


# Keywords that indicate a simple status/info query the head can handle directly.
_SIMPLE_QUERY_PATTERNS = [
    # GPU / hardware
    r"\bgpu\b.*\b(status|info|check|what|show|tell)\b",
    r"\b(status|info|check|what|show|tell)\b.*\bgpu\b",
    # Models loaded
    r"\b(loaded|running)\b.*\bmodel",
    r"\bmodel.*\b(loaded|running|status)\b",
    r"\blist\b.*\bmodel",
    r"\bwhat.*model.*\b(loaded|running|available)\b",
    # n8n status
    r"\bn8n\b.*\b(running|status|started|up)\b",
    r"\b(is|check|status)\b.*\bn8n\b",
    # ComfyUI status
    r"\bcomfyui?\b.*\b(running|status|started|up)\b",
    r"\b(is|check|status)\b.*\bcomfyui?\b",
    # Generic system status
    r"\bsystem\b.*\bstatus\b",
    r"\bstatus\b.*\bsystem\b",
]
import re as _simple_re
_SIMPLE_QUERY_RE = [_simple_re.compile(p, _simple_re.IGNORECASE) for p in _SIMPLE_QUERY_PATTERNS]

# Keywords that indicate a complex creative/build task → always delegate.
_COMPLEX_TASK_KEYWORDS = [
    "build", "create", "make", "generate", "design", "write", "implement",
    "deploy", "set up", "setup", "configure", "install", "develop",
    "refactor", "optimize", "analyze", "research", "plan",
    "workflow", "automation", "pipeline", "template",
]


def _should_delegate(message: str) -> bool:
    """
    Decide if the head agent should delegate to a worker or handle directly.

    Returns True if the task should be delegated (complex/creative work).
    Returns False if the head can handle it directly (simple status queries).
    """
    t = (message or "").strip().lower()
    if not t:
        return True  # Empty → delegate for safety

    # Short messages that match simple query patterns → head handles directly
    word_count = len(t.split())

    # Check for complex task keywords first — these always delegate
    for kw in _COMPLEX_TASK_KEYWORDS:
        if kw in t:
            return True

    # Check if it matches a simple status/info query pattern
    for pattern in _SIMPLE_QUERY_RE:
        if pattern.search(t):
            return False  # Head handles it

    # Short questions (< 20 words) without complex keywords → head handles directly
    if word_count <= 20 and any(t.startswith(q) for q in ("what ", "which ", "how many ", "is ", "are ", "check ", "show ", "list ", "tell ", "count ", "do ")):
        return False

    # Default: delegate for anything that doesn't clearly match a simple query
    return True


async def _build_worker_preflight_context(tool_router, message: str) -> str:
    """Gather concise system preflight context to prepend to worker task."""
    lines = []
    try:
        gpu = await tool_router.system_tools.get_gpu_status()
        if gpu.get("success"):
            gpus = gpu.get("gpus", []) or []
            if gpus:
                summary = []
                for g in gpus[:8]:
                    idx = g.get("index")
                    name = g.get("name") or f"GPU {idx}"
                    util = g.get("utilization")
                    vram = g.get("vram_used_mb")
                    summary.append(f"{idx}:{name} util={util}% vram_used={vram}MB")
                lines.append("GPU status: " + "; ".join(summary))
            else:
                lines.append("GPU status: no GPU devices reported")
    except Exception:
        pass

    try:
        loaded = await tool_router.model_tools.list_loaded_models()
        if loaded.get("success"):
            inst = loaded.get("instances", []) or []
            if inst:
                model_bits = [f"{i.get('instance_id')}[{i.get('provider')}] gpu={i.get('gpu')}" for i in inst[:8]]
                lines.append("Loaded models: " + "; ".join(model_bits))
            else:
                lines.append("Loaded models: none")
    except Exception:
        pass

    if _is_comfy_intent(message):
        api_running = False
        try:
            comfy = await tool_router.comfyui_tools.comfyui_status()
            if comfy.get("success"):
                api_running = bool(comfy.get("api_running"))
                lines.append(f"ComfyUI API running: {api_running}")
                if not comfy.get("bootstrapped"):
                    lines.append("ComfyUI not bootstrapped: run comfyui_install first")
                elif not api_running:
                    lines.append("ComfyUI API not running: run comfyui_start_api first")
        except Exception:
            pass
        try:
            if api_running:
                inst = await tool_router.comfyui_tools.comfyui_list_instances()
                if inst.get("success"):
                    instances = inst.get("instances", []) or []
                    if instances:
                        info = [f"{x.get('instance_id')}@{x.get('port')} running={x.get('running')}" for x in instances[:8]]
                        lines.append("ComfyUI instances: " + "; ".join(info))
                    else:
                        lines.append("ComfyUI instances: none")
        except Exception:
            pass
        try:
            if api_running:
                g = await tool_router.comfyui_tools.comfyui_list_gpus()
                if g.get("success"):
                    gpus = g.get("gpus", []) or []
                    if gpus:
                        ginfo = [f"{x.get('id')}:{x.get('name')} free={x.get('memory_free')}" for x in gpus[:8]]
                        lines.append("ComfyUI GPUs: " + "; ".join(ginfo))
        except Exception:
            pass
        try:
            if api_running:
                m = await tool_router.comfyui_tools.comfyui_list_models()
                if m.get("success"):
                    count = 0
                    models = m.get("models")
                    if isinstance(models, dict):
                        count = sum(len(v or []) for v in models.values())
                    elif isinstance(models, list):
                        count = len(models)
                    lines.append(f"ComfyUI models available: {count}")
        except Exception:
            pass
        try:
            if api_running:
                p = await tool_router.comfyui_tools.comfyui_pool_status()
                if p.get("success"):
                    lines.append(
                        f"ComfyUI pool: running={p.get('running_instances', 0)}/{p.get('total_instances', 0)}"
                    )
        except Exception:
            pass

    if _is_n8n_intent(message):
        try:
            n8n = await tool_router.n8n_tools.get_n8n_status()
            if n8n.get("success"):
                lines.append(f"n8n status: running={n8n.get('running')} port={n8n.get('port')}")
            else:
                lines.append("n8n status: not running/unknown")
        except Exception:
            pass
        try:
            inst = await tool_router.n8n_tools.list_n8n_instances()
            if inst.get("success"):
                instances = inst.get("instances", []) or []
                if instances:
                    bits = [f"{x.get('port')} running={x.get('running')}" for x in instances[:8]]
                    lines.append("n8n instances: " + "; ".join(bits))
        except Exception:
            pass

    if not lines:
        return ""
    return "Supervisor preflight context:\n- " + "\n- ".join(lines)


class CreatePersonaRequest(BaseModel):
    """Request to create a custom persona."""
    id: str
    name: str
    description: str
    system_prompt: str
    tools: List[str] = Field(default_factory=list)
    include_system_state: bool = False
    temperature: float = 0.7


class UpdatePersonaRequest(BaseModel):
    """Request to update a custom persona."""
    name: Optional[str] = None
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    tools: Optional[List[str]] = None
    include_system_state: Optional[bool] = None
    temperature: Optional[float] = None


class RenameConversationRequest(BaseModel):
    """Request to rename a conversation."""
    name: str


class SetPersonaRequest(BaseModel):
    """Request to change a conversation's persona."""
    persona_id: str


class AgentRespondRequest(BaseModel):
    """User response to an ask_user tool invocation."""
    abort_id: str
    response: str


class AgentWorkerModelSwitchRequest(BaseModel):
    """Manual worker model switch request."""
    agent_id: str
    instance_id: str


class SaveConversationRequest(BaseModel):
    """Request to save a regular chat conversation."""
    messages: List[Dict[str, str]]
    name: str = "Untitled"
    persona_id: str = "none"
    model_id: Optional[str] = None
    conv_type: str = "chat"


@router.get("/list")
async def list_tools(request: Request):
    """List all available tools."""
    from backend.tools import AVAILABLE_TOOLS

    return {
        "tools": [
            {
                "name": t["name"],
                "description": t["description"],
                "parameters": t.get("parameters", {}).get("properties", {})
            }
            for t in AVAILABLE_TOOLS
        ]
    }


@router.get("/info/{tool_name}")
async def get_tool_info(request: Request, tool_name: str):
    """Get detailed info about a specific tool."""
    from backend.tools import AVAILABLE_TOOLS

    for tool in AVAILABLE_TOOLS:
        if tool["name"] == tool_name:
            return {"tool": tool}

    return {"error": f"Tool not found: {tool_name}"}


@router.post("/call")
async def call_tool(request: Request, body: ToolCallRequest):
    """Directly call a tool (bypassing LLM)."""
    tool_router = request.app.state.tool_router
    result = await tool_router.execute(body.tool, body.arguments)
    return result


# =============================================================================
# Persona Endpoints
# =============================================================================

@router.get("/personas")
async def list_personas(request: Request):
    """List all available personas."""
    persona_manager = request.app.state.persona_manager

    return {
        "personas": [
            p.to_dict() for p in persona_manager.list_all()
        ]
    }


@router.get("/personas/{persona_id}")
async def get_persona(request: Request, persona_id: str):
    """Get a persona by ID."""
    persona_manager = request.app.state.persona_manager

    persona = persona_manager.get(persona_id)
    if not persona:
        raise HTTPException(status_code=404, detail=f"Persona not found: {persona_id}")

    return {"persona": persona.to_dict()}


@router.post("/personas")
async def create_persona(request: Request, body: CreatePersonaRequest):
    """Create a custom persona."""
    from backend.personas import Persona

    persona_manager = request.app.state.persona_manager

    persona = Persona(
        id=body.id,
        name=body.name,
        description=body.description,
        system_prompt=body.system_prompt,
        tools=body.tools,
        include_system_state=body.include_system_state,
        temperature=body.temperature,
        predefined=False,
    )

    try:
        created = persona_manager.create_custom(persona)
        return {"persona": created.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/personas/{persona_id}")
async def update_persona(request: Request, persona_id: str, body: UpdatePersonaRequest):
    """Update a custom persona."""
    persona_manager = request.app.state.persona_manager

    # Build updates dict, excluding None values
    updates = {k: v for k, v in body.model_dump().items() if v is not None}

    if not updates:
        raise HTTPException(status_code=400, detail="No updates provided")

    updated = persona_manager.update_custom(persona_id, updates)
    if not updated:
        raise HTTPException(
            status_code=404,
            detail=f"Persona not found or is predefined: {persona_id}"
        )

    return {"persona": updated.to_dict()}


@router.delete("/personas/{persona_id}")
async def delete_persona(request: Request, persona_id: str):
    """Delete a custom persona."""
    persona_manager = request.app.state.persona_manager

    success = persona_manager.delete_custom(persona_id)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Persona not found or is predefined: {persona_id}"
        )

    return {"success": True}


# =============================================================================
# Conversation Endpoints
# =============================================================================

@router.get("/conversations")
async def list_conversations(request: Request):
    """List saved conversations."""
    conversation_store = request.app.state.conversation_store

    return {
        "conversations": [
            m.to_dict() for m in conversation_store.list_all(saved_only=True)
        ]
    }


@router.post("/conversations/save")
async def save_conversation(request: Request, body: SaveConversationRequest):
    """Save a regular chat conversation (creates new saved conversation with messages)."""
    conversation_store = request.app.state.conversation_store

    conv_id = conversation_store.create_saved(
        messages=body.messages,
        name=body.name,
        persona_id=body.persona_id,
        model_id=body.model_id,
        conv_type=body.conv_type,
    )

    return {"success": True, "conversation_id": conv_id}


@router.get("/conversations/{conv_id}")
async def get_conversation(request: Request, conv_id: str):
    """Get a conversation with all messages."""
    conversation_store = request.app.state.conversation_store

    conv = conversation_store.get(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail=f"Conversation not found: {conv_id}")

    return {"conversation": conv.to_dict()}


@router.post("/conversations/{conv_id}/rename")
async def rename_conversation(request: Request, conv_id: str, body: RenameConversationRequest):
    """Rename a conversation."""
    conversation_store = request.app.state.conversation_store

    success = conversation_store.rename(conv_id, body.name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Conversation not found: {conv_id}")

    return {"success": True}


@router.post("/conversations/{conv_id}/mark-saved")
async def mark_conversation_saved(request: Request, conv_id: str, body: RenameConversationRequest):
    """Mark an existing agent conversation as saved."""
    conversation_store = request.app.state.conversation_store

    success = conversation_store.mark_saved(conv_id, body.name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Conversation not found: {conv_id}")

    return {"success": True}


@router.post("/conversations/{conv_id}/persona")
async def set_conversation_persona(request: Request, conv_id: str, body: SetPersonaRequest):
    """Change the persona for a conversation."""
    conversation_store = request.app.state.conversation_store
    persona_manager = request.app.state.persona_manager

    # Validate persona exists
    persona = persona_manager.get(body.persona_id)
    if not persona:
        raise HTTPException(status_code=400, detail=f"Persona not found: {body.persona_id}")

    success = conversation_store.set_persona(conv_id, body.persona_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Conversation not found: {conv_id}")

    return {"success": True}


@router.delete("/conversations/{conv_id}")
async def delete_conversation(request: Request, conv_id: str):
    """Delete a conversation."""
    conversation_store = request.app.state.conversation_store

    success = conversation_store.delete(conv_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Conversation not found: {conv_id}")

    return {"success": True}


@router.post("/conversations/batch-delete")
async def batch_delete_conversations(request: Request):
    """Delete conversations matching criteria. Body: {filter: 'untitled'|'all', conv_type: 'agent'|'chat'|null}"""
    conversation_store = request.app.state.conversation_store
    body = await request.json()
    filter_type = body.get("filter", "untitled")
    conv_type = body.get("conv_type")

    all_convos = conversation_store.list_all(saved_only=True)
    deleted = 0
    for conv in all_convos:
        should_delete = False
        if filter_type == "untitled" and (conv.name or "Untitled") == "Untitled":
            should_delete = True
        elif filter_type == "all":
            should_delete = True

        if conv_type and conv.conv_type != conv_type:
            should_delete = False

        if should_delete:
            if conversation_store.delete(conv.id):
                deleted += 1

    return {"success": True, "deleted": deleted}


# =============================================================================
# Agent Chat Endpoints
# =============================================================================

def _build_system_prompt(persona, dynamic_state: str, tools_prompt: str,
                         additional_instructions: str = None,
                         working_memory=None,
                         agent_memory_section: str = None) -> str:
    """Build the full system prompt for a persona.

    The prompt is layered:
    1. Persona's core system prompt (required - has tool calling instructions)
    2. Persistent agent memory (cross-conversation facts)
    3. Working memory (task progress tracking)
    4. Dynamic system state (if persona.include_system_state)
    5. Available tools list
    6. User's additional instructions (optional - tone, style, focus)
    """
    sections = [persona.system_prompt]

    # Inject persistent memory if available
    if agent_memory_section:
        sections.append(agent_memory_section)

    # Inject working memory if it has content
    if working_memory and working_memory.has_content():
        sections.append(working_memory.to_prompt_section())

    if dynamic_state:
        sections.append(dynamic_state)

    if tools_prompt:
        sections.append(tools_prompt)

    # Add user's additional instructions at the end (layered on top)
    if additional_instructions and additional_instructions.strip():
        sections.append(f"## Additional Instructions\n{additional_instructions.strip()}")

    full_prompt = "\n\n".join(sections)
    logger.info(f"Built system prompt for persona '{persona.id}' ({len(full_prompt)} chars, working_memory={'yes' if working_memory and working_memory.has_content() else 'no'}, additional_instructions={'yes' if additional_instructions else 'no'})")
    logger.debug(f"System prompt preview: {full_prompt[:500]}...")
    return full_prompt


@router.get("/agent/debug")
async def agent_debug(request: Request, persona_id: str = "system_agent"):
    """Debug endpoint to see what system prompt would be built."""
    persona_manager = request.app.state.persona_manager
    tool_router = request.app.state.tool_router

    persona = persona_manager.get(persona_id)
    if not persona:
        return {
            "error": f"Persona not found: {persona_id}",
            "available": [p.id for p in persona_manager.list_all()]
        }

    dynamic_state = await tool_router.build_dynamic_prompt(persona)
    tools_prompt = tool_router.get_tools_prompt_for_persona(persona)
    system_prompt = _build_system_prompt(persona, dynamic_state, tools_prompt)

    return {
        "persona_id": persona.id,
        "persona_name": persona.name,
        "persona_tools": persona.tools,
        "include_system_state": persona.include_system_state,
        "dynamic_state_length": len(dynamic_state),
        "tools_prompt_length": len(tools_prompt),
        "system_prompt_length": len(system_prompt),
        "system_prompt_preview": system_prompt[:2000] + "..." if len(system_prompt) > 2000 else system_prompt,
        "tools_found": [line.strip("## ") for line in tools_prompt.split("\n") if line.startswith("## ")]
    }


def _detect_tool_name(response_text: str, tool_router) -> Optional[str]:
    """Quick regex to extract tool name from LLM response without executing.
    Returns the tool name even if unknown (for progress events and logging)."""
    patterns = [
        r'<tool_call>\s*\{[^}]*"tool"\s*:\s*"([^"]+)"',
        r'```json\s*\{[^}]*"tool"\s*:\s*"([^"]+)"',
        r'\{\s*"tool"\s*:\s*"([^"]+)"',
    ]
    for pattern in patterns:
        match = _re.search(pattern, response_text, _re.DOTALL)
        if match:
            name = match.group(1)
            return name  # Return any detected name (even unknown) for visibility
    return None


def _select_auto_persona(message: str, persona_manager) -> str:
    """
    Resolve the 'auto' persona to a concrete persona ID using lightweight intent heuristics.

    Keeps default UX simple (no persona picker required) while still steering weaker
    models toward smaller, domain-specific tool sets.
    """
    text = (message or "").lower()

    # Pure conversational requests should stay in a lightweight chat persona.
    simple_chat_patterns = [
        r"^\s*(hi|hello|hey|thanks|thank you|ok|okay|cool|nice|bye)\b",
        r"^\s*(who|what) are you\b",
        r"^\s*help\b",
    ]
    if any(_re.match(p, text) for p in simple_chat_patterns):
        if persona_manager.get("general_assistant"):
            return "general_assistant"

    domain_tokens = {
        "image_creator": [
            "comfy", "sdxl", "stable diffusion", "flux", "lora", "checkpoint",
            "image", "img2img", "txt2img", "inpaint", "upscale", "video generation",
        ],
        "automator": [
            "n8n", "workflow", "automation", "webhook", "trigger",
            "deploy workflow", "schedule", "credential",
        ],
        "coder": [
            "code", "python", "javascript", "script", "debug",
            "refactor", "function", "class", "stack trace",
        ],
        "researcher": [
            "research", "search", "find sources", "latest", "news",
            "compare", "summarize", "web",
        ],
    }

    # Score domains by token hits; multi-word phrases get higher weight.
    scores: Dict[str, int] = {}
    for persona_id, tokens in domain_tokens.items():
        score = 0
        for tok in tokens:
            if tok in text:
                score += 2 if " " in tok else 1
        if score > 0:
            scores[persona_id] = score

    if not scores:
        return "system_agent"

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_persona, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0

    # Conservative routing: require decent signal and clear lead.
    if top_score < 2 or (top_score - second_score) < 1:
        return "system_agent"

    if persona_manager.get(top_persona):
        return top_persona

    # If selected persona is unavailable, fall back safely.
    return "system_agent"


@router.post("/agent/stream")
async def agent_chat_stream(request: Request, body: AgentChatRequest):
    """
    Streaming version of agent chat - sends tokens as they're generated.
    Uses Server-Sent Events (SSE) format.
    """
    from backend.tools import ToolRouter
    from providers.base import InferenceRequest, ChatMessage

    orchestrator = request.app.state.orchestrator
    persona_manager = request.app.state.persona_manager
    conversation_store = request.app.state.conversation_store

    # Agent streams need isolated ToolRouter (concurrent sessions share _agent_context)
    tool_router = ToolRouter(
        orchestrator, request.app.state.n8n_manager, request.app.state.settings,
        getattr(request.app.state, 'comfyui_manager', None),
        getattr(request.app.state, 'media_catalog', None),
        comfyui_pool=getattr(request.app.state, 'comfyui_pool', None)
    )

    requested_persona_id = (body.persona_id or "auto").strip()
    resolved_persona_id = _select_auto_persona(body.message, persona_manager) \
        if requested_persona_id == "auto" else requested_persona_id

    # Get persona
    persona = persona_manager.get(resolved_persona_id)
    if not persona:
        logger.error(f"[stream] Persona not found: {resolved_persona_id}")
        async def error_gen():
            yield f"data: {json.dumps({'type': 'error', 'error': f'Persona not found: {resolved_persona_id}'})}\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    logger.info(
        f"[stream] Using persona: {persona.id} (requested={requested_persona_id}, "
        f"tools={persona.tools}, include_state={persona.include_system_state}, autonomous={body.autonomous})"
    )

    # Resolve and enforce persona tool allowlist server-side.
    allowed_tools = persona_manager.get_tools_for_persona(persona, tool_router.get_tool_list())
    tool_router.set_allowed_tools(allowed_tools if persona.tools else [])

    # Get or create conversation
    conv_id = body.conversation_id
    if not conv_id:
        conv_id = conversation_store.create(
            persona_id=persona.id,
            model_id=body.instance_id,
        )
    else:
        conv = conversation_store.get(conv_id)
        if not conv:
            conv_id = conversation_store.create(
                persona_id=persona.id,
                model_id=body.instance_id,
            )

    # Get model to use
    requested_instance_id = body.instance_id
    instances = orchestrator.get_loaded_instances()
    if not instances:
        async def error_gen():
            yield f"data: {json.dumps({'type': 'error', 'error': 'No models loaded. Load a model first.'})}\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    # Head agent model selection.
    # When pin_head_to_openrouter is ON, force an OpenRouter model (legacy behavior).
    # When OFF (default), any model can serve as head agent.
    pin_to_openrouter = False
    app_settings = request.app.state.settings
    if app_settings and hasattr(app_settings, 'get'):
        try:
            pin_to_openrouter = bool(app_settings.get("agent.pin_head_to_openrouter", False))
        except Exception:
            pin_to_openrouter = False

    head_model_forced = False
    if pin_to_openrouter:
        # Legacy behavior: force OpenRouter
        openrouter_instances = [
            inst for inst in instances
            if getattr(getattr(inst, "provider_type", None), "value", "").lower() == "openrouter"
        ]
        if not openrouter_instances:
            async def error_gen():
                yield f"data: {json.dumps({'type': 'error', 'error': 'Head agent is pinned to OpenRouter but none loaded. Load an OpenRouter model or disable pin_head_to_openrouter in settings.'})}\n\n"
            return StreamingResponse(error_gen(), media_type="text/event-stream")

        if requested_instance_id:
            requested_inst = orchestrator.get_instance(requested_instance_id)
            requested_provider = getattr(getattr(requested_inst, "provider_type", None), "value", "").lower() if requested_inst else ""
            if requested_provider == "openrouter":
                instance_id = requested_instance_id
            else:
                instance_id = openrouter_instances[0].id
                head_model_forced = True
                logger.info(f"[stream] Overriding requested non-OpenRouter head model '{requested_instance_id}' -> '{instance_id}'")
        else:
            instance_id = openrouter_instances[0].id
    else:
        # Default: use the requested model (sidebar selection) or first loaded model
        if requested_instance_id and orchestrator.get_instance(requested_instance_id):
            instance_id = requested_instance_id
        else:
            instance_id = instances[0].id

    # Guardrail: clamp autonomous tool-call budget to prevent runaway sessions.
    max_tool_calls = _clamp_max_tool_calls(body.max_tool_calls)
    if max_tool_calls != body.max_tool_calls:
        logger.info(f"[stream] Clamped max_tool_calls from {body.max_tool_calls} to {max_tool_calls}")

    # Update conversation with model ID
    conversation_store.set_model(conv_id, instance_id)

    # Inject agent context for sub-agent spawning
    tool_router.set_agent_context(
        instance_id,
        conversation_store,
        persona_manager,
        parent_conv_id=conv_id,
        persona_id=persona.id,
    )

    # Per-panel routing preset override
    if body.routing_preset_id:
        tool_router.set_routing_preset_override(body.routing_preset_id)

    # Tools that change system state and require prompt rebuild
    SYSTEM_CHANGING_TOOLS = {
        "load_model", "unload_model", "spawn_n8n", "stop_n8n", "quick_setup",
        "comfyui_start_api", "comfyui_stop_api", "comfyui_install",
        "comfyui_add_instance", "comfyui_start_instance", "comfyui_stop_instance",
        "provision_models", "load_from_preset", "flash_workflow",
        "configure_workflow",
    }

    async def generate():
        from backend.agent_intelligence import (
            WorkingMemory, AgentPlan, RetryState, ErrorType,
            needs_planning, generate_plan, select_tool_categories,
            categorize_error, build_retry_prompt,
            detect_thinking_content, is_likely_thinking_model,
            should_summarize, summarize_old_messages,
            truncate_tool_result, update_working_memory,
        )
        from backend.personas import CATEGORY_DESCRIPTIONS

        # ---- STEP 1: INIT ----
        conv_metadata = conversation_store.get_metadata(conv_id)
        memory = WorkingMemory.from_dict(conv_metadata.get("working_memory"))
        retry_state = RetryState()

        # Detect thinking model (cache in metadata, use model name not UUID)
        is_thinking = conv_metadata.get("is_thinking_model")
        if is_thinking is None:
            model_name = instance_id or ""
            inst = orchestrator.get_instance(instance_id) if instance_id else None
            if inst:
                model_name = inst.model_identifier or inst.display_name or instance_id
            is_thinking = is_likely_thinking_model(model_name)
            conversation_store.set_metadata(conv_id, "is_thinking_model", is_thinking)

        # Build dynamic system prompt with current state
        dynamic_state = await tool_router.build_dynamic_prompt(persona)
        tools_prompt = tool_router.get_tools_prompt_for_persona(persona)

        # ---- STEP 2: PLANNING ----
        plan = None
        has_all_tools = "all" in persona.tools if persona.tools else False

        if persona.tools and needs_planning(body.message):
            category_list = ", ".join(CATEGORY_DESCRIPTIONS.keys())
            plan = await generate_plan(
                orchestrator, instance_id, body.message,
                persona, memory, category_list
            )

            # Send plan to frontend
            yield f"data: {json.dumps({'type': 'plan', 'plan': plan.to_dict()})}\n\n"

            # Update working memory with plan
            memory = update_working_memory(memory, message=body.message, plan=plan)

        # ---- STEP 3: TOOL SELECTION (narrow for "all" tool personas) ----
        if has_all_tools and persona.tools:
            cached_categories = conv_metadata.get("selected_categories")

            if cached_categories:
                categories = cached_categories
            elif plan and plan.selected_categories:
                categories = plan.selected_categories
            else:
                categories = await select_tool_categories(
                    orchestrator, instance_id, body.message, plan, CATEGORY_DESCRIPTIONS
                )

            # Always include system + utility as baseline
            for base_cat in ["system", "utility"]:
                if base_cat not in categories:
                    categories.append(base_cat)

            # Cache for follow-up turns
            conversation_store.set_metadata(conv_id, "selected_categories", categories)

            # Get narrowed tools prompt
            tools_prompt = tool_router.get_tools_prompt_for_categories(persona, categories)

        # ---- STEP 4: BUILD SYSTEM PROMPT ----
        logger.info(f"[stream] Dynamic state: {len(dynamic_state)} chars, Tools: {len(tools_prompt)} chars")
        agent_memory_section = tool_router.agent_tools.memory.get_prompt_section(5)

        # Inject model assignment guidance when multiple models are loaded
        combined_instructions = body.additional_instructions or ""
        if len(instances) > 1:
            model_guidance = (
                "\n\n## Model Assignment for Sub-Agents\n"
                "Multiple models are loaded. When spawning sub-agents, assign the best "
                "model for each task using the instance_id parameter:\n"
                "- Code/technical tasks → prefer local models (fast, free)\n"
                "- Complex reasoning → prefer cloud/larger models\n"
                "- Simple lookups/tool calls → prefer smallest/fastest model\n"
                "- Use list_loaded_models if you need to check what's available.\n"
                "If unsure, omit instance_id and the system will auto-assign."
            )
            combined_instructions = (combined_instructions + model_guidance).strip()

        system_prompt = _build_system_prompt(
            persona, dynamic_state, tools_prompt,
            combined_instructions if combined_instructions else None,
            working_memory=memory,
            agent_memory_section=agent_memory_section
        )

        # Add user message to conversation
        conversation_store.append_message(conv_id, "user", body.message)

        # Send conversation ID and resolved persona first.
        yield f"data: {json.dumps({'type': 'conv_id', 'conversation_id': conv_id, 'persona_id': persona.id, 'requested_persona_id': requested_persona_id, 'resolved_persona_id': persona.id, 'autonomous': body.autonomous, 'instance_id': instance_id, 'head_model_forced': head_model_forced, 'max_tool_calls': max_tool_calls})}\n\n"

        # Register abort signal if provided
        abort_signals = getattr(request.app.state, 'abort_signals', {})
        if body.abort_id:
            # Preserve pre-existing abort requests set before registration.
            abort_signals.setdefault(body.abort_id, "running")

        # Smart delegation: simple status/info queries run on head directly.
        # Complex/creative tasks get delegated to a worker agent.
        delegate_enabled = True
        if app_settings and hasattr(app_settings, 'get'):
            try:
                delegate_enabled = bool(app_settings.get("agent.delegate_all", True))
            except Exception:
                delegate_enabled = True

        should_delegate = delegate_enabled and _should_delegate(body.message)
        logger.info(f"[stream] Delegation decision: {should_delegate} (enabled={delegate_enabled}, msg={body.message[:80]})")

        if should_delegate:
            preflight = await _build_worker_preflight_context(tool_router, body.message)
            worker_task = body.message if not preflight else (
                f"{body.message}\n\n{preflight}"
            )

            # Single worker — tool-level racing happens inside the worker
            # via race_executor.py when a raceable tool is called.
            spawn = await tool_router.agent_tools.spawn_agent(
                task=worker_task,
                persona_id=persona.id,
                name=None,
                max_tool_calls=max_tool_calls,
                _instance_id=instance_id,
                _conversation_store=conversation_store,
                _persona_manager=persona_manager,
                _routing_preset_id=body.routing_preset_id,
                _parent_conv_id=conv_id,
            )

            if not spawn.get("success"):
                spawn_err = spawn.get("error", "unknown error")
                err_payload = {"type": "error", "error": f"Failed to spawn worker: {spawn_err}"}
                yield f"data: {json.dumps(err_payload)}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'tool_calls_total': 0})}\n\n"
                return

            delegated_msg = "Working on it..."
            conversation_store.append_message(conv_id, "assistant", delegated_msg)

            # Emit initial worker status immediately for UI tab creation.
            scoped_agents = tool_router.agent_tools.get_all_agent_status(parent_conv_id=conv_id)
            if scoped_agents:
                yield f"data: {json.dumps({'type': 'sub_agent_update', 'agents': scoped_agents})}\n\n"

            yield f"data: {json.dumps({'type': 'token', 'text': delegated_msg})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'tool_calls_total': 0, 'delegated': True})}\n\n"
            return

        # Track tool calls
        tool_call_count = 0
        recent_tool_calls = []  # Track (tool_name, args_hash) for loop detection
        loop_warning_count = 0  # Hard-stop after 2 loop warnings
        aborted_session = False
        last_sub_agent_status_hash = None
        last_supervisor_tick = 0.0
        supervisor_interval_s = 20.0

        # Helper: emit sub-agent progress updates
        def _drain_sub_agent_events():
            """Yield SSE events for any sub-agent progress updates."""
            nonlocal last_sub_agent_status_hash
            events = tool_router.agent_tools.drain_events(parent_conv_id=conv_id)
            scoped_agents = tool_router.agent_tools.get_all_agent_status(parent_conv_id=conv_id)
            payload = {"type": "sub_agent_update"}
            has_payload = False
            if scoped_agents:
                status_hash = hash(json.dumps(scoped_agents, sort_keys=True, default=str))
                if status_hash != last_sub_agent_status_hash:
                    last_sub_agent_status_hash = status_hash
                    payload["agents"] = scoped_agents
                    has_payload = True
            if events:
                payload["events"] = events
                has_payload = True
            if has_payload:
                return f"data: {json.dumps(payload)}\n\n"
            return None

        async def _supervisor_tick_if_due(force: bool = False):
            nonlocal last_supervisor_tick
            now = time.time()
            if not force and (now - last_supervisor_tick) < supervisor_interval_s:
                return None
            last_supervisor_tick = now
            if not body.autonomous:
                return None
            summary = None
            try:
                sup = await tool_router.agent_tools.supervise_workers(
                    parent_conv_id=conv_id,
                    head_instance_id=instance_id,
                )
                if sup and sup.get("running_workers", 0) > 0:
                    summary = {
                        "running_workers": sup.get("running_workers", 0),
                        "nudged": len(sup.get("nudged_workers", [])),
                    }
            except Exception as e:
                logger.debug(f"[stream] Supervisor tick skipped: {e}")
            if summary:
                return f"data: {json.dumps({'type': 'head_heartbeat', **summary})}\n\n"
            return None

        # ---- STEP 5: EXECUTION LOOP ----
        try:
          while True:
            hb = await _supervisor_tick_if_due()
            if hb:
                yield hb

            # Emit sub-agent progress updates
            sub_update = _drain_sub_agent_events()
            if sub_update:
                yield sub_update

            # Check for abort signal
            if body.abort_id and abort_signals.get(body.abort_id) == "abort":
                logger.info(f"[stream] Agent aborted by user (signal: {body.abort_id})")
                # Abort any running sub-agents
                tool_router.agent_tools.abort_all(parent_conv_id=conv_id)
                aborted_session = True
                yield f"data: {json.dumps({'type': 'aborted', 'tool_calls_total': tool_call_count})}\n\n"
                break

            # 5a. Context check: summarize if conversation is too long
            recent_messages = conversation_store.get_messages(conv_id, limit=20)
            if should_summarize(recent_messages):
                # Check if we already have a cached summary that covers most of these messages
                cached_summary = conv_metadata.get("context_summary")
                msg_count = len(conversation_store.get_messages(conv_id, limit=0))
                last_summarized_at = conv_metadata.get("summarized_at_count", 0)

                if not cached_summary or msg_count - last_summarized_at >= 6:
                    logger.info(f"[stream] Context too large ({msg_count} msgs), summarizing older messages")
                    recent_messages = await summarize_old_messages(
                        orchestrator, instance_id, recent_messages, keep=6
                    )
                    # Cache summary to avoid re-triggering on next loop iteration
                    if recent_messages and recent_messages[0].get("role") == "system":
                        conversation_store.update_metadata(conv_id, {
                            "context_summary": recent_messages[0]["content"],
                            "summarized_at_count": msg_count,
                        })
                        conv_metadata["context_summary"] = recent_messages[0]["content"]
                        conv_metadata["summarized_at_count"] = msg_count
                elif cached_summary:
                    # Reuse cached summary instead of re-calling LLM
                    keep_msgs = recent_messages[-6:] if len(recent_messages) > 6 else recent_messages
                    recent_messages = [{"role": "system", "content": cached_summary}] + keep_msgs

            # Build messages
            messages = [
                ChatMessage(role="system", content=system_prompt)
            ] + [
                ChatMessage(role=m["role"], content=m["content"])
                for m in recent_messages
            ]

            # 5b. Generate response
            user_params = body.params or {}
            # Floor max_tokens at 2048 for agent mode — tool call JSON can be large
            agent_max_tokens = max(user_params.get('max_tokens', 4096), 2048)
            inference_request = InferenceRequest(
                messages=messages,
                max_tokens=agent_max_tokens,
                temperature=user_params.get('temperature', persona.temperature),
                top_p=user_params.get('top_p', 0.95),
                top_k=user_params.get('top_k', 40),
                repeat_penalty=user_params.get('repeat_penalty', 1.1),
            )

            full_response = ""
            generation_aborted = False
            async for response in orchestrator.chat(instance_id, inference_request):
                # Check abort mid-generation
                if body.abort_id and abort_signals.get(body.abort_id) == "abort":
                    logger.info(f"[stream] Abort detected during LLM generation")
                    generation_aborted = True
                    break
                if response.text:
                    full_response += response.text
                    yield f"data: {json.dumps({'type': 'token', 'text': response.text})}\n\n"
                # Drain worker events during inference to keep worker streaming smooth
                sub_update = _drain_sub_agent_events()
                if sub_update:
                    yield sub_update

            if generation_aborted:
                tool_router.agent_tools.abort_all(parent_conv_id=conv_id)
                aborted_session = True
                yield f"data: {json.dumps({'type': 'aborted', 'tool_calls_total': tool_call_count})}\n\n"
                break

            # 5c. Thinking model detection
            if is_thinking:
                thinking_content, clean_response = detect_thinking_content(full_response)
                if thinking_content:
                    yield f"data: {json.dumps({'type': 'thinking', 'content': thinking_content})}\n\n"
                    full_response = clean_response

            # 5d. Check if response contains a tool call (only if persona has tools)
            tool_result = None
            tool_aborted = False
            if persona.tools:
                # Log full response for debugging tool call detection
                logger.info(f"[stream] LLM response length: {len(full_response)} chars, first 500: {full_response[:500]}")
                # Detect tool name early for progress events
                detected_tool = _detect_tool_name(full_response, tool_router)

                # Run tool as a task so we can poll abort signal + yield progress
                tool_task = asyncio.create_task(tool_router.parse_and_execute(full_response))
                tool_start = time.time()

                while not tool_task.done():
                    hb = await _supervisor_tick_if_due()
                    if hb:
                        yield hb

                    # Check abort signal
                    if body.abort_id and abort_signals.get(body.abort_id) == "abort":
                        tool_task.cancel()
                        try:
                            await tool_task
                        except (asyncio.CancelledError, Exception):
                            pass
                        tool_aborted = True
                        logger.info(f"[stream] Agent aborted during tool execution: {detected_tool}")
                        tool_router.agent_tools.abort_all(parent_conv_id=conv_id)
                        aborted_session = True
                        yield f"data: {json.dumps({'type': 'aborted', 'tool_calls_total': tool_call_count})}\n\n"
                        break

                    # Drain sub-agent events
                    sub_update = _drain_sub_agent_events()
                    if sub_update:
                        yield sub_update

                    # Yield progress event only when we can identify a tool name.
                    if detected_tool:
                        elapsed = round(time.time() - tool_start, 1)
                        yield f"data: {json.dumps({'type': 'tool_executing', 'tool': detected_tool, 'elapsed': elapsed})}\n\n"

                    await asyncio.sleep(1.0)

                if not tool_aborted and tool_task.done():
                    try:
                        tool_result = tool_task.result()
                    except Exception as e:
                        logger.error(f"Tool task failed: {e}")
                        tool_result = None

            if tool_aborted:
                break

            if tool_result:
                tool_call_count += 1
                tool_name = tool_result.get('tool', 'unknown')
                tool_args = tool_result.get('arguments', {})
                tool_exec_result = tool_result.get('result', {})
                tool_success = tool_exec_result.get('success', True) if isinstance(tool_exec_result, dict) else True
                logger.info(f"[stream/autonomous] Tool call {tool_call_count}/{max_tool_calls}: {tool_name} (success={tool_success})")

                # Loop detection: track recent tool calls and detect repeated patterns
                # Polling tools are legitimate to call repeatedly (they check async job status)
                POLLING_TOOLS = {
                    "comfyui_get_result", "comfyui_await_result", "comfyui_await_job",
                    "comfyui_job_status", "gguf_download_status",
                }
                args_hash = hash(json.dumps(tool_args, sort_keys=True, default=str))
                recent_tool_calls.append((tool_name, args_hash))
                is_looping = False
                # Check 1: 3 consecutive identical calls (same tool + same args)
                # Skip for polling tools — repeated status checks are expected behavior
                if len(recent_tool_calls) >= 3 and tool_name not in POLLING_TOOLS:
                    last_3 = recent_tool_calls[-3:]
                    if last_3[0] == last_3[1] == last_3[2]:
                        is_looping = True
                        logger.warning(f"[stream] Loop detected: {tool_name} called 3+ times with same args")
                # Check 2: Ping-pong/cycling pattern — last 6 calls use ≤2 unique tools
                if not is_looping and len(recent_tool_calls) >= 6:
                    last_6_names = [t[0] for t in recent_tool_calls[-6:]]
                    if len(set(last_6_names)) <= 2:
                        is_looping = True
                        logger.warning(f"[stream] Ping-pong loop detected: last 6 calls cycle between {set(last_6_names)}")
                # Check 3: Saturation — last 8 calls use ≤3 unique tools
                if not is_looping and len(recent_tool_calls) >= 8:
                    last_8_names = [t[0] for t in recent_tool_calls[-8:]]
                    if len(set(last_8_names)) <= 3:
                        is_looping = True
                        logger.warning(f"[stream] Saturation loop detected: last 8 calls cycle between {set(last_8_names)}")
                if is_looping:
                    loop_warning_count += 1

                # Add assistant's tool call to conversation
                conversation_store.append_message(conv_id, "assistant", full_response)

                # ---- ASK USER: pause and wait for user response ----
                if tool_name == "ask_user" and body.autonomous and body.abort_id:
                    question = tool_exec_result.get("question", "")
                    options = tool_exec_result.get("options", [])

                    # Yield ask_user event to frontend
                    yield f"data: {json.dumps({'type': 'ask_user', 'question': question, 'options': options, 'tool_number': tool_call_count})}\n\n"

                    # Set up wait mechanism
                    wait_event = asyncio.Event()
                    if not hasattr(request.app.state, 'agent_responses'):
                        request.app.state.agent_responses = {}
                    request.app.state.agent_responses[body.abort_id] = {
                        "event": wait_event,
                        "response": None,
                    }

                    # Poll: wait for user response or abort (check every 0.5s)
                    aborted_during_ask = False
                    while not wait_event.is_set():
                        hb = await _supervisor_tick_if_due()
                        if hb:
                            yield hb

                        sub_update = _drain_sub_agent_events()
                        if sub_update:
                            yield sub_update

                        if abort_signals.get(body.abort_id) == "abort":
                            aborted_during_ask = True
                            break
                        await asyncio.sleep(0.5)

                    # Clean up wait state
                    resp_data = request.app.state.agent_responses.pop(body.abort_id, {})

                    if aborted_during_ask:
                        logger.info(f"[stream] Agent aborted during ask_user (signal: {body.abort_id})")
                        tool_router.agent_tools.abort_all(parent_conv_id=conv_id)
                        aborted_session = True
                        yield f"data: {json.dumps({'type': 'aborted', 'tool_calls_total': tool_call_count})}\n\n"
                        break

                    # Got user response — feed it back as continuation
                    user_answer = resp_data.get("response", "No response received")
                    logger.info(f"[stream] ask_user response received: {user_answer[:100]}")

                    memory = update_working_memory(
                        memory, tool_name="ask_user",
                        tool_result={"question": question, "user_response": user_answer}
                    )

                    conversation_store.append_message(
                        conv_id, "user",
                        f"User responded to your question \"{question}\": {user_answer}"
                    )
                    conversation_store.set_metadata(conv_id, "working_memory", memory.to_dict())

                    yield f"data: {json.dumps({'type': 'continuing', 'tool_calls_so_far': tool_call_count})}\n\n"
                    sub_update = _drain_sub_agent_events()
                    if sub_update:
                        yield sub_update
                    continue

                # Send tool call info
                yield f"data: {json.dumps({'type': 'tool_call', 'tool': tool_name, 'tool_number': tool_call_count, 'arguments': tool_result.get('arguments', {}), 'result': tool_exec_result})}\n\n"

                # 5e. Result handling: truncate before storing
                result_str = json.dumps(tool_exec_result if isinstance(tool_exec_result, dict) else {"result": tool_exec_result}, indent=2)
                result_str = truncate_tool_result(result_str)

                # Update working memory
                memory = update_working_memory(
                    memory, tool_name=tool_name, tool_result=tool_exec_result
                )

                # Refresh system prompt if tool changed system state or memory
                if tool_name in SYSTEM_CHANGING_TOOLS or tool_name == "remember":
                    logger.info(f"[stream] Tool '{tool_name}' changed state, rebuilding prompt")
                    dynamic_state = await tool_router.build_dynamic_prompt(persona)
                    agent_memory_section = tool_router.agent_tools.memory.get_prompt_section(5)
                    system_prompt = _build_system_prompt(
                        persona, dynamic_state, tools_prompt,
                        body.additional_instructions, working_memory=memory,
                        agent_memory_section=agent_memory_section
                    )

                # Hard stop: if looping detected 2+ times, force final summary
                if is_looping and loop_warning_count >= 2:
                    logger.warning(f"[stream] Hard-stopping agent after {loop_warning_count} loop warnings ({tool_call_count} tool calls)")
                    tool_call_count = max_tool_calls  # Force into final summary branch

                if body.autonomous and tool_call_count < max_tool_calls:
                    # Autonomous continuation with smart retry logic
                    if not tool_success:
                        error_msg = tool_exec_result.get('error', 'Unknown error') if isinstance(tool_exec_result, dict) else str(tool_exec_result)
                        error_type = categorize_error(tool_name, error_msg)
                        retry_state.record_attempt(tool_name, tool_result.get('arguments', {}), error_msg, error_type)

                        # Build smart retry prompt
                        continuation = (
                            f"Tool result:\n```json\n{result_str}\n```\n\n"
                            + build_retry_prompt(tool_name, error_msg, error_type, retry_state)
                        )
                    elif is_looping:
                        # Loop detected - force the agent to try something different
                        continuation = (
                            f"Tool result:\n```json\n{result_str}\n```\n\n"
                            f"WARNING: You are stuck in a loop, calling the same tools repeatedly. "
                            f"You MUST stop calling tools and provide a final summary NOW. "
                            f"Summarize ALL the information you've gathered from tool results so far. "
                            f"Do NOT make another tool call."
                        )
                    else:
                        continuation = (
                            f"Tool result:\n```json\n{result_str}\n```\n\n"
                            f"Tool call succeeded. If the task is complete, provide a clear summary of what was accomplished. "
                            f"If more steps are needed, execute the next tool call."
                        )
                    conversation_store.append_message(conv_id, "user", continuation)

                    # Save working memory periodically
                    conversation_store.set_metadata(conv_id, "working_memory", memory.to_dict())

                    yield f"data: {json.dumps({'type': 'continuing', 'tool_calls_so_far': tool_call_count})}\n\n"
                    # Emit sub-agent progress after each tool call
                    sub_update = _drain_sub_agent_events()
                    if sub_update:
                        yield sub_update
                    continue
                else:
                    # Final tool call - get follow-up summary
                    conversation_store.append_message(
                        conv_id, "user",
                        f"Tool result:\n```json\n{result_str}\n```\n\n"
                        f"Provide a clear summary of what was accomplished and the results."
                    )

                    # Get follow-up response with streaming
                    recent_messages = conversation_store.get_messages(conv_id, limit=20)
                    follow_up_messages = [
                        ChatMessage(role="system", content=system_prompt)
                    ] + [
                        ChatMessage(role=m["role"], content=m["content"])
                        for m in recent_messages
                    ]

                    follow_up_request = InferenceRequest(
                        messages=follow_up_messages,
                        max_tokens=user_params.get('max_tokens', 4096),
                        temperature=user_params.get('temperature', persona.temperature),
                    )

                    yield f"data: {json.dumps({'type': 'followup_start'})}\n\n"

                    follow_up = ""
                    followup_aborted = False
                    async for response in orchestrator.chat(instance_id, follow_up_request):
                        if body.abort_id and abort_signals.get(body.abort_id) == "abort":
                            logger.info(f"[stream] Abort detected during follow-up generation")
                            followup_aborted = True
                            break
                        if response.text:
                            follow_up += response.text
                            yield f"data: {json.dumps({'type': 'token', 'text': response.text})}\n\n"

                    if followup_aborted:
                        tool_router.agent_tools.abort_all(parent_conv_id=conv_id)
                        aborted_session = True
                        yield f"data: {json.dumps({'type': 'aborted', 'tool_calls_total': tool_call_count})}\n\n"
                        break
                    conversation_store.append_message(conv_id, "assistant", follow_up)
                    break

            else:
                # No tool call - we're done
                conversation_store.append_message(conv_id, "assistant", full_response)
                break

          # Final sub-agent status flush
          sub_update = _drain_sub_agent_events()
          if sub_update:
              yield sub_update

          # Signal completion with summary unless session was aborted.
          if not aborted_session:
              yield f"data: {json.dumps({'type': 'done', 'tool_calls_total': tool_call_count})}\n\n"

        finally:
            # Always cleanup, even on client disconnect
            conversation_store.set_metadata(conv_id, "working_memory", memory.to_dict())
            conversation_store.flush(conv_id)  # Write deferred changes to disk
            if body.abort_id:
                abort_signals.pop(body.abort_id, None)
                agent_responses = getattr(request.app.state, 'agent_responses', {})
                agent_responses.pop(body.abort_id, None)

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/agent/abort")
async def agent_abort(request: Request, body: dict):
    """Abort a running agent session by its abort_id."""
    abort_id = body.get("abort_id")
    if not abort_id:
        raise HTTPException(status_code=400, detail="abort_id is required")

    abort_signals = getattr(request.app.state, 'abort_signals', {})
    if abort_id in abort_signals:
        abort_signals[abort_id] = "abort"
        logger.info(f"Abort signal set for agent session: {abort_id}")
        return {"success": True, "message": f"Abort signal sent for {abort_id}"}
    else:
        # Signal might not be registered yet or already completed - set it anyway
        abort_signals[abort_id] = "abort"
        return {"success": True, "message": f"Abort signal set for {abort_id} (may not be active)"}


@router.post("/agent/respond")
async def agent_respond(request: Request, body: AgentRespondRequest):
    """Send a user response to a paused ask_user tool invocation."""
    agent_responses = getattr(request.app.state, 'agent_responses', {})

    entry = agent_responses.get(body.abort_id)
    if not entry:
        raise HTTPException(
            status_code=404,
            detail=f"No pending question for session {body.abort_id}"
        )

    # Store the response and signal the waiting generator
    entry["response"] = body.response
    entry["event"].set()
    logger.info(f"User response received for session {body.abort_id}: {body.response[:100]}")
    return {"success": True, "message": "Response delivered to agent"}


@router.get("/agent/workers/{conv_id}/stream")
async def agent_workers_stream(request: Request, conv_id: str):
    """SSE stream of real-time worker events for a conversation."""
    conversation_store = request.app.state.conversation_store
    if not conversation_store.get(conv_id):
        raise HTTPException(status_code=404, detail=f"Conversation not found: {conv_id}")

    tool_router = getattr(request.app.state, "tool_router", None)
    if tool_router is None:
        from backend.tools import ToolRouter
        tool_router = ToolRouter(
            request.app.state.orchestrator,
            request.app.state.n8n_manager,
            request.app.state.settings,
            getattr(request.app.state, 'comfyui_manager', None),
            media_catalog=getattr(request.app.state, 'media_catalog', None),
            comfyui_pool=getattr(request.app.state, 'comfyui_pool', None)
        )

    sub_queue = tool_router.agent_tools.subscribe_events(conv_id)

    async def event_generator():
        try:
            # Immediately emit current agent status so frontend can create tabs
            agents = tool_router.agent_tools.get_all_agent_status(parent_conv_id=conv_id)
            if agents:
                yield f"data: {json.dumps({'type': 'sub_agent_update', 'agents': agents})}\n\n"

            last_status_hash = None
            while True:
                # Wait for events with a timeout so we can send keepalives & check termination
                try:
                    event = await asyncio.wait_for(sub_queue.get(), timeout=2.0)
                    # Batch: drain any additional events that arrived
                    batch = [event]
                    while not sub_queue.empty():
                        try:
                            batch.append(sub_queue.get_nowait())
                        except asyncio.QueueEmpty:
                            break

                    # Check for agent status changes in batch
                    has_status_change = any(e.get("event") == "agent_done" for e in batch)

                    # Emit events
                    yield f"data: {json.dumps({'type': 'worker_events', 'events': batch})}\n\n"

                    # On status change, also send updated agent list
                    if has_status_change:
                        agents = tool_router.agent_tools.get_all_agent_status(parent_conv_id=conv_id)
                        yield f"data: {json.dumps({'type': 'sub_agent_update', 'agents': agents})}\n\n"

                        # Check if all workers are done
                        all_done = all(
                            a.get("status") in ("completed", "failed", "timeout", "aborted")
                            for a in agents
                        ) if agents else True
                        if all_done:
                            yield f"data: {json.dumps({'type': 'workers_done'})}\n\n"
                            return

                except asyncio.TimeoutError:
                    # No events in 2s — send keepalive and check if workers still running
                    agents = tool_router.agent_tools.get_all_agent_status(parent_conv_id=conv_id)
                    if not agents:
                        yield f"data: {json.dumps({'type': 'workers_done'})}\n\n"
                        return

                    status_hash = hash(json.dumps(agents, sort_keys=True, default=str))
                    if status_hash != last_status_hash:
                        last_status_hash = status_hash
                        yield f"data: {json.dumps({'type': 'sub_agent_update', 'agents': agents})}\n\n"

                    all_done = all(
                        a.get("status") in ("completed", "failed", "timeout", "aborted")
                        for a in agents
                    )
                    if all_done:
                        yield f"data: {json.dumps({'type': 'workers_done'})}\n\n"
                        return

                    # Keepalive comment
                    yield ": keepalive\n\n"

        except asyncio.CancelledError:
            pass
        finally:
            tool_router.agent_tools.unsubscribe_events(conv_id, sub_queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.get("/agent/workers/{conv_id}")
async def agent_workers_status(request: Request, conv_id: str):
    """
    Poll worker status for a parent/head conversation.
    Enables head-panel supervision updates even when the request stream is closed.
    """
    conversation_store = request.app.state.conversation_store
    if not conversation_store.get(conv_id):
        raise HTTPException(status_code=404, detail=f"Conversation not found: {conv_id}")

    tool_router = getattr(request.app.state, "tool_router", None)
    if tool_router is None:
        from backend.tools import ToolRouter
        tool_router = ToolRouter(
            request.app.state.orchestrator,
            request.app.state.n8n_manager,
            request.app.state.settings,
            getattr(request.app.state, 'comfyui_manager', None),
            media_catalog=getattr(request.app.state, 'media_catalog', None),
            comfyui_pool=getattr(request.app.state, 'comfyui_pool', None)
        )
    agents = tool_router.agent_tools.get_all_agent_status(parent_conv_id=conv_id)
    events = tool_router.agent_tools.drain_events(parent_conv_id=conv_id, max_events=400)
    running = sum(1 for a in agents if a.get("status") == "running")
    loaded_instances = []
    for inst in request.app.state.orchestrator.get_loaded_instances() or []:
        loaded_instances.append({
            "instance_id": inst.id,
            "model": getattr(inst, "display_name", None) or getattr(inst, "model_identifier", None) or inst.id,
            "provider": getattr(getattr(inst, "provider_type", None), "value", None),
            "gpu": getattr(inst, "gpu_index", None),
        })
    return {
        "success": True,
        "conversation_id": conv_id,
        "agents": agents,
        "events": events,
        "loaded_models": loaded_instances,
        "summary": {
            "total": len(agents),
            "running": running,
            "terminal": len(agents) - running,
        }
    }


@router.post("/agent/workers/{conv_id}/switch-model")
async def agent_switch_worker_model(request: Request, conv_id: str, body: AgentWorkerModelSwitchRequest):
    """Manually switch a worker to another loaded model instance."""
    conversation_store = request.app.state.conversation_store
    if not conversation_store.get(conv_id):
        raise HTTPException(status_code=404, detail=f"Conversation not found: {conv_id}")

    tool_router = getattr(request.app.state, "tool_router", None)
    if tool_router is None:
        tool_router = ToolRouter(
            request.app.state.orchestrator,
            request.app.state.n8n_manager,
            request.app.state.settings,
            getattr(request.app.state, 'comfyui_manager', None),
            media_catalog=getattr(request.app.state, 'media_catalog', None),
            comfyui_pool=getattr(request.app.state, 'comfyui_pool', None)
        )
    result = await tool_router.agent_tools.switch_worker_model(
        agent_id=body.agent_id,
        instance_id=body.instance_id,
        _parent_conv_id=conv_id,
    )
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to switch worker model"))
    return result
