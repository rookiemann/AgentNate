"""
OpenAI-Compatible API Endpoints

Exposes /v1/chat/completions and /v1/models in OpenAI format so that
external tools (n8n AI Agent nodes, LangChain, etc.) can use AgentNate's
loaded models with standard OpenAI client libraries.

Base URL: http://localhost:8000/v1
API Key: any value (local models, no auth needed)
"""

import json
import time
import uuid
import logging
from typing import List, Optional

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from providers.base import InferenceRequest, ChatMessage as ProviderChatMessage
from backend.utils.token_utils import estimate_messages_tokens, calculate_safe_max_tokens

logger = logging.getLogger("openai_compat")
router = APIRouter()


# ==================== Request/Response Models ====================

class OpenAIMessage(BaseModel):
    role: str
    content: str


class OpenAIChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[OpenAIMessage]
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0


# ==================== Helpers ====================

def _resolve_model(orchestrator, model_field: Optional[str]):
    """
    Resolve model field to (instance_id, display_name).

    Accepts: None/empty (first loaded), UUID, or name substring.
    Returns: (instance_id, display_name) or raises ValueError.
    """
    instances = orchestrator.get_loaded_instances()
    if not instances:
        raise ValueError("No models loaded")

    # Empty/None → first loaded
    if not model_field:
        inst = instances[0]
        return inst.id, inst.display_name or inst.model_identifier

    # Exact UUID match
    for inst in instances:
        if inst.id == model_field:
            return inst.id, inst.display_name or inst.model_identifier

    # Name substring match (case-insensitive)
    model_lower = model_field.lower()
    for inst in instances:
        name = (inst.display_name or "").lower()
        ident = (inst.model_identifier or "").lower()
        if model_lower in name or model_lower in ident:
            return inst.id, inst.display_name or inst.model_identifier

    raise ValueError(f"Model '{model_field}' not found. Loaded: {[i.display_name or i.model_identifier for i in instances]}")


def _gen_id():
    """Generate OpenAI-style completion ID."""
    return f"chatcmpl-{uuid.uuid4().hex[:12]}"


# ==================== Endpoints ====================

@router.get("/models")
async def list_models(request: Request):
    """List loaded models in OpenAI format."""
    orchestrator = request.app.state.orchestrator
    instances = orchestrator.get_loaded_instances()

    data = []
    for inst in instances:
        data.append({
            "id": inst.id,
            "object": "model",
            "created": int(inst.created_at),
            "owned_by": inst.provider_type.value,
            "name": inst.display_name or inst.model_identifier,
        })

    return {"object": "list", "data": data}


@router.get("/models/{model_id}")
async def get_model(request: Request, model_id: str):
    """Get a specific model by ID."""
    orchestrator = request.app.state.orchestrator

    try:
        instance_id, display_name = _resolve_model(orchestrator, model_id)
    except ValueError as e:
        return JSONResponse(status_code=404, content={"error": {"message": str(e), "type": "invalid_request_error"}})

    inst = orchestrator.get_instance(instance_id)
    return {
        "id": inst.id,
        "object": "model",
        "created": int(inst.created_at),
        "owned_by": inst.provider_type.value,
        "name": inst.display_name or inst.model_identifier,
    }


@router.post("/chat/completions")
async def chat_completions(request: Request, body: OpenAIChatRequest):
    """
    OpenAI-compatible chat completions.

    Supports both streaming (SSE) and non-streaming modes.
    Model field accepts: instance UUID, model name, or empty (first loaded).
    API key in Authorization header is accepted but ignored (local models).
    """
    orchestrator = request.app.state.orchestrator
    settings = request.app.state.settings

    # Resolve model
    try:
        instance_id, model_name = _resolve_model(orchestrator, body.model)
    except ValueError as e:
        return JSONResponse(
            status_code=404,
            content={"error": {"message": str(e), "type": "model_not_found"}},
        )

    # Convert messages
    messages = [
        ProviderChatMessage(role=m.role, content=m.content)
        for m in body.messages
    ]

    # Calculate safe max_tokens
    instance = orchestrator.get_instance(instance_id)
    context_length = instance.context_length if instance else 4096
    msg_dicts = [{"role": m.role, "content": m.content} for m in messages]
    input_tokens = estimate_messages_tokens(msg_dicts)
    requested_max = body.max_tokens or settings.get("inference.default_max_tokens", 1024)
    safe_max = calculate_safe_max_tokens(context_length, input_tokens, requested_max, 0.05)

    # Build inference request
    inference_req = InferenceRequest(
        request_id=f"oai-{uuid.uuid4().hex[:8]}",
        messages=messages,
        max_tokens=safe_max,
        temperature=body.temperature if body.temperature is not None else 0.7,
        top_p=body.top_p if body.top_p is not None else 0.95,
        presence_penalty=body.presence_penalty or 0.0,
        frequency_penalty=body.frequency_penalty or 0.0,
        stop=body.stop,
    )

    completion_id = _gen_id()
    created = int(time.time())

    if body.stream:
        return StreamingResponse(
            _stream_response(orchestrator, instance_id, inference_req, completion_id, created, model_name),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    else:
        return await _complete_response(orchestrator, instance_id, inference_req, completion_id, created, model_name)


async def _complete_response(orchestrator, instance_id, inference_req, completion_id, created, model_name):
    """Non-streaming: collect full response and return."""
    full_text = ""
    usage = {}

    try:
        async for response in orchestrator.chat(instance_id, inference_req):
            full_text += response.text
            if response.done and response.usage:
                usage = response.usage
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "server_error"}},
        )

    # Estimate tokens if provider didn't report them
    if not usage:
        msg_dicts = [{"role": m.role, "content": m.content} for m in inference_req.messages]
        prompt_tokens = estimate_messages_tokens(msg_dicts)
        completion_tokens = len(full_text) // 4 + 1
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": full_text},
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
    }


async def _stream_response(orchestrator, instance_id, inference_req, completion_id, created, model_name):
    """Streaming: yield SSE chunks in OpenAI format."""
    # First chunk with role
    first_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(first_chunk)}\n\n"

    try:
        async for response in orchestrator.chat(instance_id, inference_req):
            if response.text:
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"content": response.text}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
    except Exception as e:
        logger.error(f"Stream error: {e}")
        error_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"

    # Final chunk with finish_reason
    final_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"
