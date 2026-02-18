"""
Chat Routes

REST and WebSocket endpoints for chat inference with streaming.
Includes PDF upload and RAG (Retrieval-Augmented Generation) support.
"""

import json
import asyncio
import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from backend.utils.token_utils import estimate_messages_tokens, calculate_safe_max_tokens

logger = logging.getLogger("chat")
router = APIRouter()


# ==================== PDF RAG State ====================

# Import PDF/RAG utilities (lazy import to avoid startup errors)
_embedding_manager = None
_session_stores: Dict[str, Any] = {}  # session_id -> InMemoryVectorStore
_session_last_access: Dict[str, float] = {}  # session_id -> timestamp
_SESSION_TTL_SECONDS = 1800  # 30 minutes
_SESSION_CLEANUP_THRESHOLD = 10  # Cleanup when we have this many sessions


def _get_embedding_manager(settings: Dict[str, Any]):
    """Get or create the embedding manager singleton."""
    global _embedding_manager
    if _embedding_manager is None:
        from backend.utils.embedding_manager import EmbeddingManager
        _embedding_manager = EmbeddingManager(settings)
    return _embedding_manager


def _cleanup_expired_sessions():
    """Remove sessions that haven't been accessed recently."""
    import time
    now = time.time()
    expired = [
        sid for sid, last_access in _session_last_access.items()
        if now - last_access > _SESSION_TTL_SECONDS
    ]
    for sid in expired:
        if sid in _session_stores:
            store = _session_stores[sid]
            if hasattr(store, 'clear'):
                store.clear()
            del _session_stores[sid]
        _session_last_access.pop(sid, None)
    if expired:
        logger.info(f"Cleaned up {len(expired)} expired PDF sessions")


def _get_session_store(session_id: str):
    """Get or create a vector store for a session."""
    import time
    global _session_stores, _session_last_access

    # Periodic cleanup when we have many sessions
    if len(_session_stores) >= _SESSION_CLEANUP_THRESHOLD:
        _cleanup_expired_sessions()

    # Update last access time
    _session_last_access[session_id] = time.time()

    if session_id not in _session_stores:
        from backend.utils.vector_store import InMemoryVectorStore
        _session_stores[session_id] = InMemoryVectorStore()
    return _session_stores[session_id]


class ChatMessage(BaseModel):
    role: str  # user, assistant, system
    content: str
    images: Optional[List[str]] = None  # base64 data URIs for vision models


class ChatRequest(BaseModel):
    instance_id: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repeat_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    mirostat: Optional[int] = None
    mirostat_tau: Optional[float] = None
    mirostat_eta: Optional[float] = None
    typical_p: Optional[float] = None
    tfs_z: Optional[float] = None
    stream: Optional[bool] = True


class ChatResponse(BaseModel):
    request_id: str
    content: str
    finish_reason: Optional[str] = None


def calculate_dynamic_max_tokens(orchestrator, instance_id, messages, requested_max_tokens, log):
    """Calculate dynamically capped max_tokens based on available context space."""
    instance = orchestrator.get_instance(instance_id)
    if not instance:
        return requested_max_tokens, False

    context_length = instance.context_length or 4096
    messages_dicts = [{"role": m.role, "content": m.content} for m in messages]
    input_tokens = estimate_messages_tokens(messages_dicts)

    safe_max = calculate_safe_max_tokens(
        context_length, input_tokens, requested_max_tokens, 0.05
    )
    was_capped = safe_max < requested_max_tokens

    if was_capped:
        log.info(f"max_tokens: {requested_max_tokens} -> {safe_max} (ctx={context_length}, in~{input_tokens})")

    return safe_max, was_capped


# Active WebSocket connections for streaming
active_connections: Dict[str, WebSocket] = {}


@router.post("/completions")
async def chat_completions(request: Request, body: ChatRequest):
    """
    Non-streaming chat completion.
    For streaming, use the WebSocket endpoint.
    """
    from providers.base import InferenceRequest, ChatMessage as ProviderChatMessage

    orchestrator = request.app.state.orchestrator
    settings = request.app.state.settings

    try:
        # Convert messages (including images for vision models)
        messages = [
            ProviderChatMessage(role=m.role, content=m.content, images=m.images)
            for m in body.messages
        ]

        # Calculate dynamic max_tokens based on available context
        requested_max = body.max_tokens or settings.get("inference.default_max_tokens", 1024)
        safe_max_tokens, _ = calculate_dynamic_max_tokens(
            orchestrator, body.instance_id, messages, requested_max, logger
        )

        # Create inference request with all parameters
        inference_req = InferenceRequest(
            request_id=f"rest-{id(body)}",
            messages=messages,
            max_tokens=safe_max_tokens,
            temperature=body.temperature or settings.get("inference.default_temperature", 0.7),
            top_p=body.top_p or settings.get("inference.default_top_p", 0.95),
            top_k=body.top_k or settings.get("inference.default_top_k", 40),
            repeat_penalty=body.repeat_penalty or settings.get("inference.default_repeat_penalty", 1.1),
            presence_penalty=body.presence_penalty or settings.get("inference.default_presence_penalty", 0.0),
            frequency_penalty=body.frequency_penalty or settings.get("inference.default_frequency_penalty", 0.0),
            mirostat=body.mirostat if body.mirostat is not None else settings.get("inference.default_mirostat", 0),
            mirostat_tau=body.mirostat_tau or settings.get("inference.default_mirostat_tau", 5.0),
            mirostat_eta=body.mirostat_eta or settings.get("inference.default_mirostat_eta", 0.1),
            typical_p=body.typical_p or settings.get("inference.default_typical_p", 1.0),
            tfs_z=body.tfs_z or settings.get("inference.default_tfs_z", 1.0),
        )

        # Collect full response
        full_response = ""
        async for response in orchestrator.chat(body.instance_id, inference_req):
            full_response += response.text

        return ChatResponse(
            request_id=inference_req.request_id,
            content=full_response,
            finish_reason="stop"
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return {"error": str(e)}


@router.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for streaming chat.

    Protocol:
    - Client sends: {"action": "chat", "instance_id": "...", "messages": [...], ...}
    - Server sends: {"type": "token", "content": "..."} for each token
    - Server sends: {"type": "done", "request_id": "..."} when complete
    - Server sends: {"type": "error", "error": "..."} on error
    """
    from providers.base import InferenceRequest, ChatMessage as ProviderChatMessage

    await websocket.accept()
    connection_id = str(id(websocket))
    active_connections[connection_id] = websocket
    logger.info(f"WebSocket connected: {connection_id}")

    # Track active chat tasks for this connection
    active_tasks: Dict[str, asyncio.Task] = {}

    async def process_chat(request_id: str, instance_id: str, inference_req: InferenceRequest, panel_id: str = None):
        """Process a single chat request as a background task."""
        logger.info(f"=== process_chat started: {request_id} ===")
        try:
            orchestrator = websocket.app.state.orchestrator
            logger.info(f"Calling orchestrator.chat({instance_id})")
            # Stream tokens with batching (reduce WebSocket frames)
            import time as _time
            response_count = 0
            token_buffer = []
            last_flush = _time.monotonic()
            BATCH_INTERVAL = 0.016  # ~16ms (one display frame at 60fps)

            async for response in orchestrator.chat(instance_id, inference_req):
                response_count += 1
                if response_count <= 3:
                    logger.info(f"Response {response_count}: text={bool(response.text)}, error={response.error}, done={response.done}")
                if response.error:
                    logger.error(f"Response error: {response.error}")
                    # Flush any buffered tokens before error
                    if token_buffer:
                        msg = {"type": "token", "request_id": request_id, "content": "".join(token_buffer)}
                        if panel_id: msg["panel_id"] = panel_id
                        await websocket.send_text(json.dumps(msg))
                        token_buffer.clear()
                    msg = {"type": "error", "request_id": request_id, "error": response.error}
                    if panel_id: msg["panel_id"] = panel_id
                    await websocket.send_text(json.dumps(msg))
                    return

                if response.text:
                    token_buffer.append(response.text)

                now = _time.monotonic()
                if token_buffer and (now - last_flush) >= BATCH_INTERVAL:
                    msg = {"type": "token", "request_id": request_id, "content": "".join(token_buffer)}
                    if panel_id: msg["panel_id"] = panel_id
                    await websocket.send_text(json.dumps(msg))
                    token_buffer.clear()
                    last_flush = now

            # Flush remaining tokens
            if token_buffer:
                msg = {"type": "token", "request_id": request_id, "content": "".join(token_buffer)}
                if panel_id: msg["panel_id"] = panel_id
                await websocket.send_text(json.dumps(msg))

            logger.info(f"Stream complete: {response_count} responses")
            # Done
            msg = {"type": "done", "request_id": request_id}
            if panel_id: msg["panel_id"] = panel_id
            await websocket.send_text(json.dumps(msg))

        except asyncio.CancelledError:
            logger.info(f"Chat task cancelled: {request_id}")
            msg = {"type": "cancelled", "request_id": request_id}
            if panel_id: msg["panel_id"] = panel_id
            await websocket.send_json(msg)
        except Exception as e:
            logger.error(f"Chat stream error: {type(e).__name__}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            msg = {"type": "error", "request_id": request_id, "error": str(e)}
            if panel_id: msg["panel_id"] = panel_id
            await websocket.send_json(msg)
        finally:
            if request_id in active_tasks:
                del active_tasks[request_id]

    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            action = data.get("action")

            if action == "chat":
                logger.info(f"=== WebSocket chat request ===")
                settings = websocket.app.state.settings

                instance_id = data.get("instance_id")
                messages_data = data.get("messages", [])
                params = data.get("params", {})

                logger.info(f"Instance ID: {instance_id}")
                logger.info(f"Messages count: {len(messages_data)}")

                # Log if any messages have images
                for i, m in enumerate(messages_data):
                    has_imgs = m.get("images") and len(m.get("images", [])) > 0
                    logger.info(f"  Message {i}: role={m.get('role')}, has_images={has_imgs}")

                # Convert messages (including images for vision models)
                messages = [
                    ProviderChatMessage(
                        role=m["role"],
                        content=m["content"],
                        images=m.get("images")
                    )
                    for m in messages_data
                ]

                # Calculate dynamic max_tokens based on available context
                orchestrator = websocket.app.state.orchestrator
                requested_max = params.get("max_tokens", settings.get("inference.default_max_tokens", 1024))
                safe_max_tokens, _ = calculate_dynamic_max_tokens(
                    orchestrator, instance_id, messages, requested_max, logger
                )

                # Create request with all inference parameters
                request_id = data.get("request_id", f"ws-{connection_id}-{id(data)}")
                inference_req = InferenceRequest(
                    request_id=request_id,
                    messages=messages,
                    max_tokens=safe_max_tokens,
                    temperature=params.get("temperature", settings.get("inference.default_temperature", 0.7)),
                    top_p=params.get("top_p", settings.get("inference.default_top_p", 0.95)),
                    top_k=params.get("top_k", settings.get("inference.default_top_k", 40)),
                    repeat_penalty=params.get("repeat_penalty", settings.get("inference.default_repeat_penalty", 1.1)),
                    presence_penalty=params.get("presence_penalty", settings.get("inference.default_presence_penalty", 0.0)),
                    frequency_penalty=params.get("frequency_penalty", settings.get("inference.default_frequency_penalty", 0.0)),
                    mirostat=params.get("mirostat", settings.get("inference.default_mirostat", 0)),
                    mirostat_tau=params.get("mirostat_tau", settings.get("inference.default_mirostat_tau", 5.0)),
                    mirostat_eta=params.get("mirostat_eta", settings.get("inference.default_mirostat_eta", 0.1)),
                    typical_p=params.get("typical_p", settings.get("inference.default_typical_p", 1.0)),
                    tfs_z=params.get("tfs_z", settings.get("inference.default_tfs_z", 1.0)),
                )

                # Spawn task for concurrent processing
                panel_id = data.get("panel_id")
                task = asyncio.create_task(process_chat(request_id, instance_id, inference_req, panel_id=panel_id))
                active_tasks[request_id] = task

                # Yield to event loop to allow task to start before processing next message
                # This ensures concurrent requests actually run concurrently
                await asyncio.sleep(0)

            elif action == "ping":
                await websocket.send_json({"type": "pong"})

            elif action == "cancel":
                request_id = data.get("request_id")
                panel_id = data.get("panel_id")
                if request_id:
                    # Cancel the task if it exists
                    if request_id in active_tasks:
                        active_tasks[request_id].cancel()
                    else:
                        orchestrator = websocket.app.state.orchestrator
                        await orchestrator.request_queue.cancel(request_id)
                    msg = {"type": "cancelled", "request_id": request_id}
                    if panel_id:
                        msg["panel_id"] = panel_id
                    await websocket.send_json(msg)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Cancel all active tasks on disconnect
        for task in active_tasks.values():
            task.cancel()
        if connection_id in active_connections:
            del active_connections[connection_id]


@router.get("/queue")
async def get_queue_status(request: Request):
    """Get request queue status."""
    orchestrator = request.app.state.orchestrator
    queue = orchestrator.request_queue

    return {
        "queue_length": queue.get_queue_length(),
        "processing_count": queue.get_processing_count(),
        "pending": queue.get_pending_requests(),
    }


@router.delete("/queue/{request_id}")
async def cancel_request(request: Request, request_id: str):
    """Cancel a queued request."""
    orchestrator = request.app.state.orchestrator
    success = await orchestrator.request_queue.cancel(request_id)
    return {"success": success}


# ==================== PDF RAG Endpoints ====================


class PdfUploadResponse(BaseModel):
    success: bool
    filename: str = ""
    page_count: int = 0
    chunk_count: int = 0
    token_estimate: int = 0
    embedding_provider: Optional[str] = None
    error: Optional[str] = None
    fallback_mode: bool = False


class RetrieveRequest(BaseModel):
    query: str
    session_id: str = "default"
    top_k: int = 5


class RetrieveResponse(BaseModel):
    success: bool
    chunks: List[Dict[str, Any]] = []
    error: Optional[str] = None


@router.post("/upload-pdf", response_model=PdfUploadResponse)
async def upload_pdf(
    request: Request,
    file: UploadFile = File(...),
    session_id: str = Form(default="default")
):
    """
    Upload PDF, extract text, chunk, embed, and store for RAG.

    If no embedding provider is available, falls back to simple mode
    (stores text without embeddings for full-context injection).
    """
    from backend.utils.pdf_processor import extract_and_chunk_pdf, extract_pdf_text_simple

    settings = request.app.state.settings

    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        return PdfUploadResponse(
            success=False,
            error="File must be a PDF"
        )

    # Read file content
    content = await file.read()

    # Check file size (max 50MB for PDFs)
    max_size_mb = 50
    if len(content) > max_size_mb * 1024 * 1024:
        return PdfUploadResponse(
            success=False,
            error=f"PDF too large (max {max_size_mb}MB)"
        )

    # Extract and chunk PDF
    result = await extract_and_chunk_pdf(content, file.filename)

    if not result.success:
        return PdfUploadResponse(
            success=False,
            filename=file.filename,
            page_count=result.page_count,
            error=result.error
        )

    # Get embedding manager and try to embed
    embedding_manager = _get_embedding_manager(settings.get_all())
    store = _get_session_store(session_id)

    try:
        # Auto-load embedding model and embed chunks
        chunk_texts = [c.text for c in result.chunks]
        embeddings = await embedding_manager.embed(chunk_texts)

        # Store in vector store
        chunk_dicts = [c.to_dict() for c in result.chunks]
        store.add(chunk_dicts, embeddings)

        status = embedding_manager.get_status()

        logger.info(
            f"PDF '{file.filename}' processed with RAG: "
            f"{len(result.chunks)} chunks embedded via {status['active_provider']}"
        )

        return PdfUploadResponse(
            success=True,
            filename=file.filename,
            page_count=result.page_count,
            chunk_count=len(result.chunks),
            token_estimate=result.token_estimate,
            embedding_provider=status.get("active_provider"),
            fallback_mode=False
        )

    except RuntimeError as e:
        # No embedding provider available - fall back to simple mode
        logger.warning(f"Embedding failed, using fallback mode: {e}")

        # Store chunks without embeddings (for simple injection)
        # We'll store them with zero vectors
        chunk_dicts = [c.to_dict() for c in result.chunks]

        # Store in a special fallback structure
        if not hasattr(store, '_fallback_chunks'):
            store._fallback_chunks = {}
        store._fallback_chunks[file.filename] = {
            "chunks": chunk_dicts,
            "full_text": "\n\n".join(c.text for c in result.chunks),
            "page_count": result.page_count,
            "token_estimate": result.token_estimate,
        }

        return PdfUploadResponse(
            success=True,
            filename=file.filename,
            page_count=result.page_count,
            chunk_count=len(result.chunks),
            token_estimate=result.token_estimate,
            embedding_provider=None,
            fallback_mode=True,
            error="No embedding service available - using full-text mode"
        )


@router.post("/retrieve-context", response_model=RetrieveResponse)
async def retrieve_context(request: Request, body: RetrieveRequest):
    """
    Retrieve relevant chunks for a query using vector similarity.

    If in fallback mode (no embeddings), returns all text for the session.
    """
    settings = request.app.state.settings

    store = _session_stores.get(body.session_id)
    if not store:
        return RetrieveResponse(
            success=False,
            error="No PDF session found. Upload a PDF first."
        )

    # Check for fallback mode
    if hasattr(store, '_fallback_chunks') and store._fallback_chunks:
        # Return full text in fallback mode
        chunks = []
        for filename, data in store._fallback_chunks.items():
            chunks.append({
                "text": data["full_text"],
                "filename": filename,
                "page": 1,
                "score": 1.0,
                "fallback": True
            })
        return RetrieveResponse(success=True, chunks=chunks)

    # Normal RAG retrieval
    if not store:
        return RetrieveResponse(
            success=False,
            error="No documents in session"
        )

    try:
        embedding_manager = _get_embedding_manager(settings.get_all())

        # Embed query
        query_embeddings = await embedding_manager.embed([body.query])
        query_embedding = query_embeddings[0]

        # Search vector store
        results = store.search(query_embedding, top_k=body.top_k)

        return RetrieveResponse(success=True, chunks=results)

    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return RetrieveResponse(success=False, error=str(e))


@router.post("/clear-pdf-session")
async def clear_pdf_session(request: Request, session_id: str = "default"):
    """
    Clear vector store and unload embedding model for a session.

    Called when user clicks "Clear" to start a new chat.
    """
    global _session_stores, _embedding_manager

    # Clear session store
    if session_id in _session_stores:
        store = _session_stores[session_id]
        store.clear()
        del _session_stores[session_id]
        logger.info(f"Cleared PDF session: {session_id}")

    # Auto-unload embedding model if no more sessions
    if not _session_stores and _embedding_manager:
        await _embedding_manager.auto_unload()
        logger.info("Embedding model unloaded (no active sessions)")

    return {"success": True, "session_id": session_id}


@router.get("/pdf-status")
async def get_pdf_status(request: Request, session_id: str = "default"):
    """Get PDF/RAG session status."""
    settings = request.app.state.settings

    store = _session_stores.get(session_id)
    embedding_manager = _get_embedding_manager(settings.get_all()) if _embedding_manager else None

    store_stats = store.get_stats() if store else None
    embedding_status = embedding_manager.get_status() if embedding_manager else None

    # Check for fallback mode
    fallback_files = []
    if store and hasattr(store, '_fallback_chunks'):
        fallback_files = list(store._fallback_chunks.keys())

    return {
        "session_id": session_id,
        "has_documents": store is not None and (len(store) > 0 or bool(fallback_files)),
        "store_stats": store_stats,
        "embedding_status": embedding_status,
        "fallback_mode": bool(fallback_files),
        "fallback_files": fallback_files,
    }


@router.get("/pdf-providers")
async def get_pdf_providers(request: Request):
    """Check available embedding providers."""
    settings = request.app.state.settings
    embedding_manager = _get_embedding_manager(settings.get_all())

    providers = await embedding_manager.detect_providers()

    return {
        "providers": [
            {
                "name": p.provider.value,
                "available": p.available,
                "model": p.model_id if p.available else None,
                "error": p.error if not p.available else None,
            }
            for p in providers
        ]
    }


# ==================== Debate Mode ====================

class DebateRequest(BaseModel):
    topic: str
    model1_id: str
    model2_id: str
    model1_position: Optional[str] = "for"  # "for" or "against"
    rounds: Optional[int] = 3
    max_tokens_per_turn: Optional[int] = 300


class DebateTurn(BaseModel):
    round: int
    model_id: str
    model_name: str
    position: str
    content: str


class DebateResponse(BaseModel):
    topic: str
    model1_name: str
    model2_name: str
    turns: List[DebateTurn]
    winner: Optional[str] = None


@router.post("/debate")
async def run_debate(request: Request, body: DebateRequest):
    """
    Run a multi-turn debate between two models on a topic.
    Streams each turn token-by-token via SSE so the user watches it unfold live.
    """
    from providers.base import InferenceRequest, ChatMessage as ProviderChatMessage

    orchestrator = request.app.state.orchestrator

    # Validate models upfront before entering the generator
    instance1 = orchestrator.get_instance(body.model1_id)
    instance2 = orchestrator.get_instance(body.model2_id)

    if not instance1 or not instance2:
        missing = body.model1_id if not instance1 else body.model2_id
        async def error_gen():
            yield f"data: {json.dumps({'type': 'error', 'error': f'Model not found: {missing}'})}\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    model1_name = instance1.display_name
    model2_name = instance2.display_name
    pos1 = body.model1_position or "for"
    pos2 = "against" if pos1 == "for" else "for"

    async def generate():
        try:
            # System prompts for each debater
            system1 = f"""You are participating in a formal debate on the topic: "{body.topic}"

Your position: You are ARGUING {pos1.upper()} this topic.

Rules:
- Make clear, logical arguments supporting your position
- Respond to your opponent's points when relevant
- Be persuasive but respectful
- Keep responses concise (2-3 paragraphs max)
- Do not break character or acknowledge you are an AI"""

            system2 = f"""You are participating in a formal debate on the topic: "{body.topic}"

Your position: You are ARGUING {pos2.upper()} this topic.

Rules:
- Make clear, logical arguments supporting your position
- Respond to your opponent's points when relevant
- Be persuasive but respectful
- Keep responses concise (2-3 paragraphs max)
- Do not break character or acknowledge you are an AI"""

            yield f"data: {json.dumps({'type': 'debate_start', 'topic': body.topic, 'model1_name': model1_name, 'model2_name': model2_name, 'model1_position': pos1, 'model2_position': pos2, 'rounds': body.rounds})}\n\n"

            debate_history = []

            for round_num in range(1, body.rounds + 1):
                # --- Model 1's turn ---
                messages1 = [ProviderChatMessage(role="system", content=system1)]
                if round_num == 1:
                    messages1.append(ProviderChatMessage(
                        role="user",
                        content=f"The debate topic is: \"{body.topic}\"\n\nYou are arguing {pos1.upper()}. Please give your opening argument."
                    ))
                else:
                    for turn in debate_history:
                        if turn["model_id"] == body.model1_id:
                            messages1.append(ProviderChatMessage(role="assistant", content=turn["content"]))
                        else:
                            messages1.append(ProviderChatMessage(role="user", content=f"[Opponent's argument]: {turn['content']}"))
                    messages1.append(ProviderChatMessage(
                        role="user",
                        content="Please respond to your opponent's points and continue your argument."
                    ))

                req1 = InferenceRequest(
                    request_id=f"debate-{round_num}-1",
                    messages=messages1,
                    max_tokens=body.max_tokens_per_turn,
                    temperature=0.8,
                )

                yield f"data: {json.dumps({'type': 'turn_start', 'round': round_num, 'model_name': model1_name, 'position': pos1})}\n\n"

                response1 = ""
                async for resp in orchestrator.chat(body.model1_id, req1):
                    if resp.text:
                        response1 += resp.text
                        yield f"data: {json.dumps({'type': 'token', 'content': resp.text})}\n\n"

                debate_history.append({
                    "round": round_num,
                    "model_id": body.model1_id,
                    "model_name": model1_name,
                    "position": pos1,
                    "content": response1.strip()
                })

                yield f"data: {json.dumps({'type': 'turn_end', 'round': round_num})}\n\n"

                # --- Model 2's turn ---
                messages2 = [ProviderChatMessage(role="system", content=system2)]
                for turn in debate_history:
                    if turn["model_id"] == body.model2_id:
                        messages2.append(ProviderChatMessage(role="assistant", content=turn["content"]))
                    else:
                        messages2.append(ProviderChatMessage(role="user", content=f"[Opponent's argument]: {turn['content']}"))

                if round_num == 1:
                    messages2.append(ProviderChatMessage(
                        role="user",
                        content="Please respond to your opponent's opening argument."
                    ))
                else:
                    messages2.append(ProviderChatMessage(
                        role="user",
                        content="Please respond to your opponent's points and continue your argument."
                    ))

                req2 = InferenceRequest(
                    request_id=f"debate-{round_num}-2",
                    messages=messages2,
                    max_tokens=body.max_tokens_per_turn,
                    temperature=0.8,
                )

                yield f"data: {json.dumps({'type': 'turn_start', 'round': round_num, 'model_name': model2_name, 'position': pos2})}\n\n"

                response2 = ""
                async for resp in orchestrator.chat(body.model2_id, req2):
                    if resp.text:
                        response2 += resp.text
                        yield f"data: {json.dumps({'type': 'token', 'content': resp.text})}\n\n"

                debate_history.append({
                    "round": round_num,
                    "model_id": body.model2_id,
                    "model_name": model2_name,
                    "position": pos2,
                    "content": response2.strip()
                })

                yield f"data: {json.dumps({'type': 'turn_end', 'round': round_num})}\n\n"

            yield f"data: {json.dumps({'type': 'complete'})}\n\n"

        except Exception as e:
            logger.error(f"Debate error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ==================== LLM Judge ====================

class JudgeRequest(BaseModel):
    prompt: str
    response_a: str
    response_b: str
    model_a_name: Optional[str] = "Model A"
    model_b_name: Optional[str] = "Model B"
    judge_model_id: Optional[str] = None  # defaults to first loaded model


@router.post("/judge")
async def run_judge(request: Request, body: JudgeRequest):
    """
    Have an LLM judge two responses to the same prompt.
    Streams the verdict via SSE.
    """
    from providers.base import InferenceRequest, ChatMessage as ProviderChatMessage

    orchestrator = request.app.state.orchestrator

    # Find judge model
    judge_id = body.judge_model_id
    if not judge_id:
        instances = orchestrator.get_loaded_instances()
        if not instances:
            async def error_gen():
                yield f"data: {json.dumps({'type': 'error', 'error': 'No models loaded to act as judge'})}\n\n"
            return StreamingResponse(error_gen(), media_type="text/event-stream")
        judge_id = instances[0].id

    instance = orchestrator.get_instance(judge_id)
    if not instance:
        async def error_gen():
            yield f"data: {json.dumps({'type': 'error', 'error': f'Judge model not found: {judge_id}'})}\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    async def generate():
        try:
            system_prompt = """You are an impartial judge evaluating two AI responses to the same prompt.

Compare them on these criteria:
- **Accuracy**: Are the facts correct?
- **Helpfulness**: Does the response address what was asked?
- **Clarity**: Is the response well-written and easy to understand?
- **Completeness**: Does it cover the topic thoroughly?

Give a brief analysis of each response, then declare a winner: "Response A", "Response B", or "Tie".
End with a single line: VERDICT: [Response A / Response B / Tie]"""

            user_content = f"""**Original Prompt:**
{body.prompt}

**Response A ({body.model_a_name}):**
{body.response_a}

**Response B ({body.model_b_name}):**
{body.response_b}

Please evaluate both responses and declare a winner."""

            messages = [
                ProviderChatMessage(role="system", content=system_prompt),
                ProviderChatMessage(role="user", content=user_content),
            ]

            req = InferenceRequest(
                request_id=f"judge-{id(body)}",
                messages=messages,
                max_tokens=600,
                temperature=0.3,
            )

            yield f"data: {json.dumps({'type': 'judge_start', 'judge_model': instance.display_name})}\n\n"

            async for resp in orchestrator.chat(judge_id, req):
                if resp.text:
                    yield f"data: {json.dumps({'type': 'token', 'content': resp.text})}\n\n"

            yield f"data: {json.dumps({'type': 'complete'})}\n\n"

        except Exception as e:
            logger.error(f"Judge error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
