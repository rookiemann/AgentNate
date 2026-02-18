"""
AgentNate Backend Server

FastAPI server providing:
- REST API for model management
- WebSocket for streaming inference
- n8n process management
- Static file serving for UI
"""

import os
import sys
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory for imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from settings.settings_manager import SettingsManager
from orchestrator.orchestrator import ModelOrchestrator
from backend.n8n_manager import N8nManager, N8nQueueManager, ProcessRegistry, _kill_n8n_port_orphans
from backend.personas import PersonaManager
from backend.conversation_store import ConversationStore
from backend.routes import models, chat, n8n, workflows, tools, system, settings, presets, marketplace, comfyui, media, routing, openai_compat, comfyui_pool as comfyui_pool_routes, tts, music, gguf as gguf_routes
from backend.comfyui_manager import ComfyUIManager
from backend.tts_manager import TTSManager
from backend.music_manager import MusicManager
from backend.gguf_downloader import GGUFDownloader
from backend.comfyui_pool import ComfyUIPool
from backend.media_catalog import MediaCatalog
from backend.middleware.debug_logger import init_debug_logger, debug_logger, DebugLoggerMiddleware, get_log_file_path, clear_log

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("AgentNate")

# Global instances
settings_manager: Optional[SettingsManager] = None
orchestrator: Optional[ModelOrchestrator] = None
n8n_manager: Optional[N8nManager] = None
n8n_queue_manager: Optional[N8nQueueManager] = None
process_registry: Optional[ProcessRegistry] = None
persona_manager: Optional[PersonaManager] = None
conversation_store: Optional[ConversationStore] = None


def _get_config_dir() -> str:
    """Get the appropriate config directory for the OS."""
    if os.name == 'nt':  # Windows
        base = os.environ.get('APPDATA', os.path.expanduser('~'))
        return os.path.join(base, 'AgentNate')
    else:  # Linux/Mac
        base = os.environ.get('XDG_CONFIG_HOME', os.path.expanduser('~/.config'))
        return os.path.join(base, 'AgentNate')


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global settings_manager, orchestrator, n8n_manager, n8n_queue_manager, process_registry, persona_manager, conversation_store

    logger.info("Starting AgentNate Backend...")

    # Get config directory
    config_dir = _get_config_dir()
    os.makedirs(config_dir, exist_ok=True)

    # Initialize settings
    settings_manager = SettingsManager(settings_dir=BASE_DIR)

    # Initialize n8n data directory and process registry FIRST
    n8n_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ".n8n-instances"
    )
    os.makedirs(n8n_data_dir, exist_ok=True)

    # Initialize debug logger (truncates log for fresh session)
    init_debug_logger(n8n_data_dir)

    # Create process registry and kill orphans from previous crashed server
    registry_path = os.path.join(n8n_data_dir, "process_registry.json")
    process_registry = ProcessRegistry(registry_path)
    orphans_killed = process_registry.kill_orphans()

    # Also scan n8n ports for orphaned node.exe processes not in registry
    # (handles pre-registry orphans and edge cases)
    port_orphans = _kill_n8n_port_orphans(port_range=(5678, 5778))
    orphans_killed += port_orphans

    if orphans_killed > 0:
        # Wait for Windows file lock release after killing orphans
        logger.info(f"Killed {orphans_killed} orphan(s), waiting for file lock release...")
        await asyncio.sleep(3)

    # Record our PID
    process_registry.set_server_pid()

    # Register atexit handler for safety net cleanup
    import atexit
    def _atexit_cleanup():
        if process_registry:
            killed = process_registry.kill_all_registered()
            if killed > 0:
                logger.info(f"atexit: killed {killed} remaining process(es)")
    atexit.register(_atexit_cleanup)

    # Initialize orchestrator
    orchestrator = ModelOrchestrator(settings_manager)
    await orchestrator.start()

    # Initialize n8n managers with portable Node.js
    node_exe = "node.exe" if os.name == "nt" else "node"
    n8n_path = os.path.join(BASE_DIR, "node_modules", "n8n", "bin", "n8n")
    node_path = os.path.join(BASE_DIR, "node", node_exe)

    # Legacy manager (for backwards compatibility)
    n8n_manager = N8nManager(
        base_port=5678,
        n8n_path=n8n_path,
        node_path=node_path,
        registry=process_registry,
    )

    # New queue manager with isolated worker databases
    n8n_queue_manager = N8nQueueManager(
        n8n_path=n8n_path,
        node_path=node_path,
        main_port=5678,
        worker_port_range=(5679, 5778),
        registry=process_registry,
    )

    # Start credential sync and worker monitor for queue manager
    await n8n_queue_manager.start_credential_sync(interval=30)
    await n8n_queue_manager.start_worker_monitor(interval=10)

    # Initialize persona manager
    from pathlib import Path
    persona_manager = PersonaManager(config_dir=Path(config_dir))
    logger.info(f"Loaded {len(persona_manager.list_all())} personas")

    # Initialize conversation store
    conversation_store = ConversationStore()
    logger.info(f"Loaded {len(conversation_store.list_all())} conversations")

    # Initialize ComfyUI manager
    modules_dir = Path(BASE_DIR) / "modules"
    comfyui_mgr = ComfyUIManager(modules_dir=modules_dir, process_registry=process_registry)
    logger.info(f"ComfyUI module: downloaded={comfyui_mgr.is_module_downloaded()}, "
                f"bootstrapped={comfyui_mgr.is_bootstrapped()}, "
                f"installed={comfyui_mgr.is_comfyui_installed()}")

    # Initialize TTS manager
    tts_mgr = TTSManager(modules_dir=modules_dir, process_registry=process_registry)
    logger.info(f"TTS module: downloaded={tts_mgr.is_module_downloaded()}, "
                f"bootstrapped={tts_mgr.is_bootstrapped()}, "
                f"installed={tts_mgr.is_installed()}")

    # Initialize Music manager
    music_mgr = MusicManager(modules_dir=modules_dir, process_registry=process_registry)
    logger.info(f"Music module: downloaded={music_mgr.is_module_downloaded()}, "
                f"bootstrapped={music_mgr.is_bootstrapped()}, "
                f"installed={music_mgr.is_installed()}")

    # Initialize ComfyUI pool for multi-instance dispatch
    comfyui_pool = ComfyUIPool(comfyui_mgr)

    # Initialize GGUF downloader
    gguf_downloader = GGUFDownloader(settings_manager)
    logger.info(f"GGUF downloader: models_dir={gguf_downloader._get_models_directory()}")

    # Store in app state for route access
    app.state.settings = settings_manager
    app.state.orchestrator = orchestrator
    app.state.n8n_manager = n8n_manager
    app.state.n8n_queue_manager = n8n_queue_manager
    app.state.process_registry = process_registry
    app.state.persona_manager = persona_manager
    app.state.conversation_store = conversation_store
    app.state.comfyui_manager = comfyui_mgr
    app.state.comfyui_pool = comfyui_pool
    app.state.tts_manager = tts_mgr
    app.state.music_manager = music_mgr
    app.state.gguf_downloader = gguf_downloader
    app.state.abort_signals = {}  # Agent abort signal tracking

    # Initialize media catalog for ComfyUI generation tracking
    catalog_db = os.path.join(config_dir, "media_catalog.db")
    comfyui_output = str(modules_dir / "comfyui" / "comfyui" / "output")
    app.state.media_catalog = MediaCatalog(catalog_db, comfyui_output)

    # Give pool access to media catalog for auto-scanning after completions
    comfyui_pool.media_catalog = app.state.media_catalog

    # Generate codebase manifest (cached, only regenerates if stale)
    from backend.codebase_manifest import ManifestCache
    manifest_cache = ManifestCache(project_root=BASE_DIR)
    manifest = manifest_cache.get_or_refresh()
    app.state.manifest_cache = manifest_cache
    logger.info(
        f"Codebase manifest: {manifest['stats']['total_files']} files, "
        f"{manifest['stats']['total_tools']} tools, "
        f"{manifest['stats']['total_endpoints']} endpoints"
    )

    # Start ComfyUI pool background polling
    await comfyui_pool.start()

    # Create shared ToolRouter (reused across requests instead of per-request instantiation)
    from backend.tools.tool_router import ToolRouter
    app.state.tool_router = ToolRouter(
        orchestrator, n8n_manager, settings_manager,
        comfyui_mgr, app.state.media_catalog,
        comfyui_pool=comfyui_pool,
        tts_manager=tts_mgr,
        music_manager=music_mgr,
        gguf_downloader=gguf_downloader,
    )

    logger.info("AgentNate Backend ready")

    yield

    # Shutdown
    logger.info("Shutting down...")
    await comfyui_pool.stop()
    await orchestrator.stop()
    await n8n_manager.stop_all()
    await n8n_queue_manager.shutdown()
    await comfyui_mgr.shutdown()
    await tts_mgr.shutdown()
    await music_mgr.shutdown()
    await gguf_downloader.shutdown()
    # Final safety net: kill anything still registered
    if process_registry:
        process_registry.kill_all_registered()
        process_registry.clear()
    logger.info("Shutdown complete")


# Create app
app = FastAPI(
    title="AgentNate",
    description="Multi-provider LLM orchestration backend",
    version="2.0.0",
    lifespan=lifespan
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(DebugLoggerMiddleware)

# Include routers
app.include_router(models.router, prefix="/api/models", tags=["Models"])
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
app.include_router(n8n.router, prefix="/api/n8n", tags=["n8n"])
app.include_router(workflows.router, prefix="/api/workflows", tags=["Workflows"])
app.include_router(tools.router, prefix="/api/tools", tags=["Tools"])
app.include_router(system.router, prefix="/api/system", tags=["System"])
app.include_router(settings.router, prefix="/api/settings", tags=["Settings"])
app.include_router(presets.router, prefix="/api/presets", tags=["Presets"])
app.include_router(marketplace.router, prefix="/api/marketplace", tags=["Marketplace"])
app.include_router(comfyui_pool_routes.router, prefix="/api/comfyui/pool", tags=["ComfyUI Pool"])
app.include_router(comfyui.router, prefix="/api/comfyui", tags=["ComfyUI"])
app.include_router(media.router, prefix="/api/comfyui", tags=["Media"])
app.include_router(tts.router, prefix="/api/tts", tags=["TTS"])
app.include_router(music.router, prefix="/api/music", tags=["Music"])
app.include_router(gguf_routes.router, prefix="/api/gguf", tags=["GGUF Downloads"])
app.include_router(routing.router, prefix="/api/routing", tags=["Routing"])
app.include_router(openai_compat.router, prefix="/v1", tags=["OpenAI Compatible"])


# Debug log endpoints
@app.post("/api/debug/log", tags=["Debug"])
async def debug_log_frontend(request: Request):
    """Receives frontend debug events and appends to debug.log."""
    try:
        body = await request.json()
        action = body.get("action", "unknown")
        detail = body.get("detail", "")
        debug_logger.info(f"[UI] {action} {detail}")
        return {"ok": True}
    except Exception:
        return {"ok": False}


@app.delete("/api/debug/log", tags=["Debug"])
async def clear_debug_log():
    """Clears the debug log file."""
    clear_log()
    return {"ok": True}


# UI directory for static files
UI_DIR = os.path.join(BASE_DIR, "ui")


@app.get("/static/{path:path}")
async def serve_static(path: str):
    """Serve UI static files, fallback to n8n for missing files."""
    # First check if file exists in UI folder
    file_path = os.path.join(UI_DIR, path)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        # Ensure .js files get correct MIME type (required for ES modules)
        media_type = "application/javascript" if path.endswith(".js") else None
        # Disable caching for JS and CSS to pick up changes immediately
        no_cache = path.endswith(".js") or path.endswith(".css")
        headers = {"Cache-Control": "no-cache, no-store, must-revalidate"} if no_cache else None
        return FileResponse(file_path, media_type=media_type, headers=headers)

    # Fallback to n8n proxy if n8n is running
    if n8n_manager and n8n_manager.instances:
        from fastapi.responses import RedirectResponse
        port = next(iter(n8n_manager.instances.keys()))
        return RedirectResponse(
            url=f"/api/n8n/{port}/proxy/static/{path}",
            status_code=307
        )

    # Return 404 if file not found
    from fastapi import HTTPException
    raise HTTPException(status_code=404, detail=f"Static file not found: {path}")


@app.get("/")
async def root():
    """Serve main UI."""
    index_path = os.path.join(UI_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "AgentNate Backend", "status": "running"}


@app.get("/favicon.svg")
async def favicon_svg():
    """Serve favicon as SVG."""
    favicon_path = os.path.join(UI_DIR, "favicon.svg")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path, media_type="image/svg+xml")
    from fastapi import HTTPException
    raise HTTPException(status_code=404, detail="Favicon not found")


@app.get("/favicon.ico")
async def favicon_ico():
    """Serve favicon.ico (returns SVG for browsers that support it)."""
    favicon_path = os.path.join(UI_DIR, "favicon.svg")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path, media_type="image/svg+xml")
    from fastapi import HTTPException
    raise HTTPException(status_code=404, detail="Favicon not found")


# Fallback routes for n8n assets/REST that bypass the proxy path
# These catch requests from n8n JavaScript that uses absolute paths
from fastapi import Request, WebSocket
from fastapi.responses import RedirectResponse

@app.api_route("/assets/{path:path}", methods=["GET"])
async def n8n_assets_fallback(request: Request, path: str):
    """Forward n8n asset requests to the first running instance."""
    if n8n_manager and n8n_manager.instances:
        port = next(iter(n8n_manager.instances.keys()))
        query = f"?{request.url.query}" if request.url.query else ""
        return RedirectResponse(
            url=f"/api/n8n/{port}/proxy/assets/{path}{query}",
            status_code=307
        )
    return {"error": "No n8n instance running"}


@app.api_route("/rest/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def n8n_rest_fallback(request: Request, path: str):
    """Forward n8n REST requests to the first running instance."""
    if n8n_manager and n8n_manager.instances:
        port = next(iter(n8n_manager.instances.keys()))
        query = f"?{request.url.query}" if request.url.query else ""
        return RedirectResponse(
            url=f"/api/n8n/{port}/proxy/rest/{path}{query}",
            status_code=307
        )
    return {"error": "No n8n instance running"}


@app.websocket("/push")
async def n8n_push_fallback(websocket: WebSocket):
    """Forward n8n WebSocket push to the first running instance."""
    import websockets
    import asyncio
    from backend.routes.n8n import _get_or_create_auth

    if not n8n_manager or not n8n_manager.instances:
        await websocket.close(code=1011, reason="No n8n instance running")
        return

    port = next(iter(n8n_manager.instances.keys()))
    await websocket.accept()

    auth_cookie = await _get_or_create_auth(port)
    headers = {}
    if auth_cookie:
        headers["Cookie"] = f"n8n-auth={auth_cookie}"

    try:
        async with websockets.connect(
            f"ws://127.0.0.1:{port}/rest/push",
            additional_headers=headers
        ) as n8n_ws:
            async def forward_to_client():
                try:
                    async for msg in n8n_ws:
                        await websocket.send_text(msg)
                except:
                    pass

            async def forward_to_n8n():
                try:
                    while True:
                        data = await websocket.receive_text()
                        await n8n_ws.send(data)
                except:
                    pass

            await asyncio.gather(forward_to_client(), forward_to_n8n())
    except Exception as e:
        try:
            await websocket.close(code=1011, reason=str(e))
        except:
            pass


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "orchestrator": orchestrator is not None,
        "n8n_manager": n8n_manager is not None,
    }


def run():
    """Run the server."""
    import uvicorn
    uvicorn.run(
        "backend.server:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    run()
