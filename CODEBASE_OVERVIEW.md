# AgentNate v2.0 - Codebase Overview

**AgentNate** is a high-throughput, multi-provider LLM orchestration system with a powerful Meta Agent, 73+ tools, n8n workflow integration, and vision capabilities.

---

## Project Status: ~85% Complete

| Component | Status | Notes |
|-----------|--------|-------|
| Orchestrator & Model Pooling | 95% | Multi-provider, JIT loading, request queue (memory leak fixed) |
| Providers (5 total) | 85% | llama.cpp, vLLM (custom Windows build), LM Studio, Ollama, OpenRouter |
| Meta Agent Tools | 40% | 73 tools defined, execution loop not wired to main chat |
| Persona System | 80% | CRUD works, not used in chat flow yet |
| n8n Integration | 95% | Queue manager + legacy manager, process registry, workers |
| Workflow Generation | 70% | Templates + LLM generation, webhook activation limitation |
| Vision Support | 50% | Detection works, chat flow untested |
| ComfyUI Module | 100% | Full lifecycle: download, bootstrap, install, multi-instance GPU |
| Process Registry | 100% | Persistent PID tracking, orphan cleanup on startup |
| Debug Logging | 100% | Middleware + manager + frontend logging to file |
| Queue Execution | 100% | Sequential + parallel modes, play/pause/enqueue controls |
| Workflow Suite | 100% | 6 pre-built workflows ready to import |
| UI Module System | 100% | 18+ ES modules, 7 tabs, lazy initialization |

**Remaining critical gap**: Tool execution is not connected to the main chat flow.

---

## IMPORTANT: Portable Python Environment

This project uses a **portable Python installation**. All dependencies are in `python/`.

```bash
# Correct - use portable Python
E:\AgentNate\python\python.exe run.py

# Install packages to portable env
E:\AgentNate\python\python.exe -m pip install <package> --target E:\AgentNate\python\Lib\site-packages
```

**NEVER use system Python or pip directly.**

---

## Quick Start

```bash
# Start the server (opens in browser)
E:\AgentNate\python\python.exe run.py

# Or server-only mode
E:\AgentNate\python\python.exe run.py --mode server

# API available at http://localhost:8000
# API docs at http://localhost:8000/docs
```

---

## Architecture Overview

### Process Model - Maximum Throughput

```
┌─────────────────────────────────────────────────────────────────────┐
│  AgentNate Backend (FastAPI)                            Port 8000   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Orchestrator                                                       │
│  ├── Request Queue (priority-based, concurrent)                     │
│  ├── Health Monitor (30s intervals)                                 │
│  ├── Instance Pool: Dict[instance_id, ModelInstance]                │
│  │                                                                  │
│  ├── llama_cpp_provider                                             │
│  │   └── workers: subprocess.Popen → inference_worker.py            │
│  │                                                                  │
│  ├── lm_studio_provider                                             │
│  │   └── SDK client + OpenAI API fallback                           │
│  │                                                                  │
│  ├── vllm_provider                                                  │
│  │   └── Custom Windows launcher → vllm_launcher.py                 │
│  │                                                                  │
│  ├── ollama_provider                                                │
│  │   └── HTTP API → localhost:11434                                 │
│  │                                                                  │
│  └── openrouter_provider                                            │
│      └── HTTP API → openrouter.ai                                   │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  Meta Agent System                                                  │
│  ├── Personas: system_agent, researcher, coder, power_agent, etc.   │
│  ├── Tool Router: 73 tools across 12 categories                     │
│  ├── Conversation Store: persistent chat history                    │
│  └── Suggestions Engine: contextual action recommendations          │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  Process Registry                                                   │
│  └── Persistent PID tracking via .n8n-instances/process_registry.json│
│      ├── Kills orphans from dead previous server on startup         │
│      └── Safety net kill_all_registered() on shutdown               │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  n8n Queue Manager                                                  │
│  ├── Main instance :5678 → shared database                          │
│  ├── Workers: isolated instances with queue execution               │
│  │   ├── Sequential mode (one at a time)                            │
│  │   └── Parallel mode (configurable batch)                         │
│  └── Legacy N8nManager for meta agent tool compatibility            │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  ComfyUI Manager                                                    │
│  ├── Portable installer (auto git clone + headless bootstrap)       │
│  ├── Management API proxy → port 5000                               │
│  └── Up to 8 GPU instances → ports 8188-8199                        │
└─────────────────────────────────────────────────────────────────────┘
```

### Why Subprocesses?

1. **True parallelism** - Python GIL doesn't block inference
2. **GPU isolation** - Each worker pins to specific GPU via CUDA_VISIBLE_DEVICES
3. **Fault isolation** - Crashed worker doesn't take down the server
4. **Memory isolation** - Each model has its own memory space
5. **Scalability** - Add more workers without coordination overhead

---

## Meta Agent & Tool System

The Meta Agent is an AI assistant that can use tools to accomplish tasks.

### Tool Categories (73 tools)

| Category | Tools | Description |
|----------|-------|-------------|
| **Model Management** | 5 | load_model, unload_model, list_loaded_models, etc. |
| **System Status** | 6 | get_gpu_status, get_system_health, quick_setup, etc. |
| **Workflow Automation** | 4 | generate_workflow, deploy_workflow, etc. |
| **n8n Instances** | 4 | spawn_n8n, stop_n8n, list_n8n_instances, etc. |
| **Web & Browser** | 10 | web_search, fetch_url, browser_open, browser_click, etc. |
| **File Operations** | 9 | read_file, write_file, search_files, search_content, etc. |
| **Code Execution** | 4 | run_python, run_javascript, run_shell, run_powershell |
| **Communication** | 5 | send_discord, send_slack, send_email, send_telegram, etc. |
| **Data & APIs** | 5 | http_request, parse_json, parse_html, database_query, etc. |
| **Utilities** | 8 | calculate, get_datetime, generate_uuid, hash_text, etc. |
| **Vision & Images** | 6 | analyze_image, analyze_screenshot, extract_text_from_image, etc. |
| **Codebase Guide** | 7 | scan_codebase, explain_file, get_architecture, etc. |

### Personas

Personas define the Meta Agent's identity, available tools, and behavior:

| Persona | Tools | Use Case |
|---------|-------|----------|
| `system_agent` | all | Control AgentNate system |
| `power_agent` | all | Full-featured assistant |
| `researcher` | web, data, utility, vision | Web research |
| `coder` | code, files, utility | Code execution |
| `automator` | workflow, n8n, code, communication | Automation |
| `data_analyst` | data, files, code, utility | Data analysis |
| `vision_agent` | vision, web, files | Image analysis |
| `codebase_guide` | codebase, files | Explain AgentNate code |
| `general_assistant` | none | Pure chat |
| `code_assistant` | none | Code help (no execution) |

### Condensed Tool Format

Tool prompts use a condensed format to save ~57% tokens:

```
## Web & Browser
  - web_search(query*, num_results=5): Search the web using DuckDuckGo
  - browser_open(url*, wait_for=load): Open a URL in the automated browser
  ...
```

---

## API Endpoints

### Models

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/models/list` | List available models from all providers |
| GET | `/api/models/loaded` | List currently loaded model instances |
| POST | `/api/models/load` | Load a model (with GPU selection) |
| POST | `/api/models/load-jit` | Load model JIT (reuse if exists) |
| DELETE | `/api/models/{instance_id}` | Unload a model instance |
| GET | `/api/models/{instance_id}` | Get instance details |
| GET | `/api/models/health/all` | Health check all providers |
| GET | `/api/models/providers` | List enabled providers |

### Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat/completions` | Non-streaming chat completion |
| WS | `/api/chat/stream` | WebSocket for streaming chat |
| GET | `/api/chat/queue` | Get request queue status |
| DELETE | `/api/chat/queue/{request_id}` | Cancel queued request |

### Tools & Meta Agent

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/tools/list` | List all available tools |
| GET | `/api/tools/info/{tool_name}` | Get tool details |
| POST | `/api/tools/call` | Direct tool execution (bypass LLM) |
| POST | `/api/tools/agent` | Agent chat with tool calling |
| POST | `/api/tools/agent/stream` | Streaming agent chat |
| GET | `/api/tools/agent/debug` | Debug system prompt size |

### Personas

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/tools/personas` | List all personas |
| GET | `/api/tools/personas/{id}` | Get persona details |
| POST | `/api/tools/personas` | Create custom persona |
| PUT | `/api/tools/personas/{id}` | Update persona |
| DELETE | `/api/tools/personas/{id}` | Delete custom persona |

### Conversations

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/tools/conversations` | List conversations |
| GET | `/api/tools/conversations/{id}` | Get conversation with messages |
| POST | `/api/tools/conversations/{id}/rename` | Rename conversation |
| POST | `/api/tools/conversations/{id}/persona` | Change persona |
| DELETE | `/api/tools/conversations/{id}` | Delete conversation |

### Workflows

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/workflows/generate` | Generate workflow from description |
| POST | `/api/workflows/quick` | Create from template |
| GET | `/api/workflows/templates` | List available templates |

### n8n

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/n8n/list` | List running n8n instances |
| POST | `/api/n8n/spawn` | Spawn new n8n instance |
| DELETE | `/api/n8n/{port}` | Stop n8n instance |
| GET | `/api/n8n/{port}` | Get instance details |
| GET | `/api/n8n/{port}/health` | Health check instance |
| DELETE | `/api/n8n/all` | Stop all instances |

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/system/health` | System health check |
| GET | `/api/system/gpu` | GPU status and memory |
| GET | `/api/settings` | Get settings |
| PUT | `/api/settings` | Update settings |

---

## Directory Structure

```
E:\AgentNate/
├── run.py                      # Application launcher
├── settings.json               # Configuration
├── config.py                   # Config utilities
├── gpu_utils.py                # GPU detection utilities
├── inference_worker.py         # Subprocess worker for llama.cpp
├── CODEBASE_OVERVIEW.md        # This file
├── README.md                   # Project readme
│
├── backend/                    # FastAPI backend
│   ├── __init__.py
│   ├── server.py               # Main FastAPI app + lifespan
│   ├── n8n_manager.py          # Multi-instance n8n process manager
│   ├── personas.py             # Persona definitions & manager
│   ├── conversation_store.py   # Conversation persistence
│   ├── workflow_generator.py   # LLM-based workflow generation
│   ├── workflow_templates.py   # Workflow template library (68KB)
│   │
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── chat.py             # /api/chat/* + WebSocket streaming
│   │   ├── models.py           # /api/models/* endpoints
│   │   ├── n8n.py              # /api/n8n/* endpoints
│   │   ├── settings.py         # /api/settings/* endpoints
│   │   ├── system.py           # /api/system/* endpoints
│   │   ├── tools.py            # /api/tools/* + Meta Agent
│   │   └── workflows.py        # /api/workflows/* endpoints
│   │
│   ├── tools/                  # Meta Agent tool implementations (15 modules)
│   │   ├── __init__.py
│   │   ├── tool_router.py      # Central dispatcher + condensed format
│   │   ├── browser_tools.py    # Playwright browser automation
│   │   ├── codebase_tools.py   # Dynamic codebase exploration
│   │   ├── code_tools.py       # Code execution (sandboxed)
│   │   ├── communication_tools.py  # Discord, Slack, email, etc.
│   │   ├── data_tools.py       # HTTP, JSON, HTML, databases
│   │   ├── file_tools.py       # File operations
│   │   ├── model_tools.py      # Model management
│   │   ├── n8n_tools.py        # n8n instance management
│   │   ├── suggestions.py      # Contextual action suggestions
│   │   ├── system_tools.py     # System status tools
│   │   ├── utility_tools.py    # Datetime, math, encoding, etc.
│   │   ├── vision_tools.py     # Image analysis with vision LLMs
│   │   ├── web_tools.py        # Web search, URL fetching
│   │   └── workflow_tools.py   # Workflow operations
│   │
│   └── utils/                  # Backend utilities
│       ├── __init__.py
│       ├── embedding_manager.py # Embedding generation & caching
│       ├── pdf_processor.py    # PDF text extraction
│       └── vector_store.py     # Vector similarity search
│
├── orchestrator/               # Model orchestration
│   ├── __init__.py
│   ├── orchestrator.py         # Central coordinator
│   └── request_queue.py        # Priority queue with concurrency
│
├── providers/                  # LLM provider implementations (5 total)
│   ├── __init__.py
│   ├── base.py                 # Base classes, ChatMessage, etc.
│   ├── llama_cpp_provider.py   # Local GPU via subprocess workers
│   ├── vllm_provider.py        # vLLM (custom Windows build from source)
│   ├── lm_studio_provider.py   # LM Studio SDK + OpenAI API
│   ├── ollama_provider.py      # Ollama HTTP API
│   └── openrouter_provider.py  # OpenRouter cloud API
│
├── workers/                    # Subprocess workers
│   └── vllm_launcher.py        # Custom FastAPI launcher for vLLM on Windows
│
├── modules/                    # External modules (git-cloned at runtime)
│   └── comfyui/                # Portable ComfyUI installer + management API
│
├── workflows/
│   └── suite/                  # 6 pre-built n8n workflows
│
├── core/                       # Core utilities
│   ├── __init__.py
│   └── signals.py              # Simple pub/sub signal system
│
├── settings/                   # Settings management
│   ├── __init__.py
│   └── settings_manager.py     # JSON settings with dot notation
│
├── ui/                         # Web frontend (served at /, 7 tabs)
│   ├── index.html              # Main HTML page
│   ├── styles.css              # Stylesheet
│   ├── favicon.svg             # Site icon
│   ├── prompts-data.js         # Prompt templates & categories (exported)
│   └── js/                     # ES modules (18+ files)
│       ├── app.js              # Entry point - imports all, wires window onclick
│       ├── state.js            # Shared state object & constants
│       ├── utils.js            # Logging, escapeHtml, apiFetch, debugLog
│       ├── chat.js             # WebSocket, sendMessage, streaming
│       ├── agent.js            # Agent mode, personas, SSE, tool calling
│       ├── models.js           # Load/unload/select models, load modal
│       ├── model-settings.js   # Inference params panel, per-model overrides
│       ├── images.js           # Image upload, preview, vision UI
│       ├── pdf.js              # PDF upload, RAG retrieval, sessions
│       ├── prompts.js          # System prompts UI (library + custom + AI)
│       ├── presets.js          # Model load presets
│       ├── n8n.js              # n8n queue, workers, tabs, polling
│       ├── comfyui.js          # ComfyUI module management (913 lines)
│       ├── settings.js         # Settings modal, save/load/reset
│       ├── arena.js            # Model comparison + debate mode
│       ├── gpu.js              # GPU dashboard, charts, auto-refresh
│       ├── workflows.js        # Marketplace, deploy, param editor
│       ├── conversations.js    # Save/load/rename/delete conversations
│       └── tabs.js             # Tab switching + lazy init
│
├── plans/                      # System prompts library
│   ├── system-prompt-manager.md
│   └── system-prompts-library.md
│
├── tests/                      # Integration tests (30 files)
│   └── test_*.py               # API/provider/system tests
│
├── _archive/                   # Archived legacy code
│
├── python/                     # Portable Python 3.14
├── node/                       # Portable Node.js
├── node_modules/               # n8n and dependencies
├── vllm-source/                # vLLM built from source (Windows, 370 files)
│
└── .n8n-instances/             # n8n data folders (auto-created)
    └── process_registry.json   # Persistent PID tracking
```

---

## Provider Details

### llama.cpp Provider (70% complete)

- **Subprocess isolation**: Each model runs in `inference_worker.py`
- **GPU selection**: Set `gpu_index` in load options
- **Multi-GPU**: Uses `CUDA_VISIBLE_DEVICES` for isolation
- **Streaming**: Token-by-token via stdout JSON
- **Vision support**: Detects vision-capable models

### LM Studio Provider (75% complete)

- **SDK + API**: Uses LM Studio SDK when available, falls back to OpenAI API
- **JIT Loading**: Models load on first inference request
- **GPU selection**: Pass `gpu_index` to SDK load config
- **Vision support**: Detects and handles vision models

### Ollama Provider (80% complete)

- **HTTP API**: Calls `localhost:11434`
- **Model pulling**: Can pull models on demand
- **Streaming**: Native streaming support
- **Keep-alive**: Automatic model warming

### OpenRouter Provider (70% complete)

- **Cloud API**: Calls OpenRouter servers
- **API key**: Set in settings.json
- **Streaming**: SSE token streaming
- **Many models**: Access to 100+ cloud models

### vLLM Provider (90% complete)

- **Custom Windows build**: Built from source with 370/370 CUDA files compiled
- **GGUF support**: Scans GGUF files like llama.cpp, uses gguf_utils for context length
- **Custom launcher**: `workers/vllm_launcher.py` - FastAPI wrapping sync LLM class
- **Windows patches**: InprocClient, TCP ZMQ, spawn context, fake distributed backend
- **GPU isolation**: CUDA_VISIBLE_DEVICES, FLASH_ATTN backend
- **Performance**: 202.8 tok/s concurrent-8 (Qwen2.5-1.5B Q8)
- **Limitation**: Single-GPU only (no NCCL on Windows)

---

## Configuration (settings.json)

```json
{
    "providers": {
        "llama_cpp": {
            "enabled": true,
            "models_directory": "E:\\LL STUDIO",
            "default_n_ctx": 4096,
            "default_n_gpu_layers": 99
        },
        "lm_studio": {
            "enabled": true,
            "base_url": "http://localhost:1234/v1"
        },
        "ollama": {
            "enabled": true,
            "base_url": "http://localhost:11434"
        },
        "openrouter": {
            "enabled": true,
            "api_key": "sk-or-..."
        }
    },
    "inference": {
        "default_max_tokens": 1024,
        "default_temperature": 0.7
    },
    "orchestrator": {
        "max_concurrent_inferences": 4,
        "health_check_interval": 30
    },
    "tools": {
        "file_tools": {"base_path": "C:/Users/chris/AgentWorkspace"},
        "code_tools": {"timeout_seconds": 30},
        "browser_tools": {"headless": true}
    }
}
```

---

## What Works Now

1. Load models from 5 providers (llama.cpp, vLLM, LM Studio, Ollama, OpenRouter)
2. Chat with loaded models (streaming via WebSocket)
3. Queue-based n8n workflow execution (sequential + parallel modes)
4. Generate and deploy workflow templates
5. Import/manage marketplace workflows (official n8n API)
6. ComfyUI image generation with multi-instance GPU management
7. Process registry with orphan cleanup across restarts
8. Debug logging middleware (request/response/timing to file)
9. GPU dashboard with real-time monitoring
10. Model arena for side-by-side comparison
11. 6 pre-built workflow suite ready to import
12. Direct tool execution via `/api/tools/call`
13. System prompt library (43 prompts across 9 categories)

## What's Not Wired Up Yet

1. **Tool execution in chat flow** - The Meta Agent doesn't execute tools during conversation
2. **Personas in inference** - Persona prompts not used in main chat
3. **Conversation persistence** - Store exists but not integrated
4. **Vision chat** - Detection works, needs end-to-end testing
5. **Agent swarms** - Infrastructure exists, coordination logic missing
6. **Settings hot-reload** - Changes require restart
7. **Export/import configurations** - Not implemented

---

## Hardware Detected

- **GPU 0**: NVIDIA GeForce RTX 3060 (12GB)
- **GPU 1**: NVIDIA GeForce RTX 3090 (24GB)

---

## Key Design Decisions

1. **FastAPI over Qt** - Pure async, no UI blocking inference
2. **Subprocess workers** - True parallelism, GPU isolation
3. **WebSocket streaming** - Real-time token delivery
4. **Condensed tool format** - 57% token reduction for tool prompts
5. **Multi-provider abstraction** - Same interface for local and cloud
6. **n8n integration** - Workflow automation via HTTP API

---

## Known Issues / TODO

1. **Tool execution loop** - Critical: wire tools to chat flow
2. **LM Studio orphan models** - SDK may keep models loaded after API unload
3. **Ollama offline** - Requires Ollama to be running separately
4. **Settings hot-reload** - Changes require restart
5. **n8n webhook registration** - Webhooks don't register via API-only activation (n8n limitation)
6. **vLLM single-GPU** - No NCCL on Windows, limited to world_size=1

---

## Test Files Archive

All test files are located in the **`tests/`** folder (`E:\AgentNate\tests\test_*.py`). These are integration tests that make HTTP requests to the API at localhost:8000.

### Test Files by Category (30 files total)

**API & Integration Testing:**
- `test_api_comprehensive.py` - Comprehensive API testing for Meta Agent system
- `test_integration.py` - Deep functionality testing
- `test_comprehensive.py` - Full system testing
- `test_final_summary.py` - Final summary of all features
- `test_direct_chat.py` - Direct chat endpoint testing

**llama.cpp Provider Testing:**
- `test_llamacpp.py` - Basic llama.cpp provider tests
- `test_llamacpp_fixed.py` - Fixed version with improvements
- `test_llamacpp_pool.py` - Model pooling tests
- `test_llamacpp_stress.py` - Stress testing
- `test_worker_direct.py` - Direct subprocess worker testing

**LM Studio Provider Testing:**
- `test_lmstudio_chat.py` - Basic chat testing
- `test_lmstudio_chat2.py` - Enhanced chat testing
- `test_lmstudio_direct_chat.py` - Direct chat API testing
- `test_lmstudio_full.py` - Full feature testing
- `test_lmstudio_gpu.py` - GPU selection testing
- `test_lmstudio_gpu2.py` - Alternative GPU testing
- `test_lmstudio_gpu_final.py` - Final GPU testing version
- `test_lmstudio_load.py` - Model loading testing
- `test_lmstudio_parallel.py` - Parallel inference testing
- `test_lmstudio_provider_gpu.py` - Provider-level GPU testing
- `test_lmstudio_sdk.py` - LM Studio SDK testing

**Other Provider & System Testing:**
- `test_ollama.py` - Ollama provider testing
- `test_openrouter_credits.py` - OpenRouter API credit testing
- `test_openrouter_safe.py` - OpenRouter safe execution testing
- `test_providers.py` - Provider base testing
- `test_orchestrator.py` - Model orchestrator testing
- `test_meta_agent.py` - Meta Agent and tool system testing
- `test_n8n.py` - n8n workflow automation testing
- `test_n8n_stress.py` - n8n stress testing

**Chat Testing:**
- `test_chat.py` - Basic chat generation

### Running Tests

Tests are standalone Python scripts (no pytest/unittest framework):

```bash
# Run a specific test
E:\AgentNate\python\python.exe tests\test_api_comprehensive.py

# Tests require the server to be running first
E:\AgentNate\python\python.exe run.py
```

### Archived Code

Legacy/obsolete code is stored in `_archive/`:
- `_archive/core/` - Old core modules
- `_archive/main.py` - Old entry point
- `_archive/model_manager.py` - Replaced by orchestrator
- `_archive/static/` - Old static files
- `_archive/templates/` - Old HTML templates

---

*Last updated: 2026-02-12*
