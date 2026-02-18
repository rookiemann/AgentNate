# AgentNate Development Plan

> **Purpose**: This file persists across sessions. Update the checklist as work progresses.
>
> **Last Updated**: 2026-02-12
> **Current Phase**: Phase 3 mostly complete, Phase 5 (ComfyUI) complete

---

## Vision

Transform AgentNate into a **self-orchestrating AI platform** where:
- A central "Meta Agent" chat can control the entire system
- Users describe automations in natural language → working n8n workflows
- Multiple models collaborate on complex tasks
- The system can improve itself by creating new workflows

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      AgentNate Meta Agent                        │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  Central Chat Interface                                      │ │
│  │  - Natural language input                                    │ │
│  │  - Tool calling (load models, create workflows, etc.)        │ │
│  │  - Multi-turn conversation with memory                       │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                              │                                    │
│                              ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  Tool System                                                 │ │
│  │  ├── ModelTools: load, unload, list, status                  │ │
│  │  ├── WorkflowTools: generate, deploy, run, list              │ │
│  │  ├── SystemTools: gpu_status, health, settings               │ │
│  │  └── N8nTools: spawn, stop, list instances                   │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                              │                                    │
│         ┌────────────────────┼────────────────────┐               │
│         ▼                    ▼                    ▼               │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐         │
│  │ LLM Pool    │    │ n8n Instances │    │ Workflow     │         │
│  │ GPU 0: φ    │    │ :5678 :5679   │    │ Templates    │         │
│  │ GPU 1: φ φ  │    │ :5680 ...     │    │ Library      │         │
│  └─────────────┘    └──────────────┘    └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Workflow Generator
**Goal**: Natural language → n8n workflow JSON

### Tasks
- [x] **1.1** Create workflow templates library (common n8n patterns)
- [x] **1.2** Build workflow generator prompt with n8n node examples
- [x] **1.3** Add `/api/workflows/generate` endpoint
- [x] **1.4** Add `/api/workflows/deploy` endpoint (pushes to n8n)
- [x] **1.5** Add UI panel for workflow generation
- [ ] **1.6** Test with 5+ real workflow scenarios

### Files to Create/Modify
- `backend/routes/workflows.py` - New workflow endpoints
- `backend/workflow_generator.py` - Prompt engineering + generation logic
- `backend/workflow_templates.py` - Template library
- `ui/app.js` - Add workflow tab
- `ui/index.html` - Add workflow UI

### Workflow Template Categories
1. **Triggers**: Webhook, Schedule, Email, RSS, File watch
2. **AI Nodes**: HTTP to local LLM, summarize, classify, extract
3. **Actions**: Discord, Slack, Email, HTTP, File write
4. **Data**: Transform, filter, merge, split

---

## Phase 2: Meta Agent Tool System
**Goal**: Chat that can execute actions via tool calls

### Tasks
- [x] **2.1** Design tool schema (OpenAI function calling format)
- [x] **2.2** Implement `ModelTools` (load, unload, list, status)
- [x] **2.3** Implement `WorkflowTools` (generate, deploy, run, list)
- [x] **2.4** Implement `SystemTools` (gpu_status, health)
- [x] **2.5** Implement `N8nTools` (spawn, stop, list)
- [x] **2.6** Build tool execution router
- [x] **2.7** Add tool results back into conversation
- [x] **2.8** Update UI to show tool calls visually

### Files to Create/Modify
- `backend/tools/` - New directory for tool implementations
  - `__init__.py`
  - `model_tools.py`
  - `workflow_tools.py`
  - `system_tools.py`
  - `n8n_tools.py`
  - `tool_router.py`
- `backend/routes/chat.py` - Add tool execution to chat flow
- `ui/app.js` - Visual tool call display

### Tool Schemas
```python
TOOLS = [
    {
        "name": "load_model",
        "description": "Load an LLM model onto a GPU",
        "parameters": {
            "model_name": "string - partial name to match",
            "gpu_index": "int - GPU to load on (0, 1, or -1 for CPU)",
            "n_ctx": "int - context length (default 4096)"
        }
    },
    {
        "name": "generate_workflow",
        "description": "Generate an n8n workflow from description",
        "parameters": {
            "description": "string - what the workflow should do",
            "trigger_type": "string - webhook, schedule, manual"
        }
    },
    # ... more tools
]
```

---

## Phase 3: Advanced Features
**Goal**: Make it impressive and shareable

### Tasks
- [x] **3.1** Model Arena - race models side by side
- [x] **3.2** Multi-agent debate mode
- [x] **3.3** GPU utilization dashboard (real-time charts)
- [x] **3.4** Workflow template marketplace/library UI (switched to official n8n API)
- [x] **3.5** One-click automation recipes (6 quick recipes)
- [x] **3.6** vLLM provider integration (custom Windows build from source)
- [x] **3.7** Process registry for robust lifecycle management
- [x] **3.8** Debug logging middleware (request/response/timing)
- [x] **3.9** Queue-based execution controls (sequential + parallel)
- [x] **3.10** Pre-built workflow suite (6 workflows)
- [x] **3.11** UI module split (monolith → 18 ES modules)
- [ ] **3.12** Export/import configurations
- [ ] **3.13** Self-improving workflows (agent creates workflows that help it)

---

## Phase 5: ComfyUI Image Generation
**Goal**: Integrate portable ComfyUI for local image generation

### Tasks
- [x] **5.1** ComfyUI module manager (download, bootstrap, lifecycle)
- [x] **5.2** Headless bootstrap fix (strip Tkinter GUI from install.bat)
- [x] **5.3** Management API proxy (30 backend routes)
- [x] **5.4** Frontend module (913 lines, 4 subtabs: overview/instances/models/nodes)
- [x] **5.5** One-click full install wizard (auto-chain: download → bootstrap → API → install)
- [x] **5.6** Multi-instance GPU management (up to 8 instances)
- [x] **5.7** Model registry browsing + HuggingFace search/download
- [x] **5.8** Custom node management (16 curated packs)
- [x] **5.9** External ComfyUI support (manage existing installations)
- [x] **5.10** Process registry integration (orphan cleanup)
- [x] **5.11** Debug logging for ComfyUI routes

### Files Created
- `backend/comfyui_manager.py` - Lifecycle manager (380 lines)
- `backend/routes/comfyui.py` - 30 API routes (327 lines)
- `ui/js/comfyui.js` - Frontend module (913 lines)
- `modules/comfyui/` - Git-cloned portable installer (runtime)

---

## Phase 4: Polish & Share
**Goal**: Make it ready for others to use

### Tasks
- [ ] **4.1** Installation script / one-click setup
- [ ] **4.2** Documentation with examples
- [ ] **4.3** Demo video / GIFs
- [ ] **4.4** GitHub README
- [ ] **4.5** Example workflow library (10+ ready-to-use)
- [ ] **4.6** Error handling and edge cases
- [ ] **4.7** Settings UI for configuration

---

## Progress Log

| Date | What Was Done | Notes |
|------|---------------|-------|
| 2026-01-31 | Fixed llama.cpp multi-GPU parallel inference | LFM2.5 has issues, avoid it |
| 2026-01-31 | Verified all GPU features work | Multi-model per GPU, pools, parallel |
| 2026-01-31 | Created development plan | This file |
| 2026-01-31 | Created workflow templates library | `backend/workflow_templates.py` |
| 2026-01-31 | Built workflow generator with LLM | `backend/workflow_generator.py` |
| 2026-01-31 | Created Tool System | `backend/tools/` - model, workflow, system, n8n tools |
| 2026-01-31 | Added API routes for workflows and tools | `/api/workflows/*`, `/api/tools/*` |
| 2026-01-31 | Added Meta Agent endpoint | `/api/tools/agent` - chat with tool calling |
| 2026-01-31 | Fixed model tools provider access | Use ProviderType enum for dict keys |
| 2026-01-31 | Tested full tool system | All 16 tools working |
| 2026-01-31 | Fixed system_tools.py bugs | health_check→check_all_health, ProviderType enum keys |
| 2026-01-31 | Fixed Meta Agent 500 error | Export get_tools_for_prompt from __init__.py |
| 2026-01-31 | Fixed n8n_tools.py bugs | Use N8nInstance.port and .is_running instead of raw process |
| 2026-01-31 | Fixed model_tools.py bugs | Use orchestrator.load_model() to register instances |
| 2026-01-31 | Chat API now works | Models generate actual text responses |
| 2026-01-31 | All tests passing | 12/12 API tests + 4/4 integration tests |
| 2026-01-31 | Added Workflows UI tab | Template browser, quick create, LLM generation |
| 2026-01-31 | Added Agent UI tab | Meta Agent chat with visual tool call display |
| 2026-01-31 | Phase 1.5 & 2.8 complete | UI work done, moving to Phase 3 |
| 2026-01-31 | Implemented GPU Dashboard | Real-time charts, memory/util bars, model map |
| 2026-01-31 | Implemented Multi-agent Debate | Arena mode toggle, debate API, vote for winner |
| 2026-01-31 | Comprehensive GPU/Model Testing | Fixed GPU stats to show all providers |
| 2026-01-31 | Fixed LM Studio JIT gpu_index | Now preserves gpu_index in JIT fallback |
| 2026-01-31 | Workflow Marketplace UI | Category tabs, search, recipe cards, template modal |
| 2026-01-31 | Quick Recipes | 6 pre-built workflow recipes (Discord bot, Sentiment, etc.) |
| 2026-02-03 | Workflow debug session | Fixed connections, executionOrder, Code node limitations |
| 2026-02-04 | Fixed memory leak in request queue | Orphaned asyncio tasks accumulating per second |
| 2026-02-05 | UI module split | Monolithic app.js (7,369 lines) → 18 ES modules |
| 2026-02-05 | MIME type fix | Explicit application/javascript for .js files |
| 2026-02-06 | Process registry | Persistent PID tracking, orphan cleanup on startup |
| 2026-02-06 | Debug logging system | Middleware + manager + frontend logging to debug.log |
| 2026-02-06 | Legacy vs Queue data path fix | Both managers now use .n8n-instances/main/ |
| 2026-02-06 | Port-in-use adoption | start_main() adopts existing n8n on port 5678 |
| 2026-02-07 | Queue-based execution controls | Play/pause, sequential/parallel, enqueue stacking |
| 2026-02-07 | Workflow suite | 6 pre-built workflows in workflows/suite/ |
| 2026-02-07 | Marketplace fix | Switched to official n8n API (api.n8n.io) |
| 2026-02-07 | Batch workflow operations | Archive-before-delete, batch delete with asyncio.gather |
| 2026-02-08 | vLLM provider integration | Replaced SGLang/ExLlamaV2, GGUF scanning, cancel_load |
| 2026-02-09 | vLLM Windows build complete | 370/370 CUDA files compiled from source |
| 2026-02-09 | Custom vLLM launcher | FastAPI wrapping sync LLM class for Windows |
| 2026-02-09 | vLLM integration tests | 202.8 tok/s concurrent-8 (Qwen2.5-1.5B Q8) |
| 2026-02-11 | ComfyUI module integration | Full lifecycle: download, bootstrap, install, multi-instance |
| 2026-02-11 | ComfyUI backend | Manager (380 lines) + 30 API routes (327 lines) |
| 2026-02-11 | ComfyUI frontend | 913-line module with 4 subtabs, job polling, auto-chain |
| 2026-02-11 | Headless bootstrap fix | Strip Tkinter GUI from install.bat, stdin=DEVNULL |
| | | |

---

## Technical Notes

### n8n Workflow JSON Structure
```json
{
  "name": "My Workflow",
  "nodes": [
    {
      "id": "uuid",
      "name": "Webhook",
      "type": "n8n-nodes-base.webhook",
      "position": [250, 300],
      "parameters": {
        "path": "my-webhook",
        "httpMethod": "POST"
      }
    },
    {
      "id": "uuid2",
      "name": "HTTP Request",
      "type": "n8n-nodes-base.httpRequest",
      "position": [450, 300],
      "parameters": {
        "url": "http://localhost:8000/api/chat/completions",
        "method": "POST",
        "body": "json"
      }
    }
  ],
  "connections": {
    "Webhook": {
      "main": [[{"node": "HTTP Request", "type": "main", "index": 0}]]
    }
  }
}
```

### Key n8n Node Types for AI Workflows
- `n8n-nodes-base.webhook` - Receive HTTP requests
- `n8n-nodes-base.httpRequest` - Call APIs (including local LLM)
- `n8n-nodes-base.scheduleTrigger` - Cron-based triggers
- `n8n-nodes-base.code` - JavaScript processing
- `n8n-nodes-base.set` - Set/transform data
- `n8n-nodes-base.if` - Conditional branching
- `n8n-nodes-base.discord` - Discord integration
- `n8n-nodes-base.slack` - Slack integration

### Local LLM API Call from n8n
```json
{
  "url": "http://localhost:8000/api/chat/completions",
  "method": "POST",
  "headers": {"Content-Type": "application/json"},
  "body": {
    "instance_id": "{{$env.DEFAULT_MODEL_ID}}",
    "messages": [{"role": "user", "content": "{{$json.input}}"}],
    "max_tokens": 500
  }
}
```

---

## Quick Commands

```bash
# Start AgentNate
E:\AgentNate\python\python.exe run.py

# Run tests
E:\AgentNate\python\python.exe test_final_summary.py

# Check GPU status
nvidia-smi --query-gpu=index,name,memory.free --format=csv
```

---

## Ideas Backlog
- Voice input/output (Whisper integration - see chat_ideas.md)
- ~~Image generation integration~~ (Done - ComfyUI module)
- RAG with local documents (embeddings via LM Studio - see chat_ideas.md)
- Browser automation via n8n
- Mobile-friendly UI
- Plugin system for custom tools
- Workflow version control
- Collaborative editing
- Wire tool execution into main chat flow (critical gap)
- Agent swarm coordination
- Settings hot-reload without restart

---

*Update this file as you make progress. It will be here next session.*
