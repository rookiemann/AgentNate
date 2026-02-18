"""
Agent Tools - Sub-agent spawning, monitoring, result retrieval,
persistent memory, and user interaction.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Any, Optional, List

logger = logging.getLogger("agent_tools")


# Default max concurrent sub-agents (configurable via settings)
DEFAULT_MAX_SUB_AGENTS = 4

# Default timeout per sub-agent in seconds
DEFAULT_TIMEOUT = 300  # 5 minutes



TOOL_DEFINITIONS = [
    {
        "name": "spawn_agent",
        "description": "Spawn a parallel sub-agent to work on a task concurrently. The sub-agent runs autonomously with its own conversation and tool access. Use this to parallelize independent tasks (e.g. research multiple topics simultaneously).",
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Clear task description for the sub-agent. Be specific about what you want it to accomplish and return."
                },
                "persona_id": {
                    "type": "string",
                    "description": "Persona for the sub-agent (e.g. 'researcher', 'coder', 'general_assistant'). Defaults to 'general_assistant'."
                },
                "name": {
                    "type": "string",
                    "description": "Short label for this sub-agent (e.g. 'competitor-research', 'data-analysis'). Used in status reports."
                },
                "max_tool_calls": {
                    "type": "integer",
                    "description": "Max tool calls for the sub-agent (default: 8)."
                },
                "instance_id": {
                    "type": "string",
                    "description": "Override: specific loaded model instance_id for this sub-agent. Bypasses routing presets."
                },
            },
            "required": ["task"]
        }
    },
    {
        "name": "batch_spawn_agents",
        "description": "Spawn multiple sub-agents in one call. Example: {\"tool\":\"batch_spawn_agents\",\"arguments\":{\"agents\":[{\"task\":\"Find 3 facts about AI\",\"persona_id\":\"researcher\",\"name\":\"fact-finder\"},{\"task\":\"Write a hello world function\",\"persona_id\":\"coder\",\"name\":\"code-bot\"}]}}",
        "parameters": {
            "type": "object",
            "properties": {
                "agents": {
                    "type": "array",
                    "description": "Array of agent specs. IMPORTANT: task = full task description (a sentence), persona_id = one of: researcher, coder, general_assistant, automator, data_analyst. name = short label.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "Full task description sentence (e.g. 'Find 3 interesting facts about quantum computing')"
                            },
                            "persona_id": {
                                "type": "string",
                                "description": "Must be one of: researcher, coder, general_assistant, automator, data_analyst"
                            },
                            "name": {
                                "type": "string",
                                "description": "Short label like 'fact-finder' or 'code-bot'"
                            },
                            "max_tool_calls": {
                                "type": "integer",
                                "description": "Max tool calls (default: 8)"
                            }
                        },
                        "required": ["task"]
                    }
                }
            },
            "required": ["agents"]
        }
    },
    {
        "name": "check_agents",
        "description": "Check the status of all spawned sub-agents. Returns which are running, completed, or failed.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_agent_result",
        "description": "Get the result from a completed sub-agent. Returns the agent's final response and tool call history.",
        "parameters": {
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "ID of the sub-agent to get results from (from spawn_agent or check_agents)."
                },
            },
            "required": ["agent_id"]
        }
    },
    {
        "name": "remember",
        "description": "Store a fact or preference in persistent memory. Survives across conversations. Use this to remember user preferences, important decisions, or useful information.",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Short identifier for this memory (e.g. 'preferred_language', 'project_structure')"
                },
                "value": {
                    "type": "string",
                    "description": "The fact to remember (e.g. 'User prefers Python over JavaScript')"
                },
                "category": {
                    "type": "string",
                    "description": "Optional grouping: preference, fact, decision, or general (default: general)"
                },
            },
            "required": ["key", "value"]
        }
    },
    {
        "name": "recall",
        "description": "Search persistent memory for stored facts. Use this to look up previously remembered information about the user, project, or past decisions.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search term to find relevant memories (searches keys, values, and categories)"
                },
            },
            "required": ["query"]
        }
    },
    {
        "name": "ask_user",
        "description": "Pause execution and ask the user a question. Use when you need clarification before proceeding. The user will see your question and can respond.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the user"
                },
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of choices (e.g. ['Option A', 'Option B']). User can also type a free response."
                },
            },
            "required": ["question"]
        }
    },
    # --- Model routing tools ---
    {
        "name": "list_routing_presets",
        "description": "List all model routing presets and current routing status. Routing presets map personas to specific models so sub-agents automatically use the best model for their role (e.g., coding model for coder persona, cloud model for orchestration).",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "save_routing_preset",
        "description": "Save a model routing preset. Maps persona IDs to {provider, model_match, label} so spawn_agent auto-routes sub-agents to the right model. Use recommend_routing first to get suggested routes.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Preset name (e.g. 'Cloud Head + Local Coders')"
                },
                "routes": {
                    "type": "object",
                    "description": "Maps persona_id to route. Example: {\"coder\": {\"provider\": \"vllm\", \"model_match\": \"DeepSeek\", \"label\": \"DeepSeek local\"}}"
                },
                "description": {
                    "type": "string",
                    "description": "Optional description of this routing strategy"
                },
                "activate": {
                    "type": "boolean",
                    "description": "Immediately activate this preset after saving (default: true)"
                }
            },
            "required": ["name", "routes"]
        }
    },
    {
        "name": "activate_routing",
        "description": "Enable or disable model routing. When active, spawn_agent auto-routes sub-agents to the best model per persona based on the active routing preset. When disabled, all sub-agents use the parent's model.",
        "parameters": {
            "type": "object",
            "properties": {
                "enabled": {
                    "type": "boolean",
                    "description": "True to enable routing, false to disable"
                },
                "preset_id": {
                    "type": "string",
                    "description": "ID of routing preset to activate (required when enabling)"
                }
            },
            "required": ["enabled"]
        }
    },
    {
        "name": "recommend_routing",
        "description": "Analyze currently loaded models and recommend optimal routing. Maps models to personas based on name keywords and provider type (cloud models for orchestration, coding models for code tasks, etc.). Use save_routing_preset to save the recommendation.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "provision_models",
        "description": "Check what models are needed for a task type and load missing ones from saved presets. Optionally sets up routing automatically. Task types: 'coding_swarm' (needs coding model), 'research' (needs general model), 'image_generation' (needs any model + ComfyUI), 'multi_model' (loads all presets), 'check_only' (report without loading).",
        "parameters": {
            "type": "object",
            "properties": {
                "task_type": {
                    "type": "string",
                    "description": "What you need models for",
                    "enum": ["coding_swarm", "research", "image_generation", "multi_model", "check_only"]
                },
                "auto_route": {
                    "type": "boolean",
                    "description": "Automatically run recommend_routing and save the result (default: true)"
                }
            },
            "required": ["task_type"]
        }
    },
    {
        "name": "generate_preset_workflow",
        "description": "Generate an n8n workflow from a routing preset pattern. Creates workflow JSON that uses runtime routing resolution (works with whatever models are loaded). Patterns: 'swarm' (parallel agents), 'pipeline' (sequential stages), 'multi_coder' (N coders + reviewer), 'image_pipeline' (LLM prompt -> ComfyUI image).",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Workflow pattern",
                    "enum": ["swarm", "pipeline", "multi_coder", "image_pipeline"]
                },
                "config": {
                    "type": "object",
                    "description": "Pattern config. swarm: {personas, task_field, system_prompts}. pipeline: {stages: [{persona_id, system_prompt, output_field}]}. multi_coder: {coder_count, reviewer_persona}. image_pipeline: {prompt_persona}. All accept optional name, webhook_path."
                }
            },
            "required": ["pattern", "config"]
        }
    },
]


class SubAgentState:
    """Tracks state for a single sub-agent."""

    def __init__(self, agent_id: str, name: str, task: str, persona_id: str,
                 conv_id: str, instance_id: str, parent_conv_id: Optional[str] = None,
                 conversation_store=None, race_id: Optional[str] = None,
                 candidate_id: Optional[int] = None, candidate_total: Optional[int] = None):
        self.agent_id = agent_id
        self.name = name
        self.task = task
        self.persona_id = persona_id
        self.conv_id = conv_id
        self.instance_id = instance_id
        self.parent_conv_id = parent_conv_id
        self.conversation_store = conversation_store
        self.status = "running"  # running, completed, failed, aborted, timeout
        self.result = None       # Dict from run_agent_loop
        self.error = None        # Error message if failed
        self.started_at = time.time()
        self.finished_at = None
        self.asyncio_task = None  # The asyncio.Task handle
        self._abort = False
        self.last_progress_at = self.started_at
        self.auto_nudges = 0
        self._guidance_queue: List[str] = []
        self.race_id = race_id
        self.candidate_id = candidate_id
        self.candidate_total = candidate_total
        self.is_race_winner = False
        self.tokens_generated = 0
        self.last_event = "started"

    def should_abort(self) -> bool:
        return self._abort

    def abort(self):
        self._abort = True

    def touch_progress(self):
        self.last_progress_at = time.time()

    def record_event(self, event_type: str, data: Optional[Dict[str, Any]] = None):
        self.last_event = event_type
        if event_type == "token":
            text = (data or {}).get("text", "")
            if text:
                self.tokens_generated += max(1, len(text.strip().split()))
        self.touch_progress()

    def enqueue_guidance(self, message: str):
        if message:
            self._guidance_queue.append(message)
            self.touch_progress()

    def drain_guidance(self) -> List[str]:
        if not self._guidance_queue:
            return []
        items = list(self._guidance_queue)
        self._guidance_queue.clear()
        return items

    def elapsed(self) -> float:
        end = self.finished_at or time.time()
        return round(end - self.started_at, 1)

    def to_status_dict(self) -> Dict[str, Any]:
        d = {
            "agent_id": self.agent_id,
            "name": self.name,
            "task": self.task[:100] + ("..." if len(self.task) > 100 else ""),
            "persona_id": self.persona_id,
            "conversation_id": self.conv_id,
            "instance_id": self.instance_id,
            "status": self.status,
            "elapsed_seconds": self.elapsed(),
            "stale_seconds": round(time.time() - self.last_progress_at, 1),
            "auto_nudges": self.auto_nudges,
            "tokens_generated": self.tokens_generated,
            "tokens_per_second": round(self.tokens_generated / max(0.1, self.elapsed()), 2),
            "last_event": self.last_event,
        }
        if self.race_id:
            d["race_id"] = self.race_id
            d["candidate_id"] = self.candidate_id
            d["candidate_total"] = self.candidate_total
            d["is_race_winner"] = self.is_race_winner
        if self.result:
            d["tool_calls_made"] = self.result.get("tool_call_count", 0)
            d["response_preview"] = self.result.get("response", "")[:200]
            d["response_full"] = self.result.get("response", "")
        if self.error:
            d["error"] = self.error
        return d


class AgentTools:
    """Manages sub-agent spawning, monitoring, memory, and user interaction."""
    _shared_agents: Dict[str, SubAgentState] = {}
    _shared_event_queue: Optional[asyncio.Queue] = None
    _shared_event_subscribers: Dict[str, List[asyncio.Queue]] = {}  # parent_conv_id → subscriber queues

    def __init__(self, orchestrator, n8n_manager, settings, comfyui_manager=None, comfyui_pool=None):
        self.orchestrator = orchestrator
        self.n8n_manager = n8n_manager
        self.settings = settings
        self.comfyui_manager = comfyui_manager
        self.comfyui_pool = comfyui_pool

        # Shared sub-agent state across requests/sessions.
        self._agents = AgentTools._shared_agents


        # Shared event queue for sub-agent progress (drained by SSE generators).
        if AgentTools._shared_event_queue is None:
            AgentTools._shared_event_queue = asyncio.Queue()
        self._event_queue = AgentTools._shared_event_queue

        # Persistent memory
        from backend.agent_memory import AgentMemory
        self.memory = AgentMemory()

        # Routing presets
        from backend.routing_presets import RoutingPresetManager
        self.routing = RoutingPresetManager(settings)

        # Get max concurrency from settings
        self._max_sub_agents = DEFAULT_MAX_SUB_AGENTS
        if settings and hasattr(settings, 'get'):
            val = settings.get("agent.max_sub_agents")
            if val is not None:
                self._max_sub_agents = int(val)

        # Timeout per sub-agent
        self._timeout = DEFAULT_TIMEOUT
        if settings and hasattr(settings, 'get'):
            val = settings.get("agent.sub_agent_timeout")
            if val is not None:
                self._timeout = int(val)

    def _pick_non_openrouter_instance(self) -> Optional[str]:
        """Pick a non-OpenRouter model instance, preferring local providers."""
        instances = self.orchestrator.get_loaded_instances()
        if not instances:
            return None

        preferred_order = ["lm_studio", "vllm", "llama_cpp", "ollama"]

        ready_local = []
        other_non_openrouter = []
        for inst in instances:
            provider = getattr(getattr(inst, "provider_type", None), "value", "").lower()
            if provider == "openrouter":
                continue
            if provider in preferred_order:
                ready_local.append(inst)
            else:
                other_non_openrouter.append(inst)

        if ready_local:
            ready_local.sort(key=lambda x: preferred_order.index(x.provider_type.value.lower()))
            return ready_local[0].id
        if other_non_openrouter:
            return other_non_openrouter[0].id
        return None

    def _score_instance_for_task(self, inst, task: str, persona_id: str) -> int:
        """Heuristic score for assigning a worker model instance."""
        task_l = (task or "").lower()
        persona_l = (persona_id or "").lower()
        display_name = getattr(inst, "display_name", None)
        model_identifier = getattr(inst, "model_identifier", None)
        model_l = ((display_name or model_identifier or "") or "").lower()
        provider = getattr(getattr(inst, "provider_type", None), "value", "").lower()

        score = 0
        # Local-first baseline
        if provider == "lm_studio":
            score += 80
        elif provider == "vllm":
            score += 70
        elif provider == "llama_cpp":
            score += 60
        elif provider == "ollama":
            score += 50
        elif provider == "openrouter":
            score += 10

        # Readiness
        status_v = getattr(getattr(inst, "status", None), "value", "").lower()
        if status_v in ("ready", "loaded", "active"):
            score += 10

        # Persona/task affinity hints
        if persona_l in ("coder", "code_assistant") or any(k in task_l for k in ("code", "debug", "script", "python", "refactor")):
            if any(k in model_l for k in ("coder", "code", "deepseek", "qwen", "starcoder", "codestral")):
                score += 35
            elif any(k in model_l for k in ("instruct", "chat")):
                score += 15
        elif persona_l in ("researcher", "general_assistant", "system_agent", "power_agent"):
            if any(k in model_l for k in ("instruct", "chat", "reason", "mixtral", "llama", "qwen")):
                score += 20

        if any(k in task_l for k in ("comfy", "image", "workflow", "n8n", "automation")):
            if any(k in model_l for k in ("instruct", "chat", "reason")):
                score += 10

        # Prefer larger context when available
        try:
            if int(getattr(inst, "context_length", 0) or 0) >= 8192:
                score += 5
        except Exception:
            pass

        return score

    def _pick_worker_instance_for_task(self, task: str, persona_id: str, current_instance_id: Optional[str]) -> Optional[str]:
        """Choose best currently loaded model for a worker task, local-first."""
        instances = self.orchestrator.get_loaded_instances() or []
        if not instances:
            return current_instance_id

        # Settings gate for local-first worker routing.
        prefer_local = bool(self._setting("agent.workers_prefer_local", True))
        scored = []
        for inst in instances:
            provider = getattr(getattr(inst, "provider_type", None), "value", "").lower()
            if prefer_local and provider == "openrouter":
                # Keep cloud as fallback only.
                pass
            scored.append((self._score_instance_for_task(inst, task, persona_id), inst))

        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0][1] if scored else None
        if not best:
            return current_instance_id

        if prefer_local:
            best_provider = getattr(getattr(best, "provider_type", None), "value", "").lower()
            if best_provider == "openrouter":
                local = self._pick_non_openrouter_instance()
                if local:
                    return local

        return best.id

    def _pick_alternative_instance(self, task: str, persona_id: str, current_instance_id: str) -> Optional[str]:
        """Pick a different loaded model for worker failover."""
        instances = [i for i in (self.orchestrator.get_loaded_instances() or []) if i.id != current_instance_id]
        if not instances:
            return None
        scored = [(self._score_instance_for_task(inst, task, persona_id), inst) for inst in instances]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1].id if scored else None

    def _build_worker_instructions(self, task: str, persona_id: str, resolved_instance_id: str, resolution_method: str) -> str:
        """Compact worker brief — just enough context for the model to act."""
        task_lower = (task or "").lower()
        execution_hint = ""
        if any(k in task_lower for k in ("workflow", "comfy", "n8n", "sdxl", "stable diffusion", "generate image", "image generation")):
            execution_hint = "Execute with real tools — don't just describe a plan.\n"
        return (
            f"Worker agent. Persona: {persona_id}. Model: {resolved_instance_id}.\n"
            f"{execution_hint}"
            "Use tools to complete the task. If blocked, state the blocker and try one diagnostic step.\n"
            f"Task:\n{task}"
        )

    def _running_count(self) -> int:
        return sum(1 for a in self._agents.values() if a.status == "running")

    def _setting(self, key: str, default=None):
        """Settings getter compatible with both get(key) and get(key, default) implementations."""
        if not self.settings or not hasattr(self.settings, "get"):
            return default
        try:
            value = self.settings.get(key, default)
        except TypeError:
            value = self.settings.get(key)
        return default if value is None else value

    @staticmethod
    def _extract_recent_messages(messages: List[Dict[str, Any]], max_items: int = 8,
                                 max_chars: int = 1200) -> str:
        if not messages:
            return ""
        clipped = messages[-max_items:]
        lines = []
        budget = max_chars
        for m in clipped:
            role = m.get("role", "unknown")
            content = (m.get("content") or "").strip().replace("\n", " ")
            if not content:
                continue
            content = content[:300]
            line = f"{role}: {content}"
            if len(line) > budget:
                break
            lines.append(line)
            budget -= len(line)
            if budget <= 0:
                break
        return "\n".join(lines)

    def _build_contextual_nudge(self, state: SubAgentState, head_instance_id: Optional[str] = None) -> str:
        """Deterministic supervisor nudge — no LLM call, just state-based heuristics."""
        elapsed = state.elapsed()
        stale = round(time.time() - state.last_progress_at, 1)

        if stale > 120:
            return (
                f"Supervisor: you have been stale for {stale:.0f}s. "
                "Summarize your blocker in one sentence, then call one concrete tool to make progress."
            )
        if elapsed > 180:
            return (
                f"Supervisor: {elapsed:.0f}s elapsed. Wrap up — return your best result now. "
                "If blocked, state the exact prerequisite that is missing."
            )
        return (
            "Supervisor: you appear stuck. State your blocker, then run the next diagnostic or completion step."
        )

    async def spawn_agent(self, task: str, persona_id: str = "general_assistant",
                          name: str = None, max_tool_calls: int = 8,
                          instance_id: str = None,
                          # Injected by parent context:
                          _instance_id: str = None,
                          _conversation_store=None,
                          _persona_manager=None,
                          _routing_preset_id: str = None,
                          _parent_conv_id: str = None,
                          _race_id: str = None,
                          _candidate_id: int = None,
                          _candidate_total: int = None,
                          _extra_forbidden_tools: Optional[List[str]] = None,
                          ) -> Dict[str, Any]:
        """Spawn a sub-agent to work on a task in parallel."""

        # Validate we have required context
        if not _instance_id or not _conversation_store or not _persona_manager:
            return {
                "success": False,
                "error": "Sub-agent spawning requires model and conversation context. This is an internal error."
            }

        # Normalize weak-model argument drift.
        if not persona_id:
            persona_id = "general_assistant"
        if isinstance(max_tool_calls, bool) or not isinstance(max_tool_calls, int) or max_tool_calls <= 0:
            max_tool_calls = 8

        # --- Model Resolution ---
        resolved_instance_id = _instance_id  # default: parent's model
        resolution_method = "parent"

        if instance_id:  # Priority 1: explicit override from LLM
            if self.orchestrator.get_instance(instance_id):
                resolved_instance_id = instance_id
                resolution_method = "explicit"
            else:
                return {"success": False, "error": f"Instance not found: {instance_id}"}
        else:
            # Priority 2: per-panel routing preset override (passed from frontend)
            preset_id = _routing_preset_id
            # Priority 3: global routing preset from settings
            if not preset_id and self._setting("agent.routing_enabled", False):
                preset_id = self._setting("agent.active_routing_preset_id")
            if preset_id:
                try:
                    routed_id = self.routing.resolve(
                        preset_id, persona_id,
                        self.orchestrator.get_loaded_instances()
                    )
                except Exception as e:
                    logger.warning(f"[spawn] Routing preset resolve failed ({preset_id}): {e}")
                    routed_id = None
                if routed_id:
                    resolved_instance_id = routed_id
                    resolution_method = "routed"

        # Keep worker/sub-agent compute local/non-cloud by default when possible.
        # If parent/routed model is OpenRouter and no explicit override was requested,
        # auto-fallback to a non-OpenRouter loaded model.
        if resolution_method != "explicit":
            resolved_inst = self.orchestrator.get_instance(resolved_instance_id)
            resolved_provider = getattr(getattr(resolved_inst, "provider_type", None), "value", "").lower()
            if resolved_provider == "openrouter":
                fallback_id = self._pick_non_openrouter_instance()
                if fallback_id and fallback_id != resolved_instance_id:
                    resolved_instance_id = fallback_id
                    resolution_method = "auto_non_openrouter"

        # Task-aware worker model assignment (local-first) for non-explicit requests.
        if resolution_method != "explicit":
            smart_pick = self._pick_worker_instance_for_task(task, persona_id, resolved_instance_id)
            if smart_pick and smart_pick != resolved_instance_id:
                resolved_instance_id = smart_pick
                resolution_method = "task_heuristic"

        # Check concurrency limit
        running = self._running_count()
        if running >= self._max_sub_agents:
            return {
                "success": False,
                "error": f"Maximum concurrent sub-agents reached ({self._max_sub_agents}). "
                         f"Wait for running agents to complete or get their results first.",
                "running_agents": [
                    a.to_status_dict() for a in self._agents.values()
                    if a.status == "running"
                ]
            }

        # Validate persona
        persona = _persona_manager.get(persona_id)
        if not persona:
            return {
                "success": False,
                "error": f"Persona '{persona_id}' not found. Available: {[p.id for p in _persona_manager.list_all()]}"
            }

        # Generate IDs
        agent_id = f"sub-{uuid.uuid4().hex[:8]}"
        if not name:
            name = agent_id

        # Create a fresh conversation for this sub-agent
        conv_id = _conversation_store.create(
            persona_id=persona_id,
            model_id=resolved_instance_id,
        )
        _conversation_store.set_model(conv_id, resolved_instance_id)

        # Create state tracker
        state = SubAgentState(
            agent_id=agent_id,
            name=name,
            task=task,
            persona_id=persona_id,
            conv_id=conv_id,
            instance_id=resolved_instance_id,
            parent_conv_id=_parent_conv_id,
            conversation_store=_conversation_store,
            race_id=_race_id,
            candidate_id=_candidate_id,
            candidate_total=_candidate_total,
        )
        self._agents[agent_id] = state

        logger.info(
            f"[spawn] Spawning sub-agent '{name}' ({agent_id}) with persona={persona_id}, "
            f"model={resolved_instance_id}, resolution={resolution_method}"
        )

        # Event callback that queues progress updates for the parent SSE stream
        def _on_sub_event(event_type, data):
            state.record_event(event_type, data)
            event_payload = {
                "agent_id": agent_id,
                "name": name,
                "parent_conv_id": _parent_conv_id,
                "event": event_type,
                "race_id": state.race_id,
                "candidate_id": state.candidate_id,
                "candidate_total": state.candidate_total,
                **data,
            }
            self._event_queue.put_nowait(event_payload)
            # Push to SSE subscribers for this conversation
            subs = AgentTools._shared_event_subscribers.get(_parent_conv_id, [])
            for q in subs:
                try:
                    q.put_nowait(event_payload)
                except asyncio.QueueFull:
                    pass  # Drop if subscriber is slow

        # Launch the agent loop as a background task
        async def _run():
            try:
                from backend.agent_loop import run_agent_loop
                from backend.tools import ToolRouter

                # Each sub-agent gets its own ToolRouter (stateless, cheap)
                sub_router = ToolRouter(
                    self.orchestrator,
                    self.n8n_manager,
                    self.settings,
                    self.comfyui_manager,
                    getattr(self, 'media_catalog', None),
                    comfyui_pool=getattr(self, 'comfyui_pool', None),
                )

                # Tool-level race settings
                _race_enabled = bool(self._setting("agent.tool_race_enabled", True))
                _race_candidates = int(self._setting("agent.tool_race_candidates", 3))

                result = await asyncio.wait_for(
                    run_agent_loop(
                        orchestrator=self.orchestrator,
                        tool_router=sub_router,
                        conversation_store=_conversation_store,
                        persona=persona,
                        persona_manager=_persona_manager,
                        instance_id=resolved_instance_id,
                        message=task,
                        conv_id=conv_id,
                        max_tool_calls=max_tool_calls,
                        additional_instructions=self._build_worker_instructions(
                            task=task,
                            persona_id=persona_id,
                            resolved_instance_id=resolved_instance_id,
                            resolution_method=resolution_method,
                        ),
                        abort_check=state.should_abort,
                        guidance_provider=state.drain_guidance,
                        instance_id_provider=lambda: state.instance_id,
                        extra_forbidden_tools=_extra_forbidden_tools,
                        on_event=_on_sub_event,
                        race_enabled=_race_enabled,
                        race_candidates=_race_candidates,
                    ),
                    timeout=self._timeout,
                )

                state.result = result
                state.status = "aborted" if result.get("aborted") else "completed"
                state.finished_at = time.time()

                logger.info(
                    f"[spawn] Sub-agent '{name}' ({agent_id}) {state.status} "
                    f"in {state.elapsed()}s with {result.get('tool_call_count', 0)} tool calls"
                )

            except asyncio.TimeoutError:
                state.status = "timeout"
                state.error = f"Sub-agent timed out after {self._timeout}s"
                state.finished_at = time.time()
                logger.warning(f"[spawn] Sub-agent '{name}' ({agent_id}) timed out after {self._timeout}s")

            except Exception as e:
                state.status = "failed"
                state.error = str(e)
                state.finished_at = time.time()
                logger.error(f"[spawn] Sub-agent '{name}' ({agent_id}) failed: {e}")

            finally:
                # Push terminal status + full response to SSE subscribers
                done_data = {
                    "status": state.status,
                    "elapsed_seconds": state.elapsed(),
                }
                if state.result:
                    resp_text = state.result.get("response", "")
                    done_data["response"] = resp_text
                    done_data["tool_call_count"] = state.result.get("tool_call_count", 0)
                    logger.info(f"[spawn] agent_done: status={state.status}, response_len={len(resp_text)}, first_100={resp_text[:100]!r}")
                else:
                    logger.warning(f"[spawn] agent_done: status={state.status}, NO result (state.result is None)")
                _on_sub_event("agent_done", done_data)

        task_handle = asyncio.create_task(_run())
        state.asyncio_task = task_handle

        return {
            "success": True,
            "agent_id": agent_id,
            "name": name,
            "persona_id": persona_id,
            "conversation_id": conv_id,
            "model_resolution": resolution_method,
            "instance_id": resolved_instance_id,
            "message": f"Sub-agent '{name}' spawned and running. Use check_agents to monitor progress, get_agent_result to retrieve results when done.",
            "running_count": self._running_count(),
            "max_concurrent": self._max_sub_agents,
        }

    async def batch_spawn_agents(self, agents: list = None,
                                  tasks: list = None,  # alias for agents (8B models use this)
                                  # Injected by parent context:
                                  _instance_id: str = None,
                                  _conversation_store=None,
                                  _persona_manager=None,
                                  _routing_preset_id: str = None,
                                  _parent_conv_id: str = None,
                                  ) -> Dict[str, Any]:
        """Spawn multiple sub-agents in one call (8B-model-friendly batch)."""
        # Accept both 'agents' and 'tasks' keys (8B models often use 'tasks')
        agents = agents or tasks
        if not agents or not isinstance(agents, list):
            return {"success": False, "error": "agents must be a non-empty list of {task, persona_id, name}"}

        # Valid persona IDs for fuzzy matching
        _VALID_PERSONAS = {
            "researcher", "coder", "general_assistant", "automator",
            "data_analyst", "vision_agent", "codebase_guide",
            "image_creator", "ai_creative", "workflow_builder",
            "code_assistant", "power_agent", "system_agent",
        }
        _PERSONA_ALIASES = {
            "research": "researcher", "code": "coder", "coding": "coder",
            "writer": "general_assistant", "writing": "general_assistant",
            "assistant": "general_assistant", "general": "general_assistant",
            "data": "data_analyst", "analyst": "data_analyst",
            "image": "image_creator", "creative": "ai_creative",
            "vision": "vision_agent", "automation": "automator",
        }

        def _resolve_persona(raw: str) -> str:
            if not raw:
                return "general_assistant"
            raw = raw.strip().lower().replace("-", "_").replace(" ", "_")
            if raw in _VALID_PERSONAS:
                return raw
            if raw in _PERSONA_ALIASES:
                return _PERSONA_ALIASES[raw]
            # Substring match
            for pid in _VALID_PERSONAS:
                if raw in pid or pid in raw:
                    return pid
            return "general_assistant"

        results = []
        for spec in agents:
            if isinstance(spec, str):
                spec = {"task": spec}
            if not isinstance(spec, dict) or "task" not in spec:
                results.append({"success": False, "error": "Each agent needs at least a 'task' field"})
                continue
            # Resolve persona_id with fuzzy matching
            spec["persona_id"] = _resolve_persona(spec.get("persona_id", "general_assistant"))
            result = await self.spawn_agent(
                task=spec["task"],
                persona_id=spec.get("persona_id", "general_assistant"),
                name=spec.get("name"),
                max_tool_calls=spec.get("max_tool_calls", 8),
                instance_id=spec.get("instance_id"),
                _instance_id=_instance_id,
                _conversation_store=_conversation_store,
                _persona_manager=_persona_manager,
                _routing_preset_id=_routing_preset_id,
                _parent_conv_id=_parent_conv_id,
            )
            results.append(result)

        spawned = [r for r in results if r.get("success")]
        failed = [r for r in results if not r.get("success")]
        return {
            "success": len(spawned) > 0,
            "spawned": len(spawned),
            "failed": len(failed),
            "agents": results,
            "message": f"Spawned {len(spawned)}/{len(agents)} sub-agents. Use check_agents to monitor, get_agent_result to retrieve results.",
        }

    async def check_agents(self, _parent_conv_id: str = None) -> Dict[str, Any]:
        """Check status of all sub-agents."""
        if not self._agents:
            return {
                "success": True,
                "agents": [],
                "message": "No sub-agents have been spawned.",
                "summary": {"running": 0, "completed": 0, "failed": 0},
            }

        states = list(self._agents.values())
        if _parent_conv_id:
            states = [s for s in states if s.parent_conv_id == _parent_conv_id]
        if not states:
            return {
                "success": True,
                "agents": [],
                "message": "No sub-agents have been spawned for this conversation.",
                "summary": {"running": 0, "completed": 0, "failed": 0, "timeout": 0, "aborted": 0},
                "max_concurrent": self._max_sub_agents,
            }

        agents = []
        summary = {"running": 0, "completed": 0, "failed": 0, "timeout": 0, "aborted": 0}

        for state in states:
            agents.append(state.to_status_dict())
            if state.status in summary:
                summary[state.status] += 1

        return {
            "success": True,
            "agents": agents,
            "summary": summary,
            "max_concurrent": self._max_sub_agents,
        }

    async def get_agent_result(self, agent_id: str, _parent_conv_id: str = None) -> Dict[str, Any]:
        """Get the result from a sub-agent."""
        state = self._agents.get(agent_id)

        if not state or (_parent_conv_id and state.parent_conv_id != _parent_conv_id):
            available = list(self._agents.keys())
            if _parent_conv_id:
                available = [aid for aid, s in self._agents.items() if s.parent_conv_id == _parent_conv_id]
            return {
                "success": False,
                "error": f"Sub-agent '{agent_id}' not found. Use check_agents to see available agents.",
                "available": available,
            }

        if state.status == "running":
            return {
                "success": False,
                "error": f"Sub-agent '{state.name}' is still running ({state.elapsed()}s elapsed). Wait for it to complete.",
                "status": state.to_status_dict(),
            }

        if state.status == "failed":
            return {
                "success": False,
                "agent_id": agent_id,
                "name": state.name,
                "status": "failed",
                "error": state.error,
                "elapsed_seconds": state.elapsed(),
            }

        if state.status == "timeout":
            return {
                "success": False,
                "agent_id": agent_id,
                "name": state.name,
                "status": "timeout",
                "error": state.error,
                "elapsed_seconds": state.elapsed(),
            }

        # Completed or aborted — return result
        result = state.result or {}
        return {
            "success": True,
            "agent_id": agent_id,
            "name": state.name,
            "status": state.status,
            "response": result.get("response", ""),
            "tool_calls": [
                {"tool": tc["tool"], "success": tc.get("result", {}).get("success", True) if isinstance(tc.get("result"), dict) else True}
                for tc in result.get("tool_calls", [])
            ],
            "tool_call_count": result.get("tool_call_count", 0),
            "conversation_id": result.get("conversation_id", ""),
            "elapsed_seconds": state.elapsed(),
        }

    async def switch_worker_model(self, agent_id: str, instance_id: str, _parent_conv_id: str = None) -> Dict[str, Any]:
        """Switch a worker to a different loaded model instance."""
        state = self._agents.get(agent_id)
        if not state or (_parent_conv_id and state.parent_conv_id != _parent_conv_id):
            return {"success": False, "error": f"Sub-agent '{agent_id}' not found."}

        target = self.orchestrator.get_instance(instance_id)
        if not target:
            return {"success": False, "error": f"Instance not found: {instance_id}"}

        previous = state.instance_id
        if previous == instance_id:
            return {
                "success": True,
                "agent_id": agent_id,
                "instance_id": instance_id,
                "message": "Worker already using requested model.",
            }

        state.instance_id = instance_id
        state.enqueue_guidance(
            f"Supervisor manually switched your model from {previous} to {instance_id}. "
            "Continue task execution using current context."
        )
        self._event_queue.put_nowait({
            "parent_conv_id": state.parent_conv_id,
            "agent_id": state.agent_id,
            "name": state.name,
            "event": "worker_model_switched",
            "from_instance_id": previous,
            "to_instance_id": instance_id,
        })
        return {
            "success": True,
            "agent_id": agent_id,
            "from_instance_id": previous,
            "to_instance_id": instance_id,
        }

    async def supervise_workers(self, parent_conv_id: str = None, stuck_after_s: int = 45,
                                nudge_after_s: int = 75, max_auto_nudges: int = 2,
                                head_instance_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Lightweight supervisor heartbeat for running sub-agents.
        Detects stale workers and queues guidance nudges to keep autonomy moving.
        """
        now = time.time()
        running_states = [
            s for s in self._agents.values()
            if s.status == "running" and (not parent_conv_id or s.parent_conv_id == parent_conv_id)
        ]

        inspected = []
        nudged = []
        switched = []
        allow_failover = bool(self._setting("agent.worker_model_failover", True))
        for state in running_states:
            elapsed = now - state.started_at
            stale = now - state.last_progress_at
            looks_stuck = elapsed >= stuck_after_s and stale >= nudge_after_s
            action = "none"

            # Optional extreme-case model failover for stuck workers.
            if looks_stuck and allow_failover and state.auto_nudges >= 1:
                alt = self._pick_alternative_instance(state.task, state.persona_id, state.instance_id)
                if alt and alt != state.instance_id:
                    prev = state.instance_id
                    state.instance_id = alt
                    switched.append({
                        "agent_id": state.agent_id,
                        "name": state.name,
                        "from_instance_id": prev,
                        "to_instance_id": alt,
                    })
                    state.enqueue_guidance(
                        f"Supervisor switched your model from {prev} to {alt} due to stalling. "
                        "Continue from current context and execute the next concrete step."
                    )
                    self._event_queue.put_nowait({
                        "parent_conv_id": state.parent_conv_id,
                        "agent_id": state.agent_id,
                        "name": state.name,
                        "event": "worker_model_switched",
                        "from_instance_id": prev,
                        "to_instance_id": alt,
                    })
                    action = "model_switched"

            if looks_stuck and state.auto_nudges < max_auto_nudges:
                state.auto_nudges += 1
                action = "nudged"
                guidance = self._build_contextual_nudge(state, head_instance_id=head_instance_id)
                state.enqueue_guidance(guidance)
                nudged.append({
                    "agent_id": state.agent_id,
                    "name": state.name,
                    "auto_nudges": state.auto_nudges,
                    "elapsed_seconds": round(elapsed, 1),
                    "stale_seconds": round(stale, 1),
                })
                self._event_queue.put_nowait({
                    "parent_conv_id": state.parent_conv_id,
                    "agent_id": state.agent_id,
                    "name": state.name,
                    "event": "supervisor_nudge",
                    "auto_nudges": state.auto_nudges,
                    "elapsed_seconds": round(elapsed, 1),
                    "stale_seconds": round(stale, 1),
                })

            inspected.append({
                "agent_id": state.agent_id,
                "name": state.name,
                "elapsed_seconds": round(elapsed, 1),
                "stale_seconds": round(stale, 1),
                "auto_nudges": state.auto_nudges,
                "action": action,
            })

        return {
            "success": True,
            "running_workers": len(running_states),
            "nudged_workers": nudged,
            "switched_workers": switched,
            "inspected": inspected,
            "stuck_after_s": stuck_after_s,
            "nudge_after_s": nudge_after_s,
            "max_auto_nudges": max_auto_nudges,
        }

    def abort_all(self, parent_conv_id: str = None):
        """Abort running sub-agents, optionally only those belonging to one parent conversation."""
        for state in self._agents.values():
            if parent_conv_id and state.parent_conv_id != parent_conv_id:
                continue
            if state.status == "running":
                state.abort()
                logger.info(f"[spawn] Aborting sub-agent '{state.name}' ({state.agent_id})")

    def cleanup(self):
        """Clean up completed/failed agents to free memory."""
        to_remove = [
            aid for aid, state in self._agents.items()
            if state.status in ("completed", "failed", "timeout", "aborted")
        ]
        for aid in to_remove:
            del self._agents[aid]
        return len(to_remove)

    # ---- Event Queue (for sub-agent progress visibility) ----

    def drain_events(self, parent_conv_id: str = None, max_events: int = 500) -> List[Dict]:
        """Drain queued sub-agent events.

        If parent_conv_id is set, keep non-matching events queued.
        max_events caps per-drain payload so UI polls stay responsive under high token throughput.
        """
        events = []
        keep = []
        while not self._event_queue.empty():
            try:
                event = self._event_queue.get_nowait()
                if parent_conv_id and event.get("parent_conv_id") != parent_conv_id:
                    keep.append(event)
                else:
                    if max_events and len(events) >= max_events:
                        keep.append(event)
                        break
                    events.append(event)
            except asyncio.QueueEmpty:
                break
        for event in keep:
            self._event_queue.put_nowait(event)
        return events

    def get_all_agent_status(self, parent_conv_id: str = None) -> List[Dict]:
        """Get status of agents, optionally filtered by parent conversation."""
        states = self._agents.values()
        if parent_conv_id:
            states = [s for s in states if s.parent_conv_id == parent_conv_id]
        return [state.to_status_dict() for state in states]

    # ---- SSE Event Subscriptions (real-time streaming to frontend) ----

    def subscribe_events(self, parent_conv_id: str) -> asyncio.Queue:
        """Subscribe to real-time worker events for a conversation. Returns a Queue."""
        q = asyncio.Queue(maxsize=2000)
        if parent_conv_id not in AgentTools._shared_event_subscribers:
            AgentTools._shared_event_subscribers[parent_conv_id] = []
        AgentTools._shared_event_subscribers[parent_conv_id].append(q)
        logger.debug(f"[subscribe] {parent_conv_id} now has {len(AgentTools._shared_event_subscribers[parent_conv_id])} subscribers")
        return q

    def unsubscribe_events(self, parent_conv_id: str, q: asyncio.Queue):
        """Remove an event subscription."""
        subs = AgentTools._shared_event_subscribers.get(parent_conv_id, [])
        try:
            subs.remove(q)
        except ValueError:
            pass
        if not subs and parent_conv_id in AgentTools._shared_event_subscribers:
            del AgentTools._shared_event_subscribers[parent_conv_id]
        logger.debug(f"[unsubscribe] {parent_conv_id} now has {len(AgentTools._shared_event_subscribers.get(parent_conv_id, []))} subscribers")

    # ---- Persistent Memory Tools ----

    async def remember(self, key: str, value: str, category: str = "general") -> Dict[str, Any]:
        """Store a fact in persistent memory."""
        entry = self.memory.store(key, value, category)
        return {
            "success": True,
            "message": f"Remembered: {key}",
            "entry": entry,
            "total_memories": self.memory.count(),
        }

    async def recall(self, query: str) -> Dict[str, Any]:
        """Search persistent memory for stored facts."""
        matches = self.memory.recall(query)
        if not matches:
            return {
                "success": True,
                "matches": [],
                "message": f"No memories found matching '{query}'.",
                "total_memories": self.memory.count(),
            }
        return {
            "success": True,
            "matches": matches,
            "message": f"Found {len(matches)} matching memories.",
            "total_memories": self.memory.count(),
        }

    # ---- Ask User Tool ----

    async def ask_user(self, question: str, options: List[str] = None) -> Dict[str, Any]:
        """
        Returns the question structure. The actual pause/resume is handled
        by the SSE generator in routes/tools.py which detects this tool
        and yields an ask_user event.
        """
        return {
            "success": True,
            "type": "ask_user",
            "question": question,
            "options": options or [],
        }

    # ---- Model Routing Tools ----

    async def list_routing_presets(self) -> Dict[str, Any]:
        """List all routing presets and current routing status."""
        presets = self.routing.list_presets()
        return {
            "success": True,
            "presets": presets,
            "routing_enabled": self._setting("agent.routing_enabled", False),
            "active_preset_id": self._setting("agent.active_routing_preset_id"),
        }

    async def save_routing_preset(self, name: str, routes: dict,
                                   description: str = "",
                                   activate: bool = True) -> Dict[str, Any]:
        """Save a routing preset and optionally activate it."""
        if not name or not routes:
            return {"success": False, "error": "name and routes are required."}

        preset = self.routing.save_preset(name, routes, description)

        if activate:
            self.settings.set("agent.routing_enabled", True)
            self.settings.set("agent.active_routing_preset_id", preset["id"])

        return {
            "success": True,
            "preset": preset,
            "activated": activate,
        }

    async def activate_routing(self, enabled: bool,
                                preset_id: str = None) -> Dict[str, Any]:
        """Enable or disable model routing."""
        if enabled:
            if not preset_id:
                return {
                    "success": False,
                    "error": "preset_id is required when enabling routing.",
                }
            preset = self.routing.get_preset(preset_id)
            if not preset:
                return {
                    "success": False,
                    "error": f"Routing preset not found: {preset_id}",
                }
            self.settings.set("agent.routing_enabled", True)
            self.settings.set("agent.active_routing_preset_id", preset_id)
            return {
                "success": True,
                "routing_enabled": True,
                "active_preset": preset["name"],
                "routes": preset.get("routes", {}),
            }
        else:
            self.settings.set("agent.routing_enabled", False)
            self.settings.set("agent.active_routing_preset_id", None)
            return {
                "success": True,
                "routing_enabled": False,
                "message": "Routing disabled. Sub-agents will use the parent model.",
            }

    async def recommend_routing(self) -> Dict[str, Any]:
        """Analyze loaded models and recommend persona→model routing."""
        instances = self.orchestrator.get_loaded_instances()
        return self.routing.recommend(instances)

    # ---- Model Provisioning ----

    async def provision_models(self, task_type: str, auto_route: bool = True) -> Dict[str, Any]:
        """Check what models are needed and load missing ones from presets."""
        from backend.tools.model_tools import ModelTools
        model_tools = ModelTools(self.orchestrator)

        # Check what's currently loaded
        loaded = await model_tools.list_loaded_models()
        loaded_models = loaded.get("instances", [])
        loaded_count = len(loaded_models)

        # Check available presets
        presets_result = await model_tools.list_model_presets()
        available_presets = presets_result.get("presets", [])

        if task_type == "check_only":
            recommendation = self.routing.recommend(self.orchestrator.get_loaded_instances())
            return {
                "success": True,
                "loaded_models": loaded_models,
                "loaded_count": loaded_count,
                "available_presets": available_presets,
                "routing_recommendation": recommendation,
            }

        actions_taken = []
        _CODING_KW = ["code", "deepseek", "starcoder", "codellama", "codestral"]

        def _has_coding():
            return any(
                any(kw in (m.get("model", "") + " " + m.get("name", "")).lower() for kw in _CODING_KW)
                for m in loaded_models
            )

        needs_coding = task_type in ("coding_swarm", "multi_model")
        needs_general = task_type in ("research", "multi_model", "image_generation")

        if needs_coding and not _has_coding():
            coding_preset = next(
                (p for p in available_presets if any(
                    kw in (p.get("model_name") or "").lower() for kw in _CODING_KW)),
                None,
            )
            if coding_preset:
                result = await model_tools.load_from_preset(coding_preset["name"])
                actions_taken.append(
                    f"Loaded coding model: {coding_preset['name']} (success={result.get('success')})"
                )

        if needs_general and loaded_count == 0:
            general_preset = next(
                (p for p in available_presets if not any(
                    kw in (p.get("model_name") or "").lower() for kw in _CODING_KW)),
                None,
            )
            if general_preset:
                result = await model_tools.load_from_preset(general_preset["name"])
                actions_taken.append(
                    f"Loaded general model: {general_preset['name']} (success={result.get('success')})"
                )

        if task_type == "multi_model":
            for preset in available_presets:
                name = preset.get("model_name", "").lower()
                already = any(
                    name in (m.get("model", "").lower()) for m in loaded_models
                )
                if not already:
                    result = await model_tools.load_from_preset(preset["name"])
                    actions_taken.append(
                        f"Loaded: {preset['name']} (success={result.get('success')})"
                    )

        # Auto-route
        routing_result = None
        if auto_route:
            instances = self.orchestrator.get_loaded_instances()
            if instances:
                recommendation = self.routing.recommend(instances)
                if recommendation.get("success") and recommendation.get("recommended_routes"):
                    preset = self.routing.save_preset(
                        f"Auto-{task_type}",
                        recommendation["recommended_routes"],
                        f"Auto-provisioned for {task_type}",
                    )
                    self.settings.set("agent.routing_enabled", True)
                    self.settings.set("agent.active_routing_preset_id", preset["id"])
                    routing_result = {
                        "preset_id": preset["id"],
                        "preset_name": preset["name"],
                        "routes": recommendation["recommended_routes"],
                    }
                    actions_taken.append(f"Activated routing preset: {preset['name']}")

        return {
            "success": True,
            "task_type": task_type,
            "actions_taken": actions_taken,
            "loaded_models_after": len(self.orchestrator.get_loaded_instances()),
            "routing": routing_result,
            "message": (
                f"Provisioning complete. {len(actions_taken)} actions taken."
                if actions_taken
                else "All needed models already loaded."
            ),
        }

    # ---- Preset-to-Workflow Bridge ----

    async def generate_preset_workflow(self, pattern: str, config: dict) -> Dict[str, Any]:
        """Generate an n8n workflow from the active routing preset + pattern."""
        preset_id = self._setting("agent.active_routing_preset_id")
        if not preset_id:
            return {
                "success": False,
                "error": "No active routing preset. Use save_routing_preset or activate_routing first.",
            }

        config = dict(config)
        config["preset_id"] = preset_id
        preset = self.routing.get_preset(preset_id)
        if preset:
            config.setdefault("name", f"{preset['name']} - {pattern}")

        from backend.workflow_bridge import generate_workflow, PATTERNS
        try:
            workflow_json = generate_workflow(pattern, config)
            return {
                "success": True,
                "workflow": workflow_json,
                "pattern": pattern,
                "preset_id": preset_id,
                "hint": "Use deploy_workflow to deploy this, or flash_workflow to deploy+run+cleanup in one step.",
            }
        except ValueError as e:
            return {"success": False, "error": str(e), "available_patterns": list(PATTERNS.keys())}
