"""
Agent Loop - Reusable agent execution loop.

Extracted core logic for running an agent conversation with tool calling.
Used by both the SSE streaming endpoint (parent agents) and sub-agent spawning.
"""

import json
import logging
from typing import Dict, Any, Optional, List, Callable

logger = logging.getLogger("agent_loop")


async def run_agent_loop(
    orchestrator,
    tool_router,
    conversation_store,
    persona,
    persona_manager,
    instance_id: str,
    message: str,
    conv_id: str,
    max_tool_calls: int = 10,
    additional_instructions: str = None,
    abort_check: Callable[[], bool] = None,
    guidance_provider: Callable[[], List[str]] = None,
    instance_id_provider: Callable[[], Optional[str]] = None,
    extra_forbidden_tools: Optional[List[str]] = None,
    on_event: Callable[[str, Dict], None] = None,
    race_enabled: bool = False,
    race_candidates: int = 3,
) -> Dict[str, Any]:
    """
    Run a full agent conversation loop: message → LLM → tool → repeat → result.

    This is the core execution engine used by sub-agents. It runs the same
    logic as agent_chat_stream but returns a result dict instead of yielding SSE.

    Args:
        orchestrator: The model orchestrator for LLM inference
        tool_router: ToolRouter instance for executing tools
        conversation_store: Conversation persistence store
        persona: The persona object defining agent identity and tools
        persona_manager: For resolving persona tools
        instance_id: Which loaded model to use
        message: The user/parent message to process
        conv_id: Conversation ID (pre-created by caller)
        max_tool_calls: Max tools before forcing completion
        additional_instructions: Extra instructions layered on system prompt
        abort_check: Optional callback returning True if agent should abort
        guidance_provider: Optional callback returning queued supervisor guidance strings
        on_event: Optional callback for progress events (type, data)

    Returns:
        {
            "success": bool,
            "conversation_id": str,
            "response": str,           # Final agent response text
            "tool_calls": list,         # [{tool, arguments, result}, ...]
            "aborted": bool,
        }
    """
    from backend.agent_intelligence import (
        WorkingMemory, RetryState, ErrorType,
        categorize_error, build_retry_prompt,
        detect_thinking_content, is_likely_thinking_model,
        should_summarize, summarize_old_messages,
        truncate_tool_result, update_working_memory,
    )
    from providers.base import InferenceRequest, ChatMessage

    def emit(event_type: str, data: Dict = None):
        if on_event:
            try:
                on_event(event_type, data or {})
            except Exception:
                pass

    # Tools that change system state and require prompt rebuild
    SYSTEM_CHANGING_TOOLS = {
        "load_model", "unload_model", "spawn_n8n", "stop_n8n", "quick_setup",
        "comfyui_start_api", "comfyui_stop_api", "comfyui_install",
        "comfyui_add_instance", "comfyui_start_instance", "comfyui_stop_instance",
        "provision_models", "load_from_preset", "flash_workflow",
        "configure_workflow",
    }

    # Tools that sub-agents must NOT use
    FORBIDDEN_TOOLS = {
        "spawn_agent", "check_agents", "get_agent_result", "ask_user",
    }
    if extra_forbidden_tools:
        FORBIDDEN_TOOLS = FORBIDDEN_TOOLS.union(set(extra_forbidden_tools))

    # ---- INIT ----
    conv_metadata = conversation_store.get_metadata(conv_id)
    memory = WorkingMemory.from_dict(conv_metadata.get("working_memory"))
    retry_state = RetryState()

    is_thinking = conv_metadata.get("is_thinking_model")
    if is_thinking is None:
        is_thinking = is_likely_thinking_model(instance_id or "")
        conversation_store.set_metadata(conv_id, "is_thinking_model", is_thinking)

    # Build system prompt
    dynamic_state = await tool_router.build_dynamic_prompt(persona)
    tools_prompt = tool_router.get_tools_prompt_for_persona(persona)

    # Enforce persona tool restrictions server-side for sub-agent execution.
    allowed_tools = persona_manager.get_tools_for_persona(persona, tool_router.get_tool_list())
    tool_router.set_allowed_tools(allowed_tools if persona.tools else [])

    from backend.routes.tools import _build_system_prompt
    system_prompt = _build_system_prompt(
        persona, dynamic_state, tools_prompt,
        additional_instructions, working_memory=memory
    )

    # Add user message
    conversation_store.append_message(conv_id, "user", message)

    # Track state
    tool_calls_made = []
    tool_call_count = 0
    recent_tool_hashes = []  # For loop detection
    loop_warning_count = 0  # Hard-stop after 2 loop warnings
    final_response = ""
    aborted = False

    emit("started", {"conv_id": conv_id, "message": message})

    # ---- EXECUTION LOOP ----
    while True:
        # Check abort
        if abort_check and abort_check():
            logger.info(f"[agent_loop] Sub-agent {conv_id} aborted")
            aborted = True
            break

        # Apply any supervisor guidance (head-agent heartbeat/nudges).
        if guidance_provider:
            try:
                guidance_items = guidance_provider() or []
            except Exception:
                guidance_items = []
            if guidance_items:
                guidance_text = "\n".join(f"- {g}" for g in guidance_items)
                conversation_store.append_message(
                    conv_id, "user",
                    "Supervisor guidance received:\n"
                    f"{guidance_text}\n\n"
                    "Continue autonomously and prioritize unblocking progress."
                )
                emit("supervisor_guidance", {"count": len(guidance_items)})

        # Context management
        recent_messages = conversation_store.get_messages(conv_id, limit=20)
        if should_summarize(recent_messages):
            cached_summary = conv_metadata.get("context_summary")
            msg_count = len(conversation_store.get_messages(conv_id, limit=0))
            last_summarized_at = conv_metadata.get("summarized_at_count", 0)

            if not cached_summary or msg_count - last_summarized_at >= 6:
                recent_messages = await summarize_old_messages(
                    orchestrator, instance_id, recent_messages, keep=6
                )
                if recent_messages and recent_messages[0].get("role") == "system":
                    conversation_store.update_metadata(conv_id, {
                        "context_summary": recent_messages[0]["content"],
                        "summarized_at_count": msg_count,
                    })
                    conv_metadata["context_summary"] = recent_messages[0]["content"]
                    conv_metadata["summarized_at_count"] = msg_count
            elif cached_summary:
                keep_msgs = recent_messages[-6:] if len(recent_messages) > 6 else recent_messages
                recent_messages = [{"role": "system", "content": cached_summary}] + keep_msgs

        # Build messages
        messages = [
            ChatMessage(role="system", content=system_prompt)
        ] + [
            ChatMessage(role=m["role"], content=m["content"])
            for m in recent_messages
        ]

        # Allow supervisor to switch worker model between loop iterations.
        current_instance_id = instance_id_provider() if instance_id_provider else None
        if not current_instance_id:
            current_instance_id = instance_id

        # Generate LLM response
        inference_request = InferenceRequest(
            messages=messages,
            max_tokens=4096,
            temperature=persona.temperature,
            top_p=0.95,
            top_k=40,
            repeat_penalty=1.1,
        )

        full_response = ""
        async for response in orchestrator.chat(current_instance_id, inference_request):
            if response.text:
                full_response += response.text
                emit("token", {"text": response.text})

        # Handle thinking models
        if is_thinking:
            thinking_content, clean_response = detect_thinking_content(full_response)
            if thinking_content:
                full_response = clean_response

        # Check for tool calls — parse first, then race or execute
        tool_result = None
        if persona.tools:
            if race_enabled and race_candidates >= 2:
                # Parse WITHOUT executing to check if raceable
                from backend.race_executor import run_tool_race, RACEABLE_TOOLS as _RACEABLE
                parsed = tool_router.parse_tool_call(full_response)
                if parsed and parsed.get("known", True) and parsed["tool"] in _RACEABLE:
                    # Race ALL candidates in parallel (C1 + variants)
                    logger.info(f"[agent_loop] Racing {race_candidates} candidates for {parsed['tool']}")
                    recent_msgs = conversation_store.get_messages(conv_id, limit=20)
                    tool_result = await run_tool_race(
                        tool_router=tool_router,
                        orchestrator=orchestrator,
                        instance_id=current_instance_id,
                        tool_name=parsed["tool"],
                        original_args=parsed["arguments"],
                        original_response=full_response,
                        conversation_messages=recent_msgs,
                        system_prompt=system_prompt,
                        persona=persona,
                        emit=emit,
                        num_candidates=race_candidates,
                    )
                else:
                    # Not raceable — normal parse + execute
                    tool_result = await tool_router.parse_and_execute(full_response)
            else:
                tool_result = await tool_router.parse_and_execute(full_response)

        # Filter forbidden tools for sub-agents
        if tool_result and tool_result.get("tool") in FORBIDDEN_TOOLS:
            forbidden_tool = tool_result["tool"]
            if forbidden_tool in {"spawn_agent", "check_agents", "get_agent_result", "ask_user"}:
                reason = f"Sub-agents cannot use '{forbidden_tool}'. Only the parent agent can orchestrate worker management tools."
            else:
                reason = f"Sub-agents cannot use '{forbidden_tool}' in this run. This tool is restricted by supervisor policy."
            tool_result = {
                "tool": forbidden_tool,
                "result": {
                    "success": False,
                    "error": reason,
                }
            }

        if tool_result:
            tool_call_count += 1
            tool_name = tool_result.get("tool", "unknown")
            tool_exec_result = tool_result.get("result", {})
            tool_success = tool_exec_result.get("success", True) if isinstance(tool_exec_result, dict) else True

            logger.info(f"[agent_loop] Sub-agent {conv_id} tool {tool_call_count}/{max_tool_calls}: {tool_name} (success={tool_success})")

            # Loop detection
            # Polling tools are legitimate to call repeatedly (they check async job status)
            POLLING_TOOLS = {
                "comfyui_get_result", "comfyui_await_result", "comfyui_await_job",
                "comfyui_job_status", "gguf_download_status",
            }
            tool_args = tool_result.get("arguments", {})
            args_hash = hash(json.dumps(tool_args, sort_keys=True, default=str))
            recent_tool_hashes.append((tool_name, args_hash))
            is_looping = False
            # Check 1: 3 consecutive identical calls (same tool + same args)
            # Skip for polling tools — repeated status checks are expected behavior
            if len(recent_tool_hashes) >= 3 and tool_name not in POLLING_TOOLS:
                last_3 = recent_tool_hashes[-3:]
                if last_3[0] == last_3[1] == last_3[2]:
                    is_looping = True
                    logger.warning(f"[agent_loop] Loop detected: {tool_name} called 3+ times with same args")
            # Check 2: Ping-pong/cycling — last 6 calls use ≤2 unique tools
            if not is_looping and len(recent_tool_hashes) >= 6:
                last_6_names = [t[0] for t in recent_tool_hashes[-6:]]
                if len(set(last_6_names)) <= 2:
                    is_looping = True
                    logger.warning(f"[agent_loop] Ping-pong loop detected: last 6 calls cycle between {set(last_6_names)}")
            # Check 3: Saturation — last 8 calls use ≤3 unique tools
            if not is_looping and len(recent_tool_hashes) >= 8:
                last_8_names = [t[0] for t in recent_tool_hashes[-8:]]
                if len(set(last_8_names)) <= 3:
                    is_looping = True
                    logger.warning(f"[agent_loop] Saturation loop detected: last 8 calls cycle between {set(last_8_names)}")
            if is_looping:
                loop_warning_count += 1

            # Store assistant message
            conversation_store.append_message(conv_id, "assistant", full_response)

            # Track tool call
            tool_calls_made.append({
                "tool": tool_name,
                "arguments": tool_result.get("arguments", {}),
                "result": tool_exec_result,
            })

            emit("tool_call", {
                "tool": tool_name,
                "number": tool_call_count,
                "success": tool_success,
            })

            # Truncate result for conversation
            result_str = json.dumps(
                tool_exec_result if isinstance(tool_exec_result, dict)
                else {"result": tool_exec_result},
                indent=2
            )
            result_str = truncate_tool_result(result_str)

            # Update working memory
            memory = update_working_memory(
                memory, tool_name=tool_name, tool_result=tool_exec_result
            )

            # Rebuild prompt if system state changed
            if tool_name in SYSTEM_CHANGING_TOOLS:
                dynamic_state = await tool_router.build_dynamic_prompt(persona)
                system_prompt = _build_system_prompt(
                    persona, dynamic_state, tools_prompt,
                    additional_instructions, working_memory=memory
                )

            # Hard stop: if looping detected 2+ times, force final summary
            if is_looping and loop_warning_count >= 2:
                logger.warning(f"[agent_loop] Hard-stopping after {loop_warning_count} loop warnings ({tool_call_count} tool calls)")
                tool_call_count = max_tool_calls  # Force into final summary branch

            if tool_call_count < max_tool_calls:
                # Continue autonomously
                if not tool_success:
                    error_msg = tool_exec_result.get("error", "Unknown error") if isinstance(tool_exec_result, dict) else str(tool_exec_result)
                    error_type = categorize_error(tool_name, error_msg)
                    retry_state.record_attempt(tool_name, tool_result.get("arguments", {}), error_msg, error_type)

                    continuation = (
                        f"Tool result:\n```json\n{result_str}\n```\n\n"
                        + build_retry_prompt(tool_name, error_msg, error_type, retry_state)
                    )
                elif is_looping:
                    continuation = (
                        f"Tool result:\n```json\n{result_str}\n```\n\n"
                        f"WARNING: You are stuck in a loop, calling the same tools repeatedly. "
                        f"You MUST stop calling tools and provide a final summary NOW. "
                        f"Summarize ALL the information you've gathered from tool results so far. "
                        f"Do NOT make another tool call."
                    )
                elif isinstance(tool_exec_result, dict) and tool_exec_result.get("_race_failed"):
                    race_notes = tool_exec_result.get("_race_failure_notes", "")
                    continuation = (
                        f"Tool result:\n```json\n{result_str}\n```\n\n"
                        f"IMPORTANT: Multiple approaches were tried automatically and ALL failed.\n"
                        f"{race_notes}\n\n"
                        f"You MUST try a fundamentally different approach. Consider different "
                        f"tools, different parameters, or ask for clarification."
                    )
                else:
                    continuation = (
                        f"Tool result:\n```json\n{result_str}\n```\n\n"
                        f"Tool call succeeded. If the task is complete, provide a clear summary of what was accomplished. "
                        f"If more steps are needed, execute the next tool call."
                    )
                conversation_store.append_message(conv_id, "user", continuation)
                conversation_store.set_metadata(conv_id, "working_memory", memory.to_dict())
                continue
            else:
                # Max tool calls reached — get final summary
                conversation_store.append_message(
                    conv_id, "user",
                    f"Tool result:\n```json\n{result_str}\n```\n\n"
                    f"Maximum tool calls reached. Provide a clear summary of what was accomplished."
                )

                recent_messages = conversation_store.get_messages(conv_id, limit=20)
                follow_up_messages = [
                    ChatMessage(role="system", content=system_prompt)
                ] + [
                    ChatMessage(role=m["role"], content=m["content"])
                    for m in recent_messages
                ]

                follow_up_request = InferenceRequest(
                    messages=follow_up_messages,
                    max_tokens=4096,
                    temperature=persona.temperature,
                )

                follow_up = ""
                final_instance_id = instance_id_provider() if instance_id_provider else None
                if not final_instance_id:
                    final_instance_id = instance_id
                async for response in orchestrator.chat(final_instance_id, follow_up_request):
                    if response.text:
                        follow_up += response.text

                conversation_store.append_message(conv_id, "assistant", follow_up)
                final_response = follow_up
                break
        else:
            # No tool call — final text response
            # Creative racing disabled: fires on task summaries producing empty variants.
            # TODO: Re-enable with heuristic to only race genuinely creative content.
            conversation_store.append_message(conv_id, "assistant", full_response)
            final_response = full_response
            break

    # Save final memory
    conversation_store.set_metadata(conv_id, "working_memory", memory.to_dict())

    # Flush deferred conversation writes to disk
    conversation_store.flush(conv_id)

    emit("done", {
        "tool_calls": tool_call_count,
        "aborted": aborted,
    })

    return {
        "success": not aborted,
        "conversation_id": conv_id,
        "response": final_response,
        "tool_calls": tool_calls_made,
        "tool_call_count": tool_call_count,
        "aborted": aborted,
    }
