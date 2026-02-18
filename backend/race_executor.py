"""
Race Executor — 3 concurrent LLM inferences for workflow tool calls.

When a worker calls a raceable tool (build_workflow, comfyui_build_workflow),
all 3 candidates fire simultaneously as LLM re-inferences with the same prompt
but slightly different temperatures. First valid wins, losers are cancelled.

Uses asyncio.wait(FIRST_COMPLETED) for early exit + real task.cancel().
"""

import asyncio
import json
import logging
import re
import time
import uuid
from typing import Dict, Any, List, Callable, Tuple, Optional

logger = logging.getLogger("race_executor")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RACEABLE_TOOLS = {
    "build_workflow",
    "comfyui_build_workflow",
}

RACE_VALIDATORS: Dict[str, Callable[[Dict], Tuple[bool, str]]] = {
    "build_workflow": lambda r: (
        bool(r.get("success", False)),
        "success" if r.get("success") else r.get("error", "build_failed"),
    ),
    "comfyui_build_workflow": lambda r: (
        bool(r.get("success", False)),
        "success" if r.get("success") else r.get("error", "build_failed"),
    ),
}

_SAFE_ARG_OVERRIDES = {
    "build_workflow": {"deploy": False, "activate": False},
    "comfyui_build_workflow": {"execute": False},
}

# Keys that indicate the JSON is tool arguments (not a tool call wrapper)
_ARG_KEYS = {
    "build_workflow": {"nodes"},
    "comfyui_build_workflow": {"nodes", "workflow"},
}


# ---------------------------------------------------------------------------
# Raw argument extraction (handles model output without wrapper format)
# ---------------------------------------------------------------------------

def _extract_raw_args(text: str, tool_name: str) -> Optional[Dict]:
    """Extract tool arguments from model output that omits the wrapper.

    Models often output: {"name": "...", "nodes": [...]}
    Instead of: {"tool": "build_workflow", "arguments": {"name": "...", "nodes": [...]}}

    Also handles markdown fences, truncated JSON, and preamble text.
    """
    expected_keys = _ARG_KEYS.get(tool_name, set())
    if not expected_keys:
        return None

    # Strip markdown fences
    clean = text
    if "```" in clean:
        clean = re.sub(r'```(?:json)?\s*', '', clean)
        clean = clean.replace('```', '')

    # Try every { in the text as a potential JSON start
    for i in range(len(clean)):
        if clean[i] != '{':
            continue
        depth = 0
        for j in range(i, len(clean)):
            if clean[j] == '{':
                depth += 1
            elif clean[j] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(clean[i:j + 1])
                        if isinstance(obj, dict) and expected_keys & set(obj.keys()):
                            logger.info(f"[race] _extract_raw_args found args at chars {i}-{j}")
                            return obj
                    except json.JSONDecodeError:
                        pass
                    break

    # Last resort: try to fix truncated JSON by adding closing brackets
    for i in range(len(clean)):
        if clean[i] != '{':
            continue
        fragment = clean[i:].rstrip()
        opens = fragment.count('{') - fragment.count('}')
        open_sq = fragment.count('[') - fragment.count(']')
        if opens > 0 or open_sq > 0:
            fixed = fragment + ']' * max(open_sq, 0) + '}' * max(opens, 0)
            try:
                obj = json.loads(fixed)
                if isinstance(obj, dict) and expected_keys & set(obj.keys()):
                    logger.info(f"[race] _extract_raw_args recovered truncated JSON")
                    return obj
            except json.JSONDecodeError:
                pass
        break

    return None


# ---------------------------------------------------------------------------
# Workflow Racing — 3 concurrent LLM inferences, first valid wins
# ---------------------------------------------------------------------------

async def run_tool_race(
    tool_router,
    orchestrator,
    instance_id: str,
    tool_name: str,
    original_args: Dict[str, Any],
    original_response: str,
    conversation_messages: List[Dict],
    system_prompt: str,
    persona,
    emit: Callable,
    num_candidates: int = 3,
    race_timeout: float = 120.0,
) -> Dict[str, Any]:
    """
    3 concurrent LLM inferences — first valid candidate wins, losers cancelled.

    All candidates get the same prompt with slight temperature variation.
    Uses asyncio.wait(FIRST_COMPLETED) for early exit + real task.cancel().
    """
    from providers.base import InferenceRequest, ChatMessage

    race_id = f"trc-{uuid.uuid4().hex[:8]}"
    validator = RACE_VALIDATORS.get(
        tool_name, lambda r: (bool(r.get("success", False)), "default")
    )
    safe_overrides = _SAFE_ARG_OVERRIDES.get(tool_name, {})
    actual_candidates = min(num_candidates, 3)

    emit("race_started", {
        "race_id": race_id,
        "tool_name": tool_name,
        "num_candidates": actual_candidates,
    })

    # Extract user's original request from conversation
    user_request = ""
    for m in conversation_messages:
        if m.get("role") == "user":
            user_request = m["content"]
            break

    # Build minimal system prompt: ONLY the tool definition for the tool being raced.
    # The full system_prompt has all persona instructions + all tool definitions which
    # easily exceeds the model's context window, leaving 0 tokens for generation.
    from backend.tools.tool_router import AVAILABLE_TOOLS
    tool_def = next((t for t in AVAILABLE_TOOLS if t["name"] == tool_name), None)
    if tool_def:
        params_text = ""
        props = tool_def.get("parameters", {}).get("properties", {})
        required = tool_def.get("parameters", {}).get("required", [])
        for pname, pinfo in props.items():
            req = "*" if pname in required else ""
            params_text += f"  - {pname}{req} ({pinfo.get('type', 'any')}): {pinfo.get('description', '')}\n"
        mini_system = (
            f"You are an AI assistant that builds workflows. "
            f"Call tools with: `{{\"tool\": \"name\", \"arguments\": {{...}}}}`\n\n"
            f"## {tool_name}\n{tool_def.get('description', '')}\n"
            f"Parameters:\n{params_text}"
        )
    else:
        mini_system = (
            f"You are an AI assistant. "
            f"Call tools with: `{{\"tool\": \"{tool_name}\", \"arguments\": {{...}}}}`"
        )

    logger.info(f"[race] {race_id}: {actual_candidates} candidates, mini prompt {len(mini_system)} chars")

    # The prompt suffix — same for all candidates
    prompt_suffix = (
        f"\n\nRespond with ONLY this exact JSON format, nothing else:\n"
        f'```json\n{{"tool": "{tool_name}", "arguments": {{"name": "...", "nodes": [...]}}}}\n```'
    )

    # ---- Define single candidate coroutine ----

    async def run_candidate(candidate_id: int) -> Dict:
        """Run one candidate: LLM inference → parse → execute → validate."""
        emit("race_candidate_status", {
            "race_id": race_id, "candidate_id": candidate_id, "status": "inferring",
        })

        messages = [
            ChatMessage(role="system", content=mini_system),
            ChatMessage(role="user", content=user_request + prompt_suffix),
        ]

        # Same prompt, slightly different temperature per candidate
        temp = min(persona.temperature + 0.1 * (candidate_id - 1), 1.0)
        inference_request = InferenceRequest(
            messages=messages,
            max_tokens=2048,
            temperature=temp,
            top_p=0.95,
        )

        vstart = time.time()
        response_text = ""

        # Phase 1: LLM inference (streaming)
        try:
            async for resp in orchestrator.chat(instance_id, inference_request):
                if resp.text:
                    response_text += resp.text
                    emit("race_candidate_token", {
                        "race_id": race_id,
                        "candidate_id": candidate_id,
                        "text": resp.text,
                    })
        except asyncio.CancelledError:
            velapsed = round(time.time() - vstart, 2)
            logger.info(f"[race] C{candidate_id} cancelled during inference at {velapsed}s")
            emit("race_candidate_status", {
                "race_id": race_id, "candidate_id": candidate_id, "status": "canceled",
            })
            raise
        except Exception as exc:
            velapsed = round(time.time() - vstart, 2)
            emit("race_candidate_evaluated", {
                "race_id": race_id, "candidate_id": candidate_id,
                "is_valid": False,
                "reason": f"inference_error: {str(exc)[:100]}",
                "elapsed": velapsed,
            })
            return {"candidate_id": candidate_id, "valid": False,
                    "reason": "inference_error", "elapsed": velapsed}

        # Phase 2: Parse tool call
        parsed = tool_router.parse_tool_call(response_text)
        if not parsed or parsed.get("tool") != tool_name:
            # Model often outputs arguments directly without wrapper
            extracted_args = _extract_raw_args(response_text, tool_name)
            if extracted_args:
                parsed = {"tool": tool_name, "arguments": extracted_args, "known": True}
                logger.info(f"[race] C{candidate_id}: extracted raw args ({len(str(extracted_args))} chars)")
            else:
                wrong = parsed.get("tool", "none") if parsed else "no_tool_call"
                velapsed = round(time.time() - vstart, 2)
                emit("race_candidate_evaluated", {
                    "race_id": race_id, "candidate_id": candidate_id,
                    "is_valid": False, "reason": f"wrong_tool: {wrong}",
                    "elapsed": velapsed,
                })
                return {"candidate_id": candidate_id, "valid": False,
                        "reason": f"wrong_tool: {wrong}", "elapsed": velapsed}

        # Phase 3: Execute with safe overrides (no side effects)
        emit("race_candidate_status", {
            "race_id": race_id, "candidate_id": candidate_id, "status": "executing",
        })

        try:
            v_safe_args = {**parsed["arguments"], **safe_overrides}
            v_result = await tool_router.execute(tool_name, v_safe_args)
            is_valid, reason = validator(v_result)
            velapsed = round(time.time() - vstart, 2)
        except asyncio.CancelledError:
            velapsed = round(time.time() - vstart, 2)
            logger.info(f"[race] C{candidate_id} cancelled during execution at {velapsed}s")
            emit("race_candidate_status", {
                "race_id": race_id, "candidate_id": candidate_id, "status": "canceled",
            })
            raise
        except Exception as exc:
            velapsed = round(time.time() - vstart, 2)
            emit("race_candidate_evaluated", {
                "race_id": race_id, "candidate_id": candidate_id,
                "is_valid": False, "reason": str(exc)[:200],
                "elapsed": velapsed,
            })
            return {"candidate_id": candidate_id, "valid": False,
                    "reason": str(exc)[:200], "elapsed": velapsed}

        emit("race_candidate_evaluated", {
            "race_id": race_id, "candidate_id": candidate_id,
            "is_valid": is_valid, "reason": reason, "elapsed": velapsed,
        })

        return {
            "candidate_id": candidate_id,
            "valid": is_valid,
            "reason": reason,
            "elapsed": velapsed,
            "parsed_args": parsed["arguments"] if is_valid else None,
            "result": v_result if is_valid else None,
        }

    # ---- Launch ALL candidates simultaneously ----

    tasks: Dict[asyncio.Task, int] = {}
    for cid in range(1, actual_candidates + 1):
        task = asyncio.create_task(
            run_candidate(cid),
            name=f"race-{race_id}-c{cid}",
        )
        tasks[task] = cid

    logger.info(f"[race] {race_id}: launched {len(tasks)} concurrent LLM inferences")

    # ---- FIRST_COMPLETED loop — early exit on winner ----

    pending = set(tasks.keys())
    winner_result = None
    failure_reasons = []

    try:
        while pending:
            done, pending = await asyncio.wait(
                pending,
                timeout=race_timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if not done:
                # Timeout — cancel everything remaining
                logger.warning(f"[race] {race_id}: timeout after {race_timeout}s")
                for t in pending:
                    t.cancel()
                if pending:
                    await asyncio.wait(pending, timeout=2.0)
                failure_reasons.append("timeout")
                break

            for completed_task in done:
                cid = tasks[completed_task]
                try:
                    result = completed_task.result()
                except asyncio.CancelledError:
                    continue
                except Exception as exc:
                    failure_reasons.append(f"C{cid}: {str(exc)[:100]}")
                    continue

                if result.get("valid"):
                    # WINNER — cancel all remaining immediately
                    winner_result = result
                    logger.info(f"[race] {race_id}: C{cid} wins at {result['elapsed']}s — cancelling {len(pending)} others")
                    for t in pending:
                        t.cancel()
                    # Brief wait for cancellation to propagate
                    if pending:
                        await asyncio.wait(pending, timeout=2.0)
                    break
                else:
                    failure_reasons.append(f"C{cid}: {result.get('reason', 'unknown')}")

            if winner_result:
                break

    except Exception:
        # Safety net: cancel everything
        for t in set(tasks.keys()):
            if not t.done():
                t.cancel()
        raise

    # ---- Handle winner or all-failed ----

    if winner_result:
        winner_id = winner_result["candidate_id"]
        emit("race_winner_selected", {"race_id": race_id, "winner": winner_id})
        emit("race_completed", {
            "race_id": race_id, "winner": winner_id,
            "total": actual_candidates, "valid_count": 1,
        })

        # Re-execute winner with side effects (deploy, activate, etc.)
        if safe_overrides:
            final_result = await tool_router.execute(tool_name, winner_result["parsed_args"])
        else:
            final_result = winner_result["result"]

        return {
            "tool": tool_name,
            "arguments": winner_result["parsed_args"],
            "result": final_result,
        }

    # All failed
    logger.info(f"[race] {race_id}: all {actual_candidates} candidates failed: {failure_reasons}")
    emit("race_failed", {
        "race_id": race_id,
        "total": actual_candidates,
        "failure_notes": f"All {actual_candidates} approaches failed. Reasons: {'; '.join(failure_reasons[:3])}",
    })

    return {
        "tool": tool_name,
        "arguments": original_args,
        "result": {
            "success": False,
            "error": failure_reasons[0] if failure_reasons else "all_failed",
            "_race_failed": True,
            "_race_failure_notes": f"Tried {actual_candidates} approaches, all failed.",
        },
    }
