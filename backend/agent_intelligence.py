"""
Agent Intelligence Module

Planning, tool selection, structured retries, context management,
working memory, and thinking model support for the Meta Agent.
"""

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("agent_intelligence")


# =============================================================================
# Constants
# =============================================================================

PLANNING_PROMPT = """Analyze this request and create a brief action plan.

Tool categories available: {category_list}

User request: {user_message}

Respond with ONLY valid JSON (no markdown, no explanation):
{{"summary": "one-line summary of what to do", "steps": [{{"step": "description", "tool_category": "category_name"}}], "needs_tools": true, "complexity": "simple"}}

Rules:
- complexity is one of: simple, moderate, complex
- needs_tools is false only for pure conversation/questions with no actions needed
- Keep steps concise (max 5 steps)
- tool_category must be from the available categories list, or "none" for non-tool steps"""

SELECTOR_PROMPT = """Which tool categories are needed for this task? Pick ONLY relevant ones.

Categories:
{category_descriptions}

Task: {user_message}

Respond with ONLY a comma-separated list of category names. Example: web,files,code"""

THINKING_MODEL_PATTERNS = ["deepseek-r1", "qwq", "qwen-qwq", "qwen3", "o1-", "o1mini"]

# Keyword-to-category map for deterministic category boosting.
# If any keyword appears in the user message, that category is always included.
KEYWORD_CATEGORY_MAP: Dict[str, List[str]] = {
    "comfyui": ["comfyui"],
    "comfy ui": ["comfyui"],
    "stable diffusion": ["comfyui"],
    "sdxl": ["comfyui"],
    "sd3": ["comfyui"],
    "flux": ["comfyui"],
    "ltx": ["comfyui"],
    "text to image": ["comfyui"],
    "text-to-image": ["comfyui"],
    "txt2img": ["comfyui"],
    "image generation": ["comfyui"],
    "image gen": ["comfyui"],
    "text to video": ["comfyui"],
    "text-to-video": ["comfyui"],
    "video generation": ["comfyui"],
    "video gen": ["comfyui"],
    "img2img": ["comfyui"],
    "inpainting": ["comfyui"],
    "upscale": ["comfyui"],
    "controlnet": ["comfyui"],
    "lora": ["comfyui"],
    "checkpoint": ["comfyui"],
    "workflow": ["workflow"],
    "n8n": ["workflow", "n8n"],
    "automation": ["workflow", "n8n"],
    "deploy workflow": ["workflow", "n8n"],
    "web search": ["web"],
    "search the web": ["web"],
    "browse": ["web"],
    "fetch url": ["web"],
    "scrape": ["web"],
    "load model": ["model"],
    "unload model": ["model"],
    "gpu": ["model", "system"],
    "sub-agent": ["agents"],
    "sub agent": ["agents"],
    "spawn agent": ["agents"],
    "parallel agent": ["agents"],
    "routing preset": ["agents"],
    "marketplace": ["marketplace"],
    "credential": ["workflow"],
    "webhook": ["workflow"],
    "read file": ["files"],
    "write file": ["files"],
    "run python": ["code"],
    "run script": ["code"],
    "execute code": ["code"],
    "discord": ["communication"],
    "slack": ["communication"],
    "email": ["communication"],
    "telegram": ["communication"],
    "codebase": ["codebase"],
    "architecture": ["codebase"],
    "manifest": ["codebase"],
}


def extract_keyword_categories(message: str) -> List[str]:
    """Deterministic keyword matching for tool category boosting."""
    msg_lower = message.lower()
    categories = set()
    for keyword, cats in KEYWORD_CATEGORY_MAP.items():
        if keyword in msg_lower:
            categories.update(cats)
    return list(categories)


# Simple queries that don't need planning (regex patterns)
SIMPLE_QUERY_PATTERNS = [
    r"^(hi|hello|hey|thanks|thank you|ok|okay|sure|yes|no|bye)\b",
    r"^what (time|date) is it",
    r"^(who|what) are you",
    r"^(help|status|info)$",
]

TOOL_FALLBACKS: Dict[str, List[str]] = {
    "web_search": ["google_search", "duckduckgo_search", "fetch_url"],
    "google_search": ["serper_search", "duckduckgo_search"],
    "serper_search": ["google_search", "duckduckgo_search"],
    "duckduckgo_search": ["web_search", "google_search"],
    "fetch_url": ["browser_open", "http_request"],
    "run_python": ["run_shell", "run_powershell"],
    "run_javascript": ["run_python", "run_shell"],
    "run_shell": ["run_powershell", "run_python"],
    "run_powershell": ["run_shell", "run_python"],
    "http_request": ["fetch_url"],
    "browser_open": ["fetch_url"],
    "read_file": ["search_content", "list_directory"],
    "write_file": ["run_python"],
    "send_discord": ["send_webhook"],
    "send_slack": ["send_webhook"],
    "send_email": ["send_webhook"],
    "analyze_image": ["describe_ui", "extract_text_from_image"],
}


# =============================================================================
# Data Structures
# =============================================================================

class ErrorType(Enum):
    TRANSIENT = "transient"
    WRONG_ARGS = "wrong_args"
    CAPABILITY = "capability"
    FATAL = "fatal"


@dataclass
class WorkingMemory:
    """Structured scratchpad for multi-step task tracking."""
    goal: str = ""
    completed_steps: List[str] = field(default_factory=list)
    gathered_facts: Dict[str, str] = field(default_factory=dict)
    remaining_steps: List[str] = field(default_factory=list)
    failed_approaches: List[Dict[str, str]] = field(default_factory=list)

    def to_prompt_section(self) -> str:
        """Format as markdown for system prompt injection."""
        if not self.has_content():
            return ""

        lines = ["## Working Memory (Task Progress)"]
        if self.goal:
            lines.append(f"**Goal:** {self.goal}")

        if self.completed_steps:
            lines.append("**Completed:**")
            for step in self.completed_steps[-10:]:  # Show last 10
                lines.append(f"  - {step}")

        if self.gathered_facts:
            lines.append("**Key Facts:**")
            for k, v in list(self.gathered_facts.items())[-10:]:
                lines.append(f"  - {k}: {v}")

        if self.remaining_steps:
            lines.append("**Remaining:**")
            for step in self.remaining_steps[:5]:  # Show next 5
                lines.append(f"  - {step}")

        if self.failed_approaches:
            lines.append("**Failed (avoid repeating):**")
            for fa in self.failed_approaches[-5:]:
                lines.append(f"  - {fa.get('tool', '?')}: {fa.get('reason', '?')}")

        return "\n".join(lines)

    def has_content(self) -> bool:
        return bool(
            self.goal or self.completed_steps or self.gathered_facts
            or self.remaining_steps or self.failed_approaches
        )

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "WorkingMemory":
        if not data:
            return cls()
        return cls(
            goal=data.get("goal", ""),
            completed_steps=data.get("completed_steps", []),
            gathered_facts=data.get("gathered_facts", {}),
            remaining_steps=data.get("remaining_steps", []),
            failed_approaches=data.get("failed_approaches", []),
        )


@dataclass
class AgentPlan:
    """Lightweight plan generated before tool execution."""
    summary: str = ""
    steps: List[Dict[str, str]] = field(default_factory=list)
    needs_tools: bool = True
    complexity: str = "simple"
    selected_categories: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "AgentPlan":
        if not data:
            return cls()
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class RetryState:
    """Tracks tool retry attempts within a single agent session."""
    tool_attempts: Dict[str, List[Dict]] = field(default_factory=dict)
    retry_budget: int = 2

    def record_attempt(self, tool_name: str, args: Dict, error: str, error_type: ErrorType):
        if tool_name not in self.tool_attempts:
            self.tool_attempts[tool_name] = []
        self.tool_attempts[tool_name].append({
            "args": args,
            "error": error,
            "error_type": error_type.value,
        })

    def can_retry(self, tool_name: str) -> bool:
        attempts = self.tool_attempts.get(tool_name, [])
        return len(attempts) < self.retry_budget

    def get_attempt_count(self, tool_name: str) -> int:
        return len(self.tool_attempts.get(tool_name, []))

    def get_failed_tools(self) -> List[str]:
        return [
            tool for tool, attempts in self.tool_attempts.items()
            if any(a.get("error") for a in attempts)
        ]


# =============================================================================
# Planning Functions
# =============================================================================

def needs_planning(message: str) -> bool:
    """Heuristic: skip planning for simple greetings/queries."""
    msg = message.strip().lower()

    # Very short messages are usually simple
    if len(msg) < 15:
        return False

    # Check against simple patterns
    for pattern in SIMPLE_QUERY_PATTERNS:
        if re.match(pattern, msg, re.IGNORECASE):
            return False

    # Messages with multiple sentences or action words likely need planning
    action_words = ["search", "find", "create", "build", "write", "read", "deploy",
                    "install", "setup", "configure", "download", "analyze", "compare",
                    "research", "summarize", "send", "fetch", "run", "execute",
                    "delete", "move", "copy", "start", "stop", "load", "unload"]

    msg_words = set(msg.split())
    action_count = len(msg_words.intersection(action_words))

    # Multiple actions or long complex requests need planning
    if action_count >= 2 or len(msg) > 100:
        return True

    # Single action on short message - still plan if it seems multi-step
    if "and" in msg or "then" in msg or "," in msg:
        return True

    return action_count >= 1 and len(msg) > 40


async def generate_plan(orchestrator, instance_id: str, message: str,
                        persona, memory: WorkingMemory,
                        category_list: str) -> AgentPlan:
    """Generate a lightweight plan via LLM call."""
    from providers.base import InferenceRequest, ChatMessage

    prompt = PLANNING_PROMPT.format(
        category_list=category_list,
        user_message=message,
    )

    # Add memory context if available
    if memory.has_content():
        prompt += f"\n\nCurrent task progress:\n{memory.to_prompt_section()}"

    messages = [
        ChatMessage(role="system", content="You are a task planner. Respond with ONLY valid JSON."),
        ChatMessage(role="user", content=prompt),
    ]

    request = InferenceRequest(
        messages=messages,
        max_tokens=512,
        temperature=0.3,
    )

    full_response = ""
    try:
        async for response in orchestrator.chat(instance_id, request):
            full_response += response.text

        # Parse JSON from response (handle markdown wrapping)
        json_str = full_response.strip()
        if json_str.startswith("```"):
            json_str = re.sub(r"^```(?:json)?\s*", "", json_str)
            json_str = re.sub(r"\s*```$", "", json_str)

        data = json.loads(json_str)

        plan = AgentPlan(
            summary=data.get("summary", ""),
            steps=data.get("steps", []),
            needs_tools=data.get("needs_tools", True),
            complexity=data.get("complexity", "simple"),
        )

        # Extract categories from steps
        categories = set()
        for step in plan.steps:
            cat = step.get("tool_category", "")
            if cat and cat != "none":
                categories.add(cat)

        # Merge with deterministic keyword matches
        keyword_cats = extract_keyword_categories(message)
        categories.update(keyword_cats)

        plan.selected_categories = list(categories)

        logger.info(f"Plan generated: {plan.summary} ({plan.complexity}, {len(plan.steps)} steps, categories={plan.selected_categories})")
        return plan

    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Plan generation failed: {e}, response was: {full_response[:200]}")
        # Return a minimal plan that doesn't block execution
        # Still apply keyword matching so tool categories are correct
        keyword_cats = extract_keyword_categories(message)
        return AgentPlan(
            summary="Execute user request",
            steps=[{"step": message, "tool_category": "none"}],
            needs_tools=True,
            complexity="simple",
            selected_categories=keyword_cats,
        )


async def select_tool_categories(orchestrator, instance_id: str,
                                 message: str, plan: Optional[AgentPlan],
                                 category_descriptions: Dict[str, str]) -> List[str]:
    """Pick relevant tool categories via keyword matching + LLM call."""
    from providers.base import InferenceRequest, ChatMessage

    # Always start with deterministic keyword matches
    keyword_cats = extract_keyword_categories(message)
    if keyword_cats:
        logger.info(f"Keyword-matched categories: {keyword_cats}")

    # If plan already has categories, merge with keyword matches
    if plan and plan.selected_categories:
        merged = list(set(plan.selected_categories + keyword_cats))
        logger.info(f"Plan + keyword categories: {merged}")
        return merged

    # If keyword matching already found categories, skip LLM call
    if keyword_cats:
        return keyword_cats

    # Fall back to LLM-based selection
    desc_lines = "\n".join(f"- {k}: {v}" for k, v in category_descriptions.items())

    prompt = SELECTOR_PROMPT.format(
        category_descriptions=desc_lines,
        user_message=message,
    )

    messages = [
        ChatMessage(role="user", content=prompt),
    ]

    request = InferenceRequest(
        messages=messages,
        max_tokens=100,
        temperature=0.1,
    )

    try:
        full_response = ""
        async for response in orchestrator.chat(instance_id, request):
            full_response += response.text

        # Parse comma-separated categories
        raw = full_response.strip().lower()
        logger.info(f"LLM category selection raw response: {raw}")
        # Clean up common LLM formatting issues
        raw = raw.replace("\n", ",").replace(";", ",")
        categories = [c.strip() for c in raw.split(",") if c.strip()]

        # Validate against known categories
        valid = [c for c in categories if c in category_descriptions]

        if valid:
            logger.info(f"Selected tool categories (LLM): {valid}")
            return valid

    except Exception as e:
        logger.warning(f"Category selection failed: {e}")

    # Fallback: return all categories
    logger.warning("Category selection returned no valid categories, using ALL")
    return list(category_descriptions.keys())


# =============================================================================
# Error Handling / Retries
# =============================================================================

def categorize_error(tool_name: str, error: str) -> ErrorType:
    """Classify an error to determine retry strategy."""
    error_lower = error.lower()

    # Transient errors (retry same)
    transient_patterns = ["timeout", "timed out", "connection refused",
                          "connection reset", "temporarily unavailable",
                          "rate limit", "429", "503", "502"]
    for pattern in transient_patterns:
        if pattern in error_lower:
            return ErrorType.TRANSIENT

    # Wrong arguments (fix args)
    args_patterns = ["typeerror", "invalid argument", "missing required",
                     "expected", "invalid value", "validation error",
                     "must be", "is required", "not a valid"]
    for pattern in args_patterns:
        if pattern in error_lower:
            return ErrorType.WRONG_ARGS

    # Capability errors (try different tool)
    capability_patterns = ["not found", "not supported", "not implemented",
                           "no such file", "permission denied", "access denied",
                           "404", "not installed", "unavailable"]
    for pattern in capability_patterns:
        if pattern in error_lower:
            return ErrorType.CAPABILITY

    # Fatal errors (give up)
    fatal_patterns = ["critical", "fatal", "out of memory", "disk full",
                      "authentication failed", "unauthorized", "forbidden"]
    for pattern in fatal_patterns:
        if pattern in error_lower:
            return ErrorType.FATAL

    # Default to capability (try a different approach)
    return ErrorType.CAPABILITY


def build_retry_prompt(tool_name: str, error: str, error_type: ErrorType,
                       retry_state: RetryState) -> str:
    """Build a smart continuation prompt based on error type."""
    attempt_count = retry_state.get_attempt_count(tool_name)

    if error_type == ErrorType.TRANSIENT:
        if retry_state.can_retry(tool_name):
            return (
                f"The tool `{tool_name}` failed with a transient error: {error}\n"
                f"This may be temporary. Try calling `{tool_name}` again with the same arguments."
            )
        else:
            fallbacks = TOOL_FALLBACKS.get(tool_name, [])
            if fallbacks:
                return (
                    f"`{tool_name}` failed {attempt_count} times with transient errors.\n"
                    f"Try an alternative tool: {', '.join(fallbacks)}"
                )

    elif error_type == ErrorType.WRONG_ARGS:
        return (
            f"The tool `{tool_name}` was called with incorrect arguments: {error}\n"
            f"Review the tool's parameter requirements and try again with corrected arguments."
        )

    elif error_type == ErrorType.CAPABILITY:
        fallbacks = TOOL_FALLBACKS.get(tool_name, [])
        if fallbacks:
            return (
                f"`{tool_name}` cannot complete this task: {error}\n"
                f"Try an alternative approach using: {', '.join(fallbacks)}"
            )
        return (
            f"`{tool_name}` failed: {error}\n"
            f"Try a different tool or approach to accomplish this step."
        )

    elif error_type == ErrorType.FATAL:
        return (
            f"`{tool_name}` encountered a fatal error: {error}\n"
            f"This cannot be retried. Explain to the user what happened and suggest alternatives."
        )

    # Generic fallback
    return (
        f"Tool `{tool_name}` failed (attempt {attempt_count}): {error}\n"
        f"Analyze the error and try a different approach."
    )


# =============================================================================
# Thinking Model Support
# =============================================================================

def detect_thinking_content(response_text: str) -> Tuple[Optional[str], str]:
    """
    Extract <think>...</think> blocks from model output.

    Returns (thinking_content, clean_response).
    If no thinking tags found, returns (None, original_text).
    """
    pattern = r"<think>(.*?)</think>"
    matches = re.findall(pattern, response_text, re.DOTALL)

    if not matches:
        return None, response_text

    thinking = "\n\n".join(m.strip() for m in matches)
    # Remove think tags, collapse any resulting double-spaces, then strip
    clean = re.sub(pattern, " ", response_text, flags=re.DOTALL)
    clean = re.sub(r"  +", " ", clean).strip()

    return thinking, clean


def is_likely_thinking_model(model_id: str) -> bool:
    """Name-based heuristic for known thinking models."""
    if not model_id:
        return False
    model_lower = model_id.lower()
    return any(pattern in model_lower for pattern in THINKING_MODEL_PATTERNS)


# =============================================================================
# Context Management
# =============================================================================

def estimate_token_count(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


def should_summarize(messages: List[Dict], max_tokens: int = 6000) -> bool:
    """Check if conversation context exceeds budget."""
    total_chars = sum(len(m.get("content", "")) for m in messages)
    return (total_chars // 4) > max_tokens


async def summarize_old_messages(orchestrator, instance_id: str,
                                 messages: List[Dict], keep: int = 6) -> List[Dict]:
    """
    Compress older messages into a summary, keeping recent ones verbatim.

    Returns a new message list with a summary message replacing old ones.
    """
    from providers.base import InferenceRequest, ChatMessage

    if len(messages) <= keep:
        return messages

    old_messages = messages[:-keep]
    recent_messages = messages[-keep:]

    # Build summary of old messages
    old_text_parts = []
    for m in old_messages:
        role = m.get("role", "unknown")
        content = m.get("content", "")
        # Truncate individual messages for the summary prompt
        if len(content) > 500:
            content = content[:500] + "..."
        old_text_parts.append(f"[{role}]: {content}")

    old_text = "\n".join(old_text_parts)

    summary_prompt = (
        f"Summarize this conversation history into a brief paragraph. "
        f"Focus on: what the user wanted, what tools were used, what results were obtained, "
        f"and what's still pending.\n\n{old_text}"
    )

    request = InferenceRequest(
        messages=[ChatMessage(role="user", content=summary_prompt)],
        max_tokens=300,
        temperature=0.3,
    )

    try:
        summary = ""
        async for response in orchestrator.chat(instance_id, request):
            summary += response.text

        summary_message = {
            "role": "system",
            "content": f"[Conversation Summary] {summary.strip()}"
        }

        logger.info(f"Summarized {len(old_messages)} old messages into summary block")
        return [summary_message] + recent_messages

    except Exception as e:
        logger.warning(f"Message summarization failed: {e}, keeping original messages")
        return messages


def truncate_tool_result(result_str: str, max_chars: int = 2000) -> str:
    """Truncate large tool results while preserving structure."""
    if len(result_str) <= max_chars:
        return result_str

    # Try to preserve JSON structure
    try:
        data = json.loads(result_str)
        # If it's a dict with a large value, truncate the value
        if isinstance(data, dict):
            truncated = {}
            remaining = max_chars - 100  # Reserve space for structure + formatting
            for key, value in data.items():
                val_str = json.dumps(value)
                if len(val_str) > remaining // max(1, len(data)):
                    if isinstance(value, str):
                        truncated[key] = value[:remaining // len(data)] + "...[truncated]"
                    elif isinstance(value, list):
                        truncated[key] = value[:3]
                        if len(value) > 3:
                            truncated[key].append(f"...and {len(value) - 3} more items")
                    else:
                        truncated[key] = value
                else:
                    truncated[key] = value
            return json.dumps(truncated, indent=2)
    except (json.JSONDecodeError, TypeError):
        pass

    # Plain text truncation
    return result_str[:max_chars - 20] + "\n...[truncated]"


# =============================================================================
# Working Memory
# =============================================================================

def update_working_memory(memory: WorkingMemory, message: str = "",
                          tool_name: str = "", tool_result: Optional[Dict] = None,
                          plan: Optional[AgentPlan] = None,
                          response: str = "") -> WorkingMemory:
    """Rule-based memory update (no LLM call)."""

    # Set goal from plan or first user message
    if plan and plan.summary and not memory.goal:
        memory.goal = plan.summary

    if not memory.goal and message:
        # Use first ~100 chars of user message as goal
        memory.goal = message[:100] + ("..." if len(message) > 100 else "")

    # Update remaining steps from plan
    if plan and plan.steps and not memory.remaining_steps:
        memory.remaining_steps = [s.get("step", "") for s in plan.steps]

    # Track completed tool calls
    if tool_name and tool_result:
        success = tool_result.get("success", True) if isinstance(tool_result, dict) else True

        if success:
            step_desc = f"{tool_name} completed"
            # Extract a key fact from the result
            if isinstance(tool_result, dict):
                # Look for common result fields
                for key in ["data", "result", "output", "content", "text", "url", "path"]:
                    if key in tool_result:
                        val = str(tool_result[key])
                        if len(val) > 100:
                            val = val[:100] + "..."
                        memory.gathered_facts[f"{tool_name}.{key}"] = val
                        break

            memory.completed_steps.append(step_desc)

            # Remove from remaining if it matches
            if memory.remaining_steps:
                # Pop the first remaining step (assumes sequential execution)
                memory.remaining_steps.pop(0)
        else:
            error = tool_result.get("error", "unknown error") if isinstance(tool_result, dict) else "failed"
            memory.failed_approaches.append({
                "tool": tool_name,
                "reason": str(error)[:100],
            })

    # Cap sizes
    memory.completed_steps = memory.completed_steps[-20:]
    if len(memory.gathered_facts) > 30:
        keys = list(memory.gathered_facts.keys())
        for k in keys[:-30]:
            del memory.gathered_facts[k]
    memory.failed_approaches = memory.failed_approaches[-10:]

    return memory
