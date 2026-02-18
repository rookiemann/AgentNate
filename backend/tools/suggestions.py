"""
Suggestion Engine - Provides contextual suggestions for what the user should do next.
"""

from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum


class SuggestionPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Suggestion:
    priority: SuggestionPriority
    message: str
    tool: str
    example_args: Dict[str, Any]


class SuggestionEngine:
    """Rule-based engine that generates contextual suggestions."""

    def __init__(self):
        self.rules = [
            self._check_no_models,
            self._check_no_n8n,
            self._check_gpu_capacity,
            self._check_queue_depth,
            self._check_idle_models,
        ]

    def generate_suggestions(self, snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate prioritized suggestions based on system state.

        Args:
            snapshot: System snapshot from get_system_snapshot()

        Returns:
            List of suggestions sorted by priority
        """
        suggestions = []

        for rule in self.rules:
            suggestion = rule(snapshot)
            if suggestion:
                suggestions.append(suggestion)

        # Sort by priority
        priority_order = {
            SuggestionPriority.CRITICAL: 0,
            SuggestionPriority.HIGH: 1,
            SuggestionPriority.MEDIUM: 2,
            SuggestionPriority.LOW: 3,
        }

        suggestions.sort(key=lambda s: priority_order[s.priority])

        # Convert to dicts for JSON serialization
        return [
            {
                "priority": s.priority.value,
                "message": s.message,
                "tool": s.tool,
                "example_args": s.example_args
            }
            for s in suggestions
        ]

    def _check_no_models(self, snapshot: Dict[str, Any]) -> Suggestion | None:
        """Check if no models are loaded."""
        models = snapshot.get("models", [])
        if len(models) == 0:
            return Suggestion(
                priority=SuggestionPriority.CRITICAL,
                message="No models loaded. Load a model to enable chat functionality.",
                tool="load_model",
                example_args={"model_name": "phi-4"}
            )
        return None

    def _check_no_n8n(self, snapshot: Dict[str, Any]) -> Suggestion | None:
        """Check if no n8n instances are running."""
        n8n_instances = snapshot.get("n8n_instances", [])
        running = [i for i in n8n_instances if i.get("running")]

        if len(running) == 0:
            return Suggestion(
                priority=SuggestionPriority.HIGH,
                message="No n8n instances running. Start one to enable workflow automation.",
                tool="spawn_n8n",
                example_args={}
            )
        return None

    def _check_gpu_capacity(self, snapshot: Dict[str, Any]) -> Suggestion | None:
        """Check if there's significant free GPU memory."""
        gpus = snapshot.get("gpus", [])
        models = snapshot.get("models", [])

        # Only suggest if at least one model is already loaded
        if len(models) == 0:
            return None

        for gpu in gpus:
            free_mb = gpu.get("memory_free_mb", 0)
            # 8GB free is enough for most models
            if free_mb >= 8000:
                return Suggestion(
                    priority=SuggestionPriority.MEDIUM,
                    message=f"GPU {gpu.get('index', 0)} has {free_mb}MB free. You could load another model.",
                    tool="load_model",
                    example_args={"model_name": "llama-3", "gpu_index": gpu.get("index", 0)}
                )
        return None

    def _check_queue_depth(self, snapshot: Dict[str, Any]) -> Suggestion | None:
        """Check if request queue is getting deep."""
        queue = snapshot.get("queue", {})
        pending = queue.get("pending", 0)

        if pending > 5:
            return Suggestion(
                priority=SuggestionPriority.MEDIUM,
                message=f"Request queue has {pending} pending requests. Consider loading additional model instances for parallel processing.",
                tool="load_model",
                example_args={"model_name": "phi-4"}
            )
        return None

    def _check_idle_models(self, snapshot: Dict[str, Any]) -> Suggestion | None:
        """Check if all models are idle (nothing happening)."""
        models = snapshot.get("models", [])
        queue = snapshot.get("queue", {})

        if len(models) > 0:
            all_ready = all(m.get("status") == "ready" for m in models)
            no_queue = queue.get("pending", 0) == 0 and queue.get("processing", 0) == 0

            if all_ready and no_queue:
                return Suggestion(
                    priority=SuggestionPriority.LOW,
                    message="System is idle. Try chatting with a model or creating a workflow.",
                    tool="list_loaded_models",
                    example_args={}
                )
        return None


def format_suggestions_for_prompt(suggestions: List[Dict[str, Any]]) -> str:
    """Format suggestions as markdown for the system prompt."""
    if not suggestions:
        return ""

    lines = ["## Suggested Actions"]
    for i, s in enumerate(suggestions, 1):
        priority_icon = {
            "critical": "🔴",
            "high": "🟠",
            "medium": "🟡",
            "low": "🟢"
        }.get(s["priority"], "•")

        lines.append(f"{i}. {priority_icon} **[{s['priority'].upper()}]** {s['message']}")
        lines.append(f"   Use `{s['tool']}` tool" + (f" with args: `{s['example_args']}`" if s['example_args'] else ""))

    return "\n".join(lines)
