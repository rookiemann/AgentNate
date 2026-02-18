"""
Agent Memory - Persistent key-value memory across conversations.

Allows the meta agent to store and recall facts, user preferences,
and accumulated knowledge that persists across sessions.
"""

import json
import os
import logging
from typing import List, Dict, Optional
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("agent_memory")

MAX_ENTRIES = 200


class AgentMemory:
    """Persistent key-value memory store backed by a JSON file."""

    def __init__(self, storage_dir: str = None):
        if storage_dir:
            self._dir = Path(storage_dir)
        else:
            # Default: %APPDATA%/AgentNate/ or ~/.config/AgentNate/
            if os.name == 'nt':
                base = os.environ.get('APPDATA', os.path.expanduser('~'))
            else:
                base = os.environ.get('XDG_CONFIG_HOME', os.path.expanduser('~/.config'))
            self._dir = Path(base) / "AgentNate"

        self._dir.mkdir(parents=True, exist_ok=True)
        self._file = self._dir / "agent_memory.json"
        self._memories: List[Dict] = []
        self._load()

    def _load(self):
        """Load memories from disk."""
        if self._file.exists():
            try:
                with open(self._file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._memories = data.get("memories", [])
                logger.info(f"Loaded {len(self._memories)} memories from {self._file}")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load agent memory: {e}")
                self._memories = []
        else:
            self._memories = []

    def _save(self):
        """Persist memories to disk."""
        try:
            with open(self._file, 'w', encoding='utf-8') as f:
                json.dump({"memories": self._memories}, f, indent=2, ensure_ascii=False)
        except IOError as e:
            logger.error(f"Failed to save agent memory: {e}")

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def store(self, key: str, value: str, category: str = "general") -> Dict:
        """
        Store a memory entry. If key already exists, updates it.

        Args:
            key: Short identifier (e.g. "user_language_preference")
            value: The fact to remember (e.g. "User prefers Python over JavaScript")
            category: Optional grouping (e.g. "preference", "fact", "decision")

        Returns:
            The stored memory entry
        """
        key = key.strip()
        value = value.strip()

        # Check for existing key — update instead of duplicate
        for mem in self._memories:
            if mem["key"].lower() == key.lower():
                mem["value"] = value
                mem["category"] = category
                mem["updated_at"] = self._now()
                self._save()
                logger.info(f"Updated memory: {key}")
                return mem

        # New entry
        entry = {
            "key": key,
            "value": value,
            "category": category,
            "created_at": self._now(),
            "updated_at": self._now(),
        }
        self._memories.append(entry)

        # Auto-prune if over limit
        if len(self._memories) > MAX_ENTRIES:
            removed = len(self._memories) - MAX_ENTRIES
            self._memories = self._memories[removed:]
            logger.info(f"Auto-pruned {removed} oldest memories")

        self._save()
        logger.info(f"Stored memory: {key}")
        return entry

    def recall(self, query: str) -> List[Dict]:
        """
        Search memories by substring match on key and value.

        Args:
            query: Search term (case-insensitive)

        Returns:
            List of matching memory entries, newest first
        """
        query_lower = query.lower().strip()
        matches = []
        for mem in self._memories:
            if (query_lower in mem["key"].lower() or
                    query_lower in mem["value"].lower() or
                    query_lower in mem.get("category", "").lower()):
                matches.append(mem)

        # Return newest first
        matches.sort(key=lambda m: m.get("updated_at", ""), reverse=True)
        return matches

    def list_recent(self, limit: int = 10) -> List[Dict]:
        """Get the N most recently updated memories."""
        sorted_mems = sorted(
            self._memories,
            key=lambda m: m.get("updated_at", ""),
            reverse=True
        )
        return sorted_mems[:limit]

    def delete(self, key: str) -> bool:
        """Delete a memory by key."""
        key_lower = key.lower().strip()
        before = len(self._memories)
        self._memories = [m for m in self._memories if m["key"].lower() != key_lower]
        if len(self._memories) < before:
            self._save()
            logger.info(f"Deleted memory: {key}")
            return True
        return False

    def count(self) -> int:
        """Return total number of stored memories."""
        return len(self._memories)

    def get_prompt_section(self, limit: int = 5) -> str:
        """
        Get a formatted string of recent memories for system prompt injection.

        Returns empty string if no memories exist.
        """
        recent = self.list_recent(limit)
        if not recent:
            return ""

        lines = ["## Agent Memory (persistent across conversations)"]
        for mem in recent:
            lines.append(f"- **{mem['key']}**: {mem['value']}")
        lines.append("")
        return "\n".join(lines)
