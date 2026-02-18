"""
Conversation Store for Meta Agent

Persistent storage for conversation histories that survive restarts.
"""

import json
import logging
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("conversation_store")


def _get_config_dir() -> Path:
    """Get the appropriate config directory for the OS."""
    if os.name == 'nt':  # Windows
        base = os.environ.get('APPDATA', os.path.expanduser('~'))
        return Path(base) / 'AgentNate' / 'conversations'
    else:  # Linux/Mac
        base = os.environ.get('XDG_CONFIG_HOME', os.path.expanduser('~/.config'))
        return Path(base) / 'AgentNate' / 'conversations'


@dataclass
class ConversationMetadata:
    """
    Lightweight metadata about a conversation for listing.

    Attributes:
        id: Unique conversation ID
        persona_id: Which persona is active
        name: User-given name (optional)
        created_at: When the conversation was created
        updated_at: When the conversation was last updated
        message_count: Number of messages in the conversation
        model_id: Which model instance was used (if any)
    """
    id: str
    persona_id: str
    name: Optional[str]
    created_at: str  # ISO format
    updated_at: str  # ISO format
    message_count: int
    model_id: Optional[str] = None
    saved: bool = False
    conv_type: str = "agent"  # "chat" or "agent"

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "ConversationMetadata":
        """Create from dictionary. Backward-compatible with old conversations."""
        if "saved" not in data:
            data["saved"] = True  # Old conversations treated as saved
        if "conv_type" not in data:
            data["conv_type"] = "agent"
        return cls(**data)


@dataclass
class Conversation:
    """
    Full conversation data including messages.

    Attributes:
        id: Unique conversation ID
        persona_id: Which persona is active
        name: User-given name (optional)
        created_at: When the conversation was created
        updated_at: When the conversation was last updated
        messages: List of message dicts with role and content
        model_id: Which model instance was used (if any)
    """
    id: str
    persona_id: str
    name: Optional[str]
    created_at: str
    updated_at: str
    messages: List[Dict] = field(default_factory=list)
    model_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    saved: bool = False
    conv_type: str = "agent"  # "chat" or "agent"

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "Conversation":
        """Create from dictionary. Backward-compatible with old conversations."""
        if "metadata" not in data:
            data["metadata"] = {}
        if "saved" not in data:
            data["saved"] = True  # Old conversations treated as saved
        if "conv_type" not in data:
            data["conv_type"] = "agent"
        return cls(**data)

    def to_metadata(self) -> ConversationMetadata:
        """Extract metadata from this conversation."""
        return ConversationMetadata(
            id=self.id,
            persona_id=self.persona_id,
            name=self.name,
            created_at=self.created_at,
            updated_at=self.updated_at,
            message_count=len(self.messages),
            model_id=self.model_id,
            saved=self.saved,
            conv_type=self.conv_type,
        )


class ConversationStore:
    """
    Persistent storage for conversations.

    Uses JSON files for storage with an in-memory cache for performance.
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize the conversation store.

        Args:
            storage_dir: Directory for storing conversations. Defaults to OS-appropriate config dir.
        """
        self.storage_dir = Path(storage_dir) if storage_dir else _get_config_dir()
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self._cache: Dict[str, Conversation] = {}
        self._index: Dict[str, ConversationMetadata] = {}
        self._dirty: set = set()  # conv_ids with unsaved changes
        self._cache: Dict[str, Conversation] = {}  # in-memory cache

        self._load_index()
        cleaned = self.cleanup_unsaved()
        if cleaned:
            logger.info(f"Cleaned up {cleaned} unsaved conversations from previous session")

    def _get_conv_path(self, conv_id: str) -> Path:
        """Get the file path for a conversation."""
        return self.storage_dir / f"{conv_id}.json"

    def _load_index(self) -> None:
        """Load the conversation index for quick listing."""
        index_path = self.storage_dir / "index.json"

        if index_path.exists():
            try:
                with open(index_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for item in data.get("conversations", []):
                    meta = ConversationMetadata.from_dict(item)
                    self._index[meta.id] = meta
                logger.info(f"Loaded {len(self._index)} conversations from index")
            except Exception as e:
                logger.error(f"Failed to load index: {e}")
                self._rebuild_index()
        else:
            self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Rebuild the index from conversation files."""
        self._index.clear()

        for file_path in self.storage_dir.glob("conv-*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                conv = Conversation.from_dict(data)
                self._index[conv.id] = conv.to_metadata()
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")

        self._save_index()
        logger.info(f"Rebuilt index with {len(self._index)} conversations")

    def _save_index(self) -> None:
        """Save the conversation index."""
        index_path = self.storage_dir / "index.json"

        try:
            data = {
                "conversations": [m.to_dict() for m in self._index.values()]
            }
            with open(index_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save index: {e}")

    def _save_to_disk(self, conv: Conversation) -> None:
        """Mark conversation for deferred save. Call flush() to write."""
        self._cache[conv.id] = conv
        self._index[conv.id] = conv.to_metadata()
        self._dirty.add(conv.id)

    def _save_to_disk_now(self, conv: Conversation) -> None:
        """Immediately write a conversation to disk."""
        file_path = self._get_conv_path(conv.id)
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(conv.to_dict(), f, indent=2)
            self._index[conv.id] = conv.to_metadata()
        except Exception as e:
            logger.error(f"Failed to save conversation {conv.id}: {e}")

    def flush(self, conv_id: str = None) -> int:
        """
        Write dirty conversations to disk.

        Args:
            conv_id: If specified, flush only this conversation.
                     If None, flush all dirty conversations.

        Returns:
            Number of conversations flushed.
        """
        if conv_id:
            ids_to_flush = {conv_id} & self._dirty
        else:
            ids_to_flush = set(self._dirty)

        if not ids_to_flush:
            return 0

        count = 0
        for cid in ids_to_flush:
            conv = self._cache.get(cid)
            if conv:
                self._save_to_disk_now(conv)
                count += 1
            self._dirty.discard(cid)

        # Save index once for all flushed conversations
        if count > 0:
            self._save_index()

        return count

    def _load_from_disk(self, conv_id: str) -> Optional[Conversation]:
        """Load a conversation from disk."""
        file_path = self._get_conv_path(conv_id)

        if not file_path.exists():
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return Conversation.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load conversation {conv_id}: {e}")
            return None

    def create(self, persona_id: str, name: Optional[str] = None,
               model_id: Optional[str] = None) -> str:
        """
        Create a new conversation.

        Args:
            persona_id: Which persona to use
            name: Optional name for the conversation
            model_id: Optional model instance ID

        Returns:
            The new conversation ID
        """
        now = datetime.utcnow().isoformat() + "Z"
        conv_id = f"conv-{uuid.uuid4().hex[:8]}"

        conv = Conversation(
            id=conv_id,
            persona_id=persona_id,
            name=name,
            created_at=now,
            updated_at=now,
            messages=[],
            model_id=model_id,
        )

        self._cache[conv_id] = conv
        self._save_to_disk(conv)

        logger.info(f"Created conversation: {conv_id}")
        return conv_id

    def get(self, conv_id: str) -> Optional[Conversation]:
        """
        Get a conversation by ID.

        Args:
            conv_id: The conversation ID

        Returns:
            The conversation, or None if not found
        """
        # Check cache first
        if conv_id in self._cache:
            return self._cache[conv_id]

        # Load from disk
        conv = self._load_from_disk(conv_id)
        if conv:
            self._cache[conv_id] = conv

        return conv

    def append_message(self, conv_id: str, role: str, content: str) -> bool:
        """
        Append a message to a conversation.

        Args:
            conv_id: The conversation ID
            role: Message role (user, assistant, system)
            content: Message content

        Returns:
            True if successful, False if conversation not found
        """
        conv = self.get(conv_id)
        if not conv:
            return False

        conv.messages.append({"role": role, "content": content})
        conv.updated_at = datetime.utcnow().isoformat() + "Z"

        self._save_to_disk(conv)
        return True

    def get_messages(self, conv_id: str, limit: int = 10) -> List[Dict]:
        """
        Get recent messages from a conversation.

        Args:
            conv_id: The conversation ID
            limit: Maximum number of messages to return (from end)

        Returns:
            List of messages, or empty list if not found
        """
        conv = self.get(conv_id)
        if not conv:
            return []

        if limit <= 0:
            return conv.messages

        return conv.messages[-limit:]

    def list_all(self, saved_only: bool = True) -> List[ConversationMetadata]:
        """
        List conversations.

        Args:
            saved_only: If True (default), only return saved conversations.

        Returns:
            List of conversation metadata, sorted by updated_at descending
        """
        if saved_only:
            metadata = [m for m in self._index.values() if m.saved]
        else:
            metadata = list(self._index.values())
        metadata.sort(key=lambda m: m.updated_at, reverse=True)
        return metadata

    def delete(self, conv_id: str) -> bool:
        """
        Delete a conversation.

        Args:
            conv_id: The conversation ID

        Returns:
            True if deleted, False if not found
        """
        if conv_id not in self._index:
            return False

        # Remove from cache
        self._cache.pop(conv_id, None)

        # Remove from index
        del self._index[conv_id]
        self._save_index()

        # Delete file
        file_path = self._get_conv_path(conv_id)
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}")

        logger.info(f"Deleted conversation: {conv_id}")
        return True

    def rename(self, conv_id: str, name: str) -> bool:
        """
        Rename a conversation.

        Args:
            conv_id: The conversation ID
            name: New name

        Returns:
            True if renamed, False if not found
        """
        conv = self.get(conv_id)
        if not conv:
            return False

        conv.name = name
        conv.updated_at = datetime.utcnow().isoformat() + "Z"
        self._save_to_disk(conv)

        logger.info(f"Renamed conversation {conv_id} to: {name}")
        return True

    def set_persona(self, conv_id: str, persona_id: str) -> bool:
        """
        Change the persona for a conversation.

        Args:
            conv_id: The conversation ID
            persona_id: New persona ID

        Returns:
            True if updated, False if not found
        """
        conv = self.get(conv_id)
        if not conv:
            return False

        conv.persona_id = persona_id
        conv.updated_at = datetime.utcnow().isoformat() + "Z"

        # Clear stale intelligence caches that depend on persona
        conv.metadata.pop("selected_categories", None)
        conv.metadata.pop("context_summary", None)
        conv.metadata.pop("summarized_at_count", None)

        self._save_to_disk(conv)

        logger.info(f"Changed conversation {conv_id} persona to: {persona_id}")
        return True

    def set_model(self, conv_id: str, model_id: str) -> bool:
        """
        Set the model ID for a conversation.

        Args:
            conv_id: The conversation ID
            model_id: Model instance ID

        Returns:
            True if updated, False if not found
        """
        conv = self.get(conv_id)
        if not conv:
            return False

        conv.model_id = model_id
        conv.updated_at = datetime.utcnow().isoformat() + "Z"
        self._save_to_disk(conv)

        return True

    def get_metadata(self, conv_id: str) -> Dict[str, Any]:
        """Get conversation metadata dict. Returns empty dict if not found."""
        conv = self.get(conv_id)
        if not conv:
            return {}
        return conv.metadata

    def set_metadata(self, conv_id: str, key: str, value: Any) -> bool:
        """Set a single metadata key. Returns False if conversation not found."""
        conv = self.get(conv_id)
        if not conv:
            return False
        conv.metadata[key] = value
        self._save_to_disk(conv)
        return True

    def update_metadata(self, conv_id: str, updates: Dict[str, Any]) -> bool:
        """Merge multiple metadata keys. Returns False if conversation not found."""
        conv = self.get(conv_id)
        if not conv:
            return False
        conv.metadata.update(updates)
        self._save_to_disk(conv)
        return True

    def mark_saved(self, conv_id: str, name: str = None) -> bool:
        """Mark a conversation as saved (opt-in persistence)."""
        conv = self.get(conv_id)
        if not conv:
            return False
        conv.saved = True
        if name:
            conv.name = name
        conv.updated_at = datetime.utcnow().isoformat() + "Z"
        self._save_to_disk(conv)
        self.flush(conv_id)  # Immediately persist saved conversations
        logger.info(f"Marked conversation {conv_id} as saved")
        return True

    def create_saved(self, messages: List[Dict], name: str = "Untitled",
                     persona_id: str = "none", model_id: Optional[str] = None,
                     conv_type: str = "chat") -> str:
        """Create a new conversation that is already saved with pre-populated messages."""
        now = datetime.utcnow().isoformat() + "Z"
        conv_id = f"conv-{uuid.uuid4().hex[:8]}"

        conv = Conversation(
            id=conv_id,
            persona_id=persona_id,
            name=name,
            created_at=now,
            updated_at=now,
            messages=messages,
            model_id=model_id,
            saved=True,
            conv_type=conv_type,
        )

        self._cache[conv_id] = conv
        self._save_to_disk(conv)

        logger.info(f"Created saved conversation: {conv_id} ({conv_type})")
        return conv_id

    def cleanup_unsaved(self) -> int:
        """Delete all unsaved conversations (stale agent sessions from previous runs)."""
        to_delete = [cid for cid, meta in self._index.items() if not meta.saved]
        for cid in to_delete:
            self.delete(cid)
        return len(to_delete)

    def clear_cache(self) -> None:
        """Clear the in-memory cache."""
        self._cache.clear()
