"""
Tests for Chat Saving Rework
"""
import sys
import os
import json
import tempfile
import shutil

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from backend.conversation_store import ConversationStore, Conversation, ConversationMetadata


def test_backward_compat_conversation():
    """Old conversations without saved/conv_type fields should load correctly."""
    old_data = {
        "id": "conv-abc123",
        "persona_id": "system_agent",
        "name": "Old Chat",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "messages": [{"role": "user", "content": "hello"}],
        "model_id": "test-model",
    }
    conv = Conversation.from_dict(old_data)
    assert conv.saved == True, f"Old conversation should default to saved=True, got {conv.saved}"
    assert conv.conv_type == "agent", f"Old conversation should default to conv_type='agent', got {conv.conv_type}"
    assert conv.metadata == {}, "Old conversation should get empty metadata"
    print("  PASS: backward_compat_conversation")


def test_backward_compat_metadata():
    """Old metadata without saved/conv_type fields should load correctly."""
    old_data = {
        "id": "conv-abc123",
        "persona_id": "system_agent",
        "name": "Old Chat",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "message_count": 5,
        "model_id": "test-model",
    }
    meta = ConversationMetadata.from_dict(old_data)
    assert meta.saved == True, f"Old metadata should default to saved=True, got {meta.saved}"
    assert meta.conv_type == "agent", f"Old metadata should default to conv_type='agent', got {meta.conv_type}"
    print("  PASS: backward_compat_metadata")


def test_new_conversation_defaults():
    """New conversations should have saved=False by default."""
    conv = Conversation(
        id="conv-new123",
        persona_id="power_agent",
        name=None,
        created_at="2026-02-12T00:00:00Z",
        updated_at="2026-02-12T00:00:00Z",
    )
    assert conv.saved == False, f"New conversation should default to saved=False, got {conv.saved}"
    assert conv.conv_type == "agent", f"New conversation should default to conv_type='agent', got {conv.conv_type}"
    print("  PASS: new_conversation_defaults")


def test_to_metadata_includes_new_fields():
    """to_metadata() should include saved and conv_type."""
    conv = Conversation(
        id="conv-test1",
        persona_id="power_agent",
        name="Test",
        created_at="2026-02-12T00:00:00Z",
        updated_at="2026-02-12T00:00:00Z",
        saved=True,
        conv_type="chat",
    )
    meta = conv.to_metadata()
    assert meta.saved == True
    assert meta.conv_type == "chat"
    print("  PASS: to_metadata_includes_new_fields")


def test_to_dict_includes_new_fields():
    """to_dict() should include saved and conv_type."""
    conv = Conversation(
        id="conv-test2",
        persona_id="power_agent",
        name="Test",
        created_at="2026-02-12T00:00:00Z",
        updated_at="2026-02-12T00:00:00Z",
        saved=True,
        conv_type="chat",
    )
    d = conv.to_dict()
    assert d["saved"] == True
    assert d["conv_type"] == "chat"
    print("  PASS: to_dict_includes_new_fields")


def test_store_create_saved():
    """create_saved() should create a conversation with saved=True."""
    tmp_dir = tempfile.mkdtemp()
    try:
        store = ConversationStore(storage_dir=Path(tmp_dir))
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        conv_id = store.create_saved(
            messages=messages,
            name="My Chat",
            persona_id="none",
            model_id="test-model",
            conv_type="chat",
        )
        conv = store.get(conv_id)
        assert conv is not None
        assert conv.saved == True
        assert conv.conv_type == "chat"
        assert conv.name == "My Chat"
        assert len(conv.messages) == 2
        assert conv.messages[0]["content"] == "hello"
        print("  PASS: store_create_saved")
    finally:
        shutil.rmtree(tmp_dir)


def test_store_mark_saved():
    """mark_saved() should flip saved to True and optionally set name."""
    tmp_dir = tempfile.mkdtemp()
    try:
        store = ConversationStore(storage_dir=Path(tmp_dir))
        conv_id = store.create(persona_id="power_agent")

        conv = store.get(conv_id)
        assert conv.saved == False, "Newly created conv should be unsaved"

        success = store.mark_saved(conv_id, name="My Agent Chat")
        assert success == True

        conv = store.get(conv_id)
        assert conv.saved == True, "Should now be saved"
        assert conv.name == "My Agent Chat"
        print("  PASS: store_mark_saved")
    finally:
        shutil.rmtree(tmp_dir)


def test_list_all_saved_only():
    """list_all(saved_only=True) should exclude unsaved conversations."""
    tmp_dir = tempfile.mkdtemp()
    try:
        store = ConversationStore(storage_dir=Path(tmp_dir))

        # Create 2 unsaved (agent auto-created)
        id1 = store.create(persona_id="power_agent")
        id2 = store.create(persona_id="system_agent")

        # Create 1 saved
        id3 = store.create_saved(
            messages=[{"role": "user", "content": "test"}],
            name="Saved One",
        )

        # Mark one agent conv as saved
        store.mark_saved(id1, name="Saved Agent")

        saved = store.list_all(saved_only=True)
        all_convs = store.list_all(saved_only=False)

        assert len(saved) == 2, f"Expected 2 saved, got {len(saved)}"
        assert len(all_convs) == 3, f"Expected 3 total, got {len(all_convs)}"

        saved_ids = {m.id for m in saved}
        assert id1 in saved_ids
        assert id3 in saved_ids
        assert id2 not in saved_ids
        print("  PASS: list_all_saved_only")
    finally:
        shutil.rmtree(tmp_dir)


def test_cleanup_unsaved():
    """cleanup_unsaved() should remove only unsaved conversations."""
    tmp_dir = tempfile.mkdtemp()
    try:
        store = ConversationStore(storage_dir=Path(tmp_dir))

        # Create 2 unsaved
        id1 = store.create(persona_id="power_agent")
        id2 = store.create(persona_id="system_agent")

        # Create 1 saved
        id3 = store.create_saved(
            messages=[{"role": "user", "content": "test"}],
            name="Keep This",
        )

        # Verify all 3 exist
        assert len(store.list_all(saved_only=False)) == 3

        cleaned = store.cleanup_unsaved()
        assert cleaned == 2, f"Expected 2 cleaned, got {cleaned}"

        remaining = store.list_all(saved_only=False)
        assert len(remaining) == 1, f"Expected 1 remaining, got {len(remaining)}"
        assert remaining[0].id == id3
        assert remaining[0].name == "Keep This"

        # Verify files are gone
        assert not (Path(tmp_dir) / f"{id1}.json").exists()
        assert not (Path(tmp_dir) / f"{id2}.json").exists()
        assert (Path(tmp_dir) / f"{id3}.json").exists()
        print("  PASS: cleanup_unsaved")
    finally:
        shutil.rmtree(tmp_dir)


def test_startup_cleanup():
    """ConversationStore should clean up unsaved on init."""
    tmp_dir = tempfile.mkdtemp()
    try:
        # First session: create conversations
        store1 = ConversationStore(storage_dir=Path(tmp_dir))
        id1 = store1.create(persona_id="power_agent")  # unsaved
        id2 = store1.create_saved(
            messages=[{"role": "user", "content": "keep me"}],
            name="Persistent Chat",
        )
        del store1

        # Second session: re-init should clean up unsaved
        store2 = ConversationStore(storage_dir=Path(tmp_dir))
        all_convs = store2.list_all(saved_only=False)
        assert len(all_convs) == 1, f"Expected 1 after cleanup, got {len(all_convs)}"
        assert all_convs[0].id == id2
        assert all_convs[0].name == "Persistent Chat"
        print("  PASS: startup_cleanup")
    finally:
        shutil.rmtree(tmp_dir)


def test_roundtrip_json():
    """saved and conv_type should survive JSON serialization roundtrip."""
    conv = Conversation(
        id="conv-rt123",
        persona_id="none",
        name="Roundtrip Test",
        created_at="2026-02-12T00:00:00Z",
        updated_at="2026-02-12T00:00:00Z",
        messages=[{"role": "user", "content": "test"}],
        saved=True,
        conv_type="chat",
    )
    json_str = json.dumps(conv.to_dict())
    loaded = Conversation.from_dict(json.loads(json_str))
    assert loaded.saved == True
    assert loaded.conv_type == "chat"
    assert loaded.messages[0]["content"] == "test"
    print("  PASS: roundtrip_json")


if __name__ == "__main__":
    print("=== Chat Saving Rework Tests ===\n")

    tests = [
        test_backward_compat_conversation,
        test_backward_compat_metadata,
        test_new_conversation_defaults,
        test_to_metadata_includes_new_fields,
        test_to_dict_includes_new_fields,
        test_store_create_saved,
        test_store_mark_saved,
        test_list_all_saved_only,
        test_cleanup_unsaved,
        test_startup_cleanup,
        test_roundtrip_json,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed == 0:
        print("All tests passed!")
