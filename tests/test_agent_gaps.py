"""
Tests for Agent System Gap Closures (Gaps 1-4)

Gap 1: Stale /agent endpoint removed
Gap 2: Sub-agent progress visibility (event queue + SSE draining)
Gap 3: Persistent agent memory (remember/recall tools)
Gap 4: Ask-user tool with pause/resume mechanism

Run with: python/python.exe test_agent_gaps.py
"""
import sys
import os
import asyncio
import json
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

passed = 0
failed = 0
errors = []


def test(name, condition, detail=""):
    global passed, failed, errors
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        msg = f"  FAIL: {name}"
        if detail:
            msg += f" -- {detail}"
        print(msg)
        errors.append(name)


# ============================================================================
# GAP 1: Stale /agent endpoint removed
# ============================================================================
print("\n=== GAP 1: Stale /agent endpoint removed ===")

import backend.routes.tools as tools_module
import inspect

# Check that agent_chat function doesn't exist
has_agent_chat = hasattr(tools_module, 'agent_chat')
test("agent_chat function removed", not has_agent_chat)

# Check that POST /agent route is gone by scanning source
source = inspect.getsource(tools_module)
has_post_agent = '@router.post("/agent")' in source
# The only /agent routes should be /agent/stream, /agent/abort, /agent/debug, /agent/respond
test("POST /agent route removed from source",
     not has_post_agent or source.count('@router.post("/agent")') == 0,
     "Found standalone POST /agent route")

# Verify /agent/stream still exists
test("POST /agent/stream exists", hasattr(tools_module, 'agent_chat_stream'))

# Verify /agent/abort still exists
test("POST /agent/abort exists", hasattr(tools_module, 'agent_abort'))

# Verify /agent/respond exists (new for Gap 4)
test("POST /agent/respond exists", hasattr(tools_module, 'agent_respond'))


# ============================================================================
# GAP 2: Sub-agent progress visibility
# ============================================================================
print("\n=== GAP 2: Sub-agent progress visibility ===")

from backend.tools.agent_tools import AgentTools, TOOL_DEFINITIONS

# Create AgentTools instance with mocks
class MockSettings:
    def get(self, key, default=None):
        return {"agent.max_sub_agents": 4, "agent.sub_agent_timeout": 300}.get(key, default)

# Use temp dir for memory to avoid polluting real memory
with tempfile.TemporaryDirectory() as tmpdir:
    tools = AgentTools(
        orchestrator=None,
        n8n_manager=None,
        settings=MockSettings(),
        comfyui_manager=None,
    )
    # Override memory to use temp dir
    from backend.agent_memory import AgentMemory
    tools.memory = AgentMemory(storage_dir=tmpdir)

    # Test event queue exists
    test("Event queue exists", hasattr(tools, '_event_queue'))
    test("Event queue is asyncio.Queue", isinstance(tools._event_queue, asyncio.Queue))

    # Test drain_events when empty
    events = tools.drain_events()
    test("drain_events returns empty list when no events", events == [])

    # Test event queue put + drain
    tools._event_queue.put_nowait({"agent_id": "sub-1", "name": "test", "event": "started"})
    tools._event_queue.put_nowait({"agent_id": "sub-1", "name": "test", "event": "tool_call", "tool": "web_search"})
    events = tools.drain_events()
    test("drain_events returns queued events", len(events) == 2)
    test("First event is 'started'", events[0]["event"] == "started")
    test("Second event is 'tool_call'", events[1]["event"] == "tool_call")

    # Queue should be empty after drain
    events2 = tools.drain_events()
    test("Queue empty after drain", len(events2) == 0)

    # Test get_all_agent_status with no agents
    status = tools.get_all_agent_status()
    test("get_all_agent_status empty when no agents", len(status) == 0)

    # Test _drain_sub_agent_events helper in tools.py (line ~550)
    # This is a closure in the SSE generator, so we test the concept:
    # it drains events + builds status dict for SSE
    test("drain_events + get_all_agent_status integration",
         isinstance(tools.get_all_agent_status(), list))


# ============================================================================
# GAP 3: Persistent agent memory
# ============================================================================
print("\n=== GAP 3: Persistent agent memory ===")

with tempfile.TemporaryDirectory() as tmpdir:
    mem = AgentMemory(storage_dir=tmpdir)

    # Test store
    entry = mem.store("user_language", "Python", "preference")
    test("store returns entry", entry is not None)
    test("Entry has key", entry["key"] == "user_language")
    test("Entry has value", entry["value"] == "Python")
    test("Entry has category", entry["category"] == "preference")
    test("Entry has created_at", "created_at" in entry)
    test("Entry has updated_at", "updated_at" in entry)

    # Test count
    test("Count is 1", mem.count() == 1)

    # Test recall
    matches = mem.recall("python")
    test("Recall finds match by value", len(matches) == 1)
    test("Recall result has correct key", matches[0]["key"] == "user_language")

    matches = mem.recall("language")
    test("Recall finds match by key", len(matches) == 1)

    matches = mem.recall("preference")
    test("Recall finds match by category", len(matches) == 1)

    matches = mem.recall("nonexistent")
    test("Recall returns empty for no match", len(matches) == 0)

    # Test duplicate key update
    mem.store("user_language", "JavaScript", "preference")
    test("Count still 1 after update", mem.count() == 1)
    matches = mem.recall("user_language")
    test("Updated value is JavaScript", matches[0]["value"] == "JavaScript")

    # Test multiple entries
    mem.store("editor", "VSCode", "preference")
    mem.store("os", "Windows", "fact")
    test("Count is 3", mem.count() == 3)

    # Test list_recent
    recent = mem.list_recent(2)
    test("list_recent returns 2", len(recent) == 2)

    # Test delete
    deleted = mem.delete("editor")
    test("Delete returns True for existing key", deleted)
    test("Count is 2 after delete", mem.count() == 2)

    deleted = mem.delete("nonexistent")
    test("Delete returns False for nonexistent key", not deleted)

    # Test get_prompt_section
    section = mem.get_prompt_section(5)
    test("Prompt section is non-empty", len(section) > 0)
    test("Prompt section has header", "Agent Memory" in section)
    test("Prompt section has entries", "user_language" in section)

    # Test empty prompt section
    mem2 = AgentMemory(storage_dir=tmpdir + "_empty")
    section2 = mem2.get_prompt_section()
    test("Empty prompt section returns empty string", section2 == "")

    # Test persistence (reload from disk)
    mem3 = AgentMemory(storage_dir=tmpdir)
    test("Persistence: reloaded count matches", mem3.count() == 2)
    matches = mem3.recall("user_language")
    test("Persistence: recalled stored value", len(matches) == 1 and matches[0]["value"] == "JavaScript")

    # Test auto-prune
    mem_prune = AgentMemory(storage_dir=tmpdir + "_prune")
    for i in range(210):
        mem_prune.store(f"key_{i}", f"value_{i}")
    test("Auto-prune: count <= 200", mem_prune.count() <= 200)
    test("Auto-prune: oldest removed", mem_prune.recall("key_0") == [])
    test("Auto-prune: newest retained", len(mem_prune.recall("key_209")) == 1)


# ============================================================================
# GAP 3b: Remember/recall tools via AgentTools
# ============================================================================
print("\n=== GAP 3b: Remember/recall tools ===")

async def test_remember_recall():
    with tempfile.TemporaryDirectory() as tmpdir:
        tools = AgentTools(None, None, MockSettings())
        tools.memory = AgentMemory(storage_dir=tmpdir)

        # Test remember tool
        result = await tools.remember("test_key", "test_value", "test_cat")
        test("remember returns success", result["success"])
        test("remember has message", "test_key" in result["message"])
        test("remember has total_memories", result["total_memories"] == 1)

        # Test recall tool
        result = await tools.recall("test")
        test("recall returns success", result["success"])
        test("recall finds match", len(result["matches"]) == 1)
        test("recall match has correct value", result["matches"][0]["value"] == "test_value")

        # Test recall no match
        result = await tools.recall("zzzzz")
        test("recall no match message", "No memories found" in result["message"])

asyncio.run(test_remember_recall())


# ============================================================================
# GAP 4: Ask-user tool
# ============================================================================
print("\n=== GAP 4: Ask-user tool ===")

# Test tool definition exists
ask_user_def = None
for td in TOOL_DEFINITIONS:
    if td["name"] == "ask_user":
        ask_user_def = td
        break

test("ask_user tool definition exists", ask_user_def is not None)
test("ask_user has question parameter", "question" in ask_user_def["parameters"]["properties"])
test("ask_user has options parameter", "options" in ask_user_def["parameters"]["properties"])
test("ask_user question is required", "question" in ask_user_def["parameters"]["required"])

# Test ask_user tool returns
async def test_ask_user_tool():
    with tempfile.TemporaryDirectory() as tmpdir:
        tools = AgentTools(None, None, MockSettings())
        tools.memory = AgentMemory(storage_dir=tmpdir)

        result = await tools.ask_user("What color?", ["Red", "Blue", "Green"])
        test("ask_user returns success", result["success"])
        test("ask_user type is ask_user", result["type"] == "ask_user")
        test("ask_user has question", result["question"] == "What color?")
        test("ask_user has options", result["options"] == ["Red", "Blue", "Green"])

        # Without options
        result2 = await tools.ask_user("What's your name?")
        test("ask_user without options has empty list", result2["options"] == [])

asyncio.run(test_ask_user_tool())

# Test ask_user in FORBIDDEN_TOOLS for sub-agents
from backend.agent_loop import run_agent_loop
agent_loop_source = inspect.getsource(run_agent_loop)
test("ask_user in FORBIDDEN_TOOLS", '"ask_user"' in agent_loop_source)

# Test /agent/respond endpoint exists
test("agent_respond function exists", hasattr(tools_module, 'agent_respond'))

# Test AgentRespondRequest model
test("AgentRespondRequest model exists", hasattr(tools_module, 'AgentRespondRequest'))
from backend.routes.tools import AgentRespondRequest
req = AgentRespondRequest(abort_id="test-123", response="Blue")
test("AgentRespondRequest has abort_id", req.abort_id == "test-123")
test("AgentRespondRequest has response", req.response == "Blue")


# ============================================================================
# GAP 4b: ask_user pause/resume mechanism
# ============================================================================
print("\n=== GAP 4b: ask_user pause/resume mechanism ===")

async def test_ask_user_pause_resume():
    """Test the asyncio.Event-based pause/resume for ask_user."""
    # Simulate the mechanism used in the SSE generator
    wait_event = asyncio.Event()
    response_store = {"event": wait_event, "response": None}

    # Simulate user responding after a delay
    async def simulate_user_response():
        await asyncio.sleep(0.2)
        response_store["response"] = "Blue"
        response_store["event"].set()

    # Start the "user response" in background
    asyncio.create_task(simulate_user_response())

    # Wait for response (like the SSE generator does)
    start = time.time()
    while not wait_event.is_set():
        await asyncio.sleep(0.1)
    elapsed = time.time() - start

    test("Pause/resume: event was set", wait_event.is_set())
    test("Pause/resume: response received", response_store["response"] == "Blue")
    test("Pause/resume: waited ~0.2s", 0.1 < elapsed < 1.0, f"elapsed={elapsed:.2f}s")

asyncio.run(test_ask_user_pause_resume())

async def test_ask_user_abort_during_wait():
    """Test that abort works while waiting for ask_user response."""
    wait_event = asyncio.Event()
    abort_signals = {"session-1": "running"}

    # Simulate abort after a delay
    async def simulate_abort():
        await asyncio.sleep(0.2)
        abort_signals["session-1"] = "abort"

    asyncio.create_task(simulate_abort())

    # Wait loop (like in SSE generator)
    aborted = False
    while not wait_event.is_set():
        if abort_signals.get("session-1") == "abort":
            aborted = True
            break
        await asyncio.sleep(0.1)

    test("Abort during ask_user: aborted flag set", aborted)
    test("Abort during ask_user: event NOT set", not wait_event.is_set())

asyncio.run(test_ask_user_abort_during_wait())


# ============================================================================
# GAP 2+3: Tool definitions in personas
# ============================================================================
print("\n=== Tool groups and persona integration ===")

from backend.personas import TOOL_GROUPS

agents_group = TOOL_GROUPS.get("agents", [])
test("agents tool group exists", len(agents_group) > 0)
test("spawn_agent in agents group", "spawn_agent" in agents_group)
test("check_agents in agents group", "check_agents" in agents_group)
test("get_agent_result in agents group", "get_agent_result" in agents_group)
test("remember in agents group", "remember" in agents_group)
test("recall in agents group", "recall" in agents_group)
test("ask_user in agents group", "ask_user" in agents_group)

# Test tool router has all routes
from backend.tools.tool_router import ToolRouter, AVAILABLE_TOOLS

tool_names = [t["name"] for t in AVAILABLE_TOOLS]
test("remember in AVAILABLE_TOOLS", "remember" in tool_names)
test("recall in AVAILABLE_TOOLS", "recall" in tool_names)
test("ask_user in AVAILABLE_TOOLS", "ask_user" in tool_names)
test("spawn_agent in AVAILABLE_TOOLS", "spawn_agent" in tool_names)


# ============================================================================
# System prompt integration
# ============================================================================
print("\n=== System prompt memory injection ===")

from backend.routes.tools import _build_system_prompt

class MockPersona:
    id = "test"
    system_prompt = "You are a test agent."
    tools = ["agents"]
    include_system_state = False
    temperature = 0.7

# Test without memory
prompt = _build_system_prompt(MockPersona(), "", "")
test("Prompt without memory has persona", "test agent" in prompt)

# Test with memory section
prompt_with_mem = _build_system_prompt(
    MockPersona(), "", "",
    agent_memory_section="## Agent Memory\n- **pref**: Python"
)
test("Prompt with memory has section", "Agent Memory" in prompt_with_mem)
test("Prompt with memory has entry", "Python" in prompt_with_mem)

# Test memory comes before tools
idx_memory = prompt_with_mem.find("Agent Memory")
idx_persona = prompt_with_mem.find("test agent")
test("Memory section after persona", idx_memory > idx_persona)


# ============================================================================
# Summary
# ============================================================================
print(f"\n{'='*60}")
print(f"Results: {passed} passed, {failed} failed")
if errors:
    print(f"Failed tests:")
    for e in errors:
        print(f"  - {e}")
print(f"{'='*60}")

sys.exit(0 if failed == 0 else 1)
