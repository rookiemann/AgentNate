"""
Benchmark rapid tool execution - measures per-tool overhead
without requiring a loaded LLM model.

Tests:
1. Raw ToolRouter.execute() throughput (50 calls)
2. parse_and_execute overhead (simulated LLM responses)
3. Conversation store append overhead
"""
import asyncio
import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.conversation_store import ConversationStore
from backend.tools.tool_router import ToolRouter


# Mock dependencies
class MockOrch:
    def get_loaded_instances(self): return []
    async def get_all_providers_health(self): return {}

class MockN8n:
    instances = {}
    main = None
    async def get_main_info(self): return None

class MockSettings:
    tools = {}
    def get(self, k, d=None): return d


async def bench_raw_execute(tr: ToolRouter, n: int):
    """Benchmark raw tool execution (fast tools only)."""
    start = time.perf_counter()
    for _ in range(n):
        await tr.execute("get_datetime", {})
    elapsed = time.perf_counter() - start
    print(f"  Raw execute ({n}x get_datetime): {elapsed*1000:.1f}ms  "
          f"({elapsed/n*1000:.3f}ms/call, {n/elapsed:.0f} calls/s)")


async def bench_parse_and_execute(tr: ToolRouter, n: int):
    """Benchmark parse_and_execute with simulated LLM responses."""
    # Simulate LLM tool call response format
    responses = []
    for i in range(n):
        r = json.dumps({"tool": "get_datetime", "arguments": {}})
        responses.append(f"I'll check the time.\n```json\n{r}\n```")

    start = time.perf_counter()
    for resp in responses:
        await tr.parse_and_execute(resp)
    elapsed = time.perf_counter() - start
    print(f"  Parse+execute ({n}x): {elapsed*1000:.1f}ms  "
          f"({elapsed/n*1000:.3f}ms/call)")


def bench_conversation_store(n: int):
    """Benchmark conversation store append speed."""
    import tempfile
    import shutil

    tmpdir = tempfile.mkdtemp(prefix="conv_bench_")
    store = ConversationStore(storage_dir=tmpdir)
    conv_id = store.create("bench_test")

    # Benchmark appending messages
    start = time.perf_counter()
    for i in range(n):
        store.append_message(conv_id, "assistant", f"Tool result {i}: success")
    store.flush(conv_id)  # Include the final disk write
    elapsed = time.perf_counter() - start
    print(f"  Conversation append ({n}x + flush): {elapsed*1000:.1f}ms  "
          f"({elapsed/n*1000:.3f}ms/call, {n/elapsed:.0f} writes/s)")

    # Check file size
    conv_path = os.path.join(tmpdir, f"{conv_id}.json")
    if os.path.exists(conv_path):
        size = os.path.getsize(conv_path)
        print(f"  Final file size: {size/1024:.1f}KB ({n} messages)")

    shutil.rmtree(tmpdir, ignore_errors=True)


async def main():
    print("=" * 65)
    print("  Tool Execution Rapid Benchmark")
    print("=" * 65)

    tr = ToolRouter(MockOrch(), MockN8n(), MockSettings())

    print("\n1. Raw tool execution:")
    await bench_raw_execute(tr, 50)
    await bench_raw_execute(tr, 200)

    print("\n2. Parse + execute (regex + JSON + dispatch):")
    await bench_parse_and_execute(tr, 50)

    print("\n3. Conversation store writes (sync file I/O):")
    bench_conversation_store(50)
    bench_conversation_store(200)

    print("\n4. Simulated 50-tool agent loop overhead:")
    # Simulate: parse -> execute -> store result -> build continuation
    import tempfile, shutil
    tmpdir = tempfile.mkdtemp(prefix="agent_bench_")
    store = ConversationStore(storage_dir=tmpdir)
    conv_id = store.create("agent_bench")

    start = time.perf_counter()
    for i in range(50):
        # Parse tool call
        resp = json.dumps({"tool": "get_datetime", "arguments": {}})
        llm_output = f"Checking time.\n```json\n{resp}\n```"
        result = await tr.parse_and_execute(llm_output)

        # Store tool call + result
        store.append_message(conv_id, "assistant", llm_output)
        store.append_message(conv_id, "system",
                             f"Tool result: {json.dumps(result)}")

    store.flush(conv_id)  # Include final disk write
    elapsed = time.perf_counter() - start
    print(f"  50-tool loop (no LLM): {elapsed*1000:.1f}ms  "
          f"({elapsed/50*1000:.2f}ms/tool)")

    shutil.rmtree(tmpdir, ignore_errors=True)
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(main())
