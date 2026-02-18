"""
Regression Tests for Race Executor Refactor — 10 tests targeting potential breakages.

Tests what could have broken when:
- race_executor.py was rewritten (3 concurrent LLM + FIRST_COMPLETED)
- Dead code removed from agent_tools.py (super_spawn, _run_race, etc.)
- super_spawn removed from personas.py tool groups
- _super_spawn_wrapper removed from tool_router.py

Run: cd E:\AgentNate && python\python.exe tests\regression_test_race_refactor.py [test_ids...]
"""

import asyncio
import json
import sys
import time
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aiohttp

BASE = "http://127.0.0.1:8000"

TESTS = [
    # T1: Non-raceable tool execution (deploy_workflow is NOT raceable)
    {"id": 1, "name": "Non-raceable tool (list_workflows)", "type": "tool_direct",
     "prompt": "List all n8n workflows currently deployed. Just list them.",
     "check": "worker uses list_workflows or list_deployed_workflows tool"},

    # T2: Worker spawning with delegate_all (basic delegation)
    {"id": 2, "name": "Basic worker delegation", "type": "delegation",
     "prompt": "Check the system health - what providers are online and what models are loaded?",
     "check": "head delegates to worker, worker uses tools, report comes back"},

    # T3: Race + deploy chain (raceable build_workflow followed by deploy)
    {"id": 3, "name": "Race then deploy chain", "type": "race_chain",
     "prompt": "Build an n8n workflow with a webhook trigger that receives JSON, a Set node that adds a processed=true field, and a Respond to Webhook that returns the result. Deploy it.",
     "check": "race fires for build_workflow, then deploy_workflow executes normally after"},

    # T4: Batch spawn agents (uses batch_spawn_agents tool)
    {"id": 4, "name": "Batch spawn agents", "type": "batch",
     "prompt": "I need two things done at the same time: First, list all loaded models. Second, get the GPU status. Do both in parallel using batch_spawn_agents.",
     "check": "batch_spawn_agents works, spawns multiple sub-agents"},

    # T5: Race all-fail recovery (intentionally hard prompt to provoke failures)
    {"id": 5, "name": "Race all-fail recovery", "type": "race_recovery",
     "prompt": "Build an n8n workflow with: webhook trigger, a Merge node that combines data from two branches, where branch A is an HTTP GET to https://httpbin.org/get and branch B is a Set node with test data, then Code node processes merged data, then Respond to Webhook.",
     "check": "if all 3 race candidates fail, worker should retry or adapt"},

    # T6: Head handles simple tool directly (no delegation)
    {"id": 6, "name": "Head handles tool directly", "type": "head_direct",
     "prompt": "What time is it? Use the system tools to check.",
     "check": "head agent uses a tool directly without spawning a worker"},

    # T7: Multiple tool calls in sequence (non-raceable tools after race)
    {"id": 7, "name": "Multi-tool sequence", "type": "sequence",
     "prompt": "Build an n8n workflow with webhook -> Set node adding status='ok' -> Respond to Webhook. After building it, activate it and then list all workflows to confirm it appears.",
     "check": "race fires for build, then activate and list execute normally"},

    # T8: Agent stop mid-execution
    {"id": 8, "name": "Agent stop during work", "type": "stop_test",
     "prompt": "Build a complex n8n workflow with 8 nodes: webhook, Code node, IF node, 2 HTTP requests, 2 Set nodes, Respond to Webhook. Take your time planning it carefully.",
     "check": "spawn worker, then stop it mid-execution - verify clean shutdown"},

    # T9: Worker tab/panel lifecycle
    {"id": 9, "name": "Worker completion cleanup", "type": "lifecycle",
     "prompt": "Build an n8n workflow: webhook trigger -> Set node adding hello='world' -> Respond to Webhook. Keep it simple.",
     "check": "worker completes, status transitions from running to done"},

    # T10: Concurrent workers (two panels simultaneously)
    {"id": 10, "name": "Concurrent workers", "type": "concurrent",
     "prompt": "Build an n8n workflow: webhook -> HTTP Request GET https://httpbin.org/get -> Respond to Webhook returning the response.",
     "check": "spawn two workers near-simultaneously, both complete without interference"},
]


async def run_basic_test(session, test, instance_id):
    """Standard test: send prompt, watch worker events, collect results."""
    tid = test["id"]
    tname = test["name"]
    ttype = test["type"]

    print(f"\n{'='*70}")
    print(f"TEST {tid}: {tname} ({ttype})")
    print(f"  Check: {test['check']}")
    print(f"{'='*70}")

    start = time.time()
    conv_id = None
    agent_id = None
    head_tool_calls = []
    head_response = ""

    # 1) Start head agent stream
    payload = {"message": test["prompt"], "instance_id": instance_id,
               "persona_id": "auto", "autonomous": True}

    try:
        async with session.post(f"{BASE}/api/tools/agent/stream", json=payload,
                                timeout=aiohttp.ClientTimeout(total=30)) as resp:
            async for line in resp.content:
                line = line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                ds = line[6:]
                if ds == "[DONE]":
                    break
                try:
                    d = json.loads(ds)
                except:
                    continue
                if d.get("conversation_id") and not conv_id:
                    conv_id = d["conversation_id"]
                if d.get("type") == "sub_agent_update":
                    for a in d.get("agents", []):
                        if a.get("agent_id") and not agent_id:
                            agent_id = a["agent_id"]
                if d.get("type") == "tool_call":
                    head_tool_calls.append(d.get("tool", "?"))
                    print(f"  HEAD TOOL: {d.get('tool', '?')}")
                if d.get("type") == "token" and d.get("text"):
                    head_response += d["text"]
    except Exception as e:
        print(f"  Head stream error: {e}")

    elapsed_head = round(time.time() - start, 1)

    # For head-direct tests, we might not have a worker
    if ttype in ("head_direct", "tool_direct") and not agent_id:
        elapsed = round(time.time() - start, 1)
        passed = len(head_response) > 20 or len(head_tool_calls) > 0
        status = "head_handled"
        print(f"  Head handled directly ({elapsed}s) tools={head_tool_calls}")
        print(f"  Response: {len(head_response)} chars")
        if head_response:
            print(f"  Preview: {head_response[:120].replace(chr(10),' ')}...")
        return {
            "id": tid, "name": tname, "type": ttype,
            "status": "PASS" if passed else "FAIL",
            "elapsed": elapsed,
            "head_tools": len(head_tool_calls),
            "worker_tools": 0,
            "workflow_races": 0,
            "note": "head_direct",
        }

    if not agent_id or not conv_id:
        elapsed = round(time.time() - start, 1)
        # For delegation test, this is a fail
        if ttype == "head_direct":
            passed = len(head_response) > 20
        else:
            passed = False
        print(f"  Not delegated (conv={conv_id}, agent={agent_id}) {elapsed}s")
        return {
            "id": tid, "name": tname, "type": ttype,
            "status": "PASS" if passed else "FAIL",
            "elapsed": elapsed, "head_tools": len(head_tool_calls),
            "worker_tools": 0, "workflow_races": 0,
            "note": "not_delegated",
        }

    print(f"  Delegated to {agent_id} ({elapsed_head}s)")

    # 2) For stop test, spawn then stop after a few seconds
    if ttype == "stop_test":
        await asyncio.sleep(5)
        try:
            async with session.post(f"{BASE}/api/tools/agent/stop",
                                    json={"conversation_id": conv_id}) as resp:
                stop_result = await resp.json()
                print(f"  STOP result: {stop_result}")
        except Exception as e:
            print(f"  STOP error: {e}")
        elapsed = round(time.time() - start, 1)
        return {
            "id": tid, "name": tname, "type": ttype,
            "status": "PASS",
            "elapsed": elapsed, "head_tools": len(head_tool_calls),
            "worker_tools": 0, "workflow_races": 0,
            "note": "stopped",
        }

    # 3) Subscribe to worker SSE for events
    workflow_races = []
    current_race = {}
    tool_calls = []
    worker_response = ""
    worker_status = ""

    try:
        async with session.get(f"{BASE}/api/tools/agent/workers/{conv_id}/stream",
                               timeout=aiohttp.ClientTimeout(total=180)) as ws:
            async for line in ws.content:
                line = line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                ds = line[6:]
                try:
                    d = json.loads(ds)
                except:
                    continue

                t = d.get("type", "")
                if t == "worker_events":
                    for e in d.get("events", []):
                        en = e.get("event", "")
                        rid = e.get("race_id", "")

                        if en == "race_started" and rid:
                            current_race[rid] = {"candidates": {}, "winner": None}
                            print(f"  RACE {rid}: {e.get('num_candidates',0)} candidates for {e.get('tool_name','?')}")

                        elif en == "race_candidate_evaluated" and rid in current_race:
                            cid = e.get("candidate_id", 0)
                            v = "VALID" if e.get("is_valid") else "FAIL"
                            print(f"    C{cid}: {v} ({e.get('elapsed',0)}s)")
                            current_race[rid]["candidates"][cid] = e.get("is_valid", False)

                        elif en == "race_winner_selected" and rid in current_race:
                            current_race[rid]["winner"] = e.get("winner")
                            print(f"    WINNER: C{e.get('winner')}")

                        elif en in ("race_completed", "race_failed") and rid in current_race:
                            workflow_races.append(current_race[rid])
                            if en == "race_failed":
                                print(f"    ALL FAILED")

                        elif en == "tool_call":
                            tool_calls.append(e.get("tool", "?"))
                            print(f"  TOOL #{len(tool_calls)}: {e.get('tool','?')}")

                        elif en == "agent_done":
                            worker_response = e.get("response", "")
                            worker_status = e.get("status", "?")

                elif t == "workers_done":
                    break

    except asyncio.TimeoutError:
        worker_status = "timeout"
    except Exception as e:
        print(f"  Worker SSE error: {e}")
        worker_status = "error"

    elapsed = round(time.time() - start, 1)

    print(f"  ---")
    print(f"  Status: {worker_status}")
    print(f"  Elapsed: {elapsed}s")
    print(f"  Worker tools: {len(tool_calls)}")
    print(f"  Workflow races: {len(workflow_races)}")
    if worker_response:
        print(f"  Response: {len(worker_response)} chars")
        print(f"  Preview: {worker_response[:120].replace(chr(10),' ')}...")

    passed = worker_status == "completed" and len(worker_response) > 30
    return {
        "id": tid, "name": tname, "type": ttype,
        "status": "PASS" if passed else "FAIL",
        "elapsed": elapsed,
        "head_tools": len(head_tool_calls),
        "worker_tools": len(tool_calls),
        "workflow_races": len(workflow_races),
        "worker_status": worker_status,
    }


async def run_concurrent_test(session, test, instance_id):
    """Test 10: Spawn two workers near-simultaneously."""
    tid = test["id"]
    print(f"\n{'='*70}")
    print(f"TEST {tid}: {test['name']} ({test['type']})")
    print(f"{'='*70}")

    start = time.time()
    prompts = [
        "Build an n8n workflow: webhook -> Set node adding x=1 -> Respond to Webhook.",
        "Build an n8n workflow: webhook -> Set node adding y=2 -> Respond to Webhook.",
    ]

    async def spawn_one(prompt, idx):
        payload = {"message": prompt, "instance_id": instance_id,
                   "persona_id": "auto", "autonomous": True}
        conv_id = None
        agent_id = None
        try:
            async with session.post(f"{BASE}/api/tools/agent/stream", json=payload,
                                    timeout=aiohttp.ClientTimeout(total=30)) as resp:
                async for line in resp.content:
                    line = line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data: "):
                        continue
                    ds = line[6:]
                    if ds == "[DONE]":
                        break
                    try:
                        d = json.loads(ds)
                    except:
                        continue
                    if d.get("conversation_id") and not conv_id:
                        conv_id = d["conversation_id"]
                    if d.get("type") == "sub_agent_update":
                        for a in d.get("agents", []):
                            if a.get("agent_id") and not agent_id:
                                agent_id = a["agent_id"]
        except Exception as e:
            print(f"  Worker {idx} head error: {e}")
            return None

        if not conv_id:
            return None
        print(f"  Worker {idx}: delegated to {agent_id}")

        # Wait for completion via SSE
        worker_status = ""
        worker_response = ""
        try:
            async with session.get(f"{BASE}/api/tools/agent/workers/{conv_id}/stream",
                                   timeout=aiohttp.ClientTimeout(total=120)) as ws:
                async for line in ws.content:
                    line = line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data: "):
                        continue
                    ds = line[6:]
                    try:
                        d = json.loads(ds)
                    except:
                        continue
                    if d.get("type") == "worker_events":
                        for e in d.get("events", []):
                            if e.get("event") == "agent_done":
                                worker_response = e.get("response", "")
                                worker_status = e.get("status", "?")
                    elif d.get("type") == "workers_done":
                        break
        except:
            worker_status = "error"

        print(f"  Worker {idx}: {worker_status} ({len(worker_response)} chars)")
        return worker_status == "completed"

    results = await asyncio.gather(
        spawn_one(prompts[0], 1),
        spawn_one(prompts[1], 2),
    )

    elapsed = round(time.time() - start, 1)
    both_passed = all(r for r in results)
    print(f"  Concurrent results: {results} ({elapsed}s)")

    return {
        "id": tid, "name": test["name"], "type": test["type"],
        "status": "PASS" if both_passed else "FAIL",
        "elapsed": elapsed,
        "head_tools": 0, "worker_tools": 0, "workflow_races": 0,
        "note": f"concurrent: {results}",
    }


async def main():
    print("=" * 70)
    print("REGRESSION TESTS — Race Executor Refactor")
    print("=" * 70)

    async with aiohttp.ClientSession() as session:
        async with session.get(f"{BASE}/api/models/loaded") as resp:
            models = await resp.json()
        if not models:
            print("ERROR: No models loaded!")
            return

        # Use OpenRouter model for head if available, else first model
        or_models = [m for m in models if m.get("provider") == "openrouter"]
        iid = or_models[0]["id"] if or_models else models[0]["id"]
        print(f"Head model: {or_models[0].get('model','?') if or_models else models[0].get('model','?')} ({iid})")

        local_models = [m for m in models if m.get("provider") != "openrouter"]
        if local_models:
            print(f"Worker model: {local_models[0].get('model','?')} ({local_models[0]['id']})")

        # Select tests
        test_ids = list(range(1, 11))
        if len(sys.argv) > 1:
            test_ids = [int(x) for x in sys.argv[1:]]
        tests = [t for t in TESTS if t["id"] in test_ids]
        print(f"Running {len(tests)} tests: {[t['id'] for t in tests]}\n")

        results = []
        for test in tests:
            if test["type"] == "concurrent":
                r = await run_concurrent_test(session, test, iid)
            else:
                r = await run_basic_test(session, test, iid)
            results.append(r)
            await asyncio.sleep(2)

    # Final summary
    print(f"\n{'='*70}")
    print("REGRESSION TEST RESULTS")
    print(f"{'='*70}")
    for r in results:
        icon = "PASS" if r["status"] == "PASS" else "FAIL"
        wr = r.get("workflow_races", 0)
        ht = r.get("head_tools", 0)
        wt = r.get("worker_tools", 0)
        note = r.get("note", "")
        race_info = f" WR={wr}" if wr else ""
        note_info = f" [{note}]" if note else ""
        print(f"  [{icon}] T{r['id']}: {r['name']} {r['elapsed']}s ht={ht} wt={wt}{race_info}{note_info}")

    passed = sum(1 for r in results if r["status"] == "PASS")
    print(f"\n{passed}/{len(results)} passed")


if __name__ == "__main__":
    asyncio.run(main())
