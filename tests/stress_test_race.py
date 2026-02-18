"""
Stress Tests for Race Executor — 10 tests covering workflow racing and creative racing.

Tests 1-5: Workflow racing (first-valid-wins) — build_workflow tool calls
Tests 6-10: Creative racing (score-all) — long creative text responses

Run: cd E:\AgentNate && python\python.exe tests\stress_test_race.py [test_ids...]
Examples:
  python\python.exe tests\stress_test_race.py          # Run all 10
  python\python.exe tests\stress_test_race.py 1 6      # Run tests 1 and 6
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
    # --- Workflow Racing (1-5) ---
    {"id": 1, "name": "5-node webhook pipeline", "type": "workflow",
     "prompt": "Build me an n8n workflow with 5 nodes: a webhook trigger, an IF node that checks if the input has a 'type' field equal to 'urgent', if true: an HTTP Request node that POSTs to https://httpbin.org/post with the webhook data, if false: a Set node that adds a field 'status' = 'queued', and finally a Respond to Webhook node that returns the result."},
    {"id": 2, "name": "Data transform pipeline", "type": "workflow",
     "prompt": "Build an n8n workflow: webhook trigger -> Code node that transforms the input JSON (extracts 'name' and 'email' fields, lowercases the email) -> HTTP Request that POSTs the transformed data to https://httpbin.org/post -> Respond to Webhook that returns the HTTP response body."},
    {"id": 3, "name": "Multi-branch workflow", "type": "workflow",
     "prompt": "Create an n8n workflow with a webhook trigger connected to a Switch node that routes based on $json.action: case 'create' -> Set node (adds created_at timestamp) -> Respond to Webhook, case 'delete' -> Set node (adds deleted=true flag) -> Respond to Webhook, default -> Set node (adds error='unknown action') -> Respond to Webhook."},
    {"id": 4, "name": "API chain workflow", "type": "workflow",
     "prompt": "Build an n8n workflow: Webhook -> HTTP Request to https://httpbin.org/get -> Code node that extracts the 'origin' IP from the response -> another HTTP Request that POSTs {ip: extracted_ip} to https://httpbin.org/post -> Respond to Webhook returning the final result."},
    {"id": 5, "name": "Conditional echo workflow", "type": "workflow",
     "prompt": "Create an n8n workflow: Webhook trigger that receives JSON with 'message' and 'repeat' fields. IF node checks if repeat > 3, true branch: Code node that repeats the message 3 times, false branch: Set node that passes message unchanged. Both branches connect to Respond to Webhook."},
    # --- Creative Racing (6-10) ---
    {"id": 6, "name": "Epic fantasy story", "type": "creative",
     "prompt": "Write me a detailed epic fantasy story (at least 500 words) about a young blacksmith who discovers they can forge weapons that contain living spirits. Include vivid descriptions of the forge, the first magical weapon created, and the moment the spirit awakens. Make it dramatic and immersive."},
    {"id": 7, "name": "Sci-fi first contact", "type": "creative",
     "prompt": "Write a detailed science fiction narrative (at least 500 words) describing humanity's first contact with an alien civilization that communicates through colors and light patterns. Describe the alien ship, the translation challenges, and the moment of breakthrough. Include rich sensory details and emotional depth."},
    {"id": 8, "name": "Mystery short story", "type": "creative",
     "prompt": "Write a gripping mystery short story (at least 500 words) set in a lighthouse during a storm. The lighthouse keeper finds a locked room that shouldn't exist. Inside is a journal from 100 years ago that predicts events happening right now. Build tension gradually and include a twist ending."},
    {"id": 9, "name": "Philosophical essay", "type": "creative",
     "prompt": "Write a thoughtful philosophical essay (at least 500 words) exploring the question: 'If an AI becomes conscious, does it have the right to refuse its purpose?' Consider multiple perspectives: utilitarian, deontological, and existentialist. Use concrete examples and thought experiments."},
    {"id": 10, "name": "Adventure travelogue", "type": "creative",
     "prompt": "Write a vivid, detailed travelogue (at least 500 words) about an expedition to a newly discovered cave system beneath the Sahara desert. Describe the descent, the geological formations, the underground river, and the discovery of ancient cave paintings that rewrite human history."},
]


async def run_test(session, test, instance_id):
    tid = test["id"]
    tname = test["name"]
    ttype = test["type"]

    print(f"\n{'='*70}")
    print(f"TEST {tid}: {tname} ({ttype})")
    print(f"{'='*70}")

    start = time.time()
    conv_id = None
    agent_id = None

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
    except Exception as e:
        print(f"  Head stream error: {e}")
        return {"id": tid, "name": tname, "type": ttype, "status": "ERROR",
                "elapsed": round(time.time() - start, 1), "error": str(e)[:100],
                "workflow_races": 0, "creative_races": 0}

    if not agent_id or not conv_id:
        elapsed = round(time.time() - start, 1)
        print(f"  Not delegated (conv={conv_id}, agent={agent_id}) {elapsed}s")
        return {"id": tid, "name": tname, "type": ttype, "status": "FAIL",
                "elapsed": elapsed, "error": "not_delegated",
                "workflow_races": 0, "creative_races": 0}

    print(f"  Delegated to {agent_id} ({round(time.time()-start,1)}s)")

    # 2) Subscribe to worker SSE for race events
    workflow_races = []  # List of race summaries
    creative_races = []
    current_race = {}  # {race_id: {type, candidates: {cid: {status, valid, reason, elapsed}}, winner}}
    tool_calls = []
    worker_response = ""
    worker_status = ""

    try:
        async with session.get(f"{BASE}/api/tools/agent/workers/{conv_id}/stream",
                               timeout=aiohttp.ClientTimeout(total=300)) as ws:
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
                            race_type = "creative" if rid.startswith("crc-") else "workflow"
                            current_race[rid] = {
                                "type": race_type,
                                "tool": e.get("tool_name", "?"),
                                "num": e.get("num_candidates", 0),
                                "candidates": {},
                                "winner": None,
                                "outcome": None,
                            }
                            print(f"  RACE [{race_type}] {rid}: {e.get('num_candidates',0)} candidates for {e.get('tool_name','?')}")

                        elif en == "race_candidate_evaluated" and rid in current_race:
                            cid = e.get("candidate_id", 0)
                            current_race[rid]["candidates"][cid] = {
                                "valid": e.get("is_valid", False),
                                "reason": e.get("reason", ""),
                                "elapsed": e.get("elapsed", 0),
                            }
                            v = "VALID" if e.get("is_valid") else "FAIL"
                            print(f"    C{cid}: {v} — {e.get('reason','')[:60]} ({e.get('elapsed',0)}s)")

                        elif en == "race_winner_selected" and rid in current_race:
                            current_race[rid]["winner"] = e.get("winner")
                            print(f"    WINNER: C{e.get('winner')}")

                        elif en == "race_completed" and rid in current_race:
                            current_race[rid]["outcome"] = "completed"
                            r = current_race[rid]
                            if r["type"] == "workflow":
                                workflow_races.append(r)
                            else:
                                creative_races.append(r)

                        elif en == "race_failed" and rid in current_race:
                            current_race[rid]["outcome"] = "failed"
                            r = current_race[rid]
                            if r["type"] == "workflow":
                                workflow_races.append(r)
                            else:
                                creative_races.append(r)
                            print(f"    ALL FAILED")

                        elif en == "tool_call":
                            tool_calls.append(e.get("tool", "?"))
                            print(f"  TOOL #{e.get('number','?')}: {e.get('tool','?')}")

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

    # Summary
    print(f"  ---")
    print(f"  Status: {worker_status}")
    print(f"  Elapsed: {elapsed}s")
    print(f"  Tool calls: {len(tool_calls)}")
    print(f"  Workflow races: {len(workflow_races)}")
    print(f"  Creative races: {len(creative_races)}")
    print(f"  Response: {len(worker_response)} chars")
    if worker_response:
        print(f"  Preview: {worker_response[:120].replace(chr(10),' ')}...")

    passed = worker_status == "completed" and len(worker_response) > 50
    return {
        "id": tid, "name": tname, "type": ttype,
        "status": "PASS" if passed else "FAIL",
        "elapsed": elapsed,
        "tool_calls": len(tool_calls),
        "workflow_races": len(workflow_races),
        "creative_races": len(creative_races),
        "response_len": len(worker_response),
        "worker_status": worker_status,
    }


async def main():
    print("=" * 70)
    print("RACE EXECUTOR STRESS TESTS")
    print("=" * 70)

    async with aiohttp.ClientSession() as session:
        async with session.get(f"{BASE}/api/models/loaded") as resp:
            models = await resp.json()
        if not models:
            print("ERROR: No models loaded!")
            return

        iid = models[0]["id"]
        print(f"Model: {models[0].get('model','?')} ({iid})")

        async with session.get(f"{BASE}/api/settings") as resp:
            settings = await resp.json()
        race_on = settings.get("agent", {}).get("tool_race_enabled", False)
        race_n = settings.get("agent", {}).get("tool_race_candidates", 3)
        print(f"Race: enabled={race_on}, candidates={race_n}")

        # Select tests
        test_ids = list(range(1, 11))
        if len(sys.argv) > 1:
            test_ids = [int(x) for x in sys.argv[1:]]
        tests = [t for t in TESTS if t["id"] in test_ids]
        print(f"Running {len(tests)} tests: {[t['id'] for t in tests]}\n")

        results = []
        for test in tests:
            r = await run_test(session, test, iid)
            results.append(r)
            await asyncio.sleep(2)

    # Final summary
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    for r in results:
        icon = "PASS" if r["status"] == "PASS" else "FAIL"
        wr = r.get("workflow_races", 0)
        cr = r.get("creative_races", 0)
        race_info = ""
        if wr:
            race_info += f" WR={wr}"
        if cr:
            race_info += f" CR={cr}"
        extra = f" tools={r.get('tool_calls',0)} resp={r.get('response_len',0)}ch"
        print(f"  [{icon}] T{r['id']}: {r['name']} ({r['type']}) {r['elapsed']}s{extra}{race_info}")

    passed = sum(1 for r in results if r["status"] == "PASS")
    total_wr = sum(r.get("workflow_races", 0) for r in results)
    total_cr = sum(r.get("creative_races", 0) for r in results)
    print(f"\n{passed}/{len(results)} passed | Workflow races: {total_wr} | Creative races: {total_cr}")


if __name__ == "__main__":
    asyncio.run(main())
