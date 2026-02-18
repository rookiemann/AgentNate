"""
Test script for queue-based execution controls.

Usage:
    python test_queue.py deploy     # Deploy the test workflow + spawn worker
    python test_queue.py enqueue N  # Enqueue N runs (default 5)
    python test_queue.py parallel N # Enqueue N runs in parallel mode
    python test_queue.py pause PORT # Pause worker
    python test_queue.py clear PORT # Clear queue + reset counter
    python test_queue.py log        # Show execution log
    python test_queue.py clearlog   # Clear execution log
    python test_queue.py status     # Show worker status

Assumes the AgentNate server is running on http://localhost:8000
"""

import sys
import json
import time
import requests

API = "http://localhost:8000/api/n8n"

# Test workflow: Manual Trigger → Code (generate data) → HTTP Request (POST to test-log)
TEST_WORKFLOW = {
    "name": "Queue Test - Execution Logger",
    "active": True,
    "nodes": [
        {
            "id": "trigger-1",
            "name": "Manual Trigger",
            "type": "n8n-nodes-base.manualTrigger",
            "typeVersion": 1,
            "position": [250, 300],
            "parameters": {}
        },
        {
            "id": "code-1",
            "name": "Prepare Log Entry",
            "type": "n8n-nodes-base.code",
            "typeVersion": 2,
            "position": [500, 300],
            "parameters": {
                "mode": "runOnceForAllItems",
                "jsCode": "const now = new Date();\nreturn [{ json: { timestamp: now.toISOString(), timestamp_ms: now.getTime(), run_id: Math.floor(Math.random() * 100000), message: 'Workflow execution completed' } }];"
            }
        },
        {
            "id": "http-1",
            "name": "Log to Server",
            "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4.2,
            "position": [750, 300],
            "parameters": {
                "method": "POST",
                "url": "http://127.0.0.1:8000/api/n8n/test-log",
                "sendHeaders": True,
                "headerParameters": {
                    "parameters": [
                        {"name": "Content-Type", "value": "application/json"}
                    ]
                },
                "sendBody": True,
                "specifyBody": "json",
                "jsonBody": "={{ JSON.stringify($json) }}"
            }
        }
    ],
    "connections": {
        "Manual Trigger": {
            "main": [[{"node": "Prepare Log Entry", "type": "main", "index": 0}]]
        },
        "Prepare Log Entry": {
            "main": [[{"node": "Log to Server", "type": "main", "index": 0}]]
        }
    },
    "settings": {
        "executionOrder": "v1"
    }
}

def deploy():
    """Deploy the test workflow and spawn a worker."""
    print("Deploying test workflow...")

    resp = requests.post(f"{API}/deploy-and-run", json={
        "workflow_json": TEST_WORKFLOW,
        "mode": "once",
    })
    data = resp.json()

    if not data.get("success"):
        print(f"FAILED: {data.get('error', 'Unknown error')}")
        return None

    worker = data["worker"]
    port = worker["port"]
    wf_id = data["workflow_id"]

    print(f"Deployed! Worker on port {port}, workflow ID: {wf_id}")
    print(f"  trigger_count: {worker.get('trigger_count', '?')}")
    print(f"  mode: {worker.get('mode', '?')}")
    print()
    print(f"Next steps:")
    print(f"  python test_queue.py enqueue 5       # Queue 5 sequential runs")
    print(f"  python test_queue.py parallel 10     # Queue 10 parallel runs")
    print(f"  python test_queue.py log             # View execution log")
    print(f"  python test_queue.py status          # Check worker status")

    return port


def get_workers():
    """Get all workers."""
    resp = requests.get(f"{API}/workers")
    return resp.json().get("workers", [])


def find_test_worker():
    """Find the test workflow worker."""
    workers = get_workers()
    for w in workers:
        if "Queue Test" in (w.get("workflow_name") or ""):
            return w
    # Fall back to first worker
    if workers:
        return workers[0]
    return None


def enqueue(count=5, parallel=False):
    """Enqueue N runs on the test worker."""
    worker = find_test_worker()
    if not worker:
        print("No test worker found. Run: python test_queue.py deploy")
        return

    port = worker["port"]
    mode = "parallel" if parallel else "sequential"
    count_display = count if count else "infinite"
    print(f"Enqueueing {count_display} {mode} runs on :{port}...")

    resp = requests.post(f"{API}/workers/{port}/enqueue", json={
        "count": count,
        "parallel": parallel,
    })
    data = resp.json()

    if data.get("success"):
        total = data.get("queued_total")
        print(f"OK. Queue total: {total if total is not None else 'infinite'}")
    else:
        print(f"FAILED: {data}")


def pause(port=None):
    """Pause a worker."""
    if port is None:
        worker = find_test_worker()
        if not worker:
            print("No test worker found.")
            return
        port = worker["port"]

    print(f"Pausing worker :{port}...")
    resp = requests.post(f"{API}/workers/{port}/pause")
    print(resp.json())


def clear(port=None):
    """Clear queue and reset counter."""
    if port is None:
        worker = find_test_worker()
        if not worker:
            print("No test worker found.")
            return
        port = worker["port"]

    print(f"Clearing queue on :{port}...")
    resp = requests.post(f"{API}/workers/{port}/reset-counter")
    print(resp.json())


def show_log():
    """Show the execution log."""
    resp = requests.get(f"{API}/test-log")
    data = resp.json()
    entries = data.get("entries", [])

    if not entries:
        print("No entries yet.")
        return

    print(f"Execution log ({len(entries)} entries):")
    print("-" * 70)

    for i, entry in enumerate(entries):
        port = entry.get("instance_port", "?")
        ts = entry.get("timestamp", "?")
        exec_id = entry.get("execution_id", "?")
        print(f"  [{i+1:3d}] port={port}  exec={exec_id}  time={ts}")

    # Show timing between entries
    if len(entries) >= 2:
        print()
        print("Timing analysis:")
        print("-" * 70)
        times = [e.get("timestamp_ms", 0) for e in entries if e.get("timestamp_ms")]
        if len(times) >= 2:
            diffs = [times[i+1] - times[i] for i in range(len(times)-1)]
            avg_ms = sum(diffs) / len(diffs)
            min_ms = min(diffs)
            max_ms = max(diffs)
            total_s = (times[-1] - times[0]) / 1000
            print(f"  Total time: {total_s:.1f}s")
            print(f"  Avg gap: {avg_ms:.0f}ms")
            print(f"  Min gap: {min_ms:.0f}ms  Max gap: {max_ms:.0f}ms")

            # Detect parallel vs sequential
            if min_ms < 500:
                print(f"  Pattern: PARALLEL (gaps < 500ms detected)")
            else:
                print(f"  Pattern: SEQUENTIAL (all gaps > 500ms)")


def clear_log():
    """Clear the execution log."""
    resp = requests.delete(f"{API}/test-log")
    print(resp.json())


def show_status():
    """Show all workers."""
    workers = get_workers()
    if not workers:
        print("No workers running.")
        return

    print(f"Workers ({len(workers)}):")
    print("-" * 80)
    for w in workers:
        name = w.get("workflow_name", "?")
        port = w.get("port", "?")
        qt = w.get("queued_total")
        qp = w.get("queued_parallel", False)
        paused = w.get("paused", False)
        processing = w.get("processing", False)
        exec_count = w.get("execution_count", 0)
        trigger_count = w.get("trigger_count", 0)

        qt_display = qt if qt is not None else "inf"
        mode_str = "PAR" if qp else "SEQ"
        state = "PROCESSING" if processing and not paused else ("PAUSED" if paused else "IDLE")

        print(f"  :{port} [{state}] {name}")
        print(f"    queue={qt_display} mode={mode_str} completed={exec_count} triggers={trigger_count}")


def watch_log(interval=2):
    """Watch the execution log in real-time."""
    print(f"Watching execution log (refresh every {interval}s, Ctrl+C to stop)...")
    print("=" * 70)
    last_count = 0

    try:
        while True:
            resp = requests.get(f"{API}/test-log")
            data = resp.json()
            entries = data.get("entries", [])
            count = len(entries)

            if count > last_count:
                for entry in entries[last_count:]:
                    port = entry.get("instance_port", "?")
                    ts = entry.get("timestamp", "?")
                    exec_id = entry.get("execution_id", "?")
                    print(f"  [{count:3d}] port={port}  exec={exec_id}  time={ts}")
                last_count = count

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "deploy":
        deploy()
    elif cmd == "enqueue":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        enqueue(n, parallel=False)
    elif cmd == "parallel":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        enqueue(n, parallel=True)
    elif cmd == "pause":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else None
        pause(port)
    elif cmd == "clear":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else None
        clear(port)
    elif cmd == "log":
        show_log()
    elif cmd == "clearlog":
        clear_log()
    elif cmd == "status":
        show_status()
    elif cmd == "watch":
        interval = float(sys.argv[2]) if len(sys.argv) > 2 else 2
        watch_log(interval)
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)
