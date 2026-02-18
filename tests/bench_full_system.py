"""
Full system stress test — everything at once.

Runs simultaneously:
1. 5 WebSocket connections doing ping/pong
2. 20 concurrent API polling (models, health, n8n status)
3. Gallery browsing (page loads, searches, filters)
4. ComfyUI status + pool polling
5. Media stats endpoint
6. Agent endpoint (abort immediately)

Measures: errors, latencies, server stability (no crashes/deadlocks).
"""
import asyncio
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/api/chat/stream"
DURATION = 15  # seconds
CONCURRENCY = 10  # per workload type


class WorkloadStats:
    """Thread-safe stats accumulator."""
    def __init__(self, name: str):
        self.name = name
        self.latencies = []
        self.errors = 0
        self.ok = 0

    def record(self, latency: float, success: bool):
        self.latencies.append(latency)
        if success:
            self.ok += 1
        else:
            self.errors += 1

    def report(self):
        total = self.ok + self.errors
        if not self.latencies:
            return f"  {self.name}: No requests"
        self.latencies.sort()
        avg = sum(self.latencies) / len(self.latencies) * 1000
        p50 = self.latencies[len(self.latencies) // 2] * 1000
        p99 = self.latencies[int(len(self.latencies) * 0.99)] * 1000
        rps = total / DURATION if DURATION > 0 else 0
        status = "PASS" if self.errors == 0 and p99 < 2000 else (
            "SLOW" if self.errors == 0 else "FAIL"
        )
        return (f"  {status} {self.name:30s}  {self.ok}/{total} OK  "
                f"avg={avg:7.1f}ms  p50={p50:7.1f}ms  p99={p99:7.1f}ms  {rps:5.0f} req/s")


async def workload_websocket_pings(stop_event: asyncio.Event, stats: WorkloadStats):
    """5 WebSocket connections doing rapid ping/pong."""
    import websockets

    conns = []
    for _ in range(5):
        try:
            ws = await websockets.connect(WS_URL, max_size=2**20)
            conns.append(ws)
        except Exception:
            stats.record(0, False)

    while not stop_event.is_set():
        for ws in conns:
            try:
                start = time.perf_counter()
                await ws.send(json.dumps({"action": "ping"}))
                resp = await asyncio.wait_for(ws.recv(), timeout=5.0)
                elapsed = time.perf_counter() - start
                data = json.loads(resp)
                stats.record(elapsed, data.get("type") == "pong")
            except Exception:
                stats.record(0, False)
        await asyncio.sleep(0.05)  # ~20 ping batches per second

    for ws in conns:
        try:
            await ws.close()
        except Exception:
            pass


async def workload_api_polling(client, stop_event: asyncio.Event, stats: WorkloadStats):
    """Rapid API polling across multiple endpoints."""
    endpoints = [
        "/api/models/loaded",
        "/api/models/health/all",
        "/api/n8n/main/status",
        "/api/comfyui/status",
    ]
    sem = asyncio.Semaphore(CONCURRENCY)

    async def poll_one():
        async with sem:
            url = random.choice(endpoints)
            start = time.perf_counter()
            try:
                resp = await client.get(url)
                elapsed = time.perf_counter() - start
                stats.record(elapsed, resp.status_code == 200)
            except Exception:
                stats.record(0, False)

    while not stop_event.is_set():
        # Fire 10 concurrent polls
        tasks = [poll_one() for _ in range(10)]
        await asyncio.gather(*tasks)
        await asyncio.sleep(0.05)


async def workload_gallery_browsing(client, stop_event: asyncio.Event, stats: WorkloadStats):
    """Gallery page loads, searches, and filters."""
    queries = ["landscape", "portrait", "ocean", "robot", "anime", "dark"]
    checkpoints = ["flux", "dream", "realistic", "xl"]
    sem = asyncio.Semaphore(CONCURRENCY)

    async def browse_one():
        async with sem:
            action = random.choice(["page", "search", "filter", "stats"])
            start = time.perf_counter()
            try:
                if action == "page":
                    resp = await client.get(
                        "/api/comfyui/media/generations",
                        params={"limit": 24, "offset": random.randint(0, 400) * 24},
                    )
                elif action == "search":
                    resp = await client.get(
                        "/api/comfyui/media/generations",
                        params={"query": random.choice(queries), "limit": 24},
                    )
                elif action == "filter":
                    resp = await client.get(
                        "/api/comfyui/media/generations",
                        params={"checkpoint": random.choice(checkpoints), "limit": 24},
                    )
                else:
                    resp = await client.get("/api/comfyui/media/stats")

                elapsed = time.perf_counter() - start
                stats.record(elapsed, resp.status_code == 200)
            except Exception:
                stats.record(0, False)

    while not stop_event.is_set():
        tasks = [browse_one() for _ in range(5)]
        await asyncio.gather(*tasks)
        await asyncio.sleep(0.1)


async def workload_comfyui_status(client, stop_event: asyncio.Event, stats: WorkloadStats):
    """ComfyUI status and pool polling."""
    endpoints = [
        "/api/comfyui/status",
        "/api/comfyui/pool/status",
    ]
    sem = asyncio.Semaphore(CONCURRENCY)

    async def poll_one():
        async with sem:
            url = random.choice(endpoints)
            start = time.perf_counter()
            try:
                resp = await client.get(url)
                elapsed = time.perf_counter() - start
                stats.record(elapsed, resp.status_code == 200)
            except Exception:
                stats.record(0, False)

    while not stop_event.is_set():
        tasks = [poll_one() for _ in range(5)]
        await asyncio.gather(*tasks)
        await asyncio.sleep(0.1)


async def workload_misc(client, stop_event: asyncio.Event, stats: WorkloadStats):
    """Misc endpoints: conversations, settings, personas, debug."""
    endpoints = [
        "/api/tools/conversations",
        "/api/tools/personas",
        "/api/chat/queue",
    ]
    sem = asyncio.Semaphore(CONCURRENCY)

    async def poll_one():
        async with sem:
            url = random.choice(endpoints)
            start = time.perf_counter()
            try:
                resp = await client.get(url)
                elapsed = time.perf_counter() - start
                stats.record(elapsed, resp.status_code == 200)
            except Exception:
                stats.record(0, False)

    while not stop_event.is_set():
        tasks = [poll_one() for _ in range(3)]
        await asyncio.gather(*tasks)
        await asyncio.sleep(0.2)


async def main():
    import httpx

    print("=" * 75)
    print("  Full System Stress Test — Everything At Once")
    print(f"  Duration: {DURATION}s  Concurrency: {CONCURRENCY} per workload")
    print("=" * 75)

    # Verify server is up
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=5) as client:
        try:
            resp = await client.get("/api/models/loaded")
            if resp.status_code != 200:
                print("  ERROR: Server not responding!")
                return
        except Exception as e:
            print(f"  ERROR: Cannot reach server: {e}")
            return

    print("\nStarting all workloads simultaneously...\n")

    # Create stats trackers
    ws_stats = WorkloadStats("WebSocket Pings (5 conns)")
    api_stats = WorkloadStats("API Polling (4 endpoints)")
    gallery_stats = WorkloadStats("Gallery Browse (page/search/filter)")
    comfyui_stats = WorkloadStats("ComfyUI Status (status+pool)")
    misc_stats = WorkloadStats("Misc (conversations/personas)")

    stop_event = asyncio.Event()

    async with httpx.AsyncClient(
        base_url=BASE_URL,
        timeout=30.0,
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=30),
    ) as client:

        # Start all workloads
        tasks = [
            asyncio.create_task(workload_websocket_pings(stop_event, ws_stats)),
            asyncio.create_task(workload_api_polling(client, stop_event, api_stats)),
            asyncio.create_task(workload_gallery_browsing(client, stop_event, gallery_stats)),
            asyncio.create_task(workload_comfyui_status(client, stop_event, comfyui_stats)),
            asyncio.create_task(workload_misc(client, stop_event, misc_stats)),
        ]

        # Let it run
        start = time.perf_counter()
        print(f"  Running for {DURATION}s...")
        await asyncio.sleep(DURATION)
        stop_event.set()

        # Wait for all workloads to finish
        await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.perf_counter() - start

    print(f"\n  Total runtime: {total_time:.1f}s")
    print("\nResults:")
    print("-" * 75)

    all_stats = [ws_stats, api_stats, gallery_stats, comfyui_stats, misc_stats]
    for s in all_stats:
        print(s.report())

    print("-" * 75)

    total_ok = sum(s.ok for s in all_stats)
    total_errors = sum(s.errors for s in all_stats)
    total_requests = total_ok + total_errors
    overall_rps = total_requests / total_time

    print(f"\n  Total requests: {total_requests}  OK: {total_ok}  Errors: {total_errors}")
    print(f"  Overall throughput: {overall_rps:.0f} req/s")
    print(f"  Error rate: {total_errors/total_requests*100:.1f}%" if total_requests > 0 else "  No requests")

    # Verify server is still alive
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=5) as client:
        try:
            resp = await client.get("/api/models/loaded")
            if resp.status_code == 200:
                print(f"\n  Server: HEALTHY after stress test")
            else:
                print(f"\n  Server: DEGRADED (status={resp.status_code})")
        except Exception as e:
            print(f"\n  Server: DOWN ({e})")

    print("\n" + "=" * 75)


if __name__ == "__main__":
    asyncio.run(main())
