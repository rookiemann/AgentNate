"""
Full system stress test v2 — with per-endpoint breakdown.
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
DURATION = 15


class Stats:
    def __init__(self, name):
        self.name = name
        self.latencies = []
        self.errors = 0
        self.ok = 0

    def record(self, latency, success):
        self.latencies.append(latency)
        if success:
            self.ok += 1
        else:
            self.errors += 1

    def report(self):
        total = self.ok + self.errors
        if not self.latencies:
            return f"  {self.name}: No data"
        self.latencies.sort()
        avg = sum(self.latencies) / len(self.latencies) * 1000
        p50 = self.latencies[len(self.latencies) // 2] * 1000
        p99 = self.latencies[int(len(self.latencies) * 0.99)] * 1000
        rps = total / DURATION
        errs = f"  {self.errors}err" if self.errors else ""
        tag = "PASS" if p99 < 1000 and self.errors == 0 else ("SLOW" if self.errors == 0 else "FAIL")
        return (f"  {tag} {self.name:35s} {total:4d} reqs  "
                f"avg={avg:7.1f}ms  p50={p50:7.1f}ms  p99={p99:7.1f}ms  {rps:5.1f}/s{errs}")


async def main():
    import httpx
    import websockets

    print("=" * 80)
    print("  Full System Stress Test v2 — Per-Endpoint Breakdown")
    print(f"  Duration: {DURATION}s")
    print("=" * 80)

    # Per-endpoint stats
    stats = {
        "ws_ping": Stats("WS Ping (5 conns)"),
        "models_loaded": Stats("GET /models/loaded"),
        "health_all": Stats("GET /models/health/all"),
        "n8n_status": Stats("GET /n8n/main/status"),
        "comfyui_status": Stats("GET /comfyui/status"),
        "pool_status": Stats("GET /comfyui/pool/status"),
        "media_stats": Stats("GET /comfyui/media/stats"),
        "gallery_page": Stats("GET /media/generations (page)"),
        "gallery_search": Stats("GET /media/generations (search)"),
        "conversations": Stats("GET /conversations"),
        "personas": Stats("GET /personas"),
        "chat_queue": Stats("GET /chat/queue"),
    }

    stop = asyncio.Event()

    # WebSocket pings
    async def ws_work():
        conns = []
        for _ in range(5):
            try:
                ws = await websockets.connect(WS_URL, max_size=2**20)
                conns.append(ws)
            except Exception:
                stats["ws_ping"].record(0, False)
        while not stop.is_set():
            for ws in conns:
                try:
                    t = time.perf_counter()
                    await ws.send(json.dumps({"action": "ping"}))
                    r = await asyncio.wait_for(ws.recv(), timeout=5)
                    stats["ws_ping"].record(time.perf_counter() - t, json.loads(r).get("type") == "pong")
                except Exception:
                    stats["ws_ping"].record(0, False)
            await asyncio.sleep(0.05)
        for ws in conns:
            try: await ws.close()
            except: pass

    # HTTP polling
    async def http_work(client):
        endpoints = {
            "/api/models/loaded": "models_loaded",
            "/api/models/health/all": "health_all",
            "/api/n8n/main/status": "n8n_status",
            "/api/comfyui/status": "comfyui_status",
            "/api/comfyui/pool/status": "pool_status",
            "/api/comfyui/media/stats": "media_stats",
            "/api/tools/conversations": "conversations",
            "/api/tools/personas": "personas",
            "/api/chat/queue": "chat_queue",
        }
        sem = asyncio.Semaphore(15)

        while not stop.is_set():
            async def hit(url, key):
                async with sem:
                    t = time.perf_counter()
                    try:
                        r = await client.get(url)
                        stats[key].record(time.perf_counter() - t, r.status_code == 200)
                    except Exception:
                        stats[key].record(0, False)

            # Hit all endpoints
            tasks = [hit(url, key) for url, key in endpoints.items()]
            await asyncio.gather(*tasks)
            await asyncio.sleep(0.1)

    # Gallery browsing
    async def gallery_work(client):
        queries = ["landscape", "portrait", "ocean", "robot", "anime"]
        sem = asyncio.Semaphore(10)

        while not stop.is_set():
            async def browse():
                async with sem:
                    action = random.choice(["page", "page", "search"])
                    t = time.perf_counter()
                    try:
                        if action == "page":
                            r = await client.get("/api/comfyui/media/generations",
                                params={"limit": 24, "offset": random.randint(0, 200) * 24})
                            stats["gallery_page"].record(time.perf_counter() - t, r.status_code == 200)
                        else:
                            r = await client.get("/api/comfyui/media/generations",
                                params={"query": random.choice(queries), "limit": 24})
                            stats["gallery_search"].record(time.perf_counter() - t, r.status_code == 200)
                    except Exception:
                        stats["gallery_page"].record(0, False)

            tasks = [browse() for _ in range(5)]
            await asyncio.gather(*tasks)
            await asyncio.sleep(0.1)

    print("\nStarting all workloads...\n")

    async with httpx.AsyncClient(
        base_url=BASE_URL, timeout=30,
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=30),
    ) as client:
        tasks = [
            asyncio.create_task(ws_work()),
            asyncio.create_task(http_work(client)),
            asyncio.create_task(gallery_work(client)),
        ]

        start = time.perf_counter()
        await asyncio.sleep(DURATION)
        stop.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        total = time.perf_counter() - start

    print(f"  Runtime: {total:.1f}s\n")
    print("Per-Endpoint Results:")
    print("-" * 80)

    total_reqs = 0
    total_errs = 0
    for s in stats.values():
        print(s.report())
        total_reqs += s.ok + s.errors
        total_errs += s.errors

    print("-" * 80)
    print(f"\n  Total: {total_reqs} requests, {total_errs} errors ({total_errs/total_reqs*100:.1f}% error rate)" if total_reqs else "")
    print(f"  Aggregate throughput: {total_reqs/total:.0f} req/s")

    # Health check
    import httpx as h
    async with h.AsyncClient(timeout=5) as c:
        try:
            r = await c.get(f"{BASE_URL}/api/models/loaded")
            print(f"\n  Server: {'HEALTHY' if r.status_code == 200 else 'DEGRADED'}")
        except:
            print(f"\n  Server: DOWN")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
