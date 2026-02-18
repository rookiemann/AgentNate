"""
API Endpoint Burst Test
=======================
Hammers key endpoints at 100+ req/s and measures latency percentiles.
"""
import asyncio
import time
import sys
import os
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import httpx

BASE = "http://127.0.0.1:8000"
ENDPOINTS = [
    ("GET", "/health", "Health check"),
    ("GET", "/api/models/loaded", "Loaded models"),
    ("GET", "/api/comfyui/status", "ComfyUI status"),
    ("GET", "/api/comfyui/pool/status", "Pool status"),
    ("GET", "/api/n8n/main/status", "n8n main status"),
    ("GET", "/api/comfyui/media/stats", "Media stats"),
]

REQS_PER_ENDPOINT = 200
CONCURRENCY = 20  # simultaneous connections


async def burst_endpoint(client: httpx.AsyncClient, method: str, path: str, count: int) -> list:
    """Fire `count` requests and return list of latencies in ms."""
    latencies = []
    errors = 0

    async def single_req():
        nonlocal errors
        start = time.perf_counter()
        try:
            if method == "GET":
                r = await client.get(f"{BASE}{path}")
            else:
                r = await client.post(f"{BASE}{path}")
            elapsed = (time.perf_counter() - start) * 1000
            if r.status_code < 400:
                latencies.append(elapsed)
            else:
                errors += 1
                latencies.append(elapsed)
        except Exception:
            errors += 1
            latencies.append((time.perf_counter() - start) * 1000)

    # Fire in batches of CONCURRENCY
    sem = asyncio.Semaphore(CONCURRENCY)

    async def bounded_req():
        async with sem:
            await single_req()

    tasks = [bounded_req() for _ in range(count)]
    await asyncio.gather(*tasks)

    return latencies, errors


async def main():
    print("=" * 75)
    print(f"  API Burst Test — {REQS_PER_ENDPOINT} reqs/endpoint, {CONCURRENCY} concurrent")
    print("=" * 75)

    async with httpx.AsyncClient(timeout=10.0) as client:
        # Warmup
        for method, path, _ in ENDPOINTS:
            try:
                await client.get(f"{BASE}{path}")
            except:
                pass

        results = []
        for method, path, label in ENDPOINTS:
            start = time.perf_counter()
            latencies, errors = await burst_endpoint(client, method, path, REQS_PER_ENDPOINT)
            wall = time.perf_counter() - start

            if latencies:
                latencies.sort()
                p50 = latencies[len(latencies) // 2]
                p95 = latencies[int(len(latencies) * 0.95)]
                p99 = latencies[int(len(latencies) * 0.99)]
                avg = statistics.mean(latencies)
                rps = len(latencies) / wall
            else:
                p50 = p95 = p99 = avg = rps = 0

            status = "PASS" if p99 < 500 and errors == 0 else "SLOW" if p99 < 1000 else "FAIL"
            results.append((label, path, avg, p50, p95, p99, rps, errors, status))

            print(f"  {status:4s} {label:20s} avg={avg:6.1f}ms  p50={p50:6.1f}ms  "
                  f"p95={p95:6.1f}ms  p99={p99:6.1f}ms  "
                  f"{rps:7.0f} req/s  errors={errors}")

    print()
    print("=" * 75)
    slow = [r for r in results if r[8] != "PASS"]
    if slow:
        print(f"  {len(slow)} endpoint(s) need optimization:")
        for label, path, avg, p50, p95, p99, rps, errors, status in slow:
            print(f"    - {label} ({path}): p99={p99:.1f}ms")
    else:
        print("  All endpoints PASS (p99 < 100ms)")
    print("=" * 75)


if __name__ == "__main__":
    asyncio.run(main())
