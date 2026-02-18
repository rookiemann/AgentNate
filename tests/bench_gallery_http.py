"""
HTTP-level gallery stress test against running server.

Seeds the media catalog with 10k records, then hammers gallery endpoints
with concurrent requests to measure latency and throughput.
"""
import asyncio
import os
import random
import sqlite3
import sys
import time
import uuid

_this_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.path.join(os.getcwd(), 'tests')
sys.path.insert(0, os.path.dirname(_this_dir))
sys.path.insert(0, _this_dir)

# Reuse seeding logic
from bench_gallery import CHECKPOINTS, TAGS_POOL, SAMPLERS, SCHEDULERS, seed_database, create_dummy_images


async def main():
    import httpx

    print("=" * 65)
    print("  Gallery HTTP Stress Test")
    print("=" * 65)

    # Find the production media catalog DB
    appdata = os.environ.get("APPDATA", os.path.expanduser("~"))
    db_dir = os.path.join(appdata, "AgentNate", "comfyui")
    db_path = os.path.join(db_dir, "media_catalog.db")

    os.makedirs(db_dir, exist_ok=True)

    # Check current count
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        count = conn.execute("SELECT COUNT(*) as cnt FROM generations").fetchone()["cnt"]
        print(f"\nCurrent DB has {count} generation records")
    except Exception:
        count = 0
        print("\nNo existing media_catalog.db — will seed fresh")
    conn.close()

    # Seed to 10k if needed
    target = 10000
    if count < target:
        needed = target - count
        print(f"Seeding {needed} additional records...")
        start = time.perf_counter()
        n_gens, n_files = seed_database(db_path, needed)
        elapsed = time.perf_counter() - start
        print(f"  Seeded {n_gens} generations + {n_files} files in {elapsed:.1f}s")

    # Also create dummy images in the ComfyUI output dir (if it exists)
    output_dir = os.path.join(db_dir, "output")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    existing_imgs = len([f for f in os.listdir(output_dir) if f.endswith(".png")])
    if existing_imgs < 50:
        print(f"Creating dummy PNG files for image proxy test...")
        create_dummy_images(output_dir, 100)

    # Run HTTP tests
    N_CONCURRENT = 20
    BASE_URL = "http://localhost:8000"

    async with httpx.AsyncClient(
        base_url=BASE_URL,
        timeout=30.0,
        limits=httpx.Limits(max_connections=N_CONCURRENT, max_keepalive_connections=N_CONCURRENT),
    ) as client:

        # ---- Test 1: Random page pagination ----
        print(f"\n1. Gallery pagination ({N_CONCURRENT} concurrent, 200 requests):")
        latencies = []
        sem = asyncio.Semaphore(N_CONCURRENT)

        async def fetch_page(page: int):
            async with sem:
                start = time.perf_counter()
                resp = await client.get(
                    "/api/comfyui/media/generations",
                    params={"limit": 24, "offset": page * 24, "sort": "newest"},
                )
                elapsed = time.perf_counter() - start
                latencies.append(elapsed)
                return resp.status_code

        pages = [random.randint(0, 400) for _ in range(200)]
        start = time.perf_counter()
        results = await asyncio.gather(*[fetch_page(p) for p in pages])
        total_elapsed = time.perf_counter() - start

        ok = sum(1 for r in results if r == 200)
        errors = sum(1 for r in results if r != 200)
        latencies.sort()
        p50 = latencies[len(latencies) // 2] * 1000
        p99 = latencies[int(len(latencies) * 0.99)] * 1000
        avg = sum(latencies) / len(latencies) * 1000
        rps = len(results) / total_elapsed

        status = "PASS" if p99 < 500 else ("SLOW" if p99 < 2000 else "FAIL")
        print(f"  {status} Random pages: {ok}/{len(results)} OK  avg={avg:.1f}ms  p50={p50:.1f}ms  p99={p99:.1f}ms  {rps:.0f} req/s")

        # ---- Test 2: Text search burst ----
        print(f"\n2. Text search burst ({N_CONCURRENT} concurrent, 100 requests):")
        latencies = []
        queries = ["landscape", "portrait", "ocean", "robot", "anime", "dark", "epic", "mountain"]

        async def fetch_search(q: str):
            async with sem:
                start = time.perf_counter()
                resp = await client.get(
                    "/api/comfyui/media/generations",
                    params={"query": q, "limit": 24, "offset": 0},
                )
                elapsed = time.perf_counter() - start
                latencies.append(elapsed)
                return resp.status_code

        tasks = [fetch_search(random.choice(queries)) for _ in range(100)]
        start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        total_elapsed = time.perf_counter() - start

        latencies.sort()
        p50 = latencies[len(latencies) // 2] * 1000
        p99 = latencies[int(len(latencies) * 0.99)] * 1000
        avg = sum(latencies) / len(latencies) * 1000
        rps = len(results) / total_elapsed

        status = "PASS" if p99 < 500 else ("SLOW" if p99 < 2000 else "FAIL")
        print(f"  {status} Search: avg={avg:.1f}ms  p50={p50:.1f}ms  p99={p99:.1f}ms  {rps:.0f} req/s")

        # ---- Test 3: Filtered queries ----
        print(f"\n3. Filtered queries (checkpoint + tags + favorites):")
        latencies = []

        async def fetch_filtered():
            async with sem:
                filter_type = random.choice(["checkpoint", "tags", "favorites", "combined"])
                params = {"limit": 24, "offset": 0}
                if filter_type == "checkpoint":
                    params["checkpoint"] = random.choice(["flux", "dream", "realistic", "xl"])
                elif filter_type == "tags":
                    params["tags"] = random.choice(TAGS_POOL)
                elif filter_type == "favorites":
                    params["favorite"] = "true"
                else:
                    params["checkpoint"] = random.choice(["flux", "dream"])
                    params["tags"] = random.choice(TAGS_POOL)

                start = time.perf_counter()
                resp = await client.get("/api/comfyui/media/generations", params=params)
                elapsed = time.perf_counter() - start
                latencies.append(elapsed)
                return resp.status_code

        tasks = [fetch_filtered() for _ in range(100)]
        start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        total_elapsed = time.perf_counter() - start

        latencies.sort()
        p50 = latencies[len(latencies) // 2] * 1000
        p99 = latencies[int(len(latencies) * 0.99)] * 1000
        avg = sum(latencies) / len(latencies) * 1000
        rps = len(results) / total_elapsed

        status = "PASS" if p99 < 500 else ("SLOW" if p99 < 2000 else "FAIL")
        print(f"  {status} Filtered: avg={avg:.1f}ms  p50={p50:.1f}ms  p99={p99:.1f}ms  {rps:.0f} req/s")

        # ---- Test 4: Stats endpoint burst ----
        print(f"\n4. Stats endpoint burst (200 requests):")
        latencies = []

        async def fetch_stats():
            async with sem:
                start = time.perf_counter()
                resp = await client.get("/api/comfyui/media/stats")
                elapsed = time.perf_counter() - start
                latencies.append(elapsed)
                return resp.status_code

        start = time.perf_counter()
        results = await asyncio.gather(*[fetch_stats() for _ in range(200)])
        total_elapsed = time.perf_counter() - start

        latencies.sort()
        p50 = latencies[len(latencies) // 2] * 1000
        p99 = latencies[int(len(latencies) * 0.99)] * 1000
        avg = sum(latencies) / len(latencies) * 1000
        rps = len(results) / total_elapsed

        status = "PASS" if p99 < 500 else ("SLOW" if p99 < 2000 else "FAIL")
        print(f"  {status} Stats: avg={avg:.1f}ms  p50={p50:.1f}ms  p99={p99:.1f}ms  {rps:.0f} req/s")

        # ---- Test 5: Image proxy throughput ----
        files = [f for f in os.listdir(output_dir) if f.endswith(".png")][:50]
        if files:
            print(f"\n5. Image proxy ({N_CONCURRENT} concurrent, 200 requests, {len(files)} files):")
            latencies = []
            bytes_total = 0

            async def fetch_image(filename: str):
                nonlocal bytes_total
                async with sem:
                    start = time.perf_counter()
                    resp = await client.get(f"/api/comfyui/images/{filename}")
                    elapsed = time.perf_counter() - start
                    latencies.append(elapsed)
                    if resp.status_code == 200:
                        bytes_total += len(resp.content)
                    return resp.status_code

            tasks = [fetch_image(random.choice(files)) for _ in range(200)]
            start = time.perf_counter()
            results = await asyncio.gather(*tasks)
            total_elapsed = time.perf_counter() - start

            ok = sum(1 for r in results if r == 200)
            latencies.sort()
            p50 = latencies[len(latencies) // 2] * 1000
            p99 = latencies[int(len(latencies) * 0.99)] * 1000
            avg = sum(latencies) / len(latencies) * 1000
            rps = len(results) / total_elapsed

            status = "PASS" if p99 < 200 else ("SLOW" if p99 < 1000 else "FAIL")
            print(f"  {status} Images: {ok}/{len(results)} OK  avg={avg:.1f}ms  p50={p50:.1f}ms  p99={p99:.1f}ms  {rps:.0f} req/s")
            print(f"       {bytes_total/1024/1024:.1f}MB transferred in {total_elapsed:.1f}s ({bytes_total/1024/1024/total_elapsed:.1f} MB/s)")

        # ---- Test 6: Mixed workload (gallery browsing simulation) ----
        print(f"\n6. Mixed workload (simulated gallery browsing, 200 requests):")
        latencies = []

        async def gallery_browse():
            async with sem:
                action = random.choice(["page", "search", "filter", "stats", "stats", "stats"])
                start = time.perf_counter()

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
                        params={"checkpoint": random.choice(["flux", "dream"]), "limit": 24},
                    )
                else:  # stats
                    resp = await client.get("/api/comfyui/media/stats")

                elapsed = time.perf_counter() - start
                latencies.append(elapsed)
                return resp.status_code

        tasks = [gallery_browse() for _ in range(200)]
        start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        total_elapsed = time.perf_counter() - start

        ok = sum(1 for r in results if r == 200)
        latencies.sort()
        p50 = latencies[len(latencies) // 2] * 1000
        p99 = latencies[int(len(latencies) * 0.99)] * 1000
        avg = sum(latencies) / len(latencies) * 1000
        rps = len(results) / total_elapsed

        status = "PASS" if p99 < 500 else ("SLOW" if p99 < 2000 else "FAIL")
        print(f"  {status} Mixed: {ok}/{len(results)} OK  avg={avg:.1f}ms  p50={p50:.1f}ms  p99={p99:.1f}ms  {rps:.0f} req/s")

    print("\n" + "=" * 65)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
