"""
Concurrent WebSocket connection stress test.

Tests:
1. Rapid connection establishment (20 concurrent)
2. Ping/pong latency on all connections
3. Rapid connect/disconnect cycling
4. Error handling (chat without model)
5. Connection cleanup verification
"""
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WS_URL = "ws://localhost:8000/api/chat/stream"


async def main():
    import websockets

    print("=" * 65)
    print("  WebSocket Concurrent Connection Stress Test")
    print("=" * 65)

    # ---- Test 1: Rapid concurrent connection ----
    print("\n1. Open 20 WebSocket connections simultaneously:")
    connections = []
    connect_times = []

    async def connect_one():
        start = time.perf_counter()
        ws = await websockets.connect(WS_URL, max_size=2**20)
        elapsed = time.perf_counter() - start
        connect_times.append(elapsed)
        return ws

    start = time.perf_counter()
    tasks = [connect_one() for _ in range(20)]
    connections = await asyncio.gather(*tasks, return_exceptions=True)
    total = time.perf_counter() - start

    ok_conns = [c for c in connections if not isinstance(c, Exception)]
    errors = [c for c in connections if isinstance(c, Exception)]
    connect_times.sort()

    print(f"  Connected: {len(ok_conns)}/20  Errors: {len(errors)}")
    if connect_times:
        print(f"  Connect time: avg={sum(connect_times)/len(connect_times)*1000:.1f}ms"
              f"  p99={connect_times[-1]*1000:.1f}ms  total={total*1000:.0f}ms")

    if errors:
        for e in errors[:3]:
            print(f"  Error: {e}")

    # ---- Test 2: Ping/pong latency on all connections ----
    print(f"\n2. Ping/pong latency on {len(ok_conns)} connections:")
    latencies = []

    async def ping_one(ws, idx):
        for _ in range(10):  # 10 pings each
            msg = json.dumps({"action": "ping"})
            start = time.perf_counter()
            await ws.send(msg)
            resp = await asyncio.wait_for(ws.recv(), timeout=5.0)
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)
            data = json.loads(resp)
            if data.get("type") != "pong":
                print(f"  Unexpected response on conn {idx}: {data}")
                return False
        return True

    start = time.perf_counter()
    results = await asyncio.gather(
        *[ping_one(ws, i) for i, ws in enumerate(ok_conns)],
        return_exceptions=True,
    )
    total = time.perf_counter() - start

    ok_pings = sum(1 for r in results if r is True)
    errors = [r for r in results if isinstance(r, Exception)]
    latencies.sort()

    total_pings = len(ok_conns) * 10
    print(f"  {len(latencies)}/{total_pings} pings OK, {len(errors)} connection errors")
    if latencies:
        avg = sum(latencies) / len(latencies) * 1000
        p50 = latencies[len(latencies) // 2] * 1000
        p99 = latencies[int(len(latencies) * 0.99)] * 1000
        print(f"  Latency: avg={avg:.2f}ms  p50={p50:.2f}ms  p99={p99:.2f}ms")
        print(f"  Total: {total:.1f}s ({total_pings/total:.0f} pings/s)")

    # ---- Test 3: Chat error handling (no model loaded) ----
    print(f"\n3. Chat error handling on {len(ok_conns)} connections:")

    async def send_chat(ws, idx):
        msg = json.dumps({
            "action": "chat",
            "instance_id": "nonexistent-model",
            "request_id": f"stress-{idx}",
            "messages": [{"role": "user", "content": "Hello"}],
            "params": {},
        })
        start = time.perf_counter()
        await ws.send(msg)
        # Should get an error response
        try:
            resp = await asyncio.wait_for(ws.recv(), timeout=10.0)
            elapsed = time.perf_counter() - start
            data = json.loads(resp)
            return {"ok": True, "type": data.get("type"), "elapsed": elapsed}
        except asyncio.TimeoutError:
            return {"ok": False, "type": "timeout", "elapsed": 10.0}
        except Exception as e:
            return {"ok": False, "type": str(type(e).__name__), "elapsed": 0}

    # Use first 10 connections
    test_conns = ok_conns[:10]
    results = await asyncio.gather(
        *[send_chat(ws, i) for i, ws in enumerate(test_conns)],
        return_exceptions=True,
    )

    error_types = {}
    for r in results:
        if isinstance(r, Exception):
            t = type(r).__name__
        else:
            t = r.get("type", "unknown")
        error_types[t] = error_types.get(t, 0) + 1

    print(f"  Response types: {error_types}")
    # Show timing for successful error responses
    timings = [r["elapsed"] for r in results if isinstance(r, dict) and r["ok"]]
    if timings:
        print(f"  Error response time: avg={sum(timings)/len(timings)*1000:.1f}ms")

    # ---- Test 4: Rapid connect/disconnect cycling ----
    print(f"\n4. Rapid connect/disconnect cycling (50 cycles):")

    async def connect_disconnect():
        ws = await websockets.connect(WS_URL, max_size=2**20)
        # Quick ping
        await ws.send(json.dumps({"action": "ping"}))
        await asyncio.wait_for(ws.recv(), timeout=5.0)
        await ws.close()

    cycle_times = []
    for batch in range(5):
        batch_start = time.perf_counter()
        tasks = [connect_disconnect() for _ in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        batch_elapsed = time.perf_counter() - batch_start
        cycle_times.append(batch_elapsed)
        ok = sum(1 for r in results if not isinstance(r, Exception))
        errs = sum(1 for r in results if isinstance(r, Exception))
        if errs:
            print(f"  Batch {batch+1}: {ok}/10 OK, {errs} errors ({batch_elapsed*1000:.0f}ms)")

    avg_batch = sum(cycle_times) / len(cycle_times) * 1000
    total_cycles = 50
    total_time = sum(cycle_times)
    print(f"  50 connect+ping+disconnect cycles: {total_time:.1f}s  ({total_cycles/total_time:.0f} cycles/s)")
    print(f"  Per-batch (10 concurrent): avg={avg_batch:.0f}ms")

    # ---- Test 5: Verify cleanup ----
    print(f"\n5. Connection cleanup verification:")

    # Close all remaining connections
    for ws in ok_conns:
        try:
            await ws.close()
        except Exception:
            pass

    # Give server a moment to clean up
    await asyncio.sleep(0.5)

    # Check active connections via API
    import httpx
    async with httpx.AsyncClient(timeout=5) as client:
        resp = await client.get("http://localhost:8000/api/chat/queue")
        if resp.status_code == 200:
            queue = resp.json()
            print(f"  Queue status: {json.dumps(queue)}")

    # Try opening a fresh connection to verify server still healthy
    try:
        ws = await websockets.connect(WS_URL, max_size=2**20)
        await ws.send(json.dumps({"action": "ping"}))
        resp = await asyncio.wait_for(ws.recv(), timeout=5.0)
        data = json.loads(resp)
        await ws.close()
        print(f"  Post-cleanup: Server healthy (ping → {data.get('type')})")
    except Exception as e:
        print(f"  Post-cleanup: Server error: {e}")

    print("\n" + "=" * 65)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
