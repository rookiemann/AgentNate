# -*- coding: utf-8 -*-
"""
n8n Stress Test - Tests scaling to 10 instances
"""

import asyncio
import httpx
import sys
import time

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

API_BASE = "http://localhost:8000/api/n8n"

async def stress_test():
    print("=" * 60)
    print("n8n STRESS TEST - 10 Instances")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=120) as client:
        # Clear first
        print("\n[1] Clearing existing instances...")
        await client.delete(f"{API_BASE}/all")
        await asyncio.sleep(2)

        # Spawn 10 instances
        print("\n[2] Spawning 10 instances...")
        start_time = time.time()
        ports = []

        for i in range(10):
            try:
                resp = await client.post(f"{API_BASE}/spawn", timeout=120)
                data = resp.json()
                if data.get("success"):
                    port = data["instance"]["port"]
                    ports.append(port)
                    print(f"  Instance {i+1}/10 spawned on port {port}")
                else:
                    print(f"  Instance {i+1}/10 FAILED: {data.get('error', 'unknown')}")
            except Exception as e:
                print(f"  Instance {i+1}/10 ERROR: {e}")

            # Small delay between spawns to avoid overwhelming
            if i < 9:
                await asyncio.sleep(3)

        spawn_time = time.time() - start_time
        print(f"\n  Spawned {len(ports)}/10 in {spawn_time:.1f}s")

        # Wait for all to be ready
        print("\n[3] Waiting for all instances to be ready...")
        await asyncio.sleep(60)

        # Check health of all
        print("\n[4] Checking health of all instances...")
        healthy = []
        unhealthy = []

        for port in ports:
            try:
                resp = await client.get(f"http://localhost:{port}/rest/workflows", timeout=15)
                if resp.status_code == 200:
                    healthy.append(port)
                    print(f"  Port {port}: OK")
                else:
                    unhealthy.append(port)
                    print(f"  Port {port}: HTTP {resp.status_code}")
            except Exception as e:
                unhealthy.append(port)
                print(f"  Port {port}: ERROR - {str(e)[:40]}")

        print(f"\n  Healthy: {len(healthy)}/{len(ports)}")
        print(f"  Unhealthy: {len(unhealthy)}/{len(ports)}")

        # Create workflow on first instance
        print("\n[5] Creating workflow on first instance...")
        if healthy:
            try:
                resp = await client.post(
                    f"http://localhost:{healthy[0]}/rest/workflows",
                    json={"name": "Stress Test Workflow", "nodes": [], "connections": {}, "active": False},
                    timeout=30
                )
                if resp.status_code == 200:
                    print(f"  Workflow created successfully")
                else:
                    print(f"  Workflow creation failed: HTTP {resp.status_code}")
            except Exception as e:
                print(f"  Workflow creation error: {e}")

        # Get system stats
        print("\n[6] Final instance list...")
        try:
            resp = await client.get(f"{API_BASE}/list", timeout=30)
            instances = resp.json().get("instances", [])
            print(f"  Total instances: {len(instances)}")
            for inst in instances:
                print(f"    - Port {inst['port']} (PID: {inst['pid']}, running: {inst['is_running']})")
        except Exception as e:
            print(f"  Error: {e}")

        # Cleanup
        print("\n[7] Cleaning up...")
        try:
            resp = await client.delete(f"{API_BASE}/all", timeout=120)
            print("  All instances stopped")
        except Exception as e:
            print(f"  Cleanup error: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("STRESS TEST SUMMARY")
    print("=" * 60)
    print(f"  Instances spawned: {len(ports)}/10")
    print(f"  Instances healthy: {len(healthy)}/{len(ports)}")
    print(f"  Spawn time: {spawn_time:.1f}s")
    print(f"  Result: {'PASS' if len(healthy) >= 8 else 'FAIL'}")
    print("=" * 60)

    return len(healthy) >= 8

if __name__ == "__main__":
    result = asyncio.run(stress_test())
    sys.exit(0 if result else 1)
