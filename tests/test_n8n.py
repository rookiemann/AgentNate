# -*- coding: utf-8 -*-
"""
n8n Multi-Instance Test Suite
Tests the n8n management system for AgentNate
"""

import asyncio
import httpx
import sys

# Fix Windows console encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

API_BASE = "http://localhost:8000/api/n8n"

async def test_n8n_system():
    results = []

    async with httpx.AsyncClient(timeout=120) as client:
        print("=" * 60)
        print("n8n Multi-Instance Test Suite")
        print("=" * 60)

        # Test 1: Clear all instances
        print("\n[Test 1] Clearing all n8n instances...")
        try:
            resp = await client.delete(f"{API_BASE}/all", timeout=120)
            data = resp.json()
            if data.get("success"):
                print("  [PASS] All instances cleared")
                results.append(("Clear all instances", True))
            else:
                print(f"  [FAIL] {data}")
                results.append(("Clear all instances", False))
        except Exception as e:
            print(f"  [FAIL] Error: {e}")
            results.append(("Clear all instances", False))

        await asyncio.sleep(2)

        # Test 2: Spawn single instance
        print("\n[Test 2] Spawning single n8n instance...")
        port1 = None
        try:
            resp = await client.post(f"{API_BASE}/spawn", timeout=120)
            data = resp.json()
            if data.get("success"):
                port1 = data["instance"]["port"]
                print(f"  [PASS] Instance spawned on port {port1}")
                results.append(("Spawn single instance", True))
            else:
                print(f"  [FAIL] {data}")
                results.append(("Spawn single instance", False))
        except Exception as e:
            print(f"  [FAIL] Error: {e}")
            results.append(("Spawn single instance", False))

        if not port1:
            print("\n[ABORT] Cannot continue without instance")
            return results

        # Wait for instance to be ready
        print("\n[Test 3] Waiting for instance to be ready...")
        ready = False
        for i in range(30):
            try:
                resp = await client.get(f"{API_BASE}/{port1}/ready", timeout=10)
                if resp.json().get("ready"):
                    print(f"  [PASS] Instance ready after {i+1} attempts")
                    ready = True
                    results.append(("Instance ready check", True))
                    break
            except:
                pass
            await asyncio.sleep(2)

        if not ready:
            print("  [FAIL] Instance never became ready")
            results.append(("Instance ready check", False))
            return results

        # Test 4: API access without auth
        print("\n[Test 4] Testing API access without authentication...")
        try:
            resp = await client.get(f"http://localhost:{port1}/rest/workflows", timeout=30)
            if resp.status_code == 200:
                print(f"  [PASS] REST API accessible (HTTP {resp.status_code})")
                results.append(("API without auth", True))
            else:
                print(f"  [FAIL] API returned {resp.status_code}")
                results.append(("API without auth", False))
        except Exception as e:
            print(f"  [FAIL] Error: {e}")
            results.append(("API without auth", False))

        # Test 5: Create workflow
        print("\n[Test 5] Creating workflow programmatically...")
        workflow_data = {
            "name": "Test Workflow",
            "nodes": [],
            "connections": {},
            "active": False
        }
        try:
            resp = await client.post(
                f"http://localhost:{port1}/rest/workflows",
                json=workflow_data,
                timeout=30
            )
            if resp.status_code == 200:
                wf_id = resp.json().get("data", {}).get("id")
                print(f"  [PASS] Workflow created with ID: {wf_id}")
                results.append(("Create workflow", True))
            else:
                print(f"  [FAIL] Create failed: HTTP {resp.status_code}")
                results.append(("Create workflow", False))
        except Exception as e:
            print(f"  [FAIL] Error: {e}")
            results.append(("Create workflow", False))

        # Test 6: Spawn 3 more instances sequentially (more stable than parallel)
        print("\n[Test 6] Spawning 3 more instances...")
        spawned = 0
        for i in range(3):
            try:
                resp = await client.post(f"{API_BASE}/spawn", timeout=120)
                if resp.json().get("success"):
                    spawned += 1
                    print(f"  Instance {i+1} spawned")
            except Exception as e:
                print(f"  Instance {i+1} failed: {e}")
            await asyncio.sleep(5)

        print(f"  [{'PASS' if spawned >= 2 else 'FAIL'}] {spawned}/3 instances spawned")
        results.append(("Spawn multiple instances", spawned >= 2))

        # Wait for instances
        print("\n  Waiting for instances to be ready...")
        await asyncio.sleep(45)

        # Test 7: Check all instances healthy
        print("\n[Test 7] Checking all instances are healthy...")
        try:
            resp = await client.get(f"{API_BASE}/list", timeout=30)
            instances = resp.json().get("instances", [])
            print(f"  Found {len(instances)} instances")

            healthy = 0
            for inst in instances:
                port = inst["port"]
                try:
                    r = await client.get(f"http://localhost:{port}/rest/workflows", timeout=15)
                    if r.status_code == 200:
                        healthy += 1
                        print(f"    Port {port}: healthy")
                    else:
                        print(f"    Port {port}: HTTP {r.status_code}")
                except Exception as e:
                    print(f"    Port {port}: error - {str(e)[:50]}")

            results.append(("All instances healthy", healthy == len(instances)))
            print(f"  [{'PASS' if healthy == len(instances) else 'FAIL'}] {healthy}/{len(instances)} healthy")
        except Exception as e:
            print(f"  [FAIL] Error: {e}")
            results.append(("All instances healthy", False))

        # Test 8: Stop single instance
        print("\n[Test 8] Stopping a single instance...")
        try:
            resp = await client.get(f"{API_BASE}/list", timeout=30)
            instances = resp.json().get("instances", [])
            if instances:
                port_to_stop = instances[-1]["port"]
                resp = await client.delete(f"{API_BASE}/{port_to_stop}", timeout=30)
                if resp.json().get("success"):
                    print(f"  [PASS] Instance on port {port_to_stop} stopped")
                    results.append(("Stop single instance", True))
                else:
                    print(f"  [FAIL] Stop failed")
                    results.append(("Stop single instance", False))
        except Exception as e:
            print(f"  [FAIL] Error: {e}")
            results.append(("Stop single instance", False))

        # Test 9: Stop all
        print("\n[Test 9] Stopping all instances...")
        try:
            resp = await client.delete(f"{API_BASE}/all", timeout=120)
            if resp.json().get("success"):
                print("  [PASS] All instances stopped")
                results.append(("Stop all instances", True))
            else:
                print("  [FAIL] Stop all failed")
                results.append(("Stop all instances", False))
        except Exception as e:
            print(f"  [FAIL] Error: {e}")
            results.append(("Stop all instances", False))

        # Verify all stopped
        await asyncio.sleep(2)
        try:
            resp = await client.get(f"{API_BASE}/list", timeout=30)
            count = resp.json().get("count", -1)
            print(f"  Verified: {count} instances remaining")
        except:
            pass

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\nPassed: {passed}/{total}")
    print()
    for name, success in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"  {status}: {name}")

    print("\n" + "=" * 60)
    return results

if __name__ == "__main__":
    results = asyncio.run(test_n8n_system())
    sys.exit(0 if all(s for _, s in results) else 1)
