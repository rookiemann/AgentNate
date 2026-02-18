"""Benchmark ToolRouter instantiation overhead."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock minimal objects
class MockOrch:
    def get_loaded_instances(self): return []
    async def get_all_providers_health(self): return {}
class MockN8n:
    instances = {}
    main = None
    async def get_main_info(self): return None
class MockSettings:
    tools = {}
    def get(self, k, d=None): return d

from backend.tools.tool_router import ToolRouter

orch = MockOrch()
n8n = MockN8n()
settings = MockSettings()

# Warmup
for _ in range(3):
    ToolRouter(orch, n8n, settings)

# Benchmark
N = 100
start = time.perf_counter()
for _ in range(N):
    tr = ToolRouter(orch, n8n, settings)
elapsed = time.perf_counter() - start

print(f"ToolRouter instantiation: {N} iterations")
print(f"  Total: {elapsed*1000:.1f}ms")
print(f"  Per call: {elapsed/N*1000:.2f}ms")
print(f"  Rate: {N/elapsed:.0f} creates/sec")

# Also benchmark execute overhead
import asyncio
async def bench_execute():
    tr = ToolRouter(orch, n8n, settings)
    N2 = 50
    start2 = time.perf_counter()
    for _ in range(N2):
        try:
            await tr.execute("get_system_health", {})
        except:
            pass
    elapsed2 = time.perf_counter() - start2
    print(f"\nTool execute (get_system_health): {N2} calls")
    print(f"  Total: {elapsed2*1000:.1f}ms")
    print(f"  Per call: {elapsed2/N2*1000:.2f}ms")

asyncio.run(bench_execute())
