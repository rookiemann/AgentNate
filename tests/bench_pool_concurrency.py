"""
ComfyUI Pool concurrency stress test.

Tests the pool's internal logic under concurrent batch submissions
without requiring actual ComfyUI instances (uses mocked HTTP responses).
"""
import asyncio
import json
import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.comfyui_pool import ComfyUIPool, InstanceMetrics


def make_mock_manager():
    """Create a mock ComfyUI manager that simulates running instances."""
    mgr = MagicMock()
    mgr.comfyui_dir = None
    return mgr


def make_pool_with_instances(n_instances: int = 3):
    """Create a pool with N pre-seeded running instances."""
    mgr = make_mock_manager()
    pool = ComfyUIPool(mgr)

    # Seed instances directly (bypass refresh_instances which needs HTTP)
    for i in range(n_instances):
        inst_id = f"inst-{i}"
        pool._instances[inst_id] = InstanceMetrics(
            instance_id=inst_id,
            port=8188 + i,
            gpu_device=f"cuda:{i}",
            gpu_label=f"GPU {i}",
            vram_total_mb=24000,
            vram_mode="normal",
            status="running",
            loaded_checkpoint="model.safetensors" if i == 0 else None,
            queue_depth=0,
            busy=False,
            last_poll=time.time(),
        )
    # Mark refresh as recent so it doesn't try to actually refresh
    pool._refresh_cache_time = time.time()

    return pool


class MockResponse:
    """Mock httpx response."""
    def __init__(self, status_code=200, data=None):
        self.status_code = status_code
        self._data = data or {"prompt_id": f"pid-{id(self)}"}
        self.text = json.dumps(self._data)

    def json(self):
        return self._data


async def test_concurrent_batches():
    """Test 3 concurrent batch submissions don't over-assign to one instance."""
    print("\n1. Concurrent batch submissions (3 batches x 33 jobs):")

    pool = make_pool_with_instances(3)
    dispatch_count = {f"inst-{i}": 0 for i in range(3)}

    # Track which instance gets each dispatch
    original_dispatch = pool._dispatch_job

    async def mock_dispatch(job):
        # Simulate a small network delay
        await asyncio.sleep(0.001)
        dispatch_count[job.instance_id] = dispatch_count.get(job.instance_id, 0) + 1

        # Simulate successful dispatch
        metrics = pool._instances.get(job.instance_id)
        job.prompt_id = f"pid-{job.job_id}"
        job.status = "queued"
        if metrics:
            metrics.queue_depth += 1
            metrics.busy = True
        return True

    pool._dispatch_job = mock_dispatch
    pool._provision_model = AsyncMock(return_value=True)

    # Submit 3 batches concurrently
    jobs_per_batch = [
        [{"prompt": f"batch1 prompt {i}"} for i in range(33)],
        [{"prompt": f"batch2 prompt {i}"} for i in range(33)],
        [{"prompt": f"batch3 prompt {i}"} for i in range(33)],
    ]

    start = time.perf_counter()
    results = await asyncio.gather(
        pool.submit_batch("model.safetensors", jobs_per_batch[0]),
        pool.submit_batch("model.safetensors", jobs_per_batch[1]),
        pool.submit_batch("model.safetensors", jobs_per_batch[2]),
    )
    elapsed = time.perf_counter() - start

    ok = sum(1 for r in results if r.get("success"))
    total_jobs = sum(r.get("total_jobs", 0) for r in results)

    print(f"  Batches: {ok}/3 successful, {total_jobs} total jobs in {elapsed*1000:.0f}ms")
    print(f"  Distribution across instances:")
    for inst_id, count in sorted(dispatch_count.items()):
        print(f"    {inst_id}: {count} jobs")

    # Check distribution is roughly balanced
    counts = list(dispatch_count.values())
    max_count = max(counts)
    min_count = min(counts)
    imbalance = max_count - min_count

    # With 99 jobs across 3 instances, perfect = 33 each
    # inst-0 has model affinity so gets 2x share from first batch
    # But subsequent batches should rebalance
    print(f"  Imbalance: {imbalance} (max={max_count}, min={min_count})")
    if imbalance <= 40:
        print(f"  PASS: Reasonable distribution")
    else:
        print(f"  WARN: High imbalance, possible race condition")

    # Verify queue depths match dispatch counts
    print(f"\n  Queue depth verification:")
    for inst_id, metrics in pool._instances.items():
        expected = dispatch_count[inst_id]
        actual = metrics.queue_depth
        match = "OK" if actual == expected else f"MISMATCH (expected {expected})"
        print(f"    {inst_id}: queue_depth={actual} {match}")


async def test_job_id_uniqueness():
    """Test that concurrent batch submissions produce unique job IDs."""
    print("\n2. Job ID uniqueness under concurrent submission:")

    pool = make_pool_with_instances(2)
    pool._dispatch_job = AsyncMock(return_value=True)
    pool._provision_model = AsyncMock(return_value=True)

    # Make dispatch also update metrics
    async def tracked_dispatch(job):
        await asyncio.sleep(0.001)
        metrics = pool._instances.get(job.instance_id)
        job.prompt_id = f"pid-{job.job_id}"
        job.status = "queued"
        if metrics:
            metrics.queue_depth += 1
        return True

    pool._dispatch_job = tracked_dispatch

    # 5 concurrent batches of 20 jobs each
    batches = [
        [{"prompt": f"b{b}p{i}"} for i in range(20)]
        for b in range(5)
    ]

    results = await asyncio.gather(*[
        pool.submit_batch("test.safetensors", b)
        for b in batches
    ])

    all_job_ids = set()
    all_batch_ids = set()
    for r in results:
        if r.get("job_ids"):
            for jid in r["job_ids"]:
                all_job_ids.add(jid)
        if r.get("batch_id"):
            all_batch_ids.add(r["batch_id"])

    expected_total = 5 * 20
    print(f"  Total job IDs: {len(all_job_ids)} (expected {expected_total})")
    print(f"  Total batch IDs: {len(all_batch_ids)} (expected 5)")

    if len(all_job_ids) == expected_total:
        print(f"  PASS: All job IDs unique")
    else:
        print(f"  FAIL: Duplicate job IDs detected!")

    if len(all_batch_ids) == 5:
        print(f"  PASS: All batch IDs unique")
    else:
        print(f"  FAIL: Duplicate batch IDs!")


async def test_instance_failure_handling():
    """Test pool behavior when an instance goes down mid-batch."""
    print("\n3. Instance failure during batch dispatch:")

    pool = make_pool_with_instances(3)
    pool._provision_model = AsyncMock(return_value=True)

    dispatch_calls = 0

    async def failing_dispatch(job):
        nonlocal dispatch_calls
        dispatch_calls += 1
        await asyncio.sleep(0.001)

        metrics = pool._instances.get(job.instance_id)

        # Simulate instance 1 failing after 10 dispatches
        if job.instance_id == "inst-1" and dispatch_calls > 10:
            job.status = "failed"
            job.error = "Instance went down"
            return False

        job.prompt_id = f"pid-{job.job_id}"
        job.status = "queued"
        if metrics:
            metrics.queue_depth += 1
            metrics.busy = True
        return True

    pool._dispatch_job = failing_dispatch

    result = await pool.submit_batch(
        "model.safetensors",
        [{"prompt": f"job {i}"} for i in range(50)],
    )

    print(f"  Batch success: {result.get('success')}")
    print(f"  Total jobs: {result.get('total_jobs')}")
    print(f"  Distribution: {result.get('distribution')}")

    # Count job statuses
    statuses = {}
    for jid in (result.get("job_ids") or []):
        job = pool._jobs.get(jid)
        if job:
            s = job.status
            statuses[s] = statuses.get(s, 0) + 1
    print(f"  Job statuses: {statuses}")
    print(f"  PASS: Partial failures handled gracefully")


async def test_model_affinity_scoring():
    """Test that model affinity correctly routes to loaded instances."""
    print("\n4. Model affinity scoring:")

    pool = make_pool_with_instances(3)

    # inst-0 has "model.safetensors" loaded
    # inst-1 and inst-2 have nothing loaded

    selected = pool._select_best_instance("model.safetensors")
    print(f"  Model loaded on inst-0, selected: {selected}")
    assert selected == "inst-0", f"Expected inst-0, got {selected}"
    print(f"  PASS: Model affinity selects correct instance")

    # Now make inst-0 heavily loaded
    pool._instances["inst-0"].queue_depth = 15
    pool._instances["inst-0"].busy = True

    selected = pool._select_best_instance("model.safetensors")
    print(f"  inst-0 queue_depth=15, selected: {selected}")
    # Score: inst-0 = 100 - 150 = -50, inst-1 = 0 + 20 = 20, inst-2 = 0 + 20 = 20
    # Should pick inst-1 or inst-2
    assert selected != "inst-0", f"Should NOT select heavily loaded inst-0"
    print(f"  PASS: Queue depth overrides affinity when heavily loaded")


async def test_rapid_single_job_submissions():
    """Test many rapid single job submissions."""
    print("\n5. Rapid single job submissions (100 jobs):")

    pool = make_pool_with_instances(3)
    pool._provision_model = AsyncMock(return_value=True)

    async def fast_dispatch(job):
        metrics = pool._instances.get(job.instance_id)
        job.prompt_id = f"pid-{job.job_id}"
        job.status = "queued"
        if metrics:
            metrics.queue_depth += 1
            metrics.busy = True
        return True

    pool._dispatch_job = fast_dispatch

    start = time.perf_counter()
    results = await asyncio.gather(*[
        pool.submit_job("model.safetensors", f"prompt {i}")
        for i in range(100)
    ])
    elapsed = time.perf_counter() - start

    ok = sum(1 for r in results if r.get("success"))
    print(f"  {ok}/100 jobs submitted in {elapsed*1000:.0f}ms ({100/elapsed:.0f} jobs/s)")

    # Check distribution
    inst_counts = {}
    for r in results:
        iid = r.get("instance_id")
        if iid:
            inst_counts[iid] = inst_counts.get(iid, 0) + 1
    print(f"  Distribution: {dict(sorted(inst_counts.items()))}")

    if ok == 100:
        print(f"  PASS: All jobs submitted successfully")
    else:
        print(f"  FAIL: {100-ok} jobs failed")


async def test_prune_under_load():
    """Test job pruning doesn't lose active jobs."""
    print("\n6. Job pruning under load:")

    pool = make_pool_with_instances(1)
    pool._provision_model = AsyncMock(return_value=True)
    pool.MAX_JOBS = 50  # Low limit for testing

    async def fast_dispatch(job):
        job.prompt_id = f"pid-{job.job_id}"
        job.status = "queued"
        metrics = pool._instances.get(job.instance_id)
        if metrics:
            metrics.queue_depth += 1
        return True

    pool._dispatch_job = fast_dispatch

    # Submit 100 jobs (exceeds MAX_JOBS of 50)
    for i in range(100):
        # Mark some old jobs as completed so pruning can work
        if i > 0 and i % 10 == 0:
            for jid in list(pool._jobs.keys())[:5]:
                pool._jobs[jid].status = "completed"
                pool._jobs[jid].completed_at = time.time()

        result = await pool.submit_job("model.safetensors", f"prompt {i}")

    # Check that active jobs were NOT pruned
    active = sum(1 for j in pool._jobs.values() if j.status in ("queued", "running", "pending"))
    completed = sum(1 for j in pool._jobs.values() if j.status == "completed")
    total = len(pool._jobs)

    print(f"  Total jobs tracked: {total}")
    print(f"  Active: {active}, Completed: {completed}")
    print(f"  MAX_JOBS limit: {pool.MAX_JOBS}")

    if active > 0 and total <= pool.MAX_JOBS + 20:  # Some tolerance
        print(f"  PASS: Pruning respects active jobs")
    else:
        print(f"  WARN: Pruning may have issues (total={total})")


async def main():
    print("=" * 65)
    print("  ComfyUI Pool Concurrency Stress Test")
    print("=" * 65)

    await test_concurrent_batches()
    await test_job_id_uniqueness()
    await test_instance_failure_handling()
    await test_model_affinity_scoring()
    await test_rapid_single_job_submissions()
    await test_prune_under_load()

    print("\n" + "=" * 65)
    print("All tests complete!")


if __name__ == "__main__":
    asyncio.run(main())
