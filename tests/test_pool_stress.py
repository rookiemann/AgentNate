"""
ComfyUI Pool Stress Test: 99 Cat Images Across 3 Instances
==========================================================
- Instance 8188 (RTX 3060, cuda:0): 33 cat images
- Instance 8189 (RTX 3090, cuda:1): 33 cat images
- Instance 8190 (RTX 3090, cuda:1): 33 cat images

Uses SDXL Turbo (4 steps, fast generation) with unique seeds per image.
Each instance gets its queue filled with all 33 jobs at once.
"""

import asyncio
import time
import random
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx

COMFYUI_PORTS = [8188, 8189, 8190]
CHECKPOINT = "sd_xl_turbo_1.0_fp16.safetensors"
JOBS_PER_INSTANCE = 33
TOTAL_JOBS = JOBS_PER_INSTANCE * len(COMFYUI_PORTS)

# SDXL Turbo settings - optimized for speed
WIDTH = 512
HEIGHT = 512
STEPS = 4
CFG = 1.0
SAMPLER = "euler"
SCHEDULER = "normal"

# Cat prompt variations for variety
CAT_STYLES = [
    "a fluffy orange tabby cat sitting on a windowsill, sunlight streaming in",
    "a black cat with bright green eyes in a mysterious dark alley",
    "a cute kitten playing with a ball of yarn, soft focus",
    "a majestic persian cat with long white fur, portrait style",
    "a siamese cat lounging on a velvet cushion, elegant",
    "a calico cat chasing butterflies in a garden, dynamic action",
    "a grey striped cat sleeping curled up on a cozy blanket",
    "a scottish fold cat with round eyes looking curious",
    "a ginger cat stretching on a wooden fence at sunset",
    "a tuxedo cat in a tiny bow tie, formal portrait",
    "a bengal cat with leopard spots prowling through grass",
    "a ragdoll cat being held like a baby, fluffy and relaxed",
    "a cat wearing a tiny hat sitting at a miniature desk",
    "an orange cat sitting in a cardboard box that is too small",
    "a fluffy white cat in a snowy winter scene, soft lighting",
    "a tabby cat watching fish in a koi pond, reflections",
    "a maine coon cat with magnificent fur, majestic pose",
    "a cat sitting on top of a stack of books in a library",
    "a sphynx cat with wrinkly skin looking regal on a throne",
    "a cat napping in a sunbeam on a hardwood floor",
    "two kittens cuddling together in a wicker basket",
    "a cat mid-jump catching a feather toy, action shot",
    "a russian blue cat with silvery fur in studio lighting",
    "a cat peeking out from behind a curtain, only eyes visible",
    "a marmalade cat sitting in front of a fireplace, warm",
    "a tiny kitten with oversized paws sitting on a mushroom",
    "a cat on a rooftop silhouetted against a full moon",
    "a grumpy looking persian cat with a squished face",
    "a cat in a field of lavender flowers, purple haze",
    "a norwegian forest cat in autumn leaves, beautiful colors",
    "an abyssinian cat with warm ticked coat, elegant pose",
    "a cat inside a paper bag, peeking out playfully",
    "a black and white cat sitting on piano keys, artistic",
]


def build_txt2img_workflow(prompt: str, seed: int) -> dict:
    """Build a ComfyUI API-format txt2img workflow for SDXL Turbo."""
    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": STEPS,
                "cfg": CFG,
                "sampler_name": SAMPLER,
                "scheduler": SCHEDULER,
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": CHECKPOINT},
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": WIDTH, "height": HEIGHT, "batch_size": 1},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": ["4", 1],
            },
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "blurry, low quality, distorted, deformed, ugly, bad anatomy",
                "clip": ["4", 1],
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "cat_stress_test", "images": ["8", 0]},
        },
    }


async def queue_jobs_on_instance(client: httpx.AsyncClient, port: int, jobs: list[dict]) -> list[str]:
    """Queue all jobs on a single instance. Returns list of prompt_ids."""
    prompt_ids = []
    for job in jobs:
        try:
            resp = await client.post(
                f"http://127.0.0.1:{port}/prompt",
                json={"prompt": job["workflow"]},
                timeout=10.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                pid = data.get("prompt_id", "unknown")
                prompt_ids.append(pid)
            else:
                print(f"  [!] Port {port}: queue failed ({resp.status_code}): {resp.text[:100]}")
                prompt_ids.append(None)
        except Exception as e:
            print(f"  [!] Port {port}: queue error: {e}")
            prompt_ids.append(None)
    return prompt_ids


async def wait_for_completion(client: httpx.AsyncClient, port: int, prompt_ids: list[str], label: str) -> dict:
    """Poll instance until all queued jobs are done. Returns timing stats."""
    total = len([p for p in prompt_ids if p])
    completed = set()
    failed = set()
    start = time.time()
    first_complete_time = None

    while len(completed) + len(failed) < total:
        await asyncio.sleep(2)
        try:
            # Check queue status
            resp = await client.get(f"http://127.0.0.1:{port}/queue", timeout=5.0)
            if resp.status_code == 200:
                q = resp.json()
                running = len(q.get("queue_running", []))
                pending = len(q.get("queue_pending", []))
            else:
                running, pending = "?", "?"

            # Check history for completed jobs
            resp = await client.get(f"http://127.0.0.1:{port}/history", timeout=5.0)
            if resp.status_code == 200:
                history = resp.json()
                for pid in prompt_ids:
                    if pid and pid not in completed and pid not in failed:
                        if pid in history:
                            entry = history[pid]
                            status = entry.get("status", {})
                            if status.get("completed", False):
                                completed.add(pid)
                                if first_complete_time is None:
                                    first_complete_time = time.time() - start
                            elif status.get("status_str") == "error":
                                failed.add(pid)

            elapsed = time.time() - start
            done = len(completed) + len(failed)
            print(f"  [{label}] {done}/{total} done (running={running}, pending={pending}) [{elapsed:.1f}s]")

        except Exception as e:
            print(f"  [{label}] poll error: {e}")

    end = time.time()
    return {
        "label": label,
        "port": port,
        "total": total,
        "completed": len(completed),
        "failed": len(failed),
        "elapsed": end - start,
        "first_complete": first_complete_time,
    }


async def main():
    print("=" * 70)
    print("  ComfyUI Pool Stress Test: 99 Cat Images")
    print("=" * 70)
    print(f"  Checkpoint: {CHECKPOINT}")
    print(f"  Resolution: {WIDTH}x{HEIGHT}")
    print(f"  Steps: {STEPS} (SDXL Turbo)")
    print(f"  Instances: {len(COMFYUI_PORTS)} ({JOBS_PER_INSTANCE} jobs each)")
    print(f"  Total: {TOTAL_JOBS} images")
    print("=" * 70)

    # Verify all instances are running
    async with httpx.AsyncClient() as client:
        for port in COMFYUI_PORTS:
            try:
                resp = await client.get(f"http://127.0.0.1:{port}/system_stats", timeout=5.0)
                if resp.status_code == 200:
                    stats = resp.json()
                    devices = stats.get("devices", [{}])
                    gpu_name = devices[0].get("name", "unknown") if devices else "unknown"
                    vram = devices[0].get("vram_total", 0) / (1024**3) if devices else 0
                    print(f"  Port {port}: OK ({gpu_name}, {vram:.1f}GB VRAM)")
                else:
                    print(f"  Port {port}: WARNING - status {resp.status_code}")
            except Exception as e:
                print(f"  Port {port}: FAILED - {e}")
                return

    print()

    # Generate jobs with unique seeds and varied prompts
    rng = random.Random(42)  # Deterministic for reproducibility
    all_jobs = {}
    for i, port in enumerate(COMFYUI_PORTS):
        jobs = []
        for j in range(JOBS_PER_INSTANCE):
            idx = i * JOBS_PER_INSTANCE + j
            prompt = CAT_STYLES[idx % len(CAT_STYLES)]
            seed = rng.randint(0, 2**32 - 1)
            jobs.append({
                "prompt": prompt,
                "seed": seed,
                "workflow": build_txt2img_workflow(prompt, seed),
            })
        all_jobs[port] = jobs

    # Phase 1: Queue all jobs on all instances simultaneously
    print("Phase 1: Queuing jobs...")
    queue_start = time.time()

    async with httpx.AsyncClient() as client:
        queue_tasks = []
        labels = {
            8188: "3060",
            8189: "3090-A",
            8190: "3090-B",
        }
        for port in COMFYUI_PORTS:
            queue_tasks.append(queue_jobs_on_instance(client, port, all_jobs[port]))

        results = await asyncio.gather(*queue_tasks)

    queue_time = time.time() - queue_start
    instance_prompt_ids = dict(zip(COMFYUI_PORTS, results))

    for port, pids in instance_prompt_ids.items():
        ok = len([p for p in pids if p])
        print(f"  Port {port} ({labels[port]}): {ok}/{JOBS_PER_INSTANCE} queued")

    print(f"  Queue time: {queue_time:.2f}s")
    print()

    # Phase 2: Wait for all instances to finish
    print("Phase 2: Generating images...")
    gen_start = time.time()

    async with httpx.AsyncClient() as client:
        wait_tasks = []
        for port in COMFYUI_PORTS:
            wait_tasks.append(
                wait_for_completion(client, port, instance_prompt_ids[port], labels[port])
            )

        stats = await asyncio.gather(*wait_tasks)

    total_gen_time = time.time() - gen_start
    total_time = time.time() - queue_start

    # Print results
    print()
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    total_completed = 0
    total_failed = 0
    for s in stats:
        total_completed += s["completed"]
        total_failed += s["failed"]
        first = f", first at {s['first_complete']:.1f}s" if s["first_complete"] else ""
        print(f"  {s['label']} (port {s['port']}): "
              f"{s['completed']}/{s['total']} completed, "
              f"{s['failed']} failed, "
              f"{s['elapsed']:.1f}s{first}")

    print(f"\n  Total: {total_completed}/{TOTAL_JOBS} images generated")
    if total_failed > 0:
        print(f"  Failed: {total_failed}")
    print(f"  Queue time: {queue_time:.2f}s")
    print(f"  Generation time: {total_gen_time:.1f}s")
    print(f"  Total wall time: {total_time:.1f}s")
    if total_completed > 0:
        print(f"  Throughput: {total_completed / total_gen_time:.2f} images/sec")
        print(f"  Avg per image: {total_gen_time / total_completed:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
