"""
ComfyUI Instance Pool

Dispatch layer that wraps the existing InstanceManager to provide:
- Model-aware routing (prefer instances with checkpoint already loaded)
- Queue depth tracking per instance
- Batch job distribution across multiple instances
- Automatic model provisioning (download missing checkpoints before dispatch)
- Pool-level job tracking with aggregated results
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import httpx

from backend.comfyui_utils import detect_model_defaults, build_txt2img_workflow, estimate_vram_mb

logger = logging.getLogger("comfyui_pool")

# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #

@dataclass
class InstanceMetrics:
    """Tracked state of a single ComfyUI instance."""
    instance_id: str
    port: int
    gpu_device: str
    gpu_label: str
    vram_total_mb: int
    vram_mode: str
    status: str                                # running / stopped / error
    loaded_checkpoint: Optional[str] = None    # detected from /history
    queue_depth: int = 0                       # queue_running + queue_pending
    busy: bool = False                         # queue_running > 0
    last_poll: float = 0.0


@dataclass
class PoolJob:
    """A single generation job tracked by the pool."""
    job_id: str
    status: str                                # pending / provisioning / queued / running / completed / failed
    checkpoint: str
    prompt: str
    negative_prompt: str
    params: Dict[str, Any]
    instance_id: Optional[str] = None
    prompt_id: Optional[str] = None
    batch_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    result: Optional[Dict] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "checkpoint": self.checkpoint,
            "instance_id": self.instance_id,
            "prompt_id": self.prompt_id,
            "batch_id": self.batch_id,
            "params": self.params,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error,
        }


@dataclass
class BatchJob:
    """Aggregated tracking for a batch of pool jobs."""
    batch_id: str
    total: int
    completed: int = 0
    failed: int = 0
    status: str = "running"
    job_ids: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "batch_id": self.batch_id,
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "status": self.status,
            "job_ids": self.job_ids,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


# --------------------------------------------------------------------------- #
# Pool Manager
# --------------------------------------------------------------------------- #

class ComfyUIPool:
    """
    Pool dispatch layer wrapping the existing ComfyUIManager.

    Provides smart routing, batch distribution, auto-provisioning,
    and pool-level job tracking across multiple ComfyUI instances.
    """

    POLL_INTERVAL = 10      # seconds between background polls
    MAX_JOBS = 500          # prune oldest completed jobs when exceeded
    PROVISION_TIMEOUT = 1800  # 30 min max wait for model download

    def __init__(self, comfyui_manager, media_catalog=None):
        self.manager = comfyui_manager
        self.media_catalog = media_catalog
        self._instances: Dict[str, InstanceMetrics] = {}
        self._jobs: Dict[str, PoolJob] = {}
        self._batches: Dict[str, BatchJob] = {}
        self._lock = asyncio.Lock()
        self._poll_task: Optional[asyncio.Task] = None
        self._download_locks: Dict[str, asyncio.Event] = {}  # checkpoint → event
        self._needs_scan = False  # flag to trigger media scan after completions
        # Shared HTTP client for instance polling (connection pooling)
        self._http_client: Optional[httpx.AsyncClient] = None
        # TTL cache for pool status
        self._pool_status_cache: Optional[Dict] = None
        self._pool_status_cache_time: float = 0.0
        self._POOL_STATUS_CACHE_TTL: float = 5.0  # seconds
        # TTL cache for refresh_instances (avoid re-polling on every status call)
        self._refresh_cache_time: float = 0.0
        self._REFRESH_CACHE_TTL: float = 5.0
        self._refresh_lock: Optional[asyncio.Lock] = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def start(self):
        """Start the background polling loop."""
        if self._poll_task is None or self._poll_task.done():
            self._poll_task = asyncio.create_task(self._background_poll())
            logger.info("[pool] Background polling started")

    async def stop(self):
        """Stop polling and clean up."""
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        self._poll_task = None
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None
        logger.info("[pool] Stopped")

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the shared HTTP client for instance polling."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=5,
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
        return self._http_client

    # ------------------------------------------------------------------ #
    # Instance tracking
    # ------------------------------------------------------------------ #

    async def refresh_instances(self, force: bool = False):
        """Fetch instance list from the management API and poll each running one."""
        now = time.time()
        if not force and (now - self._refresh_cache_time) < self._REFRESH_CACHE_TTL:
            return

        # Stampede lock: only one caller fetches at a time
        if self._refresh_lock is None:
            self._refresh_lock = asyncio.Lock()
        async with self._refresh_lock:
            # Double-check after acquiring lock
            now = time.time()
            if not force and (now - self._refresh_cache_time) < self._REFRESH_CACHE_TTL:
                return
            if not self.manager or not await self.manager.is_api_running():
                return

            try:
                data = await self.manager.proxy("GET", "/api/instances")
                instances = data if isinstance(data, list) else data.get("instances", [])
            except Exception as e:
                logger.debug(f"[pool] Failed to fetch instances: {e}")
                return

            # Detect GPU VRAM for mapping
            gpu_vram = await self._get_gpu_vram_map()

            seen_ids = set()
            for inst in instances:
                iid = str(inst.get("instance_id") or inst.get("id", ""))
                if not iid:
                    continue
                seen_ids.add(iid)

                port = inst.get("port", 0)
                gpu_device = str(inst.get("gpu_device", "cpu"))
                status = "running" if inst.get("is_running") else (inst.get("status", "stopped"))

                metrics = self._instances.get(iid)
                if not metrics:
                    metrics = InstanceMetrics(
                        instance_id=iid,
                        port=port,
                        gpu_device=gpu_device,
                        gpu_label=inst.get("gpu_label", f"GPU {gpu_device}"),
                        vram_total_mb=gpu_vram.get(gpu_device, 0),
                        vram_mode=inst.get("vram_mode", "normal"),
                        status=status,
                    )
                    self._instances[iid] = metrics
                else:
                    metrics.port = port
                    metrics.status = status

                # Poll queue + checkpoint for running instances
                if status == "running":
                    await self._poll_instance(metrics)

            # Remove stale entries
            for iid in list(self._instances.keys()):
                if iid not in seen_ids:
                    del self._instances[iid]

            self._refresh_cache_time = time.time()

    async def _poll_instance(self, metrics: InstanceMetrics):
        """Poll a running instance for queue depth and loaded checkpoint."""
        port = metrics.port
        try:
            client = await self._get_client()
            # Queue depth
            r = await client.get(f"http://127.0.0.1:{port}/queue")
            if r.status_code == 200:
                q = r.json()
                running = len(q.get("queue_running", []))
                pending = len(q.get("queue_pending", []))
                metrics.queue_depth = running + pending
                metrics.busy = running > 0

            # Detect loaded checkpoint from recent history
            ckpt = await self._detect_loaded_checkpoint(port)
            if ckpt:
                metrics.loaded_checkpoint = ckpt

            metrics.last_poll = time.time()
        except Exception as e:
            logger.debug(f"[pool] Poll failed for {metrics.instance_id}: {e}")

    async def _detect_loaded_checkpoint(self, port: int) -> Optional[str]:
        """
        Inspect the most recent history entry to determine which checkpoint
        the instance has loaded in VRAM.

        ComfyUI keeps models loaded until a different one is requested.
        """
        try:
            client = await self._get_client()
            r = await client.get(
                f"http://127.0.0.1:{port}/history",
                params={"max_items": "1"},
            )
            if r.status_code != 200:
                return None

            history = r.json()
            for _prompt_id, entry in history.items():
                prompt_data = entry.get("prompt", [])
                # prompt_data is a list: [number, prompt_id, workflow_dict, ...]
                workflow = None
                if isinstance(prompt_data, list) and len(prompt_data) >= 3:
                    workflow = prompt_data[2]
                elif isinstance(prompt_data, dict):
                    workflow = prompt_data

                if not isinstance(workflow, dict):
                    continue

                for _node_id, node in workflow.items():
                    ct = node.get("class_type", "")
                    if ct in ("CheckpointLoaderSimple", "CheckpointLoader",
                              "UNETLoader", "DiffusersLoader"):
                        inputs = node.get("inputs", {})
                        return (inputs.get("ckpt_name")
                                or inputs.get("unet_name")
                                or inputs.get("model_path"))
        except Exception:
            pass
        return None

    async def _get_gpu_vram_map(self) -> Dict[str, int]:
        """Get GPU device → VRAM (MB) mapping."""
        try:
            data = await self.manager.proxy("GET", "/api/gpus")
            gpus = data if isinstance(data, list) else data.get("gpus", [])
            return {
                str(g.get("index", g.get("device", ""))): g.get("memory_total_mb", 0)
                for g in gpus
            }
        except Exception:
            return {}

    # ------------------------------------------------------------------ #
    # Smart routing
    # ------------------------------------------------------------------ #

    def _select_best_instance(self, checkpoint: str,
                              vram_needed: int = 0) -> Optional[str]:
        """
        Pick the best running instance for a job.

        Scoring:
        - +100 if checkpoint already loaded (avoid model swap)
        - +20 if instance is idle (queue empty)
        - -10 per queued item
        Hard filters:
        - Skip non-running instances
        - Skip if VRAM insufficient for model type
        """
        candidates = []

        for m in self._instances.values():
            if m.status != "running":
                continue

            # VRAM hard filter
            if vram_needed > 0 and m.vram_total_mb > 0:
                # lowvram/novram modes can squeeze more, give 50% headroom
                effective_vram = m.vram_total_mb
                if m.vram_mode in ("low", "none"):
                    effective_vram = int(m.vram_total_mb * 1.5)  # offloading helps
                if effective_vram < vram_needed * 0.7:  # 30% tolerance
                    continue

            score = 0
            # Model affinity
            if m.loaded_checkpoint and m.loaded_checkpoint == checkpoint:
                score += 100
            # Idle bonus
            if not m.busy and m.queue_depth == 0:
                score += 20
            # Queue penalty
            score -= m.queue_depth * 10

            candidates.append((score, m.instance_id))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    # ------------------------------------------------------------------ #
    # Auto-provisioning
    # ------------------------------------------------------------------ #

    async def _provision_model(self, checkpoint: str) -> str:
        """
        Ensure a checkpoint exists locally. Downloads if missing.

        Returns the actual filename (with extension) for ComfyUI, or empty string on failure.
        Accepts checkpoint as name (no ext), filename, or path basename.
        """
        # Check if already downloading
        if checkpoint in self._download_locks:
            logger.info(f"[pool] Waiting for existing download of {checkpoint}")
            try:
                await asyncio.wait_for(
                    self._download_locks[checkpoint].wait(),
                    timeout=self.PROVISION_TIMEOUT,
                )
                return checkpoint
            except asyncio.TimeoutError:
                return ""

        # Check if already installed
        try:
            local = await self.manager.proxy("GET", "/api/models/local")
            models_by_folder = local if isinstance(local, dict) else {}
            # The local API nests under "models": {"checkpoints": [...], ...}
            if "models" in models_by_folder:
                models_by_folder = models_by_folder["models"]
            for _folder, model_list in models_by_folder.items():
                if not isinstance(model_list, list):
                    continue
                for m in model_list:
                    model_name = m.get("name", "")
                    model_filename = m.get("filename", "")
                    model_path = m.get("path", "")
                    path_basename = model_path.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] if model_path else ""
                    if checkpoint in (model_name, model_filename, path_basename):
                        # Return the actual filename ComfyUI needs
                        return path_basename or model_filename or checkpoint
        except Exception as e:
            logger.warning(f"[pool] Model scan failed: {e}")

        # Search registry for downloadable match
        logger.info(f"[pool] Checkpoint '{checkpoint}' not found locally, searching registry...")
        event = asyncio.Event()
        self._download_locks[checkpoint] = event

        try:
            registry = await self.manager.proxy("GET", "/api/models/registry")
            reg_models = registry.get("models", []) if isinstance(registry, dict) else registry
            if not isinstance(reg_models, list):
                reg_models = []

            # Match by filename or name substring
            ckpt_lower = checkpoint.lower()
            matching = [
                m for m in reg_models
                if m.get("filename", "").lower() == ckpt_lower
                or ckpt_lower in m.get("name", "").lower()
            ]

            if not matching:
                logger.warning(f"[pool] No registry match for '{checkpoint}'")
                return ""

            best = matching[0]
            resolved_filename = best.get("filename", checkpoint)
            model_ids = [best.get("id") or best.get("name")]
            logger.info(f"[pool] Auto-downloading: {model_ids[0]}")

            result = await self.manager.proxy(
                "POST", "/api/models/download",
                json={"model_ids": model_ids},
            )
            job_id = result.get("job_id") if isinstance(result, dict) else None

            if not job_id:
                logger.error(f"[pool] Download request failed: {result}")
                return ""

            # Poll until complete
            deadline = time.time() + self.PROVISION_TIMEOUT
            while time.time() < deadline:
                await asyncio.sleep(5)
                try:
                    status = await self.manager.proxy("GET", f"/api/jobs/{job_id}")
                    job_status = status.get("status") if isinstance(status, dict) else None
                    if job_status == "completed":
                        logger.info(f"[pool] Download complete: {resolved_filename}")
                        return resolved_filename
                    if job_status == "failed":
                        logger.error(f"[pool] Download failed: {status.get('error')}")
                        return ""
                except Exception:
                    pass

            logger.error(f"[pool] Download timeout for {checkpoint}")
            return ""

        except Exception as e:
            logger.error(f"[pool] Provision error: {e}")
            return ""
        finally:
            event.set()
            self._download_locks.pop(checkpoint, None)

    # ------------------------------------------------------------------ #
    # Job dispatch
    # ------------------------------------------------------------------ #

    async def submit_job(self, checkpoint: str, prompt: str,
                         negative_prompt: str = "blurry, low quality, distorted",
                         width: int = None, height: int = None,
                         steps: int = None, cfg: float = None,
                         seed: int = -1, sampler_name: str = None,
                         scheduler: str = None,
                         instance_id: str = None) -> Dict:
        """Submit a single generation job to the pool."""
        # Refresh instance state
        await self.refresh_instances()

        running = [m for m in self._instances.values() if m.status == "running"]
        if not running:
            return {"success": False, "error": "No running ComfyUI instances. Start instances first."}

        # Fill defaults from checkpoint type
        defaults = detect_model_defaults(checkpoint)
        width = width or defaults["width"]
        height = height or defaults["height"]
        steps = steps or defaults["steps"]
        cfg = cfg if cfg is not None else defaults["cfg"]
        sampler_name = sampler_name or defaults["sampler_name"]
        scheduler = scheduler or defaults["scheduler"]

        # Create pool job
        job = PoolJob(
            job_id=str(uuid.uuid4())[:12],
            status="pending",
            checkpoint=checkpoint,
            prompt=prompt,
            negative_prompt=negative_prompt,
            params={
                "width": width, "height": height, "steps": steps,
                "cfg": cfg, "seed": seed, "sampler_name": sampler_name,
                "scheduler": scheduler,
            },
        )
        self._jobs[job.job_id] = job

        # Auto-provision: checks local models, resolves to actual filename
        job.status = "provisioning"
        resolved = await self._provision_model(checkpoint)
        if not resolved:
            job.status = "failed"
            job.error = f"Checkpoint '{checkpoint}' not found locally and could not be downloaded."
            return {"success": False, "error": job.error, "job_id": job.job_id}
        # Use the actual filename (with extension) for the workflow
        job.checkpoint = resolved

        # Select best instance
        if instance_id:
            if instance_id not in self._instances:
                job.status = "failed"
                job.error = f"Instance '{instance_id}' not found."
                return {"success": False, "error": job.error}
            selected = instance_id
        else:
            vram_needed = estimate_vram_mb(checkpoint)
            selected = self._select_best_instance(checkpoint, vram_needed)
            if not selected:
                job.status = "failed"
                job.error = "No suitable instance found (VRAM or status mismatch)."
                return {"success": False, "error": job.error}

        job.instance_id = selected

        # Dispatch
        result = await self._dispatch_job(job)
        if not result:
            return {"success": False, "error": job.error or "Dispatch failed", "job_id": job.job_id}

        self._prune_jobs()

        return {
            "success": True,
            "job_id": job.job_id,
            "instance_id": job.instance_id,
            "prompt_id": job.prompt_id,
            "status": job.status,
            "message": f"Job queued on instance {selected}. Use comfyui_pool_results to track.",
        }

    async def submit_batch(self, checkpoint: str, jobs: List[Dict],
                           steps: int = None, cfg: float = None,
                           width: int = None, height: int = None,
                           sampler_name: str = None,
                           scheduler: str = None) -> Dict:
        """Submit N jobs distributed across pool instances."""
        await self.refresh_instances()

        running = [m for m in self._instances.values() if m.status == "running"]
        if not running:
            return {"success": False, "error": "No running ComfyUI instances."}

        if not jobs:
            return {"success": False, "error": "No jobs provided."}

        # Fill defaults
        defaults = detect_model_defaults(checkpoint)
        width = width or defaults["width"]
        height = height or defaults["height"]
        steps = steps or defaults["steps"]
        cfg = cfg if cfg is not None else defaults["cfg"]
        sampler_name = sampler_name or defaults["sampler_name"]
        scheduler = scheduler or defaults["scheduler"]

        # Provision model once (resolves to actual filename)
        resolved = await self._provision_model(checkpoint)
        if not resolved:
            return {"success": False, "error": f"Checkpoint '{checkpoint}' not available."}
        checkpoint = resolved

        # Create batch
        batch_id = f"batch-{str(uuid.uuid4())[:8]}"
        batch = BatchJob(batch_id=batch_id, total=len(jobs))
        self._batches[batch_id] = batch

        # Create pool jobs
        vram_needed = estimate_vram_mb(checkpoint)
        pool_jobs = []
        for j in jobs:
            pj = PoolJob(
                job_id=str(uuid.uuid4())[:12],
                status="pending",
                checkpoint=checkpoint,
                prompt=j.get("prompt", ""),
                negative_prompt=j.get("negative_prompt", "blurry, low quality, distorted"),
                params={
                    "width": j.get("width", width),
                    "height": j.get("height", height),
                    "steps": steps, "cfg": cfg,
                    "seed": j.get("seed", -1),
                    "sampler_name": sampler_name,
                    "scheduler": scheduler,
                },
                batch_id=batch_id,
            )
            self._jobs[pj.job_id] = pj
            batch.job_ids.append(pj.job_id)
            pool_jobs.append(pj)

        # Distribute across instances
        # Sort instances: has-checkpoint first, then by queue depth ascending
        scored = []
        for m in running:
            score = 0
            if m.loaded_checkpoint and m.loaded_checkpoint == checkpoint:
                score += 100
            score -= m.queue_depth * 10
            if m.vram_total_mb > 0 and m.vram_total_mb * 0.7 < vram_needed:
                continue  # skip undersize
            scored.append((score, m.instance_id))
        scored.sort(key=lambda x: x[0], reverse=True)

        if not scored:
            batch.status = "failed"
            return {"success": False, "error": "No suitable instances for this model."}

        # Round-robin weighted: instances with model loaded get 2x share
        instance_ids = [s[1] for s in scored]
        weights = []
        for s in scored:
            iid = s[1]
            m = self._instances[iid]
            w = 2 if (m.loaded_checkpoint and m.loaded_checkpoint == checkpoint) else 1
            weights.append(w)

        # Build assignment buckets
        total_weight = sum(weights)
        assignments: Dict[str, List[PoolJob]] = {iid: [] for iid in instance_ids}
        job_idx = 0
        for i, iid in enumerate(instance_ids):
            share = max(1, round(len(pool_jobs) * weights[i] / total_weight))
            for _ in range(share):
                if job_idx >= len(pool_jobs):
                    break
                assignments[iid].append(pool_jobs[job_idx])
                job_idx += 1

        # Assign any remaining jobs (rounding leftovers)
        while job_idx < len(pool_jobs):
            # Pick instance with fewest assigned jobs
            least_loaded = min(instance_ids, key=lambda x: len(assignments[x]))
            assignments[least_loaded].append(pool_jobs[job_idx])
            job_idx += 1

        # Dispatch all jobs
        distribution = {}
        for iid, assigned_jobs in assignments.items():
            if not assigned_jobs:
                continue
            distribution[iid] = len(assigned_jobs)
            for pj in assigned_jobs:
                pj.instance_id = iid
                await self._dispatch_job(pj)

        self._prune_jobs()

        return {
            "success": True,
            "batch_id": batch_id,
            "total_jobs": len(pool_jobs),
            "distribution": distribution,
            "job_ids": batch.job_ids,
            "message": f"Batch of {len(pool_jobs)} jobs distributed across {len(distribution)} instance(s).",
        }

    async def _dispatch_job(self, job: PoolJob) -> bool:
        """Build workflow and queue on the assigned instance."""
        metrics = self._instances.get(job.instance_id)
        if not metrics or metrics.status != "running":
            job.status = "failed"
            job.error = f"Instance {job.instance_id} not available."
            return False

        try:
            workflow = build_txt2img_workflow(
                checkpoint=job.checkpoint,
                prompt=job.prompt,
                negative_prompt=job.negative_prompt,
                width=job.params["width"],
                height=job.params["height"],
                steps=job.params["steps"],
                cfg=job.params["cfg"],
                seed=job.params["seed"],
                sampler_name=job.params["sampler_name"],
                scheduler=job.params["scheduler"],
            )

            client = await self._get_client()
            r = await client.post(
                f"http://127.0.0.1:{metrics.port}/prompt",
                json={"prompt": workflow},
                timeout=30,
            )
            if r.status_code >= 400:
                job.status = "failed"
                job.error = f"ComfyUI returned {r.status_code}: {r.text[:300]}"
                return False

            data = r.json()
            job.prompt_id = data.get("prompt_id")
            job.status = "queued"

            # Update local queue depth so concurrent batch submissions
            # see accurate counts and don't over-assign to one instance
            metrics.queue_depth += 1
            metrics.busy = True

            logger.info(f"[pool] Job {job.job_id} -> instance {job.instance_id} "
                        f"(prompt_id={job.prompt_id}, queue_depth={metrics.queue_depth})")
            return True

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            logger.error(f"[pool] Dispatch failed for {job.job_id}: {e}")
            return False

    # ------------------------------------------------------------------ #
    # Result tracking
    # ------------------------------------------------------------------ #

    async def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get current status of a pool job. Polls instance if still running."""
        job = self._jobs.get(job_id)
        if not job:
            return None

        # If still in-flight, poll the instance
        if job.status in ("queued", "running") and job.prompt_id and job.instance_id:
            await self._check_single_job(job)

        d = job.to_dict()
        d["prompt"] = job.prompt[:100]
        return d

    async def get_batch_status(self, batch_id: str) -> Optional[Dict]:
        """Get aggregated status for a batch."""
        batch = self._batches.get(batch_id)
        if not batch:
            return None

        # Refresh counts
        completed = 0
        failed = 0
        results = []
        for jid in batch.job_ids:
            job = self._jobs.get(jid)
            if not job:
                continue
            # Poll if still in-flight
            if job.status in ("queued", "running") and job.prompt_id:
                await self._check_single_job(job)
            if job.status == "completed":
                completed += 1
                results.append({"job_id": jid, "result": job.result})
            elif job.status == "failed":
                failed += 1

        batch.completed = completed
        batch.failed = failed

        if completed + failed >= batch.total:
            batch.status = "completed" if failed == 0 else "partial_failure"
            batch.completed_at = time.time()

        d = batch.to_dict()
        d["results"] = results
        return d

    async def get_pool_status(self) -> Dict:
        """Overview of the pool: instances, queues, active jobs (cached with 5s TTL)."""
        now = time.time()
        if self._pool_status_cache and (now - self._pool_status_cache_time) < self._POOL_STATUS_CACHE_TTL:
            return self._pool_status_cache

        # refresh_instances has its own stampede lock, safe to call from multiple coroutines
        await self.refresh_instances()

        instances = []
        for m in self._instances.values():
            instances.append({
                "instance_id": m.instance_id,
                "port": m.port,
                "gpu_device": m.gpu_device,
                "gpu_label": m.gpu_label,
                "vram_total_mb": m.vram_total_mb,
                "vram_mode": m.vram_mode,
                "status": m.status,
                "loaded_checkpoint": m.loaded_checkpoint,
                "queue_depth": m.queue_depth,
                "busy": m.busy,
            })

        active_jobs = [
            j.to_dict() for j in self._jobs.values()
            if j.status in ("pending", "provisioning", "queued", "running")
        ]

        active_batches = [
            b.to_dict() for b in self._batches.values()
            if b.status == "running"
        ]

        result = {
            "instances": instances,
            "running_count": sum(1 for m in self._instances.values() if m.status == "running"),
            "total_instances": len(self._instances),
            "active_jobs": len(active_jobs),
            "active_batches": len(active_batches),
            "jobs": active_jobs,
            "batches": active_batches,
        }
        self._pool_status_cache = result
        self._pool_status_cache_time = time.time()
        return result

    # ------------------------------------------------------------------ #
    # Background polling
    # ------------------------------------------------------------------ #

    async def _background_poll(self):
        """Periodically refresh instances and check job completions."""
        while True:
            try:
                await asyncio.sleep(self.POLL_INTERVAL)
                await self.refresh_instances(force=True)
                await self._check_completions()
                # Auto-scan media catalog when new images completed
                if self._needs_scan and self.media_catalog:
                    try:
                        self.media_catalog.scan_output_directory()
                        logger.debug("[pool] Auto-scanned media catalog after completions")
                    except Exception as scan_err:
                        logger.debug(f"[pool] Media scan failed: {scan_err}")
                    self._needs_scan = False
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"[pool] Poll error: {e}")

    async def _check_completions(self):
        """Check all in-flight jobs for completion."""
        for job in list(self._jobs.values()):
            if job.status in ("queued", "running") and job.prompt_id and job.instance_id:
                await self._check_single_job(job)

    async def _check_single_job(self, job: PoolJob):
        """Poll a single job's ComfyUI instance for completion."""
        metrics = self._instances.get(job.instance_id)
        if not metrics or metrics.status != "running":
            return

        try:
            client = await self._get_client()
            r = await client.get(
                f"http://127.0.0.1:{metrics.port}/history/{job.prompt_id}",
                timeout=10,
            )
            if r.status_code != 200:
                return

            history = r.json()
            entry = history.get(job.prompt_id)
            if not entry:
                # Still in queue or running
                job.status = "running"
                return

            # Check for execution errors
            status_info = entry.get("status", {})
            if status_info.get("status_str") == "error":
                job.status = "failed"
                job.error = str(status_info.get("messages", "Execution error"))
                job.completed_at = time.time()
                self._update_batch_counts(job)
                return

            # Extract images from outputs
            outputs = entry.get("outputs", {})
            images = []
            for _node_id, node_output in outputs.items():
                for img in node_output.get("images", []):
                    images.append({
                        "filename": img.get("filename"),
                        "subfolder": img.get("subfolder", ""),
                        "type": img.get("type", "output"),
                        "url": f"http://127.0.0.1:{metrics.port}/view?"
                               f"filename={img.get('filename')}"
                               f"&subfolder={img.get('subfolder', '')}"
                               f"&type={img.get('type', 'output')}",
                    })

            job.status = "completed"
            job.completed_at = time.time()
            job.result = {"images": images, "image_count": len(images)}
            self._update_batch_counts(job)
            self._needs_scan = True

            logger.info(f"[pool] Job {job.job_id} completed: {len(images)} image(s)")

        except Exception as e:
            logger.debug(f"[pool] Check job {job.job_id} failed: {e}")

    def _update_batch_counts(self, job: PoolJob):
        """Update batch completion counts when a job finishes."""
        if not job.batch_id:
            return
        batch = self._batches.get(job.batch_id)
        if not batch:
            return
        # Recount from source of truth
        completed = sum(1 for jid in batch.job_ids
                        if self._jobs.get(jid) and self._jobs[jid].status == "completed")
        failed = sum(1 for jid in batch.job_ids
                     if self._jobs.get(jid) and self._jobs[jid].status == "failed")
        batch.completed = completed
        batch.failed = failed
        if completed + failed >= batch.total:
            batch.status = "completed" if failed == 0 else "partial_failure"
            batch.completed_at = time.time()

    # ------------------------------------------------------------------ #
    # Housekeeping
    # ------------------------------------------------------------------ #

    def _prune_jobs(self):
        """Remove oldest completed/failed jobs when over MAX_JOBS."""
        if len(self._jobs) <= self.MAX_JOBS:
            return

        finished = [
            (j.completed_at or j.created_at, jid)
            for jid, j in self._jobs.items()
            if j.status in ("completed", "failed")
        ]
        finished.sort()

        to_remove = len(self._jobs) - self.MAX_JOBS
        for i in range(min(to_remove, len(finished))):
            del self._jobs[finished[i][1]]
