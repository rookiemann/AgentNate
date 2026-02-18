"""
LlamaCpp Provider - Local llama.cpp inference via subprocess.

Features:
- GPU assignment via CUDA_VISIBLE_DEVICES
- Subprocess isolation per model (true multi-model)
- Multi-GPU support with per-worker assignment
- Multiple models per GPU (memory-aware)
- CPU-only fallback when no GPU available
- Streaming token output
- Model family auto-detection
- Request queuing per worker
- Clean loading/unloading
"""

import asyncio
import json
import os
import sys
import subprocess
from typing import AsyncIterator, Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
import time
import logging

from .base import (
    BaseProvider,
    ProviderType,
    ModelInstance,
    ModelStatus,
    InferenceRequest,
    InferenceResponse,
    ChatMessage,
)

# Setup logging
logger = logging.getLogger(__name__)


def is_vision_model(model_path: str) -> bool:
    """Detect if model supports vision based on name patterns."""
    vision_patterns = [
        'vision', 'llava', 'bakllava', 'moondream', 'cogvlm',
        'minicpm-v', 'qwen-vl', 'internvl', 'yi-vl', 'deepseek-vl',
        'llava-v', 'obsidian', 'pixtral', 'molmo'
    ]
    name_lower = os.path.basename(model_path).lower()
    return any(p in name_lower for p in vision_patterns)


def get_available_gpus() -> List[Dict[str, Any]]:
    """
    Detect available NVIDIA GPUs via nvidia-smi.

    Returns list of dicts with index, name, memory info.
    Returns empty list if no GPUs or nvidia-smi unavailable.
    """
    gpus = []

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )

        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 5:
                        memory_total = int(parts[2])
                        memory_free = int(parts[3])
                        memory_used = int(parts[4])
                        gpus.append({
                            "index": int(parts[0]),
                            "name": parts[1],
                            "memory_total_mb": memory_total,
                            "memory_free_mb": memory_free,
                            "memory_used_mb": memory_used,
                            "memory_mb": memory_total,  # Backwards compat
                            "display": f"GPU {parts[0]}: {parts[1]} ({memory_free} MB free / {memory_total} MB)",
                        })
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
        logger.debug(f"GPU detection failed: {e}")

    return gpus


def get_device_options() -> List[Dict[str, Any]]:
    """Get available device options for model loading."""
    options = [{"index": -1, "name": "CPU", "display": "CPU (no GPU)"}]
    options.extend(get_available_gpus())
    return options


def estimate_model_vram(model_path: str, n_ctx: int = 4096) -> int:
    """
    Estimate VRAM needed for a model in MB.

    This is a rough estimate based on file size and context length.
    Actual usage varies by quantization and model architecture.
    """
    try:
        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        # Model weights + KV cache overhead
        # Context adds roughly 2MB per 1K tokens for most models
        context_overhead = (n_ctx / 1024) * 2
        # Add 20% buffer for misc overhead
        estimated = (file_size_mb + context_overhead) * 1.2
        return int(estimated)
    except:
        return 0


@dataclass
class WorkerState:
    """Tracks state of a worker subprocess."""
    instance_id: str
    model_path: str
    gpu_index: Optional[int]
    n_ctx: int
    process: Optional[asyncio.subprocess.Process] = None
    loaded: bool = False
    busy: bool = False
    request_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    current_request_id: Optional[str] = None
    error_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)


class LlamaCppWorker:
    """
    Manages a single llama.cpp worker subprocess.

    Each worker runs in its own process with isolated GPU assignment.
    Uses JSON protocol over stdin/stdout for communication.
    """

    def __init__(
        self,
        instance_id: str,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = 99,
        gpu_index: Optional[int] = None,
        draft_model_path: Optional[str] = None,
        mmproj_path: Optional[str] = None,
        use_mmap: bool = True,
        flash_attn: bool = True,
        use_mlock: bool = False,
    ):
        self.instance_id = instance_id
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.gpu_index = gpu_index
        self.draft_model_path = draft_model_path
        self.mmproj_path = mmproj_path
        self.use_mmap = use_mmap
        self.flash_attn = flash_attn
        self.use_mlock = use_mlock

        self.process: Optional[asyncio.subprocess.Process] = None
        self._loaded = False
        self._busy = False
        self._lock = asyncio.Lock()
        self._read_task: Optional[asyncio.Task] = None
        self._response_queues: Dict[str, asyncio.Queue] = {}
        self._default_queue: asyncio.Queue = asyncio.Queue()

    async def start(self, timeout: float = 300.0) -> bool:
        """Start the worker subprocess and load the model."""
        # Build environment with GPU assignment
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        if self.gpu_index is not None:
            if self.gpu_index < 0:
                # CPU only - hide all GPUs
                env["CUDA_VISIBLE_DEVICES"] = ""
            else:
                # Specific GPU - CUDA will see it as device 0
                env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_index)
                env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

        # Find inference_worker.py
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        worker_path = os.path.join(base_dir, "inference_worker.py")

        if not os.path.exists(worker_path):
            raise FileNotFoundError(f"Worker script not found: {worker_path}")

        # Find Python executable
        python_exe = os.path.join(base_dir, "python", "python.exe")
        if not os.path.exists(python_exe):
            python_exe = sys.executable

        logger.info(f"Starting worker for {os.path.basename(self.model_path)} on GPU {self.gpu_index}")

        # Start subprocess
        try:
            self.process = await asyncio.create_subprocess_exec(
                python_exe, worker_path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to start worker process: {e}")

        # Start background reader for stderr (logging)
        asyncio.create_task(self._read_stderr())

        # Wait for ready signal
        try:
            ready = await asyncio.wait_for(self._wait_for_ready(), timeout=10.0)
            if not ready:
                raise RuntimeError("Worker did not send ready signal")
        except asyncio.TimeoutError:
            await self.stop()
            raise RuntimeError("Worker startup timeout")

        # Send load command
        load_cmd = {
            "command": "load",
            "model_path": self.model_path,
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "use_mmap": self.use_mmap,
            "flash_attn": self.flash_attn,
            "use_mlock": self.use_mlock,
            "is_cpu_only": self.gpu_index is not None and self.gpu_index < 0,
        }

        if self.mmproj_path:
            load_cmd["mmproj_path"] = self.mmproj_path
        if self.draft_model_path:
            load_cmd["draft_model_path"] = self.draft_model_path

        await self._send_command(load_cmd)

        # Wait for loaded response
        try:
            loaded = await asyncio.wait_for(self._wait_for_loaded(), timeout=timeout)
            if not loaded:
                raise RuntimeError("Model load failed")
            self._loaded = True
            logger.info(f"Worker loaded: {os.path.basename(self.model_path)}")
            return True
        except asyncio.TimeoutError:
            await self.stop()
            raise RuntimeError(f"Model load timeout after {timeout}s")

    async def _wait_for_ready(self) -> bool:
        """Wait for worker ready signal."""
        async for line in self._read_stdout_lines():
            try:
                data = json.loads(line)
                if data.get("status") == "ready":
                    return True
                if "error" in data:
                    raise RuntimeError(data["error"])
            except json.JSONDecodeError:
                continue
        return False

    async def _wait_for_loaded(self) -> bool:
        """Wait for model loaded signal."""
        async for line in self._read_stdout_lines():
            try:
                data = json.loads(line)
                if data.get("status") == "loaded":
                    return True
                if "error" in data:
                    raise RuntimeError(data["error"])
            except json.JSONDecodeError:
                continue
        return False

    async def _read_stderr(self):
        """Background task to read and log stderr."""
        if not self.process or not self.process.stderr:
            return
        try:
            while True:
                line = await self.process.stderr.readline()
                if not line:
                    break
                text = line.decode().strip()
                if text:
                    logger.debug(f"[Worker {self.instance_id[:8]}] {text}")
        except Exception:
            pass

    async def stop(self):
        """Stop the worker subprocess quickly."""
        if not self.process:
            return

        logger.info(f"Stopping worker {self.instance_id[:8]}...")

        try:
            # Send exit command
            if self.process.stdin and not self.process.stdin.is_closing():
                try:
                    await self._send_command({"command": "exit"})
                except Exception:
                    pass

            # Very short wait for graceful exit (VRAM is freed immediately,
            # we just need subprocess cleanup)
            try:
                await asyncio.wait_for(self.process.wait(), timeout=0.5)
            except asyncio.TimeoutError:
                # Terminate immediately - don't wait around
                logger.debug(f"Worker {self.instance_id[:8]} terminating...")
                self.process.terminate()
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=0.5)
                except asyncio.TimeoutError:
                    logger.debug(f"Worker {self.instance_id[:8]} killing...")
                    self.process.kill()
                    # Don't wait for kill - it's forceful
                    try:
                        await asyncio.wait_for(self.process.wait(), timeout=0.2)
                    except asyncio.TimeoutError:
                        pass  # Process will be cleaned up by OS

        except Exception as e:
            logger.error(f"Error stopping worker: {e}")
            if self.process and self.process.returncode is None:
                self.process.kill()

        self.process = None
        self._loaded = False

        # Clear CUDA cache in parent process
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        logger.info(f"Worker {self.instance_id[:8]} stopped")

    async def generate(
        self,
        request: InferenceRequest
    ) -> AsyncIterator[InferenceResponse]:
        """Generate streaming response with proper locking."""
        if not self._loaded or not self.process:
            logger.error(f"Worker generate called but not loaded: _loaded={self._loaded}, process={self.process is not None}")
            yield InferenceResponse(
                request_id=request.request_id,
                error="Worker not loaded"
            )
            return

        # Check if process is still running
        if self.process.returncode is not None:
            logger.error(f"Worker process has exited with code {self.process.returncode}")
            self._loaded = False
            yield InferenceResponse(
                request_id=request.request_id,
                error=f"Worker process crashed (exit code {self.process.returncode})"
            )
            return

        # Use lock to ensure only one request at a time per worker
        async with self._lock:
            if self._busy:
                yield InferenceResponse(
                    request_id=request.request_id,
                    error="Worker is busy (concurrent request)"
                )
                return

            self._busy = True
            start_time = time.time()
            logger.debug(f"Worker {self.instance_id[:8]} starting generation")

            try:
                # Build chat command
                chat_cmd = {
                    "command": "chat",
                    "request_id": request.request_id,
                    "messages": request.to_messages_dict(),
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "top_k": request.top_k,
                    "repeat_penalty": request.repeat_penalty,
                    "presence_penalty": request.presence_penalty,
                    "frequency_penalty": request.frequency_penalty,
                    "mirostat": request.mirostat,
                    "mirostat_tau": request.mirostat_tau,
                    "mirostat_eta": request.mirostat_eta,
                    "typical_p": request.typical_p,
                    "tfs_z": request.tfs_z,
                }

                if request.stop:
                    chat_cmd["stop"] = request.stop

                logger.debug(f"Worker {self.instance_id[:8]} sending chat command")
                await self._send_command(chat_cmd)
                logger.debug(f"Worker {self.instance_id[:8]} waiting for response...")

                # Stream responses
                got_response = False
                async for line in self._read_stdout_lines():
                    try:
                        data = json.loads(line)

                        # Filter by request_id if present
                        resp_id = data.get("request_id", request.request_id)
                        if resp_id != request.request_id:
                            continue

                        if "text" in data:
                            got_response = True
                            yield InferenceResponse(
                                request_id=request.request_id,
                                text=data["text"]
                            )
                        elif data.get("done"):
                            yield InferenceResponse(
                                request_id=request.request_id,
                                done=True,
                                total_time=time.time() - start_time,
                                usage=data.get("usage", {})
                            )
                            return
                        elif "error" in data:
                            error_msg = data["error"]
                            is_recoverable = data.get("recoverable", False)

                            # Smart error classification (from WritePost pattern)
                            # Context overflow is not fatal - model stays loaded
                            if is_recoverable or "context" in error_msg.lower() or "exceed" in error_msg.lower() or "requested tokens" in error_msg.lower():
                                logger.info(f"Recoverable error (context overflow): {error_msg}")
                                yield InferenceResponse(
                                    request_id=request.request_id,
                                    text="\n\n[Context too long - generation truncated]\n",
                                )
                                yield InferenceResponse(
                                    request_id=request.request_id,
                                    done=True,
                                    total_time=time.time() - start_time,
                                )
                            else:
                                # Truly fatal error
                                yield InferenceResponse(
                                    request_id=request.request_id,
                                    error=error_msg
                                )
                            return

                    except json.JSONDecodeError:
                        # Non-JSON output (shouldn't happen with new worker)
                        continue

                # If we get here without done signal, something went wrong
                if not got_response:
                    yield InferenceResponse(
                        request_id=request.request_id,
                        error="No response from worker"
                    )

            except Exception as e:
                logger.error(f"Generate error: {e}")
                yield InferenceResponse(
                    request_id=request.request_id,
                    error=str(e)
                )

            finally:
                self._busy = False

    async def _send_command(self, cmd: Dict[str, Any]):
        """Send JSON command to worker."""
        if self.process and self.process.stdin and not self.process.stdin.is_closing():
            try:
                line = json.dumps(cmd) + "\n"
                self.process.stdin.write(line.encode())
                await self.process.stdin.drain()
            except Exception as e:
                logger.error(f"Failed to send command: {e}")
                raise

    async def _read_stdout_lines(self) -> AsyncIterator[str]:
        """Read lines from worker stdout."""
        if not self.process or not self.process.stdout:
            return

        while True:
            try:
                line = await asyncio.wait_for(
                    self.process.stdout.readline(),
                    timeout=300.0  # 5 minute timeout for long generations
                )
                if not line:
                    break
                yield line.decode().strip()
            except asyncio.TimeoutError:
                logger.warning("Read timeout from worker")
                break
            except Exception as e:
                logger.error(f"Read error: {e}")
                break

    async def ping(self) -> bool:
        """Check if worker is responsive."""
        if not self.process or not self._loaded:
            return False

        try:
            await self._send_command({"command": "ping"})
            async for line in self._read_stdout_lines():
                try:
                    data = json.loads(line)
                    if data.get("status") == "pong":
                        return True
                except json.JSONDecodeError:
                    continue
                break
        except Exception:
            pass
        return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self.process is not None and self.process.returncode is None

    @property
    def is_busy(self) -> bool:
        return self._busy


class GPUMemoryTracker:
    """Tracks GPU memory usage across workers."""

    def __init__(self):
        self._allocations: Dict[str, Dict[str, int]] = {}  # gpu_index -> {instance_id: mb}

    def allocate(self, gpu_index: int, instance_id: str, mb: int):
        """Record memory allocation."""
        key = str(gpu_index)
        if key not in self._allocations:
            self._allocations[key] = {}
        self._allocations[key][instance_id] = mb

    def release(self, instance_id: str):
        """Release memory allocation."""
        for gpu_allocs in self._allocations.values():
            if instance_id in gpu_allocs:
                del gpu_allocs[instance_id]

    def get_allocated(self, gpu_index: int) -> int:
        """Get total allocated MB for a GPU."""
        key = str(gpu_index)
        if key not in self._allocations:
            return 0
        return sum(self._allocations[key].values())

    def get_available(self, gpu_index: int) -> int:
        """Get available MB for a GPU."""
        gpus = get_available_gpus()
        for gpu in gpus:
            if gpu["index"] == gpu_index:
                return gpu["memory_free_mb"] - self.get_allocated(gpu_index)
        return 0

    def can_fit(self, gpu_index: int, required_mb: int, buffer_mb: int = 500) -> bool:
        """Check if a model can fit on a GPU with buffer."""
        available = self.get_available(gpu_index)
        return available >= (required_mb + buffer_mb)


class ModelPool:
    """
    Manages a pool of workers for the same model across multiple devices.

    Enables round-robin load balancing for parallel inference.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.workers: List[LlamaCppWorker] = []
        self._next_idx = 0
        self._lock = asyncio.Lock()

    def add_worker(self, worker: LlamaCppWorker):
        """Add a worker to the pool."""
        self.workers.append(worker)

    async def get_available_worker(self) -> Optional[LlamaCppWorker]:
        """Get first available (non-busy) worker, or None."""
        async with self._lock:
            for worker in self.workers:
                if worker.is_loaded and not worker.is_busy:
                    return worker
        return None

    async def get_next_worker(self) -> Optional[LlamaCppWorker]:
        """Get next worker in round-robin fashion (may be busy - will queue)."""
        async with self._lock:
            if not self.workers:
                return None

            # Find next loaded worker
            for _ in range(len(self.workers)):
                worker = self.workers[self._next_idx]
                self._next_idx = (self._next_idx + 1) % len(self.workers)
                if worker.is_loaded:
                    return worker

        return None

    @property
    def size(self) -> int:
        return len(self.workers)

    @property
    def available_count(self) -> int:
        return sum(1 for w in self.workers if w.is_loaded and not w.is_busy)


class LlamaCppProvider(BaseProvider):
    """
    Provider for local llama.cpp inference via subprocess workers.

    Features:
    - GPU assignment via CUDA_VISIBLE_DEVICES
    - Multiple concurrent models (one subprocess each)
    - Multiple models per GPU (memory-aware)
    - Multi-GPU pools for same model (parallel inference)
    - CPU-only fallback
    - Streaming token output
    - Clean loading/unloading
    """

    def __init__(
        self,
        models_directory: str = "",
        settings: Optional[Any] = None,
    ):
        super().__init__(ProviderType.LLAMA_CPP)
        self.models_directory = models_directory
        self.settings = settings
        self._workers: Dict[str, LlamaCppWorker] = {}
        self._loading_workers: Dict[str, LlamaCppWorker] = {}  # Track workers during loading
        self._pools: Dict[str, ModelPool] = {}  # model_path -> pool
        self._memory_tracker = GPUMemoryTracker()
        self._lock = asyncio.Lock()

        # Set CUDA ordering for consistent GPU indexing
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    async def load_model(
        self,
        model_identifier: str,
        n_ctx: Optional[int] = None,
        n_gpu_layers: Optional[int] = None,
        gpu_index: Optional[int] = None,
        draft_model_path: Optional[str] = None,
        mmproj_path: Optional[str] = None,
        auto_gpu: bool = True,
        **kwargs
    ) -> ModelInstance:
        """
        Load a .gguf model via subprocess worker.

        Args:
            model_identifier: Full path to .gguf file
            n_ctx: Context length (default from settings)
            n_gpu_layers: GPU layers (default from settings)
            gpu_index: GPU to use (None=auto, -1=CPU, 0+=specific)
            draft_model_path: Path to draft model for speculative decoding
            mmproj_path: Path to mmproj for vision models
            auto_gpu: If True and gpu_index is None, pick GPU with most free memory

        Returns:
            ModelInstance
        """
        async with self._lock:
            # Get defaults from settings
            if n_ctx is None:
                n_ctx = (self.settings.get("providers.llama_cpp.default_n_ctx", 4096)
                         if self.settings else 4096)
            if n_gpu_layers is None:
                n_gpu_layers = (self.settings.get("providers.llama_cpp.default_n_gpu_layers", 99)
                               if self.settings else 99)

            use_mmap = (self.settings.get("providers.llama_cpp.use_mmap", True)
                        if self.settings else True)
            flash_attn = (self.settings.get("providers.llama_cpp.flash_attn", True)
                          if self.settings else True)
            use_mlock = (self.settings.get("providers.llama_cpp.use_mlock", False)
                         if self.settings else False)

            # Estimate memory requirement
            estimated_mb = estimate_model_vram(model_identifier, n_ctx)

            # Validate gpu_index if specified
            if gpu_index is not None and gpu_index >= 0:
                gpus = get_available_gpus()
                num_gpus = len(gpus)
                if num_gpus == 0:
                    logger.warning(f"No GPUs detected, falling back to CPU (requested GPU {gpu_index})")
                    gpu_index = -1
                    n_gpu_layers = 0
                elif gpu_index >= num_gpus:
                    raise ValueError(
                        f"GPU {gpu_index} requested but only {num_gpus} GPU(s) available (0-{num_gpus-1})"
                    )

            # Auto-select GPU if not specified
            if gpu_index is None and auto_gpu:
                gpus = get_available_gpus()
                if gpus:
                    # Find GPU with most free memory that can fit the model
                    best_gpu = None
                    best_free = 0
                    for gpu in gpus:
                        free = gpu["memory_free_mb"] - self._memory_tracker.get_allocated(gpu["index"])
                        if free > estimated_mb + 500 and free > best_free:
                            best_gpu = gpu["index"]
                            best_free = free

                    if best_gpu is not None:
                        gpu_index = best_gpu
                        logger.info(f"Auto-selected GPU {gpu_index} ({best_free} MB free)")
                    else:
                        logger.warning("No GPU has enough free memory, using CPU")
                        gpu_index = -1
                        n_gpu_layers = 0

            # Auto-detect vision model (mmproj)
            if mmproj_path is None:
                model_dir = os.path.dirname(model_identifier)
                if os.path.exists(model_dir):
                    for f in os.listdir(model_dir):
                        if "mmproj" in f.lower() and f.endswith(".gguf"):
                            mmproj_path = os.path.join(model_dir, f)
                            break

            # Determine if this is a vision model
            model_is_vision = is_vision_model(model_identifier)
            has_vision = mmproj_path is not None or model_is_vision

            # Create instance
            instance = ModelInstance(
                provider_type=self.provider_type,
                model_identifier=model_identifier,
                display_name=os.path.basename(model_identifier),
                context_length=n_ctx,
                status=ModelStatus.LOADING,
                gpu_index=gpu_index,
                metadata={
                    "n_gpu_layers": n_gpu_layers,
                    "draft_model": draft_model_path,
                    "mmproj": mmproj_path,
                    "is_vision": has_vision,
                    "has_vision": has_vision,
                    "estimated_vram_mb": estimated_mb,
                }
            )
            self.instances[instance.id] = instance

            # Create and start worker
            worker = LlamaCppWorker(
                instance_id=instance.id,
                model_path=model_identifier,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                gpu_index=gpu_index,
                draft_model_path=draft_model_path,
                mmproj_path=mmproj_path,
                use_mmap=use_mmap,
                flash_attn=flash_attn,
                use_mlock=use_mlock,
            )

            # Track worker during loading so cancel_load can stop it
            self._loading_workers[instance.id] = worker

            try:
                await worker.start()
                # Move from loading to ready
                self._loading_workers.pop(instance.id, None)
                self._workers[instance.id] = worker
                instance.status = ModelStatus.READY

                # Track memory allocation
                if gpu_index is not None and gpu_index >= 0:
                    self._memory_tracker.allocate(gpu_index, instance.id, estimated_mb)

            except asyncio.CancelledError:
                # Load was cancelled - clean up worker
                logger.info(f"Load cancelled for {instance.id[:8]}, stopping worker")
                self._loading_workers.pop(instance.id, None)
                await worker.stop()
                instance.status = ModelStatus.ERROR
                del self.instances[instance.id]
                raise

            except Exception as e:
                # Load failed - clean up worker
                self._loading_workers.pop(instance.id, None)
                await worker.stop()
                instance.status = ModelStatus.ERROR
                instance.metadata["error"] = str(e)
                del self.instances[instance.id]
                raise

            return instance

    async def unload_model(self, instance_id: str) -> bool:
        """Unload a model and stop its worker cleanly with CUDA cache cleanup."""
        async with self._lock:
            if instance_id in self._workers:
                worker = self._workers[instance_id]
                gpu_index = worker.gpu_index
                await worker.stop()
                del self._workers[instance_id]

                # Release memory tracking
                self._memory_tracker.release(instance_id)

                # Additional CUDA cache cleanup at provider level
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.debug(f"CUDA cache cleared after unloading from GPU {gpu_index}")
                except ImportError:
                    pass

            if instance_id in self.instances:
                del self.instances[instance_id]
                return True

            return False

    async def cancel_load(self, pending_id: str) -> bool:
        """
        Cancel a model that is currently loading.

        This stops the worker subprocess even if it's mid-load.
        """
        # Check if there's a loading worker for this pending_id
        # The pending_id might not match instance_id directly, so check all loading workers
        worker = self._loading_workers.get(pending_id)
        if worker:
            logger.info(f"Cancelling load for {pending_id[:8]}, stopping worker subprocess")
            self._loading_workers.pop(pending_id, None)
            await worker.stop()

            # Clean up instance if it exists
            if pending_id in self.instances:
                del self.instances[pending_id]

            return True

        # Also check workers dict in case it just finished loading
        if pending_id in self._workers:
            logger.info(f"Model {pending_id[:8]} already loaded, unloading instead")
            return await self.unload_model(pending_id)

        return False

    async def chat(
        self,
        instance_id: str,
        request: InferenceRequest
    ) -> AsyncIterator[InferenceResponse]:
        """Stream chat via worker with proper error handling."""
        logger.info(f"[LlamaCpp] chat called for instance {instance_id[:8]}...")

        worker = self._workers.get(instance_id)
        if not worker:
            logger.error(f"[LlamaCpp] Worker not found for {instance_id[:8]}")
            yield InferenceResponse(
                request_id=request.request_id,
                error="Worker not found"
            )
            return

        if not worker.is_loaded:
            logger.error(f"[LlamaCpp] Worker for {instance_id[:8]} is not loaded")
            yield InferenceResponse(
                request_id=request.request_id,
                error="Worker not loaded"
            )
            return

        instance = self.instances.get(instance_id)
        if instance:
            instance.status = ModelStatus.BUSY
            instance.touch()

        logger.info(f"[LlamaCpp] Starting generation for {instance_id[:8]}...")
        token_count = 0
        try:
            async for response in worker.generate(request):
                if response.text:
                    token_count += 1
                yield response
        except asyncio.CancelledError:
            logger.info(f"[LlamaCpp] Generation cancelled for request {request.request_id}")
            yield InferenceResponse(
                request_id=request.request_id,
                error="Generation cancelled"
            )
            return
        except Exception as e:
            logger.error(f"[LlamaCpp] Error during generation: {e}")
            yield InferenceResponse(
                request_id=request.request_id,
                error=str(e)
            )
        finally:
            logger.info(f"[LlamaCpp] Generation complete for {instance_id[:8]}, {token_count} tokens")
            if instance:
                instance.status = ModelStatus.READY

    async def list_models(self) -> List[Dict[str, Any]]:
        """Scan models directory for .gguf files (runs in executor to avoid blocking)."""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._list_models_sync)

    def _list_models_sync(self) -> List[Dict[str, Any]]:
        """Synchronous model listing (called from executor)."""
        models = []

        if not self.models_directory or not os.path.exists(self.models_directory):
            return models

        # NOTE: We intentionally do NOT read GGUF metadata here for performance.
        # Reading metadata from every file is very slow (hundreds of key-value pairs per file).
        # Context length is fetched on-demand when user selects a model in the dropdown.

        for root, _, files in os.walk(self.models_directory):
            for f in files:
                if f.lower().endswith(".gguf") and "mmproj" not in f.lower():
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, self.models_directory)

                    # Check for companion mmproj file
                    has_mmproj = False
                    mmproj_file = None
                    for other in files:
                        if "mmproj" in other.lower() and other.endswith(".gguf"):
                            has_mmproj = True
                            mmproj_file = os.path.join(root, other)
                            break

                    # has_vision is True if:
                    # 1. There's a companion mmproj file, OR
                    # 2. Model name matches vision patterns (llava, moondream, etc.)
                    model_is_vision = is_vision_model(full_path)
                    has_vision = has_mmproj or model_is_vision

                    try:
                        size_bytes = os.path.getsize(full_path)
                    except:
                        size_bytes = 0

                    models.append({
                        "id": full_path,
                        "name": f,
                        "path": full_path,
                        "relative_path": rel_path,
                        "provider": "llama_cpp",
                        "has_vision": has_vision,
                        "is_vision": has_vision,  # Backwards compat
                        "has_mmproj": has_mmproj,
                        "mmproj_path": mmproj_file,
                        "size_bytes": size_bytes,
                        # Context length fetched on-demand when model selected
                        "context_length": None,
                    })

        return models

    async def create_pool(
        self,
        model_path: str,
        gpu_indices: Optional[List[int]] = None,
        n_ctx: Optional[int] = None,
        n_gpu_layers: Optional[int] = None,
        **kwargs
    ) -> Tuple[ModelPool, List[ModelInstance]]:
        """
        Create a pool of workers for the same model across multiple GPUs.

        Args:
            model_path: Path to the .gguf model file
            gpu_indices: List of GPU indices to use (None = auto-detect all)
                        Use [-1] for CPU-only, or [0, 1, 2] for specific GPUs
            n_ctx: Context length
            n_gpu_layers: GPU layers per worker
            **kwargs: Additional args passed to load_model

        Returns:
            Tuple of (ModelPool, list of ModelInstances)
        """
        # Auto-detect GPUs if not specified
        if gpu_indices is None:
            gpus = get_available_gpus()
            if gpus:
                gpu_indices = [g["index"] for g in gpus]
            else:
                gpu_indices = [-1]  # CPU fallback

        # Create pool
        pool = ModelPool(model_path)
        instances = []

        for gpu_idx in gpu_indices:
            try:
                instance = await self.load_model(
                    model_identifier=model_path,
                    n_ctx=n_ctx,
                    n_gpu_layers=n_gpu_layers if gpu_idx >= 0 else 0,
                    gpu_index=gpu_idx,
                    auto_gpu=False,  # Explicit GPU selection
                    **kwargs
                )
                instances.append(instance)

                # Add worker to pool
                worker = self._workers.get(instance.id)
                if worker:
                    pool.add_worker(worker)
                    logger.info(f"Added worker to pool on GPU {gpu_idx}")

            except Exception as e:
                logger.error(f"Failed to load on GPU {gpu_idx}: {e}")
                continue

        # Register pool
        self._pools[model_path] = pool

        return pool, instances

    def get_pool(self, model_path: str) -> Optional[ModelPool]:
        """Get existing pool for a model path."""
        return self._pools.get(model_path)

    async def chat_pooled(
        self,
        model_path: str,
        request: InferenceRequest
    ) -> AsyncIterator[InferenceResponse]:
        """
        Chat using pooled workers with automatic load balancing.

        Finds first available worker in the pool for the model.
        """
        pool = self._pools.get(model_path)
        if not pool:
            yield InferenceResponse(
                request_id=request.request_id,
                error=f"No pool found for model: {model_path}"
            )
            return

        # Get available worker (prefer non-busy)
        worker = await pool.get_available_worker()
        if not worker:
            # All busy, use round-robin (will queue)
            worker = await pool.get_next_worker()

        if not worker:
            yield InferenceResponse(
                request_id=request.request_id,
                error="No workers available in pool"
            )
            return

        # Generate using the selected worker
        async for response in worker.generate(request):
            yield response

    async def health_check(self) -> Dict[str, Any]:
        """Check provider health including GPU availability."""
        models_exist = (
            self.models_directory and
            os.path.exists(self.models_directory)
        )

        gpus = get_available_gpus()

        # Check worker health
        workers_healthy = 0
        for worker in self._workers.values():
            if worker.is_loaded:
                workers_healthy += 1

        return {
            "provider": "llama_cpp",
            "status": "healthy" if models_exist else "no_models_directory",
            "models_directory": self.models_directory,
            "loaded_count": len(self._workers),
            "healthy_count": workers_healthy,
            "pool_count": len(self._pools),
            "gpu_count": len(gpus),
            "gpus": gpus,
            "has_cuda": len(gpus) > 0,
            "workers": {
                wid: {
                    "loaded": w.is_loaded,
                    "busy": w.is_busy,
                    "gpu": w.gpu_index,
                    "model": os.path.basename(w.model_path),
                }
                for wid, w in self._workers.items()
            },
            "pools": {
                path: {
                    "size": pool.size,
                    "available": pool.available_count,
                }
                for path, pool in self._pools.items()
            },
            "memory_tracking": {
                gpu_idx: self._memory_tracker.get_allocated(int(gpu_idx))
                for gpu_idx in self._memory_tracker._allocations.keys()
            }
        }

    async def get_status(self, instance_id: str) -> ModelStatus:
        """Get status of an instance."""
        instance = self.instances.get(instance_id)
        if not instance:
            return ModelStatus.UNLOADED

        worker = self._workers.get(instance_id)
        if not worker or not worker.is_loaded:
            return ModelStatus.ERROR

        return ModelStatus.BUSY if worker.is_busy else ModelStatus.READY

    async def get_gpu_info(self) -> List[Dict[str, Any]]:
        """Get detailed GPU info including per-GPU model list."""
        gpus = get_available_gpus()

        for gpu in gpus:
            gpu["models_loaded"] = []
            gpu["allocated_mb"] = self._memory_tracker.get_allocated(gpu["index"])
            gpu["effective_free_mb"] = gpu["memory_free_mb"] - gpu["allocated_mb"]

            for wid, worker in self._workers.items():
                if worker.gpu_index == gpu["index"]:
                    gpu["models_loaded"].append({
                        "instance_id": wid,
                        "model": os.path.basename(worker.model_path),
                        "busy": worker.is_busy,
                    })

        return gpus

    async def close(self):
        """Close all workers cleanly."""
        logger.info("Closing LlamaCpp provider...")

        # Close all workers
        for instance_id in list(self._workers.keys()):
            await self.unload_model(instance_id)

        # Clear pools
        self._pools.clear()

        logger.info("LlamaCpp provider closed")
