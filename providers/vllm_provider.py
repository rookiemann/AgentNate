"""
vLLM Provider - High-throughput inference via subprocess servers.

Features:
- GPU assignment via CUDA_VISIBLE_DEVICES
- Subprocess isolation per model
- PagedAttention for efficient memory
- High concurrent request throughput (continuous batching)
- Optional multi-GPU pooling for extra throughput
- Broad model format support (HF, AWQ, GPTQ)
- OpenAI-compatible HTTP API
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
import uuid

try:
    import aiohttp
except ImportError:
    aiohttp = None

from .base import (
    BaseProvider,
    ProviderType,
    ModelInstance,
    ModelStatus,
    InferenceRequest,
    InferenceResponse,
)

logger = logging.getLogger(__name__)


def get_available_gpus() -> List[Dict[str, Any]]:
    """Detect available NVIDIA GPUs via nvidia-smi."""
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
                        gpus.append({
                            "index": int(parts[0]),
                            "name": parts[1],
                            "memory_total_mb": int(parts[2]),
                            "memory_free_mb": int(parts[3]),
                            "memory_used_mb": int(parts[4]),
                        })
    except Exception as e:
        logger.debug(f"GPU detection failed: {e}")
    return gpus


def estimate_model_vram(model_path: str, max_model_len: int = 4096) -> int:
    """Estimate VRAM needed for a model in MB."""
    try:
        if os.path.isdir(model_path):
            total_size = 0
            for f in os.listdir(model_path):
                if f.endswith(('.safetensors', '.bin', '.pt')):
                    total_size += os.path.getsize(os.path.join(model_path, f))
            file_size_mb = total_size / (1024 * 1024)
        elif os.path.isfile(model_path):
            file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        else:
            model_lower = model_path.lower()
            if '70b' in model_lower:
                file_size_mb = 35000
            elif '34b' in model_lower or '35b' in model_lower:
                file_size_mb = 17000
            elif '13b' in model_lower or '14b' in model_lower:
                file_size_mb = 7000
            elif '7b' in model_lower or '8b' in model_lower:
                file_size_mb = 4000
            elif '3b' in model_lower:
                file_size_mb = 2000
            elif '1b' in model_lower:
                file_size_mb = 1000
            else:
                file_size_mb = 8000

        context_overhead = (max_model_len / 1024) * 2
        estimated = (file_size_mb + context_overhead) * 1.25
        return int(estimated)
    except:
        return 8000


class GPUMemoryTracker:
    """Tracks GPU memory usage across servers."""

    def __init__(self):
        self._allocations: Dict[str, Dict[str, int]] = {}

    def allocate(self, gpu_index: int, instance_id: str, mb: int):
        key = str(gpu_index)
        if key not in self._allocations:
            self._allocations[key] = {}
        self._allocations[key][instance_id] = mb

    def release(self, instance_id: str):
        for gpu_allocs in self._allocations.values():
            if instance_id in gpu_allocs:
                del gpu_allocs[instance_id]

    def get_allocated(self, gpu_index: int) -> int:
        key = str(gpu_index)
        if key not in self._allocations:
            return 0
        return sum(self._allocations[key].values())

    def get_available(self, gpu_index: int) -> int:
        gpus = get_available_gpus()
        for gpu in gpus:
            if gpu["index"] == gpu_index:
                return gpu["memory_free_mb"] - self.get_allocated(gpu_index)
        return 0

    def can_fit(self, gpu_index: int, required_mb: int, buffer_mb: int = 500) -> bool:
        available = self.get_available(gpu_index)
        return available >= (required_mb + buffer_mb)


@dataclass
class ServerProcess:
    """Tracks a running vLLM server."""
    process: asyncio.subprocess.Process
    port: int
    gpu_index: int
    model_path: str
    instance_id: str
    started_at: float = field(default_factory=time.time)

    @property
    def is_running(self) -> bool:
        return self.process.returncode is None


class ServerPool:
    """Manages a pool of servers for the same model across multiple GPUs."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.servers: List[ServerProcess] = []
        self._next_idx = 0
        self._lock = asyncio.Lock()

    def add_server(self, server: ServerProcess):
        self.servers.append(server)

    def remove_server(self, instance_id: str):
        self.servers = [s for s in self.servers if s.instance_id != instance_id]

    async def get_next_server(self) -> Optional[ServerProcess]:
        async with self._lock:
            if not self.servers:
                return None
            for _ in range(len(self.servers)):
                server = self.servers[self._next_idx]
                self._next_idx = (self._next_idx + 1) % len(self.servers)
                if server.is_running:
                    return server
        return None

    @property
    def size(self) -> int:
        return len(self.servers)

    @property
    def running_count(self) -> int:
        return sum(1 for s in self.servers if s.is_running)


class VLLMProvider(BaseProvider):
    """
    Provider for vLLM inference via subprocess servers.

    Features:
    - PagedAttention for efficient memory
    - High concurrent request throughput
    - GPU assignment via CUDA_VISIBLE_DEVICES
    - Optional multi-GPU pooling
    - OpenAI-compatible HTTP API
    """

    def __init__(
        self,
        env_path: str = "envs/vllm",
        port_start: int = 8100,
        settings: Optional[Any] = None,
        models_directory: str = "",
    ):
        super().__init__(ProviderType.VLLM)
        self.env_path = env_path
        self._port_start = port_start
        self._next_port = port_start
        self.settings = settings
        self.models_directory = models_directory

        self._servers: Dict[str, ServerProcess] = {}
        self._pools: Dict[str, ServerPool] = {}
        self._memory_tracker = GPUMemoryTracker()
        self._lock = asyncio.Lock()
        self._stderr_tasks: Dict[str, asyncio.Task] = {}

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    def _get_env_python(self) -> str:
        """Get Python executable from isolated environment."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        if sys.platform == "win32":
            env_python = os.path.join(base_dir, self.env_path, "Scripts", "python.exe")
        else:
            env_python = os.path.join(base_dir, self.env_path, "bin", "python")

        if os.path.exists(env_python):
            return env_python

        logger.warning(f"vLLM env not found at {env_python}, using system Python")
        return sys.executable

    def _allocate_port(self) -> int:
        port = self._next_port
        self._next_port += 1
        return port

    def _select_gpu(self, model_path: str, required_mb: int) -> int:
        gpus = get_available_gpus()
        if not gpus:
            return -1

        best_gpu = None
        best_free = 0

        for gpu in gpus:
            available = gpu["memory_free_mb"] - self._memory_tracker.get_allocated(gpu["index"])
            if available >= required_mb + 500 and available > best_free:
                best_gpu = gpu["index"]
                best_free = available

        if best_gpu is not None:
            return best_gpu

        if gpus:
            best = max(gpus, key=lambda g: g["memory_free_mb"] - self._memory_tracker.get_allocated(g["index"]))
            return best["index"]

        return -1

    async def _spawn_server(
        self,
        model_path: str,
        port: int,
        gpu_index: int,
        instance_id: str,
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: Optional[float] = None,
        tensor_parallel_size: int = 1,
        enforce_eager: bool = False,
    ) -> ServerProcess:
        """Spawn a vLLM server subprocess."""
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        if gpu_index >= 0:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""

        # Windows-specific env vars required for vLLM
        if sys.platform == "win32":
            env["VLLM_HOST_IP"] = "127.0.0.1"
            env["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"

        python_exe = self._get_env_python()

        # Use our launcher wrapper (stubs uvloop on Windows)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        launcher = os.path.join(base_dir, "workers", "vllm_launcher.py")

        # vLLM server command
        cmd = [
            python_exe,
            launcher,
            "--model", model_path,
            "--port", str(port),
            "--host", "127.0.0.1",
        ]

        # Add optional parameters
        if max_model_len:
            cmd.extend(["--max-model-len", str(max_model_len)])

        if gpu_memory_utilization:
            cmd.extend(["--gpu-memory-utilization", str(gpu_memory_utilization)])
        else:
            default_util = self.settings.get("providers.vllm.default_gpu_memory_utilization", 0.9) if self.settings else 0.9
            if default_util:
                cmd.extend(["--gpu-memory-utilization", str(default_util)])

        if tensor_parallel_size > 1:
            cmd.extend(["--tensor-parallel-size", str(tensor_parallel_size)])

        if enforce_eager:
            cmd.append("--enforce-eager")
        elif self.settings and self.settings.get("providers.vllm.enforce_eager", False):
            cmd.append("--enforce-eager")
        elif sys.platform == "win32":
            # CUDA graphs require Triton (not available on Windows)
            cmd.append("--enforce-eager")

        logger.info(f"Spawning vLLM server: {' '.join(cmd)}")
        logger.info(f"GPU assignment: CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', 'not set')}")

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to spawn vLLM server: {e}")

        server = ServerProcess(
            process=process,
            port=port,
            gpu_index=gpu_index,
            model_path=model_path,
            instance_id=instance_id,
        )

        task = asyncio.create_task(self._read_stderr(server))
        self._stderr_tasks[instance_id] = task

        return server

    async def _read_stderr(self, server: ServerProcess):
        """Read and log stderr from server process."""
        try:
            while server.is_running:
                if server.process.stderr:
                    line = await server.process.stderr.readline()
                    if line:
                        text = line.decode().strip()
                        if text:
                            logger.debug(f"[vLLM:{server.port}] {text}")
                    else:
                        break
                else:
                    break
        except Exception as e:
            logger.debug(f"Stderr reader error: {e}")

    async def _wait_for_ready(self, server: ServerProcess, timeout: float = 600.0) -> bool:
        """Wait for server to become ready."""
        if aiohttp is None:
            raise ImportError("aiohttp is required for vLLM provider")

        start = time.time()
        last_log = start

        while time.time() - start < timeout:
            if not server.is_running:
                raise RuntimeError(f"vLLM server died during startup (exit code: {server.process.returncode})")

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://127.0.0.1:{server.port}/health",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as resp:
                        if resp.status == 200:
                            logger.info(f"vLLM server ready on port {server.port}")
                            return True
            except Exception:
                pass

            if time.time() - last_log >= 30:
                elapsed = int(time.time() - start)
                logger.info(f"Waiting for vLLM server... ({elapsed}s elapsed)")
                last_log = time.time()

            await asyncio.sleep(2.0)

        raise TimeoutError(f"vLLM server failed to start within {timeout}s")

    async def _stop_server(self, server: ServerProcess):
        """Stop server gracefully, then forcefully if needed."""
        if not server.is_running:
            return

        logger.info(f"Stopping vLLM server on port {server.port}")

        server.process.terminate()
        try:
            await asyncio.wait_for(server.process.wait(), timeout=10.0)
            logger.info(f"vLLM server stopped gracefully")
        except asyncio.TimeoutError:
            logger.warning(f"Force killing vLLM server")
            server.process.kill()
            await server.process.wait()

        if server.instance_id in self._stderr_tasks:
            self._stderr_tasks[server.instance_id].cancel()
            del self._stderr_tasks[server.instance_id]

    async def load_model(
        self,
        model_identifier: str,
        **kwargs
    ) -> ModelInstance:
        """
        Load a model by spawning a vLLM server.

        Args:
            model_identifier: HuggingFace model ID or local path
            gpu_index: GPU to use (-1 for CPU, None for auto)
            max_model_len: Maximum sequence length (None for model default)
            gpu_memory_utilization: GPU memory fraction (0.0-1.0)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            enforce_eager: Disable CUDA graphs for debugging
        """
        async with self._lock:
            instance_id = str(uuid.uuid4())

            gpu_index = kwargs.get("gpu_index")
            max_model_len = kwargs.get("max_model_len") or kwargs.get("n_ctx")
            gpu_memory_utilization = kwargs.get("gpu_memory_utilization")
            tensor_parallel_size = kwargs.get("tensor_parallel_size", 1)
            enforce_eager = kwargs.get("enforce_eager", False)

            if max_model_len is None and self.settings:
                max_model_len = self.settings.get("providers.vllm.default_max_model_len")

            ctx = max_model_len or 4096
            estimated_mb = estimate_model_vram(model_identifier, ctx)

            if gpu_index is None:
                gpu_index = self._select_gpu(model_identifier, estimated_mb)

            logger.info(f"Loading model {model_identifier} on GPU {gpu_index} (estimated {estimated_mb}MB)")

            display_name = os.path.basename(model_identifier) if os.path.sep in model_identifier else model_identifier
            instance = ModelInstance(
                id=instance_id,
                provider_type=self.provider_type,
                model_identifier=model_identifier,
                status=ModelStatus.LOADING,
                display_name=f"vLLM: {display_name}",
                context_length=ctx,
                gpu_index=gpu_index,
                metadata={
                    "estimated_vram_mb": estimated_mb,
                    "provider": "vllm",
                }
            )
            self.instances[instance_id] = instance

            port = self._allocate_port()

            try:
                server = await self._spawn_server(
                    model_path=model_identifier,
                    port=port,
                    gpu_index=gpu_index,
                    instance_id=instance_id,
                    max_model_len=max_model_len,
                    gpu_memory_utilization=gpu_memory_utilization,
                    tensor_parallel_size=tensor_parallel_size,
                    enforce_eager=enforce_eager,
                )

                load_timeout = self.settings.get("providers.vllm.load_timeout", 600) if self.settings else 600
                await self._wait_for_ready(server, timeout=load_timeout)

                self._servers[instance_id] = server
                if gpu_index >= 0:
                    self._memory_tracker.allocate(gpu_index, instance_id, estimated_mb)

                if model_identifier not in self._pools:
                    self._pools[model_identifier] = ServerPool(model_identifier)
                self._pools[model_identifier].add_server(server)

                instance.status = ModelStatus.READY
                instance.metadata["port"] = port
                instance.metadata["server_pid"] = server.process.pid

                logger.info(f"Model loaded successfully: {instance_id}")
                return instance

            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                instance.status = ModelStatus.ERROR
                instance.metadata["error"] = str(e)

                if instance_id in self._servers:
                    await self._stop_server(self._servers[instance_id])
                    del self._servers[instance_id]

                raise

    async def unload_model(self, instance_id: str) -> bool:
        """Unload a model by stopping its server."""
        if instance_id not in self.instances:
            return False

        instance = self.instances[instance_id]
        logger.info(f"Unloading model: {instance_id}")

        if instance_id in self._servers:
            server = self._servers[instance_id]
            await self._stop_server(server)

            if server.model_path in self._pools:
                self._pools[server.model_path].remove_server(instance_id)
                if self._pools[server.model_path].size == 0:
                    del self._pools[server.model_path]

            del self._servers[instance_id]

        self._memory_tracker.release(instance_id)
        del self.instances[instance_id]

        return True

    async def cancel_load(self, pending_id: str) -> bool:
        """
        Cancel a model that is currently loading.

        Stops the vLLM server subprocess, releases memory, and cleans up the instance.
        """
        server = self._servers.get(pending_id)
        if server:
            logger.info(f"Cancelling load for {pending_id[:8]}, stopping vLLM server")
            await self._stop_server(server)

            # Clean up pool reference
            if server.model_path in self._pools:
                self._pools[server.model_path].remove_server(pending_id)
                if self._pools[server.model_path].size == 0:
                    del self._pools[server.model_path]

            del self._servers[pending_id]
            self._memory_tracker.release(pending_id)

            if pending_id in self.instances:
                del self.instances[pending_id]

            return True

        # Instance exists but no server (maybe failed before spawn)
        if pending_id in self.instances:
            del self.instances[pending_id]
            return True

        return False

    async def chat(
        self,
        instance_id: str,
        request: InferenceRequest
    ) -> AsyncIterator[InferenceResponse]:
        """Stream chat completions from vLLM server."""
        if aiohttp is None:
            yield InferenceResponse(request_id=request.request_id, error="aiohttp not installed")
            return

        if instance_id not in self._servers:
            yield InferenceResponse(request_id=request.request_id, error="Model not loaded")
            return

        server = self._servers[instance_id]
        instance = self.instances[instance_id]

        if not server.is_running:
            yield InferenceResponse(request_id=request.request_id, error="Server not running")
            return

        instance.touch()

        payload = {
            "model": instance.model_identifier,
            "messages": request.to_messages_dict(),
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": True,
        }

        if request.stop:
            payload["stop"] = request.stop
        if request.presence_penalty != 0.0:
            payload["presence_penalty"] = request.presence_penalty
        if request.frequency_penalty != 0.0:
            payload["frequency_penalty"] = request.frequency_penalty
        if request.top_k and request.top_k > 0:
            payload["top_k"] = request.top_k
        if request.repeat_penalty != 1.0:
            payload["repetition_penalty"] = request.repeat_penalty

        url = f"http://127.0.0.1:{server.port}/v1/chat/completions"
        start_time = time.time()
        first_token_time = None
        total_tokens = 0

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        yield InferenceResponse(
                            request_id=request.request_id,
                            error=f"Server error {resp.status}: {error_text}"
                        )
                        return

                    async for line in resp.content:
                        line = line.decode().strip()

                        if not line:
                            continue

                        if line.startswith("data: "):
                            data_str = line[6:]

                            if data_str == "[DONE]":
                                break

                            try:
                                data = json.loads(data_str)
                                choices = data.get("choices", [])

                                if choices:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content", "")

                                    if content:
                                        if first_token_time is None:
                                            first_token_time = time.time()

                                        total_tokens += 1
                                        yield InferenceResponse(
                                            request_id=request.request_id,
                                            text=content,
                                        )

                                    finish_reason = choices[0].get("finish_reason")
                                    if finish_reason:
                                        break

                            except json.JSONDecodeError:
                                continue

            total_time = time.time() - start_time
            ttft = first_token_time - start_time if first_token_time else None
            tps = total_tokens / total_time if total_time > 0 else 0

            yield InferenceResponse(
                request_id=request.request_id,
                done=True,
                time_to_first_token=ttft,
                total_time=total_time,
                tokens_per_second=tps,
                usage={"completion_tokens": total_tokens},
            )

        except asyncio.CancelledError:
            yield InferenceResponse(request_id=request.request_id, done=True)
        except Exception as e:
            logger.error(f"Chat error: {e}")
            yield InferenceResponse(request_id=request.request_id, error=str(e))

    async def list_models(self) -> List[Dict[str, Any]]:
        """Scan models directory for .gguf files (runs in executor to avoid blocking)."""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._list_models_sync)

    def _list_models_sync(self) -> List[Dict[str, Any]]:
        """Synchronous model listing (called from executor)."""
        models = []

        if not self.models_directory or not os.path.exists(self.models_directory):
            # Fall back to listing loaded instances
            for instance in self.instances.values():
                models.append({
                    "id": instance.id,
                    "name": instance.display_name,
                    "provider": "vllm",
                    "status": instance.status.value,
                    "gpu_index": instance.gpu_index,
                    "context_length": instance.context_length,
                })
            return models

        for root, _, files in os.walk(self.models_directory):
            for f in files:
                if f.lower().endswith(".gguf") and "mmproj" not in f.lower():
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, self.models_directory)

                    try:
                        size_bytes = os.path.getsize(full_path)
                    except Exception:
                        size_bytes = 0

                    models.append({
                        "id": full_path,
                        "name": f,
                        "path": full_path,
                        "relative_path": rel_path,
                        "provider": "vllm",
                        "size_bytes": size_bytes,
                        "context_length": None,
                    })

        return models

    async def health_check(self) -> Dict[str, Any]:
        """Check provider health."""
        python_exe = self._get_env_python()
        env_exists = os.path.exists(python_exe)
        running = sum(1 for s in self._servers.values() if s.is_running)

        return {
            "provider": "vllm",
            "status": "healthy" if env_exists else "env_missing",
            "env_path": self.env_path,
            "env_exists": env_exists,
            "loaded_count": len(self._servers),
            "running_count": running,
            "pool_count": len(self._pools),
            "servers": {
                sid: {
                    "port": s.port,
                    "gpu": s.gpu_index,
                    "running": s.is_running,
                    "model": s.model_path,
                }
                for sid, s in self._servers.items()
            }
        }

    async def get_status(self, instance_id: str) -> ModelStatus:
        """Get status of a specific instance."""
        instance = self.instances.get(instance_id)
        if not instance:
            return ModelStatus.UNLOADED

        server = self._servers.get(instance_id)
        if not server or not server.is_running:
            return ModelStatus.ERROR

        return instance.status

    async def create_pool(
        self,
        model_identifier: str,
        gpu_indices: List[int],
        **kwargs
    ) -> Tuple[ServerPool, List[ModelInstance]]:
        """Load same model on multiple GPUs for pooled throughput."""
        instances = []

        for gpu_idx in gpu_indices:
            instance = await self.load_model(
                model_identifier,
                gpu_index=gpu_idx,
                **kwargs
            )
            instances.append(instance)

        pool = self._pools.get(model_identifier)
        return pool, instances

    async def chat_pooled(
        self,
        model_identifier: str,
        request: InferenceRequest
    ) -> AsyncIterator[InferenceResponse]:
        """Chat using pooled servers with automatic load balancing."""
        pool = self._pools.get(model_identifier)
        if not pool:
            yield InferenceResponse(request_id=request.request_id, error="No pool for model")
            return

        server = await pool.get_next_server()
        if not server:
            yield InferenceResponse(request_id=request.request_id, error="No available servers in pool")
            return

        async for response in self.chat(server.instance_id, request):
            yield response

    async def close(self):
        """Clean up all servers."""
        logger.info("Closing vLLM provider...")

        for instance_id in list(self._servers.keys()):
            await self.unload_model(instance_id)

        for task in self._stderr_tasks.values():
            task.cancel()
        self._stderr_tasks.clear()

    async def get_gpu_info(self) -> List[Dict[str, Any]]:
        """Get GPU info with vLLM allocations."""
        gpus = get_available_gpus()

        for gpu in gpus:
            gpu["models_loaded"] = []
            gpu["allocated_mb"] = self._memory_tracker.get_allocated(gpu["index"])
            gpu["effective_free_mb"] = gpu["memory_free_mb"] - gpu["allocated_mb"]

            for instance_id, server in self._servers.items():
                if server.gpu_index == gpu["index"]:
                    instance = self.instances.get(instance_id)
                    gpu["models_loaded"].append({
                        "instance_id": instance_id,
                        "model": os.path.basename(server.model_path),
                        "port": server.port,
                        "display_name": instance.display_name if instance else "Unknown",
                    })

        return gpus
