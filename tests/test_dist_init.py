"""Test vLLM distributed init on Windows with fake backend."""
import sys
import os

sys.path.insert(0, r"E:\AgentNate\envs\vllm\Lib\site-packages")
os.environ["CUDA_HOME"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
os.environ["VLLM_HOST_IP"] = "127.0.0.1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

print("Testing vLLM distributed init on Windows...")

# Test the helper function
from vllm.distributed.parallel_state import _get_cpu_backend
cpu_backend = _get_cpu_backend()
print(f"CPU backend: {cpu_backend}")

# Test distributed init
from vllm.distributed.parallel_state import init_distributed_environment
from vllm.utils.network_utils import get_distributed_init_method, get_ip, get_open_port

ip = get_ip()
port = get_open_port()
init_method = get_distributed_init_method(ip, port)
print(f"Init method: {init_method}")

try:
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=init_method,
        local_rank=0,
        backend="nccl",
    )
    import torch.distributed as dist
    print(f"Distributed init: OK, rank={dist.get_rank()}, world={dist.get_world_size()}")
except Exception as e:
    print(f"Distributed init FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("Done.")
