"""Test gloo distributed backend on Windows."""
import torch
import os
import sys
import tempfile

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"Gloo available: {torch.distributed.is_gloo_available()}")
print(f"NCCL available: {torch.distributed.is_nccl_available()}")

# Test 1: file:// init method
print("\n=== Test 1: Gloo with file:// ===")
tmp = os.path.join(tempfile.gettempdir(), "vllm_gloo_test")
if os.path.exists(tmp):
    os.remove(tmp)
try:
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file:///{tmp}",
        world_size=1,
        rank=0
    )
    print("SUCCESS")
    torch.distributed.destroy_process_group()
except Exception as e:
    print(f"FAILED: {e}")

# Test 2: env:// init method
print("\n=== Test 2: Gloo with env:// ===")
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
try:
    torch.distributed.init_process_group(
        backend="gloo",
        init_method="env://",
        world_size=1,
        rank=0
    )
    print("SUCCESS")
    torch.distributed.destroy_process_group()
except Exception as e:
    print(f"FAILED: {e}")

# Test 3: tcp:// init method
print("\n=== Test 3: Gloo with tcp://127.0.0.1 ===")
try:
    torch.distributed.init_process_group(
        backend="gloo",
        init_method="tcp://127.0.0.1:29501",
        world_size=1,
        rank=0
    )
    print("SUCCESS")
    torch.distributed.destroy_process_group()
except Exception as e:
    print(f"FAILED: {e}")

# Test 4: tcp:// with GLOO_SOCKET_IFNAME set
print("\n=== Test 4: Gloo with tcp:// + GLOO_SOCKET_IFNAME=127.0.0.1 ===")
os.environ["GLOO_SOCKET_IFNAME"] = "127.0.0.1"
try:
    torch.distributed.init_process_group(
        backend="gloo",
        init_method="tcp://127.0.0.1:29502",
        world_size=1,
        rank=0
    )
    print("SUCCESS")
    torch.distributed.destroy_process_group()
except Exception as e:
    print(f"FAILED: {e}")

print("\nDone.")
