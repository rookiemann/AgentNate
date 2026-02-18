"""Check attention backend availability."""
import sys
import os
sys.path.insert(0, r"E:\AgentNate\envs\vllm\Lib\site-packages")
os.environ["VLLM_HOST_IP"] = "127.0.0.1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Check vllm_flash_attn
try:
    from vllm_flash_attn import flash_attn_varlen_func
    print("vllm_flash_attn: OK")
except ImportError as e:
    print(f"vllm_flash_attn: {e}")

# Check flash_attn (standard package)
try:
    from flash_attn import flash_attn_varlen_func
    print("flash_attn (standard): OK")
except ImportError as e:
    print(f"flash_attn (standard): {e}")

# Check fa_utils
try:
    from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
    v = get_flash_attn_version()
    print(f"Flash attn version from fa_utils: {v}")
except Exception as e:
    print(f"fa_utils: {e}")

# Check flex attention
try:
    from torch.nn.attention.flex_attention import flex_attention
    print("flex_attention: OK")
except ImportError as e:
    print(f"flex_attention: {e}")
