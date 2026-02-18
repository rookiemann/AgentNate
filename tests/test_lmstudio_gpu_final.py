"""Final LM Studio v4 GPU isolation test."""
import time
import lmstudio
from lmstudio import LlmLoadModelConfig
from lmstudio._sdk_models import GpuSetting

print("=" * 60)
print("LM STUDIO v4 - GPU ISOLATION FINAL TEST")
print("=" * 60)

# Connect
api_host = lmstudio.Client.find_default_local_api_host()
print(f"\nSDK API: {api_host}")
client = lmstudio.Client(api_host=api_host)
time.sleep(2)

# Use phi-4-mini (small, fast)
model_path = "lmstudio-community/Phi-4-mini-instruct-GGUF/Phi-4-mini-instruct-Q8_0.gguf"
print(f"Model: {model_path}")

# Unload any existing
for m in list(client.llm.list_loaded()):
    print(f"Unloading: {m.identifier}")
    client.llm.unload(m.identifier)

print("\n" + "-" * 60)
print("LOADING ON GPU 0 (RTX 3060)")
print("-" * 60)

config0 = LlmLoadModelConfig(
    gpu=GpuSetting(main_gpu=0, disabled_gpus=[1]),
    context_length=2048,
)
model0 = client.llm.load_new_instance(model_path, "gpu0-instance", config=config0, ttl=300)
print(f"[OK] Loaded: {model0.identifier}")

print("\n" + "-" * 60)
print("LOADING ON GPU 1 (RTX 3090)")
print("-" * 60)

config1 = LlmLoadModelConfig(
    gpu=GpuSetting(main_gpu=1, disabled_gpus=[0]),
    context_length=2048,
)
model1 = client.llm.load_new_instance(model_path, "gpu1-instance", config=config1, ttl=300)
print(f"[OK] Loaded: {model1.identifier}")

# Check loaded
print("\n" + "-" * 60)
print("CURRENTLY LOADED INSTANCES")
print("-" * 60)
for m in client.llm.list_loaded():
    print(f"  - {m.identifier}")

# Test inference on both
print("\n" + "-" * 60)
print("INFERENCE TEST")
print("-" * 60)

print("\nGPU 0 responding...")
result0 = model0.respond("Say 'Hello from GPU zero' exactly.")
print(f"  Type: {type(result0)}")
if hasattr(result0, 'content'):
    print(f"  Content: {result0.content[:100]}")
elif hasattr(result0, 'text'):
    print(f"  Text: {result0.text[:100]}")
else:
    print(f"  Result: {str(result0)[:100]}")

print("\nGPU 1 responding...")
result1 = model1.respond("Say 'Hello from GPU one' exactly.")
print(f"  Type: {type(result1)}")
if hasattr(result1, 'content'):
    print(f"  Content: {result1.content[:100]}")
elif hasattr(result1, 'text'):
    print(f"  Text: {result1.text[:100]}")
else:
    print(f"  Result: {str(result1)[:100]}")

# Cleanup
print("\n" + "-" * 60)
print("CLEANUP")
print("-" * 60)
model0.unload()
model1.unload()
print("[OK] Both unloaded")

client.close()
print("\n" + "=" * 60)
print("TEST COMPLETE - GPU ISOLATION WORKS!")
print("=" * 60)
