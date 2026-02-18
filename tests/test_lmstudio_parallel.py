"""Test parallel inference on ONE loaded model in LM Studio v4."""
import time
import threading
import lmstudio
from lmstudio import LlmLoadModelConfig

print("=" * 60)
print("LM STUDIO v4 - PARALLEL INFERENCE ON SINGLE MODEL")
print("=" * 60)

# Connect
api_host = lmstudio.Client.find_default_local_api_host()
print(f"\nSDK API: {api_host}")
client = lmstudio.Client(api_host=api_host)
time.sleep(2)

# Load one model
model_path = "lmstudio-community/Phi-4-mini-instruct-GGUF/Phi-4-mini-instruct-Q8_0.gguf"
print(f"Model: {model_path}")

# Unload existing
for m in list(client.llm.list_loaded()):
    client.llm.unload(m.identifier)

print("\nLoading single model instance...")
config = LlmLoadModelConfig(context_length=4096)
model = client.llm.load_new_instance(model_path, "single-instance", config=config, ttl=300)
print(f"[OK] Loaded: {model.identifier}")

# Test parallel requests to SAME model
print("\n" + "-" * 60)
print("TEST: Send 3 requests simultaneously to ONE model")
print("-" * 60)

results = {}
timings = {}

def request_task(task_id, prompt):
    start = time.time()
    try:
        result = model.respond(prompt, config={"max_tokens": 30})
        results[task_id] = result.content if hasattr(result, 'content') else str(result)
        timings[task_id] = time.time() - start
    except Exception as e:
        results[task_id] = f"ERROR: {e}"
        timings[task_id] = time.time() - start

# Launch 3 parallel requests
threads = []
prompts = [
    ("A", "Count from 1 to 5."),
    ("B", "Count from 10 to 15."),
    ("C", "Count from 100 to 105."),
]

print("\nStarting 3 parallel requests...")
overall_start = time.time()

for task_id, prompt in prompts:
    t = threading.Thread(target=request_task, args=(task_id, prompt))
    threads.append(t)
    t.start()
    print(f"  Started request {task_id}")

# Wait for all
for t in threads:
    t.join()

overall_time = time.time() - overall_start

print("\n" + "-" * 60)
print("RESULTS")
print("-" * 60)

for task_id, prompt in prompts:
    print(f"\nRequest {task_id} ({timings.get(task_id, 0):.2f}s):")
    print(f"  Prompt: {prompt}")
    print(f"  Response: {results.get(task_id, 'N/A')[:80]}")

print(f"\nTotal wall-clock time: {overall_time:.2f}s")

# Analysis
individual_sum = sum(timings.values())
print(f"Sum of individual times: {individual_sum:.2f}s")

if overall_time < individual_sum * 0.7:
    print("\n[PARALLEL] Requests were processed in PARALLEL!")
    print(f"  Speedup: {individual_sum / overall_time:.1f}x")
else:
    print("\n[SEQUENTIAL] Requests were processed SEQUENTIALLY.")
    print("  (Each request waited for previous to complete)")

# Cleanup
print("\n" + "-" * 60)
model.unload()
client.close()
print("Done")
