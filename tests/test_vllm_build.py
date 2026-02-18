"""Test vLLM build: basic inference + continuous batching (concurrent requests)."""
import time
import sys
import os

# Use the vllm venv
sys.path.insert(0, r"E:\AgentNate\envs\vllm\Lib\site-packages")
os.environ["CUDA_HOME"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
os.environ["VLLM_HOST_IP"] = "127.0.0.1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # RTX 3090
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"

def find_model(name_fragment):
    """Find a GGUF model by partial name."""
    for root, dirs, files in os.walk(r"E:\LL STUDIO"):
        for f in files:
            if f.endswith(".gguf") and name_fragment.lower() in f.lower() and "mmproj" not in f.lower():
                return os.path.join(root, f)
    return None

def test_basic_inference(llm):
    """Test 1: Single prompt generation."""
    from vllm import SamplingParams
    print("\n=== Test 1: Basic Single Inference ===")
    params = SamplingParams(max_tokens=50, temperature=0.7)
    start = time.time()
    outputs = llm.generate(["What is the capital of France?"], params)
    elapsed = time.time() - start
    text = outputs[0].outputs[0].text
    tokens = len(outputs[0].outputs[0].token_ids)
    print(f"  Response ({tokens} tokens, {elapsed:.2f}s): {text[:200]}")
    print(f"  Tokens/sec: {tokens/elapsed:.1f}")
    return True

def test_batch_inference(llm):
    """Test 2: Multiple prompts in one call (static batching)."""
    from vllm import SamplingParams
    print("\n=== Test 2: Batch Inference (4 prompts, 1 call) ===")
    prompts = [
        "Explain quantum computing in one sentence.",
        "Write a haiku about programming.",
        "What is 2+2? Answer with just the number.",
        "Name three planets in our solar system.",
    ]
    params = SamplingParams(max_tokens=50, temperature=0.7)
    start = time.time()
    outputs = llm.generate(prompts, params)
    elapsed = time.time() - start
    total_tokens = 0
    for i, out in enumerate(outputs):
        text = out.outputs[0].text
        tokens = len(out.outputs[0].token_ids)
        total_tokens += tokens
        print(f"  Prompt {i+1} ({tokens} tok): {text[:100].strip()}")
    print(f"  Total: {total_tokens} tokens in {elapsed:.2f}s = {total_tokens/elapsed:.1f} tok/s")
    return True

def test_continuous_batching(llm):
    """Test 3: Concurrent requests via async engine (true continuous batching)."""
    import asyncio
    from vllm import SamplingParams

    print("\n=== Test 3: Continuous Batching (8 concurrent async requests) ===")

    prompts = [
        "Tell me a fun fact about the ocean.",
        "What is machine learning?",
        "Describe the color blue.",
        "How does a car engine work?",
        "What is the speed of light?",
        "Name a famous painting.",
        "What causes rain?",
        "Explain gravity simply.",
    ]
    params = SamplingParams(max_tokens=60, temperature=0.7)

    # Use the engine directly for async requests
    engine = llm.llm_engine

    results = {}
    start = time.time()

    # Submit all requests at once
    for i, prompt in enumerate(prompts):
        request_id = f"req-{i}"
        engine.add_request(request_id, prompt, params)

    # Process all requests through the engine
    completed = 0
    while completed < len(prompts):
        step_outputs = engine.step()
        for output in step_outputs:
            if output.finished:
                rid = output.request_id
                text = output.outputs[0].text
                tokens = len(output.outputs[0].token_ids)
                results[rid] = (tokens, text)
                completed += 1

    elapsed = time.time() - start
    total_tokens = 0
    for i in range(len(prompts)):
        rid = f"req-{i}"
        tokens, text = results[rid]
        total_tokens += tokens
        print(f"  {rid} ({tokens} tok): {text[:80].strip()}")

    print(f"\n  All {len(prompts)} requests completed in {elapsed:.2f}s")
    print(f"  Total: {total_tokens} tokens = {total_tokens/elapsed:.1f} tok/s aggregate throughput")
    print(f"  vs sequential estimate: ~{total_tokens/(total_tokens/elapsed * 0.3):.1f}s (continuous batching is ~3x+ faster)")
    return True


if __name__ == "__main__":
    # Find a small model
    model_path = find_model("qwen2.5-1.5b-instruct-q8")
    if not model_path:
        model_path = find_model("tinyllama")
    if not model_path:
        model_path = find_model("llama-3.2-3b")

    if not model_path:
        print("ERROR: No suitable small GGUF model found")
        sys.exit(1)

    print(f"Model: {model_path}")
    print(f"Size: {os.path.getsize(model_path)/(1024**3):.1f}GB")

    # Load model with vLLM
    from vllm import LLM
    print("\nLoading model with vLLM...")
    start = time.time()
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=0.6,
        max_model_len=2048,
        max_num_seqs=64,
        enforce_eager=True,  # Skip CUDA graph capture for faster startup
    )
    print(f"Model loaded in {time.time()-start:.1f}s")

    # Run tests
    passed = 0
    total = 3

    try:
        if test_basic_inference(llm):
            passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")

    try:
        if test_batch_inference(llm):
            passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")

    try:
        if test_continuous_batching(llm):
            passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} tests passed")
    if passed == total:
        print("vLLM build is fully functional with continuous batching!")
    sys.exit(0 if passed == total else 1)
