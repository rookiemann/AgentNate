"""Test the inference worker directly to diagnose empty responses."""
import asyncio
import subprocess
import json
import sys
import os

async def test_worker():
    # Start worker
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    cmd = [
        r"E:\AgentNate\python\python.exe",
        r"E:\AgentNate\inference_worker.py"
    ]

    print("Starting worker...")
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    )

    async def send(data):
        line = json.dumps(data) + "\n"
        proc.stdin.write(line.encode())
        await proc.stdin.drain()
        print(f"SENT: {data.get('command', data)}")

    async def read_line(timeout=60):
        try:
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=timeout)
            return line.decode().strip() if line else None
        except asyncio.TimeoutError:
            return None

    async def read_stderr_bg():
        while True:
            line = await proc.stderr.readline()
            if not line:
                break
            print(f"STDERR: {line.decode().strip()}")

    # Start stderr reader
    asyncio.create_task(read_stderr_bg())

    # Wait for ready
    print("Waiting for ready...")
    line = await read_line(30)
    print(f"GOT: {line}")

    # Load model
    await send({
        "command": "load",
        "model_path": r"E:\LL STUDIO\TheBloke\TinyLlama-1.1B-Chat-v1.0-GGUF\tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
        "n_ctx": 2048,
        "n_gpu_layers": 99
    })

    print("Waiting for model load...")
    while True:
        line = await read_line(120)
        print(f"GOT: {line}")
        if not line:
            print("No more output")
            break
        try:
            data = json.loads(line)
            if data.get("status") == "loaded":
                print("Model loaded!")
                break
            if "error" in data:
                print(f"Load error: {data['error']}")
                return
        except:
            continue

    # Send chat
    await send({
        "command": "chat",
        "request_id": "test-123",
        "messages": [{"role": "user", "content": "Say hello!"}],
        "max_tokens": 20,
        "temperature": 0.7
    })

    print("\nWaiting for chat response...")
    while True:
        line = await read_line(30)
        print(f"GOT: {line}")
        if not line:
            print("No more output")
            break
        try:
            data = json.loads(line)
            if data.get("done"):
                print("Chat complete!")
                break
            if "error" in data:
                print(f"Chat error: {data['error']}")
                break
        except:
            continue

    # Exit
    await send({"command": "exit"})
    await proc.wait()
    print("Worker exited")

if __name__ == "__main__":
    asyncio.run(test_worker())
