"""Direct test of chat inference to diagnose empty responses."""
import asyncio
import aiohttp
import json

BASE_URL = "http://localhost:8000"

async def test():
    async with aiohttp.ClientSession() as session:
        # Load model
        print("Loading model...")
        async with session.post(
            f"{BASE_URL}/api/tools/call",
            json={"tool": "load_model", "arguments": {
                "model_name": "tinyllama",
                "gpu_index": 0
            }},
            timeout=aiohttp.ClientTimeout(total=120)
        ) as resp:
            data = await resp.json()
            print(f"Load result: {data}")
            if not data.get("success"):
                return
            instance_id = data["instance_id"]

        # Direct chat
        print(f"\nChatting with instance {instance_id[:8]}...")
        async with session.post(
            f"{BASE_URL}/api/chat/completions",
            json={
                "instance_id": instance_id,
                "messages": [{"role": "user", "content": "Hello! Count to 5."}],
                "max_tokens": 100,
                "temperature": 0.7
            },
            timeout=aiohttp.ClientTimeout(total=60)
        ) as resp:
            print(f"Status: {resp.status}")
            raw = await resp.text()
            print(f"Raw response: {raw[:500]}")
            try:
                data = json.loads(raw)
                print(f"Parsed: {json.dumps(data, indent=2)}")
            except:
                pass

        # Unload
        await session.post(
            f"{BASE_URL}/api/tools/call",
            json={"tool": "unload_model", "arguments": {"instance_id": instance_id}}
        )
        print("\nModel unloaded")

if __name__ == "__main__":
    asyncio.run(test())
