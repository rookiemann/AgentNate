"""Test direct chat to a specific model in LM Studio v4."""
import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import asyncio
import aiohttp
import json

async def test_direct_chat():
    print("=== Testing Direct Chat to Specific Model ===")

    # Try chatting directly with phi-4 model
    model_id = "phi-4"
    print(f"Attempting to chat with: {model_id}")

    payload = {
        "model": model_id,  # Specify the exact model
        "messages": [
            {"role": "user", "content": "Say hello in exactly 5 words."}
        ],
        "max_tokens": 50,
        "temperature": 0.7,
        "stream": True,
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                "http://localhost:1234/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                print(f"Response status: {response.status}")

                if response.status != 200:
                    error_text = await response.text()
                    print(f"Error: {error_text}")
                    return

                print("Response: ", end="", flush=True)
                async for line in response.content:
                    line = line.decode().strip()
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            choices = chunk.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    print(content, end="", flush=True)
                        except json.JSONDecodeError:
                            pass

                print("\n\n=== Test Complete ===")

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_direct_chat())

