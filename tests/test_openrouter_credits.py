"""Check OpenRouter credit balance."""
import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import asyncio
from providers.openrouter_provider import OpenRouterProvider

API_KEY = "your-api-key-here"

async def check_credits():
    print("=== OpenRouter Credit Check ===")

    orr = OpenRouterProvider(api_key=API_KEY)

    # Check credits
    credits = await orr.get_credits()
    print(f"\nAccount: {credits.get('label', 'Unknown')}")

    if "error" not in credits:
        usage_total = credits.get("usage_total", 0) or 0
        usage_monthly = credits.get("usage_monthly", 0) or 0
        limit = credits.get("limit")

        print(f"Total usage: ${usage_total:.2f}")
        print(f"This month: ${usage_monthly:.2f}")

        if limit:
            remaining = credits.get("limit_remaining", 0) or 0
            print(f"Limit: ${limit:.2f}")
            print(f"Remaining: ${remaining:.2f}")
        else:
            print("Account type: Pay-as-you-go (no prepaid limit)")

        if credits.get("is_free_tier"):
            print("Tier: Free tier")
    else:
        print(f"Could not get info: {credits.get('error')}")

    # Count free vs paid models
    print("\n--- Model Summary ---")
    all_models = await orr.list_models()
    free_models = await orr.list_free_models()

    print(f"Total models: {len(all_models)}")
    print(f"Free models: {len(free_models)}")
    print(f"Paid models: {len(all_models) - len(free_models)}")

    # Show some popular free models
    print("\n--- Popular Free Models ---")
    popular_free = [
        "meta-llama/llama-3.3-70b-instruct:free",
        "google/gemma-2-9b-it:free",
        "mistralai/mistral-7b-instruct:free",
        "qwen/qwen-2-7b-instruct:free",
    ]

    for model_id in popular_free:
        exists = any(m["id"] == model_id for m in free_models)
        status = "Available" if exists else "Not found"
        print(f"  {model_id}: {status}")

    await orr.close()

if __name__ == "__main__":
    asyncio.run(check_credits())

