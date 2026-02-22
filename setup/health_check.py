#!/usr/bin/env python3
import asyncio
import aiohttp
import sys
from typing import Dict, List

ENGINES = {
    "vllm": "http://localhost:8001/v1/completions",
    "sglang": "http://localhost:8002/generate",
    "llamacpp": "http://localhost:8003/completion",
    "lemonade": "http://localhost:8000/api/v1/completions",
}

async def check_engine(name: str, url: str, payload: dict) -> bool:
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            async with session.post(url, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # Check if logprobs are present
                    if "choices" in data and len(data["choices"]) > 0:
                        choice = data["choices"][0]
                        if "logprobs" in choice:
                            print(f"✓ {name}: OK (logprobs present)")
                            return True
                        else:
                            print(f"⚠ {name}: OK but no logprobs")
                            return False
                    else:
                        print(f"⚠ {name}: Unexpected response format")
                        return False
                else:
                    print(f"✗ {name}: HTTP {resp.status}")
                    return False
    except Exception as e:
        print(f"✗ {name}: {e}")
        return False

async def main():
    print("Health checking engines...")
    prompt = "Hello world"
    payloads = {
        "vllm": {
            "model": "test",
            "prompt": prompt,
            "max_tokens": 1,
            "logprobs": 5,
            "temperature": 0.0,
        },
        "sglang": {
            "text": prompt,
            "max_tokens": 1,
            "return_logprob": True,
            "temperature": 0.0,
        },
        "llamacpp": {
            "prompt": prompt,
            "n_predict": 1,
            "n_probs": 5,
            "temperature": 0.0,
        },
        "lemonade": {
            "model": "test",
            "prompt": prompt,
            "max_tokens": 1,
            "logprobs": 5,
            "temperature": 0.0,
            "stream": False,
        },
    }
    
    tasks = []
    for name, url in ENGINES.items():
        tasks.append(check_engine(name, url, payloads[name]))
    
    results = await asyncio.gather(*tasks)
    
    if all(results):
        print("All engines healthy.")
        sys.exit(0)
    else:
        print("Some engines failed.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())