"""
Pings every model in configs/models.yaml with a trivial prompt and prints
latency, token usage, and estimated cost.

Run this BEFORE the resolution loop to catch:
  - bad/missing OPENROUTER_API_KEY
  - typos in model IDs
  - :free-tier outages for a specific provider

Usage:
    python3.12 -m tools.smoke_openrouter
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

# Import after load_dotenv so the client picks up the key.
from server.cost.usage import cost_for_call  # noqa: E402
from server.openrouter_client import OpenRouterError, query_model  # noqa: E402

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "models.yaml"


async def ping(model_id: str) -> None:
    t0 = time.perf_counter()
    try:
        r = await query_model(
            model_id,
            [{"role": "user", "content": "Reply with exactly the word: pong"}],
            max_tokens=8,
            temperature=0.0,
        )
        dt = time.perf_counter() - t0
        cost = cost_for_call(model_id, r["usage"])
        print(
            f"OK   {model_id:<40} {dt:5.2f}s  "
            f"in={r['usage'].get('prompt_tokens', 0):>4} "
            f"out={r['usage'].get('completion_tokens', 0):>4} "
            f"cost=${cost:.6f}  text={r['text']!r}"
        )
    except OpenRouterError as e:
        dt = time.perf_counter() - t0
        print(f"FAIL {model_id:<40} {dt:5.2f}s  {type(e).__name__}: {e}")
    except Exception as e:  # noqa: BLE001
        dt = time.perf_counter() - t0
        print(f"ERR  {model_id:<40} {dt:5.2f}s  {type(e).__name__}: {e}")


async def main() -> None:
    models = list(yaml.safe_load(CONFIG.read_text()).keys())
    if not models:
        print(f"No models in {CONFIG}")
        return
    print(f"Pinging {len(models)} model(s) from {CONFIG}\n")
    await asyncio.gather(*(ping(m) for m in models))


if __name__ == "__main__":
    asyncio.run(main())
