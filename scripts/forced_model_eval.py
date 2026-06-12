"""
Router-vs-individual-model ablation driver.

Runs the SAME fixed, balanced problem set through:
  1. the adaptive router (granite -> capability ladder), and
  2. each lineup model PINNED as a single solver (no escalation, via SolveRequest.pin_model),

so individual-model accuracy/cost can be compared apples-to-apples against the router.

Every pass writes to its own run_id in the server's metrics DB:
  cmp_router, cmp_<model-shortname>.

Usage (on the VM, against a server that holds an API key):
  ROUTER_BASE_URL=http://127.0.0.1:7654/v1 \
  .venv/bin/python scripts/forced_model_eval.py \
      --server-url http://127.0.0.1:8011 --per-domain 8 --max-active 2
"""

import argparse
import asyncio
import sys
from pathlib import Path

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "local-inference"))
from utils import LogicManager, load_domain_problems  # noqa: E402

# Large reasoning models (e.g. nemotron-550b) can take minutes per problem, so the
# forced passes use a generous HTTP timeout rather than the driver default.
SOLVE_TIMEOUT_S = 900.0

LINEUP = [
    "openai/gpt-oss-20b:free",
    "openai/gpt-oss-120b:free",
    "qwen/qwen3-coder:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
]
DOMAINS = ["math", "reasoning", "factual", "code"]


def short_name(model_id: str) -> str:
    tail = model_id.split("/")[-1].replace(":free", "")
    return tail.replace("-", "_").replace(".", "_")


def select_problems(per_domain: int) -> pd.DataFrame:
    frames = []
    for dom in DOMAINS:
        df = load_domain_problems(dom)
        if "problem_id" not in df.columns:
            df["problem_id"] = range(len(df))
        df = df.sample(frac=1, random_state=42).reset_index(drop=True).head(per_domain).copy()
        df["__domain"] = dom
        frames.append(df)
    out = pd.concat(frames).reset_index(drop=True)
    print(f"Selected {len(out)} problems ({per_domain}/domain across {DOMAINS}).")
    return out


def _val(rec, key, default=None):
    v = rec.get(key, default)
    try:
        if pd.isna(v):
            return default
    except (TypeError, ValueError):
        pass
    return v


def build_payload(run_id: str, rec: dict, model_id: str, max_attempts: int, pin: bool) -> dict:
    return {
        "run_id": run_id,
        "problem_id": int(rec["problem_id"]),
        "problem": str(rec["problem"]),
        "answer": None if _val(rec, "answer") is None else str(_val(rec, "answer")),
        "verify": str(_val(rec, "verify", "match")),
        "difficulty": str(_val(rec, "difficulty", "unknown")),
        "category": _val(rec, "__domain"),
        "assert_cases": _val(rec, "assert_cases"),
        "source_url": _val(rec, "source_url"),
        "validator": _val(rec, "validator"),
        "model_id": model_id,
        "max_attempts": max_attempts,
        "pin_model": pin,
    }


async def run_router_pass(problems: pd.DataFrame, server_url: str, max_active: int, max_attempts: int):
    lm = LogicManager(
        prompt_type="cache",
        max_active=max_active,
        server_url=server_url,
        run_id="cmp_router",
        max_attempts=max_attempts,
    )
    sem = asyncio.Semaphore(max_active)

    async def one(rec):
        async with sem:
            try:
                await lm.handle_problem(rec)
            except Exception as e:  # noqa: BLE001
                print(f"[router] problem_id={rec.get('problem_id')} FAILED: {e!r}")

    print("=== ROUTER pass (run_id=cmp_router) ===")
    await asyncio.gather(*(one(r._asdict() if hasattr(r, "_asdict") else dict(r))
                           for _, r in problems.iterrows()))


async def run_forced_pass(problems: pd.DataFrame, model_id: str, server_url: str,
                          max_active: int, max_attempts: int):
    run_id = f"cmp_{short_name(model_id)}"
    server_url = server_url.rstrip("/")
    sem = asyncio.Semaphore(max_active)
    counters = {"solved": 0, "n": 0, "err": 0}

    async def one(client, rec):
        async with sem:
            payload = build_payload(run_id, rec, model_id, max_attempts, pin=True)
            try:
                resp = await client.post(f"{server_url}/solve", json=payload)
                resp.raise_for_status()
                data = resp.json()
                counters["n"] += 1
                counters["solved"] += int(bool(data.get("solved")))
                if data.get("error"):
                    counters["err"] += 1
            except Exception as e:  # noqa: BLE001
                counters["n"] += 1
                counters["err"] += 1
                print(f"[{run_id}] problem_id={rec.get('problem_id')} FAILED: {e!r}", flush=True)

    print(f"=== FORCED pass {model_id} (run_id={run_id}) ===", flush=True)
    async with httpx.AsyncClient(timeout=SOLVE_TIMEOUT_S) as client:
        await asyncio.gather(*(one(client, dict(r)) for _, r in problems.iterrows()))
    print(f"  -> {counters['solved']}/{counters['n']} solved, {counters['err']} errors", flush=True)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-url", default="http://127.0.0.1:8011")
    ap.add_argument("--per-domain", type=int, default=8)
    ap.add_argument("--max-active", type=int, default=2)
    ap.add_argument("--max-attempts", type=int, default=2)
    ap.add_argument("--skip-router", action="store_true")
    ap.add_argument("--models", default=None, help="Comma-separated subset of lineup model ids.")
    args = ap.parse_args()

    problems = select_problems(args.per_domain)
    models = args.models.split(",") if args.models else LINEUP

    if not args.skip_router:
        await run_router_pass(problems, args.server_url, args.max_active, args.max_attempts)
    for model_id in models:
        await run_forced_pass(problems, model_id, args.server_url, args.max_active, args.max_attempts)

    print("=== DONE forced_model_eval ===")


if __name__ == "__main__":
    asyncio.run(main())
