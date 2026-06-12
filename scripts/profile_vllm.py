"""
Basic vLLM throughput profiler for the local router model.

Cost is one axis of "profiling the agent"; serving throughput is the other.
This drives the running vLLM OpenAI-compatible endpoint (the router, e.g.
ibm-granite/granite-4.1-3b from run.sh) with a controlled synthetic workload and
reports the numbers that characterize serving performance:

  - output throughput   (generated tokens / sec, the headline number)
  - total throughput    ((prompt + generated) tokens / sec)
  - request throughput  (requests / sec)
  - latency             p50/p90/p95/p99/mean end-to-end seconds
  - TTFT                (time-to-first-token, streaming only)
  - TPOT                (mean time-per-output-token, streaming only)

It tags each run with the quantization config (DTYPE / QUANTIZE_KV_CACHE / ACCEL)
so bf16 vs fp8 (or TPU tpu_int8) and fp8-kv-cache on/off are directly comparable.
For an apples-to-apples decode measurement it sets ignore_eos so every request
generates exactly --max-tokens output tokens.

This profiles whatever model is ALREADY serving on --base-url; it does not start
or stop vLLM. Use scripts/profile_quant_sweep.sh to sweep quantization configs.

Usage:
    # profile the currently-running router
    python scripts/profile_vllm.py --num-requests 64 --concurrency 8 --max-tokens 256

    # tag the run with the active quant config and save JSON
    DTYPE=fp8 QUANTIZE_KV_CACHE=true ACCEL=tpu \\
        python scripts/profile_vllm.py --out /tmp/prof/fp8.json

Requirements: a reachable vLLM OpenAI endpoint (no API key needed; uses a dummy).
"""

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from openai import AsyncOpenAI


DEFAULT_BASE_URL = os.environ.get("ROUTER_BASE_URL", "http://127.0.0.1:7654/v1")

# A filler sentence ~= 10 tokens; repeated to hit an approximate prompt length.
# Exact prompt/completion token counts come from the server's usage accounting.
_FILLER = "The quick brown fox jumps over the lazy dog. "


def build_prompt(target_tokens: int) -> str:
    repeats = max(1, target_tokens // 10)
    return (
        "You are a benchmark workload. Ignore the content below and keep writing.\n"
        + (_FILLER * repeats)
    )


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile; pct in [0, 100]. Empty -> 0.0."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def summarize(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "mean": statistics.fmean(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


async def detect_model(base_url: str) -> str:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{base_url.rstrip('/')}/models")
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            raise RuntimeError(f"No models served at {base_url}")
        return data[0]["id"]


async def scrape_server_metrics(base_url: str) -> dict:
    """Best-effort scrape of vLLM's Prometheus /metrics for cross-checking.

    Returns a small dict of selected gauges/counters, or {} on any failure so
    profiling never depends on the metrics endpoint being present.
    """
    # /metrics lives at the server root, not under /v1.
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    wanted_prefixes = (
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:gpu_cache_usage_perc",
        "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
        "vllm:request_success_total",
    )
    out: dict[str, float] = {}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{root}/metrics")
            resp.raise_for_status()
        for line in resp.text.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            if not line.startswith(wanted_prefixes):
                continue
            name, _, value = line.partition(" ")
            try:
                out[name] = float(value)
            except ValueError:
                continue
    except Exception:
        return {}
    return out


async def one_request(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    max_tokens: int,
    stream: bool,
    ignore_eos: bool,
    sem: asyncio.Semaphore,
) -> dict:
    messages = [{"role": "user", "content": prompt}]
    extra_body = {"ignore_eos": ignore_eos}
    async with sem:
        start = time.perf_counter()
        ttft = None
        try:
            if stream:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                    max_tokens=max_tokens,
                    stream=True,
                    stream_options={"include_usage": True},
                    extra_body=extra_body,
                )
                usage = None
                async for chunk in resp:
                    if chunk.choices and chunk.choices[0].delta.content:
                        if ttft is None:
                            ttft = time.perf_counter() - start
                    if getattr(chunk, "usage", None):
                        usage = chunk.usage
                latency = time.perf_counter() - start
                prompt_toks = getattr(usage, "prompt_tokens", 0) or 0
                completion_toks = getattr(usage, "completion_tokens", 0) or 0
            else:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                    max_tokens=max_tokens,
                    stream=False,
                    extra_body=extra_body,
                )
                latency = time.perf_counter() - start
                usage = resp.usage
                prompt_toks = getattr(usage, "prompt_tokens", 0) or 0
                completion_toks = getattr(usage, "completion_tokens", 0) or 0
        except Exception as exc:  # noqa: BLE001 - record, don't crash the sweep
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    return {
        "ok": True,
        "latency": latency,
        "ttft": ttft,
        "prompt_tokens": prompt_toks,
        "completion_tokens": completion_toks,
    }


async def run_profile(args) -> dict:
    base_url = args.base_url
    model = args.model or await detect_model(base_url)
    prompt = build_prompt(args.prompt_tokens)
    client = AsyncOpenAI(api_key="dummy", base_url=base_url)
    sem = asyncio.Semaphore(args.concurrency)

    async def fire(n: int) -> list[dict]:
        return await asyncio.gather(
            *(
                one_request(
                    client, model, prompt, args.max_tokens, not args.no_stream, not args.allow_eos, sem
                )
                for _ in range(n)
            )
        )

    if args.warmup > 0:
        print(f"[profile] warmup: {args.warmup} request(s)...", flush=True)
        await fire(args.warmup)

    print(
        f"[profile] model={model} requests={args.num_requests} concurrency={args.concurrency} "
        f"max_tokens={args.max_tokens} stream={not args.no_stream}",
        flush=True,
    )
    wall_start = time.perf_counter()
    results = await fire(args.num_requests)
    wall_time = time.perf_counter() - wall_start

    ok = [r for r in results if r["ok"]]
    errors = [r for r in results if not r["ok"]]
    latencies = [r["latency"] for r in ok]
    ttfts = [r["ttft"] for r in ok if r["ttft"] is not None]
    prompt_total = sum(r["prompt_tokens"] for r in ok)
    completion_total = sum(r["completion_tokens"] for r in ok)

    # TPOT: mean per-request decode time per output token (excludes prefill/TTFT).
    tpots = []
    for r in ok:
        if r["ttft"] is not None and r["completion_tokens"] > 1:
            decode_time = r["latency"] - r["ttft"]
            tpots.append(decode_time / (r["completion_tokens"] - 1))

    server_metrics = await scrape_server_metrics(base_url)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "base_url": base_url,
        "config": {
            "label": args.label,
            "dtype": args.dtype,
            "kv_cache_fp8": args.kv_cache,
            "accel": args.accel,
            "max_num_seqs": args.max_num_seqs,
        },
        "workload": {
            "num_requests": args.num_requests,
            "concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "prompt_tokens_target": args.prompt_tokens,
            "ignore_eos": not args.allow_eos,
            "stream": not args.no_stream,
        },
        "results": {
            "wall_time_s": wall_time,
            "completed": len(ok),
            "errors": len(errors),
            "error_samples": [e["error"] for e in errors[:5]],
            "prompt_tokens_total": prompt_total,
            "output_tokens_total": completion_total,
            "output_throughput_tok_s": (completion_total / wall_time) if wall_time else 0.0,
            "total_throughput_tok_s": ((prompt_total + completion_total) / wall_time) if wall_time else 0.0,
            "request_throughput_req_s": (len(ok) / wall_time) if wall_time else 0.0,
            "latency_s": summarize(latencies),
            "ttft_s": summarize(ttfts),
            "tpot_s_mean": statistics.fmean(tpots) if tpots else 0.0,
        },
        "server_metrics": server_metrics,
    }


def print_summary(report: dict) -> None:
    cfg = report["config"]
    res = report["results"]
    lat = res["latency_s"]
    ttft = res["ttft_s"]
    print("\n=== vLLM THROUGHPUT ===")
    print(f"model:            {report['model']}")
    print(
        f"config:           label={cfg['label']} dtype={cfg['dtype']} "
        f"kv_cache_fp8={cfg['kv_cache_fp8']} accel={cfg['accel']} max_num_seqs={cfg['max_num_seqs']}"
    )
    print(
        f"workload:         {report['workload']['num_requests']} req @ conc "
        f"{report['workload']['concurrency']}, max_tokens={report['workload']['max_tokens']}, "
        f"stream={report['workload']['stream']}, ignore_eos={report['workload']['ignore_eos']}"
    )
    print(f"completed/errors: {res['completed']} / {res['errors']}")
    print(f"wall time:        {res['wall_time_s']:.2f} s")
    print(f"output tok/s:     {res['output_throughput_tok_s']:.1f}   (the headline number)")
    print(f"total  tok/s:     {res['total_throughput_tok_s']:.1f}")
    print(f"requests/s:       {res['request_throughput_req_s']:.2f}")
    print(f"output tokens:    {res['output_tokens_total']}  prompt tokens: {res['prompt_tokens_total']}")
    print(
        f"latency (s):      mean={lat['mean']:.3f} p50={lat['p50']:.3f} "
        f"p95={lat['p95']:.3f} p99={lat['p99']:.3f}"
    )
    if report["workload"]["stream"]:
        print(f"TTFT (s):         mean={ttft['mean']:.3f} p50={ttft['p50']:.3f} p95={ttft['p95']:.3f}")
        print(f"TPOT (s/tok):     mean={res['tpot_s_mean']:.4f}  (~{(1/res['tpot_s_mean']) if res['tpot_s_mean'] else 0:.1f} tok/s/req decode)")
    if res["error_samples"]:
        print(f"errors (sample):  {res['error_samples']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile vLLM serving throughput for the local router model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="vLLM OpenAI base URL (…/v1).")
    parser.add_argument("--model", default=None, help="Model id. Default: auto-detect from /v1/models.")
    parser.add_argument("--num-requests", type=int, default=64, help="Total requests to send (measured).")
    parser.add_argument("--concurrency", type=int, default=8, help="Max in-flight requests.")
    parser.add_argument("--max-tokens", type=int, default=256, help="Output tokens generated per request.")
    parser.add_argument("--prompt-tokens", type=int, default=512, help="Approx prompt length (tokens).")
    parser.add_argument("--warmup", type=int, default=4, help="Warmup requests (not measured).")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming (skips TTFT/TPOT).")
    parser.add_argument(
        "--allow-eos",
        action="store_true",
        help="Allow early stop on EOS. Default keeps ignore_eos on so every request emits exactly --max-tokens.",
    )
    # Config tags (default from the env that run.sh / the sweep set).
    parser.add_argument("--label", default=os.environ.get("PROFILE_LABEL"), help="Run label tag.")
    parser.add_argument("--dtype", default=os.environ.get("DTYPE", "bf16"), help="Weight dtype/quant tag.")
    parser.add_argument(
        "--kv-cache",
        default=os.environ.get("QUANTIZE_KV_CACHE", "true"),
        help="fp8 kv-cache tag (true/false).",
    )
    parser.add_argument("--accel", default=os.environ.get("ACCEL", "auto"), help="Accelerator tag.")
    parser.add_argument(
        "--max-num-seqs",
        default=os.environ.get("MAX_NUM_SEQS", ""),
        help="max_num_seqs tag (informational).",
    )
    parser.add_argument("--out", type=Path, default=None, help="Write the full report JSON here.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = asyncio.run(run_profile(args))
    except Exception as exc:  # noqa: BLE001
        print(f"[profile] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print_summary(report)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n[profile] wrote {args.out}")

    if report["results"]["completed"] == 0:
        print("[profile] no successful requests", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
