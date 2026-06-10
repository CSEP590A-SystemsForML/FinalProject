"""
Live end-to-end eval: real external models, no oracle mocking.

Unlike scripts/e2e_smoke.py (which mocks the solver with each problem's known
answer to test wiring deterministically), this drives the REAL resolution
pipeline against real external models via the server's `query_model`
(OpenRouter). Use it to measure how often the actual models solve the imported
problems, and at what cost.

What is and isn't mocked:
- Solver / judge: REAL external model calls (needs an API key + network).
- Router: a deterministic difficulty -> model mapping (same as the smoke test),
  because the real router needs the local vLLM model on a GPU (`run.sh`). Pass
  --model to force every problem to a single solver model instead.

Requirements:
- API_TOKEN (or OPENROUTER_API_KEY / LITELLM_API_KEY) in the environment.

Usage:
    export API_TOKEN=sk-or-...
    python scripts/live_eval.py --domain math --per-domain 5
    python scripts/live_eval.py --domain code --per-domain 10 --analyze
    python scripts/live_eval.py --domain math --per-domain 5 --model openai/gpt-oss-120b:free
    python scripts/live_eval.py --domain math --per-domain 5 --optimizations caveman

Results persist to a SQLite DB (printed at the end) so you can re-run analysis:
    python server/metrics/analysis_script.py    # if you pass --db-path .../metrics.db
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Reuse the smoke harness helpers so routing/sampling/analysis stay in one place.
import scripts.e2e_smoke as smoke


def run(
    domain: str | None,
    per_domain: int,
    run_id: str,
    db_path: Path,
    max_attempts: int,
    model_override: str | None,
    optimizations: list[str],
    analyze: bool,
) -> int:
    from server.utils import get_external_api_key

    if not get_external_api_key():
        raise SystemExit(
            "No API key found. Set API_TOKEN (or OPENROUTER_API_KEY / LITELLM_API_KEY) "
            "before running a live eval."
        )

    import server.server as srv
    from server.metrics import create_run

    db_path.parent.mkdir(parents=True, exist_ok=True)
    srv.get_db_path = lambda: db_path

    from fastapi.testclient import TestClient

    problems = smoke.sample_problems(domain, per_domain)
    if not problems:
        raise SystemExit("No problems sampled.")

    flags = {flag: True for flag in optimizations} or {"baseline": True}

    print(f"Live eval: run_id={run_id}  problems={len(problems)}  db={db_path}")
    print(f"Optimizations: {sorted(flags)}")
    if model_override:
        print(f"Forcing solver model: {model_override}")
    print("-" * 60)

    with TestClient(srv.app) as client:
        health = client.get("/health").json()
        if health.get("status") != "ok":
            raise SystemExit(f"Server unhealthy: {health}")

        create_run.upsert_optimization_run(
            db_path=db_path,
            run_id=run_id,
            label="Live eval",
            description="Real external-model end-to-end run.",
            flags=flags,
        )

        for i, problem in enumerate(problems, start=1):
            payload = smoke.build_payload(problem, run_id)
            payload["max_attempts"] = max_attempts
            if model_override:
                payload["model_id"] = model_override

            response = client.post("/solve", json=payload)
            response.raise_for_status()
            body = response.json()
            status = "OK " if body.get("solved") else "MISS"
            print(
                f"[{i:>3}/{len(problems)}] {status} "
                f"pid={problem['problem_id']} "
                f"diff={problem.get('difficulty')} cat={problem.get('category')} "
                f"model={body.get('model_id')} attempts={body.get('attempts')} "
                f"escalated={body.get('escalated')} cost={body.get('total_cost'):.6f}"
            )
            if body.get("error"):
                print(f"        error: {body['error']}")

        metrics = client.get("/metrics").json()

    print("\n=== LIVE RESULTS ===")
    print(f"problems:     {metrics['problem_solving_count']}")
    print(f"solved:       {metrics['solved_problems']}  (solve_rate={metrics['solve_rate']:.2%})")
    print(f"escalations:  {metrics['escalations']}")
    print(f"total cost:   {metrics['total_cost']:.6f}")
    print(f"model usage:  {metrics['model_usage']}")
    print(f"\nMetrics DB: {db_path}")

    if analyze:
        return smoke.run_analysis(db_path)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Live end-to-end eval against real external models.")
    parser.add_argument("--domain", default=None, help="Restrict to one domain (e.g. code, math).")
    parser.add_argument("--per-domain", type=int, default=5, help="Problems sampled per domain (live calls cost money).")
    parser.add_argument("--run-id", default=None, help="Run id (default: live_<timestamp>).")
    parser.add_argument("--db-path", type=Path, default=None, help="Metrics DB path (default: server/metrics/<run_id>.db).")
    parser.add_argument("--max-attempts", type=int, default=2, help="Max attempts before escalation.")
    parser.add_argument("--model", default=None, help="Force every problem to this solver model id.")
    parser.add_argument(
        "--optimizations",
        nargs="*",
        default=[],
        help="Optimization flags to enable for this run (e.g. caveman long_context_compression_lemma).",
    )
    parser.add_argument("--analyze", action="store_true", help="Run analysis over the populated DB afterward.")
    args = parser.parse_args()

    from server.metrics.create_run import OPTIMIZATION_FLAGS

    unknown = [f for f in args.optimizations if f not in OPTIMIZATION_FLAGS]
    if unknown:
        raise SystemExit(f"Unknown optimization(s): {unknown}. Valid: {OPTIMIZATION_FLAGS}")

    run_id = args.run_id or f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    db_path = args.db_path or (REPO_ROOT / "server" / "metrics" / f"{run_id}.db")

    raise SystemExit(
        run(
            domain=args.domain,
            per_domain=args.per_domain,
            run_id=run_id,
            db_path=db_path,
            max_attempts=args.max_attempts,
            model_override=args.model,
            optimizations=args.optimizations,
            analyze=args.analyze,
        )
    )


if __name__ == "__main__":
    main()
