"""
End-to-end smoke test for the resolution pipeline.

Exercises the REAL FastAPI resolution server in-process (via Starlette's
TestClient) so a run touches every server-owned stage:

    /solve  ->  routing log  ->  resolution loop  ->  validation  ->  cost  ->  SQLite metrics  ->  /metrics

Only the two *external network* calls are mocked, because this environment has
no GPU (for the vLLM router) and no external API key (for the solver models):

- Router: a deterministic difficulty -> model mapping (no vLLM call).
- Solver/judge: an oracle that returns each problem's known-good answer
  (its `canonical_solution` or `answer`), so validation actually passes and we
  prove the match / tests / heuristic / judge paths and the cost + metrics
  wiring all work together.

This is intentionally lightweight and CI-friendly. A real run against a live
vLLM router + OpenRouter is described in the repo README.

Usage:
    python scripts/e2e_smoke.py                       # a few problems from every domain
    python scripts/e2e_smoke.py --domain math         # one domain only
    python scripts/e2e_smoke.py --per-domain 5        # N problems per domain
    python scripts/e2e_smoke.py --domain code --per-domain 200 --analyze
                                                      # run all code problems + analysis report
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOMAINS_DIR = REPO_ROOT / "local-inference" / "problems" / "domains"

# Difficulty -> model, mirroring the routing heuristic the real router learns.
MOCK_ROUTER = {
    "very_easy": "openai/gpt-oss-20b:free",
    "easy": "openai/gpt-oss-20b:free",
    "medium": "openai/gpt-oss-120b:free",
    "hard": "deepseek/deepseek-v4-flash:free",
    "very_hard": "moonshotai/kimi-k2.6:free",
}
DEFAULT_MODEL = "openai/gpt-oss-120b:free"


def mock_route(problem: dict) -> str:
    return MOCK_ROUTER.get(str(problem.get("difficulty")), DEFAULT_MODEL)


def oracle_answer(problem: dict) -> str:
    """The known-good answer the mock solver should return for this problem."""
    if problem.get("canonical_solution"):
        return str(problem["canonical_solution"])
    if problem.get("answer"):
        return str(problem["answer"])
    return str(problem.get("problem", ""))


def is_oracle_solvable(problem: dict) -> bool:
    """
    True when we can synthesize a known-good answer (a canonical solution or a
    reference answer). Problems with neither (e.g. open-ended keyword-heuristic
    prompts) can't be deterministically "solved" by the mock, so the smoke test
    skips them to keep the signal clean.
    """
    return bool(problem.get("canonical_solution") or problem.get("answer"))


def load_domain_files(domain: str | None) -> dict[str, list[dict]]:
    if not DOMAINS_DIR.exists():
        raise SystemExit(
            f"No domain datasets in {DOMAINS_DIR}. Run: python local-inference/problems/build.py"
        )
    if domain:
        path = DOMAINS_DIR / f"{domain}.json"
        if not path.exists():
            available = sorted(p.stem for p in DOMAINS_DIR.glob("*.json"))
            raise SystemExit(f"Unknown domain '{domain}'. Available: {available}")
        return {domain: json.loads(path.read_text())}
    return {p.stem: json.loads(p.read_text()) for p in sorted(DOMAINS_DIR.glob("*.json"))}


def sample_problems(domain: str | None, per_domain: int) -> list[dict]:
    sampled: list[dict] = []
    for _, problems in load_domain_files(domain).items():
        solvable = [p for p in problems if is_oracle_solvable(p)]
        sampled.extend(solvable[:per_domain])
    return sampled


def build_payload(problem: dict, run_id: str) -> dict:
    return {
        "run_id": run_id,
        "problem_id": int(problem["problem_id"]),
        "problem": str(problem["problem"]),
        "answer": problem.get("answer"),
        "verify": str(problem.get("verify", "match")),
        "difficulty": str(problem.get("difficulty", "unknown")),
        "category": problem.get("category"),
        "assert_cases": problem.get("assert_cases"),
        "source_url": problem.get("source_url"),
        "validator": problem.get("validator"),
        "model_id": mock_route(problem),
        "router_reasoning": f"mock router: difficulty={problem.get('difficulty')}",
        "max_attempts": 2,
    }


def install_mock_models(current: dict) -> None:
    """
    Patch the solver/judge model calls. `current["answer"]` is set before each
    /solve so the oracle returns the right thing for that problem.
    """
    from server.interfaces import ModelCallResult
    import server.resolution.resolution as resolution
    import server.utils as utils

    def fake_query_model(model_id, prompt_or_messages, *args, **kwargs):
        if isinstance(prompt_or_messages, str):
            content = prompt_or_messages
        else:
            content = " ".join(str(m.get("content", "")) for m in prompt_or_messages)

        # The LLM-judge validation path uses a grading prompt; answer it "yes".
        text = "yes" if "grading assistant" in content else current["answer"]
        return ModelCallResult(
            text=text,
            prompt_tokens=50,
            completion_tokens=20,
            model_id=model_id,
            error=None,
        )

    resolution.query_model = fake_query_model
    utils.query_model = fake_query_model


def run_analysis(db_path: Path) -> int:
    """
    Run the metrics analysis over the populated DB and print the key tables.

    Proves the analysis path works on a real run (not just that rows landed).
    Requires pandas; matplotlib is optional (plots are skipped if absent).
    """

    try:
        import sqlite3

        import server.metrics.analysis_script as az
    except ModuleNotFoundError as e:
        print(f"\n[analyze] skipped: missing dependency ({e.name}). pip install pandas to enable.")
        return 1

    conn = sqlite3.connect(db_path)
    try:
        sections = {
            "RUN SUMMARY": az.run_summary_with_optimizations(conn),
            "BASELINE COMPARISON": az.baseline_comparison(conn),
            "MODEL USAGE": az.model_usage_summary(conn),
            "COST BY DIFFICULTY": az.cost_by_difficulty(conn),
            "COST VS SOLVE RATE": az.cost_vs_solve_rate_frontier(conn),
            "TOKEN BREAKDOWN": az.token_breakdown(conn),
            "ROUTER CALIBRATION": az.router_calibration(conn),
        }
        print("\n=== ANALYSIS ===")
        for title, df in sections.items():
            print(f"\n--- {title} ---")
            print("No data." if df.empty else df.to_string(index=False))

        if sections["RUN SUMMARY"].empty:
            print("\n[analyze] FAILED: run summary is empty.")
            return 1
        print("\n[analyze] OK: analysis ran over the populated metrics DB.")
        return 0
    finally:
        conn.close()


def run(domain: str | None, per_domain: int, analyze: bool = False) -> int:
    sys.path.insert(0, str(REPO_ROOT))

    tmp_db = Path(tempfile.mkdtemp(prefix="e2e-metrics-")) / "metrics.db"

    import server.server as srv
    from server.metrics import create_run

    # Route all server DB access to a throwaway metrics DB.
    srv.get_db_path = lambda: tmp_db

    current = {"answer": ""}
    install_mock_models(current)

    from fastapi.testclient import TestClient

    problems = sample_problems(domain, per_domain)
    if not problems:
        raise SystemExit("No problems sampled.")

    run_id = "e2e_smoke"

    with TestClient(srv.app) as client:
        health = client.get("/health").json()
        if health.get("status") != "ok":
            raise SystemExit(f"Server unhealthy: {health}")

        create_run.upsert_optimization_run(
            db_path=tmp_db,
            run_id=run_id,
            label="E2E smoke",
            description="In-process mock E2E run.",
            flags={"baseline": True},
        )

        for problem in problems:
            current["answer"] = oracle_answer(problem)
            response = client.post("/solve", json=build_payload(problem, run_id))
            response.raise_for_status()

        metrics = client.get("/metrics").json()

    print("\n=== E2E smoke results ===")
    print(f"domain(s):           {domain or 'all'}")
    print(f"problems submitted:  {len(problems)}")
    print(f"routing rows:        {metrics['routing_count']}")
    print(f"problem_solving rows:{metrics['problem_solving_count']}")
    print(f"solved:              {metrics['solved_problems']}  (solve_rate={metrics['solve_rate']:.2%})")
    print(f"escalations:         {metrics['escalations']}")
    print(f"total cost:          {metrics['total_cost']:.6f}")
    print(f"long-ctx chars saved:{metrics.get('long_context_chars_saved', 0)}")
    print(f"model usage:         {metrics['model_usage']}")

    failures = []
    if metrics["routing_count"] != len(problems):
        failures.append("routing_count mismatch")
    if metrics["problem_solving_count"] != len(problems):
        failures.append("problem_solving_count mismatch")
    if metrics["solved_problems"] == 0:
        failures.append("no problems solved (validation/oracle wiring broken)")

    if failures:
        print("\nFAILED:", "; ".join(failures))
        return 1

    print("\nPASSED: pipeline wired end-to-end (routing -> resolution -> validation -> cost -> metrics).")

    if analyze:
        return run_analysis(tmp_db)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the E2E pipeline smoke test.")
    parser.add_argument("--domain", default=None, help="Restrict to one domain (default: all).")
    parser.add_argument("--per-domain", type=int, default=3, help="Problems sampled per domain.")
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="After the run, run metrics analysis over the populated DB (needs pandas).",
    )
    args = parser.parse_args()
    raise SystemExit(run(args.domain, args.per_domain, analyze=args.analyze))


if __name__ == "__main__":
    main()
