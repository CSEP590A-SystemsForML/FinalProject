"""
Manual test for local -> OpenRouter escalation.

Runs every problem in local-inference/problems/problems.json through
`server.solver.solve_with_escalation` and prints a per-problem summary
plus a final aggregate row.

Usage:
    # Full ladder (local vLLM first, then OpenRouter):
    python3.12 -m tools.test_escalation

    # Skip local vLLM entirely (handy if you don't have it running):
    SKIP_LOCAL=1 python3.12 -m tools.test_escalation

    # Run just one problem id:
    PROBLEM_ID=4 python3.12 -m tools.test_escalation
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)

from server.solver import (  # noqa: E402
    DEFAULT_LADDER,
    LOCAL_MODEL_ID,
    solve_with_escalation,
)

PROBLEMS_PATH = (
    Path(__file__).resolve().parent.parent
    / "local-inference" / "problems" / "problems.json"
)


async def main() -> int:
    problems = json.loads(PROBLEMS_PATH.read_text())

    if pid := os.environ.get("PROBLEM_ID"):
        problems = [p for p in problems if str(p["id"]) == pid]
        if not problems:
            print(f"No problem with id={pid}")
            return 2

    ladder = list(DEFAULT_LADDER)
    if os.environ.get("SKIP_LOCAL"):
        ladder = [m for m in ladder if m != LOCAL_MODEL_ID]
        print(f"[ladder] {ladder} (local skipped)\n")
    else:
        print(f"[ladder] {ladder}\n")

    results = []
    for problem in problems:
        print(f"\n>>> problem {problem['id']} ({problem.get('difficulty','?')}): "
              f"{problem['problem'][:80]}...")
        result = await solve_with_escalation(problem, ladder=ladder)
        results.append(result)

        chain = " -> ".join(a.model_id for a in result.attempts)
        print(
            f"<<< {'SUCCESS' if result.success else 'FAIL'} "
            f"final={result.final_model_id} escalated={result.escalated} "
            f"cost=${result.total_cost:.6f}"
        )
        print(f"    chain: {chain}")
        for a in result.attempts:
            preview = (a.text or "").replace("\n", "\\n")[:80]
            tag = "OK  " if a.success else "MISS"
            print(
                f"      [{tag}] {a.model_id:<32} "
                f"in={a.prompt_tokens:>4} out={a.completion_tokens:>4} "
                f"cost=${a.cost:.6f}  text={preview!r}"
            )

    # Aggregate
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    n = len(results)
    ok = sum(1 for r in results if r.success)
    esc = sum(1 for r in results if r.escalated)
    total = sum(r.total_cost for r in results)
    print(f"  problems:   {n}")
    print(f"  succeeded:  {ok}/{n}")
    print(f"  escalated:  {esc}/{n}")
    print(f"  total cost: ${total:.6f}")
    return 0 if ok == n else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
