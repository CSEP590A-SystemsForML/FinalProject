"""
Behavioral tests for the two resolution strategies, run head-to-head on the same
SolveRequest with a stubbed model so they are deterministic and network-free:

    python scripts/test_routing_strategies.py

`query_model` is monkeypatched to return scripted answers per model id, so we can
assert *which* models each strategy calls and how it escalates:
- difficulty (legacy): router pick, then a single jump to the strongest model.
- confidence (ladder): start rung from difficulty/confidence, then climb ONE rung
  at a time, accepting on validation (ground truth) or self-confidence (no GT).
"""

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from server.interfaces import ModelCallResult, SolveRequest  # noqa: E402
from server.resolution import resolution  # noqa: E402
from server.resolution.ladder import model_ladder  # noqa: E402
from server.resolution.resolution import (  # noqa: E402
    STRONGEST_MODEL_ID,
    normalize_routing_strategy,
    solve_problem,
)

LADDER = model_ladder()
CHEAP, MID = LADDER[0], LADDER[1]


def _install_fake_model(script: dict) -> None:
    """Patch query_model to return scripted text per model id (text or callable)."""

    def fake_query_model(model_id, prompt_or_messages, *a, **kw):
        value = script.get(model_id, "")
        text = value(prompt_or_messages) if callable(value) else value
        return ModelCallResult(text=text, prompt_tokens=10, completion_tokens=5, model_id=model_id, error=None)

    resolution.query_model = fake_query_model


def _req(**kw) -> SolveRequest:
    base = dict(run_id="t", problem_id=1, problem="2+2?", model_id=CHEAP, max_attempts=1)
    base.update(kw)
    return SolveRequest(**base)


def test_strategy_normalizer() -> None:
    assert normalize_routing_strategy("ladder") == "confidence"
    assert normalize_routing_strategy("confidence") == "confidence"
    assert normalize_routing_strategy("difficulty") == "difficulty"
    assert normalize_routing_strategy("legacy") == "difficulty"
    assert normalize_routing_strategy(None) == "confidence"  # default
    assert normalize_routing_strategy("garbage") == "confidence"  # safe fallback
    assert normalize_routing_strategy(None, {"routing_strategy": "difficulty"}) == "difficulty"


def test_difficulty_jumps_straight_to_strongest() -> None:
    _install_fake_model({CHEAP: "wrong", STRONGEST_MODEL_ID: "4"})
    req = _req(verify="match", answer="4", routing_strategy="difficulty", max_attempts=2)
    resp = asyncio.run(solve_problem(req))
    assert resp.solved is True
    assert resp.escalated is True
    assert resp.model_id == STRONGEST_MODEL_ID  # single jump, not the next rung
    assert resp.attempts == 3  # 2 cheap attempts + 1 escalation


def test_confidence_climbs_one_rung() -> None:
    _install_fake_model({CHEAP: "wrong", MID: "4", STRONGEST_MODEL_ID: "4"})
    req = _req(verify="match", answer="4", routing_strategy="confidence",
               difficulty_pred="easy", confidence=0.9)
    resp = asyncio.run(solve_problem(req))
    assert resp.solved is True
    assert resp.model_id == MID  # climbed exactly one rung, did NOT jump to strongest
    assert resp.escalated is True
    assert resp.attempts == 2


def test_confidence_low_confidence_raises_start_rung() -> None:
    # Low router confidence (<0.45) should start at MID and solve there immediately.
    _install_fake_model({CHEAP: "wrong", MID: "4"})
    req = _req(verify="match", answer="4", routing_strategy="confidence",
               difficulty_pred="easy", confidence=0.1)
    resp = asyncio.run(solve_problem(req))
    assert resp.solved is True
    assert resp.model_id == MID
    assert resp.escalated is False  # started at MID, so no escalation occurred


def test_confidence_no_ground_truth_uses_self_confidence() -> None:
    # No answer key: accept on self-reported confidence, escalate when too low.
    _install_fake_model({
        CHEAP: '{"answer": "maybe", "confidence": 0.2}',
        MID: '{"answer": "final", "confidence": 0.95}',
    })
    req = _req(verify="none", answer=None, routing_strategy="confidence",
               difficulty_pred="easy", confidence=0.9)
    resp = asyncio.run(solve_problem(req))
    assert resp.solved is True  # accepted by the confidence gate
    assert resp.model_id == MID
    assert resp.final_answer == "final"
    assert resp.escalated is True


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\n{len(tests)} routing-strategy tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
