"""
Unit tests for confidence-routed ladder escalation.

Dependency-free (plain asserts + __main__) so CI can run it without pytest:

    python scripts/test_escalation_ladder.py

Covers the prompt-only routing logic that decides which model rung to start on
and the no-ground-truth accept/give-up signal:
- model_ladder ordering (weakest -> strongest by params)
- difficulty -> starting rung
- confidence nudging the start rung up
- has_ground_truth detection
- defensive parsing of the solver's {"answer","confidence"} JSON
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from server.interfaces import SolveRequest  # noqa: E402
from server.resolution.ladder import (  # noqa: E402
    difficulty_start_index,
    ladder_index,
    model_ladder,
    resolve_start_index,
    strongest_model_id,
)
from server.resolution.resolution import (  # noqa: E402
    LOW_CONFIDENCE_THRESHOLD,
    has_ground_truth,
    parse_answer_confidence,
)

CHEAP = "openai/gpt-oss-20b:free"
STRONG = "nvidia/nemotron-3-ultra-550b-a55b:free"


def _req(**kw) -> SolveRequest:
    base = dict(run_id="t", problem_id=1, problem="2+2?", model_id=CHEAP)
    base.update(kw)
    return SolveRequest(**base)


def test_ladder_is_ordered_weak_to_strong() -> None:
    ladder = model_ladder()
    assert ladder[0] == CHEAP, ladder
    assert ladder[-1] == STRONG, ladder
    assert strongest_model_id() == STRONG
    assert ladder_index(CHEAP) == 0
    assert ladder_index(STRONG) == len(ladder) - 1
    # Unknown model defaults to the cheapest rung.
    assert ladder_index("nope/none:free") == 0


def test_difficulty_maps_to_rung() -> None:
    last = len(model_ladder()) - 1
    assert difficulty_start_index("very_easy") == 0
    assert difficulty_start_index("easy") == 0
    assert difficulty_start_index("very_hard") == last
    # Unknown / None difficulty starts cheap.
    assert difficulty_start_index(None) == 0
    assert difficulty_start_index("bogus") == 0
    # Monotonic non-decreasing across the ordered labels.
    order = ["very_easy", "easy", "medium", "hard", "very_hard"]
    idxs = [difficulty_start_index(d) for d in order]
    assert idxs == sorted(idxs), idxs


def test_low_confidence_bumps_start_rung() -> None:
    # Easy + confident -> cheapest rung.
    assert resolve_start_index(CHEAP, "easy", 0.9, LOW_CONFIDENCE_THRESHOLD) == 0
    # Easy but low confidence -> one rung higher.
    assert resolve_start_index(CHEAP, "easy", 0.1, LOW_CONFIDENCE_THRESHOLD) == 1
    # Hard takes the difficulty-implied rung even if the pick was cheap.
    hard = resolve_start_index(CHEAP, "hard", 0.9, LOW_CONFIDENCE_THRESHOLD)
    assert hard == difficulty_start_index("hard")
    # Never exceeds the top rung.
    top = len(model_ladder()) - 1
    assert resolve_start_index(STRONG, "very_hard", 0.0, LOW_CONFIDENCE_THRESHOLD) == top


def test_start_rung_takes_stronger_of_pick_and_difficulty() -> None:
    # Router picked a strong model but called it "easy": respect the stronger pick.
    idx = resolve_start_index(STRONG, "easy", 0.9, LOW_CONFIDENCE_THRESHOLD)
    assert idx == ladder_index(STRONG)


def test_has_ground_truth() -> None:
    assert has_ground_truth(_req(verify="match", answer="4"))
    assert not has_ground_truth(_req(verify="match", answer=None))
    assert has_ground_truth(_req(verify="tests", assert_cases="assert f(1)==1"))
    assert not has_ground_truth(_req(verify="tests", assert_cases=None))
    assert has_ground_truth(_req(verify="heuristic", validator="numeric_match"))
    # General/specialized use: only a prompt -> no ground truth.
    assert not has_ground_truth(_req(verify="none", answer=None))


def test_parse_answer_confidence() -> None:
    a, c = parse_answer_confidence('{"answer": "42", "confidence": 0.91}')
    assert a == "42" and abs(c - 0.91) < 1e-9
    # Single quotes tolerated.
    a, c = parse_answer_confidence("{'answer': 'foo', 'confidence': 0.3}")
    assert a == "foo" and abs(c - 0.3) < 1e-9
    # Confidence clamped to [0,1].
    _, c = parse_answer_confidence('{"answer": "x", "confidence": 1.7}')
    assert c == 1.0
    # No JSON, no confidence -> unknown (escalate).
    a, c = parse_answer_confidence("just prose, no json")
    assert a == "just prose, no json" and c is None
    # Confidence recoverable via regex even if JSON is malformed.
    a, c = parse_answer_confidence('answer: stuff "confidence": 0.55 trailing')
    assert abs(c - 0.55) < 1e-9
    # Empty input.
    assert parse_answer_confidence("") == ("", None)


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\n{len(tests)} escalation-ladder tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
