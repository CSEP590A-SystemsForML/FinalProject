"""
Unit tests for the cost accumulator (server/cost/accumulator.py).

Dependency-free (plain asserts + __main__) so CI can run it without pytest.
Proves that per-call costs/tokens fold correctly into the per-problem totals
that flow into the SolveResponse and on into the metrics DB:

    python scripts/test_cost_accumulator.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from server.cost.accumulator import CostAccumulator  # noqa: E402
from server.cost.cost_function import calculate_model_call_cost  # noqa: E402
from server.interfaces import ModelCallResult  # noqa: E402

CHEAP = "openai/gpt-oss-20b:free"
STRONG = "nvidia/nemotron-3-ultra-550b-a55b:free"


def _call(model_id: str, prompt: int, completion: int, error: str | None = None) -> ModelCallResult:
    return ModelCallResult(
        text="x",
        prompt_tokens=prompt,
        completion_tokens=completion,
        model_id=model_id,
        error=error,
    )


def test_empty_accumulator_is_zero() -> None:
    acc = CostAccumulator()
    assert acc.cost == 0.0
    assert acc.prompt_tokens == 0
    assert acc.completion_tokens == 0
    assert acc.num_calls == 0


def test_add_sums_tokens_and_cost() -> None:
    acc = CostAccumulator()
    c1 = acc.add(_call(CHEAP, 1000, 100))
    c2 = acc.add(_call(STRONG, 2000, 300))

    assert c1 == calculate_model_call_cost(CHEAP, 1000, 100)
    assert c2 == calculate_model_call_cost(STRONG, 2000, 300)
    assert acc.prompt_tokens == 3000
    assert acc.completion_tokens == 400
    assert acc.num_calls == 2
    assert abs(acc.cost - (c1 + c2)) < 1e-12
    # Escalating to a far bigger model must cost strictly more for the same work.
    assert c2 > c1


def test_failed_call_counts_tokens_but_not_cost() -> None:
    acc = CostAccumulator()
    returned = acc.add(_call(CHEAP, 500, 0, error="boom"))
    assert returned == 0.0
    assert acc.cost == 0.0
    assert acc.prompt_tokens == 500
    assert acc.num_calls == 1


def test_image_tokens_add_a_premium() -> None:
    base = CostAccumulator()
    base.add(_call(CHEAP, 1000, 100))

    with_img = CostAccumulator()
    with_img.add(_call(CHEAP, 1000, 100), image_tokens=2000)

    assert with_img.image_tokens == 2000
    assert with_img.cost > base.cost


def test_unknown_model_does_not_raise() -> None:
    acc = CostAccumulator()
    # A bad model id must never break resolution: cost falls back to 0.
    cost = acc.add(_call("does/not-exist", 100, 100))
    assert cost == 0.0
    assert acc.prompt_tokens == 100


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\n{len(tests)} cost-accumulator tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
