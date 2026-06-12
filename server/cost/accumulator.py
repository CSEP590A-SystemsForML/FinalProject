"""
Cost accumulator.

A single place to accumulate the cost (and token usage) of every model call
that goes into resolving one problem: the router-selected attempts, the
solve->run->repair retries, and the escalation hop. The resolution loop adds
each `ModelCallResult` as it happens; the totals then flow into the
`SolveResponse` and on into the SQLite metrics, where `/metrics` and
`analysis_script` sum them per run.

Cost is computed with the project's custom `cost_function` (params + tokens),
not provider prices. A failed call contributes its token estimate but $0 cost,
and a costing error never breaks resolution (it counts as $0).
"""

from __future__ import annotations

from dataclasses import dataclass

from server.cost.cost_function import calculate_model_call_cost
from server.interfaces import ModelCallResult


@dataclass
class CostAccumulator:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    image_tokens: int = 0
    cost: float = 0.0
    num_calls: int = 0

    def add(self, call_result: ModelCallResult, image_tokens: int = 0) -> float:
        """
        Fold one model call into the running totals. Returns that call's cost.
        """

        self.prompt_tokens += call_result.prompt_tokens
        self.completion_tokens += call_result.completion_tokens
        self.image_tokens += image_tokens
        self.num_calls += 1

        call_cost = self._call_cost(call_result, image_tokens)
        self.cost += call_cost
        return call_cost

    @staticmethod
    def _call_cost(call_result: ModelCallResult, image_tokens: int) -> float:
        # A failed call still consumed (estimated) tokens but should not be billed.
        if call_result.error:
            return 0.0
        try:
            return calculate_model_call_cost(
                call_result.model_id,
                call_result.prompt_tokens,
                call_result.completion_tokens,
                image_tokens=image_tokens,
            )
        except Exception:
            # Cost must never break the resolution loop.
            return 0.0
