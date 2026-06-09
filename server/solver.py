"""
Cross-provider escalation engine.

Walks a "ladder" of models starting with the local vLLM and stepping up
through OpenRouter models on each failure:

    local Qwen3-4B  ->  openai/gpt-oss-20b:free
                    ->  openai/gpt-oss-120b:free
                    ->  moonshotai/kimi-k2.6:free   (final fallback)

A step "fails" if the model errors out OR if `validation.dispatch.verify`
says the answer is wrong. The first successful step short-circuits the
ladder.

All attempts are logged to stdout AND returned in the result so callers
(or the future /solve endpoint) can persist them.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import AsyncOpenAI

from server.cost.usage import cost_for_call
from server.openrouter_client import OpenRouterError, query_model
from server.validation.dispatch import verify

log = logging.getLogger(__name__)

# ID we use in metrics / logs for the local vLLM model. Not an OpenRouter
# model id - cost_for_call will treat it as unknown and return $0.
LOCAL_MODEL_ID = "local/vllm"

DEFAULT_LADDER: list[str] = [
    LOCAL_MODEL_ID,
    "openai/gpt-oss-20b:free",
    "openai/gpt-oss-120b:free",
    "moonshotai/kimi-k2.6:free",
]

SOLVE_SYSTEM_PROMPT = (
    "You are a problem solver. Read the user's problem and respond with "
    "ONLY the final answer. Do not show your work, do not add commentary, "
    "do not greet the user. If the problem asks for code, respond with a "
    "single fenced python code block and nothing else."
)


@dataclass
class Attempt:
    model_id: str
    text: str
    success: bool
    cost: float
    prompt_tokens: int
    completion_tokens: int
    error: Optional[str] = None


@dataclass
class SolveResult:
    problem_id: Any
    success: bool
    final_model_id: str
    escalated: bool
    total_cost: float
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def escalated_from(self) -> Optional[str]:
        return self.attempts[0].model_id if self.escalated and self.attempts else None


# --------------------------------------------------------------------------- #
# Local vLLM client (lazy)
# --------------------------------------------------------------------------- #


_local_client: AsyncOpenAI | None = None
_local_model_name: str | None = None


def _get_local_client() -> tuple[AsyncOpenAI, str]:
    """Return a process-wide AsyncOpenAI pointed at the local vLLM server."""
    global _local_client, _local_model_name
    if _local_client is None:
        base_url = os.environ.get("LOCAL_VLLM_BASE_URL", "http://localhost:8000/v1")
        _local_client = AsyncOpenAI(api_key="dummy", base_url=base_url, timeout=120.0)
        _local_model_name = os.environ.get("LOCAL_VLLM_MODEL", "Qwen/Qwen3-4B")
    assert _local_model_name is not None
    return _local_client, _local_model_name


async def _call_local(messages: list[dict[str, Any]], max_tokens: int) -> dict:
    client, model_name = _get_local_client()
    resp = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.2,
        max_completion_tokens=max_tokens,
        # Qwen3 emits <think>...</think> by default which can eat the whole
        # token budget on small prompts. Turn it off so the local rung
        # actually produces visible content.
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    choice = resp.choices[0]
    usage = resp.usage.model_dump() if resp.usage is not None else {
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
    }
    return {
        "text": choice.message.content or "",
        "finish_reason": choice.finish_reason or "",
        "usage": usage,
    }


async def _call_any_model(
    model_id: str, messages: list[dict[str, Any]], max_tokens: int
) -> dict:
    """Uniform wrapper: dispatches to local vLLM or OpenRouter based on model_id."""
    if model_id == LOCAL_MODEL_ID:
        return await _call_local(messages, max_tokens)
    return await query_model(
        model_id, messages, max_tokens=max_tokens, temperature=0.2,
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def _build_messages(problem: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SOLVE_SYSTEM_PROMPT},
        {"role": "user", "content": str(problem["problem"])},
    ]


def _budget_for(problem: dict[str, Any]) -> int:
    """Crude per-difficulty token cap so easy problems don't waste tokens."""
    diff = str(problem.get("difficulty", "")).strip().lower()
    return {
        "very_easy": 128,
        "easy": 512,
        "medium": 2048,
        "hard": 4096,
    }.get(diff, 1024)


async def solve_with_escalation(
    problem: dict[str, Any],
    ladder: Optional[list[str]] = None,
) -> SolveResult:
    """
    Try `problem` against each model in `ladder`. Stop at the first one
    whose answer passes validation.

    `problem` shape (mirrors local-inference/problems/problems.json):
        {
            "id":         int | str,
            "problem":    str,
            "answer":     str,
            "verify":     "match" | "code_tests" | "model_judge" | shorthand,
            "difficulty": "very_easy" | "easy" | "medium" | "hard",
            "tests":      list[str]   (only for code_tests)
        }
    """
    ladder = list(ladder or DEFAULT_LADDER)
    messages = _build_messages(problem)
    budget = _budget_for(problem)
    problem_id = problem.get("id", "?")

    attempts: list[Attempt] = []

    for idx, model_id in enumerate(ladder):
        log.info(
            "[solver] problem=%s attempt=%d model=%s budget=%d",
            problem_id, idx + 1, model_id, budget,
        )
        try:
            r = await _call_any_model(model_id, messages, max_tokens=budget)
        except OpenRouterError as e:
            log.warning(
                "[solver] problem=%s model=%s OpenRouter error: %s",
                problem_id, model_id, e,
            )
            attempts.append(Attempt(
                model_id=model_id, text="", success=False, cost=0.0,
                prompt_tokens=0, completion_tokens=0, error=repr(e),
            ))
            continue
        except Exception as e:  # noqa: BLE001
            log.warning(
                "[solver] problem=%s model=%s call failed: %s",
                problem_id, model_id, e,
            )
            attempts.append(Attempt(
                model_id=model_id, text="", success=False, cost=0.0,
                prompt_tokens=0, completion_tokens=0, error=repr(e),
            ))
            continue

        text = r.get("text", "") or ""
        usage = r.get("usage", {}) or {}
        cost = cost_for_call(model_id, usage)
        ok = await verify(problem, text)

        attempts.append(Attempt(
            model_id=model_id,
            text=text,
            success=ok,
            cost=cost,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        ))

        if ok:
            log.info(
                "[solver] problem=%s SOLVED by %s (cost=$%.6f)",
                problem_id, model_id, cost,
            )
            break

        if idx + 1 < len(ladder):
            log.info(
                "[ESCALATE] problem=%s %s -> %s (validation failed)",
                problem_id, model_id, ladder[idx + 1],
            )

    success = bool(attempts and attempts[-1].success)
    final_model_id = attempts[-1].model_id if attempts else (ladder[0] if ladder else "")
    escalated = len(attempts) > 1
    total_cost = sum(a.cost for a in attempts)

    return SolveResult(
        problem_id=problem_id,
        success=success,
        final_model_id=final_model_id,
        escalated=escalated,
        total_cost=total_cost,
        attempts=attempts,
    )
