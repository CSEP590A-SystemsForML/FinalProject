"""
Validation helpers used by `server/validation/registry.py`.

Heads-up: `model_judge` is async because it goes through the OpenRouter
client. Callers in `registry.py` await it. Synchronous helpers
(`direct_match`, `run_code`) are unchanged so they can still be used in
non-async contexts.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def run_code(function: str, assert_cases: str) -> bool:
    """
    Executes the given Python function definition combined with assert_cases.
    Returns True if execution succeeds without raising any exceptions, and False otherwise.
    """
    if not function:
        return False
    try:
        env: dict = {}
        exec(f"{function}\n{assert_cases}", env)
        return True
    except Exception:
        return False


def direct_match(model_answer: str, correct_answer: str) -> bool:
    """
    Compares the model's answer directly with the correct answer.
    Returns True if they are identical after stripping leading/trailing whitespace.
    """
    if model_answer is None or correct_answer is None:
        return False
    return model_answer.strip() == correct_answer.strip()


async def model_judge(
    question: str,
    model_answer: str,
    correct_answer: str,
    *,
    judge_model_id: str = "openai/gpt-oss-120b:free",
) -> bool:
    """
    Uses an LLM judge over OpenRouter to determine whether the student's
    answer matches the reference.

    Returns False (rather than raising) if the underlying call fails so
    a flaky judge does not fail the whole resolution loop.
    """
    if not question or not model_answer or not correct_answer:
        return False

    from server.openrouter_client import OpenRouterError, query_model

    prompt = (
        "You are a grading assistant.\n"
        "Evaluate whether the student's answer is correct based on the question "
        "and the reference correct answer.\n\n"
        f"Question: {question}\n"
        f"Correct Answer: {correct_answer}\n"
        f"Student Answer: {model_answer}\n\n"
        "Is the Student Answer correct? Reply with exactly 'yes' if it is correct, "
        "or 'no' if it is incorrect. Do not add any other words or explanation."
    )

    try:
        result = await query_model(
            judge_model_id,
            [{"role": "user", "content": prompt}],
            max_tokens=8,
            temperature=0.0,
        )
    except OpenRouterError as e:
        log.warning("model_judge: OpenRouter call failed: %s", e)
        return False

    res_clean = result["text"].strip().lower().strip(" .!,;:")
    return "yes" in res_clean or "true" in res_clean
