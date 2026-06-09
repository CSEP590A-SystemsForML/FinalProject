"""
Verifier-kind dispatch driven by the `verify` field on each problem.

Normalizes the messy short codes already present in `problems.json`
(`m`, `match`, `t`, `tests`, `h`, ...) into three canonical kinds:

    "match"        -> case-insensitive substring/exact compare
    "code_tests"   -> execute the model's code with attached asserts
    "model_judge"  -> ask a cheap OpenRouter model to grade free-form text

Defaults to "model_judge" for any unrecognized kind.
"""
from __future__ import annotations

import logging
from typing import Any

from server.validation.extract import extract_python_code
from server.validation.utils import direct_match, model_judge, run_code

log = logging.getLogger(__name__)

_KIND_ALIASES = {
    "match": "match", "m": "match", "exact": "match",
    "code_tests": "code_tests", "tests": "code_tests",
    "t": "code_tests", "code": "code_tests",
    "model_judge": "model_judge", "judge": "model_judge",
    "h": "model_judge", "llm": "model_judge",
}


def _normalize_kind(kind: str) -> str:
    return _KIND_ALIASES.get((kind or "").strip().lower(), "model_judge")


def _loose_match(answer: str, expected: str) -> bool:
    """Case-insensitive compare; tolerates trailing punctuation and a
    short preamble (e.g. 'The answer is Paris.' matches 'Paris')."""
    if not answer or not expected:
        return False

    def norm(s: str) -> str:
        return s.strip().lower().rstrip(".!?,;:").strip()

    a, e = norm(answer), norm(expected)
    if not a or not e:
        return False
    # Exact, then contained-as-substring (so verbose answers still pass).
    return a == e or e in a or direct_match(a, e)


async def verify(problem: dict[str, Any], answer: str) -> bool:
    """
    Validate `answer` against `problem`. Never raises; returns False on
    any internal error so the escalation loop just moves on.
    """
    kind = _normalize_kind(problem.get("verify", ""))
    try:
        if kind == "match":
            expected = problem.get("answer", "")
            return _loose_match(answer, str(expected))

        if kind == "code_tests":
            tests = "\n".join(problem.get("tests") or [])
            if not tests:
                log.warning(
                    "code_tests verifier called but no `tests` on problem %s",
                    problem.get("id"),
                )
                return False
            code = extract_python_code(answer)
            return run_code(code, tests)

        if kind == "model_judge":
            return await model_judge(
                problem.get("problem", ""),
                answer,
                str(problem.get("answer", "")),
            )
    except Exception as e:  # noqa: BLE001
        log.warning(
            "verify failed for problem %s (%s): %s",
            problem.get("id"), kind, e,
        )
        return False

    return False
