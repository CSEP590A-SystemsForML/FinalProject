"""
Async-aware dispatch into per-problem validation functions.

`test` is now async because `model_judge`-backed validators need to await
the OpenRouter client. Existing synchronous validators in
`validation_functions.py` are still called the same way; we just await
the call site even though the return is plain bool.
"""
from __future__ import annotations

import asyncio
import inspect
from typing import Any, Awaitable, Callable, Dict, Union

from server.validation import validation_functions

ValidatorReturn = Union[bool, Awaitable[bool]]
Validator = Callable[..., ValidatorReturn]

_REGISTRY: Dict[int, Validator] = {
    0: validation_functions.validate_math_problem,
    1: validation_functions.validate_cuda_kernel,
    2: validation_functions.validate_sky_color,
    3: validation_functions.validate_rome_capital,
    4: validation_functions.validate_smart_pointers,
    5: validation_functions.validate_spin_lock,
    6: validation_functions.validate_array_rotation,
}


async def test(
    problem_id: int,
    previous_result: bool,
    *args: Any,
    **kwargs: Any,
) -> bool:
    """
    Executes the validation function for the given problem_id if previous_result is True.

    If previous_result is False, the failure is propagated immediately, returning False
    without running the validation function.

    If problem_id is not mapped in the registry, the previous_result is propagated.

    The function is async so judges that need to call out to OpenRouter can
    `await`; validators that return plain bools are also supported.
    """
    if not previous_result:
        return False

    if problem_id not in _REGISTRY:
        return previous_result

    validator = _REGISTRY[problem_id]
    result = validator(*args, **kwargs)
    if inspect.isawaitable(result):
        result = await result
    return bool(result)


def test_sync(
    problem_id: int,
    previous_result: bool,
    *args: Any,
    **kwargs: Any,
) -> bool:
    """Synchronous convenience wrapper around `test` for non-async callers."""
    return asyncio.run(test(problem_id, previous_result, *args, **kwargs))
