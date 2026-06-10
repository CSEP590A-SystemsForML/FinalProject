from typing import Any, Callable, Dict

from server.validation import validation_functions


# Name-keyed registry. Problems reference a validator by name via the
# "validator" field, so the registry scales independently of problem_id.
_BY_NAME: Dict[str, Callable[..., bool]] = {
    "numeric_match": validation_functions.validate_numeric_match,
    "text_equals_ci": validation_functions.validate_text_equals_ci,
    "math_problem": validation_functions.validate_math_problem,
    "cuda_kernel": validation_functions.validate_cuda_kernel,
    "sky_color": validation_functions.validate_sky_color,
    "rome_capital": validation_functions.validate_rome_capital,
    "smart_pointers": validation_functions.validate_smart_pointers,
    "spin_lock": validation_functions.validate_spin_lock,
    "array_rotation": validation_functions.validate_array_rotation,
}

# Backward-compatible mapping for the original 7-problem sample, which keyed
# heuristic validators by problem_id. New problems should use names instead.
_LEGACY_BY_ID: Dict[int, Callable[..., bool]] = {
    0: validation_functions.validate_math_problem,
    1: validation_functions.validate_cuda_kernel,
    2: validation_functions.validate_sky_color,
    3: validation_functions.validate_rome_capital,
    4: validation_functions.validate_smart_pointers,
    5: validation_functions.validate_spin_lock,
    6: validation_functions.validate_array_rotation,
}


def get_validator(
    validator_name: str | None = None,
    problem_id: int | None = None,
) -> Callable[..., bool] | None:
    """
    Resolve a heuristic validator by name first, then fall back to the legacy
    problem_id mapping. Returns None if neither resolves.
    """

    if validator_name:
        validator = _BY_NAME.get(validator_name.strip())
        if validator is not None:
            return validator

    if problem_id is not None and problem_id in _LEGACY_BY_ID:
        return _LEGACY_BY_ID[problem_id]

    return None


def validate_named(
    model_answer: str,
    expected_answer: str,
    validator_name: str | None = None,
    problem_id: int | None = None,
) -> bool:
    """
    Run the resolved heuristic validator. Returns False if no validator matches,
    so an unregistered validator name fails closed rather than silently passing.
    """

    validator = get_validator(validator_name, problem_id)
    if validator is None:
        return False

    return validator(model_answer, expected_answer)


def test(problem_id: int, previous_result: bool, *args: Any, **kwargs: Any) -> bool:
    """
    Deprecated legacy entry point kept for backward compatibility.

    Executes the validation function for the given problem_id if previous_result
    is True. If problem_id is not mapped, previous_result is propagated.
    """

    if not previous_result:
        return False

    validator = _LEGACY_BY_ID.get(problem_id)
    if validator is None:
        return previous_result

    return validator(*args, **kwargs)
