from typing import Callable, Dict, Any
from server.validation import validation_functions

_REGISTRY: Dict[int, Callable[..., bool]] = {
    0: validation_functions.validate_math_problem,
    1: validation_functions.validate_cuda_kernel,
    2: validation_functions.validate_sky_color,
    3: validation_functions.validate_rome_capital,
    4: validation_functions.validate_smart_pointers,
    5: validation_functions.validate_spin_lock,
    6: validation_functions.validate_array_rotation,
}

def test(problem_id: int, previous_result: bool, *args: Any, **kwargs: Any) -> bool:
    """
    Executes the validation function for the given problem_id if previous_result is True.
    
    If previous_result is False, the failure is propagated immediately, returning False
    without running the validation function.
    
    If problem_id is not mapped in the registry, the previous_result is propagated.
    """
    if not previous_result:
        return False
        
    if problem_id not in _REGISTRY:
        return previous_result
        
    validator = _REGISTRY[problem_id]
    return validator(*args, **kwargs)