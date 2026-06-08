import re
from typing import Any

def validate_math_problem(answer: str, expected: str, *args: Any, **kwargs: Any) -> bool:
    """
    Validate math problem (ID 0) by checking if the answer matches expected.
    """
    if not answer or not expected:
        return False
    return answer.strip() == expected.strip()

def validate_cuda_kernel(code: str, *args: Any, **kwargs: Any) -> bool:
    """
    Validate CUDA kernel (ID 1) by checking for CUDA/PTX keywords.
    """
    if not code:
        return False
    required = ["global", "threadIdx", "blockIdx"]
    return any(req in code for req in required)

def validate_sky_color(answer: str, expected: str, *args: Any, **kwargs: Any) -> bool:
    """
    Validate sky color (ID 2).
    """
    if not answer or not expected:
        return False
    return answer.strip().lower() == expected.strip().lower()

def validate_rome_capital(answer: str, expected: str, *args: Any, **kwargs: Any) -> bool:
    """
    Validate capital of Rome (ID 3).
    """
    if not answer or not expected:
        return False
    return answer.strip().lower() == expected.strip().lower()

def validate_smart_pointers(text: str, *args: Any, **kwargs: Any) -> bool:
    """
    Validate explanation of smart pointers (ID 4) by checking for key smart pointer terms.
    """
    if not text:
        return False
    keywords = ["unique_ptr", "shared_ptr", "memory"]
    return any(kw in text.lower() for kw in keywords)

def validate_spin_lock(code: str, *args: Any, **kwargs: Any) -> bool:
    """
    Validate spin lock implementation in C (ID 5).
    """
    if not code:
        return False
    keywords = ["atomic", "lock", "while", "spin"]
    return any(kw in code.lower() for kw in keywords)

def validate_array_rotation(code: str, *args: Any, **kwargs: Any) -> bool:
    """
    Validate 2D array rotation in Python (ID 6).
    """
    if not code:
        return False
    return "def " in code or "rotate" in code.lower()