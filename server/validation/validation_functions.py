import re
from typing import Any

_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _last_number(text: str) -> float | None:
    """Extract the last numeric value from text, tolerating commas, $ and %."""
    if text is None:
        return None
    matches = _NUMBER_RE.findall(str(text))
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return None


def validate_numeric_match(answer: str, expected: str, *args: Any, **kwargs: Any) -> bool:
    """
    Compare the final numeric value in the model answer against the expected
    number. Robust to surrounding prose, currency symbols, and thousands commas.
    """
    got = _last_number(answer)
    want = _last_number(expected)
    if got is None or want is None:
        return False
    return abs(got - want) < 1e-6


def validate_text_equals_ci(answer: str, expected: str, *args: Any, **kwargs: Any) -> bool:
    """
    Case-insensitive equality after trimming whitespace and surrounding
    punctuation. Good for short factual answers (names, symbols, single words).
    """
    if not answer or not expected:
        return False
    strip_chars = " \t\n.!,;:'\"()"
    return answer.strip().strip(strip_chars).lower() == expected.strip().strip(strip_chars).lower()


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