"""
Helpers for extracting structured content out of LLM responses.
"""
from __future__ import annotations

import re

# Match ```python ... ``` or ``` ... ``` fenced blocks. Greedy across newlines.
_FENCED_RE = re.compile(
    r"```(?:python|py)?\s*\n?(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def extract_python_code(text: str) -> str:
    """
    Pull the first fenced Python code block out of `text`. If no fenced
    block is present, return the original text (some models just emit raw
    code without fences).
    """
    if not text:
        return ""
    match = _FENCED_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()
