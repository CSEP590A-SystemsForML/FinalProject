from pathlib import Path
import re

import yaml


DEFAULT_CAVEMAN_PROMPT = (
    "Respond like smart caveman. All technical substance stay. Only fluff die. "
    "Short fragments ok. Technical terms exact. Code unchanged."
)

DEFAULT_MAX_COMPRESSED_CHARS = 10_000
DEFAULT_LONG_CONTEXT_CHARS = 8_000


def _load_caveman_prompt() -> str:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "prompts.yaml"
    try:
        with open(config_path, "r") as f:
            prompts = yaml.safe_load(f) or {}
        return prompts.get("optimizations", {}).get("caveman") or DEFAULT_CAVEMAN_PROMPT
    except Exception:
        return DEFAULT_CAVEMAN_PROMPT


def caveman_prompt(prompt: str) -> str:
    """
    Add the caveman optimization instruction to a system prompt.
    """

    caveman = _load_caveman_prompt().strip()
    prompt = (prompt or "").strip()

    if caveman in prompt:
        return prompt

    if not prompt:
        return caveman

    return f"{prompt}\n\nOptimization instruction:\n{caveman}"


def _squash_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def compress_web_search(web_text: str, max_chars: int = DEFAULT_MAX_COMPRESSED_CHARS) -> str:
    """
    Deterministic MVP web compression.

    This is intentionally not AI summarization yet. It removes repeated
    whitespace and caps length so web context cannot explode prompt tokens.
    """

    compressed = _squash_whitespace(web_text)
    if len(compressed) <= max_chars:
        return compressed

    return compressed[:max_chars] + " [compressed/truncated for MVP]"


def long_context_compression_lemma(
    long_context: str,
    max_chars: int = DEFAULT_LONG_CONTEXT_CHARS,
) -> str:
    """
    Heuristic long-context compression.

    Keep the beginning and end because prompts often place setup first and the
    actual question/constraints near the end.
    """

    context = _squash_whitespace(long_context)
    if len(context) <= max_chars:
        return context

    head_chars = max_chars // 2
    tail_chars = max_chars - head_chars
    return (
        context[:head_chars]
        + " [middle omitted by heuristic compression] "
        + context[-tail_chars:]
    )


def long_context_compression_ai(long_context: str) -> str:
    """
    Placeholder for AI compression.

    MVP behavior delegates to deterministic heuristic compression so callers can
    toggle this path without requiring a second model call yet.
    """

    return long_context_compression_lemma(long_context)