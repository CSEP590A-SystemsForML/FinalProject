from pathlib import Path
import re

import yaml

import server.utils as _utils


DEFAULT_CAVEMAN_PROMPT = (
    "Respond like smart caveman. All technical substance stay. Only fluff die. "
    "Short fragments ok. Technical terms exact. Code unchanged."
)

DEFAULT_MAX_COMPRESSED_CHARS = 10_000
DEFAULT_LONG_CONTEXT_CHARS = 8_000
DEFAULT_LONG_CONTEXT_AI_MODEL = "openai/gpt-oss-20b:free"
DEFAULT_LONG_CONTEXT_AI_PROMPT = (
    "You compress text without losing solving-critical detail. "
    "Preserve every instruction, constraint, number, variable name, function "
    "or test signature, code block, example, and any explicit must/return "
    "requirement exactly. Drop only filler, repetition, and prose that does "
    "not change the answer. The result must be strictly shorter than the "
    "input and self-contained. Return only the compressed text."
)


def _load_caveman_prompt() -> str:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "prompts.yaml"
    try:
        with open(config_path, "r") as f:
            prompts = yaml.safe_load(f) or {}
        return prompts.get("optimizations", {}).get("caveman") or DEFAULT_CAVEMAN_PROMPT
    except Exception:
        return DEFAULT_CAVEMAN_PROMPT


def _load_long_context_ai_prompt() -> str:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "prompts.yaml"
    try:
        with open(config_path, "r") as f:
            prompts = yaml.safe_load(f) or {}
        return (
            prompts.get("optimizations", {}).get("long_context_compression_ai")
            or DEFAULT_LONG_CONTEXT_AI_PROMPT
        )
    except Exception:
        return DEFAULT_LONG_CONTEXT_AI_PROMPT


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


def long_context_compression_ai(
    long_context: str,
    max_chars: int = DEFAULT_LONG_CONTEXT_CHARS,
    model_id: str | None = None,
) -> str:
    """
    AI long-context compression.

    Below-threshold inputs are returned with only whitespace squashed, matching
    the lemma variant so behavior is comparable. Above-threshold inputs are
    handed to a small/cheap external model with an instruction to preserve
    every constraint, number, code block, and example. The call is routed
    through `server.utils.query_model` via the module attribute so test
    monkeypatches (e.g. `scripts/e2e_smoke.py`) keep working.

    Safety:
      - any model error or empty completion -> fall back to lemma.
      - if the model returns something not strictly shorter -> fall back to
        lemma (a compressor that does not compress is a bug, not a feature).
    """

    context = _squash_whitespace(long_context)
    if len(context) <= max_chars:
        return context

    system_prompt = _load_long_context_ai_prompt().strip()
    # Give the model enough room to write a compressed version up to ~max_chars
    # plus some slack; estimate_tokens uses ~4 chars/token.
    target_completion_tokens = max(512, max_chars // 4 + 256)

    result = _utils.query_model(
        model_id=model_id or DEFAULT_LONG_CONTEXT_AI_MODEL,
        prompt_or_messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context},
        ],
        temperature=0,
        max_completion_tokens=target_completion_tokens,
    )

    if result.error:
        return long_context_compression_lemma(context, max_chars=max_chars)

    compressed = (result.text or "").strip()
    if not compressed or len(compressed) >= len(context):
        return long_context_compression_lemma(context, max_chars=max_chars)

    return compressed