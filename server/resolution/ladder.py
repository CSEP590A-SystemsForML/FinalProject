"""
Capability-ordered escalation ladder.

The router picks a *starting* model from the prompt alone; the resolution loop
then walks "up" this ladder (cheapest/smallest -> strongest/largest) whenever the
current model fails or is not confident enough. Ordering the models in one place
keeps the escalation policy honest: it is purely a function of model size, read
from configs/models.yaml (total params, with active params as a tie-breaker).

Difficulty is mapped to a starting rung so a prompt the router judges "hard" can
skip the cheapest tiers instead of paying for them to fail first.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

_FALLBACK_LADDER = [
    "openai/gpt-oss-20b:free",
    "openai/gpt-oss-120b:free",
    "qwen/qwen3-coder:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
]

# Router-predicted difficulty -> fraction of the way up the ladder to start.
# easy -> cheapest rung, very_hard -> top rung. Clamped to the ladder length.
_DIFFICULTY_START_FRACTION = {
    "very_easy": 0.0,
    "easy": 0.0,
    "medium": 0.34,
    "hard": 0.67,
    "very_hard": 1.0,
}


def _models_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "models.yaml"


@lru_cache(maxsize=1)
def model_ladder() -> tuple[str, ...]:
    """
    Model ids ordered weakest -> strongest by (total_params, active_params).

    Cached because configs/models.yaml does not change within a process. Falls
    back to a known ordering if the config is missing or unreadable.
    """

    try:
        with open(_models_config_path(), "r") as f:
            models = yaml.safe_load(f) or {}
        if not models:
            return tuple(_FALLBACK_LADDER)
        ordered = sorted(
            models.items(),
            key=lambda kv: (
                float((kv[1] or {}).get("total_params", 0) or 0),
                float((kv[1] or {}).get("active_params", 0) or 0),
            ),
        )
        return tuple(model_id for model_id, _ in ordered)
    except Exception:
        return tuple(_FALLBACK_LADDER)


def ladder_index(model_id: str | None) -> int:
    """Rung of model_id on the ladder; 0 (cheapest) when unknown/missing."""

    ladder = model_ladder()
    if model_id and model_id in ladder:
        return ladder.index(model_id)
    return 0


def strongest_model_id() -> str:
    """Top rung of the ladder (the escalation target of last resort)."""

    ladder = model_ladder()
    return ladder[-1] if ladder else _FALLBACK_LADDER[-1]


def difficulty_start_index(difficulty: str | None) -> int:
    """
    Map a router-predicted difficulty label to a starting rung index.

    Unknown/None difficulty starts at the cheapest rung (index 0), so the
    behavior degrades gracefully to "try cheap first" when no signal is given.
    """

    ladder = model_ladder()
    if not ladder:
        return 0
    fraction = _DIFFICULTY_START_FRACTION.get(
        str(difficulty or "").strip().lower(), 0.0
    )
    idx = round(fraction * (len(ladder) - 1))
    return max(0, min(idx, len(ladder) - 1))


def resolve_start_index(
    model_id: str | None,
    difficulty: str | None,
    confidence: float | None,
    low_confidence_threshold: float,
) -> int:
    """
    Decide which rung to begin solving on, combining three prompt-only signals:

    - the router's explicit model pick (its ladder rung),
    - the router's predicted difficulty (a difficulty-implied rung),
    - the router's confidence (a low score nudges one rung higher).

    We take the *higher* (stronger) of the model-pick rung and the
    difficulty-implied rung, then bump one more rung if confidence is low. This
    keeps a confident "easy" routing cheap while letting a low-confidence or
    "hard" judgment skip rungs that would only fail.
    """

    ladder = model_ladder()
    if not ladder:
        return 0
    last = len(ladder) - 1

    start = max(ladder_index(model_id), difficulty_start_index(difficulty))
    if confidence is not None and confidence < low_confidence_threshold:
        start = min(start + 1, last)
    return max(0, min(start, last))
