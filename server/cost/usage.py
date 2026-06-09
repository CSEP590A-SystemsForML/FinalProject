"""
Convenience helpers that translate an OpenRouter `query_model` result into a
dollar-cost estimate using `server.cost.cost_function.calculate_cost`.

Anyone who calls `query_model` should also call `cost_for_call` (or use the
combined `query_and_cost`) so cost tracking stays consistent.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from server.cost.cost_function import calculate_cost
from server.interfaces import CompletionConfig, InferenceConfig, ModelConfig
from server.openrouter_client import query_model

log = logging.getLogger(__name__)

_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "configs" / "models.yaml"
)


@lru_cache(maxsize=1)
def _models_config() -> dict[str, dict[str, Any]]:
    return yaml.safe_load(_CONFIG_PATH.read_text()) or {}


def model_config_for(model_id: str) -> ModelConfig:
    """Build a ModelConfig from configs/models.yaml. Raises KeyError if unknown."""
    cfg = _models_config()
    if model_id not in cfg:
        raise KeyError(
            f"Model {model_id!r} not found in {_CONFIG_PATH}. "
            f"Known models: {sorted(cfg.keys())}"
        )
    info = cfg[model_id]
    return ModelConfig(
        id=model_id,
        source=info.get("source", ""),
        source_url=info.get("source_url", ""),
        total_params=info["total_params"],
        active_params=info["active_params"],
    )


def cost_for_call(model_id: str, usage: dict[str, Any]) -> float:
    """
    Estimate the dollar cost of a single OpenRouter completion given its
    `usage` block.

    Returns 0.0 (and logs a warning) if the model is not in models.yaml so
    a missing config never crashes the resolution loop.
    """
    try:
        mc = model_config_for(model_id)
    except KeyError as e:
        log.warning("cost_for_call: %s", e)
        return 0.0

    comp = CompletionConfig(
        prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
        completion_tokens=int(usage.get("completion_tokens", 0) or 0),
    )
    return calculate_cost(mc, InferenceConfig(completions=[comp]))


async def query_and_cost(
    model_id: str,
    messages: list[dict[str, Any]],
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Same as `query_model`, but additionally stamps an estimated `cost`
    (USD, float) into the returned dict.
    """
    result = await query_model(model_id, messages, **kwargs)
    result["cost"] = cost_for_call(model_id, result["usage"])
    return result
