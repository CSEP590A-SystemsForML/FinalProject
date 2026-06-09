"""
Project-wide helpers. The OpenRouter call interface is exported here so that
existing imports like `from server.utils import query_model` keep working.
"""
from server.openrouter_client import (  # noqa: F401
    OpenRouterAuthError,
    OpenRouterEmptyResponse,
    OpenRouterError,
    OpenRouterModelUnavailable,
    OpenRouterRetriesExhausted,
    query_model,
)
