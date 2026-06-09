"""
Thin, project-wide async client for OpenRouter (OpenAI-compatible API).
"""
from __future__ import annotations

import logging
import os
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    AsyncOpenAI,
    AuthenticationError,
    NotFoundError,
    RateLimitError,
)
from tenacity import (
    AsyncRetrying,
    RetryError,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger(__name__)

_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
_TIMEOUT_SECONDS = float(os.environ.get("OPENROUTER_TIMEOUT_SECONDS", "60"))
_MAX_RETRIES = int(os.environ.get("OPENROUTER_MAX_RETRIES", "4"))

# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class OpenRouterError(Exception):
    """Base class for all OpenRouter client errors."""


class OpenRouterAuthError(OpenRouterError):
    """401/403 from OpenRouter. Almost certainly a bad/missing API key."""


class OpenRouterModelUnavailable(OpenRouterError):
    """404 / provider routed-but-failed / model not found."""


class OpenRouterEmptyResponse(OpenRouterError):
    """200 OK but the model produced no text *and* no tool calls."""


class OpenRouterRetriesExhausted(OpenRouterError):
    """All retry attempts failed with retryable errors."""


# --------------------------------------------------------------------------- #
# Client (lazy singleton)
# --------------------------------------------------------------------------- #


_client: AsyncOpenAI | None = None


def _build_client() -> AsyncOpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise OpenRouterAuthError(
            "OPENROUTER_API_KEY is not set. "
            "Copy .env.example to .env and fill it in."
        )
    default_headers: dict[str, str] = {}
    if referer := os.environ.get("OPENROUTER_REFERER"):
        default_headers["HTTP-Referer"] = referer
    if title := os.environ.get("OPENROUTER_APP_TITLE"):
        default_headers["X-Title"] = title
    return AsyncOpenAI(
        api_key=api_key,
        base_url=_BASE_URL,
        timeout=_TIMEOUT_SECONDS,
        default_headers=default_headers or None,
    )


def get_client() -> AsyncOpenAI:
    """Return the process-wide AsyncOpenAI client, building it on first use."""
    global _client
    if _client is None:
        _client = _build_client()
    return _client


def reset_client() -> None:
    """Drop the cached client so the next call rebuilds it. Useful in tests."""
    global _client
    _client = None


# --------------------------------------------------------------------------- #
# Retry policy
# --------------------------------------------------------------------------- #


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (RateLimitError, APIConnectionError)):
        return True
    if isinstance(exc, APIStatusError):
        # 5xx and the occasional 408/425/429 escaping the typed RateLimitError.
        return exc.status_code >= 500 or exc.status_code in (408, 425, 429)
    return False


def _classify_status_error(exc: APIStatusError) -> OpenRouterError:
    if exc.status_code in (401, 403):
        return OpenRouterAuthError(str(exc))
    if exc.status_code == 404:
        return OpenRouterModelUnavailable(str(exc))
    return OpenRouterError(f"HTTP {exc.status_code}: {exc}")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


async def query_model(
    model_id: str,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict] | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.2,
    extra_body: dict | None = None,
) -> dict[str, Any]:
    """
    Single round-trip to OpenRouter.

    Returns:
        {
            "text":          str,              # may be "" when only tool calls were emitted
            "tool_calls":    list | None,      # OpenAI ToolCall objects, or None
            "finish_reason": str,
            "model":         str,              # what OpenRouter actually billed
            "usage": {
                "prompt_tokens":     int,
                "completion_tokens": int,
                "total_tokens":      int,
            },
            "raw": ChatCompletion,             # full SDK object, for debugging
        }

    Raises:
        OpenRouterAuthError:        401/403 (bail fast — do not retry/escalate).
        OpenRouterModelUnavailable: 404 or provider-pass-through failure.
        OpenRouterEmptyResponse:    200 OK but no text and no tool calls.
        OpenRouterRetriesExhausted: all retries failed with retryable errors.
        OpenRouterError:            any other 4xx surfaced from OpenRouter.
    """
    client = get_client()
    kwargs: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    if extra_body:
        kwargs["extra_body"] = extra_body

    try:
        async for attempt in AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(_MAX_RETRIES),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            retry=retry_if_exception(_is_retryable),
            before_sleep=before_sleep_log(log, logging.WARNING),
        ):
            with attempt:
                resp = await client.chat.completions.create(**kwargs)
    except AuthenticationError as e:
        raise OpenRouterAuthError(str(e)) from e
    except NotFoundError as e:
        raise OpenRouterModelUnavailable(str(e)) from e
    except APIStatusError as e:
        # Non-retryable 4xx (or retryable that exhausted attempts).
        raise _classify_status_error(e) from e
    except (RateLimitError, APIConnectionError) as e:
        raise OpenRouterRetriesExhausted(str(e)) from e
    except RetryError as e:  # Should be unreachable thanks to reraise=True.
        raise OpenRouterRetriesExhausted(str(e)) from e

    choice = resp.choices[0]
    text = choice.message.content or ""
    tool_calls = choice.message.tool_calls
    finish_reason = choice.finish_reason or ""

    # OpenRouter sometimes returns 200 with a provider-side failure baked in.
    if finish_reason == "error":
        raise OpenRouterModelUnavailable(
            f"Provider returned finish_reason=error for {model_id}"
        )

    if not text and not tool_calls:
        raise OpenRouterEmptyResponse(
            f"Model {model_id} returned empty content and no tool calls "
            f"(finish_reason={finish_reason!r})."
        )

    usage = (
        resp.usage.model_dump()
        if resp.usage is not None
        else {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    )

    return {
        "text": text,
        "tool_calls": tool_calls,
        "finish_reason": finish_reason,
        "model": getattr(resp, "model", model_id),
        "usage": usage,
        "raw": resp,
    }
