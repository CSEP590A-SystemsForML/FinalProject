import os
from typing import Any

from openai import OpenAI

from server.interfaces import ModelCallResult


def estimate_tokens(text: str | None) -> int:
    """
    Cheap MVP token estimate.

    Most English/code tokens are roughly 3-5 chars. Use 4 chars/token so the
    resolution loop can still estimate cost when the provider does not return
    usage metadata.
    """

    if not text:
        return 0
    return max(1, int(len(text) / 4))


def normalize_messages(prompt_or_messages: str | list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Accept either a plain prompt string or OpenAI-compatible chat messages.
    """

    if isinstance(prompt_or_messages, str):
        return [{"role": "user", "content": prompt_or_messages}]

    if isinstance(prompt_or_messages, list):
        return prompt_or_messages

    raise TypeError("prompt_or_messages must be a string or list of chat messages.")


def get_external_api_key() -> str | None:
    """
    Returns the configured external model API key.

    API_TOKEN is the project MVP env var. OPENROUTER_API_KEY and LITELLM_API_KEY
    are also supported for compatibility with common provider/proxy naming.
    """

    return (
        os.environ.get("API_TOKEN")
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("LITELLM_API_KEY")
    )


def get_external_base_url() -> str:
    """
    Returns the OpenAI-compatible base URL for external model calls.
    """

    return os.environ.get("EXTERNAL_MODEL_BASE_URL", "https://openrouter.ai/api/v1")


def get_external_client() -> OpenAI:
    """
    Builds an OpenAI-compatible client for external model providers.
    """

    api_key = get_external_api_key()
    if not api_key:
        raise RuntimeError("Missing API_TOKEN, OPENROUTER_API_KEY, or LITELLM_API_KEY.")

    return OpenAI(
        api_key=api_key,
        base_url=get_external_base_url(),
    )


def _usage_value(usage: Any, key: str) -> int | None:
    if usage is None:
        return None

    if isinstance(usage, dict):
        value = usage.get(key)
    else:
        value = getattr(usage, key, None)

    if value is None:
        return None

    return int(value)


def query_model(
    model_id: str,
    prompt_or_messages: str | list[dict[str, str]],
    temperature: float = 0,
    max_completion_tokens: int = 4096,
) -> ModelCallResult:
    """
    Calls an external OpenAI-compatible model and returns text plus token usage.

    This function intentionally does not write metrics. The resolution/server
    layer owns all metrics collection for the MVP.
    """

    messages = normalize_messages(prompt_or_messages)

    try:
        client = get_external_client()
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
        )

        text = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)

        prompt_tokens = _usage_value(usage, "prompt_tokens")
        completion_tokens = _usage_value(usage, "completion_tokens")

        if prompt_tokens is None:
            prompt_tokens = estimate_tokens(
                "\n".join(str(message.get("content", "")) for message in messages)
            )

        if completion_tokens is None:
            completion_tokens = estimate_tokens(text)

        return ModelCallResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model_id=model_id,
            error=None,
        )

    except Exception as e:
        return ModelCallResult(
            text="",
            prompt_tokens=estimate_tokens(
                "\n".join(str(message.get("content", "")) for message in messages)
            ),
            completion_tokens=0,
            model_id=model_id,
            error=repr(e),
        )