import importlib.util
from pathlib import Path
from typing import Any

from pydantic import BaseModel


MAX_WEB_CONTEXT_CHARS = 10_000


class ToolCallResult(BaseModel):
    name: str
    ok: bool
    output: str = ""
    error: str | None = None
    metadata: dict[str, Any] = {}


def _load_tool_server_module():
    """
    Load tool-server/server.py despite the hyphen in the directory name.

    MVP choice: call the async tool functions directly rather than adding MCP
    transport/client complexity yet.
    """

    tool_server_path = Path(__file__).resolve().parents[1] / "tool-server" / "server.py"
    spec = importlib.util.spec_from_file_location("tool_server_server", tool_server_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load tool server module from {tool_server_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def run_python_code_tool(code: str, timeout_seconds: int = 20) -> ToolCallResult:
    try:
        module = _load_tool_server_module()
        result = await module.run_python_code(code=code, timeout_seconds=timeout_seconds)
        return ToolCallResult(
            name="run_python_code",
            ok=bool(result.get("ok")),
            output=(result.get("stdout") or "") + (result.get("stderr") or ""),
            error=result.get("error"),
            metadata=result,
        )
    except Exception as e:
        return ToolCallResult(
            name="run_python_code",
            ok=False,
            error=repr(e),
        )


async def web_search_tool(url: str, timeout_seconds: int = 20) -> ToolCallResult:
    try:
        module = _load_tool_server_module()
        result = await module.web_search(url=url, timeout_seconds=timeout_seconds)
        text = result.get("text") or ""
        return ToolCallResult(
            name="web_search",
            ok=bool(result.get("ok")),
            output=text,
            error=result.get("error"),
            metadata={
                "url": result.get("url"),
                "final_url": result.get("final_url"),
                "status_code": result.get("status_code"),
                "content_type": result.get("content_type"),
            },
        )
    except Exception as e:
        return ToolCallResult(
            name="web_search",
            ok=False,
            error=repr(e),
        )


def truncate_web_context(text: str, max_chars: int = MAX_WEB_CONTEXT_CHARS) -> str:
    if not text:
        return ""

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "\n\n[truncated for MVP]"