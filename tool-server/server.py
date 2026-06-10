from __future__ import annotations

from typing import Optional

from fastmcp import FastMCP

import core


mcp = FastMCP("tools")

DEFAULT_TIMEOUT_SECONDS = core.DEFAULT_TIMEOUT_SECONDS
MAX_TIMEOUT_SECONDS = core.MAX_TIMEOUT_SECONDS


@mcp.tool()
async def run_python_code(
    code: str,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    cwd: Optional[str] = None,
) -> dict:
    """
    Execute Python code in a subprocess and return stdout, stderr, exit code,
    and timeout status.
    """

    return await core.run_python_code(code=code, timeout_seconds=timeout_seconds, cwd=cwd)


@mcp.tool()
async def web_search(
    url: str,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict:
    """
    Fetch a webpage from a URL and return status code, final URL, content type,
    headers, and text content.

    This fetches a specific URL. It is not a search-engine query tool.
    """

    return await core.web_search(url=url, timeout_seconds=timeout_seconds)


if __name__ == "__main__":
    mcp.run()
