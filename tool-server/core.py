"""
Plain tool executors with no MCP/transport dependency.

Both the FastMCP server (`server.py`) and the in-process resolution-server
wrappers (`server/tools.py`) call these. Keeping them free of the `fastmcp`
import lets the resolution path use the sandboxed Python runner without pulling
the MCP stack (important for lightweight CI).
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx


DEFAULT_TIMEOUT_SECONDS = 20
MAX_TIMEOUT_SECONDS = 300


def clamp_timeout(timeout_seconds: int) -> int:
    return max(1, min(timeout_seconds, MAX_TIMEOUT_SECONDS))


def _decode(data: bytes | None) -> str:
    if not data:
        return ""
    return data.decode("utf-8", errors="replace")


async def run_python_code(
    code: str,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    cwd: Optional[str] = None,
) -> dict:
    """
    Execute Python code in a subprocess and return stdout, stderr, exit code,
    and timeout status.

    By default, each run gets its own temporary working directory so concurrent
    requests do not collide through shared local files.

    If cwd is provided, the code runs from that directory. In that case, concurrent
    calls can still collide if the code mutates the same files.
    """

    timeout_seconds = clamp_timeout(timeout_seconds)

    with tempfile.TemporaryDirectory(prefix="mcp-python-run-") as temp_dir:
        temp_path = Path(temp_dir)
        script_path = temp_path / "main.py"
        script_path.write_text(code, encoding="utf-8")

        if cwd is None:
            working_dir = temp_path
        else:
            working_dir = Path(cwd).expanduser().resolve()

            if not working_dir.exists() or not working_dir.is_dir():
                return {
                    "ok": False,
                    "error": f"Invalid cwd: {str(working_dir)}",
                    "stdout": "",
                    "stderr": "",
                    "returncode": None,
                    "timed_out": False,
                }

        env = os.environ.copy()
        env["MCP_RUN_TEMP_DIR"] = str(temp_path)

        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(script_path),
            cwd=str(working_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout_seconds,
            )

            return {
                "ok": process.returncode == 0,
                "stdout": _decode(stdout),
                "stderr": _decode(stderr),
                "returncode": process.returncode,
                "timed_out": False,
                "temp_dir": str(temp_path),
            }

        except asyncio.TimeoutError:
            process.kill()
            stdout, stderr = await process.communicate()

            return {
                "ok": False,
                "stdout": _decode(stdout),
                "stderr": _decode(stderr),
                "returncode": process.returncode,
                "timed_out": True,
                "error": f"Python execution timed out after {timeout_seconds} seconds.",
                "temp_dir": str(temp_path),
            }


async def web_search(
    url: str,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict:
    """
    Fetch a webpage from a URL and return status code, final URL, content type,
    headers, and text content.

    This fetches a specific URL. It is not a search-engine query tool.
    """

    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        return {
            "ok": False,
            "url": url,
            "error": "Only http:// and https:// URLs are allowed.",
        }

    timeout_seconds = clamp_timeout(timeout_seconds)

    headers = {
        "User-Agent": "basic-dev-tools-mcp/1.0",
        "Accept": "text/html,text/plain,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=timeout_seconds,
            headers=headers,
        ) as client:
            response = await client.get(url)

        return {
            "ok": 200 <= response.status_code < 400,
            "url": url,
            "final_url": str(response.url),
            "status_code": response.status_code,
            "content_type": response.headers.get("content-type", ""),
            "headers": dict(response.headers),
            "text": response.text,
        }

    except httpx.TimeoutException:
        return {
            "ok": False,
            "url": url,
            "error": f"Request timed out after {timeout_seconds} seconds.",
        }

    except Exception as e:
        return {
            "ok": False,
            "url": url,
            "error": repr(e),
        }
