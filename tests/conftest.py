"""Shared fixtures: the scripted ACP agent and a helper that drives a tool the way Strands does."""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
from strands.types.tools import ToolContext

from strands_acp_client import make_acp_agent

FAKE_AGENT = Path(__file__).with_name("fake_acp_agent.py")
FAKE_ARGV = (sys.executable, str(FAKE_AGENT))


@pytest.fixture
def fake_argv() -> tuple[str, ...]:
    return FAKE_ARGV


@pytest.fixture
async def make_tool(tmp_path: Path) -> AsyncIterator[Callable[..., Any]]:
    """Build tools bound to the fake agent; their agent processes and threads end with the test."""
    tools: list[Any] = []

    def factory(**overrides: Any) -> Any:
        options: dict[str, Any] = {"agents": {"fake": FAKE_ARGV}, "cwd": tmp_path, "timeout": 10.0, "cancel_grace": 2.0}
        options.update(overrides)
        built = make_acp_agent(**options)
        tools.append(built)
        return built

    yield factory

    for built in tools:
        await built.aclose()


RunTool = Callable[..., Awaitable[tuple[list[dict[str, Any]], dict[str, Any]]]]


@pytest.fixture
def run_tool() -> RunTool:
    """Invoke a tool through its ``stream`` method, returning (progress events, final tool result)."""

    async def run(
        built: Any, *, cancel_signal: threading.Event | None = None, **inputs: Any
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        tool_use = {"toolUseId": "test-use", "name": built.tool_name, "input": inputs}
        context = ToolContext(tool_use=tool_use, agent=None, invocation_state={})
        if cancel_signal is not None:
            context.cancel_signal = cancel_signal
        events: list[dict[str, Any]] = []
        result: dict[str, Any] | None = None
        async for event in built.stream(tool_use, {}, _tool_context=context):
            payload = getattr(event, "data", None)
            if event.get("type") == "tool_stream":
                data = event["tool_stream_event"]["data"]
                if not (isinstance(data, dict) and "status" in data and "content" in data):
                    events.append(data)  # the final result is also echoed as a stream event; skip it
            elif "tool_result" in event:
                result = event["tool_result"]
            else:
                events.append(payload or dict(event))
        assert result is not None, "the tool never produced a result"
        return events, result

    return run


def result_text(result: dict[str, Any]) -> str:
    return "".join(block.get("text", "") for block in result["content"])


def result_json(result: dict[str, Any]) -> dict[str, Any]:
    for block in result["content"]:
        if "json" in block:
            return dict(block["json"])
    raise AssertionError("no json block in the tool result")


@pytest.fixture
def pid_file(tmp_path: Path) -> str:
    """A path for ``FAKE_ACP_PID_FILE``: the fake agent writes its process id there when it starts."""
    return str(tmp_path / "agent.pid")


def read_pid(pid_file: str) -> int:
    return int(Path(pid_file).read_text())


def is_running(pid: int) -> bool:
    """Whether a process with this id exists. A zombie still does, until its parent collects it."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


async def eventually(condition: Callable[[], bool], timeout: float = 5.0) -> bool:
    """Poll ``condition`` until it holds or ``timeout`` seconds have passed."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        await asyncio.sleep(0.05)
    return condition()
