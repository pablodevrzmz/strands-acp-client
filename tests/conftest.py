"""Shared fixtures: the scripted ACP agent and a helper that drives a tool the way Strands does."""

from __future__ import annotations

import sys
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
    """Build tools bound to the fake agent and close their sessions when the test ends."""
    tools: list[Any] = []

    def factory(**overrides: Any) -> Any:
        options: dict[str, Any] = {"agents": {"fake": FAKE_ARGV}, "cwd": tmp_path, "timeout": 10.0, "cancel_grace": 2.0}
        options.update(overrides)
        built = make_acp_agent(**options)
        tools.append(built)
        return built

    yield factory

    for built in tools:
        await built.acp_sessions.close_all()


RunTool = Callable[..., Awaitable[tuple[list[dict[str, Any]], dict[str, Any]]]]


@pytest.fixture
def run_tool() -> RunTool:
    """Invoke a tool through its ``stream`` method, returning (progress events, final tool result)."""

    async def run(built: Any, **inputs: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        tool_use = {"toolUseId": "test-use", "name": built.tool_name, "input": inputs}
        context = ToolContext(tool_use=tool_use, agent=None, invocation_state={})
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
