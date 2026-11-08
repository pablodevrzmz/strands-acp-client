"""Behaviour of the acp_agent tool against the scripted ACP agent."""

from __future__ import annotations

import os
import sys
import time

import pytest

from strands_acp_client import KNOWN_AGENTS, AgentCommand, make_acp_agent
from tests.conftest import FAKE_ARGV, result_json, result_text


def test_tool_spec_lists_agents_and_annotations(tmp_path):
    built = make_acp_agent(agents={"fake": FAKE_ARGV, "claude": "claude"}, cwd=tmp_path)
    spec = built.tool_spec
    assert built.tool_name == "acp_agent"
    schema = spec["inputSchema"]["json"]
    assert schema["required"] == ["task"]
    assert schema["properties"]["agent_name"]["enum"] == ["fake", "claude"]
    assert "session_id" in schema["properties"]
    assert "Claude Code" in spec["description"]
    assert "read_only" in spec["description"]
    assert spec["annotations"]["readOnlyHint"] is True
    assert spec["annotations"]["destructiveHint"] is False


def test_allow_all_is_marked_destructive(tmp_path):
    built = make_acp_agent(agents={"fake": FAKE_ARGV}, cwd=tmp_path, permissions="allow_all")
    assert built.tool_spec["annotations"]["readOnlyHint"] is False
    assert built.tool_spec["annotations"]["destructiveHint"] is True


def test_agents_argument_shapes(tmp_path):
    by_names = make_acp_agent(agents=["claude", "codex"], cwd=tmp_path)
    assert by_names.tool_spec["inputSchema"]["json"]["properties"]["agent_name"]["enum"] == ["claude", "codex"]
    single = make_acp_agent(agents="gemini", cwd=tmp_path)
    assert single.tool_spec["inputSchema"]["json"]["properties"]["agent_name"]["enum"] == ["gemini"]
    custom = make_acp_agent(agents={"mine": AgentCommand(("my-agent", "--acp"), "my agent")}, cwd=tmp_path)
    assert "mine (my agent)" in custom.tool_spec["description"]
    assert set(KNOWN_AGENTS) >= {"claude", "codex", "gemini", "copilot", "kiro", "opencode", "strands"}


def test_factory_rejects_bad_arguments(tmp_path):
    with pytest.raises(ValueError):
        make_acp_agent(agents={}, cwd=tmp_path)
    with pytest.raises(ValueError):
        make_acp_agent(agents={"fake": FAKE_ARGV}, cwd=tmp_path, permissions="sometimes")
    with pytest.raises(ValueError):
        make_acp_agent(agents={"fake": FAKE_ARGV}, cwd=tmp_path, default_agent="other")
    with pytest.raises(ValueError):
        make_acp_agent(agents=["not-a-known-agent"], cwd=tmp_path)
    with pytest.raises(ValueError):
        make_acp_agent(agents={"fake": FAKE_ARGV}, cwd=tmp_path / "missing")


async def test_echo_returns_text_and_metadata(make_tool, run_tool):
    built = make_tool()
    events, result = await run_tool(built, task="hello there")

    assert result["status"] == "success"
    assert result_text(result) == "You said: hello there"
    details = result_json(result)
    assert details["agent"] == "fake"
    assert details["agent_info"] == {"name": "fake-agent", "version": "0.0.1"}
    assert details["stop_reason"] == "end_turn"
    assert details["session_id"].startswith("fake-")
    assert details["resumed"] is False
    assert details["permissions"] == {"policy": "read_only", "allowed": 0, "denied": 0}
    assert [e["type"] for e in events] == ["message_delta", "message_delta"]


async def test_read_only_allows_reads_and_refuses_edits(make_tool, run_tool):
    built = make_tool()
    events, read = await run_tool(built, task="tool read")
    assert "permission allowed via allow-once" in result_text(read)
    assert result_json(read)["permissions"] == {"policy": "read_only", "allowed": 1, "denied": 0}
    assert [e["type"] for e in events] == ["tool_call", "permission", "tool_call_update", "message_delta"]
    assert events[1] == {
        "type": "permission",
        "tool_call_id": "call-1",
        "title": "read example.txt",
        "kind": "read",
        "allowed": True,
    }

    _, edit = await run_tool(built, task="tool edit")
    assert "permission denied via reject-once" in result_text(edit)
    details = result_json(edit)
    assert details["permissions"]["denied"] == 1
    assert details["tool_calls"] == [
        {"id": "call-1", "title": "edit example.txt", "kind": "edit", "status": "failed", "locations": []}
    ]


async def test_allow_all_and_deny_all(make_tool, run_tool):
    _, allowed = await run_tool(make_tool(permissions="allow_all"), task="tool execute")
    assert "permission allowed via allow-once" in result_text(allowed)
    _, denied = await run_tool(make_tool(permissions="deny_all"), task="tool read")
    assert "permission denied via reject-once" in result_text(denied)


async def test_custom_permission_handler(make_tool, run_tool):
    seen = []

    async def handler(request):
        seen.append(request)
        return "allow-always" if request.kind == "edit" else False

    built = make_tool(permissions=handler)
    _, result = await run_tool(built, task="tool edit")
    assert "permission allowed via allow-always" in result_text(result)
    assert result_json(result)["permissions"]["policy"] == "custom"
    assert seen[0].agent == "fake" and seen[0].kind == "edit" and seen[0].title == "edit example.txt"
    assert seen[0].locations[0].endswith("example.txt")
    assert [o.kind for o in seen[0].options] == ["allow_once", "allow_always", "reject_once", "reject_always"]


async def test_session_continues_in_the_same_process(make_tool, run_tool):
    built = make_tool()
    _, first = await run_tool(built, task="turn")
    session_id = result_json(first)["session_id"]
    assert result_text(first) == "turn 1"
    assert len(built.acp_sessions) == 1

    _, second = await run_tool(built, task="turn", session_id=session_id)
    assert result_text(second) == "turn 2"
    assert result_json(second)["session_id"] == session_id
    assert len(built.acp_sessions) == 1

    _, fresh = await run_tool(built, task="turn")
    assert result_text(fresh) == "turn 1"
    assert result_json(fresh)["session_id"] != session_id
    assert len(built.acp_sessions) == 2


async def test_keep_sessions_false_closes_the_process(make_tool, run_tool):
    built = make_tool(keep_sessions=False)
    _, result = await run_tool(built, task="turn")
    assert result["status"] == "success"
    assert len(built.acp_sessions) == 0


async def test_unknown_session_without_load_support_is_an_error(make_tool, run_tool):
    built = make_tool()
    _, result = await run_tool(built, task="turn", session_id="never-seen")
    assert result["status"] == "error"
    assert "cannot resume sessions" in result_text(result)
    assert len(built.acp_sessions) == 0


async def test_session_resumes_through_load_session(make_tool, run_tool):
    built = make_tool(env={"FAKE_ACP_LOAD_SESSION": "1", "FAKE_ACP_LOADED_TURNS": "4"})
    _, result = await run_tool(built, task="turn", session_id="earlier-session")
    assert result["status"] == "success"
    assert result_text(result) == "turn 5"
    details = result_json(result)
    assert details["session_id"] == "earlier-session"
    assert details["resumed"] is True


async def test_max_sessions_evicts_the_least_recently_used(make_tool, run_tool):
    built = make_tool(max_sessions=2)
    _, a = await run_tool(built, task="turn")
    _, b = await run_tool(built, task="turn")
    _, c = await run_tool(built, task="turn")
    assert len(built.acp_sessions) == 2
    assert result_json(a)["session_id"] not in built.acp_sessions.session_ids
    assert {result_json(b)["session_id"], result_json(c)["session_id"]} == set(built.acp_sessions.session_ids)


async def test_timeout_cancels_a_slow_turn(make_tool, run_tool):
    built = make_tool(timeout=0.5)
    started = time.perf_counter()
    events, result = await run_tool(built, task="slow 30")
    elapsed = time.perf_counter() - started
    assert elapsed < 5
    assert result["status"] == "error"
    assert "did not finish within 0s" in result_text(result) or "did not finish within 1s" in result_text(result)
    assert "working..." in result_text(result)
    assert result_json(result)["stop_reason"] == "timeout"
    assert len(built.acp_sessions) == 0


async def test_cancel_signal_stops_the_turn(make_tool, run_tool, monkeypatch):
    import threading

    from strands.types.tools import ToolContext

    built = make_tool(timeout=10)
    signal = threading.Event()
    tool_use = {"toolUseId": "cancel-use", "name": built.tool_name, "input": {"task": "slow 30"}}
    context = ToolContext(tool_use=tool_use, agent=None, invocation_state={}, cancel_signal=signal)
    result = None
    started = time.perf_counter()
    async for event in built.stream(tool_use, {}, _tool_context=context):
        if event.get("type") == "tool_stream":
            signal.set()
        if "tool_result" in event:
            result = event["tool_result"]
    assert result is not None
    assert time.perf_counter() - started < 5
    assert result["status"] == "error"
    assert result_json(result)["stop_reason"] == "cancelled"


async def test_usage_is_reported(make_tool, run_tool):
    _, result = await run_tool(make_tool(), task="usage")
    assert result_json(result)["usage"] == {"total_tokens": 30, "input_tokens": 20, "output_tokens": 10}


async def test_refusal_is_an_error(make_tool, run_tool):
    _, result = await run_tool(make_tool(), task="refuse")
    assert result["status"] == "error"
    assert "refused" in result_text(result)


async def test_agent_errors_are_quoted(make_tool, run_tool):
    built = make_tool()
    _, auth = await run_tool(built, task="fail")
    assert auth["status"] == "error"
    assert "requires authentication: Authentication required" in result_text(auth)
    _, internal = await run_tool(built, task="error")
    assert internal["status"] == "error"
    assert "returned an error (code -32603)" in result_text(internal)
    assert len(built.acp_sessions) == 0


async def test_protocol_version_mismatch_is_reported(make_tool, run_tool):
    built = make_tool(env={"FAKE_ACP_PROTOCOL_VERSION": "2"})
    _, result = await run_tool(built, task="hello")
    assert result["status"] == "error"
    assert "protocol version 2" in result_text(result)
    assert len(built.acp_sessions) == 0


async def test_agent_crash_mid_turn_is_reported(make_tool, run_tool):
    built = make_tool()
    _, result = await run_tool(built, task="crash")
    assert result["status"] == "error"
    assert "failed mid-turn" in result_text(result)
    assert len(built.acp_sessions) == 0


async def test_unknown_agent_and_bad_cwd_do_not_spawn(make_tool, run_tool, tmp_path):
    built = make_tool()
    _, result = await run_tool(built, task="hello", agent_name="nope")
    assert result["status"] == "error" and "unknown agent" in result_text(result)
    _, result = await run_tool(built, task="hello", cwd=str(tmp_path / "missing"))
    assert result["status"] == "error" and "not a directory" in result_text(result)
    _, result = await run_tool(built, task="   ")
    assert result["status"] == "error" and "empty" in result_text(result)
    assert len(built.acp_sessions) == 0


async def test_missing_program_is_reported(make_tool, run_tool):
    built = make_tool(agents={"ghost": ("definitely-not-a-real-program-xyz",)})
    _, result = await run_tool(built, task="hello", agent_name="ghost")
    assert result["status"] == "error"
    assert "was not found on PATH" in result_text(result)


async def test_agent_runs_in_the_requested_cwd(make_tool, run_tool, tmp_path):
    other = tmp_path / "elsewhere"
    other.mkdir()
    seen = []

    def handler(request):
        seen.append(request.locations[0])
        return True

    built = make_tool(permissions=handler)
    await run_tool(built, task="tool read", cwd=str(other))
    assert os.path.dirname(seen[0]) == str(other.resolve())


def test_fake_agent_runs_under_this_interpreter():
    assert FAKE_ARGV[0] == sys.executable
