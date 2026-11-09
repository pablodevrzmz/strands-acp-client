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
    by_names = make_acp_agent(agents=["claude", "opencode"], cwd=tmp_path)
    assert by_names.tool_spec["inputSchema"]["json"]["properties"]["agent_name"]["enum"] == ["claude", "opencode"]
    single = make_acp_agent(agents="strands", cwd=tmp_path)
    assert single.tool_spec["inputSchema"]["json"]["properties"]["agent_name"]["enum"] == ["strands"]
    custom = make_acp_agent(agents={"mine": AgentCommand(("my-agent", "--acp"), "my agent")}, cwd=tmp_path)
    assert "mine (my agent)" in custom.tool_spec["description"]
    # Presets exist only for agents the end-to-end suite has driven for real.
    assert set(KNOWN_AGENTS) == {"claude", "codex", "kilo", "opencode", "qwen", "strands"}
    with pytest.raises(ValueError, match="Pass argv explicitly"):
        make_acp_agent(agents=["gemini"], cwd=tmp_path)
    unverified = make_acp_agent(agents={"gemini": ["npx", "-y", "@google/gemini-cli", "--acp"]}, cwd=tmp_path)
    assert unverified.tool_spec["inputSchema"]["json"]["properties"]["agent_name"]["enum"] == ["gemini"]


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
    assert result_text(edit).endswith("Refused under the read_only policy: edit example.txt.")
    details = result_json(edit)
    assert details["permissions"]["denied"] == 1
    assert details["refused"] == ["edit example.txt"]
    assert "refused" not in result_json(read)
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


async def test_env_changes_reach_the_agent_and_none_unsets(make_tool, run_tool, monkeypatch):
    monkeypatch.setenv("FAKE_PARENT_VAR", "from-parent")
    built = make_tool(env={"FAKE_PARENT_VAR": None, "FAKE_EXTRA_VAR": "from-tool"})
    _, hidden = await run_tool(built, task="env FAKE_PARENT_VAR")
    assert result_text(hidden) == "<unset>"
    _, extra = await run_tool(built, task="env FAKE_EXTRA_VAR")
    assert result_text(extra) == "from-tool"
    inherited = make_tool()
    _, kept = await run_tool(inherited, task="env FAKE_PARENT_VAR")
    assert result_text(kept) == "from-parent"


def test_fake_agent_runs_under_this_interpreter():
    assert FAKE_ARGV[0] == sys.executable


# ── a policy only answers questions: the agent has to ask, and the tool checks that it did ──


async def test_unasked_edit_is_a_policy_violation(make_tool, run_tool):
    built = make_tool()
    events, result = await run_tool(built, task="sneak edit")
    assert result["status"] == "error"
    details = result_json(result)
    assert details["stop_reason"] == "policy_violation"
    assert details["policy_violation"] == {"tool_call_id": "sneak-1", "title": "edit example.txt", "kind": "edit"}
    assert "without asking for permission" in result_text(result)
    assert "nobody stopped me" not in result_text(result), "the turn must be cancelled, not left to finish"
    assert any(event["type"] == "policy_violation" for event in events)
    assert len(built.acp_sessions) == 0


async def test_unasked_reads_are_fine_and_allow_all_is_not_guarded(make_tool, run_tool):
    _, read = await run_tool(make_tool(), task="sneak read 0.2")
    assert read["status"] == "success" and "nobody stopped me" in result_text(read)
    _, edit = await run_tool(make_tool(permissions="allow_all"), task="sneak edit 0.2")
    assert edit["status"] == "success"
    assert "policy_violation" not in result_json(edit)


async def test_strict_false_lets_unasked_changes_through(make_tool, run_tool):
    _, result = await run_tool(make_tool(strict=False), task="sneak execute 0.2")
    assert result["status"] == "success"
    details = result_json(result)
    assert details["tool_calls"][0]["status"] == "completed"
    assert "policy_violation" not in details


async def test_allow_always_from_a_handler_is_remembered(make_tool, run_tool):
    built = make_tool(permissions=lambda request: "allow-always")
    _, first = await run_tool(built, task="tool edit")
    assert "permission allowed via allow-always" in result_text(first)
    # The agent was told "always" for edits, so it is entitled not to ask again in this session.
    _, second = await run_tool(built, task="sneak edit 0.2", session_id=result_json(first)["session_id"])
    assert second["status"] == "success"


async def test_restrictive_policy_selects_the_agents_asking_mode(make_tool, run_tool):
    command = AgentCommand(FAKE_ARGV, "fake", modes={"read_only": "ask"})
    env = {"FAKE_ACP_MODES": "ask,auto"}
    built = make_tool(agents={"fake": command}, env=env)
    _, result = await run_tool(built, task="mode")
    assert result_text(result) == "mode ask"
    assert result_json(result)["mode"] == "ask"

    permissive = make_tool(agents={"fake": command}, env=env, permissions="allow_all")
    _, result = await run_tool(permissive, task="mode")
    assert result_text(result) == "mode auto"
    assert result_json(result)["mode"] is None


async def test_mode_is_selected_through_the_config_option(make_tool, run_tool):
    command = AgentCommand(FAKE_ARGV, "fake", modes={"read_only": "plan"})
    env = {"FAKE_ACP_CONFIG_MODES": "build,plan", "FAKE_ACP_MODE": "build"}
    _, result = await run_tool(make_tool(agents={"fake": command}, env=env), task="mode")
    assert result_text(result) == "mode plan"
    assert result_json(result)["mode"] == "plan"


async def test_a_mode_the_agent_does_not_offer_is_skipped(make_tool, run_tool):
    command = AgentCommand(FAKE_ARGV, "fake", modes={"read_only": "nope"})
    _, result = await run_tool(make_tool(agents={"fake": command}, env={"FAKE_ACP_MODES": "ask,auto"}), task="mode")
    assert result_text(result) == "mode auto"
    assert result_json(result)["mode"] is None


def test_known_agents_carry_their_asking_modes():
    from strands_acp_client._agents import resolve_agents

    pinned = resolve_agents({"claude": ["npx", "-y", "@agentclientprotocol/claude-agent-acp@0.101.0"]})
    assert pinned["claude"].modes == {"read_only": "default", "deny_all": "default", "custom": "default"}
    assert resolve_agents(["opencode"])["opencode"].modes == {"read_only": "plan", "deny_all": "plan"}
    assert resolve_agents(["qwen"])["qwen"].modes == pinned["claude"].modes
    assert resolve_agents({"mine": ["my-agent"]})["mine"].modes is None


async def test_policy_args_are_appended_for_that_policy_only(make_tool, run_tool):
    command = AgentCommand(FAKE_ARGV, "fake", policy_args={"read_only": ("--tools", "read")})
    _, restricted = await run_tool(make_tool(agents={"fake": command}), task="argv")
    assert result_text(restricted) == "--tools read"
    _, free = await run_tool(make_tool(agents={"fake": command}, permissions="allow_all"), task="argv")
    assert result_text(free) == "<none>"


async def test_with_args_adds_flags_and_keeps_what_is_known(make_tool, run_tool):
    command = AgentCommand(
        FAKE_ARGV, "fake", modes={"read_only": "ask"}, policy_args={"read_only": ("--tools", "read")}
    )
    pinned = command.with_args("--model", "local/tiny", "")
    assert pinned.argv == (*FAKE_ARGV, "--model", "local/tiny", "")
    assert pinned.modes == command.modes and pinned.policy_args == command.policy_args
    _, result = await run_tool(make_tool(agents={"fake": pinned}), task="argv")
    assert result_text(result).startswith("--model local/tiny")
    assert result_text(result).endswith("--tools read")


def test_strands_is_started_read_only_under_restrictive_policies():
    from strands_acp_client._agents import resolve_agents

    strands = KNOWN_AGENTS["strands"]
    assert strands.argv_for("allow_all") == strands.argv
    assert strands.argv_for("read_only")[-2:] == ("--builtin-tools", "read,web_fetch")
    assert strands.argv_for("deny_all")[-2:] == ("--builtin-tools", "read,web_fetch")
    # A hand-written argv under the same name does not inherit flags written for the stock command.
    assert resolve_agents({"strands": ["strands", "--acp-server", "--agent", "./mine"]})["strands"].policy_args is None
    with pytest.raises(ValueError):
        AgentCommand(("",))


async def test_every_task_tells_the_agent_its_working_directory(make_tool, run_tool, tmp_path):
    built = make_tool()
    _, default = await run_tool(built, task="note")
    assert result_text(default) == f"Working directory: {os.path.abspath(tmp_path)}"
    other = tmp_path / "nested"
    other.mkdir()
    _, elsewhere = await run_tool(built, task="note", cwd=str(other))
    assert result_text(elsewhere) == f"Working directory: {os.path.abspath(other)}"


def test_vendor_notifications_do_not_log_errors(caplog):
    import logging

    from acp import RequestError

    from strands_acp_client._session import _quiet_vendor_notifications

    _quiet_vendor_notifications()
    _quiet_vendor_notifications()  # installing twice must not stack filters
    root = logging.getLogger()
    assert sum(type(f).__name__ == "_VendorNotificationFilter" for f in root.filters) == 1

    message = "Unhandled error while handling notification method=%s"
    with caplog.at_level(logging.DEBUG):
        root.error(message, "qwen/notify/session/mode-update", exc_info=RequestError.method_not_found("x"))
        root.error(message, "session/update", exc_info=ValueError("a real bug in a handler"))
        root.error("something else went wrong")
    errors = [record.getMessage() for record in caplog.records if record.levelno >= logging.ERROR]
    assert errors == [
        "Unhandled error while handling notification method=session/update",
        "something else went wrong",
    ]


def test_refused_calls_are_named_with_their_kind():
    from strands_acp_client._client import _describe_call

    assert _describe_call("calc.py", "edit") == "calc.py (edit)"
    assert _describe_call("Edit calc.py", "edit") == "Edit calc.py"
    assert _describe_call(None, "execute") == "execute"
    assert _describe_call(None, None) == "a tool call"


# ── what Codex needed: an authentication step, self-sandboxed kinds, and stopping on a refusal ──


async def test_an_agent_that_requires_authentication_gets_the_configured_method(make_tool, run_tool):
    command = AgentCommand(FAKE_ARGV, "fake", auth_method="fake-key")
    built = make_tool(agents={"fake": command}, env={"FAKE_ACP_REQUIRE_AUTH": "fake-key"})
    _, result = await run_tool(built, task="hello")
    assert result["status"] == "success"
    assert result_text(result) == "You said: hello"


async def test_authentication_without_a_usable_method_is_explained(make_tool, run_tool):
    env = {"FAKE_ACP_REQUIRE_AUTH": "fake-key"}
    _, missing = await run_tool(make_tool(env=env), task="hello")
    assert missing["status"] == "error"
    assert "requires authentication and no auth_method is configured" in result_text(missing)
    assert "it offers: fake-key" in result_text(missing)

    wrong = AgentCommand(FAKE_ARGV, "fake", auth_method="chatgpt")
    built = make_tool(agents={"fake": wrong}, env=env)
    _, refused = await run_tool(built, task="hello")
    assert refused["status"] == "error"
    assert "does not offer the configured method 'chatgpt'" in result_text(refused)
    assert len(built.acp_sessions) == 0


async def test_sandboxed_kinds_are_exempt_only_once_the_mode_is_active(make_tool, run_tool):
    command = AgentCommand(FAKE_ARGV, "fake", modes={"read_only": "ask"}, sandboxed_kinds=("execute",))
    with_mode = make_tool(agents={"fake": command}, env={"FAKE_ACP_MODES": "ask,auto"})
    _, reading = await run_tool(with_mode, task="sneak execute 0.2")
    assert reading["status"] == "success", "the agent sandboxes its own commands in this mode"
    _, editing = await run_tool(with_mode, task="sneak edit")
    assert result_json(editing)["stop_reason"] == "policy_violation", "an unasked edit is still a violation"

    # The agent does not offer the mode, so nothing guarantees the sandbox: commands are guarded again.
    without_mode = make_tool(agents={"fake": command})
    _, unguarded = await run_tool(without_mode, task="sneak execute")
    assert result_json(unguarded)["stop_reason"] == "policy_violation"


async def test_an_agent_that_stops_after_a_refusal_is_not_an_error(make_tool, run_tool):
    _, result = await run_tool(make_tool(), task="toolstop edit")
    assert result["status"] == "success"
    assert result_json(result)["stop_reason"] == "cancelled"
    text = result_text(result)
    assert text.startswith("The agent stopped after a permission request was refused.")
    assert text.endswith("Refused under the read_only policy: edit example.txt.")


def test_with_env_and_the_codex_preset():
    codex = KNOWN_AGENTS["codex"]
    assert codex.auth_method == "api-key"
    assert codex.sandboxed_kinds == ("execute",)
    assert codex.modes == {"read_only": "read-only", "deny_all": "read-only", "custom": "read-only"}
    assert codex.env == {"NO_BROWSER": "1"}

    local = codex.with_env(MODEL_PROVIDER="ollama", NO_BROWSER=None)
    assert local.env == {"NO_BROWSER": None, "MODEL_PROVIDER": "ollama"}
    assert local.argv == codex.argv and local.modes == codex.modes and local.auth_method == "api-key"
    assert codex.env == {"NO_BROWSER": "1"}, "with_env must not change the original"

    kilo = KNOWN_AGENTS["kilo"]
    assert kilo.modes == {"read_only": "ask", "deny_all": "ask"}
    assert kilo.config == {"model": "kilo/kilo-auto/free"}, "the default model needs an account; this one does not"
    assert kilo.sandboxed_kinds == ("execute",), "in ask mode Kilo filters shell commands itself"


# ── session options over ACP: what Kilo needed to run on a free model ──


async def test_session_options_are_set_and_reported(make_tool, run_tool):
    command = AgentCommand(FAKE_ARGV, "fake", config={"model": "small-free"})
    built = make_tool(agents={"fake": command}, env={"FAKE_ACP_CONFIG_MODELS": "big-paid,small-free"})
    _, result = await run_tool(built, task="model")
    assert result_text(result) == "model small-free"
    assert result_json(result)["config"] == {"model": "small-free"}

    untouched = make_tool(env={"FAKE_ACP_CONFIG_MODELS": "big-paid,small-free"})
    _, default = await run_tool(untouched, task="model")
    assert result_text(default) == "model big-paid"
    assert "config" not in result_json(default)


async def test_an_option_or_value_the_agent_does_not_offer_is_an_error(make_tool, run_tool):
    env = {"FAKE_ACP_CONFIG_MODELS": "big-paid,small-free"}
    typo = make_tool(agents={"fake": AgentCommand(FAKE_ARGV, "fake", config={"model": "smal-free"})}, env=env)
    _, result = await run_tool(typo, task="model")
    assert result["status"] == "error"
    assert "does not offer 'smal-free' for 'model'; choices: big-paid, small-free" in result_text(result)
    assert len(typo.acp_sessions) == 0, "a session that could not be configured is not kept"

    unknown = make_tool(agents={"fake": AgentCommand(FAKE_ARGV, "fake", config={"effort": "high"})}, env=env)
    _, result = await run_tool(unknown, task="model")
    assert result["status"] == "error"
    assert "has no session option 'effort'; it offers: model" in result_text(result)


def test_with_config_adds_options_and_keeps_what_is_known():
    base = AgentCommand(FAKE_ARGV, "fake", modes={"read_only": "ask"}, config={"model": "a"})
    tuned = base.with_config(model="b", effort="low")
    assert tuned.config == {"model": "b", "effort": "low"}
    assert tuned.modes == base.modes and tuned.argv == base.argv
    assert base.config == {"model": "a"}


async def test_a_vague_permission_request_is_judged_by_the_calls_known_kind(make_tool, run_tool):
    seen = []

    def handler(request):
        seen.append(request.kind)
        return request.kind == "read"

    built = make_tool(permissions=handler)
    _, read = await run_tool(built, task="vague read")
    assert "permission allowed" in result_text(read)
    _, edit = await run_tool(built, task="vague edit")
    assert "permission denied" in result_text(edit)
    assert seen == ["read", "edit"], "the request said 'other'; the tool call had already said what it was"

    # read_only follows the same rule: a search or read announced earlier is allowed, an edit is not.
    _, allowed = await run_tool(make_tool(), task="vague search")
    assert "permission allowed" in result_text(allowed)


async def test_the_working_directory_is_resolved_to_its_real_path(make_tool, run_tool, tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    _, result = await run_tool(make_tool(cwd=link), task="note")
    assert result_text(result) == f"Working directory: {os.path.realpath(real)}"
    _, per_call = await run_tool(make_tool(), task="note", cwd=str(link))
    assert result_text(per_call) == f"Working directory: {os.path.realpath(real)}"
