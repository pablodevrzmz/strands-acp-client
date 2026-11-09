"""The tool's own event loop: sessions that outlive a call's loop, and how the tool is closed."""

from __future__ import annotations

import asyncio
import gc
import threading
import time
from typing import Any

from strands import Agent
from strands.tools.decorator import DecoratedFunctionTool
from strands.types.tools import AgentTool, ToolContext

from strands_acp_client import AcpAgentTool, SessionRegistry, make_acp_agent
from strands_acp_client import _registry as registry_module
from strands_acp_client import _runtime as runtime_module
from strands_acp_client import tool as tool_module
from tests.conftest import FAKE_ARGV, eventually, is_running, read_pid, result_json, result_text
from tests.scripted_model import ScriptedModel


def _last_tool_result(agent: Agent) -> dict[str, Any]:
    for message in reversed(agent.messages):
        for block in message["content"]:
            if "toolResult" in block:
                return dict(block["toolResult"])
    raise AssertionError("the agent never called the tool")


def test_the_factory_returns_a_typed_strands_tool(tmp_path):
    built = make_acp_agent(agents={"fake": FAKE_ARGV}, cwd=tmp_path)
    assert isinstance(built, AcpAgentTool) and isinstance(built, AgentTool)
    assert isinstance(built, DecoratedFunctionTool), "the type Strands looks for when it loads tools by name"
    assert isinstance(built.acp_sessions, SessionRegistry)
    assert built.tool_type == "function"
    assert built.get_display_properties() == {"Name": "acp_agent", "Type": "function", "Function": "acp_agent"}
    built.close()  # nothing was started: closing is a no-op, and so is closing twice
    built.close()
    assert not built._runtime.running, "a tool that was never called starts no thread"


def test_strands_finds_the_tool_by_module_name(run_tool):
    """``Agent(tools=["package.module:name"])``, which is how a configuration file names a tool."""
    from tests import module_level_tool

    agent = Agent(model=ScriptedModel("acp_agent"), tools=["tests.module_level_tool:acp_agent"], callback_handler=None)
    assert agent.tool_names == ["acp_agent"]
    assert agent.tool_registry.registry["acp_agent"] is module_level_tool.acp_agent
    try:
        _, result = asyncio.run(run_tool(module_level_tool.acp_agent, task="turn"))
        assert result["status"] == "success" and result_text(result) == "turn 1"
    finally:
        module_level_tool.acp_agent.close()


def test_the_tool_is_not_rebound_when_kept_on_a_class(tmp_path):
    built = make_acp_agent(agents={"fake": FAKE_ARGV}, cwd=tmp_path)

    class Holder:
        acp_agent = built

    assert Holder().acp_agent is built, "a decorated method is bound to its instance; this tool is not a method"


# ── sessions outlive the event loop of the call that started them ──


def test_a_session_carries_over_between_synchronous_agent_calls(tmp_path):
    """The README quick start, twice. Strands runs every ``agent(...)`` on a new event loop."""
    built = make_acp_agent(agents={"fake": FAKE_ARGV}, cwd=tmp_path, timeout=10)
    try:
        model = ScriptedModel(built.tool_name)
        agent = Agent(model=model, tools=[built], callback_handler=None)

        model.calls.append({"task": "turn"})
        agent("Ask the coding agent which turn this is.")
        first = _last_tool_result(agent)
        assert first["status"] == "success", result_text(first)
        assert result_text(first) == "turn 1"
        session_id = result_json(first)["session_id"]

        model.calls.append({"task": "turn", "session_id": session_id})
        agent("Ask it again, in the same session.")
        second = _last_tool_result(agent)
        assert second["status"] == "success", result_text(second)
        assert result_text(second) == "turn 2", "the second call must reach the process of the first"
        assert result_json(second)["session_id"] == session_id
        assert result_json(second)["resumed"] is False
        assert built.acp_sessions.session_ids == [session_id]
    finally:
        built.close()


def test_sessions_outlive_the_event_loop_of_a_call(tmp_path, run_tool):
    built = make_acp_agent(agents={"fake": FAKE_ARGV}, cwd=tmp_path, timeout=10)
    try:
        _, first = asyncio.run(run_tool(built, task="turn"))
        session_id = result_json(first)["session_id"]
        _, second = asyncio.run(run_tool(built, task="turn", session_id=session_id))
        assert result_text(second) == "turn 2"

        # And from another thread altogether, which is where Strands puts a synchronous call.
        from_thread: list[Any] = []
        worker = threading.Thread(
            target=lambda: from_thread.append(asyncio.run(run_tool(built, task="turn", session_id=session_id)))
        )
        worker.start()
        worker.join(20)
        assert result_text(from_thread[0][1]) == "turn 3"
        assert len(built.acp_sessions) == 1
    finally:
        built.close()


async def test_a_permission_handler_runs_on_the_loop_of_the_call(make_tool, run_tool):
    here = (threading.current_thread(), asyncio.get_running_loop())
    seen: list[Any] = []

    async def asynchronous(request):
        seen.append((threading.current_thread(), asyncio.get_running_loop()))
        return True

    def plain(request):
        seen.append((threading.current_thread(), asyncio.get_running_loop()))
        return False

    _, allowed = await run_tool(make_tool(permissions=asynchronous), task="tool edit")
    assert "permission allowed" in result_text(allowed)
    _, refused = await run_tool(make_tool(permissions=plain), task="tool edit")
    assert "permission denied" in result_text(refused)
    assert seen == [here, here], "application code runs where the application is, not on the tool's thread"


async def test_a_handler_that_raises_refuses_the_call(make_tool, run_tool):
    def broken(request):
        raise RuntimeError("the handler has a bug")

    _, result = await run_tool(make_tool(permissions=broken), task="tool edit")
    assert result["status"] == "success"
    assert "permission denied" in result_text(result)
    assert result_json(result)["permissions"]["denied"] == 1


# ── a call nobody waits for any more ──


async def test_a_call_whose_caller_stopped_listening_is_cancelled(make_tool, run_tool):
    built = make_tool(timeout=30)
    tool_use = {"toolUseId": "abandoned", "name": built.tool_name, "input": {"task": "slow 30"}}
    context = ToolContext(tool_use=tool_use, agent=None, invocation_state={})
    stream = built.stream(tool_use, {}, _tool_context=context)
    async for event in stream:
        if event.get("type") == "tool_stream":
            break
    await stream.aclose()

    session_id = built.acp_sessions.session_ids[0]
    started = time.perf_counter()
    _, after = await run_tool(built, task="turn", session_id=session_id)
    assert result_text(after) == "turn 2"
    assert time.perf_counter() - started < 5, "the abandoned turn was left running instead of being cancelled"


async def test_closing_the_tool_ends_a_call_in_flight(make_tool):
    built = make_tool(timeout=30)
    tool_use = {"toolUseId": "in-flight", "name": built.tool_name, "input": {"task": "slow 30"}}
    context = ToolContext(tool_use=tool_use, agent=None, invocation_state={})
    closing: asyncio.Future[None] | None = None
    result = None
    started = time.perf_counter()
    async for event in built.stream(tool_use, {}, _tool_context=context):
        if event.get("type") == "tool_stream" and closing is None:
            closing = asyncio.ensure_future(built.aclose())
        if "tool_result" in event:
            result = event["tool_result"]
    assert closing is not None
    await closing
    assert time.perf_counter() - started < 10
    assert result is not None and result["status"] == "error"
    assert "closed while the agent was working" in result_text(result)
    assert len(built.acp_sessions) == 0


# ── closing ──


def test_with_block_closes_the_tool_and_it_can_be_used_again(tmp_path, run_tool, pid_file):
    env = {"FAKE_ACP_PID_FILE": pid_file}
    with make_acp_agent(agents={"fake": FAKE_ARGV}, cwd=tmp_path, env=env) as built:
        _, first = asyncio.run(run_tool(built, task="turn"))
        assert first["status"] == "success"
        first_pid = read_pid(pid_file)
        thread = built._runtime._thread
        assert is_running(first_pid) and thread is not None and thread.is_alive()
    assert not is_running(first_pid)
    assert not thread.is_alive(), "close() also stops the tool's background thread"
    assert len(built.acp_sessions) == 0

    _, again = asyncio.run(run_tool(built, task="turn"))
    assert again["status"] == "success" and result_text(again) == "turn 1"
    assert read_pid(pid_file) != first_pid
    built.close()
    assert not is_running(read_pid(pid_file))


async def test_async_with_closes_without_blocking_the_loop(tmp_path, run_tool, pid_file):
    async with make_acp_agent(agents={"fake": FAKE_ARGV}, cwd=tmp_path, env={"FAKE_ACP_PID_FILE": pid_file}) as built:
        _, result = await run_tool(built, task="turn")
        assert result["status"] == "success"
        pid = read_pid(pid_file)
    assert not is_running(pid)
    assert not built._runtime.running


async def test_one_session_can_be_closed_by_id(make_tool, run_tool):
    built = make_tool()
    _, first = await run_tool(built, task="turn")
    _, second = await run_tool(built, task="turn")
    first_id, second_id = result_json(first)["session_id"], result_json(second)["session_id"]
    await built.acp_sessions.close(first_id)
    await built.acp_sessions.close("never-seen")
    assert built.acp_sessions.session_ids == [second_id]
    _, gone = await run_tool(built, task="turn", session_id=first_id)
    assert gone["status"] == "error" and "cannot resume sessions" in result_text(gone)


async def test_a_tool_that_is_dropped_ends_its_processes(tmp_path, run_tool, pid_file):
    built = make_acp_agent(agents={"fake": FAKE_ARGV}, cwd=tmp_path, env={"FAKE_ACP_PID_FILE": pid_file})
    await run_tool(built, task="turn")
    pid = read_pid(pid_file)
    assert is_running(pid)
    del built
    gc.collect()
    assert await eventually(lambda: not is_running(pid)), "a tool nobody closed left its agent running"


async def test_a_forked_child_starts_afresh_and_leaves_the_parents_agents_alone(make_tool, run_tool, pid_file):
    built = make_tool(env={"FAKE_ACP_PID_FILE": pid_file})
    await run_tool(built, task="turn")
    parents_pid = read_pid(pid_file)
    parents_loop, parents_thread = built._runtime._loop, built._runtime._thread
    parents_sessions = list(built.acp_sessions._sessions.values())
    assert parents_loop is not None and parents_thread is not None
    try:
        tool_module._forget_in_forked_child()  # what a child process runs right after a fork
        assert len(built.acp_sessions) == 0
        _, fresh = await run_tool(built, task="turn")
        assert fresh["status"] == "success" and result_text(fresh) == "turn 1"
        assert read_pid(pid_file) != parents_pid
        assert is_running(parents_pid), "the parent's agent is not the child's to stop"
    finally:
        # This test is its own parent: put the first runtime's process and thread away by hand.
        for session in parents_sessions:
            asyncio.run_coroutine_threadsafe(session.close(), parents_loop).result(10)
        parents_loop.call_soon_threadsafe(parents_loop.stop)
        parents_thread.join(10)
        registry_module._PARENTS_SESSIONS.clear()
        runtime_module._ORPHANED_LOOPS.clear()
    assert not is_running(parents_pid)
