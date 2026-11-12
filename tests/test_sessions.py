"""Which session a call gets: continuing by id, resuming, sharing, and what is closed when."""

from __future__ import annotations

import asyncio
import os
import signal
import threading
import time
from typing import Any

from strands.types.tools import ToolContext

from tests.conftest import FAKE_ARGV, eventually, is_running, read_pid, result_json, result_text

# ── continuing a session ──


async def test_a_follow_up_needs_only_the_session_id(make_tool, run_tool, tmp_path):
    built = make_tool(agents={"first": FAKE_ARGV, "second": FAKE_ARGV})
    _, started = await run_tool(built, task="turn", agent_name="second")
    session_id = result_json(started)["session_id"]

    # The model rarely repeats agent_name on a follow-up. The session knows whose it is.
    _, followed = await run_tool(built, task="turn", session_id=session_id)
    assert followed["status"] == "success", result_text(followed)
    assert result_text(followed) == "turn 2"
    assert result_json(followed)["agent"] == "second"

    _, wrong_agent = await run_tool(built, task="turn", session_id=session_id, agent_name="first")
    assert wrong_agent["status"] == "error"
    assert "belongs to agent 'second', not 'first'" in result_text(wrong_agent)

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    _, wrong_cwd = await run_tool(built, task="turn", session_id=session_id, cwd=str(elsewhere))
    assert wrong_cwd["status"] == "error"
    assert "cannot move to" in result_text(wrong_cwd)

    # Neither mistake cost the session.
    _, still = await run_tool(built, task="turn", session_id=session_id, agent_name="second", cwd=str(tmp_path))
    assert result_text(still) == "turn 3"
    assert built.acp_sessions.session_ids == [session_id]


async def test_a_session_stays_in_the_directory_it_started_in(make_tool, run_tool, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    built = make_tool()
    _, first = await run_tool(built, task="note", cwd=str(elsewhere))
    _, second = await run_tool(built, task="note", session_id=result_json(first)["session_id"])
    assert result_text(second) == f"Working directory: {os.path.realpath(elsewhere)}"


async def test_empty_optional_inputs_mean_the_defaults(make_tool, run_tool):
    # Models fill optional parameters with empty strings more often than they leave them out.
    _, result = await run_tool(make_tool(), task="turn", agent_name="", session_id="", cwd="")
    assert result["status"] == "success", result_text(result)
    assert result_text(result) == "turn 1"


# ── resuming a session whose process is gone ──


async def test_an_evicted_session_is_resumed_with_its_own_agent(make_tool, run_tool):
    env = {"FAKE_ACP_LOAD_SESSION": "1", "FAKE_ACP_LOADED_TURNS": "7"}
    built = make_tool(agents={"first": FAKE_ARGV, "second": FAKE_ARGV}, env=env, max_sessions=1)
    _, started = await run_tool(built, task="turn", agent_name="second")
    session_id = result_json(started)["session_id"]
    await run_tool(built, task="turn")  # a session of the default agent takes the only place
    assert session_id not in built.acp_sessions.session_ids

    _, back = await run_tool(built, task="turn", session_id=session_id)
    assert back["status"] == "success", result_text(back)
    details = result_json(back)
    assert details["agent"] == "second", "resumed by the agent that owns it, not by the default one"
    assert details["resumed"] is True and details["session_id"] == session_id
    assert result_text(back) == "turn 8"


async def test_two_calls_resuming_one_session_share_one_process(make_tool, run_tool):
    built = make_tool(env={"FAKE_ACP_LOAD_SESSION": "1", "FAKE_ACP_LOADED_TURNS": "4"})
    results = await asyncio.gather(*(run_tool(built, task="turn", session_id="earlier") for _ in range(3)))
    assert sorted(result_text(result) for _, result in results) == ["turn 5", "turn 6", "turn 7"]
    assert built.acp_sessions.session_ids == ["earlier"]


async def test_a_session_whose_process_died_is_noticed(make_tool, run_tool, pid_file):
    built = make_tool(env={"FAKE_ACP_PID_FILE": pid_file})
    _, first = await run_tool(built, task="turn")
    pid = read_pid(pid_file)
    session_id = result_json(first)["session_id"]
    session = built.acp_sessions._sessions[session_id]
    os.kill(pid, signal.SIGKILL)
    assert await eventually(lambda: not is_running(pid))
    assert await eventually(lambda: not session.alive), "the tool's loop never heard that the process died"

    _, after = await run_tool(built, task="turn", session_id=session_id)
    assert after["status"] == "error"
    assert "cannot resume sessions; start a new one" in result_text(after)
    assert len(built.acp_sessions) == 0


# ── several calls at once ──


async def test_calls_on_one_session_take_turns(make_tool, run_tool):
    built = make_tool()
    _, first = await run_tool(built, task="turn")
    session_id = result_json(first)["session_id"]
    results = await asyncio.gather(*(run_tool(built, task="turn", session_id=session_id) for _ in range(3)))
    assert sorted(result_text(result) for _, result in results) == ["turn 2", "turn 3", "turn 4"]
    assert built.acp_sessions.session_ids == [session_id]


async def test_sessions_in_use_are_never_evicted(make_tool, run_tool):
    built = make_tool(max_sessions=1)
    results = await asyncio.gather(run_tool(built, task="slow 1.5"), run_tool(built, task="slow 1.5"))
    for _, result in results:
        assert result["status"] == "success", result_text(result)
        assert "done sleeping" in result_text(result)
    assert len(built.acp_sessions) == 1, "back under the limit once both calls are over"
    # The session that was given up to the limit is not offered for follow-ups; the other one is.
    offered = [result_json(result)["session_id"] for _, result in results]
    assert built.acp_sessions.session_ids[0] in offered
    assert None in offered


async def test_a_cancelled_call_waiting_for_a_busy_session_returns_at_once(make_tool, run_tool):
    built = make_tool(timeout=30)
    _, first = await run_tool(built, task="turn")
    session_id = result_json(first)["session_id"]

    working = asyncio.Event()

    async def long_turn() -> dict[str, Any]:
        tool_use = {"toolUseId": "busy", "name": built.tool_name, "input": {"task": "slow 2", "session_id": session_id}}
        context = ToolContext(tool_use=tool_use, agent=None, invocation_state={})
        result: dict[str, Any] = {}
        async for event in built.stream(tool_use, {}, _tool_context=context):
            if event.get("type") == "tool_stream":
                working.set()
            if "tool_result" in event:
                result = event["tool_result"]
        return result

    busy = asyncio.ensure_future(long_turn())
    await asyncio.wait_for(working.wait(), 15)
    cancel_signal = threading.Event()
    cancel_signal.set()
    started = time.perf_counter()
    _, waiting = await run_tool(built, task="turn", session_id=session_id, cancel_signal=cancel_signal)
    assert time.perf_counter() - started < 1.5, "it must not wait for the turn in front of it"
    assert waiting["status"] == "error"
    assert result_json(waiting)["stop_reason"] == "cancelled"
    assert "before the agent started it" in result_text(waiting)

    finished = await busy
    assert finished["status"] == "success", "the turn in front was not disturbed"
    assert result_json(finished)["session_id"] == session_id


# ── what a result offers for follow-ups ──


async def test_a_closed_session_is_not_offered_for_follow_ups(make_tool, run_tool):
    _, gone = await run_tool(make_tool(keep_sessions=False), task="turn")
    assert gone["status"] == "success"
    details = result_json(gone)
    assert details["session_id"] is None and details["session_closed"] is True

    # An agent that can reload a session by id is another matter: the id still works.
    reloading = make_tool(keep_sessions=False, env={"FAKE_ACP_LOAD_SESSION": "1"})
    _, kept = await run_tool(reloading, task="turn")
    details = result_json(kept)
    assert details["session_id"] and "session_closed" not in details
    assert len(reloading.acp_sessions) == 0
    _, again = await run_tool(reloading, task="turn", session_id=details["session_id"])
    assert again["status"] == "success" and result_json(again)["resumed"] is True


async def test_a_session_does_not_outlive_a_policy_violation(make_tool, run_tool):
    built = make_tool()
    _, result = await run_tool(built, task="sneak edit")
    details = result_json(result)
    assert details["stop_reason"] == "policy_violation"
    assert details["session_id"] is None and details["session_closed"] is True
    assert len(built.acp_sessions) == 0


async def test_a_turn_waiting_behind_a_policy_violation_does_not_run(make_tool, run_tool):
    built = make_tool()
    _, first = await run_tool(built, task="turn")
    session_id = result_json(first)["session_id"]
    # Both calls reach the session in this order; the second waits for the first one's turn to end.
    sneaking = asyncio.ensure_future(run_tool(built, task="sneak edit 2", session_id=session_id))
    await asyncio.sleep(0)
    waiting = asyncio.ensure_future(run_tool(built, task="turn", session_id=session_id))
    (_, violated), (_, behind) = await sneaking, await waiting
    assert result_json(violated)["stop_reason"] == "policy_violation"
    assert behind["status"] == "error", "the agent that broke the policy must not get another turn"
    assert "turn 3" not in result_text(behind)
    assert "closed" in result_text(behind)
    assert len(built.acp_sessions) == 0
