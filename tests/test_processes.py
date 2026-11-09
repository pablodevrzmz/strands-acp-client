"""The agent process itself: what it writes to stderr, how long it may take to start, how it ends."""

from __future__ import annotations

import asyncio
import shlex
import sys
import threading
import time

import pytest

from strands_acp_client import _process
from tests.conftest import FAKE_ARGV, eventually, is_running, read_pid, result_json, result_text

# ── what it writes ──


async def test_stderr_is_read_so_a_chatty_agent_never_blocks(make_tool, run_tool):
    # 512 KiB is several times what a pipe holds. Unread, the agent would stall inside the write.
    built = make_tool(timeout=20)
    started = time.perf_counter()
    _, result = await run_tool(built, task="noisy 512")
    assert result["status"] == "success", result_text(result)
    assert result_text(result) == "wrote 512 KiB to stderr"
    assert time.perf_counter() - started < 10


async def test_a_message_longer_than_one_buffered_line_arrives_whole(make_tool, run_tool):
    _, result = await run_tool(make_tool(), task="big 300")
    assert result["status"] == "success"
    assert result_text(result) == "y" * (300 * 1024)


# ── how it starts ──


async def test_an_agent_that_never_answers_is_stopped_at_startup(make_tool, run_tool, pid_file):
    env = {
        "FAKE_ACP_HANG": "initialize",
        "FAKE_ACP_PID_FILE": pid_file,
        "FAKE_ACP_STDERR": "waiting for a login that never comes",
    }
    built = make_tool(startup_timeout=1.5, env=env)
    started = time.perf_counter()
    _, result = await run_tool(built, task="hello")
    assert time.perf_counter() - started < 10
    assert result["status"] == "error"
    text = result_text(result)
    assert "did not finish starting within 1.5s" in text
    assert "waiting for a login that never comes" in text, "what the agent said on stderr explains the wait"
    assert len(built.acp_sessions) == 0
    assert not is_running(read_pid(pid_file)), "the agent that never answered must not be left running"


async def test_a_session_that_never_opens_is_stopped_too(make_tool, run_tool, pid_file):
    built = make_tool(startup_timeout=1.5, env={"FAKE_ACP_HANG": "new_session", "FAKE_ACP_PID_FILE": pid_file})
    _, result = await run_tool(built, task="hello")
    assert result["status"] == "error"
    assert "did not finish starting within 1.5s" in result_text(result)
    assert not is_running(read_pid(pid_file))


async def test_an_agent_that_dies_at_startup_is_quoted(make_tool, run_tool):
    built = make_tool(env={"FAKE_ACP_STDERR": "error: you are not logged in", "FAKE_ACP_EXIT": "7"})
    _, result = await run_tool(built, task="hello")
    assert result["status"] == "error"
    text = result_text(result)
    assert "agent 'fake' failed during setup" in text
    assert "the process exited with code 7" in text
    assert "error: you are not logged in" in text
    assert len(built.acp_sessions) == 0


async def test_cancelling_while_the_agent_starts_stops_it(make_tool, run_tool, pid_file):
    built = make_tool(env={"FAKE_ACP_HANG": "initialize", "FAKE_ACP_PID_FILE": pid_file})
    cancel_signal = threading.Event()
    timer = threading.Timer(1.0, cancel_signal.set)
    timer.start()
    started = time.perf_counter()
    try:
        _, result = await run_tool(built, task="hello", cancel_signal=cancel_signal)
    finally:
        timer.cancel()
    assert time.perf_counter() - started < 8, (
        "the startup timeout is two minutes; the cancellation must not wait for it"
    )
    assert result["status"] == "error"
    assert "cancelled by the caller while the agent was starting" in result_text(result)
    assert len(built.acp_sessions) == 0
    assert not is_running(read_pid(pid_file))


# ── how it ends ──


async def test_a_crash_mid_turn_says_how_the_agent_died(make_tool, run_tool):
    built = make_tool()
    _, result = await run_tool(built, task="crash")
    assert result["status"] == "error"
    text = result_text(result)
    assert "the agent failed mid-turn" in text
    assert "the process exited with code 3" in text
    assert "out of cheese" in text, "the agent's last words on stderr belong in the error"
    details = result_json(result)
    assert details["session_id"] is None and details["session_closed"] is True
    assert len(built.acp_sessions) == 0


async def test_an_agent_that_ignores_cancellation_is_ended(make_tool, run_tool, pid_file):
    built = make_tool(timeout=0.3, cancel_grace=0.3, env={"FAKE_ACP_PID_FILE": pid_file})
    started = time.perf_counter()
    _, result = await run_tool(built, task="stubborn 30")
    assert time.perf_counter() - started < 8
    assert result["status"] == "error"
    assert "did not react to cancellation, so its process was ended" in result_text(result)
    details = result_json(result)
    assert details["stop_reason"] == "timeout"
    assert details["session_id"] is None and details["session_closed"] is True, "there is nothing left to continue"
    assert len(built.acp_sessions) == 0
    assert not is_running(read_pid(pid_file))


@pytest.mark.skipif(sys.platform == "win32", reason="process groups are a POSIX notion")
@pytest.mark.parametrize("wrapped", [True, False], ids=["behind-a-wrapper", "on-its-own"])
async def test_closing_ends_an_agent_that_will_not_leave(make_tool, run_tool, pid_file, monkeypatch, wrapped):
    monkeypatch.setattr(_process, "EXIT_GRACE", 0.3)
    # This agent stays when its stdin closes and ignores SIGTERM. Behind a wrapper (a shell that
    # starts it and waits, the way npx starts node) a signal to the wrapper alone would leave it
    # running for good; only a signal to the whole process group reaches it.
    command = " ".join(shlex.quote(part) for part in FAKE_ARGV)
    argv = ("sh", "-c", f"{command}; exit $?") if wrapped else FAKE_ARGV
    env = {"FAKE_ACP_PID_FILE": pid_file, "FAKE_ACP_IGNORE_EOF": "1", "FAKE_ACP_IGNORE_TERM": "1"}
    built = make_tool(agents={"fake": argv}, env=env)
    _, result = await run_tool(built, task="hello")
    assert result["status"] == "success", result_text(result)
    agent_pid = read_pid(pid_file)
    assert is_running(agent_pid)

    await built.acp_sessions.close_all()
    assert len(built.acp_sessions) == 0
    assert await eventually(lambda: not is_running(agent_pid)), "the agent was left running after close"


# ── found by the review of the first version ──


async def test_a_cancellation_while_a_setup_error_is_reported_still_stops_the_agent(
    make_tool, run_tool, pid_file, monkeypatch
):
    cancel_signal = threading.Event()
    last_words = _process.AgentProcess.last_words

    async def interrupted(self, wait=1.0):
        cancel_signal.set()  # the caller gives up while the error is being put together
        return await last_words(self, wait)

    monkeypatch.setattr(_process.AgentProcess, "last_words", interrupted)
    env = {"FAKE_ACP_LOAD_SESSION": "1", "FAKE_ACP_LOAD_FAIL": "1", "FAKE_ACP_PID_FILE": pid_file}
    built = make_tool(env=env)
    _, result = await run_tool(built, task="turn", session_id="earlier", cancel_signal=cancel_signal)
    assert result["status"] == "error"
    assert await eventually(lambda: not is_running(read_pid(pid_file))), "the agent outlived its failed start"


async def test_an_agent_that_stops_reading_its_stdin_cannot_hold_a_turn(make_tool, run_tool, pid_file, monkeypatch):
    monkeypatch.setattr(_process, "EXIT_GRACE", 0.3)
    built = make_tool(timeout=1, cancel_grace=0.5, env={"FAKE_ACP_PID_FILE": pid_file})
    _, first = await run_tool(built, task="wedge 60")
    assert result_text(first) == "wedging"
    await asyncio.sleep(0.5)  # its event loop is now blocked, and nothing reads its stdin

    started = time.perf_counter()
    _, stuck = await run_tool(built, task="x" * 300_000, session_id=result_json(first)["session_id"])
    assert time.perf_counter() - started < 8, "a task larger than the pipe must not outlast timeout and grace"
    assert stuck["status"] == "error"
    assert result_json(stuck)["stop_reason"] == "timeout"
    assert await eventually(lambda: not is_running(read_pid(pid_file)))

    started = time.perf_counter()
    await built.aclose()
    assert time.perf_counter() - started < 3
