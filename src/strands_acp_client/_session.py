"""Agent processes, ACP sessions and the registry that keeps them alive between tool calls."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from acp import PROTOCOL_VERSION, RequestError, spawn_agent_process
from acp.schema import ClientCapabilities, Implementation, TextContentBlock

from strands_acp_client._agents import AgentCommand
from strands_acp_client._client import StrandsAcpClient, Turn
from strands_acp_client._permissions import PermissionPolicy, policy_label

__all__ = ["AcpAgentError", "AcpSession", "SessionRegistry", "TurnResult"]

logger = logging.getLogger(__name__)

try:
    _VERSION = version("strands-acp-client")
except PackageNotFoundError:  # pragma: no cover - source checkout without metadata
    _VERSION = "0.0.0"

_CLIENT_INFO = Implementation(name="strands-acp-client", title="Strands ACP client tool", version=_VERSION)


class AcpAgentError(Exception):
    """The agent could not be started, resumed or driven through a turn."""


@dataclass
class TurnResult:
    """Outcome of one prompt, ready to be turned into a tool result."""

    text: str
    stop_reason: str
    usage: dict[str, Any] | None
    tool_calls: list[dict[str, Any]]
    allowed: int
    denied: int
    thought_chars: int
    elapsed: float
    error: str | None = None
    violation: dict[str, Any] | None = None


class AcpSession:
    """One agent subprocess holding one ACP session."""

    def __init__(
        self,
        agent_name: str,
        command: AgentCommand,
        cwd: str,
        policy: PermissionPolicy,
        env: Mapping[str, str | None] | None = None,
        cancel_grace: float = 10.0,
        strict: bool = True,
    ) -> None:
        self.agent_name = agent_name
        self.command = command
        self.cwd = cwd
        self.cancel_grace = cancel_grace
        self.policy_label = policy_label(policy)
        self.client = StrandsAcpClient(agent_name, policy, strict=strict)
        self.session_id: str = ""
        self.mode: str | None = None
        self.resumed = False
        self.agent_info: dict[str, Any] | None = None
        self.supports_load = False
        self.loop: asyncio.AbstractEventLoop | None = None
        self.last_used = time.monotonic()
        self._env = env
        self._stack: AsyncExitStack | None = None
        self._conn: Any = None
        self._proc: Any = None
        self._lock: asyncio.Lock | None = None
        self._closed = False

    # ── lifecycle ──

    async def start(self, session_id: str | None = None) -> None:
        """Spawn the agent, negotiate ACP v1 and open (or resume) a session."""
        self.loop = asyncio.get_running_loop()
        self._lock = asyncio.Lock()
        env = dict(os.environ)
        for changes in (self.command.env, self._env):
            for key, value in (changes or {}).items():
                if value is None:
                    env.pop(key, None)
                else:
                    env[key] = value

        stack = AsyncExitStack()
        try:
            self._conn, self._proc = await stack.enter_async_context(
                spawn_agent_process(self.client, *self.command.argv_for(self.policy_label), cwd=self.cwd, env=env)
            )
        except FileNotFoundError:
            await stack.aclose()
            raise AcpAgentError(
                f"cannot start agent {self.agent_name!r}: {self.command.argv[0]!r} was not found on PATH"
            ) from None
        except Exception as exc:
            await stack.aclose()
            raise AcpAgentError(f"cannot start agent {self.agent_name!r}: {exc}") from exc
        self._stack = stack

        try:
            init = await self._conn.initialize(
                protocol_version=PROTOCOL_VERSION,
                client_capabilities=ClientCapabilities(),
                client_info=_CLIENT_INFO,
            )
            if init.protocol_version != PROTOCOL_VERSION:
                raise AcpAgentError(
                    f"agent {self.agent_name!r} answered with ACP protocol version {init.protocol_version}; "
                    f"this client speaks version {PROTOCOL_VERSION}"
                )
            if init.agent_info is not None:
                self.agent_info = {"name": init.agent_info.name, "version": init.agent_info.version}
            self.supports_load = bool(init.agent_capabilities and init.agent_capabilities.load_session)

            if session_id is None:
                opened = await self._conn.new_session(cwd=self.cwd, mcp_servers=[])
                self.session_id = opened.session_id
            else:
                if not self.supports_load:
                    raise AcpAgentError(
                        f"session {session_id!r} is not held by this process and agent {self.agent_name!r} "
                        "cannot resume sessions; start a new one"
                    )
                opened = await self._conn.load_session(cwd=self.cwd, session_id=session_id, mcp_servers=[])
                self.session_id = session_id
                self.resumed = True
            await self._select_mode(opened)
        except AcpAgentError:
            await self.close()
            raise
        except Exception as exc:
            await self.close()
            raise AcpAgentError(f"agent {self.agent_name!r} failed during setup: {exc}") from exc

    async def _select_mode(self, opened: Any) -> None:
        """Switch the session to the mode this agent needs for the current policy, if it offers it."""
        wanted = (self.command.modes or {}).get(self.policy_label)
        if not wanted or opened is None:
            return
        modes = getattr(opened, "modes", None)
        if modes is not None and any(mode.id == wanted for mode in modes.available_modes):
            if modes.current_mode_id != wanted:
                await self._conn.set_session_mode(session_id=self.session_id, mode_id=wanted)
            self.mode = wanted
            return
        for option in getattr(opened, "config_options", None) or []:
            if getattr(option, "id", None) == "mode" and wanted in _select_values(option):
                if getattr(option, "current_value", None) != wanted:
                    await self._conn.set_config_option(config_id="mode", session_id=self.session_id, value=wanted)
                self.mode = wanted
                return
        logger.info("agent %r does not offer mode %r; relying on its own defaults", self.agent_name, wanted)

    @property
    def alive(self) -> bool:
        return not self._closed and self._proc is not None and self._proc.returncode is None

    @property
    def busy(self) -> bool:
        return self._lock is not None and self._lock.locked()

    def touch(self) -> None:
        self.last_used = time.monotonic()

    async def close(self) -> None:
        """Close the connection and terminate the agent process."""
        if self._closed:
            return
        self._closed = True
        stack, self._stack = self._stack, None
        if stack is not None:
            with contextlib.suppress(Exception):
                await stack.aclose()
        self.kill()

    def kill(self) -> None:
        """Best-effort synchronous termination, for sessions bound to a loop that is gone."""
        self._closed = True
        proc = self._proc
        if proc is not None and proc.returncode is None:
            with contextlib.suppress(ProcessLookupError, OSError):
                proc.kill()

    # ── one turn ──

    async def prompt(
        self,
        text: str,
        *,
        timeout: float,
        cancel_signal: Any = None,
    ) -> AsyncIterator[dict[str, Any] | TurnResult]:
        """Send one prompt and stream progress events; the last item yielded is the ``TurnResult``."""
        if self._lock is None or self._conn is None:
            raise AcpAgentError("session was not started")
        async with self._lock:
            self.touch()
            turn = Turn()
            self.client.turn = turn
            loop = asyncio.get_running_loop()
            started = loop.time()
            deadline = started + timeout
            # The agent knows its working directory from session/new, but its model often does not:
            # one that cannot run ``pwd`` (a read-only toolset) has no way to turn the relative paths
            # in a task into the absolute ones its file tools want. So every task says where it is.
            blocks = [TextContentBlock(text=text), TextContentBlock(text=f"Working directory: {self.cwd}")]
            task = asyncio.ensure_future(self._conn.prompt(session_id=self.session_id, prompt=blocks))
            cancel_reason: str | None = None
            try:
                while not task.done():
                    if turn.violation is not None:
                        cancel_reason = "policy_violation"
                        break
                    if cancel_signal is not None and cancel_signal.is_set():
                        cancel_reason = "cancelled"
                        break
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        cancel_reason = "timeout"
                        break
                    getter = asyncio.ensure_future(turn.events.get())
                    done, _ = await asyncio.wait(
                        {task, getter}, timeout=min(0.25, remaining), return_when=asyncio.FIRST_COMPLETED
                    )
                    if getter in done:
                        yield getter.result()
                    else:
                        getter.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await getter
                if cancel_reason is None and turn.violation is not None:
                    cancel_reason = "policy_violation"
                if cancel_reason is not None and not task.done():
                    await self._cancel_turn(task)
                while not turn.events.empty():
                    yield turn.events.get_nowait()
                yield self._finish(turn, task, cancel_reason, loop.time() - started, timeout)
            finally:
                if not task.done():
                    # The consumer stopped listening (an error upstream, or the agent loop was torn down):
                    # do not leave the coding agent working for nobody.
                    await self._cancel_turn(task)
                self.client.turn = None
                self.touch()

    async def _cancel_turn(self, task: asyncio.Future[Any]) -> None:
        """Ask the agent to stop, give it a grace period, then kill the process if it ignores us."""
        with contextlib.suppress(Exception):
            await self._conn.cancel(session_id=self.session_id)
        done, _ = await asyncio.wait({task}, timeout=self.cancel_grace)
        if task not in done:
            logger.warning("agent %r ignored session/cancel for %.0fs; killing it", self.agent_name, self.cancel_grace)
            task.cancel()
            with contextlib.suppress(BaseException):
                await task
            await self.close()

    def _finish(
        self, turn: Turn, task: asyncio.Future[Any], cancel_reason: str | None, elapsed: float, timeout: float
    ) -> TurnResult:
        usage: dict[str, Any] | None = None
        error: str | None = None
        stop_reason = "error"
        finished = task.done() and not task.cancelled() and task.exception() is None
        if finished:
            response = task.result()
            stop_reason = str(response.stop_reason)
            if response.usage is not None:
                usage = response.usage.model_dump(exclude_none=True, by_alias=False)

        if cancel_reason == "timeout":
            stop_reason = "timeout"
            reaction = "was cancelled" if finished else "did not react to cancellation"
            error = f"the agent did not finish within {timeout:.0f}s and {reaction}"
        elif cancel_reason == "cancelled":
            stop_reason = "cancelled"
            error = "the task was cancelled by the caller"
        elif cancel_reason == "policy_violation":
            stop_reason = "policy_violation"
            violation = turn.violation or {}
            error = (
                f"the agent ran {violation.get('title') or 'a tool'!r} ({violation.get('kind')}) without asking "
                f"for permission, so the {self.policy_label} policy could not be enforced and the turn was "
                "stopped; that change may already have been made"
            )
        elif not finished:
            exc = task.exception() if task.done() and not task.cancelled() else None
            error = _describe_failure(exc)
        return TurnResult(
            text=turn.text,
            stop_reason=stop_reason,
            usage=usage,
            tool_calls=list(turn.tool_calls.values()),
            allowed=turn.allowed,
            denied=turn.denied,
            thought_chars=turn.thought_chars,
            elapsed=elapsed,
            error=error,
            violation=turn.violation,
        )


def _select_values(option: Any) -> list[str]:
    """Values of a select config option, whether it lists options directly or in groups."""
    values: list[str] = []
    for item in getattr(option, "options", None) or []:
        nested = getattr(item, "options", None)
        if nested is not None:
            values.extend(str(inner.value) for inner in nested)
        else:
            values.append(str(item.value))
    return values


def _describe_failure(exc: BaseException | None) -> str:
    """Turn what broke the turn into one sentence the model can act on."""
    if exc is None:
        return "the agent failed mid-turn"
    if isinstance(exc, RequestError):
        if exc.code == -32000:
            return f"the agent requires authentication: {exc}"
        return f"the agent returned an error (code {exc.code}): {exc}"
    return f"the agent failed mid-turn: {exc}"


class SessionRegistry:
    """Live sessions of one tool, keyed by ACP session id.

    Sessions are bound to the event loop that created them. A session created in an
    earlier ``agent()`` call (Strands runs each synchronous call on its own loop) cannot
    be driven from a new loop: it is dropped, and the agent is asked to resume the session
    by id when it supports ``session/load``.
    """

    def __init__(self, max_sessions: int = 8) -> None:
        if max_sessions < 1:
            raise ValueError("max_sessions must be at least 1")
        self.max_sessions = max_sessions
        self._sessions: dict[str, AcpSession] = {}

    def __len__(self) -> int:
        return len(self._sessions)

    @property
    def session_ids(self) -> list[str]:
        return list(self._sessions)

    async def acquire(
        self,
        agent_name: str,
        command: AgentCommand,
        cwd: str,
        session_id: str | None,
        policy: PermissionPolicy,
        env: Mapping[str, str | None] | None,
        cancel_grace: float,
        strict: bool = True,
    ) -> AcpSession:
        loop = asyncio.get_running_loop()
        if session_id is not None:
            held = self._sessions.get(session_id)
            if held is not None:
                if held.loop is loop and held.alive and held.agent_name == agent_name:
                    held.touch()
                    return held
                await self._drop(held)
        session = AcpSession(agent_name, command, cwd, policy, env=env, cancel_grace=cancel_grace, strict=strict)
        await session.start(session_id=session_id)
        self._sessions[session.session_id] = session
        await self._evict()
        return session

    async def release(self, session: AcpSession, keep: bool) -> None:
        if not keep or not session.alive:
            await self._drop(session)

    async def close(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        if session is not None:
            await self._drop(session)

    async def close_all(self) -> None:
        for session in list(self._sessions.values()):
            await self._drop(session)

    async def _drop(self, session: AcpSession) -> None:
        self._sessions.pop(session.session_id, None)
        if session.loop is asyncio.get_running_loop():
            await session.close()
        else:
            session.kill()

    async def _evict(self) -> None:
        while len(self._sessions) > self.max_sessions:
            idle = [s for s in self._sessions.values() if not s.busy]
            if not idle:
                return
            await self._drop(min(idle, key=lambda s: s.last_used))
