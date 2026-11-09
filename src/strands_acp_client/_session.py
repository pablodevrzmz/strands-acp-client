"""One agent process holding one ACP session, and what a single prompt to it produces."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import threading
import time
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

from acp import PROTOCOL_VERSION, RequestError, connect_to_agent
from acp.schema import ClientCapabilities, Implementation, TextContentBlock

from strands_acp_client._agents import AgentCommand
from strands_acp_client._client import StrandsAcpClient, Turn
from strands_acp_client._permissions import PermissionPolicy, policy_label
from strands_acp_client._process import EXIT_GRACE, AgentProcess
from strands_acp_client._runtime import CallCancelled, unless_cancelled

if TYPE_CHECKING:
    from acp.client.connection import ClientSideConnection

__all__ = ["AcpAgentError", "AcpSession", "SessionSpec", "TurnResult"]

logger = logging.getLogger(__name__)

try:
    _VERSION = version("strands-acp-client")
except PackageNotFoundError:  # pragma: no cover - source checkout without metadata
    _VERSION = "0.0.0"

_CLIENT_INFO = Implementation(name="strands-acp-client", title="Strands ACP client tool", version=_VERSION)
_AUTH_REQUIRED = -32000
_POLL = 0.25
"""Seconds between two looks at the clock and at the caller's cancellation signal during a turn."""


class _VendorNotificationFilter(logging.Filter):
    """Drop the SDK's error record for a notification this client has no handler for.

    Agents send notifications of their own outside the ACP namespace (Qwen Code emits
    ``qwen/notify/session/mode-update``). ACP lets a receiver ignore unknown notifications, but
    the Python SDK logs each one on the root logger as "Unhandled error while handling
    notification" with a traceback. That is noise in an application's log, so it is turned
    into a debug line here; every other record passes untouched.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not str(record.msg).startswith("Unhandled error while handling notification"):
            return True
        exc = record.exc_info[1] if record.exc_info else None
        if isinstance(exc, RequestError) and exc.code == -32601:
            method = record.args[0] if isinstance(record.args, tuple) and record.args else ""
            logger.debug("ignored vendor notification %s", method)
            return False
        return True


_vendor_filter = _VendorNotificationFilter()


def _quiet_vendor_notifications() -> None:
    root = logging.getLogger()
    if _vendor_filter not in root.filters:
        root.addFilter(_vendor_filter)


class AcpAgentError(Exception):
    """The agent could not be started, resumed or driven through a turn."""


@dataclass(frozen=True)
class SessionSpec:
    """What it takes to start one agent process and open a session in it."""

    agent_name: str
    command: AgentCommand
    cwd: str
    policy: PermissionPolicy
    env: Mapping[str, str | None] | None = None
    strict: bool = True
    cancel_grace: float = 10.0
    startup_timeout: float = 120.0


@dataclass
class TurnResult:
    """Outcome of one prompt, ready to be turned into a tool result."""

    text: str
    stop_reason: str
    usage: dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    allowed: int = 0
    denied: int = 0
    thought_chars: int = 0
    elapsed: float = 0.0
    error: str | None = None
    violation: dict[str, Any] | None = None
    refused: tuple[str, ...] = ()
    stderr: str = ""


class AcpSession:
    """One agent subprocess holding one ACP session.

    Everything here runs on the tool's own event loop (see ``Runtime``), whichever loop the tool
    call came from.
    """

    def __init__(self, spec: SessionSpec) -> None:
        self.spec = spec
        self.agent_name = spec.agent_name
        self.command = spec.command
        self.cwd = spec.cwd
        self.policy_label = policy_label(spec.policy)
        self.client = StrandsAcpClient(spec.agent_name, spec.policy, strict=spec.strict)
        self.session_id: str = ""
        self.mode: str | None = None
        self.config: dict[str, str] = {}
        self.resumed = False
        self.agent_info: dict[str, Any] | None = None
        self.supports_load = False
        self.last_used = time.monotonic()
        # Kept by the registry: how many tool calls hold this session, and whether it is to be
        # closed once the last of them lets go.
        self.leases = 0
        self.retired = False
        self._conn: ClientSideConnection | None = None
        self._process: AgentProcess | None = None
        self._lock = asyncio.Lock()
        self._closing: asyncio.Future[None] | None = None
        self._closed = False

    # ── lifecycle ──

    async def start(self, session_id: str | None = None) -> None:
        """Spawn the agent, negotiate ACP v1 and open (or resume) a session.

        All of it has ``startup_timeout`` seconds. An agent that never answers, because it is
        waiting for a login in a browser or is still downloading itself, is stopped and reported
        instead of holding the tool call for ever.
        """
        _quiet_vendor_notifications()
        try:
            await asyncio.wait_for(self._start(session_id), self.spec.startup_timeout)
        except BaseException as exc:
            # The process is stopped whatever happens next, including a cancellation that arrives
            # while the error message is being put together: a handler that raises skips its siblings.
            try:
                error = await self._start_error(exc)
            finally:
                await self.close()
            if error is None:
                raise
            if error is exc:
                raise
            raise error from (None if isinstance(exc, asyncio.TimeoutError) else exc)

    async def _start_error(self, exc: BaseException) -> AcpAgentError | None:
        """The error to report for a failed start, or ``None`` to let ``exc`` (a cancellation) through."""
        if isinstance(exc, AcpAgentError):
            return exc
        if isinstance(exc, asyncio.TimeoutError):
            waited = f"did not finish starting within {self.spec.startup_timeout:g}s"
            return AcpAgentError(await self._with_last_words(f"agent {self.agent_name!r} {waited}", wait=0))
        if isinstance(exc, Exception):
            return AcpAgentError(await self._with_last_words(f"agent {self.agent_name!r} failed during setup: {exc}"))
        return None

    async def _start(self, session_id: str | None) -> None:
        argv = self.command.argv_for(self.policy_label)
        try:
            self._process = await AgentProcess.spawn(argv, cwd=self.cwd, env=self._environment(), label=self.agent_name)
        except FileNotFoundError:
            if not os.path.isdir(self.cwd):
                raise AcpAgentError(
                    f"cannot start agent {self.agent_name!r}: {self.cwd!r} is not a directory"
                ) from None
            raise AcpAgentError(f"cannot start agent {self.agent_name!r}: {argv[0]!r} was not found on PATH") from None
        except OSError as exc:
            raise AcpAgentError(f"cannot start agent {self.agent_name!r}: {exc}") from exc
        self._conn = conn = connect_to_agent(self.client, self._process.stdin, self._process.stdout)

        init = await conn.initialize(
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

        if session_id is not None and not self.supports_load:
            raise AcpAgentError(
                f"session {session_id!r} is not held by this process and agent {self.agent_name!r} "
                "cannot resume sessions; start a new one"
            )
        try:
            opened = await self._open(conn, session_id)
        except RequestError as exc:
            if exc.code != _AUTH_REQUIRED:
                raise
            await self._authenticate(conn, init.auth_methods or [])
            opened = await self._open(conn, session_id)
        await self._select_mode(conn, opened)
        await self._apply_config(conn, opened)
        if self.mode is not None:
            # The agent is in the mode where it confines these kinds itself; see AgentCommand.
            self.client.exempt_kinds = frozenset(self.command.sandboxed_kinds)

    def _environment(self) -> dict[str, str]:
        """This process's environment with the agent's and the tool's changes applied, in that order."""
        env = dict(os.environ)
        for changes in (self.command.env, self.spec.env):
            for key, value in (changes or {}).items():
                if value is None:
                    env.pop(key, None)
                else:
                    env[key] = value
        return env

    async def _open(self, conn: ClientSideConnection, session_id: str | None) -> Any:
        if session_id is None:
            created = await conn.new_session(cwd=self.cwd, mcp_servers=[])
            self.session_id = created.session_id
            return created
        loaded = await conn.load_session(cwd=self.cwd, session_id=session_id, mcp_servers=[])
        self.session_id = session_id
        self.resumed = True
        return loaded

    async def _authenticate(self, conn: ClientSideConnection, offered: list[Any]) -> None:
        """Run the ACP authentication step with the method configured for this agent."""
        ids = [str(method.id) for method in offered]
        method = self.command.auth_method
        if method is None:
            raise AcpAgentError(
                f"agent {self.agent_name!r} requires authentication and no auth_method is configured for it; "
                f"it offers: {', '.join(ids) or 'none'}"
            )
        if method not in ids:
            raise AcpAgentError(
                f"agent {self.agent_name!r} requires authentication but does not offer the configured method "
                f"{method!r}; it offers: {', '.join(ids) or 'none'}"
            )
        await conn.authenticate(method_id=method)

    async def _select_mode(self, conn: ClientSideConnection, opened: Any) -> None:
        """Switch the session to the mode this agent needs for the current policy, if it offers it."""
        wanted = (self.command.modes or {}).get(self.policy_label)
        if not wanted or opened is None:
            return
        modes = getattr(opened, "modes", None)
        if modes is not None and any(mode.id == wanted for mode in modes.available_modes):
            if modes.current_mode_id != wanted:
                await conn.set_session_mode(session_id=self.session_id, mode_id=wanted)
            self.mode = wanted
            return
        for option in getattr(opened, "config_options", None) or []:
            if getattr(option, "id", None) == "mode" and wanted in _select_values(option):
                if getattr(option, "current_value", None) != wanted:
                    await conn.set_config_option(config_id="mode", session_id=self.session_id, value=wanted)
                self.mode = wanted
                return
        logger.info("agent %r does not offer mode %r; relying on its own defaults", self.agent_name, wanted)

    async def _apply_config(self, conn: ClientSideConnection, opened: Any) -> None:
        """Set the session options configured for this agent (its model, for instance)."""
        wanted = self.command.config or {}
        if not wanted:
            return
        offered = {str(getattr(option, "id", "")): option for option in getattr(opened, "config_options", None) or []}
        for option_id, value in wanted.items():
            option = offered.get(option_id)
            if option is None:
                raise AcpAgentError(
                    f"agent {self.agent_name!r} has no session option {option_id!r}; "
                    f"it offers: {', '.join(sorted(offered)) or 'none'}"
                )
            choices = _select_values(option)
            if choices and value not in choices:
                shown = ", ".join(choices[:12]) + (f" and {len(choices) - 12} more" if len(choices) > 12 else "")
                raise AcpAgentError(
                    f"agent {self.agent_name!r} does not offer {value!r} for {option_id!r}; choices: {shown}"
                )
            if getattr(option, "current_value", None) != value:
                await conn.set_config_option(config_id=option_id, session_id=self.session_id, value=value)
            self.config[option_id] = value

    async def _with_last_words(self, message: str, wait: float = 1.0) -> str:
        """Add to ``message`` how the process ended and what it last said on stderr, when that is known."""
        if self._process is None:
            return message
        note, stderr = await self._process.last_words(wait)
        if note:
            message += f" ({note})"
        if stderr:
            message += f". Its last output on stderr:\n{stderr}"
        return message

    @property
    def alive(self) -> bool:
        return not self._closed and self._process is not None and self._process.returncode is None

    def touch(self) -> None:
        self.last_used = time.monotonic()

    async def close(self) -> None:
        """Close the connection and end the agent process.

        Every caller waits for the same shutdown, and the shutdown carries on if a caller is
        cancelled while waiting for it.
        """
        if self._closing is None:
            self._closed = True
            self._closing = asyncio.ensure_future(self._close())
        await asyncio.shield(self._closing)

    async def _close(self) -> None:
        # The process first. Closing the connection waits for its writer, and the writer can be stuck
        # for good on an agent that stopped reading stdin; once the process is gone that write fails.
        if self._process is not None:
            await self._process.stop()
        if self._conn is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._conn.close(), EXIT_GRACE)

    def kill(self) -> None:
        """End the agent process at once, without an event loop: the last resort when the loop is gone."""
        self._closed = True
        if self._process is not None:
            self._process.kill()

    # ── one turn ──

    async def prompt(
        self,
        text: str,
        *,
        timeout: float,
        cancel_signal: threading.Event | None = None,
        home: asyncio.AbstractEventLoop | None = None,
    ) -> AsyncGenerator[dict[str, Any] | TurnResult, None]:
        """Send one prompt and stream progress events; the last item yielded is the ``TurnResult``.

        Turns on one session run one after another: a second call waits for the first to finish.
        ``home`` is the event loop of the tool call, where a custom permission handler is run.
        """
        conn = self._conn
        if conn is None:
            raise AcpAgentError("session was not started")
        try:
            await unless_cancelled(self._lock.acquire(), cancel_signal)
        except CallCancelled:
            yield TurnResult("", "cancelled", error="the task was cancelled by the caller before the agent started it")
            return
        try:
            if not self.alive:
                yield TurnResult("", "error", error="the session was closed before the agent could start the task")
                return
            self.touch()
            turn = Turn(home)
            self.client.turn = turn
            loop = asyncio.get_running_loop()
            started = loop.time()
            deadline = started + timeout
            # The agent knows its working directory from session/new, but its model often does not:
            # one that cannot run ``pwd`` (a read-only toolset) has no way to turn the relative paths
            # in a task into the absolute ones its file tools want. So every task says where it is.
            note = TextContentBlock(text=f"Working directory: {self.cwd}")
            request: asyncio.Future[Any] = asyncio.ensure_future(
                conn.prompt(session_id=self.session_id, prompt=[TextContentBlock(text=text), note])
            )
            pending: asyncio.Future[dict[str, Any]] | None = None
            stopped: str | None = None
            try:
                while stopped is None:
                    if pending is None:
                        pending = asyncio.ensure_future(turn.events.get())
                    wait = min(_POLL, max(deadline - loop.time(), 0.0))
                    waiting: set[asyncio.Future[Any]] = {request, pending}
                    done, _ = await asyncio.wait(waiting, timeout=wait, return_when=asyncio.FIRST_COMPLETED)
                    if pending in done:
                        event, pending = pending.result(), None
                        yield event
                    if turn.violation is not None:
                        stopped = "policy_violation"
                    elif request.done():
                        break
                    elif cancel_signal is not None and cancel_signal.is_set():
                        stopped = "cancelled"
                    elif loop.time() >= deadline:
                        stopped = "timeout"
                if stopped is not None and not request.done():
                    await self._cancel_turn(conn, request)
                if pending is not None:
                    # A waiter that is cancelled before it wakes up leaves its event in the queue.
                    if pending.done() and not pending.cancelled():
                        yield pending.result()
                    else:
                        pending.cancel()
                    pending = None
                while not turn.events.empty():
                    yield turn.events.get_nowait()
                yield await self._finish(turn, request, stopped, loop.time() - started, timeout)
            finally:
                if pending is not None:
                    pending.cancel()
                if not request.done():
                    # The consumer stopped listening (an error upstream, or the call was cancelled):
                    # do not leave the coding agent working for nobody.
                    await self._cancel_turn(conn, request)
                self.client.turn = None
                self.touch()
        finally:
            self._lock.release()

    async def _cancel_turn(self, conn: ClientSideConnection, request: asyncio.Future[Any]) -> None:
        """Ask the agent to stop, give it a grace period, then end the process if it ignores us.

        The grace period covers sending the cancellation too: an agent that stopped reading its
        stdin never receives it, and must not hold the turn past its deadline.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.spec.cancel_grace
        with contextlib.suppress(Exception):
            await asyncio.wait_for(conn.cancel(session_id=self.session_id), self.spec.cancel_grace)
        done, _ = await asyncio.wait({request}, timeout=max(deadline - loop.time(), 0.0))
        if request not in done:
            logger.warning(
                "agent %r ignored session/cancel for %gs; ending its process", self.agent_name, self.spec.cancel_grace
            )
            request.cancel()
            with contextlib.suppress(BaseException):
                await request
            await self.close()

    async def _finish(
        self, turn: Turn, request: asyncio.Future[Any], stopped: str | None, elapsed: float, timeout: float
    ) -> TurnResult:
        usage: dict[str, Any] | None = None
        error: str | None = None
        stderr = ""
        stop_reason = "error"
        finished = request.done() and not request.cancelled() and request.exception() is None
        if finished:
            response = request.result()
            stop_reason = str(response.stop_reason)
            if response.usage is not None:
                usage = response.usage.model_dump(exclude_none=True, by_alias=False)

        if stopped == "timeout":
            stop_reason = "timeout"
            reaction = "was cancelled" if finished else "did not react to cancellation, so its process was ended"
            error = f"the agent did not finish within {timeout:.0f}s and {reaction}"
        elif stopped == "cancelled":
            stop_reason = "cancelled"
            error = "the task was cancelled by the caller"
        elif stopped == "policy_violation":
            stop_reason = "policy_violation"
            violation = turn.violation or {}
            error = (
                f"the agent ran {violation.get('title') or 'a tool'!r} ({violation.get('kind')}) without asking "
                f"for permission, so the {self.policy_label} policy could not be enforced and the turn was "
                "stopped; that change may already have been made"
            )
        elif not finished:
            exc = request.exception() if request.done() and not request.cancelled() else None
            if isinstance(exc, RequestError):
                error = _describe_failure(exc)
            elif self._closed:
                error = "the session was closed while the agent was working"
            else:
                # No answer and no error from the agent: its end of the pipe closed. Say how it died.
                error = _describe_failure(exc)
                if self._process is not None:
                    note, stderr = await self._process.last_words()
                    if note:
                        error += f" ({note})"
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
            refused=tuple(turn.refused),
            stderr=stderr,
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
        if exc.code == _AUTH_REQUIRED:
            return f"the agent requires authentication: {exc}"
        return f"the agent returned an error (code {exc.code}): {exc}"
    return f"the agent failed mid-turn: {exc}"
