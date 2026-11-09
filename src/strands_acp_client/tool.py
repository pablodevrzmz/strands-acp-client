"""The ``acp_agent`` tool: delegate a task to a coding agent over the Agent Client Protocol."""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import os
import threading
import weakref
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import Any

from strands import tool
from strands.tools.decorator import DecoratedFunctionTool
from strands.types.tools import ToolContext

from strands_acp_client._agents import AgentCommand, AgentsArg, resolve_agents
from strands_acp_client._permissions import PermissionPolicy, policy_label, validate_policy
from strands_acp_client._registry import SessionRegistry
from strands_acp_client._runtime import CallCancelled, Runtime, RuntimeClosed, unless_cancelled
from strands_acp_client._session import AcpAgentError, AcpSession, SessionSpec, TurnResult

__all__ = ["AcpAgentTool", "make_acp_agent"]

_SUCCESS_STOP_REASONS = frozenset({"end_turn", "max_tokens", "max_turn_requests"})

_POLICY_NOTES: dict[str, str] = {
    "read_only": "the agent can read, search and fetch, but every edit, delete or command it proposes is refused, "
    "and a change made without asking stops the task with an error",
    "allow_all": "every permission the agent asks for is granted, so it can edit files and run commands",
    "deny_all": "every permission the agent asks for is refused",
    "custom": "permissions are decided by a handler supplied by the application",
}


class AcpAgentTool(DecoratedFunctionTool[..., Any]):
    """The tool :func:`make_acp_agent` builds: a Strands function tool, and the agent processes behind it.

    Give it to ``Agent(tools=[...])`` like any other tool. The agent processes it starts stay
    alive between calls so that a ``session_id`` can be continued. :attr:`acp_sessions` lists
    them, and :meth:`close` ends them, as does leaving a ``with`` block::

        with make_acp_agent(agents=["claude"]) as acp_agent:
            Agent(tools=[acp_agent])("Ask claude what src/http.py does.")

    A tool that is never closed ends its processes when it is garbage collected or when the
    interpreter exits.
    """

    acp_sessions: SessionRegistry
    """The live sessions of this tool."""

    def __init__(self, decorated: DecoratedFunctionTool[..., Any], registry: SessionRegistry, runtime: Runtime) -> None:
        # This object takes over the state of the tool that ``@tool`` built, instead of building
        # it again through a constructor whose arguments are the SDK's own business. It has to be
        # a ``DecoratedFunctionTool`` and not a wrapper around one: that is the type Strands looks
        # for when tools are named by module ("my_app.tools:acp_agent") or loaded from a file.
        self.__dict__.update(vars(decorated))
        self._runtime = runtime
        self.acp_sessions = registry
        _OPEN_TOOLS.add(self)

    def __del__(self) -> None:
        # Dropped without ``close()``. Whoever triggered the collection must not be held up, so the
        # loop is told to end its processes and then itself. At interpreter exit ``_close_at_exit``
        # has already closed everything by the time this runs.
        try:
            self._runtime.abandon(self.acp_sessions.drop_all)
        except Exception:  # half-built, or the interpreter is being torn down
            pass

    def __get__(self, instance: Any, obj_type: type | None = None) -> AcpAgentTool:
        # The function behind this tool is not a method. Kept as a class attribute, the tool must
        # not be re-bound to whichever instance reaches it, which is what a decorated method does.
        return self

    def close(self) -> None:
        """End every agent process and stop the tool's background thread.

        Blocks until they are gone, which takes a few seconds at most; in async code await
        :meth:`aclose` instead. The tool can be used again afterwards.
        """
        self._runtime.stop(self.acp_sessions.drop_all)
        self.acp_sessions.kill_all()  # anything the loop did not get to

    async def aclose(self) -> None:
        """:meth:`close`, without holding up the event loop that awaits it."""
        await asyncio.to_thread(self.close)

    def __enter__(self) -> AcpAgentTool:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self.close()

    async def __aenter__(self) -> AcpAgentTool:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        await self.aclose()


_OPEN_TOOLS: weakref.WeakSet[AcpAgentTool] = weakref.WeakSet()


@atexit.register
def _close_at_exit() -> None:
    """End the agent processes of every tool still open when the interpreter exits."""
    for built in list(_OPEN_TOOLS):
        with contextlib.suppress(Exception):
            built.close()


def _forget_in_forked_child() -> None:
    """Start afresh in a forked child: the loop thread did not survive the fork, and the agent
    processes belong to the parent, which is still using them."""
    for built in list(_OPEN_TOOLS):
        built._runtime.forget()
        built.acp_sessions.forget()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forget_in_forked_child)


def make_acp_agent(
    *,
    agents: AgentsArg = None,
    default_agent: str | None = None,
    cwd: str | os.PathLike[str] | None = None,
    timeout: float = 600.0,
    startup_timeout: float = 120.0,
    permissions: PermissionPolicy = "read_only",
    strict: bool = True,
    env: Mapping[str, str | None] | None = None,
    keep_sessions: bool = True,
    max_sessions: int = 8,
    cancel_grace: float = 10.0,
    name: str = "acp_agent",
    description: str | None = None,
) -> AcpAgentTool:
    """Build a tool that runs ACP coding agents as subprocesses and returns what they answer.

    Args:
        agents: Which agents the model may pick: a mapping of name to launch argv (or to a
            :class:`~strands_acp_client.AgentCommand`, or to a known-agent alias), a list of
            known-agent names, or ``None`` for Claude Code only. See ``KNOWN_AGENTS``.
        default_agent: Agent used when the model does not pick one. Defaults to the first configured.
        cwd: Working directory the agents are started in. Defaults to the current directory.
            The model may pass another directory per call.
        timeout: Seconds a single task may run before the turn is cancelled.
        startup_timeout: Seconds an agent may take to start and open its session. A first run
            through ``npx`` downloads the agent, which is why the default is generous.
        permissions: How the agent's ``session/request_permission`` calls are answered:
            ``"read_only"`` (default), ``"allow_all"``, ``"deny_all"`` or a callable that receives a
            :class:`~strands_acp_client.PermissionRequest` and returns ``True``, ``False`` or an option id.
            A callable runs on the event loop of the tool call it answers for.
        strict: Under any policy but ``allow_all``, stop the turn and return an error when the agent
            completes an edit, delete, move or command it never asked permission for. A policy can only
            answer questions; this is what catches an agent that is configured not to ask.
        env: Environment changes for every agent process, applied over this process's environment.
            A value of ``None`` removes the variable.
        keep_sessions: Keep agent processes alive after a call so the model can continue the
            conversation with ``session_id``. ``False`` closes the process after every call.
        max_sessions: Maximum number of live sessions; the least recently used idle one is closed first.
        cancel_grace: Seconds to wait for the agent to honour a cancellation before its process is ended.
        name: Tool name shown to the model.
        description: Replace the generated tool description.

    Returns:
        An :class:`AcpAgentTool`. Its live sessions are reachable as ``tool.acp_sessions`` (a
        :class:`~strands_acp_client.SessionRegistry`), and ``tool.close()`` ends every agent process.
    """
    resolved = resolve_agents(agents)
    validate_policy(permissions)
    default = default_agent or next(iter(resolved))
    if default not in resolved:
        raise ValueError(f"default_agent {default!r} is not one of the configured agents: {', '.join(resolved)}")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if startup_timeout <= 0:
        raise ValueError("startup_timeout must be positive")
    if cancel_grace < 0:
        raise ValueError("cancel_grace must not be negative")
    # The real path, not a symlinked one: agents compare the files they touch with the session's
    # directory, and a path that only looks different sends them asking about an "external" folder.
    base_cwd = os.path.realpath(os.fspath(cwd)) if cwd is not None else os.path.realpath(os.getcwd())
    if not os.path.isdir(base_cwd):
        raise ValueError(f"cwd {base_cwd!r} is not a directory")

    settings = _Settings(
        agents=resolved,
        default_agent=default,
        base_cwd=base_cwd,
        timeout=timeout,
        startup_timeout=startup_timeout,
        permissions=permissions,
        strict=strict,
        env=env,
        keep_sessions=keep_sessions,
        cancel_grace=cancel_grace,
    )
    runtime = Runtime(f"strands-acp-client:{name}")
    registry = SessionRegistry(runtime, max_sessions=max_sessions)
    delegator = _Delegator(settings, registry, runtime)

    async def acp_agent(
        task: str,
        agent_name: str | None = None,
        session_id: str | None = None,
        cwd: str | None = None,
        *,
        tool_context: ToolContext,
    ) -> AsyncGenerator[Any, None]:
        cancel_signal = getattr(tool_context, "cancel_signal", None)
        async with contextlib.aclosing(delegator.stream(task, agent_name, session_id, cwd, cancel_signal)) as items:
            async for item in items:
                yield item

    decorated = tool(
        name=name,
        description=description or _describe(settings),
        inputSchema={"json": _input_schema(settings)},
        context="tool_context",
    )(acp_agent)
    decorated.tool_spec = {**decorated.tool_spec, "annotations": _annotations(settings.policy)}
    return AcpAgentTool(decorated, registry, runtime)


# ── one call ──


@dataclass(frozen=True)
class _Settings:
    """What :func:`make_acp_agent` was asked for, once checked."""

    agents: Mapping[str, AgentCommand]
    default_agent: str
    base_cwd: str
    timeout: float
    startup_timeout: float
    permissions: PermissionPolicy
    strict: bool
    env: Mapping[str, str | None] | None
    keep_sessions: bool
    cancel_grace: float

    @property
    def policy(self) -> str:
        return policy_label(self.permissions)


@dataclass(frozen=True)
class _Call:
    """One tool call, on its way from the caller's event loop to the tool's own."""

    task: str
    agent_name: str | None
    session_id: str | None
    cwd: str | None
    cancel_signal: threading.Event | None
    home: asyncio.AbstractEventLoop


class _Delegator:
    """What a tool call does: check its inputs, find or start a session, run the turn, report."""

    def __init__(self, settings: _Settings, registry: SessionRegistry, runtime: Runtime) -> None:
        self._settings = settings
        self._registry = registry
        self._runtime = runtime

    async def stream(
        self,
        task: str,
        agent_name: str | None,
        session_id: str | None,
        cwd: str | None,
        cancel_signal: threading.Event | None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Progress events of the delegated task, then its tool result. Runs on the caller's loop."""
        agents = self._settings.agents
        if agent_name and agent_name not in agents:
            yield _error(f"unknown agent {agent_name!r}; configured agents: {', '.join(agents)}")
            return
        if not task or not task.strip():
            yield _error("task must not be empty")
            return
        work_dir = os.path.realpath(cwd) if cwd else None
        if work_dir is not None and not os.path.isdir(work_dir):
            yield _error(f"cwd {work_dir!r} is not a directory")
            return
        call = _Call(task, agent_name or None, session_id or None, work_dir, cancel_signal, asyncio.get_running_loop())
        try:
            async with contextlib.aclosing(self._runtime.stream(lambda: self._run(call))) as items:
                async for item in items:
                    yield item
        except RuntimeClosed:
            yield _error("the tool was closed while the agent was working")

    async def _run(self, call: _Call) -> AsyncGenerator[dict[str, Any], None]:
        """The same, where the agent processes live: on the tool's own loop."""
        settings = self._settings
        try:
            spec = self._spec_for(call)
        except AcpAgentError as exc:
            yield _error(str(exc))
            return

        acquiring = asyncio.ensure_future(self._registry.acquire(spec, call.session_id))
        try:
            session = await unless_cancelled(acquiring, call.cancel_signal)
        except CallCancelled:
            await self._give_back(acquiring)
            yield _error("the task was cancelled by the caller while the agent was starting")
            return
        except AcpAgentError as exc:
            yield _error(str(exc))
            return
        except BaseException:
            await self._give_back(acquiring)
            raise

        result: TurnResult | None = None
        try:
            turn = session.prompt(call.task, timeout=settings.timeout, cancel_signal=call.cancel_signal, home=call.home)
            async with contextlib.aclosing(turn) as items:
                async for item in items:
                    if isinstance(item, TurnResult):
                        result = item
                    else:
                        yield item
        finally:
            # A session outlives a turn that failed, timed out or was cancelled, as long as its agent
            # is still there to talk to. It does not outlive a change made behind the policy's back.
            keep = settings.keep_sessions and result is not None and result.violation is None
            await self._registry.release(session, keep)

        if result is None:
            yield _error(f"agent {session.agent_name!r} produced no result")
            return
        continuable = self._registry.holds(session) or session.supports_load
        yield _tool_result(session, settings.policy, result, continuable)

    def _spec_for(self, call: _Call) -> SessionSpec:
        """Settle which agent and directory a call means. A session it continues has both already."""
        settings = self._settings
        owner = self._registry.owner(call.session_id) if call.session_id else None
        agent_name = call.agent_name or (owner.agent_name if owner else settings.default_agent)
        cwd = call.cwd or (owner.cwd if owner else settings.base_cwd)
        if owner is not None and agent_name != owner.agent_name:
            raise AcpAgentError(
                f"session {call.session_id!r} belongs to agent {owner.agent_name!r}, not {agent_name!r}; leave "
                f"agent_name out to continue it, or leave session_id out to start a new session with {agent_name!r}"
            )
        if owner is not None and cwd != owner.cwd:
            raise AcpAgentError(
                f"session {call.session_id!r} works in {owner.cwd} and cannot move to {cwd}; leave cwd out to "
                "continue it, or leave session_id out to start a new session there"
            )
        return SessionSpec(
            agent_name=agent_name,
            command=settings.agents[agent_name],
            cwd=cwd,
            policy=settings.permissions,
            env=settings.env,
            strict=settings.strict,
            cancel_grace=settings.cancel_grace,
            startup_timeout=settings.startup_timeout,
        )

    async def _give_back(self, acquiring: asyncio.Future[AcpSession]) -> None:
        """Release a session that was leased for a call which is not going to use it after all."""
        if acquiring.done() and not acquiring.cancelled() and acquiring.exception() is None:
            await self._registry.release(acquiring.result())


# ── model-facing surface ──


def _describe(settings: _Settings) -> str:
    agents = "; ".join(f"{n} ({c.description})" if c.description else n for n, c in settings.agents.items())
    lines = [
        "Delegate a task to a coding agent that speaks the Agent Client Protocol and wait for its answer. "
        "The agent runs on this machine with its own tools, in a working directory, and returns a final message. "
        "Give it a complete, self-contained task.",
        f"Agents: {agents}.",
        f"Default agent: {settings.default_agent}. Default working directory: {settings.base_cwd}.",
        f"Permissions: {settings.policy} ({_POLICY_NOTES[settings.policy]}).",
        "The result includes a session_id: pass it back, on its own, to continue the same conversation with "
        "follow-up tasks.",
    ]
    return " ".join(lines)


def _input_schema(settings: _Settings) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "What the coding agent should do, with the context it needs to work on its own.",
            },
            "agent_name": {
                "type": "string",
                "enum": list(settings.agents),
                "description": f"Which agent to delegate to. Defaults to {settings.default_agent}. Leave it out "
                "when continuing a session: the session already has its agent.",
            },
            "session_id": {
                "type": "string",
                "description": "A session_id returned by an earlier call, to continue that conversation "
                "instead of starting a new one.",
            },
            "cwd": {
                "type": "string",
                "description": f"Working directory for the agent. Defaults to {settings.base_cwd}. Leave it out "
                "when continuing a session: the session stays in its own directory.",
            },
        },
        "required": ["task"],
    }


def _annotations(label: str) -> dict[str, object]:
    return {
        "title": "ACP coding agent",
        "readOnlyHint": label in ("read_only", "deny_all"),
        "destructiveHint": label in ("allow_all", "custom"),
        "idempotentHint": False,
        "openWorldHint": True,
    }


# ── results ──


def _tool_result(session: AcpSession, label: str, result: TurnResult, continuable: bool) -> dict[str, Any]:
    # Some agents (Codex) end the turn as "cancelled" when a permission is refused. That is the
    # policy doing its job, not a failure of the delegation.
    stopped_on_refusal = result.stop_reason == "cancelled" and result.error is None and bool(result.refused)
    finished = result.stop_reason in _SUCCESS_STOP_REASONS or stopped_on_refusal
    status = "success" if result.error is None and finished else "error"
    text = result.text.strip()
    if stopped_on_refusal:
        text = "The agent stopped after a permission request was refused." + (f"\n\n{text}" if text else "")
    elif result.error is not None:
        text = f"{result.error}." + (f"\n\nPartial output:\n{text}" if text else "")
        if result.stderr:
            text += f"\n\nLast output of the agent on stderr:\n{result.stderr}"
    elif result.stop_reason == "refusal":
        text = "The agent refused the task." + (f"\n\n{text}" if text else "")
    elif result.stop_reason in ("max_tokens", "max_turn_requests"):
        text = f"The agent stopped early ({result.stop_reason})." + (f"\n\n{text}" if text else "")
    elif not text:
        text = "The agent finished without returning any text."
    if result.refused:
        # Agents do not always say that they were refused; the caller needs to know either way.
        shown = "; ".join(result.refused[:5])
        more = f"; and {len(result.refused) - 5} more" if len(result.refused) > 5 else ""
        text += f"\n\nRefused under the {label} policy: {shown}{more}."
    details: dict[str, Any] = {
        "agent": session.agent_name,
        "agent_info": session.agent_info,
        # Only an id that a follow-up can use. A closed session whose agent cannot reload it is over.
        "session_id": session.session_id if continuable else None,
        "resumed": session.resumed,
        "mode": session.mode,
        "stop_reason": result.stop_reason,
        "usage": result.usage,
        "tool_calls": result.tool_calls,
        "permissions": {"policy": label, "allowed": result.allowed, "denied": result.denied},
        "elapsed_seconds": round(result.elapsed, 2),
    }
    if not continuable:
        details["session_closed"] = True
    if session.config:
        details["config"] = dict(session.config)
    if result.refused:
        details["refused"] = list(result.refused)
    if result.violation is not None:
        details["policy_violation"] = result.violation
    return {"status": status, "content": [{"text": text}, {"json": details}]}


def _error(message: str) -> dict[str, Any]:
    return {"status": "error", "content": [{"text": message}]}
