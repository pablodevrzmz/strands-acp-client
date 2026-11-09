"""The ``acp_agent`` tool: delegate a task to a coding agent over the Agent Client Protocol."""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator, Mapping
from typing import Any

from strands import tool
from strands.tools.decorator import DecoratedFunctionTool
from strands.types.tools import ToolContext

from strands_acp_client._agents import AgentCommand, AgentsArg, resolve_agents
from strands_acp_client._permissions import PermissionPolicy, policy_label, validate_policy
from strands_acp_client._session import AcpAgentError, AcpSession, SessionRegistry, TurnResult

__all__ = ["make_acp_agent"]

_SUCCESS_STOP_REASONS = frozenset({"end_turn", "max_tokens", "max_turn_requests"})

_POLICY_NOTES: dict[str, str] = {
    "read_only": "the agent can read, search and fetch, but every edit, delete or command it proposes is refused, "
    "and a change made without asking stops the task with an error",
    "allow_all": "every permission the agent asks for is granted, so it can edit files and run commands",
    "deny_all": "every permission the agent asks for is refused",
    "custom": "permissions are decided by a handler supplied by the application",
}


def make_acp_agent(
    *,
    agents: AgentsArg = None,
    default_agent: str | None = None,
    cwd: str | os.PathLike[str] | None = None,
    timeout: float = 600.0,
    permissions: PermissionPolicy = "read_only",
    strict: bool = True,
    env: Mapping[str, str | None] | None = None,
    keep_sessions: bool = True,
    max_sessions: int = 8,
    cancel_grace: float = 10.0,
    name: str = "acp_agent",
    description: str | None = None,
) -> DecoratedFunctionTool[..., Any]:
    """Build a tool that runs ACP coding agents as subprocesses and returns what they answer.

    Args:
        agents: Which agents the model may pick: a mapping of name to launch argv (or to a
            :class:`~strands_acp_client.AgentCommand`, or to a known-agent alias), a list of
            known-agent names, or ``None`` for Claude Code only. See ``KNOWN_AGENTS``.
        default_agent: Agent used when the model does not pick one. Defaults to the first configured.
        cwd: Working directory the agents are started in. Defaults to the current directory.
            The model may pass another directory per call.
        timeout: Seconds a single task may run before the turn is cancelled.
        permissions: How the agent's ``session/request_permission`` calls are answered:
            ``"read_only"`` (default), ``"allow_all"``, ``"deny_all"`` or a callable that receives a
            :class:`~strands_acp_client.PermissionRequest` and returns ``True``, ``False`` or an option id.
        strict: Under any policy but ``allow_all``, stop the turn and return an error when the agent
            completes an edit, delete, move or command it never asked permission for. A policy can only
            answer questions; this is what catches an agent that is configured not to ask.
        env: Environment changes for every agent process, applied over this process's environment.
            A value of ``None`` removes the variable.
        keep_sessions: Keep agent processes alive after a call so the model can continue the
            conversation with ``session_id``. ``False`` closes the process after every call.
        max_sessions: Maximum number of live sessions; the least recently used is closed first.
        cancel_grace: Seconds to wait for the agent to honour a cancellation before its process is killed.
        name: Tool name shown to the model.
        description: Replace the generated tool description.

    Returns:
        A Strands tool. Its live sessions are reachable as ``tool.acp_sessions`` (a
        :class:`~strands_acp_client.SessionRegistry`); call ``await tool.acp_sessions.close_all()``
        to stop every agent process.
    """
    resolved = resolve_agents(agents)
    validate_policy(permissions)
    default = default_agent or next(iter(resolved))
    if default not in resolved:
        raise ValueError(f"default_agent {default!r} is not one of the configured agents: {', '.join(resolved)}")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    # The real path, not a symlinked one: agents compare the files they touch with the session's
    # directory, and a path that only looks different sends them asking about an "external" folder.
    base_cwd = os.path.realpath(os.fspath(cwd)) if cwd is not None else os.path.realpath(os.getcwd())
    if not os.path.isdir(base_cwd):
        raise ValueError(f"cwd {base_cwd!r} is not a directory")

    registry = SessionRegistry(max_sessions=max_sessions)
    label = policy_label(permissions)
    tool_description = description or _describe(resolved, default, label, base_cwd)
    input_schema = _input_schema(resolved, default, base_cwd)

    async def acp_agent(
        task: str,
        agent_name: str | None = None,
        session_id: str | None = None,
        cwd: str | None = None,
        tool_context: ToolContext = None,  # type: ignore[assignment]
    ) -> AsyncGenerator[Any, None]:
        chosen = agent_name or default
        if chosen not in resolved:
            yield _error(f"unknown agent {chosen!r}; configured agents: {', '.join(resolved)}")
            return
        if not task or not task.strip():
            yield _error("task must not be empty")
            return
        work_dir = os.path.realpath(cwd) if cwd else base_cwd
        if not os.path.isdir(work_dir):
            yield _error(f"cwd {work_dir!r} is not a directory")
            return
        cancel_signal = getattr(tool_context, "cancel_signal", None)

        try:
            session = await registry.acquire(
                chosen, resolved[chosen], work_dir, session_id, permissions, env, cancel_grace, strict
            )
        except AcpAgentError as exc:
            yield _error(str(exc))
            return

        result: TurnResult | None = None
        try:
            async for item in session.prompt(task, timeout=timeout, cancel_signal=cancel_signal):
                if isinstance(item, TurnResult):
                    result = item
                else:
                    yield item
        except AcpAgentError as exc:
            await registry.close(session.session_id)
            yield _error(str(exc))
            return
        finally:
            keep = keep_sessions and result is not None and result.error is None
            await registry.release(session, keep)

        if result is None:
            yield _error(f"agent {chosen!r} produced no result")
            return
        yield _tool_result(chosen, session, label, result)

    decorated = tool(
        name=name, description=tool_description, inputSchema={"json": input_schema}, context="tool_context"
    )(acp_agent)
    decorated.tool_spec = {**decorated.tool_spec, "annotations": _annotations(label)}
    decorated.acp_sessions = registry  # type: ignore[attr-defined]
    return decorated


# ── model-facing surface ──


def _describe(agents: Mapping[str, AgentCommand], default: str, label: str, base_cwd: str) -> str:
    lines = [
        "Delegate a task to a coding agent that speaks the Agent Client Protocol and wait for its answer. "
        "The agent runs on this machine with its own tools, in a working directory, and returns a final message. "
        "Give it a complete, self-contained task.",
        "Agents: " + "; ".join(f"{n} ({c.description})" if c.description else n for n, c in agents.items()) + ".",
        f"Default agent: {default}. Default working directory: {base_cwd}.",
        f"Permissions: {label} ({_POLICY_NOTES[label]}).",
        "The result includes a session_id: pass it back to continue the same conversation with follow-up tasks.",
    ]
    return " ".join(lines)


def _input_schema(agents: Mapping[str, AgentCommand], default: str, base_cwd: str) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "What the coding agent should do, with the context it needs to work on its own.",
            },
            "agent_name": {
                "type": "string",
                "enum": list(agents),
                "description": f"Which agent to delegate to. Defaults to {default}.",
            },
            "session_id": {
                "type": "string",
                "description": "A session_id returned by an earlier call, to continue that conversation "
                "instead of starting a new one.",
            },
            "cwd": {
                "type": "string",
                "description": f"Working directory for the agent. Defaults to {base_cwd}.",
            },
        },
        "required": ["task"],
    }


def _annotations(label: str) -> dict[str, Any]:
    return {
        "title": "ACP coding agent",
        "readOnlyHint": label in ("read_only", "deny_all"),
        "destructiveHint": label in ("allow_all", "custom"),
        "idempotentHint": False,
        "openWorldHint": True,
    }


# ── results ──


def _tool_result(agent_name: str, session: AcpSession, label: str, result: TurnResult) -> dict[str, Any]:
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
        "agent": agent_name,
        "agent_info": session.agent_info,
        "session_id": session.session_id,
        "resumed": session.resumed,
        "mode": session.mode,
        "stop_reason": result.stop_reason,
        "usage": result.usage,
        "tool_calls": result.tool_calls,
        "permissions": {"policy": label, "allowed": result.allowed, "denied": result.denied},
        "elapsed_seconds": round(result.elapsed, 2),
    }
    if session.config:
        details["config"] = dict(session.config)
    if result.refused:
        details["refused"] = list(result.refused)
    if result.violation is not None:
        details["policy_violation"] = result.violation
    return {"status": status, "content": [{"text": text}, {"json": details}]}


def _error(message: str) -> dict[str, Any]:
    return {"status": "error", "content": [{"text": message}]}
