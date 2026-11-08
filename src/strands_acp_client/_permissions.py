"""Answering ``session/request_permission`` on behalf of the Strands agent.

Under ACP the client owns permissions: every time the coding agent wants to run a tool
that its own rules gate (editing a file, running a command), it asks the client. This
module turns that question into an answer from a policy name or a caller-supplied
function, so the Strands side never adds a second permission system on top of the
intervention handlers it already has.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Union

from acp.schema import AllowedOutcome, DeniedOutcome, PermissionOption, RequestPermissionResponse

__all__ = [
    "READ_ONLY_KINDS",
    "PermissionDecision",
    "PermissionHandler",
    "PermissionPolicy",
    "PermissionPolicyName",
    "PermissionRequest",
    "decide",
    "policy_label",
    "validate_policy",
]

PermissionPolicyName = Literal["allow_all", "read_only", "deny_all"]

READ_ONLY_KINDS: frozenset[str] = frozenset({"read", "search", "fetch", "think"})
"""ACP tool kinds that the ``read_only`` policy allows; everything else (edit, delete, move, execute...) is refused."""


@dataclass(frozen=True)
class PermissionRequest:
    """What the coding agent asked for, in plain Python for custom handlers.

    Attributes:
        agent: The tool-facing name of the agent that is asking.
        session_id: The ACP session the request belongs to.
        tool_call_id: The agent's id for the gated tool call.
        title: Human-readable description of the call (e.g. ``"Edit src/app.py"``).
        kind: ACP tool kind: read, edit, delete, move, search, execute, think, fetch, switch_mode or other.
        locations: File paths the call touches, when the agent reports them.
        options: The answers the agent offers (allow_once, allow_always, reject_once, reject_always).
        raw_input: The tool input as the agent reported it, if any.
    """

    agent: str
    session_id: str
    tool_call_id: str
    title: str | None
    kind: str | None
    locations: tuple[str, ...]
    options: tuple[PermissionOption, ...]
    raw_input: Any = None


PermissionDecision = Union[bool, str]
"""``True`` to allow, ``False`` to refuse, or the ``option_id`` of one of the offered options."""

PermissionHandler = Callable[[PermissionRequest], Union[PermissionDecision, Awaitable[PermissionDecision]]]
PermissionPolicy = Union[PermissionPolicyName, PermissionHandler]

_POLICY_NAMES: tuple[str, ...] = ("allow_all", "read_only", "deny_all")


def validate_policy(policy: object) -> None:
    """Raise ``ValueError`` unless ``policy`` is a known name or a callable."""
    if callable(policy):
        return
    if policy not in _POLICY_NAMES:
        raise ValueError(f"permissions must be one of {_POLICY_NAMES} or a callable, got {policy!r}")


def policy_label(policy: PermissionPolicy) -> str:
    """A short name for the policy, for tool descriptions and results."""
    return "custom" if callable(policy) else str(policy)


async def decide(policy: PermissionPolicy, request: PermissionRequest) -> tuple[RequestPermissionResponse, bool]:
    """Apply ``policy`` to ``request``. Returns the wire response and whether the call was allowed."""
    decision: PermissionDecision
    if callable(policy):
        raw = policy(request)
        decision = await raw if inspect.isawaitable(raw) else raw
    elif policy == "allow_all":
        decision = True
    elif policy == "deny_all":
        decision = False
    elif policy == "read_only":
        decision = request.kind in READ_ONLY_KINDS
    else:
        raise ValueError(f"unknown permission policy {policy!r}")
    return build_response(request.options, decision)


def build_response(
    options: Sequence[PermissionOption], decision: PermissionDecision
) -> tuple[RequestPermissionResponse, bool]:
    """Pick the offered option that matches ``decision``.

    A boolean selects the first ``allow_*`` or ``reject_*`` option (once before always). A
    string selects that ``option_id``. When no offered option fits, the request is answered
    as cancelled, which every agent treats as a refusal.
    """
    if isinstance(decision, str):
        for option in options:
            if option.option_id == decision:
                return _select(option), option.kind.startswith("allow")
        raise ValueError(f"permission handler chose option {decision!r}, offered: {[o.option_id for o in options]}")

    wanted = ("allow_once", "allow_always") if decision else ("reject_once", "reject_always")
    for kind in wanted:
        for option in options:
            if option.kind == kind:
                return _select(option), bool(decision)
    return RequestPermissionResponse(outcome=DeniedOutcome()), False


def _select(option: PermissionOption) -> RequestPermissionResponse:
    return RequestPermissionResponse(outcome=AllowedOutcome(option_id=option.option_id))
