"""Known ACP agents and how to launch them as subprocesses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

__all__ = ["KNOWN_AGENTS", "AgentCommand", "resolve_agents"]


@dataclass(frozen=True)
class AgentCommand:
    """How to start an agent that speaks the Agent Client Protocol on stdin/stdout.

    Attributes:
        argv: The program and its arguments, e.g. ``("npx", "-y", "@agentclientprotocol/claude-agent-acp")``.
        description: One line shown to the model so it can pick the right agent.
        env: Environment changes for the subprocess, applied over the parent's environment. A value of
            ``None`` removes the variable, which is how you hide a variable the agent must not inherit.
        modes: Session mode to select for each permission policy, keyed by policy name (``read_only``,
            ``deny_all``, ``custom``, ``allow_all``). A policy can only answer the questions an agent asks,
            and many agents start in a mode that does not ask. This names the mode in which the agent
            defers to the client, e.g. ``{"read_only": "default"}`` for Claude Code. It is applied when
            the agent offers that mode, through ``session/set_mode`` or the ``mode`` config option.
        policy_args: Extra arguments appended to ``argv`` for a permission policy, keyed the same way.
            For agents whose restricted behaviour is chosen at launch rather than per session, e.g.
            ``{"read_only": ("--builtin-tools", "read,web_fetch")}`` for the Strands CLI.
    """

    argv: tuple[str, ...]
    description: str = ""
    env: Mapping[str, str | None] | None = None
    modes: Mapping[str, str] | None = None
    policy_args: Mapping[str, Sequence[str]] | None = None

    def __post_init__(self) -> None:
        if not self.argv or not self.argv[0] or not all(isinstance(part, str) for part in self.argv):
            raise ValueError("argv must be a non-empty sequence of strings starting with the program to run")

    def with_args(self, *args: str) -> AgentCommand:
        """A copy of this command with more arguments, keeping its description, modes and policy arguments.

        ``KNOWN_AGENTS["strands"].with_args("--model", "ollama/qwen3.8:27b")`` pins a model without
        losing what the package knows about the agent.
        """
        return replace(self, argv=(*self.argv, *args))

    def argv_for(self, policy: str) -> tuple[str, ...]:
        """The full command line for a permission policy."""
        return (*self.argv, *(self.policy_args or {}).get(policy, ()))


# Modes verified against the agents themselves. Claude Code's "default" is "always ask before
# making changes"; it starts in whatever the user's settings say, often "auto". OpenCode's
# "plan" is its read-only agent; it starts in "build", which edits without asking. The Strands
# CLI has no way to ask the client over ACP (0.2.x), so read-only is a matter of which built-in
# tools it is started with.
_ASK_FIRST = {"read_only": "default", "deny_all": "default", "custom": "default"}
_PLAN_ONLY = {"read_only": "plan", "deny_all": "plan"}
_READ_TOOLS = ("--builtin-tools", "read,web_fetch")

KNOWN_AGENTS: dict[str, AgentCommand] = {
    "claude": AgentCommand(
        ("npx", "-y", "@agentclientprotocol/claude-agent-acp"),
        "Claude Code through the official ACP adapter",
        modes=_ASK_FIRST,
    ),
    "opencode": AgentCommand(
        ("opencode", "acp"),
        "OpenCode (the opencode binary must be installed)",
        modes=_PLAN_ONLY,
    ),
    "strands": AgentCommand(
        ("npx", "-y", "@strands-agents/cli", "--acp-server"),
        "A Strands agent served over ACP by the Strands CLI",
        policy_args={"read_only": _READ_TOOLS, "deny_all": _READ_TOOLS},
    ),
}
"""Agents this package has been driven against end to end, with what was learned about each.

An agent is added here only after the suite in ``tests/e2e`` passes against the real thing. Any other
agent that speaks ACP works by passing its command (the ACP registry lists them:
https://github.com/agentclientprotocol/registry), but nothing here has checked how it asks for permission.
"""

AgentsArg = Mapping[str, "Sequence[str] | AgentCommand | str"] | Sequence[str] | str | None
"""Accepted shapes for the ``agents`` argument of :func:`strands_acp_client.make_acp_agent`."""


def resolve_agents(agents: AgentsArg) -> dict[str, AgentCommand]:
    """Normalize the ``agents`` argument into a mapping of tool-facing name to launch command.

    Accepts a mapping of name to argv / :class:`AgentCommand` / known-agent alias, a list of
    known-agent names, a single known-agent name, or ``None`` (Claude Code only). A plain argv
    given under a known name keeps that agent's description and modes, so pinning a version
    does not lose them; its policy arguments are not carried over, because they are written for
    the stock command. To add flags to a known agent, use :meth:`AgentCommand.with_args`.
    """
    if agents is None:
        return {"claude": KNOWN_AGENTS["claude"]}
    if isinstance(agents, str):
        agents = [agents]

    resolved: dict[str, AgentCommand] = {}
    if isinstance(agents, Mapping):
        for name, spec in agents.items():
            _check_name(name)
            if isinstance(spec, AgentCommand):
                resolved[name] = spec
            elif isinstance(spec, str):
                resolved[name] = _known(spec)
            else:
                known = KNOWN_AGENTS.get(name)
                resolved[name] = AgentCommand(
                    tuple(spec),
                    known.description if known else "",
                    modes=known.modes if known else None,
                )
    else:
        for name in agents:
            _check_name(name)
            resolved[name] = _known(name)

    if not resolved:
        raise ValueError("agents must not be empty")
    return resolved


def _known(name: str) -> AgentCommand:
    try:
        return KNOWN_AGENTS[name]
    except KeyError:
        known = ", ".join(sorted(KNOWN_AGENTS))
        raise ValueError(f"unknown agent {name!r}; known agents: {known}. Pass argv explicitly for others.") from None


def _check_name(name: object) -> None:
    if not isinstance(name, str) or not name or any(ch.isspace() for ch in name):
        raise ValueError(f"agent names must be non-empty strings without whitespace, got {name!r}")
