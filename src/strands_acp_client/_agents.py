"""Known ACP agents and how to launch them as subprocesses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

__all__ = ["KNOWN_AGENTS", "AgentCommand", "resolve_agents"]


@dataclass(frozen=True)
class AgentCommand:
    """How to start an agent that speaks the Agent Client Protocol on stdin/stdout.

    Attributes:
        argv: The program and its arguments, e.g. ``("npx", "-y", "@agentclientprotocol/claude-agent-acp")``.
        description: One line shown to the model so it can pick the right agent.
        env: Extra environment variables for the subprocess, merged over the parent's environment.
    """

    argv: tuple[str, ...]
    description: str = ""
    env: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if not self.argv or not all(isinstance(part, str) and part for part in self.argv):
            raise ValueError("argv must be a non-empty sequence of non-empty strings")


KNOWN_AGENTS: dict[str, AgentCommand] = {
    "claude": AgentCommand(
        ("npx", "-y", "@agentclientprotocol/claude-agent-acp"),
        "Claude Code through the official ACP adapter",
    ),
    "codex": AgentCommand(
        ("npx", "-y", "@agentclientprotocol/codex-acp"),
        "OpenAI Codex CLI through the official ACP adapter",
    ),
    "gemini": AgentCommand(("npx", "-y", "@google/gemini-cli", "--acp"), "Gemini CLI"),
    "copilot": AgentCommand(("npx", "-y", "@github/copilot", "--acp"), "GitHub Copilot CLI"),
    "qwen": AgentCommand(("npx", "-y", "@qwen-code/qwen-code", "--acp"), "Qwen Code"),
    "kiro": AgentCommand(("kiro-cli", "acp"), "Kiro CLI (kiro-cli must be installed)"),
    "opencode": AgentCommand(("opencode", "acp"), "OpenCode (the opencode binary must be installed)"),
    "strands": AgentCommand(
        ("npx", "-y", "@strands-agents/cli", "--acp-server"),
        "A Strands agent served over ACP by the Strands CLI",
    ),
}
"""Launch commands taken from the ACP registry (https://github.com/agentclientprotocol/registry)."""

AgentsArg = Mapping[str, "Sequence[str] | AgentCommand | str"] | Sequence[str] | str | None
"""Accepted shapes for the ``agents`` argument of :func:`strands_acp_client.make_acp_agent`."""


def resolve_agents(agents: AgentsArg) -> dict[str, AgentCommand]:
    """Normalize the ``agents`` argument into a mapping of tool-facing name to launch command.

    Accepts a mapping of name to argv / :class:`AgentCommand` / known-agent alias, a list of
    known-agent names, a single known-agent name, or ``None`` (Claude Code only).
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
                resolved[name] = AgentCommand(
                    tuple(spec), KNOWN_AGENTS[name].description if name in KNOWN_AGENTS else ""
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
