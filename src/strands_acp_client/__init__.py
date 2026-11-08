"""Use ACP coding agents (Claude Code, Codex, Gemini CLI, Kiro CLI, Strands CLI...) as tools from a Strands agent."""

from importlib.metadata import PackageNotFoundError, version

from strands_acp_client._agents import KNOWN_AGENTS, AgentCommand
from strands_acp_client._permissions import READ_ONLY_KINDS, PermissionPolicy, PermissionRequest
from strands_acp_client._session import AcpAgentError, SessionRegistry
from strands_acp_client.tool import make_acp_agent

try:
    __version__ = version("strands-acp-client")
except PackageNotFoundError:  # pragma: no cover - source checkout without metadata
    __version__ = "0.0.0"

__all__ = [
    "KNOWN_AGENTS",
    "READ_ONLY_KINDS",
    "AcpAgentError",
    "AgentCommand",
    "PermissionPolicy",
    "PermissionRequest",
    "SessionRegistry",
    "__version__",
    "make_acp_agent",
]
