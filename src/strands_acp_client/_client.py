"""The ACP ``Client`` that receives what the coding agent streams back."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from acp import Client, RequestError
from acp.interfaces import Agent
from acp.schema import (
    AgentMessageChunk,
    AgentThoughtChunk,
    ConnectMcpResponse,
    CreateElicitationResponse,
    CreateTerminalResponse,
    DeclineElicitationResponse,
    DeniedOutcome,
    DisconnectMcpResponse,
    ElicitationMode,
    EnvVariable,
    KillTerminalResponse,
    PermissionOption,
    ReadTextFileResponse,
    ReleaseTerminalResponse,
    RequestPermissionResponse,
    TerminalOutputResponse,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    ToolCallUpdate,
    WaitForTerminalExitResponse,
    WriteTextFileResponse,
)

from strands_acp_client._permissions import PermissionPolicy, PermissionRequest, decide

__all__ = ["StrandsAcpClient", "Turn"]

logger = logging.getLogger(__name__)


class Turn:
    """Everything the agent reports during one ``session/prompt``.

    Text chunks are joined into the final answer, tool calls are tracked by id, and a
    queue of progress events feeds the tool's stream so the Strands side can watch the
    agent work.
    """

    def __init__(self) -> None:
        self.text_parts: list[str] = []
        self.thought_chars = 0
        self.tool_calls: dict[str, dict[str, Any]] = {}
        self.allowed = 0
        self.denied = 0
        self.events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    @property
    def text(self) -> str:
        return "".join(self.text_parts)

    def push(self, event: dict[str, Any]) -> None:
        self.events.put_nowait(event)


class StrandsAcpClient(Client):
    """Client-side handler for one agent process.

    Only ``session_update`` and ``session/request_permission`` do real work. File system,
    terminal and MCP proxying are not advertised in the client capabilities, so a
    well-behaved agent never calls them; they answer "method not found" if one does. The
    agent uses its own tools, directly in the working directory it was started in.
    """

    def __init__(self, agent_name: str, policy: PermissionPolicy) -> None:
        self.agent_name = agent_name
        self.policy = policy
        self.turn: Turn | None = None

    def on_connect(self, conn: Agent) -> None:
        return None

    # ── what the agent streams ──

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        turn = self.turn
        if turn is None:
            # History replayed by session/load, or an update outside a prompt: nothing to collect.
            return
        if isinstance(update, AgentMessageChunk):
            text = _text_of(update.content)
            if text:
                turn.text_parts.append(text)
                turn.push({"type": "message_delta", "text": text})
        elif isinstance(update, AgentThoughtChunk):
            turn.thought_chars += len(_text_of(update.content))
        elif isinstance(update, ToolCallStart):
            entry = {
                "id": update.tool_call_id,
                "title": update.title,
                "kind": update.kind,
                "status": update.status or "pending",
                "locations": [location.path for location in update.locations or []],
            }
            turn.tool_calls[update.tool_call_id] = entry
            turn.push({"type": "tool_call", **entry})
        elif isinstance(update, ToolCallProgress):
            entry = turn.tool_calls.setdefault(
                update.tool_call_id,
                {"id": update.tool_call_id, "title": None, "kind": None, "status": "pending", "locations": []},
            )
            if update.title:
                entry["title"] = update.title
            if update.kind:
                entry["kind"] = update.kind
            if update.status:
                entry["status"] = update.status
            if update.locations:
                entry["locations"] = [location.path for location in update.locations]
            if update.status in ("completed", "failed"):
                turn.push({"type": "tool_call_update", **entry})
        # Plans, available commands, mode and config changes are not needed for a delegated task.

    # ── what the agent asks ──

    async def request_permission(
        self, session_id: str, tool_call: ToolCallUpdate, options: list[PermissionOption], **kwargs: Any
    ) -> RequestPermissionResponse:
        request = PermissionRequest(
            agent=self.agent_name,
            session_id=session_id,
            tool_call_id=tool_call.tool_call_id,
            title=tool_call.title,
            kind=tool_call.kind,
            locations=tuple(location.path for location in tool_call.locations or []),
            options=tuple(options),
            raw_input=tool_call.raw_input,
        )
        try:
            response, allowed = await decide(self.policy, request)
        except Exception:
            logger.exception("permission handler failed for %s; refusing the call", request.title)
            response, allowed = RequestPermissionResponse(outcome=DeniedOutcome()), False

        turn = self.turn
        if turn is not None:
            if allowed:
                turn.allowed += 1
            else:
                turn.denied += 1
            turn.push(
                {
                    "type": "permission",
                    "tool_call_id": request.tool_call_id,
                    "title": request.title,
                    "kind": request.kind,
                    "allowed": allowed,
                }
            )
        return response

    async def create_elicitation(self, message: str, mode: ElicitationMode, **kwargs: Any) -> CreateElicitationResponse:
        # Nobody is at the keyboard on this side: decline, and let the agent carry on or stop.
        return DeclineElicitationResponse(action="decline")

    async def complete_elicitation(self, elicitation_id: str, **kwargs: Any) -> None:
        return None

    # ── capabilities this client does not offer ──

    async def read_text_file(
        self, session_id: str, path: str, line: int | None = None, limit: int | None = None, **kwargs: Any
    ) -> ReadTextFileResponse:
        raise RequestError.method_not_found("fs/read_text_file")

    async def write_text_file(
        self, session_id: str, path: str, content: str, **kwargs: Any
    ) -> WriteTextFileResponse | None:
        raise RequestError.method_not_found("fs/write_text_file")

    async def create_terminal(
        self,
        session_id: str,
        command: str,
        args: list[str] | None = None,
        env: list[EnvVariable] | None = None,
        cwd: str | None = None,
        output_byte_limit: int | None = None,
        **kwargs: Any,
    ) -> CreateTerminalResponse:
        raise RequestError.method_not_found("terminal/create")

    async def terminal_output(self, session_id: str, terminal_id: str, **kwargs: Any) -> TerminalOutputResponse:
        raise RequestError.method_not_found("terminal/output")

    async def release_terminal(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> ReleaseTerminalResponse | None:
        raise RequestError.method_not_found("terminal/release")

    async def wait_for_terminal_exit(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> WaitForTerminalExitResponse:
        raise RequestError.method_not_found("terminal/wait_for_exit")

    async def kill_terminal(self, session_id: str, terminal_id: str, **kwargs: Any) -> KillTerminalResponse | None:
        raise RequestError.method_not_found("terminal/kill")

    async def connect_mcp(self, server_id: str, **kwargs: Any) -> ConnectMcpResponse:
        raise RequestError.method_not_found("mcp/connect")

    async def disconnect_mcp(self, connection_id: str, **kwargs: Any) -> DisconnectMcpResponse:
        raise RequestError.method_not_found("mcp/disconnect")

    async def mcp_message(
        self, connection_id: str, method: str, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any:
        raise RequestError.method_not_found("mcp/message")

    async def notify_mcp(
        self, connection_id: str, method: str, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        return None

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        raise RequestError.method_not_found(method)

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        return None


def _text_of(content: Any) -> str:
    if isinstance(content, TextContentBlock):
        return content.text
    return ""
