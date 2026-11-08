"""A scripted ACP agent for the tests: no model, no API keys, deterministic behaviour.

Run with ``python tests/fake_acp_agent.py``. The first word of each prompt picks a script:

- ``tool <kind>``: report a tool call of that kind, ask permission for it, then say whether it was allowed
- ``slow <seconds>``: keep the turn open until cancelled or the time elapses
- ``turn``: answer with the number of prompts seen in this session
- ``usage``: answer with token usage attached
- ``refuse``: end the turn with stop reason ``refusal``
- ``fail``: answer with an "authentication required" error; ``error``: answer with an internal error
- ``crash``: exit the process mid-turn
- anything else: echo the prompt back in two chunks

Environment: ``FAKE_ACP_PROTOCOL_VERSION`` overrides the negotiated version, ``FAKE_ACP_LOAD_SESSION=1``
advertises ``session/load`` and ``FAKE_ACP_LOADED_TURNS`` seeds the turn counter of a loaded session.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from acp import PROTOCOL_VERSION, Agent, RequestError, run_agent
from acp.interfaces import Client
from acp.schema import (
    AgentCapabilities,
    AgentMessageChunk,
    AgentThoughtChunk,
    AllowedOutcome,
    AuthenticateResponse,
    Implementation,
    InitializeResponse,
    LoadSessionResponse,
    NewSessionResponse,
    PermissionOption,
    PromptResponse,
    SetSessionModeResponse,
    TextContentBlock,
    ToolCallLocation,
    ToolCallProgress,
    ToolCallStart,
    ToolCallUpdate,
    Usage,
)

OPTIONS = [
    PermissionOption(option_id="allow-once", name="Allow once", kind="allow_once"),
    PermissionOption(option_id="allow-always", name="Always allow", kind="allow_always"),
    PermissionOption(option_id="reject-once", name="Reject once", kind="reject_once"),
    PermissionOption(option_id="reject-always", name="Always reject", kind="reject_always"),
]


class FakeAgent(Agent):
    def __init__(self) -> None:
        self._conn: Client | None = None
        self._next = 0
        self._turns: dict[str, int] = {}
        self._cancelled: dict[str, asyncio.Event] = {}

    def on_connect(self, conn: Client) -> None:
        self._conn = conn

    @property
    def conn(self) -> Client:
        assert self._conn is not None
        return self._conn

    async def initialize(self, protocol_version: int, **kwargs: Any) -> InitializeResponse:
        version = int(os.environ.get("FAKE_ACP_PROTOCOL_VERSION", PROTOCOL_VERSION))
        load = os.environ.get("FAKE_ACP_LOAD_SESSION") == "1"
        return InitializeResponse(
            protocol_version=version,
            agent_capabilities=AgentCapabilities(load_session=load),
            agent_info=Implementation(name="fake-agent", version="0.0.1"),
        )

    async def authenticate(self, method_id: str, **kwargs: Any) -> AuthenticateResponse:
        return AuthenticateResponse()

    async def new_session(self, cwd: str, **kwargs: Any) -> NewSessionResponse:
        session_id = f"fake-{os.getpid()}-{self._next}"
        self._next += 1
        self._turns[session_id] = 0
        return NewSessionResponse(session_id=session_id)

    async def load_session(self, cwd: str, session_id: str, **kwargs: Any) -> LoadSessionResponse:
        self._turns[session_id] = int(os.environ.get("FAKE_ACP_LOADED_TURNS", "0"))
        # Replay a bit of history, like a real agent does; the client must not collect it.
        await self._say(session_id, "replayed history")
        return LoadSessionResponse()

    async def set_session_mode(self, session_id: str, mode_id: str, **kwargs: Any) -> SetSessionModeResponse:
        return SetSessionModeResponse()

    async def prompt(self, session_id: str, prompt: list[Any], **kwargs: Any) -> PromptResponse:
        text = " ".join(block.text for block in prompt if isinstance(block, TextContentBlock))
        self._turns[session_id] = self._turns.get(session_id, 0) + 1
        cancelled = asyncio.Event()
        self._cancelled[session_id] = cancelled
        words = text.split()
        command = words[0] if words else ""

        if command == "tool":
            kind = words[1] if len(words) > 1 else "edit"
            call_id = f"call-{self._turns[session_id]}"
            location = [ToolCallLocation(path=os.path.join(os.getcwd(), "example.txt"))]
            await self.conn.session_update(
                session_id,
                ToolCallStart(tool_call_id=call_id, title=f"{kind} example.txt", kind=kind, status="in_progress"),
            )
            response = await self.conn.request_permission(
                session_id=session_id,
                tool_call=ToolCallUpdate(
                    tool_call_id=call_id,
                    title=f"{kind} example.txt",
                    kind=kind,
                    status="in_progress",
                    locations=location,
                ),
                options=list(OPTIONS),
            )
            selected = response.outcome.option_id if isinstance(response.outcome, AllowedOutcome) else "cancelled"
            allowed = selected.startswith("allow")
            await self.conn.session_update(
                session_id, ToolCallProgress(tool_call_id=call_id, status="completed" if allowed else "failed")
            )
            await self._say(session_id, f"permission {'allowed' if allowed else 'denied'} via {selected}")
            return PromptResponse(stop_reason="end_turn")

        if command == "slow":
            seconds = float(words[1]) if len(words) > 1 else 5.0
            await self._say(session_id, "working...")
            try:
                await asyncio.wait_for(cancelled.wait(), timeout=seconds)
            except asyncio.TimeoutError:
                await self._say(session_id, "done sleeping")
                return PromptResponse(stop_reason="end_turn")
            return PromptResponse(stop_reason="cancelled")

        if command == "turn":
            await self._say(session_id, f"turn {self._turns[session_id]}")
            return PromptResponse(stop_reason="end_turn")

        if command == "usage":
            await self._say(session_id, "counted")
            return PromptResponse(
                stop_reason="end_turn", usage=Usage(total_tokens=30, input_tokens=20, output_tokens=10)
            )

        if command == "refuse":
            return PromptResponse(stop_reason="refusal")

        if command == "fail":
            raise RequestError.auth_required()

        if command == "error":
            raise RequestError.internal_error({"detail": "boom"})

        if command == "crash":
            os._exit(3)

        await self.conn.session_update(session_id, AgentThoughtChunk(content=TextContentBlock(text="thinking")))
        await self._say(session_id, "You said: ")
        await self._say(session_id, text)
        return PromptResponse(stop_reason="end_turn")

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        event = self._cancelled.get(session_id)
        if event is not None:
            event.set()

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        return {}

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        return None

    async def _say(self, session_id: str, text: str) -> None:
        await self.conn.session_update(session_id, AgentMessageChunk(content=TextContentBlock(text=text)))


if __name__ == "__main__":
    asyncio.run(run_agent(FakeAgent()))
