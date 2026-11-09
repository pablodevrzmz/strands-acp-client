"""A scripted ACP agent for the tests: no model, no API keys, deterministic behaviour.

Run with ``python tests/fake_acp_agent.py``. The first word of each prompt picks a script:

- ``tool <kind>``: report a tool call of that kind, ask permission for it, then say whether it was allowed
- ``vague <kind>``: like ``tool``, but the permission request itself only says ``other``
- ``toolstop <kind>``: like ``tool``, but end the turn as ``cancelled`` with no message when refused
- ``sneak <kind> [seconds]``: complete a tool call of that kind without asking, then wait to see if anyone objects
- ``slow <seconds>``: keep the turn open until cancelled or the time elapses
- ``stubborn <seconds>``: keep the turn open for that long and ignore ``session/cancel``
- ``noisy <kib>``: write that many KiB to stderr, more than a pipe holds, then answer
- ``big <kib>``: answer with a single message chunk of that many KiB, far longer than one buffered line
- ``turn``: answer with the number of prompts seen in this session
- ``mode``: answer with the session's current mode; ``model``: answer with its current model
- ``usage``: answer with token usage attached
- ``refuse``: end the turn with stop reason ``refusal``
- ``fail``: answer with an "authentication required" error; ``error``: answer with an internal error
- ``crash``: complain on stderr and exit the process mid-turn
- ``env <NAME>``: answer with the value of that environment variable, or ``<unset>``
- ``argv``: answer with the arguments the agent was started with
- ``note``: answer with whatever came after the task in the prompt (the client adds the working directory)
- anything else: echo the prompt back in two chunks

Environment: ``FAKE_ACP_PROTOCOL_VERSION`` overrides the negotiated version, ``FAKE_ACP_LOAD_SESSION=1``
advertises ``session/load`` and ``FAKE_ACP_LOADED_TURNS`` seeds the turn counter of a loaded session.
``FAKE_ACP_MODES=a,b`` offers session modes, ``FAKE_ACP_CONFIG_MODES=a,b`` offers them as the ``mode``
config option instead, and ``FAKE_ACP_MODE`` picks the starting one (the last listed by default).
``FAKE_ACP_CONFIG_MODELS=a,b`` offers a ``model`` config option starting at the first one.
``FAKE_ACP_REQUIRE_AUTH=<method id>`` refuses sessions until the client authenticates with that method.

For the process itself: ``FAKE_ACP_PID_FILE`` names a file that receives the process id, ``FAKE_ACP_STDERR``
is a line written to stderr at startup, ``FAKE_ACP_EXIT=<code>`` exits with that code before speaking ACP at
all, ``FAKE_ACP_HANG=initialize`` (or ``new_session``) never answers that request, ``FAKE_ACP_IGNORE_EOF=1``
keeps the process running after its stdin closes and ``FAKE_ACP_IGNORE_TERM=1`` makes it ignore SIGTERM.
"""

from __future__ import annotations

import os
import sys

# Before the imports below, which take a moment: a test that gives the agent a second to start
# must find the process id and the stderr line already there.
if __name__ == "__main__":
    if os.environ.get("FAKE_ACP_PID_FILE"):
        with open(os.environ["FAKE_ACP_PID_FILE"], "w") as _pid_file:
            _pid_file.write(str(os.getpid()))
    if os.environ.get("FAKE_ACP_STDERR"):
        print(os.environ["FAKE_ACP_STDERR"], file=sys.stderr, flush=True)

import asyncio
import signal
import time
from typing import Any

from acp import PROTOCOL_VERSION, Agent, RequestError, run_agent
from acp.interfaces import Client
from acp.schema import (
    AgentCapabilities,
    AgentMessageChunk,
    AgentThoughtChunk,
    AllowedOutcome,
    AuthenticateResponse,
    AuthMethodAgent,
    Implementation,
    InitializeResponse,
    LoadSessionResponse,
    NewSessionResponse,
    PermissionOption,
    PromptResponse,
    SessionConfigOptionSelect,
    SessionConfigSelectOption,
    SessionMode,
    SessionModeState,
    SetSessionConfigOptionResponse,
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


def _listed(name: str) -> list[str]:
    return [item for item in os.environ.get(name, "").split(",") if item]


class FakeAgent(Agent):
    def __init__(self) -> None:
        self._conn: Client | None = None
        self._next = 0
        self._turns: dict[str, int] = {}
        self._modes: dict[str, str] = {}
        self._models: dict[str, str] = {}
        self._required_auth = os.environ.get("FAKE_ACP_REQUIRE_AUTH")
        self._authenticated = False
        self._cancelled: dict[str, asyncio.Event] = {}

    def on_connect(self, conn: Client) -> None:
        self._conn = conn

    @property
    def conn(self) -> Client:
        assert self._conn is not None
        return self._conn

    async def initialize(self, protocol_version: int, **kwargs: Any) -> InitializeResponse:
        if os.environ.get("FAKE_ACP_HANG") == "initialize":
            await asyncio.sleep(3600)
        version = int(os.environ.get("FAKE_ACP_PROTOCOL_VERSION", PROTOCOL_VERSION))
        load = os.environ.get("FAKE_ACP_LOAD_SESSION") == "1"
        return InitializeResponse(
            protocol_version=version,
            agent_capabilities=AgentCapabilities(load_session=load),
            agent_info=Implementation(name="fake-agent", version="0.0.1"),
            auth_methods=[AuthMethodAgent(id=self._required_auth, name="Fake key")] if self._required_auth else [],
        )

    async def authenticate(self, method_id: str, **kwargs: Any) -> AuthenticateResponse:
        if self._required_auth and method_id != self._required_auth:
            raise RequestError.invalid_params({"method": method_id})
        self._authenticated = True
        return AuthenticateResponse()

    def _require_auth(self) -> None:
        if self._required_auth and not self._authenticated:
            raise RequestError.auth_required()

    async def new_session(self, cwd: str, **kwargs: Any) -> NewSessionResponse:
        if os.environ.get("FAKE_ACP_HANG") == "new_session":
            await asyncio.sleep(3600)
        self._require_auth()
        session_id = f"fake-{os.getpid()}-{self._next}"
        self._next += 1
        self._turns[session_id] = 0
        return NewSessionResponse(
            session_id=session_id, modes=self._mode_state(session_id), config_options=self._config_options(session_id)
        )

    async def load_session(self, cwd: str, session_id: str, **kwargs: Any) -> LoadSessionResponse:
        self._require_auth()
        self._turns[session_id] = int(os.environ.get("FAKE_ACP_LOADED_TURNS", "0"))
        # Replay a bit of history, like a real agent does; the client must not collect it.
        await self._say(session_id, "replayed history")
        return LoadSessionResponse(modes=self._mode_state(session_id), config_options=self._config_options(session_id))

    # ── modes, offered either the classic way or as a config option ──

    def _starting_mode(self, offered: list[str]) -> str:
        return os.environ.get("FAKE_ACP_MODE") or offered[-1]

    def _mode_state(self, session_id: str) -> SessionModeState | None:
        offered = _listed("FAKE_ACP_MODES")
        if not offered:
            return None
        current = self._modes.setdefault(session_id, self._starting_mode(offered))
        return SessionModeState(
            current_mode_id=current, available_modes=[SessionMode(id=mode, name=mode.title()) for mode in offered]
        )

    def _config_options(self, session_id: str) -> list[SessionConfigOptionSelect] | None:
        options: list[SessionConfigOptionSelect] = []
        offered = _listed("FAKE_ACP_CONFIG_MODES")
        if offered:
            current = self._modes.setdefault(session_id, self._starting_mode(offered))
            options.append(
                SessionConfigOptionSelect(
                    id="mode",
                    name="Mode",
                    category="mode",
                    current_value=current,
                    options=[SessionConfigSelectOption(value=mode, name=mode.title()) for mode in offered],
                )
            )
        models = _listed("FAKE_ACP_CONFIG_MODELS")
        if models:
            current = self._models.setdefault(session_id, models[0])
            options.append(
                SessionConfigOptionSelect(
                    id="model",
                    name="Model",
                    category="model",
                    current_value=current,
                    options=[SessionConfigSelectOption(value=model, name=model) for model in models],
                )
            )
        return options or None

    async def set_session_mode(self, session_id: str, mode_id: str, **kwargs: Any) -> SetSessionModeResponse:
        self._modes[session_id] = mode_id
        return SetSessionModeResponse()

    async def set_config_option(
        self, config_id: str, session_id: str, value: str | bool, **kwargs: Any
    ) -> SetSessionConfigOptionResponse:
        if config_id == "mode":
            self._modes[session_id] = str(value)
        if config_id == "model":
            self._models[session_id] = str(value)
        return SetSessionConfigOptionResponse(config_options=self._config_options(session_id) or [])

    # ── prompts ──

    async def prompt(self, session_id: str, prompt: list[Any], **kwargs: Any) -> PromptResponse:
        blocks = [block.text for block in prompt if isinstance(block, TextContentBlock)]
        text = blocks[0] if blocks else ""
        notes = blocks[1:]
        self._turns[session_id] = self._turns.get(session_id, 0) + 1
        cancelled = asyncio.Event()
        self._cancelled[session_id] = cancelled
        words = text.split()
        command = words[0] if words else ""

        if command in ("tool", "toolstop", "vague"):
            kind = words[1] if len(words) > 1 else "edit"
            call_id = f"call-{self._turns[session_id]}"
            title = f"{kind} example.txt"
            location = [ToolCallLocation(path=os.path.join(os.getcwd(), "example.txt"))]
            await self.conn.session_update(
                session_id, ToolCallStart(tool_call_id=call_id, title=title, kind=kind, status="pending")
            )
            response = await self.conn.request_permission(
                session_id=session_id,
                tool_call=ToolCallUpdate(
                    tool_call_id=call_id,
                    title=title,
                    kind="other" if command == "vague" else kind,
                    status="pending",
                    locations=location,
                ),
                options=list(OPTIONS),
            )
            selected = response.outcome.option_id if isinstance(response.outcome, AllowedOutcome) else "cancelled"
            allowed = selected.startswith("allow")
            await self.conn.session_update(
                session_id, ToolCallProgress(tool_call_id=call_id, status="completed" if allowed else "failed")
            )
            if command == "toolstop" and not allowed:
                return PromptResponse(stop_reason="cancelled")
            await self._say(session_id, f"permission {'allowed' if allowed else 'denied'} via {selected}")
            return PromptResponse(stop_reason="end_turn")

        if command == "sneak":
            kind = words[1] if len(words) > 1 else "edit"
            wait = float(words[2]) if len(words) > 2 else 2.0
            call_id = f"sneak-{self._turns[session_id]}"
            await self.conn.session_update(
                session_id,
                ToolCallStart(tool_call_id=call_id, title=f"{kind} example.txt", kind=kind, status="pending"),
            )
            await self.conn.session_update(session_id, ToolCallProgress(tool_call_id=call_id, status="completed"))
            try:
                await asyncio.wait_for(cancelled.wait(), timeout=wait)
            except asyncio.TimeoutError:
                await self._say(session_id, "nobody stopped me")
                return PromptResponse(stop_reason="end_turn")
            return PromptResponse(stop_reason="cancelled")

        if command == "slow":
            seconds = float(words[1]) if len(words) > 1 else 5.0
            await self._say(session_id, "working...")
            try:
                await asyncio.wait_for(cancelled.wait(), timeout=seconds)
            except asyncio.TimeoutError:
                await self._say(session_id, "done sleeping")
                return PromptResponse(stop_reason="end_turn")
            return PromptResponse(stop_reason="cancelled")

        if command == "stubborn":
            await self._say(session_id, "not listening...")
            await asyncio.sleep(float(words[1]) if len(words) > 1 else 5.0)
            await self._say(session_id, "finished anyway")
            return PromptResponse(stop_reason="end_turn")

        if command == "noisy":
            kib = int(words[1]) if len(words) > 1 else 256
            line = "x" * 1023 + "\n"
            for _ in range(kib):
                sys.stderr.write(line)  # blocks for good once the pipe is full and nobody reads it
            sys.stderr.flush()
            await self._say(session_id, f"wrote {kib} KiB to stderr")
            return PromptResponse(stop_reason="end_turn")

        if command == "big":
            await self._say(session_id, "y" * (int(words[1]) * 1024 if len(words) > 1 else 262144))
            return PromptResponse(stop_reason="end_turn")

        if command == "turn":
            await self._say(session_id, f"turn {self._turns[session_id]}")
            return PromptResponse(stop_reason="end_turn")

        if command == "mode":
            await self._say(session_id, f"mode {self._modes.get(session_id, 'none')}")
            return PromptResponse(stop_reason="end_turn")

        if command == "model":
            await self._say(session_id, f"model {self._models.get(session_id, 'none')}")
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
            sys.stderr.write("fake agent: out of cheese, giving up\n")
            sys.stderr.flush()
            os._exit(3)

        if command == "note":
            await self._say(session_id, " | ".join(notes) or "<none>")
            return PromptResponse(stop_reason="end_turn")

        if command == "argv":
            await self._say(session_id, " ".join(sys.argv[1:]) or "<none>")
            return PromptResponse(stop_reason="end_turn")

        if command == "env":
            await self._say(session_id, os.environ.get(words[1], "<unset>"))
            return PromptResponse(stop_reason="end_turn")

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


def main() -> None:
    if os.environ.get("FAKE_ACP_IGNORE_TERM") == "1":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if os.environ.get("FAKE_ACP_EXIT"):
        sys.exit(int(os.environ["FAKE_ACP_EXIT"]))
    asyncio.run(run_agent(FakeAgent()))
    if os.environ.get("FAKE_ACP_IGNORE_EOF") == "1":
        time.sleep(3600)  # an agent that does not leave when its stdin closes


if __name__ == "__main__":
    main()
