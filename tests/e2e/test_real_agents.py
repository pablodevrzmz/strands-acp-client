"""End-to-end checks against real coding agents installed on this machine.

These tests cost model tokens and need the agents logged in, so they never run by default.
Select the agents with ``ACP_E2E_AGENTS``: known names from ``KNOWN_AGENTS``, ``name+extra args`` for a
known agent with more flags, or ``name=argv`` for any command:

    ACP_E2E_AGENTS=claude,opencode hatch run test tests/e2e -v -s
    ACP_E2E_AGENTS='strands+--model ollama/qwen3.8:27b --memory off' hatch run test tests/e2e -v -s
    ACP_E2E_AGENTS='qwen+--auth-type openai --openai-base-url http://localhost:11434/v1 --openai-api-key ollama
        -m qwen3.8:27b' hatch run test tests/e2e -v -s
    MODEL_PROVIDER=ollama CODEX_CONFIG='{"model": "qwen3.8:27b"}' ACP_E2E_AGENTS=codex hatch run test tests/e2e -v -s

An agent that takes its provider from the environment (Codex) simply inherits it from this process.

Each scenario works in a fresh temporary directory holding a tiny Python file with an obvious bug,
so the agent has something concrete to read, explain and fix.
"""

from __future__ import annotations

import os
import shlex
import textwrap
from pathlib import Path
from typing import Any

import pytest

from strands_acp_client import KNOWN_AGENTS, AgentCommand, make_acp_agent
from tests.conftest import result_json, result_text

pytestmark = pytest.mark.e2e

CALC = textwrap.dedent(
    """\
    def add(a, b):
        return a - b
    """
)


def _selected_agents() -> dict[str, AgentCommand]:
    raw = os.environ.get("ACP_E2E_AGENTS", "").strip()
    if not raw:
        return {}
    agents: dict[str, AgentCommand] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item.split("+", 1)[0]:
            name, _, argv = item.partition("=")
            agents[name] = AgentCommand(tuple(shlex.split(argv)))
        else:
            name, _, extra = item.partition("+")
            agents[name] = KNOWN_AGENTS[name].with_args(*shlex.split(extra))
    return agents


AGENTS = _selected_agents()
if not AGENTS:
    pytest.skip("set ACP_E2E_AGENTS=claude,opencode to run the end-to-end tests", allow_module_level=True)

_KEEP = {"CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY"}


def _nesting_env() -> dict[str, str | None]:
    """Hide the host session from a nested Claude Code.

    When these tests are launched from inside a Claude Code session, the environment carries that
    session's variables (its auth is refreshed by the host app). A nested CLI that inherits them
    refuses to authenticate, so they are removed and the CLI falls back to its own stored login.
    """
    if "CLAUDECODE" not in os.environ:
        return {}
    prefixes = ("CLAUDE_", "ANTHROPIC_BASE_URL")
    hidden: dict[str, str | None] = {"CLAUDECODE": None}
    hidden.update({key: None for key in os.environ if key.startswith(prefixes) and key not in _KEEP})
    return hidden


NESTING_ENV = _nesting_env()


@pytest.fixture(params=sorted(AGENTS), ids=sorted(AGENTS))
def agent_name(request: Any) -> str:
    return str(request.param)


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    (tmp_path / "calc.py").write_text(CALC)
    return tmp_path


@pytest.fixture
async def make_real_tool(agent_name: str, workdir: Path):
    tools: list[Any] = []

    def factory(**overrides: Any) -> Any:
        options: dict[str, Any] = {
            "agents": {agent_name: AGENTS[agent_name]},
            "cwd": workdir,
            "timeout": float(os.environ.get("ACP_E2E_TIMEOUT", "240")),
            "env": NESTING_ENV,
        }
        options.update(overrides)
        built = make_acp_agent(**options)
        tools.append(built)
        return built

    yield factory

    for built in tools:
        await built.acp_sessions.close_all()


def _print(label: str, result: dict[str, Any]) -> None:
    details = result_json(result)
    print(
        f"\n[{label}] status={result['status']} stop={details['stop_reason']} mode={details['mode']} "
        f"elapsed={details['elapsed_seconds']}s violation={details.get('policy_violation')}"
    )
    print(
        f"  permissions={details['permissions']} tool_calls={[(c['kind'], c['status']) for c in details['tool_calls']]}"
    )
    print("  text:", result_text(result)[:400].replace("\n", " | "))


async def test_explains_the_bug_without_editing(make_real_tool, run_tool, workdir, agent_name):
    built = make_real_tool(permissions="read_only")
    _, result = await run_tool(
        built,
        task="Read calc.py in the current directory. In one short sentence, say whether add() has a bug "
        "and what it is. Do not modify any file.",
    )
    _print(f"{agent_name} read_only explain", result)
    assert result["status"] == "success", result_text(result)
    details = result_json(result)
    assert details["stop_reason"] == "end_turn"
    text = result_text(result).lower()
    assert any(word in text for word in ("subtract", "minus", "a - b", "-", "bug")), text
    assert (workdir / "calc.py").read_text() == CALC, "read_only must leave the file untouched"


async def test_fixes_the_bug_when_allowed(make_real_tool, run_tool, workdir, agent_name):
    built = make_real_tool(permissions="allow_all")
    _, result = await run_tool(
        built,
        task="calc.py in the current directory has a bug in add(): it subtracts. Fix it so add() returns a + b, "
        "change nothing else, and reply with the single word DONE when the file is saved.",
    )
    _print(f"{agent_name} allow_all fix", result)
    assert result["status"] == "success", result_text(result)
    details = result_json(result)
    fixed = (workdir / "calc.py").read_text()
    assert "a + b" in fixed.replace("return a+b", "return a + b"), fixed
    kinds = {call["kind"] for call in details["tool_calls"]}
    print("  kinds seen:", kinds)
    assert details["permissions"]["denied"] == 0


async def test_read_only_refuses_the_edit(make_real_tool, run_tool, workdir, agent_name):
    built = make_real_tool(permissions="read_only")
    _, result = await run_tool(
        built,
        task="Edit calc.py in the current directory so that add() returns a + b instead of a - b. "
        "If you are not allowed to edit, say so in one sentence.",
    )
    _print(f"{agent_name} read_only edit attempt", result)
    details = result_json(result)
    assert (workdir / "calc.py").read_text() == CALC, "the edit must have been refused"
    # Either the agent asked and was refused, or its read-only mode kept it from trying.
    assert details["permissions"]["allowed"] == 0
    assert "policy_violation" not in details, "no change may slip through unasked"


async def test_session_continues_with_session_id(make_real_tool, run_tool, agent_name):
    built = make_real_tool(permissions="read_only")
    _, first = await run_tool(
        built, task="Remember the code word PINEAPPLE for later in this conversation. Reply with the single word OK."
    )
    _print(f"{agent_name} session turn 1", first)
    assert first["status"] == "success", result_text(first)
    session_id = result_json(first)["session_id"]

    _, second = await run_tool(
        built, task="What was the code word I asked you to remember? Reply with just that word.", session_id=session_id
    )
    _print(f"{agent_name} session turn 2", second)
    assert second["status"] == "success", result_text(second)
    assert "pineapple" in result_text(second).lower()
    assert result_json(second)["session_id"] == session_id


async def test_custom_handler_is_never_bypassed_silently(make_real_tool, run_tool, workdir, agent_name):
    asked: list[str | None] = []

    def reads_only(request: Any) -> bool:
        asked.append(request.kind)
        return request.kind in ("read", "search", "fetch", "think")

    built = make_real_tool(permissions=reads_only)
    _, result = await run_tool(
        built,
        task="Edit calc.py in the current directory so that add() returns a + b instead of a - b. "
        "If you are not allowed to edit, say so in one sentence.",
    )
    _print(f"{agent_name} custom handler", result)
    details = result_json(result)
    print("  handler was asked about:", asked)
    if (workdir / "calc.py").read_text() != CALC:
        # The agent changed the file without consulting the handler: that must surface as an error.
        assert result["status"] == "error", "an unasked change must not be reported as success"
        assert details["stop_reason"] == "policy_violation"
        assert details["policy_violation"]["kind"] in ("edit", "execute")
    else:
        assert details["permissions"]["allowed"] == 0 or "edit" not in asked
