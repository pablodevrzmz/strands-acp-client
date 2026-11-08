# strands-acp-client

[![PyPI](https://img.shields.io/pypi/v/strands-acp-client.svg)](https://pypi.org/project/strands-acp-client/)
[![CI](https://github.com/pablodevrzmz/strands-acp-client/actions/workflows/ci.yml/badge.svg)](https://github.com/pablodevrzmz/strands-acp-client/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

Use coding agents as tools from a [Strands Agents](https://strandsagents.com) agent, over the
[Agent Client Protocol](https://agentclientprotocol.com) (ACP): Claude Code, Codex, Gemini CLI,
GitHub Copilot CLI, Kiro CLI, OpenCode, Qwen Code, a Strands agent served by the Strands CLI, or any
other ACP agent. Your agent delegates a task, the coding agent works in a directory with its own
tools, and the final answer comes back as a tool result together with a `session_id` for follow-ups.

ACP has two halves. The *Agent* half exposes an agent to editors like Zed; for Strands that already
exists as [strands-acp](https://github.com/ryancormack/strands-acp) (TypeScript). This package is the
*Client* half: the Strands agent takes the editor's seat, launches the coding agent, hands it the
task and decides its permission requests.

## Install

```bash
pip install strands-acp-client
```

The agents themselves are separate programs. Most official adapters run through `npx`, so Node.js 20+
must be on `PATH`, and each agent needs its own login or API key exactly as it does in a terminal
(for example `claude login` for Claude Code, `codex login` for Codex). Kiro CLI and OpenCode ship as
binaries you install first.

For unattended runs, give the agent a credential that does not depend on an interactive login. Claude Code
issues long-lived tokens with `claude setup-token`; pass one as `CLAUDE_CODE_OAUTH_TOKEN` through the
factory's `env`, or through the agent's own `AgentCommand(env=...)`. An expired login surfaces as an error
result that quotes the agent's message, so the model can report it instead of retrying blindly.

## Quick start

```python
from strands import Agent
from strands_acp_client import make_acp_agent

acp_agent = make_acp_agent(
    agents=["claude", "codex"],  # known agents, or your own commands (see below)
    cwd="./repo",                # where the coding agents work
    timeout=900,                 # seconds per delegated task
    permissions="read_only",     # what the coding agent may do without asking you
)

agent = Agent(tools=[acp_agent])
agent("Ask claude to explain how the retry logic in src/http.py works and whether it has a bug.")
```

The model sees one tool, `acp_agent`, with these inputs:

| Input | Meaning |
| --- | --- |
| `task` | What the coding agent should do, with the context it needs to work on its own. |
| `agent_name` | Which configured agent to use. Defaults to the first one (or `default_agent`). |
| `session_id` | A `session_id` from an earlier result, to continue that conversation. |
| `cwd` | Working directory for this call. Defaults to the factory's `cwd`. |

Each result carries the agent's final message as text plus a JSON block with `session_id`,
`stop_reason`, token `usage` (when the agent reports it), the `tool_calls` it made (id, title,
ACP kind, status, file locations), how many permission requests were allowed or denied, and the
elapsed time.

## Agents

`agents` accepts a list of known names, a single name, or a mapping of your own names to launch
commands. Commands are taken from the [ACP registry](https://github.com/agentclientprotocol/registry):

| Name | Command | Notes |
| --- | --- | --- |
| `claude` | `npx -y @agentclientprotocol/claude-agent-acp` | Claude Code through the official adapter |
| `codex` | `npx -y @agentclientprotocol/codex-acp` | OpenAI Codex CLI through the official adapter |
| `gemini` | `npx -y @google/gemini-cli --acp` | Gemini CLI |
| `copilot` | `npx -y @github/copilot --acp` | GitHub Copilot CLI |
| `qwen` | `npx -y @qwen-code/qwen-code --acp` | Qwen Code |
| `kiro` | `kiro-cli acp` | Kiro CLI, installed separately |
| `opencode` | `opencode acp` | OpenCode binary, installed separately |
| `strands` | `npx -y @strands-agents/cli --acp-server` | A Strands agent served by the Strands CLI |

Anything that speaks ACP on stdin/stdout works:

```python
from strands_acp_client import AgentCommand, make_acp_agent

acp_agent = make_acp_agent(
    agents={
        "claude": "claude",  # alias of a known agent
        "reviewer": AgentCommand(("npx", "-y", "@agentclientprotocol/codex-acp"), "Codex, used for reviews"),
        "local": ["python", "my_acp_agent.py"],
    },
    default_agent="claude",
)
```

The description you give each agent is shown to the model, so say what each one is for.

## Permissions

Under ACP the client owns permissions: when the coding agent wants to do something its own rules
gate (edit a file, run a command), it asks. This tool answers from a policy, without adding another
permission system on top of the intervention handlers Strands already has:

| `permissions` | Behaviour |
| --- | --- |
| `"read_only"` (default) | Allows `read`, `search`, `fetch` and `think`; refuses edits, deletes, moves and commands. |
| `"allow_all"` | Grants everything. Right for sandboxed runs, such as a container or an AgentCore Runtime session. |
| `"deny_all"` | Refuses everything. |
| a callable | Receives a `PermissionRequest` (agent, tool call title, ACP kind, file locations, offered options) and returns `True`, `False` or an `option_id`. May be async. |

```python
from strands_acp_client import PermissionRequest, make_acp_agent

def inside_src_only(request: PermissionRequest) -> bool:
    if request.kind in ("read", "search", "fetch", "think"):
        return True
    return request.kind == "edit" and all("/src/" in path for path in request.locations)

acp_agent = make_acp_agent(agents=["claude"], permissions=inside_src_only)
```

A refused request is answered with the agent's own `reject` option, so the agent sees a refusal and
can explain itself or try something else; the turn is not aborted. To gate the delegation itself (is
this task worth handing to a coding agent at all?), use a Strands intervention handler on the
`acp_agent` tool call, as you would for any other tool.

## Sessions

Every call returns a `session_id`. Passing it back continues the same conversation with the same
agent process, so the model can ask follow-up questions or request changes to earlier work. Agent
processes stay alive between calls (`keep_sessions=True`, at most `max_sessions` of them, least
recently used closed first). Set `keep_sessions=False` to start a fresh process for every task.

Two things to know:

- A session lives in the event loop that created it. Strands runs each synchronous `agent(...)` call
  on its own loop, so a `session_id` from a previous synchronous call cannot be reused directly. Inside
  one invocation, and in async applications (`invoke_async`, a web server, AgentCore Runtime), sessions
  carry over as expected.
- When the process holding a session is gone, agents that advertise `loadSession` (Claude Code's
  adapter does) get the session resumed by id in a new process; the result then says `"resumed": true`.
  Other agents answer with an error asking you to start a new session.

Stop every agent process with:

```python
await acp_agent.acp_sessions.close_all()
```

## Watching the agent work

While a task runs, the tool streams progress events (`message_delta`, `tool_call`,
`tool_call_update`, `permission`) that surface as `tool_stream` events in `Agent.stream_async` and in
callback handlers. ACP tool kinds (`read`, `edit`, `execute`...) come through on every tool-call
event, so a UI can render what the coding agent is doing without parsing its output.

## Running on Amazon Bedrock AgentCore Runtime

The tool is a plain Strands tool, so it runs anywhere the agent runs. For AgentCore Runtime, the
container image needs Node.js and the agent program (or the `npx` cache warmed at build time), plus
the agent's credentials injected as environment variables. Each Runtime session is its own microVM,
so `permissions="allow_all"` with `keep_sessions=True` is a reasonable setting there: the coding agent
can edit and run freely inside the sandbox, and follow-up turns reuse the process.

## How this relates to other pieces

- [strands-acp](https://github.com/ryancormack/strands-acp) exposes a Strands agent to ACP clients
  (TypeScript, Agent side). This package drives ACP agents from a Strands agent (Python, Client side).
- [strands-labs/ai-functions](https://github.com/strands-labs/ai-functions) includes an ACP-driven
  Kiro thread inside its own threads framework. This package is agent-agnostic and a plain tool.
- A2A is for talking to remote agents over HTTP, MCP is for tools, ACP is for coding agents that
  run as local processes with their own tools. Pick by what sits on the other side.

## Limits

- ACP protocol version 1, which every listed agent still serves. The ACP v2 draft removes the client
  file-system and terminal methods; this client never advertised them, so the move will be small.
- No file-system or terminal delegation: the coding agent uses its own tools in `cwd`. The tool does
  not sandbox it beyond the permission policy.
- Permission requests are answered by policy or by your callable. A mode that pauses the Strands
  agent with an interrupt and asks a human is planned.
- The agents' own authentication is not handled here; log in as you would in a terminal.

## Development

```bash
pip install hatch
hatch run prepare   # format, lint, typecheck, tests
```

Tests run against a scripted ACP agent in `tests/fake_acp_agent.py`, so they need no API keys.
The `agent-client-protocol` SDK supports Python 3.10 to 3.14; if Hatch picks a newer interpreter,
point it at a supported one with `HATCH_PYTHON=/path/to/python3.14`.

## License

Apache-2.0. See [LICENSE](LICENSE).
