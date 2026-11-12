# strands-acp-client

[![PyPI](https://img.shields.io/pypi/v/strands-acp-client.svg)](https://pypi.org/project/strands-acp-client/)
[![CI](https://github.com/pablodevrzmz/strands-acp-client/actions/workflows/ci.yml/badge.svg)](https://github.com/pablodevrzmz/strands-acp-client/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

Use coding agents as tools from a [Strands Agents](https://strandsagents.com) agent, over the
[Agent Client Protocol](https://agentclientprotocol.com) (ACP). Verified end to end with Claude Code,
Codex, Kilo, OpenCode, Qwen Code and a Strands agent served by the Strands CLI; any other ACP agent
plugs in by its command.
Your agent delegates a task, the coding agent works in a directory with its own tools, and the final
answer comes back as a tool result together with a `session_id` for follow-ups.

ACP has two halves. The *Agent* half exposes an agent to editors like Zed; for Strands that already
exists as [strands-acp](https://github.com/ryancormack/strands-acp) (TypeScript). This package is the
*Client* half: the Strands agent takes the editor's seat, launches the coding agent, hands it the
task and decides its permission requests.

## Install

```bash
pip install strands-acp-client
```

The agents themselves are separate programs. Claude Code's adapter and the Strands CLI run through
`npx`, so Node.js 20+ must be on `PATH`; OpenCode is a binary you install first. Each agent needs its
own login or model provider exactly as it does in a terminal (for example `claude auth login` for
Claude Code, or a local Ollama model for the Strands CLI).

Codex authenticates over ACP rather than from a saved login. With `CODEX_API_KEY` or `OPENAI_API_KEY` in
the environment the tool completes that step by itself (`auth_method="api-key"` on the preset); a
local model provider needs no key at all.

For unattended runs, give the agent a credential that does not depend on an interactive login. Claude Code
issues long-lived tokens with `claude setup-token`; pass one as `CLAUDE_CODE_OAUTH_TOKEN` through the
factory's `env`, or through the agent's own `AgentCommand(env=...)`. An expired login surfaces as an error
result that quotes the agent's message, so the model can report it instead of retrying blindly.

`env` is applied over the environment of your process, and a value of `None` removes a variable. That
matters when your code itself runs inside a Claude Code session: the nested Claude Code inherits the
host session's variables and refuses to authenticate. Hide them and it uses its own stored login:

```python
import os

hide_host_session = {
    key: None
    for key in os.environ
    if key == "CLAUDECODE" or key.startswith(("CLAUDE_CODE_", "CLAUDE_AGENT_SDK_"))
}
acp_agent = make_acp_agent(agents=["claude"], env=hide_host_session)
```

## Quick start

```python
from strands import Agent
from strands_acp_client import make_acp_agent

acp_agent = make_acp_agent(
    agents=["claude", "opencode"],  # verified agents, or your own commands (see below)
    cwd="./repo",                # where the coding agents work
    timeout=900,                 # seconds per delegated task
    permissions="read_only",     # what the coding agent may do without asking you
)

agent = Agent(tools=[acp_agent])
agent("Ask claude to explain how the retry logic in src/http.py works and whether it has a bug.")
agent("Ask it, in the same session, how it would fix that.")

acp_agent.close()  # ends the agent processes; `with make_acp_agent(...) as acp_agent:` does it for you
```

The model sees one tool, `acp_agent`, with these inputs:

| Input | Meaning |
| --- | --- |
| `task` | What the coding agent should do, with the context it needs to work on its own. |
| `agent_name` | Which configured agent to use. Defaults to the first one (or `default_agent`). Left out when continuing a session. |
| `session_id` | A `session_id` from an earlier result, to continue that conversation. |
| `cwd` | Working directory for this call: the factory's `cwd` (the default) or a directory inside it. Left out when continuing a session. |

Every task is sent together with the absolute working directory, so relative paths in the task
resolve even for an agent whose model cannot run `pwd`. The directory is resolved to its real path
first: an agent that sees a symlinked path treats its own project as an external folder.

Each result carries the agent's final message as text plus a JSON block with `session_id`,
`stop_reason`, token `usage` (when the agent reports it), the `tool_calls` it made (id, title,
ACP kind, status, file locations), how many permission requests were allowed or denied, and the
elapsed time. When requests were refused, the text ends with a line naming them, because an agent
does not always say that it was turned down. An agent that ends its turn right after a refusal, as
Codex does, is reported as a success with that explanation, not as an error.

When something goes wrong the result is an error the model can act on. An agent that dies is reported
with its exit code and the last lines it wrote to stderr, which is where an agent says that it is not
logged in or that a package could not be found. An agent gets `startup_timeout` seconds (120 by
default, because a first run through `npx` downloads the agent) to start and open its session; one that
never answers, for instance because it is waiting for a login in a browser, is stopped and reported the
same way.

## Agents

`agents` accepts a list of known names, a single name, or a mapping of your own names to launch
commands. A name becomes known only after the end-to-end suite has driven that agent for real, so the
list is short on purpose:

| Name | Command | Verified with |
| --- | --- | --- |
| `claude` | `npx -y @agentclientprotocol/claude-agent-acp@0.101.0` | Claude Code 2.1, adapter 0.101 |
| `codex` | `npx -y @agentclientprotocol/codex-acp@2.6.0` | Codex adapter 2.6 on a local Ollama model |
| `kilo` | `npx -y @kilocode/cli@7.8.27 acp` | Kilo 7.8 on its free model, no account |
| `opencode` | `opencode acp` | OpenCode 1.18 |
| `qwen` | `npx -y @qwen-code/qwen-code@0.27.0 --acp` | Qwen Code 0.27 on a local Ollama model |
| `strands` | `npx -y @strands-agents/cli@0.2.4 --acp-server` | Strands CLI 0.2.4 on a local Ollama model |

Each `npx` command names the version that passed the suite. `npx -y` on a bare package name runs whatever
its latest release happens to be, and a new release can change how the agent asks for permission. To run
another version, pass its command under the same name; it keeps everything else the package knows about
that agent:

```python
acp_agent = make_acp_agent(agents={"claude": ["npx", "-y", "@agentclientprotocol/claude-agent-acp@0.102.0"]})
```

The model can send an agent to a directory inside the factory's `cwd`, by an absolute path or one relative
to it, and nowhere else: the directory is the model's choice, and a prompt injection should not be able
to point a coding agent at the rest of the machine. `confine_cwd=False` lifts that.

Anything else that speaks ACP on stdin/stdout works by passing its command; the
[ACP registry](https://github.com/agentclientprotocol/registry) lists them. An agent outside the table
has not been checked here, so find out how it asks for permission before trusting a restrictive policy
with it (see [A policy only answers questions](#a-policy-only-answers-questions)):

```python
from strands_acp_client import AgentCommand, make_acp_agent

acp_agent = make_acp_agent(
    agents={
        "claude": "claude",  # alias of a known agent
        "reviewer": AgentCommand(("my-review-agent", "--acp"), "an agent tuned for code review"),
        "local": ["python", "my_acp_agent.py"],
    },
    default_agent="claude",
)
```

The description you give each agent is shown to the model, so say what each one is for.

To add flags to a known agent, for example to choose its model, extend it instead of rewriting the
command, so it keeps what the package knows about it:

```python
from strands_acp_client import KNOWN_AGENTS, make_acp_agent

local_strands = KNOWN_AGENTS["strands"].with_args("--model", "ollama/qwen3.8:27b", "--memory", "off")
acp_agent = make_acp_agent(agents={"strands": local_strands})
```

That last example costs nothing to run: the Strands CLI serves a Strands agent over ACP on a local
Ollama model, and another Strands agent delegates to it. Qwen Code runs on the same local model
through Ollama's OpenAI-compatible endpoint:

```python
local_qwen = KNOWN_AGENTS["qwen"].with_args(
    "--auth-type", "openai",
    "--openai-base-url", "http://localhost:11434/v1",
    "--openai-api-key", "ollama",
    "-m", "qwen3.8:27b",
)
```

Codex takes its provider from the environment instead, so extend it with `with_env`:

```python
local_codex = KNOWN_AGENTS["codex"].with_env(MODEL_PROVIDER="ollama", CODEX_CONFIG='{"model": "qwen3.8:27b"}')
```

On a model Codex does not know, each answer starts with a one-line warning about missing model
metadata. It comes from Codex and is left as it is.

### Choosing a model

Agents announce their session options (model, effort, mode...) and the choices for each when a
session opens. Set them per agent with `config`, or `with_config` on a known agent:

```python
sonnet = KNOWN_AGENTS["claude"].with_config(model="sonnet")
```

The `kilo` preset already does this. Kilo's default model needs a Kilo account, so the preset selects
its free router, `kilo/kilo-auto/free`, which does not. An option or a value the agent does not offer
is an error that lists the choices, so a misspelled model never falls back silently to a default you
may be paying for. The options in effect are reported as `config` in the result.

Kilo's free router comes with an anonymous allowance. Once it is used up, Kilo answers with a request
to sign up (free) or come back later, which the tool reports as an error quoting that message. To run
Kilo on a local model instead, declare the provider in Kilo's own `kilo.json` and select it the same way:

```json
{
  "provider": {
    "ollama": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://localhost:11434/v1" },
      "models": { "qwen3.8:27b": {} }
    }
  }
}
```

```python
local_kilo = KNOWN_AGENTS["kilo"].with_config(model="ollama/qwen3.8:27b")
```

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

A handler is your code, so it runs on the event loop of the tool call it answers for, where the rest of
your application runs, and not on the tool's own thread (see [Sessions](#sessions)).

A refused request is answered with the agent's own `reject` option, so the agent sees a refusal and
can explain itself or try something else; the turn is not aborted. To gate the delegation itself (is
this task worth handing to a coding agent at all?), use a Strands intervention handler on the
`acp_agent` tool call, as you would for any other tool.

### A policy only answers questions

A permission policy decides what to answer when the agent asks. Whether the agent asks is up to the
agent, and many start in a mode that does not: Claude Code often starts in `auto`, OpenCode in `build`.
Tested against the real agents, both edited a file under `read_only` without sending a single
permission request. So the tool does two more things under every policy except `allow_all`:

1. **It puts the agent in a mode that respects the policy.** Each known agent declares, per policy,
   either the session mode in which it defers to the client, selected through ACP (`session/set_mode`
   or the `mode` config option) and reported as `mode` in the result, or the launch arguments that
   restrict what it can do. An agent that does not offer the mode declared for it is not started: the
   call returns an error listing the modes it does offer.
2. **It checks that the agent asked, and that it took no for an answer** (`strict=True`, the default).
   If an edit, delete, move or command completes without a permission request, or after its request
   was refused, the turn is cancelled at once and the tool returns an error with
   `stop_reason: "policy_violation"` naming the call, instead of a success that hides it.

| Agent | Starts in | Under a restrictive policy | An edit under `read_only` |
| --- | --- | --- | --- |
| `claude` | your Claude Code settings, often `auto` | switched to `default` ("always ask before making changes") | the agent asks, the policy refuses |
| `codex` | `agent`, where it runs and edits on its own | switched to `read-only` | its sandbox lets commands read; it asks before any change and the policy refuses |
| `qwen` | `auto`, where a classifier decides | switched to `default` | the agent asks, the policy refuses |
| `kilo` | `code`, which edits and runs commands unasked | switched to `ask` for `read_only` and `deny_all` | the agent declines on its own, and its shell policy blocks commands that write |
| `opencode` | `build` | switched to `plan` for `read_only` and `deny_all` | the agent declines on its own |
| `strands` | every built-in tool, nobody asked | started with `--builtin-tools read,web_fetch` for `read_only` and `deny_all` | the agent has no tool to edit with |
| others | their own default | nothing known yet | reported by `strict` if they do not ask |

For your own agents, declare the modes or the arguments on the command:

```python
AgentCommand(("my-agent", "--acp"), "my agent", modes={"read_only": "ask", "deny_all": "ask", "custom": "ask"})
AgentCommand(("my-agent", "--acp"), "my agent", policy_args={"read_only": ("--no-write",)})
```

Two more fields cover what Codex needed. `auth_method` names the ACP authentication method to run when
the agent answers "authentication required". `sandboxed_kinds` lists tool kinds the agent confines by
itself once its restricted mode is active: Codex reads files with shell commands inside an
operating-system sandbox, and Kilo's `ask` mode filters shell commands through its own allowlist, so an
unasked `execute` is expected there and is not treated as a violation.

OpenCode, Kilo and the Strands CLI (0.2.x) have no way to consult a custom handler: OpenCode's `build`
mode, Kilo's `code` mode and the Strands CLI's stock tools act without asking, and `strict` reports
that as a violation. For OpenCode, make it ask in its own configuration (`"permission": {"edit": "ask", "bash": "ask"}` in
`opencode.json`); Kilo's equivalent setting has not been verified here. For the Strands CLI, give the agent only the tools the task needs.

`strict` detects a violation after the first unasked change has happened; it stops the rest and tells
you, it cannot undo that change. Treat the policy as a contract with a cooperative agent, not as a
sandbox. When you need a hard guarantee, run the agent where it cannot do damage: a container, a
read-only checkout, or an AgentCore Runtime session.

## Sessions

Every call returns a `session_id`. Passing it back, on its own, continues the same conversation with
the same agent process. The session knows which agent it belongs to and which directory it works in, so
`agent_name` and `cwd` only matter when a session starts; naming a different agent or directory for an
existing session is an error, and the session carries on as it was.

Agent processes stay alive between calls (`keep_sessions=True`). There are at most `max_sessions` of
them: beyond that the least recently used idle session is closed, never one that a call is using.
Calls that share a session take turns. Set `keep_sessions=False` to start a fresh process for every
task.

The processes live on an event loop that belongs to the tool, in a background thread it starts on first
use. That is why a session outlives the call that started it. Strands runs each synchronous
`agent(...)` call on a new event loop and closes it afterwards, so the two calls in the quick start run
on two different loops and still reach the same Claude Code process. The same holds for `invoke_async`,
a web server, AgentCore Runtime, or several threads at once.

A session survives a task that failed, timed out or was cancelled, as long as the agent stopped when it
was asked to, so the model can pick the conversation up again. It is closed when its process dies, when
the agent ignores a cancellation for `cancel_grace` seconds, and after a policy violation. What happens
next depends on the agent:

- Agents that advertise `loadSession` (Claude Code's adapter does) get the session resumed by id in a
  new process, by the agent that owns it. The result then says `"resumed": true`.
- For the others the session is over. The result carries `"session_id": null` and
  `"session_closed": true`, so the model does not try to continue something that is gone.

### Closing

```python
acp_agent.close()                             # every agent process, and the background thread
await acp_agent.aclose()                      # the same, without holding up your event loop
await acp_agent.acp_sessions.close_all()      # only the processes
await acp_agent.acp_sessions.close(session_id)
```

`with make_acp_agent(...) as acp_agent:` and `async with` close on the way out. A closed tool can be
used again; it starts new processes when it needs them. A tool you never close ends its processes when
it is garbage collected and when the interpreter exits.

Each agent is started as the leader of its own process group, and closing addresses the group: its
stdin is closed first, which is how an agent on stdio is told to leave, then the group gets SIGTERM, then
SIGKILL. An agent launched through a wrapper, as `npx` does, cannot outlive its wrapper this way. On
Windows, where there is no such group, only the launched process is terminated.

The same group keeps a Ctrl-C in your terminal from reaching the agents. A Strands agent does not stop
a running tool on Ctrl-C either, in a synchronous `agent(...)` call or under `asyncio.run`, so the
interrupt only takes effect once the delegated turn is over, which can take up to `timeout` seconds. To
stop it at once, cancel the agent when Ctrl-C arrives. The tool watches the agent's cancellation signal
and cancels the coding agent's turn with it:

```python
import signal

signal.signal(signal.SIGINT, lambda *_: agent.cancel())
result = agent("Ask claude to explain src/http.py.")  # ends with stop_reason "cancelled" on Ctrl-C
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
- No file-system or terminal delegation: the coding agent uses its own tools in `cwd`. The permission
  policy and `strict` hold a cooperative agent to the rules; they are not a sandbox.
- Permission requests are answered by policy or by your callable. A mode that pauses the Strands
  agent with an interrupt and asks a human is planned.
- The agents' own authentication is not handled here; log in as you would in a terminal.
- A process that forks after using the tool gets a child with no sessions of its own: the agents stay
  with the parent, and the child starts new ones on its first call. While such a child is alive it also
  holds the agents' pipes open, so when the parent closes the tool its agents do not see their input end
  and are stopped with SIGTERM after two seconds instead. Create the tool after forking (in each worker of
  a prefork server, for instance) to avoid that.

## Development

```bash
pip install hatch
hatch run prepare   # format, lint, typecheck, tests
```

Tests run against a scripted ACP agent in `tests/fake_acp_agent.py`, so they need no API keys.

An opt-in end-to-end suite drives real agents installed on your machine. It spends model tokens and
needs the agents logged in, so it only runs when you name them:

```bash
ACP_E2E_AGENTS=claude,opencode hatch run test tests/e2e -v -s
```

Each scenario gives the agent a small file with an obvious bug and checks five things: it can explain
the bug without touching the file, it fixes it under `allow_all`, the edit does not happen under
`read_only`, a custom handler is either consulted or the unasked change is reported as a violation,
and a second task with the returned `session_id` remembers the first. Claude Code 2.1 (adapter 0.101),
OpenCode 1.18 and Kilo 7.8 pass all five, and so do Codex (adapter 2.6), Qwen Code 0.27 and the Strands
CLI 0.2.4 on a local Ollama model. Kilo was verified both on its free router and on local Ollama;
it and those three need no account and no key:

```bash
ACP_E2E_AGENTS='strands+--model ollama/qwen3.8:27b --memory off' hatch run test tests/e2e -v -s
```
The `agent-client-protocol` SDK supports Python 3.10 to 3.14; if Hatch picks a newer interpreter,
point it at a supported one with `HATCH_PYTHON=/path/to/python3.14`.

## License

Apache-2.0. See [LICENSE](LICENSE).
