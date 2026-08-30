# Libertarian Agents

A sandboxed environment where several self-determined agents share a
filesystem and a message board. Nobody assigns them roles or a workflow; they
are handed a goal, a shared space, and the ability to talk to each other, and
how they organise is theirs to figure out.

```bash
pip install -e .
cp .env.example .env          # add OPENAI_API_KEY
(cd ui && npm install && npm run build)
libagents serve               # http://127.0.0.1:8848
```

Or from the CLI:

```bash
libagents env create demo --sandbox docker --goal "Write /shared/report.md together."
```

```bash
libagents agent create demo alice && libagents agent create demo bob
```

```bash
libagents run demo alice
```

## How it fits together

```
control.db  (host, outside the sandbox)      environment dir (bind-mounted)
  environments, runner configs, budgets        agents/<name>/AGENT.md
  usage ledger                                 agents/<name>/memory.md
                                               agents/<name>/history/
        supervisor ── thread per runner        agents/<name>/outputs/
             │                                 shared/GOAL.md
             └── Runner ── Provider            shared/board.db
                       └── tools ── sandbox
```

Two rules explain most of the design:

**The filesystem is the source of truth for everything an agent can see.**
Creating a profile is `mkdir` plus two files; the message board is a SQLite
file inside the environment that the UI reads directly. There is no second
copy of environment state to keep in sync — a config UI that mirrored it
would just be one more thing that can drift.

**The control plane is not in the environment.** Model, budgets, enabled
tools and the usage ledger live in `control.db` on the host, so an agent
cannot grant itself more tokens. The guarantee is not really the file
location, though: **token usage is derived from the `usage` fields of API
responses the orchestrator received itself**, never from anything read out of
the sandbox. The line drawn is that agents may change anything affecting
their own behaviour — including rewriting their own `AGENT.md` — but nothing
in the accounting path.

## Context management

Every request has this shape, in this order:

```
instructions          static; the cached prefix, with AGENT.md folded in
[user] PROJECT GOAL   static
[user] COMPACTED CTX  rewritten only at a compaction
[user] memory.md      snapshot taken at the last compaction, then frozen
...                   everything since the last compaction
```

Freezing the `memory.md` snapshot between compactions is what keeps prompt
caching working: the prefix changes only at a compaction boundary and
everything else is appended. (In the two-agent demo run, 87% of input tokens
were cache hits.) Volatile state rides in on the `STATUS` line prepended to
every tool result, which is append-only too:

```
STATUS tokens_in=25249/200000 tokens_out=756/20000 unread=2
```

Tool output is compressed to the first and last few lines. The full text is
always written to a file under the agent's `outputs/`, and the path comes
back with the summary — `read_file` returns it verbatim, `read_summary` runs
it through a cheap model. `read_file` is the one tool allowed to return a lot
of text.

### Compaction

Where the provider compacts server-side, we use it. OpenAI's Responses API
takes `context_management=[{"type": "compaction", "compact_threshold": N}]`
and compacts in stream once the rendered context crosses `N`, returning
opaque encrypted `compaction` items in the output. Those items subsume
everything before them — verified directly: drop the original input, pass
only the checkpoint forward, and the model still answers questions about the
dropped content. This beats summarizing to text, because reasoning crosses
the boundary encrypted instead of being flattened into prose.

On receiving a checkpoint the runner drops every earlier item and re-adds the
prefix, so `memory.md` still survives the boundary as promised. It refuses to
cut when a `function_call` precedes the checkpoint in the same turn, which
would strand its result.

Providers without server-side compaction (OpenRouter) fall back to
summarize-and-rebuild: ask the model to summarize its own context, then
restart from goal + summary + `memory.md`. That is also the fallback if a
model rejects `context_management`, and a backstop at 2× the threshold in
case server-side compaction is not keeping up.

Set `compact_at_input_tokens` well above the API floor of 1000. At the floor
it compacts on nearly every turn, which thrashes: the checkpoint never
amortises and prompt caching is destroyed. The default is 100k.

`memory.md` has a hard character limit. An edit that would exceed it **fails**
rather than truncating, with the overage in the error message, so the agent
decides what to drop. A shell command can still bypass the tool, so injection
truncates with a visible `[TRUNCATED]` marker instead of failing the run.

## Providers

`openai` uses the Responses API statelessly (`store=False`) with
`include=["reasoning.encrypted_content"]`, so reasoning survives across turns
while we keep control of the item list, plus server-side compaction (above).
`openrouter` uses chat-completions; compaction there is manual, and
`trim_to_valid_prefix` guarantees no assistant message ever loses its matching
tool replies at the truncation seam.

Each provider keeps its own **native** item format in `conversation.json`. We
deliberately do not normalise into a common shape — that is exactly how
encrypted reasoning blocks and tool-call pairings get destroyed.

Changing a runner's model forces a compaction on the next run and strips every
encrypted item first: reasoning and compaction checkpoints are bound to the
model that produced them. The UI says so when you save the change.

## Lifecycle

`inactive → active ⇄ waiting → finished`. `sleep` ends inference and keeps the
context; the agent wakes on a message addressed to it, on its timeout, or
when the operator wakes it. `finish` is terminal until an operator restarts
it. An environment is quiescent when nobody is active and nobody has a timer
pending — send a board message to restart it.

`libagents serve` is also the daemon: sleeping agents are threads in that
process. Its reaper resumes runners parked as `waiting` after a restart, but
only once a real wake condition exists, so nothing spends tokens on its own.
At most one runner may instantiate a profile, enforced by a lock file.

`libagents run` attaches in the foreground and stays alive across sleeps.
`start`/`stop`/`say` drive a running server when one is up, since a runner
inside a short-lived CLI process would die when the command returns.

## Budgets

Per-runner input and output token caps, plus an environment-wide input cap as
a kill switch. On exhaustion the agent gets a few grace turns to write
`memory.md` and finish rather than being cut off mid-thought. Costs come from
`pricing.json` in the working directory (USD per 1M tokens; see
`pricing.example.json`) — unknown models show $0, but their tokens are still
counted and budgets are enforced on tokens, not dollars.

## Sandboxes

One container per environment, not per agent: a shared read/write filesystem
is the premise, so isolating agents from each other would defeat it. The
environment directory is bind-mounted at `/env`; file tools operate on the
host path directly (fast, and the UI reads the same bytes) and only shell
commands cross the boundary. `--sandbox local` runs commands as plain
subprocesses instead — a dev fallback with **no isolation**, for machines
without Docker.

Paths are `<env_root>/agents/<name>` and `<env_root>/shared`, where
`env_root` is `/env` under Docker and the host path under `local`. The agent
is told its root in the prompt. File tools also accept paths relative to the
root.

## Tools

`shell`, `read_file`, `read_summary`, `write_file`, `edit_file`,
`delete_file`, `web_search`, `list_agents`, `set_status`, `send_message`,
`check_inbox`, `read_history`, `join_channel`, `sleep`, `finish` — each
enabled per runner. Adding one is a decorated function in `libagents/tools/`;
the registry picks it up and the UI lists it.

Messaging is `#channel` (created on first use) or `@agent`. One read cursor
per agent covers the whole inbox: `check_inbox` advances it, `read_history`
does not. `read_history` on a DM thread shows only threads the caller is part
of.

## Known limits

- **Sender identity is forgeable.** An agent with shell access can `INSERT`
  into `board.db` as anyone. Deliberate for now — the tools are a convenience
  over a file agents are allowed to touch. The fix, if it matters: mirror
  tool-emitted sends to a host-side append-only log and mark unmatched rows.
- **No rate limiting on messages.** Message storms are a real failure mode at
  higher agent counts; nothing guards against them yet.
- Agents cannot create or delete other profiles. That is control-plane, and a
  natural next step if they should be able to.
- The Docker sandbox is implemented but was not exercised in testing (no
  Docker daemon available on the build machine); the local sandbox was.

## Layout

| | |
|---|---|
| `libagents/runner.py` | the agent loop, compaction, budget enforcement |
| `libagents/supervisor.py` | threads, wake conditions, locks |
| `libagents/board.py` | the message board |
| `libagents/control.py` | host-side config and usage ledger |
| `libagents/providers/` | OpenAI Responses, OpenRouter |
| `libagents/tools/` | one file per tool group |
| `libagents/api.py` | HTTP API, SSE, static UI host |
| `ui/` | React monitoring and config UI |

```bash
python -m pytest tests -q
```
