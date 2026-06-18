# Connect coding agents to Krill

Krill is a drop-in local backend for the popular terminal coding agents. One
command wires an agent to your local server:

```bash
krill launch <agent>
```

`krill launch` (no agent) prints the roster. It picks the first installed
model (or `--model <name>`), makes sure a server is running with that model
loaded (auto-starting one if needed), writes the agent's config / exports its
env, and then execs the agent so it owns your terminal.

```bash
krill launch                 # list agents
krill launch claude          # boot Claude Code wired to Krill
krill launch codex --model gemma-4-e2b
krill launch opencode --port 11434           # use a specific server port
krill launch claude -- --resume              # pass args after -- to the agent
```

Flags: `--model <name>`, `--port <port>`, `--host <host>`, `--no-serve` (require
an already-running server instead of auto-starting one). Anything after `--`
is forwarded to the agent binary.

> Coding agents have large system prompts + tool schemas. Prefer a model and a
> server context window of **at least 32k-64k tokens**, or the agent's prompt
> gets truncated and it behaves strangely.

## How it works: one server, three wire protocols

There is no per-agent proxy. Krill's server speaks the three wire protocols
these agents use, and `launch` just points each agent at the matching endpoint:

| Wire protocol | Krill endpoint | Agents |
|---|---|---|
| Anthropic Messages | `POST /v1/messages` | Claude Code |
| OpenAI Chat Completions | `POST /v1/chat/completions` | OpenCode, Hermes, Pi, Copilot CLI, Droid |
| OpenAI Responses | `POST /v1/responses` | Codex |

## Supported agents (`krill launch`)

| `launch` id | Agent | Wire | What it wires |
|---|---|---|---|
| `claude` | Claude Code | Anthropic | `ANTHROPIC_BASE_URL` + auth token + model aliases |
| `codex` | Codex CLI | Responses | isolated `config.toml` under a krill-owned `CODEX_HOME` (your real `~/.codex` is untouched) |
| `opencode` | OpenCode | OpenAI Chat | `krill` provider deep-merged into `~/.config/opencode/opencode.json` (`.bak` kept) |
| `hermes` | Hermes Agent | OpenAI Chat | `hermes config set model.*` |
| `pi` | Pi | OpenAI Chat | `krill` provider merged into `~/.pi/agent/models.json` |
| `copilot` | Copilot CLI | OpenAI Chat | `COPILOT_PROVIDER_BASE_URL` + `COPILOT_MODEL` env |
| `droid` | Droid (Factory) | OpenAI Chat | `custom_models` entry appended to `~/.factory/config.json` |

`claude`, `codex`, and `opencode` are verified end-to-end. `hermes`, `pi`,
`copilot`, and `droid` follow each tool's documented local-endpoint config and
are exercised once their binaries are present.

## Manual setup for agents not yet in `launch`

These speak the same protocols; point them at the server by hand. Start
`krill serve --model <name>` first (default port 57455).

### Codex desktop app (`codex-app`)

The desktop app reads your real `~/.codex/config.toml` (it does not inherit a
shell `CODEX_HOME`). Add a provider + profile, then open the app and pick the
`krill` profile/model:

```toml
[model_providers.krill]
name = "Krill"
base_url = "http://127.0.0.1:57455/v1"
wire_api = "responses"
env_key = "KRILL_API_KEY"

[profiles.krill]
model = "<model>"
model_provider = "krill"
```

### OpenClaw (`openclaw`)

OpenClaw is OpenAI-compatible; configure its provider with base URL
`http://127.0.0.1:57455/v1` and any non-empty API key (the local server ignores
it), then select your local model. Exact config path varies by version.

## Adding a new agent

Agent knowledge lives in one declarative table:
[`Sources/KrillCLI/AgentProfiles.swift`](../Sources/KrillCLI/AgentProfiles.swift).
Append one `AgentProfile` literal:

- `wire`: which protocol it speaks (picks the endpoint).
- `env`: env vars to export (templated over the server base URL + model).
- `configFiles`: files to `write` (krill-owned paths) or `mergeJSON` (deep-merge
  into the user's config, `.bak` kept, arrays concatenate + dedup).
- `preExec`: setup commands to run before launch (e.g. a `config set`).
- `binary` / `args` / `notInstalledHint`.

The `LaunchCommand` flow stays generic over the table; no other change is
needed beyond the one literal.

See also [`docs/SERVER_API.md`](SERVER_API.md) for the raw endpoint shapes.
