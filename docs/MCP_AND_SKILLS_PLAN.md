# MCP + Skills support for Krill

Status: Proposed &nbsp;·&nbsp; Date: 2026-08-02 &nbsp;·&nbsp; Owner: Sourav

## 0. TL;DR

1. **New `KrillMCP` target**: a Model Context Protocol *client* (streamable-HTTP + stdio transports, OAuth 2.1 PKCE, Keychain token storage) that connects to **any** MCP server — INDmoney is the first real consumer, not the design boundary.
2. **`MCPTool: Tool` adapter**: every remote MCP tool becomes an ordinary `ToolRegistry` entry (`mcp__<server>__<tool>`), so `AgentLoop` needs **zero changes** — it is already schema-driven.
3. **Collapse the three hard-coded toolset arrays** into one shared `defaultToolset()` before appending MCP tools, or the TUI/voice paths silently diverge.
4. **Skills**: a `SkillsLoader` in KrillHarness reading `SKILL.md` packages (Claude-compatible format) with progressive disclosure — names+descriptions in the system prompt, bodies loaded on demand through the existing tool-observation channel.
5. **Config**: sidecar `~/.krill/mcp.json` + `krill mcp add|list|login|remove` CLI; the flat TOML parser cannot express server tables and should not be upgraded just for this.
6. Ship in four phases; each lands with tests and is independently useful.

## 1. Context

`krill code` has a capable agentic loop (`Sources/KrillHarness/AgentLoop.swift`) with
grammar-constrained tool calling (`--constrain-args`, `Sources/KrillGrammar/`), a real
permission model (`Sources/KrillHarness/Permission.swift`), and OpenAI-compatible
server-side tool calls (`Sources/KrillServer/Server.swift:1186 handleToolChat`). But its
tool surface is closed: three call sites hard-code the built-in tool arrays
(`Sources/KrillCLI/Code/CodeCommand.swift:129`, `Sources/KrillCLI/TUI/ChatTUI.swift:1218
agentTools()`, `Sources/KrillCLI/Voice/VoicePanel.swift:552`), and there is no way to
attach external capabilities without recompiling.

MCP is the industry-standard answer (Claude, Codex, and every serious agent host speak
it), and real servers we already want exist today: INDmoney's portfolio server
(`https://mcp.indmoney.com/mcp`, OAuth 2.1, read-only), GitHub, browser automation, and
every stdio server on npm/PyPI. Skills are the complementary standard for *packaged
instructions* — versioned prompt+resource bundles the model pulls in on demand.

Constraints discovered in the codebase survey (2026-08-02):

- **No streaming HTTP client exists.** SSE is only ever *written* (`Server.writeSSEJSON`);
  `WebFetchTool`'s `WebFetcher`/`BoundedCollector` is request/response only.
- **No OAuth, no Keychain usage anywhere.** API keys live in plaintext
  `~/.krill/config.toml` (`KrillConfig`, `Sources/KrillRegistry/Config.swift`).
- **The TOML parser is flat** — `mergeFromTOML` (Config.swift:221) skips `[section]`
  headers entirely; `[[mcp_servers]]` tables cannot be expressed.
- **Arbitrary JSON Schemas are a real hazard**: `SchemaGrammar.compile` is total but
  relaxes unsupported keywords; `EngineGenerator.completeConstrained` caps constrained
  output at 256 tokens; `ToolNameAutomaton` constrains the name slot to offered names.
  MCP schemas arrive `$ref`-laden and unbounded.
- **KrillAgent's `OperatorTool` is a separate, deliberately isolated system** — MCP
  support goes into the KrillHarness/KrillTooling world only; the self-operator stays
  scoped away from external capability.

## 2. Decision

### 2.1 New target: `KrillMCP`

SwiftPM target depending on `KrillTooling` only (mirroring how KrillHarness stays
MLX-free). Added to `Package.swift` and the `docs/ARCHITECTURE.md` dependency graph.

```
KrillMCP
├── MCPClient.swift        // session lifecycle: initialize → tools/list → tools/call
├── MCPTransport.swift     // protocol MCPTransport { send(request) async throws -> stream }
├── HTTPTransport.swift    // streamable HTTP: POST JSON-RPC, parse SSE *and* plain JSON
│                          // responses, Mcp-Session-Id header, resumability optional
├── StdioTransport.swift   // child process, newline-delimited JSON-RPC over pipes
├── MCPAuth.swift          // OAuth 2.1 + PKCE: RFC 9728 protected-resource discovery,
│                          // RFC 8414 metadata, RFC 7591 dynamic client registration,
│                          // loopback redirect listener, refresh-token rotation
├── TokenStore.swift       // Keychain (kSecClassGenericPassword, service "krill-mcp");
│                          // first Keychain use in the codebase — deliberate: OAuth
│                          // tokens are credentials to live financial data and do NOT
│                          // follow api-keys-in-plaintext-TOML precedent
├── MCPToolAdapter.swift   // MCPTool: Tool (see 2.2)
└── MCPConfig.swift        // ~/.krill/mcp.json load/save
```

JSON-RPC layer is hand-rolled `Codable` (the protocol surface we need — `initialize`,
`notifications/initialized`, `tools/list`, `tools/call`, `ping` — is small; no dependency
on a third-party MCP SDK). Protocol revision pinned to `2025-06-18` with the
`MCP-Protocol-Version` header on HTTP.

Explicitly **out of scope for v1**: resources, prompts, sampling, elicitation,
notifications-driven tool-list refresh (we re-list on session start), and acting as an
MCP *server*. Each can be added later without reshaping this design.

### 2.2 Tool adapter — the load-bearing simplification

```swift
struct MCPTool: Tool {
    let serverName: String
    let remote: MCPToolDescriptor      // name, description, inputSchema, annotations
    let client: MCPClient
    var name: String { "mcp__\(serverName)__\(remote.name)" }
    var parametersJSON: String { remote.inputSchemaJSON }   // verbatim passthrough
    var isReadOnly: Bool { remote.annotations.readOnlyHint ?? false }
    func run(argumentsJSON: String) async -> ToolResult { ... tools/call ... }
}
```

`AgentLoop` already does everything else: schema injection
(`ToolCalling.injectToolSystem`), extraction, name repair, permission gating, arg
repair, observation feedback. An `MCPTool` in the registry is indistinguishable from
`GrepTool`. This is why the adapter — not a parallel "MCP loop" — is the design.

Two verified hazards and their mitigations:

- **Name charset**: `mcp__server__tool` must survive `ToolNameAutomaton` and every
  `ToolFormat`'s rendering. Server names are normalised to `[a-z0-9_-]` at config time;
  remote tool names pass through unless they contain characters the automaton can't
  spell, in which case the tool is skipped with a stderr warning (never silently
  renamed — a name the model calls must round-trip to `tools/call` exactly).
- **Schema size/$refs**: before handing `inputSchema` to `parametersJSON`, inline
  local `$ref`s (`#/definitions/...`, `#/$defs/...`) with a depth cap of 8 and a
  post-inline size cap of 16 KiB; over-cap schemas degrade to
  `{"type":"object"}` + the textual description (grammar constraint relaxes to "any",
  which `SchemaCompiler` already does for unsupported keywords — same failure mode,
  same visibility rule: one stderr note). The 256-token constrained-args cap in
  `EngineGenerator.completeConstrained` rises to a per-call
  `max(256, schemaFieldCount × 64)` bounded at 1024.

### 2.3 Registration: one toolset builder

New `Sources/KrillHarness/Toolset.swift`:

```swift
public func defaultToolset(bash: Bool, dispatch: Bool, mcp: [MCPTool]) -> ToolRegistry
```

`CodeCommand.swift:129`, `ChatTUI.agentTools()` (1218) and `VoicePanel.swift:552` all
call it. This refactor lands *first* (Phase 0) — it is pure consolidation, testable in
isolation, and without it every later phase has three drift-prone copies.

MCP connection happens once per process at toolset build: read `mcp.json`, connect
enabled servers concurrently with a per-server timeout (default 10 s), `tools/list`,
adapt. A server that fails to connect degrades to a one-line stderr note and an entry
in `krill mcp list` output — never a startup failure (`krill code` must work offline).

### 2.4 Permissions

No new types. `PermissionPolicy` deny/allow are name-based and wildcards are added to
matching (`mcp__indmoney__*`), which also benefits built-ins. Defaults:

- `readOnlyHint == true` ⇒ `isReadOnly` ⇒ auto-allowed in every mode including
  `.plan` — but the hint is *advisory* (a lying server could mutate), so a config
  per-server `"trust_read_only": false` forces `.ask` regardless of hint.
  Default is `true` for HTTPS servers, `false` for stdio (arbitrary local binaries).
- Non-read-only MCP tools follow the mode exactly like `BashTool` does today
  (`.ask` prompts, `.acceptAll` runs, `.plan` denies).
- Per-server `allow`/`deny` arrays in `mcp.json` merge into the policy ahead of CLI
  flags (CLI still wins on conflict via existing precedence: deny > allow).

### 2.5 Config + CLI

The flat TOML parser stays untouched. Sidecar `~/.krill/mcp.json` (same
`mergeJSON` habit as `AgentProfiles.swift`), project-level `.krill/mcp.json` merges
over it:

```json
{
  "servers": {
    "indmoney": { "url": "https://mcp.indmoney.com/mcp" },
    "github":   { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"],
                  "env": { "GITHUB_TOKEN": "..." }, "enabled": true },
    "local-db": { "command": "...", "trust_read_only": false,
                  "allow": ["query"], "deny": ["execute_*"] }
  }
}
```

New CLI (`krill mcp`): `add [--url | --command] <name>`, `list` (with live/failed/auth
status), `login <name>` (runs the OAuth flow in the browser, stores tokens in
Keychain), `logout <name>`, `remove <name>`, `tools <name>` (lists adapted tool names —
the exact strings for `--allow-tool`/`--deny-tool`). `krill code --mcp <name>` /
`--no-mcp` toggle per run.

`krill serve` gains nothing in v1: the server already accepts client-supplied `tools`
and emits `tool_calls` (`ServerParsing.swift:179`, `Server.swift:1303`), so any HTTP
client that wants MCP does its own hosting. A later phase may host MCP server-side so
plain chat clients benefit; that is a separate decision.

### 2.6 Skills

Claude-compatible `SKILL.md` packages, because the format is a de-facto standard and
Sourav already maintains skills in it:

```
~/.krill/skills/<name>/SKILL.md      # user scope
.krill/skills/<name>/SKILL.md        # project scope (wins on name collision)
```

Frontmatter `name:` + `description:` are required; everything else is body. A
`SkillsLoader` in KrillHarness:

1. **Discovery** at system-prompt assembly — both call sites:
   `CodeCommand.effectiveSystem` (CodeCommand.swift:140) and
   `ChatTUI.ensureAgentSeed()` (ChatTUI.swift:1226). Injected fragment is only
   `name — description` per skill (progressive disclosure; bodies never bloat the
   context of a small local model up front).
2. **Invocation** via a built-in `SkillTool` (`skill` — args `{"name": "..."}`), whose
   `ToolResult` is the skill body. It rides the existing `role:"user"` observation
   channel in `AgentLoop.record` — no loop changes. `isReadOnly = true`.
3. **TUI slash command**: `/skillname args` in ChatTUI expands to a user turn that
   instructs the model to run the named skill with those args (parity with how users
   already drive Claude Code skills).
4. Files referenced by a skill body are read by the model through the ordinary
   `ReadTool` under the ordinary permission policy — a skill grants *instructions*,
   never capability.

Not in v1: executable hooks, skill-bundled scripts auto-run, remote skill registries.

### 2.7 First consumer: artha / INDmoney

`krill mcp add --url https://mcp.indmoney.com/mcp indmoney && krill mcp login indmoney`
then `krill code --allow-tool 'mcp__indmoney__*' "what's my MF XIRR?"` works with any
local model. Small-model reliability guidance (gemma-4-e2b struggles with wide tool
menus): artha-style callers should pass `--mcp indmoney --no-bash` so the offered
toolset stays narrow — the fewer tools offered, the better `--constrain-args` repairs
land. artha's production integration continues to prefer its own snapshot-injection
design (holdings.json refreshed out-of-band); direct Krill→MCP is for interactive use.

## 3. Alternatives considered

- **Vendor an existing Swift MCP SDK** — the official swift-sdk is heavy (NIO
  dependency chain) and moves fast; our client surface is 5 methods. Hand-rolled
  Codable keeps the dependency graph clean (house rule: KrillHarness stays lean).
- **Upgrade the TOML parser for `[[mcp_servers]]`** — touches every config consumer to
  express one feature; JSON sidecar matches an existing habit and ships now.
- **A separate MCP agent loop** — rejected outright; `AgentLoop` is already
  schema-driven and the adapter proves it. Two loops would fork permission and
  constraint behaviour forever.
- **Tokens in `config.toml` like API keys** — rejected; OAuth refresh tokens to a
  brokerage are not equivalent to a search-engine key. Keychain now, and a follow-up
  ADR may migrate the existing keys too.
- **Skills as always-injected prompt text** — kills small-model context budgets;
  progressive disclosure is the entire point of the format.

## 4. Consequences

- Three new seams get tests for free by construction: `MCPTransport` (protocol ⇒ mock
  transport in tests, like `WebFetcher`), `TokenStore` (protocol ⇒ in-memory fake),
  `SkillsLoader` (pure function of a directory).
- First Keychain dependency; CI on Linux must stub `TokenStore` (Keychain is
  Darwin-only — acceptable, Krill is Mac-native by charter).
- `mcp__server__tool` names are long; `ToolFormat` renderers and the TUI tool-event
  display should be spot-checked for truncation.
- A hostile MCP server is a live prompt-injection channel (tool descriptions and
  results are model-visible text). Mitigations: descriptions pass through
  `injectToolSystem` unmodified but the permission gate covers every call; stdio
  servers default to `trust_read_only: false`; and the existing runaway-guard dedupe
  in `AgentLoop` bounds repeated-call loops. A fuller taint model is future work.

## 5. Phases + verification

| Phase | Lands | Verified by |
|---|---|---|
| 0 | `defaultToolset()` consolidation (3 call sites → 1) | existing harness tests still green; new test that CLI/TUI/voice registries are identical |
| 1 | `KrillMCP` target: transports, client, adapter; stdio + non-auth HTTP servers; `krill mcp add/list/tools/remove`; permission wildcards | unit tests against a mock transport; integration test against `@modelcontextprotocol/server-everything` (stdio); `krill code` round-trip with a local echo server |
| 2 | OAuth 2.1 + Keychain + `krill mcp login`; INDmoney end-to-end | manual OAuth against mcp.indmoney.com; token-refresh unit tests with a fake clock; revocation → visible re-auth error, never a silent empty toolset |
| 3 | Skills: loader, `SkillTool`, TUI slash commands | fixture skill dirs in tests; small-model dogfood (gemma-4-e2b invokes a skill by name); docs/ARCHITECTURE.md + README updated |

Each phase gets its own ADR in `docs/decisions/` if it deviates from this plan; this
document is the plan of record until then.

---

*Survey references (2026-08-02): Tool protocol `Sources/KrillHarness/Tool.swift`; loop
`Sources/KrillHarness/AgentLoop.swift`; wire formats `Sources/KrillTooling/ToolCalling.swift`;
permissions `Sources/KrillHarness/Permission.swift`; grammar `Sources/KrillGrammar/SchemaCompiler.swift`,
`ToolNameGrammar.swift`; server tools `Sources/KrillServer/Server.swift:1186`; config
`Sources/KrillRegistry/Config.swift:221`; toolset call sites `Sources/KrillCLI/Code/CodeCommand.swift:129`,
`Sources/KrillCLI/TUI/ChatTUI.swift:1218`, `Sources/KrillCLI/Voice/VoicePanel.swift:552`.*
