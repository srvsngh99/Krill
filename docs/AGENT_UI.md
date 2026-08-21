# Krill Agent UI — remote `krill code` from any device

`krill serve` hosts agent sessions (the same loop as `krill code`) and serves a
phone-quality web app at **`/ui`**, embedded in the binary. Point any browser at
it — your Mac, a laptop on the LAN, or your phone from anywhere via a VPN — and
you get: sessions rooted in a chosen repo, a live transcript of assistant turns
and tool calls, and approval prompts for mutating tools you answer with a tap.

## Quick start (phone)

```sh
# On the Mac — one command:
krill ui
```

`krill ui` starts `krill serve` in the background bound on `0.0.0.0` (it
survives closing the terminal), generates an API key on first run and saves it
to `~/.krill/config.toml` (`server_api_key`), pre-loads your default model,
prints the links, and opens the UI in the Mac's browser. It reuses a server
that is already running on the port.

1. Reach the Mac from the phone — open the printed **Phone link**:
   - **Same Wi-Fi**: the `Same Wi-Fi` address.
   - **From anywhere (recommended): [Tailscale](https://tailscale.com)** on the
     Mac and phone (free, ~10 min). `krill ui` prints the `Tailscale` address
     when it is up. Zero ports exposed to the internet; no TLS needed.
   The phone link ends in `#k=<key>`: the app stores the key and strips it from
   the address bar (URL fragments are never sent over the network). Without the
   fragment, enter the key on the connect screen.
2. iOS: Share → **Add to Home Screen** for a full-screen standalone app.
3. Leave it on: `krill ui --install` registers the server as a **login item**
   (launchd): it starts when you log in and restarts if it exits, with logs in
   `~/.krill/ui/serve.log`. `krill ui --uninstall` removes it.

| `krill ui` flag | What |
| --- | --- |
| `--model NAME` | Model to pre-load (default: `default_model` from config, else the first installed). |
| `--port N` / `--host H` | Port (default 57455) and bind address (default `0.0.0.0`). |
| `--no-open` | Don't open the Mac's browser. |
| `--foreground` | Run in this terminal instead of detaching (Ctrl+C stops). |
| `--status` | Reprint the links and key for the running server. |
| `--stop` | Stop the detached server (`~/.krill/ui/serve.pid`). |
| `--install` / `--uninstall` | Add / remove the launchd login item (`~/Library/LaunchAgents/ai.souravailabs.krill.serve.plist`). |

Manual equivalent, if you'd rather own the process:
`KRILL_API_KEY=your-secret krill serve --host 0.0.0.0` (a non-loopback bind
requires a key).

Krill never bundles or requires a specific network layer — any path that routes
HTTP to the Mac works (LAN, WireGuard, Cloudflare Tunnel + a real cert, ...).
Do NOT port-forward `krill serve` straight to the public internet: the agent
runs tools on your machine; keep it behind a VPN or an authenticated tunnel.

## Using the app, step by step

Shipped in Krill **v0.21.0** (`brew upgrade krill`). Everything below is the
embedded app at `/ui`; nothing to install on the phone.

1. **Connect.** Opened from the `krill ui` phone link, this step is skipped —
   the key is adopted from the link. Otherwise the first screen asks for the
   **server** (pre-filled with the address you opened the page from; otherwise
   `http://<mac-ip>:57455`) and the **API key** (`server_api_key` /
   `KRILL_API_KEY`; leave blank only for a loopback server). Tap **Connect**.
   Both are remembered in the browser. The **⋮** button on the
   sessions screen brings you back here to change them.
2. **Sessions list.** The header shows a status dot and the model the server
   has loaded (`no model loaded` / `unreachable` are diagnostics, see below).
   Tap **+ New session**.
3. **New session sheet.** Three choices:
   - **Workspace** — a folder browser rooted at your home directory. Tap a
     folder to enter it, **‹** to go up; git repos are flagged. The session's
     file tools, `bash`, and `Krill.md` brief all resolve inside this folder.
   - **Model** — any installed model (`krill list`); the server loads it on
     demand, so the first turn on a new model takes a while.
   - **Permissions** — **Plan** (read-only), **Ask** (every mutating tool
     prompts; the default), **Edits** (file edits auto-apply, commands prompt),
     **Auto** (nothing prompts).
   Tap **Start session**.
4. **Talk to the agent.** Type in *Message the agent…* and send with **↑**
   (on a desktop browser, Enter sends and Shift+Enter inserts a newline). The
   pill in the header shows `idle` / `running` / `waiting`, and a *working…*
   indicator sits under the transcript while a turn is in flight. The
   transcript shows your messages, the assistant's replies (Markdown), and a
   **tool card** per call — the tool name plus a one-line summary. Tap a card
   to expand its arguments and result; a red dot marks a failed call, and
   failures open automatically.
5. **Approve or deny.** When a tool needs permission the pill flips to
   `waiting for approval…`, the agent pauses, and a card shows the call with
   **Deny / Allow / Always**. *Always* whitelists that tool name for the rest
   of the session (typical: allow `bash` once for `swift build`, then Always).
   The loop resumes as soon as you answer.
6. **Stop or delete.** While a turn runs, the send button becomes **■** — tap
   it to stop. The **⋯** menu has **Stop current run** and **Delete session**.
7. **Leave and come back.** Sessions live in the server's memory: lock the
   phone, switch networks, close the app — reopening replays the transcript and
   resumes the live stream. They last until `krill serve` restarts.
8. **On a laptop** the same page opens as a two-pane layout (sessions on the
   left); it is the same app, not a separate desktop build.

### Troubleshooting

| You see | Cause / fix |
| --- | --- |
| Model chip says **unreachable** | The phone can't reach the server: wrong address, `krill serve` not running, or the phone isn't on the same Wi-Fi / tailnet. Test the URL in a plain browser tab first. |
| **401** on connect | Wrong key. It must match `KRILL_API_KEY` exactly. |
| Server exits on start with `--host 0.0.0.0` | A non-loopback bind requires an API key — set `KRILL_API_KEY` (or pass `--allow-remote-unauthenticated` on a trusted private network only). |
| *No model available for this session* | Start the server with a model (`krill serve --model gemma-4-e2b`) or pick an installed model in the new-session sheet. |
| A turn ends with *Stopped at the iteration limit* | The run hit `max_iterations` (24 by default). Send the next step as a new message — context carries over — or create the session over HTTP with a higher limit (table below). |
| `krill ui --install` reports the server never answered | launchd cannot execute a binary inside Desktop, Documents, Downloads or iCloud Drive (macOS privacy protection). A dev build under `~/Desktop` hits this; `brew install krill` or `make install` (→ `/usr/local/bin`) do not. Check `~/.krill/ui/serve.log` and `launchctl print gui/$(id -u)/ai.souravailabs.krill.serve`. |
| The stream goes quiet on iOS | Safari suspends background tabs; the app reconnects and replays when you return. Add to Home Screen for the most reliable behaviour. |

## Sessions

A session = a workspace directory + a model + a permission posture + its
conversation. File tools and `bash` resolve inside the session's workspace (not
the server's cwd), `Krill.md` at the workspace root is loaded as the project
brief, and multi-turn context carries across messages. Sessions live in server
memory: they survive phone disconnects/reconnects, but not a server restart.

Permission postures (same semantics as `krill code`): `plan` (read-only),
`ask` (default here: every mutating tool prompts), `accept-edits` (file edits
auto-apply, commands prompt), `accept-all`. "Always allow" from a prompt
whitelists that tool for the rest of the session.

## HTTP API

All routes require `Authorization: Bearer <key>` when a key is set. The `/ui*`
shell (page, manifest, icon — no data) is exempt, as are `OPTIONS`/`/healthz`.

| Route | What |
| --- | --- |
| `POST /v1/agent/sessions` | Create. Body: `workspace?` (default: server cwd), `model?` (installed name; default: active engine), `permission_mode?` (default `ask`), `title?`, `max_tokens?` (default 1024), `max_iterations?` (default 24). |
| `GET /v1/agent/sessions` | List (newest first, with status + pending approval). |
| `GET /v1/agent/sessions/{id}` | Meta + full event replay (`events[]`, each with `seq`). |
| `DELETE /v1/agent/sessions/{id}` | Cancel + remove. |
| `POST /v1/agent/sessions/{id}/messages` | `{"text": …}` → `202`; starts a turn (409 if one is running). |
| `GET /v1/agent/sessions/{id}/events` | SSE. `?since=N` (or `Last-Event-ID`) replays `seq > N`, then tails live. `id:` line = seq; `: ping` heartbeat every 25s. |
| `POST /v1/agent/sessions/{id}/approvals` | `{"id"?: …, "allow": bool, "always"?: bool}` → 409 if nothing (or a different request) is pending. |
| `POST /v1/agent/sessions/{id}/cancel` | Stop the active turn (also resolves a parked approval as deny). |
| `GET /v1/agent/workspaces?path=~` | Shallow directory browser for the workspace picker (marks git repos). |

Event types on the stream: `user`, `assistant`, `assistant_final`, `tool_call`
(`name`, `args` JSON string), `tool_result` (`content`, `is_error`), `note`,
`status` (`idle|running|waiting|cancelled`), `approval_request`
(`id`, `tool`, `args`), `approval_resolved` (`id`, `allow`).

## Notes

- The model loads on demand per session (same engine pool as every generate
  route); agent turns share the generation queue with chat requests, so a phone
  session and an API client interleave rather than colliding on the GPU.
- Events carry cleaned whole turns, not raw token deltas (raw streaming would
  leak tool-call markers): the UI shows a working indicator during a turn.
- The toolset is `krill code`'s minus `dispatch_agent` (background fan-out
  stays a TUI feature for now).

## Development notes (for contributors and agents)

**Code map** — the feature is three server files plus one harness seam:

| File | Owns |
| --- | --- |
| `Sources/KrillHarness/AgentWorkspace.swift` | Task-local workspace root. `FileToolSupport`, `BashTool` (`currentDirectoryURL`), and `AgentEnvironment` resolve against it; unbound = process cwd, so CLI surfaces are untouched. Bind with `AgentWorkspace.$root.withValue(url) { await loop.run(…) }`. |
| `Sources/KrillServer/AgentSessions.swift` | `RemoteAgentSession` (lock-guarded seq-numbered event log + subscriber fan-out + the `priorMessages` seam), `RemoteApprover` (a `PermissionGate` that parks the loop on a continuation until an HTTP answer; sticky always-allow), `ServerHarnessGenerator` (engine + `GenerationQueue` binding; one queue slot **per completion**, not per run), `AgentSessionStore` (one per `KrillServer`, shared across connections). |
| `Sources/KrillServer/AgentAPI.swift` | All `/v1/agent/*` routes as an `HTTPHandler` extension, the run-task orchestration (engine activate → retain → loop → release), and the SSE tail (head written synchronously in the channelRead chain — NIO requires it; heartbeat task + `channelInactive` teardown live on the handler). |
| `Sources/KrillServer/WebUI.swift` | The `/ui` shell: page HTML as a Swift raw string, PWA manifest, and the brand icon PNGs embedded base64. |

**Event contract**: `RemoteAgentSession.push(_:)` is the wire mirror of
`foldAgentEvent` — if you add an `AgentEvent` case, extend both, and add the
type to the table above so clients know it. Never renumber or reuse seqs; SSE
replay correctness depends on them being append-only.

**Editing the web app**: the page is a single self-contained HTML file inlined
in `WebUI.swift` (deliberate — no resource bundle to lose in `make dist`). Edit
the string directly, or extract to a scratch file and re-embed between the
`indexHTML = #"""` / `"""#` markers. No external assets beyond Google Fonts
(degrades to system fonts offline); everything else must stay same-origin.

**Brand rules are law here**: the page follows the SAI suite spine — ink
`#1a1a1a` on paper `#efece4`, dark `#161310` terminal panels, hairline 1.5px
borders, corners ≤ 6px, zero elevation (no shadows/glow/glass/gradients),
Space Grotesk + JetBrains Mono, whisper-dot texture. Colour appears **only** in
the product symbol (the `</>` tile and the `›Krill_` mark's Ember cursor);
status colours (`ok/warn/err/idle`) are functional state only, never chrome.
The icon PNGs are the brand kit's own files — don't regenerate or restyle them.

**Tests**: `Tests/KrillHarnessTests/AgentWorkspaceTests.swift` (task-local
threading incl. bash cwd), `Tests/KrillServerTests/AgentSessionTests.swift`
(approver, event folding, replay, store), `Tests/KrillServerTests/AgentAPITests.swift`
(routes via `EmbeddedChannel`, auth exemption). Run:
`swift test --filter "AgentWorkspaceTests|AgentSessionTests|AgentAPITests"`.

**Known gaps / natural next steps**: sessions are in-memory only (a restart
drops them — a small JSONL store under `~/.krill/sessions` would fix resume);
`dispatch_agent` is excluded from the hosted toolset; plain-chat turns don't
stream token-by-token to the UI; one pending approval at a time per session
(matches the loop's sequential tool execution); a QR code in the serve banner
would remove the type-the-URL step on phones.
