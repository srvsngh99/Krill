# Krill Agent UI — remote `krill code` from any device

`krill serve` hosts agent sessions (the same loop as `krill code`) and serves a
phone-quality web app at **`/ui`**, embedded in the binary. Point any browser at
it — your Mac, a laptop on the LAN, or your phone from anywhere via a VPN — and
you get: sessions rooted in a chosen repo, a live transcript of assistant turns
and tool calls, and approval prompts for mutating tools you answer with a tap.

## Quick start (phone)

```sh
# On the Mac: bind beyond loopback and set a key (required for non-loopback).
KRILL_API_KEY=your-secret krill serve --host 0.0.0.0
```

1. Reach the Mac from the phone:
   - **Same Wi-Fi**: open `http://<mac-ip>:57455/ui`.
   - **From anywhere (recommended): [Tailscale](https://tailscale.com)** on the
     Mac and phone (free, ~10 min). The startup banner prints the ready-to-open
     tailnet URL. Zero ports exposed to the internet; no TLS needed.
2. Enter the API key on the connect screen (stored in the browser).
3. iOS: Share → **Add to Home Screen** for a full-screen standalone app.

Krill never bundles or requires a specific network layer — any path that routes
HTTP to the Mac works (LAN, WireGuard, Cloudflare Tunnel + a real cert, ...).
Do NOT port-forward `krill serve` straight to the public internet: the agent
runs tools on your machine; keep it behind a VPN or an authenticated tunnel.

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
