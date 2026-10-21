# Observability

Krill keeps a **local** record of what it did so that "what went wrong last Tuesday?" has an
answer. Nothing here is telemetry: no network call, no analytics, no remote reporting. Everything
lives in `~/.krill` and leaves the machine only if you attach a bug-report bundle yourself.

## What is written, and where

| What | Where | Content |
|---|---|---|
| Server event log | `~/.krill/logs/serve-YYYY-MM-DD.jsonl` (CLI surfaces: `cli-YYYY-MM-DD.jsonl`) | One JSON object per line: requests and lifecycle events. **No prompt or completion text.** |
| Agent session transcripts | `~/.krill/sessions/YYYY/MM/DD/<HHMMSS>-<id>.jsonl` | Per turn: your text, the model's text, tool calls with arguments and results, permission decisions, retries, errors, stop reason. |
| Session index | `~/.krill/sessions/index.jsonl` | One row per session: id, start, end, project dir, surface, model, provider, outcome. |
| Run markers | `~/.krill/run/<kind>-<pid>.json` | Written at start, removed on a clean exit. A marker whose process is gone means a crash or kill. |

Files are created `0600` in `0700` directories.

### Server event log

Written by `krill serve` (default on), from an async queue: logging never blocks a request, the
queue is bounded (a `log_dropped` line records any loss), and it is flushed on shutdown.
A day file is also rotated (`serve-DATE.1.jsonl`, ...) once it passes 16 MB.

`request` events (health probes, `/metrics` and `/ui` assets are only logged when they fail):

```json
{"ts":"2026-10-21T20:49:45.867Z","event":"request","pid":94159,"port":57490,"request_id":"94211fa6017c",
 "method":"POST","endpoint":"/v1/chat/completions","model":"llama-3.2-1b","status":200,"error":null,
 "ttft_ms":151.8,"total_ms":332.7,"prompt_tokens":40,"completion_tokens":2,"tok_s":82,"queue_wait_ms":0,
 "session_id":"demo-session"}
```

Embedding requests add `embed_device` (`gpu`/`ane`) and `inputs`. `session_id` comes from the
`X-Krill-Session` request header (set by agents so `krill logs --session <id>` can join a
transcript with the server requests it made). Every response carries `X-Krill-Request-Id`.
`ttft_ms` is time to the first streamed byte for streaming requests, queue wait plus prefill time
otherwise.

Lifecycle events: `server_start`, `server_stop`, `shutdown_signal`, `model_load` (with `ms`, MLX
and footprint MB), `model_load_failed`, `model_unload`, `model_evict`, `keepalive_unload`,
`server_error`, `unclean_exit`, and `log` (every warning or error from any Krill logger, whatever
the stderr level). They are written at full detail regardless of `KRILL_LOG_LEVEL`.

### Agent session transcripts

Written for `krill run` (chat and agent), `krill code`, hosted `/ui` sessions and `krill launch`
(a start record only: `launch` replaces itself with the agent, which owns its own transcript).
Each field is truncated at 16 KB. `krill logs --project .` lists the sessions started in a
directory; `krill logs --session <id>` shows one.

### Crash breadcrumbs

On start, every stale run marker (process gone) becomes an `unclean_exit` event carrying the dead
process's last events and any `~/Library/Logs/DiagnosticReports/krill*.ips` written since it started.

## Knobs

| Config key | Env | Default | Meaning |
|---|---|---|---|
| `event_log` | `KRILL_EVENT_LOG` | `on` | `off` writes no event log and no run markers |
| `session_log` | `KRILL_SESSION_LOG` | `full` | `full`, `meta` (structure only: no text, no tool arguments or results) or `off` |
| `log_retention_days` | `KRILL_LOG_RETENTION_DAYS` | `30` | Older log and session files are deleted |
| `log_max_mb` | `KRILL_LOG_MAX_MB` | `1024` | Logs plus sessions stay under this size, oldest deleted first |
| | `KRILL_OBS_HOME` | `~/.krill` | Relocates logs, sessions and run markers only (tests, containers) |

Retention runs at startup (skipped if it ran within the hour) and once a day in `krill serve`.
It is one directory walk. Today's active files are never deleted by the size cap.

The four config keys can be set with `/config key=value` in the TUI or `krill --config key=value`.

## Stderr logging

| Env | Effect |
|---|---|
| `KRILL_LOG_LEVEL` | `trace`, `debug`, `info`, `notice` (default), `warning`, `error`, `critical`. Controls the terminal/launchd stderr only. |
| `KRILL_DEBUG` | Implies `debug` when `KRILL_LOG_LEVEL` is unset. |
| `KRILL_TOOL_DEBUG` | Server: prints tool-argument regeneration decisions to stderr. |
| `KRILL_DEBUG_PROMPT` | Engine: dumps the token ids a request really feeds the model. Contains your prompt. |
| `KRILL_DEBUG_FIRST_TOKEN` | Engine: dumps first-token logits. |
| `KRILL_EMBED_LOG_MEM` | Embedding engine: prints MLX peak/active/cache memory after loads. |

The event log and `KRILL_LOG_LEVEL` are independent: lifecycle events are always in the file.

## Commands

```
krill logs [--today] [--since 2h|7d|2026-10-20] [--errors] [--session <id>] [--project <dir>] [--json] [-f] [-n 50]
krill doctor [--json] [--strict]
krill bug-report [--days 3] [--session <id> [--include-content]] [--output <dir>]
```

* `logs` is read-only. Default: the last 50 events.
* `doctor` is read-only and never starts, stops, loads or unloads anything. One screen: version and
  binary paths, whether several installs exist and which one each launchd job runs, codesign state,
  launchd jobs that run krill (loaded or not, last exit reason), ports and whether `/healthz`
  answers, disk free, the last crash and recent errors. `--strict` exits 1 on any failure.
* `bug-report` writes ONE zip to the current directory and prints its manifest. It sends nothing.
  It includes `doctor.txt`, the config with keys and tokens redacted, `KRILL_*` / `OLLAMA_*` / `HF_*`
  environment (secret-looking values hidden), the last N days of event logs, crash markers and
  macOS crash reports. A session is included only with `--session <id>`, with its text and tool
  payloads removed unless `--include-content`. Redaction covers API keys, bearer tokens,
  `server_api_key`, secret-named config/env values, e-mail addresses and the home path (to `~`).
  It then prints a prefilled GitHub issue URL (title and body only; GitHub cannot take a file
  through a URL) and offers to open it only when run in a terminal and you answer `y`.

## Metrics

`GET /metrics` (Prometheus text, same auth as the rest of the API) reports:

* `krill_uptime_seconds`: age of the server **process** (kernel start time).
* `krill_memory_footprint_mb`: physical footprint, which includes Metal/MLX buffers;
  `krill_resident_memory_mb` is the resident set and does **not** include model weights.
* `krill_mlx_active_memory_mb`, `krill_mlx_peak_memory_mb`.
* `krill_http_requests_total{endpoint,code}`, `krill_http_errors_total{endpoint}`,
  `krill_http_request_duration_seconds` (histogram), `krill_prompt_tokens_total`,
  `krill_completion_tokens_total`, `krill_http_in_flight`. `endpoint` is a normalised route, never a raw path.

`/v1/status` carries the same `uptime_seconds`, `memory_mb` (footprint), `mlx_active_mb`, `mlx_peak_mb`.
