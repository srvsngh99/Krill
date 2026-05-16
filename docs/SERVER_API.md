# Server API Reference

## Status and Scope

This build is a release-readiness baseline, not a production release. The HTTP server supports **text generation on every model family** plus **Gemma 4 image and audio input**. Image input runs through the native Swift SigLIP2 vision encoder; audio input is routed to the `mlx-vlm` Python bridge (the same path the CLI uses for `--audio`). When a request supplies both image and audio, the entire request goes through the bridge to match CLI behavior. Non-Gemma 4 models reject image/audio payloads with HTTP 400. See [`RELEASE_READINESS_REMEDIATION.md`](RELEASE_READINESS_REMEDIATION.md) for full status.

**Limits:**
- 1 image per request maximum (Gemma 4 supports a single image per turn).
- 1 audio clip per request.
- 25 MB per decoded media item.
- 10 MB total HTTP body (`ServerLimits.maxBodySize`).

## Starting the Server

```bash
krillm serve --model llama-3.2-1b --port 11435
```

Default: `127.0.0.1:11435`

### Compat mode and the Ollama port

`krillm serve` accepts `--compat ollama|openai|both` (default `both`):

- `both` — `/api/*` and `/v1/*` are both served (default).
- `ollama` — only `/api/*` + `/healthz` + `/metrics`.
- `openai` — only `/v1/*` + `/healthz` + `/metrics`.

Endpoints disabled by the compat mode return `404` with an explanatory
`error` body, so client protocol-probing behaves as if the server simply
does not speak that protocol.

**Port deferral (T0-1):** the default port intentionally remains `11435`.
For an Ollama drop-in run `krillm serve --port 11434 --compat both`. The
default flip to `11434` is deliberately deferred until the `mac_parity`
gate is green — see [`OLLAMA_MAC_PARITY_PLAN.md`](OLLAMA_MAC_PARITY_PLAN.md)
§4 WS-A1. Verify parity progress with `make parity-gate`.

## OpenAI-Compatible Endpoints

### POST /v1/chat/completions

```bash
curl http://127.0.0.1:11435/v1/chat/completions -d '{
  "model": "llama-3.2-1b",
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": true,
  "max_tokens": 64,
  "temperature": 0.7
}'
```

Streaming: SSE format with `data: {...}\n\n` and `data: [DONE]\n\n` terminator.

### POST /v1/completions

```bash
curl http://127.0.0.1:11435/v1/completions -d '{
  "prompt": "The meaning of life is",
  "max_tokens": 32
}'
```

### GET /v1/models

Returns list of installed models.

### POST /v1/models/load

```bash
curl -X POST http://127.0.0.1:11435/v1/models/load -d '{"model": "llama-3.2-1b"}'
```

### POST /v1/models/unload

Unloads the current model from memory.

### GET /v1/status

Returns server status, memory, uptime, loaded model info.

## Ollama-Compatible Endpoints

### POST /api/generate

Text generation on any model. With Gemma 4 loaded, you may also include `images` (base64-encoded array, max 1 element) and/or a single `audio` field (base64 string; default format `wav`, override with `audio_format`).

```bash
# Text
curl http://127.0.0.1:11435/api/generate -d '{
  "model": "llama-3.2-1b",
  "prompt": "Hello",
  "stream": true,
  "options": {"temperature": 0, "num_predict": 32}
}'

# Image (Gemma 4 only — native Swift vision path)
curl http://127.0.0.1:11435/api/generate -d '{
  "model": "gemma-4-e2b",
  "prompt": "What is in this image?",
  "images": ["'"$(base64 -i photo.png)"'"]
}'

# Audio (Gemma 4 only — routed through mlx-vlm bridge)
curl http://127.0.0.1:11435/api/generate -d '{
  "model": "gemma-4-e2b",
  "prompt": "Transcribe this clip.",
  "audio": "'"$(base64 -i clip.wav)"'",
  "audio_format": "wav"
}'
```

**Timing fields in final `done` chunk:**
- `total_duration` — total request time (nanoseconds)
- `prompt_eval_count` — number of prompt tokens
- `prompt_eval_duration` — prefill time (nanoseconds)
- `eval_count` — number of generated tokens
- `eval_duration` — decode time (nanoseconds)
- `ttft_ns` — time to first token (nanoseconds, server-side)

### POST /api/chat

Chat-style endpoint. Each message accepts an optional `images` array (Gemma 4 only) and/or `audio` field. The server collects per-message media into a request-level payload and applies the same per-request limit (1 image, 1 audio).

```json
{
  "model": "gemma-4-e2b",
  "messages": [
    {
      "role": "user",
      "content": "What's in this picture?",
      "images": ["<base64-png>"]
    }
  ]
}
```

### GET /api/tags

Returns installed models in Ollama format.

### GET /api/version

Returns `{"version": "<ollama-compat>", "krillm_version": "<krillm>"}`. The
advertised Ollama-compat version is spoofable via the
`KRILL_OLLAMA_COMPAT_VERSION` env var so version-gated clients proceed.

### GET /api/ps

Lists the currently-loaded model with `size`, `details`, and a best-effort
`expires_at` (derived from `idle_timeout` until per-request `keep_alive`
lands in a later phase). Empty `models` list when nothing is loaded.

### POST /api/show

Body: `{"model": "<name>", "verbose": false}`. Returns `modelfile`,
`parameters`, `template`, `system`, `details`, `model_info`,
`capabilities`, `modified_at`. `404` if the model is not installed.

### POST /api/pull

Body: `{"model": "<alias-or-hf-repo>", "stream": true}`. Streams NDJSON
progress (`pulling manifest` → `downloading …` → `success`); set
`stream: false` for a single terminal JSON. Names resolve through the same
alias map as `krillm pull`.

### DELETE /api/delete

Body: `{"model": "<name>"}`. Removes the manifest + blobs. `404` if absent.

### POST /api/copy

Body: `{"source": "<name>", "destination": "<name>"}`. Duplicates the
model + manifest under a new name.

### HEAD|POST /api/blobs/:digest

Digest blob store backing `ollama create` uploads. `HEAD` → `200` if the
blob exists else `404`; `POST` stores the body. (`/api/create` itself is a
later phase; see the parity plan.)

### GET /v1/models/{id}

OpenAI single-model lookup. Returns the model object or `404`.

## Embeddings

KrillLM serves embeddings from a **dedicated sentence-embedding model**
(BERT/RoBERTa/MiniLM/BGE/E5 — `bert` family), independent of any loaded
chat model. Pull one first, e.g. `krillm pull all-minilm` (also
`bge-small-en`, `bge-base-en`). Vectors are mean-pooled (override with
`KRILL_EMBED_POOLING=cls`) and L2-normalized.

### POST /api/embed

Body: `{"model": "all-minilm", "input": "text" | ["t1","t2"]}`. Returns
`{"model", "embeddings": [[...]], "prompt_eval_count", "total_duration"}`.

### POST /api/embeddings (legacy)

Body: `{"model": "all-minilm", "prompt": "text"}`. Returns
`{"embedding": [...]}` (single vector).

### POST /v1/embeddings

OpenAI shape. Body: `{"model": "all-minilm", "input": "text" | [...]}`.
Returns `{"object":"list","data":[{"object":"embedding","index":0,
"embedding":[...]}],"model","usage"}`.

Requesting embeddings against a non-embedding (chat) model returns `400`;
an uninstalled model returns `404` with a `krillm pull` hint.

## Tool / Function Calling

`tools: [{type:"function", function:{name, description, parameters}}]` is
accepted on `POST /v1/chat/completions` and `POST /api/chat`. KrillLM
injects the tool schemas as a system turn and instructs the model to emit
`<tool_call>{"name":...,"arguments":{...}}</tool_call>`; extraction is
tolerant of a missing close tag, backticks, fenced blocks, and bare JSON.

- OpenAI: a tool call yields `choices[0].message.tool_calls` (arguments as
  a JSON **string**), `content:null`, `finish_reason:"tool_calls"`.
- Ollama: `message.tool_calls` (arguments as a decoded **object**),
  `done_reason:"tool_calls"`.
- Multi-turn: send the assistant `tool_calls` message back followed by a
  `{"role":"tool","name":...,"content":...}` result; both round-trip.

Tool-call *quality* depends on the model (small local models are weaker —
the same as Ollama). Token-level streaming tool deltas are Phase 4; with
`stream:true` the assembled call is emitted as one SSE/NDJSON chunk.

## Multimodal Notes

- Image input is supported on all four chat/generate endpoints when a Gemma 4 model is loaded; it is rejected with HTTP 400 for any other model family.
- Audio input is also Gemma 4 only and is routed through the `mlx-vlm` Python bridge; if `mlx-vlm` is not installed the server returns HTTP 503 with an installation hint (`make setup-mlx-vlm`).
- OpenAI `/v1/chat/completions` accepts both string content and the standard content-block array form: `{"type": "text"}`, `{"type": "image_url", "image_url": {"url": "data:..."}}`, and `{"type": "input_audio", "input_audio": {"data": "...", "format": "wav"}}`. Only `data:` URLs are accepted for images (no remote fetching).
- OpenAI `/v1/completions` remains text-only (parity with the upstream API).
- Decoded media is written to `FileManager.default.temporaryDirectory` and removed when the request completes.

### OpenAI image example

```bash
curl http://127.0.0.1:11435/v1/chat/completions -d '{
  "model": "gemma-4-e2b",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "What is in this image?"},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,'"$(base64 -i photo.png)"'"}}
    ]
  }]
}'
```

## Health and Monitoring

### GET /healthz

```json
{"status": "ok", "model_loaded": true, "model": "llama-3.2-1b", "family": "llama"}
```

### GET /metrics

Prometheus text format:
```
krillm_up 1
krillm_model_loaded 1
krillm_resident_memory_mb 1234.5
krillm_uptime_seconds 3600
```

## Request Validation

- Model name validation: rejects requests for mismatched models
- Unsupported fields rejected: `tools`, `function_call`, `format`, `context`
- Max body size: 10 MB
- Returns 503 during model swap

## Streaming Performance

Per-token streaming uses direct JSON string formatting (not JSONSerialization) for the hot path to minimize overhead. Complex responses (final stats, non-streaming) still use JSONSerialization.
