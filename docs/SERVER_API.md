# Server API Reference

## Starting the Server

```bash
krillm serve --model llama-3.2-1b --port 11435
```

Default: `127.0.0.1:11435`

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

Text-only generation. **Does not accept `images` or media payloads** — the `images` field from Ollama's API is rejected. Image/audio generation is only available via the CLI (`krillm run --image`), not the server API.

```bash
curl http://127.0.0.1:11435/api/generate -d '{
  "model": "llama-3.2-1b",
  "prompt": "Hello",
  "stream": true,
  "options": {"temperature": 0, "num_predict": 32}
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

Text-only chat. Same media limitation as `/api/generate` — no image/audio payloads.

```json
{"model": "...", "messages": [{"role": "user", "content": "..."}]}
```

### GET /api/tags

Returns installed models in Ollama format.

## Multimodal Limitations

The server currently supports **text-only** generation. Image and audio are not supported through any server endpoint:

- `/api/generate` rejects the `images` field
- `/api/chat` does not pass image/audio data to the engine
- Server-mode benchmarks skip media tasks for this reason

For multimodal inference, use the CLI directly:
```bash
krillm run gemma-4-e2b "Describe this image." --image photo.png
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
