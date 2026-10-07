# 0006. Multimodal embeddings request format (`dimensions`, `task`, content parts)

Status: adopted. Date: 2026-10-08. Owner: unassigned. Scope: the request fields
and limits `/v1/embeddings` and `/api/embed` gained with EmbeddingGemma 2 (PRs
#329, #330, #331). Model behaviour, parity numbers and measurements live in
`docs/EMBEDDINGGEMMA2.md`; this record is the request contract and why it is
shaped this way.

---

## 0. TL;DR

- **`dimensions`** follows the OpenAI name. A model with Matryoshka (MRL) sizes
  truncates and re-normalises; a model without them ignores a valid value, as
  every embedding model did before the field existed.
- **`task`** names a prompt from the model's sentence-transformers prompt table.
  A model with no table answers `400`. It is mutually exclusive with the older
  `instruction` field (`400`).
- **`input` items may be content parts**: `text`, `image_url` / `input_image`,
  `input_audio` / `audio_url`, `video_url` / `input_video`. One item is one
  vector; part order is token order.
- **Data URLs or base64 only.** The server never fetches a URL.
- **The existing 10 MB body cap** applies; no separate media limit.
- **Over-limit input is refused, not truncated**: an item over the 8,192-token
  context is `400`, and audio over 30 s is `400`.
- **Towers load lazily**, vision and audio separately, so text-only users keep
  their memory profile.
- **float16 is not accepted** for EmbeddingGemma 2.

---

## Context

Before EmbeddingGemma 2 the three embeddings endpoints took strings, an optional
literal `instruction` prefix, and (on `/v1/embeddings`) an OpenAI-shaped body.
EmbeddingGemma 2 needed more:

- Its checkpoint ships a `config_sentence_transformers.json` `prompts` table
  (`SearchQuery`, `Document`, ...), and the model card says quality is best with
  those prefixes.
- It is a Matryoshka model (768 / 512 / 256 / 128).
- It embeds images, audio and video into the same space as text. The reference
  (sentence-transformers) takes a message with ordered content parts, and an
  image with its caption is one joint vector, not two.
- Media is large and arrives in a request body, and the model has hard limits
  (an 8,192-token context; the audio feature extractor truncates at 30 s).

Plain-string requests had to keep working unchanged for every existing model.

## Decision

**1. `dimensions` is OpenAI-style and permissive for non-MRL models.** For
EmbeddingGemma 2 only 768, 512, 256 and 128 are valid; anything else is `400`
and lists the valid sizes. The vector is the first N components, re-normalised to
unit length. For every other model a valid positive integer is ignored, exactly
as before (`EmbeddingEngine.swift`, "Non-MRL models: `dimensions` is ignored").
A non-integer, boolean or non-positive value is `400` for all models
(`EmbeddingOptions.parse`).

**2. `task` names the prompt; `instruction` stays the literal prefix.** `task` is
looked up exactly, then case-insensitively, in the model's prompt table. An
unknown task is `400` and lists the valid ones. A model with no prompt table
answers `400` ("use 'instruction' for a literal prefix"). `task` and
`instruction` together are `400` (`EmbeddingOptions.parse`). With neither, no
prefix is added. For items with media, the prefix goes in front of the first
text only; media-only items get none (the reference and model card do the same).

**3. Per-input content parts.** `input` may be a string, an array of strings
(both unchanged), or an array whose items are strings or
`{"content": [parts]}`. Part types: `text`; `image_url` (object or bare string)
and `input_image`; `input_audio` (OpenAI shape) and `audio_url`; `video_url` and
`input_video`. The parser is `EmbeddingInputParser`
(`Sources/KrillEngine/EmbeddingInputs.swift`). The legacy `/api/embeddings`
stays text-only. Media sent to a text-only embedding model is `400`.

**4. Base64 / data URLs only; no remote fetch.** `http(s)://`, `file:` and paths
are `400`. Nothing in a request makes the server open a connection or read a
local file.

**5. The existing 10 MB body cap** (`ServerLimits.maxBodySize`) is the only size
limit on the request. A larger body is `413`. A consequence stated in the docs:
8,192 tokens of raw 16 kHz PCM16 WAV (about 13 MB) cannot be sent in one request,
while compressed audio or video can.

**6. Over-limit input is refused, never truncated.** An item whose media tokens
plus text exceed the 8,192-token context is `400`, and the message says how long
an input can be. Audio over 30 s (480,000 samples at 16 kHz) and audio under
0.1 s are `400`. Text-only items keep the old truncation to the context (keeping
`<eos>`). The reference truncates audio silently at 30 s; Krill does not copy that.

**7. Towers load lazily and separately.** The vision tower (shared by images and
video) loads on the first request with an image or video part; the audio tower on
the first audio part. Text-only requests touch neither
(`ensureVisionTower`, `ensureAudioTower`; `isVisionTowerLoaded` /
`isAudioTowerLoaded` are false after text requests, tested with weights).

**8. float16 is not accepted.** EmbeddingGemma 2 computes in float32 (default) or
bfloat16. `KRILL_EMBED_DTYPE` recognises only those, `setComputeDtype` rejects
anything else, and NaN / Inf output is never returned (`500`).

## Alternatives considered

- **`dimensions` rejected for models without MRL.** Not chosen: it would change
  the behaviour of every existing embedding model for a field that was ignored
  before (`EmbeddingEngine.swift` states this explicitly).
- **`task` folded into `instruction`.** Not chosen in the code: `instruction`
  remains a literal prefix and `task` a table lookup, kept as separate fields that
  cannot be combined.
- **Remote URL fetch for media.** Not chosen; the docs state the server never
  fetches anything for a request. A fetch would make a request open outbound
  connections.
- **Truncating over-long input, or windowing long audio.** Over-long media is
  refused. Windowing audio is listed as open in `docs/BACKLOG.md`.
- **A separate endpoint or a separate size limit for media.** Not chosen: the
  content-part items share `/v1/embeddings` and `/api/embed` and the existing body
  cap.
- **Loading all towers with the model.** Not chosen: text-only users would pay
  the memory of towers they never use (see Consequences).

## Why this over the others

Plain strings and existing models are untouched, so nothing already deployed
changes behaviour. The part shapes follow the OpenAI / sentence-transformers
message format that clients already produce. Refusing instead of truncating means
a vector is never silently computed from less input than the caller sent; the
`400` message tells the caller the limit. Lazy towers keep the text-only profile
(measured: 1551 MB fp32 / 524 MB bf16 MLX peak, tower not loaded).

## Consequences

- Clients can embed an image, audio clip or video with its caption as one vector,
  and pick retrieval prefixes by name (`SearchQuery` / `Document`).
- `usage.prompt_tokens` counts every token, soft tokens and markers included.
- The first request carrying a modality pays the tower load (about 0.2-1.05 s in
  the measurements) and then holds the tower: about 0.7 GB (fp32) / 0.35 GB (bf16)
  of weights for vision, about 1.2 GB MLX peak (fp32) for audio.
- A large video or audio file must fit in 10 MB of body after base64.
- Other embedding models still return `400` for `task` and ignore `dimensions`.

## Testing / verification

- Weight-free unit tests: `EmbeddingInputsTests` (part parsing, limits, `400`
  cases), `EmbeddingGemma2AudioTests`, `EmbeddingGemma2VideoTests`, the sequence
  builder tests, and a lazy-load test that asserts both towers are unloaded after
  text requests.
- Parity tests (skip without `KRILL_EG2_DIR`) against sentence-transformers fp32
  reference vectors in `Tests/KrillEngineTests/Fixtures/eg2_mm/`; gate fp32
  cosine >= 0.999. Worst measured: image 0.99941, audio / mixed 0.99952, video
  0.99948.
- `docs/EMBEDDINGGEMMA2.md` lists the measured limits and what was not tested.

## Key files

- `Sources/KrillEngine/EmbeddingOptions.swift` — `dimensions` / `task` /
  `instruction` parsing and the prompt table.
- `Sources/KrillEngine/EmbeddingInputs.swift` — content-part parsing.
- `Sources/KrillEngine/EmbeddingEngine.swift` — prefix resolution, `dimensions`
  validation, lazy towers, limits.
- `Sources/KrillServer/ServerParsing.swift` — `maxBodySize`.
- `docs/EMBEDDINGGEMMA2.md`, `docs/SERVER_API.md` — the user-facing reference.
