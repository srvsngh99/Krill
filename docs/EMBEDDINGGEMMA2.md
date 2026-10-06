# EmbeddingGemma 2 (`google/embeddinggemma-2`)

Krill serves EmbeddingGemma 2 natively (Swift + MLX) on its three embeddings
endpoints. **Milestone 1 is text only.** Image, audio and video are
Milestone 2 (design at the end of this file).

```
krill pull embeddinggemma-2        # 1.49 GB, Apache-2.0, not gated
krill serve
curl localhost:57455/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model":"embeddinggemma-2","input":"What causes the northern lights?","task":"SearchQuery"}'
```

## What is supported (Milestone 1)

| | |
|---|---|
| Model | `embedding_gemma2` (`EmbeddingGemma2Model`), 740M params in one `model.safetensors`; the text path is the remainder after the 170M vision and 300M audio towers, ~270M |
| Text backbone | 24 layers, d_model 512, FFN 2048 (GELU-tanh gated), 4 heads, head_dim 256 / 512, 2 KV heads (sliding) / 1 (global), sliding window 512, 5 sliding : 1 global, vocab 262,144 |
| Head | mean pool over real tokens, 512 -> 768 projection, L2 normalise |
| Context | 8,192 tokens (inputs are truncated to this, keeping `<eos>`) |
| Dimensions | 768 (default), 512, 256, 128 (Matryoshka) |
| Compute dtype | **float32 (default)** or bfloat16. **float16 is refused**: it is unsafe for this model. |
| Languages checked | English, French, code, Hindi, Kannada, Sanskrit (Devanagari) |
| Not served | vision / audio / video inputs (their tensors are skipped at load) |

### It is not "Gemma 4 run bidirectionally"

Verified against `transformers/models/embedding_gemma2/modeling_embedding_gemma2.py`
(5.19.0) and the checkpoint header, so `Gemma4TextModel` was **not** reused:

- attention is bidirectional on every layer; sliding layers allow
  `|i - j| <= 512`; there is no KV cache, no KV sharing, no MoE, no `lm_head`;
- full-attention layers (5, 11, 17, 23) use head_dim 512 and one KV head
  (`per_layer_config`), sliding layers head_dim 256 and two KV heads;
- RoPE is plain rotate-half over the whole head (theta 1e6 full / 1e4
  sliding). Gemma 4's proportional / partial RoPE does not apply;
- the per-layer-embedding block is projection-only
  (`ple.per_layer_model_projection(embeds) * 512^-0.5`, RMSNorm); there is no
  `embed_tokens_per_layer` table;
- `v_norm` is a scale-free RMS norm on V, attention scale is 1.0.

### Strict weight binding

The VL loaders use lax `verify: []`, where an unbound weight silently yields
plausible garbage. `loadEmbeddingGemma2` instead:

1. partitions every checkpoint key (`language_model.*` = text;
   `vision_tower.*` / `embed_vision.*` / `audio_tower.*` / `embed_audio.*` =
   skipped; **anything else throws**);
2. binds the text keys with `.allModelKeysSet + .shapeMismatch +
   .noUnusedKeys`, so a missing, mis-shaped or extra text tensor fails the load.

On the real checkpoint it logs
`bound 413 text tensors (strict: every one consumed, none missing); skipped 963 multimodal tensors (211 vision, 752 audio)`
(413 + 963 = 1376 tensors; the header has 1377 entries including `__metadata__`).

### Tokenizer

swift-transformers' BPE seeds merges from Swift `Character`s (grapheme
clusters). For Devanagari / Kannada / Sanskrit a cluster is not a vocabulary
entry, so text fell into per-byte `<0xHH>` fallback: 2-3x too many, wrong tokens
(cosine 0.73-0.92 to the reference). Latin text and code were unaffected, which
is why it looks fine on English. EmbeddingGemma 2 therefore uses
`CodePointBPETokenizer` (`Sources/KrillTokenizer`), which seeds from Unicode
scalars like HF `tokenizers`. It supports only the Gemma `tokenizer.json` shape
(Replace `" "->"▁"` normalizer, byte-fallback BPE, `<bos> A <eos>` template) and
rejects anything else at load. The other embedding models still use the library
tokenizer (unchanged). A test pins its ids to HF `tokenizers` for 20 strings.

## Request fields

Same fields on `POST /v1/embeddings`, `POST /api/embed`, `POST /api/embeddings`
(legacy; `prompt` instead of `input`):

| Field | Type | Meaning |
|---|---|---|
| `model` | string | `embeddinggemma-2` |
| `input` / `prompt` | string or [string] | texts |
| `task` | string | selects a prefix from the model's `config_sentence_transformers.json` `prompts` table (case-insensitive): `SearchQuery`, `Document`, `Retrieval-query`, `Retrieval-document`, `Classification`, `Clustering`, `SentenceSimilarity`, `QuestionAnswering`, `FactChecking`, `CodeRetrieval`, `Reranking`, `Summarization`, `BitextMining`, ... An unknown task is `400` and lists the valid ones. |
| `instruction` | string | literal prefix (existing field, unchanged). Mutually exclusive with `task`. |
| `dimensions` | int | 768 / 512 / 256 / 128: first N components then re-normalised (unit norm). Any other value is `400`. |

**Default: no prefix** (the model works, but the card says quality is best with
the prompts). For retrieval send `task: "SearchQuery"` for queries and
`task: "Document"` for documents. `dimensions` defaults to 768.

For every other embedding model behaviour is unchanged: `dimensions` is ignored
as before, and `task` returns `400` because those models define no prompt table.
NaN / Inf output is never returned: the request fails with `500` instead.

Batches are sorted by length and run as padded batches with a key-padding mask
(<= 32 texts and <= 8,192 padded tokens per forward); results come back in input
order. A text embedded alone and inside a mixed batch agree (cos 1.0000000 fp32,
0.99987 bf16).

Env: `KRILL_EMBED_DTYPE=float32|bfloat16` (EmbeddingGemma 2 only),
`KRILL_EMBED_LOG_MEM=1` logs the MLX peak-memory high-water mark per request.

## Measured numbers

All numbers below are from real runs on this Mac (Apple Silicon, shared with
other jobs; throwaway `HOME`, own port, `make release` binary). Reference =
sentence-transformers fp32 on MPS for 21 strings (7 English, 1 French, 4 code, 3
Hindi, 2 Kannada, 3 Sanskrit, 1 ~3k-token document that exercises the sliding
window), raw and under all 8 prompt modes (9 modes x 21 strings = 189 vectors).

### Parity (cosine to reference, over all 9 prompt modes)

| Group | fp32 min | fp32 mean | bf16 min | bf16 mean |
|---|---|---|---|---|
| English | 0.99999982 | 1.00000000 | 0.99988234 | 0.99995923 |
| French | 1.00000000 | 1.00000000 | 0.99995714 | 0.99996799 |
| Code | 0.99999994 | 1.00000000 | 0.99986959 | 0.99994320 |
| Hindi | 0.99999988 | 1.00000000 | 0.99989474 | 0.99996430 |
| Kannada | 0.99999994 | 1.00000000 | 0.99986100 | 0.99993759 |
| Sanskrit | 1.00000000 | 1.00000000 | 0.99990207 | 0.99994838 |
| ~3k-token doc | 1.00000000 | 1.00000000 | 0.99997687 | 0.99997985 |
| **All** | **0.99999982** | **1.00000000** | **0.99986100** | **0.99995476** |

Gate was cosine >= 0.999 (not an elementwise tolerance): both pass. MRL dims
768/512/256/128 are unit-norm and match the truncated reference (fp32 cos
1.000000; bf16 >= 0.99992). Card example (query vs relevant / France / cats)
gives 0.8707 / 0.559 / 0.530, relevant first.

### Speed and memory vs sentence-transformers

Krill numbers are end-to-end over HTTP on loopback (tokenise + forward + JSON),
median of 3 for batches, 50 requests for the single query. The
sentence-transformers column is the earlier reference run (MPS, bf16,
text-only, `encode`); the two are not the same harness, so treat small
differences as noise.

| | sentence-transformers MPS bf16 | Krill bf16 | Krill fp32 (default) |
|---|---|---|---|
| single query p50 | 25 ms | 14.3 ms (p95 20.0) | 15.4 ms (p95 26.5) |
| batch 32 x ~256 tok | 30 docs/s | 52.9 docs/s | 45.1 docs/s |
| batch 32 x ~1k tok | 4.9 docs/s | 10.6 docs/s | 9.3 docs/s |
| peak GPU memory | ~9 GB (driver) | 1.55 GB (MLX peak) | 3.3 GB (MLX peak) |
| load time | (not measured here) | ~0.8 s warm cache (first request, includes load) | ~0.8 s |

Load time is the first-request wall time with the checkpoint already in the
page cache; a cold read of 1.49 GB from disk will be slower. Peak memory is
`Memory.peakMemory` after the 32 x ~1k-token batch (RSS does not see MLX memory).

---

# Milestone 2 design: image, audio, video

Source of truth: `config.json`, `processor_config.json`, the safetensors header
(1,377 entries) and `transformers/models/embedding_gemma2/*` 5.19.0. Nothing in
this section has been run: the key/shape checks are by reading Krill's module
declarations against the header, **no numeric parity was attempted**. Milestone 2
must produce its own sentence-transformers reference vectors per modality and
gate on cosine >= 0.999 like the text path.

## How the model fuses modalities

There is no separate pooled "image embedding" head. The processor expands the
prompt with placeholder tokens; the tower output is projected into the text
space and **scattered into the text sequence** at those positions
(`masked_scatter`, no sqrt(512) scaling for the soft tokens; PLE is computed
after the merge); then the one text backbone runs and the same mean pool ->
512->768 -> L2 produces the vector. Pooling (`include_prompt: true`) covers
every token, soft tokens and the `<boi>/<eoi>/<boa>/<eoa>` markers included.
Layout from `processing_embedding_gemma2.py`:

- image: `<boi>` + `<image>` x N + `<eoi>` (`boi_token_id` 255999 text token,
  `image_token_id` 258880, `eoi_token_id` 258882)
- video: one image-style block per frame using `video_token_id` 258884, no
  timestamps by default (`add_timestamps: false`)
- audio: `<boa>` (256000) + `<audio>` x N (258881) + `<eoa>` (258883)

`embed_vision` / `embed_audio` = scale-free RMSNorm then bias-free Linear to 512.

## Image

- **Keys** (211 tensors incl. embed_vision): `vision_tower.patch_embedder.input_proj.weight [768,768]`,
  `vision_tower.patch_embedder.position_embedding_table [2,10240,768]`,
  `vision_tower.encoder.layers.{0..15}.{input_layernorm, post_attention_layernorm, pre_feedforward_layernorm, post_feedforward_layernorm}.weight [768]`,
  `...self_attn.{q,k,v,o}_proj.linear.weight [768,768]`, `...self_attn.{q_norm,k_norm}.weight [64]`,
  `...mlp.{gate,up}_proj.linear.weight [3072,768]`, `...mlp.down_proj.linear.weight [768,3072]`,
  `embed_vision.embedding_projection.weight [512,768]`.
- **Config** (`gemma4_vision`): hidden 768, 16 layers, 12 heads, 12 KV heads,
  head_dim 64, FFN 3072, patch 16, pooling kernel 3, 280 output soft tokens,
  position table 10240, axial RoPE theta 100, GELU-tanh, eps 1e-6,
  `use_clipped_linears: false`, `standardize: false`.
- **Does Krill's encoder match?** `VisionEncoder` defaults are exactly these
  numbers and every `@ModuleInfo` key path matches the header
  (`patch_embedder.{input_proj,position_embedding_table}`, `encoder.layers.N.*`,
  `.linear.weight` under `ClippableLinear`). `MultimodalEmbedder` (RMSNorm
  no-scale -> Linear, key `embedding_projection`) matches `embed_vision`.
- **Gaps:**
  1. `ClippableLinear` declares `input_min/max/output_min/output_max`, which the
     checkpoint does not have for the vision tower (clipping off). With the
     strict verify used for text this fails; either load with a `clipping:
     false` mode (plain `Linear`, same `.linear.weight` path) or default those
     scalars and exclude them from `.allModelKeysSet`. Do not fall back to
     `verify: []`.
  2. The preprocessor is different. `Gemma4ImageProcessor` here: RGB, bicubic
     (`resample: 3`), aspect-ratio-preserving resize so the patch grid has at most
     `max_soft_tokens * 9 = 2520` patches, rescale 1/255,
     **no normalise**, variable soft-token count up to 280. Krill's
     `preprocessImage` resizes the longest side to 672 and **pads with white** to
     a block multiple (a chat-model choice), so soft-token counts and pixels
     differ. Needs a new preprocessor, or a parameter, that reproduces the HF
     resize rule and returns the actual per-image soft-token count.
  3. Confirm the pooler: Krill's `VisionPooler` multiplies by sqrt(768); confirm
     against the Gemma4 vision pooler HF uses in this checkpoint (numeric test).
  4. Model plumbing: a sequence builder (tokens + scatter), PLE after the merge.
     `EG2` text model needs a `forward(inputsEmbeds:)` entry (it currently
     takes token ids).
- **Cost:** vision tower 170M params. Memory ~0.34 GB bf16 extra.

## Audio

- **Keys** (752 tensors incl. embed_audio): `audio_tower.subsample_conv_projection.{layer0,layer1}.{conv.weight [128,1,3,3]/[32,128,3,3], norm.weight}`,
  `...input_proj_linear.weight [1024,1024]`, 12 conformer layers
  `audio_tower.layers.N.{feed_forward1,feed_forward2}.{ffw_layer_1,ffw_layer_2}.{linear.weight, input_min/max, output_min/max}`,
  `.{pre,post}_layer_norm.weight`, `.self_attn.{q,k,v}_proj / post` (clipped
  linears, `[1024,1024]`), `.self_attn.relative_k_proj.weight`,
  `.self_attn.per_dim_scale [128]`, `.lconv1d.{linear_start [2048,1024], depthwise_conv1d.weight [1024,1,5], conv_norm, linear_end, pre_layer_norm}`,
  `.norm_pre_attn / norm_post_attn / norm_out.weight`;
  `audio_tower.output_proj.{weight [1536,1024], bias [1536]}`;
  `embed_audio.embedding_projection.weight [512,1536]`.
- **Config** (`gemma4_audio`): hidden 1024, 12 layers, 8 heads, conv kernel 5,
  chunk 12, left context 13, right 0, logit cap 50, residual weight 0.5,
  `use_clipped_linears: true`, output_proj 1536.
- **Does Krill match?** Yes on structure: `AudioConfig` defaults equal every
  value above, `AudioConformerBlock` / `SubSampleConvProjection` / `output_proj`
  (bias: true) key paths match the header, and clipped linears carry the
  min/max scalars the checkpoint has. `AudioPreprocessor` constants match the
  feature extractor (16 kHz, 128 mel, frame 320, hop 160, fft 512, floor 1e-3,
  0-8000 Hz, no dither / preemphasis / per-bin stats).
- **Gaps:** token budget and clip length. The processor caps soft tokens at
  `audio_seq_length = 280` (40 ms/token), Krill's constants are 750 and a 30 s
  / 480,000-sample cap: needs the 280 cap and the HF count formula
  (`(n + 160 - 321)/160 + 1` mel frames, then two stride-2 k3 p1 convs, min 280).
  Also the audio marker ids (`boa/eoa`) and the same `embed_audio` +
  scatter-merge plumbing as image. Decide behaviour for audio > 11.2 s (truncate
  or window-and-mean); not checked what the HF extractor does.
- **Cost:** audio tower 300M params, ~0.6 GB bf16.

## Video

- **Keys:** none of its own. Frames go through the **vision tower and
  `embed_vision`** (same 211 tensors); only the placeholder id differs
  (`video_token_id` 258884).
- **Config** (`EmbeddingGemma2VideoProcessor`): sample 1 fps uniformly,
  `max_frames: 32` (`overflow_strategy: uniform`), `max_soft_tokens: 140` per frame
  (so max 1,260 patches per frame), bicubic resize, rescale 1/255, no normalise,
  no timestamps. One `<boi>[video]xN<eoi>` block per frame; mean pooled with
  everything else, so a 32-frame clip adds up to 32 x (140 + 2) = 4,544 tokens (fits 8,192).
- **Does Krill match?** The encoder is the same as image, so everything in the
  Image section applies. Krill has **no video frame sampler or decoder** for
  embeddings; chat video (if any) does not use the 140-token budget.
- **Gaps:** frame extraction (AVFoundation `AVAssetImageGenerator` at 1 fps,
  uniform down-sample to 32), the 140-token resize budget, per-frame blocks, and
  accepting a pre-decoded list of frames as the portable input.

## Proposed request format

Keep text requests exactly as they are. Add an optional OpenAI-compatible
multimodal `input` item array (mirrors the sentence-transformers "message"
modality config). Each item is one embedding; an item is a string (text) or an
object with `content` parts, whose order is the token order that gets mean-pooled
together (so image + caption in one item becomes one joint vector):

```json
{
  "model": "embeddinggemma-2",
  "task": "Document",
  "dimensions": 512,
  "input": [
    "plain text still works",
    {"content": [{"type": "text", "text": "a red bicycle"},
                 {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]},
    {"content": [{"type": "input_audio", "input_audio": {"data": "<base64 wav/m4a>", "format": "wav"}}]},
    {"content": [{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,..."}}]}
  ]
}
```

Rules: `data:` URLs or base64 only (no network fetch); `task` prefix applies to the
text of each item only, placed first; `usage.prompt_tokens` counts all tokens
including soft tokens; unsupported media type -> `400`; a request with media sent
to a build/model without the towers -> `400` (not a silent text-only result);
size limits: image up to the 2520-patch budget, audio up to 11.2 s (decision
above), video up to 32 frames. `/api/embed` takes the same item shapes; the legacy
`/api/embeddings` stays text-only.

## Milestone 2 work order

1. Reference vectors per modality from sentence-transformers (images, a few
   clips, short videos) as fixtures; 2. text model `forward(inputsEmbeds:)` and a
   sequence builder; 3. image preprocessor + vision loader (strict, clipping off);
   parity >= 0.999; 4. audio (token cap 280) parity; 5. video frame sampler;
   6. request parsing + docs. Load the towers lazily so text-only users keep the
   0.8 s / 1.5 GB profile.
