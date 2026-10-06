# EmbeddingGemma 2 (`google/embeddinggemma-2`)

Krill serves EmbeddingGemma 2 natively (Swift + MLX) on its three embeddings
endpoints. **Text (Milestone 1) and images, alone or interleaved with text
(Milestone 2a), are supported.** Audio and video are Milestones 2b / 2c (design
and an extension guide at the end of this file).

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
| Images | yes, see "Images and mixed input" (vision tower loaded lazily on the first image) |
| Not served | audio / video inputs (tensors skipped at load; the request parts answer `400 not yet supported`) |

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

## Images and mixed input (Milestone 2a)

Request format on `POST /v1/embeddings` and `POST /api/embed` (the legacy
`/api/embeddings` stays text-only). `input` is a string, an array of strings
(both unchanged, for every model) or an array whose items may also be
**content-part items**; one item is one embedding and part order is token order,
so an image plus its caption is one joint vector:

```bash
IMG=$(base64 < photo.png | tr -d '\n')
curl localhost:57455/v1/embeddings -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $KRILL_API_KEY" -d '{
  "model": "embeddinggemma-2",
  "task": "Document",
  "dimensions": 512,
  "input": [
    "plain text still works",
    {"content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,'"$IMG"'"}}]},
    {"content": [{"type": "text", "text": "a red bicycle"},
                 {"type": "image_url", "image_url": {"url": "data:image/png;base64,'"$IMG"'"}}]}
  ]}'
```

| Part | Shape |
|---|---|
| text | `{"type":"text","text":"..."}` |
| image | `{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}` (`image_url` may also be a bare string), or `{"type":"input_image","image_url":"data:..."}` / `{"type":"input_image","data":"<base64>"}` |
| audio, video | `input_audio`, `video_url` and friends are recognised and answer `400 ... not yet supported` |

Rules (all enforced and tested):

- **Data URLs or base64 only.** `http(s)://`, `file:` and paths are `400`: the
  server never fetches anything for a request. The whole body is capped at 10 MB.
- Decodes whatever ImageIO reads (PNG, JPEG, GIF, TIFF, HEIC, BMP); alpha is
  dropped like PIL `convert("RGB")`, EXIF orientation is not applied (like PIL
  `Image.open`).
- **Task prefix:** applied only to items that contain text, as a string prefix in
  front of the first text (the reference prepends it as a system message); an
  image-only item gets no prefix (model card: media are passed without one).
- Touching text parts are one string for the tokenizer (the reference chat
  template concatenates them); media break the string. Text parts must not
  contain the literal placeholders (`<|image|>`, `<|image>`, ...): `400`.
- **No truncation of media inputs.** The context is 8,192 tokens (an image is up
  to 280 soft tokens + 2 markers, about 29 images); an item over it is `400`
  with the token count. Text-only items keep the old truncation.
- `usage.prompt_tokens` / `prompt_eval_count` count every token (soft tokens and
  markers included). `dimensions` and `task` work on media items.
- Media sent to a text-only embedding model is `400`; plain strings take the
  unchanged code path for every model.

### Image parity (fp32 sentence-transformers reference, cosine)

Reference: sentence-transformers 6.1 / transformers 5.19, fp32, CPU, one input per
call. Fixtures: `Tests/KrillEngineTests/Fixtures/eg2_mm/` (synthetic images, see its
README). Gate: fp32 cosine >= 0.999.

| Input | fp32 | bf16 |
|---|---|---|
| landscape 640x480 PNG | 0.99999997 | 0.99995039 |
| square 224 PNG | 0.99999558 | 0.99998077 |
| tiny 100x37 PNG | 0.99996220 | 0.99994092 |
| RGBA 320x240 PNG (partial alpha) | 0.99998877 | 0.99996646 |
| grayscale 300x200 PNG | 0.99998540 | 0.99996077 |
| portrait 480x800 JPEG | 0.99977382 | 0.99974129 |
| large 1920x1080 JPEG | 0.99981213 | 0.99978765 |
| panorama 1200x300 JPEG | 0.99941155 | 0.99939396 |
| mixed: text then image | 0.99999996 | 0.99995416 |
| mixed: image then text (portrait JPEG) | 0.99957879 | 0.99956215 |
| mixed: text, image, text | 0.99997338 | 0.99995006 |
| mixed: two images + caption | 0.99989432 | 0.99989301 |
| mixed + `Document` task prefix | 0.99996060 | 0.99993259 |
| mixed + `SearchQuery` task prefix | 0.99999375 | 0.99997316 |
| **worst** | **0.99941155** | **0.99939396** |

All 14 pass the 0.999 gate. The sequences are identical to the ids
sentence-transformers fed the model (asserted for all 14), and patch grids / soft-token
counts equal the reference for every image. **The residual is the JPEG decoder, not
the model or the resize:** Apple ImageIO and libjpeg (PIL) differ in chroma
upsampling (mean 0.2-0.4 of 255, up to 40-66 levels on 3-6% of pixels at chroma
edges). Feeding the PIL-decoded pixels of the three JPEGs (as PNG) to the server gives
0.999996 / 0.999995 / 0.999987 in fp32. PNG inputs are limited only by the resize
kernel (PIL-style 8-bit separable bicubic vs torchvision's: max 2 levels on < 0.5% of
pixels, mean abs <= 0.006). Real photos are smoother than these synthetic edges, so
expect the JPEG gap to be smaller in practice, but it is a known property of the
platform decoder. The bf16 floor is the same 0.9994 (the decoder dominates).

### Image speed and memory

Measured over HTTP on loopback with the `make release` binary on this Mac
(shared with other jobs; 640x480 image = 266 soft tokens, 270 tokens total):

| | fp32 (default) | bf16 |
|---|---|---|
| text-only first request incl. load | 1.18 s | 1.26 s |
| text-only single query p50 | 17.2 ms | 15.7 ms |
| text-only MLX peak / RSS after first request | 1551 MB / 698 MB | 524 MB / 696 MB |
| first image request incl. lazy tower load | 1.05 s | 0.88 s |
| 1 image p50 (p95) | 376 ms (384) | 396 ms (403) |
| 8 images, one request of 8 items | 3.14 s = 392 ms/image | 3.20 s = 400 ms/image |
| 6 images in ONE item (1,586 tokens) | 2.44 s | 2.41 s |
| MLX peak / RSS with the tower loaded, after image work | 3112 MB / 1002 MB | 2117 MB / 1601 MB |
| text query p50 after the tower is resident | 14.9 ms | 11.8 ms |

Text-only users are unaffected: the tower is not loaded (verified by a test and by
the numbers above: peak and RSS before any image are the text-only profile, and
text latency does not move once the tower is resident). The tower costs ~0.7 GB
(fp32) / ~0.35 GB (bf16) of weights on top, plus attention activations while an
image runs (peak 1.3-1.6 GB above the text peak for a 2,400-patch image). Images
are processed one at a time, ~0.4 s each; a large share of that is the CPU resize
and patchify, which has not been optimised (the tower itself runs only real
patches). The Milestone-1 text numbers above are from a different request mix, so only the
before/after-the-tower comparison within this table is like-for-like.

### How it works (verified against transformers 5.19 and sentence-transformers' own token ids)

1. **Preprocess** (`EG2ImagePreprocessor`; the Gemma 4 chat `preprocessImage` is a
   different recipe and is untouched): decode, resize aspect-preserving so the
   16-px patch grid has at most `280 * 9 = 2520` patches with both sides a
   multiple of 48 (`get_aspect_ratio_preserving_size`, ported and pinned by
   tests), PIL-style antialiased bicubic, rescale 1/255, **no normalisation**, cut
   into 16x16x3 patches (row-major; each patch is (py, px, c)). Small images are
   upscaled to fill the budget (100x37 becomes 480x1296, 270 soft tokens).
2. **Tower** (`EG2VisionTower` = `vision_tower.*` + `embed_vision.*`): Krill's
   existing `VisionEncoder` runs this checkpoint unchanged (16 layers, 2-D axial
   RoPE, factored position table, pooler x sqrt(768)); clipping is off, so the
   missing clip scalars are defaulted. Only the real patches are run (no pad to
   2520): padding keys are masked in the reference, so it is the same function (a
   unit test compares it with the padded forward on a tiny tower). Then 3x3
   position pooling to `patches / 9` tokens, scale-free RMSNorm, 768->512 linear.
3. **Sequence** (`EG2SequenceBuilder`): `<bos> ... <boi> <image>xN <eoi> ... <eos>`;
   the soft tokens replace the placeholder rows (not multiplied by sqrt(512)), PLE
   is computed after the merge, the one text backbone runs and the same mean pool
   -> 512->768 -> L2 gives the vector. The mean covers every position (soft
   tokens, `<boi>/<eoi>`, `<bos>/<eos>` and the task prefix), exactly the
   reference's `include_prompt: true` over the full attention mask. A test pins
   the ids to the ones sentence-transformers fed the model for all 14 fixtures.
4. **Strict binding** (`loadEG2VisionTower`): every `vision_tower.*` /
   `embed_vision.*` tensor must bind and every parameter must be covered with the
   right shape. The only parameters the checkpoint lacks are 448 clip scalars,
   defaulted only because `use_clipped_linears` is false in its config (if it
   were true they would be required). Real checkpoint: 211 tensors bound.
5. **Lazy:** the tower loads on the first request that carries an image;
   text-only requests never touch it.

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

# Milestone 2 design: image (DONE in 2a), audio, video

> **Status.** Image was implemented and measured in Milestone 2a; the sections
> below are kept as the design record. What turned out wrong or incomplete in this
> design is listed here; the rest of the image section held up.
>
> - *"Audio capped at 280 soft tokens (min 280)"* is **wrong** for the real model path: the
>   reference does not cap (a 23.3 s clip gave 583 `<audio>` tokens; 25 tokens per second);
>   `audio_seq_length: 280` is only used by a serving-framework helper.
> - *Gap 1 (clip scalars)*: confirmed, solved by defaulting exactly those scalars when
>   `use_clipped_linears` is false (counted, strict otherwise).
> - *Gap 2 (preprocessor)*: confirmed; additionally the resize is torchvision's
>   antialiased bicubic on uint8 (ported PIL-style, mean error <= 0.006/255).
> - *Gap 3 (pooler x sqrt(768))*: confirmed by parity, no change needed.
> - *"Memory ~0.34 GB bf16 extra"*: right for weights; add ~1.3-1.6 GB peak activations
>   while an image of ~2,400 patches runs.
> - The reference runs images padded to 2,520 patches; Krill runs only the real
>   patches (identical result, tested).
> - The task prefix is applied by sentence-transformers to text-bearing items only (as a
>   system message, i.e. a string prefix); media-only items get none.
> - `usage.prompt_tokens` counts soft tokens and markers (the design said the same).

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

1. Reference vectors per modality (done, 2a); 2. `forward(inputsEmbeds:)` and sequence
   builder (done, 2a); 3. image preprocessor + vision loader, parity >= 0.999 (done, 2a);
   4. audio parity (no 280 cap, see above); 5. video frame sampler; 6. request parsing
   + docs (done for images; audio/video parts answer 400 until their tower lands). Load the towers lazily so text-only users keep the
   0.8 s / 1.5 GB profile.

---

## Extension guide for audio/video

Milestone 2a deliberately built the shared plumbing so a new modality is a tower,
a preprocessor and a request part type. Nothing else changes. What already exists
and what each next agent adds:

**Already there (do not duplicate)**

| Piece | Where |
|---|---|
| Marker / placeholder ids (`boa`, `audio`, `eoa`, `video`, `boi`, `eoi`) read from `config.json` | `EG2ModalityTokens` in `Sources/KrillCore/EmbeddingGemma2Sequence.swift` (`EmbeddingGemma2Config.modalityTokens`) |
| Layout of every modality's block: audio `<boa> <audio>xN <eoa>`, video one `<boi> <video>xN <eoi>` per frame | `EG2MediaBlock(.audio, softTokensPerBlock:)` / `EG2MediaBlock(.video, softTokensPerBlock:, blocks: frames)` and `EG2SequenceBuilder.build` (same file; already unit-tested for audio and video layouts) |
| Scatter into the text embeddings + count checks | `EmbeddingGemma2Model.mergedEmbeddings(_:features:)` where `features[.audio]` / `features[.video]` is one `[softTokens, 512]` array per block, in order (video: one array per frame, or one per video if you split by `blocks`: the builder emits one span per frame, so supply one `[n, 512]` per frame) |
| Backbone on merged embeddings, mean pool over every position | `EmbeddingGemma2Model.forward(inputsEmbeds:lengths:)`, `pooled(inputsEmbeds:lengths:)` |
| Lazy tower slot, 400 mapping, MRL / task handling, token accounting | `EmbeddingEngine.embedMediaItem` / `ensureVisionTower` in `Sources/KrillEngine/EmbeddingEngine.swift` |
| Request parsing with `input_audio` / `video_url` already recognised (they answer `400 not yet supported`) | `EmbeddingInputParser.notYetSupported` and `parsePart` in `Sources/KrillEngine/EmbeddingInputs.swift` |
| Strict loader pattern, clip-scalar rule | `loadEG2VisionTower` in `Sources/KrillCore/EmbeddingGemma2Vision.swift` |
| Reference vectors, ids and media for audio and video | `Tests/KrillEngineTests/Fixtures/eg2_mm/` (`reference_audio.json`, `reference_video.json`, `audio/`, `video/`, README) |

**Audio (2b) adds**

1. `Sources/KrillCore/EmbeddingGemma2Audio.swift`: `EG2AudioTower` (`audio_tower.*` +
   `embed_audio.*`, keys equal to the checkpoint's, 752 tensors; the clipped linears
   DO have scalars here, `use_clipped_linears: true`), a strict `loadEG2AudioTower`
   mirroring `loadEG2VisionTower`, and a feature extractor/preprocessor
   (16 kHz mono, 128 mel, frame 320, hop 160, fft 512; see `processor_config.json`).
   Soft tokens per clip: `(n + 160 - 321)/160 + 1` mel frames, then two stride-2
   k3 p1 convs. **The reference does not cap at 280** (measured: 23.3 s -> 583
   tokens), only the 8,192 context limits it.
2. `EmbeddingPart.audio(Data, format: String)` (the commented case in
   `EmbeddingInputs.swift`), a `case "input_audio"` in `parsePart` (remove its
   `notYetSupported` entry), a decoder for wav (and whatever else you accept) to
   float32 mono 16 kHz.
3. In `embedMediaItem`: a `.audio` case that sizes the block
   (`EG2MediaBlock(.audio, softTokensPerBlock: n)`), an `ensureAudioTower()`
   twin of `ensureVisionTower()` (lazy, own lock), and `features[.audio] = [...]`.
4. Parity against `reference_audio.json` (fp32 >= 0.999).

**Video (2c) adds**

1. Frame extraction (AVFoundation 1 fps, uniform down-sample to at most 32 frames;
   accept a pre-decoded frame list as the portable input) producing `EG2RGBImage`s.
2. Per frame `EG2ImagePreprocessor.prepare(rgb, maxSoftTokens: EG2ImagePreprocessor.videoFrameSoftTokens)`
   (140-token budget, already supported and tested) and the SAME `EG2VisionTower.softTokens`
   (video has no tower of its own: `vision_tower` + `embed_vision`, placeholder id
   `video`). Layout `EG2MediaBlock(.video, softTokensPerBlock: n, blocks: frames)`;
   `features[.video]` = one `[n, 512]` array per frame.
3. `EmbeddingPart.video(Data)`, `parsePart` case for `video_url` / `input_video`.
4. Parity against `reference_video.json` (3 s -> 3 frames, 40 s -> 32 frames; 130 soft
   tokens per 320x240 frame). A 32-frame clip is about 4.2k tokens, so the sliding
   window (512) is exercised.

Also update `docs/SERVER_API.md`, this file's request table and `CHANGELOG.md`.
