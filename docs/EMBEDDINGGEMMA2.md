# EmbeddingGemma 2 (`google/embeddinggemma-2`)

Krill serves EmbeddingGemma 2 natively (Swift + MLX) on its three embeddings
endpoints. **Text (Milestone 1), images (2a), audio and video (2b), alone or
interleaved with text, are all supported.** The design record and the notes on
what it got wrong are at the end of this file.

```
krill pull embeddinggemma-2        # 1.49 GB, Apache-2.0, not gated
# smaller builds, 1.01 GB down to 0.44 GB (8bit, 6bit, 5bit, 4bit-dyn, nvfp4, ...): see "Quantized builds"
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
| Audio | yes, up to 30 s per clip, see "Audio and video" (audio tower loaded lazily on the first audio part) |
| Video | yes, 1 fps, at most 32 frames, see "Audio and video" (shares the vision tower; no audio track, no timestamps) |

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
| audio, video | `input_audio`, `video_url`: see "Audio and video" below |

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

## Audio and video (Milestone 2b)

Audio and video use the same request shape as images (content-part items on
`/v1/embeddings` and `/api/embed`; one item is one vector; part order is token
order; data URLs or base64 only, no fetching; whole body <= 10 MB):

```bash
# an audio clip (small clips fit on the command line; use -d @file for anything near 1 MB)
AUD=$(base64 < question.wav | tr -d '\n')
curl localhost:57455/v1/embeddings -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $KRILL_API_KEY" -d '{
  "model": "embeddinggemma-2",
  "task": "SearchQuery",
  "input": [
    {"content": [{"type": "input_audio", "input_audio": {"data": "'"$AUD"'", "format": "wav"}}]},
    {"content": [{"type": "text", "text": "spoken query: "},
                 {"type": "input_audio", "input_audio": {"data": "'"$AUD"'", "format": "wav"}}]}
  ]}'

# a video clip (mp4 / mov / m4v); build big bodies in a file
VID=$(base64 < clip.mp4 | tr -d '\n')
printf '{"model":"embeddinggemma-2","dimensions":512,"input":[{"content":[{"type":"text","text":"a cooking video: "},{"type":"video_url","video_url":{"url":"data:video/mp4;base64,%s"}}]}]}' "$VID" > body.json
curl localhost:57455/v1/embeddings -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $KRILL_API_KEY" -d @body.json
```

| Part | Shape |
|---|---|
| audio | `{"type":"input_audio","input_audio":{"data":"<base64 or data: URL>","format":"wav"}}` (OpenAI shape; `format` is a hint: `wav`, `mp3`, `m4a`, `flac`, `aiff`, `caf`, `ogg`; when absent the bytes are sniffed). Also `{"type":"audio_url","audio_url":{"url":"data:audio/...;base64,..."}}` |
| video | `{"type":"video_url","video_url":{"url":"data:video/mp4;base64,..."}}` (`video_url` may be a bare string); also `{"type":"input_video","input_video":{"data":"<base64>","format":"mp4"}}` or `{"type":"input_video","data":"<base64>"}` |

Rules and limits (all enforced and tested):

- **Audio is at most 30 s per clip** (480,000 samples at 16 kHz). The reference
  feature extractor silently truncates longer audio at 30 s; Krill answers `400`
  with the limit instead. Shorter than 0.1 s is `400`. A 30 s clip is 750 soft
  tokens (25 per second); the 280 cap in the old design note does not exist on
  the model path.
- **Video: 1 frame per second, at most 32 frames**, chosen exactly like the
  reference (see "How it works"); each frame is up to 140 soft tokens + 2 markers
  (130 for 320x240, 120 for 480x270). The audio track of a video is ignored (the
  reference never reads it); no timestamp tokens (`add_timestamps` is false in the
  checkpoint's processor config, so the layout has none either).
- **The 8,192-token context is never exceeded silently**: an item over it (any
  mix of text, images, audio, video) is `400`, and the message says how long an
  input can be ("... about 327 s of audio (25 tokens per second, each clip up to
  30 s), up to 32 video frames at up to 142 tokens each, or about 29 images ...").
  The 10 MB body cap answers `413`. Because of the body cap, 8,192 tokens of raw
  16 kHz PCM16 WAV (about 13 MB) cannot be sent in one request; compressed audio
  (mp3 / m4a / flac) or video can.
- **Task prefix**: applied only to items that contain text, as a string prefix in
  front of the first text; audio-only and video-only items get none (same rule as
  images).
- Undecodable bytes are `400` (`audio could not be decoded ...` / `video could
  not be decoded ...`); remote and `file:` URLs are `400`; media sent to a
  text-only embedding model is `400`.
- Each tower is loaded **lazily and separately**: the audio tower on the first
  audio part, the vision tower (shared by images and video) on the first image
  or video part. Text-only requests touch neither (asserted by tests and by the
  memory numbers below).

**Formats that decode.** Audio goes through `AVAudioFile` (CoreAudio), then is
mixed to mono (channel average) and resampled to 16 kHz with `AVAudioConverter`
when needed. Checked with ffmpeg-made files: WAV (PCM 8 / 16 / 24-bit, float32,
mu-law, IMA ADPCM; 8 / 16 / 44.1 / 48 kHz, mono and stereo), MP3 (the LAME/Xing
encoder-delay and padding are trimmed so the length matches ffmpeg), AAC in
M4A / MP4 and raw ADTS `.aac`, ALAC, FLAC, AIFF, CAF and Ogg Opus. **Not
decoded:** WebM / Matroska containers, and anything CoreAudio does not know
(Vorbis, WMA, AMR). Video goes through `AVAssetReader`: `mp4`, `mov`, `m4v` with
H.264 / HEVC (anything VideoToolbox decodes); not WebM / MKV / AVI. Rotation
metadata is not applied (the reference does not either).

### How audio works (verified against transformers 5.19 and sentence-transformers' own token ids)

1. **Features** (`EG2AudioPreprocessor`; shares the Gemma 4 chat path's
   `AudioPreprocessor.features`, which matches `Gemma4AudioFeatureExtractor`
   exactly: 16 kHz mono, pad to a multiple of 128 samples, semicausal left pad of
   160, 320-sample periodic-Hann frames at hop 160, rfft 512, |.|, HTK mel
   0-8000 Hz x 128 (no norm), `log(mel + 1e-3)`, padded frames zeroed).
2. **Soft tokens** = `replace_audio_token`: the per-frame validity mask (frame i
   is real iff its last sample is) sub-sampled by the two stride-2 convs, i.e.
   every 4th frame. Pinned by tests to 52 / 146 / 583 tokens for the three WAV
   fixtures (about 25 per second, no cap).
3. **Tower** (`EG2AudioTower` = `audio_tower.*` + `embed_audio.*`): Krill's
   `AudioEncoder` (the USM conformer used for Gemma 4 chat audio) runs this
   checkpoint's tower unchanged: 12 layers, d 1024, 8 heads, chunked local
   attention, clipped linears (the scalars ARE in the checkpoint here), `output_proj`
   1024 -> 1536 **with bias**; then only the valid frames, scale-free RMSNorm and a
   1536 -> 512 linear.
4. **Strict binding** (`loadEG2AudioTower`): every one of the 752 `audio_tower.*` /
   `embed_audio.*` tensors must bind and every parameter must be covered with the
   right shape, otherwise the load throws (nothing is defaulted: the config says
   `use_clipped_linears: true` and a config that said false is refused).
   **Weight layout gotcha:** this HF checkpoint stores the two conv kinds in
   PyTorch layout (`subsample_conv_projection.*.conv.weight` `[out,in,kH,kW]`,
   `lconv1d.depthwise_conv1d.weight` `[C,1,K]`), unlike the channel-last mlx-vlm
   weights the Gemma 4 loader sees (`[out,kH,kW,in]`, `[C,K,1]`; the Gemma 4
   loader comments say "no transpose"). The loader converts each, and only when the
   transposed shape is exactly the module's; a test runs the same tiny tower from
   both layouts.
5. **Sequence**: `<bos> ... <boa> <audio>xN <eoa> ... <eos>`, soft tokens replace the
   placeholder rows, then the same backbone and mean pool as everything else.

### How video works

1. **Frame sampling** (`EG2VideoSampler`, a port of
   `EmbeddingGemma2VideoProcessor.sample_frames`): `step = native_fps / 1`,
   `num_sampled = max(1, int(duration))`,
   `indices = [min(total - 1, int(i * step)) ...]`; when there are more than 32,
   `np.linspace(0, n - 1, 32, dtype=int)` of them. Tests pin 14 (frames, fps,
   duration) cases to values produced by the reference code and numpy (3 s at 10
   fps -> `[0, 10, 20]`; 40 s at 10 fps -> 32 frames ending at 390; 29.97 fps x
   4.004 s -> `[0, 29, 59, 89]`; 0.5 fps, 59.94 fps and 5-minute clips included).
2. **Decode** (`EG2VideoSource`): the file is written to a temp dir (removed when
   the request ends), the container is read for fps / duration / size and the
   compressed samples are counted (no decode) so the prompt can be sized and an
   over-long item rejected before any pixel work; then `AVAssetReader` decodes
   sequentially and only the sampled frames are kept, each resized and patchified
   as soon as it is decoded.
3. **Per frame**: the 140-token budget (`EG2ImagePreprocessor.prepare(.., 140)`, 1,260
   patches), the SAME vision tower and `embed_vision` as images, one `[n, 512]` array
   per frame.
4. **Sequence**: one `<boi> <|video|>xN <eoi>` block per frame, concatenated, no
   timestamps, mean pooled with everything else (a 32-frame clip is 4,226 tokens, so
   the sliding window of 512 is exercised).

### Decoder differences found (and what was done)

- **Video pixels** were the one real gap. AVFoundation's own BGRA output upsamples
  chroma smoothly, the reference decoder (torchcodec / ffmpeg / swscale) replicates
  it: that alone gave mean 2.9 / 255 pixel error (5% of pixels off by > 8) and
  cosines of 0.9934-0.9986. Asking AVFoundation for the decoder's 8-bit 4:2:0 planes
  and converting them the swscale way (nearest chroma, the stream's matrix with
  BT.601 when untagged, the stream's range) measures mean 0.19 / 255, max 1, against
  torchcodec frames, and every video case is >= 0.99948. A weight-free test compares
  12 frames of 4 fixtures with the reference decoder's frames (12x16 block means,
  `reference_video_frames.json`); 10-bit, 4:2:2 and 4:4:4 sources are converted by
  VideoToolbox to 8-bit 4:2:0 first (not measured).
- **MP3**: AVAudioFile does not honour the LAME/Xing delay / padding tag that ffmpeg
  does (it left 1,532 extra samples = 3 extra soft tokens on a 5.8 s clip, cosine
  0.9989 -> below the gate). Krill now trims the tag's delay and padding (AVFoundation
  already removes the 529-sample decoder delay; proven by cross-correlating the two
  decoders, lag = the tag's delay). Without a tag nothing is trimmed, like ffmpeg.
- **M4A (AAC)**: AVFoundation applies the edit list (93,520 samples); ffmpeg, which
  made the reference, returned 94,208, so the clip is 1 soft token shorter than the
  reference's. The vector passes the gate (0.99952) but its token count and ids
  differ by that one token; **feeding the ffmpeg-decoded samples reaches the
  reference's ids and 0.99987** (`a8_m4a_ffmpeg16k.wav`), proving the cause is the
  decoder, not the model path. In this one case AVFoundation is arguably the more
  correct decoder.
- **FLAC 44.1 kHz stereo**: same 33,315 samples as ffmpeg; the resamplers differ
  (`AVAudioConverter` vs swr), 0.99957.
- The audio cosines are a uniform ~0.9997-0.9999, not 0.999999 like text: all of
  it is below the 1e-3 gate and the same in fp32 and bf16, so it is neither the dtype
  nor the tokens. I did not chase it further; the likely sources are the float32 mel
  (DFT as a matmul vs numpy's float64 FFT, then `log`) and the sound-alike
  resampler, but this is **not verified**.

### Audio and video parity (fp32 sentence-transformers reference, cosine)

Reference: sentence-transformers 6.1 / transformers 5.19, fp32, CPU, one input per
call (audio as 16 kHz float arrays, ffmpeg-decoded where the file is not 16 kHz mono
PCM; video as mp4 paths through torchcodec). Gate: fp32 cosine >= 0.999; bf16
gated at >= 0.99. The sequence ids equal the ids sentence-transformers fed the model
for every case except `a8.m4a` (see above), and `usage.prompt_tokens` equals its
`n_tokens`.

| Input | fp32 | bf16 |
|---|---|---|
| audio a3.wav (2.1 s, 52 tokens) | 0.99970342 | 0.99964802 |
| audio a8.wav (5.8 s, 146) | 0.99990566 | 0.99988864 |
| audio a20.wav (23.3 s, 583) | 0.99976995 | 0.99975483 |
| audio 44.1 kHz stereo FLAC (a3) | 0.99957053 | 0.99965542 |
| audio MP3 (a8) | 0.99988533 | 0.99987294 |
| audio M4A / AAC (a8; ids differ by 1 token, see above) | 0.99951660 | 0.99949544 |
| mixed: text then audio | 0.99973738 | 0.99971141 |
| mixed: audio then text, `SearchQuery` | 0.99990973 | 0.99988684 |
| mixed: text then audio, `Document` | 0.99974860 | 0.99974093 |
| mixed: text, audio, image | 0.99990054 | 0.99988709 |
| video v3 (3 s, 3 frames, 390 soft tokens) | 0.99948133 | 0.99942534 |
| video v40 (40 s -> 32 frames, 4,160 soft, 4,226 tokens) | 0.99983380 | 0.99983823 |
| video v_ntsc (29.97 fps, 480x270, 4 frames, 120/frame) | 0.99981755 | 0.99977050 |
| video with an audio track (v_audio, 2 frames) | 0.99954901 | 0.99951622 |
| mixed: text, video, text | 0.99968090 | 0.99960508 |
| mixed: video then text, `Document` | 0.99980937 | 0.99979126 |
| **worst audio / mixed** | **0.99951660** | **0.99949544** |
| **worst video / mixed** | **0.99948133** | **0.99942534** |

The same vectors come back over HTTP (`/v1/embeddings`, data URLs / base64) at
the same cosines (worst 0.99957 audio, 0.99948 video, fp32; 0.99965 / 0.99943 bf16).

### Audio and video speed and memory

Measured over HTTP on loopback with the `make release` binary (shared Mac, fresh
server per run, `KRILL_EMBED_LOG_MEM=1`). Audio is decoded, featurised and run
at roughly 8-18 ms per second of audio; a video frame costs ~0.15-0.2 s (the vision
tower is not optimised, and a 32-frame clip is one 4,226-token backbone pass).

| | fp32 (default) | bf16 |
|---|---|---|
| text-only MLX peak / RSS after first request | 1551 MB / 702 MB | 524 MB / 696 MB |
| text single query p50 (before any tower) | 14.2 ms | 13.1 ms |
| first audio request incl. lazy audio-tower load | 0.97 s | 0.21 s |
| audio 2.1 s / 5.8 s / 23.3 s, median of 3 | 37 / 62 / 186 ms | 39 / 62 / 173 ms |
| per audio second (2.1 s / 5.8 s / 23.3 s clip) | 18 / 11 / 8 ms | 19 / 11 / 7 ms |
| MLX peak after audio work (audio tower resident) | 2783 MB | 1713 MB |
| first video request incl. lazy vision-tower load | 0.75 s | 0.61 s |
| video 3 frames (398 tokens) | 539 ms | 507 ms |
| video 4 frames, 480x270 (490 tokens) | 645 ms | 601 ms |
| video 32 frames (4,226 tokens) | 6.1-6.2 s | 6.4 s |
| MLX peak after video work, fresh process (vision tower only) | 3427 MB | not run separately |
| MLX peak, both towers resident after the 32-frame clip | 4593 MB | 2339 MB |

The text-only profile is unchanged: before any media request the peak and RSS are
the Milestone-1 numbers above (1551 MB fp32 / 524 MB bf16), neither tower is loaded
(`isAudioTowerLoaded` / `isVisionTowerLoaded` are false after text requests, tested
with weights), and an audio request does not load the vision tower or the reverse.
The audio tower adds ~1.2 GB MLX peak (fp32; its weights are ~1.2 GB of the 3 GB
fp32 total, 300M params) and the vision tower ~0.9 GB before a video runs; the peak
is a high-water mark, so the 32-frame figures include that clip's activations.
RSS does not see MLX memory and moves around because the OS reclaims pages.

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


## Quantized builds (the ladder)

The full measured ladder (quality, speed, memory, size, and a comparison with Unsloth's
GGUFs and Ollama) is in [`docs/bench/embeddinggemma2-2026-10-08.md`](bench/embeddinggemma2-2026-10-08.md).
Every build below is a complete folder (safetensors with a `quantization` block in

All of these builds, alongside Google's original `google/embeddinggemma-2`, are collected on Hugging Face in the [EmbeddingGemma 2 MLX quantization ladder](https://huggingface.co/collections/srv-sngh/embeddinggemma-2-mlx-quantization-ladder-6ac7b2592bbdabdd74dc5ba7).
`config.json`, tokenizer, processor and the sentence-transformers files) that loads with
the same strict binder as the bf16 repo.

```
krill pull embeddinggemma-2-8bit          # 0.81 GB, affine 8-bit, group 64
krill pull embeddinggemma-2-6bit          # 0.62 GB
krill pull embeddinggemma-2-5bit          # 0.53 GB
krill pull embeddinggemma-2-4bit-dyn      # 0.47 GB, 4-bit with the sensitive text tensors at 8-bit
krill pull embeddinggemma-2-4bit-dyn-text # 0.51 GB, 4-bit with every text layer at 8-bit
krill pull embeddinggemma-2-4bit-g32      # 0.49 GB
krill pull embeddinggemma-2-mxfp8         # 1.01 GB, text layers + vision/audio attention dense
krill pull embeddinggemma-2-nvfp4         # 0.44 GB
krill pull embeddinggemma-2-6bit-dyn      # 0.66 GB (passes the gate, no measurable gain over 6bit)
```

| Build (alias `embeddinggemma-2-<name>`) | Bits | MB | Text cos mean | SciFact nDCG@10 (bf16: 86.92) | Hindi (bf16: 72.78) | Gate |
|---|---|---|---|---|---|---|
| `mxfp8` | 8 (mxfp8 mixed) | 1007 | 0.9977 | 86.80 | 72.58 | pass |
| `8bit` | 8 | 806 | 0.9999 | 86.95 | 72.81 | pass |
| `6bit` | 6 | 624 | 0.9988 | 86.98 | 72.81 | pass |
| `6bit-dyn` | 6 / 8 text | 657 | 0.9998 | 86.70 | 72.74 | pass |
| `5bit` | 5 | 533 | 0.9953 | 86.51 | 72.59 | pass |
| `4bit-dyn-text` | 4 / 8 text layers | 507 | 0.9980 | 85.32 | 72.38 | pass |
| `4bit-dyn` | 4 / 8 attn+ple+proj+last2 | 473 | 0.9927 | 86.30 | 72.03 | pass |
| `4bit-g32` | 4 (g32) | 488 | 0.9849 | 83.93 | 72.04 | pass |
| `nvfp4` | 4 (nvfp4) | 442 | 0.9809 | 84.43 | 72.44 | pass |
| `4bit` | 4 (g64) | 442 | 0.9806 | 83.03 | 71.42 | not published |
| `mxfp4` | 4 (mxfp4) | 419 | 0.9732 | 82.71 | 71.31 | not published |
| `3bit` | 3 | 351 | 0.9234 | 75.62 | 70.26 | not published |

The **publish gate is Claude's proposal**, not a decision of the maintainer: against bf16, the 8-bit class (mxfp8, 8bit,
6bit, 6bit-dyn) may lose at most 1.0 nDCG point on SciFact and on Hindi and 1.0 point of image Recall@1; the 4/5-bit class at most 3.0
points on each and needs text fidelity mean >= 0.97. Plain `4bit` (g64), `mxfp4` and `3bit` fail it and are not published
(`krill quantize` still builds them). The pass/fail decision uses small test sets (300 SciFact queries); read it as "no measurable
loss", not "equal".

A module is quantized exactly when `<module>.scales` exists in the checkpoint; its mode, bits and group come
from `config.quantization` (with per-path overrides), and a checkpoint and
config that disagree fail the load instead of producing plausible garbage. The
vision tower, the audio tower and the text model all load quantized leaves, at any
affine width (2, 3, 4, 5, 6, 8 bits) or `mxfp4` / `nvfp4` / `mxfp8`.
Compute is still fp32 by default (`KRILL_EMBED_DTYPE`).

### Dynamic builds: what is kept at 8-bit

`krill quantize --protect` quantizes matching modules at the protect precision (`--protect-bits`, default 8) and
`--skip` keeps them dense. Both take a module-path substring, or a regex when the pattern starts with `re:`
(needed to tell the text tower's `mlp.down_proj` from the vision tower's, which a substring cannot). The set was
chosen from a per-class sensitivity sweep (protect one class at 8-bit on top of a plain 4-bit build, measure the
fixture fidelity through the server; the tables are in the bench doc):

- the text tower is the only sensitive part at 4 bit; the 134M-parameter embedding table, vision, audio and the multimodal
  projectors gain nothing measurable from 8-bit and stay at 4-bit;
- best value per MB: the text projection (0.2 MB), the last two text layers, `ple_block`, `o_proj`, the global-attention layers;
- `4bit-dyn` protects the text projection, text `self_attn`, text `ple_block` and the last two layers (+31 MB);
  `4bit-dyn-text` protects the whole text stack (+65 MB); `6bit-dyn` is the same for 6 bit and is not worth it.

### Commands for every build

Run from the bf16 repo folder (`<bf16-dir>`); `--output-dir` writes a folder and does not register anything:

```
krill quantize <bf16-dir> --bits 8 --group-size 64 --dtype bf16 --output-dir embeddinggemma-2-mlx-8bit
krill quantize <bf16-dir> --bits 6 --group-size 64 --dtype bf16 --output-dir embeddinggemma-2-mlx-6bit
krill quantize <bf16-dir> --bits 5 --group-size 64 --dtype bf16 --output-dir embeddinggemma-2-mlx-5bit
krill quantize <bf16-dir> --bits 4 --group-size 32 --dtype bf16 --output-dir embeddinggemma-2-mlx-4bit-g32
krill quantize <bf16-dir> --mode nvfp4 --dtype bf16 --output-dir embeddinggemma-2-mlx-nvfp4
krill quantize <bf16-dir> --mode mxfp8 --dtype bf16 \
  --skip language_model.layers --skip self_attn --skip patch_embedder \
  --output-dir embeddinggemma-2-mlx-mixed-mxfp8

# dynamic 4-bit: sensitive text tensors at 8-bit
krill quantize <bf16-dir> --bits 4 --group-size 64 --dtype bf16 --protect-bits 8 --protect-group-size 64 \
  --protect 're:^language_model\.embedding_projection$' \
  --protect 're:^language_model\.layers\.\d+\.self_attn\.' \
  --protect 're:^language_model\.layers\.\d+\.ple_block\.' \
  --protect 're:^language_model\.layers\.2[23]\.' \
  --output-dir embeddinggemma-2-mlx-4bit-dyn

# dynamic 4-bit, every text layer at 8-bit
krill quantize <bf16-dir> --bits 4 --group-size 64 --dtype bf16 --protect-bits 8 --protect-group-size 64 \
  --protect 're:^language_model\.embedding_projection$' --protect 're:^language_model\.layers\.' \
  --output-dir embeddinggemma-2-mlx-4bit-dyn-text

# dynamic 6-bit (no measurable gain over 6bit)
krill quantize <bf16-dir> --bits 6 --group-size 64 --dtype bf16 --protect-bits 8 --protect-group-size 64 \
  --protect 're:^language_model\.embedding_projection$' --protect 're:^language_model\.layers\.' \
  --output-dir embeddinggemma-2-mlx-6bit-dyn
```

Built but not published (they fail the gate): `--bits 4 --group-size 64` (3.9 SciFact points lost), `--mode mxfp4`,
`--bits 3`.

### The experiment history (mxfp8 and nvfp4 recipes)

The two float-format recipes came from the 17-row experiment below.

### What is quantized, and why (measured)

Naive "quantize every 2-D weight" costs about 0.008 of cosine on every modality
(mxfp8: mean 0.992, worst text 0.974) because the error is spread over every
layer rather than concentrated in a few tensors, so the recipe is found by
keeping the most sensitive tensors dense. Every row below is the whole fixture
set (text 42 vectors = 21 strings raw + `Document`; image 8, mixed 12, audio 6,
video 4) through the release build over HTTP, cosine to the sentence-transformers
fp32 reference, `min / mean`. Text ranking: pairwise-similarity Spearman vs
Krill fp32 (raw / `Document`) and top-1 neighbour agreement (of 21).

| Variant (mxfp8 unless noted; "dense" = kept bf16; `--skip` substring match, so a bare name also hits the vision/audio tower copy) | text | image | mixed | audio | video | Spearman raw / Doc | top-1 raw / Doc | size |
|---|---|---|---|---|---|---|---|---|
| everything quantized (468 tensors) | .974 / .992 | .981 / .991 | .991 / .992 | .988 / .992 | .992 / .993 | .972 / .990 | 20 / 19 | 783 MB |
| dense `embed_tokens` | .978 / .993 | .980 / .992 | .991 / .992 | .989 / .992 | .992 / .993 | .975 / .990 | 20 / 19 | 913 MB |
| + dense `embedding_projection` | .979 / .994 | .982 / .993 | .992 / .994 | .990 / .994 | .993 / .995 | .977 / .992 | 20 / 19 | 915 MB |
| + dense PLE (per-layer input) | .983 / .995 | .983 / .994 | .993 / .995 | .991 / .994 | .993 / .994 | .981 / .993 | 20 / 19 | 933 MB |
| dense every `mlp` (text and vision) | .986 / .995 | .992 / .995 | .994 / .995 | .993 / .994 | .995 / .996 | .990 / .991 | 21 / 21 | 967 MB |
| dense `down_proj`, `o_proj`, projectors, patch/conv | .982 / .996 | .989 / .996 | .993 / .996 | .992 / .995 | .995 / .996 | .985 / .993 | 20 / 21 | - |
| dense whole vision tower | .974 / .992 | .995 / .995 | .991 / .993 | .988 / .992 | .992 / .993 | .972 / .990 | 20 / 19 | - |
| dense all text layers | .994 / .998 | .984 / .994 | .993 / .996 | .996 / .996 | .996 / .996 | .998 / .997 | 21 / 21 | - |
| dense text layers + whole vision tower | .994 / .998 | .997 / .998 | .997 / .998 | .996 / .996 | .998 / .998 | - | 21 / 21 | 1,056 MB |
| dense text layers + vision `down_proj`, `o_proj`, patch | .994 / .998 | .989 / .995 | .994 / .997 | .996 / .996 | .995 / .996 | - | 21 / 21 | 955 MB |
| **dense text layers + vision `self_attn` + patch embedder (shipped)** | **.994 / .998** | **.992 / .996** | **.996 / .997** | **.996 / .997** | **.996 / .997** | **.998 / .997** | **21 / 21** | **1,007 MB** |
| nvfp4, everything quantized (**shipped**) | .952 / .981 | .970 / .983 | .972 / .982 | .984 / .985 | .981 / .981 | .973 / .968 | 20 / 18 | 442 MB |
| nvfp4, dense `embed_tokens` | .936 / .981 | .970 / .983 | .973 / .982 | .984 / .985 | .981 / .981 | .976 / .971 | 20 / 18 | 635 MB |
| nvfp4, dense `embed_tokens` + `embedding_projection` | .942 / .985 | .974 / .986 | .976 / .985 | .987 / .989 | .983 / .985 | .981 / .973 | 20 / 18 | 637 MB |
| nvfp4 + mxfp8 on `down_proj`, `o_proj`, `embedding_projection`, patch embedder | .973 / .986 | .978 / .987 | .978 / .987 | .990 / .991 | .987 / .988 | .979 / .975 | 19 / 17 | 484 MB |
| nvfp4 + mxfp8 on all text layers, `embedding_projection`, patch embedder | .973 / .990 | .978 / .990 | .981 / .989 | .991 / .992 | .979 / .986 | .971 / .983 | 19 / 19 | 504 MB |

Targets: mxfp8 mean >= 0.995 and min >= 0.99 per modality; nvfp4 mean >= 0.97 and
min >= 0.95. Findings:

- **mxfp8 (E4M3 elements, power-of-two group scales) needs the text layers dense.**
  Text min only clears 0.99 once every text transformer layer is dense; the text
  MLP is the single most sensitive class, then attention, then the per-layer input
  block. `embed_tokens` is the largest tensor (262,144 x 512) and quantizing it
  is nearly free (dense `embed_tokens` moves text mean by 0.0004 for +130 MB).
- **The image tower is the other sensitive part**: quantized vision attention
  costs about 0.007 of image min cosine. Keeping `self_attn` and the patch
  embedder dense (and quantizing the vision MLP) is the cheapest way to reach
  image min >= 0.99; keeping only `down_proj`/`o_proj` dense missed it (0.989).
- **The audio tower tolerates mxfp8 fully quantized** (min 0.996 once the text
  layers are dense), so it is where most of the saving comes from.
- **nvfp4 (4-bit, group 16, fp8 scales) meets its target with every linear
  quantized**, which is the smallest recipe that does. Text min 0.952 is the thin
  margin: keeping `embed_tokens` or the projection dense does not help it
  (0.936 / 0.942), while protecting `down_proj`, `o_proj`, `embedding_projection`
  and the patch embedder at mxfp8 lifts text min to 0.973 for +42 MB (484 MB) but
  did not improve top-1 neighbour agreement (19/17 of 21 vs 20/18), so the plain
  build ships. If you need the extra headroom, use the protected command below.
- A `--skip` must hit the same module in every layer. Keeping layer 0 dense and
  quantizing the rest crashes the MLX leaf swap, so `krill quantize` refuses it.

### Parity of the shipped builds (cosine to the sentence-transformers fp32 reference)

| | mxfp8 min / mean | nvfp4 min / mean | Krill bf16/fp32 floor |
|---|---|---|---|
| text (42) | 0.9937 / 0.9977 | 0.9523 / 0.9809 | >= 0.9994 |
| image (8) | 0.9925 / 0.9961 | 0.9701 / 0.9828 | >= 0.9994 |
| mixed (12) | 0.9959 / 0.9970 | 0.9717 / 0.9817 | >= 0.9994 |
| audio (6) | 0.9963 / 0.9968 | 0.9839 / 0.9851 | >= 0.9994 |
| video (4) | 0.9955 / 0.9966 | 0.9806 / 0.9815 | >= 0.9994 |

Text ranking vs Krill fp32 (21 strings): mxfp8 Spearman 0.998 raw / 0.997 `Document`,
top-1 neighbour 21/21 both; nvfp4 Spearman 0.973 / 0.968, top-1 20/21 raw and
18/21 `Document`. nvfp4 is a retrieval-grade compromise, not a drop-in: expect
some neighbour reordering among near ties.

### Size, speed and memory

> **Superseded.** The earlier single-run timings in this section (bf16 / mxfp8 / nvfp4, measured on a loaded, shared
> Mac) have been replaced by a gated protocol (load, free memory, thermal and AC checks before every repetition; 3 repetitions with ranges; re-measured until spreads are <= 10%) and all contenders in
> [`docs/bench/embeddinggemma2-2026-10-08.md`](bench/embeddinggemma2-2026-10-08.md). The conclusion held: quantization saves
> disk and memory and changes time little (single-query p50 8.8-10.5 ms and 34-39 docs/s at batch 32 x ~256 tokens across the Krill ladder), because compute is fp32 and the model is small. Sizes below are the
> `model.safetensors` file: bf16 1,489 MB, mxfp8 1,007 MB, nvfp4 442 MB.

### Rebuilding the first two builds

`--output-dir` writes a complete folder (it does not register the model or touch
`~/.krill`); `--skip` keeps modules whose path contains the substring at full
precision. Run from the bf16 repo folder:

```
# mxfp8: text layers dense, vision attention + patch embedder dense, rest mxfp8
krill quantize <bf16-dir> --mode mxfp8 --dtype bf16 \
  --skip language_model.layers --skip self_attn --skip patch_embedder \
  --output-dir embeddinggemma-2-mlx-mixed-mxfp8

# nvfp4: every linear and the embedding table (group 16)
krill quantize <bf16-dir> --mode nvfp4 --dtype bf16 \
  --output-dir embeddinggemma-2-mlx-nvfp4

# nvfp4 with more headroom (484 MB; text min 0.973)
krill quantize <bf16-dir> --mode nvfp4 --dtype bf16 --protect-mode mxfp8 \
  --protect down_proj --protect o_proj --protect embedding_projection \
  --protect patch_embedder --output-dir embeddinggemma-2-mlx-nvfp4-protected
```

Note `--skip self_attn` also matches the audio conformer's
`audio_tower.layers.N.self_attn.*`, so audio attention stays dense in the shipped
mxfp8 build; the sizes and parity above include that. All sizes are the
`model.safetensors` file in decimal MB.

Weighted parity tests: `KRILL_EG2_MXFP8_DIR` / `KRILL_EG2_NVFP4_DIR` point
`EmbeddingGemma2QuantizedParityTests` at a built folder (skipped otherwise); they
gate at the targets above per modality.

---

# Milestone 2 design record: image (done in 2a), audio and video (done in 2b)

> **Status.** Image (2a) and audio + video (2b) are implemented and measured (see
> "Images and mixed input" and "Audio and video" above); the sections below are kept
> as the design record. What turned out wrong or incomplete in this
> design is listed here; the rest of the image section held up.
>
> - *"Audio capped at 280 soft tokens (min 280)"* is **wrong** for the real model path: the
>   reference does not cap (a 23.3 s clip gave 583 `<audio>` tokens; 25 tokens per second);
>   `audio_seq_length: 280` is only used by a serving-framework helper. What does limit
>   audio is the feature extractor's `max_length=480000` (30 s), which TRUNCATES silently;
>   Krill answers `400` for a longer clip instead. The "audio up to 11.2 s" figure in the
>   request-format section is therefore also wrong (it is 30 s).
> - *"Audio conv weights match Krill's `AudioEncoder`, structure and keys"*: structure and
>   keys yes (752 tensors, `output_proj` bias included), but the HF checkpoint stores the
>   conv weights in **PyTorch layout** (`[out,in,kH,kW]`, depthwise `[C,1,K]`) where the
>   Gemma 4 loader's weights are channel-last; the shapes quoted in the Audio section below
>   are the PyTorch ones, and the loader transposes them (strict on the result).
> - *Video: "decode with AVAssetImageGenerator"*: not used. Frames are decoded sequentially
>   with `AVAssetReader`; asking AVFoundation for BGRA costs ~0.005 cosine against the
>   torchcodec reference (chroma upsampling), so the decoder's 4:2:0 planes are converted
>   the swscale way instead. The 140-token budget, 1 fps, 32-frame uniform cap and "no
>   timestamps" held up; a 320x240 frame is 130 soft tokens.
> - *Video "Keys: none of its own"*: right; it shares `vision_tower` / `embed_vision`.
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
  scatter-merge plumbing as image. (Resolved in 2b: the reference truncates at 30 s, Krill refuses; was: decide behaviour for audio > 11.2 s (truncate
  or window-and-mean); not checked what the HF extractor does.)
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
size limits: image up to the 2520-patch budget, audio up to 30 s (was 11.2 s; decision
above), video up to 32 frames. `/api/embed` takes the same item shapes; the legacy
`/api/embeddings` stays text-only.

## Milestone 2 work order

All done: reference vectors per modality (2a, 2b), `forward(inputsEmbeds:)` and the
sequence builder (2a), image preprocessor + vision loader (2a), strict audio tower +
feature extractor + `input_audio` parts (2b), video frame sampler + decoder +
`video_url` parts (2b), all gated at fp32 cosine >= 0.999 and measured. The towers load
lazily and separately so text-only users keep the 0.8 s / 1.5 GB profile.

---

## Extension guide for audio/video

Milestone 2a deliberately built the shared plumbing so a new modality is a tower,
a preprocessor and a request part type. Nothing else changes. What already exists
(audio and video, added in 2b, used all of it; where they landed is listed below):

**Shared plumbing (do not duplicate)**

| Piece | Where |
|---|---|
| Marker / placeholder ids (`boa`, `audio`, `eoa`, `video`, `boi`, `eoi`) read from `config.json` | `EG2ModalityTokens` in `Sources/KrillCore/EmbeddingGemma2Sequence.swift` (`EmbeddingGemma2Config.modalityTokens`) |
| Layout of every modality's block: audio `<boa> <audio>xN <eoa>`, video one `<boi> <video>xN <eoi>` per frame | `EG2MediaBlock(.audio, softTokensPerBlock:)` / `EG2MediaBlock(.video, softTokensPerBlock:, blocks: frames)` and `EG2SequenceBuilder.build` (same file; already unit-tested for audio and video layouts) |
| Scatter into the text embeddings + count checks | `EmbeddingGemma2Model.mergedEmbeddings(_:features:)` where `features[.audio]` / `features[.video]` is one `[softTokens, 512]` array per block, in order (video: one array per frame, or one per video if you split by `blocks`: the builder emits one span per frame, so supply one `[n, 512]` per frame) |
| Backbone on merged embeddings, mean pool over every position | `EmbeddingGemma2Model.forward(inputsEmbeds:lengths:)`, `pooled(inputsEmbeds:lengths:)` |
| Lazy tower slot, 400 mapping, MRL / task handling, token accounting | `EmbeddingEngine.embedMediaItem` / `ensureVisionTower` in `Sources/KrillEngine/EmbeddingEngine.swift` |
| Request parsing (`input_audio` / `audio_url` / `video_url` / `input_video` now parsed) | `parsePart` in `Sources/KrillEngine/EmbeddingInputs.swift` |
| Strict loader pattern, clip-scalar rule | `loadEG2VisionTower` in `Sources/KrillCore/EmbeddingGemma2Vision.swift` |
| Reference vectors, ids and media for audio and video | `Tests/KrillEngineTests/Fixtures/eg2_mm/` (`reference_audio.json`, `reference_video.json`, `audio/`, `video/`, README) |

**Where audio and video landed (2b)**

| Piece | Where |
|---|---|
| Audio tower, strict loader (incl. the PyTorch conv-layout conversion), feature extractor, soft-token count, AVFoundation decoder | `Sources/KrillCore/EmbeddingGemma2Audio.swift` (`EG2AudioTower`, `loadEG2AudioTower`, `EG2AudioPreprocessor`) |
| Frame sampler, AVAssetReader decoder, swscale-matched YUV conversion | `Sources/KrillCore/EmbeddingGemma2Video.swift` (`EG2VideoSampler`, `EG2VideoSource`) |
| Request parts `input_audio` / `audio_url`, `video_url` / `input_video` | `EmbeddingPart.audio` / `.video`, `parsePart` in `Sources/KrillEngine/EmbeddingInputs.swift` |
| Lazy `ensureAudioTower()`, sizing before any tower work, 400 mapping | `embedMediaItem` in `Sources/KrillEngine/EmbeddingEngine.swift` |
| Weight-free tests | `Tests/KrillCoreTests/EmbeddingGemma2AudioTests.swift`, `EmbeddingGemma2VideoTests.swift`, `Tests/KrillEngineTests/EmbeddingInputsTests.swift` |
| Parity tests (skip without `KRILL_EG2_DIR`) | `Tests/KrillEngineTests/EmbeddingGemma2AudioParityTests.swift`, `EmbeddingGemma2VideoParityTests.swift` |
| Reference vectors, ids, frame thumbnails, media | `Tests/KrillEngineTests/Fixtures/eg2_mm/` (README: how each was made) |

Not done / not verified: bf16 audio / video floors are measured on the fixtures only;
10-bit / 4:2:2 video and rotated phone video were not tested; Ogg Vorbis, WebM and
Matroska do not decode; timestamps (`add_timestamps: true`) are not offered; audio
longer than 30 s is refused rather than windowed.
