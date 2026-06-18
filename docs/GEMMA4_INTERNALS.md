# Gemma 4 Internals

Gemma 4 is the most complex model family in Krill. This document explains every architectural detail needed to understand, debug, and extend the implementation.

## Architecture Overview

Gemma 4 E2B (2B parameters) is a multimodal model with:
- 35 transformer layers with mixed attention types
- Per-Layer Embedding (PLE) gating
- KV sharing between layers 15-34
- Text: native Swift on both CLI and server
- Image: native Swift via SigLIP2 vision encoder (supported on both CLI and server)
- Audio: routed through the `mlx-vlm` Python bridge (CLI and server both supported; native conformer rewrite pending). When audio is combined with an image, the entire request goes through `mlx-vlm`.
- Server API: text on every endpoint; image (native) and audio (bridge) on `/api/generate`, `/api/chat`, and `/v1/chat/completions`. `/v1/completions` remains text-only to match upstream OpenAI semantics.

This is a release-readiness baseline, not a production release. See [`RELEASE_READINESS_REMEDIATION.md`](RELEASE_READINESS_REMEDIATION.md) for status.

## Files

| File | What it contains |
|------|-----------------|
| `Sources/KrillCore/Gemma4Model.swift` | Text model, attention, MLP, PLE, multimodal wrapper |
| `Sources/KrillCore/VisionEncoder.swift` | SigLIP2 vision encoder, image preprocessing |
| `Sources/KrillCore/AudioEncoder.swift` | Conformer audio encoder, WAV/mel preprocessing |
| `Sources/KrillCore/ModelLoader.swift` | Weight loading, quantization, tied embeddings |
| `Sources/KrillEngine/InferenceEngine.swift` | Multimodal generate path |
| `Sources/KrillTokenizer/TokenizerWrapper.swift` | Gemma4 chat template (direct token IDs) |

## Config (Gemma4Config)

From `config.json` (text_config section):

| Field | Value (E2B) | Purpose |
|-------|-------------|---------|
| hiddenSize | 1536 | Main hidden dimension |
| numHiddenLayers | 35 | Total transformer layers |
| numAttentionHeads | 8 | Query heads |
| numKeyValueHeads | 1 | KV heads (GQA) |
| headDim | 256 | Sliding layer head dimension |
| globalHeadDim | 512 | Full attention layer head dimension |
| slidingWindow | 512 | Window size for sliding layers |
| slidingWindowPattern | 5 | Every 5th layer is full attention |
| hiddenSizePerLayerInput | 256 | PLE dimension |
| numKVSharedLayers | 20 | Layers 15-34 share KV from donors |
| finalLogitSoftcapping | 30.0 | Logit capping value |
| vocabSize | 262144 | Vocabulary size |

## Layer Types

Layers alternate between sliding window and full attention:
- Sliding: layers 0-3, 5-8, 10-13, 15-18, 20-23, 25-28, 30-33
- Full: layers 4, 9, 14, 19, 24, 29, 34

**Sliding-window masking (IMPLEMENTED on the solo path).** A `sliding_attention`
layer attends only the last `slidingWindow` (512) keys; a `full_attention` layer
attends the whole causal context. The solo forward (`callAsFunction`) builds both
a plain causal mask and a `createSlidingWindowCausalMask` and selects per layer
via `isFullAttention(layerIdx:)`. This is REQUIRED for correctness: the sliding
layers are trained on a 512-token window, so feeding them the full context at long
prompts is out-of-distribution and collapses generation to an immediate stop
(empty output past ~2x the window). For prompts shorter than the window the two
masks are identical, so short requests are byte-identical to the non-windowed
path. The concurrent batched-decode path does NOT yet apply the window (follow-up).

```swift
config.isFullAttention(layerIdx: 4)  // true
config.isFullAttention(layerIdx: 3)  // false
```

## Attention (Gemma4Attention)

### Sliding Layers (headDim=256)
- Standard RoPE with base=10000
- All 256 dims rotated
- Q norm (RMSNorm) + K norm (RMSNorm) + V norm (parameter-free RMSNorm)
- Scale = 1.0 (not 1/sqrt(d))

### Full Attention Layers (headDim=512)
- ProportionalRoPE with base=1e6
- Only 128 of 512 dims rotated (split-half: 64 from left, 64 from right)
- Adjusted base: `pow(1e6, 128/512) = 31.623`
- Same norms and scale

### KV Sharing (CRITICAL)

**Layers 15-34 do NOT compute their own K/V projections.** They reuse the donor layer's K/V directly:

```swift
if let shared = sharedCache, let sharedSnap = shared.snapshot() {
    // Reuse donor's K/V — do NOT compute new projections
    k = sharedSnap.keys
    v = sharedSnap.values
}
```

Donors:
- Sliding shared layer -> last non-shared sliding layer (layer 14 for sliding)
- Full shared layer -> last non-shared full layer (layer 14 for full)

**Shared-layer query RoPE offset.** A shared layer keeps an EMPTY own cache (it
never writes K/V), so a naive `offset = cache.sequenceLength` is 0. That rotates
this forward's first query token at position 0 — correct only when the span
actually starts at position 0:
- **Cold full prefill** (span `[0, N)`): offset 0 == true positions. ✓
- **Single-token decode / full-MATCH 1-token re-forward** (`L == 1`): legacy
  offset-0 path, left as-is. ✓ (gated by the int8 full-match replay test)
- **Partial-prefix RESUME** (`L > 1`, donor already holds the prefix): the span
  is the diverging SUFFIX at true positions `[LCP, count)`. Offset 0 would
  rotate it at `0..L-1`, misaligned with the donor's restored K (rotated at
  their true positions) → RoPE mismatch → divergent output. The shared layer
  instead derives base `donorLen - L` from the donor's POST-update length (the
  donor ran earlier this forward and appended the same L-token span), giving
  `LCP`. So the suffix Q lands at its true positions and the resume is bit-exact
  to a cold prefill. See `Gemma4Attention.callAsFunction` and
  `Gemma4PartialReuseLiveTests`.

**BUG HISTORY**: Previously, shared layers computed their own K/V and wrote to separate caches. This produced wrong attention context for 20 of 35 layers, causing gibberish output. Fixed by reusing donor's K/V snapshot directly.

## Per-Layer Embedding (PLE)

PLE adds per-layer learned embeddings to the hidden state at each layer:

```
1. pleEmbed = embedPerLayer(tokens) * sqrt(256)         -> [B, L, 35*256]
2. projection = perLayerProj(h) * (1/sqrt(1536))        -> [B, L, 35*256]
3. projNormed = perLayerNorm(reshape to [*, 256])        -> [B, L, 35*256]
4. combined = (projNormed + pleEmbed) * (1/sqrt(2))      -> [B, L, 35*256]
5. Per layer: slice [i*256:(i+1)*256] -> gate -> project -> norm -> residual add
```

The PLE gate uses `geluApproximate` (not `gelu` — this matters for numerical accuracy).

**Multimodal PLE**: Image/audio token positions are zeroed in the PLE input to avoid corrupting per-layer embeddings with placeholder token IDs.

## Block Forward (Gemma4Block)

Each block has 4 norms (vs the standard 2):

```
h = x + postAttnNorm(selfAttn(inputLayernorm(x)))     // attention with post-norm
h = h + postFfnNorm(mlp(preFfnNorm(h)))                // FFN with pre+post norm
h = h + pleNorm(pleProj(geluApproximate(pleGate(h)) * ple))  // PLE gating
h = h * layerScalar                                     // per-layer scaling
```

## MLP (Gemma4MLP)

GeGLU activation:
```swift
downProj(geluApproximate(gateProj(x)) * upProj(x))
```

KV-shared layers use double-wide MLP (2x intermediate size).

## LM Head (Tied Embeddings)

Gemma 4 does NOT have a separate `lm_head` weight. It reuses `embed_tokens.asLinear()`:

```swift
private func lmHead(_ hidden: MLXArray) -> MLXArray {
    model.embedTokens.asLinear(hidden)
}
```

**BUG HISTORY**: Previously created a separate `lm_head` Linear and copied embed_tokens weights. This produced different results from `QuantizedEmbedding.asLinear()` due to different dequantization paths.

## Logit Softcapping

```swift
tanh(logits / 30.0) * 30.0
```

Clamps logits to [-30, 30] range, preventing extreme values.

## Vision Encoder (SigLIP2)

### Pipeline
```
Image (PNG/JPEG)
  -> preprocessImage() -> [1, 3, H, W] in [0,1] (channel-first)
  -> VisionPatchEmbedder: patchify + position embeddings -> [1, numPatches, 768]
  -> VisionTransformerModel: 16 layers of ClippableLinear attention + GeGLU MLP
  -> VisionPooler: position-aware average pooling -> [1, ~256, 768]
  -> MultimodalEmbedder: RMSNormNoScale + 4-bit projection -> [1, ~256, 1536]
  -> maskedScatter into token embeddings at <|image|> positions
```

### Image Preprocessing (preprocessImage)
1. Decode PNG/JPEG via CoreGraphics
2. Resize: minimum 768px for small images, longest side to 672px for large
3. Pad to nearest multiple of 48 (patchSize * poolingKernel)
4. CGContext renders with white background (row flip for top-to-bottom)
5. Extract RGB channel-first [1, 3, H, W] in [0, 1] float32

### ClippableLinear
All vision/audio attention projections use ClippableLinear:
```swift
func callAsFunction(_ x: MLXArray) -> MLXArray {
    var h = MLX.clip(x, min: inputMin, max: inputMax)
    h = linear(h)
    return MLX.clip(h, min: outputMin, max: outputMax)
}
```

Weight key structure: `xxx.linear.weight`, `xxx.input_min`, `xxx.input_max`, etc.

### 2D Multidimensional RoPE
Vision attention uses 2D RoPE on patch grid coordinates:
- Splits headDim=64 into 2x32 (one per spatial axis)
- Independent rotate_half per axis
- Base frequency = 100.0

### Embedding Injection (maskedScatter)
```swift
// cumsum-based scatter: source[0] -> first True mask position, etc.
let indices = MLX.cumsum(maskFlat, axis: 0) - 1
let aligned = sourceFlat.take(indices % sourceSize, axis: 0)
return MLX.where(maskFlat, aligned, inputTensor.flattened())
```

## Audio Encoder (USM Conformer)

Gemma 4 E2B audio is a Universal Speech Model (USM) Conformer. Discovered
from the local checkpoint (`config.json` `audio_config`, `model.safetensors`
`audio_tower.*` / `embed_audio.*`) and cross-checked against the `mlx-vlm`
reference (`mlx_vlm/models/gemma4/audio.py`,
`audio_feature_extractor.py`), which is the correctness oracle.

### audio_config (checkpoint defaults)
```
hidden_size            1024     num_hidden_layers        12
num_attention_heads    8        head_dim                 128 (=1024/8)
subsampling_conv_channels [128, 32]   conv_kernel_size    5
attention_chunk_size   12       attention_context_left   13
attention_context_right 0       attention_logit_cap      50.0
attention_invalid_logits_value -1e9
residual_weight        0.5      rms_norm_eps             1e-6
gradient_clipping      1e10     output_proj_dims         1536
hidden_act             silu     use_clipped_linears      true
```
Tokens: `audio_token_id` 258881, `boa_token_id` 256000,
`eoa_token_id`/`eoa_token_index` 258883, `image_token_id` 258880.

### Preprocessing (Gemma4AudioFeatureExtractor defaults)
The HF/local checkpoint ships **no** `feature_extractor` config, so all
USM defaults apply (fixed for every Gemma 4 model):
```
feature_size (mel bins)  128      sampling_rate       16000
frame_length_ms 20 -> 320 samples hop_length_ms 10 -> 160 samples
fft_length 2^ceil(log2(320)) = 512 num_freq_bins  257 (=512/2+1)
min_freq 0.0  max_freq 8000.0      mel_scale "htk", norm None
preemphasis 0.0 (=> frame = unfold[..., :-1], size 320)
window: periodic Hann, w[n]=0.5-0.5*cos(2*pi*n/320)
mel_floor 1e-3   log_mel = log(|rfft| @ mel_filters + 1e-3)
per_bin_mean/stddev None (no normalization)
semicausal left-pad = frame_length//2 = 160 zeros
frame_size_for_unfold = frame_length + 1 = 321, step = hop = 160
max_length 480000 samples (30 s), pad_to_multiple_of 128, truncation on
mask frame end index = arange(T)*hop + frame_size_for_unfold - 1
padded spectrogram positions are zeroed
```
Soft-token count: `ceil(duration_ms / 40)` capped at `audio_seq_length`
750; prompt expands to `boa + <|audio|>*N + eoa`. The 2x stride-2 conv
subsampling turns 10 ms-hop frames (100/s) into ~25/s = 40 ms/token, so
the tower output length matches N by construction.

### Audio tower (`audio_tower.*`, BF16)
- `subsample_conv_projection`: `layer0` Conv2d [128,3,3,1] +
  LayerNorm(128) + ReLU, `layer1` Conv2d [32,3,3,128] + LayerNorm(32) +
  ReLU (both stride 2, symmetric pad (1,1,1,1)); flatten F*C then
  `input_proj_linear` [1024,1024]. Mask downsampled by time stride.
- 12x `layers.N` Conformer block (macaron):
  `feed_forward1` -> attention -> `lconv1d` -> `feed_forward2` ->
  clip -> `norm_out`.
  - `feed_forwardX`: `pre_layer_norm` (AudioRMSNorm) ->
    `ffw_layer_1` ClippableLinear [4096,1024] -> SiLU ->
    `ffw_layer_2` ClippableLinear [1024,4096] -> clip ->
    `post_layer_norm`; out = residual + x*`residual_weight`(0.5).
  - `self_attn`: `q/k/v_proj` ClippableLinear [1024,1024],
    `relative_k_proj` [1024,1024], `per_dim_scale` [128], `post`
    ClippableLinear [1024,1024]. Chunked local attention (chunk 12,
    past 12, future 0, context 24) with sinusoidal relative-position
    bias, logit softcap `tanh(l/50)*50`, validity+causal mask.
    `q_scale = head_dim^-0.5 / ln2`, `k_scale = ln(1+e)/ln2`,
    `per_dim_scale` via softplus.
  - `lconv1d`: `pre_layer_norm` -> `linear_start` ClippableLinear
    [2048,1024] -> GLU -> causal `depthwise_conv1d` [1024,5,1]
    (groups=1024) -> clip -> `conv_norm` -> SiLU -> `linear_end`
    ClippableLinear [1024,1024] + residual.
- `output_proj`: Linear with bias, [1536,1024] -> 1536.

### embed_audio (`embed_audio.*`, 4-bit quantized)
`MultimodalEmbedder`: RMSNormNoScale(1536, eps 1e-6) then
`embedding_projection` Linear 1536->1536 (weight U32 [1536,192] +
scales/biases [1536,24], group_size 64, bits 4). Identical structure to
`embed_vision`; output scattered into `<|audio|>` (258881) positions via
the shared `maskedScatter`.

### Native vs bridge
Native Swift+MLX path lands in `Sources/KrillCore/AudioPreprocessor.swift`
(frontend) + `Sources/KrillCore/AudioEncoder.swift` (tower, rewritten from
the placeholder). The `mlx-vlm` bridge stays as fallback/oracle behind
`KRILL_AUDIO_BRIDGE_ONLY=1`; native is gated by `KRILL_NATIVE_AUDIO`.

## Tokenizer (Gemma4 Special Tokens)

| Token | ID | Purpose |
|-------|-----|---------|
| `<bos>` | 2 | Beginning of sequence |
| `<|turn>` | 105 | Turn start marker |
| `<turn\|>` | 106 | Turn end marker |
| `\n` | 107 | Newline |
| `<\|image\|>` | 258880 | Image placeholder |
| `<\|audio\|>` | 258881 | Audio placeholder |

**CRITICAL**: Chat template must use direct token IDs via `formatGemma4TokenIds()`, NOT text encoding. The text `<|turn>` encodes to multiple tokens instead of the single ID 105.

**BUG HISTORY**: `applyChatTemplate` decoded token IDs to text then re-encoded, losing special token identity. Fixed by returning token IDs directly for Gemma4.

## Weight Loading (ModelLoader)

### Safetensors Key Structure
```
language_model.model.embed_tokens.*          -> QuantizedEmbedding
language_model.model.layers.N.*              -> QuantizedLinear (except PLE)
language_model.model.per_layer_model_projection.* -> Linear (BF16, not quantized)
vision_tower.patch_embedder.*                -> Linear + position table
vision_tower.encoder.layers.N.*             -> ClippableLinear (BF16)
embed_vision.embedding_projection.*          -> QuantizedLinear (4-bit)
audio_tower.*                                -> BF16 (native USM Conformer, see Audio Encoder)
embed_audio.embedding_projection.*           -> QuantizedLinear (4-bit)
```

### Quantization Filter
Language model layers are quantized (4-bit, group_size=64) EXCEPT:
- `per_layer_model_projection` (stays BF16)
- `per_layer_projection_norm` (stays BF16)
- Vision tower (stays BF16)
- Audio tower (stays BF16)
- `embed_vision.embedding_projection` and `embed_audio.embedding_projection` (quantized)

### Conv Weight Sanitization
- Conv2d: PyTorch [out, in, kH, kW] -> MLX [out, kH, kW, in]
- Conv1d: PyTorch [out, in, kW] -> MLX [out, kW, in]

## Performance Characteristics

| Metric | Value (M4 Pro 24GB, 4-bit) |
|--------|---------------------------|
| Text decode | ~125 tok/s |
| Text prefill | ~390 tok/s |
| Image + text TTFT | ~4s (includes vision encoder) |
| Model load | ~1.1s |
| Peak memory (text) | ~3.6 GB |
| Peak memory (image) | ~4.4 GB |
