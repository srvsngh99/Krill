# Gemma 4 Internals

Gemma 4 is the most complex model family in KrillLM. This document explains every architectural detail needed to understand, debug, and extend the implementation.

## Architecture Overview

Gemma 4 E2B (2B parameters) is a multimodal model with:
- 35 transformer layers with mixed attention types
- Per-Layer Embedding (PLE) gating
- KV sharing between layers 15-34
- Native vision encoder (SigLIP2)
- Audio support via Python bridge (conformer rewrite pending)

## Files

| File | What it contains |
|------|-----------------|
| `Sources/KLMCore/Gemma4Model.swift` | Text model, attention, MLP, PLE, multimodal wrapper |
| `Sources/KLMCore/VisionEncoder.swift` | SigLIP2 vision encoder, image preprocessing |
| `Sources/KLMCore/AudioEncoder.swift` | Conformer audio encoder, WAV/mel preprocessing |
| `Sources/KLMCore/ModelLoader.swift` | Weight loading, quantization, tied embeddings |
| `Sources/KLMEngine/InferenceEngine.swift` | Multimodal generate path |
| `Sources/KLMTokenizer/TokenizerWrapper.swift` | Gemma4 chat template (direct token IDs) |

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
audio_tower.*                                -> BF16 (not loaded into Swift modules yet)
embed_audio.*                                -> QuantizedLinear (4-bit, not loaded yet)
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
