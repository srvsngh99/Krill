# Pitfalls and Lessons Learned

This document records bugs found during development, their root causes, and how to avoid repeating them. Each entry is a real bug that caused incorrect behavior in production.

## 1. Gemma4 KV Sharing: Don't Compute K/V for Shared Layers

**Bug**: Gemma4 native text generation produced gibberish.

**Root cause**: KV-shared layers (15-34) computed their own K/V projections and wrote to separate caches. The Python reference passes the donor's `(keys, values)` tuple directly and skips K/V computation entirely.

**Fix**: When `sharedCache` is provided, use the donor's K/V snapshot directly:
```swift
// WRONG: compute new K/V for shared layers
var newK = kProj(x)...
(k, v) = cache.update(keys: newK, values: newV)

// CORRECT: reuse donor's K/V
if let shared = sharedCache, let snap = shared.snapshot() {
    k = snap.keys
    v = snap.values
}
```

**How to avoid**: When implementing KV sharing for any model, verify against the reference: does the shared layer compute new K/V or reuse the donor's? For Gemma4, shared layers only compute Q.

**Files**: `Sources/KrillCore/Gemma4Model.swift`

---

## 2. Gemma4 Tokenizer: Special Token Round-Trip Loss

**Bug**: Gemma4 produced wrong output because special tokens (105, 106, 107) were corrupted.

**Root cause**: `applyChatTemplate()` built token IDs correctly (e.g., `[2, 105, 2364, 107, ...]`) then **decoded to text** and the engine **re-encoded from text**. The decode->encode round-trip turned single token 105 (`<|turn>`) into multiple tokens.

**Fix**: Return token IDs directly for Gemma4:
```swift
// WRONG: decode then re-encode
let text = tokenizer.decode(tokens: tokenIds)  // loses special tokens
let reEncoded = tokenizer.encode(text: text)    // different IDs!

// CORRECT: pass token IDs directly
public func formatGemma4TokenIds(messages:) -> [Int] {
    var tokens: [Int] = [2]  // BOS
    tokens.append(105)       // <|turn|>
    tokens += tokenizer.encode(text: role)
    // ...
    return tokens  // use directly, no round-trip
}
```

**How to avoid**: Any model with special tokens that don't survive decode->encode round-trips needs a direct token ID path. Check by decoding special tokens and re-encoding: `encode(decode([specialId])) == [specialId]`?

**Files**: `Sources/KrillTokenizer/TokenizerWrapper.swift`, `Sources/KrillEngine/InferenceEngine.swift`

---

## 3. Tied Embeddings: Don't Create Separate lm_head

**Bug**: Gemma4 output quality was wrong even with correct tokens and weights.

**Root cause**: Created a separate `lm_head` Linear and copied `embed_tokens` weights. But `QuantizedEmbedding.asLinear()` uses a different dequantization/matmul path than a standalone `QuantizedLinear`. The results diverge.

**Fix**: Use `embed_tokens.asLinear()` directly:
```swift
// WRONG: separate lm_head
@ModuleInfo(key: "lm_head") var lmHead: Linear
let logits = lmHead(hidden)

// CORRECT: tied embeddings
private func lmHead(_ hidden: MLXArray) -> MLXArray {
    model.embedTokens.asLinear(hidden)
}
```

**How to avoid**: Check if the Python reference has a separate `lm_head` or uses `embed_tokens.as_linear()`. If the checkpoint has no `lm_head.*` keys, it's tied.

**Files**: `Sources/KrillCore/Gemma4Model.swift`, `Sources/KrillCore/ModelLoader.swift`

---

## 4. GELU vs GELU Approximate

**Bug**: Numerical differences accumulated across 35 layers.

**Root cause**: Used `gelu()` (exact) but the reference uses `gelu_approx()` (tanh approximation). Over 35 layers, small per-activation differences compound.

**Fix**: Use `geluApproximate()` everywhere Gemma4 uses it:
```swift
// WRONG
downProj(gelu(gateProj(x)) * upProj(x))

// CORRECT
downProj(geluApproximate(gateProj(x)) * upProj(x))
```

**How to avoid**: Check the Python model's activation function. `nn.gelu_approx`, `nn.gelu`, `F.gelu(approximate='tanh')` are all different. Match exactly.

**Files**: `Sources/KrillCore/Gemma4Model.swift`

---

## 5. Vision Encoder: Architecture Must Match Safetensors Exactly

**Bug**: Native image inference crashed with shape mismatches.

**Root cause**: VisionEncoder was written speculatively without matching the actual checkpoint. Key mismatches:
- Patch embedding: Conv2d (wrong) vs Linear on flattened patches (correct)
- MLP: 2-layer fc1/fc2 (wrong) vs GeGLU gate/up/down (correct)
- Norms: 2 per block (wrong) vs 4 per block (correct)
- Hidden size: 1152 (wrong) vs 768 (correct)
- Bias: true (wrong) vs false (correct)
- Attention: plain Linear (wrong) vs ClippableLinear (correct)

**Fix**: Full rewrite matching safetensors key structure exactly.

**How to avoid**: Before implementing ANY encoder, dump the safetensors keys and shapes:
```python
arrays = mx.load("model.safetensors")
for k in sorted(arrays):
    if "vision" in k:
        print(f"{k}: {arrays[k].shape}")
```
Then design the Swift modules so `@ModuleInfo` keys produce identical paths.

**Files**: `Sources/KrillCore/VisionEncoder.swift`

---

## 6. Image Preprocessing: Channel Order and Row Flip

**Bug**: Vision encoder produced wrong embeddings.

**Root causes**:
1. Used NHWC format `[1, H, W, 3]` but model expects NCHW `[1, 3, H, W]`
2. CGContext stores pixels bottom-to-top; model expects top-to-bottom
3. Used wrong target size (672 instead of 768 for small images)
4. Used bfloat16 output but model expects float32 input

**Fix**: Channel-first with row flip:
```swift
// Channel-first with row flip
for row in 0 ..< newH {
    let flippedRow = newH - 1 - row  // CG bottom -> array top
    for col in 0 ..< newW {
        floats[dstIdx] = Float(ptr[srcIdx]) / 255.0           // R plane
        floats[pixelCount + dstIdx] = Float(ptr[srcIdx+1]) / 255.0  // G plane
        floats[2*pixelCount + dstIdx] = Float(ptr[srcIdx+2]) / 255.0  // B plane
    }
}
```

**How to avoid**: Check the Python processor's output shape and dtype. Print `processor(images=[img])['pixel_values'].shape` and `.dtype`.

**Files**: `Sources/KrillCore/VisionEncoder.swift`

---

## 7. Embedding Injection: Use masked_scatter, Not Positional Replace

**Bug**: Image embeddings were placed at wrong positions.

**Root cause**: Initial implementation put `replacement[i]` at position `i` in the sequence. But the correct behavior (masked_scatter) puts `replacement[0]` at the first mask-True position, `replacement[1]` at the second, etc.

**Fix**: Use cumsum-based masked_scatter:
```swift
let indices = MLX.cumsum(maskFlat, axis: 0) - 1
let aligned = sourceFlat.take(indices % sourceSize, axis: 0)
return MLX.where(maskFlat, aligned, inputTensor.flattened())
```

**How to avoid**: Check the Python model's `get_input_embeddings` method. Look for `masked_scatter` or equivalent.

**Files**: `Sources/KrillCore/Gemma4Model.swift`

---

## 8. Dynamic Image Token Count

**Bug**: Hardcoded 280 image tokens, but actual count depends on image size.

**Root cause**: `vision_soft_tokens_per_image=280` in config is the maximum, not the fixed count. A 256x256 image resized to 768x768 produces 256 tokens: `(768/16)^2 / 9 = 256`.

**Fix**: Compute token count from actual preprocessed image dimensions:
```swift
func computeImageTokenCount(imageData: Data) -> Int {
    let tensor = try preprocessImage(imageData)
    let pH = tensor.dim(2) / 16
    let pW = tensor.dim(3) / 16
    return (pH * pW) / (3 * 3)
}
```

**How to avoid**: Never hardcode token counts from config maximums. Compute from the actual preprocessed input.

**Files**: `Sources/KrillEngine/InferenceEngine.swift`

---

## 9. Prefix Cache Threshold Too High

**Bug**: Prefix cache never activated for benchmark prompts.

**Root cause**: Minimum prefix length was 32 tokens, but benchmark prompts were only 16 tokens. Repeated server requests paid full prefill every time.

**Fix**: Lowered threshold from 32 to 8.

**How to avoid**: Set cache thresholds based on expected workload. For server benchmarks with short prompts, 8 is reasonable.

**Files**: `Sources/KrillCache/PrefixCache.swift`, `Sources/KrillEngine/InferenceEngine.swift`

---

## 10. Server Streaming JSON: JSONSerialization is Expensive Per-Token

**Bug**: Server decode throughput 17% lower than CLI (105 vs 126 tok/s).

**Root cause**: `JSONSerialization.data(withJSONObject:)` called on every token event in the streaming hot path. Foundation JSON serialization has significant overhead for simple objects.

**Fix**: Direct string formatting for the per-token streaming path:
```swift
// WRONG: JSONSerialization per token
let chunk: [String: Any] = ["model": name, "response": text, "done": false]
let data = try! JSONSerialization.data(withJSONObject: chunk)

// CORRECT: direct string formatting
let escaped = escapeJSON(event.text)
let line = "{\"model\":\"\(name)\",\"response\":\"\(escaped)\",\"done\":false}\n"
```

**How to avoid**: Profile the hot path. For streaming, avoid Foundation JSON on every token.

**Files**: `Sources/KrillServer/Server.swift`

---

## 11. Benchmark Equivalence: Don't Compare Different Workloads

**Bug**: Server multimodal benchmark compared Krill text-only prompts against Ollama processing real images.

**Root cause**: `--krill-url` server path only sent text prompts to Krill but Ollama received base64-encoded images. Different work, invalid comparison.

**Fix**: Server benchmark skips image/audio tasks with explicit message.

**How to avoid**: Always verify that both engines receive equivalent inputs. Check prompt token counts: if one side has 20 tokens and the other has 277, the workloads are different.

**Files**: `tools/gemma4_multimodal_benchmark.py`

---

## 12. BPE Tokenizer: Grapheme Clusters Break Indic Text

**Bug**: EmbeddingGemma 2 embeddings of Hindi, Kannada and Sanskrit text had cosine 0.73-0.92 to the reference. English, French and code were fine (1.0), so the model looked correct.

**Root cause**: swift-transformers' BPE seeds its merges from Swift `Character`s, which are grapheme clusters. For Devanagari and Kannada a cluster is not a vocabulary entry, so the text fell into per-byte `<0xHH>` fallback: 2-3x too many tokens, and the wrong ones. HF `tokenizers` seeds from Unicode scalars.

**Fix**: `CodePointBPETokenizer` seeds from Unicode scalars. It supports only the Gemma `tokenizer.json` shape (Replace `" "->"▁"` normalizer, byte-fallback BPE, `<bos> A <eos>` template) and rejects anything else at load. A test pins its ids to HF `tokenizers` for 20 strings.

**How to avoid**: Test a new tokenizer with non-Latin scripts, not just English. Compare token ids against the HF tokenizer, not only the final vector.

**Files**: `Sources/KrillTokenizer/CodePointBPETokenizer.swift`, `docs/EMBEDDINGGEMMA2.md`

---

## 13. JPEG Decode: Apple ImageIO vs libjpeg Chroma Upsampling

**Bug**: EmbeddingGemma 2 image embeddings of JPEGs matched the reference at 0.9994-0.9998, while PNGs matched at >= 0.99996.

**Root cause**: Apple ImageIO and libjpeg (PIL) upsample chroma differently: mean 0.2-0.4 of 255, up to 40-66 levels on 3-6% of pixels at chroma edges. The model and resize were not the cause: feeding the PIL-decoded pixels of the same JPEGs to the server as PNG gave 0.999996 / 0.999995 / 0.999987 in fp32.

**Fix**: None. It is a property of the platform decoder and stays above the 0.999 gate. Real photos are smoother than the synthetic edges in the fixtures, so the gap should be smaller in practice.

**How to avoid**: When one input type is worse than the rest, decode with the reference's decoder and feed the pixels back in as a lossless format. That separates decoder error from model error.

**Files**: `Sources/KrillCore/EmbeddingGemma2Vision.swift`

---

## 14. Video Frames: AVFoundation BGRA vs torchcodec / swscale

**Bug**: EmbeddingGemma 2 video cosines were 0.9934-0.9986, below the other modalities.

**Root cause**: AVFoundation's own BGRA output upsamples chroma smoothly. The reference decoder (torchcodec / ffmpeg / swscale) replicates it. That alone gave mean 2.9 / 255 pixel error, with 5% of pixels off by more than 8.

**Fix**: Ask AVFoundation for the decoder's 8-bit 4:2:0 planes and convert them the swscale way: nearest chroma, the stream's matrix (BT.601 when untagged) and the stream's range. This measures mean 0.19 / 255, max 1, against torchcodec frames, and every video case is >= 0.99948. A weight-free test compares 12 frames of 4 fixtures with the reference decoder's frames.

**How to avoid**: Compare decoded pixels with the reference decoder before comparing embeddings. 10-bit, 4:2:2 and 4:4:4 sources are converted by VideoToolbox to 8-bit 4:2:0 first and were not measured.

**Files**: `Sources/KrillCore/EmbeddingGemma2Video.swift`

---

## 15. MP3 Decode: AVAudioFile Ignores the LAME Delay and Padding

**Bug**: An MP3 clip came out 1,532 samples (3 soft tokens) longer than the reference's, with cosine 0.9989, below the 0.999 gate.

**Root cause**: `AVAudioFile` removes the decoder delay (529 samples) but does not honour the encoder delay and padding in the LAME/Xing tag, which ffmpeg does.

**Fix**: Read the delay and padding from the first frame's Xing/Info tag and trim them. Without a tag nothing is trimmed, like ffmpeg. Checked by cross-correlating the two decoders: the lag equals the tag's delay.

**How to avoid**: Compare the decoded sample count with the reference decoder for every container format. A length difference changes the soft-token count, not just the values.

**Files**: `Sources/KrillCore/EmbeddingGemma2Audio.swift`

---

## 16. AAC Decode: AVFoundation Applies the Edit List, ffmpeg Keeps the Priming

**Bug**: An M4A (AAC) clip was 1 soft token shorter than the reference's (93,520 samples vs 94,208), so the token ids differ from the reference.

**Root cause**: AVFoundation applies the container's edit list; ffmpeg, which made the reference, keeps the priming samples.

**Fix**: None, by choice. The vector passes the gate (0.99952). Feeding the ffmpeg-decoded samples reaches the reference's ids and 0.99987, which shows the cause is the decoder and not the model path. In this one case AVFoundation is arguably the more correct decoder.

**How to avoid**: Do not read a token-count mismatch as a model bug until you have fed the reference's decoded samples through the same path.

**Files**: `Sources/KrillCore/EmbeddingGemma2Audio.swift`, `docs/EMBEDDINGGEMMA2.md`

---

## 17. Audio Conv Weights: PyTorch Layout vs Krill's Channel-Last Loader

**Bug**: Risk of a silent mis-load. The Gemma 4 audio loader assumes channel-last conv weights ("no transpose"), but the EmbeddingGemma 2 checkpoint stores them differently.

**Root cause**: The HF checkpoint keeps `subsample_conv_projection.*.conv.weight` as `[out,in,kH,kW]` and `lconv1d.depthwise_conv1d.weight` as `[C,1,K]` (PyTorch layout). The mlx-vlm weights the Gemma 4 loader sees are `[out,kH,kW,in]` and `[C,K,1]`.

**Fix**: `loadEG2AudioTower` converts each conv weight, and only when the transposed shape is exactly the module's. Binding is strict (all 752 tensors, nothing defaulted). A test runs the same tiny tower from both layouts.

**How to avoid**: Check the layout of every conv weight when reusing a module for a new checkpoint source. Keep the strict verify; `verify: []` would hide this.

**Files**: `Sources/KrillCore/EmbeddingGemma2Audio.swift`

---

## 18. Audio Length: The 280-Token Cap Does Not Exist

**Bug**: The design note said audio is capped at 280 soft tokens (about 11.2 s). The real model path has no such cap.

**Root cause**: `audio_seq_length: 280` in the processor config is only used by a serving-framework helper. The reference gives about 25 soft tokens per second (a 23.3 s clip gave 583). What limits audio is the feature extractor's `max_length=480000` (30 s), which silently truncates longer audio.

**Fix**: Krill uses the real token count and answers `400` for a clip over 30 s (480,000 samples at 16 kHz) instead of silently truncating. Clips under 0.1 s are also `400`.

**How to avoid**: Count tokens from a real reference run, not from a config constant. Pin the counts in a test (52 / 146 / 583 for the three WAV fixtures).

**Files**: `Sources/KrillCore/EmbeddingGemma2Audio.swift`, `docs/EMBEDDINGGEMMA2.md`

---

## 19. float16 Is Unsafe for EmbeddingGemma 2

**Bug**: Google's model card warns that EmbeddingGemma 2's activations exceed float16's range, so fp16 can return NaN or silently degraded vectors. (One sentence-transformers fp16 check here returned no NaN, so the failure is input-dependent, not guaranteed.)

**Root cause**: activation range beyond float16's dynamic range (model card); see the comments in `EmbeddingGemma2Model.swift` and `EmbeddingEngine.swift`.

**Fix**: Compute dtype is float32 (default) or bfloat16 only. `KRILL_EMBED_DTYPE` honours only those two (any other value, float16 included, is ignored and fp32 is used), `setComputeDtype` rejects anything else, and NaN / Inf output is never returned: the request fails with `500`.

**How to avoid**: Other embedders here handle fp16 differently (an fp32 upcast of `embed_tokens` for the Mistral-backbone ones, for example), so do not copy a dtype choice between embedding models without checking parity.

**Files**: `Sources/KrillCore/EmbeddingGemma2Model.swift`, `Sources/KrillEngine/EmbeddingEngine.swift`

---

## General Debugging Strategy for Model Output Issues

When a model produces gibberish:

1. **Check tokenizer**: Are the token IDs correct? Compare `tokens` array with Python reference.
2. **Check embeddings**: Does `embed_tokens(token_id)` produce the same values?
3. **Check with/without cache**: Python models may require KV cache even for prefill.
4. **Check layer output incrementally**: Compare hidden state after each layer.
5. **Check activation functions**: `gelu` vs `gelu_approx` vs `silu` matter.
6. **Check norm behavior**: `RMSNorm` with vs without +1 offset, parameter-free variants.
7. **Check weight loading**: Does `model.update(parameters:, verify: [])` silently skip mismatched keys?
8. **Check the reference call path**: The Python `model(input_ids)` may do things differently from calling layers manually.
