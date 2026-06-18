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

**Files**: `Sources/KLMCore/Gemma4Model.swift`

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

**Files**: `Sources/KLMTokenizer/TokenizerWrapper.swift`, `Sources/KLMEngine/InferenceEngine.swift`

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

**Files**: `Sources/KLMCore/Gemma4Model.swift`, `Sources/KLMCore/ModelLoader.swift`

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

**Files**: `Sources/KLMCore/Gemma4Model.swift`

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

**Files**: `Sources/KLMCore/VisionEncoder.swift`

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

**Files**: `Sources/KLMCore/VisionEncoder.swift`

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

**Files**: `Sources/KLMCore/Gemma4Model.swift`

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

**Files**: `Sources/KLMEngine/InferenceEngine.swift`

---

## 9. Prefix Cache Threshold Too High

**Bug**: Prefix cache never activated for benchmark prompts.

**Root cause**: Minimum prefix length was 32 tokens, but benchmark prompts were only 16 tokens. Repeated server requests paid full prefill every time.

**Fix**: Lowered threshold from 32 to 8.

**How to avoid**: Set cache thresholds based on expected workload. For server benchmarks with short prompts, 8 is reasonable.

**Files**: `Sources/KLMCache/PrefixCache.swift`, `Sources/KLMEngine/InferenceEngine.swift`

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

**Files**: `Sources/KLMServer/Server.swift`

---

## 11. Benchmark Equivalence: Don't Compare Different Workloads

**Bug**: Server multimodal benchmark compared Krill text-only prompts against Ollama processing real images.

**Root cause**: `--krill-url` server path only sent text prompts to Krill but Ollama received base64-encoded images. Different work, invalid comparison.

**Fix**: Server benchmark skips image/audio tasks with explicit message.

**How to avoid**: Always verify that both engines receive equivalent inputs. Check prompt token counts: if one side has 20 tokens and the other has 277, the workloads are different.

**Files**: `tools/gemma4_multimodal_benchmark.py`

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
