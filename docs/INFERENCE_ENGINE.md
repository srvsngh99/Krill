# Inference Engine Internals

## InferenceEngine

`Sources/KLMEngine/InferenceEngine.swift` — the central orchestrator.

### State

| Property | Type | Purpose |
|----------|------|---------|
| `loadedModel` | `LoadedModel?` | Current model (module + forward closures) |
| `tokenizer` | `KLMTokenizer?` | Tokenizer for the loaded model |
| `modelDirectory` | `URL` | Path to model weights |
| `prefixCache` | `PrefixCache` | Shared prefix cache |
| `specDecoder` | `SpeculativeDecoder?` | Optional draft model |
| `_isSwapping` | `OSAllocatedUnfairLock<Bool>` | Swap-in-progress guard |
| `loadedAt` | `Date?` | When model was loaded |

### Model Lifecycle

```
load() -> load model + tokenizer from disk
swap(modelDirectory:) -> load new model, then replace (atomic)
unload() -> release model + tokenizer
loadDraftModel(from:) -> load speculative decoder
```

`swap()` loads the new model into temporaries first. If loading fails, the old model stays active.

### Generation Flow

```swift
generate(messages:, params:, maxTokens:, imageData:, audioData:)
  -> (AsyncStream<TokenEvent>, () -> GenerationStats?)
```

#### Step 1: Tokenize
- Gemma4: `formatGemma4TokenIds()` (direct token IDs, no round-trip)
- Others: `applyChatTemplate()` then `encodeWithoutExtraBOS()`
- Multimodal (CLI image only): prepend N copies of `<|image|>` based on `computeImageTokenCount()`
- Audio: handled by Python bridge in RunCommand before reaching InferenceEngine

#### Step 2: Create KV Caches
- `makeKVCaches(numLayers)` — one empty cache per layer

#### Step 3: Prefix Cache Lookup
- Full-hit only (prefixLength == promptTokens.count)
- Partial hits rejected (causal mask shape mismatch)
- On hit: restore all layer caches, truncate to length-1, forward last token only

#### Step 4: Prefill
```swift
if multimodal && imageData != nil {
    pixelValues = preprocessImage(imageData)
    logits = multimodalForward(tokens, caches, pixelValues, nil)
} else {
    logits = forward(tokens, caches)
}
MLX.eval(logits)
```

#### Step 5: Prefix Cache Store (write-behind)
- If tokens >= 8: snapshot all caches, async write to disk
- Never blocks generation

#### Step 6: Decode Loop

**Standard path:**
```
while generatedCount < maxTokens:
    if nextToken == EOS: break
    yield TokenEvent(tokenId, text, elapsed)
    logits = forward([nextToken], caches)
    nextToken = sampler.sample(logits)
```

**Speculative path:**
```
while generatedCount < maxTokens:
    accepted = specDecoder.step(lastToken, targetCaches, draftCaches)
    for token in accepted:
        yield TokenEvent(...)
    lastToken = accepted.last
```

### Stats

`GenerationStats` captures:
- `promptTokens`, `generatedTokens`
- `prefillTime`, `decodeTime`
- Derived: `prefillTokensPerSecond`, `decodeTokensPerSecond`, `ttft`, `totalTime`

## KV Cache

`Sources/KLMCache/KVCache.swift`

### Batched Concatenation Strategy

Per-token `concatenated()` is expensive (creates new array every step). Instead:

1. New K/V slices go into `_pendingKeys`/`_pendingValues` arrays
2. `update()` returns the full K/V by concatenating `_keys + pending` for attention
3. At 8 pending slices, `compact()` merges into `_keys`/`_values`

This reduces allocations by ~8x during decode.

### Layout
K/V arrays: `[B, numKVHeads, seqLen, headDim]` — sequence on axis 2.

## Prefix Cache

`Sources/KLMCache/PrefixCache.swift`

### Two-Tier Architecture

1. **Memory LRU** (default 8 entries): fast, no I/O
2. **Disk** (`~/.krill/cache/`): safetensors, persistent across restarts

### Cache Key
FNV-1a hash of `modelId + token_bytes`. Not cryptographic, just for keying.

### Lookup
Progressive from full prompt length, step down by `minPrefixLength/2`:
```
checkLen = promptTokens.count
while checkLen >= 8:
    check memory -> check disk
    checkLen -= step
```

### Write-Behind
Store happens in `Task.detached` — non-blocking on generation path.

## Speculative Decoding

`Sources/KLMEngine/SpeculativeDecoder.swift`

### Algorithm
1. Draft model generates K tokens greedily
2. Target model verifies all K in single batched forward
3. Accept up to first rejection, replace rejected with target's token
4. Roll back KV cache on rejection
5. Bonus token if all K accepted

### Adaptive K
- Tracks last 16 acceptance rates
- rate < 0.4 -> K-=1 (min 2)
- rate > 0.8 -> K+=1 (max 6)
- Default K=4

### Draft Pairs
```swift
"llama-3.1-8b": "llama-3.2-1b",
"qwen2.5-7b": "qwen2.5-0.5b",
"gemma-4-e4b": "gemma-2-2b",
```

## Sampler

`Sources/KLMSampler/Sampler.swift`

### Pipeline
```
logits -> temperature scaling -> top-K filter -> top-P filter -> softmax -> categorical sample
```

If temperature <= 0: greedy (argmax).

### Presets
- `.greedy`: temp=0
- `.creative`: temp=0.7, topP=0.9
