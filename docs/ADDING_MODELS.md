# Adding New Model Families

> Adding a model of an **existing** family needs no code change at all:
> put it in the model catalog (`krill catalog`, see
> [MODEL_CATALOG.md](MODEL_CATALOG.md)). The steps below are for adding
> a new model *architecture* / `ModelFamily`.

## Steps

### 1. Create the model file

Add `Sources/KrillCore/NewModel.swift` with:

```swift
// Config struct conforming to ModelConfig
public struct NewConfig: Decodable, Sendable, ModelConfig {
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let vocabSize: Int
    public let quantization: QuantizationConfig?
    // ... model-specific fields
}

// Model classes
class NewAttention: Module { ... }
class NewMLP: Module { ... }
class NewBlock: Module { ... }
class NewModelInner: Module { ... }  // embed + layers + norm
public class NewForCausalLM: Module { ... }  // model + lm_head
```

### 2. Match weight keys exactly

Dump the safetensors keys:
```python
import mlx.core as mx
arrays = mx.load("model.safetensors")
for k in sorted(arrays):
    print(f"{k}: {arrays[k].shape} {arrays[k].dtype}")
```

Your `@ModuleInfo(key:)` annotations must produce paths that match these keys. For example, if the safetensors has `model.layers.0.self_attn.q_proj.weight`, your code needs:

```swift
class NewForCausalLM: Module {
    @ModuleInfo(key: "model") var model: NewModelInner
    // produces: model.layers.0.self_attn.q_proj.weight
}
```

### 3. Add loader in ModelLoader.swift

```swift
private func loadNew(configData: Data, directory: URL) throws -> LoadedModel {
    let config = try JSONDecoder().decode(NewConfig.self, from: configData)
    let model = NewForCausalLM(config)
    try loadWeights(into: model, from: directory, quantization: config.quantization)

    return LoadedModel(
        module: model,
        numLayers: config.numHiddenLayers,
        family: "new",
        forward: { tokens, caches in model(tokens, caches: caches) },
        multimodalForward: nil,
        vocabSize: config.vocabSize
    )
}
```

### 4. Add detection in loadModel()

```swift
} else if arch.contains("new") || modelType == "new" {
    return try loadNew(configData: configData, directory: directory)
}
```

Detection order matters: check specific patterns before generic ones.

### 5. Add chat template (if needed)

If the model uses special chat tokens that don't survive decode->encode, add a direct token ID path in `TokenizerWrapper.swift`.

### 6. Add draft pair (if applicable)

```swift
// In SpeculativeDecoder.swift
"new-7b": "new-1b",
```

### 7. Declare the family's runtime adapter (new family only)

Adding a whole new `ModelFamily` (not just an alias of an existing
one) touches more of the registry than it looks like. PR #315
(`ModelFamily.prismHadamardQwen35`) is the reference: verify against
`git show 32e5f0d` rather than trusting this list, because it is easy
to under-scope this step. Five files, all exhaustively `switch
family` (or a `ModelFamily` lookup), so most of these will not
compile until every case is given a value - but two of them
(`ModelManifest.swift`'s `detect`/`fromModelType` and
`ModelProfiles.swift`) are plain functions that silently do the wrong
thing instead of failing to build, so the compiler will not catch a
skipped one:

- `ModelManifest.swift`: the `ModelFamily` enum case itself, an arm in
  `detect(from:)`'s arch-substring chain (ordered before any generic
  arm whose substring could also match), and a case in
  `fromModelType`.
- `ModelCapabilities.swift`: a `capabilities(for:)` case and a
  `supportTier(for:)` case.
- `ModelAdapter.swift`: **five** exhaustive switches, not three -
  `chatRouting`, `requiresImageInput`, `chatTemplate`,
  `tokenizerPrompt`, and `kvCacheQuantization` all need a value.
  `chatRouting` is `.denseEngine` for every native Swift+MLX family
  today (the MoE-sidecar case this doc used to point at was deleted
  once the last MoE family went native); `kvCacheQuantization` is
  `.fp16Only` unless the family's forward closure genuinely accepts
  `[QuantizedKVCache]` (only Gemma 4 does today).
- `ModelProfiles.swift`: a `profile(for family:)` case, so `/model`'s
  deep-dive does not silently fall through to `nil` (no curated
  story) for the new family.

**The trap that will not show up in any test.**
`InferenceEngine.capabilities` (`Sources/KrillEngine/InferenceEngine.swift`,
around line 151) does `ModelFamily(rawValue: loaded.family)` and, when
that lookup FAILS, silently returns an EMPTY capability set - no
error, no crash, nothing. `loaded.family` is the plain string the
loader sets on `LoadedModel` (e.g. `family: "prism_hadamard_qwen35"`
in `loadPrismHadamardQwen35`), matched against the new `ModelFamily`
case's `rawValue`, NOT its Swift case name. These two strings are a
contract: if they do not match exactly, the checkpoint loads fine,
every load-time test passes, and the server then refuses even plain
text generation, with nothing obviously broken anywhere - the failure
surfaces as "the model will not chat," several layers away from the
actual bug. This is precisely the mistake #315 nearly made: reusing
`.qwen35` (`rawValue == "qwen3_5"`) for a loader that sets `family:
"prism_hadamard_qwen35"` would have left every request to that family
silently capability-less. Give the new case a `rawValue` string that
is IDENTICAL to the literal the loader passes to `LoadedModel(family:)`,
and grep for that literal to confirm the two actually agree.

The server's `dispatchFamilyChat` and `ToolFormat.forFamily` then
pick the family up automatically — do not add a `family == …` branch
in `Server.swift`.

## Checklist

- [ ] Config decodes from the model's `config.json`
- [ ] `@ModuleInfo` keys match safetensors key paths
- [ ] Quantization filter excludes the right layers
- [ ] Forward pass matches Python reference (check with same tokens)
- [ ] Chat template produces correct token IDs
- [ ] RMSNorm variant is correct (standard, +1 offset, parameter-free)
- [ ] Activation function is correct (gelu, gelu_approx, silu, relu)
- [ ] Attention scale is correct (1/sqrt(d) vs 1.0)
- [ ] Bias presence matches (bias: true vs false)
- [ ] RoPE base and dimensions are correct
- [ ] New family only: the `ModelFamily` case's `rawValue` matches the
      loader's `LoadedModel(family:)` string exactly (see section 7 -
      a mismatch fails silently, not at build or test time)

## Common Config Fields

Most models share these (via `ModelConfig` protocol):

| Field | CodingKey | Purpose |
|-------|-----------|---------|
| hiddenSize | hidden_size | Main hidden dimension |
| intermediateSize | intermediate_size | MLP intermediate |
| numAttentionHeads | num_attention_heads | Query heads |
| numKeyValueHeads | num_key_value_heads | KV heads (GQA) |
| numHiddenLayers | num_hidden_layers | Layer count |
| vocabSize | vocab_size | Vocabulary size |
| rmsNormEps | rms_norm_eps | Norm epsilon |
| ropeTheta | rope_theta | RoPE base frequency |
| quantization | quantization | Quantization config |
