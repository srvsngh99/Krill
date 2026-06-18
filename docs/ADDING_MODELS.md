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
one) means declaring its server-side contract in the registry so the
server routes it without a new hand-written branch:

- `ModelCapabilities.swift`: add a `capabilities(for:)` and a
  `supportTier(for:)` case.
- `ModelAdapter.swift`: the `switch family` in `chatRouting`,
  `requiresImageInput`, and `chatTemplate` is exhaustive, so a new
  `ModelFamily` case will not compile until each is given a value.
  Pick `.denseEngine` for a native Swift+MLX text/vision family
  (Qwen 2.5-VL is one); `.mixtureOfExperts` for the MoE path.

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
