import Foundation
import MLX
import MLXNN
import KrillCache

// Native Swift+MLX runtime for NVIDIA LocateAnything-3B (`locateanything`) — a
// visual-grounding VLM that emits bounding boxes as ordinary text tokens
// (`<box><x1><y1><x2><y2></box>`, coords normalized 0-1000). Composition:
//
//   MoonViT vision tower (MoonViTVisionModel) -> 2x2 merge -> mlp1 connector
//   (LocateAnythingConnector) -> spliced into the Qwen2.5-3B token embeddings
//   at the image-token positions -> Qwen2.5 decoder (QwenForCausalLM).
//
// The model ships a `slow` pure-autoregressive generation mode, so this port
// runs the standard Krill AR decode loop — the custom "Parallel Box Decoding"
// (MTP) throughput trick is intentionally NOT ported. Box/coord tokens are
// already in the tokenizer (vocab 152681), so grounding output falls straight
// out of normal decoding.
//
// Weight subtrees: `vision_model.*` (MoonViT), `mlp1.*` (connector),
// `language_model.model.*` / `language_model.lm_head.*` (Qwen2.5). Parity:
// tools/verify_locateanything_parity.py (vision path); the text stack reuses
// the parity-gated `QwenForCausalLM`.

// MARK: - Config

public struct LocateAnythingConfig: Decodable, Sendable {
    public let imageTokenIndex: Int
    public let visionConfig: MoonViTVisionConfig
    public let textConfig: QwenConfig

    enum CodingKeys: String, CodingKey {
        case imageTokenIndex = "image_token_index"
        case visionConfig = "vision_config"
        case textConfig = "text_config"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        imageTokenIndex = try c.decodeIfPresent(Int.self, forKey: .imageTokenIndex) ?? 151665
        visionConfig = try c.decode(MoonViTVisionConfig.self, forKey: .visionConfig)
        textConfig = try c.decode(QwenConfig.self, forKey: .textConfig)
    }
}

// MARK: - Model

public final class LocateAnythingForConditionalGeneration: Module {
    @ModuleInfo(key: "vision_model") var visionModel: MoonViTVisionModel
    @ModuleInfo(key: "mlp1") var connector: LocateAnythingConnector
    @ModuleInfo(key: "language_model") var languageModel: QwenForCausalLM

    public let config: LocateAnythingConfig
    let imageTokenIndex: Int

    public init(_ config: LocateAnythingConfig) {
        self.config = config
        self.imageTokenIndex = config.imageTokenIndex
        _visionModel = ModuleInfo(wrappedValue: MoonViTVisionModel(config.visionConfig), key: "vision_model")
        _connector = ModuleInfo(
            wrappedValue: LocateAnythingConnector(
                vitHidden: config.visionConfig.hiddenSize, llmHidden: config.textConfig.hiddenSize),
            key: "mlp1")
        _languageModel = ModuleInfo(wrappedValue: QwenForCausalLM(config.textConfig), key: "language_model")
    }

    /// MoonViT tower + `mlp1` connector: `[N, C*ph*pw]` flattened patches and
    /// per-image `(h,w)` grids -> `[numMergedTokens, textHidden]` LM features.
    public func imageFeatures(_ pixelValues: MLXArray, grids: [(h: Int, w: Int)]) -> MLXArray {
        connector(visionModel(pixelValues, grids: grids))
    }

    /// Text-only forward (no image in the turn).
    public func callAsFunction(
        _ tokens: MLXArray, caches: [KVCache]? = nil, lastTokenOnly: Bool = false
    ) -> MLXArray {
        languageModel(tokens, caches: caches, lastTokenOnly: lastTokenOnly)
    }

    /// Multimodal forward: embed tokens, splice the projected MoonViT features
    /// over the contiguous `<image>` run, and run the Qwen2.5 decoder. The image
    /// tokens form one contiguous block (the processor emits `numMergedTokens`
    /// copies of `image_token_index`), matching the LLaVA splice pattern.
    public func callAsFunction(
        _ inputIds: MLXArray, pixelValues: MLXArray, grids: [(h: Int, w: Int)],
        caches: [KVCache]? = nil, lastTokenOnly: Bool = false
    ) -> MLXArray {
        var inputEmbeds = languageModel.model.embedTokens(inputIds)   // [1, L, H]
        let features = imageFeatures(pixelValues, grids: grids)       // [n, H]
        let n = features.dim(0)

        eval(inputIds)
        let ids = inputIds.reshaped(-1).asArray(Int32.self)
        let imagePositions = ids.indices.filter { ids[$0] == Int32(imageTokenIndex) }
        precondition(imagePositions.count == n,
            "LocateAnything expects one image token per vision feature "
            + "(\(imagePositions.count) tokens, \(n) features)")
        let start = imagePositions.first ?? 0
        let L = inputEmbeds.dim(1)
        let before = inputEmbeds[0..., 0 ..< start, 0...]
        let feat3 = features.expandedDimensions(axis: 0).asType(inputEmbeds.dtype)  // [1, n, H]
        let after = inputEmbeds[0..., (start + n) ..< L, 0...]
        inputEmbeds = concatenated([before, feat3, after], axis: 1)

        return languageModel(inputsEmbeds: inputEmbeds, caches: caches, lastTokenOnly: lastTokenOnly)
    }

    /// Remap the checkpoint into this module's key layout:
    ///   * reshape the MoonViT Conv2d patch kernel to a matmul weight, and
    ///   * rewrite the `mlp1` `nn.Sequential` numeric indices to named children
    ///     (`mlp1.0`->`mlp1.norm`, `mlp1.1`->`mlp1.fc1`, `mlp1.3`->`mlp1.fc2`).
    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out = MoonViTVisionModel.sanitize(weights)
        let remap: [String: String] = ["mlp1.0": "mlp1.norm", "mlp1.1": "mlp1.fc1", "mlp1.3": "mlp1.fc2"]
        for (from, to) in remap {
            for (k, v) in out where k.hasPrefix(from + ".") {
                out[to + String(k.dropFirst(from.count))] = v
                out.removeValue(forKey: k)
            }
        }
        return out
    }
}
