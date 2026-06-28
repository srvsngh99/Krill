import Foundation
import MLX
import MLXNN

// MARK: - DeepEncoder (Unlimited-OCR / DeepSeek-OCR vision front-end)

/// Ties the two validated towers + projector into the vision feature extractor.
/// Per `modeling_unlimitedocr`:
///   sam_feat  = sam_model(image)                  [B,1024,16,16]
///   clip_feat = vision_model(image, sam_feat)     [B,257,1024]
///   vis = cat(clip_feat[:,1:], sam_feat.flatten(2).permute(0,2,1), -1)  [B,256,2048]
///   features  = projector(vis)                    [B,256,1280]
/// The 256 feature tokens per 1024-image (tile) are what splice into the LM.

final class DeepEncoderProjector: Module {
    @ModuleInfo(key: "layers") var layers: Linear   // linear projector_type
    init(inputDim: Int, nEmbed: Int) {
        _layers = ModuleInfo(wrappedValue: Linear(inputDim, nEmbed, bias: true), key: "layers")
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { layers(x) }
}

public final class DeepEncoder: Module {
    @ModuleInfo(key: "sam_model") var samModel: DeepEncoderSAM
    @ModuleInfo(key: "vision_model") var visionModel: DeepEncoderCLIP
    @ModuleInfo(key: "projector") var projector: DeepEncoderProjector

    public init(samConfig: SAMConfig = SAMConfig(), clipConfig: CLIPVisionConfig = CLIPVisionConfig(),
                nEmbed: Int = 1280) {
        _samModel = ModuleInfo(wrappedValue: DeepEncoderSAM(samConfig), key: "sam_model")
        _visionModel = ModuleInfo(wrappedValue: DeepEncoderCLIP(clipConfig), key: "vision_model")
        // input_dim 2048 = CLIP(1024) ++ SAM(1024)
        _projector = ModuleInfo(
            wrappedValue: DeepEncoderProjector(inputDim: 2048, nEmbed: nEmbed), key: "projector")
    }

    /// `image`: channels-last [B, 1024, 1024, 3]. Returns LM-space vision
    /// features [B, 256, nEmbed].
    public func callAsFunction(image: MLXArray) -> MLXArray {
        let samFeat = samModel(image: image)                       // [B,1024,16,16]
        let clipFeat = visionModel(patchEmbeds: samFeat)           // [B,257,1024]
        let B = samFeat.dim(0)
        let clipNoCls = clipFeat[0..., 1..., 0...]                 // [B,256,1024]
        let samFlat = samFeat.reshaped(B, samFeat.dim(1), -1).transposed(0, 2, 1)  // [B,256,1024]
        let vis = concatenated([clipNoCls, samFlat], axis: -1)     // [B,256,2048]
        return projector(vis)                                      // [B,256,nEmbed]
    }
}

/// Base (single-view, non-tiled) multimodal token assembly (G5), mirroring
/// modeling_unlimitedocr (lines 551-573): lay the 16x16 feature grid out, append
/// `imageNewline` as an extra column per row, flatten, then append one
/// `viewSeparator`. `features`: [1, hw, n]; returns [h*(w+1) + 1, n] (273 for a
/// 1024 image). These embeddings masked-scatter into `embed_tokens` at the
/// `<image>` placeholder positions.
public func assembleBaseViewTokens(
    features: MLXArray, imageNewline: MLXArray, viewSeparator: MLXArray
) -> MLXArray {
    let hw = features.dim(1), n = features.dim(2)
    let h = Int(Double(hw).squareRoot().rounded())
    let w = h
    let grid = features.reshaped(h, w, n)
    let newlineCol = broadcast(imageNewline.reshaped(1, 1, n), to: [h, 1, n])
    let withNewlines = concatenated([grid, newlineCol], axis: 1)   // [h, w+1, n]
    let flat = withNewlines.reshaped(h * (w + 1), n)               // [h*(w+1), n]
    return concatenated([flat, viewSeparator.reshaped(1, n)], axis: 0)
}
