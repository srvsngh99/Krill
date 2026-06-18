import XCTest
import MLX
@testable import KrillCore
@testable import KrillTokenizer

/// Engine image-serving wiring for the native LLaVA-1.5 runtime (PR #129 landed
/// the model math). These pin the two family-specific seams the generic
/// multimodal path needs: CLIP image preprocessing and the vicuna prompt with
/// the image-token run placed directly.
final class LlavaImageServingTests: XCTestCase {

    #if canImport(CoreGraphics) && canImport(ImageIO)
    /// Encode a solid-color image as PNG bytes.
    private func solidPNG(width: Int, height: Int,
                          r: UInt8, g: UInt8, b: UInt8) -> Data {
        let cs = CGColorSpaceCreateDeviceRGB()
        let ctx = CGContext(
            data: nil, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4, space: cs,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.setFillColor(CGColor(
            red: CGFloat(r) / 255, green: CGFloat(g) / 255,
            blue: CGFloat(b) / 255, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        let image = ctx.makeImage()!
        let out = NSMutableData()
        let dest = CGImageDestinationCreateWithData(
            out, "public.png" as CFString, 1, nil)!
        CGImageDestinationAddImage(dest, image, nil)
        XCTAssertTrue(CGImageDestinationFinalize(dest))
        return out as Data
    }

    // MARK: - Preprocessing

    func testPreprocessReturnsSquareChannelFirstTensor() throws {
        // A non-square image must be resized (shortest edge -> 336) and
        // center-cropped to a square [1, 3, 336, 336].
        let png = solidPNG(width: 500, height: 280, r: 200, g: 30, b: 30)
        let pixels = try LlavaImagePreprocessor.preprocess(png, imageSize: 336)
        XCTAssertEqual(pixels.shape, [1, 3, 336, 336])
    }

    func testPreprocessNormalizesWithClipStats() throws {
        // A pure-black image (all channels 0) maps to (0 - mean) / std per
        // channel, so each channel plane is a constant equal to that value.
        let png = solidPNG(width: 336, height: 336, r: 0, g: 0, b: 0)
        let pixels = try LlavaImagePreprocessor.preprocess(png, imageSize: 336)
        eval(pixels)
        let arr = pixels.asArray(Float.self)
        let plane = 336 * 336
        for c in 0 ..< 3 {
            let expected = (0 - LlavaImagePreprocessor.imageMean[c])
                / LlavaImagePreprocessor.imageStd[c]
            XCTAssertEqual(arr[c * plane], expected, accuracy: 1e-3,
                "channel \(c) must be normalized with the CLIP mean/std")
        }
        // The R channel constant (mean 0.481, smallest std) is the most
        // negative; B (largest mean offset/std) differs -- the three planes
        // are NOT identical, confirming per-channel normalization.
        XCTAssertNotEqual(arr[0], arr[2 * plane], accuracy: 1e-4)
    }

    func testPreprocessRejectsEmptyData() {
        XCTAssertThrowsError(try LlavaImagePreprocessor.preprocess(Data(), imageSize: 336))
    }
    #endif

    // MARK: - Minimal text_config decoding

    /// The canonical mlx-community/llava-1.5-7b-4bit `config.json` ships a
    /// MINIMAL `text_config` that omits hidden_size / intermediate_size /
    /// num_attention_heads / num_hidden_layers / rope_theta and relies on
    /// transformers' LlamaConfig defaults (vicuna-7b dims). `LlavaConfig` must
    /// decode it by filling the HF base-Llama defaults rather than failing on
    /// the missing keys.
    func testDecodesMinimalTextConfigWithLlamaDefaults() throws {
        let json = """
        {
          "architectures": ["LlavaForConditionalGeneration"],
          "image_token_index": 32000,
          "model_type": "llava",
          "quantization": {"group_size": 64, "bits": 4},
          "text_config": {
            "architectures": ["LlamaForCausalLM"],
            "max_position_embeddings": 4096,
            "model_type": "llama",
            "rms_norm_eps": 1e-05,
            "vocab_size": 32064
          },
          "vision_config": {
            "hidden_size": 1024, "image_size": 336, "intermediate_size": 4096,
            "model_type": "clip_vision_model", "num_attention_heads": 16,
            "num_hidden_layers": 24, "patch_size": 14
          },
          "vision_feature_layer": -2,
          "vision_feature_select_strategy": "default",
          "vocab_size": 32064
        }
        """
        let cfg = try JSONDecoder().decode(LlavaConfig.self, from: Data(json.utf8))
        // HF base-Llama defaults == vicuna-7b dims.
        XCTAssertEqual(cfg.textConfig.hiddenSize, 4096)
        XCTAssertEqual(cfg.textConfig.intermediateSize, 11_008)
        XCTAssertEqual(cfg.textConfig.numAttentionHeads, 32)
        XCTAssertEqual(cfg.textConfig.numKeyValueHeads, 32)
        XCTAssertEqual(cfg.textConfig.numHiddenLayers, 32)
        // rope_theta must default to the vicuna/Llama-2 value (10000), NOT
        // Krill's Llama-3 default (500000).
        XCTAssertEqual(cfg.textConfig.ropeTheta, 10_000.0)
        // Present keys are honored.
        XCTAssertEqual(cfg.textConfig.maxPositionEmbeddings, 4096)
        XCTAssertEqual(cfg.textConfig.vocabSize, 32_064)
        XCTAssertEqual(cfg.imageTokenIndex, 32_000)
        XCTAssertEqual(cfg.visionConfig.numPatches, 576, "(336/14)^2")
    }
}
