import XCTest
@testable import KLMCore

final class Gemma4MultimodalTests: XCTestCase {
    func testNativeImagePreprocessingFailsLoudlyForNonEmptyData() {
        XCTAssertThrowsError(try preprocessImage(Data([0x89, 0x50, 0x4E, 0x47]))) { error in
            XCTAssertEqual(
                String(describing: error),
                MultimodalPreprocessingError.imagePreprocessingUnavailable.description)
        }
    }

    func testNativeImagePreprocessingRejectsEmptyData() {
        XCTAssertThrowsError(try preprocessImage(Data())) { error in
            XCTAssertEqual(
                String(describing: error),
                MultimodalPreprocessingError.emptyImageData.description)
        }
    }

    func testNativeAudioPreprocessingFailsLoudly() {
        XCTAssertThrowsError(try computeMelSpectrogram()) { error in
            XCTAssertEqual(
                String(describing: error),
                MultimodalPreprocessingError.audioPreprocessingUnavailable.description)
        }
    }

    func testGemma4ConfigDecodesInstalledConditionalGenerationShape() throws {
        let json = """
        {
            "architectures": ["Gemma4ForConditionalGeneration"],
            "model_type": "gemma4",
            "image_token_id": 258880,
            "audio_token_id": 258881,
            "vision_soft_tokens_per_image": 280,
            "text_config": {
                "hidden_size": 1536,
                "intermediate_size": 6144,
                "num_attention_heads": 8,
                "num_key_value_heads": 1,
                "num_hidden_layers": 35,
                "vocab_size": 262144,
                "head_dim": 256,
                "global_head_dim": 512,
                "sliding_window": 512,
                "num_kv_shared_layers": 20,
                "use_double_wide_mlp": true,
                "tie_word_embeddings": true
            },
            "quantization": {
                "group_size": 64,
                "bits": 4,
                "mode": "affine"
            }
        }
        """

        let config = try JSONDecoder().decode(Gemma4Config.self, from: Data(json.utf8))

        XCTAssertEqual(config.hiddenSize, 1536)
        XCTAssertEqual(config.numHiddenLayers, 35)
        XCTAssertEqual(config.vocabSize, 262144)
        XCTAssertEqual(config.quantization?.bits, 4)
        XCTAssertTrue(config.isFullAttention(layerIdx: 4))
        XCTAssertFalse(config.isFullAttention(layerIdx: 3))
    }
}
