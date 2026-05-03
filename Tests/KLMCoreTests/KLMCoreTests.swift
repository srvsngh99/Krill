import XCTest
@testable import KLMCore
@testable import KLMCache

final class KLMCoreTests: XCTestCase {
    func testLlamaConfigDecoding() throws {
        let json = """
        {
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_hidden_layers": 32,
            "vocab_size": 128256,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "max_position_embeddings": 131072
        }
        """
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(LlamaConfig.self, from: data)

        XCTAssertEqual(config.hiddenSize, 4096)
        XCTAssertEqual(config.intermediateSize, 14336)
        XCTAssertEqual(config.numAttentionHeads, 32)
        XCTAssertEqual(config.numKeyValueHeads, 8)
        XCTAssertEqual(config.numHiddenLayers, 32)
        XCTAssertEqual(config.vocabSize, 128256)
        XCTAssertEqual(config.headDim, 128)
        XCTAssertNil(config.quantization)
    }

    func testLlamaConfigWithQuantization() throws {
        let json = """
        {
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_hidden_layers": 32,
            "vocab_size": 128256,
            "quantization": {
                "group_size": 64,
                "bits": 4
            }
        }
        """
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(LlamaConfig.self, from: data)

        XCTAssertNotNil(config.quantization)
        XCTAssertEqual(config.quantization?.groupSize, 64)
        XCTAssertEqual(config.quantization?.bits, 4)
    }

    func testLlamaConfigDefaults() throws {
        let json = """
        {
            "hidden_size": 2048,
            "intermediate_size": 5632,
            "num_attention_heads": 16,
            "num_hidden_layers": 16,
            "vocab_size": 32000
        }
        """
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(LlamaConfig.self, from: data)

        // num_key_value_heads defaults to num_attention_heads
        XCTAssertEqual(config.numKeyValueHeads, 16)
        // rms_norm_eps defaults to 1e-5
        XCTAssertEqual(config.rmsNormEps, 1e-5)
        // rope_theta defaults to 500000
        XCTAssertEqual(config.ropeTheta, 500_000.0)
    }

    func testKVCacheEmpty() {
        let cache = KVCache()
        XCTAssertEqual(cache.sequenceLength, 0)
    }

    func testManualConfig() {
        let config = LlamaConfig(
            hiddenSize: 256,
            intermediateSize: 512,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            numHiddenLayers: 2,
            vocabSize: 1000
        )
        XCTAssertEqual(config.headDim, 64) // 256 / 4
        XCTAssertNil(config.quantization)
    }
}
