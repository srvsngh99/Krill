import XCTest
import MLX
import MLXRandom
@testable import KLMCore

/// WS5 shape tests for the native USM Conformer. With default-initialized
/// weights these are not numerically meaningful, but running the full graph
/// (conv subsampling -> 12 macaron blocks -> chunked relative-position
/// attention -> output_proj) end to end catches reshape/transpose bugs the
/// live tests would otherwise surface only with a real checkpoint.
final class AudioEncoderShapeTests: XCTestCase {

    func testAudioConfigParsesDictWithDefaults() {
        let def = AudioConfig()
        XCTAssertEqual(def.hiddenSize, 1024)
        XCTAssertEqual(def.numHiddenLayers, 12)
        XCTAssertEqual(def.numAttentionHeads, 8)
        XCTAssertEqual(def.headDim, 128)
        XCTAssertEqual(def.outputProjDims, 1536)

        let c = AudioConfig(from: ["hidden_size": 512,
                                   "num_hidden_layers": 4,
                                   "num_attention_heads": 4,
                                   "output_proj_dims": 768])
        XCTAssertEqual(c.hiddenSize, 512)
        XCTAssertEqual(c.numHiddenLayers, 4)
        XCTAssertEqual(c.headDim, 128)            // 512 / 4
        XCTAssertEqual(c.outputProjDims, 768)
        // Untouched keys keep USM defaults.
        XCTAssertEqual(c.attentionChunkSize, 12)
    }

    func testEncoderForwardProducesProjectedFrames() {
        // Small config keeps the test fast while exercising every module.
        var cfg = AudioConfig()
        cfg.numHiddenLayers = 2
        let enc = AudioEncoder(cfg)

        let T = 200
        let mel = MLXRandom.normal([1, T, AudioPreprocessor.melBins])
        let mask = MLXArray.ones([1, T]).asType(.bool)   // all valid

        let (out, outMask) = enc(mel, validMask: mask)
        MLX.eval(out, outMask)

        XCTAssertEqual(out.dim(0), 1)
        XCTAssertEqual(out.dim(2), cfg.outputProjDims)    // 1536
        // Two stride-2 conv blocks => ~T/4 frames, > 0, mask agrees.
        XCTAssertGreaterThan(out.dim(1), 0)
        XCTAssertLessThan(out.dim(1), T)
        XCTAssertEqual(out.dim(1), outMask.dim(1))
    }

    func testEncoderHandlesPartialValidityMask() {
        var cfg = AudioConfig()
        cfg.numHiddenLayers = 1
        let enc = AudioEncoder(cfg)

        let T = 160
        let mel = MLXRandom.normal([1, T, AudioPreprocessor.melBins])
        // Second half is padding (invalid).
        var flags = [Int32](repeating: 1, count: T)
        for i in (T / 2) ..< T { flags[i] = 0 }
        let mask = MLXArray(flags, [1, T]).asType(.bool)

        let (out, _) = enc(mel, validMask: mask)
        MLX.eval(out)
        XCTAssertEqual(out.dim(2), cfg.outputProjDims)
        XCTAssertGreaterThan(out.dim(1), 0)
    }
}
