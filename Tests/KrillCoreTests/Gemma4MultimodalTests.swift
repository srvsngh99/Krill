import XCTest
import MLX
@testable import KrillCore

final class Gemma4MultimodalTests: XCTestCase {

    // MARK: - Image Preprocessing Tests

    func testNativeImagePreprocessingRejectsEmptyData() {
        XCTAssertThrowsError(try preprocessImage(Data())) { error in
            XCTAssertEqual(
                String(describing: error),
                MultimodalPreprocessingError.emptyImageData.description)
        }
    }

    func testNativeImagePreprocessingProducesCorrectShape() throws {
        // Create a minimal valid 2x2 red PNG
        let pngData = createTestPNG(width: 2, height: 2, r: 255, g: 0, b: 0)
        let tensor = try preprocessImage(pngData, targetSize: 48)

        // Shape must be [1, 3, H, W] (channel-first) where H,W are multiples of 48
        XCTAssertEqual(tensor.ndim, 4)
        XCTAssertEqual(tensor.dim(0), 1)
        XCTAssertEqual(tensor.dim(1), 3, "Channel dim must be 3")
        XCTAssertTrue(tensor.dim(2) % 48 == 0, "Height must be divisible by 48, got \(tensor.dim(2))")
        XCTAssertTrue(tensor.dim(3) % 48 == 0, "Width must be divisible by 48, got \(tensor.dim(3))")
    }

    func testNativeImagePreprocessingNormalizesToExpectedRange() throws {
        // White image: all pixels should map to ~1.0 (255/255 = 1.0)
        let pngData = createTestPNG(width: 4, height: 4, r: 255, g: 255, b: 255)
        let tensor = try preprocessImage(pngData, targetSize: 48)

        // Check that values are in [0, 1] range (channel-first format, normalized to [0,1])
        let maxVal = MLX.max(tensor).item(Float.self)
        let minVal = MLX.min(tensor).item(Float.self)
        XCTAssertGreaterThanOrEqual(minVal, -0.01)
        XCTAssertLessThanOrEqual(maxVal, 1.01)
    }

    // MARK: - Audio Preprocessing Tests

    func testNativeAudioPreprocessingWithNoWaveformThrows() {
        XCTAssertThrowsError(try computeMelSpectrogram()) { error in
            XCTAssertEqual(
                String(describing: error),
                MultimodalPreprocessingError.audioPreprocessingUnavailable.description)
        }
    }

    func testNativeAudioPreprocessingProducesCorrectShape() throws {
        // 1 second of 16kHz audio (sine wave)
        let sampleRate = 16000
        let duration: Float = 1.0
        let numSamples = Int(duration * Float(sampleRate))
        var waveform = [Float](repeating: 0, count: numSamples)
        for i in 0 ..< numSamples {
            waveform[i] = sin(2.0 * .pi * 440.0 * Float(i) / Float(sampleRate))
        }

        let melSpec = try computeMelSpectrogram(
            waveform: waveform,
            sampleRate: sampleRate,
            melBins: 128,
            frameMs: 40
        )

        // Shape: [1, numFrames, 128]
        XCTAssertEqual(melSpec.ndim, 3)
        XCTAssertEqual(melSpec.dim(0), 1)
        XCTAssertGreaterThan(melSpec.dim(1), 0, "Should have at least one frame")
        XCTAssertEqual(melSpec.dim(2), 128)

        // Expected frames: (16000 - 640) / 320 + 1 = 48
        let expectedFrames = (numSamples - sampleRate * 40 / 1000) / (sampleRate * 20 / 1000) + 1
        XCTAssertEqual(melSpec.dim(1), expectedFrames)
    }

    func testWAVLoadingRoundTrip() throws {
        // Create a minimal WAV file (16-bit PCM, mono, 16kHz)
        let wavData = createTestWAV(sampleRate: 16000, durationMs: 100, frequency: 440)
        let (samples, sr) = try loadWAV(from: wavData)

        XCTAssertEqual(sr, 16000)
        XCTAssertGreaterThan(samples.count, 0)
        // 100ms at 16kHz = 1600 samples
        XCTAssertEqual(samples.count, 1600)
    }

    func testMelSpectrogramFromWAV() throws {
        let wavData = createTestWAV(sampleRate: 16000, durationMs: 500, frequency: 440)
        let melSpec = try computeMelSpectrogramFromWAV(wavData)

        XCTAssertEqual(melSpec.ndim, 3)
        XCTAssertEqual(melSpec.dim(0), 1)
        XCTAssertGreaterThan(melSpec.dim(1), 0)
        XCTAssertEqual(melSpec.dim(2), 128)
    }

    // MARK: - Config Tests

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

// MARK: - Test Asset Helpers

import CoreGraphics
import ImageIO

/// Create a minimal PNG image for testing.
func createTestPNG(width: Int, height: Int, r: UInt8, g: UInt8, b: UInt8) -> Data {
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    guard let ctx = CGContext(
        data: nil, width: width, height: height,
        bitsPerComponent: 8, bytesPerRow: width * 4,
        space: colorSpace,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else { return Data() }

    ctx.setFillColor(CGColor(
        red: CGFloat(r) / 255.0,
        green: CGFloat(g) / 255.0,
        blue: CGFloat(b) / 255.0,
        alpha: 1.0))
    ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))

    guard let image = ctx.makeImage() else { return Data() }
    let mutableData = NSMutableData()
    guard let dest = CGImageDestinationCreateWithData(mutableData, "public.png" as CFString, 1, nil) else { return Data() }
    CGImageDestinationAddImage(dest, image, nil)
    CGImageDestinationFinalize(dest)
    return mutableData as Data
}

/// Create a minimal WAV file with a sine wave for testing.
func createTestWAV(sampleRate: Int, durationMs: Int, frequency: Float) -> Data {
    let numSamples = sampleRate * durationMs / 1000
    var data = Data()

    // RIFF header
    data.append(contentsOf: [0x52, 0x49, 0x46, 0x46])  // "RIFF"
    let dataSize = numSamples * 2
    let fileSize = UInt32(36 + dataSize)
    data.append(contentsOf: withUnsafeBytes(of: fileSize.littleEndian) { Array($0) })
    data.append(contentsOf: [0x57, 0x41, 0x56, 0x45])  // "WAVE"

    // fmt chunk
    data.append(contentsOf: [0x66, 0x6D, 0x74, 0x20])  // "fmt "
    data.append(contentsOf: withUnsafeBytes(of: UInt32(16).littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) })  // PCM
    data.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) })  // mono
    data.append(contentsOf: withUnsafeBytes(of: UInt32(sampleRate).littleEndian) { Array($0) })
    let byteRate = UInt32(sampleRate * 2)
    data.append(contentsOf: withUnsafeBytes(of: byteRate.littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: UInt16(2).littleEndian) { Array($0) })  // block align
    data.append(contentsOf: withUnsafeBytes(of: UInt16(16).littleEndian) { Array($0) }) // bits per sample

    // data chunk
    data.append(contentsOf: [0x64, 0x61, 0x74, 0x61])  // "data"
    data.append(contentsOf: withUnsafeBytes(of: UInt32(dataSize).littleEndian) { Array($0) })

    for i in 0 ..< numSamples {
        let sample = sin(2.0 * .pi * frequency * Float(i) / Float(sampleRate))
        let int16 = Int16(sample * 32767.0)
        data.append(contentsOf: withUnsafeBytes(of: int16.littleEndian) { Array($0) })
    }

    return data
}
