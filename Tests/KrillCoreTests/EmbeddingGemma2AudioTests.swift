import XCTest
import Foundation
import MLX
import MLXNN
@testable import KrillCore

/// EmbeddingGemma 2 audio path without the real checkpoint: soft-token
/// arithmetic (pinned to the reference), feature shapes, limits, decoding and
/// resampling, and STRICT weight binding on a tiny synthetic tower.
final class EmbeddingGemma2AudioTests: XCTestCase {

    // MARK: Soft tokens / frames

    func testSoftTokenCountMatchesReferenceProcessor() {
        // (samples, <audio> placeholders sentence-transformers produced), from reference_audio*.json
        for (n, want) in [(33_315, 52), (93_508, 146), (373_028, 583)] {
            XCTAssertEqual(EG2AudioPreprocessor.softTokenCount(samples: n), want, "\(n) samples")
        }
        // The cap of 280 is NOT applied (23.3 s gives 583); 30 s gives 750.
        XCTAssertEqual(EG2AudioPreprocessor.softTokenCount(samples: 480_000), 750)
    }

    func testSoftTokenCountIsMonotonicAndCloseTo25PerSecond() {
        var prev = 0
        for n in stride(from: 1_600, through: 480_000, by: 997) {
            let t = EG2AudioPreprocessor.softTokenCount(samples: n)
            XCTAssertGreaterThanOrEqual(t, prev, "\(n)")
            prev = t
            XCTAssertLessThanOrEqual(abs(Double(t) - Double(n) / 640.0), 3, "\(n)")
        }
    }

    func testMelFrameCountFormula() {
        // (n + 160 - 321) / 160 + 1 on the 128-padded length
        XCTAssertEqual(EG2AudioPreprocessor.melFrames(samples: 16_000), (16_000 + 160 - 321) / 160 + 1)
        XCTAssertEqual(EG2AudioPreprocessor.melFrames(samples: 16_001), (16_128 + 160 - 321) / 160 + 1)
        XCTAssertEqual(EG2AudioPreprocessor.melFrames(samples: 100), 0)
    }

    func testFeaturesShapeAndValidMask() throws {
        let n = 20_000
        let w = (0 ..< n).map { Float(sin(Double($0) * 0.05)) * 0.3 }
        let p = try EG2AudioPreprocessor.prepare(waveform: w)
        let T = EG2AudioPreprocessor.melFrames(samples: n)
        XCTAssertEqual(p.mel.shape, [1, T, 128])
        XCTAssertEqual(p.validMask.shape, [1, T])
        XCTAssertEqual(p.softTokens, EG2AudioPreprocessor.softTokenCount(samples: n))
        // Valid frames are a prefix; padded frames are zeroed in the mel.
        let valid = p.validMask.asType(.int32).asArray(Int32.self)
        let firstInvalid = valid.firstIndex(of: 0) ?? T
        XCTAssertFalse(valid[firstInvalid...].contains(1), "valid frames must be a prefix")
        if firstInvalid < T {
            let tail = p.mel[0..., firstInvalid..., 0...].abs().max().item(Float.self)
            XCTAssertEqual(tail, 0)
        }
        XCTAssertEqual(p.seconds, Double(n) / 16_000, accuracy: 1e-9)
    }

    func testLimits() {
        XCTAssertThrowsError(try EG2AudioPreprocessor.softTokens(forWaveform: [Float](repeating: 0, count: 1_599))) {
            guard case EG2AudioError.tooShort = $0 else { return XCTFail("\($0)") }
        }
        XCTAssertNoThrow(try EG2AudioPreprocessor.softTokens(forWaveform: [Float](repeating: 0, count: 480_000)))
        XCTAssertThrowsError(try EG2AudioPreprocessor.softTokens(forWaveform: [Float](repeating: 0, count: 480_001))) {
            guard case EG2AudioError.tooLong = $0 else { return XCTFail("\($0)") }
            XCTAssertTrue("\($0)".contains("maximum is 30 s"), "\($0)")
        }
    }

    // MARK: Decode

    /// 16-bit PCM WAV bytes.
    private func wav(_ channels: [[Float]], rate: Int) -> Data {
        let n = channels[0].count, nch = channels.count
        var d = Data()
        func u32(_ v: Int) { var x = UInt32(v).littleEndian; d.append(Data(bytes: &x, count: 4)) }
        func u16(_ v: Int) { var x = UInt16(v).littleEndian; d.append(Data(bytes: &x, count: 2)) }
        d.append(Data("RIFF".utf8)); u32(36 + n * nch * 2); d.append(Data("WAVEfmt ".utf8))
        u32(16); u16(1); u16(nch); u32(rate); u32(rate * nch * 2); u16(nch * 2); u16(16)
        d.append(Data("data".utf8)); u32(n * nch * 2)
        for i in 0 ..< n {
            for c in 0 ..< nch {
                var x = Int16(max(-1, min(1, channels[c][i])) * 32767).littleEndian
                d.append(Data(bytes: &x, count: 2))
            }
        }
        return d
    }

    func testDecodeMono16kIsExact() throws {
        let src: [Float] = (0 ..< 4_000).map { Float(sin(Double($0) * 0.1)) * 0.5 }
        let out = try EG2AudioPreprocessor.decode(wav([src], rate: 16_000))
        XCTAssertEqual(out.count, src.count)
        for i in stride(from: 0, to: src.count, by: 37) { XCTAssertEqual(out[i], src[i], accuracy: 1e-4) }
    }

    func testDecodeStereo44kResamplesAndDownmixes() throws {
        let rate = 44_100, n = 44_100  // 1 s of a 440 Hz tone, identical in both channels
        let tone: [Float] = (0 ..< n).map { Float(sin(2 * Double.pi * 440 * Double($0) / Double(rate))) * 0.5 }
        let out = try EG2AudioPreprocessor.decode(wav([tone, tone], rate: rate), formatHint: "wav")
        XCTAssertEqual(Double(out.count), 16_000, accuracy: 64)
        let mid = out.dropFirst(800).dropLast(800)
        let rms = (mid.reduce(0) { $0 + $1 * $1 } / Float(mid.count)).squareRoot()
        XCTAssertEqual(Double(rms), 0.5 / 2.0.squareRoot(), accuracy: 0.02, "tone amplitude must survive resampling")
        // Opposite-phase channels cancel in the mono mix.
        let anti = try EG2AudioPreprocessor.decode(wav([tone, tone.map { -$0 }], rate: rate))
        XCTAssertLessThan(anti.map { abs($0) }.max() ?? 1, 1e-3)
    }

    private static func fixture(_ rel: String, _ file: StaticString = #filePath) -> URL {
        URL(fileURLWithPath: "\(file)").deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("KrillEngineTests/Fixtures/eg2_mm/\(rel)")
    }

    func testFixtureFormatsDecodeToTheExpectedLengths() throws {
        // WAV 16 kHz: 33315 samples (the reference's count).
        XCTAssertEqual(try EG2AudioPreprocessor.decode(Data(contentsOf: Self.fixture("audio/a3.wav"))).count, 33_315)
        // FLAC 44.1 kHz stereo -> 16 kHz mono: 33315 (+-2) samples.
        let flac = try EG2AudioPreprocessor.decode(Data(contentsOf: Self.fixture("audio/a3_44k_stereo.flac")))
        XCTAssertEqual(Double(flac.count), 33_315, accuracy: 2)
        // MP3: LAME-tag delay/padding trimmed like ffmpeg: exactly the original 93508 samples.
        let mp3 = try EG2AudioPreprocessor.decode(Data(contentsOf: Self.fixture("audio/a8.mp3")), formatHint: "mp3")
        XCTAssertEqual(mp3.count, 93_508)
        // M4A (AAC): AVFoundation applies the edit list.
        let m4a = try EG2AudioPreprocessor.decode(Data(contentsOf: Self.fixture("audio/a8.m4a")), formatHint: "m4a")
        XCTAssertEqual(Double(m4a.count), 93_508, accuracy: 100)
    }

    func testMp3GaplessTrimReadsTheLameTag() throws {
        let mp3 = try Data(contentsOf: Self.fixture("audio/a8.mp3"))
        let t = EG2AudioPreprocessor.mp3GaplessTrim(mp3)
        XCTAssertEqual(t?.start, 576)
        XCTAssertEqual(t?.end, 956)
        XCTAssertNil(EG2AudioPreprocessor.mp3GaplessTrim(Data("ID3 not really an mp3".utf8)))
        XCTAssertNil(EG2AudioPreprocessor.mp3GaplessTrim(Data()))
    }

    func testDecodeRejectsGarbage() {
        XCTAssertThrowsError(try EG2AudioPreprocessor.decode(Data()))
        XCTAssertThrowsError(try EG2AudioPreprocessor.decode(Data("this is not audio at all".utf8), formatHint: "wav")) {
            guard case EG2AudioError.undecodable = $0 else { return XCTFail("\($0)") }
        }
    }

    func testFormatHintAndSniffing() {
        XCTAssertEqual(EG2AudioPreprocessor.sanitizedExtension("audio/mpeg"), "mp3")
        XCTAssertEqual(EG2AudioPreprocessor.sanitizedExtension("x-wav"), "wav")
        XCTAssertEqual(EG2AudioPreprocessor.sanitizedExtension("MP3"), "mp3")
        XCTAssertEqual(EG2AudioPreprocessor.sanitizedExtension("aac"), "m4a")
        XCTAssertNil(EG2AudioPreprocessor.sanitizedExtension("../../etc/passwd"))
        XCTAssertNil(EG2AudioPreprocessor.sanitizedExtension(""))
        XCTAssertEqual(EG2AudioPreprocessor.sniffExtension(Data("RIFF\0\0\0\0WAVE".utf8)), "wav")
        XCTAssertEqual(EG2AudioPreprocessor.sniffExtension(Data("fLaC....".utf8)), "flac")
        XCTAssertEqual(EG2AudioPreprocessor.sniffExtension(Data("ID3\u{3}".utf8)), "mp3")
    }

    // MARK: Tiny tower, strict binding

    private func tinyConfigJSON(clipped: Bool = true, withAudio: Bool = true) -> String {
        let audio = withAudio ? """
        , "audio_config": {"hidden_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2,
           "subsampling_conv_channels": [4, 4], "conv_kernel_size": 5, "residual_weight": 0.5,
           "attention_chunk_size": 4, "attention_context_left": 5, "attention_context_right": 0,
           "attention_logit_cap": 50.0, "rms_norm_eps": 1e-6, "output_proj_dims": 24,
           "use_clipped_linears": \(clipped)}
        """ : ""
        return """
        {"model_type": "embedding_gemma2", "boa_token_id": 256000, "eoa_token_index": 258883, "audio_token_id": 258881,
         "text_config": {"embedding_dim": 12, "head_dim": 8, "hidden_size": 16, "hidden_size_per_layer_input": 8,
           "intermediate_size": 32, "num_attention_heads": 2, "num_hidden_layers": 2, "num_key_value_heads": 1,
           "per_layer_config": {"1": {"head_dim": 16, "num_key_value_heads": 1}}, "sliding_window": 4, "vocab_size": 50}\(audio)}
        """
    }

    private func makeTinyTower() throws -> EG2AudioTower {
        let root = try JSONSerialization.jsonObject(with: Data(tinyConfigJSON().utf8)) as! [String: Any]
        let t = EG2AudioTower(AudioConfig(from: root["audio_config"] as? [String: Any]), textHidden: 16)
        MLXRandom.seed(5)
        let p = t.parameters().flattened().map { (k, v) -> (String, MLXArray) in
            if k.hasSuffix("_min") { return (k, MLXArray(Float(-1e4))) }
            if k.hasSuffix("_max") { return (k, MLXArray(Float(1e4))) }
            if k.hasSuffix("norm.weight") { return (k, MLXArray.ones(v.shape)) }
            return (k, MLXRandom.normal(v.shape) * 0.1)
        }
        t.update(parameters: ModuleParameters.unflattened(p))
        t.setComputeDtype(.float32)
        eval(t)
        return t
    }

    private func writeTinyCheckpoint(mutate: (inout [String: MLXArray]) -> Void = { _ in },
                                     clipped: Bool = true, withAudio: Bool = true,
                                     pytorchLayout: Bool = true) throws -> URL {
        let tower = try makeTinyTower()
        var arrays: [String: MLXArray] = [:]
        for (k, v) in tower.parameters().flattened() {
            // The real checkpoint stores convs in PyTorch layout.
            if pytorchLayout, k.hasSuffix(".conv.weight") { arrays[k] = v.transposed(0, 3, 1, 2) }
            else if pytorchLayout, k.hasSuffix("depthwise_conv1d.weight") { arrays[k] = v.transposed(0, 2, 1) }
            else { arrays[k] = v }
        }
        mutate(&arrays)
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("eg2-audio-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data(tinyConfigJSON(clipped: clipped, withAudio: withAudio).utf8)
            .write(to: dir.appendingPathComponent("config.json"))
        try save(arrays: arrays, url: dir.appendingPathComponent("model.safetensors"))
        return dir
    }

    func testStrictBindingAcceptsExactCheckpointAndRuns() throws {
        let dir = try writeTinyCheckpoint()
        defer { try? FileManager.default.removeItem(at: dir) }
        let (tower, report) = try loadEG2AudioTower(directory: dir, dtype: .float32)
        XCTAssertGreaterThan(report.bound, 50)
        let w = (0 ..< 4_000).map { Float(sin(Double($0) * 0.07)) * 0.3 }
        let p = try EG2AudioPreprocessor.prepare(waveform: w)
        let s = try tower.softTokens(p)
        XCTAssertEqual(s.shape, [p.softTokens, 16])
        XCTAssertTrue(s.asArray(Float.self).allSatisfy { $0.isFinite })
    }

    func testConvLayoutConversionMatchesChannelLastCheckpointToo() throws {
        // PyTorch-layout (the real checkpoint) and channel-last tensors give the same tower.
        let w = (0 ..< 2_400).map { Float(sin(Double($0) * 0.05)) * 0.2 }
        let p = try EG2AudioPreprocessor.prepare(waveform: w + w + w + w)
        var outs: [[Float]] = []
        for pt in [true, false] {
            let dir = try writeTinyCheckpoint(pytorchLayout: pt)
            defer { try? FileManager.default.removeItem(at: dir) }
            outs.append(try loadEG2AudioTower(directory: dir).tower.softTokens(p).asArray(Float.self))
        }
        XCTAssertEqual(outs[0].count, outs[1].count)
        for (a, b) in zip(outs[0], outs[1]) { XCTAssertEqual(a, b, accuracy: 1e-6) }
    }

    func testConvLayoutConversionOnlyFiresOnTheExactPytorchShape() {
        let v = MLXArray.zeros([128, 1, 3, 3])
        XCTAssertEqual(eg2AudioConvToMLXLayout(key: "a.conv.weight", value: v, expected: [128, 3, 3, 1]).shape, [128, 3, 3, 1])
        XCTAssertEqual(eg2AudioConvToMLXLayout(key: "a.conv.weight", value: v, expected: [128, 9, 3, 1]).shape, [128, 1, 3, 3],
                       "a shape that matches neither layout is left for strict binding to reject")
        XCTAssertEqual(eg2AudioConvToMLXLayout(key: "x.weight", value: v, expected: [128, 3, 3, 1]).shape, [128, 1, 3, 3])
        let d = MLXArray.zeros([1024, 1, 5])
        XCTAssertEqual(eg2AudioConvToMLXLayout(key: "l.depthwise_conv1d.weight", value: d, expected: [1024, 5, 1]).shape, [1024, 5, 1])
    }

    func testStrictBindingRejectsUnknownKey() throws {
        let dir = try writeTinyCheckpoint { $0["audio_tower.layers.0.bogus.weight"] = MLXArray.zeros([3]) }
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertThrowsError(try loadEG2AudioTower(directory: dir))
    }

    func testStrictBindingRejectsMissingTensorsIncludingBiasAndClipScalars() throws {
        for key in ["embed_audio.embedding_projection.weight",
                    "audio_tower.output_proj.bias",
                    "audio_tower.layers.0.self_attn.per_dim_scale",
                    "audio_tower.layers.0.feed_forward1.ffw_layer_1.input_min"] {
            let dir = try writeTinyCheckpoint { $0.removeValue(forKey: key) }
            defer { try? FileManager.default.removeItem(at: dir) }
            XCTAssertThrowsError(try loadEG2AudioTower(directory: dir), key)
        }
    }

    func testStrictBindingRejectsWrongShape() throws {
        let dir = try writeTinyCheckpoint { $0["audio_tower.output_proj.weight"] = MLXArray.zeros([24, 31]) }
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertThrowsError(try loadEG2AudioTower(directory: dir))
    }

    func testUnclippedAudioConfigIsRefused() throws {
        let dir = try writeTinyCheckpoint(clipped: false)
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertThrowsError(try loadEG2AudioTower(directory: dir)) {
            guard case EG2AudioLoadError.unsupported = $0 else { return XCTFail("\($0)") }
        }
    }

    func testNoAudioTowerIsReported() throws {
        let dir = try writeTinyCheckpoint(mutate: { arrays in
            for k in arrays.keys { arrays.removeValue(forKey: k) }
            arrays["language_model.embed_tokens.weight"] = MLXArray.zeros([2, 2])
        }, withAudio: false)
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertThrowsError(try loadEG2AudioTower(directory: dir)) {
            guard case EG2AudioLoadError.noAudioTower = $0 else { return XCTFail("\($0)") }
        }
    }
}
