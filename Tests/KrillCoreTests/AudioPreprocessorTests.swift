import XCTest
import MLX
@testable import KrillCore
#if canImport(AVFoundation)
@preconcurrency import AVFoundation
#endif

/// WS5 unit tests for the native Gemma 4 USM audio frontend. Deterministic
/// in-memory WAVs (no external assets) pin shapes, the soft-token cadence,
/// padding-zeroing, and error handling against the documented USM defaults.
final class AudioPreprocessorTests: XCTestCase {

    /// Build a 16-bit PCM mono WAV from float samples in [-1, 1].
    private func makeWAV(_ samples: [Float], sampleRate: Int = 16_000) -> Data {
        var d = Data()
        func u32(_ v: UInt32) { var x = v.littleEndian; d.append(Data(bytes: &x, count: 4)) }
        func u16(_ v: UInt16) { var x = v.littleEndian; d.append(Data(bytes: &x, count: 2)) }
        let dataBytes = samples.count * 2
        d.append("RIFF".data(using: .ascii)!); u32(UInt32(36 + dataBytes))
        d.append("WAVE".data(using: .ascii)!)
        d.append("fmt ".data(using: .ascii)!); u32(16); u16(1); u16(1)
        u32(UInt32(sampleRate)); u32(UInt32(sampleRate * 2)); u16(2); u16(16)
        d.append("data".data(using: .ascii)!); u32(UInt32(dataBytes))
        for s in samples {
            let c = max(-1.0, min(1.0, s))
            u16(UInt16(bitPattern: Int16(c * 32767.0)))
        }
        return d
    }

    private func sine(seconds: Double, freq: Double = 440, sr: Int = 16_000) -> [Float] {
        let n = Int(seconds * Double(sr))
        return (0 ..< n).map { Float(0.5 * sin(2.0 * .pi * freq * Double($0) / Double(sr))) }
    }

#if canImport(AVFoundation)
    private func makeM4A(_ samples: [Float], sampleRate: Int = 16_000) throws -> Data {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krill-audio-test-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("tone.m4a")
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatMPEG4AAC,
            AVSampleRateKey: Double(sampleRate),
            AVNumberOfChannelsKey: 1,
        ]
        do {
            let file = try AVAudioFile(forWriting: url, settings: settings)
            guard let format = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: Double(sampleRate),
                channels: 1,
                interleaved: false),
                  let buffer = AVAudioPCMBuffer(
                    pcmFormat: format,
                    frameCapacity: AVAudioFrameCount(samples.count)) else {
                throw MultimodalPreprocessingError.audioPreprocessingUnavailable
            }
            buffer.frameLength = AVAudioFrameCount(samples.count)
            guard let ch = buffer.floatChannelData?[0] else {
                throw MultimodalPreprocessingError.audioPreprocessingUnavailable
            }
            for (i, sample) in samples.enumerated() { ch[i] = sample }
            try file.write(from: buffer)
        }
        return try Data(contentsOf: url)
    }
#endif

    func testMelShapeAndTokenCadenceForOneSecond() throws {
        let wav = makeWAV(sine(seconds: 1.0))
        let f = try AudioPreprocessor.features(fromWAV: wav)
        MLX.eval(f.mel, f.validMask)
        XCTAssertEqual(f.mel.dim(0), 1)
        XCTAssertEqual(f.mel.dim(2), AudioPreprocessor.melBins)   // 128
        XCTAssertEqual(f.validMask.dim(0), 1)
        // mel time dim and the per-frame mask must agree.
        XCTAssertEqual(f.mel.dim(1), f.validMask.dim(1))
        XCTAssertGreaterThan(f.mel.dim(1), 0)
        // 1000 ms / 40 ms-per-token = 25 soft tokens.
        XCTAssertEqual(f.numTokens, 25)
    }

    func testGenericAudioDecoderHandlesWAVAndM4A() throws {
        let samples = sine(seconds: 1.0)
        let wav = makeWAV(samples)
        let wavFeatures = try AudioPreprocessor.features(fromAudio: wav)
        XCTAssertEqual(wavFeatures.numTokens, 25)
        XCTAssertEqual(wavFeatures.mel.dim(2), AudioPreprocessor.melBins)

#if canImport(AVFoundation)
        let m4a: Data
        do {
            m4a = try makeM4A(samples)
        } catch {
            throw XCTSkip("M4A encoder unavailable in this environment: \(error)")
        }
        let m4aFeatures = try AudioPreprocessor.features(fromAudio: m4a)
        XCTAssertEqual(m4aFeatures.numTokens, 25)
        XCTAssertEqual(m4aFeatures.mel.dim(2), AudioPreprocessor.melBins)
#endif
    }

    func testTokenCadenceScalesAndCapsAt750() throws {
        let three = try AudioPreprocessor.features(fromWAV: makeWAV(sine(seconds: 3.0)))
        XCTAssertEqual(three.numTokens, 75)            // ceil(3000/40)
        // 40 s would be 1000 tokens uncapped; must clamp to 750.
        let long = try AudioPreprocessor.features(fromWAV: makeWAV(sine(seconds: 40.0)))
        XCTAssertEqual(long.numTokens, 750)
    }

    func testDeterministicAcrossRuns() throws {
        let wav = makeWAV(sine(seconds: 0.5))
        let a = try AudioPreprocessor.features(fromWAV: wav)
        let b = try AudioPreprocessor.features(fromWAV: wav)
        MLX.eval(a.mel, b.mel)
        XCTAssertTrue(MLX.allClose(a.mel, b.mel, rtol: 0, atol: 0).item(Bool.self))
        XCTAssertEqual(a.numTokens, b.numTokens)
        XCTAssertEqual(a.mel.dim(1), b.mel.dim(1))
    }

    func testPaddedTailFramesAreZeroedAndMaskedInvalid() throws {
        // 0.31 s is not a multiple of 128 samples, so the extractor pads and
        // the trailing frames must be invalid (mask false) and zeroed.
        let f = try AudioPreprocessor.features(fromWAV: makeWAV(sine(seconds: 0.31)))
        MLX.eval(f.mel, f.validMask)
        let valid = f.validMask.reshaped([f.validMask.dim(1)])
        // Some frame is valid and the very last frame is padding.
        XCTAssertTrue(valid.any().item(Bool.self))
        let last = valid[valid.dim(0) - 1].item(Bool.self)
        XCTAssertFalse(last, "trailing padded frame must be invalid")
        let lastRow = f.mel[0, f.mel.dim(1) - 1]
        let maxAbs = MLX.max(MLX.abs(lastRow)).item(Float.self)
        XCTAssertEqual(maxAbs, 0.0, "padded spectrogram rows must be zeroed")
    }

    func testFilterBankAndDFTBasisShapes() {
        let mel = AudioPreprocessor.melFilterBank()
        XCTAssertEqual(mel.shape, [AudioPreprocessor.numFreqBins, AudioPreprocessor.melBins])
        let (cosB, sinB) = AudioPreprocessor.dftBasis()
        XCTAssertEqual(cosB.shape, [AudioPreprocessor.frameLength, AudioPreprocessor.numFreqBins])
        XCTAssertEqual(sinB.shape, [AudioPreprocessor.frameLength, AudioPreprocessor.numFreqBins])
    }

    func testEmptyAndInvalidInputsThrow() {
        XCTAssertThrowsError(try AudioPreprocessor.features(waveform: []))
        XCTAssertThrowsError(try AudioPreprocessor.features(fromWAV: Data([0, 1, 2, 3])))
    }

    /// PR #21 rereview P1b: undecodable bytes on the generic path must
    /// throw — the engine turns this into a loud error stream rather than
    /// a silent text-only answer. Covers a non-WAV garbage blob and a
    /// RIFF/WAVE header with a corrupt body (isWAV true, decode fails).
    func testGenericDecoderThrowsOnUndecodableAudio() {
        XCTAssertThrowsError(
            try AudioPreprocessor.features(fromAudio: Data([0, 1, 2, 3, 4])))
        var corruptWav = Data("RIFF".utf8)
        corruptWav.append(Data([0xff, 0xff, 0xff, 0xff]))
        corruptWav.append(Data("WAVE".utf8))
        corruptWav.append(Data(repeating: 0x7f, count: 8))   // no fmt/data
        XCTAssertTrue(AudioPreprocessor.isWAV(corruptWav))
        XCTAssertThrowsError(
            try AudioPreprocessor.features(fromAudio: corruptWav))
    }

    /// `isWAV` keeps the exact PCM decoder path available for RIFF/WAVE while
    /// non-WAV containers go through the platform decoder.
    func testIsWAVDiscriminatesContainers() {
        XCTAssertTrue(AudioPreprocessor.isWAV(makeWAV(sine(seconds: 0.05))))
        XCTAssertFalse(AudioPreprocessor.isWAV(Data()))
        XCTAssertFalse(AudioPreprocessor.isWAV(Data([0, 1, 2, 3])))
        // ID3-tagged MP3 header.
        XCTAssertFalse(AudioPreprocessor.isWAV(Data("ID3\u{04}\u{00}\u{00}\u{00}\u{00}\u{00}\u{00}\u{00}\u{00}".utf8)))
        // RIFF but not WAVE (e.g. AVI) must be rejected.
        var riffNotWave = Data("RIFF".utf8)
        riffNotWave.append(Data([0, 0, 0, 0]))
        riffNotWave.append(Data("AVI ".utf8))
        XCTAssertFalse(AudioPreprocessor.isWAV(riffNotWave))
    }
}
