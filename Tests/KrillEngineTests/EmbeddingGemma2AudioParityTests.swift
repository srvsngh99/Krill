import XCTest
import Foundation
import KrillCore
import MLX
import KrillTokenizer
@testable import KrillEngine

/// EmbeddingGemma 2 audio (and mixed text+audio) parity against fp32
/// sentence-transformers 6.1 (CPU) vectors, plus the sequence ids the reference
/// fed the model. Fixtures: `Fixtures/eg2_mm/` (README explains how each was
/// made; the `*_extra` / `mixed_av` references decode the non-16 kHz files with
/// ffmpeg, which is the decoder Krill's AVFoundation path is compared with).
///
/// Weighted tests skip when the checkpoint is absent (`KRILL_EG2_DIR`).
/// Gate: cosine >= 0.999 per case in fp32; bf16 is gated at >= 0.99 and its
/// measured floor is in docs/EMBEDDINGGEMMA2.md.
final class EmbeddingGemma2AudioParityTests: XCTestCase {
    struct Part: Decodable { let type: String; let file: String?; let text: String? }
    struct Case: Decodable {
        let name: String
        let modality: String
        let task: String?
        let parts: [Part]
        let input_ids_rle: [[Int]]
        let n_tokens: Int
        let embedding: [Float]
        let audio_soft_tokens: Int?
        let samples: Int?
    }
    struct Reference: Decodable { let cases: [Case] }

    private static func fixtureDir(_ file: StaticString = #filePath) -> URL {
        URL(fileURLWithPath: "\(file)").deletingLastPathComponent().appendingPathComponent("Fixtures/eg2_mm")
    }
    private func reference(_ name: String) throws -> [Case] {
        let data = try Data(contentsOf: Self.fixtureDir().appendingPathComponent("reference_\(name).json"))
        return try JSONDecoder().decode(Reference.self, from: data).cases
    }
    private func media(_ sub: String, _ file: String) throws -> Data {
        try Data(contentsOf: Self.fixtureDir().appendingPathComponent("\(sub)/\(file)"))
    }
    private func weightsDir() throws -> URL {
        var candidates: [String] = []
        if let e = ProcessInfo.processInfo.environment["KRILL_EG2_DIR"] { candidates.append(e) }
        candidates.append(NSHomeDirectory() + "/.krill/models/embeddinggemma-2")
        for c in candidates where FileManager.default.fileExists(atPath: c + "/model.safetensors") {
            return URL(fileURLWithPath: c)
        }
        throw XCTSkip("EmbeddingGemma 2 weights not present (set KRILL_EG2_DIR)")
    }
    private func cosine(_ a: [Float], _ b: [Float]) -> Double {
        var d = 0.0, na = 0.0, nb = 0.0
        for i in 0 ..< min(a.count, b.count) {
            d += Double(a[i] * b[i]); na += Double(a[i] * a[i]); nb += Double(b[i] * b[i])
        }
        return d / (na.squareRoot() * nb.squareRoot())
    }

    private func format(of file: String) -> String { (file as NSString).pathExtension }

    /// Audio cases (+ mixed text/audio/image) only; video is covered by its own test file.
    private func audioCases() throws -> [Case] {
        let all = try reference("audio") + reference("audio_extra") + reference("mixed_av")
        return all.filter { c in !c.parts.contains { $0.type == "video" } }
    }

    private func input(_ c: Case) throws -> EmbeddingInput {
        EmbeddingInput(parts: try c.parts.map { p in
            switch p.type {
            case "text": return .text(p.text ?? "")
            case "image": return .image(try media("images", p.file ?? ""))
            case "audio": return .audio(try media("audio", p.file ?? ""), format: format(of: p.file ?? ""))
            default: throw XCTSkip("part type \(p.type) not covered here")
            }
        })
    }

    /// `a8.m4a`: the reference decoder (ffmpeg) returns 94208 samples (it keeps the AAC
    /// priming), AVFoundation applies the edit list and returns 93520, so the clip is
    /// 1 soft token shorter. The vector still passes the gate; ids / prompt-token count
    /// are asserted instead on the ffmpeg-decoded samples (`a8_m4a_ffmpeg16k.wav`) in
    /// `testReferenceDecodedSamplesReachIdenticalIdsAndParity`.
    static let decoderLengthDiffers: Set<String> = ["a8.m4a"]

    // MARK: Needs tokenizer.json only

    func testSequenceIdsEqualWhatSentenceTransformersFed() throws {
        let dir = try weightsDir()
        let tok = try CodePointBPETokenizer(directory: dir)
        let builder = EG2SequenceBuilder(tokenizer: tok, tokens: .checkpointDefaults)
        let prompts = EmbeddingPromptTable.load(directory: dir)
        for c in try audioCases() {
            var segs: [EG2Segment] = []
            for p in c.parts {
                switch p.type {
                case "text": segs.append(.text(p.text!))
                case "audio":
                    let w = try EG2AudioPreprocessor.decode(try media("audio", p.file!), formatHint: format(of: p.file!))
                    segs.append(.media(EG2MediaBlock(.audio, softTokensPerBlock: try EG2AudioPreprocessor.softTokens(forWaveform: w))))
                case "image":
                    let img = try EG2ImagePreprocessor.prepare(try media("images", p.file!))
                    segs.append(.media(EG2MediaBlock(.image, softTokensPerBlock: img.softTokens)))
                default: XCTFail(p.type)
                }
            }
            let prefix = c.task.flatMap { prompts?.prefix(for: $0) } ?? ""
            let s = try builder.build(segs, prefix: prefix)
            let want = c.input_ids_rle.flatMap { [Int](repeating: $0[0], count: $0[1]) }
            if Self.decoderLengthDiffers.contains(c.name) { continue }  // see the reference-decoded test below
            XCTAssertEqual(s.ids.map { Int($0) }, want, c.name)
            XCTAssertEqual(s.count, c.n_tokens, c.name)
        }
    }

    // MARK: Needs weights

    private func parity(dtype: String, minCos: Double) async throws {
        let dir = try weightsDir()
        setenv("KRILL_EMBED_DTYPE", dtype, 1)
        defer { unsetenv("KRILL_EMBED_DTYPE") }
        let engine = EmbeddingEngine()
        try await engine.load(directory: dir)
        XCTAssertFalse(engine.isAudioTowerLoaded, "audio tower must load lazily")
        var rows: [String] = []
        var worst = 1.0
        for c in try audioCases() {
            let r = try engine.embed(inputs: [try input(c)], options: EmbeddingRequestOptions(task: c.task))
            if !Self.decoderLengthDiffers.contains(c.name) { XCTAssertEqual(r.promptTokens, c.n_tokens, c.name) }
            XCTAssertEqual(r.vectors[0].count, 768)
            let cos = cosine(r.vectors[0], c.embedding)
            worst = min(worst, cos)
            rows.append(String(format: "%-34@ %@ cos %.8f tokens %d", c.name as NSString, dtype as NSString, cos, r.promptTokens))
            XCTAssertGreaterThanOrEqual(cos, minCos, "\(dtype) \(c.name)")
        }
        print("EG2 audio parity (\(dtype)) worst \(worst)\n" + rows.joined(separator: "\n"))
        XCTAssertTrue(engine.isAudioTowerLoaded)
    }

    func testAudioAndMixedParityFloat32Above0999() async throws {
        try await parity(dtype: "float32", minCos: 0.999)
    }

    func testAudioAndMixedParityBFloat16Above099() async throws {
        try await parity(dtype: "bfloat16", minCos: 0.99)
    }

    /// Proof that the m4a gap is the decoder, not the model path: the same clip decoded
    /// by ffmpeg (as the reference does) reaches the reference's ids and vector.
    func testReferenceDecodedSamplesReachIdenticalIdsAndParity() async throws {
        let dir = try weightsDir()
        let engine = EmbeddingEngine()
        try await engine.load(directory: dir)
        let c = try reference("audio_extra").first { $0.name == "a8.m4a" }!
        let item = EmbeddingInput(parts: [.audio(try media("audio", "a8_m4a_ffmpeg16k.wav"), format: "wav")])
        let r = try engine.embed(inputs: [item])
        XCTAssertEqual(r.promptTokens, c.n_tokens)
        let cos = cosine(r.vectors[0], c.embedding)
        print("EG2 m4a with ffmpeg-decoded samples: cos \(cos) tokens \(r.promptTokens)")
        XCTAssertGreaterThanOrEqual(cos, 0.999)
    }

    func testAudioTowerIsLazyAndSeparateFromVision() async throws {
        let dir = try weightsDir()
        let engine = EmbeddingEngine()
        try await engine.load(directory: dir)
        _ = try engine.embed(["hello world"])
        XCTAssertFalse(engine.isAudioTowerLoaded)
        XCTAssertFalse(engine.isVisionTowerLoaded)
        let c = try reference("audio").first { $0.name == "a3.wav" }!
        _ = try engine.embed(inputs: [try input(c)])
        XCTAssertTrue(engine.isAudioTowerLoaded)
        XCTAssertFalse(engine.isVisionTowerLoaded, "an audio request must not load the vision tower")
    }

    func testBatchOfMixedItemsKeepsOrderAndMatchesSolo() async throws {
        let dir = try weightsDir()
        let engine = EmbeddingEngine()
        try await engine.load(directory: dir)
        let cases = try reference("audio")
        let a3 = try input(cases.first { $0.name == "a3.wav" }!)
        let a8 = try input(cases.first { $0.name == "a8.wav" }!)
        let items = [EmbeddingInput(text: "a plain sentence"), a3, a8]
        let r = try engine.embed(inputs: items)
        XCTAssertEqual(r.vectors.count, 3)
        XCTAssertGreaterThan(cosine(r.vectors[1], try engine.embed(inputs: [a3]).vectors[0]), 0.99999)
        XCTAssertGreaterThan(cosine(r.vectors[0], try engine.embed(["a plain sentence"]).vectors[0]), 0.99999)
        XCTAssertLessThan(cosine(r.vectors[1], r.vectors[2]), 0.999)
    }

    func testMRLAndTaskOnAudioItems() async throws {
        let dir = try weightsDir()
        let engine = EmbeddingEngine()
        try await engine.load(directory: dir)
        let c = try reference("mixed_av").first { $0.name == "text_then_audio" }!
        let r = try engine.embed(inputs: [try input(c)], options: EmbeddingRequestOptions(dimensions: 256))
        XCTAssertEqual(r.vectors[0].count, 256)
        XCTAssertEqual(r.vectors[0].reduce(0) { $0 + $1 * $1 }, 1, accuracy: 1e-4)
        XCTAssertThrowsError(try engine.embed(inputs: [try input(c)], options: EmbeddingRequestOptions(dimensions: 100)))
    }

    // MARK: Limits

    func testOverlongAudioInputIsRejectedNotTruncated() async throws {
        let dir = try weightsDir()
        let engine = EmbeddingEngine()
        try await engine.load(directory: dir)
        let a20 = try media("audio", "a20.wav")  // 585 tokens each
        let item = EmbeddingInput(parts: [EmbeddingPart](repeating: .audio(a20, format: "wav"), count: 15))
        XCTAssertThrowsError(try engine.embed(inputs: [item])) {
            guard case EmbeddingError.invalidOption(let m) = $0 else { return XCTFail("\($0)") }
            XCTAssertTrue(m.contains("8192"), m)
        }
        XCTAssertFalse(engine.isAudioTowerLoaded, "the length check happens before the tower is loaded")
    }

    func testAudioOver30SecondsIsRejectedWithTheLimit() async throws {
        let dir = try weightsDir()
        let engine = EmbeddingEngine()
        try await engine.load(directory: dir)
        // 31 s of silence, 16 kHz mono PCM16 WAV
        let n = 31 * 16_000
        var d = Data()
        func u32(_ v: Int) { var x = UInt32(v).littleEndian; d.append(Data(bytes: &x, count: 4)) }
        func u16(_ v: Int) { var x = UInt16(v).littleEndian; d.append(Data(bytes: &x, count: 2)) }
        d.append(Data("RIFF".utf8)); u32(36 + n * 2); d.append(Data("WAVEfmt ".utf8))
        u32(16); u16(1); u16(1); u32(16_000); u32(32_000); u16(2); u16(16)
        d.append(Data("data".utf8)); u32(n * 2); d.append(Data(count: n * 2))
        XCTAssertThrowsError(try engine.embed(inputs: [EmbeddingInput(parts: [.audio(d, format: "wav")])])) {
            guard case EmbeddingError.invalidOption(let m) = $0 else { return XCTFail("\($0)") }
            XCTAssertTrue(m.contains("maximum is 30 s"), m)
        }
        XCTAssertThrowsError(try engine.embed(inputs: [EmbeddingInput(parts: [.audio(Data("junk".utf8), format: "wav")])])) {
            guard case EmbeddingError.invalidOption = $0 else { return XCTFail("\($0)") }
        }
    }

    // MARK: Text-only models refuse audio

    func testTextOnlyModelRejectsAudioParts() async throws {
        guard let d = ProcessInfo.processInfo.environment["KRILL_TEXT_ONLY_EMBED_DIR"],
              FileManager.default.fileExists(atPath: d + "/config.json") else {
            throw XCTSkip("set KRILL_TEXT_ONLY_EMBED_DIR to any non-EmbeddingGemma-2 embedding model dir")
        }
        let engine = EmbeddingEngine()
        try await engine.load(directory: URL(fileURLWithPath: d))
        let item = EmbeddingInput(parts: [.text("x"), .audio(try media("audio", "a3.wav"), format: "wav")])
        XCTAssertThrowsError(try engine.embed(inputs: [item]))
    }
}
