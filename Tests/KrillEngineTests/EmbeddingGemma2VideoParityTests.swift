import XCTest
import Foundation
import KrillCore
import MLX
import KrillTokenizer
@testable import KrillEngine

/// EmbeddingGemma 2 video (and mixed text+video) parity against fp32
/// sentence-transformers 6.1 (CPU, torchcodec/ffmpeg decode) vectors, plus the
/// sequence ids the reference fed the model. Fixtures: `Fixtures/eg2_mm/`.
/// Weighted tests skip when the checkpoint is absent (`KRILL_EG2_DIR`).
/// Gate: cosine >= 0.999 in fp32; bf16 gated at >= 0.99 (floor in the docs).
final class EmbeddingGemma2VideoParityTests: XCTestCase {
    struct Part: Decodable { let type: String; let file: String?; let text: String? }
    struct Case: Decodable {
        let name: String
        let modality: String
        let task: String?
        let parts: [Part]
        let input_ids_rle: [[Int]]
        let n_tokens: Int
        let embedding: [Float]
        let video_soft_tokens_total: Int?
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

    private func videoCases() throws -> [Case] {
        let all = try reference("video") + reference("video_extra") + reference("mixed_av")
        return all.filter { $0.parts.contains { $0.type == "video" } }
    }

    private func input(_ c: Case) throws -> EmbeddingInput {
        EmbeddingInput(parts: try c.parts.map { p in
            switch p.type {
            case "text": return .text(p.text ?? "")
            case "video": return .video(try media("video", p.file ?? ""), format: "mp4")
            default: throw XCTSkip("part type \(p.type) not covered here")
            }
        })
    }

    private func segments(_ c: Case) throws -> [EG2Segment] {
        var segs: [EG2Segment] = []
        for p in c.parts {
            if p.type == "text" { segs.append(.text(p.text!)); continue }
            let src = try EG2VideoSource(data: try media("video", p.file!))
            let idx = EG2VideoSampler.frameIndices(totalFrames: src.info.totalFrames, fps: src.info.fps,
                                                   duration: src.info.duration)
            let n = try EG2ImagePreprocessor.softTokens(width: src.info.width, height: src.info.height,
                                                        maxSoftTokens: EG2ImagePreprocessor.videoFrameSoftTokens)
            segs.append(.media(EG2MediaBlock(.video, softTokensPerBlock: n, blocks: idx.count)))
        }
        return segs
    }

    // MARK: Needs tokenizer.json only

    func testSequenceIdsEqualWhatSentenceTransformersFed() throws {
        let dir = try weightsDir()
        let tok = try CodePointBPETokenizer(directory: dir)
        let builder = EG2SequenceBuilder(tokenizer: tok, tokens: .checkpointDefaults)
        let prompts = EmbeddingPromptTable.load(directory: dir)
        for c in try videoCases() {
            let prefix = c.task.flatMap { prompts?.prefix(for: $0) } ?? ""
            let s = try builder.build(try segments(c), prefix: prefix)
            let want = c.input_ids_rle.flatMap { [Int](repeating: $0[0], count: $0[1]) }
            XCTAssertEqual(s.ids.map { Int($0) }, want, c.name)
            XCTAssertEqual(s.count, c.n_tokens, c.name)
            XCTAssertEqual(s.softTokenCount(.video), c.video_soft_tokens_total, c.name)
        }
    }

    // MARK: Needs weights

    private func parity(dtype: String, minCos: Double) async throws {
        let dir = try weightsDir()
        setenv("KRILL_EMBED_DTYPE", dtype, 1)
        defer { unsetenv("KRILL_EMBED_DTYPE") }
        let engine = EmbeddingEngine()
        try await engine.load(directory: dir)
        XCTAssertFalse(engine.isVisionTowerLoaded, "vision tower must load lazily")
        var rows: [String] = []
        var worst = 1.0
        for c in try videoCases() {
            let t0 = CFAbsoluteTimeGetCurrent()
            let r = try engine.embed(inputs: [try input(c)], options: EmbeddingRequestOptions(task: c.task))
            let ms = (CFAbsoluteTimeGetCurrent() - t0) * 1000
            XCTAssertEqual(r.promptTokens, c.n_tokens, c.name)
            XCTAssertEqual(r.vectors[0].count, 768)
            let cos = cosine(r.vectors[0], c.embedding)
            worst = min(worst, cos)
            rows.append(String(format: "%-34@ %@ cos %.8f tokens %d  %.0f ms", c.name as NSString, dtype as NSString,
                               cos, r.promptTokens, ms))
            XCTAssertGreaterThanOrEqual(cos, minCos, "\(dtype) \(c.name)")
        }
        print("EG2 video parity (\(dtype)) worst \(worst)\n" + rows.joined(separator: "\n"))
        XCTAssertTrue(engine.isVisionTowerLoaded)
        XCTAssertFalse(engine.isAudioTowerLoaded, "a video request must not load the audio tower")
    }

    func testVideoAndMixedParityFloat32Above0999() async throws {
        try await parity(dtype: "float32", minCos: 0.999)
    }

    func testVideoAndMixedParityBFloat16Above099() async throws {
        try await parity(dtype: "bfloat16", minCos: 0.99)
    }

    func testBatchOfMixedItemsKeepsOrderAndMatchesSolo() async throws {
        let dir = try weightsDir()
        let engine = EmbeddingEngine()
        try await engine.load(directory: dir)
        let v3 = try input(try reference("video").first { $0.name == "v3.mp4" }!)
        let items = [EmbeddingInput(text: "a plain sentence"), v3]
        let r = try engine.embed(inputs: items)
        XCTAssertEqual(r.vectors.count, 2)
        XCTAssertGreaterThan(cosine(r.vectors[1], try engine.embed(inputs: [v3]).vectors[0]), 0.99999)
        XCTAssertGreaterThan(cosine(r.vectors[0], try engine.embed(["a plain sentence"]).vectors[0]), 0.99999)
    }

    func testVideoWithAudioTrackEqualsItsSilentTwin() async throws {
        // The audio track is never read: the reference vector (made from the same file by
        // torchcodec, which ignores audio) is matched by `testVideoAndMixedParity...`; here we
        // also check no audio tower is touched.
        let dir = try weightsDir()
        let engine = EmbeddingEngine()
        try await engine.load(directory: dir)
        _ = try engine.embed(inputs: [EmbeddingInput(parts: [.video(try media("video", "v_audio.mp4"), format: "mp4")])])
        XCTAssertFalse(engine.isAudioTowerLoaded)
        XCTAssertTrue(engine.isVisionTowerLoaded)
    }

    // MARK: Limits

    func testOverlongVideoInputIsRejectedNotTruncated() async throws {
        let dir = try weightsDir()
        let engine = EmbeddingEngine()
        try await engine.load(directory: dir)
        // v40 is 4226 tokens; two of them are 8452 > 8192.
        let v40 = try media("video", "v40.mp4")
        let item = EmbeddingInput(parts: [.video(v40, format: "mp4"), .video(v40, format: "mp4")])
        XCTAssertThrowsError(try engine.embed(inputs: [item])) {
            guard case EmbeddingError.invalidOption(let m) = $0 else { return XCTFail("\($0)") }
            XCTAssertTrue(m.contains("8192"), m)
            XCTAssertTrue(m.contains("32 video frames"), m)
        }
        XCTAssertFalse(engine.isVisionTowerLoaded, "the length check happens before the tower is loaded")
    }

    func testUndecodableVideoIsAClientError() async throws {
        let dir = try weightsDir()
        let engine = EmbeddingEngine()
        try await engine.load(directory: dir)
        XCTAssertThrowsError(try engine.embed(inputs: [EmbeddingInput(parts: [.video(Data("junk".utf8), format: "mp4")])])) {
            guard case EmbeddingError.invalidOption(let m) = $0 else { return XCTFail("\($0)") }
            XCTAssertTrue(m.hasPrefix("video:"), m)
        }
    }

    // MARK: Text-only models refuse video

    func testTextOnlyModelRejectsVideoParts() async throws {
        guard let d = ProcessInfo.processInfo.environment["KRILL_TEXT_ONLY_EMBED_DIR"],
              FileManager.default.fileExists(atPath: d + "/config.json") else {
            throw XCTSkip("set KRILL_TEXT_ONLY_EMBED_DIR to any non-EmbeddingGemma-2 embedding model dir")
        }
        let engine = EmbeddingEngine()
        try await engine.load(directory: URL(fileURLWithPath: d))
        let item = EmbeddingInput(parts: [.text("x"), .video(try media("video", "v3.mp4"), format: "mp4")])
        XCTAssertThrowsError(try engine.embed(inputs: [item]))
    }
}
