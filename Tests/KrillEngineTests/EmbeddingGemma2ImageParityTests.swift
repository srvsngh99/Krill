import XCTest
import Foundation
import KrillCore
import MLX
import KrillTokenizer
@testable import KrillEngine

/// EmbeddingGemma 2 image / mixed-input parity against fp32
/// sentence-transformers 6.1 (CPU) vectors, plus the sequence ids the reference
/// actually fed the model. Fixtures: `Fixtures/eg2_mm/` (see its README for how
/// every file was made).
///
/// Weight-free tests (preprocessor grid vs the reference) always run. The rest
/// skip when the weights are absent: point `KRILL_EG2_DIR` at a directory with
/// the checkpoint, or install `embeddinggemma-2` in ~/.krill/models.
/// Gate: cosine >= 0.999 per input in fp32. bf16 is gated at >= 0.99; the
/// measured floor is in docs/EMBEDDINGGEMMA2.md.
final class EmbeddingGemma2ImageParityTests: XCTestCase {
    struct Part: Decodable { let type: String; let file: String?; let text: String? }
    struct Grid: Decodable { let patches: Int; let pW: Int; let pH: Int; let soft_tokens: Int }
    struct Case: Decodable {
        let name: String
        let modality: String
        let task: String?
        let parts: [Part]
        let input_ids_rle: [[Int]]
        let n_tokens: Int
        let embedding: [Float]
        let grid: Grid?
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

    private func input(_ c: Case) throws -> EmbeddingInput {
        EmbeddingInput(parts: try c.parts.map { p in
            switch p.type {
            case "text": return .text(p.text ?? "")
            case "image": return .image(try media("images", p.file ?? ""))
            default: throw XCTSkip("part type \(p.type) not covered here")
            }
        })
    }

    // MARK: Weight-free

    func testPreprocessorGridMatchesReferenceForEveryImage() throws {
        for c in try reference("image") {
            let f = c.parts[0].file!
            let p = try EG2ImagePreprocessor.prepare(try media("images", f))
            let g = c.grid!
            XCTAssertEqual(p.gridW, g.pW, f)
            XCTAssertEqual(p.gridH, g.pH, f)
            XCTAssertEqual(p.numPatches, g.patches, f)
            XCTAssertEqual(p.softTokens, g.soft_tokens, f)
        }
    }

    // MARK: Needs tokenizer.json only

    func testSequenceIdsEqualWhatSentenceTransformersFed() throws {
        let dir = try weightsDir()
        let tok = try CodePointBPETokenizer(directory: dir)
        let builder = EG2SequenceBuilder(tokenizer: tok, tokens: .checkpointDefaults)
        let prompts = EmbeddingPromptTable.load(directory: dir)
        for c in try reference("image") + reference("mixed") {
            var segs: [EG2Segment] = []
            for p in c.parts {
                if p.type == "text" { segs.append(.text(p.text!)) }
                else {
                    let img = try EG2ImagePreprocessor.prepare(try media("images", p.file!))
                    segs.append(.media(EG2MediaBlock(.image, softTokensPerBlock: img.softTokens)))
                }
            }
            let prefix = c.task.flatMap { prompts?.prefix(for: $0) } ?? ""
            let s = try builder.build(segs, prefix: prefix)
            let want = c.input_ids_rle.flatMap { [Int](repeating: $0[0], count: $0[1]) }
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
        XCTAssertFalse(engine.isVisionTowerLoaded, "vision tower must load lazily")
        var rows: [String] = []
        var worst = 1.0
        for c in try reference("image") + reference("mixed") {
            let r = try engine.embed(inputs: [try input(c)],
                                     options: EmbeddingRequestOptions(task: c.task))
            XCTAssertEqual(r.promptTokens, c.n_tokens, c.name)
            XCTAssertEqual(r.vectors[0].count, 768)
            let cos = cosine(r.vectors[0], c.embedding)
            worst = min(worst, cos)
            rows.append(String(format: "%-34@ %@ cos %.8f", c.name as NSString, dtype as NSString, cos))
            XCTAssertGreaterThanOrEqual(cos, minCos, "\(dtype) \(c.name)")
        }
        print("EG2 image parity (\(dtype)) worst \(worst)\n" + rows.joined(separator: "\n"))
        XCTAssertTrue(engine.isVisionTowerLoaded)
    }

    func testImageAndMixedParityFloat32Above0999() async throws {
        try await parity(dtype: "float32", minCos: 0.999)
    }

    func testImageAndMixedParityBFloat16Above099() async throws {
        try await parity(dtype: "bfloat16", minCos: 0.99)
    }

    func testBatchOfMixedItemsKeepsOrderAndMatchesSolo() async throws {
        let dir = try weightsDir()
        let engine = EmbeddingEngine()
        try await engine.load(directory: dir)
        let imgs = try reference("image")
        let items = [EmbeddingInput(text: "a plain sentence"), try input(imgs[0]),
                     EmbeddingInput(text: "another"), try input(imgs[3])]
        let r = try engine.embed(inputs: items)
        XCTAssertEqual(r.vectors.count, 4)
        let solo = try engine.embed(inputs: [items[1]]).vectors[0]
        XCTAssertGreaterThan(cosine(r.vectors[1], solo), 0.99999)
        let text = try engine.embed(["a plain sentence", "another"]).vectors
        XCTAssertGreaterThan(cosine(r.vectors[0], text[0]), 0.99999)
        XCTAssertGreaterThan(cosine(r.vectors[2], text[1]), 0.99999)
        // image items are not text items
        XCTAssertLessThan(cosine(r.vectors[1], r.vectors[0]), 0.95)
    }

    func testTextOnlyRequestNeverLoadsTheVisionTower() async throws {
        let dir = try weightsDir()
        let engine = EmbeddingEngine()
        try await engine.load(directory: dir)
        _ = try engine.embed(["hello world"])
        _ = try engine.embed(inputs: [EmbeddingInput(text: "hello"), EmbeddingInput(text: "world")])
        XCTAssertFalse(engine.isVisionTowerLoaded)
    }

    func testMRLAndTaskOnMediaItems() async throws {
        let dir = try weightsDir()
        let engine = EmbeddingEngine()
        try await engine.load(directory: dir)
        let c = try reference("mixed")[0]
        let r = try engine.embed(inputs: [try input(c)], options: EmbeddingRequestOptions(dimensions: 256))
        XCTAssertEqual(r.vectors[0].count, 256)
        XCTAssertEqual(r.vectors[0].reduce(0) { $0 + $1 * $1 }, 1, accuracy: 1e-4)
        XCTAssertThrowsError(try engine.embed(inputs: [try input(c)], options: EmbeddingRequestOptions(dimensions: 100)))
    }

    func testOverlongMediaInputIsRejectedNotTruncated() async throws {
        let dir = try weightsDir()
        let engine = EmbeddingEngine()
        try await engine.load(directory: dir)
        let png = try media("images", "img_tiny_100x37.png")
        // 31 images x 272 tokens = 8432 > 8192
        let item = EmbeddingInput(parts: [EmbeddingPart](repeating: .image(png), count: 31))
        XCTAssertThrowsError(try engine.embed(inputs: [item])) {
            guard case EmbeddingError.invalidOption(let m) = $0 else { return XCTFail("\($0)") }
            XCTAssertTrue(m.contains("8192"), m)
        }
    }

    // MARK: Text-only models refuse media

    func testTextOnlyModelRejectsImageParts() async throws {
        guard let d = ProcessInfo.processInfo.environment["KRILL_TEXT_ONLY_EMBED_DIR"],
              FileManager.default.fileExists(atPath: d + "/config.json") else {
            throw XCTSkip("set KRILL_TEXT_ONLY_EMBED_DIR to any non-EmbeddingGemma-2 embedding model dir")
        }
        let engine = EmbeddingEngine()
        try await engine.load(directory: URL(fileURLWithPath: d))
        let item = EmbeddingInput(parts: [.text("x"), .image(try media("images", "img_tiny_100x37.png"))])
        XCTAssertThrowsError(try engine.embed(inputs: [item])) {
            guard case EmbeddingError.invalidOption = $0 else { return XCTFail("\($0)") }
        }
        // plain strings still work, via the unchanged path
        XCTAssertFalse(try engine.embed(inputs: [EmbeddingInput(text: "hello")]).vectors[0].isEmpty)
    }
}
