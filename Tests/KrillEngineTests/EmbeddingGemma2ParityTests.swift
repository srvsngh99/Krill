import XCTest
import Foundation
import KrillCore
import MLX
import KrillTokenizer
@testable import KrillEngine

/// Parity of the native EmbeddingGemma 2 text path against fp32
/// sentence-transformers vectors (`Fixtures/eg2_reference.json`: 21 strings -
/// English, French, code, Hindi, Kannada, Sanskrit and one ~3k-token document
/// that exercises the sliding window; raw and `Document`-prompted).
///
/// Skips when the weights are absent. Point `KRILL_EG2_DIR` at a directory
/// holding the checkpoint, or install `embeddinggemma-2` (~/.krill/models).
/// Gate: cosine >= 0.999 per string (fp32). bf16 is only checked >= 0.99
/// (measured floor is documented in docs/EMBEDDINGGEMMA2.md).
final class EmbeddingGemma2ParityTests: XCTestCase {
    private struct Fixture: Decodable {
        let strings: [String]
        let token_ids: [[Int]]
        let emb_raw: [[Float]]
        let emb_Document: [[Float]]
    }

    private func fixture(file: StaticString = #filePath) throws -> Fixture {
        let here = URL(fileURLWithPath: "\(file)").deletingLastPathComponent()
        let data = try Data(contentsOf: here.appendingPathComponent("Fixtures/eg2_reference.json"))
        return try JSONDecoder().decode(Fixture.self, from: data)
    }

    private func weightsDir() throws -> URL {
        let fm = FileManager.default
        var candidates: [String] = []
        if let e = ProcessInfo.processInfo.environment["KRILL_EG2_DIR"] { candidates.append(e) }
        candidates.append(NSHomeDirectory() + "/.krill/models/embeddinggemma-2")
        for c in candidates where fm.fileExists(atPath: c + "/model.safetensors") {
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

    func testTokenizerMatchesHFTokenizers() throws {
        let dir = try weightsDir()
        let f = try fixture()
        let tok = try CodePointBPETokenizer(directory: dir)
        for (i, ids) in f.token_ids.enumerated() {
            XCTAssertEqual(tok.encode(f.strings[i]), ids, "string \(i): \(f.strings[i].prefix(30))")
        }
    }

    private func run(dtype: DType, minCos: Double) async throws {
        let dir = try weightsDir()
        let f = try fixture()
        let engine = EmbeddingEngine()
        setenv("KRILL_EMBED_DTYPE", dtype == .bfloat16 ? "bfloat16" : "float32", 1)
        defer { unsetenv("KRILL_EMBED_DTYPE") }
        try await engine.load(directory: dir)
        for (task, ref) in [(nil as String?, f.emb_raw), ("Document", f.emb_Document)] {
            let r = try engine.embed(f.strings, options: EmbeddingRequestOptions(task: task))
            XCTAssertEqual(r.vectors.count, f.strings.count)
            for i in f.strings.indices {
                let c = cosine(r.vectors[i], ref[i])
                XCTAssertGreaterThanOrEqual(c, minCos, "\(dtype) task=\(task ?? "raw") string \(i)")
            }
        }
        // MRL: truncated + re-normalised vector is a unit vector on the prefix.
        let m = try engine.embed([f.strings[0]], options: EmbeddingRequestOptions(dimensions: 256))
        XCTAssertEqual(m.vectors[0].count, 256)
        XCTAssertEqual(m.vectors[0].reduce(0) { $0 + $1 * $1 }, 1, accuracy: 1e-4)
    }

    func testFloat32ParityAbove0999() async throws { try await run(dtype: .float32, minCos: 0.999) }
    func testBFloat16ParityAbove099() async throws { try await run(dtype: .bfloat16, minCos: 0.99) }

    func testRejectsBadOptions() async throws {
        let dir = try weightsDir()
        let engine = EmbeddingEngine()
        try await engine.load(directory: dir)
        XCTAssertThrowsError(try engine.embed(["x"], options: EmbeddingRequestOptions(dimensions: 100)))
        XCTAssertThrowsError(try engine.embed(["x"], options: EmbeddingRequestOptions(task: "Nope")))
    }
}
