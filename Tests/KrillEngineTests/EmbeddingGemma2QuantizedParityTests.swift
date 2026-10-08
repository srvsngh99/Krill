import XCTest
import Foundation
import KrillCore
import MLX
@testable import KrillEngine

/// Quantized EmbeddingGemma 2 builds against the same fp32 sentence-transformers
/// vectors the bf16 parity tests use: text (`Fixtures/eg2_reference.json`, raw
/// and `Document`) plus every multimodal reference (`Fixtures/eg2_mm/`).
///
/// Skips unless `KRILL_EG2_MXFP8_DIR` / `KRILL_EG2_NVFP4_DIR` points at a built
/// folder (see "Quantized builds" in docs/EMBEDDINGGEMMA2.md). Gates are the
/// recipe targets, per modality: mxfp8 mean >= 0.995 and min >= 0.99; nvfp4 mean
/// >= 0.97 and min >= 0.95.
final class EmbeddingGemma2QuantizedParityTests: XCTestCase {
    private struct Part: Decodable { let type: String; let file: String?; let text: String? }
    private struct Case: Decodable { let name: String; let task: String?; let parts: [Part]; let embedding: [Float] }
    private struct Reference: Decodable { let cases: [Case] }
    private struct TextFixture: Decodable { let strings: [String]; let emb_raw: [[Float]]; let emb_Document: [[Float]] }

    private static func fixtures(_ file: StaticString = #filePath) -> URL {
        URL(fileURLWithPath: "\(file)").deletingLastPathComponent().appendingPathComponent("Fixtures")
    }

    private func dir(_ env: String) throws -> URL {
        guard let p = ProcessInfo.processInfo.environment[env],
              FileManager.default.fileExists(atPath: p + "/model.safetensors") else {
            throw XCTSkip("quantized EmbeddingGemma 2 build not present (set \(env))")
        }
        return URL(fileURLWithPath: p)
    }

    private func cosine(_ a: [Float], _ b: [Float]) -> Double {
        var d = 0.0, na = 0.0, nb = 0.0
        for i in 0 ..< min(a.count, b.count) {
            d += Double(a[i] * b[i]); na += Double(a[i] * a[i]); nb += Double(b[i] * b[i])
        }
        return d / (na.squareRoot() * nb.squareRoot())
    }

    private func input(_ c: Case) throws -> EmbeddingInput {
        let mm = Self.fixtures().appendingPathComponent("eg2_mm")
        func media(_ sub: String, _ f: String) throws -> Data { try Data(contentsOf: mm.appendingPathComponent("\(sub)/\(f)")) }
        return EmbeddingInput(parts: try c.parts.map { p in
            let f = p.file ?? ""
            switch p.type {
            case "text": return .text(p.text ?? "")
            case "image": return .image(try media("images", f))
            case "audio": return .audio(try media("audio", f), format: (f as NSString).pathExtension)
            case "video": return .video(try media("video", f), format: "mp4")
            default: throw XCTSkip("part type \(p.type)")
            }
        })
    }

    private func run(env: String, minMean: Double, minMin: Double) async throws {
        let engine = EmbeddingEngine()
        try await engine.load(directory: try dir(env))
        var groups: [String: [Double]] = [:]

        let t = try JSONDecoder().decode(
            TextFixture.self, from: Data(contentsOf: Self.fixtures().appendingPathComponent("eg2_reference.json")))
        for (task, ref) in [(nil as String?, t.emb_raw), ("Document", t.emb_Document)] {
            let r = try engine.embed(t.strings, options: EmbeddingRequestOptions(task: task))
            for i in t.strings.indices { groups["text", default: []].append(cosine(r.vectors[i], ref[i])) }
        }
        let groupOf = ["image": "image", "mixed": "mixed", "mixed_av": "mixed", "audio": "audio",
                       "audio_extra": "audio", "video": "video", "video_extra": "video"]
        for (file, group) in groupOf.sorted(by: { $0.key < $1.key }) {
            let data = try Data(contentsOf: Self.fixtures().appendingPathComponent("eg2_mm/reference_\(file).json"))
            for c in try JSONDecoder().decode(Reference.self, from: data).cases {
                let r = try engine.embed(inputs: [try input(c)], options: EmbeddingRequestOptions(task: c.task))
                groups[group, default: []].append(cosine(r.vectors[0], c.embedding))
            }
        }
        XCTAssertEqual(Set(groups.keys), ["text", "image", "mixed", "audio", "video"])
        for (g, cs) in groups.sorted(by: { $0.key < $1.key }) {
            let mean = cs.reduce(0, +) / Double(cs.count), mn = cs.min()!
            print(String(format: "EG2 %@ %@ n=%d min %.5f mean %.5f", env as NSString, g as NSString, cs.count, mn, mean))
            XCTAssertGreaterThanOrEqual(mean, minMean, "\(env) \(g) mean")
            XCTAssertGreaterThanOrEqual(mn, minMin, "\(env) \(g) min")
        }
    }

    func testMXFP8Parity() async throws { try await run(env: "KRILL_EG2_MXFP8_DIR", minMean: 0.995, minMin: 0.99) }
    func testNVFP4Parity() async throws { try await run(env: "KRILL_EG2_NVFP4_DIR", minMean: 0.97, minMin: 0.95) }

    /// A quantized build still serves the two-step MRL path and rejects bad options.
    func testQuantizedMRLAndOptions() async throws {
        let engine = EmbeddingEngine()
        let d: URL
        if let n = try? dir("KRILL_EG2_NVFP4_DIR") { d = n } else { d = try dir("KRILL_EG2_MXFP8_DIR") }
        try await engine.load(directory: d)
        let m = try engine.embed(["hello"], options: EmbeddingRequestOptions(dimensions: 256))
        XCTAssertEqual(m.vectors[0].count, 256)
        XCTAssertEqual(m.vectors[0].reduce(0) { $0 + $1 * $1 }, 1, accuracy: 1e-4)
        XCTAssertThrowsError(try engine.embed(["x"], options: EmbeddingRequestOptions(dimensions: 100)))
    }
}
