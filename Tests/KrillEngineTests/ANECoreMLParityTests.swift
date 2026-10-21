import XCTest
import Foundation
@testable import KrillEngine

#if canImport(CoreML) && arch(arm64)
import CoreML

/// Parity of the real Core ML path (host gather -> compiled bucket -> prediction) against PyTorch.
/// The fixture is a tiny seeded random-weight model (Fixtures/ane_tiny, regenerate with
/// tools/ane/make_test_fixture.py): a 3-layer hidden-64 stack with S=16 and S=32 fp16 buckets and
/// the PyTorch fp32 output for several token sequences. Compute units are CPU-only so CI runners
/// without a Neural Engine run it too; the graph is identical.
final class ANECoreMLParityTests: XCTestCase {
    private struct Ref: Decodable { let ids: [Int32]; let bucket: Int; let vector: [Float] }

    private func fixtureDir(file: StaticString = #filePath) -> URL {
        URL(fileURLWithPath: "\(file)").deletingLastPathComponent().appendingPathComponent("Fixtures/ane_tiny")
    }

    private func cosine(_ a: [Float], _ b: [Float]) -> Float {
        var d: Float = 0, x: Float = 0, y: Float = 0
        for i in 0 ..< a.count { d += a[i] * b[i]; x += a[i] * a[i]; y += b[i] * b[i] }
        return d / (x.squareRoot() * y.squareRoot())
    }

    func testCoreMLMatchesPyTorchReference() throws {
        let dir = fixtureDir()
        let manifest = try ANEPackageManifest.load(directory: dir)
        let refs = try JSONDecoder().decode([Ref].self, from: Data(contentsOf: dir.appendingPathComponent("reference.json")))
        XCTAssertFalse(refs.isEmpty)
        let embedder = try ANEEmbedder.coreML(directory: dir, manifest: manifest, units: .cpuOnly)
        for r in refs {
            let (v, bucket) = try embedder.embed(tokenIDs: r.ids)
            XCTAssertEqual(bucket, r.bucket, "bucket choice for \(r.ids.count) tokens")
            XCTAssertEqual(v.count, manifest.outputDim)
            XCTAssertGreaterThan(cosine(v, r.vector), 0.999, "\(r.ids.count) tokens, bucket \(r.bucket)")
        }
        XCTAssertEqual(embedder.loadedBuckets, [16, 32], "both buckets were needed and loaded lazily")
    }

    func testRepeatedPredictionsAreStable() throws {
        let dir = fixtureDir()
        let embedder = try ANEEmbedder.coreML(directory: dir, manifest: try ANEPackageManifest.load(directory: dir),
                                              units: .cpuOnly)
        let first = try embedder.embed(tokenIDs: [5, 6, 7, 8]).vector
        for _ in 0 ..< 200 { XCTAssertEqual(try embedder.embed(tokenIDs: [5, 6, 7, 8]).vector, first) }
        // The reused input buffers must not leak the previous (longer) request into a shorter one.
        _ = try embedder.embed(tokenIDs: Array(repeating: 9, count: 16))
        XCTAssertEqual(try embedder.embed(tokenIDs: [5, 6, 7, 8]).vector, first)
    }

    func testConcurrentCoreMLPredictions() throws {
        let dir = fixtureDir()
        let embedder = try ANEEmbedder.coreML(directory: dir, manifest: try ANEPackageManifest.load(directory: dir),
                                              units: .cpuOnly)
        let want = try embedder.embed(tokenIDs: [3, 4, 5]).vector
        let bad = NSLock()
        nonisolated(unsafe) var mismatches = 0
        DispatchQueue.concurrentPerform(iterations: 8) { _ in
            for _ in 0 ..< 40 {
                if (try? embedder.embed(tokenIDs: [3, 4, 5]).vector) != want { bad.lock(); mismatches += 1; bad.unlock() }
            }
        }
        XCTAssertEqual(mismatches, 0)
    }
}
#endif
