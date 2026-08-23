import Foundation
import XCTest
@testable import KrillEngine

/// Model-free regression for the dense generation contract. A loaded checkpoint
/// is not available in unit tests, so this pins every terminal yield in the
/// production dense path to a preceding stats publication.
final class InferenceEngineStatsOrderingTests: XCTestCase {
    func testDensePathPublishesStatsBeforeEveryTerminalEvent() throws {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // KrillEngineTests
            .deletingLastPathComponent() // Tests
            .deletingLastPathComponent() // repository root
        let sourceURL = root.appendingPathComponent("Sources/KrillEngine/InferenceEngine.swift")
        let source = try String(contentsOf: sourceURL, encoding: .utf8)

        guard let start = source.range(of: "// -- Decode loop --"),
              let end = source.range(
                of: "return (stream, { statsHolder.stats })", range: start.upperBound..<source.endIndex)
        else {
            return XCTFail("dense generation block markers changed")
        }
        let dense = source[start.lowerBound..<end.lowerBound]
        let marker = "isEnd: true"
        var searchStart = dense.startIndex
        var terminalCount = 0
        while let terminal = dense.range(of: marker, range: searchStart..<dense.endIndex) {
            terminalCount += 1
            let prefixStart = dense.index(
                terminal.lowerBound, offsetBy: -min(700, dense.distance(from: dense.startIndex, to: terminal.lowerBound)))
            let prefix = dense[prefixStart..<terminal.lowerBound]
            XCTAssertTrue(
                prefix.contains("publishStats()"),
                "terminal event \(terminalCount) can become visible before GenerationStats")
            searchStart = terminal.upperBound
        }
        XCTAssertEqual(terminalCount, 7, "audit new/removed dense terminal paths explicitly")
    }
}
