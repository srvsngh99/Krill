import XCTest
import MLX
import KrillRuntime
import KrillTokenizer
@testable import KrillCore

/// In-process latency of `DecisionModel`. Opt-in: set `KRILL_DECISION_BENCH` and
/// `KRILL_DECISION_MODEL_PATH`, then `swift test -c release --filter DecisionModelBenchmarkTests`.
final class DecisionModelBenchmarkTests: XCTestCase {
    private func time(_ calls: Int, _ body: () throws -> Void) rethrows -> (median: Double, p90: Double) {
        var ms: [Double] = []
        for _ in 0 ..< calls {
            let start = DispatchTime.now().uptimeNanoseconds
            try body()
            ms.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6)
        }
        ms.sort()
        return (ms[ms.count / 2], ms[Int(Double(ms.count) * 0.9)])
    }

    func testLatency() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["KRILL_DECISION_BENCH"] != nil, let path = env["KRILL_DECISION_MODEL_PATH"] else {
            throw XCTSkip("Set KRILL_DECISION_MODEL_PATH and KRILL_DECISION_BENCH to run the decision-model benchmark")
        }
        guard MLXMetalRuntime.canInitializeMLXForTests else { throw XCTSkip("MLX Metal runtime is not available") }
        let dir = URL(fileURLWithPath: path)
        let model = try await DecisionModel.load(directory: dir)
        let tokenizer = try await KrillTokenizer(from: dir)

        let short = "My card was charged twice for the same order this month."
        let three = ["billing question", "password reset", "shipping delay"]
        let many = (0 ..< 77).map { "category number \($0) for routing" }
        let sentence = "The customer wrote that the parcel arrived late and the invoice shows a duplicate charge. "
        let long = String(repeating: sentence, count: 28)
        let mid = String(repeating: sentence, count: 6)
        let four = three + ["account closure"]
        let ten = (0 ..< 10).map { "topic \($0) about orders" }
        print("long text tokens: \(tokenizer.encode(long).count)")

        _ = try model.choose([short], options: three)
        let a = try time(50) { _ = try model.choose([short], options: three) }
        print("short text, 3 options: median \(a.median) ms, p90 \(a.p90) ms")

        let first = try time(1) { _ = try model.choose([short], options: many) }
        print("short text, 77 options, first call: \(first.median) ms")
        let b = try time(50) { _ = try model.choose([short], options: many) }
        print("short text, 77 options, repeat calls: median \(b.median) ms, p90 \(b.p90) ms")

        _ = try model.choose([mid], options: four)
        let m = try time(50) { _ = try model.choose([mid], options: four) }
        print("~100-token text, 4 options: median \(m.median) ms, p90 \(m.p90) ms")

        _ = try model.choose([long], options: ten)
        let c = try time(50) { _ = try model.choose([long], options: ten) }
        print("~500-token text, 10 options: median \(c.median) ms, p90 \(c.p90) ms")
        _ = try model.yesNo(texts: [short], question: "is this about billing?")[0]
        let y = try time(50) { _ = try model.yesNo(texts: [short], question: "is this about billing?")[0] }
        print("yes / no: median \(y.median) ms, p90 \(y.p90) ms")
        print("MLX peak memory: \(GPU.peakMemory / 1_048_576) MB")
    }
}
