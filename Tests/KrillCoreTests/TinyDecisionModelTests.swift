import XCTest
import MLX
import MLXNN
import KrillRuntime
import KrillTokenizer
@testable import KrillCore

/// Runs the decision maths in CI against a tiny random-weight bundle of the same architecture
/// (three layers, hidden size 32) with goldens from the Python reference run on that bundle.
final class TinyDecisionModelTests: XCTestCase {
    private typealias Golden = DecisionModelParityTests.Golden

    private struct ReaderCheck: Decodable {
        let ids: [Int]
        let hidden: [[Float]]
    }

    private struct Check: Decodable {
        let readerCheck: ReaderCheck
        enum CodingKeys: String, CodingKey { case readerCheck = "reader_check" }
    }

    private let fixtures = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .appendingPathComponent("Fixtures/decision")
    private var bundle: URL { fixtures.appendingPathComponent("tiny") }

    private func fixture() async throws -> (model: DecisionModel, golden: Golden) {
        guard MLXMetalRuntime.canInitializeMLXForTests else {
            throw XCTSkip("MLX Metal runtime is not available in this test process")
        }
        let data = try Data(contentsOf: fixtures.appendingPathComponent("golden_tiny.json"))
        return (try await DecisionModel.load(directory: bundle), try JSONDecoder().decode(Golden.self, from: data))
    }

    func testTokenizerMatchesReference() async throws {
        let (_, golden) = try await fixture()
        let tokenizer = try await KrillTokenizer(from: bundle)
        for c in golden.cases {
            let state = c.question.flatMap { c.kind == "choice" ? "\(c.text)\n\n\($0)" : nil } ?? c.text
            XCTAssertEqual(tokenizer.encode(state), c.stateIds, "tokens differ for \(c.id)")
        }
    }

    private func assertMatchesReference(_ model: DecisionModel, _ golden: Golden) throws -> Int {
        var none = 0
        for c in golden.cases {
            if c.kind == "yesno" {
                let r = try model.yesNo(texts: [c.text], question: c.question!)[0]
                XCTAssertEqual(r.answer, c.answer, c.id)
                XCTAssertEqual(r.pYes, c.pYes!, accuracy: 1e-4, c.id)
                continue
            }
            let r = try model.choose([c.text], options: c.options!, question: c.question, allowNone: c.allowNone!)[0]
            XCTAssertEqual(r.index, c.index, "decision differs for \(c.id)")
            XCTAssertEqual(r.pNone, c.pNone!, accuracy: 1e-4, c.id)
            for (a, b) in zip(r.scores, c.scores!) { XCTAssertEqual(a, b, accuracy: 1e-4, c.id) }
            if r.index == nil { none += 1 }
        }
        return none
    }

    func testDecisionsAndProbabilitiesMatchReference() async throws {
        let (model, golden) = try await fixture()
        XCTAssertTrue(golden.cases.contains { $0.options?.count == 1 })
        XCTAssertGreaterThan(try assertMatchesReference(model, golden), 0)
    }

    func testClsPoolingMatchesReference() async throws {
        _ = try await fixture()
        let copy = FileManager.default.temporaryDirectory.appendingPathComponent("krill-cls-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: copy) }
        try FileManager.default.copyItem(at: bundle, to: copy)
        let raw = try Data(contentsOf: copy.appendingPathComponent("config.json"))
        var json = try XCTUnwrap(JSONSerialization.jsonObject(with: raw) as? [String: Any])
        json["pooling"] = "cls"
        try JSONSerialization.data(withJSONObject: json).write(to: copy.appendingPathComponent("config.json"))
        let golden = try JSONDecoder().decode(
            Golden.self, from: Data(contentsOf: fixtures.appendingPathComponent("golden_tiny_cls.json")))
        _ = try assertMatchesReference(try await DecisionModel.load(directory: copy), golden)
    }

    func testBatchEqualsSingle() async throws {
        let (model, golden) = try await fixture()
        let rows = golden.cases.filter { golden.batch.contains($0.id) }
        XCTAssertTrue(rows.allSatisfy { $0.options == rows[0].options && $0.question == rows[0].question })
        let batched = try model.choose(rows.map(\.text), options: rows[0].options!, question: rows[0].question)
        for (row, b) in zip(rows, batched) {
            XCTAssertEqual(b.index, row.index, row.id)
            let single = try model.choose([row.text], options: row.options!, question: row.question)[0]
            for (x, y) in zip(b.scores, single.scores) { XCTAssertEqual(x, y, accuracy: 1e-5, row.id) }
        }
    }

    func testChunkedReadMatchesSingleForManyTexts() async throws {
        let (model, _) = try await fixture()
        let texts = (0 ..< 20).map { "ticket \($0): " + String(repeating: "the parcel is late ", count: $0 % 5 + 1) }
        let options = ["billing question", "shipping delay", "password reset"]
        let batched = try model.choose(texts, options: options)
        for (text, b) in zip(texts, batched) {
            let single = try model.choose([text], options: options)[0]
            XCTAssertEqual(b.index, single.index)
            for (x, y) in zip(b.scores, single.scores) { XCTAssertEqual(x, y, accuracy: 1e-5) }
        }
    }

    func testRequestLimitsAndEmptyInputsThrow() async throws {
        let (model, _) = try await fixture()
        XCTAssertThrowsError(try model.choose(Array(repeating: "hi", count: DecisionModel.maxTexts + 1), options: ["a"]))
        XCTAssertThrowsError(try model.choose(["hi"], options: (0 ... DecisionModel.maxOptions).map { "option \($0)" }))
        XCTAssertThrowsError(try model.yesNo(texts: Array(repeating: "hi", count: DecisionModel.maxTexts + 1), question: "Is it?"))
        XCTAssertThrowsError(try model.choose([], options: ["a"]))
        XCTAssertThrowsError(try model.choose([""], options: ["a", "b"]))
        XCTAssertThrowsError(try model.choose(["hi"], options: ["a", ""]))
        XCTAssertThrowsError(try model.yesNo(texts: [""], question: "Is it?"))
        XCTAssertNoThrow(try model.choose(["hi"], options: (0 ..< DecisionModel.maxOptions).map { "option \($0)" }))
        XCTAssertThrowsError(try model.choose([String(repeating: "word ", count: 3000)], options: ["a", "b"]))
    }

    func testConfigRejectsUnknownPoolingAndRule() throws {
        let raw = try Data(contentsOf: bundle.appendingPathComponent("config.json"))
        var json = try XCTUnwrap(JSONSerialization.jsonObject(with: raw) as? [String: Any])
        XCTAssertNoThrow(try JSONDecoder().decode(DecisionConfig.self, from: raw))
        json["pooling"] = "max"
        XCTAssertThrowsError(try JSONDecoder().decode(
            DecisionConfig.self, from: JSONSerialization.data(withJSONObject: json)))
        json["pooling"] = "mean"
        var none = try XCTUnwrap(json["none"] as? [String: Any])
        none["short"] = ["rule": "sum", "tau": 0.1]
        json["none"] = none
        XCTAssertThrowsError(try JSONDecoder().decode(
            DecisionConfig.self, from: JSONSerialization.data(withJSONObject: json)))
    }

    func testBundleDetectionNeedsReaderHeadAndPooling() throws {
        XCTAssertTrue(DecisionModel.isBundle(bundle))
        XCTAssertEqual(DecisionFamily.detect(bundle), .reader)
        XCTAssertFalse(DecisionModel.isDecisionConfig(["reader": 1, "head": 2, "pooling": "cls"]))
        XCTAssertFalse(DecisionModel.isDecisionConfig(["reader": [String: Any](), "head": [String: Any]()]))
    }

    // MARK: - Shared encoder

    private func explicitNorm(_ x: MLXArray, _ w: MLXArray, eps: Float) -> MLXArray {
        let mu = MLX.mean(x, axis: -1, keepDims: true)
        let v = MLX.variance(x, axis: -1, keepDims: true)
        return w * (x - mu) * MLX.rsqrt(v + eps)
    }

    func testFusedNormEqualsExplicitFormula() throws {
        guard MLXMetalRuntime.canInitializeMLXForTests else { throw XCTSkip("MLX Metal runtime is not available") }
        MLXRandom.seed(11)
        let norm = ModernBertNorm(64, eps: 1e-5)
        let w = MLXRandom.normal([64]) * 0.5 + 1
        try norm.update(parameters: ModuleParameters.unflattened(["weight": w]), verify: [.all])
        for (dtype, tol) in [(DType.float32, Float(1e-5)), (DType.float16, Float(2e-2))] {
            let x = (MLXRandom.normal([3, 17, 64]) * 2 + 0.5).asType(dtype)
            let fused = norm(x).asType(.float32)
            let reference = explicitNorm(x, w.asType(dtype), eps: 1e-5).asType(.float32)
            XCTAssertLessThan(MLX.abs(fused - reference).max().item(Float.self), tol, "\(dtype)")
        }
    }

    private func reader() throws -> (ModernBertEmbeddingModel, DecisionConfig) {
        let config = try JSONDecoder().decode(
            DecisionConfig.self, from: Data(contentsOf: bundle.appendingPathComponent("config.json")))
        let model = ModernBertEmbeddingModel(config.reader)
        let weights = try loadWeightArrays(from: bundle)
            .filter { !["option_mlp.", "none_mlp.", "interaction_proj."].contains(where: $0.key.hasPrefix) }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.allModelKeysSet, .shapeMismatch])
        return (model, config)
    }

    func testEmbeddingForwardMatchesReference() throws {
        guard MLXMetalRuntime.canInitializeMLXForTests else { throw XCTSkip("MLX Metal runtime is not available") }
        let data = try Data(contentsOf: fixtures.appendingPathComponent("golden_tiny.json"))
        let check = try JSONDecoder().decode(Check.self, from: data).readerCheck
        let (model, _) = try reader()
        let out = model.lastHiddenState(MLXArray(check.ids.map(Int32.init), [1, check.ids.count]))[0]
        let expected = MLXArray(check.hidden.flatMap { $0 }, [check.ids.count, check.hidden[0].count])
        XCTAssertLessThan(MLX.abs(out - expected).max().item(Float.self), 1e-4)
    }

    func testPaddedMaskMatchesUnpaddedInFp16() throws {
        guard MLXMetalRuntime.canInitializeMLXForTests else { throw XCTSkip("MLX Metal runtime is not available") }
        let (model, _) = try reader()
        model.update(parameters: model.parameters().mapValues { $0.asType(.float16) })
        let tokens = MLXArray((0 ..< 3 * 40).map { Int32(5 + $0 % 37) }, [3, 40])
        let lengths = [40, 20, 3]
        let mask = MLXArray(0 ..< Int32(40)).reshaped(1, 40) .< MLXArray(lengths.map(Int32.init)).reshaped(-1, 1)
        let padded = model.hiddenStates(tokens, attentionMask: mask).asType(.float32)
        XCTAssertFalse(MLX.isNaN(padded).any().item(Bool.self))
        for (row, n) in lengths.enumerated() {
            let alone = model.hiddenStates(tokens[row ..< row + 1, 0 ..< n]).asType(.float32)[0]
            XCTAssertLessThan(MLX.abs(padded[row, 0 ..< n] - alone).max().item(Float.self), 0.1, "row \(row)")
        }
    }
}
