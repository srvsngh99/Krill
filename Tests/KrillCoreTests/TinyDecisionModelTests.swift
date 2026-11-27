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

    func testTuneCorrectsHeldOutChoicesAndReloads() async throws {
        let (model, _) = try await fixture()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("krill-tune-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let output = root.appendingPathComponent("tuned")
        let text = "My card was charged twice for the same order this month."
        let choices = ["billing question", "password reset", "cancel subscription", "shipping delay",
                       "product defect", "general feedback"]
        let baseScores = try model.choose([text], options: choices)[0]
        let texts = (0 ..< 20).map { "\(text) Reference \($0)." }
        let heldOut = [12, 7, 11, 10].map { texts[$0] }
        let previous = try model.choose(heldOut, options: choices).compactMap(\.label)
        let label = try XCTUnwrap(choices.first { $0 != baseScores.label && !previous.contains($0) })
        let examples = try texts.enumerated().map { i, text in
            try DecisionTuneExample(json: ["text": text, "options": choices, "label": label], number: i + 1)
        }
        let head = bundle.appendingPathComponent("head.safetensors")
        let calibration = bundle.appendingPathComponent("calibration.json")
        let originalHead = try Data(contentsOf: head)
        let originalCalibration = try Data(contentsOf: calibration)
        let report = try model.tune(examples: examples, options: .init(epochs: 40, lr: 0.03), output: output)
        XCTAssertTrue(report.accepted)
        XCTAssertEqual(report.holdoutExamples, 4)
        XCTAssertEqual(report.trainExamples, 16)
        XCTAssertGreaterThan(report.after.accuracy, report.before.accuracy)
        XCTAssertEqual(report.after.accuracy, 1)

        let tuned = try await DecisionModel.load(directory: output)
        let reloaded = try await DecisionModel.load(directory: output)
        let first = try tuned.choose([text], options: choices)[0]
        let second = try reloaded.choose([text], options: choices)[0]
        XCTAssertEqual(first.label, label)
        XCTAssertEqual(first.scores, second.scores)
        XCTAssertEqual(first.pNone, second.pNone)
        XCTAssertEqual(try model.choose([text], options: choices)[0].scores, baseScores.scores)
        XCTAssertEqual(try Data(contentsOf: head), originalHead)
        XCTAssertEqual(try Data(contentsOf: calibration), originalCalibration)
        XCTAssertEqual(try Data(contentsOf: output.appendingPathComponent("calibration.json")), originalCalibration)

        let fm = FileManager.default
        let reader = "reader.safetensors"
        let baseReader = bundle.appendingPathComponent(reader)
        let tunedReader = output.appendingPathComponent(reader)
        let baseAttributes = try fm.attributesOfItem(atPath: baseReader.path)
        let tunedAttributes = try fm.attributesOfItem(atPath: tunedReader.path)
        if baseAttributes[.systemNumber] as? NSNumber == tunedAttributes[.systemNumber] as? NSNumber {
            XCTAssertEqual(baseAttributes[.systemFileNumber] as? NSNumber, tunedAttributes[.systemFileNumber] as? NSNumber)
        }
        XCTAssertEqual(try Data(contentsOf: baseReader), try Data(contentsOf: tunedReader))
        let baseWeights = try loadArrays(url: head)
        let tunedWeights = try loadArrays(url: output.appendingPathComponent(head.lastPathComponent))
        XCTAssertEqual(Set(baseWeights.keys), Set(tunedWeights.keys))
        for (key, weight) in baseWeights {
            XCTAssertEqual(weight.shape, tunedWeights[key]?.shape, key)
            XCTAssertEqual(weight.dtype, tunedWeights[key]?.dtype, key)
        }
        let config = try JSONSerialization.jsonObject(with: Data(contentsOf: output.appendingPathComponent("config.json")))
        let provenance = (config as? [String: Any])?["tuned_from"] as? [String: String]
        XCTAssertEqual(provenance?["version"], "tiny")
        XCTAssertEqual(provenance?["head_sha256"]?.count, 64)
    }

    func testTuneNoneAndYesNoWithFixedCalibration() async throws {
        let (model, _) = try await fixture()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("krill-tune-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let text = "My card was charged twice for the same order this month."
        let choices = ["billing question", "password reset", "cancel subscription", "shipping delay",
                       "product defect", "general feedback"]
        XCTAssertNotNil(try model.choose([text], options: choices)[0].label)
        let neither = try DecisionTuneExample(json: ["text": text, "options": choices, "label": NSNull()], number: 1)
        let noneOutput = root.appendingPathComponent("none")
        let noneReport = try model.tune(examples: Array(repeating: neither, count: 20),
                                        options: .init(lr: 0.03), output: noneOutput)
        XCTAssertEqual(noneReport.after.noneRecall, 1)
        XCTAssertEqual(noneReport.after.nonePrecision, 1)
        let noneModel = try await DecisionModel.load(directory: noneOutput)
        XCTAssertNil(try noneModel.choose([text], options: choices)[0].label)

        let question = "Is this about a payment?"
        let before = try model.yesNo(texts: [text], question: question)[0]
        let answer = before.answer == "yes" ? "no" : "yes"
        let binary = try DecisionTuneExample(json: ["text": text, "question": question, "answer": answer], number: 1)
        let binaryOutput = root.appendingPathComponent("binary")
        let binaryReport = try model.tune(examples: Array(repeating: binary, count: 20),
                                          options: .init(lr: 0.03), output: binaryOutput)
        XCTAssertEqual(binaryReport.before.accuracy, 0)
        XCTAssertEqual(binaryReport.after.accuracy, 1)
        let binaryModel = try await DecisionModel.load(directory: binaryOutput)
        XCTAssertEqual(try binaryModel.yesNo(texts: [text], question: question)[0].answer, answer)
        XCTAssertEqual(try model.yesNo(texts: [text], question: question)[0].pYes, before.pYes)
        let calibration = "calibration.json"
        XCTAssertEqual(try Data(contentsOf: binaryOutput.appendingPathComponent(calibration)),
                       try Data(contentsOf: bundle.appendingPathComponent(calibration)))
    }

    func testTuneRejectsWorseHoldoutUnlessForced() async throws {
        _ = try await fixture()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("krill-tune-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let base = root.appendingPathComponent("base")
        try FileManager.default.copyItem(at: bundle, to: base)
        let head = base.appendingPathComponent("head.safetensors")
        let weights = try loadArrays(url: head).mapValues { MLXArray.zeros(like: $0) }
        try MLX.save(arrays: weights, url: head)
        let calibration = "calibration.json"
        let original = Data(#"{"yes_no_logit_offset":{"default":0.25,"custom":0.2},"version":"tiny"}"#.utf8)
        try original.write(to: base.appendingPathComponent(calibration))
        let model = try await DecisionModel.load(directory: base)
        let holdout = Set(decisionTuneOrder(20, seed: 0, epoch: 0).prefix(4))
        let examples = try (0 ..< 20).map { i in
            let label: Any = holdout.contains(i) ? NSNull() : "billing"
            return try DecisionTuneExample(json: ["text": "The card was charged.", "options": ["billing", "access"],
                                                  "label": label], number: i + 1)
        }
        let output = root.appendingPathComponent("tuned")
        let rejected = try model.tune(examples: examples, options: .init(epochs: 1, lr: 0.5), output: output)
        XCTAssertEqual(rejected.before.accuracy, 1)
        XCTAssertEqual(rejected.after.accuracy, 0)
        XCTAssertFalse(rejected.accepted)
        XCTAssertFalse(FileManager.default.fileExists(atPath: output.path))
        let forced = try model.tune(examples: examples, options: .init(epochs: 1, lr: 0.5, force: true), output: output)
        XCTAssertTrue(forced.accepted)
        XCTAssertTrue(DecisionModel.isBundle(output))
        XCTAssertEqual(try Data(contentsOf: output.appendingPathComponent(calibration)), original)
        XCTAssertEqual(try Data(contentsOf: base.appendingPathComponent(calibration)), original)
    }

    func testTuneSplitWeightsAndValidation() throws {
        let heldOut = decisionTuneOrder(20, seed: 0, epoch: 0).prefix(4)
        XCTAssertEqual(Array(heldOut), [12, 7, 11, 10])
        let counts = ["common": 240, "rare": 12, "none": 1]
        let weights = decisionTuneClassWeights(counts)
        XCTAssertEqual(weights["rare"], weights["none"])
        XCTAssertEqual(weights["rare"]! / weights["common"]!, 10, accuracy: 1e-5)
        XCTAssertEqual(counts.reduce(Float(0)) { $0 + Float($1.value) * weights[$1.key]! }, 253, accuracy: 1e-4)
        XCTAssertNoThrow(try DecisionModel.requireTunable(.reader))
        for family in [DecisionFamily.amber, .glide, .dmjepa, .espresso] {
            XCTAssertThrowsError(try DecisionModel.requireTunable(family)) {
                XCTAssertTrue(String(describing: $0).contains("only reader decision models support tuning"))
            }
        }
        XCTAssertThrowsError(try DecisionTuneExample(json: ["text": "example", "options": ["a"]], number: 1))
        XCTAssertThrowsError(try DecisionTuneExample(json: ["text": "example", "question": "q", "answer": NSNull()], number: 1))
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
