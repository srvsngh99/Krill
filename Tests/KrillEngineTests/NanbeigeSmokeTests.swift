import XCTest
import Foundation
import KrillSampler
@testable import KrillEngine

/// Live smoke gate for the native Nanbeige 4.2 runtime. Gated on
/// `KRILL_NANBEIGE_MODEL_PATH` pointing at a checkpoint directory (the nvfp4
/// build is the shipped one); skipped when unset.
///
/// Logit parity proves the forward is right for ONE token. These assert the
/// whole serving path — prompt build → looped prefill → 44-cache decode loop →
/// reasoning filter → detokenizer — produces text a user would accept. Every
/// bug this family actually shipped was invisible to parity and visible here:
///
///  - the sampler's double-softmax (correct greedy, garbage at temperature > 0)
///  - the tokenizer's per-token `Strip`, which ate every space
///    ("ThecapitalofFranceisParis.")
///  - dropped byte-fallback newlines, which flattened lists onto one line
///  - a 512-token default spent entirely inside `<think>`, returning an EMPTY
///    reply on the golden path
final class NanbeigeSmokeTests: XCTestCase {

    private func requireModel() throws -> URL {
        guard let path = ProcessInfo.processInfo
            .environment["KRILL_NANBEIGE_MODEL_PATH"], !path.isEmpty else {
            throw XCTSkip("KRILL_NANBEIGE_MODEL_PATH not set")
        }
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(
            atPath: path, isDirectory: &isDir), isDir.boolValue else {
            throw XCTSkip("KRILL_NANBEIGE_MODEL_PATH is not a directory: \(path)")
        }
        return URL(fileURLWithPath: path)
    }

    private func loadEngine() async throws -> InferenceEngine {
        let engine = InferenceEngine(modelDirectory: try requireModel())
        try await engine.load()
        return engine
    }

    /// Drain the generation stream into a single string, as the CLI does.
    private func run(
        _ engine: InferenceEngine, _ prompt: String,
        params: SamplingParams = .greedy, maxTokens: Int = 4096
    ) async -> String {
        let (stream, _) = engine.generate(
            prompt: prompt, params: params, maxTokens: maxTokens)
        var out = ""
        for await ev in stream { out += ev.text; if ev.isEnd { break } }
        return out
    }

    /// A looped model must report `num_loops * num_hidden_layers` KV caches.
    /// 22 instead of 44 corrupts every decode step past the first.
    func testLoadsAsNanbeigeWithLoopedCacheCount() async throws {
        let engine = try await loadEngine()
        XCTAssertEqual(engine.family, "nanbeige")
        XCTAssertTrue(engine.emitsReasoningBlock,
            "Nanbeige's template opens `<think>`; the CLI budgets tokens on this")
    }

    /// The golden path: a plain factual question, greedy, with the CLI's own
    /// reasoning-aware budget. Must produce non-empty text containing the answer.
    func testAnswersFactualQuestion() async throws {
        let engine = try await loadEngine()
        let out = await run(
            engine, "What is the capital of France? Answer in one short sentence.")
        let visible = stripReasoning(out)
        XCTAssertFalse(visible.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            "empty reply - the whole budget went into a reasoning block")
        XCTAssertTrue(visible.lowercased().contains("paris"),
            "expected 'Paris', got: \(visible.prefix(300))")
    }

    /// Word separators must survive per-token streaming decode. The `Strip`
    /// decoder bug produced "ThecapitalofFranceisParis." - fluent, and unusable.
    func testOutputHasWordSpacing() async throws {
        let engine = try await loadEngine()
        let out = await run(
            engine, "What is the capital of France? Answer in one short sentence.")
        let visible = stripReasoning(out)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let words = visible.split(separator: " ")
        XCTAssertGreaterThan(words.count, 3,
            "output has no spaces - per-token decode is eating separators: \(visible.prefix(200))")
    }

    /// Byte-fallback newlines (`<0x0A>`) must survive too, or every list and
    /// code block collapses onto a single line.
    func testMultiLineOutputKeepsNewlines() async throws {
        let engine = try await loadEngine()
        let out = await run(
            engine, "List exactly three primary colors as a numbered list, nothing else.")
        let visible = stripReasoning(out)
        XCTAssertTrue(visible.contains("\n"),
            "no newlines - byte-fallback tokens are being dropped: \(visible.prefix(200))")
    }

    /// Sampling at the model's recommended temperature must stay coherent. The
    /// double-softmax bug returned uniformly random vocabulary here while greedy
    /// looked perfect, so greedy-only coverage would have missed it entirely.
    func testTemperatureSamplingStaysCoherent() async throws {
        let engine = try await loadEngine()
        let out = await run(
            engine, "What is the capital of France? Answer in one short sentence.",
            params: SamplingParams(temperature: 0.6, topP: 0.95, seed: 42))
        let visible = stripReasoning(out)
        XCTAssertTrue(visible.lowercased().contains("paris"),
            "temperature sampling produced incoherent text: \(visible.prefix(300))")
    }

    /// Drop a leading `<think>…</think>` block the way the CLI/server surfaces do.
    private func stripReasoning(_ text: String) -> String {
        guard let close = text.range(of: "</think>") else { return text }
        return String(text[close.upperBound...])
    }
}
