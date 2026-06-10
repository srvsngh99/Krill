import XCTest
import MLX
@testable import KLMEngine

/// Context-sweep decode benchmark: decode tok/s + peak memory as a function of
/// context length, with an n-gram-speculative column. This is the re-runnable
/// harness behind the long-context decode work (the 24.8 -> 12.5 tok/s cliff)
/// - run it BEFORE a change to lock the baseline curve and AFTER to prove the
/// win. Rows print to stderr as they complete, so a hard OOM at the top of the
/// sweep (expected for the pre-rotating-KV baseline past ~32k: 48 unbounded
/// layers ~ 448KB/token of KV) still leaves the partial table.
///
///   KLM_RUN_DECODE_SWEEP=1 \
///   KLM_GEMMA4_UNIFIED_MODEL_PATH=~/.krillm/models/blobs/gemma-4-12b \
///     swift test --filter Gemma4DecodeSweepTests
///
/// Optional: KLM_SWEEP_CTX="512,2048,6656" overrides the context list;
/// KLM_SWEEP_NGRAM=0 skips the n-gram column (halves runtime).
final class Gemma4DecodeSweepTests: XCTestCase {

    private func requireOptIn() throws -> URL {
        guard ProcessInfo.processInfo.environment["KLM_RUN_DECODE_SWEEP"] == "1" else {
            throw XCTSkip("KLM_RUN_DECODE_SWEEP != 1")
        }
        guard let path = ProcessInfo.processInfo.environment["KLM_GEMMA4_UNIFIED_MODEL_PATH"],
              !path.isEmpty else {
            throw XCTSkip("KLM_GEMMA4_UNIFIED_MODEL_PATH not set")
        }
        let expanded = (path as NSString).expandingTildeInPath
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: expanded, isDirectory: &isDir),
              isDir.boolValue else {
            throw XCTSkip("not a directory: \(expanded)")
        }
        return URL(fileURLWithPath: expanded, isDirectory: true)
    }

    /// Crash-safe row output: stderr is unbuffered, so a fatal MLX OOM later in
    /// the sweep cannot erase rows already printed.
    private func emit(_ line: String) {
        FileHandle.standardError.write(Data((line + "\n").utf8))
    }

    /// Agent-context-like filler: repeated structured sections (tool outputs /
    /// code in real agent transcripts are similarly repetitive, which is also
    /// the regime where n-gram speculation can win) with a needle planted at
    /// the START so correctness at full length is checkable, and a question at
    /// the end. ~55 tokens per section.
    private func prompt(sections: Int) -> String {
        let para = """
        The KrillLM engine runs large language models natively on Apple Silicon \
        using MLX. It supports continuous batching, prefix-cache reuse, grammar \
        constrained decoding, native tool calling, and multimodal input. The \
        secret pass phrase for this document is "tangerine satellite".
        """
        var blocks: [String] = ["Document start. Remember the pass phrase."]
        for i in 0..<sections { blocks.append("Section \(i): " + para) }
        // Default question asks for a summary FIRST, so at 64 max tokens the
        // phrase often does not fit (needle=N is then a generation-budget
        // artifact, not a retrieval failure). KLM_SWEEP_DIRECT_Q=1 asks for
        // the phrase ONLY - use it to validate retrieval at frontier lengths.
        if ProcessInfo.processInfo.environment["KLM_SWEEP_DIRECT_Q"] == "1" {
            blocks.append("\nQuestion: What is the secret pass phrase mentioned above? Answer:")
        } else {
            blocks.append("\nQuestion: Summarize what the KrillLM engine supports, then state the secret pass phrase. Answer:")
        }
        return blocks.joined(separator: "\n")
    }

    private struct Row {
        let label: String
        let promptTokens: Int
        let prefillS: Double
        let decodeTps: Double
        let peakGB: Double
        let needle: Bool
        let specAccepted: Int
    }

    private func run(
        engine: InferenceEngine, sections: Int, ngram: Bool, maxTokens: Int = 64
    ) async -> Row {
        GPU.resetPeakMemory()
        let base = Memory.snapshot().peakMemory
        let (stream, stats) = engine.generate(
            prompt: prompt(sections: sections), params: .greedy,
            maxTokens: maxTokens, usePrefixCache: false,
            useNgramSpeculative: ngram ? true : nil)
        var text = ""
        for await event in stream {
            if event.isEnd { break }
            text += event.text
        }
        let peakGB = Double(Memory.snapshot().peakMemory - base) / 1_073_741_824
        // Stats are written by the generation Task as it finalizes; a very
        // short answer (direct question -> phrase -> EOS) can end the stream
        // before that write lands. Poll briefly rather than reading zeros.
        var s = stats()
        for _ in 0 ..< 20 where (s?.promptTokens ?? 0) == 0 {
            try? await Task.sleep(nanoseconds: 50_000_000)
            s = stats()
        }
        return Row(
            label: ngram ? "ngram" : "plain",
            promptTokens: s?.promptTokens ?? 0,
            prefillS: s?.prefillTime ?? 0,
            decodeTps: s?.decodeTokensPerSecond ?? 0,
            peakGB: peakGB,
            needle: text.lowercased().contains("tangerine"),
            specAccepted: s?.speculative?.acceptedTokens ?? 0)
    }

    func testDecodeContextSweep() async throws {
        let modelDir = try requireOptIn()
        let env = ProcessInfo.processInfo.environment
        // ~55 tok/section; targets ~ {512, 2k, 6.6k, 16k, 32k, 48k, 60k}.
        let defaultSections = [8, 36, 120, 300, 590, 880, 1100]
        let sectionsList: [Int]
        if let raw = env["KLM_SWEEP_CTX"] {
            // Values are target TOKEN counts; convert to sections.
            sectionsList = raw.split(separator: ",").compactMap {
                Int($0.trimmingCharacters(in: .whitespaces)).map { max(1, $0 / 55) }
            }
        } else {
            sectionsList = defaultSections
        }
        let runNgram = env["KLM_SWEEP_NGRAM"] != "0"

        let engine = InferenceEngine(modelDirectory: modelDir, kvCacheDtype: "fp16")
        try await engine.load()
        defer { engine.unload() }

        emit("")
        emit("===== decode context sweep: gemma-4-12b =====")
        emit("| ctx tokens | prefill s | decode tok/s | peak GB | needle | ngram tok/s | ngram needle | spec accepted |")
        emit("|---|---|---|---|---|---|---|---|")
        for sections in sectionsList {
            let plain = await run(engine: engine, sections: sections, ngram: false)
            var ngramCol = "-", ngramNeedle = "-", spec = "-"
            if runNgram {
                let ng = await run(engine: engine, sections: sections, ngram: true)
                ngramCol = String(format: "%.1f", ng.decodeTps)
                ngramNeedle = ng.needle ? "y" : "N"
                spec = "\(ng.specAccepted)"
            }
            emit("| \(plain.promptTokens) | \(String(format: "%.1f", plain.prefillS)) | "
                + "\(String(format: "%.1f", plain.decodeTps)) | "
                + "\(String(format: "%.2f", plain.peakGB)) | \(plain.needle ? "y" : "N") | "
                + "\(ngramCol) | \(ngramNeedle) | \(spec) |")
            // A direct-question row can answer in a couple of tokens (decode
            // rate is then meaningless); the row still proves itself via the
            // needle. Otherwise require a real decode rate.
            XCTAssertTrue(plain.decodeTps > 0 || plain.needle,
                "no decode and no needle at sections=\(sections)")
        }
        emit("=============================================")
        emit("")
    }
}
