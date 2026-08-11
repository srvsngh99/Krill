import ArgumentParser
import Foundation
import MLX
import MLXNN
import KrillCache
import KrillCore
import KrillEngine
import KrillRegistry
import KrillTokenizer

/// Measure a checkpoint's LANGUAGE-MODELLING QUALITY, which no other Krill
/// surface reports.
///
/// `krill bench` answers "how fast", and the parity oracles answer "does this
/// runtime match the reference implementation". Neither answers "what did this
/// QUANTIZATION cost me" — the parity gates compare a Krill build against the
/// same quantized reference, so a quantization that degrades the model equally
/// in both passes them cleanly. Choosing between 4-bit affine, nvfp4 and 3-bit
/// needs a number that moves when accuracy moves.
///
/// Reports two:
///   * **perplexity** — `exp(mean NLL per token)`. Comparable ONLY between
///     builds that share a tokenizer (the same base model at different
///     quantizations, which is the intended use). Across families it is
///     meaningless: a tokenizer that packs more text per token gets a lower
///     number for free.
///   * **bits per byte** — total NLL over the UTF-8 byte count. Tokenizer
///     -independent, so this is the honest cross-family column.
///
/// Both are computed over non-overlapping windows, so each token is scored
/// exactly once and the result does not depend on a stride heuristic. The first
/// token of each window has no context and is not scored.
struct PerplexityCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "perplexity",
        abstract: "Measure language-modelling quality (perplexity + bits/byte)"
    )

    @Argument(help: "Model name (from registry) or path to model directory")
    var model: String

    @Option(name: .long, help: "UTF-8 text file to evaluate on")
    var text: String

    @Option(name: .long, help: "Tokens per window (lower this for big models: a window holds window x vocab floats)")
    var window: Int = 512

    @Option(name: .long, help: "Stop after this many tokens (0 = whole file)")
    var limit: Int = 0

    func run() async throws {
        let registry = Registry()
        let modelDir = registry.hasModel(model)
            ? registry.modelPath(model)
            : URL(fileURLWithPath: model)
        guard FileManager.default.fileExists(atPath: modelDir.path) else {
            print("Error: model '\(model)' not found.")
            throw ExitCode.failure
        }
        guard let raw = try? String(contentsOfFile: text, encoding: .utf8) else {
            print("Error: could not read --text file: \(text)")
            throw ExitCode.failure
        }
        guard window >= 2 else {
            print("Error: --window must be at least 2.")
            throw ExitCode.failure
        }

        let loaded = try loadModel(from: modelDir)
        let tokenizer = try await KrillTokenizer(from: modelDir)

        // Raw text, NOT a chat template: perplexity must measure the language
        // model, not the prompt scaffold.
        var ids = tokenizer.encode(raw)
        let truncated = limit > 0 && ids.count > limit
        if truncated { ids = Array(ids.prefix(limit)) }
        guard ids.count >= 2 else {
            print("Error: text tokenized to \(ids.count) token(s); need at least 2.")
            throw ExitCode.failure
        }
        // Bits/byte must divide by the bytes actually SCORED. With `--limit`,
        // the whole file's byte count covers text the model never saw, which
        // silently deflates the number — and deflates it by a different factor
        // per model, since a limit in TOKENS covers a different amount of text
        // under each tokenizer. Decoding the (possibly truncated) ids back
        // gives the exact span that was measured.
        let byteCount = truncated
            ? Data(tokenizer.decode(ids).utf8).count
            : Data(raw.utf8).count

        print("Krill Perplexity")
        print("================")
        print("Model:  \(model) (family: \(loaded.family))")
        print("Text:   \(text) — \(byteCount) bytes, \(ids.count) tokens")
        print("Window: \(window) tokens, non-overlapping")
        print()

        var totalNLL = Float(0)
        var scored = 0
        var windowIndex = 0

        var start = 0
        while start + 2 <= ids.count {
            let end = Swift.min(start + window, ids.count)
            let chunk = Array(ids[start ..< end]).map { Int32($0) }
            // Fresh caches per window: windows are independent, and a carried
            // cache would silently turn this into a sliding-context measurement.
            let caches = makeKVCaches(spec: loaded.cacheSpec, numLayers: loaded.numLayers)
            let input = MLXArray(chunk).reshaped(1, chunk.count)

            // Full forward — we need a distribution at EVERY position, so the
            // last-token-only `prefillForward` is deliberately not used here.
            let logits = loaded.forward(input, caches)
            // Predict token t+1 from position t.
            let pred = logits[0, 0 ..< (chunk.count - 1), 0...]
            let targets = Array(chunk[1...])

            totalNLL += PerplexityMath.totalNLL(logits: pred, targets: targets)
            scored += chunk.count - 1
            windowIndex += 1
            if windowIndex % 8 == 0 {
                print("  \(scored)/\(ids.count - 1) tokens scored...")
            }
            start = end
        }

        guard scored > 0 else {
            print("Error: no tokens scored.")
            throw ExitCode.failure
        }

        let meanNLL = Double(totalNLL) / Double(scored)
        let ppl = PerplexityMath.perplexity(totalNLL: Double(totalNLL), tokens: scored)
        let bpb = PerplexityMath.bitsPerByte(totalNLL: Double(totalNLL), bytes: byteCount)

        print()
        print("Results")
        print("-------")
        print(String(format: "Tokens scored:  %d (%d window(s))", scored, windowIndex))
        print(String(format: "Mean NLL:       %.4f nats/token", meanNLL))
        print(String(format: "Perplexity:     %.3f", ppl))
        print(String(format: "Bits per byte:  %.4f", bpb))
        print()
        print("Perplexity compares only across builds sharing a tokenizer;")
        print("use bits/byte to compare different model families.")
    }
}
