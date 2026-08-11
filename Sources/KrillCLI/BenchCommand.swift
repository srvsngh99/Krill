import ArgumentParser
import Foundation
import MLX
import KrillEngine
import KrillCore
import KrillCache
import KrillSampler
import KrillRegistry

struct BenchCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "bench",
        abstract: "Run standardized inference benchmarks"
    )

    @Argument(help: "Model name (from registry) or path to model directory")
    var model: String

    @Option(name: .long, help: "Prompt length in tokens for prefill benchmark")
    var promptLen: Int = 512

    @Option(name: .long, help: "Number of tokens to generate")
    var genLen: Int = 256

    @Option(name: .long, help: "Number of benchmark runs (results averaged)")
    var runs: Int = 3

    @Option(name: .long, help: "Warmup runs before measuring (default 1)")
    var warmup: Int = 1

    func run() async throws {
        let modelDir: URL
        let registry = Registry()
        if registry.hasModel(model) {
            modelDir = registry.modelPath(model)
        } else {
            modelDir = URL(fileURLWithPath: model)
        }

        guard FileManager.default.fileExists(atPath: modelDir.path) else {
            print("Error: model '\(model)' not found.")
            throw ExitCode.failure
        }

        print("Krill Benchmark")
        print("=================")
        print("Model: \(model)")
        print("Prompt: \(promptLen) tokens, Generate: \(genLen) tokens, Runs: \(runs)")
        print()

        // Load model
        print("Loading model...")
        let loadStart = CFAbsoluteTimeGetCurrent()
        let loaded = try loadModel(from: modelDir)
        let loadTime = CFAbsoluteTimeGetCurrent() - loadStart
        print(String(format: "Loaded in %.2fs (family: %@)", loadTime, loaded.family))
        print()

        // Create synthetic prompt tokens (random valid token IDs)
        let promptTokens = (0 ..< promptLen).map { _ in Int32.random(in: 1 ..< Int32(min(loaded.vocabSize, 10000))) }

        // Warmup
        if warmup > 0 {
            print("Warming up (\(warmup) run\(warmup > 1 ? "s" : ""))...")
            for _ in 0 ..< warmup {
                _ = try await benchmarkRun(
                    model: loaded, promptTokens: promptTokens, genLen: 16)
            }
        }

        // Benchmark runs
        print("Benchmarking...")
        var results: [BenchResult] = []

        for i in 0 ..< runs {
            let result = try await benchmarkRun(
                model: loaded, promptTokens: promptTokens, genLen: genLen)
            results.append(result)
            print(String(format: "  Run %d: prefill %.1f tok/s, decode %.1f tok/s, TTFT %.0fms",
                         i + 1, result.prefillToksPerSec, result.decodeToksPerSec, result.ttftMs))
        }

        // Compute averages
        let avgPrefill = results.map(\.prefillToksPerSec).reduce(0, +) / Double(runs)
        let avgDecode = results.map(\.decodeToksPerSec).reduce(0, +) / Double(runs)
        let avgTTFT = results.map(\.ttftMs).reduce(0, +) / Double(runs)
        let peakMemMB = results.map(\.peakMemoryMB).max() ?? 0

        print()
        print("Results (avg of \(runs) runs)")
        print("-----------------------------")
        print(String(format: "Prefill:     %.1f tok/s (%d tokens)", avgPrefill, promptLen))
        print(String(format: "Decode:      %.1f tok/s (%d tokens)", avgDecode, genLen))
        print(String(format: "TTFT:        %.0f ms", avgTTFT))
        print(String(format: "Peak memory: %.0f MB", peakMemMB))
        print(String(format: "Load time:   %.2f s", loadTime))
    }
}

// MARK: - Benchmark Runner

private struct BenchResult {
    let prefillToksPerSec: Double
    let decodeToksPerSec: Double
    let ttftMs: Double
    let peakMemoryMB: Double
}

private func benchmarkRun(
    model: LoadedModel,
    promptTokens: [Int32],
    genLen: Int
) async throws -> BenchResult {
    // Allocate caches from the family's `cacheSpec`, exactly as the serving
    // engine does. The spec-less overload gives every layer a full-history
    // `KVCache`, which is wrong for any family with windowed layers (Gemma 4,
    // Muse Glimmer: 39 of 52 layers are `.rotating(window: 2048)`) and makes
    // the benchmark measure a configuration the server never runs.
    let caches: [KVCacheProtocol] = makeKVCaches(
        spec: model.cacheSpec, numLayers: model.numLayers)
    let sampler = Sampler(params: .init(temperature: 0.0))

    // Prefill
    let inputArray = MLXArray(promptTokens).reshaped(1, promptTokens.count)

    let prefillStart = CFAbsoluteTimeGetCurrent()
    // Prefer the family's last-token-only prefill, like the engine. The full
    // forward materializes logits for EVERY prompt position — at 512 tokens on
    // a 202k vocab that is ~414 MB of throwaway floats, which on a machine
    // already near its memory ceiling is enough to tip the run into swap and
    // report a decode rate the real serving path never produces.
    let prefillLogits = (model.prefillForward ?? model.forward)(inputArray, caches)
    MLX.eval(prefillLogits)
    let prefillTime = CFAbsoluteTimeGetCurrent() - prefillStart

    var nextToken = sampler.sample(prefillLogits)

    // Decode
    let decodeStart = CFAbsoluteTimeGetCurrent()
    for _ in 0 ..< genLen {
        let tokenInput = MLXArray([Int32(nextToken)]).reshaped(1, 1)
        let logits = model.forward(tokenInput, caches)
        MLX.eval(logits)
        nextToken = sampler.sample(logits)
    }
    let decodeTime = CFAbsoluteTimeGetCurrent() - decodeStart

    // Peak MLX allocation, measured from process start (MLX exposes no reset in
    // this version), so it covers load + every run — which is what the caller's
    // max() across runs wanted anyway. `resident_size` was useless here: MLX
    // holds weights in unified-memory buffers the task's RSS does not count, so
    // an 18 GB model reported 28 MB.
    let peakMemMB = Double(MLX.GPU.peakMemory) / 1_048_576

    let prefillTps = Double(promptTokens.count) / prefillTime
    let decodeTps = Double(genLen) / decodeTime
    let ttftMs = prefillTime * 1000

    return BenchResult(
        prefillToksPerSec: prefillTps,
        decodeToksPerSec: decodeTps,
        ttftMs: ttftMs,
        peakMemoryMB: peakMemMB
    )
}
