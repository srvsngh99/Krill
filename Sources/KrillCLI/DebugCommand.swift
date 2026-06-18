import ArgumentParser
import Foundation
import MLX
import KrillEngine
import KrillCore
import KrillCache
import KrillTokenizer
import KrillSampler

struct DebugCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "debug",
        abstract: "Debug model loading and inference (diagnostic tool)"
    )

    @Argument(help: "Model path")
    var modelPath: String

    func run() async throws {
        let dir = URL(fileURLWithPath: modelPath)

        print("=== Krill Debug ===")
        print()

        // 1. Load model
        print("[1] Loading model from \(modelPath)...")
        let loaded = try loadModel(from: dir)
        print("    Family: \(loaded.family)")
        print("    Layers: \(loaded.numLayers)")
        print("    Vocab: \(loaded.vocabSize)")
        print()

        // 2. Load tokenizer
        print("[2] Loading tokenizer...")
        let tokenizer = try await KrillTokenizer(from: dir)
        print("    EOS token ID: \(tokenizer.eosTokenId)")
        print()

        // 3. Test raw encode/decode
        print("[3] Tokenizer encode/decode test:")
        let testStr = "Hello, world!"
        let encoded = tokenizer.encode(testStr)
        let decoded = tokenizer.decode(encoded)
        print("    Encode '\(testStr)' -> \(encoded.prefix(10))... (\(encoded.count) tokens)")
        print("    Decode back -> '\(decoded)'")
        print()

        // 4. Chat template
        print("[4] Chat template:")
        let messages: [[String: String]] = [["role": "user", "content": "What is 2+2?"]]
        let formatted = tokenizer.applyChatTemplate(messages: messages)
        let fmtPreview = String(formatted.prefix(200))
        print("    Output: '\(fmtPreview)'")
        let chatTokens = tokenizer.encode(formatted)
        print("    Token count: \(chatTokens.count)")
        print("    First 10 tokens: \(Array(chatTokens.prefix(10)))")
        print()

        // 5. Forward pass
        print("[5] Forward pass:")
        let caches = makeKVCaches(numLayers: loaded.numLayers)
        let input = MLXArray(chatTokens.map { Int32($0) }).reshaped(1, chatTokens.count)
        print("    Input shape: \(input.shape)")

        let logits = loaded.forward(input, caches)
        MLX.eval(logits)
        print("    Logits shape: \(logits.shape)")

        // 6. Inspect logits at last position
        print()
        print("[6] Logits analysis (last position):")
        let lastLogits: MLXArray
        if logits.ndim == 3 {
            lastLogits = logits[0, logits.dim(1) - 1]
        } else {
            lastLogits = logits[0]
        }
        MLX.eval(lastLogits)

        let minVal = MLX.min(lastLogits).item(Float.self)
        let maxVal = MLX.max(lastLogits).item(Float.self)
        let meanVal = MLX.mean(lastLogits).item(Float.self)
        print("    Min: \(minVal), Max: \(maxVal), Mean: \(meanVal)")

        // Check if logits are all the same (broken model)
        let stdVal = MLX.sqrt(MLX.mean((lastLogits - MLX.mean(lastLogits)) * (lastLogits - MLX.mean(lastLogits)))).item(Float.self)
        print("    Std: \(stdVal)")

        if stdVal < 0.001 {
            print("    WARNING: Logits have near-zero variance - model may not be loaded correctly!")
        }

        // Top-5 tokens
        print()
        print("[7] Top-5 predicted tokens (argmax):")
        // Get top indices by sorting
        let negLogits = MLXArray(0) - lastLogits
        let sortedIdx = argSort(negLogits, axis: -1)
        MLX.eval(sortedIdx)

        for i in 0..<5 {
            let idx = sortedIdx[i].item(Int.self)
            let logitVal = lastLogits[idx].item(Float.self)
            let text = tokenizer.decode(token: idx)
            print("    #\(i+1): token \(idx) logit=\(String(format: "%.2f", logitVal)) -> '\(text)'")
        }

        // 8. Greedy sample
        print()
        print("[8] Greedy sample (via Sampler):")
        let sampler = Sampler(params: .greedy)
        let sampledToken = sampler.sample(logits)
        let sampledText = tokenizer.decode(token: sampledToken)
        print("    Token ID: \(sampledToken) -> '\(sampledText)'")

        // 9. Generate 5 tokens
        print()
        print("[9] Generate 5 tokens:")
        var nextToken = sampledToken
        var generated: [String] = [sampledText]
        for _ in 0..<4 {
            let tokenInput = MLXArray([Int32(nextToken)]).reshaped(1, 1)
            let nextLogits = loaded.forward(tokenInput, caches)
            MLX.eval(nextLogits)
            nextToken = sampler.sample(nextLogits)
            generated.append(tokenizer.decode(token: nextToken))
        }
        print("    Output: '\(generated.joined())'")
        print()
        print("=== Done ===")
    }
}
