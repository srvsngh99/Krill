import Foundation
import MLX
import MLXNN

// MARK: - Native Whisper transcription runtime

/// End-to-end native Whisper ASR: a converted KrillLM model dir
/// (`tools/convert_whisper.py`: `model.safetensors` + `config.json` +
/// `vocab.json`) loaded into `WhisperModel` + `WhisperTokenizer`, driving the
/// mel front-end, encoder, and a greedy autoregressive decode loop.
///
/// Built for dictation: English, no timestamps, one 30 s window per call.
public final class WhisperRuntime {
    public let model: WhisperModel
    public let tokenizer: WhisperTokenizer
    public let config: WhisperConfig

    public enum RuntimeError: Error, CustomStringConvertible {
        case missingFile(String)
        public var description: String {
            switch self {
            case .missingFile(let f): return "Whisper model dir is missing \(f)"
            }
        }
    }

    public init(modelDir: URL) throws {
        let st = modelDir.appendingPathComponent("model.safetensors")
        guard FileManager.default.fileExists(atPath: st.path) else {
            throw RuntimeError.missingFile("model.safetensors")
        }
        // config.json is optional (defaults cover whisper-small.en).
        var cfg = WhisperConfig()
        let cfgURL = modelDir.appendingPathComponent("config.json")
        if let data = try? Data(contentsOf: cfgURL),
           let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            cfg = WhisperConfig(from: dict)
        }
        config = cfg

        let model = WhisperModel(cfg)
        var weights = try loadWeightArrays(from: modelDir)
        weights = Self.normalizeWeights(weights)
        let nested = ModuleParameters.unflattened(weights.map { ($0.key, $0.value) })
        try model.update(parameters: nested, verify: [.all])
        eval(model)
        self.model = model

        tokenizer = try WhisperTokenizer(
            vocabURL: modelDir.appendingPathComponent("vocab.json"))
    }

    /// Cast to float32 and, when the checkpoint uses the HuggingFace
    /// `transformers` layout (`model.encoder.*` / `model.decoder.*`), remap to
    /// the OpenAI/mlx key hierarchy the modules expect. This lets a raw HF
    /// download load with no Python conversion step.
    static func normalizeWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        let isHF = weights.keys.contains { $0.hasPrefix("model.encoder.") || $0.hasPrefix("model.decoder.") }
        var out = [String: MLXArray]()
        for (k, v) in weights {
            guard isHF else { out[k] = v.asType(.float32); continue }
            guard let mapped = remapHFKey(k) else { continue }   // drop computed tensors
            var t = v.asType(.float32)
            // HF stores Conv1d weights as [out, in, kW]; MLX wants [out, kW, in].
            if mapped.hasSuffix("conv1.weight") || mapped.hasSuffix("conv2.weight"), t.ndim == 3 {
                t = t.transposed(0, 2, 1)
            }
            out[mapped] = t
        }
        return out
    }

    /// Rewrite one HF key to the OpenAI/mlx layout, or nil to drop it (the
    /// encoder positions are computed sinusoids, not loaded).
    private static func remapHFKey(_ key: String) -> String? {
        var k = key
        if k.hasPrefix("model.") { k.removeFirst("model.".count) }
        if k == "encoder.embed_positions.weight" { return nil }
        // Order matters: longer fragments first.
        let subs: [(String, String)] = [
            ("self_attn_layer_norm", "attn_ln"),
            ("encoder_attn_layer_norm", "cross_attn_ln"),
            ("encoder_attn", "cross_attn"),
            ("self_attn", "attn"),
            ("final_layer_norm", "mlp_ln"),
            (".layers.", ".blocks."),
            (".fc1.", ".mlp1."),
            (".fc2.", ".mlp2."),
            (".q_proj.", ".query."),
            (".k_proj.", ".key."),
            (".v_proj.", ".value."),
            (".out_proj.", ".out."),
            ("decoder.embed_tokens.weight", "decoder.token_embedding.weight"),
            ("decoder.embed_positions.weight", "decoder.positional_embedding"),
            ("encoder.layer_norm.", "encoder.ln_post."),
            ("decoder.layer_norm.", "decoder.ln."),
        ]
        for (a, b) in subs { k = k.replacingOccurrences(of: a, with: b) }
        return k
    }

    /// Transcribe a mono 16 kHz waveform. `maxTokens` caps the decode length
    /// (a 30 s window of speech is well under the default).
    public func transcribe(waveform: [Float], maxTokens: Int = 224) -> String {
        let mel = WhisperMel.logMel(waveform: waveform, nMels: config.nMels)
            .expandedDimensions(axis: 0)                   // [1, 3000, nMels]
        let audio = model.encoder(mel)                     // [1, nAudioCtx, D]

        let cache = model.newCache()
        // Prefill the fixed English no-timestamp prompt.
        let prompt = MLXArray(WhisperTokenizer.promptTokens.map { Int32($0) },
                              [1, WhisperTokenizer.promptTokens.count])
        var logits = model.decoder(prompt, audioFeatures: audio, cache: cache)
        var next = greedyText(logits[0, -1])

        let limit = min(maxTokens, config.nTextCtx - WhisperTokenizer.promptTokens.count)
        var generated = [Int]()
        for _ in 0 ..< limit {
            if next == WhisperTokenizer.endOfText { break }
            generated.append(next)
            let tok = MLXArray([Int32(next)], [1, 1])
            logits = model.decoder(tok, audioFeatures: audio, cache: cache)
            next = greedyText(logits[0, 0])
        }
        return tokenizer.decode(generated).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Argmax over the text vocabulary only. Restricting to ids
    /// `[0, startOfTranscript)` suppresses every special and timestamp token
    /// while keeping `<|endoftext|>` (50256) available as the stop signal.
    private func greedyText(_ logitsRow: MLXArray) -> Int {
        let textLogits = logitsRow[0 ..< WhisperTokenizer.startOfTranscript]
        return Int(MLX.argMax(textLogits).item(Int32.self))
    }
}
