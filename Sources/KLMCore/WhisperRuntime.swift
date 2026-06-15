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
        weights = weights.mapValues { $0.asType(.float32) }
        let nested = ModuleParameters.unflattened(weights.map { ($0.key, $0.value) })
        try model.update(parameters: nested, verify: [.all])
        eval(model)
        self.model = model

        tokenizer = try WhisperTokenizer(
            vocabURL: modelDir.appendingPathComponent("vocab.json"))
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
