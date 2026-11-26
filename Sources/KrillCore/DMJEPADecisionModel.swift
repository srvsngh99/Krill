import Foundation
import MLX
import MLXNN
import KrillTokenizer

enum DMJEPAPrompt {
    private static let whitespace = CharacterSet.whitespacesAndNewlines
        .subtracting(CharacterSet(charactersIn: "\u{200b}"))
        .union(CharacterSet(charactersIn: "\u{1c}\u{1d}\u{1e}\u{1f}"))

    static func state(_ state: String, instructions: String) -> String {
        let state = trim(state)
        return "<|state|>\n\(state.isEmpty ? "none" : state)\n<|instructions|>\n\(trim(instructions))"
    }

    static func option(label: String, criterion: String = "") -> String {
        let label = trim(label)
        let criterion = trim(criterion)
        guard !criterion.isEmpty else { return "<|option|> \(label)" }
        let isNumeric = !label.isEmpty && label.unicodeScalars.allSatisfy {
            $0.properties.numericType == .decimal || $0.properties.numericType == .digit
        }
        if label.hasPrefix("k_") || label.hasPrefix("option_") || isNumeric {
            return "<|option|> \(criterion)"
        }
        return "<|option|> \(label): \(criterion)"
    }

    static func trim(_ text: String) -> String {
        text.trimmingCharacters(in: whitespace)
    }
}

private final class DMJEPAPredictorCell: Module {
    let norm0: LayerNorm
    let norm1: LayerNorm
    let norm2: LayerNorm
    let optionAttention: MultiHeadAttention
    let crossAttention: MultiHeadAttention
    let ffn: Sequential
    let gate: Linear

    init(hidden: Int, heads: Int, width: Int) {
        norm0 = LayerNorm(dimensions: hidden)
        norm1 = LayerNorm(dimensions: hidden)
        norm2 = LayerNorm(dimensions: hidden)
        optionAttention = MultiHeadAttention(dimensions: hidden, numHeads: heads, bias: true)
        crossAttention = MultiHeadAttention(dimensions: hidden, numHeads: heads, bias: true)
        ffn = Sequential(layers: Linear(hidden, width), GELU(), Linear(width, hidden))
        gate = Linear(2 * hidden, hidden)
    }

    func callAsFunction(_ latent: MLXArray, state: MLXArray) -> MLXArray {
        let query = norm0(latent)
        let z = latent + optionAttention(query, keys: query, values: query)
        let attended = z + crossAttention(norm1(z), keys: state, values: state)
        let candidate = attended + ffn(norm2(attended))
        let mix = sigmoid(gate(concatenated([z, candidate], axis: -1)))
        return mix * candidate + (1 - mix) * z
    }
}

private final class DMJEPAHead: Module {
    let pool: Sequential
    let cell: DMJEPAPredictorCell
    let output: Sequential
    let logTemperature = MLXArray(Float(0))
    let bias = MLXArray.zeros([1])
    let steps: Int

    init(hidden: Int, heads: Int, width: Int, steps: Int) {
        pool = Sequential(layers: Linear(hidden, hidden), GELU(), LayerNorm(dimensions: hidden))
        cell = DMJEPAPredictorCell(hidden: hidden, heads: heads, width: width)
        output = Sequential(layers: LayerNorm(dimensions: hidden), Linear(hidden, hidden), LayerNorm(dimensions: hidden))
        self.steps = steps
    }

    func callAsFunction(state: MLXArray, pooledOptions: MLXArray) -> MLXArray {
        let options = pool(pooledOptions).expandedDimensions(axis: 0)
        var latent = options
        for _ in 0 ..< steps {
            latent = cell(latent, state: state)
        }
        let scores = (l2Normalized(output(latent)) * l2Normalized(options)).sum(axis: -1)
        return softmax(exp(logTemperature) * scores + bias, axis: -1)[0]
    }
}

final class DMJEPADecisionModel: DecisionScoring {
    private struct Config: Decodable {
        struct Predictor: Decodable {
            let numSteps: Int
            let nhead: Int
            let dFf: Int
        }

        let version: String
        let backboneModel: String
        let dModel: Int
        let latentPredictor: Predictor
        let maxStateLength: Int
        let maxOptionLength: Int
    }

    private let config: Config
    private let reader: ModernBertEmbeddingModel
    private let head: DMJEPAHead
    private let tokenizer: KrillTokenizer
    private static let padID = 50283

    static func isConfig(_ json: [String: Any]) -> Bool {
        json["model_type"] as? String == "dm_jepa"
    }

    static func load(directory: URL) async throws -> DMJEPADecisionModel {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let config = try decoder.decode(Config.self, from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        let predictor = config.latentPredictor
        guard config.version == "1.1", config.backboneModel == "answerdotai/ModernBERT-base",
              config.dModel == 768, predictor.numSteps == 4, predictor.nhead == 8, predictor.dFf == 2048,
              (1 ... 16384).contains(config.maxStateLength), (1 ... 512).contains(config.maxOptionLength) else {
            throw DecisionError.invalid("unsupported DM-JEPA configuration")
        }

        let backboneConfig = Data("""
            {"hidden_size":768,"num_hidden_layers":22,"num_attention_heads":12,"intermediate_size":1152,"vocab_size":50368}
            """.utf8)
        let reader = ModernBertEmbeddingModel(try JSONDecoder().decode(ModernBertConfig.self, from: backboneConfig))
        let head = DMJEPAHead(hidden: config.dModel, heads: predictor.nhead, width: predictor.dFf, steps: predictor.numSteps)
        let tokenizer = try await KrillTokenizer(from: directory)
        guard tokenizer.tokenID(for: "[PAD]") == padID else {
            throw DecisionError.invalid("DM-JEPA needs the ModernBERT-base tokenizer")
        }

        var readerWeights: [String: MLXArray] = [:]
        var headWeights: [String: MLXArray] = [:]
        let backbone = "criteria_encoder.backbone."
        for (key, value) in try loadArrays(url: directory.appendingPathComponent("model.safetensors")) {
            guard value.dtype == .float32 else { throw DecisionError.invalid("DM-JEPA weights must be float32") }
            // PyTorch loads both copies into the shared encoder, with the criteria copy last.
            if key.hasPrefix("state_encoder.backbone.") { continue }
            if key.hasPrefix(backbone) {
                readerWeights[String(key.dropFirst(backbone.count))] = value
                continue
            }
            let name = key.replacingOccurrences(of: "criteria_encoder.pool_proj.", with: "pool.layers.")
                .replacingOccurrences(of: "latent_predictor.out_proj.", with: "output.layers.")
                .replacingOccurrences(of: "latent_predictor.cell.", with: "cell.")
                .replacingOccurrences(of: "cell.ffn.0.", with: "cell.ffn.layers.0.")
                .replacingOccurrences(of: "cell.ffn.3.", with: "cell.ffn.layers.2.")
                .replacingOccurrences(of: "cell.gate.0.", with: "cell.gate.")
                .replacingOccurrences(of: "cell.option_self_attn.", with: "cell.optionAttention.")
                .replacingOccurrences(of: "cell.cross_attn.", with: "cell.crossAttention.")
                .replacingOccurrences(of: "scorer.log_temperature", with: "logTemperature")
                .replacingOccurrences(of: "scorer.bias", with: "bias")
            if let packed = name.range(of: ".in_proj_") {
                guard value.shape.first == 3 * config.dModel else {
                    throw DecisionError.invalid("invalid DM-JEPA attention projection")
                }
                let slices = MLX.split(value, parts: 3, axis: 0)
                for (projection, weight) in zip(["query_proj", "key_proj", "value_proj"], slices) {
                    headWeights["\(name[..<packed.lowerBound]).\(projection).\(name[packed.upperBound...])"] = weight
                }
            } else {
                headWeights[name] = value
            }
        }
        let strict: Module.VerifyUpdate = [.allModelKeysSet, .shapeMismatch, .noUnusedKeys]
        try reader.update(parameters: ModuleParameters.unflattened(readerWeights), verify: strict)
        try head.update(parameters: ModuleParameters.unflattened(headWeights), verify: strict)
        eval(reader, head)
        return DMJEPADecisionModel(config: config, reader: reader, head: head, tokenizer: tokenizer)
    }

    private init(config: Config, reader: ModernBertEmbeddingModel, head: DMJEPAHead, tokenizer: KrillTokenizer) {
        self.config = config
        self.reader = reader
        self.head = head
        self.tokenizer = tokenizer
    }

    private func prepareOptions(_ options: [String], labels: [String]? = nil) throws
        -> (pooled: MLXArray, inputTokens: Int)
    {
        let prompts: [String]
        if let labels {
            guard labels.count == options.count, Set(labels).count == labels.count,
                  labels.allSatisfy({ !DMJEPAPrompt.trim($0).isEmpty }) else {
                throw DecisionError.invalid("provide one distinct non-blank label per option")
            }
            let criteria = Dictionary(uniqueKeysWithValues: zip(labels, options))
            prompts = zip(labels, options).map { label, criterion in
                let fallback: [String]
                switch label.lowercased() {
                case "no", "0": fallback = ["false", "no", "0"]
                case "yes", "1": fallback = ["true", "yes", "1"]
                case "false": fallback = ["no", "0"]
                case "true": fallback = ["yes", "1"]
                default: fallback = []
                }
                let description = criterion.isEmpty
                    ? fallback.lazy.compactMap { criteria[$0] }.first { !$0.isEmpty } ?? ""
                    : criterion
                return DMJEPAPrompt.option(label: label, criterion: description)
            }
        } else {
            guard options.allSatisfy({ !DMJEPAPrompt.trim($0).isEmpty }) else {
                throw DecisionError.invalid("an option must not be empty")
            }
            prompts = options.map { DMJEPAPrompt.option(label: $0) }
        }
        let ids = prompts.map { tokenizer.encode($0) }
        let lengths = ids.map(\.count)
        let longest = lengths.max() ?? 0
        guard longest <= config.maxOptionLength else {
            throw DecisionError.tooLong(what: "an option", tokens: longest, limit: config.maxOptionLength)
        }
        let (tokens, mask) = paddedDecisionTokens(ids, lengths: lengths, longest: longest, padId: Self.padID)
        let hidden = reader.hiddenStates(tokens, attentionMask: mask)
        let weights = mask.asType(.float32).expandedDimensions(axis: -1)
        let pooled = (hidden * weights).sum(axis: 1) / MLX.maximum(weights.sum(axis: 1), 1e-9)
        return (pooled, lengths.reduce(0, +))
    }

    private func score(text: String, question: String, options: MLXArray) throws -> (scores: [Float], inputTokens: Int) {
        let ids = tokenizer.encode(DMJEPAPrompt.state(text, instructions: question))
        guard ids.count <= config.maxStateLength else {
            throw DecisionError.tooLong(what: "the text and question", tokens: ids.count, limit: config.maxStateLength)
        }
        let state = reader.hiddenStates(MLXArray(ids.map(Int32.init), [1, ids.count]))
        return (head(state: state, pooledOptions: options).asArray(Float.self), ids.count)
    }

    func choose(_ texts: [String], options: [String], question: String?, allowNone: Bool) throws -> [DecisionChoice] {
        let (pooled, _) = try prepareOptions(options)
        return try texts.map { text in
            let (p, _) = try score(text: text, question: question ?? "", options: pooled)
            let best = p.indices.dropFirst().reduce(0) { p[$1] > p[$0] ? $1 : $0 }
            return DecisionChoice(label: options[best], index: best, scores: p, pNone: 0)
        }
    }

    func candidateProbabilities(text: String, questions: [String], options: [String]) throws
        -> (probabilities: [[Float]], inputTokens: Int)
    {
        try candidateProbabilities(text: text, questions: questions, options: options, labels: nil)
    }

    func candidateProbabilities(text: String, questions: [String], options: [String], labels: [String]?) throws
        -> (probabilities: [[Float]], inputTokens: Int)
    {
        let (pooled, tokens) = try prepareOptions(options, labels: labels)
        let rows = try questions.map { try score(text: text, question: $0, options: pooled) }
        return (rows.map(\.scores), rows.reduce(0) { $0 + $1.inputTokens + tokens })
    }

    func yesNo(texts: [String], question: String) throws -> [DecisionYesNo] {
        let (pooled, tokens) = try prepareOptions(["no", "yes"], labels: ["false", "true"])
        return try texts.map { text in
            let (p, count) = try score(text: text, question: question, options: pooled)
            return DecisionYesNo(answer: p[1] > p[0] ? "yes" : "no", pYes: p[1], inputTokens: count + tokens)
        }
    }
}
