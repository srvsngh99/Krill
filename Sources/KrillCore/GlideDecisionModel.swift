import Foundation
import MLX
import MLXNN
import KrillTokenizer

final class GlideDecisionModel: DecisionScoring {
    private static let configFile = "decision_config.json"
    private static let readoutFile = "readout.safetensors"

    private struct Config: Decodable {
        let formatVersion: Int
        let backbone: String
        let attentionMode: String
        let pooling: String
        let promptFormat: String
        let promptLayout: String
        let codes: [String]
        let maxLen: Int
        let truncHead: Int
        let temperature: Float
    }

    private let config: Config
    private let reader: ModernBertEmbeddingModel
    private let head: Linear
    private let tokenizer: KrillTokenizer
    private let cls: Int32
    private let sep: Int32

    static func isBundle(_ directory: URL) -> Bool {
        [configFile, readoutFile].allSatisfy {
            FileManager.default.fileExists(atPath: directory.appendingPathComponent($0).path)
        }
    }

    static func load(directory: URL) async throws -> GlideDecisionModel {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let config = try decoder.decode(Config.self, from: Data(contentsOf: directory.appendingPathComponent(configFile)))
        let readerConfig = try JSONDecoder().decode(
            ModernBertConfig.self, from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        guard config.formatVersion == 1, config.backbone == "encoder",
              config.attentionMode == "noncausal_bidirectional_encoder", config.pooling == "last",
              config.promptFormat == "quick-empty-think-medium-v1", config.promptLayout == "std",
              config.codes.count == DecisionModel.maxSystemOneOptions,
              config.maxLen >= 3, config.maxLen <= readerConfig.maxTokens,
              (0 ..< config.maxLen - 2).contains(config.truncHead),
              config.temperature.isFinite, config.temperature > 0 else {
            throw DecisionError.invalid("unsupported GLiDE configuration")
        }

        let tokenizer = try await KrillTokenizer(from: directory)
        guard let cls = tokenizer.tokenID(for: "[CLS]"), let sep = tokenizer.tokenID(for: "[SEP]"), cls != sep,
              [cls, sep].allSatisfy({ (0 ..< readerConfig.vocabSize).contains($0) }) else {
            throw DecisionError.invalid("GLiDE tokenizer needs distinct [CLS] and [SEP] tokens")
        }
        let reader = ModernBertEmbeddingModel(readerConfig)
        let head = Linear(readerConfig.hiddenSize, config.codes.count, bias: false)
        var weights = try loadWeightArrays(from: directory)
        guard let readout = weights.removeValue(forKey: "weight"), weights.keys.allSatisfy({ $0.hasPrefix("model.") }) else {
            throw DecisionError.invalid("GLiDE needs model.* encoder weights and a readout weight")
        }
        let readerWeights = Dictionary(uniqueKeysWithValues: weights.map { key, value in
            (String(key.dropFirst("model.".count)), value.asType(.bfloat16))
        })
        let strict: Module.VerifyUpdate = [.allModelKeysSet, .shapeMismatch, .noUnusedKeys]
        try reader.update(parameters: ModuleParameters.unflattened(readerWeights), verify: strict)
        try head.update(parameters: ModuleParameters.unflattened(["weight": readout.asType(.bfloat16)]), verify: strict)
        eval(reader, head)
        return GlideDecisionModel(config: config, reader: reader, head: head, tokenizer: tokenizer,
                                  cls: Int32(cls), sep: Int32(sep))
    }

    private init(config: Config, reader: ModernBertEmbeddingModel, head: Linear, tokenizer: KrillTokenizer,
                 cls: Int32, sep: Int32) {
        self.config = config
        self.reader = reader
        self.head = head
        self.tokenizer = tokenizer
        self.cls = cls
        self.sep = sep
    }

    static func prompt(state: String, instructions: String, options: [String], codes: [String]) -> String {
        let lines = zip(codes, options).map { "(\($0)) \($1)" }.joined(separator: "\n")
        let body = "Context:\n\(state)\n\nQuestion: \(instructions)\n\nOptions:\n\(lines)"
            .trimmingCharacters(in: .whitespacesAndNewlines)
        // The render tokenizer's single-user template at medium effort has no system turn.
        return "<|im_start|>user\n\(body)\n\nThink it through briefly, then answer with the option letter only."
            + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer: ("
    }

    static func wrap(_ ids: [Int32], cls: Int32, sep: Int32, limit: Int, head: Int) -> [Int32] {
        let capacity = limit - 2
        if ids.count <= capacity { return [cls] + ids + [sep] }
        return [cls] + ids.prefix(head) + ids.suffix(capacity - head) + [sep]
    }

    private func score(text: String, question: String, options: [String]) -> (probabilities: [Float], inputTokens: Int) {
        let prompt = Self.prompt(state: text, instructions: question, options: options, codes: config.codes)
        let ids = Self.wrap(tokenizer.encodeWithoutSpecialTokens(prompt).map(Int32.init), cls: cls, sep: sep,
                            limit: config.maxLen, head: config.truncHead)
        let hidden = reader.hiddenStates(MLXArray(ids, [1, ids.count]))
        let logits = head(hidden[0, ids.count - 2, 0...])[0 ..< options.count].asType(.float32)
        return (softmax(logits / config.temperature, axis: -1).asArray(Float.self), ids.count)
    }

    func choose(_ texts: [String], options: [String], question: String?, allowNone: Bool) throws -> [DecisionChoice] {
        guard !options.contains(where: { $0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }) else {
            throw DecisionError.invalid("an option must not be empty")
        }
        return texts.map { text in
            let (p, _) = score(text: text, question: question ?? "", options: options)
            let best = p.indices.reduce(0) { p[$1] > p[$0] ? $1 : $0 }
            return DecisionChoice(label: options[best], index: best, scores: p, pNone: 0)
        }
    }

    func candidateProbabilities(text: String, questions: [String], options: [String]) throws
        -> (probabilities: [[Float]], inputTokens: Int)
    {
        let rows = questions.map { score(text: text, question: $0, options: options) }
        return (rows.map(\.probabilities), rows.reduce(0) { $0 + $1.inputTokens })
    }

    func yesNo(texts: [String], question: String) throws -> [DecisionYesNo] {
        texts.map { text in
            let (p, tokens) = score(text: text, question: question, options: ["true: Yes", "false: No"])
            return DecisionYesNo(answer: p[0] >= p[1] ? "yes" : "no", pYes: p[0], inputTokens: tokens)
        }
    }
}
