import Foundation
import MLX
import MLXNN
import KrillTokenizer

struct AmberSequence {
    let ids: [Int32]
    let markers: [Int32]
    let starts: [Int32]
    let ends: [Int32]

    init(instructions: [Int32], state: [Int32], options: [[Int32]], cls: Int32, sep: Int32, mask: Int32,
         limit: Int) throws {
        let count = 4 + instructions.count + state.count + options.reduce(0) { $0 + $1.count + 1 }
        guard count <= limit else {
            throw DecisionError.tooLong(what: "the question, state and options", tokens: count, limit: limit)
        }
        var ids = [cls] + instructions + [sep] + state + [sep]
        ids.reserveCapacity(count)
        var markers: [Int32] = []
        var ends: [Int32] = []
        for option in options {
            markers.append(Int32(ids.count))
            ids.append(mask)
            ids.append(contentsOf: option)
            ends.append(Int32(ids.count))
        }
        ids.append(sep)
        self.ids = ids
        self.markers = markers
        self.starts = markers.map { $0 + 1 }
        self.ends = ends
    }

    func features(_ states: MLXArray) -> MLXArray {
        let h = states.asType(.float32)
        let prefix = h.cumsum(axis: 0, inclusive: false)
        let start = MLXArray(starts), end = MLXArray(ends)
        let length = MLX.maximum(end - start, 1).asType(.float32).expandedDimensions(axis: -1)
        let mean = (prefix[end] - prefix[start]) / length
        let cls = broadcast(h[0], to: [markers.count, h.dim(1)])
        return concatenated([h[MLXArray(markers)], mean, cls], axis: -1)
    }
}

final class AmberDecisionModel: DecisionScoring {
    static let configFile = "uni_config.json"

    private struct Config: Decodable {
        let layout: String
        let hidden: Int
        let width: Int
        let orders: Int?
    }

    private let reader: ModernBertEmbeddingModel
    private let head: DecisionMLP
    private let tokenizer: KrillTokenizer
    private let cls: Int32
    private let sep: Int32
    private let mask: Int32

    static func isBundle(_ directory: URL) -> Bool {
        FileManager.default.fileExists(atPath: directory.appendingPathComponent(configFile).path)
    }

    static func load(directory: URL) async throws -> AmberDecisionModel {
        let decoder = JSONDecoder()
        let config = try decoder.decode(Config.self, from: Data(contentsOf: directory.appendingPathComponent(configFile)))
        let encoder = directory.appendingPathComponent("encoder")
        let encoderData = try Data(contentsOf: encoder.appendingPathComponent("config.json"))
        let readerConfig = try decoder.decode(ModernBertConfig.self, from: encoderData)
        guard config.layout == "uni", (config.orders ?? 1) == 1,
              config.hidden > 0, config.hidden == readerConfig.hiddenSize, config.width > 0 else {
            throw DecisionError.invalid("unsupported Amber configuration")
        }
        let tokenizer = try await KrillTokenizer(from: encoder)
        let reader = ModernBertEmbeddingModel(readerConfig)
        let head = DecisionMLP(3 * config.hidden, config.width)
        var readerWeights: [String: MLXArray] = [:]
        var headWeights: [String: MLXArray] = [:]
        for (key, value) in try loadWeightArrays(from: directory) {
            let weight = value.asType(.bfloat16)
            if key.hasPrefix("encoder.") {
                readerWeights[String(key.dropFirst("encoder.".count))] = weight
            } else {
                let name = key.replacingOccurrences(of: "head.0.", with: "first.")
                    .replacingOccurrences(of: "head.3.", with: "last.")
                headWeights[name] = weight
            }
        }
        let strict: Module.VerifyUpdate = [.allModelKeysSet, .shapeMismatch, .noUnusedKeys]
        try reader.update(parameters: ModuleParameters.unflattened(readerWeights), verify: strict)
        try head.update(parameters: ModuleParameters.unflattened(headWeights), verify: strict)
        eval(reader, head)
        return try AmberDecisionModel(reader: reader, head: head, tokenizer: tokenizer)
    }

    private init(reader: ModernBertEmbeddingModel, head: DecisionMLP, tokenizer: KrillTokenizer) throws {
        guard let cls = tokenizer.tokenID(for: "[CLS]"), let sep = tokenizer.tokenID(for: "[SEP]"),
              let mask = tokenizer.tokenID(for: "[MASK]"),
              [cls, sep, mask].allSatisfy({ (0 ..< reader.config.vocabSize).contains($0) }),
              Set([cls, sep, mask]).count == 3 else {
            throw DecisionError.invalid("Amber tokenizer needs distinct [CLS], [SEP], and [MASK] tokens")
        }
        self.reader = reader
        self.head = head
        self.tokenizer = tokenizer
        self.cls = Int32(cls)
        self.sep = Int32(sep)
        self.mask = Int32(mask)
    }

    private func encode(_ text: String) -> [Int32] {
        tokenizer.encodeWithoutSpecialTokens(text).map(Int32.init)
    }

    private func score(instructions: [Int32], state: [Int32], options: [[Int32]]) throws
        -> (probabilities: [Float], inputTokens: Int)
    {
        let sequence = try AmberSequence(instructions: instructions, state: state, options: options,
                                         cls: cls, sep: sep, mask: mask, limit: reader.config.maxTokens)
        let tokens = MLXArray(sequence.ids, [1, sequence.ids.count])
        let hidden = reader.hiddenStates(tokens)[0]
        let features = sequence.features(hidden).asType(.bfloat16)
        let logits = head(features).squeezed(axis: -1).asType(.float32)
        return (softmax(logits, axis: -1).asArray(Float.self), sequence.ids.count)
    }

    private func encodeOptions(_ options: [String]) throws -> [[Int32]] {
        guard !options.contains(where: { $0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }) else {
            throw DecisionError.invalid("an option must not be empty")
        }
        return options.map(encode)
    }

    func choose(_ texts: [String], options: [String], question: String?, allowNone: Bool) throws -> [DecisionChoice] {
        if (question ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
           texts.contains(where: { $0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }) {
            throw DecisionError.invalid("the text must not be empty")
        }
        let instructions = encode(question ?? "")
        let optionIDs = try encodeOptions(options)
        return try texts.map { text in
            let (p, _) = try score(instructions: instructions, state: encode(text), options: optionIDs)
            let best = p.indices.reduce(0) { p[$1] > p[$0] ? $1 : $0 }
            return DecisionChoice(label: options[best], index: best, scores: p, pNone: 0)
        }
    }

    func candidateProbabilities(text: String, questions: [String], options: [String]) throws
        -> (probabilities: [[Float]], inputTokens: Int)
    {
        let state = encode(text)
        let optionIDs = try encodeOptions(options)
        let rows = try questions.map { try score(instructions: encode($0), state: state, options: optionIDs) }
        return (rows.map(\.probabilities), rows.reduce(0) { $0 + $1.inputTokens })
    }

    func yesNo(texts: [String], question: String) throws -> [DecisionYesNo] {
        let instructions = encode(question)
        let options = ["yes", "no"].map(encode)
        return try texts.map { text in
            let (p, tokens) = try score(instructions: instructions, state: encode(text), options: options)
            return DecisionYesNo(answer: p[0] >= p[1] ? "yes" : "no", pYes: p[0], inputTokens: tokens)
        }
    }
}
