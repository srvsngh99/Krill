import Foundation
import MLX
import MLXNN
import KrillTokenizer

struct EspressoDecisionRule: Decodable {
    let deltaNone: Double
    let bPass: Double
    let tau: Double
    let minOptions: Int

    func apply(_ probabilities: [Double], keys: [String])
        -> (scores: [Double], best: Int, review: Bool, pNone: Double)
    {
        let none = keys.firstIndex(of: "none_of_these")
        let scoped = none != nil && keys.count >= minOptions
        let pass = keys.count == 2 && keys.contains("FAIL") ? keys.firstIndex(of: "PASS") : nil
        let noneBias = scoped ? deltaNone : 0
        let passBias = pass != nil ? bPass : 0
        var scores = probabilities
        if noneBias != 0 || passBias != 0 {
            var logs = scores.map { log(max($0, 1e-12)) }
            if let none { logs[none] += noneBias }
            if let pass { logs[pass] += passBias }
            scores = Self.softmax(logs)
        }
        let best = scores.indices.dropFirst().reduce(0) { scores[$1] > scores[$0] ? $1 : $0 }
        let review = scoped && tau > 0 && scores[best] / scores.reduce(0, +) < tau
        return (scores, best, review, none.map { scores[$0] } ?? 0)
    }

    static func softmax(_ logits: [Double]) -> [Double] {
        let top = logits.reduce(-Double.infinity, max)
        let values = logits.map { exp($0 - top) }
        let total = values.reduce(0, +)
        return values.map { $0 / total }
    }
}

private final class EspressoHead: Module {
    @ModuleInfo var norm: LayerNorm
    @ModuleInfo(key: "w_p") var stateProjection: Linear
    @ModuleInfo(key: "w_q") var questionProjection: Linear
    @ModuleInfo(key: "w_o") var optionProjection: Linear
    @ModuleInfo(key: "cls_mlp") var classifier: [Linear]
    @ParameterInfo(key: "log_tau") var logTau: MLXArray

    init(hidden: Int, projection: Int) {
        _norm = ModuleInfo(wrappedValue: LayerNorm(dimensions: hidden))
        _stateProjection = ModuleInfo(wrappedValue: Linear(hidden, projection), key: "w_p")
        _questionProjection = ModuleInfo(wrappedValue: Linear(hidden, projection), key: "w_q")
        _optionProjection = ModuleInfo(wrappedValue: Linear(hidden, projection), key: "w_o")
        _classifier = ModuleInfo(wrappedValue: [Linear(hidden, 2 * hidden), Linear(2 * hidden, 1)], key: "cls_mlp")
        _logTau = ParameterInfo(wrappedValue: MLXArray(Float(0)), key: "log_tau")
    }

    func callAsFunction(_ hidden: MLXArray, pool: Range<Int>, markers: [Int32]) -> MLXArray {
        let options = hidden[MLXArray(markers)]
        let state = l2Normalized(stateProjection(norm(hidden[pool].mean(axis: 0, keepDims: true))))
        let question = questionProjection(norm(hidden[1 ..< 2]))
        let projected = l2Normalized(optionProjection(norm(options)) + question)
        let learned = classifier[1](relu(classifier[0](options))).squeezed(axis: -1)
        return matmul(projected, state.squeezed(axis: 0)) / MLX.exp(logTau) + learned
    }
}

final class EspressoDecisionModel: DecisionScoring {
    private static let headsFile = "heads.safetensors"
    private static let ruleFile = "decision_rule.json"

    private struct Config: Decodable {
        let baseModel: String
        let markerIds: [String: Int32]
        let bosTokenId: Int32
        let eosTokenId: Int32
        let maxLen: Int
        let tokenBudget: Int
    }

    private let encoder: EspressoEncoder
    private let head: EspressoHead
    private let tokenizer: KrillTokenizer
    private let config: Config
    private let rule: EspressoDecisionRule
    private let questionMarker: Int32
    private let optionMarker: Int32
    private let schemaMarker: Int32
    private let stateMarker: Int32

    static func configURL(in directory: URL) -> URL {
        let standard = directory.appendingPathComponent("config.json")
        return FileManager.default.fileExists(atPath: standard.path)
            ? standard : directory.appendingPathComponent("espresso_config.json")
    }

    static func isBundle(_ directory: URL) -> Bool {
        [directory.appendingPathComponent(headsFile), directory.appendingPathComponent(ruleFile), configURL(in: directory)]
            .allSatisfy { FileManager.default.fileExists(atPath: $0.path) }
    }

    static func load(directory: URL) async throws -> EspressoDecisionModel {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let config = try decoder.decode(Config.self, from: Data(contentsOf: configURL(in: directory)))
        let rule = try decoder.decode(EspressoDecisionRule.self,
                                      from: Data(contentsOf: directory.appendingPathComponent(ruleFile)))
        let tokenizer = try await KrillTokenizer(from: directory, standaloneBPE: true)
        let model = try EspressoDecisionModel(config: config, rule: rule, tokenizer: tokenizer)
        let encoderWeights = try loadArrays(url: directory.appendingPathComponent("model.safetensors"))
            .compactMap { key, value in
                key.hasPrefix("encoder.") ? (String(key.dropFirst("encoder.".count)), value) : nil
            }
        let headWeights = try loadArrays(url: directory.appendingPathComponent(headsFile)).map { key, value in
            (key.replacingOccurrences(of: "cls_mlp.3.", with: "cls_mlp.1."), value)
        }
        let strict: Module.VerifyUpdate = [.allModelKeysSet, .shapeMismatch, .noUnusedKeys]
        try model.encoder.update(parameters: ModuleParameters.unflattened(encoderWeights), verify: strict)
        try model.head.update(parameters: ModuleParameters.unflattened(headWeights), verify: strict)
        eval(model.encoder, model.head)
        return model
    }

    private init(config: Config, rule: EspressoDecisionRule, tokenizer: KrillTokenizer) throws {
        guard config.baseModel == "vllm-sr/vela3-fast",
              (6 ... 32768).contains(config.maxLen), config.maxLen <= config.tokenBudget,
              let q = config.markerIds["[Q]"], let o = config.markerIds["[O]"],
              let schema = config.markerIds["[SEP_SCHEMA]"], let state = config.markerIds["[SEG_user]"],
              Set([q, o, schema, state]).count == 4,
              [q, o, schema, state, config.bosTokenId, config.eosTokenId].allSatisfy({
                  (0 ..< EspressoEncoder.vocabSize).contains(Int($0))
              }), (0 ... 1).contains(rule.tau), rule.deltaNone.isFinite, rule.bPass.isFinite, rule.minOptions >= 0 else {
            throw DecisionError.invalid("unsupported Espresso configuration")
        }
        self.config = config
        self.rule = rule
        self.tokenizer = tokenizer
        self.questionMarker = q
        self.optionMarker = o
        self.schemaMarker = schema
        self.stateMarker = state
        self.encoder = EspressoEncoder()
        self.head = EspressoHead(hidden: EspressoEncoder.hiddenSize, projection: 256)
    }

    private func encode(_ text: String) -> [Int32] {
        text.isEmpty ? [] : tokenizer.encodeWithoutSpecialTokens(text).map(Int32.init)
    }

    private func score(state: [Int32], instructions: [Int32], options: [[Int32]]) throws
        -> (probabilities: [Double], inputTokens: Int)
    {
        let count = 5 + instructions.count + state.count + options.reduce(0) { $0 + 1 + $1.count }
        guard count <= config.maxLen else {
            throw DecisionError.tooLong(what: "the question, state and options", tokens: count, limit: config.maxLen)
        }
        var ids = [config.bosTokenId, questionMarker] + instructions
        ids.reserveCapacity(count)
        var markers: [Int32] = []
        for option in options {
            markers.append(Int32(ids.count))
            ids.append(optionMarker)
            ids.append(contentsOf: option)
        }
        ids.append(schemaMarker)
        let start = ids.count
        ids.append(stateMarker)
        ids.append(contentsOf: state)
        let stop = ids.count
        ids.append(config.eosTokenId)
        let logits = head(encoder(ids), pool: start ..< stop, markers: markers).asArray(Float.self)
        return (EspressoDecisionRule.softmax(logits.map(Double.init)), count)
    }

    func choose(_ texts: [String], options: [String], question: String?, allowNone: Bool) throws -> [DecisionChoice] {
        guard options.allSatisfy({ !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }) else {
            throw DecisionError.invalid("an option must not be empty")
        }
        let instructions = encode(question ?? "")
        let optionIDs = options.map(encode)
        return try texts.map { text in
            let (p, _) = try score(state: encode(text), instructions: instructions, options: optionIDs)
            let result = rule.apply(p, keys: options)
            let review = allowNone && result.review
            return DecisionChoice(label: review ? nil : options[result.best], index: review ? nil : result.best,
                                  scores: result.scores.map(Float.init), pNone: Float(result.pNone))
        }
    }

    static func choiceKey(_ key: String, description: String) -> String {
        description.lowercased().contains("out of scope") ? "none_of_these" : key
    }

    func candidateProbabilities(text: String, questions: [String], options: [String]) throws
        -> (probabilities: [[Float]], inputTokens: Int)
    {
        try candidateProbabilities(text: text, questions: questions, options: options, choiceKeys: nil)
    }

    func candidateProbabilities(text: String, questions: [String], options: [String], choiceKeys: [String]?) throws
        -> (probabilities: [[Float]], inputTokens: Int)
    {
        if let choiceKeys, choiceKeys.count != options.count {
            throw DecisionError.invalid("Espresso needs a choice key for every option")
        }
        let state = encode(text)
        let keys = choiceKeys.map { zip($0, options).map { Self.choiceKey($0.0, description: $0.1) } }
        if let keys, Set(keys).count != keys.count {
            throw DecisionError.invalid("Espresso needs distinct choice keys after mapping out-of-scope options")
        }
        let optionIDs = keys.map {
            zip($0, options).map { encode($0.1.isEmpty ? $0.0 : "\($0.0): \($0.1)") }
        } ?? options.map(encode)
        let rows = try questions.map { try score(state: state, instructions: encode($0), options: optionIDs) }
        return (rows.map { row in
            let probabilities = keys.map { rule.apply(row.probabilities, keys: $0).scores } ?? row.probabilities
            return probabilities.map(Float.init)
        }, rows.reduce(0) { $0 + $1.inputTokens })
    }

    func yesNo(texts: [String], question: String) throws -> [DecisionYesNo] {
        let instructions = encode(question)
        let options = ["no", "yes"].map(encode)
        return try texts.map { text in
            let (p, tokens) = try score(state: encode(text), instructions: instructions, options: options)
            return DecisionYesNo(answer: p[1] > p[0] ? "yes" : "no", pYes: Float(p[1]), inputTokens: tokens)
        }
    }
}
