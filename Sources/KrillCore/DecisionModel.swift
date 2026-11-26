import Foundation
import MLX
import MLXNN
import KrillTokenizer

// MARK: - Config

struct DecisionConfig: Decodable {
    struct Head: Decodable {
        let mlpHidden: Int
        let interactionDim: Int
        enum CodingKeys: String, CodingKey { case mlpHidden = "mlp_hidden", interactionDim = "interaction_dim" }
    }
    struct Tau: Decodable {
        let rule: String
        let tau: Float
    }
    struct NoneRule: Decodable {
        let longList: Int
        let short: Tau
        let long: Tau
        enum CodingKeys: String, CodingKey { case longList = "long_list", short, long }
    }
    private struct ReaderIds: Decodable {
        let padTokenId: Int
        enum CodingKeys: String, CodingKey { case padTokenId = "pad_token_id" }
    }

    let reader: ModernBertConfig
    let padId: Int
    let pooling: String
    let stateMaxTokens: Int
    let optionMaxTokens: Int
    let head: Head
    let none: NoneRule

    enum CodingKeys: String, CodingKey {
        case reader, pooling, head, none
        case stateMaxTokens = "state_max_tokens"
        case optionMaxTokens = "option_max_tokens"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        reader = try c.decode(ModernBertConfig.self, forKey: .reader)
        padId = try c.decode(ReaderIds.self, forKey: .reader).padTokenId
        pooling = try c.decode(String.self, forKey: .pooling)
        guard ["cls", "mean"].contains(pooling) else {
            throw DecodingError.dataCorruptedError(forKey: .pooling, in: c, debugDescription: "unknown pooling")
        }
        stateMaxTokens = try c.decode(Int.self, forKey: .stateMaxTokens)
        optionMaxTokens = try c.decode(Int.self, forKey: .optionMaxTokens)
        head = try c.decode(Head.self, forKey: .head)
        none = try c.decode(NoneRule.self, forKey: .none)
        for rule in [none.short.rule, none.long.rule] where !["p_none", "margin"].contains(rule) {
            throw DecodingError.dataCorruptedError(forKey: .none, in: c, debugDescription: "unknown none rule")
        }
    }
}

// MARK: - Head

final class DecisionMLP: Module {
    @ModuleInfo(key: "first") var first: Linear
    @ModuleInfo(key: "last") var last: Linear

    init(_ inputs: Int, _ hidden: Int) {
        _first = ModuleInfo(wrappedValue: Linear(inputs, hidden), key: "first")
        _last = ModuleInfo(wrappedValue: Linear(hidden, 1), key: "last")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let hidden = first(x)
        return last(gelu(hidden.asType(.float32)).asType(hidden.dtype))
    }
}

/// Scores options from the MaxSim term, two extra features and a projected
/// state-option interaction, and scores "none of these" from list statistics.
final class DecisionHead: Module {
    @ModuleInfo(key: "option_mlp") var optionMLP: DecisionMLP
    @ModuleInfo(key: "none_mlp") var noneMLP: DecisionMLP
    @ModuleInfo(key: "interaction_proj") var interactionProj: Linear

    init(hidden: Int, mlp: Int, interaction: Int) {
        _optionMLP = ModuleInfo(wrappedValue: DecisionMLP(3 + interaction, mlp), key: "option_mlp")
        _noneMLP = ModuleInfo(wrappedValue: DecisionMLP(hidden + 6, mlp), key: "none_mlp")
        _interactionProj = ModuleInfo(wrappedValue: Linear(hidden, interaction), key: "interaction_proj")
    }

    func options(maxsim: MLXArray, extra: MLXArray, interaction: MLXArray) -> MLXArray {
        let x = concatenated([maxsim.expandedDimensions(axis: -1), extra, interactionProj(interaction)], axis: -1)
        return optionMLP(x).squeezed(axis: -1)
    }

    func none(_ logits: MLXArray, statePooled: MLXArray) -> MLXArray {
        let n = logits.dim(0)
        let sorted = MLX.sorted(logits)
        let top = sorted[n - 1]
        let top2 = n > 1 ? sorted[n - 2] : MLXArray(Float(0))
        let std = n > 1 ? MLX.sqrt(MLX.variance(logits)) : MLXArray(Float(0))
        let p = softmax(logits, axis: -1)
        let ent = -(p * MLX.log(MLX.maximum(p, 1e-9))).sum() / log(Float(max(n, 2)))
        let agg = stacked([top, logits.mean(), std, top - top2, ent, MLXArray(log(Float(n + 1)))])
        let state = statePooled / MLX.maximum(MLX.sqrt((statePooled * statePooled).sum()), 1e-9)
        return noneMLP(concatenated([state, agg])).squeezed()
    }
}

// MARK: - Model

public enum DecisionError: Error, CustomStringConvertible {
    case tooLong(what: String, tokens: Int, limit: Int)
    case invalid(String)

    public var description: String {
        switch self {
        case .tooLong(let what, let tokens, let limit): return "\(what) has \(tokens) tokens, the limit is \(limit)"
        case .invalid(let message): return message
        }
    }
}

public struct DecisionChoice: Sendable {
    public let label: String?
    public let index: Int?
    public let scores: [Float]
    public let pNone: Float
}

public struct DecisionYesNo: Sendable {
    public let answer: String
    public let pYes: Float
    public let inputTokens: Int
}

func l2Normalized(_ x: MLXArray) -> MLXArray {
    x / MLX.maximum(MLX.sqrt((x * x).sum(axis: -1, keepDims: true)), 1e-12)
}

func paddedDecisionTokens(_ ids: [[Int]], lengths: [Int], longest: Int, padId: Int)
    -> (tokens: MLXArray, mask: MLXArray)
{
    let rows = ids.flatMap { $0 + Array(repeating: padId, count: longest - $0.count) }
    let tokens = MLXArray(rows.map(Int32.init), [ids.count, longest])
    let mask = MLXArray(0 ..< Int32(longest)).reshaped(1, longest)
        .< MLXArray(lengths.map(Int32.init)).reshaped(-1, 1)
    return (tokens, mask)
}

protocol DecisionScoring {
    func choose(_ texts: [String], options: [String], question: String?, allowNone: Bool) throws -> [DecisionChoice]
    func candidateProbabilities(text: String, questions: [String], options: [String]) throws
        -> (probabilities: [[Float]], inputTokens: Int)
    func yesNo(texts: [String], question: String) throws -> [DecisionYesNo]
}

/// Callers serialize inference. The reader-based model caches option states.
public final class DecisionModel: @unchecked Sendable {
    private let scorer: any DecisionScoring

    public static let maxTexts = 256
    public static let maxOptions = 128
    public static let maxSystemOneOptions = 255

    public var isAmber: Bool { scorer is AmberDecisionModel }
    public var isDMJEPA: Bool { scorer is DMJEPADecisionModel }

    public static func isDecisionConfig(_ json: [String: Any]) -> Bool {
        DMJEPADecisionModel.isConfig(json)
            || (json["reader"] is [String: Any] && json["head"] is [String: Any] && json["pooling"] != nil)
    }

    private static func configJSON(in directory: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("config.json")) else { return nil }
        return try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    }

    public static func isBundle(_ directory: URL) -> Bool {
        AmberDecisionModel.isBundle(directory) || configJSON(in: directory).map(isDecisionConfig) == true
    }

    public static func load(directory: URL) async throws -> DecisionModel {
        if AmberDecisionModel.isBundle(directory) {
            return DecisionModel(scorer: try await AmberDecisionModel.load(directory: directory))
        }
        if let json = configJSON(in: directory), DMJEPADecisionModel.isConfig(json) {
            return DecisionModel(scorer: try await DMJEPADecisionModel.load(directory: directory))
        }
        return DecisionModel(scorer: try await ReaderDecisionModel.load(directory: directory))
    }

    private init(scorer: any DecisionScoring) { self.scorer = scorer }

    public static func checkCounts(texts: Int, options: Int) throws {
        try checkCounts(texts: texts, options: options, optionLimit: maxOptions)
    }

    private static func checkCounts(texts: Int, options: Int, optionLimit: Int) throws {
        guard (1 ... maxTexts).contains(texts) else { throw DecisionError.invalid("provide 1 to \(maxTexts) texts") }
        guard options <= optionLimit else { throw DecisionError.invalid("at most \(optionLimit) options") }
    }

    public func choose(_ texts: [String], options: [String], question: String? = nil,
                       allowNone: Bool = true) throws -> [DecisionChoice] {
        guard !options.isEmpty else { throw DecisionError.invalid("options must not be empty") }
        try Self.checkCounts(texts: texts.count, options: options.count)
        return try scorer.choose(texts, options: options, question: question, allowNone: allowNone)
    }

    public func candidateProbabilities(text: String, questions: [String], options: [String],
                                       optionLabels: [String]? = nil) throws
        -> (probabilities: [[Float]], inputTokens: Int)
    {
        guard options.count >= 2 else { throw DecisionError.invalid("provide at least two options") }
        try Self.checkCounts(texts: questions.count, options: options.count, optionLimit: Self.maxSystemOneOptions)
        if let optionLabels, let dmjepa = scorer as? DMJEPADecisionModel {
            return try dmjepa.candidateProbabilities(text: text, questions: questions, options: options, labels: optionLabels)
        }
        return try scorer.candidateProbabilities(text: text, questions: questions, options: options)
    }

    public func yesNo(texts: [String], question: String) throws -> [DecisionYesNo] {
        try Self.checkCounts(texts: texts.count, options: 2)
        return try scorer.yesNo(texts: texts, question: question)
    }
}

private final class ReaderDecisionModel: DecisionScoring {
    private struct OptionStates {
        let tokens: MLXArray
        let mask: MLXArray
        let pooled: MLXArray
        let tokenFraction: MLXArray
        let tokenCount: Int
    }

    private let config: DecisionConfig
    private let reader: ModernBertEmbeddingModel
    private let head: DecisionHead
    private let tokenizer: KrillTokenizer
    private let yesNoOffset: Float
    private var optionCache: [[String]: OptionStates] = [:]
    private var cachedOptions = 0

    private static let readChunk = 8
    private static let cacheOptionLimit = 512

    static func load(directory: URL) async throws -> ReaderDecisionModel {
        let config = try JSONDecoder().decode(
            DecisionConfig.self, from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        let reader = ModernBertEmbeddingModel(config.reader)
        let head = DecisionHead(
            hidden: config.reader.hiddenSize, mlp: config.head.mlpHidden, interaction: config.head.interactionDim)

        var readerWeights: [String: MLXArray] = [:]
        var headWeights: [String: MLXArray] = [:]
        for (key, value) in try loadWeightArrays(from: directory) {
            let fp32 = value.asType(.float32)
            if key.hasPrefix("option_mlp.") || key.hasPrefix("none_mlp.") || key.hasPrefix("interaction_proj.") {
                let renamed = key.replacingOccurrences(of: "_mlp.0.", with: "_mlp.first.")
                    .replacingOccurrences(of: "_mlp.2.", with: "_mlp.last.")
                headWeights[renamed] = fp32
            } else {
                readerWeights[key] = fp32
            }
        }
        let strict: Module.VerifyUpdate = [.allModelKeysSet, .shapeMismatch, .noUnusedKeys]
        try reader.update(parameters: ModuleParameters.unflattened(readerWeights), verify: strict)
        try head.update(parameters: ModuleParameters.unflattened(headWeights), verify: strict)
        eval(reader, head)

        let offsets = (try? Data(contentsOf: directory.appendingPathComponent("calibration.json")))
            .flatMap { try? JSONDecoder().decode([String: [String: Float]].self, from: $0) }?["yes_no_logit_offset"]
        return ReaderDecisionModel(
            config: config, reader: reader, head: head,
            tokenizer: try await KrillTokenizer(from: directory), yesNoOffset: offsets?["default"] ?? 0)
    }

    private init(config: DecisionConfig, reader: ModernBertEmbeddingModel, head: DecisionHead,
                 tokenizer: KrillTokenizer, yesNoOffset: Float) {
        self.config = config
        self.reader = reader
        self.head = head
        self.tokenizer = tokenizer
        self.yesNoOffset = yesNoOffset
    }

    private func read(_ texts: [String], limit: Int, what: String) throws
        -> (states: MLXArray, lengths: [Int], pooled: MLXArray)
    {
        let ids = texts.map { tokenizer.encode($0) }
        let lengths = ids.map(\.count)
        guard !texts.contains(where: { $0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }) else {
            throw DecisionError.invalid("\(what) must not be empty")
        }
        let longest = lengths.max() ?? 0
        guard longest <= limit else { throw DecisionError.tooLong(what: what, tokens: longest, limit: limit) }
        let (tokens, mask) = paddedDecisionTokens(ids, lengths: lengths, longest: longest, padId: config.padId)
        let hasPadding = lengths.contains { $0 != longest }
        let states = reader.hiddenStates(tokens, attentionMask: hasPadding ? mask : nil)
        if config.pooling == "cls" { return (states, lengths, states[0..., 0, 0...]) }
        let m = mask.asType(.float32).expandedDimensions(axis: -1)
        return (states, lengths, (states * m).sum(axis: 1) / MLX.maximum(m.sum(axis: 1), 1e-6))
    }

    private func optionStates(_ options: [String]) throws -> OptionStates {
        if let cached = optionCache[options] { return cached }
        let limit = config.optionMaxTokens
        let (states, lengths, pooled) = try read(options, limit: limit, what: "an option")
        let mask = (MLXArray(0 ..< Int32(states.dim(1))).reshaped(1, -1)
            .< MLXArray(lengths.map(Int32.init)).reshaped(-1, 1)).asType(.float32)
        let built = OptionStates(
            tokens: l2Normalized(states), mask: mask, pooled: pooled,
            tokenFraction: MLX.log(mask.sum(axis: 1) + 1) / log(Float(limit + 1)),
            tokenCount: lengths.reduce(0, +))
        if cachedOptions + options.count > Self.cacheOptionLimit { optionCache.removeAll(); cachedOptions = 0 }
        optionCache[options] = built
        cachedOptions += options.count
        return built
    }

    private func optionLogits(_ opts: OptionStates, state: MLXArray, pooled: MLXArray) -> MLXArray {
        let sims = matmul(opts.tokens, l2Normalized(state).T).max(axis: -1)
        let maxsim = (sims * opts.mask).sum(axis: 1) / MLX.maximum(opts.mask.sum(axis: 1), 1)
        let optionPooled = l2Normalized(opts.pooled)
        let statePooled = l2Normalized(pooled)
        let extra = stacked([matmul(optionPooled, statePooled), opts.tokenFraction], axis: -1)
        return head.options(maxsim: maxsim, extra: extra, interaction: optionPooled * statePooled)
    }

    private func score(_ texts: [String], options: [String]) throws
        -> (logits: [MLXArray], pooled: MLXArray, inputTokens: [Int])
    {
        let opts = try optionStates(options)
        var logits: [MLXArray] = []
        var pooledRows: [MLXArray] = []
        var inputTokens: [Int] = []
        for start in stride(from: 0, to: texts.count, by: Self.readChunk) {
            let chunk = Array(texts[start ..< min(start + Self.readChunk, texts.count)])
            let (states, lengths, pooled) = try read(chunk, limit: config.stateMaxTokens, what: "the text")
            inputTokens.append(contentsOf: lengths.map { $0 + opts.tokenCount })
            for i in chunk.indices {
                logits.append(optionLogits(opts, state: states[i, 0 ..< lengths[i]], pooled: pooled[i]))
                pooledRows.append(pooled[i])
            }
            // Evaluating per chunk frees its states before the next chunk is read.
            eval(Array(logits.suffix(chunk.count)) + Array(pooledRows.suffix(chunk.count)))
        }
        return (logits, stacked(pooledRows), inputTokens)
    }

    func choose(_ texts: [String], options: [String], question: String?, allowNone: Bool) throws -> [DecisionChoice] {
        let states = question.map { q in texts.map { Self.condition($0, on: q) } } ?? texts
        let (logits, pooled, _) = try score(states, options: options)
        let t = options.count > config.none.longList ? config.none.long : config.none.short
        return logits.enumerated().map { i, lg in
            let all = concatenated([lg, head.none(lg, statePooled: pooled[i]).reshaped(1)])
            let p = softmax(all, axis: -1).asArray(Float.self)
            let real = Array(p.dropLast())
            let best = real.indices.reduce(0) { real[$1] > real[$0] ? $1 : $0 }
            let pNone = p[p.count - 1]
            let gap = t.rule == "p_none" ? pNone : pNone - real[best]
            if allowNone && gap >= t.tau { return DecisionChoice(label: nil, index: nil, scores: real, pNone: pNone) }
            return DecisionChoice(label: options[best], index: best, scores: real, pNone: pNone)
        }
    }

    private static func condition(_ text: String, on question: String) -> String { "\(text)\n\n\(question)" }

    func candidateProbabilities(text: String, questions: [String], options: [String]) throws
        -> (probabilities: [[Float]], inputTokens: Int)
    {
        let result = try score(questions.map { Self.condition(text, on: $0) }, options: options)
        return (result.logits.map { softmax($0, axis: -1).asArray(Float.self) }, result.inputTokens.reduce(0, +))
    }

    func yesNo(texts: [String], question: String) throws -> [DecisionYesNo] {
        let options = ["no: \(question)", "yes: \(question)"]
        let longest = options.map { tokenizer.encode($0).count }.max() ?? 0
        guard longest <= config.optionMaxTokens else {
            throw DecisionError.tooLong(what: "the yes_no question with its prefix", tokens: longest,
                                        limit: config.optionMaxTokens)
        }
        let result = try score(texts, options: options)
        return zip(result.logits, result.inputTokens).map { row, inputTokens in
            let lg = row.asArray(Float.self)
            let pYes = 1 / (1 + exp(-(lg[1] - lg[0] + yesNoOffset)))
            return DecisionYesNo(answer: pYes > 0.5 ? "yes" : "no", pYes: pYes, inputTokens: inputTokens)
        }
    }
}
