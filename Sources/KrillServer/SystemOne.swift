import Foundation
import NIOCore
import KrillCore
import KrillEngine

struct SystemOneRequest: Sendable {
    static let path = "/v1/systemone"
    static let maxBodySize = 32 * 1024 * 1024
    private static let maxTextSize = 64 * 1024

    enum ParseError: Error {
        case invalid(String)
        case textTooLarge
    }

    private enum Kind: String, Sendable { case choice, noul, score }

    private struct Question: Sendable {
        let kind: Kind
        let instructions: String
        let keys: [String]
        let options: [String]

        func answer(_ scores: [Float]) throws -> [String: Any] {
            let total = scores.reduce(0.0) { $0 + Double($1) }
            guard total.isFinite, total > 0 else { throw ScoreError.invalidProbabilities }
            let p = scores.map { Double($0) / total }
            if kind == .noul { return ["type": kind.rawValue, "noul": p[0]] }
            let entropy = -p.filter { $0 > 0 }.reduce(0) { $0 + $1 * log($1) }
            var result: [String: Any] = [
                "type": kind.rawValue,
                "probabilities": Dictionary(uniqueKeysWithValues: zip(keys, p)),
                "confidence": max(0, 1 - entropy / log(Double(p.count))),
            ]
            if kind == .choice {
                let best = p.indices.dropFirst().reduce(0) { p[$1] > p[$0] ? $1 : $0 }
                result["choice"] = keys[best]
            } else {
                result["score"] = p.enumerated().reduce(0) { $0 + Double($1.offset) * $1.element }
                result["legend"] = Dictionary(uniqueKeysWithValues: zip(keys, options))
            }
            return result
        }
    }

    private enum ScoreError: Error { case invalidProbabilities }

    let model: String
    let keepAlive: Int?
    private let state: String
    private let questions: [String: Question]

    init(_ body: ByteBuffer) throws {
        guard let json = ServerParsing.jsonObject(from: body) else { throw ParseError.invalid("Invalid JSON") }
        guard let model = Self.nonblank(json["model"]) else {
            throw ParseError.invalid("'model' must be a non-blank string")
        }
        self.model = model
        if let state = Self.nonblank(json["state"]) {
            self.state = state
        } else if let rawState = json["state"], JSONSerialization.isValidJSONObject(rawState) {
            let data = try JSONSerialization.data(withJSONObject: rawState, options: [.sortedKeys, .withoutEscapingSlashes])
            self.state = String(decoding: data, as: UTF8.self)
        } else {
            throw ParseError.invalid("'state' must be a non-blank string, object, or array")
        }
        if let images = json["images"] {
            guard let images = images as? [Any] else { throw ParseError.invalid("'images' must be an array") }
            guard images.isEmpty else { throw ParseError.invalid("Krill decision models are text-only and do not support images") }
        }
        self.keepAlive = KeepAliveParse.seconds(from: json["keep_alive"])
        if json["keep_alive"] != nil, keepAlive == nil {
            throw ParseError.invalid("'keep_alive' must be seconds or a duration string")
        }
        guard let rawQuestions = json["questions"] as? [String: [String: Any]],
              (1 ... 64).contains(rawQuestions.count) else {
            throw ParseError.invalid("'questions' must be an object with 1 to 64 questions")
        }
        let schema = try JSONSerialization.data(withJSONObject: rawQuestions, options: [.withoutEscapingSlashes])
        guard state.utf8.count + schema.count <= Self.maxTextSize else { throw ParseError.textTooLarge }
        let choiceNames = Set(rawQuestions.keys.filter { rawQuestions[$0]?["type"] as? String == Kind.choice.rawValue })
        let orderedKeys = choiceNames.isEmpty ? [:] : try Self.choiceKeys(in: body.readableBytesView, names: choiceNames)
        var questions: [String: Question] = [:]
        for (name, raw) in rawQuestions {
            guard name.wholeMatch(of: #/[A-Za-z0-9_.-]{1,100}/#) != nil else {
                throw ParseError.invalid("Invalid question name '\(name)'")
            }
            questions[name] = try Self.question(named: name, raw, orderedKeys: orderedKeys[name] ?? [])
        }
        self.questions = questions
    }

    private static func nonblank(_ raw: Any?) -> String? {
        guard let text = raw as? String, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
        return text
    }

    private static func question(named name: String, _ raw: [String: Any], orderedKeys: [String]) throws -> Question {
        guard let type = raw["type"] as? String, let kind = Kind(rawValue: type),
              let instructions = nonblank(raw["instructions"]) else {
            throw ParseError.invalid("Question '\(name)' needs type choice, noul, or score and non-blank instructions")
        }
        let keys: [String]
        let options: [String]
        switch kind {
        case .choice:
            let maxOptions = DecisionModel.maxSystemOneOptions
            guard let criteria = raw["criteria"] as? [String: Any],
                  (2 ... maxOptions).contains(criteria.count), orderedKeys.count == criteria.count else {
                throw ParseError.invalid("Choice '\(name)' needs an object with 2 to \(maxOptions) criteria")
            }
            keys = orderedKeys
            options = try keys.map { key in
                guard nonblank(key) != nil, let value = criteria[key],
                      let option = value is NSNull ? key : nonblank(value) else {
                    throw ParseError.invalid("Choice '\(name)' needs non-blank keys and non-blank descriptions or null")
                }
                return option
            }
        case .noul:
            keys = []
            if let value = raw["criteria"] {
                guard let criteria = value as? [String: String],
                      criteria.allSatisfy({ ["true", "false"].contains($0.key) && nonblank($0.value) != nil }) else {
                    throw ParseError.invalid("Noul '\(name)' criteria must map 'true' and/or 'false' to non-blank descriptions")
                }
                options = [criteria["true"] ?? "true", criteria["false"] ?? "false"]
            } else {
                options = []
            }
        case .score:
            guard let criteria = raw["criteria"] as? [String], (2 ... 10).contains(criteria.count),
                  criteria.allSatisfy({ nonblank($0) != nil }) else {
                throw ParseError.invalid("Score '\(name)' needs an array of 2 to 10 non-blank descriptions")
            }
            options = criteria
            keys = options.indices.map { String($0) }
        }
        return Question(kind: kind, instructions: instructions, keys: keys, options: options)
    }

    func response(using engine: DecisionEngine, model loaded: DecisionModel) throws -> [String: Any] {
        var answers: [String: [String: Any]] = [:]
        var inputTokens = 0
        for (options, group) in Dictionary(grouping: questions, by: { $0.value.options }) {
            if options.isEmpty {
                for (instructions, sameQuestion) in Dictionary(grouping: group, by: { $0.value.instructions }) {
                    let result = try engine.yesNo(loaded, [state], question: instructions)[0]
                    guard (0 ... 1).contains(result.pYes) else { throw ScoreError.invalidProbabilities }
                    inputTokens += result.inputTokens * sameQuestion.count
                    for (name, _) in sameQuestion { answers[name] = ["type": "noul", "noul": result.pYes] }
                }
            } else {
                let scored = try engine.candidateProbabilities(
                    loaded, text: state, questions: group.map { $0.value.instructions }, options: options)
                inputTokens += scored.inputTokens
                for ((name, question), p) in zip(group, scored.probabilities) {
                    answers[name] = try question.answer(p)
                }
            }
        }
        return ["model": model, "answers": answers,
                "usage": ["input_tokens": inputTokens, "output_tokens": answers.count]]
    }

    private static func choiceKeys(in bytes: ByteBufferView, names: Set<String>) throws -> [String: [String]] {
        var scanner = JSONMemberScanner(bytes)
        var keys: [String: [String]] = [:]
        try scanner.object { key, root in
            guard key == "questions" else { return root.skipValue() }
            try root.object { name, questions in
                guard names.contains(name) else { return questions.skipValue() }
                try questions.object { field, question in
                    guard field == "criteria" else { return question.skipValue() }
                    var ordered: [String] = []
                    try question.object { option, criteria in
                        ordered.append(option)
                        criteria.skipValue()
                    }
                    keys[name] = ordered
                }
            }
        }
        return keys
    }
}

// Foundation validates the JSON and decodes values. This only walks the objects
// whose member order matters, skipping the contents of other values.
private struct JSONMemberScanner {
    private let bytes: ByteBufferView
    private var index: Int
    private var current: UInt8? { index < bytes.endIndex ? bytes[index] : nil }

    init(_ bytes: ByteBufferView) {
        self.bytes = bytes
        self.index = bytes.startIndex + (bytes.starts(with: [0xef, 0xbb, 0xbf]) ? 3 : 0)
    }

    mutating func object(_ visit: (String, inout Self) throws -> Void) throws {
        let invalid = SystemOneRequest.ParseError.invalid("Invalid JSON object")
        guard consume("{") else { throw invalid }
        var seen: Set<String> = []
        while !consume("}") {
            let start = index
            guard current == UInt8(ascii: "\"") else { throw invalid }
            skipString()
            let key = try JSONDecoder().decode(String.self, from: Data(bytes[start ..< index]))
            guard seen.insert(key).inserted else {
                throw SystemOneRequest.ParseError.invalid("Duplicate JSON key '\(key)'")
            }
            guard consume(":") else { throw invalid }
            try visit(key, &self)
            if consume("}") { return }
            guard consume(",") else { throw invalid }
        }
    }

    private mutating func consume(_ token: Unicode.Scalar) -> Bool {
        while let byte = current, byte == 32 || byte == 9 || byte == 10 || byte == 13 { index += 1 }
        guard current == UInt8(ascii: token) else { return false }
        index += 1
        return true
    }

    mutating func skipValue() {
        var depth = 0
        while let byte = current {
            switch byte {
            case UInt8(ascii: "\""):
                skipString()
                continue
            case UInt8(ascii: "{"), UInt8(ascii: "["): depth += 1
            case UInt8(ascii: "}"), UInt8(ascii: "]"):
                guard depth > 0 else { return }
                depth -= 1
            case UInt8(ascii: ",") where depth == 0: return
            default: break
            }
            index += 1
        }
    }

    private mutating func skipString() {
        index += 1
        while let byte = current {
            index += 1
            if byte == UInt8(ascii: "\"") { return }
            if byte == UInt8(ascii: "\\"), current != nil { index += 1 }
        }
    }
}
