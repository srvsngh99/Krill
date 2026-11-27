import CryptoKit
import Foundation
import MLX
import MLXNN
import MLXOptimizers

public struct DecisionTuneExample: Sendable {
    let text: String
    let options: [String]
    let yesNo: Bool
    let target: Int
    let label: String

    public init(json row: [String: Any]?, number: Int) throws {
        guard let row, let text = row["text"] as? String else {
            throw DecisionError.invalid("example \(number): text must be a string")
        }
        let question = row["question"] as? String
        if let rawAnswer = row["answer"] {
            guard let question, let answer = rawAnswer as? String, ["no", "yes"].contains(answer),
                  row["options"] == nil, row["label"] == nil else {
                throw DecisionError.invalid("example \(number): yes/no needs a question and an answer of 'yes' or 'no'")
            }
            self.text = text
            options = ["no: \(question)", "yes: \(question)"]
            yesNo = true
            target = answer == "yes" ? 1 : 0
            label = answer
            return
        }

        guard let options = row["options"] as? [String], !options.isEmpty else {
            throw DecisionError.invalid("example \(number): options must be a nonempty list of strings")
        }
        guard row["question"] == nil || row["question"] is NSNull || question != nil else {
            throw DecisionError.invalid("example \(number): question must be a string")
        }
        if row["label"] is NSNull {
            target = -1
            label = "none"
        } else if let label = row["label"] as? String, let index = options.firstIndex(of: label) {
            target = index
            self.label = label
        } else {
            throw DecisionError.invalid("example \(number): label must be one of the options or null")
        }
        self.text = question.flatMap { $0.isEmpty ? nil : "\(text)\n\n\($0)" } ?? text
        self.options = options
        yesNo = false
    }
}

public struct DecisionTuneOptions: Sendable {
    public let epochs: Int
    public let lr: Float
    public let holdout: Double
    public let seed: Int
    public let force: Bool

    public init(epochs: Int = 30, lr: Float = 3e-3, holdout: Double = 0.2, seed: Int = 0, force: Bool = false) {
        self.epochs = epochs
        self.lr = lr
        self.holdout = holdout
        self.seed = seed
        self.force = force
    }

    public func validate(examples: [DecisionTuneExample]) throws {
        guard epochs > 0, lr.isFinite, lr > 0, (0 ..< 1).contains(holdout) else {
            throw DecisionError.invalid("epochs and lr must be positive, and holdout must be in [0, 1)")
        }
        guard !examples.isEmpty else { throw DecisionError.invalid("at least one example is required") }
    }
}

public struct DecisionTuneReport: Sendable {
    public struct Recall: Sendable {
        public let label: String
        public let trainCount: Int
        public let support: Int
        public let recall: Double?
    }

    public struct Metrics: Sendable {
        public let accuracy: Double
        public let nonePrecision: Double?
        public let noneRecall: Double?
        public let loss: Double
        public let rareClassRecall: [Recall]

        fileprivate var jsonObject: [String: Any] {
            ["accuracy": accuracy, "none_precision": nonePrecision as Any? ?? NSNull(),
             "none_recall": noneRecall as Any? ?? NSNull(), "loss": loss,
             "rare_class_recall": rareClassRecall.map { row in
                 ["label": row.label, "train_count": row.trainCount, "support": row.support,
                  "recall": row.recall as Any? ?? NSNull()] as [String: Any]
             }]
        }
    }

    public let accepted: Bool
    public let trainExamples: Int
    public let holdoutExamples: Int
    public let epochs: Int
    public let bestEpoch: Int
    public let before: Metrics
    public let after: Metrics

    public var jsonObject: [String: Any] {
        ["accepted": accepted, "train_examples": trainExamples, "holdout_examples": holdoutExamples,
         "evaluated_on": holdoutExamples > 0 ? "holdout" : "training", "epochs": epochs, "best_epoch": bestEpoch,
         "before": before.jsonObject, "after": after.jsonObject]
    }
}

private struct TuneFeatures {
    let maxima: MLXArray
    let state: MLXArray
    let options: [Int32]
}

private struct TuneBatch {
    struct Kind: Hashable {
        let yesNo: Bool
        let optionCount: Int
    }

    let kind: Kind
    let maxima: MLXArray
    let states: MLXArray
    let optionIDs: MLXArray
    let targets: MLXArray
    let weights: MLXArray
    let rows: [DecisionTuneExample]
}

func decisionTuneOrder(_ count: Int, seed: Int, epoch: Int) -> [Int] {
    (0 ..< count).map { index in
        (index, SHA256.hash(data: Data("\(seed):\(epoch):\(index)".utf8)))
    }.sorted { $0.1.lexicographicallyPrecedes($1.1) }.map { $0.0 }
}

func decisionTuneClassWeights(_ counts: [String: Int]) -> [String: Float] {
    let floor = Double(counts.values.max()!) / 10
    let inverse = counts.mapValues { 1 / max(Double($0), floor) }
    let total = counts.values.reduce(0, +)
    let mean = counts.keys.sorted().reduce(0.0) { $0 + Double(counts[$1]!) * inverse[$1]! } / Double(total)
    return inverse.mapValues { Float($0 / mean) }
}

extension ReaderDecisionModel {
    func tune(examples: [DecisionTuneExample], options: DecisionTuneOptions, output: URL) throws
        -> DecisionTuneReport
    {
        try options.validate(examples: examples)
        let rows = decisionTuneOrder(examples.count, seed: options.seed, epoch: 0).map { examples[$0] }
        let heldOut = rows.count >= 20 && options.holdout > 0 ? max(1, Int(Double(rows.count) * options.holdout)) : 0
        let counts = rows.dropFirst(heldOut).reduce(into: [String: Int]()) { $0[$1.label, default: 0] += 1 }
        let rare = Set(rows.map(\.label)).sorted {
            let a = counts[$0, default: 0], b = counts[$1, default: 0]
            return a == b ? $0.utf8.lexicographicallyPrecedes($1.utf8) : a < b
        }.prefix(10).map { ($0, counts[$0, default: 0]) }

        let (features, pool) = try tuneFeatures(rows)
        return try Device.withDefaultDevice(.cpu) {
            let training = Self.tuneBatches(heldOut ..< rows.count, rows: rows, features: features,
                                            weights: decisionTuneClassWeights(counts))
            let validation = heldOut == 0 ? training : Self.tuneBatches(0 ..< heldOut, rows: rows, features: features)
            let headFile = directory.appendingPathComponent("head.safetensors")
            let original = try Data(contentsOf: headFile)
            let stored = try MLX.loadArrays(url: headFile)
            let specification = stored.map { (file: $0.key, parameter: DecisionHead.parameterKey(for: $0.key), dtype: $0.value.dtype) }
            let head = DecisionHead(config: config)
            let parameters = Dictionary(uniqueKeysWithValues: stored.map {
                (DecisionHead.parameterKey(for: $0.key), $0.value.asType(.float32))
            })
            try head.update(parameters: ModuleParameters.unflattened(parameters), verify: .all)
            eval(head)
            let evalHead = DecisionHead(config: config)
            // An inactive branch must skip its Adam step, including momentum and bias correction.
            let sharedAdam = Adam(learningRate: options.lr, biasCorrection: true)
            let noneAdam = Adam(learningRate: options.lr, biasCorrection: true)
            let gradient = MLXNN.valueAndGrad(model: head) { (head: DecisionHead, batch: TuneBatch) in
                [(self.tuneLoss(head, batch: batch, pool: pool).losses * batch.weights).mean()]
            }

            let before = try tuneMetrics(head, batches: validation, pool: pool, rare: rare)
            var best = before
            var bestEpoch = 0
            var bestWeights: [String: MLXArray] = [:]
            var ran = 0
            for epoch in 1 ... options.epochs {
                ran = epoch
                for i in decisionTuneOrder(training.count, seed: options.seed, epoch: epoch) {
                    let batch = training[i]
                    let (_, gradients) = gradient(head, batch)
                    let flat = gradients.flattened()
                    sharedAdam.update(model: head, gradients: ModuleParameters.unflattened(flat.filter {
                        $0.0.hasPrefix("option_mlp.") || $0.0.hasPrefix("interaction_proj.")
                    }))
                    if !batch.kind.yesNo {
                        noneAdam.update(model: head, gradients: ModuleParameters.unflattened(flat.filter { $0.0.hasPrefix("none_mlp.") }))
                    }
                    eval(head, sharedAdam, noneAdam)
                }

                if heldOut > 0 || epoch == options.epochs {
                    let updated = Dictionary(uniqueKeysWithValues: head.parameters().flattened())
                    let rounded = specification.map {
                        ($0.file, $0.parameter, updated[$0.parameter]!.asType($0.dtype).contiguous())
                    }
                    let checkpoint = rounded.map { ($0.1, $0.2.asType(.float32)) }
                    evalHead.update(parameters: ModuleParameters.unflattened(checkpoint))
                    let metrics = try tuneMetrics(evalHead, batches: validation, pool: pool, rare: rare)
                    if bestEpoch == 0 || metrics.accuracy > best.accuracy
                        || (metrics.accuracy == best.accuracy && metrics.loss < best.loss) {
                        best = metrics
                        bestEpoch = epoch
                        bestWeights = Dictionary(uniqueKeysWithValues: rounded.map { ($0.0, $0.2) })
                        eval(Array(bestWeights.values))
                    }
                    if heldOut > 0 && epoch - bestEpoch >= 5 { break }
                }
            }

            let accepted = heldOut == 0 || best.accuracy >= before.accuracy || options.force
            if accepted {
                let headDigest = SHA256.hash(data: original).map { String(format: "%02x", $0) }.joined()
                try saveTune(to: output, weights: bestWeights, headDigest: headDigest)
            }
            return DecisionTuneReport(accepted: accepted, trainExamples: rows.count - heldOut, holdoutExamples: heldOut,
                                      epochs: ran, bestEpoch: bestEpoch, before: before, after: best)
        }
    }

    private func tuneFeatures(_ rows: [DecisionTuneExample]) throws -> ([TuneFeatures], MLXArray) {
        struct Encoded {
            let tokens: MLXArray
            let state: MLXArray
        }

        let byState = Dictionary(grouping: rows.indices) { rows[$0].text }
        var seen = Set<String>()
        let states = rows.map(\.text).filter { seen.insert($0).inserted }
        var optionIDs: [String: Int] = [:]
        var lastUse: [String: Int] = [:]
        for (i, state) in states.enumerated() {
            lastUse[state] = i
            for row in byState[state]! {
                for option in rows[row].options {
                    if optionIDs[option] == nil { optionIDs[option] = optionIDs.count }
                    lastUse[option] = i
                }
            }
        }
        var optionPool = [MLXArray?](repeating: nil, count: optionIDs.count)
        var cached: [String: Encoded] = [:]
        var features = [TuneFeatures?](repeating: nil, count: rows.count)

        for start in stride(from: 0, to: states.count, by: 16) {
            let end = min(start + 16, states.count)
            let needed = states[start ..< end].flatMap { state in
                [state] + byState[state]!.flatMap { rows[$0].options }
            }
            seen.removeAll(keepingCapacity: true)
            let missing = needed.filter { cached[$0] == nil && seen.insert($0).inserted }
            for start in stride(from: 0, to: missing.count, by: 16) {
                let chunk = Array(missing[start ..< min(start + 16, missing.count)])
                let ids = try chunk.map { text in
                    let ids = tokenizer.encode(text)
                    if optionIDs[text] != nil, ids.count > config.optionMaxTokens {
                        throw DecisionError.tooLong(what: "an option", tokens: ids.count, limit: config.optionMaxTokens)
                    }
                    if byState[text] != nil, ids.count > config.stateMaxTokens {
                        throw DecisionError.tooLong(what: "the text", tokens: ids.count, limit: config.stateMaxTokens)
                    }
                    guard !ids.isEmpty else { throw DecisionError.invalid("reader input produced no tokens") }
                    return ids
                }
                let order = ids.indices.sorted { ids[$0].count < ids[$1].count }
                var first = 0
                while first < order.count {
                    var last = first + 1
                    while last < order.count, (last - first + 1) * ids[order[last]].count <= 512 { last += 1 }
                    let indices = order[first ..< last]
                    let (tokens, lengths, pooled) = read(indices.map { ids[$0] })
                    var realized: [MLXArray] = []
                    for (i, index) in indices.enumerated() {
                        let text = chunk[index]
                        let normalized = l2Normalized(tokens[i, 0 ..< lengths[i]])
                        let state = pooled[i].asType(.float16)
                        cached[text] = Encoded(tokens: normalized, state: state)
                        realized += [normalized, state]
                        if let id = optionIDs[text] {
                            let fraction = log(Float(lengths[i] + 1)) / log(Float(config.optionMaxTokens + 1))
                            let packed = concatenated([l2Normalized(pooled[i]), MLXArray([fraction])]).asType(.float16)
                            optionPool[id] = packed
                            realized.append(packed)
                        }
                    }
                    eval(realized)
                    first = last
                }
            }

            for i in start ..< end {
                let text = states[i]
                let state = cached[text]!
                var similarities: [String: MLXArray] = [:]
                for index in byState[text]! {
                    let names = rows[index].options
                    let maxima = stacked(names.map { name in
                        if let similarity = similarities[name] { return similarity }
                        let similarity = decisionMaxSimilarity(cached[name]!.tokens, state.tokens).mean().asType(.float16)
                        similarities[name] = similarity
                        return similarity
                    })
                    eval(maxima)
                    features[index] = TuneFeatures(maxima: maxima, state: state.state,
                                                   options: names.map { Int32(optionIDs[$0]!) })
                }
                for name in byState[text]!.flatMap({ rows[$0].options }) + [text] where lastUse[name] == i {
                    cached.removeValue(forKey: name)
                }
            }
        }

        let pool = stacked(optionPool.map { $0! })
        eval(pool)
        return (features.map { $0! }, pool)
    }

    private static func tuneBatches(_ indices: Range<Int>, rows: [DecisionTuneExample], features: [TuneFeatures],
                                    weights: [String: Float] = [:]) -> [TuneBatch] {
        let groups = Dictionary(grouping: indices) {
            TuneBatch.Kind(yesNo: rows[$0].yesNo, optionCount: rows[$0].options.count)
        }
        var batches: [TuneBatch] = []
        for (kind, group) in groups.sorted(by: { $0.value[0] < $1.value[0] }) {
            for start in stride(from: 0, to: group.count, by: 16) {
                let indices = group[start ..< min(start + 16, group.count)]
                let inputs = indices.map { features[$0] }
                let samples = indices.map { rows[$0] }
                batches.append(TuneBatch(kind: kind, maxima: stacked(inputs.map(\.maxima)), states: stacked(inputs.map(\.state)),
                                         optionIDs: MLXArray(inputs.flatMap(\.options), [samples.count, kind.optionCount]),
                                         targets: MLXArray(samples.map { Int32($0.target) }),
                                         weights: MLXArray(samples.map { weights[$0.label, default: 1] }), rows: samples))
            }
        }
        return batches
    }

    private func tuneLoss(_ head: DecisionHead, batch: TuneBatch, pool: MLXArray) -> (losses: MLXArray, scores: MLXArray) {
        let pooled = batch.states.asType(.float32)
        let options = pool[batch.optionIDs].asType(.float32)
        let hidden = config.reader.hiddenSize
        let vectors = options[.ellipsis, 0 ..< hidden]
        let state = l2Normalized(pooled).expandedDimensions(axis: 1)
        let interaction = vectors * state
        let extra = stacked([interaction.sum(axis: -1), options[.ellipsis, hidden]], axis: -1)
        let logits = head.options(maxsim: batch.maxima.asType(.float32), extra: extra, interaction: interaction)
        if batch.kind.yesNo {
            let scores = yesNoLogit(logits)
            return (binaryCrossEntropy(logits: scores, targets: batch.targets.asType(.float32), reduction: .none), scores)
        }

        let scores = head.choiceLogits(logits, statePooled: pooled)
        let rule = config.none.rule(for: batch.kind.optionCount)
        var required = MLXArray(rule.tau)
        if rule.rule != "p_none" {
            let best = softmax(logits, axis: -1).max(axis: -1)
            required = (required + best) / (1 + best)
        }
        required = clip(required, min: Float(1e-6), max: Float(1 - 1e-6))
        let none = scores[.ellipsis, batch.kind.optionCount] - logits.logSumExp(axis: -1)
            - (MLX.log(required) - log1p(-required))
        let isNone = batch.targets .== -1
        let choice = crossEntropy(logits: logits, targets: maximum(batch.targets, Int32(0)))
        let loss = binaryCrossEntropy(logits: none, targets: isNone.asType(.float32), reduction: .none)
        return (which(isNone, 0, choice) + loss, scores)
    }

    private func tuneMetrics(_ head: DecisionHead, batches: [TuneBatch], pool: MLXArray, rare: [(String, Int)]) throws
        -> DecisionTuneReport.Metrics
    {
        var loss = 0.0
        var count = 0, hits = 0, predictedNone = 0, actualNone = 0, trueNone = 0
        var support: [String: Int] = [:], labelHits: [String: Int] = [:]
        for batch in batches {
            let (losses, scores) = tuneLoss(head, batch: batch, pool: pool)
            loss += Double(losses.sum().item(Float.self))
            count += batch.rows.count
            let predictions: [Int]
            if batch.kind.yesNo {
                predictions = MLX.sigmoid(scores).asArray(Float.self).map { $0 > 0.5 ? 1 : 0 }
            } else {
                let n = batch.kind.optionCount
                let rule = config.none.rule(for: n)
                let probs = softmax(scores, axis: -1).asArray(Float.self)
                predictions = stride(from: 0, to: probs.count, by: n + 1).map { start in
                    let best = (0 ..< n).reduce(0) { probs[start + $1] > probs[start + $0] ? $1 : $0 }
                    return rule.isNone(probs[start + n], best: probs[start + best]) ? -1 : best
                }
            }
            for (prediction, row) in zip(predictions, batch.rows) {
                support[row.label, default: 0] += 1
                if prediction == -1 { predictedNone += 1 }
                if row.target == -1 { actualNone += 1 }
                if prediction == row.target {
                    hits += 1
                    labelHits[row.label, default: 0] += 1
                    if prediction == -1 { trueNone += 1 }
                }
            }
        }
        guard loss.isFinite else { throw DecisionError.invalid("tuning produced a non-finite loss") }
        func ratio(_ hits: Int, _ total: Int) -> Double? { total == 0 ? nil : Double(hits) / Double(total) }
        return DecisionTuneReport.Metrics(
            accuracy: Double(hits) / Double(count), nonePrecision: ratio(trueNone, predictedNone),
            noneRecall: ratio(trueNone, actualNone), loss: loss / Double(count),
            rareClassRecall: rare.map { label, trainCount in
                let total = support[label, default: 0]
                return .init(label: label, trainCount: trainCount, support: total,
                             recall: ratio(labelHits[label, default: 0], total))
            })
    }

    private func saveTune(to output: URL, weights: [String: MLXArray], headDigest: String) throws {
        let fm = FileManager.default
        let source = directory.resolvingSymlinksInPath().standardizedFileURL
        let parent = output.deletingLastPathComponent().resolvingSymlinksInPath().standardizedFileURL
        let destination = parent.appendingPathComponent(output.lastPathComponent)
        guard !destination.pathComponents.starts(with: source.pathComponents) else {
            throw DecisionError.invalid("output must be outside the source bundle")
        }
        let stage = parent.appendingPathComponent(".tune-\(UUID().uuidString)")
        try fm.createDirectory(at: stage, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: stage) }

        let generated = ["reader.safetensors", "head.safetensors", "config.json", "__pycache__", ".cache", ".git"]
        for entry in try fm.contentsOfDirectory(at: source, includingPropertiesForKeys: nil)
            where !generated.contains(entry.lastPathComponent) {
            try fm.copyItem(at: entry.resolvingSymlinksInPath(), to: stage.appendingPathComponent(entry.lastPathComponent))
        }

        let reader = source.appendingPathComponent("reader.safetensors").resolvingSymlinksInPath()
        let readerOutput = stage.appendingPathComponent("reader.safetensors")
        do { try fm.linkItem(at: reader, to: readerOutput) }
        catch { try fm.copyItem(at: reader, to: readerOutput) }
        try MLX.save(arrays: weights, url: stage.appendingPathComponent("head.safetensors"))

        let configFile = "config.json"
        let configData = try Data(contentsOf: source.appendingPathComponent(configFile))
        guard let cfg = try JSONSerialization.jsonObject(with: configData) as? [String: Any],
              let brace = configData.firstIndex(of: UInt8(ascii: "{")) else {
            throw DecisionError.invalid("config.json must contain a JSON object")
        }
        // Splice into the original bytes: re-serialising would turn 2.0 into 2, which strict loaders reject.
        let tuned = try JSONSerialization.data(withJSONObject: ["version": cfg["version"] ?? NSNull(), "head_sha256": headDigest],
                                               options: [.sortedKeys])
        var data = configData
        data.insert(contentsOf: Data("\"tuned_from\": ".utf8) + tuned + Data(",".utf8), at: brace + 1)
        try data.write(to: stage.appendingPathComponent(configFile))
        try fm.moveItem(at: stage, to: destination)
    }
}
