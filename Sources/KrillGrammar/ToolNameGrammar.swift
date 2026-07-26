import Foundation

/// Trigger-activated grammar that constrains ONLY the tool-name slot of a tool
/// call, leaving every other token unconstrained.
///
/// # Why this exists
///
/// A model fine-tuned on another harness's tool vocabulary names the right
/// capability with the wrong word - `gemma-4-12b-agentic` asks for `Read` where
/// this harness offers `read_file`. Recovering after the fact (casing rules,
/// alias tables, a second constrained pass) is guesswork that only covers the
/// vocabularies someone thought to enumerate. The durable fix is the one the
/// wider ecosystem settled on - llama.cpp's lazy grammars, XGrammar's structural
/// tags, vLLM's guided decoding: make the invalid name **unrepresentable** at
/// sampling time, so no recovery is needed.
///
/// # Why it is "lazy"
///
/// A tool call is optional: the model must stay free to answer in prose. So the
/// grammar cannot constrain the whole generation. Instead it idles in a
/// pass-through state until the family's unambiguous tool-call sentinel appears
/// (`<tool_call>`, `[TOOL_CALLS]`, ...), constrains the characters of the name
/// value to the offered set, then goes back to pass-through. Everything else -
/// prose, reasoning, argument values - decodes normally.
///
/// # Safety
///
/// Constraining is armed only inside a sentinel-delimited call, and the name key
/// is located with a real (if small) JSON scanner that tracks string and escape
/// context. That matters: a naive search for `"name"` would fire inside an
/// argument value - e.g. `write_file` whose content happens to contain
/// `{"name": ...}` - and corrupt legitimate output. Families with no
/// unambiguous sentinel are deliberately not constrained; see
/// `ToolCallSentinels` in KrillTooling for that policy and its rationale.
///
/// If anything unexpected happens the automaton simply rejects, and the engine's
/// existing fail-open path disables the mask for the rest of the generation. The
/// worst case is therefore exactly the unconstrained behaviour that shipped
/// before, never a stuck decode.
public struct ToolNameAutomaton: GrammarAutomaton {

    public enum State: Hashable, Sendable {
        /// Free text. `opened` counts how many characters of each sentinel have
        /// matched so far, so a sentinel split across tokens still triggers.
        case scanning(progress: [Int])
        /// Inside a call, looking for the name key. Tracks JSON string/escape
        /// state so a `"name"` inside a value cannot arm the constraint.
        case seekingKey(inString: Bool, escaped: Bool, key: String, capturingKey: Bool)
        /// Name key found; consuming whitespace then the `:`.
        case awaitingColon
        /// Colon seen; consuming whitespace then the opening quote.
        case awaitingQuote
        /// Constrained: `matched` is the name prefix emitted so far.
        case inName(matched: String)
    }

    /// Unambiguous tool-call opening sentinels for the active family.
    public let sentinels: [String]
    /// The JSON key holding the tool name (`name` for every supported family).
    public let nameKey: String
    /// The offered tool names. The only values the name slot may take.
    public let names: [String]

    public init(sentinels: [String], nameKey: String = "name", names: [String]) {
        self.sentinels = sentinels
        self.nameKey = nameKey
        self.names = names
    }

    public var initialState: State {
        .scanning(progress: Array(repeating: 0, count: sentinels.count))
    }

    /// EOS is legal everywhere except mid-name: stopping halfway through a name
    /// would emit a truncated tool name. In every other state the model is free
    /// to end its turn, which is what keeps a plain prose answer possible.
    public func isComplete(_ s: State) -> Bool {
        if case .inName = s { return false }
        return true
    }

    public func step(_ s: State, _ c: Character) -> State? {
        switch s {
        case .scanning(let progress):
            return stepScanning(progress, c)

        case .seekingKey(let inString, let escaped, let key, let capturing):
            return stepSeekingKey(inString: inString, escaped: escaped, key: key,
                                  capturing: capturing, c)

        case .awaitingColon:
            if c == ":" { return .awaitingQuote }
            if c.isWhitespace { return .awaitingColon }
            // Not the shape we expected; go back to hunting for the key rather
            // than constraining something that is not a name.
            return .seekingKey(inString: false, escaped: false, key: "", capturingKey: false)

        case .awaitingQuote:
            if c == "\"" { return .inName(matched: "") }
            if c.isWhitespace { return .awaitingQuote }
            return .seekingKey(inString: false, escaped: false, key: "", capturingKey: false)

        case .inName(let matched):
            return stepInName(matched, c)
        }
    }

    // MARK: - Phases

    private func stepScanning(_ progress: [Int], _ c: Character) -> State? {
        var next = progress
        for (i, sentinel) in sentinels.enumerated() {
            let chars = Array(sentinel)
            var p = progress[i]
            // Extend the match, or restart it if this character begins the
            // sentinel afresh (handles `<<tool_call>`).
            if p < chars.count, chars[p] == c {
                p += 1
            } else {
                p = (chars.first == c) ? 1 : 0
            }
            if p == chars.count {
                return .seekingKey(inString: false, escaped: false, key: "", capturingKey: false)
            }
            next[i] = p
        }
        return .scanning(progress: next)
    }

    private func stepSeekingKey(
        inString: Bool, escaped: Bool, key: String, capturing: Bool, _ c: Character
    ) -> State? {
        if escaped {
            return .seekingKey(inString: inString, escaped: false,
                               key: capturing ? key + String(c) : key, capturingKey: capturing)
        }
        if inString {
            if c == "\\" {
                return .seekingKey(inString: true, escaped: true, key: key, capturingKey: capturing)
            }
            if c == "\"" {
                // String closed. If it was a key spelled exactly `nameKey`, the
                // colon and the value follow.
                if capturing && key == nameKey { return .awaitingColon }
                return .seekingKey(inString: false, escaped: false, key: "", capturingKey: false)
            }
            guard capturing else {
                return .seekingKey(inString: true, escaped: false, key: "", capturingKey: false)
            }
            // Keep capturing ONLY while the string is still a prefix of the name
            // key. This is a correctness-neutral bound on the state space, and it
            // matters: every distinct state costs a full vocabulary scan to build
            // its mask, so accumulating an arbitrary string value here would mean
            // one scan per character of every argument the model writes.
            let next = key + String(c)
            guard nameKey.hasPrefix(next) else {
                return .seekingKey(inString: true, escaped: false, key: "", capturingKey: false)
            }
            return .seekingKey(inString: true, escaped: false, key: next, capturingKey: true)
        }
        if c == "\"" {
            // Opening a string. Capture it as a candidate key.
            return .seekingKey(inString: true, escaped: false, key: "", capturingKey: true)
        }
        return .seekingKey(inString: false, escaped: false, key: "", capturingKey: false)
    }

    private func stepInName(_ matched: String, _ c: Character) -> State? {
        if c == "\"" {
            // Closing quote is legal only on a complete name.
            guard names.contains(matched) else { return nil }
            return .scanning(progress: Array(repeating: 0, count: sentinels.count))
        }
        let candidate = matched + String(c)
        // Any offered name still reachable from this prefix keeps the door open.
        guard names.contains(where: { $0.hasPrefix(candidate) }) else { return nil }
        return .inName(matched: candidate)
    }
}
