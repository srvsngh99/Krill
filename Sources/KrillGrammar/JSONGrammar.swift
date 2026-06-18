import Foundation

/// Incremental, character-driven automaton for a single JSON value
/// (RFC 8259). It is the grammar runtime behind token-level
/// grammar-constrained decoding (parity plan §8, Stage A): given a parser
/// `State` and the next character, `step` returns the new state or `nil`
/// if the character cannot extend the output into a valid JSON-value
/// prefix.
///
/// Pure Swift, no MLX — `JSONTokenMask` layers the tokenizer-vocab logit
/// mask on top. The automaton is a small pushdown machine: a `stack` of
/// open containers plus a lexical `Lex` sub-state. Numbers are not
/// self-delimiting, so a number lexeme completes lazily when a
/// non-number character arrives and that character is re-processed in the
/// post-value context (see the `step(completeValue(stack), c)` calls).
public enum JSONGrammar {

    /// An open structural container on the parse stack.
    public enum Container: Hashable, Sendable { case object, array }

    /// Lexical sub-state. `key: Bool` distinguishes an object key string
    /// (followed by `:`) from a value string (which completes a value).
    public enum Lex: Hashable, Sendable {
        /// Expecting the start of a value (top level, after `:`, or after
        /// `,` inside an array — none of which may close a container).
        case value
        /// A value just completed; expecting `,` or a container close
        /// (or EOS at the top level).
        case afterValue
        /// Just consumed `{`: expecting a key string or `}`.
        case objectExpectKeyOrClose
        /// Just consumed `,` inside an object: expecting a key string
        /// (a trailing comma is invalid, so no `}` here).
        case objectExpectKey
        case inString(key: Bool)
        case strEscape(key: Bool)
        case strUnicode(key: Bool, digits: Int)
        /// Key string closed: expecting `:`.
        case afterKey
        /// Just consumed `[`: expecting a value or `]`.
        case arrayExpectValueOrClose
        // Number lexemes. Terminal (a complete number) are flagged in
        // `isComplete`: numIntZero, numInt, numFrac, numExp.
        case numAfterSign     // saw '-', need a digit
        case numIntZero       // saw a lone leading '0'
        case numInt           // in integer digits (started 1-9)
        case numDotDigit      // saw '.', need a fractional digit
        case numFrac          // in fractional digits
        case numExpSign       // saw e/E, may take +/- or a digit
        case numExpDigit      // saw e/E sign, need an exponent digit
        case numExp           // in exponent digits
        // Literal lexeme: `rest` is the suffix still to be matched, e.g.
        // "rue" after consuming the 't' of `true`.
        case lit(rest: String)
    }

    public struct State: Hashable, Sendable {
        public var stack: [Container]
        public var lex: Lex
        /// True once the single top-level value is fully complete.
        public var done: Bool

        public init(stack: [Container], lex: Lex, done: Bool) {
            self.stack = stack
            self.lex = lex
            self.done = done
        }
    }

    /// Start of generation: expecting the first value, nothing emitted yet.
    public static let initialState = State(stack: [], lex: .value, done: false)

    /// Whether the output so far is a complete, balanced JSON value — i.e.
    /// whether EOS may be emitted here.
    public static func isComplete(_ s: State) -> Bool {
        guard s.stack.isEmpty else { return false }
        if s.done { return true }
        // A top-level number completes lazily, so its terminal lexemes are
        // valid stopping points even without a trailing delimiter.
        switch s.lex {
        case .numIntZero, .numInt, .numFrac, .numExp: return true
        default: return false
        }
    }

    private static func isWS(_ c: Character) -> Bool {
        c == " " || c == "\t" || c == "\n" || c == "\r"
    }
    private static func isDigit(_ c: Character) -> Bool { c >= "0" && c <= "9" }
    private static func isHex(_ c: Character) -> Bool {
        isDigit(c) || (c >= "a" && c <= "f") || (c >= "A" && c <= "F")
    }

    /// State after a value completes given the current (already-popped) stack.
    private static func completeValue(_ stack: [Container]) -> State {
        State(stack: stack, lex: .afterValue, done: stack.isEmpty)
    }

    /// Begin a value with `c` in value position (no container-close allowed).
    private static func beginValue(_ c: Character, _ stack: [Container]) -> State? {
        switch c {
        case "{": return State(stack: stack + [.object], lex: .objectExpectKeyOrClose, done: false)
        case "[": return State(stack: stack + [.array], lex: .arrayExpectValueOrClose, done: false)
        case "\"": return State(stack: stack, lex: .inString(key: false), done: false)
        case "-": return State(stack: stack, lex: .numAfterSign, done: false)
        case "0": return State(stack: stack, lex: .numIntZero, done: false)
        case "t": return State(stack: stack, lex: .lit(rest: "rue"), done: false)
        case "f": return State(stack: stack, lex: .lit(rest: "alse"), done: false)
        case "n": return State(stack: stack, lex: .lit(rest: "ull"), done: false)
        default:
            if isDigit(c) { return State(stack: stack, lex: .numInt, done: false) }  // 1-9
            return nil
        }
    }

    /// Advance the automaton by one character. Returns `nil` if `c` cannot
    /// extend the output into a valid JSON-value prefix.
    public static func step(_ s: State, _ c: Character) -> State? {
        if s.done {
            return isWS(c) ? s : nil  // only trailing whitespace after the value
        }
        let stack = s.stack
        switch s.lex {
        case .value:
            if isWS(c) { return s }
            return beginValue(c, stack)

        case .arrayExpectValueOrClose:
            if isWS(c) { return s }
            if c == "]" { var st = stack; st.removeLast(); return completeValue(st) }
            return beginValue(c, stack)

        case .objectExpectKeyOrClose:
            if isWS(c) { return s }
            if c == "}" { var st = stack; st.removeLast(); return completeValue(st) }
            if c == "\"" { return State(stack: stack, lex: .inString(key: true), done: false) }
            return nil

        case .objectExpectKey:
            if isWS(c) { return s }
            if c == "\"" { return State(stack: stack, lex: .inString(key: true), done: false) }
            return nil

        case .inString(let key):
            if c == "\"" {
                return key
                    ? State(stack: stack, lex: .afterKey, done: false)
                    : completeValue(stack)
            }
            if c == "\\" { return State(stack: stack, lex: .strEscape(key: key), done: false) }
            // Unescaped control characters are invalid inside a JSON string.
            if c.unicodeScalars.count == 1, let a = c.unicodeScalars.first, a.value < 0x20 {
                return nil
            }
            return s

        case .strEscape(let key):
            switch c {
            case "\"", "\\", "/", "b", "f", "n", "r", "t":
                return State(stack: stack, lex: .inString(key: key), done: false)
            case "u":
                return State(stack: stack, lex: .strUnicode(key: key, digits: 0), done: false)
            default:
                return nil
            }

        case .strUnicode(let key, let digits):
            guard isHex(c) else { return nil }
            return digits == 3
                ? State(stack: stack, lex: .inString(key: key), done: false)
                : State(stack: stack, lex: .strUnicode(key: key, digits: digits + 1), done: false)

        case .afterKey:
            if isWS(c) { return s }
            if c == ":" { return State(stack: stack, lex: .value, done: false) }
            return nil

        case .afterValue:
            if isWS(c) { return s }
            guard let top = stack.last else { return nil }  // top level: only trailing WS
            switch top {
            case .array:
                if c == "," { return State(stack: stack, lex: .value, done: false) }
                if c == "]" { var st = stack; st.removeLast(); return completeValue(st) }
                return nil
            case .object:
                if c == "," { return State(stack: stack, lex: .objectExpectKey, done: false) }
                if c == "}" { var st = stack; st.removeLast(); return completeValue(st) }
                return nil
            }

        case .numAfterSign:
            if c == "0" { return State(stack: stack, lex: .numIntZero, done: false) }
            if isDigit(c) { return State(stack: stack, lex: .numInt, done: false) }
            return nil

        case .numIntZero:
            if c == "." { return State(stack: stack, lex: .numDotDigit, done: false) }
            if c == "e" || c == "E" { return State(stack: stack, lex: .numExpSign, done: false) }
            if isDigit(c) { return nil }  // no leading zeros (e.g. "01")
            return step(completeValue(stack), c)

        case .numInt:
            if isDigit(c) { return s }
            if c == "." { return State(stack: stack, lex: .numDotDigit, done: false) }
            if c == "e" || c == "E" { return State(stack: stack, lex: .numExpSign, done: false) }
            return step(completeValue(stack), c)

        case .numDotDigit:
            if isDigit(c) { return State(stack: stack, lex: .numFrac, done: false) }
            return nil

        case .numFrac:
            if isDigit(c) { return s }
            if c == "e" || c == "E" { return State(stack: stack, lex: .numExpSign, done: false) }
            return step(completeValue(stack), c)

        case .numExpSign:
            if c == "+" || c == "-" { return State(stack: stack, lex: .numExpDigit, done: false) }
            if isDigit(c) { return State(stack: stack, lex: .numExp, done: false) }
            return nil

        case .numExpDigit:
            if isDigit(c) { return State(stack: stack, lex: .numExp, done: false) }
            return nil

        case .numExp:
            if isDigit(c) { return s }
            return step(completeValue(stack), c)

        case .lit(let rest):
            guard let first = rest.first else { return step(completeValue(stack), c) }
            guard c == first else { return nil }
            let r = String(rest.dropFirst())
            return r.isEmpty
                ? completeValue(stack)
                : State(stack: stack, lex: .lit(rest: r), done: false)
        }
    }

    /// Advance the automaton over every character of `piece`. Returns the
    /// resulting state, or `nil` if any character is rejected. An empty
    /// piece is a no-op (returns `s` unchanged).
    public static func advance(_ s: State, piece: String) -> State? {
        var cur = s
        for ch in piece {
            guard let next = step(cur, ch) else { return nil }
            cur = next
        }
        return cur
    }
}
