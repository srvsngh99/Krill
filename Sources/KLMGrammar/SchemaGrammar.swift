import Foundation

/// Stage B: a JSON-Schema-constrained grammar automaton (parity plan §8).
///
/// `SchemaGrammar.compile(_:)` turns a bounded subset of JSON Schema into a
/// pushdown automaton conforming to `GrammarAutomaton`, so the shared
/// `GrammarTokenMask` masking layer can constrain decoding to JSON that not
/// only is well-formed but also matches the schema's structure: declared
/// property keys, required fields, `additionalProperties`, array `items`,
/// scalar `type` (string / number / integer / boolean / null), and
/// `enum` / `const`.
///
/// Supported subset:
/// - `type`: object, array, string, number, integer, boolean, null
/// - object: `properties`, `required`, `additionalProperties` (bool or schema)
/// - array: `items` (single schema)
/// - `enum` (scalars + composite values), `const`
///
/// Deliberately unsupported (compiled to an unconstrained "any value" node so
/// output stays valid JSON, with a one-time stderr note): `anyOf` / `oneOf` /
/// `allOf` / `not` (they make the automaton nondeterministic), `$ref`,
/// `patternProperties`, `pattern`, string `format`, numeric bounds,
/// `minItems` / `maxItems`, and union `type` arrays. These are tracked as
/// Stage B follow-ups.
public struct SchemaGrammar: GrammarAutomaton {

    // MARK: Compiled nodes

    enum Additional: Hashable, Sendable {
        case any              // additionalProperties: true / absent
        case schema(Int)      // additionalProperties: { ... }
        case forbidden        // additionalProperties: false
    }

    enum Node: Hashable, Sendable {
        case any                         // any JSON value
        case string
        case number(integer: Bool)
        case boolean
        case null
        case object(props: [String: Int], required: [String], additional: Additional)
        case array(item: Int)            // item schema node id
        case enumConst([String])         // allowed values, compact-serialized
    }

    let nodes: [Node]
    let rootId: Int
    let anyId: Int
    let anyObjectId: Int
    let anyArrayId: Int

    // MARK: State

    /// One level of the parse stack. `node` is the schema node this frame
    /// validates; `step` is the lexical position within it; `seen` tracks the
    /// object keys consumed so far (for `required` / no-repeat). `seen` is
    /// empty for non-object frames.
    struct Frame: Hashable, Sendable {
        var node: Int
        var step: Step
        var seen: Set<String>
    }

    enum Step: Hashable, Sendable {
        case start                       // about to read this value (WS ok)
        // string
        case strBody
        case strEsc
        case strU(Int)
        // number
        case numAfterSign
        case numIntZero
        case numInt
        case numDotDigit
        case numFrac
        case numExpSign
        case numExpDigit
        case numExp
        // literal / enum / const matching (prefix consumed so far)
        case litMatch(String)
        // object
        case objOpen                     // consumed '{' (WS, key, or '}')
        case objKeyStart                 // after ',' (WS or key; no '}')
        case objInKey(String)            // inside key string (partial)
        case objKeyEsc(String)
        case objKeyU(String, Int)
        case objAfterKey(Int)            // key closed, expect ':' (child node id)
        case objAfterColon(Int)          // consumed ':', expect value (child id)
        case objAfterValue               // value done, expect ',' or '}'
        // array
        case arrOpen                     // consumed '[' (WS, value, or ']')
        case arrItemStart                // after ',' (WS or value; no ']')
        case arrAfterValue               // item done, expect ',' or ']'
    }

    public struct State: Hashable, Sendable {
        var stack: [Frame]
        var done: Bool
    }

    public var initialState: State {
        State(stack: [Frame(node: rootId, step: .start, seen: [])], done: false)
    }

    // MARK: - Character classes

    private static func isWS(_ c: Character) -> Bool {
        c == " " || c == "\t" || c == "\n" || c == "\r"
    }
    private static func isDigit(_ c: Character) -> Bool { c >= "0" && c <= "9" }
    private static func isHex(_ c: Character) -> Bool {
        isDigit(c) || (c >= "a" && c <= "f") || (c >= "A" && c <= "F")
    }

    // MARK: - Completeness

    /// A value is complete when the root value has fully closed, or the root
    /// frame is a lazily-terminated scalar (number / enum) sitting in an
    /// accepting position (those have no closing delimiter of their own).
    public func isComplete(_ s: State) -> Bool {
        if s.done { return true }
        guard s.stack.count == 1 else { return false }
        return frameAccepting(s.stack[0])
    }

    /// Whether a lazily-terminated scalar frame is at an accepting boundary.
    private func frameAccepting(_ f: Frame) -> Bool {
        switch f.step {
        case .numIntZero, .numInt, .numFrac, .numExp:
            return true
        case .litMatch(let prefix):
            return candidates(forNode: f.node).contains(prefix)
        default:
            return false
        }
    }

    // MARK: - Candidate literals (boolean / null / enum / const)

    private func candidates(forNode node: Int) -> [String] {
        if case .enumConst(let list) = nodes[node] { return list }
        return ["true", "false", "null"]
    }

    private func integerNode(_ node: Int) -> Bool {
        if case .number(let isInt) = nodes[node] { return isInt }
        return false
    }

    // MARK: - Step

    private enum Outcome {
        case reject
        case replace(Frame)
        /// Begin a child value: replace the parent frame, push the child, and
        /// reprocess the current char against the child.
        case pushReprocess(parent: Frame, child: Frame)
        /// The current value is complete and consumed the char (e.g. a closing
        /// quote / `}` / `]`). Pop; the parent (already in *AfterValue) takes
        /// over on the next char.
        case completeConsumed
        /// The current value is complete but did NOT consume the char (a
        /// delimiter after a lazy scalar). Pop and reprocess the char in the
        /// parent.
        case completeReprocess
    }

    public func step(_ s: State, _ c: Character) -> State? {
        if s.done { return Self.isWS(c) ? s : nil }
        var st = s
        // Bounded by stack depth: each completeReprocess pops a frame, and
        // pushReprocess is followed by a child .start that always consumes or
        // rejects c.
        while true {
            guard let top = st.stack.last else {
                // Stack emptied (root value closed). Only trailing WS allowed.
                st.done = true
                return Self.isWS(c) ? st : nil
            }
            switch advance(top, c) {
            case .reject:
                return nil
            case .replace(let f):
                st.stack[st.stack.count - 1] = f
                return st
            case .pushReprocess(let parent, let child):
                st.stack[st.stack.count - 1] = parent
                st.stack.append(child)
                continue  // reprocess c in the child
            case .completeConsumed:
                st.stack.removeLast()
                if st.stack.isEmpty { st.done = true }
                return st
            case .completeReprocess:
                st.stack.removeLast()
                if st.stack.isEmpty {
                    st.done = true
                    return Self.isWS(c) ? st : nil
                }
                continue  // reprocess c in the parent (now in *AfterValue)
            }
        }
    }

    // MARK: - Per-frame transition

    private func advance(_ top: Frame, _ c: Character) -> Outcome {
        let node = nodes[top.node]
        switch top.step {

        case .start:
            if Self.isWS(c) { return .replace(top) }
            return beginValue(top, c)

        // MARK: strings
        case .strBody:
            if c == "\"" { return .completeConsumed }
            if c == "\\" { return .replace(with(top, .strEsc)) }
            if isControl(c) { return .reject }
            return .replace(top)
        case .strEsc:
            if "\"\\/bfnrt".contains(c) { return .replace(with(top, .strBody)) }
            if c == "u" { return .replace(with(top, .strU(0))) }
            return .reject
        case .strU(let n):
            guard Self.isHex(c) else { return .reject }
            return .replace(with(top, n == 3 ? .strBody : .strU(n + 1)))

        // MARK: numbers
        case .numAfterSign:
            if c == "0" { return .replace(with(top, .numIntZero)) }
            if Self.isDigit(c) { return .replace(with(top, .numInt)) }
            return .reject
        case .numIntZero:
            if !integerNode(top.node) && c == "." { return .replace(with(top, .numDotDigit)) }
            if !integerNode(top.node) && (c == "e" || c == "E") { return .replace(with(top, .numExpSign)) }
            if Self.isDigit(c) { return .reject }      // no leading zeros
            return .completeReprocess
        case .numInt:
            if Self.isDigit(c) { return .replace(top) }
            if !integerNode(top.node) && c == "." { return .replace(with(top, .numDotDigit)) }
            if !integerNode(top.node) && (c == "e" || c == "E") { return .replace(with(top, .numExpSign)) }
            return .completeReprocess
        case .numDotDigit:
            if Self.isDigit(c) { return .replace(with(top, .numFrac)) }
            return .reject
        case .numFrac:
            if Self.isDigit(c) { return .replace(top) }
            if c == "e" || c == "E" { return .replace(with(top, .numExpSign)) }
            return .completeReprocess
        case .numExpSign:
            if c == "+" || c == "-" { return .replace(with(top, .numExpDigit)) }
            if Self.isDigit(c) { return .replace(with(top, .numExp)) }
            return .reject
        case .numExpDigit:
            if Self.isDigit(c) { return .replace(with(top, .numExp)) }
            return .reject
        case .numExp:
            if Self.isDigit(c) { return .replace(top) }
            return .completeReprocess

        // MARK: literal / enum / const
        case .litMatch(let prefix):
            let cands = candidates(forNode: top.node)
            let extended = prefix + String(c)
            if cands.contains(where: { $0.hasPrefix(extended) }) {
                return .replace(with(top, .litMatch(extended)))
            }
            // Cannot extend: complete iff the prefix is already a full literal.
            if cands.contains(prefix) { return .completeReprocess }
            return .reject

        // MARK: objects
        case .objOpen, .objKeyStart:
            if Self.isWS(c) { return .replace(top) }
            if c == "}" && top.step == .objOpen {
                return objClose(top) ? .completeConsumed : .reject
            }
            if c == "\"" { return .replace(with(top, .objInKey(""))) }
            return .reject
        case .objInKey(let partial):
            if c == "\"" { return closeKey(top, key: partial) }
            if c == "\\" { return .replace(with(top, .objKeyEsc(partial))) }
            if isControl(c) { return .reject }
            let extended = partial + String(c)
            guard keyPrefixAllowed(top, prefix: extended) else { return .reject }
            return .replace(with(top, .objInKey(extended)))
        case .objKeyEsc(let partial):
            if let u = unescape(c) {
                let extended = partial + String(u)
                guard keyPrefixAllowed(top, prefix: extended) else { return .reject }
                return .replace(with(top, .objInKey(extended)))
            }
            if c == "u" { return .replace(with(top, .objKeyU(partial, 0))) }
            return .reject
        case .objKeyU(let partial, let n):
            guard Self.isHex(c) else { return .reject }
            // We do not decode the code point for prefix matching; once a \u
            // escape appears in a key we stop constraining the key to declared
            // names (rare; declared property names are plain text).
            if n == 3 { return .replace(with(top, .objInKey(partial))) }
            return .replace(with(top, .objKeyU(partial, n + 1)))
        case .objAfterKey(let childId):
            if Self.isWS(c) { return .replace(top) }
            if c == ":" { return .replace(with(top, .objAfterColon(childId))) }
            return .reject
        case .objAfterColon(let childId):
            if Self.isWS(c) { return .replace(top) }
            let parent = with(top, .objAfterValue)
            let child = Frame(node: childId, step: .start, seen: [])
            return .pushReprocess(parent: parent, child: child)
        case .objAfterValue:
            if Self.isWS(c) { return .replace(top) }
            if c == "," { return .replace(with(top, .objKeyStart)) }
            if c == "}" { return objClose(top) ? .completeConsumed : .reject }
            return .reject

        // MARK: arrays
        case .arrOpen:
            if Self.isWS(c) { return .replace(top) }
            if c == "]" { return .completeConsumed }
            return beginItem(top, node: node, c: c)
        case .arrItemStart:
            if Self.isWS(c) { return .replace(top) }
            return beginItem(top, node: node, c: c)
        case .arrAfterValue:
            if Self.isWS(c) { return .replace(top) }
            if c == "," { return .replace(with(top, .arrItemStart)) }
            if c == "]" { return .completeConsumed }
            return .reject
        }
    }

    // MARK: - Value entry

    private func beginValue(_ top: Frame, _ c: Character) -> Outcome {
        switch nodes[top.node] {
        case .any:
            return beginAnyValue(top, c)
        case .object:
            return c == "{" ? .replace(with(top, .objOpen)) : .reject
        case .array:
            return c == "[" ? .replace(with(top, .arrOpen)) : .reject
        case .string:
            return c == "\"" ? .replace(with(top, .strBody)) : .reject
        case .number:
            return beginNumber(top, c)
        case .boolean:
            if c == "t" { return .replace(with(top, .litMatch("t"))) }
            if c == "f" { return .replace(with(top, .litMatch("f"))) }
            return .reject
        case .null:
            return c == "n" ? .replace(with(top, .litMatch("n"))) : .reject
        case .enumConst(let list):
            return list.contains(where: { $0.hasPrefix(String(c)) })
                ? .replace(with(top, .litMatch(String(c)))) : .reject
        }
    }

    /// `.any` value: dispatch by first char into the unconstrained variant,
    /// rebinding the frame's node to a canonical any-object / any-array so key
    /// and item resolution have a concrete container node.
    private func beginAnyValue(_ top: Frame, _ c: Character) -> Outcome {
        switch c {
        case "{": return .replace(Frame(node: anyObjectId, step: .objOpen, seen: []))
        case "[": return .replace(Frame(node: anyArrayId, step: .arrOpen, seen: []))
        case "\"": return .replace(with(top, .strBody))
        case "-": return .replace(with(top, .numAfterSign))
        case "0": return .replace(with(top, .numIntZero))
        case "t", "f": return .replace(with(top, .litMatch(String(c))))
        case "n": return .replace(with(top, .litMatch("n")))
        default:
            if Self.isDigit(c) { return .replace(with(top, .numInt)) }  // 1-9
            return .reject
        }
    }

    private func beginNumber(_ top: Frame, _ c: Character) -> Outcome {
        if c == "-" { return .replace(with(top, .numAfterSign)) }
        if c == "0" { return .replace(with(top, .numIntZero)) }
        if Self.isDigit(c) { return .replace(with(top, .numInt)) }   // 1-9
        return .reject
    }

    private func beginItem(_ top: Frame, node: Node, c: Character) -> Outcome {
        guard case .array(let itemId) = node else { return .reject }
        let parent = with(top, .arrAfterValue)
        let child = Frame(node: itemId, step: .start, seen: [])
        return .pushReprocess(parent: parent, child: child)
    }

    // MARK: - Object helpers

    /// `}` allowed iff every required key has been seen.
    private func objClose(_ top: Frame) -> Bool {
        guard case .object(_, let required, _) = nodes[top.node] else { return true }
        for key in required where !top.seen.contains(key) { return false }
        return true
    }

    /// Whether `prefix` can still become a legal key for this object. When
    /// `additionalProperties` is forbidden, the key must be a prefix of some
    /// declared property not yet seen; otherwise any string is allowed.
    private func keyPrefixAllowed(_ top: Frame, prefix: String) -> Bool {
        guard case .object(let props, _, let additional) = nodes[top.node] else { return true }
        if additional != .forbidden { return true }
        for (name, _) in props where !top.seen.contains(name) && name.hasPrefix(prefix) {
            return true
        }
        return false
    }

    /// Close a key string: resolve the child node, record the key as seen, and
    /// move to expect the colon.
    private func closeKey(_ top: Frame, key: String) -> Outcome {
        guard case .object(let props, _, let additional) = nodes[top.node] else { return .reject }
        // Forbid repeating a key already emitted in this object, uniformly
        // (declared or additional). A repeated member makes the object's value
        // for that key ambiguous; rejecting it keeps the constraint consistent
        // regardless of `additionalProperties` rather than only catching
        // declared-key repeats under `additionalProperties:false`.
        if top.seen.contains(key) { return .reject }
        let childId: Int
        if let declared = props[key] {
            childId = declared
        } else {
            switch additional {
            case .any: childId = anyId
            case .schema(let id): childId = id
            case .forbidden: return .reject   // undeclared key with no additional
            }
        }
        var f = top
        f.seen.insert(key)
        f.step = .objAfterKey(childId)
        return .replace(f)
    }

    // MARK: - Small utilities

    private func with(_ frame: Frame, _ step: Step) -> Frame {
        var f = frame
        f.step = step
        return f
    }

    private func isControl(_ c: Character) -> Bool {
        c.unicodeScalars.count == 1 && c.unicodeScalars.first!.value < 0x20
    }

    /// Unescape a simple two-char JSON escape; `nil` for `u` (handled
    /// separately) or invalid.
    private func unescape(_ c: Character) -> Character? {
        switch c {
        case "\"": return "\""
        case "\\": return "\\"
        case "/": return "/"
        case "b": return "\u{08}"
        case "f": return "\u{0C}"
        case "n": return "\n"
        case "r": return "\r"
        case "t": return "\t"
        default: return nil
        }
    }
}
