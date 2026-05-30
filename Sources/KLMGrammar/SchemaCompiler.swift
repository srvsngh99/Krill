import Foundation

/// Compiles a JSON Schema (as a JSON string) into a `SchemaGrammar`.
///
/// Best-effort and total: an unparseable or unsupported schema never throws.
/// Unsupported constructs compile to an unconstrained "any value" node so the
/// output stays valid JSON; a one-time stderr note records which keyword was
/// relaxed, so the relaxation is observable rather than silent.
public extension SchemaGrammar {

    /// Compile `schemaJSON`. Returns `nil` only when the input is not a JSON
    /// object/bool at the root (in which case the caller falls back to the
    /// plain JSON-validity mask).
    static func compile(_ schemaJSON: String) -> SchemaGrammar? {
        guard let data = schemaJSON.data(using: .utf8),
              let root = try? JSONSerialization.jsonObject(
                  with: data, options: [.fragmentsAllowed])
        else { return nil }

        var builder = Builder()
        // Reserve canonical "any" nodes first so their ids are stable.
        let anyId = builder.add(.any)
        let anyObjectId = builder.add(
            .object(props: [:], required: [], additional: .any))
        let anyArrayId = builder.add(.array(item: anyId))
        builder.anyId = anyId

        let rootId = builder.compile(root)
        builder.flushWarnings()

        return SchemaGrammar(
            nodes: builder.nodes,
            rootId: rootId,
            anyId: anyId,
            anyObjectId: anyObjectId,
            anyArrayId: anyArrayId)
    }

    /// Mutable compile-time scratch.
    private struct Builder {
        var nodes: [SchemaGrammar.Node] = []
        var anyId: Int = 0
        var relaxedKeywords: Set<String> = []

        mutating func add(_ node: SchemaGrammar.Node) -> Int {
            nodes.append(node)
            return nodes.count - 1
        }

        mutating func note(_ keyword: String) { relaxedKeywords.insert(keyword) }

        func flushWarnings() {
            guard !relaxedKeywords.isEmpty else { return }
            let list = relaxedKeywords.sorted().joined(separator: ", ")
            FileHandle.standardError.write(Data((
                "[KrillLM] JSON schema: unsupported keyword(s) relaxed to "
                + "\"any value\" (output stays valid JSON but is not "
                + "constrained on these): \(list).\n").utf8))
        }

        /// Compile any schema value to a node id.
        mutating func compile(_ value: Any) -> Int {
            // Boolean schema: `true` = any, `false` = nothing matchable. We map
            // both to "any" (false is rare and an empty language would force
            // fail-open anyway).
            if let b = value as? Bool {
                if !b { note("false-schema") }
                return anyId
            }
            guard let schema = value as? [String: Any] else { return anyId }

            // enum / const: an explicit value set wins over `type`.
            if let constVal = schema["const"] {
                if let s = serialize(constVal) { return add(.enumConst([s])) }
                return anyId
            }
            if let enumVals = schema["enum"] as? [Any] {
                let serialized = enumVals.compactMap { serialize($0) }
                if serialized.count == enumVals.count && !serialized.isEmpty {
                    return add(.enumConst(serialized))
                }
                return anyId
            }

            // Combinators / refs we cannot make deterministic: relax to any.
            for kw in ["anyOf", "oneOf", "allOf", "not", "$ref", "if", "then", "else"]
            where schema[kw] != nil {
                note(kw)
                return anyId
            }

            return compileTyped(schema)
        }

        mutating func compileTyped(_ schema: [String: Any]) -> Int {
            // `type` may be a string or (unsupported) an array of strings.
            let typeStr: String?
            if let t = schema["type"] as? String {
                typeStr = t
            } else if schema["type"] is [Any] {
                note("type-union")
                return anyId
            } else {
                typeStr = inferType(schema)
            }

            switch typeStr {
            case "object": return compileObject(schema)
            case "array": return compileArray(schema)
            case "string":
                for kw in ["pattern", "format"] where schema[kw] != nil { note(kw) }
                return add(.string)
            case "integer": return add(.number(integer: true))
            case "number": return add(.number(integer: false))
            case "boolean": return add(.boolean)
            case "null": return add(.null)
            default: return anyId
            }
        }

        /// Infer object/array when `type` is omitted but the shape keywords
        /// are present (common in hand-written schemas).
        func inferType(_ schema: [String: Any]) -> String? {
            if schema["properties"] != nil || schema["required"] != nil
                || schema["additionalProperties"] != nil { return "object" }
            if schema["items"] != nil { return "array" }
            return nil
        }

        mutating func compileObject(_ schema: [String: Any]) -> Int {
            if schema["patternProperties"] != nil { note("patternProperties") }

            var props: [String: Int] = [:]
            if let p = schema["properties"] as? [String: Any] {
                for (name, sub) in p { props[name] = compile(sub) }
            }
            let required = (schema["required"] as? [Any])?.compactMap { $0 as? String } ?? []

            let additional: SchemaGrammar.Additional
            if let b = schema["additionalProperties"] as? Bool {
                additional = b ? .any : .forbidden
            } else if let sub = schema["additionalProperties"] {
                additional = .schema(compile(sub))
            } else {
                additional = .any
            }
            return add(.object(props: props, required: required, additional: additional))
        }

        mutating func compileArray(_ schema: [String: Any]) -> Int {
            for kw in ["minItems", "maxItems", "uniqueItems"] where schema[kw] != nil {
                note(kw)
            }
            // Tuple `items: [..]` (per-position schemas) is not supported; relax
            // the element schema to any.
            if schema["items"] is [Any] {
                note("items-tuple")
                return add(.array(item: anyId))
            }
            let itemId = (schema["items"]).map { compile($0) } ?? anyId
            return add(.array(item: itemId))
        }

        /// Compact, deterministic JSON serialization of an enum/const value.
        /// Object keys are sorted so the model's output order is well defined.
        func serialize(_ value: Any) -> String? {
            if value is NSNull { return "null" }
            // JSONSerialization decodes both booleans and numbers as NSNumber,
            // and `NSNumber(1) as? Bool` spuriously succeeds (1 bridges to
            // true). Disambiguate by the underlying CFType so the integer `1`
            // serializes as "1", not "true".
            if let n = value as? NSNumber {
                if CFGetTypeID(n) == CFBooleanGetTypeID() {
                    return n.boolValue ? "true" : "false"
                }
                guard let data = try? JSONSerialization.data(
                    withJSONObject: n, options: [.fragmentsAllowed]),
                      let s = String(data: data, encoding: .utf8) else { return nil }
                return s
            }
            if let b = value as? Bool { return b ? "true" : "false" }
            if let s = value as? String {
                guard let data = try? JSONSerialization.data(
                    withJSONObject: s, options: [.fragmentsAllowed]),
                      let out = String(data: data, encoding: .utf8) else { return nil }
                return out
            }
            // Composite enum value: serialize with sorted keys, no whitespace.
            guard JSONSerialization.isValidJSONObject(value),
                  let data = try? JSONSerialization.data(
                      withJSONObject: value, options: [.sortedKeys]),
                  let s = String(data: data, encoding: .utf8) else { return nil }
            return s
        }
    }
}
