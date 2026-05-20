import Foundation

/// Model-agnostic tool/function calling (WS-D D1).
///
/// KrillLM does not depend on any single model's native tool-call template.
/// Instead it injects a compact instruction + the tool JSON-schemas as a
/// system turn and asks the model to emit a sentinel-wrapped call:
///
///     <tool_call>{"name": "...", "arguments": { ... }}</tool_call>
///
/// This convention (Hermes/Qwen-style) works acceptably across the Llama /
/// Qwen / Mistral / Gemma / Phi families KrillLM serves. Parsing is tolerant
/// of three shapes: `<tool_call>…</tool_call>`, the legacy Gemma
/// `<|tool_call|>…<tool_call|>`, and a bare leading JSON object that has
/// both `name` and `arguments`.
///
/// Pure and Sendable so it is unit-testable without a model or a channel.
internal enum ToolCalling {

    struct ParsedToolCall: Equatable, Sendable {
        let name: String
        /// Arguments as a JSON *string* (OpenAI wants a string; Ollama wants
        /// the decoded object - callers convert as needed).
        let argumentsJSON: String
    }

    /// Wire format a model family was fine-tuned on. Tool calling is
    /// irreducibly model-specific at the token level: there is no universal
    /// format, so the public API stays family-agnostic while this enum
    /// selects the concrete adapter for rendering and parsing.
    ///
    /// - `.gemma4`: native special-token format
    ///   (`<|tool>`/`<|tool_call>call:name{json}`/`<|tool_response>`,
    ///   token ids 46-51) the Gemma 4 checkpoint was trained on; pinned by
    ///   the model's own `tokenizer_config.json` `response_schema`.
    /// - `.hermes`: the generic `<tool_call>{"name",…,"arguments"}…` prompt.
    ///   This matches what Qwen 2.5/3 emit and works acceptably on
    ///   Llama/Mistral; it is the fallback for families without a native
    ///   adapter, NOT a universal default.
    enum ToolFormat: Sendable {
        case hermes
        case gemma4
        case llama
        case qwen

        /// Resolve the adapter from the loaded model family
        /// (`InferenceEngine.family`).
        ///
        /// `.moe` maps to `.qwen` because the only native MoE
        /// runtime today (WS6) is Qwen 3 MoE, which uses the Qwen
        /// chat / tool-call template verbatim. If a future native
        /// MoE family (Mixtral, OLMoE) needs a different template,
        /// promote the family string carried in `LoadedModel.family`
        /// to a more specific identifier and add a case here.
        static func forFamily(_ family: String?) -> ToolFormat {
            switch family {
            case "gemma4": return .gemma4
            case "llama": return .llama
            case "qwen": return .qwen
            case "moe": return .qwen
            default: return .hermes
            }
        }
    }

    // Gemma 4 native tool special-token ids (added_tokens in the
    // checkpoint's tokenizer.json; emitted as ids, never as text, exactly
    // like the turn markers 105/106/107).
    static let g4ToolOpen = 46     // <|tool>
    static let g4ToolClose = 47    // <tool|>
    static let g4RespOpen = 50     // <|tool_response>
    static let g4RespClose = 51    // <tool_response|>

    // MARK: - Prompt injection

    static func toolSystemPrompt(_ tools: [ServerToolSpec]) -> String {
        var lines = [
            "You can call tools. The available tools are listed as JSON schemas:",
            "",
        ]
        for t in tools {
            lines.append(
                "{\"name\": \"\(t.name)\", \"description\": \"\(escapeForPrompt(t.description))\", \"parameters\": \(t.parametersJSON)}")
        }
        lines.append("")
        lines.append("To call a tool, output ONLY this exact line and nothing else - no explanation, no code fences, do not repeat the schema:")
        lines.append("<tool_call>{\"name\": \"<tool-name>\", \"arguments\": {<the actual argument values>}}</tool_call>")
        lines.append("`arguments` must be the concrete values for this request, not the schema.")
        lines.append("Use multiple <tool_call> lines to call multiple tools.")
        lines.append("If no tool is needed, just answer the user normally with no <tool_call>.")
        return lines.joined(separator: "\n")
    }

    /// Prepare messages so the model sees tools in the format it was trained
    /// on. `format` defaults to `.hermes` so existing callers/tests keep
    /// their exact behavior; the server passes the resolved family adapter.
    static func injectToolSystem(
        into messages: [[String: String]],
        tools: [ServerToolSpec],
        format: ToolFormat = .hermes
    ) -> [[String: String]] {
        guard !tools.isEmpty else { return messages }
        switch format {
        case .hermes:
            return injectHermes(into: messages, tools: tools)
        case .gemma4:
            return injectGemma4(into: messages, tools: tools)
        case .llama:
            return injectLlama(into: messages, tools: tools)
        case .qwen:
            // Qwen 2.5/3's native tool format IS the Hermes
            // `<tool_call>{"name","arguments"}</tool_call>` convention, so
            // the generic injection is already Qwen-native.
            return injectHermes(into: messages, tools: tools)
        }
    }

    /// Llama 3.x native path. The Llama checkpoint ships a full chat
    /// template whose `tools_in_user_message` branch (default) prepends an
    /// exact instruction + the tool JSON schemas to the first user message;
    /// the model then emits a bare `{"name":…, "parameters":…}` object (no
    /// sentinel). swift-transformers does not pass `tools` into the Jinja
    /// context ("not supported yet"), so we reproduce that branch's output
    /// verbatim as message text - byte-for-byte what the model was trained
    /// on - instead of the foreign Hermes prompt.
    private static func injectLlama(
        into messages: [[String: String]],
        tools: [ServerToolSpec]
    ) -> [[String: String]] {
        // Mirror Ollama's llama3.2 template exactly: each tool is the
        // compact `{"type":"function","function":{…}}` JSON, one per line.
        var schemas: [String] = []
        for t in tools {
            let params = (try? JSONSerialization.jsonObject(
                with: Data(t.parametersJSON.utf8))) ?? [String: Any]()
            let fn: [String: Any] = [
                "type": "function",
                "function": ["name": t.name, "description": t.description,
                             "parameters": params],
            ]
            if let d = try? JSONSerialization.data(withJSONObject: fn),
               let s = String(data: d, encoding: .utf8) {
                schemas.append(s)
            }
        }
        // The tool block Ollama splices into the LAST user turn.
        let toolBlock =
            "Given the following functions, please respond with a JSON for a function call "
            + "with its proper arguments that best answers the given prompt.\n\n"
            + "Respond in the format {\"name\": function name, \"parameters\": "
            + "dictionary of argument name and its value}. Do not use variables.\n\n"
            + schemas.map { "\($0)\n" }.joined(separator: "\n") + "\n"

        // System guidance Ollama prepends when tools are present - without
        // it the small model over-triggers and never answers after a result.
        let sysGuide =
            "When you receive a tool call response, use the output to format "
            + "an answer to the orginal user question.\n\n"
            + "You are a helpful assistant with tool calling capabilities."

        // Un-transform the canonical Hermes turns first (so role detection
        // below sees real `assistant`/`ipython` turns), then locate the
        // last user turn to receive the tool block.
        var work: [[String: String]] = []
        for var m in messages {
            let role = m["role"] ?? "user"
            let content = m["content"] ?? ""
            if role == "user",
               let inner = between(content, "<tool_response>", "</tool_response>") {
                let result = stripNormalizerNamePrefix(inner)
                work.append(["role": "ipython", "content": result])
                continue
            }
            if role == "assistant", content.contains("<tool_call>") {
                let (calls, rest) = extractHermes(from: content)
                if !calls.isEmpty {
                    // Map ALL calls (Ollama's llama3.2 template emits one
                    // {"name","parameters"} object per ToolCall), matching
                    // injectGemma4's multi-call handling.
                    let native = calls.map {
                        "{\"name\": \"\($0.name)\", \"parameters\": \($0.argumentsJSON)}"
                    }.joined(separator: "\n")
                    m["content"] = rest.isEmpty ? native : rest + "\n" + native
                }
            }
            work.append(m)
        }
        let lastUser = work.lastIndex { ($0["role"] ?? "user") == "user" }

        var out: [[String: String]] = []
        if let first = work.first, first["role"] == "system" {
            out.append(["role": "system",
                        "content": (first["content"] ?? "") + "\n\n" + sysGuide])
        } else {
            out.append(["role": "system", "content": sysGuide])
        }
        for (i, m) in work.enumerated() {
            if i == 0, m["role"] == "system" { continue }
            if i == lastUser {
                out.append(["role": "user",
                            "content": toolBlock + (m["content"] ?? "")])
            } else {
                out.append(m)
            }
        }
        if lastUser == nil {
            out.append(["role": "user", "content": toolBlock])
        }
        return out
    }

    /// Generic Hermes/Qwen path - unchanged. Prepend the tool instruction;
    /// if a leading system message exists, append to it so a single system
    /// turn is preserved.
    private static func injectHermes(
        into messages: [[String: String]],
        tools: [ServerToolSpec]
    ) -> [[String: String]] {
        let block = toolSystemPrompt(tools)
        var out = messages
        if let first = out.first, first["role"] == "system" {
            out[0]["content"] = (first["content"] ?? "") + "\n\n" + block
        } else {
            out.insert(["role": "system", "content": block], at: 0)
        }
        return out
    }

    /// Gemma 4 native path. The model was fine-tuned on its own tool
    /// special tokens, so we inject NO foreign instruction (that prompt is
    /// exactly what corrupts Gemma 4 - see docs/NATIVE_TOOL_CALLING_PLAN.md).
    /// Instead we:
    ///  - emit the tool schemas as a `tools` role message that
    ///    `formatGemma4TokenIds` frames with `<|tool>`…`<tool|>` (ids 46/47),
    ///  - un-transform the canonical Hermes turns that `normalizeToolTurns`
    ///    produced for multi-step history back into native shapes so the
    ///    `<|tool_response>` (ids 50/51) framing and native call text apply.
    private static func injectGemma4(
        into messages: [[String: String]],
        tools: [ServerToolSpec]
    ) -> [[String: String]] {
        let schemas = tools.map {
            "{\"name\": \"\($0.name)\", \"description\": \"\(escapeForPrompt($0.description))\", \"parameters\": \($0.parametersJSON)}"
        }.joined(separator: ", ")
        var out: [[String: String]] = [["role": "tools", "content": "[\(schemas)]"]]

        for var m in messages {
            let role = m["role"] ?? "user"
            let content = m["content"] ?? ""
            // `normalizeToolTurns` rewrote tool results to a user turn whose
            // content is exactly `<tool_response>[name=… ]RESULT</tool_response>`.
            if role == "user",
               let inner = between(content, "<tool_response>", "</tool_response>") {
                let result = stripNormalizerNamePrefix(inner)
                out.append(["role": "tool", "content": result])
                continue
            }
            // …and assistant tool calls to `<tool_call>{name,arguments}</tool_call>`
            // lines. Re-render them in Gemma's native `call:NAME{args}` form
            // so the model sees its own prior calls correctly.
            if role == "assistant", content.contains("<tool_call>") {
                let (calls, rest) = extractHermes(from: content)
                if !calls.isEmpty {
                    let native = calls.map { "<|tool_call>call:\($0.name)\($0.argumentsJSON)<tool_call|>" }
                        .joined()
                    m["content"] = rest.isEmpty ? native : rest + "\n" + native
                }
            }
            out.append(m)
        }
        return out
    }

    /// Drop the `name=<tool> ` prefix that `normalizeToolTurns` prepends to
    /// a tool-result body. Anchored with `hasPrefix` so a result whose own
    /// content contains `name=` mid-string is never truncated.
    private static func stripNormalizerNamePrefix(_ s: String) -> String {
        guard s.hasPrefix("name="), let sp = s.firstIndex(of: " ")
        else { return s }
        return String(s[s.index(after: sp)...])
    }

    /// Inner text between the first `open` and the next `close`, or nil.
    private static func between(_ s: String, _ open: String, _ close: String)
        -> String?
    {
        guard let o = s.range(of: open),
              let c = s.range(of: close, range: o.upperBound ..< s.endIndex)
        else { return nil }
        return String(s[o.upperBound ..< c.lowerBound])
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Extraction

    private static let pairs: [(open: String, close: String)] = [
        ("<tool_call>", "</tool_call>"),
        ("<|tool_call|>", "<tool_call|>"),
    ]

    static func extractToolCalls(from text: String, format: ToolFormat = .hermes)
        -> (calls: [ParsedToolCall], cleanedText: String)
    {
        switch format {
        case .hermes: return extractHermes(from: text)
        case .gemma4: return extractGemma4(from: text)
        case .llama: return extractLlama(from: text)
        case .qwen: return extractQwen(from: text)
        }
    }

    /// Extract tool calls only when the request actually offered tools.
    /// The family-aware extractors (esp. `.llama`/`.qwen`) treat a bare
    /// `{"name","arguments"|"parameters"}` object as a native call, so
    /// running them on a no-tools turn would misclassify ordinary JSON
    /// output as a tool call (e.g. an Anthropic `tool_use` block). With no
    /// tools offered there can be no tool calls by definition.
    static func extractIfToolsOffered(
        from text: String, hasTools: Bool, format: ToolFormat
    ) -> (calls: [ParsedToolCall], cleanedText: String) {
        guard hasTools else { return ([], text) }
        return extractToolCalls(from: text, format: format)
    }

    /// Qwen 2.5/3 parser: the Hermes extractor, plus a leading-junk-tolerant
    /// bare-JSON fallback (the model sometimes emits a stray prefix like
    /// `>{…}` or partial `<tool_call` markers before the object). Scoped to
    /// `.qwen` so the validated Hermes/Gemma paths are unaffected.
    static func extractQwen(from text: String)
        -> (calls: [ParsedToolCall], cleanedText: String)
    {
        let h = extractHermes(from: text)
        if !h.calls.isEmpty { return h }
        // Scan every balanced object; collect ALL with `name` +
        // `arguments`, ignoring leading prose/markers. Collecting all (not
        // the first) keeps multi-tool responses intact, matching Hermes.
        var calls: [ParsedToolCall] = []
        var scan = Substring(text)
        while let (json, end) = firstJSONObject(in: scan), !json.isEmpty {
            if json.contains("\"arguments\""), let c = parseCallJSON(json),
               !c.name.isEmpty {
                calls.append(c)
            }
            scan = scan[end...]
        }
        return calls.isEmpty ? h : (calls, "")
    }

    /// Llama 3.x native parser. The model emits a bare
    /// `{"name": …, "parameters": …}` object (Llama uses `parameters`, not
    /// `arguments`; `<|python_tag|>` may prefix it for the code env). Falls
    /// back to the Hermes sentinel because the small 1B checkpoint
    /// sometimes emits `<tool_call>…</tool_call>` instead.
    static func extractLlama(from text: String)
        -> (calls: [ParsedToolCall], cleanedText: String)
    {
        var cleaned = text
        if let r = cleaned.range(of: "<|python_tag|>") {
            cleaned.removeSubrange(cleaned.startIndex ..< r.upperBound)
        }
        // Strip ```json / ``` fences the 1B model loves to add.
        cleaned = cleaned.replacingOccurrences(of: "```json", with: "")
            .replacingOccurrences(of: "```", with: "")

        // Collect EVERY balanced object that is a call: a string `name` +
        // `parameters`/`arguments`, and NOT an echoed schema (those carry a
        // top-level `type:"function"` or a nested `function`). Collecting
        // all (not the first) keeps multi-tool responses intact, matching
        // the Hermes path.
        var calls: [ParsedToolCall] = []
        var scan = Substring(cleaned)
        while let (json, end) = firstJSONObject(in: scan), !json.isEmpty {
            if let data = json.data(using: .utf8),
               let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let name = obj["name"] as? String, !name.isEmpty,
               obj["function"] == nil,
               !((obj["type"] as? String) == "function"),
               let argsAny = obj["parameters"] ?? obj["arguments"] {
                let argsString: String
                if let s = argsAny as? String {
                    argsString = s
                } else if let d = try? JSONSerialization.data(withJSONObject: argsAny) {
                    argsString = String(data: d, encoding: .utf8) ?? "{}"
                } else {
                    argsString = "{}"
                }
                calls.append(ParsedToolCall(name: name, argumentsJSON: argsString))
            }
            scan = scan[end...]
        }
        if !calls.isEmpty { return (calls, "") }
        // 1B fallback: it sometimes emits the Hermes sentinel anyway.
        return extractHermes(from: text)
    }

    /// Gemma 4 native parser. Implements the grammar pinned by the
    /// checkpoint's `response_schema`: iterate `<|tool_call> … <tool_call|>`,
    /// each body is `call:NAME{json-args}`; a leading
    /// `<|channel>thought\n…<channel|>` (or `<|think|>…`) segment is reasoning
    /// and is stripped from user-visible content. Tolerant of a missing
    /// `call:` prefix or close sentinel (small models drop both).
    static func extractGemma4(from text: String)
        -> (calls: [ParsedToolCall], cleanedText: String)
    {
        var cleaned = text

        // Strip reasoning channels from visible output.
        for (o, c) in [("<|channel>", "<channel|>"), ("<|think|>", "<think|>")] {
            while let s = cleaned.range(of: o) {
                if let e = cleaned.range(of: c, range: s.upperBound ..< cleaned.endIndex) {
                    cleaned.removeSubrange(s.lowerBound ..< e.upperBound)
                } else {
                    cleaned.removeSubrange(s.lowerBound ..< cleaned.endIndex)
                }
            }
        }

        var calls: [ParsedToolCall] = []
        while let s = cleaned.range(of: "<|tool_call>") {
            let after = cleaned[s.upperBound...]
            // Body runs to the close sentinel if present, else to end.
            let bodyEnd = after.range(of: "<tool_call|>")
            let body = String(after[after.startIndex ..< (bodyEnd?.lowerBound ?? after.endIndex)])
            if let call = parseGemma4Call(body) { calls.append(call) }
            let removeEnd = bodyEnd?.upperBound ?? cleaned.endIndex
            cleaned.removeSubrange(s.lowerBound ..< removeEnd)
        }

        // Fallback: a bare `call:NAME{…}` with no sentinels.
        if calls.isEmpty,
           let r = cleaned.range(of: "call:"),
           let call = parseGemma4Call(String(cleaned[r.lowerBound...])) {
            calls.append(call)
            cleaned = String(cleaned[..<r.lowerBound])
        }

        return (calls, cleaned.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    /// Parse one Gemma 4 call body: optional `call:` prefix, a `\w+` name,
    /// then the first balanced JSON object (the arguments).
    private static func parseGemma4Call(_ raw: String) -> ParsedToolCall? {
        var s = Substring(raw.trimmingCharacters(in: .whitespacesAndNewlines))
        if s.hasPrefix("call:") { s = s.dropFirst(5) }
        s = Substring(s.trimmingCharacters(in: .whitespacesAndNewlines))
        // Name: a quoted string, or leading word chars. The 2B model is
        // sloppy and emits all of: `add{…}`, `"add": {…}`,
        // `"multiply", "parameters": {…}` - so read the name, then take the
        // FIRST balanced object anywhere after it as the arguments
        // (Ollama's `gemma4-tool-call` parser is similarly tolerant).
        let name: String
        if let q = s.first, q == "\"" || q == "'" {
            let body = s.dropFirst()
            guard let end = body.firstIndex(of: q) else { return nil }
            name = String(body[body.startIndex ..< end])
        } else {
            name = String(s.prefix { $0.isLetter || $0.isNumber || $0 == "_" })
        }
        guard !name.isEmpty else { return nil }
        let rest = s
        var argsString = "{}"
        if let (blob, _) = firstJSONObject(in: rest) {
            // Gemma 4's `x-parser: gemma4-tool-call` arg blob is NOT strict
            // JSON: keys are often bare (`{a:12, "b":30}`) and literals may
            // be Python-style. Normalize to JSON so the OpenAI/Ollama
            // shapers (which JSON-decode it) get real values, not `{}`.
            let json = normalizeGemma4Args(blob)
            if let d = json.data(using: .utf8),
               (try? JSONSerialization.jsonObject(with: d)) != nil {
                argsString = json
            }
        }
        return ParsedToolCall(name: name, argumentsJSON: argsString)
    }

    /// Convert Gemma 4's relaxed arg syntax to strict JSON: quote bare
    /// object keys, convert single-quoted strings to double-quoted, and map
    /// Python `True`/`False`/`None` to JSON. String-aware so quotes/braces
    /// inside string values are never rewritten.
    static func normalizeGemma4Args(_ blob: String) -> String {
        var out = ""
        let chars = Array(blob)
        var i = 0
        var inStr = false
        var delim: Character = "\""
        func skipWS(_ j: Int) -> Int {
            var k = j
            while k < chars.count, chars[k] == " " || chars[k] == "\n"
                || chars[k] == "\t" { k += 1 }
            return k
        }
        while i < chars.count {
            let c = chars[i]
            if inStr {
                if c == "\\", i + 1 < chars.count {
                    out.append(c); out.append(chars[i + 1]); i += 2; continue
                }
                if c == delim { out.append("\""); inStr = false; i += 1; continue }
                if c == "\"" { out.append("\\\"") } else { out.append(c) }
                i += 1
                continue
            }
            if c == "\"" || c == "'" {
                delim = c; inStr = true; out.append("\""); i += 1; continue
            }
            if c.isLetter || c == "_" {
                var j = i
                while j < chars.count,
                      chars[j].isLetter || chars[j].isNumber || chars[j] == "_" {
                    j += 1
                }
                let word = String(chars[i ..< j])
                let next = skipWS(j)
                if next < chars.count, chars[next] == ":" {
                    out += "\"\(word)\""              // bare object key
                } else {
                    switch word {
                    case "True", "true": out += "true"
                    case "False", "false": out += "false"
                    case "None", "null": out += "null"
                    default: out += "\"\(word)\""    // bare string value
                    }
                }
                i = j
                continue
            }
            out.append(c)
            i += 1
        }
        return out
    }

    static func extractHermes(from text: String)
        -> (calls: [ParsedToolCall], cleanedText: String)
    {
        var calls: [ParsedToolCall] = []
        var cleaned = text

        // For each opening sentinel, extract the first balanced JSON object
        // after it - tolerating a missing close tag, surrounding backticks,
        // and trailing punctuation (small models routinely do all three).
        for (open, close) in pairs {
            while let s = cleaned.range(of: open) {
                let after = cleaned[s.upperBound...]
                guard let (json, jsonEnd) = Self.firstJSONObject(in: after) else {
                    // No JSON after the marker - drop the bare marker so it
                    // doesn't leak into user-visible content.
                    cleaned.removeSubrange(s.lowerBound ..< s.upperBound)
                    continue
                }
                if let c = parseCallJSON(json) { calls.append(c) }
                // Remove from the open marker through the JSON end, plus an
                // optional matching close tag / trailing junk on that span.
                var removeEnd = jsonEnd
                let tail = cleaned[jsonEnd...]
                if let cr = tail.range(of: close),
                   cr.lowerBound == tail.startIndex
                       || tail[tail.startIndex ..< cr.lowerBound]
                           .allSatisfy({ " `;\n\t".contains($0) })
                {
                    removeEnd = cr.upperBound
                }
                cleaned.removeSubrange(s.lowerBound ..< removeEnd)
            }
        }

        if calls.isEmpty {
            // Fenced ```json block whose object has BOTH `name` and
            // `arguments`. Requiring `arguments` avoids mistaking an echoed
            // tool *schema* (which has `parameters`/`properties`, not
            // `arguments`) for an actual call.
            if let f = cleaned.range(of: "```"),
               let e = cleaned.range(of: "```", range: f.upperBound ..< cleaned.endIndex)
            {
                var body = String(cleaned[f.upperBound ..< e.lowerBound])
                if body.hasPrefix("json") { body.removeFirst(4) }
                body = body.trimmingCharacters(in: .whitespacesAndNewlines)
                if body.contains("\"arguments\""), let c = parseCallJSON(body) {
                    calls.append(c)
                    cleaned.removeSubrange(f.lowerBound ..< e.upperBound)
                }
            }
        }

        if calls.isEmpty {
            // Bare JSON object with name + arguments (no sentinel).
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.hasPrefix("{"), trimmed.contains("\"arguments\""),
               let c = parseCallJSON(trimmed), !c.name.isEmpty {
                calls.append(c)
                cleaned = ""
            }
        }

        return (calls, cleaned.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    /// Find the first balanced `{...}` JSON object in `s` (string-literal
    /// aware so braces inside quotes don't miscount). Returns the object
    /// substring and the index just past its closing brace.
    private static func firstJSONObject(in s: Substring)
        -> (json: String, end: Substring.Index)?
    {
        guard let start = s.firstIndex(of: "{") else { return nil }
        var depth = 0
        var inStr = false
        var esc = false
        var i = start
        while i < s.endIndex {
            let ch = s[i]
            if inStr {
                if esc { esc = false }
                else if ch == "\\" { esc = true }
                else if ch == "\"" { inStr = false }
            } else {
                if ch == "\"" { inStr = true }
                else if ch == "{" { depth += 1 }
                else if ch == "}" {
                    depth -= 1
                    if depth == 0 {
                        let end = s.index(after: i)
                        return (String(s[start ..< end]), end)
                    }
                }
            }
            i = s.index(after: i)
        }
        return nil
    }

    private static func parseCallJSON(_ json: String) -> ParsedToolCall? {
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let name = obj["name"] as? String, !name.isEmpty
        else { return nil }

        let argsString: String
        if let a = obj["arguments"] {
            if let s = a as? String {
                argsString = s
            } else if let d = try? JSONSerialization.data(withJSONObject: a) {
                argsString = String(data: d, encoding: .utf8) ?? "{}"
            } else {
                argsString = "{}"
            }
        } else {
            argsString = "{}"
        }
        return ParsedToolCall(name: name, argumentsJSON: argsString)
    }

    // MARK: - Response shaping

    /// OpenAI `message.tool_calls` array (arguments as a JSON string).
    static func openAIToolCalls(_ calls: [ParsedToolCall]) -> [[String: Any]] {
        calls.enumerated().map { i, c in
            [
                "id": "call_\(randomId())\(i)",
                "type": "function",
                "function": ["name": c.name, "arguments": c.argumentsJSON],
            ]
        }
    }

    /// Ollama `message.tool_calls` array (arguments as a decoded object).
    static func ollamaToolCalls(_ calls: [ParsedToolCall]) -> [[String: Any]] {
        calls.map { c in
            let argsObj = (try? JSONSerialization.jsonObject(
                with: Data(c.argumentsJSON.utf8))) ?? [String: Any]()
            return ["function": ["name": c.name, "arguments": argsObj]]
        }
    }

    private static func randomId() -> String {
        String(UUID().uuidString.prefix(8)).lowercased()
    }

    private static func escapeForPrompt(_ s: String) -> String {
        s.replacingOccurrences(of: "\"", with: "'")
            .replacingOccurrences(of: "\n", with: " ")
    }
}
