import Foundation
import KLMRegistry

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
        case mistral
        case phi

        /// Resolve the wire format from the loaded model family
        /// (`InferenceEngine.family`).
        ///
        /// The family-to-template decision is owned by the registry's
        /// `ModelAdapter.chatTemplate` (WS3); this method only maps
        /// the registry's module-neutral `ChatTemplatePolicy` onto
        /// the concrete renderer/parser in this module. A new family
        /// declares its template in `ModelAdapter`, not here - which
        /// is why, e.g., `.moe` resolving to `.qwen` lives there.
        ///
        /// `family` is the loader string from `InferenceEngine.family`,
        /// resolved to a `ModelFamily` by `rawValue`. Every loader
        /// string - `gemma4`, `llama`, `qwen`, `qwen2_5_vl`, `moe`,
        /// `mistral`, `gemma`, `phi`, `glm` - equals its
        /// `ModelFamily.rawValue`, so the resolution always
        /// round-trips.
        ///
        /// A nil or unrecognized family string falls back to
        /// `.hermes`, matching the registry's own default.
        static func forFamily(_ family: String?) -> ToolFormat {
            guard let family,
                  let modelFamily = ModelFamily(rawValue: family) else {
                return .hermes
            }
            switch ModelAdapter(family: modelFamily).chatTemplate {
            case .hermes: return .hermes
            case .gemma4: return .gemma4
            case .llama: return .llama
            case .qwen: return .qwen
            case .mistral: return .mistral
            case .phi: return .phi
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
            return injectQwen(into: messages, tools: tools)
        case .mistral:
            return injectMistral(into: messages, tools: tools)
        case .phi:
            return injectPhi(into: messages, tools: tools)
        }
    }

    // MARK: - Forced (grammar-constrained) tool calls

    /// Build a JSON-schema string that a forced tool call must match, for
    /// `tool_choice = required | {function:name}`. Decoding is constrained to
    /// this schema so the emitted call is valid + name-correct (best-effort:
    /// the mask fails open if no valid token is available), and
    /// (for a single resolved tool) argument-schema-correct.
    ///
    /// Shape: `{"type":"object","properties":{"name":{...},"arguments":{...}},
    ///          "required":["name","arguments"],"additionalProperties":false}`
    /// - single resolved tool  -> name is `const`, arguments = the tool's own
    ///   parameter schema (full constraint).
    /// - multiple tools (`required`) -> name is `enum` of tool names, arguments
    ///   = any object (the schema grammar has no `oneOf`, so per-name argument
    ///   schemas cannot be expressed; name is still pinned to a valid tool).
    /// Returns nil when no tool resolves (caller falls back to unconstrained).
    static func forcedToolCallSchema(
        tools: [ServerToolSpec], choice: ServerToolChoice
    ) -> String? {
        let candidates: [ServerToolSpec]
        switch choice {
        case .function(let name):
            guard let t = tools.first(where: { $0.name == name }) else { return nil }
            candidates = [t]
        case .required:
            guard !tools.isEmpty else { return nil }
            candidates = tools
        case .auto, .none:
            return nil
        }

        let nameSchema: [String: Any]
        let argsSchema: Any
        if candidates.count == 1 {
            nameSchema = ["const": candidates[0].name]
            // Embed the tool's own parameter schema (parsed) as the arguments
            // constraint; fall back to a bare object if it does not parse.
            if let data = candidates[0].parametersJSON.data(using: .utf8),
               let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                argsSchema = obj
            } else {
                argsSchema = ["type": "object"]
            }
        } else {
            nameSchema = ["enum": candidates.map { $0.name }]
            argsSchema = ["type": "object"]
        }

        let schema: [String: Any] = [
            "type": "object",
            "properties": ["name": nameSchema, "arguments": argsSchema],
            "required": ["name", "arguments"],
            "additionalProperties": false,
        ]
        guard let out = try? JSONSerialization.data(withJSONObject: schema),
              let str = String(data: out, encoding: .utf8) else { return nil }
        return str
    }

    /// System turn for a forced tool call: list the tool schemas and instruct a
    /// bare-JSON `{"name","arguments"}` object (no family sentinel - decoding is
    /// schema-constrained, so the bare object is the whole output and parses
    /// directly). Used only when `tool_choice` forces a call.
    static func forcedToolSystemPrompt(
        tools: [ServerToolSpec], choice: ServerToolChoice
    ) -> String {
        var lines = ["You must call a tool. Available tools (JSON schemas):", ""]
        for t in tools {
            lines.append(
                "{\"name\": \"\(escapeForPrompt(t.name))\", \"description\": \"\(escapeForPrompt(t.description))\", \"parameters\": \(t.parametersJSON)}")
        }
        lines.append("")
        if case .function(let name) = choice {
            lines.append("Call the tool named \"\(escapeForPrompt(name))\".")
        }
        lines.append("Respond with ONLY a single JSON object of the form:")
        lines.append("{\"name\": \"<tool-name>\", \"arguments\": {<concrete argument values>}}")
        lines.append("No prose, no code fences, no sentinels - just the JSON object.")
        return lines.joined(separator: "\n")
    }

    /// Inject the forced-tool system turn (replaces the family-specific
    /// injection when `tool_choice` forces a call).
    static func injectForcedToolSystem(
        into messages: [[String: String]],
        tools: [ServerToolSpec],
        choice: ServerToolChoice
    ) -> [[String: String]] {
        guard !tools.isEmpty else { return messages }
        let sys = forcedToolSystemPrompt(tools: tools, choice: choice)
        // Prepend as a system turn (merge with an existing leading system msg).
        var out = messages
        if let first = out.first, first["role"] == "system" {
            out[0] = ["role": "system", "content": (first["content"] ?? "") + "\n\n" + sys]
        } else {
            out.insert(["role": "system", "content": sys], at: 0)
        }
        return out
    }

    /// Parse the bare, schema-constrained JSON object emitted for a forced
    /// tool call into a `ParsedToolCall`. Tolerant of surrounding whitespace
    /// and a stray code fence. Returns nil if it is not a `{name,arguments}`
    /// object.
    static func parseForcedToolCall(_ text: String) -> ParsedToolCall? {
        var s = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if s.hasPrefix("```") {
            s = s.replacingOccurrences(of: "```json", with: "")
                 .replacingOccurrences(of: "```", with: "")
                 .trimmingCharacters(in: .whitespacesAndNewlines)
        }
        // Take from the first '{' to the last '}' (constrained output is a bare
        // object, but be defensive).
        guard let lo = s.firstIndex(of: "{"), let hi = s.lastIndex(of: "}"), lo < hi else {
            return nil
        }
        let objStr = String(s[lo...hi])
        guard let data = objStr.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let name = obj["name"] as? String else { return nil }
        let argsValue = obj["arguments"] ?? [String: Any]()
        let argsJSON: String
        if let argsData = try? JSONSerialization.data(withJSONObject: argsValue),
           let str = String(data: argsData, encoding: .utf8) {
            argsJSON = str
        } else {
            argsJSON = "{}"
        }
        return ParsedToolCall(name: name, argumentsJSON: argsJSON)
    }

    // MARK: - Auto tool-call argument constraining (two-pass)

    /// Gate for the two-pass argument constraint: does `argumentsJSON` satisfy
    /// the tool's `parametersJSON` schema well enough to pass through? True
    /// when the args parse as a JSON object AND every `required` key in the
    /// schema is present. Tools with no required keys always pass (an empty
    /// object is valid for them, so there is nothing to re-generate).
    ///
    /// This is intentionally a presence check, not full type validation - it
    /// targets the dominant small-model failure (empty `{}` / missing required
    /// fields, e.g. codex's shell tool missing `cmd`) without forcing a second
    /// pass on otherwise-usable arguments. Pure + unit-testable.
    static func argsSatisfySchema(argumentsJSON: String, parametersJSON: String) -> Bool {
        guard let argsData = argumentsJSON.data(using: .utf8),
              let args = try? JSONSerialization.jsonObject(with: argsData) as? [String: Any]
        else { return false }  // not even a JSON object -> must re-generate
        guard let schemaData = parametersJSON.data(using: .utf8),
              let schema = try? JSONSerialization.jsonObject(with: schemaData) as? [String: Any],
              let required = schema["required"] as? [String], !required.isEmpty
        else { return true }   // no required keys -> nothing to enforce
        return required.allSatisfy { args[$0] != nil }
    }

    /// Instruction turn for pass 2: ask the model for ONLY the JSON arguments
    /// object for `tool`. Decoding is grammar-constrained to the tool's
    /// parameter schema, so this only needs to orient the model.
    static func argsRegenPrompt(tool: ServerToolSpec) -> String {
        return "Now output ONLY the JSON arguments object to call the tool "
            + "\"\(escapeForPrompt(tool.name))\". Its parameter schema is: "
            + "\(tool.parametersJSON). Respond with a single JSON object giving "
            + "concrete values for the required arguments - no prose, no code fences."
    }

    /// Parse a bare arguments object (pass-2 output) into a compact JSON
    /// string. Tolerant of surrounding whitespace and a stray code fence.
    /// Returns nil if no JSON object is present.
    static func parseArgsObject(_ text: String) -> String? {
        var s = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if s.hasPrefix("```") {
            s = s.replacingOccurrences(of: "```json", with: "")
                 .replacingOccurrences(of: "```", with: "")
                 .trimmingCharacters(in: .whitespacesAndNewlines)
        }
        guard let lo = s.firstIndex(of: "{"), let hi = s.lastIndex(of: "}"), lo < hi else {
            return nil
        }
        let objStr = String(s[lo...hi])
        guard let data = objStr.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let out = try? JSONSerialization.data(withJSONObject: obj),
              let str = String(data: out, encoding: .utf8) else { return nil }
        return str
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

    /// Qwen 2.5 / 3 native path. The call/result sentinels ARE the Hermes
    /// `<tool_call>{"name","arguments"}</tool_call>` convention (so the parser
    /// and history un-transform are shared), but the model was fine-tuned on a
    /// SPECIFIC tool-definition block - the chat template's `# Tools` section
    /// with the schemas inside `<tools></tools>` XML tags - not on the generic
    /// Hermes instruction. swift-transformers drops the `tools` Jinja variable,
    /// so (like the Llama/Mistral paths) we reproduce that block verbatim into
    /// the system message, byte-for-byte what the official template renders and
    /// what Ollama sends. The earlier generic Hermes prompt elicited tool calls
    /// less reliably on the same weights (BENCHMARK_ISSUES #6); the canonical
    /// format aligns the call decision with Ollama.
    private static func injectQwen(
        into messages: [[String: String]],
        tools: [ServerToolSpec]
    ) -> [[String: String]] {
        // Each tool is the OpenAI `{"type":"function","function":{…}}` spec, one
        // per line inside <tools>, exactly as `tool | tojson` renders it.
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
        // The official template's tool block. The closing `<|im_end|>` is added
        // by the chat-template turn wrapper, so it is intentionally omitted here.
        let block =
            "# Tools\n\nYou may call one or more functions to assist with the user query."
            + "\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>"
            + schemas.map { "\n" + $0 }.joined()
            + "\n</tools>\n\nFor each function call, return a json object with function name and"
            + " arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n"
            + "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"

        var out = messages
        if let first = out.first, first["role"] == "system" {
            // Append the block to the existing system content, matching the
            // template's `{system}\n\n# Tools...` layout.
            out[0]["content"] = (first["content"] ?? "") + "\n\n" + block
        } else {
            // No system turn: carry the block as the system message. We do NOT
            // splice in the official template's "You are Qwen..." default, since
            // the `.qwen` tool format also serves non-Qwen MoE checkpoints
            // (OLMoE / DeepSeek-V2 / Mixtral); the `# Tools` block is the part
            // the model keys on for the call decision.
            out.insert(["role": "system", "content": block], at: 0)
        }
        return out
    }

    /// Mistral native path. Mistral 7B Instruct v0.3 / Nemo / Small were
    /// fine-tuned on `[AVAILABLE_TOOLS][ … ][/AVAILABLE_TOOLS]` (the tool
    /// schemas, immediately before the final user `[INST]`), emit calls as
    /// `[TOOL_CALLS][{"name":…,"arguments":{…}}]`, and take results back as
    /// `[TOOL_RESULTS]{"content":…}[/TOOL_RESULTS]`. The checkpoint ships a
    /// chat template that renders the `[INST]` turns but - like every family
    /// here - swift-transformers drops the `tools`, so we splice the tool
    /// block into the last genuine user turn as text (`[AVAILABLE_TOOLS]` and
    /// friends are added tokens that re-encode to their special ids 5-9) and
    /// un-transform the canonical Hermes history back to Mistral's native
    /// call/result shapes.
    private static func injectMistral(
        into messages: [[String: String]],
        tools: [ServerToolSpec]
    ) -> [[String: String]] {
        // [AVAILABLE_TOOLS] is a JSON array of
        // {"type":"function","function":{name,description,parameters}}.
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
        // Match Ollama's `[AVAILABLE_TOOLS] {tools}[/AVAILABLE_TOOLS]` byte
        // layout (space after the open tag) so the model sees the same input
        // it does under Ollama.
        let toolBlock = "[AVAILABLE_TOOLS] [" + schemas.joined(separator: ", ")
            + "][/AVAILABLE_TOOLS]"

        // Un-transform the canonical Hermes turns normalizeToolTurns produced
        // back into Mistral-native text.
        var work: [[String: String]] = []
        for var m in messages {
            let role = m["role"] ?? "user"
            let content = m["content"] ?? ""
            if role == "user",
               let inner = between(content, "<tool_response>", "</tool_response>") {
                let result = stripNormalizerNamePrefix(inner)
                let payload = (try? JSONSerialization.data(
                    withJSONObject: ["content": result]))
                    .flatMap { String(data: $0, encoding: .utf8) }
                    ?? "{\"content\": \"\"}"
                m["content"] = "[TOOL_RESULTS]\(payload)[/TOOL_RESULTS]"
                work.append(m)
                continue
            }
            if role == "assistant", content.contains("<tool_call>") {
                let (calls, rest) = extractHermes(from: content)
                if !calls.isEmpty {
                    let arr = calls.map {
                        "{\"name\": \"\($0.name)\", \"arguments\": \($0.argumentsJSON)}"
                    }.joined(separator: ", ")
                    let native = "[TOOL_CALLS][\(arr)]"
                    m["content"] = rest.isEmpty ? native : rest + native
                }
            }
            work.append(m)
        }

        // Splice the tool block as a prefix to the last GENUINE user turn (not
        // a `[TOOL_RESULTS]` turn), mirroring Mistral's "tools before the last
        // user query" placement.
        let lastUser = work.lastIndex {
            ($0["role"] ?? "user") == "user"
                && !($0["content"] ?? "").hasPrefix("[TOOL_RESULTS]")
        }
        var out: [[String: String]] = []
        for (i, m) in work.enumerated() {
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

    /// Phi-3.5 / Phi-4 native path. The Phi checkpoints were fine-tuned to
    /// receive tool definitions inside the system turn, wrapped in
    /// `<|tool|> … <|/tool|>` as a JSON array of bare
    /// `{name,description,parameters}` objects, and to emit calls as
    /// `<|tool_call|>[{"name":…,"arguments":{…}}]<|/tool_call|>`. The
    /// checkpoint's chat template renders the `<|system|>`/`<|user|>` turns
    /// but (like the other families) drops `tools`, so we bake the
    /// `<|tool|>…<|/tool|>` block into the system message content - the added
    /// tokens re-encode to their special ids - and un-transform the canonical
    /// Hermes history into Phi's native call shape.
    private static func injectPhi(
        into messages: [[String: String]],
        tools: [ServerToolSpec]
    ) -> [[String: String]] {
        // Ollama renders Phi's `<|tool|>` block from the same wrapped
        // `[{"type":"function","function":{name,description,parameters}}]`
        // array it uses everywhere; mirror that exact shape so the model sees
        // identical input to the parity baseline.
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
        let toolDef = "<|tool|>[" + schemas.joined(separator: ", ") + "]<|/tool|>"

        var work: [[String: String]] = []
        for var m in messages {
            let role = m["role"] ?? "user"
            let content = m["content"] ?? ""
            if role == "user",
               let inner = between(content, "<tool_response>", "</tool_response>") {
                // Feed the tool result back as a plain user turn: Phi's chat
                // template renders `<|user|>…<|end|>`, and a `tool` role would
                // collide with the `<|tool|>` definition token.
                m["content"] = stripNormalizerNamePrefix(inner)
                work.append(m)
                continue
            }
            if role == "assistant", content.contains("<tool_call>") {
                let (calls, rest) = extractHermes(from: content)
                if !calls.isEmpty {
                    let arr = calls.map {
                        "{\"name\": \"\($0.name)\", \"arguments\": \($0.argumentsJSON)}"
                    }.joined(separator: ", ")
                    let native = "<|tool_call|>[\(arr)]<|/tool_call|>"
                    m["content"] = rest.isEmpty ? native : rest + native
                }
            }
            work.append(m)
        }

        // Bake the tool definitions into the system turn (create one if none).
        var out: [[String: String]] = []
        if let first = work.first, first["role"] == "system" {
            out.append(["role": "system",
                        "content": (first["content"] ?? "") + toolDef])
            out.append(contentsOf: work.dropFirst())
        } else {
            // Ollama's exact default system text when tools are present.
            out.append(["role": "system",
                        "content": "You are a helpful assistant with some tools." + toolDef])
            out.append(contentsOf: work)
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
        case .mistral: return extractMistral(from: text)
        case .phi: return extractPhi(from: text)
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

    /// Mistral native parser. The model emits `[TOOL_CALLS]` followed by a
    /// JSON array of `{"name":…,"arguments":{…}}` objects. Tolerant of a
    /// quantized model that drops the marker and emits a bare call array, and
    /// falls back to the Hermes sentinel.
    static func extractMistral(from text: String)
        -> (calls: [ParsedToolCall], cleanedText: String)
    {
        if let r = text.range(of: "[TOOL_CALLS]") {
            let calls = parseCallArray(text[r.upperBound...])
            if !calls.isEmpty {
                let cleaned = String(text[..<r.lowerBound])
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                return (calls, cleaned)
            }
        }
        let bare = parseCallArray(Substring(text))
        if !bare.isEmpty { return (bare, "") }
        return extractHermes(from: text)
    }

    /// Phi-3.5 / Phi-4 native parser. The model wraps calls in
    /// `<|tool_call|> … <|/tool_call|>` around a JSON array of
    /// `{"name":…,"arguments":{…}}` objects (some builds prefix `functools`
    /// or drop the wrapper). Falls back to a bare call array, then Hermes.
    static func extractPhi(from text: String)
        -> (calls: [ParsedToolCall], cleanedText: String)
    {
        if let r = text.range(of: "<|tool_call|>") {
            let after = text[r.upperBound...]
            // Body runs to the close marker if present, else to end.
            let bodyEnd = after.range(of: "<|/tool_call|>")
            let body = after[after.startIndex ..< (bodyEnd?.lowerBound ?? after.endIndex)]
            let calls = parseCallArray(body)
            if !calls.isEmpty {
                var cleaned = String(text[..<r.lowerBound])
                if let e = bodyEnd { cleaned += String(after[e.upperBound...]) }
                return (calls, cleaned.trimmingCharacters(in: .whitespacesAndNewlines))
            }
        }
        let bare = parseCallArray(Substring(text))
        if !bare.isEmpty { return (bare, "") }
        return extractHermes(from: text)
    }

    /// Parse the first balanced JSON array of `{"name":…,"arguments":…}` call
    /// objects in `s` (shared by the Mistral `[TOOL_CALLS]` and Phi
    /// `<|tool_call|>` parsers). Tolerant of leading prose; accepts
    /// `parameters` as an alias for `arguments`. Falls back to a single bare
    /// call object when there is no array.
    private static func parseCallArray(_ s: Substring) -> [ParsedToolCall] {
        guard let start = s.firstIndex(of: "[") else {
            if let (json, _) = firstJSONObject(in: s),
               json.contains("\"arguments\""), let c = parseCallJSON(json) {
                return [c]
            }
            return []
        }
        // Find the balanced `]` that closes the array (string-literal aware).
        var depth = 0, inStr = false, esc = false
        var i = start
        var end: Substring.Index?
        while i < s.endIndex {
            let ch = s[i]
            if inStr {
                if esc { esc = false }
                else if ch == "\\" { esc = true }
                else if ch == "\"" { inStr = false }
            } else if ch == "\"" { inStr = true }
            else if ch == "[" { depth += 1 }
            else if ch == "]" {
                depth -= 1
                if depth == 0 { end = s.index(after: i); break }
            }
            i = s.index(after: i)
        }
        guard let e = end,
              let data = String(s[start ..< e]).data(using: .utf8),
              let arr = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]]
        else { return [] }
        var calls: [ParsedToolCall] = []
        for obj in arr {
            guard let name = obj["name"] as? String, !name.isEmpty else { continue }
            let argsAny = obj["arguments"] ?? obj["parameters"]
            let argsString: String
            if let str = argsAny as? String {
                argsString = str
            } else if let a = argsAny,
                      let d = try? JSONSerialization.data(withJSONObject: a) {
                argsString = String(data: d, encoding: .utf8) ?? "{}"
            } else {
                argsString = "{}"
            }
            calls.append(ParsedToolCall(name: name, argumentsJSON: argsString))
        }
        return calls
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
        // Strip reasoning channels from visible output (shared with the
        // non-tool response path via ReasoningParser).
        var cleaned = ReasoningParser.stripGemmaChannels(text).visible

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
