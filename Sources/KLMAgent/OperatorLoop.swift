import Foundation

/// The ReAct-style driver behind `krillm agent`.
///
/// Each step:
///   1. Build a fresh message list (system prompt + transcript so far).
///   2. Ask the `OperatorGenerator` for one assistant turn.
///   3. Parse it for `<tool_call>` envelopes. If any, dispatch them,
///      append the assistant turn (sentinels stripped) + the tool
///      results, and loop. Otherwise treat the turn as the final answer.
///
/// Termination conditions, all surfaced as a terminal event:
///   - `goalCompleted` - a no-tool-call assistant turn (the natural finish).
///   - `budgetExhausted` - `maxSteps`, `maxToolCalls`, or `maxOutputTokens`.
///   - `stuck` - the same `(name, canonicalized-arguments)` tool call fired
///     three times in a row. Arguments are canonicalized (JSON-decoded
///     then re-encoded with sorted keys) before comparison so trivial
///     reorderings or whitespace changes do not bypass the guard.
///   - `cancelled` - the run was cancelled by the host (Ctrl-C).
///   - `failed` - the underlying generator threw.
///
/// The loop is intentionally non-reentrant: `run(goal:)` must be called
/// at most once per instance. A precondition flag enforces this so an
/// accidental double-use crashes loudly rather than producing
/// interleaved transcripts on a shared state.
public final class OperatorLoop: @unchecked Sendable {
    private let generator: any OperatorGenerator
    private let tools: OperatorToolRegistry
    private let budget: OperatorBudget
    private let systemPrompt: String
    private let toolFormat: OperatorToolFormat
    private var started = false

    public init(
        generator: any OperatorGenerator,
        tools: OperatorToolRegistry,
        systemPrompt: String,
        budget: OperatorBudget = OperatorBudget(),
        toolFormat: OperatorToolFormat = .hermes
    ) {
        self.generator = generator
        self.tools = tools
        self.systemPrompt = systemPrompt
        self.budget = budget
        self.toolFormat = toolFormat
    }

    /// Run a goal to termination, emitting events on the returned stream.
    ///
    /// The stream finishes after exactly one terminal event
    /// (`goalCompleted` / `budgetExhausted` / `stuck` / `cancelled`).
    public func run(goal: String) -> AsyncStream<OperatorEvent> {
        precondition(!started,
                     "OperatorLoop.run was called twice on the same "
                     + "instance; create one OperatorLoop per goal.")
        started = true
        return AsyncStream { continuation in
            let task = Task {
                await self.drive(goal: goal, continuation: continuation)
                continuation.finish()
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    private func drive(
        goal: String,
        continuation: AsyncStream<OperatorEvent>.Continuation
    ) async {
        continuation.yield(.goalStarted(goal: goal))

        var transcript: [[String: String]] = [
            ["role": "user", "content": goal],
        ]
        var stepsUsed = 0
        var toolCallsUsed = 0
        var tokensUsed = 0
        var lastCallSignature: String? = nil
        var lastCallStreak = 0

        while true {
            if Task.isCancelled {
                continuation.yield(.cancelled)
                return
            }
            if stepsUsed >= budget.maxSteps {
                continuation.yield(.budgetExhausted(
                    detail: "max_steps=\(budget.maxSteps) reached"))
                return
            }
            stepsUsed += 1

            let messages: [[String: String]] = [
                ["role": "system", "content": systemPrompt],
            ] + transcript

            let turn: OperatorTurn
            do {
                turn = try await generator.generate(messages: messages)
            } catch {
                continuation.yield(.failed(reason: "\(error)"))
                return
            }
            tokensUsed += turn.tokenCount
            if tokensUsed > budget.maxOutputTokens {
                continuation.yield(.budgetExhausted(
                    detail: "max_output_tokens=\(budget.maxOutputTokens) exceeded"))
                return
            }

            let (calls, cleanedText) = extract(from: turn.text)

            if calls.isEmpty {
                let trimmed = cleanedText.trimmingCharacters(
                    in: .whitespacesAndNewlines)
                if !trimmed.isEmpty {
                    continuation.yield(.assistantMessage(trimmed))
                }
                continuation.yield(.goalCompleted(
                    summary: trimmed.isEmpty ? "Done." : firstSentence(trimmed)))
                return
            }

            // Stuck detection: same (name, canonicalized-args) tool call
            // three times in a row. Canonicalize so a router that emits
            // {"x":1,"y":2} and {"y":2,"x":1} on consecutive turns is
            // still caught (raw-string comparison would let either key
            // reordering or whitespace bypass the only structural
            // guardrail against an in-budget infinite loop).
            let signature = calls.map {
                "\($0.name)|\(Self.canonicalize($0.argumentsJSON))"
            }.joined(separator: ";")
            if signature == lastCallSignature {
                lastCallStreak += 1
            } else {
                lastCallSignature = signature
                lastCallStreak = 1
            }
            if lastCallStreak >= 3 {
                continuation.yield(.stuck(
                    reason: "Same tool call repeated 3 times: \(signature)"))
                return
            }

            // Surface any prose the model wrote alongside the tool call
            // BEFORE dispatching. `cleanedText` is the assistant turn
            // with the tool-call sentinels stripped; without this yield
            // a "reasoning" prefix the model emits ("I'll check the
            // hardware first, then ...") never reaches the event
            // stream and the CLI / --json consumers see only the tool
            // call.
            let cleanedTrimmed = cleanedText.trimmingCharacters(
                in: .whitespacesAndNewlines)
            if !cleanedTrimmed.isEmpty {
                continuation.yield(.assistantMessage(cleanedTrimmed))
            }

            // Record the assistant turn (with sentinels intact so the model
            // sees its own prior calls correctly on follow-up turns).
            transcript.append([
                "role": "assistant",
                "content": turn.text,
            ])

            for call in calls {
                if toolCallsUsed >= budget.maxToolCalls {
                    continuation.yield(.budgetExhausted(
                        detail: "max_tool_calls=\(budget.maxToolCalls) reached"))
                    return
                }
                toolCallsUsed += 1
                continuation.yield(.toolCallStarted(
                    name: call.name, argumentsJSON: call.argumentsJSON))

                let result = await dispatch(call: call)
                continuation.yield(.toolCallResult(
                    name: call.name,
                    content: result.content,
                    isError: result.isError))

                // Surface as a `user` tool-response turn the model can read.
                // We keep the same wire shape the server uses
                // (`<tool_response>name=NAME RESULT</tool_response>`) so
                // future deduplication with KLMServer.ToolCalling stays
                // mechanical.
                let envelope =
                    "<tool_response>name=\(call.name) \(result.content)</tool_response>"
                transcript.append([
                    "role": "user",
                    "content": envelope,
                ])
            }
        }
    }

    private func extract(from text: String)
        -> (calls: [OperatorToolCall], cleanedText: String)
    {
        switch toolFormat {
        case .hermes:
            return HermesToolCallExtractor.extract(from: text)
        case .familyAware(let family):
            return FamilyAwareToolCallExtractor.extract(
                from: text, family: family)
        }
    }

    private func dispatch(call: OperatorToolCall) async -> OperatorToolResult {
        guard let tool = tools.tool(named: call.name) else {
            return OperatorToolResult(
                toolName: call.name,
                content: "Error: unknown tool '\(call.name)'. "
                    + "Available tools: \(tools.names.joined(separator: ", ")).",
                isError: true)
        }
        do {
            let content = try await tool.execute(arguments: call.arguments())
            return OperatorToolResult(toolName: call.name, content: content)
        } catch {
            return OperatorToolResult(
                toolName: call.name,
                content: "Error: \(error)",
                isError: true)
        }
    }

    /// Canonical JSON form of an arguments string: decoded then
    /// re-serialized with sorted keys, so key-reorderings and
    /// whitespace differences hash to the same signature. Falls
    /// through to the original string on parse failure.
    private static func canonicalize(_ json: String) -> String {
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data),
              let reencoded = try? JSONSerialization.data(
                withJSONObject: obj, options: [.sortedKeys]),
              let result = String(data: reencoded, encoding: .utf8)
        else {
            return json.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return result
    }

    private func firstSentence(_ s: String) -> String {
        if let dot = s.firstIndex(where: { ".!?\n".contains($0) }) {
            return String(s[..<dot])
        }
        return s.count > 120 ? String(s.prefix(120)) + "…" : s
    }
}
