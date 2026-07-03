import Foundation
import KrillTooling

/// One tool call the loop executed, with its observation.
public struct ToolInvocation: Sendable, Equatable {
    public let name: String
    public let argumentsJSON: String
    public let result: ToolResult
    public init(name: String, argumentsJSON: String, result: ToolResult) {
        self.name = name
        self.argumentsJSON = argumentsJSON
        self.result = result
    }
}

/// One iteration of the loop: the model's (cleaned) text plus any tool calls
/// it made. The final step has an empty `toolCalls`.
public struct AgentStep: Sendable {
    public let assistantText: String
    public let toolCalls: [ToolInvocation]
    public init(assistantText: String, toolCalls: [ToolInvocation]) {
        self.assistantText = assistantText
        self.toolCalls = toolCalls
    }
}

/// The full record of a run: every step, the final answer, whether the loop
/// stopped because the model was done or because it hit the iteration cap, and
/// the final message history (useful for rendering and for a follow-up turn).
public struct AgentTranscript: Sendable {
    public let steps: [AgentStep]
    public let finalText: String
    public let hitIterationLimit: Bool
    /// True when the run stopped because its Task was cancelled (Ctrl-C in the
    /// TUI), as opposed to finishing or hitting the iteration cap.
    public let wasCancelled: Bool
    public let messages: [[String: String]]

    public init(
        steps: [AgentStep],
        finalText: String,
        hitIterationLimit: Bool,
        wasCancelled: Bool = false,
        messages: [[String: String]]
    ) {
        self.steps = steps
        self.finalText = finalText
        self.hitIterationLimit = hitIterationLimit
        self.wasCancelled = wasCancelled
        self.messages = messages
    }
}

/// An incremental event emitted as the loop runs, so a live UI (the `code` TUI)
/// can render the conversation as it happens instead of waiting for the whole
/// run to finish. Events arrive in the order the loop produces them. The batch
/// `AgentTranscript` is still returned at the end - events are an additive seam,
/// not a replacement.
///
/// Note: events carry the CLEANED (reasoning-stripped, tool-marker-free)
/// assistant text per turn, not raw token deltas. Raw streaming would leak
/// `<tool_call>` markers mid-stream; a renderer shows a working indicator during
/// generation and reveals the clean turn here.
public enum AgentEvent: Sendable, Equatable {
    /// The cleaned preamble text of a turn that is about to call tools (the
    /// `toolStarted`/`toolFinished` events for that turn follow). The text may be
    /// empty when the model jumps straight to a tool call. The TERMINAL turn's
    /// text is delivered by `finalAnswer`, not here - so a consumer can render
    /// `assistantTurn` and `finalAnswer` without ever double-printing.
    case assistantTurn(text: String)
    /// A tool passed the permission gate and is about to run, with the final
    /// (possibly repaired) arguments it will run with.
    case toolStarted(name: String, argumentsJSON: String)
    /// A tool call resolved: it ran, was denied by the permission gate, or named
    /// an unknown tool (distinguish via `invocation.result.isError`).
    case toolFinished(ToolInvocation)
    /// The loop ended with a final answer (a turn with no tool calls). This is
    /// the only event that carries the terminal turn's text.
    case finalAnswer(String)
    /// The loop stopped at the iteration cap without a final answer.
    case iterationLimitReached
    /// The run was cancelled (its Task was cancelled, e.g. Ctrl-C in the TUI).
    case cancelled
}

/// The core agentic loop: inject the tool system turn, then repeatedly
/// generate, parse tool calls (family-aware via `KrillTooling`), execute them,
/// feed observations back, and continue until the model answers without a tool
/// call or the iteration cap is reached.
///
/// Reuses the exact `ToolCalling` rendering/parsing the HTTP server uses, so
/// the in-process harness and the server agree on tool semantics.
public struct AgentLoop: Sendable {
    public let generator: any HarnessGenerator
    public let tools: ToolRegistry
    public let maxIterations: Int
    /// When true, a tool call whose args do not satisfy the tool's JSON schema
    /// (the empty-`{}` failure small local models produce) triggers a second,
    /// grammar-constrained pass that regenerates the args. This is the
    /// in-process differentiator - it is what lets a small model complete an
    /// agentic task instead of looping on empty arguments.
    public let constrainToolArgs: Bool
    /// Permission policy consulted before every tool call. The default
    /// (`.acceptAll`) runs every tool, preserving the autonomous flow the loop
    /// shipped with; `.plan` denies mutating tools, `.ask` defers them to `gate`.
    public let permission: PermissionPolicy
    /// Interactive approver for `.ask` decisions. When the policy returns `.ask`
    /// and this is nil, the call is denied (fail-safe: no silent execution).
    public let gate: (any PermissionGate)?

    public init(
        generator: any HarnessGenerator,
        tools: ToolRegistry,
        maxIterations: Int = 12,
        constrainToolArgs: Bool = true,
        permission: PermissionPolicy = PermissionPolicy(),
        gate: (any PermissionGate)? = nil
    ) {
        self.generator = generator
        self.tools = tools
        self.maxIterations = max(1, maxIterations)
        self.constrainToolArgs = constrainToolArgs
        self.permission = permission
        self.gate = gate
    }

    /// Run the loop to completion. Pass `onEvent` to observe progress live (the
    /// `code` TUI does); it is called synchronously from the loop in event order.
    /// Omitting it preserves the original batch behavior exactly.
    ///
    /// Pass `priorMessages` (a previous run's `transcript.messages`) to CONTINUE
    /// that conversation: the tool system turn and history are already present,
    /// so they are not re-injected - only the new `user` turn is appended. This
    /// is how the multi-turn TUI carries context across follow-up tasks. When
    /// `priorMessages` is empty, the loop starts fresh (system + user + tool
    /// injection), the original behavior.
    public func run(
        user: String,
        system: String? = nil,
        priorMessages: [[String: String]] = [],
        onEvent: (@Sendable (AgentEvent) -> Void)? = nil
    ) async -> AgentTranscript {
        let format = generator.toolFormat
        let specs = tools.specs()
        let hasTools = !specs.isEmpty

        var messages: [[String: String]]
        if priorMessages.isEmpty {
            messages = []
            if let system, !system.isEmpty {
                messages.append(["role": "system", "content": system])
            }
            messages.append(["role": "user", "content": user])
            messages = ToolCalling.injectToolSystem(into: messages, tools: specs, format: format)
        } else {
            // Continue an existing conversation: tools/history already injected.
            messages = priorMessages
            messages.append(["role": "user", "content": user])
        }

        var steps: [AgentStep] = []
        var finalText = ""
        var hitLimit = false
        var cancelled = false

        // Runaway guard: signatures of calls already executed (to drop exact
        // repeats a looping model re-emits) and a total tool-call budget.
        var executedSignatures = Set<String>()
        var executedToolCalls = 0
        let maxToolCalls = maxIterations * 4

        var iteration = 0
        while true {
            // Cooperative cancellation: when the enclosing Task is cancelled
            // (the `code` TUI does this on Ctrl-C), stop between turns rather
            // than starting another generation. The generator is expected to
            // honor cancellation too, so an in-flight turn returns promptly.
            if Task.isCancelled {
                cancelled = true
                break
            }
            if iteration >= maxIterations {
                hitLimit = true
                break
            }
            iteration += 1

            let output = await generator.complete(messages: messages)
            let (calls, cleaned) = ToolCalling.extractIfToolsOffered(
                from: output, hasTools: hasTools, format: format,
                knownToolNames: specs.map { $0.name })
            // Strip reasoning (`<think>`/`<thinking>`/Gemma channels) from the
            // text we DISPLAY. The raw `output` still goes into the message
            // history below, so the model keeps its own context and tool-call
            // markers are untouched - this is display-only cleanup.
            let visible = strippedForDisplay(cleaned)

            if calls.isEmpty {
                finalText = visible
                steps.append(AgentStep(assistantText: visible, toolCalls: []))
                onEvent?(.finalAnswer(visible))
                break
            }

            // Runaway guard: drop calls whose (name, args) was already executed
            // - a model re-emitting them is looping, not progressing. If a whole
            // turn is repeats, or the total budget is spent, stop and take the
            // visible text as the final answer instead of generating forever.
            var freshCalls: [ToolCalling.ParsedToolCall] = []
            for call in calls where executedSignatures.insert(toolSignature(call)).inserted {
                freshCalls.append(call)
            }
            let remaining = maxToolCalls - executedToolCalls
            if freshCalls.isEmpty || remaining <= 0 {
                finalText = visible
                steps.append(AgentStep(assistantText: visible, toolCalls: []))
                onEvent?(.finalAnswer(visible))
                break
            }
            let toRun = Array(freshCalls.prefix(remaining))
            executedToolCalls += toRun.count

            onEvent?(.assistantTurn(text: visible))

            // Record the model's own turn (raw, so it sees the call it made),
            // then run each tool and feed the observation back as a user turn
            // (template-safe across every family - a dedicated tool role can
            // come with the richer multi-turn work in a later PR).
            messages.append(["role": "assistant", "content": output])
            var invocations: [ToolInvocation] = []
            for call in toRun {
                // Feed `result` back as the observation, record the invocation,
                // and emit it, whether the tool ran, was denied, or was unknown.
                func record(args: String, _ result: ToolResult) {
                    let invocation = ToolInvocation(
                        name: call.name, argumentsJSON: args, result: result)
                    invocations.append(invocation)
                    messages.append([
                        "role": "user",
                        "content": "Tool result (\(call.name)):\n\(result.content)",
                    ])
                    onEvent?(.toolFinished(invocation))
                }

                guard let tool = tools.tool(named: call.name), let spec = tools.spec(named: call.name) else {
                    record(args: call.argumentsJSON,
                           ToolResult(content: "Error: unknown tool '\(call.name)'. "
                               + "Available tools: \(tools.names.joined(separator: ", ")).",
                               isError: true))
                    continue
                }

                // Permission gate: decide BEFORE repairing/running so a denied
                // tool costs no extra generation. A denial is fed back so the
                // model can adapt (e.g. switch to read-only investigation).
                let decision = permission.decision(
                    toolName: call.name, isReadOnly: tool.isReadOnly, isFileEdit: tool.isFileEdit)
                if case .deny(let reason) = decision {
                    record(args: call.argumentsJSON,
                           ToolResult(content: "Permission denied: \(reason)", isError: true))
                    continue
                }

                let argsJSON = await repairedArgs(for: call, spec: spec, history: messages)

                if case .ask = decision {
                    let approved = await (gate?.approve(
                        toolName: call.name, argumentsJSON: argsJSON) ?? false)
                    if !approved {
                        record(args: argsJSON, ToolResult(
                            content: "Permission denied: the user declined to run '\(call.name)'.",
                            isError: true))
                        continue
                    }
                }

                onEvent?(.toolStarted(name: call.name, argumentsJSON: argsJSON))
                record(args: argsJSON, await tool.run(argumentsJSON: argsJSON))
            }
            steps.append(AgentStep(assistantText: visible, toolCalls: invocations))
        }

        if cancelled { onEvent?(.cancelled) }
        else if hitLimit { onEvent?(.iterationLimitReached) }

        return AgentTranscript(
            steps: steps, finalText: finalText,
            hitIterationLimit: hitLimit, wasCancelled: cancelled, messages: messages)
    }

    /// Stable identity for a parsed call: name + raw arguments. Two calls with
    /// the same signature are treated as the same action for the runaway guard.
    private func toolSignature(_ call: ToolCalling.ParsedToolCall) -> String {
        call.name + "\u{1}" + call.argumentsJSON
    }

    /// Strip ALL reasoning blocks from text bound for the display surface.
    /// `ReasoningParser.strip` removes every Gemma channel but only the FIRST
    /// `<think>`/`<thinking>` block per call (the shared server/streaming
    /// convention). Small local models - the target of `krill code` - can emit
    /// several reasoning blocks in one turn, so apply `strip` to a fixpoint:
    /// each pass removes one more block and strictly shrinks the text, so this
    /// terminates. Keeping the loop here (not in the shared parser) leaves the
    /// server/streaming single-block semantics untouched.
    private func strippedForDisplay(_ text: String) -> String {
        var visible = text
        while true {
            let next = ReasoningParser.strip(visible).visible
            if next == visible { return visible }
            visible = next
        }
    }

    /// Selective, fail-open two-pass arg repair: if the model's free-form args
    /// already satisfy the schema (the common case for capable models), use
    /// them unchanged. Otherwise ask the model to re-emit ONLY the args object,
    /// grammar-constrained to the schema so required fields cannot be omitted,
    /// and accept the regenerated args only if they now satisfy the schema.
    private func repairedArgs(
        for call: ToolCalling.ParsedToolCall,
        spec: ServerToolSpec,
        history: [[String: String]]
    ) async -> String {
        guard constrainToolArgs,
              !ToolCalling.argsSatisfySchema(
                argumentsJSON: call.argumentsJSON, parametersJSON: spec.parametersJSON)
        else { return call.argumentsJSON }

        let repairMessages = history + [["role": "user", "content": ToolCalling.argsRegenPrompt(tool: spec)]]
        let repaired = await generator.completeConstrained(
            messages: repairMessages, jsonSchema: spec.parametersJSON)
        return ToolCalling.argsSatisfySchema(
            argumentsJSON: repaired, parametersJSON: spec.parametersJSON)
            ? repaired : call.argumentsJSON
    }
}
