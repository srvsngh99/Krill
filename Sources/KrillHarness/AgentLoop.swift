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
    public let messages: [[String: String]]
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

    public init(
        generator: any HarnessGenerator,
        tools: ToolRegistry,
        maxIterations: Int = 12,
        constrainToolArgs: Bool = true
    ) {
        self.generator = generator
        self.tools = tools
        self.maxIterations = max(1, maxIterations)
        self.constrainToolArgs = constrainToolArgs
    }

    public func run(user: String, system: String? = nil) async -> AgentTranscript {
        let format = generator.toolFormat
        let specs = tools.specs()
        let hasTools = !specs.isEmpty

        var messages: [[String: String]] = []
        if let system, !system.isEmpty {
            messages.append(["role": "system", "content": system])
        }
        messages.append(["role": "user", "content": user])
        messages = ToolCalling.injectToolSystem(into: messages, tools: specs, format: format)

        var steps: [AgentStep] = []
        var finalText = ""
        var hitLimit = false

        var iteration = 0
        while true {
            if iteration >= maxIterations {
                hitLimit = true
                break
            }
            iteration += 1

            let output = await generator.complete(messages: messages)
            let (calls, cleaned) = ToolCalling.extractIfToolsOffered(
                from: output, hasTools: hasTools, format: format)
            // Strip reasoning (`<think>`/`<thinking>`/Gemma channels) from the
            // text we DISPLAY. The raw `output` still goes into the message
            // history below, so the model keeps its own context and tool-call
            // markers are untouched - this is display-only cleanup.
            let visible = strippedForDisplay(cleaned)

            if calls.isEmpty {
                finalText = visible
                steps.append(AgentStep(assistantText: visible, toolCalls: []))
                break
            }

            // Record the model's own turn (raw, so it sees the call it made),
            // then run each tool and feed the observation back as a user turn
            // (template-safe across every family - a dedicated tool role can
            // come with the richer multi-turn work in a later PR).
            messages.append(["role": "assistant", "content": output])
            var invocations: [ToolInvocation] = []
            for call in calls {
                guard let tool = tools.tool(named: call.name), let spec = tools.spec(named: call.name) else {
                    let result = ToolResult(content: "Error: unknown tool '\(call.name)'", isError: true)
                    invocations.append(ToolInvocation(
                        name: call.name, argumentsJSON: call.argumentsJSON, result: result))
                    messages.append(["role": "user", "content": "Tool result (\(call.name)):\n\(result.content)"])
                    continue
                }

                let argsJSON = await repairedArgs(for: call, spec: spec, history: messages)
                let result = await tool.run(argumentsJSON: argsJSON)
                invocations.append(ToolInvocation(
                    name: call.name, argumentsJSON: argsJSON, result: result))
                messages.append([
                    "role": "user",
                    "content": "Tool result (\(call.name)):\n\(result.content)",
                ])
            }
            steps.append(AgentStep(assistantText: visible, toolCalls: invocations))
        }

        return AgentTranscript(
            steps: steps, finalText: finalText,
            hitIterationLimit: hitLimit, messages: messages)
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
