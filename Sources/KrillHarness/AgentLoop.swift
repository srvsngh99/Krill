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

    public init(generator: any HarnessGenerator, tools: ToolRegistry, maxIterations: Int = 12) {
        self.generator = generator
        self.tools = tools
        self.maxIterations = max(1, maxIterations)
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

            if calls.isEmpty {
                finalText = cleaned
                steps.append(AgentStep(assistantText: cleaned, toolCalls: []))
                break
            }

            // Record the model's own turn (raw, so it sees the call it made),
            // then run each tool and feed the observation back as a user turn
            // (template-safe across every family - a dedicated tool role can
            // come with the richer multi-turn work in a later PR).
            messages.append(["role": "assistant", "content": output])
            var invocations: [ToolInvocation] = []
            for call in calls {
                let result: ToolResult
                if let tool = tools.tool(named: call.name) {
                    result = await tool.run(argumentsJSON: call.argumentsJSON)
                } else {
                    result = ToolResult(
                        content: "Error: unknown tool '\(call.name)'", isError: true)
                }
                invocations.append(ToolInvocation(
                    name: call.name, argumentsJSON: call.argumentsJSON, result: result))
                messages.append([
                    "role": "user",
                    "content": "Tool result (\(call.name)):\n\(result.content)",
                ])
            }
            steps.append(AgentStep(assistantText: cleaned, toolCalls: invocations))
        }

        return AgentTranscript(
            steps: steps, finalText: finalText,
            hitIterationLimit: hitLimit, messages: messages)
    }
}
