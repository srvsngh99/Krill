import Foundation
import KLMRegistry

/// `pull_model` operator tool. Wraps `AgentPuller`.
///
/// Confirmation lives at the agent boundary (the CLI emits a
/// `confirmationNeeded` event before invoking this when the
/// hardware fit is `risky` / `wontFit` AND `--yes` was not passed).
/// The tool itself never blocks: per adult mode, "warn loudly, do
/// not refuse" - the warning surfaces upstream of the call.
public struct PullModelTool: OperatorTool {
    public let name = "pull_model"
    public let description =
        "Download a model by alias (e.g. 'llama-3.2-3b'). Resolves " +
        "through the built-in alias map and the catalog. Streams " +
        "weights from HuggingFace; may take minutes on a slow link."
    public let parametersJSON = """
{"type":"object","properties":{\
"alias":{"type":"string","description":\
"Model alias from list_remote_catalog (e.g. 'llama-3.2-3b')."}\
},"required":["alias"],"additionalProperties":false}
"""

    private let puller: any AgentPuller

    public init(puller: any AgentPuller) {
        self.puller = puller
    }

    public func execute(arguments: [String: Any]) async throws -> String {
        guard let alias = arguments["alias"] as? String, !alias.isEmpty else {
            throw RegistryToolError.missingArgument("alias")
        }
        return try await puller.pull(alias: alias)
    }
}
