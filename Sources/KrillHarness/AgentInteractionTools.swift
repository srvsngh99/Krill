/// Constructs the interaction tools for a classic/line-oriented agent surface.
///
/// `ask_user` requires a real question gate. `request_execute` usually does too,
/// except adaptive mode self-promotes before consulting the gate and therefore
/// must remain available in non-interactive scripts and CI.
public enum AgentInteractionTools {
    public static func make(
        mode: PermissionMode,
        permissionBox: PermissionBox,
        questionGate: (any UserQuestionGate)?
    ) -> [any Tool] {
        var tools: [any Tool] = []
        if let questionGate {
            tools.append(AskUserTool(gate: questionGate))
            tools.append(RequestExecuteTool(
                permissionBox: permissionBox, gate: questionGate))
        } else if mode == .adaptive {
            tools.append(RequestExecuteTool(
                permissionBox: permissionBox, gate: DecliningQuestionGate()))
        }
        return tools
    }
}
