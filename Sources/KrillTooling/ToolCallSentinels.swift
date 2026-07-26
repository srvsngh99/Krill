import Foundation

/// Per-family policy for constraining the tool-name slot at sampling time.
///
/// The tool-name grammar (`ToolNameAutomaton` in KrillGrammar) only arms itself
/// inside a tool call, and it recognises a call by its family's opening
/// sentinel. This type is the single place that says, per family, what that
/// sentinel is - and, just as importantly, where there ISN'T one.
///
/// ## Why some families are excluded
///
/// Constraining needs an unambiguous marker that says "a tool name comes next".
/// Without one, the grammar would have to guess from ordinary text, and a wrong
/// guess is worse than no constraint: it would corrupt a legitimate answer. Two
/// families have no such marker:
///
/// - `.pythonic` emits `read_file(path="x")`. Until the `(` arrives, that is
///   indistinguishable from prose, and by then the name is already decoded.
/// - `.llama` emits a bare leading JSON object, `{"name": …, "parameters": …}`,
///   with no wrapper. Arming on `{"name"` alone would also fire on any JSON the
///   model legitimately writes - a real risk for an agent whose job includes
///   writing JSON files. Llama's `<|python_tag|>` form DOES carry a marker, so
///   that variant is covered; the bare form is not.
///
/// Excluded families lose nothing they had before: they keep deterministic name
/// normalisation and the grammar-constrained re-pick in `AgentLoop`. They simply
/// do not get the stronger "cannot be generated at all" guarantee.
public enum ToolCallSentinels {

    /// Opening sentinels after which a tool name is expected, or `[]` when the
    /// family has no unambiguous marker.
    public static func sentinels(for format: ToolCalling.ToolFormat) -> [String] {
        switch format {
        case .hermes, .qwen:
            // `<tool_call>{"name": …}</tool_call>` - the generic convention.
            return ["<tool_call>"]
        case .gemma4:
            // Prompted with `<tool_call>`; the legacy Gemma form is also accepted
            // by the parser, so arm on both.
            return ["<tool_call>", "<|tool_call|>"]
        case .mistral:
            // `[TOOL_CALLS][{"name": …}]`
            return ["[TOOL_CALLS]"]
        case .phi:
            // `<|tool_call|>[{"name": …}]<|/tool_call|>`
            return ["<|tool_call|>"]
        case .llama:
            // Only the tagged form is unambiguous; see the type doc.
            return ["<|python_tag|>"]
        case .pythonic:
            // No marker precedes the name; see the type doc.
            return []
        }
    }

    /// Whether the name slot can be constrained for this family.
    public static func supportsNameConstraint(_ format: ToolCalling.ToolFormat) -> Bool {
        !sentinels(for: format).isEmpty
    }

    /// The JSON key holding the tool name. Every family that carries a sentinel
    /// uses `name`; kept explicit so a future family can differ without the
    /// grammar hard-coding it.
    public static func nameKey(for format: ToolCalling.ToolFormat) -> String { "name" }
}
