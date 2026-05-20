import Foundation

/// Strips chain-of-thought / reasoning tags from a model's raw output
/// before the server formats it for any chat-completion response.
///
/// Two tag shapes leak today:
///   - `<thinking>...</thinking>` - the Anthropic-style tag KrillLM
///     itself injects when a request opts into `thinking` (see
///     `Server.swift` `/v1/messages`).
///   - `<think>...</think>` - emitted natively by Qwen 3 by default
///     (its instruct templates open the reasoning block before the
///     first user turn). Other reasoning models (DeepSeek R1
///     distills) emit the same tag.
///
/// Both shapes are stripped from the visible text returned to the
/// client. The captured inner text is returned separately so chat
/// surfaces that support a `thinking` field can populate it; surfaces
/// that do not just discard the second return value.
public enum ReasoningParser {
    /// Strip the FIRST balanced `<thinking>` or `<think>` block from
    /// `text`. Returns the cleaned text and the captured reasoning
    /// content (trimmed of surrounding whitespace). If no tag is
    /// present, returns the input unchanged and a nil reasoning
    /// string.
    ///
    /// Unbalanced tags (open without close, or vice versa) are left
    /// intact rather than truncated to half a tag, so the client at
    /// least sees the raw model output instead of a silently broken
    /// payload.
    public static func strip(_ text: String) -> (visible: String, thinking: String?) {
        for tag in ["thinking", "think"] {
            let open = "<\(tag)>"
            let close = "</\(tag)>"
            if let s = text.range(of: open), let e = text.range(of: close),
               s.upperBound <= e.lowerBound {
                let captured = String(text[s.upperBound ..< e.lowerBound])
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                var visible = text
                visible.removeSubrange(s.lowerBound ..< e.upperBound)
                return (visible.trimmingCharacters(in: .whitespacesAndNewlines), captured.isEmpty ? nil : captured)
            }
        }
        return (text, nil)
    }
}
