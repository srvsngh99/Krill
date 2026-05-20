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
    private static let tags = ["thinking", "think"]

    /// Strip the FIRST `<thinking>` or `<think>` block from `text`.
    /// Returns the cleaned text and the captured reasoning content
    /// (trimmed of surrounding whitespace).
    ///
    /// If an opening tag is present but the closing tag is missing
    /// (truncated by `max_tokens` before `</think>`, common with
    /// reasoning models on small budgets), everything from the
    /// opening tag to end-of-text is treated as reasoning and
    /// dropped from the visible output. This prevents the entire
    /// reasoning chain from leaking when the model never reached
    /// the closing tag. The pre-tag prefix (typically empty) is
    /// preserved.
    public static func strip(_ text: String) -> (visible: String, thinking: String?) {
        for tag in tags {
            let open = "<\(tag)>"
            let close = "</\(tag)>"
            guard let s = text.range(of: open) else { continue }
            if let e = text.range(of: close, range: s.upperBound ..< text.endIndex) {
                let captured = String(text[s.upperBound ..< e.lowerBound])
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                var visible = text
                visible.removeSubrange(s.lowerBound ..< e.upperBound)
                return (
                    visible.trimmingCharacters(in: .whitespacesAndNewlines),
                    captured.isEmpty ? nil : captured)
            }
            // Open tag but no close: model exceeded max_tokens mid-
            // reasoning. Discard from `<think>` onward so the client
            // never sees the raw chain. Keep any pre-tag prefix
            // (typically empty for Qwen 3, whose template opens the
            // tag as the first generated token).
            let captured = String(text[s.upperBound ..< text.endIndex])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let visible = String(text[text.startIndex ..< s.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return (visible, captured.isEmpty ? nil : captured)
        }
        return (text, nil)
    }
}

/// Stateful incremental filter for streaming token output.
///
/// The non-stream `ReasoningParser.strip` operates on a fully
/// accumulated string. Streaming clients see tokens as they arrive;
/// to keep `<think>` from leaking to those clients, we hold tokens
/// until we know whether they belong to a reasoning block.
///
/// Contract:
///   - `consume(_:)` takes the next text chunk and returns the
///     substring (possibly empty) that is safe to emit to the
///     client right now.
///   - `finish()` returns any text held in the tail buffer that the
///     filter is confident is non-reasoning (e.g. a partial
///     non-tag suffix that turned out not to be a tag). Reasoning
///     content captured during streaming is discarded; streaming
///     surfaces do not have a "thinking" field on each chunk.
///
/// State machine:
///   - `.preamble`: text before any opening tag. Tokens are held
///     until we are confident they do not start a `<think...` or
///     `<thinking...` prefix; if they do, flush the safe portion
///     and transition to `.scanningOpen`.
///   - `.scanningOpen`: we have a partial opening tag and are
///     accumulating until either a full tag is seen (transition to
///     `.insideBlock`) or the running buffer cannot be a prefix
///     of any tag (flush the buffer literally and return to
///     `.preamble`).
///   - `.insideBlock`: tokens are reasoning content; drop them
///     until `</think>` / `</thinking>` is seen, then transition to
///     `.afterBlock`.
///   - `.afterBlock`: text outside reasoning. Tokens are emitted
///     verbatim. Only one block per stream is stripped, matching
///     `strip(_:)` behavior; subsequent `<think>` tags pass through
///     untouched.
public final class StreamingReasoningFilter {
    private enum State {
        case preamble
        case scanningOpen
        case insideBlock(closing: String)
        /// Just exited the reasoning block; trim any leading
        /// whitespace from incoming chunks before emitting. Falls
        /// through to `.afterBlock` once a non-whitespace
        /// character is seen.
        case justExited
        case afterBlock
    }

    private static let openTags = ["<thinking>", "<think>"]
    private static let closingFor: [String: String] = [
        "<thinking>": "</thinking>",
        "<think>": "</think>",
    ]
    /// Max prefix length we hold while waiting to disambiguate a
    /// partial opening tag. Equal to the longest possible opening
    /// tag length.
    private static let maxOpenPrefix: Int = {
        openTags.map(\.count).max() ?? 0
    }()

    private var state: State = .preamble
    private var buffer: String = ""

    public init() {}

    /// Feed the next streamed chunk. Returns the substring that is
    /// safe to emit to the client now.
    public func consume(_ chunk: String) -> String {
        var emit = ""
        buffer += chunk

        while !buffer.isEmpty {
            switch state {
            case .afterBlock:
                emit += buffer
                buffer.removeAll(keepingCapacity: true)

            case .preamble:
                // Look for an opening tag. If a complete one is
                // present, flush text before it and switch state.
                if let (openRange, tag) = firstOpenTag(in: buffer) {
                    emit += buffer[buffer.startIndex ..< openRange.lowerBound]
                    buffer.removeSubrange(buffer.startIndex ..< openRange.upperBound)
                    let closing = Self.closingFor[tag] ?? "</think>"
                    state = .insideBlock(closing: closing)
                    continue
                }
                // No complete tag. If the buffer ends with a prefix
                // that COULD become an opening tag, hold the
                // tail; emit everything before it.
                let prefixLen = trailingOpenPrefixLength(buffer)
                if prefixLen > 0 {
                    let safeEnd = buffer.index(buffer.endIndex, offsetBy: -prefixLen)
                    emit += buffer[buffer.startIndex ..< safeEnd]
                    buffer.removeSubrange(buffer.startIndex ..< safeEnd)
                    state = .scanningOpen
                    return emit
                }
                emit += buffer
                buffer.removeAll(keepingCapacity: true)

            case .scanningOpen:
                // We already held a partial-tag prefix. Either it
                // completes into a real opening tag, definitely is
                // NOT one, or we still need more bytes.
                if let (openRange, tag) = firstOpenTag(in: buffer),
                   openRange.lowerBound == buffer.startIndex {
                    buffer.removeSubrange(buffer.startIndex ..< openRange.upperBound)
                    let closing = Self.closingFor[tag] ?? "</think>"
                    state = .insideBlock(closing: closing)
                    continue
                }
                // Still ambiguous?
                if isPrefixOfAnyOpenTag(buffer) {
                    return emit
                }
                // Buffer can no longer be a prefix - flush as
                // literal text and return to preamble.
                state = .preamble
                continue

            case .insideBlock(let closing):
                if let closeRange = buffer.range(of: closing) {
                    buffer.removeSubrange(buffer.startIndex ..< closeRange.upperBound)
                    state = .justExited
                    continue
                }
                // Hold a tail equal to (closing.count - 1) bytes
                // in case the closing tag is split across chunks;
                // drop the rest as reasoning.
                let holdLen = min(buffer.count, closing.count - 1)
                buffer = String(buffer.suffix(holdLen))
                return emit

            case .justExited:
                // Drop leading whitespace from the joined buffer
                // (which may span the chunk that closed the block
                // AND subsequent chunks) until a non-whitespace
                // character appears; then promote to .afterBlock.
                let trimmed = buffer.drop(while: { $0.isWhitespace })
                if trimmed.isEmpty {
                    buffer.removeAll(keepingCapacity: true)
                    return emit
                }
                buffer = String(trimmed)
                state = .afterBlock
            }
        }
        return emit
    }

    /// Flush any text the filter is confident about at end-of-stream.
    /// Reasoning content (including an unterminated reasoning block)
    /// is discarded so a `max_tokens`-truncated stream does not leak.
    public func finish() -> String {
        switch state {
        case .preamble, .afterBlock:
            let out = buffer
            buffer.removeAll(keepingCapacity: true)
            return out
        case .justExited:
            // Whitespace-only buffer between </think> and EOS.
            buffer.removeAll(keepingCapacity: true)
            return ""
        case .scanningOpen:
            // The held tail looked like the start of a tag but no
            // tag ever completed. Emit literally.
            let out = buffer
            buffer.removeAll(keepingCapacity: true)
            return out
        case .insideBlock:
            // Truncated mid-reasoning. Drop.
            buffer.removeAll(keepingCapacity: true)
            return ""
        }
    }

    private func firstOpenTag(in s: String) -> (Range<String.Index>, String)? {
        var best: (Range<String.Index>, String)?
        for tag in Self.openTags {
            if let r = s.range(of: tag) {
                if best == nil || r.lowerBound < best!.0.lowerBound {
                    best = (r, tag)
                }
            }
        }
        return best
    }

    private func isPrefixOfAnyOpenTag(_ s: String) -> Bool {
        for tag in Self.openTags where tag.hasPrefix(s) {
            return true
        }
        return false
    }

    private func trailingOpenPrefixLength(_ s: String) -> Int {
        // Largest n such that the last n characters of `s` form a
        // proper prefix of an opening tag.
        let limit = min(s.count, Self.maxOpenPrefix)
        for n in stride(from: limit, through: 1, by: -1) {
            let suffix = String(s.suffix(n))
            if isPrefixOfAnyOpenTag(suffix) && !Self.openTags.contains(suffix) {
                return n
            }
        }
        return 0
    }
}
