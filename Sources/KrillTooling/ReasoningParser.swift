import Foundation

/// Strips chain-of-thought / reasoning tags from a model's raw output
/// before the server formats it for any chat-completion response.
///
/// Three tag shapes leak today:
///   - `<thinking>...</thinking>` - the Anthropic-style tag Krill
///     itself injects when a request opts into `thinking` (see
///     `Server.swift` `/v1/messages`).
///   - `<think>...</think>` - emitted natively by Qwen 3 by default
///     (its instruct templates open the reasoning block before the
///     first user turn). Other reasoning models (DeepSeek R1
///     distills) emit the same tag.
///   - `<|channel>...<channel|>` (and `<|think|>...<think|>`) - Gemma 4's
///     native reasoning channel. These are plain inline text (NOT special
///     tokens); the checkpoint's own chat template strips them from prior
///     turns via its `strip_thinking` macro, and we apply the same removal
///     to the live generation. Previously these were only stripped on the
///     tool-call path (`ToolCalling.extractGemma4`), so plain chat/generate
///     responses leaked the literal markers.
///
/// All shapes are stripped from the visible text returned to the
/// client. The captured inner text is returned separately so chat
/// surfaces that support a `thinking` field can populate it; surfaces
/// that do not just discard the second return value.
public enum ReasoningParser {
    private static let tags = ["thinking", "think"]

    /// Gemma 4 native reasoning markers as (open, close) pairs. Unlike the
    /// `<x>`/`</x>` tags these are asymmetric inline text the model emits
    /// before its visible answer.
    static let gemmaMarkers: [(open: String, close: String)] = [
        ("<|channel>", "<channel|>"),
        ("<|think|>", "<think|>"),
    ]

    /// Remove EVERY Gemma-4 reasoning-marker span from `text` (the model can
    /// open more than one channel). On a missing close marker (output
    /// truncated mid-channel) everything from the open marker to end-of-text
    /// is dropped. Returns the cleaned text plus the concatenated captured
    /// reasoning (nil if none). Shared with `ToolCalling.extractGemma4`.
    static func stripGemmaChannels(_ text: String) -> (visible: String, thinking: String?) {
        var cleaned = text
        var captured = ""
        for (open, close) in gemmaMarkers {
            while let s = cleaned.range(of: open) {
                if let e = cleaned.range(of: close, range: s.upperBound ..< cleaned.endIndex) {
                    captured += cleaned[s.upperBound ..< e.lowerBound]
                    cleaned.removeSubrange(s.lowerBound ..< e.upperBound)
                } else {
                    captured += cleaned[s.upperBound ..< cleaned.endIndex]
                    cleaned.removeSubrange(s.lowerBound ..< cleaned.endIndex)
                }
            }
            // Orphan close marker (the model double-closed a channel, e.g.
            // `<|channel>t<channel|>answer<channel|>`): the text before it is
            // the visible answer, so drop just the marker.
            while let e = cleaned.range(of: close) {
                cleaned.removeSubrange(e)
            }
        }
        let t = captured.trimmingCharacters(in: .whitespacesAndNewlines)
        return (cleaned, t.isEmpty ? nil : t)
    }

    /// Strip reasoning from `text`: first every Gemma-4 channel span, then the
    /// FIRST `<thinking>`/`<think>` block. Returns the cleaned text and the
    /// captured reasoning content (trimmed; nil if empty).
    ///
    /// If a `<think>` opening tag is present but its close is missing
    /// (truncated by `max_tokens` before `</think>`, common with reasoning
    /// models on small budgets), everything from the opening tag to
    /// end-of-text is treated as reasoning and dropped from the visible
    /// output. This prevents the entire reasoning chain from leaking when the
    /// model never reached the closing tag. The pre-tag prefix (typically
    /// empty) is preserved.
    public static func strip(_ text: String) -> (visible: String, thinking: String?) {
        // Gemma-4 channels first (all occurrences), then the generic tag.
        let (afterGemma, gemmaThinking) = stripGemmaChannels(text)
        var captures: [String] = []
        if let g = gemmaThinking { captures.append(g) }

        var visible = afterGemma
        for tag in tags {
            let open = "<\(tag)>"
            let close = "</\(tag)>"
            guard let s = visible.range(of: open) else { continue }
            if let e = visible.range(of: close, range: s.upperBound ..< visible.endIndex) {
                captures.append(String(visible[s.upperBound ..< e.lowerBound]))
                visible.removeSubrange(s.lowerBound ..< e.upperBound)
            } else {
                // Open tag but no close: discard from the tag onward, keep the
                // pre-tag prefix.
                captures.append(String(visible[s.upperBound ..< visible.endIndex]))
                visible = String(visible[visible.startIndex ..< s.lowerBound])
            }
            break  // only the first matched tag is stripped per call
        }

        let thinking = captures.joined(separator: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return (
            visible.trimmingCharacters(in: .whitespacesAndNewlines),
            thinking.isEmpty ? nil : thinking)
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

    private static let openTags = ["<thinking>", "<think>", "<|channel>", "<|think|>"]
    private static let closingFor: [String: String] = [
        "<thinking>": "</thinking>",
        "<think>": "</think>",
        // Gemma 4 native reasoning channel (asymmetric open/close).
        "<|channel>": "<channel|>",
        "<|think|>": "<think|>",
    ]
    /// Gemma close markers encountered OUTSIDE a block (the model
    /// double-closed a channel). Never legitimate visible text; dropped
    /// silently while the surrounding text is passed through.
    private static let orphanCloseMarkers = ["<channel|>", "<think|>"]
    private static let allMarkers = openTags + orphanCloseMarkers
    /// Max prefix length we hold while waiting to disambiguate a
    /// partial opening tag. Equal to the longest possible opening
    /// tag length.
    private static let maxOpenPrefix: Int = {
        allMarkers.map(\.count).max() ?? 0
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
                // Look for an opening tag (or an orphan close marker to
                // drop). If a complete one is present, flush text before
                // it and switch state.
                if let (openRange, tag) = firstMarker(in: buffer) {
                    emit += buffer[buffer.startIndex ..< openRange.lowerBound]
                    buffer.removeSubrange(buffer.startIndex ..< openRange.upperBound)
                    if let closing = Self.closingFor[tag] {
                        state = .insideBlock(closing: closing)
                    }  // orphan close: drop the marker, stay in .preamble
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
                // completes into a real opening tag (or an orphan close
                // marker to drop), definitely is NOT one, or we still
                // need more bytes.
                if let (openRange, tag) = firstMarker(in: buffer),
                   openRange.lowerBound == buffer.startIndex {
                    buffer.removeSubrange(buffer.startIndex ..< openRange.upperBound)
                    if let closing = Self.closingFor[tag] {
                        state = .insideBlock(closing: closing)
                    } else {
                        state = .preamble
                    }
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
                // character appears; then resume scanning for the
                // NEXT reasoning block. Gemma 4 can emit a run of
                // channel blocks (a degenerate think loop produces
                // `<|channel>thought<channel|>` over and over); going
                // back to `.preamble` strips them all instead of
                // dumping everything after the first block as raw text.
                let trimmed = buffer.drop(while: { $0.isWhitespace })
                if trimmed.isEmpty {
                    buffer.removeAll(keepingCapacity: true)
                    return emit
                }
                buffer = String(trimmed)
                state = .preamble
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

    /// Earliest complete marker in `s`: an opening tag (look up its close
    /// in `closingFor`) or an orphan close marker (no `closingFor` entry).
    private func firstMarker(in s: String) -> (Range<String.Index>, String)? {
        var best: (Range<String.Index>, String)?
        for tag in Self.allMarkers {
            if let r = s.range(of: tag) {
                if best == nil || r.lowerBound < best!.0.lowerBound {
                    best = (r, tag)
                }
            }
        }
        return best
    }

    private func isPrefixOfAnyOpenTag(_ s: String) -> Bool {
        for tag in Self.allMarkers where tag.hasPrefix(s) {
            return true
        }
        return false
    }

    private func trailingOpenPrefixLength(_ s: String) -> Int {
        // Largest n such that the last n characters of `s` form a
        // proper prefix of a marker.
        let limit = min(s.count, Self.maxOpenPrefix)
        for n in stride(from: limit, through: 1, by: -1) {
            let suffix = String(s.suffix(n))
            if isPrefixOfAnyOpenTag(suffix) && !Self.allMarkers.contains(suffix) {
                return n
            }
        }
        return 0
    }
}
