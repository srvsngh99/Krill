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

    // MARK: - ATEM / Harmony recipient channels (Muse Glimmer)

    static let atemMessage = "<|message|>"
    static let atemStart = "<|start|>"
    /// `<|eom|>` ends a message that is NOT the end of the turn (more messages
    /// follow); `<|eot|>` ends the turn. Both terminate a channel's content.
    static let atemEnds = ["<|eom|>", "<|eot|>"]

    /// Recipient named in an ATEM message header, e.g. `assistant to=self` or
    /// ` to=web.search`. nil when the header carries no `to=` (the template's
    /// default recipient is `user`, i.e. visible).
    static func atemRecipient(in header: some StringProtocol) -> String? {
        guard let r = header.range(of: "to=") else { return nil }
        let name = header[r.upperBound...].prefix {
            !$0.isWhitespace && $0 != "<"
        }
        return name.isEmpty ? nil : String(name)
    }

    /// Split a Muse Glimmer generation into visible text and reasoning.
    ///
    /// The checkpoint's template routes each assistant message to a RECIPIENT
    /// rather than wrapping reasoning in paired tags:
    ///
    ///     [<|start|>assistant][ to=<recipient>]<|message|>content<|eom|or|eot|>
    ///
    /// `to=self` is the model's own scratchpad — the template renders a prior
    /// turn's `reasoning_content` as exactly `to=self` — so that content is
    /// reasoning and must not reach the user. No `to=` (or `to=user`) is the
    /// visible answer. The generation prompt ends at `<|start|>assistant`, so
    /// the FIRST header arrives without a `<|start|>` of its own.
    ///
    /// Any other recipient is a tool call (`<atem:function_calls>` XML). Those
    /// stay VISIBLE deliberately: Krill cannot parse this dialect yet (see
    /// `ChatTemplatePolicy.museGlimmer`), and showing an unparsed tool call is
    /// honest, where silently swallowing it would look like an empty reply.
    ///
    /// Content with no terminator (truncated by `max_tokens` mid-message) runs
    /// to end-of-text, matching how an unclosed `<think>` is treated above.
    static func stripAtemChannels(_ text: String) -> (visible: String, thinking: String?) {
        // Fast path: no ATEM structure, no work. Every other family lands here.
        guard text.contains(atemMessage) else { return (text, nil) }

        var visible = ""
        var captured = ""
        var rest = Substring(text)

        while let m = rest.range(of: atemMessage) {
            let recipient = atemRecipient(in: rest[rest.startIndex ..< m.lowerBound])
            let body = rest[m.upperBound...]
            // A message ends at its terminator, or at the next `<|start|>` if
            // the model skipped one.
            let stop = earliestRange(of: atemEnds + [atemStart], in: body)
            let content = stop.map { body[body.startIndex ..< $0.lowerBound] } ?? body
            if recipient == "self" {
                captured += content
            } else {
                visible += content
            }
            guard let stop else { rest = body[body.endIndex...]; break }
            rest = body[stop.upperBound...]
        }

        // Trailing text after the last terminator. Once ATEM structure is
        // established, prose only ever follows a `<|message|>`, so a dangling
        // header fragment (`<|start|>assistant to=user`, truncated) is scaffold
        // and is dropped rather than leaked.
        if let s = rest.range(of: atemStart) {
            visible += rest[rest.startIndex ..< s.lowerBound]
        } else {
            visible += rest
        }

        let t = captured.trimmingCharacters(in: .whitespacesAndNewlines)
        return (visible, t.isEmpty ? nil : t)
    }

    /// Earliest occurrence of any of `needles` in `haystack`.
    static func earliestRange(
        of needles: [String], in haystack: some StringProtocol
    ) -> Range<String.Index>? {
        var best: Range<String.Index>?
        for n in needles {
            if let r = haystack.range(of: n), best == nil || r.lowerBound < best!.lowerBound {
                best = r
            }
        }
        return best
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
        // ATEM recipient channels first (they wrap whole messages, including
        // any inner tags), then Gemma-4 channels, then the generic tag.
        let (afterAtem, atemThinking) = stripAtemChannels(text)
        let (afterGemma, gemmaThinking) = stripGemmaChannels(afterAtem)
        var captures: [String] = []
        if let a = atemThinking { captures.append(a) }
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
        /// Start of stream, and again after every ATEM terminator: the text so
        /// far may be an ATEM message header (` to=self<|message|>`). Held
        /// until it either completes into a header — which tells us whether the
        /// message that follows is visible — or stops being able to be one, at
        /// which point it is flushed literally and the filter behaves exactly
        /// as it always has. Non-ATEM models leave this state on their first
        /// chunk, so nothing is delayed for them.
        case atemProbe
        /// Inside an ATEM message body whose recipient we know.
        case atemContent(visible: Bool)
    }

    /// Terminators that close an ATEM message body. `<|start|>` is included
    /// because a model that skips its terminator still opens the next message.
    private static let atemStops = ReasoningParser.atemEnds + [ReasoningParser.atemStart]
    private static let maxAtemStopPrefix: Int = atemStops.map(\.count).max() ?? 0
    /// Cap on how much a header probe may hold before deciding it is not one.
    /// A real header is `<|start|>assistant to=<name><|message|>`; 64 leaves
    /// room for a long tool namespace without ever stalling a stream.
    private static let atemHeaderLimit = 64

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

    private var state: State = .atemProbe
    private var buffer: String = ""

    public init() {}

    /// Feed the next streamed chunk. Returns the substring that is
    /// safe to emit to the client now.
    public func consume(_ chunk: String) -> String {
        var emit = ""
        buffer += chunk

        while !buffer.isEmpty {
            switch state {
            case .atemProbe:
                if let r = buffer.range(of: ReasoningParser.atemMessage) {
                    // Header complete: `to=self` is the model's scratchpad and
                    // is dropped; anything else (no recipient = `user`, or a
                    // tool namespace) is shown. See `stripAtemChannels`.
                    let recipient = ReasoningParser.atemRecipient(
                        in: buffer[buffer.startIndex ..< r.lowerBound])
                    buffer.removeSubrange(buffer.startIndex ..< r.upperBound)
                    state = .atemContent(visible: recipient != "self")
                    continue
                }
                if couldBeAtemHeader(buffer) { return emit }
                // Not a header. Hand the buffer to the ordinary tag scanner
                // untouched, so every other family is unaffected.
                state = .preamble
                continue

            case .atemContent(let visible):
                if let stop = ReasoningParser.earliestRange(
                    of: Self.atemStops, in: buffer) {
                    if visible {
                        emit += buffer[buffer.startIndex ..< stop.lowerBound]
                    }
                    buffer.removeSubrange(buffer.startIndex ..< stop.upperBound)
                    state = .atemProbe
                    continue
                }
                // Hold only a tail that could still become a terminator.
                let holdLen = trailingAtemStopPrefixLength(buffer)
                let safeEnd = buffer.index(buffer.endIndex, offsetBy: -holdLen)
                if visible {
                    emit += buffer[buffer.startIndex ..< safeEnd]
                }
                buffer.removeSubrange(buffer.startIndex ..< safeEnd)
                return emit

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
        case .atemProbe:
            // Only reachable when every byte held is still a prefix of an ATEM
            // header — scaffold, never prose. A one-token reply truncated at
            // ` to` lands here; emitting it would leak the channel marker.
            buffer.removeAll(keepingCapacity: true)
            return ""
        case .atemContent(let visible):
            // Held tail of a visible message: emit it, minus any partial
            // terminator. Reasoning (`to=self`) is dropped as ever.
            let out = visible
                ? String(buffer.dropLast(trailingAtemStopPrefixLength(buffer)))
                : ""
            buffer.removeAll(keepingCapacity: true)
            return out
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

    /// True while `s` can still grow into an ATEM message header. Bounded, and
    /// deliberately narrow: an ordinary reply ("Paris…") matches none of these
    /// shapes, so it is released on the first chunk with no added latency.
    private func couldBeAtemHeader(_ s: String) -> Bool {
        if s.count > Self.atemHeaderLimit { return false }
        // A header in progress: recipient still accumulating.
        if s.hasPrefix(" to=") || s.hasPrefix(ReasoningParser.atemStart) { return true }
        // Too short to tell yet: still a prefix of a shape we recognise.
        return ReasoningParser.atemMessage.hasPrefix(s)
            || ReasoningParser.atemStart.hasPrefix(s)
            || " to=".hasPrefix(s)
    }

    /// Largest n such that the last n characters of `s` form a proper prefix
    /// of an ATEM terminator (so a `<|eom|>` split across chunks is not
    /// emitted as text).
    private func trailingAtemStopPrefixLength(_ s: String) -> Int {
        let limit = min(s.count, Self.maxAtemStopPrefix)
        for n in stride(from: limit, through: 1, by: -1) {
            let suffix = String(s.suffix(n))
            if Self.atemStops.contains(where: { $0.hasPrefix(suffix) }),
               !Self.atemStops.contains(suffix) {
                return n
            }
        }
        return 0
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
