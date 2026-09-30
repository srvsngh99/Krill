import Foundation
import KrillEngine
import KrillSampler
import KrillTooling

// Phase 1 OpenAI `logprobs` support (docs/LOGPROBS_PLAN.md). Shared by the
// non-streaming, streaming, and tool-chat `/v1/chat/completions` handlers so
// the "which tokens get an entry" rule (§4.3, §6 decision on the reasoning
// filter) lives in exactly one place.

/// The narrow slice of `InferenceEngine` that logprobs formatting needs:
/// byte-exact recovery for a token id, independent of the lossy
/// `decodeForOutput` path (docs/LOGPROBS_PLAN.md §4.2), and whether a token
/// is a structural/special one that must never reach the visible answer.
/// `InferenceEngine` already has all three methods with these exact
/// signatures, so it conforms with an empty extension below - the protocol
/// exists purely so `LogprobsAggregatorTests` can drive the aggregator with
/// a synthetic byte table instead of a real loaded tokenizer.
protocol TokenLogprobResolver {
    func rawTokenBytes(for tokenId: Int) -> [UInt8]?
    func lossyTokenString(bytes: [UInt8]) -> String
    func isOutputSuppressedToken(_ tokenId: Int) -> Bool
}

extension InferenceEngine: TokenLogprobResolver {}

/// One `logprobs.content[]` (or `top_logprobs[]`) entry as OpenAI's wire
/// shape: `{token, logprob, bytes, top_logprobs?}`. Pure - the caller
/// resolves `token`/`bytes` from the tokenizer before calling this.
func logprobEntryJSON(
    token: String, logprob: Float, bytes: [UInt8], topLogprobs: [[String: Any]]? = nil
) -> [String: Any] {
    var entry: [String: Any] = [
        "token": token,
        "logprob": logprob,
        "bytes": bytes.map { Int($0) },
    ]
    if let topLogprobs { entry["top_logprobs"] = topLogprobs }
    return entry
}

/// Resolve one sampled token's `logprobEntryJSON`, including its
/// `top_logprobs` alternates (always present as an array - possibly empty,
/// per docs/LOGPROBS_PLAN.md §3.1's decision that OpenAI SDK types treat it
/// as a required list). `eng` provides the raw-piece byte recovery
/// (`rawTokenBytes(for:)`, independent of the lossy `decodeForOutput` path -
/// §4.2).
func logprobsContentEntry(tokenId: Int, info: TokenLogprobInfo, eng: TokenLogprobResolver) -> [String: Any] {
    let bytes = eng.rawTokenBytes(for: tokenId) ?? []
    let token = eng.lossyTokenString(bytes: bytes)
    let alternates: [[String: Any]] = info.topAlternates.map { alt in
        let altBytes = eng.rawTokenBytes(for: alt.tokenId) ?? []
        return logprobEntryJSON(
            token: eng.lossyTokenString(bytes: altBytes),
            logprob: alt.logprob, bytes: altBytes)
    }
    return logprobEntryJSON(token: token, logprob: info.logprob, bytes: bytes, topLogprobs: alternates)
}

/// Collects per-token `logprobs.content[]` entries alongside the existing
/// per-token reasoning-block filtering, so a token's logprob entry travels
/// with the same "did this reach the visible answer" decision the text
/// itself already gets (docs/LOGPROBS_PLAN.md §4.3, §6 "which tokens get
/// entries").
///
/// Required behaviour (finding #1 fix, replacing the old same-call/
/// same-length rule that silently dropped any held or empty-decode token):
/// EVERY generated token that contributes to the visible answer gets exactly
/// one entry, in order; a token inside a reasoning block, or a suppressed/
/// special token, gets none. In particular:
///
/// - A token the filter HOLDS while disambiguating a possible tag prefix
///   (it buffers on `<`, which shows up in code - `x < y`, `<div>`,
///   generics) is not lost: it stays pending until the hold resolves, then
///   gets its entry (or is dropped, if the hold resolved into a real
///   reasoning tag).
/// - A token whose OWN `decodeForOutput` text is empty (one piece of a
///   multi-byte character split across tokens - byte-fallback or a partial
///   byte-level-BPE sequence) still gets a FIFO slot and, if it sits in the
///   visible region, its own entry (with real `bytes` from
///   `rawTokenBytes(for:)`, independent of the empty `decodeForOutput`
///   text) - this is exactly the gap OpenAI's `bytes` field exists to close.
/// - The final `finish()` flush resolves every token still pending (a
///   `max_tokens`-truncated stream can end mid-hold), attributing entries
///   there too, not dropping them.
///
/// Design: a FIFO of tokens fed to the filter but not yet resolved
/// (emitted or discarded). `StreamingReasoningFilter` is not made
/// token-aware itself (it is shared by CLI/TUI call sites this feature must
/// not touch); instead this aggregator owns a PRIVATE filter instance and
/// reconstructs, from `StreamingReasoningFilter.pendingUTF8Length` before
/// and after each call plus the actual emitted text, exactly how many UTF-8
/// bytes were newly resolved this call and whether they were emitted or
/// discarded (see `resolveNewlyProcessed` for the exact accounting - it
/// works in UTF-8 byte units specifically because `Character`/grapheme-
/// cluster counts are NOT additive under string concatenation, e.g. a
/// Devanagari base+matra pair split across two fed chunks would throw a
/// `Character`-count-based version of this accounting off). Draining the
/// FIFO in order then tells us, per pending token, whether it fell in the
/// resolved-emit region, the resolved-discard region, or is still pending.
final class LogprobsAggregator {
    private let filter = StreamingReasoningFilter()
    private let eng: TokenLogprobResolver
    private let enabled: Bool
    private(set) var entries: [[String: Any]] = []

    /// One token fed to the filter whose emit/discard fate is not yet fully
    /// resolved. FIFO order matches feed order.
    private struct Pending {
        let tokenId: Int
        let info: TokenLogprobInfo?
        /// This token's OWN REMAINING (not yet resolved) contribution to
        /// the filter's input stream. Starts as exactly `event.text`
        /// (possibly "" - a byte-fallback/partial byte-level-BPE piece
        /// whose own decode is empty) and is trimmed from the front as a
        /// resolution call consumes part of it without fully resolving the
        /// whole token (e.g. the first few characters of a long token are
        /// discarded as reasoning content while the rest is still held).
        /// Concatenating every still-pending token's `text`, in FIFO order,
        /// always reproduces EXACTLY the filter's own internal buffer
        /// content, which is what makes the byte-offset attribution below
        /// exact without needing to ask the filter for its buffer directly.
        var text: String
        var utf8Length: Int { text.utf8.count }
        /// This token's own decoded text exactly as fed to the filter
        /// (`event.text`), kept UNCHANGED for the life of this `Pending`
        /// (unlike `text` above, which is trimmed as the token resolves) -
        /// needed to compute `visibleRange`-relative byte slices for a
        /// partial entry (see `LogprobsAggregator.entryJSON`).
        let fullText: String
        /// True for a structural/special token that must never get an
        /// entry regardless of where it falls (`InferenceEngine.
        /// isOutputSuppressedToken`). Still occupies a FIFO slot (its own
        /// `event.text` is always "" by construction) purely so the byte
        /// tiling invariant holds even in that case.
        let suppressed: Bool
        /// True once ANY portion of this token (across however many partial
        /// resolutions it took) has overlapped the emit region.
        var sawEmit: Bool = false
        /// How many bytes have been trimmed off the FRONT of the token's
        /// original text so far (across however many partial resolutions it
        /// took) - lets `visibleRange` below be expressed in coordinates
        /// relative to the token's own original bytes even though `text`
        /// itself has already had the resolved prefix removed.
        var consumedFromFront: Int = 0
        /// Byte range, relative to the token's OWN original text, confirmed
        /// to overlap the emit region so far (nil until the first overlap).
        /// `StreamingReasoningFilter`'s whitespace-edge trimming
        /// (2026-09-30 stream/non-stream parity fix) can now split a SINGLE
        /// token's fate: e.g. a token whose text is `":\n  return \"x\"\n"`
        /// has its trailing `"\n"` held back and finally dropped at
        /// end-of-stream (matching `ReasoningParser.strip(_:)`'s trailing
        /// trim) while the rest of the same token already reached the
        /// client as visible `content`. Whitespace-edge trimming only ever
        /// removes a PREFIX (leading-whitespace-eat) or a SUFFIX (trailing-
        /// whitespace-hold) of what a token contributes - never a middle
        /// span - so the visible portion is always contiguous and this
        /// single range is enough to track it (widened, never fragmented,
        /// as more of the token resolves). A straddling token gets a
        /// PARTIAL entry (bytes trimmed to `visibleRange`, via
        /// `partialEntryBytes`) instead of its full raw bytes, keeping
        /// `bytes-concat == content` exact. The same mechanism also covers
        /// the (vanishingly rare - would need a single generated token
        /// literally spanning a reasoning-tag boundary) reasoning-boundary
        /// straddle case, which used to always get a full-bytes entry when
        /// it overlapped at all; it now gets a correctly-scoped partial one
        /// instead, which is strictly more correct and untested either way.
        var visibleRange: Range<Int>?
    }
    private var pending: [Pending] = []

    init(engine: TokenLogprobResolver, enabled: Bool) {
        self.eng = engine
        self.enabled = enabled
    }

    /// Feed one token event through the reasoning filter. Returns the text
    /// safe to emit to the client now (identical to calling the filter
    /// directly) - callers use this exactly as they used
    /// `reasoningFilter.consume(event.text)` before, whether or not
    /// logprobs were requested. When `enabled` is false this is a pure
    /// passthrough with none of the FIFO bookkeeping below, so a request
    /// that does not ask for logprobs pays nothing extra.
    func consume(_ event: TokenEvent) -> String {
        guard enabled else { return filter.consume(event.text) }
        pending.append(Pending(
            tokenId: event.tokenId, info: event.logprob, text: event.text,
            fullText: event.text,
            suppressed: eng.isOutputSuppressedToken(event.tokenId)))
        let beforeLen = filter.pendingUTF8Length
        let emitted = filter.consume(event.text)
        let afterLen = filter.pendingUTF8Length
        resolveNewlyProcessed(consumedLen: beforeLen + event.text.utf8.count - afterLen, emitted: emitted)
        return emitted
    }

    /// Flush any trailing held text at end-of-stream (mirrors
    /// `StreamingReasoningFilter.finish()`), resolving every token still
    /// pending - the filter's buffer is always empty afterward, so nothing
    /// is left unattributed.
    func finish() -> String {
        guard enabled else { return filter.finish() }
        let beforeLen = filter.pendingUTF8Length
        let output = filter.finish()
        resolveNewlyProcessed(consumedLen: beforeLen, emitted: output)
        return output
    }

    /// Drain FIFO tokens whose bytes are now fully accounted for by this
    /// call's `consumedLen` newly-resolved bytes (emitted + discarded,
    /// computed by the caller from `pendingUTF8Length` before/after), and
    /// decide which ones get an entry.
    ///
    /// `emitted`'s bytes are always either an exact PREFIX of the resolved
    /// region (ordinary "text, then a tag/reasoning-block starts") or an
    /// exact SUFFIX of it ("finishing a held reasoning block, then trailing
    /// visible text") - `StreamingReasoningFilter`'s buffer only ever holds
    /// ONE dangling ambiguous run at a time (a partial tag-prefix scan, a
    /// partial closing-tag/ATEM-terminator suffix, or a partial ATEM header
    /// probe), so a single `consume`/`finish` call resolves at most one
    /// emit-run and one discard-run, strictly in stream order. Reconstructing
    /// the resolved region's true bytes (by concatenating pending tokens'
    /// `text`, per the FIFO invariant) and locating `emitted`'s bytes as a
    /// prefix or suffix of it (byte comparison, not `Character`-based, so
    /// Unicode grapheme merging at a concatenation seam can't mislead it)
    /// recovers that split exactly.
    private func resolveNewlyProcessed(consumedLen: Int, emitted: String) {
        guard consumedLen > 0 else { return }
        var fullBytes: [UInt8] = []
        fullBytes.reserveCapacity(consumedLen)
        for tok in pending {
            if fullBytes.count >= consumedLen { break }
            fullBytes.append(contentsOf: tok.text.utf8)
        }
        let consumedBytes = Array(fullBytes.prefix(consumedLen))
        let emittedBytes = Array(emitted.utf8)

        let emitRange: Range<Int>
        if consumedBytes.starts(with: emittedBytes) {
            // Emit-first: text before a tag/reasoning-block start (or pure
            // emit, if nothing was discarded this call).
            emitRange = 0 ..< emittedBytes.count
        } else if emittedBytes.count <= consumedBytes.count,
                  Array(consumedBytes.suffix(emittedBytes.count)) == emittedBytes {
            // Discard-first: a held reasoning block/ATEM message closes,
            // then trailing visible text follows in the same call (or pure
            // discard, if `emittedBytes` is empty).
            emitRange = (consumedBytes.count - emittedBytes.count) ..< consumedBytes.count
        } else {
            // Should not happen given the filter's one-dangling-run
            // structure (see doc comment). Fail safe: attribute nothing as
            // visible rather than risk pairing the wrong bytes with the
            // wrong token - this can only under-report, never misattribute.
            emitRange = consumedLen ..< consumedLen
        }
        drain(consumedLen: consumedLen, emitRange: emitRange)
    }

    /// Walk the FIFO from the front, popping every token FULLY covered by
    /// `[0, consumedLen)` and appending an entry for each one that overlaps
    /// `emitRange` (a zero-length token - empty own decode - counts as
    /// overlapping when its offset falls anywhere in `[emitRange.lowerBound,
    /// emitRange.upperBound)`, grouping it with whichever visible text
    /// starts right there, per the "attributed together with the next
    /// visible text" requirement). A suppressed token never gets an entry
    /// regardless of overlap.
    ///
    /// A token only PARTIALLY covered by `[0, consumedLen)` (its resolution
    /// spans more than one `consume`/`finish` call - e.g. a long reasoning
    /// token whose front is discarded this call while its tail stays held)
    /// is not popped: its `text` is trimmed to the still-unresolved
    /// remainder and `sawEmit` is updated, so the NEXT call that fully
    /// resolves it has the complete picture. Trimming is always safe at a
    /// valid UTF-8 scalar boundary - `consumedLen` is derived from
    /// `StreamingReasoningFilter`'s own Character-based buffer operations,
    /// which only ever cut at whole-scalar (grapheme-cluster) boundaries of
    /// its buffer, and every pending token's own text is independently a
    /// run of whole scalars starting where the previous token's ended.
    private func drain(consumedLen: Int, emitRange: Range<Int>) {
        var offset = 0
        var i = 0
        while i < pending.count {
            let tokLen = pending[i].utf8Length
            let tokEnd = offset + tokLen
            if tokEnd <= consumedLen {
                if tokLen > 0 {
                    widenVisibleRange(
                        &pending[i], streamRange: offset ..< tokEnd,
                        callOffset: offset, emitRange: emitRange)
                }
                let overlapsEmit = tokLen > 0
                    ? (offset < emitRange.upperBound && tokEnd > emitRange.lowerBound)
                    : (offset >= emitRange.lowerBound && offset < emitRange.upperBound)
                let visible = pending[i].sawEmit || overlapsEmit
                if visible, !pending[i].suppressed, let info = pending[i].info {
                    entries.append(entryJSON(for: pending[i], info: info))
                }
                offset = tokEnd
                i += 1
            } else if offset < consumedLen {
                let resolvedHere = consumedLen - offset
                if tokLen > 0 {
                    widenVisibleRange(
                        &pending[i], streamRange: offset ..< consumedLen,
                        callOffset: offset, emitRange: emitRange)
                }
                let overlapsEmit = offset < emitRange.upperBound && consumedLen > emitRange.lowerBound
                pending[i].sawEmit = pending[i].sawEmit || overlapsEmit
                let remaining = Array(pending[i].text.utf8).dropFirst(resolvedHere)
                pending[i].text = String(decoding: remaining, as: UTF8.self)
                pending[i].consumedFromFront += resolvedHere
                break
            } else {
                break
            }
        }
        pending.removeFirst(i)
    }

    /// Intersect `streamRange` (this call's resolved span for the token, in
    /// FIFO-relative stream coordinates) with `emitRange`, convert the
    /// overlap (if any) to coordinates relative to the token's OWN original
    /// bytes (`callOffset` is where `streamRange` starts, `pending.
    /// consumedFromFront` is how much of the token's front was already
    /// resolved in earlier calls), and widen `pending.visibleRange` to
    /// cover it. A no-op when there is no overlap this call.
    private func widenVisibleRange(
        _ pending: inout Pending, streamRange: Range<Int>,
        callOffset: Int, emitRange: Range<Int>
    ) {
        let lo = max(streamRange.lowerBound, emitRange.lowerBound)
        let hi = min(streamRange.upperBound, emitRange.upperBound)
        guard lo < hi else { return }
        let tokenLo = lo - callOffset + pending.consumedFromFront
        let tokenHi = hi - callOffset + pending.consumedFromFront
        if let existing = pending.visibleRange {
            pending.visibleRange = min(existing.lowerBound, tokenLo) ..< max(existing.upperBound, tokenHi)
        } else {
            pending.visibleRange = tokenLo ..< tokenHi
        }
    }

    /// Build this token's `logprobs.content[]` entry. When `visibleRange`
    /// covers the token's full original length (the overwhelmingly common
    /// case, and the only case before the 2026-09-30 whitespace-edge fix),
    /// this is byte-for-byte `logprobsContentEntry`'s ordinary full-token
    /// entry, using `eng.rawTokenBytes(for:)` exactly as before. Otherwise
    /// (a token whose whitespace was partly trimmed as a leading/trailing
    /// answer edge, or - vanishingly rare - straddling a reasoning-tag
    /// boundary) the entry's `bytes`/`token` are trimmed to just the visible
    /// slice, computed from the token's OWN decoded text (not
    /// `rawTokenBytes`, which has no defined slicing correspondence to a
    /// byte-fallback token's text) - see `Pending.visibleRange`.
    private func entryJSON(for pending: Pending, info: TokenLogprobInfo) -> [String: Any] {
        let fullUTF8 = Array(pending.fullText.utf8)
        guard let range = pending.visibleRange, range != 0 ..< fullUTF8.count else {
            return logprobsContentEntry(tokenId: pending.tokenId, info: info, eng: eng)
        }
        let bytes = Array(fullUTF8[range])
        let token = String(decoding: bytes, as: UTF8.self)
        // Alternates are hypothetical (never-sampled) tokens, unaffected by
        // this token's own whitespace trimming - built exactly as
        // `logprobsContentEntry` builds them, so `top_logprobs` keeps its
        // usual shape (always an array for the OpenAI dialect) even on a
        // partial entry.
        let alternates: [[String: Any]] = info.topAlternates.map { alt in
            let altBytes = eng.rawTokenBytes(for: alt.tokenId) ?? []
            return logprobEntryJSON(
                token: eng.lossyTokenString(bytes: altBytes),
                logprob: alt.logprob, bytes: altBytes)
        }
        return logprobEntryJSON(token: token, logprob: info.logprob, bytes: bytes, topLogprobs: alternates)
    }
}

/// `choices[].logprobs` object for a non-streaming chat completion, or the
/// per-chunk `choices[0].logprobs` for a streaming one that carries content.
/// `NSNull()` when the request did not ask for logprobs (§3.1's documented
/// default) - callers check `wantLogprobs` before calling this.
func logprobsChoiceJSON(content: [[String: Any]]) -> [String: Any] {
    ["content": content]
}

// MARK: - Tool-call logprobs, OpenAI dialect (2026-09-30 follow-up)
//
// See docs/LOGPROBS_PLAN.md's "Tool-call logprobs (2026-09-30)" section for
// the full sourcing. Summary: `ChoiceLogprobs.content` in the OpenAI Python
// SDK's generated types (`openai/types/chat/chat_completion.py`) is
// documented as covering `message.content` specifically - there is no field
// anywhere in `ChoiceLogprobs` for tool-call argument tokens - and is typed
// `Optional[List[ChatCompletionTokenLogprob]] = None`, i.e. nullable, not a
// signal that the whole `logprobs` object goes away. A live user report on
// the OpenAI developer forum ("Can I use logprobs & function calling at the
// same time?") confirms the real shape for a pure tool-call turn:
// `logprobs=ChoiceLogprobs(content=None)` - a present object with a null
// `content`, not a bare `logprobs: null`.

/// `choices[].logprobs` for a `/v1/chat/completions` turn that went through
/// the tool-chat path (`request.tools` non-empty). `hasToolCalls` is
/// `!calls.isEmpty` - Krill's own `message.content` is always null for such
/// a turn regardless of logprobs (any leftover pre-call text is discarded,
/// matching OpenAI's own observed content/tool_calls mutual exclusivity), so
/// there is currently no "mixed" turn to populate `content` for; `content`
/// (the collected entries for a plain, non-tool-calling reply) is used only
/// when `hasToolCalls` is false.
func toolChatLogprobsJSON(wantLogprobs: Bool, hasToolCalls: Bool, content: [[String: Any]]) -> Any {
    guard wantLogprobs else { return NSNull() }
    if hasToolCalls { return ["content": NSNull(), "refusal": NSNull()] as [String: Any] }
    return logprobsChoiceJSON(content: content)
}

// MARK: - Ollama `/api/chat` + `/api/generate` (2026-09-30 follow-up)
//
// Ollama's own wire shape (docs.ollama.com/api/chat, api/generate; confirmed
// against the Go source `api/types.go`) copies OpenAI's per-token
// `{token, logprob, bytes, top_logprobs}` shape almost exactly - close
// enough that `logprobsContentEntry` above is reused as-is for the per-token
// object. The one real difference: every Ollama field uses Go's
// `json:"...,omitempty"` tag, so an EMPTY `top_logprobs` list is OMITTED
// from a Logprob object (Go: `TopLogprobs []TokenLogprob
// json:"top_logprobs,omitempty"`), never sent as `[]` the way OpenAI's chat
// endpoint always does (that shape is a required, non-optional list per the
// OpenAI SDK's own response model - see the Resolutions section of
// docs/LOGPROBS_PLAN.md). This function strips that key when empty so the
// two dialects' conventions do not bleed into each other.

/// Convert one `logprobsContentEntry(...)`-shaped dictionary into Ollama's
/// own convention: drop `top_logprobs` entirely when it is an empty array
/// (Go `omitempty`), instead of keeping it as `[]`.
func ollamaLogprobEntryJSON(_ entry: [String: Any]) -> [String: Any] {
    var e = entry
    if let alts = e["top_logprobs"] as? [[String: Any]], alts.isEmpty {
        e.removeValue(forKey: "top_logprobs")
    }
    return e
}

/// The full top-level `logprobs` array for one Ollama response object
/// (`ChatResponse.Logprobs` / `GenerateResponse.Logprobs`), built from a
/// `LogprobsAggregator`'s `entries`. Ollama's `logprobs` field is itself
/// `omitempty` - callers only set this key on the response/chunk when the
/// result is non-empty, otherwise the key must be left out entirely (this
/// function does not decide that - it just formats the array).
func ollamaLogprobsArrayJSON(entries: [[String: Any]]) -> [[String: Any]] {
    entries.map(ollamaLogprobEntryJSON)
}

// MARK: - Legacy OpenAI `/v1/completions` (2026-09-30 follow-up)
//
// The legacy completions endpoint predates the chat endpoint's
// `content[]`-array shape and uses an older, flatter, four-parallel-arrays
// response (confirmed against the OpenAI Python SDK's
// `openai/types/completion_choice.py`: `Logprobs.{tokens, token_logprobs,
// top_logprobs, text_offset}`, all `Optional`) - a materially different wire
// shape from chat's, not just a renaming. `top_logprobs` here is a list of
// STRING-KEYED DICTS (`token -> logprob`), one dict per generated-token
// position, NOT chat's separate `{token, logprob, bytes}` object array.

/// Build the legacy `/v1/completions` `choices[].logprobs` object from a
/// `LogprobsAggregator`'s per-token `content[]`-shaped `entries` (the same
/// entries chat/Ollama use - reused here, then reshaped). docs/LOGPROBS_
/// PLAN.md §3.2.
///
/// - `tokens[i]` / `token_logprobs[i]`: the sampled token's own decoded
///   string and raw logprob, in generation order - always present for every
///   entry, matching "the API will always return the logprob of the sampled
///   token" (`completion_create_params.py`'s doc comment for `logprobs`).
/// - `top_logprobs[i]`: a `{token: logprob}` dict of the position's top-N
///   alternates (N = the request's `logprobs` value). Per that same doc
///   comment ("there may be up to `logprobs+1` elements"), the sampled
///   token is folded into this SAME dict when it is not already one of the
///   alternates - so the dict has at most `logprobs+1` entries, never fewer
///   than the true top-N found. When the request's `logprobs` was `0`
///   (sampled-token-only, no alternates requested), `entries[i]` carries no
///   alternates at all and this dict is `{}` for that position - the
///   sampled token's own logprob is NOT duplicated into an otherwise-empty
///   dict, since `logprobs: 0` explicitly asked for none.
/// - `text_offset[i]`: the character offset (Unicode scalar count, matching
///   Python's code-point-based string indexing that the real API's
///   `text_offset` is defined against) of `tokens[i]`'s first character
///   within the returned completion TEXT (not prompt+completion - this
///   endpoint has no `echo` support yet, Phase 3, so there is no prompt
///   prefix to offset past).
func legacyCompletionLogprobsJSON(entries: [[String: Any]]) -> [String: Any] {
    var tokens: [String] = []
    var tokenLogprobs: [Any] = []
    var topLogprobsList: [[String: Any]] = []
    var textOffset: [Int] = []
    var offset = 0
    for entry in entries {
        let token = entry["token"] as? String ?? ""
        let logprob = entry["logprob"] ?? 0
        tokens.append(token)
        tokenLogprobs.append(logprob)
        textOffset.append(offset)
        offset += token.unicodeScalars.count

        var dict: [String: Any] = [:]
        let alternates = entry["top_logprobs"] as? [[String: Any]] ?? []
        for alt in alternates {
            guard let altToken = alt["token"] as? String else { continue }
            dict[altToken] = alt["logprob"] ?? 0
        }
        // "Up to logprobs+1 elements": fold the sampled token in when it
        // wasn't already one of the alternates - but ONLY when alternates
        // were actually requested (`alternates` non-empty). A `logprobs: 0`
        // request must get `{}`, not `{token: logprob}` for every position.
        if !alternates.isEmpty, dict[token] == nil {
            dict[token] = logprob
        }
        topLogprobsList.append(dict)
    }
    return [
        "tokens": tokens,
        "token_logprobs": tokenLogprobs,
        "top_logprobs": topLogprobsList,
        "text_offset": textOffset,
    ]
}
