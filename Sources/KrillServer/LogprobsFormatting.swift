import Foundation
import KrillEngine
import KrillSampler
import KrillTooling

// Phase 1 OpenAI `logprobs` support (docs/LOGPROBS_PLAN.md). Shared by the
// non-streaming, streaming, and tool-chat `/v1/chat/completions` handlers so
// the "which tokens get an entry" rule (§4.3, §6 decision on the reasoning
// filter) lives in exactly one place.

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
func logprobsContentEntry(tokenId: Int, info: TokenLogprobInfo, eng: InferenceEngine) -> [String: Any] {
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
/// Conservative alignment rule (documented limitation, phase 1): a token
/// gets an entry only when `StreamingReasoningFilter.consume(_:)` emits back
/// EXACTLY that token's own text for that call - i.e. the filter passed the
/// chunk straight through with no cross-token buffering. This is correct for
/// every ordinary token (buffering only ever engages while the filter is
/// disambiguating a possible `<think>`/ATEM tag prefix) and for every
/// reasoning-block-interior token (which never emits anything, len 0, and is
/// correctly dropped). The rare case this under-reports is a token whose
/// text is held because it LOOKS LIKE the start of a reasoning tag but turns
/// out not to be one (e.g. ordinary prose containing the literal fragment
/// "<th"), where the held text is later flushed together with a NEIGHBORING
/// token's text; both tokens are dropped from `logprobs.content[]` rather
/// than risking a misattributed token/bytes pair. This never duplicates an
/// entry and never attaches the wrong bytes to a token.
final class LogprobsAggregator {
    private let filter = StreamingReasoningFilter()
    private let eng: InferenceEngine
    private let enabled: Bool
    private(set) var entries: [[String: Any]] = []

    init(engine: InferenceEngine, enabled: Bool) {
        self.eng = engine
        self.enabled = enabled
    }

    /// Feed one token event through the reasoning filter. Returns the text
    /// safe to emit to the client now (identical to calling the filter
    /// directly) - callers use this exactly as they used
    /// `reasoningFilter.consume(event.text)` before, whether or not
    /// logprobs were requested.
    func consume(_ event: TokenEvent) -> String {
        let emitted = filter.consume(event.text)
        if enabled, let info = event.logprob, !emitted.isEmpty, emitted.count == event.text.count {
            entries.append(logprobsContentEntry(tokenId: event.tokenId, info: info, eng: eng))
        }
        return emitted
    }

    /// Flush any trailing held text at end-of-stream (mirrors
    /// `StreamingReasoningFilter.finish()`). Never attributed to an entry -
    /// see the conservative-rule note above.
    func finish() -> String { filter.finish() }
}

/// `choices[].logprobs` object for a non-streaming chat completion, or the
/// per-chunk `choices[0].logprobs` for a streaming one that carries content.
/// `NSNull()` when the request did not ask for logprobs (§3.1's documented
/// default) - callers check `wantLogprobs` before calling this.
func logprobsChoiceJSON(content: [[String: Any]]) -> [String: Any] {
    ["content": content]
}
