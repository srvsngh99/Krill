import Foundation

/// How many tokens a single generation may produce.
///
/// Krill used to answer this with seven unrelated constants — 256, 512, 1024,
/// 2048, 4096 — spread across the CLI, the agent loop, and each server dialect.
/// They were arbitrary, they disagreed, and the smallest of them silently cut
/// off long replies: a reasoning model can spend its entire allowance thinking
/// and return nothing at all.
///
/// The real constraint is not a constant. It is the model's context window
/// minus whatever the prompt already occupies. That is how the rest of the
/// ecosystem behaves — Ollama's `num_predict: -1` means "generate until EOS or
/// the context is full", and OpenAI treats an omitted `max_tokens` the same
/// way — so an explicit number is a *cost or latency brake*, not a default.
///
/// This type is that policy in one place. `resolve` turns an optional caller
/// request plus what we know about the model into a concrete ceiling.
public enum TokenBudget {
    /// Headroom kept between the end of a generation and the hard context edge.
    ///
    /// Chat templates append a few tokens after the prompt we counted, and a
    /// KV cache that lands exactly on the boundary is far worse than one that
    /// stops a little early. Small enough to be irrelevant beside any real
    /// context window.
    public static let contextReserve = 256

    /// Ceiling applied when the model's context window is unknown.
    ///
    /// Not a guess at what a reply "should" be — only a runaway guard for the
    /// case where we cannot compute the real limit. Deliberately far above the
    /// old 1024 so a reasoning model's think phase does not consume it.
    public static let unknownContextFallback = 8192

    /// The sentinel a caller passes for "no explicit limit" — mirrors Ollama's
    /// `num_predict: -1`, which the server already accepts and maps here.
    public static let unlimited = -1

    /// Resolve the ceiling for one generation.
    ///
    /// - Parameters:
    ///   - requested: what the caller asked for. `nil` or `unlimited` (-1) both
    ///     mean "derive it"; any positive value is honoured EXACTLY, because an
    ///     explicit budget is a deliberate cost or latency decision and second-
    ///     guessing it is the very bug this type exists to remove.
    ///   - contextWindow: the model's total context in tokens, 0 when unknown.
    ///   - promptTokens: tokens the prompt already occupies, 0 when not yet
    ///     counted.
    ///   - floor: lower bound for a DERIVED budget only, so an exhausted context
    ///     still generates something rather than nothing.
    /// - Returns: a positive token ceiling.
    public static func resolve(
        requested: Int?,
        contextWindow: Int,
        promptTokens: Int = 0,
        floor: Int = 256
    ) -> Int {
        if let requested, requested > 0 { return requested }
        guard contextWindow > 0 else { return unknownContextFallback }
        let remaining = contextWindow - promptTokens - contextReserve
        return max(floor, remaining)
    }

    /// True when `requested` means "derive the limit" rather than naming one.
    public static func isDerived(_ requested: Int?) -> Bool {
        guard let requested else { return true }
        return requested <= 0
    }

    /// Parse a CLI `--max-tokens` value, returning nil when it is not valid.
    ///
    /// Accepts a positive integer, or any of the "derive it" spellings. `-1` is
    /// the Ollama-compatible form, but a bare `--max-tokens -1` cannot be parsed
    /// as a VALUE by ArgumentParser (a leading `-` reads as another flag), so it
    /// requires `--max-tokens=-1`. `auto` exists to give that same meaning a
    /// spelling that works in both forms.
    public static func parse(_ raw: String) -> Int? {
        let value = raw.trimmingCharacters(in: .whitespaces).lowercased()
        if ["auto", "-1", "unlimited", "model", "context"].contains(value) {
            return unlimited
        }
        guard let n = Int(value), n > 0 else { return nil }
        return n
    }

    /// The value shown when a flag rejects its input.
    public static let parseHelp =
        "a positive number of tokens, or 'auto' (equivalently --max-tokens=-1) "
        + "to derive it from the model's context window"
}
