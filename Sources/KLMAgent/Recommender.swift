import Foundation
import KLMRegistry

/// One ranked candidate the recommender returns to the operator agent.
///
/// The `agent` then either presents the top-k to the user or picks
/// one via the router LLM with the fit annotation in context.
public struct RecommendedModel: Equatable, Sendable {
    /// The catalog entry being recommended.
    public let alias: String
    public let repo: String
    public let family: ModelFamily
    public let params: String
    public let context: Int
    /// Capability set declared for the family.
    public let capabilities: Set<Capability>
    /// Support tier for the family.
    public let supportTier: SupportTier
    /// Heuristic on-disk size in bytes (the recommender's estimate;
    /// see `Recommender.estimatedSize(params:quant:)`).
    public let estimatedSizeBytes: UInt64
    /// How comfortably this fits on the user's machine.
    public let fit: FitClassification
    /// Deterministic score the recommender used to rank candidates.
    /// Higher is better. Surfaced so the CLI/test can stable-sort and
    /// so callers can format a "why this ranked first" explanation.
    public let score: Double

    public init(
        alias: String, repo: String, family: ModelFamily,
        params: String, context: Int,
        capabilities: Set<Capability>, supportTier: SupportTier,
        estimatedSizeBytes: UInt64, fit: FitClassification, score: Double
    ) {
        self.alias = alias
        self.repo = repo
        self.family = family
        self.params = params
        self.context = context
        self.capabilities = capabilities
        self.supportTier = supportTier
        self.estimatedSizeBytes = estimatedSizeBytes
        self.fit = fit
        self.score = score
    }
}

/// Deterministic catalog-ranking pass used by the `recommend_model`
/// operator tool.
///
/// The flow is:
///   1. Filter the catalog down to entries whose declared
///      `capabilities(for:)` is a superset of the caller-requested
///      capability set (e.g. `[.audioInput]`).
///   2. On Intel Macs, optionally filter out families that depend on
///      Apple-Silicon-only kernels (Qwen 2.5-VL, Gemma 4 audio).
///   3. Compute a fit classification + a numeric score for each
///      surviving entry.
///   4. Return the top-N entries, sorted by score (then by alias for
///      stable ordering).
///
/// The recommender returns a *shortlist* - the operator agent's
/// router LLM picks the final one, with the shortlist in context so
/// the explanation can cite hardware fit, params, and support tier.
public enum Recommender {

    /// Heuristic on-disk size in bytes for a `params` / `quant` pair.
    ///
    /// The catalog entries don't carry `sizeBytes` (only installed
    /// `ModelManifest`s do), so the recommender uses an approximation
    /// from the parameter-count label:
    ///   bytes ≈ params × bytes_per_param + 200 MB overhead
    /// where bytes_per_param defaults to 0.55 at 4-bit (matches the
    /// observed mlx-community 4bit checkpoints to within ~10 %).
    public static func estimatedSize(params: String, quant: String) -> UInt64 {
        let lower = params.lowercased()
        let scale: Double
        if lower.hasSuffix("b") {
            scale = 1_000_000_000
        } else if lower.hasSuffix("m") {
            scale = 1_000_000
        } else {
            scale = 1_000_000_000
        }
        let numericPart = lower.dropLast()
        // MoE label shapes seen in the catalog:
        //   "8x7B"        Mixtral-style: experts × per-expert params
        //                 -> storage = experts * per-expert (the full
        //                 set lives on disk; the active subset is a
        //                 runtime concern).
        //   "30B-A3B"     Qwen3-MoE-style: total-params - active-params
        //   "1B-7B"       OLMoE-style: active-params - total-params
        //                 The on-disk footprint is the LARGER value
        //                 (total params), not whichever happens to be
        //                 written first; OLMoE and Qwen 3 disagree on
        //                 ordering. Without this, "1B-7B" sizes as 1 B
        //                 and a 4 GB checkpoint reads as 0.5 GB.
        var paramCount: Double = 0
        if numericPart.contains("x") {
            let parts = numericPart.split(separator: "x")
            if parts.count == 2,
               let a = Double(parts[0]), let b = Double(parts[1])
            {
                paramCount = a * b
            }
        } else if numericPart.contains("-") {
            // "30B-A3B" / "1B-7B" - strip a leading "a"/"A" on either
            // half ("A3B" => active 3 B) so the numbers parse, then
            // take the max.
            let parts = numericPart.split(separator: "-").map { part -> Double in
                var s = part
                if s.first == "a" { s = s.dropFirst() }
                if s.last == "b" || s.last == "m" { s = s.dropLast() }
                return Double(s) ?? 0
            }
            paramCount = parts.max() ?? 0
        } else {
            paramCount = Double(numericPart) ?? 0
        }
        let count = paramCount * scale

        let bytesPerParam: Double
        switch quant.lowercased() {
        case "fp32": bytesPerParam = 4.0
        case "fp16", "bf16": bytesPerParam = 2.0
        case "8bit", "int8": bytesPerParam = 1.05
        case "4bit": bytesPerParam = 0.55
        default: bytesPerParam = 0.55
        }
        let overhead: Double = 200 * 1024 * 1024
        return UInt64(count * bytesPerParam + overhead)
    }

    /// Build a recommendation shortlist for a capability requirement
    /// against the supplied catalog + hardware snapshot.
    ///
    /// - Parameters:
    ///   - required: capabilities the result must declare. Use
    ///     `[.textGeneration]` for "any chat model"; pass `.tools`,
    ///     `.visionInput`, `.audioInput`, etc. for specific asks.
    ///   - catalog: the candidate pool. Typically `[CatalogEntry]`
    ///     from the loaded remote catalog OR `AliasMap.allAliases`
    ///     converted into catalog entries.
    ///   - hardware: the user's machine.
    ///   - intelFilterOut: a set of families to drop entirely when
    ///     `hardware.arch == "x86_64"`. Defaults to the
    ///     Apple-Silicon-only families (Qwen 2.5-VL, Gemma 4) per the
    ///     strategic plan's §2.9 open question 6 - recommend filtering
    ///     them out of the suggestion, while still letting the user
    ///     `pull_model` them by explicit name (adult mode).
    ///   - limit: cap on the returned list. `0` means "all".
    public static func recommend(
        required: Set<Capability>,
        catalog: [CatalogEntry],
        hardware: HardwareInfo,
        intelFilterOut: Set<ModelFamily> = [.qwen25vl, .gemma4],
        limit: Int = 5
    ) -> [RecommendedModel] {
        var ranked: [RecommendedModel] = []
        for entry in catalog {
            let caps = ModelCapabilities.capabilities(for: entry.family)
            guard required.isSubset(of: caps) else { continue }
            if hardware.arch == "x86_64",
               intelFilterOut.contains(entry.family) { continue }

            let size = estimatedSize(params: entry.params, quant: entry.quant)
            let fit = hardware.classifyFit(modelSizeBytes: size)
            // Don't recommend models that won't fit at all - the
            // operator agent will still allow them via explicit
            // `pull_model`, but they have no business on a shortlist.
            if fit == .wontFit { continue }

            let tier = ModelCapabilities.supportTier(for: entry.family)
            let score = scoreFor(
                fit: fit, tier: tier,
                params: entry.params, quant: entry.quant)
            ranked.append(RecommendedModel(
                alias: entry.alias,
                repo: entry.repo,
                family: entry.family,
                params: entry.params,
                context: entry.context,
                capabilities: caps,
                supportTier: tier,
                estimatedSizeBytes: size,
                fit: fit,
                score: score))
        }
        ranked.sort { lhs, rhs in
            if lhs.score != rhs.score { return lhs.score > rhs.score }
            return lhs.alias < rhs.alias
        }
        if limit > 0, ranked.count > limit {
            ranked = Array(ranked.prefix(limit))
        }
        return ranked
    }

    /// Deterministic scoring used by `recommend`. Higher is better.
    ///
    /// Weights (§2.9 open question 3 - these match the recommended
    /// defaults and stay easy to tune later):
    ///   - support tier: production_native = +3, compatible_fallback
    ///     = +1, experimental = 0, unsupported = -10.
    ///   - fit: comfortable = +2, tight = +1, risky = -1.
    ///   - param-count nudge: prefer mid-size models on capable hardware
    ///     (params in [1.5, 14] B → +0.5; smaller still ok).
    internal static func scoreFor(
        fit: FitClassification,
        tier: SupportTier,
        params: String,
        quant: String
    ) -> Double {
        var s = 0.0
        switch tier {
        case .productionNative: s += 3
        case .compatibleFallback: s += 1
        case .experimental: s += 0
        case .unsupported: s -= 10
        }
        switch fit {
        case .comfortable: s += 2
        case .tight: s += 1
        case .risky: s -= 1
        case .wontFit: s -= 5
        }
        let lower = params.lowercased()
        if lower.hasSuffix("b"),
           let n = Double(lower.dropLast())
        {
            if n >= 1.5 && n <= 14 { s += 0.5 }
        }
        return s
    }
}

public extension CatalogEntry {
    /// Convenience adapter: convert the built-in `AliasMap.allAliases`
    /// into catalog-entry form so the recommender can rank them
    /// without a separate code path for "no remote catalog loaded yet".
    static func fromAliasMap() -> [CatalogEntry] {
        AliasMap.allAliases.map { (alias, resolved) in
            CatalogEntry(
                alias: alias,
                repo: resolved.repo,
                family: resolved.family,
                params: resolved.params,
                quant: resolved.quant,
                context: resolved.context)
        }
    }
}
