import XCTest
import KrillTokenizer

/// Prompt-tokenization gate for Muse Glimmer — TOKENIZER ONLY, no weights.
///
/// Why this exists: the native runtime is logit-parity-verified against mlx-vlm
/// on the real 30B checkpoint (argmax + cosine > 0.9999 on both prefill and
/// cached decode), yet `krill run` on that same checkpoint emits multilingual
/// token salad. Identical weights and identical math leave exactly one
/// suspect — the prompt the engine feeds in.
///
/// Muse Glimmer's tokenizer is a ~202K o200k-style BPE, the same family as
/// Phi-4-mini's, and `InferenceEngine`'s `.phiRenderReencode` case documents
/// precisely this failure for that family: swift-transformers'
/// `applyChatTemplateTokens` produces ids that *decode back to the right text*
/// but use non-canonical token boundaries, so the model sees an off-distribution
/// prompt and "degenerates into fluent garbage". Token COUNT can match while the
/// ids differ, which is why comparing counts (as an earlier check did) is not
/// enough — this compares the ids themselves.
///
/// Gated on `KRILL_MUSE_GLIMMER_MODEL_DIR` (a real checkpoint directory, for the
/// tokenizer files only) and `KRILL_MUSE_GLIMMER_PROMPT_REF` (a JSON written by
/// the transformers reference: `{"prompt": ..., "ids": [...]}`). Skipped when
/// either is unset — this needs a real tokenizer, but NOT the 18 GB of weights,
/// so it runs in seconds and costs no memory.
final class MuseGlimmerPromptTests: XCTestCase {

    private struct PromptRef: Decodable {
        let prompt: String
        let rendered: String
        let ids: [Int]
    }

    private func load() async throws -> (KrillTokenizer, PromptRef) {
        let env = ProcessInfo.processInfo.environment
        guard let dir = env["KRILL_MUSE_GLIMMER_MODEL_DIR"],
              let refPath = env["KRILL_MUSE_GLIMMER_PROMPT_REF"] else {
            throw XCTSkip("Set KRILL_MUSE_GLIMMER_MODEL_DIR and "
                          + "KRILL_MUSE_GLIMMER_PROMPT_REF")
        }
        let ref = try JSONDecoder().decode(
            PromptRef.self, from: Data(contentsOf: URL(fileURLWithPath: refPath)))
        let tok = try await KrillTokenizer(from: URL(fileURLWithPath: dir))
        return (tok, ref)
    }

    /// True when `got` and `want` differ ONLY where the chat template stamps the
    /// current date.
    ///
    /// The template renders `Current date: <today>` via `strftime_now`, so the
    /// canonical ids captured on one day stop matching the next — the reference
    /// for 2026-08-11 diverges from a 2026-08-12 run at exactly one index
    /// (`825` -> `738`, i.e. "11" -> "12"). Re-capturing the reference daily is
    /// not a gate, and letting it fail daily trains everyone to ignore it. So
    /// permit a mismatch only where BOTH sides decode to a day-of-month, and
    /// keep every other position strict — a real tokenization regression still
    /// fails, because it would land on a token that is not a small integer or
    /// would change the id COUNT.
    private func differsOnlyByDate(
        _ got: [Int], _ want: [Int], _ tok: KrillTokenizer
    ) -> Bool {
        guard got.count == want.count else { return false }
        var sawDateDiff = false
        for (g, w) in zip(got, want) where g != w {
            let gs = tok.decode(token: g).trimmingCharacters(in: .whitespaces)
            let ws = tok.decode(token: w).trimmingCharacters(in: .whitespaces)
            guard let gi = Int(gs), let wi = Int(ws),
                  (1 ... 31).contains(gi), (1 ... 31).contains(wi) else {
                return false
            }
            sawDateDiff = true
        }
        if sawDateDiff {
            print("note: ids differ only at the template's `Current date` stamp "
                  + "(reference captured on a different day) — treated as equal.")
        }
        return true
    }

    private func describe(_ label: String, _ got: [Int], _ want: [Int]) -> String {
        var s = "\(label): \(got.count) ids (reference \(want.count))"
        if got != want {
            let firstDiff = zip(got, want).enumerated().first(where: { $0.element.0 != $0.element.1 })?.offset
                ?? min(got.count, want.count)
            s += "\n    first divergence at index \(firstDiff)"
            let lo = max(0, firstDiff - 3)
            s += "\n    got   [\(lo)...]: \(Array(got.dropFirst(lo).prefix(8)))"
            s += "\n    want  [\(lo)...]: \(Array(want.dropFirst(lo).prefix(8)))"
        }
        return s
    }

    /// The image path splices a `<|patch|>` run into the user turn and relies on
    /// the chat template round-tripping it as ONE id per placeholder. If the
    /// tokenizer ever splits `<|patch|>` into pieces, the vision splice would
    /// see the wrong span and the model would assert deep in the forward — so
    /// pin the assumption here, where no weights are needed to check it.
    func testPatchPlaceholderRoundTripsAsSingleIds() async throws {
        let (tok, ref) = try await load()
        let patchId = 200_092  // config.image_token_id / added token `<|patch|>`

        for count in [1, 4, 64] {
            let run = String(repeating: "<|patch|>", count: count)
            let messages = [["role": "user", "content": run + ref.prompt]]
            let ids = try XCTUnwrap(tok.applyChatTemplateTokens(messages: messages))
            XCTAssertEqual(
                ids.filter { $0 == patchId }.count, count,
                "\(count) `<|patch|>` placeholders must survive as \(count) ids")
            // And they must be CONTIGUOUS: the splice replaces one span.
            let first = try XCTUnwrap(ids.firstIndex(of: patchId))
            let span = ids[first...].prefix { $0 == patchId }.count
            XCTAssertEqual(span, count, "the placeholder run must be contiguous")
        }
    }

    /// The two candidate policies, measured against the reference ids. This test
    /// is diagnostic: it always reports both, and only fails if NEITHER matches
    /// (which would mean the problem is not the policy choice at all).
    func testWhichPromptPolicyMatchesTheReference() async throws {
        let (tok, ref) = try await load()
        let messages = [["role": "user", "content": ref.prompt]]

        let direct = tok.applyChatTemplateTokens(messages: messages)
        let rendered = tok.applyChatTemplate(messages: messages)
        let reencoded = tok.encodeWithoutExtraBOS(rendered)

        print("=== muse_glimmer prompt tokenization ===")
        print(describe("direct (.directTokenIdsWithRenderFallback)", direct ?? [], ref.ids))
        print(describe("render+reencode (.phiRenderReencode)", reencoded, ref.ids))

        let directMatches = differsOnlyByDate(direct ?? [], ref.ids, tok)
        let reencodeMatches = differsOnlyByDate(reencoded, ref.ids, tok)
        print("direct matches: \(directMatches)   render+reencode matches: \(reencodeMatches)")

        XCTAssertTrue(directMatches || reencodeMatches,
            "neither prompt policy reproduces the reference ids — the defect is "
            + "not the policy choice; compare the RENDERED text next")
    }

    /// The actual gate: once the right policy is wired into `ModelAdapter`, the
    /// engine's chosen path must reproduce the reference ids exactly.
    func testConfiguredPolicyReproducesReferenceIds() async throws {
        let (tok, ref) = try await load()
        let messages = [["role": "user", "content": ref.prompt]]
        // Mirror what `ModelAdapter.tokenizerPrompt` currently selects for this
        // family. Update this line when the policy changes, so the test tracks
        // the engine rather than drifting from it.
        let configured = tok.applyChatTemplateTokens(messages: messages)
            ?? tok.encodeWithoutExtraBOS(tok.applyChatTemplate(messages: messages))
        XCTAssertTrue(differsOnlyByDate(configured, ref.ids, tok),
            "the policy the engine uses must reproduce the canonical prompt ids "
            + "(modulo the template's date stamp); same text with different token "
            + "boundaries is off-distribution and generates fluent garbage\n"
            + describe("configured", configured, ref.ids))
    }

    /// Guards the weaker check that misled an earlier debugging pass: a matching
    /// token COUNT does not imply matching ids.
    func testCountMatchIsNotSufficient() async throws {
        let (tok, ref) = try await load()
        let messages = [["role": "user", "content": ref.prompt]]
        guard let direct = tok.applyChatTemplateTokens(messages: messages) else {
            throw XCTSkip("no direct template path on this tokenizer")
        }
        if direct.count == ref.ids.count && !differsOnlyByDate(direct, ref.ids, tok) {
            XCTFail("token counts match (\(direct.count)) but ids differ — this is "
                    + "exactly the trap: count parity proves nothing")
        }
    }
}
