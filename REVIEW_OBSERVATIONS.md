# Krill Repo Observations

Date: 2026-05-06

> **Status (2026-05-11):** historical. Most P0 issues listed below have
> been addressed by subsequent PRs (e.g., real prefix-cache KV
> snapshot/restore, schema v2 media-hash keying, int8 KV + prefix-cache
> compose, release-gate semantics). Do not treat any item below as
> "current"; cross-reference with `OLLAMA_SPEEDUP_EXECUTION_PLAN.md` and
> `docs/RELEASE_READINESS_REMEDIATION.md` for what's still open.

This document captures review observations for follow-up implementation. The repo is a promising Swift/MLX local LLM CLI/server with a clean module split, but some advertised performance and compatibility features need correctness hardening before they should be treated as production-ready.

## Summary

Krill has a strong foundation:

- Clear SwiftPM module boundaries for core models, inference engine, cache, registry, server, CLI, tokenizer, sampler, and kernels.
- Sensible model-family abstraction through `LoadedModel`.
- Practical CLI/API surface with OpenAI and Ollama-style endpoints.
- Basic tests pass for config decoding and registry behavior.

Main risks:

- Prefix cache is currently a placeholder and can produce incorrect generations if it skips prefill without real KV restoration.
- Speculative decoding appears to mutate target KV cache before knowing which draft tokens are accepted.
- Server async/NIO concurrency needs cleanup.
- API compatibility and multimodal features are currently partial despite being exposed in public surfaces.

Verification performed:

- `swift test` passed: 13 XCTest cases, 0 failures.
- Build emitted Swift 6 concurrency warnings around `nonisolated(unsafe)` captures in engine/server code.

## P0: Correctness Issues

### 1. Prefix Cache Stores Empty KV State

Evidence:

- `Sources/KrillEngine/InferenceEngine.swift:192-203` stores prefix cache entries with `keys: []` and `values: []`.
- `Sources/KrillCache/KVCache.swift:8-9` keeps `_keys` and `_values` private and exposes no snapshot API.
- `Sources/KrillEngine/InferenceEngine.swift:158-173` treats any prefix-cache hit as valid and sets `prefillStartIdx = hit.prefixLength`.

Impact:

On a later cache hit, the engine may skip part or all of prefill without restoring real KV tensors. That can produce invalid logits/generation. This is a correctness issue, not just a performance gap.

Recommendation:

- Short-term: disable prefix cache by default or prevent storing entries until KV snapshots are real.
- Proper fix:
  - Add `KVCache.snapshot() -> (keys: MLXArray, values: MLXArray)?`.
  - Add a safe restore initializer or `restore(keys:values:)`.
  - Store per-layer KV arrays after prefill.
  - Add tests that prove cached and uncached generation produce identical next-token logits for a toy model or deterministic fixture.

### 2. Speculative Decoding Mutates Target Cache Before Acceptance

Evidence:

- `Sources/KrillEngine/SpeculativeDecoder.swift:84-86` verifies draft tokens by forwarding through `targetCaches`.
- `Sources/KrillEngine/SpeculativeDecoder.swift:92-105` may reject partway through the proposed draft sequence.
- There is no rollback/truncation of `targetCaches` after rejection.

Impact:

If a draft token is rejected, target caches may already include KV state for unaccepted draft tokens. Subsequent decoding can run from an invalid sequence state.

Recommendation:

- Implement rollback/truncation in `KVCache`, or verify on copied caches and commit only accepted tokens.
- Add deterministic tests around rejection cases.
- Consider disabling speculative decoding in user-facing paths until cache correctness is proven.

### 3. Potential Empty Prompt Crash

Evidence:

- `Sources/KrillEngine/InferenceEngine.swift:180-183` force unwraps `promptTokens.last!` on full cache hit.

Impact:

If tokenization produces an empty token list, this can crash. It may be rare because chat templates often include special tokens, but the engine should not depend on that.

Recommendation:

- Validate `promptTokens` is non-empty before generation.
- Return an error stream/event or use BOS fallback explicitly.

## P1: Server and API Robustness

### 4. NIO Writes Occur From Swift Tasks Using Captured Context

Evidence:

- `Sources/KrillServer/Server.swift:214-245`
- `Sources/KrillServer/Server.swift:254-285`
- `Sources/KrillServer/Server.swift:301-315`
- Several captures use `nonisolated(unsafe)`.

Impact:

`ChannelHandlerContext` should be used on its event loop. Writing from arbitrary Swift tasks risks race conditions and undefined behavior under concurrency. Swift 6 also emits warnings here.

Recommendation:

- Route all writes through `context.eventLoop.execute`.
- Prefer a small helper that serializes response writes back onto the channel event loop.
- Remove unnecessary `nonisolated(unsafe)` after the write path is fixed.

### 5. Ollama Streaming Response Does Not Send Headers

Evidence:

- `Sources/KrillServer/Server.swift:422-436` writes body chunks for `/api/chat` streaming but does not send an HTTP response head first.

Impact:

Clients may receive malformed responses or fail depending on NIO behavior and client strictness.

Recommendation:

- Send a `200 OK` response head with content type before streaming newline-delimited JSON.
- Add an integration test using a real HTTP client.

### 6. Request Body Size Is Unbounded

Evidence:

- `Sources/KrillServer/Server.swift:83-94` accumulates body chunks into `body` without a maximum.

Impact:

Large requests can cause unbounded memory growth.

Recommendation:

- Enforce a maximum request body size.
- Return `413 Payload Too Large` when exceeded.

### 7. API Compatibility Is Partial

Evidence:

- `Sources/KrillServer/Server.swift:173-188` extracts only the last user message and last system message.
- Sampling only maps `temperature`; fields like `top_p`, `top_k`, `stop`, `seed`, penalties, and model selection are ignored.

Impact:

OpenAI/Ollama clients expecting multi-turn history, stops, tool calls, or model routing will get surprising behavior.

Recommendation:

- Preserve full message history in `KrillTokenizer.applyChatTemplate`.
- Parse and apply common generation parameters.
- Return clear errors for unsupported features instead of silently ignoring them.

## P1: Model Pulling and Registry

### 8. Puller Lacks Resume, Retry, Auth, and Revision Pinning

Evidence:

- `Sources/KrillRegistry/Puller.swift:50-92` downloads each selected file sequentially.
- `Sources/KrillRegistry/Puller.swift:178-195` downloads to a temp file, moves it, then reads the full file into memory to hash it.

Impact:

Large model downloads are brittle. Gated/private Hugging Face repos are unsupported. Full-file hashing can spike memory.

Recommendation:

- Support `HF_TOKEN` or config-based token auth.
- Add retry with exponential backoff.
- Support revisions/commits and persist revision in manifest.
- Hash incrementally while streaming or from a file handle.
- Download into `.partial` files and resume where possible.

### 9. Registry Layout Mentions Content-Addressed Blobs But Uses Model Names

Evidence:

- `Sources/KrillRegistry/Registry.swift` docs describe content-addressed blobs.
- `Registry.modelPath(_:)` returns `blobs/<name>`.

Impact:

The implementation is fine for now, but docs imply deduplication/content addressing that does not exist.

Recommendation:

- Either update comments/docs to describe current name-based layout or implement content-addressed storage.

## P2: Multimodal and Tooling Surface

### 10. Vision and Audio Preprocessing Are Placeholders

Evidence:

- `Sources/KrillCore/VisionEncoder.swift:218-228` returns a zero image tensor.
- `Sources/KrillCore/AudioEncoder.swift:224-228` returns a zero mel spectrogram.

Impact:

Image/audio flags and multimodal modules can appear functional while producing meaningless inputs.

Recommendation:

- Mark these as experimental/unsupported in CLI and README until implemented.
- Implement real image decode/resize/normalize using CoreGraphics or a dedicated dependency.
- Implement real STFT/mel using Accelerate/vDSP.
- Add golden-shape and non-zero preprocessing tests.

### 11. CLI Exposes Unused Tool/Image/Audio Options

Evidence:

- `Sources/KrillCLI/RunCommand.swift:35-42` defines `--image`, `--audio`, and `--tools`.
- Native generation path does not use `tools`; image/audio only route through the Gemma 4 Python fallback path.

Impact:

Users can pass options that are silently ignored in common paths.

Recommendation:

- Reject unsupported combinations clearly.
- Wire tool definitions into prompts where supported.
- Document support matrix per model family.

## P2: Build, Packaging, and Docs

### 12. Swift Version Docs Do Not Match Manifest

Evidence:

- `Package.swift:1` requires Swift tools 6.2.
- `README.md` lists Swift 6.0+.

Recommendation:

- Update README prerequisites to Swift 6.2+ or lower the tools version if 6.0 compatibility is intended.

### 13. `Package.resolved` Is Ignored

Evidence:

- `.gitignore:5` ignores `Package.resolved`.

Impact:

For an executable application, committing `Package.resolved` usually improves reproducibility.

Recommendation:

- Commit `Package.resolved` unless there is a deliberate reason not to.

### 14. Release Tarball Is Tracked In Source

Evidence:

- `dist/krill-0.2.0-arm64-apple-macos.tar.gz` is tracked.

Impact:

Binary artifacts increase repo size and are better handled as release assets.

Recommendation:

- Move tarballs to GitHub Releases.
- Add `dist/` to `.gitignore` if local builds should not be committed.

### 15. README Performance Claims Need Reproducibility Details

Evidence:

- `README.md:7-16` includes performance and memory claims.

Impact:

The claims are compelling, but reviewers need enough context to reproduce them.

Recommendation:

- Add exact model repo/revision, prompt, token counts, command lines, hardware, macOS/Xcode/Swift versions, and benchmark methodology.
- Store benchmark output under a docs or benchmarks directory.

## Testing Gaps

Current tests cover:

- Llama config decoding.
- Basic KV cache empty state.
- Alias resolution.
- Registry save/load/remove.

Recommended new tests:

- Prefix cache equivalence: cached vs uncached next-token logits.
- KV cache snapshot/restore/truncate behavior.
- Speculative decoding rejection path.
- Sampler top-k/top-p behavior with deterministic logits.
- Chat-template formatting for Llama/Qwen/Gemma.
- Server request parsing, streaming headers, `/api/chat`, `/v1/chat/completions`, `/v1/models/load`.
- Puller error handling via mocked URL loading.
- CLI parser tests for unsupported option combinations.

## Suggested Implementation Order

1. Disable or complete prefix cache.
2. Disable or fix speculative decoding cache commit/rollback.
3. Fix NIO event-loop write discipline and streaming headers.
4. Add body size limits and API parameter validation.
5. Tighten README/CLI claims for unsupported multimodal/tool features.
6. Harden puller downloads.
7. Expand test coverage around the fixed areas.

