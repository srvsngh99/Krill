# PR 4 Merge-Blocking Test Coverage Plan

This plan replaces the earlier contract-only coverage request with an implementation-grade checklist. The target is to protect correctness-sensitive changes in prefix caching, speculative decoding, server protocol handling, and model pulling.

> **Note (2026-05-11):** this document captures the original PR #4 coverage
> contract. It is preserved as historical context; subsequent PRs added more
> tests on top. The current verified suite size is `make test` → 128 tests,
> 9 skipped, 0 failures. See "Coverage added since PR #4" at the bottom of
> this file for the current rosters that the original plan does not enumerate.

## Standard

Tests should exercise production code paths whenever practical. Pure contract tests are acceptable only as a supplement, not as the main coverage for changed behavior.

Required properties:

- No real network access.
- Metal/MLX-dependent tests must skip cleanly when unavailable.
- Testability seams should be small, internal where possible, and preserve production behavior.
- Server tests should use NIO embedded channels or extracted internal helpers instead of duplicating handler logic in test files.
- Puller tests should mock transport, auth, retry, resume, and hashing behavior deterministically.

## Work Slices

### 1. KVCache and PrefixCache

Files:

- `Tests/KrillCoreTests/KVCacheTests.swift`
- `Tests/KrillCoreTests/PrefixCacheTests.swift`

Required tests:

- `KVCache.update` with actual `MLXArray` values returns expected shape and sequence length.
- Multiple `update` calls concatenate on axis 2.
- `snapshot` returns the current key/value array shapes.
- `restore` overwrites previous state and sequence length.
- `truncate(to:)` slices axis 2 and is a no-op above current length.
- `reset` clears state after an update.
- `update` after `restore` concatenates correctly.
- `PrefixCache.store` and exact lookup return real KV arrays with expected `prefixLength` and shapes.
- Full-hit replay invariant with real cache arrays: restore N, truncate N-1, update one token, final length is N.
- Cross-model cache isolation.

Non-goal for this PR:

- Partial-prefix cache acceleration. The implementation intentionally skips partial hits until cache-aware masks exist.

### 2. Speculative Decoding

Files:

- `Tests/KrillEngineTests/SpeculativeDecodingTests.swift`
- `Sources/KrillEngine/SpeculativeDecoder.swift` only for narrow internal seams.

Required tests:

- Real `KVCache` rollback/truncate behavior relevant to rejected draft tokens.
- Greedy `Sampler` with real logits returns argmax.
- First-token sampling uses real logits.
- Existing first-token-emission contract remains covered.
- Existing max-token one behavior remains covered.

Nice-to-have:

- A deterministic fake-model seam that drives `SpeculativeDecoder.step` end to end. If this is too invasive, keep it out of this PR and document the limitation.

### 3. Server Integration and Parsing

Files:

- `Sources/KrillServer/Server.swift`
- Optional `Sources/KrillServer/ServerParsing.swift`
- `Tests/KrillServerTests/ServerTests.swift`

Required tests:

- Structured message conversion preserves system/user/assistant/user roles.
- OpenAI `temperature`, `top_p`, and `top_k` parsing.
- Ollama `temperature`, `top_p`, and `top_k` parsing.
- Oversized body returns `413 Payload Too Large`.
- `/healthz` without a loaded model returns `200` JSON with `model_loaded=false`.

Nice-to-have:

- Ollama streaming sends response head before body. If the model-loaded guard blocks an embedded-channel test, extract a small internal response-head helper and test it directly.

### 4. Puller Hardening

Files:

- `Sources/KrillRegistry/Puller.swift`
- `Tests/KrillRegistryTests/PullerTests.swift`

Required tests:

- Auth header is attached from an injected token provider or `HF_TOKEN`.
- Retry behavior is deterministic and does not sleep in tests.
- Resume request sends the expected `Range` header when a `.partial` file exists.
- Incremental SHA256 matches known bytes.
- HTTP 206 append path combines existing partial data and new bytes.

Implementation guidance:

- Prefer an injectable `URLSession` plus injectable sleeper/token provider, or a small `HTTPClient` protocol if `URLSession.download(for:)` is hard to mock.
- Avoid real Hugging Face calls.

## Package Wiring

The main thread owns `Package.swift` dependency changes:

- `KrillCoreTests` gets `MLX`.
- `KrillEngineTests` gets `MLX`.
- New `KrillServerTests` gets `KrillServer`, `KrillEngine`, `KrillRegistry`, `KrillSampler`, `NIOCore`, `NIOEmbedded`, and `NIOHTTP1`.
- `KrillRegistryTests` gets `Crypto`.

## Verification

Required commands:

```bash
swift build
swift test
```

Expected outcome:

- All tests pass.
- MLX/Metal tests either run or skip cleanly.
- No network access is required.
- Newly added source seams are minimal and justified by tests.

---

## Coverage added since PR #4

Append-only roster of test files added by later PRs. Use this as the
shortlist when scoping a new follow-up.

### PR #5 (merge-blocking coverage for PR #4 fixes)

- `Tests/KrillRegistryTests/PullerTests.swift`
- `Tests/KrillServerTests/ServerTests.swift`
- Expanded `Tests/KrillCoreTests/PrefixCacheTests.swift`,
  `Tests/KrillCoreTests/KVCacheTests.swift`,
  `Tests/KrillEngineTests/SpeculativeDecodingTests.swift`.

### PR #9 (`feat/release-readiness-remediation`) and follow-ups

- `Tests/KrillServerTests/MultimodalEndpointsTests.swift` — server multimodal
  request/response shape coverage and `testTwoDifferentImagesProduceDifferentOutputs`.
- `Tests/KrillEngineTests/QuantizedKVCacheUnitTests.swift` — int8 cache shape
  and quantize/dequantize unit checks.
- `Tests/KrillEngineTests/QuantizedKVCacheIntegrationTests.swift`
  (env-gated) — int8 vs fp16 greedy parity through `InferenceEngine`.
- `Tests/KrillEngineTests/MultimodalPrefixCacheTests.swift` — verifies the
  prefix cache key includes media bytes (schema v2).

### PR #11 (`perf: compose int8 KV cache with prefix cache`)

- `Tests/KrillCoreTests/QuantizedPrefixCacheTests.swift` — 4 unit tests:
  uint8 round-trip, dtype isolation in both directions, restore+truncate+update.
- `Tests/KrillEngineTests/QuantizedPrefixCacheLiveTests.swift` (env-gated) —
  `testInt8PrefixCacheReplayMatchesColdRun` asserts cold and warm runs through
  `InferenceEngine` with `kvCacheDtype: "int8"` and a shared `PrefixCache`
  produce identical greedy tokens.

### PR #12 (`feat: release_candidate gate profile + kv_cache_dtype in reports`)

- `tools/test_release_gate.py` — 7 Python unit tests covering strict vs
  release_candidate profile dispatch, audio scope skipping, hard-metric
  regression detection, missing-hard-metric → fail, and `kv_cache_dtype`
  surfacing. Run with `python3 -m unittest tools.test_release_gate`.

### PR #14 (`feat: peak-memory sampling + release_candidate memory hard-gate`)

- `tools/test_memory_sampling.py` — 17 Python unit tests covering the
  cross-platform memory sampler against a live subprocess
  (`phys_footprint` from `proc_pid_rusage` on macOS, RSS from `ps`
  elsewhere), process-tree topology walking, per-PID footprint summing,
  `pgrep` parsing and self-exclusion, override handling, daemon-thread
  join on `__exit__`, the `_MemoryProbe` report-block shape, the
  platform-conditional basis, and the `RSSSampler → MemorySampler`
  back-compat alias. Run with `python3 -m unittest tools.test_memory_sampling`.
- `tools/test_release_gate.py` — extended with 6 new tests (now 13 total)
  for `memory_ratio` semantics: recorded-when-present, advisory under
  quant-class mismatch, hard pass/fail under quant-class equality, strict
  keeps memory hard regardless, and exclusion from the geometric-mean
  speedup headline.
- `.github/workflows/tools-tests.yml` — minimal CI matrix (ubuntu-latest +
  macos-latest, Python 3.13) that runs the two suites above on every PR
  that touches `tools/**`. The macOS leg exercises the `phys_footprint`
  sampler path; the Linux leg exercises the `ps rss` fallback. Swift
  `make test` stays local-only (Apple Silicon required for mlx-swift).

