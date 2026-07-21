# Testing

## Regular CI

`swift-tests.yml` builds the release test bundle, compiles the packaged Metal
library, runs the deterministic Swift suite, publishes SwiftPM's coverage JSON
and a coverage summary, and performs dependency-free repository/release checks.
`tools-tests.yml` runs the Python helper tests on Linux and macOS. Large model
tests use `XCTSkip` unless their documented environment path is present.

Local equivalents:

```sh
make test
python3 -m unittest tools.test_release_gate tools.test_memory_sampling tools.test_installer -v
python3 tools/check_release_consistency.py
```

## Live Apple-Silicon lane

`live-model-tests.yml` runs weekly and on manual dispatch. It pulls the public
`qwen2.5-0.5b` 4-bit checkpoint on the ARM64 `macos-26` image and runs a focused set of
real-weight batching, cross-row isolation, shared-prefix, and prompt-lookup
correctness gates. The workflow is self-contained; its model cache only reduces
repeat download time.

To reproduce that lane with any supported plain-causal checkpoint:

```sh
model_path="$HOME/.krill/models/blobs/qwen2.5-0.5b"
KRILL_BATCH_MODEL_PATH="$model_path" \
KRILL_TEXT_MODEL_PATH="$model_path" \
KRILL_NGRAM_MODEL_PATH="$model_path" \
  swift test -c release --filter 'BatchedDecodeLiveTests|PrefixCachePartialReuseLiveTests|NgramLiveParityTests'
```

## Larger opt-in matrix

These lanes are intentionally not downloaded by pull-request CI. Run them on
an Apple-Silicon machine with the listed checkpoint or parity bundle already
available; each test file documents any additional fixture/reference variable.

| Area | Primary environment path | Representative suites |
|---|---|---|
| Gemma 4 text/image/audio, int8 KV | `KRILL_GEMMA4_MODEL_PATH` | `Gemma4SmokeTests`, `NativeAudioRoutingTests`, `Gemma4PartialReuseLiveTests`, `QuantizedPrefixCacheLiveTests` |
| Gemma 4 unified/long context | `KRILL_GEMMA4_UNIFIED_MODEL_PATH` | `Gemma4ChunkedPrefillTests`, `Gemma4DecodeSweepTests` |
| Qwen2.5-VL | `KRILL_QWEN25VL_MODEL_PATH` | `Qwen25VLSmokeTests`, `Qwen25VLProfileTests` |
| Qwen3.5-VL / Ornith | `KRILL_ORNITH_MODEL_PATH` | `Qwen35VLSmokeTests`, `Qwen35RealCheckpointTests` |
| LLaVA | `KRILL_LLAVA_MODEL_PATH` | `LlavaPromptTokenizationTests` |
| Llama 3.2 Vision parity | `KRILL_MLLAMA_PARITY_DIR` | `MllamaRuntimeTests` |
| Reranker | `KRILL_RERANKER_MODEL_PATH` | `RerankEngineTests` |
| Whisper | `KRILL_WHISPER_DIR`, `KRILL_WHISPER_ML_DIR` | encoder/decoder parity and end-to-end transcription tests |

Performance and deliberate OOM probes have a second explicit switch such as
`KRILL_NGRAM_PERF`, `KRILL_RUN_DECODE_SWEEP`, or `KRILL_RUN_OOM_BASELINE=1`.
They must remain opt-in and isolated from correctness CI.
