# Follow-ups: Native-Path Bugs Blocking Agent Dogfood

Discovered 2026-05-27/28 while wiring KrillLM as the backend for an external
agent harness (Claude Code via `ANTHROPIC_BASE_URL` + `/v1/messages`). The
Anthropic shim, native tool calling, and SSE streaming all work end-to-end
on **Qwen 2.5 14B Instruct** (`.qwen` family, dense engine). Larger /
agent-tuned models fail at the loader or runtime layer.

Scope: **native Swift+MLX path only.** The Python `MoEEngine`/`PythonSidecar`
bridge is a dead end per project direction and is not patched here.

---

## 1. Gemma 4 non-e2b variants crash at first inference

**Symptom.** Loading any Gemma 4 variant other than e2b on `feat/agent-mode-tools`
crashes inside the attention reshape during the first forward pass.

- `mlx-community/gemma-4-e4b-it-4bit` →
  `[reshape] Cannot reshape array of size 548864 into shape (1,268,8,512)` (½× expected)
- `mlx-community/gemma-4-26b-a4b-it-4bit` →
  `[reshape] Cannot reshape array of size 2903040 into shape (1,2520,12,64)` (1.5× expected)

Different shape mismatches in different reshape sites → multiple shape
assumptions are tied to e2b's specific config.

**Likely root cause.** `Sources/KLMCore/Gemma4Model.swift:91`:

```swift
public func isFullAttention(layerIdx: Int) -> Bool {
    (layerIdx + 1) % slidingWindowPattern == 0
}
```

Per-layer `layerHeadDim` is then picked at `Gemma4Model.swift:133`:

```swift
self.layerHeadDim = isFullAttn ? config.globalHeadDim : config.headDim
```

For e2b (35 layers, head_dim=256, global_head_dim=512) the modulo pattern
matches the checkpoint. For e4b (42 layers, kv_heads=2) and 26B-A4B
(30 layers, heads=16, kv_heads=8, sliding_window=1024) the actual
per-layer attention type is published differently (likely as a
`layer_types` list in `config.json`), and `isFullAttention` returns the
wrong value, so the wrong head_dim is used → reshape mismatch in
Q/K/V projections at `Gemma4Model.swift:213, 228-229`.

**Suggested fix.**
1. Parse a `layer_types: [...]` (or equivalent) array from `config.json` per
   variant; fall back to the modulo pattern only when absent.
2. Add a per-variant smoke test under `Tests/KLMEngineTests/` that loads
   each Gemma 4 SKU (e2b, e4b, 26B-A4B, 31B) and asserts a single token
   generates. e2b is currently the only one covered.

**Also:** Gemma 4's tool-calling parity was verified on e2b only
(PR #23). Once larger variants load, re-run the tool-calling matrix.

---

## 2. AliasMap points at non-existent `gemma-4-12b` repo

**Symptom.** `krillm pull gemma-4-12b` → `HTTP 401: Failed to list repo files`.

**Root cause.** `Sources/KLMRegistry/AliasMap.swift:185-187`:

```swift
"gemma-4-12b": ResolvedModel(
    repo: "mlx-community/gemma-4-12b-it-4bit",
    ...
```

This repo does not exist. Google's Gemma 4 lineup is **E2B / E4B /
26B-A4B / 31B** — there is no 12B SKU. HF returns 401 (which it
uses uniformly for "not found or private") rather than 404, which made
this look like a license-gate at first.

**Suggested fix.** Remove the `gemma-4-12b` entry. Optionally add
`gemma-4-26b-a4b` and `gemma-4-31b` once §1 is fixed and they actually
load. The 31B at 4-bit (~16 GB) is too tight for 16/24 GB Macs; flag
appropriately in `Recommender`.

---

## 3. Puller's file allowlist drops files the Qwen 3 tokenizer needs

**Symptom.** After `krillm pull mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit`,
the local blob is missing `chat_template.jinja`, `added_tokens.json`,
`merges.txt`, `vocab.json`, and `model.safetensors.index.json`. Loading
the tokenizer fails with "chat_template is not set."

**Root cause.** `Sources/KLMRegistry/Puller.swift:84-94`:

```swift
let essentialFiles = fileList.filter { file in
    let name = file.name.lowercased()
    return name.hasSuffix(".safetensors")
        || name == "config.json"
        || name == "tokenizer.json"
        || name == "tokenizer_config.json"
        || name == "special_tokens_map.json"
        || name == "generation_config.json"
        || name == "tokenizer.model"
}
```

The allowlist predates the newer HF convention (mid-2025+) where the
chat template ships as `chat_template.jinja` alongside `tokenizer_config.json`
instead of being embedded in it. Newer Qwen 3 MoE checkpoints (Coder,
Instruct-2507) ship the separated form. The index file is also missing,
which the sharded weight loader needs for any multi-shard checkpoint.

**Suggested fix.** Extend the allowlist:

```swift
|| name == "chat_template.jinja"
|| name == "added_tokens.json"
|| name == "model.safetensors.index.json"
|| name == "merges.txt"            // BPE tokenizers without merged tokenizer.json
|| name == "vocab.json"            // ditto
```

Or invert the policy: pull everything except a small denylist
(`README.md`, `.gitattributes`, `*.png`, `*.md`, `original/`, sample
images). Inversion is more future-proof — every new HF convention so
far has added files, not removed them.

Add a tokenizer-load smoke test that asserts `chat_template` is present
(either inline or via the separate jinja file) after `krillm pull`.

---

## 4. Native MoE runtime ignores per-module quantization overrides

**Symptom.** With `KRILL_NATIVE_MOE=1`, loading Qwen3-Coder MoE crashes:

```
[quantized_matmul] The shapes of the weight and scales are incompatible
based on bits and group_size. w.shape() == (128,512) and
scales.shape() == (128,32) with group_size=64 and bits=4
```

For 4-bit at group_size=64, scales should be (128, 8). The (128, 32)
shape is 4× too large because **that layer is actually quantized at
8-bit, not 4-bit.**

**Root cause.** Qwen3-Coder's `config.json` ships per-module quant
overrides:

```json
"quantization": {
  "group_size": 64, "bits": 4,
  "model.layers.0.mlp.gate": { "group_size": 64, "bits": 8 },
  "model.layers.1.mlp.gate": { "group_size": 64, "bits": 8 },
  // ... 48 such entries, one per MoE gate
}
```

`Sources/KLMCore/Qwen3MoEModel.swift:45,117` reads this as a single
`QuantizationConfig` and applies the top-level `(bits=4, group_size=64)`
uniformly. Mixed-precision per-module overrides are not honored, so the
8-bit gate weights get instantiated as 4-bit `QuantizedLinear`, and the
scales tensor shape mismatches.

**Suggested fix.**
1. Extend `QuantizationConfig` (or add a parallel `ModuleQuantOverrides`
   struct) to capture per-module entries.
2. At expert/router/linear instantiation time in `Qwen3MoEModel.swift`,
   look up the module's full dotted name against the override map and
   use the per-module bits/group_size when present.
3. Treat this as a generic mechanism — Qwen3-Coder is the trigger but
   the same pattern (per-module overrides) is increasingly common across
   MLX-community quants. Wire it through `Gemma4Model` and `QwenModel`
   too while you're there.

Adding the override path also lifts the artificial `KRILL_NATIVE_MOE=1`
gate, since the next blocker (scatter-dispatch perf) is correctness-
orthogonal — the runtime can ship now and optimize dispatch later.

---

## 5. Remove the MoE Python sidecar entirely

Not a bug per se, but project direction: the `MoEEngine` /
`PythonSidecar` bridge in `Sources/KLMEngine/MoEEngine.swift` is the
last `~/.krillm/venv` consumer in the chat path. WS5 already retired
the Qwen 2.5-VL bridge for the same reason. Once §3 and §4 land:

- Drop `MoEEngine` to native-only (rename `mixtureOfExperts`
  ChatRouting → `denseEngine` or merge cases; the `.moe` family no
  longer needs a separate router enum value).
- Delete `tools/moe_bridge.py` and the `KRILLM_MOE_PYTHON` /
  `KRILLM_MOE_BRIDGE` env overrides.
- Update the "compatible_fallback tier" language in
  `ModelCapabilities.swift` to reflect that MoE is now first-class.
- Remove the "opt-in until scatter-dispatch lands" gate referenced in
  the error message at `KLMCli/ServeCommand.swift`.

---

## Verification path once §1–§4 land

1. `krillm pull mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit`
2. `krillm serve --model Qwen3-Coder-30B-A3B-Instruct-4bit`
3. Plain-text smoke against `/v1/messages` (Anthropic shape).
4. Tool-calling smoke with one `tools` entry — assert the response
   `content[0].type == "tool_use"` with parsed `input`.
5. SSE streaming smoke — assert event order `message_start`,
   `content_block_start`, `content_block_delta`, `content_block_stop`,
   `message_delta`, `message_stop`.
6. Drive an external Claude Code instance via `ANTHROPIC_BASE_URL` on
   a toy multi-step task (e.g. "create hello.txt, then read it back").

All four are wired and verified for Qwen 2.5 14B today; the work above
is to extend that working surface to the agent-grade SKUs (Qwen 3 MoE,
Gemma 4 26B-A4B / 31B).
