# Native Tool-Calling Implementation Plan (Gemma 4 / Llama / Qwen / Mistral / Phi)

Created: 2026-05-20
Status: WS1-5 DONE + multi-family validated. Native tool calling for
Gemma 4, Llama 3.x, Qwen 2.5/3, **Mistral, and Phi**, default-on, each
benchmarked at parity with Ollama on a properly-sized model.
Owner: unassigned

**Mistral / Phi adapters added 2026-06-01.** Both mirror the native wire
format (verified byte-exact against Ollama's template) and are live-benchmarked
on the real checkpoints:

| Family | Native format | Krill | Ollama | Verdict |
|--|--|--|--|--|
| Mistral | `[AVAILABLE_TOOLS][…][/AVAILABLE_TOOLS]` / `[TOOL_CALLS][…]` / `[TOOL_RESULTS]` | 4/4 valid+exact | 4/4 | PASS |
| Phi-4-mini | `<\|tool\|>[…]<\|/tool\|>` defs / `<\|tool_call\|>[…]<\|/tool_call\|>` | 1/4 | 1/4 | PASS¹ |

¹ phi-4-mini is a weak tool-caller on **both** engines (it declines T1/T2/T4
and emits a tool call only rarely); identical model behaviour, parity holds -
the same pattern as the Llama-T3 note above. The Krill adapter parses the
native `<|tool_call|>` array correctly when the model does emit one.

**Phi-4-mini base-runtime fixes (2026-06-01, prerequisite for the above).**
The Phi runtime was written for Phi-3-mini and produced pure garbage on
phi-4-mini; five distinct bugs were fixed in `Sources/KLMCore/PhiModel.swift`
(+ loader/engine): (1) `partial_rotary_factor` 0.75 was ignored (full RoPE on a
partial-rotary head); (2) `tie_word_embeddings` was ignored (random `lm_head`);
(3) the checkpoint's **fused `qkv_proj`** was silently dropped because the model
declared separate `q/k/v_proj` (the dominant cause); (4) LongRoPE ("su") scaling
was absent; (5) the `<|end|>` chat terminator (declared only in `config.json`'s
`eos_token_id`, no `generation_config.json`) was not a stop token, so the
assistant ran on. After the fixes phi-4-mini answers correctly and halts.

The hard Krill>=Ollama tool-quality gate is `tools/tool_calling_benchmark.py`
(per family, temp 0); the server's `parity_gate.py` `T0-4` cell now additionally
asserts any returned `tool_calls` are structurally well-formed.

**Multi-family parity (2026-05-20, `tools/tool_calling_benchmark.py`,
temp 0; Krill vs Ollama, same model class):**

| Family | Models | Krill | Ollama | Verdict |
|--|--|--|--|--|
| Gemma 4 | gemma-4-e2b / gemma4:e2b | 4/4 | 4/4 | PASS |
| Llama 3.x | llama-3.2-3b / llama3.2:3b | 3/4 | 3/4 | PASS¹ |
| Qwen 2.5 | qwen2.5-3b / qwen2.5:3b | 4/4 | 4/4 | PASS |

¹ Llama T3 ("no-tool") fails on **both** engines - the 3B hallucinates
a `get_weather` call for "what colour is the sky"; identical model
behaviour, parity holds. (On llama-3.2-**1b** both native and Hermes
are a wash - 1B models are poor tool-callers; the 3B is the valid
target.)

**Architecture (the "solve for all models" answer):** tool calling is
irreducibly model-specific at the wire level, so `ToolCalling` is a
model-agnostic dispatcher (`ToolFormat` resolved from
`engine.family`) over per-family adapters:
- `.gemma4` - hand-built native (no shipped chat_template); ids 46-51.
- `.llama` - mirrors Ollama's exact `llama3.2` template (system
  guidance + tool block in the **last** user turn + compact tool JSON);
  parses bare `{"name","parameters"}`, rejects echoed schemas.
- `.qwen` - Qwen's native format already *is* Hermes
  `<tool_call>{"name","arguments"}</tool_call>`; reuses the Hermes
  injection + a leading-junk-tolerant extractor.
- `.mistral` - native `[AVAILABLE_TOOLS]`/`[TOOL_CALLS]`/`[TOOL_RESULTS]`
  (token ids 5-9); tool schemas spliced before the last user turn,
  calls parsed from the `[TOOL_CALLS]` JSON array.
- `.phi` - native `<|tool|>` defs baked into the system turn,
  `<|tool_call|>[…]<|/tool_call|>` calls parsed from the array.
- `.hermes` - generic fallback for GLM / Gemma(-2) / unknown (unchanged,
  byte-identical, regression-free).

The vendored swift-transformers accepts but **ignores** `tools` in
Jinja ("not supported yet"), which is *why* per-family hand adapters
(not the model's own template) are required.

---

**Original Gemma 4 before/after (the reported finding):**

| | T1 single | T2 select | T3 no-tool | T4 agentic | TOTAL |
|--|--|--|--|--|--|
| Ollama (before/after) | ✓ | ✓ | ✓ | ✓ | 4/4 |
| Krill **before** | FAIL | FAIL | ✓ | FAIL | **1/4** |
| Krill **after** | ✓ | ✓ | ✓ | ✓ | **4/4** |

GATE: PASS (Krill ≥ Ollama on valid + args_exact). Artifacts
(gitignored, regenerate via §5):
`.build/benchmarks/tool-calling-baseline-2026-05-20.json` (red),
`.build/benchmarks/tool-calling-nativefix-2026-05-20.json` (green).

**Empirical note (drove the parser design):** Gemma 4's
`x-parser: gemma4-tool-call` arg blob is NOT strict JSON. The 2B
checkpoint emits all of: `call:add{a:12, "b":30}` (bare key),
`call: "multiply", "parameters": {…}` (quoted name + label),
`call: "add": {…}`. `extractGemma4` + `normalizeGemma4Args` are
deliberately tolerant of every observed shape, mirroring Ollama's
lenient native parser.

This is the feature companion to `OLLAMA_MAC_PARITY_PLAN.md` and shares
its hard/advisory/OOS gate philosophy. It targets the
[finding] that Krill's agentic / function-calling output is
unreliable on the **same** Gemma 4 E2B weights that Ollama drives
correctly.

## 1. Problem

Krill's tool plumbing is mechanically complete - the OpenAI `tools`
param, the Ollama `tools` param, the Anthropic `tools` param, a
`tool_calls` response in all three shapes, and a tolerant
`ToolCalling.extractToolCalls` parser. The plumbing is **not** the
problem.

Controlled test (single explicit `add(a,b)` tool, temp 0, identical
Gemma 4 E2B weights):

| Engine | Output |
|--------|--------|
| Ollama `gemma4:e2b`  | valid `{"name":"add","arguments":{"a":12,"b":30}}` |
| Krill `gemma-4-e2b` | malformed `{"name":"add","arguments":{"a":{"a":12,"b":30}}}`, `finish_reason=stop`, nothing extracted |

A multi-step agentic loop produces fully garbled output. Plain-QA
accuracy is fine, so this is **not** a weights or decode-quality
problem.

## 2. Root cause (confirmed)

`Sources/KLMServer/ToolCalling.swift` hand-rolls a generic
Hermes/Qwen-style instruction (`injectToolSystem` ->
`toolSystemPrompt`) that asks the model to emit:

```
<tool_call>{"name": "<tool-name>", "arguments": {<values>}}</tool_call>
```

Gemma 4 was **never fine-tuned on that convention**. The Krill-side
checkpoint (`~/.krill/models/blobs/gemma-4-e2b/tokenizer_config.json`)
ships *no* `chat_template`, so `TokenizerWrapper.applyChatTemplate`
falls through to `formatGemma4TokenIds` - a hand-rolled turn-token
format with **zero** tool awareness. The `tools` array never reaches
the renderer; the model only ever sees the foreign Hermes prompt and,
unsurprisingly, double-nests the arguments.

The same `tokenizer_config.json` defines the **native** format
explicitly via its special tokens and `response_schema`:

| Token | Value | Role |
|-------|-------|------|
| `std` / `etd` | `<\|tool>` … `<tool\|>` | tool **definitions** block |
| `stc` / `etc` | `<\|tool_call>` … `<tool_call\|>` | model-emitted tool **call** |
| `str` / `etr` | `<\|tool_response>` … `<tool_response\|>` | tool **result** fed back |
| `sot` / `eot` | `<\|turn>` … `<turn\|>` | turn markers (ids 105/106) |
| `soc` / `eoc` | `<\|channel>` … `<channel\|>` | `thought` channel |

`response_schema` pins the call grammar exactly:

```
tool_calls : x-regex-iterator  <\|tool_call>(.*?)<tool_call\|>
function   : x-regex           call\:(?P<name>\w+)(?P<arguments>\{.*\})
arguments  : x-parser          gemma4-tool-call
thinking   : <\|channel>thought\n(?P<thinking>.*?)<channel\|>
```

So the model is trained to emit, literally:

```
<|tool_call>call:add{"a": 12, "b": 30}<tool_call|>
```

not a JSON object with `name`/`arguments` keys. Ollama's gemma4 path
uses this embedded renderer + `response_schema`; Krill ignores both.

## 3. Workstreams

Gate philosophy: every WS ships unit-tested behind a feature flag,
flips default only after the §5 benchmark goes green vs Ollama on the
M4 target, mirroring the audio plan's WS6.

### WS1 - Native renderer (request side)

`TokenizerWrapper` / `formatGemma4TokenIds` (and a new
`Gemma4ToolTemplate`): when `tools` are present, render the definition
block in `<|tool> … <tool|>` and feed prior `tool` role messages back
as `<|tool_response> … <tool_response|>`. Carry `tools` through
`InferenceEngine.generate(messages:)` (currently dropped - see
`Sources/KLMEngine/InferenceEngine.swift:260`). Verify byte-exact
against Ollama's rendered prompt via `/api/generate` `raw`/template
dump for the same tools+messages.

### WS2 - Native parser (response side)

Add a Gemma 4 branch to `ToolCalling.extractToolCalls` (or a family
dispatch) implementing the `response_schema` grammar: iterate
`<|tool_call> … <tool_call|>`, parse `call:<name>{<json>}` via the
`gemma4-tool-call` arguments parser, split the `<|channel>thought`
segment into `thinking`. Keep the legacy Hermes path as a non-Gemma
fallback. The three response shapers (OpenAI/Ollama/Anthropic) are
unchanged downstream of `ParsedToolCall`.

### WS3 - Family dispatch

`ToolCalling` becomes model-aware (Gemma 4 native; Llama 3.x / Qwen
keep the existing generic prompt until each gets its own native
format). Selection keyed off the loaded model family, not a string
match. Llama/Qwen native formats are tracked follow-ups, not WS1-2
blockers.

### WS4 - Wire to all entrypoints

`handleToolChat` (OpenAI `Server.swift:542`, Ollama `:1193`) and
`handleAnthropicMessages` (`:582`) all route through the family-aware
renderer/parser. No endpoint-shape changes.

### WS5 - Tests + benchmark gate

Swift unit tests for render + parse (golden strings from
`response_schema`). Live E2E gated on `KLM_GEMMA4_MODEL_PATH`. The §5
benchmark must go green vs Ollama before WS3 flips the Gemma 4 default
on. Then add a hard `tool_call` cell to the parity gate.

## 4. Feature flag & default flip

Implement behind `KRILL_NATIVE_TOOLS=1` (off by default). Flip default
on - and remove the Hermes path for Gemma 4 - only after WS5 is green
on the M4 target, exactly as native audio did in WS6.

## 5. Benchmarking against Ollama

`tools/tool_calling_benchmark.py` (landed) is the reference harness.
Same skip-gate (exit 77 on missing prereqs) and report-artifact
discipline as `tools/krill_vs_ollama_benchmark.py`.

It runs both engines on the **same** Gemma 4 E2B weights
(`gemma4:e2b` on Ollama `:11434`, `gemma-4-e2b` on a running Krill
server) over a fixed suite, temp 0, and scores each turn:

- **T1 single tool** - one explicit `add(a,b)`; expect exactly one
  well-formed call with correct args.
- **T2 tool selection** - 3 tools, prompt needs exactly one; wrong
  tool / extra calls = fail.
- **T3 no-tool** - answerable without tools; emitting a spurious call
  = fail.
- **T4 multi-step agentic loop** - call -> inject `tool` result ->
  follow-up call -> final answer; scores loop completion.

Per engine per task: `valid_tool_call` (parsed, right name, args
schema-valid), `args_exact` (values match), `well_formed` (no
double-nesting / leakage), latency. The metric is the
**Krill-vs-Ollama parity ratio**; the gate is "Krill ≥ Ollama on
valid_tool_call and args_exact across the suite."

Pre-fix expectation (captured as the baseline artifact): Ollama green
across T1-T4; Krill red on T1 (`args` double-nested) and T4.

Run:

```
python3 tools/tool_calling_benchmark.py \
  --krill-url http://127.0.0.1:11435 \
  --ollama-host http://127.0.0.1:11434 \
  --krill-model gemma-4-e2b --ollama-model gemma4:e2b \
  --output .build/benchmarks/tool-calling.json
```

## 6. Out of scope

Token-level streaming tool-call deltas (Phase 4 parity follow-up);
Llama/Qwen native tool formats (WS3 tracked follow-up); grammar-
constrained decode of the `response_schema`.

[finding]: agentic/tool-calling underperforms Ollama on the same
Gemma 4 weights - see `memory/finding_tool_calling_gap.md`.
