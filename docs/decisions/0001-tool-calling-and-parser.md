# 0001. Tool Calling and Parsing

Status: adopted. Date: 2026-06-27. Owner: Sourav. Scope: how Krill turns a model's
generated text into structured tool calls, for BOTH of Krill's tool-using
systems (the inference server and the in-process agent harness), and why we
chose the design we did.

---

## 0. TL;DR

- The **wire/API layer is an industry standard** (OpenAI `tools` / `tool_calls`)
  and Krill already implements it. No model-side standard exists for HOW a model
  emits a call in its generated text; every serious engine keeps **per-model
  parsers** (vLLM `--tool-call-parser`, Ollama/llama.cpp templates).
- Krill has **one** shared tool-calling module, `Sources/KrillTooling/ToolCalling.swift`,
  used by both the **inference server** (`KrillServer/Server.swift`) and the
  **harness** (`KrillHarness/AgentLoop.swift`). A change in that module affects
  both, so both test suites gate every change (see Section 8).
- **Decision: a hybrid.** (1) A **robust default** where each family's parser
  recognizes that family's full trained output space - including **pythonic**
  calls - plus universal tool-name case-normalization and a `parameters`/
  `arguments` alias. This needs zero per-model config and cannot silently
  regress. (2) An **optional per-alias parse override** (`ResolvedModel.toolFormat`)
  for the rare model you want to pin to one parser, like vLLM's per-model
  `--tool-call-parser`. Inject (the prompt) always stays the model's trained
  template; the override only changes the PARSER.

---

## 1. The two systems (both built into one binary)

Krill ships inference and an agent harness in one process. They are different
consumers of the SAME tool-calling code:

| System | Entry | What it does with tool calls |
|---|---|---|
| **Inference server** | `KrillServer/Server.swift` (`handleToolChat`, OpenAI/Ollama/Anthropic endpoints) | Parses the model's text into `tool_calls` and returns them to the API client. The CLIENT executes the tools. |
| **Harness** (agent loop) | `KrillHarness/AgentLoop.swift` | Parses the same way, then EXECUTES the tool locally, feeds the result back, and loops until a final answer. |

Both call the same three functions in `ToolCalling.swift`:
`injectToolSystem(...)` (render tools into the prompt), `extractToolCalls(...)` /
`extractIfToolsOffered(...)` (parse calls out of the text), and
`openAIToolCalls(...)` / `ollamaToolCalls(...)` (shape the wire response).

Consequence (the rule the user set): **one change must not break the other.**
Because the logic is shared, the guarantee is structural - there is no second
parser to drift - and is enforced by running BOTH the server tests
(`Tests/KrillServerTests/ServerTests.swift`) and the harness tests
(`Tests/KrillHarness*`/`AgentLoopTests`) on every change.

---

## 2. The three layers, and which are standardized

### Layer A - Wire/API: STANDARDIZED (OpenAI)
Request: `tools: [{type:"function", function:{name, description, parameters(JSON Schema)}}]`.
Response: `tool_calls: [{id, type:"function", function:{name, arguments(JSON string)}}]`.
Every engine (vLLM, Ollama, TGI, llama.cpp, SGLang) speaks this. Krill implements
it in `openAIToolCalls`/`ollamaToolCalls` (`ToolCalling.swift`). **No change.**

### Layer B - Render/Inject: standardized via the model's chat template
The convention (HF `apply_chat_template(tools=...)`) is that the MODEL ships a
Jinja template that renders tools the way it was trained. Krill currently
**hand-codes** these blocks per family (`injectHermes`, `injectQwen`,
`injectGemma4`, `injectMistral`, `injectPhi`, `injectLlama`) because
swift-transformers drops the `tools` Jinja variable. This is a known divergence
from the standard; closing it (render straight from the model template) is a
future effort, tracked separately - NOT part of this decision.

### Layer C - Parse/Extract: NO standard; per-model parsers
Each family trained on its own emit format; there is no shared token:
- Hermes / Qwen: `<tool_call>{json}</tool_call>`
- Mistral: `[TOOL_CALLS][...]`
- Llama 3.x: `<|python_tag|>` + JSON, or pythonic
- Gemma 3/4: **pythonic** ` ```tool_code\nget_weather(city="Paris")\n``` `
- Phi: `<|tool_call|>[...]<|/tool_call|>`

The industry answer is a NAMED parser chosen per model. vLLM exposes
`--tool-call-parser {hermes, mistral, llama3_json, pythonic, granite, ...}`;
`pythonic` is a first-class parser there, not a fallback. This is the layer our
decision is about.

---

## 3. How Krill resolved the format BEFORE this change

Format was derived purely from FAMILY, once, shared by inject and parse:

```
engine.family ("gemma4_unified", "qwen3_5", ...)   InferenceEngine.swift:97
   -> ToolFormat.forFamily(family)                  ToolCalling.swift:73-86
      (ModelFamily -> ModelAdapter.chatTemplate -> {hermes,gemma4,llama,qwen,mistral,phi})
   -> injectToolSystem(tools, format)               prompt block
   -> extractToolCalls(text, format)                that format's parser
```

Family-to-format map (`ModelAdapter.swift:77-106`): `gemma4`/`gemma4Unified`->gemma4;
`llama`->llama; `qwen`/`qwen35`/`moe`->qwen; `mistral`->mistral; `phi`->phi;
everything else (`gemma`,`glm`,`glm4`,`deepseek`,`qwen2_5_vl`,...)->hermes.

**Why this failed for agentic models:** `gemma-4-12b-agentic` is family
`gemma4_unified` -> format `gemma4` -> the `<|tool_call>`/`call:` sentinel parser.
The finetune actually emits **pythonic** (`get_weather(city="Paris")`), which that
parser cannot see, so it returned zero tool calls. One family = one parser could
not express "this finetune drifted to a different format."

---

## 4. The decision (adopted): hybrid - robust default + optional per-alias override

### 4a. Robust default (zero config, cannot silently regress)
In the shared parse dispatch (`extractToolCalls`):
1. Run the family's native parser (unchanged).
2. If it found nothing AND the offered tool names are known, try the new
   **pythonic** parser (`extractPythonic`) - gated on the offered tool names so a
   stray `func(...)` in prose is never mistaken for a call.
3. **Canonicalize tool-name casing** against the offered set (`Multiply` ->
   `multiply`) so a casing slip is not an "unknown tool".
4. Accept **`parameters` as an alias for `arguments`** in the JSON paths so a
   `{name, parameters}` object is not dropped.

This is "each parser covers its family's full trained output space," which now
includes pythonic. A new agentic finetune works with no registry edit.

### 4b. Optional per-alias parse override (explicit pin, like vLLM)
`ResolvedModel` (`KrillRegistry/AliasMap.swift`) gains `toolFormat: String?`
(default nil). When set (e.g. `gemma-4-12b-agentic -> "pythonic"`),
`ToolFormat.forModel(modelName, family)` uses it for the PARSER; inject still
uses `forFamily` (the model's trained prompt). `engine.modelName`
(`InferenceEngine.swift:108`, the blob dir = the alias) is available at every
resolution site, so the override needs no new plumbing through the loader.

Important: the override is **parse-only**. We never change inject from the model's
trained template, because a finetune is usually PROMPTED in its base format even
when it EMITS a drifted format (Gemma agentic is prompted gemma4, emits pythonic).
This mirrors vLLM separating `--chat-template` (render) from `--tool-call-parser`
(parse).

---

## 5. Alternatives considered, and why we did not pick them

| Option | What it is | Why not (as the sole design) |
|---|---|---|
| **A. Strict per-alias registry only** | Format comes ONLY from an explicit per-alias field; family is just fallback. (vLLM's exact model.) | Every new model must be configured or it **silently mis-parses** (falls back to the wrong family default and returns zero calls - the exact bug we debugged, but now invisible). Config burden on every pull. Local/path-loaded models (not in AliasMap) can never be overridden. We KEEP the override, but not as the only mechanism. |
| **B. Parser-handles-both only (no override)** | Each parser covers sentinel + pythonic; no registry field at all. | Great robustness, zero config - but no way to FORCE one parser for the rare model that emits two formats ambiguously, and no explicit, auditable record of a model's format. We adopted this as the DEFAULT and added the override on top. |
| **C. Stacked universal fallbacks** | Try every parser in sequence on every model. | Maximizes false positives (one model's prose parsed as another's call), unauditable, slow. Rejected. |
| **D. Chat-template-driven inject overhaul** | Fix swift-transformers so inject renders from the model's own Jinja `tools` var (the true render standard). | Correct long-term direction, but a large, higher-risk change touching every family's inject. Deferred to a separate effort; out of scope here. |

**Why the hybrid wins:** it removes the silent-failure mode of A (the default
always tries pythonic + normalizes casing), keeps the explicit control A wanted
(the per-alias override), and avoids the false positives of C (gated on offered
tool names, native parser first). It is the smallest change that makes every
model agentic without a config tax.

### Pros / cons of the adopted hybrid
- Pros: new agentic finetunes work with no edit; nothing silently regresses;
  casing/`parameters` slips no longer drop calls; explicit pin available;
  inject stays correct (trained template); one shared parser for both systems.
- Cons: two mechanisms to understand (default + override) instead of one; the
  pythonic parser is heuristic (kwargs only; positional args are skipped); the
  override is parse-only by design, so a model that needs a different INJECT
  still relies on the family template (closed only by Option D later).

---

## 6. Per-format reference (parser coverage)

Dispatch: `extractToolCalls(from:format:knownToolNames:)` -> `extractIfToolsOffered`
(`ToolCalling.swift`). `forFamily` maps family->format; `forModel` applies the
per-alias override for the PARSER.

| Format | Inject fn | Parse fn | Native call shape | Notes |
|---|---|---|---|---|
| hermes (default/fallback) | injectHermes | extractHermes | `<tool_call>{json}</tool_call>` | also the internal fallback for llama/qwen/mistral/phi |
| gemma4 | injectGemma4 | extractGemma4 | `<|tool_call>call:NAME{json}<tool_call|>` | base Gemma sentinel |
| qwen | injectQwen | extractQwen | Hermes + bare-JSON scan | Qwen2.5/3, Ornith (qwen3_5), MoE |
| llama | injectLlama | extractLlama | `<|python_tag|>{name,parameters}` | rejects echoed schemas |
| mistral | injectMistral | extractMistral | `[TOOL_CALLS][...]` | |
| phi | injectPhi | extractPhi | `<|tool_call|>[...]<|/tool_call|>` | |
| **pythonic (new)** | (inject stays family) | **extractPythonic** | `name(arg=val)`, `tools.name(...)`, ```tool_code``` / ```python``` blocks, `[f(), g()]` | gated on offered tool names |

Cross-cutting normalization applied to EVERY format after parse:
tool-name casing canonicalized to the offered set; `parameters` accepted as
`arguments`.

---

## 7. How to pin a model's parser (the override)

In `Sources/KrillRegistry/AliasMap.swift`, set `toolFormat` on the alias:

```swift
"gemma-4-12b-agentic": ResolvedModel(
    repo: "srv-sngh/gemma-4-12B-agentic-fable5-composer2.5-v2-nvfp4",
    name: "gemma-4-12b-agentic", family: .gemma4Unified, params: "12B",
    quant: "nvfp4", context: 131072,
    toolFormat: "pythonic"),   // PARSER override; inject stays gemma4
```

Valid values: `hermes`, `gemma4`, `llama`, `qwen`, `mistral`, `phi`, `pythonic`.
Omit for the robust default. Local/path-loaded models always use the default.

---

## 8. Testing contract (the "do not break the other system" guarantee)

Because inference and harness share `ToolCalling.swift`, every change to it must
keep BOTH suites green:
- Inference/parse: `Tests/KrillServerTests/ServerTests.swift` (all six formats:
  extraction, injection, `forFamily` mapping, multi-call, missing-close, bare
  JSON, no-tools gate).
- Harness: `Tests/KrillHarness*` / `AgentLoopTests` (the agent loop end to end).
- Registry policy: `Tests/KrillRegistryTests/ModelAdapterTests.swift` (the
  invariant that a tools-capable family is never `.hermes`).
- New coverage added with this change: pythonic extraction (Gemma `tool_code`,
  `tools.fn(...)`, `[f(), g()]` list), tool-name casing canonicalization,
  `parameters` alias, and the per-alias `forModel` override.

A green run of all four proves the inference and harness paths still agree.

---

## 9. Key files

- `Sources/KrillTooling/ToolCalling.swift` - the shared module: `ToolFormat`,
  `forFamily`/`forModel`, `injectToolSystem`, `extractToolCalls`/
  `extractIfToolsOffered`, `extractPythonic`, the per-format inject/parse fns,
  and the `openAIToolCalls`/`ollamaToolCalls` wire shapers.
- `Sources/KrillRegistry/ModelAdapter.swift` - family -> `ChatTemplatePolicy`
  (the family default).
- `Sources/KrillRegistry/AliasMap.swift` - `ResolvedModel.toolFormat` per-alias
  override.
- `Sources/KrillServer/Server.swift` - inference call sites (`handleToolChat`).
- `Sources/KrillHarness/AgentLoop.swift` - harness call sites.
- `Sources/KrillEngine/InferenceEngine.swift` - `family` / `modelName` used to
  resolve the format at request time.
