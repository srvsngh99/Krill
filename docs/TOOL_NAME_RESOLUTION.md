# Tool-name resolution

How Krill guarantees that a tool call names a tool the harness actually offers,
and what to do when adding a model family or a tool.

## The problem

Tool names are chosen by the caller, not the model. OpenAI, Anthropic, and MCP
all work this way: the request lists the tools, and the model is expected to
reply with one of those exact names.

Local models routinely break that expectation. A model fine-tuned on another
harness's tool vocabulary names the right capability with the wrong word:

```
> Add a docstring to fizzbuzz.py
  Read  fizzbuzz.py
  Error: unknown tool 'Read'. Available tools: read_file, list_dir, glob, ...
```

`gemma-4-12b-agentic` learned Claude Code's vocabulary, where the read tool is
`Read`. Krill offers `read_file`. The capability matched; the spelling did not,
and the agent died on its first tool call.

This is not a casing slip or a separator slip. `Read` and `read_file` are
different words, so no amount of normalisation can bridge them in general.

## The shape of the fix

Three layers. The first prevents the error; the other two recover from it where
prevention is structurally impossible.

```
                                    cost         covers
0. Constrained sampling             free         masking backends, families with a sentinel
1. Deterministic normalisation      free         casing / separators / prefixes / known synonyms
2. Grammar-constrained re-pick      1 generation anything still unresolved
   -> otherwise: "unknown tool" observation back to the model
```

### Layer 0 - make it unrepresentable (primary)

`OutputFormat.toolNames` installs a **trigger-activated** grammar
(`ToolNameAutomaton`, `Sources/KrillGrammar/ToolNameGrammar.swift`). This is the
approach the ecosystem converged on: llama.cpp's lazy grammars, XGrammar's
structural tags, vLLM's guided decoding.

A tool call is optional - the model must stay free to answer in prose - so the
grammar cannot constrain the whole generation. Instead it:

1. idles in a pass-through state, watching for the family's tool-call sentinel;
2. on `<tool_call>` (or `[TOOL_CALLS]`, ...) starts looking for the `name` key
   using a small JSON scanner that tracks string and escape context;
3. constrains the characters of the name value to the offered set;
4. disarms at the closing quote, so arguments decode freely.

Inside the name slot the sampler is masked to characters that keep the output a
prefix of some offered name. `Read` is not merely rejected after the fact - the
capital `R` has no probability mass, so it can never be sampled.

The engine's existing fail-open rule still applies: if the automaton ever
rejects an accepted token, the mask is disabled for the rest of the generation.
The worst case is the unconstrained behaviour that shipped before.

### Layer 1 - deterministic normalisation (free)

In `ToolCalling.canonicalizeNames`, in order: exact match, case-insensitive
match, namespace prefixes (`tools.`, `functions__`, ...), separator-squash
(`WriteFile` / `write-file` -> `write_file`, unique matches only), segment
splitting, and finally a small closed alias table for well-known synonyms
(`Read` -> `read_file`).

The alias table is a convenience, not the guarantee. It is deliberately small
and closed - fuzzy matching would risk dispatching the wrong tool, which is
worse than failing. Every entry is pinned to the real registry by
`ToolNameAliasConformanceTests`, so renaming a tool fails the build instead of
leaving an alias that silently never fires.

### Layer 2 - constrained re-pick (one generation)

`AgentLoop.resolvedCall` asks the model to choose again, constrained to
`{"tool": {"enum": [...offered]}}`. Accepted only if the reply names an offered
tool; otherwise the original call is left untouched and the standard
`unknown tool 'X'. Available tools: ...` observation goes back to the model.

Every layer is gated on the **offered** set. A name never resolves to a tool the
caller did not expose, even if that tool exists.

## Per-family coverage

Layer 0 needs an unambiguous marker that says "a tool name comes next". The
policy lives in one place: `ToolCallSentinels` (`Sources/KrillTooling/`).

| Family | Sentinel | Layer 0 |
|---|---|---|
| `.hermes` | `<tool_call>` | yes |
| `.qwen` | `<tool_call>` | yes |
| `.gemma4` | `<tool_call>`, `<\|tool_call\|>` | yes |
| `.mistral` | `[TOOL_CALLS]` | yes |
| `.phi` | `<\|tool_call\|>` | yes |
| `.llama` | `<\|python_tag\|>` | partial - tagged form only |
| `.pythonic` | none | no |

Two exclusions, both deliberate:

- **`.pythonic`** emits `read_file(path="x")`. Until the `(` arrives it is
  indistinguishable from prose, and by then the name has already been decoded.
- **`.llama`** also emits a bare leading JSON object with no wrapper. Arming on
  `{"name"` alone would fire on any JSON the model legitimately writes - a real
  risk for an agent whose job includes writing JSON files. The `<|python_tag|>`
  variant carries a marker and is covered.

Excluded families are not left worse off than before: they keep layers 1 and 2.
They only lose the stronger "cannot be generated at all" guarantee.

## Adding a model family

1. Add the case to `ToolCalling.ToolFormat` and its renderer/parser.
2. Add its sentinel to `ToolCallSentinels.sentinels(for:)`. If the family has no
   unambiguous marker, return `[]` and add a sentence to the type doc saying
   why - an empty list is a decision, not an oversight.
3. Add the family to the table above.
4. Extend `testEachFamilyArmsAndRejectsAnUnknownName` in
   `Tests/KrillGrammarTests/ToolNameGrammarTests.swift` with a realistic call
   opening. That test asserts both halves: the sentinel arms the constraint, and
   a name outside the offered set is rejected at the first wrong character.

## Adding or renaming a tool

Nothing to do for layer 0 - it reads the live offered set.

If you **rename** a tool, `ToolNameAliasConformanceTests` fails if an alias still
points at the old name. Fix the target or delete the entry.

Only add an alias for a name a real model actually emits. The guarantee comes
from layer 0; the table is there to save a generation on known synonyms, not to
guess.

## Testing

| Test | What it protects |
|---|---|
| `KrillGrammarTests/ToolNameGrammarTests` | prose stays unconstrained; each family arms; unknown names unrepresentable; a `"name"` inside a string value cannot arm the constraint |
| `KrillToolingTests/ToolNameRecoveryTests` | casing, separator, prefix and alias normalisation; ambiguity is left unresolved |
| `KrillHarnessTests/ToolNameAliasConformanceTests` | alias targets are real tools; no alias shadows a tool; resolution stays inside the offered set |
| `KrillHarnessTests/AgentLoopTests` | the re-pick fires only when needed, never for an alias hit or a known name, and fails open |

The safety property worth keeping in mind while changing any of this: **a wrong
guess is worse than no guess.** Failing with `unknown tool` costs an iteration;
dispatching the wrong tool can delete a file.
