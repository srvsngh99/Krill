# Decision models

Tier: **Experimental** (served over HTTP and checked against golden fixtures).

A decision model reads a text and returns a typed answer instead of generating text: one of
several options (or "none of these") with probabilities, or yes / no. It is a ModernBERT
reader plus a small decision head. The type has its own runtime path (`DecisionModel` in
`KrillCore`, `DecisionEngine` in `KrillEngine`) and serves both `POST /v1/decide` and
the Ollama-compatible `POST /v1/systemone`. Both endpoints use the same runtime.

A bundle is recognised as a decision model when its `config.json` has a `reader` and a
`head` section (both objects) and a `pooling` entry. Amber 0.4B v2 uses `uni_config.json`,
`model.safetensors`, and the tokenizer and encoder config under `encoder/`. GLiDE 0.4B uses
`decision_config.json`, `readout.safetensors`, and a sharded ModernBERT encoder. All three formats
register as the `decision` model family.

## CLI

`--model` accepts a bundle directory or the name of an installed model.

```sh
krill pull MLukauskas/amber-0.4b-v2
krill decide --model amber-0.4b-v2 --text "My card was charged twice." \
  --question "Classify the banking intent." \
  --option "duplicate charge" --option "password reset"

krill pull fastino/GLiDE-0.4B
krill decide --model glide-0.4b --text "My card was charged twice." \
  --question "Classify the banking intent." \
  --option "duplicate charge" --option "password reset"
```

Amber and GLiDE read the question, text and options together. Their probabilities sum to one
across the given options, and `p_none` is zero. Use `amber-0.4b-v2` or `glide-0.4b` as the model
name on either HTTP endpoint. GLiDE's System One prompt includes both the criterion key and its
description in the order they appear in the request.
The examples below show the reader-based bundle format and its optional abstention.

```sh
krill decide --model <bundle> --text "My card was charged twice." \
  --option "billing question" --option "password reset" --option "shipping delay"
# {"index":0,"label":"billing question","p_none":0.0016,"scores":[0.998,0.00002,0.0003]}

krill decide --model <bundle> --text "The invoice was paid in full." --yes-no "Is the invoice paid?"
# {"answer":"yes","p_yes":0.97}
```

- The reader-based bundle appends `--question <q>` to the text (`text + "\n\n" + question`).
- `--no-none` always picks an option; otherwise `label` and `index` are `null` when "none of these" wins.
- `--json-in <file|->` reads JSON lines (`text`, `question`, `options`, `allow_none`, `yes_no`)
  and prints one result per line, loading the model once.

## HTTP: `POST /v1/decide`

`model` is the name of an installed decision model, as for embeddings and rerank; a path is not
accepted over HTTP (the CLI takes one). Send
`text` (one string) or `texts` (a non-empty array, one result each), and either `options` (with an
optional `question` and `allow_none`, default `true`) or `yes_no` (the question).

```sh
curl -s http://127.0.0.1:57455/v1/decide -H 'Content-Type: application/json' -d '{
  "model": "decider",
  "texts": ["My card was charged twice.", "Where is my parcel? It is a week late."],
  "options": ["billing question", "password reset", "shipping delay"]
}'
# {"model":"decider","results":[
#   {"label":"billing question","index":0,"scores":[0.998,0.00002,0.0003],"p_none":0.0016},
#   {"label":"shipping delay","index":2,"scores":[0.001,0.0002,0.99],"p_none":0.005}]}

curl -s http://127.0.0.1:57455/v1/decide -H 'Content-Type: application/json' -d '{
  "model": "decider", "text": "The invoice was paid in full.", "yes_no": "Is the invoice paid?"}'
# {"model":"decider","results":[{"answer":"yes","p_yes":0.97}]}
```

When "none of these" wins, `label` and `index` are `null` (`scores` and `p_none` are still given);
`allow_none: false` always picks an option. Amber and GLiDE always pick an option.

Errors are `{"error": "..."}`, like the embeddings and rerank routes:

| Status | When |
|---|---|
| 400 | malformed JSON; missing `model`; no `text` / `texts` (or both); no `options` / `yes_no` (or both); empty `options`; wrong field types; an empty option; more than 256 texts or more options than the model supports; text or option rejected by the bundle's input limits; a `yes_no` question with `question`; `model` is not a decision model |
| 401 | missing or wrong bearer key, when the server has one |
| 404 | unknown model name |
| 500 | the model failed to load or run (the body says only "decide failed"; details go to the server log) |

A decision model sent to chat, embeddings or rerank is refused with a message that says so. It
shares the server's request queue, `keep_alive` field and idle unload with the embedding and rerank
models, and a registered decision model shows up in `/v1/models` and `/v1/status`.

The [Ollama-compatible System One endpoint](SERVER_API.md#post-v1systemone)
scores named choice, yes/no, and ordinal questions about one shared state. It serves
Krill decision bundles only. Krill cannot load Ollama's Nimble, Tev1, or Clef weights.

## Input limits

A `/v1/decide` request may carry up to 256 texts and 128 options, or 255 options with GLiDE.
A `/v1/systemone` choice may carry up to 255 options. Amber allows up to 7,999 tokens per
question, including the instructions, state, options and markers. Its yes/no options are
`yes` and `no`.

GLiDE runs in bf16 and reads the final prompt token before `[SEP]`. Prompts that exceed 7,999
encoder tokens keep the first 1,024 prompt tokens and the tail, including the answer cue, with
room for `[CLS]` and `[SEP]`.

Reader-based bundles set `state_max_tokens` (the text with its question) and `option_max_tokens` in their
`config.json` (2,048 and 32 for the tiny test bundle). For `yes_no` the question is scored as the
options `no: <question>` and `yes: <question>`, so it must fit `option_max_tokens` including the
prefix and special tokens; a longer one is a 400 that says "the yes_no question". Longer input is an
error that names the count and the limit; nothing is truncated. Texts go through the reader 8 at a
time and each chunk is evaluated before the next is read, so only the pooled rows and option scores of
earlier chunks stay alive; one chunk can still hold 8 texts of the full token limit. Empty or
whitespace-only texts and options are rejected. The option cache holds at most 512 options in total.

## Parity

Amber's 200-question parity gate compares options and probabilities and reports median native
scoring latency after a warmup. Set `KRILL_AMBER_MODEL_PATH`, `KRILL_AMBER_INPUTS`, and
`KRILL_AMBER_RESULTS` to the local bundle and the two JSONL files. Build the release tests and
Metal library, then run `python3 scripts/amber_parity.py`.

GLiDE's gate requires at least 198 matching choices out of 200 and reports the maximum absolute
probability difference and median native scoring latency. Set `KRILL_GLIDE_MODEL_PATH`,
`KRILL_GLIDE_INPUTS`, and `KRILL_GLIDE_RESULTS` to the local bundle and benchmark JSONL files.
Build the release tests and Metal library, then run `python3 scripts/glide_parity.py`.

Two reader-based golden fixtures, both produced by the Python reference runtime (CPU, fp32), which is not in this
repository:

- **Tiny bundle, runs in CI with no download.** `Tests/KrillCoreTests/Fixtures/decision/tiny` is a
  random-weight bundle of the same architecture (3 layers, hidden size 32, mean pooling, a 400-token
  tokenizer, 188 KB) and `golden_tiny.json` holds the reference output for 37 cases (the 35 inputs
  below plus two one-option lists), and `golden_tiny_cls.json` nine more for the same weights read
  with `cls` pooling. `TinyDecisionModelTests` checks token ids, every decision, probabilities within
  1e-4, batch equals single, the request limits and empty inputs, config validation, the fused layer norm
  against the explicit formula (fp32 and fp16), and the encoder's per-token states against the reference.
  `DecideEndpointTests` repeats the cases over HTTP. This covers masking, MaxSim, the head, the
  "none" rules and yes / no, but with random weights it says nothing about the quality of a real model.
- **Real bundle, needs the weights (not public).** `golden.json` holds 35 hand-written cases:
  support, intent, spam, moderation, tool choice, sentiment, 2- to 77-option lists, with and without
  a question, "none" and `allow_none=false`, a 1,470-token text, a padded batch, four yes / no.
  `DecisionModelParityTests` and `testMatchesRealBundleGolden` run it with
  `KRILL_DECISION_MODEL_PATH=<bundle> swift test --filter Decision` and skip when it is unset, so
  CI never runs them. Regenerated for the 1.0 bundle; verified locally on one machine: 35 of 35
  decisions identical, probabilities within 1.5e-5.

## Speed and memory

Manual measurements on one machine (M4 Pro), not run in CI. The in-process numbers come from
`DecisionModelBenchmarkTests` (opt-in with `KRILL_DECISION_BENCH=1`, release build, 50 warm calls,
medians). The comparison column is a Python reference runtime on MPS (fp32, same calls) that is not in
this repository, so it cannot be reproduced from here. Other jobs may have shared the machine.

| Call | Krill | Python reference | Ratio |
|---|---|---|---|
| short text, 3 options | 6.7 ms | 10.0 ms | 0.7 |
| short text, 77 options, first call (options not cached) | 58 ms | 76 ms | 0.75 |
| short text, 77 options, repeat calls | 6.5 ms | 10.0 ms | 0.65 |
| 99-token text, 4 options | 13.3 ms | 15.1 ms | 0.9 |
| 451-token text, 10 options | 46.0 ms | 39.2 ms | 1.15 |
| yes / no | 6.6 ms | 9.9 ms | 0.65 |

The reader's layer norm runs as one fused kernel (`MLXFast.layerNorm`). A server with the model loaded
held 776 MB (physical footprint) after a few calls, against 8 MB idle; one reading, one machine.

## Not there yet

- A release performance gate. The reader-based model reads texts in padded chunks of 8 and scores them one at a time.
- Loading is not single-flight: two cold requests can load two copies (same as the rerank engine).
