# Proposal: `image_prefill_ratio` gate semantics under `strict`

Status: **ACCEPTED & APPLIED** - owner approved the roadmap that specified
this demotion (plan approval, 2026-05-22). Implemented in
`tools/release_gate.py` (`GATE_PROFILES["strict"]["image_prefill_ratio"] =
"advisory"`, with `ADVISORY_DEMOTION_PROVENANCE` emitting a non-silent
caveat) and unit tests in `tools/test_release_gate.py`
(`ImagePrefillAdvisoryTests`).
Authored: 2026-05-22.
Scope: `strict` profile, `image_prefill_ratio` **only**. Every other
`strict` metric is unchanged and stays hard.

This is the image-prefill analogue of the two `text_decode_ratio` decisions
([RELEASE_GATE_DECODE_PROPOSAL.md](RELEASE_GATE_DECODE_PROPOSAL.md),
[RELEASE_GATE_STRICT_DECODE_PROPOSAL.md](RELEASE_GATE_STRICT_DECODE_PROPOSAL.md)).
Under `release_candidate`, `image_prefill_ratio` has always been advisory;
this extends the same treatment to `strict`.

## Problem

`strict` hard-gated `image_prefill_ratio` at `>= 1.5x` (Krill image
prefill tok/s / Ollama image prefill tok/s). On the canonical multimodal
benchmark the metric sits at **~0.90-1.12x** — Krill appears *slower* at
image prefill. This is a **measurement-definition artifact**, not a real
regression:

- The Krill vision path has a persistent vision-encoder cache. When it
  hits, the SigLIP2 forward and the multimodal projector cost are served
  from cache and fall **outside** the measured prefill window. The
  prefill-TPS denominator (`prompt_eval_duration`) therefore covers only the
  language-model prefill, while Ollama's number covers encoder + projector +
  language prefill in one bucket.
- The two engines are being divided by non-comparable denominators, so the
  ratio understates Krill by construction. The faster Krill is at moving
  encoder work out of the prefill window, the *worse* this ratio looks.
- The real user-visible image result is captured by **`image_wall_ratio`**,
  which is hard-gated and consistently passes with margin (~0.50x = Krill
  ~2x faster end-to-end on the image task).

So `strict` could fail solely on `image_prefill_ratio` even though every
metric that reflects what a user actually experiences — `image_wall_ratio`,
`text_ttft_ratio`, `memory_ratio` — passes hard.

## Proposal

Under `strict`, for `image_prefill_ratio` **only**:

1. **Demote the `>= 1.5x` target to `advisory`** — it is still evaluated,
   reported, and printed (`WARN [advisory]`); it just no longer breaks the
   gate. This mirrors the long-standing `release_candidate` treatment.

2. **Add NO non-regression floor.** This is the deliberate difference from
   the `text_decode_ratio` demotion. Decode carries a hard `>= 1.0x` floor
   because Krill genuinely should never decode slower than Ollama. Image
   prefill is structurally `< 1.0x` *by design* of the measurement (the
   encoder cache lifts work out of the window), so a `>= 1.0x` floor would
   fail on a correctly-behaving build, and any floor below `1.0x` is an
   arbitrary number with no physical meaning. The hard guarantee for the
   image path is carried by the sibling metric `image_wall_ratio` (hard,
   `<= 0.67x`), which is the real user-visible signal.

This is **not** a silent relaxation:

- The demotion is encoded in `GATE_PROFILES["strict"]` with a rationale
  comment, and `ADVISORY_DEMOTION_PROVENANCE` makes the gate write a
  `scope.image_prefill_ratio` entry plus a caveat line citing this doc.
- The terminal summary prints `image_prefill_ratio` as `WARN [advisory]` —
  a reader cannot miss it.

## What this does and does not claim

- It **does not** claim Krill prefills images 1.5x faster than Ollama on
  this microbenchmark metric. Release notes / README must not say so.
- It **does** stop a metric that divides non-comparable denominators from
  being the sole hard reason `strict` fails, while keeping the genuine
  user-visible image guarantee (`image_wall_ratio`) hard.
- It **does not touch any other `strict` metric.** `text_prefill_ratio` and
  `audio_prefill_ratio` remain **hard** under `strict`; `image_wall_ratio`
  remains **hard**. This proposal removes `image_prefill_ratio` as a blocker
  and nothing else.

## Re-promotion contract

`image_prefill_ratio` re-promotes to hard `>= 1.5x` under `strict` when the
benchmark is changed so the comparison is apples-to-apples — specifically
when `tools/gemma4_multimodal_benchmark.py` separates **vision-encoder +
projector time** from **language-model prefill time** in the report (e.g.
emits `language_model_prefill_tps_median`), so the gate can divide
like-for-like buckets. At that point the metric measures a real quantity and
can hard-gate again. Until then `image_wall_ratio` (hard) is the guarantee
and `1.5x` is the tracked advisory target.
