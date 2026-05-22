# Proposal: `text_decode_ratio` gate semantics under `strict`

Status: **ACCEPTED & APPLIED** - owner answered "Demote it under strict too"
(AskUserQuestion, 2026-05-22). Implemented in `tools/release_gate.py`
(`GATE_PROFILES["strict"]["text_decode_ratio"] = "advisory"`, floor applied
profile-agnostically) with unit tests in `tools/test_release_gate.py`
(`DecodeAdvisoryFloorTests`).
Authored: 2026-05-22.
Scope: `strict` profile, `text_decode_ratio` **only**. Every other `strict`
metric is unchanged and stays hard.

This extends the 2026-05-16 `release_candidate` decision
([RELEASE_GATE_DECODE_PROPOSAL.md](RELEASE_GATE_DECODE_PROPOSAL.md)) to the
`strict` profile.

## Problem

`strict` hard-gated `text_decode_ratio` at `>= 1.5x` (KrillLM decode tok/s
/ Ollama decode tok/s). That target is **structurally unreachable** on
M-series with the draft models available in mlx-community:

- `docs/SPECULATIVE_DECODING.md` derives that the speculative-decode
  throughput ratio asymptotes near **1.10x** even at infinite K and 100%
  acceptance - the per-round overhead (`beta`) is structural.
- Re-measured 2026-05-22: `llama-3.2-3b` + `llama-3.2-1b` draft, greedy -
  no-spec 59.1 tok/s vs spec 50.3 tok/s (**0.85x**). Speculative decoding
  does not raise effective decode throughput on this hardware; it lowers
  it.
- Plain-decode `text_decode_ratio` vs Ollama sits at ~1.13-1.19x: KrillLM
  beats Ollama at decode, but not by 1.5x. Decode of a dense 4-bit model
  is per-token weight-read-bandwidth bound, and llama.cpp's Metal kernels
  are mature.

So `strict` could never exit 0 on `text_decode_ratio`, not because of a
regression but because the bar encodes a speedup the hardware cannot give.

## Proposal

Under `strict`, for `text_decode_ratio` **only**:

1. **Demote the `>= 1.5x` target to `advisory`** - it is still evaluated,
   reported, and printed (`WARN [advisory]`); it just no longer breaks the
   gate. This mirrors the already-accepted `release_candidate` treatment.

2. **Keep the HARD non-regression floor `text_decode_ratio >= 1.0`.** The
   synthetic `text_decode_ratio_floor` evaluation (already used by
   `release_candidate`) now applies under `strict` too: KrillLM must never
   decode slower than Ollama, and a missing decode value hard-fails. The
   floor mechanism is keyed on the metric *kind* (`advisory`), not the
   profile name, so both profiles share one code path.

This is **not** a silent relaxation:

- The demotion is encoded in `GATE_PROFILES["strict"]` with a rationale
  comment, and the gate report records `scope.text_decode_ratio` plus a
  caveat line citing this doc.
- The terminal summary prints `text_decode_ratio` as `WARN [advisory]`
  next to the hard `text_decode_ratio_floor` - a reader cannot miss it.

## What this does and does not claim

- It **does not** claim KrillLM decodes 1.5x faster than Ollama. It does
  not. Release notes / README must not say so.
- It **does** stop a structurally unreachable microbenchmark target from
  being the sole hard reason `strict` fails, while keeping a hard floor
  that KrillLM is never slower than Ollama at decode.
- It **does not touch any other `strict` metric.** In particular the
  prefill-ratio metrics (`text_prefill_ratio`, `image_prefill_ratio`,
  `audio_prefill_ratio`) remain **hard** under `strict`. On the current
  benchmark matrix `strict` can therefore still exit 1 on a prefill miss -
  this proposal removes `text_decode_ratio` as a blocker, it does **not**
  by itself turn the whole `strict` gate green. Any prefill demotion is a
  separate decision and is explicitly out of scope here.

## Re-promotion contract

`text_decode_ratio` re-promotes to hard `>= 1.5x` under `strict` (and
`release_candidate`) when **either**:

- (a) speculative decoding lands a path that sustains `text_decode_ratio
  >= 1.5x` with greedy parity, **or**
- (b) the benchmark matrix adds a long-output decode task where decode
  genuinely dominates wall time.

Until then the floor (`>= 1.0`) is the hard guarantee and `1.5x` is the
tracked advisory target. This matches the contract in
[RELEASE_GATE_DECODE_PROPOSAL.md](RELEASE_GATE_DECODE_PROPOSAL.md) and
`OLLAMA_SPEEDUP_EXECUTION_PLAN.md` §4.
