# Proposal: `text_decode_ratio` gate semantics under `release_candidate`

Status: **ACCEPTED & APPLIED** — owner answered "Accept & implement"
(AskUserQuestion, 2026-05-16) and re-confirmed verification ("yes go
ahead and you do it"). Implemented in `tools/release_gate.py`
(`ADVISORY_HARD_FLOORS`) with unit tests in `tools/test_release_gate.py`
(`DecodeAdvisoryFloorTests`).
Authored: 2026-05-16, after PR #16 closed `memory_ratio`.
Scope: `release_candidate` profile only. `strict` is **unchanged**.

> **Follow-up (2026-05-22):** the same `text_decode_ratio` demotion was
> later extended to the `strict` profile - see
> [RELEASE_GATE_STRICT_DECODE_PROPOSAL.md](RELEASE_GATE_STRICT_DECODE_PROPOSAL.md).
> The "`strict` is unchanged" / "strict keeps `text_decode_ratio` hard"
> statements in this 2026-05-16 document are historical: `strict` now also
> treats `text_decode_ratio` as advisory with the same hard `>= 1.0`
> floor. Every other `strict` metric remains hard.

## Verified outcome (`.build/benchmarks/v6-mm.json`, 2026-05-16)

```text
release_candidate --allow-dtype-mismatch  -> exit 0  (GATE: PASS)
  text_decode_ratio        1.1937  WARN [advisory]  (need >= 1.5)
  text_decode_ratio_floor  1.1937  OK   [hard]      (need >= 1.0)
  memory_ratio             0.3221  OK   [hard]
  text_wall_ratio          0.6373  OK   [hard]
  text_ttft_ratio          0.2102  OK   [hard]
  image_wall_ratio         0.5645  OK   [hard]
  caveat: text_decode_ratio demoted to advisory ... hard-gated by
          text_decode_ratio_floor >= 1.0x ... (recorded in report)

strict                                    -> exit 1  (unchanged)
```

The summary still prints `text_decode_ratio` as an advisory WARN at
1.19x — the gate does **not** claim Krill hit 1.5x decode.

## Problem

After PR #16, `text_decode_ratio` is the **sole** hard `release_candidate`
gate miss. Measured across 5 fresh `native_server` runs on the M4 Pro
24 GB target:

| Metric | Range (5 runs) | Threshold | Kind | Status |
| --- | --- | --- | --- | --- |
| `text_decode_ratio` | 1.13–1.19x | ≥ 1.5x | hard | ❌ FAIL |
| `text_wall_ratio` | 0.64–0.69x (≈1.46–1.57x faster) | ≤ 0.67x | hard | ✅ |
| `text_ttft_ratio` | 0.20–0.22x (≈4.5–5.1x faster) | ≤ 0.67x | hard | ✅ |
| `image_wall_ratio` | 0.56–0.58x (≈1.71–1.77x faster) | ≤ 0.67x | hard | ✅ |
| `memory_ratio` | 0.32–0.84x | ≤ 1.0x | hard | ✅ |

Krill decodes ~103–106 tok/s; Ollama ~88–95 tok/s.

This is **not variance and not a measurement artifact** (PR #16 verified the
loop is already two-deep GPU-pipelined, sampling is on-GPU argmax, and the
`eval_duration` accounting is fair). Decode of a dense model is bounded by
per-token weight-read bandwidth; on a tiny 4-bit Gemma 4 e2b, llama.cpp's
mature hand-tuned Metal decode kernels are at parity with MLX. Pushing the
*ratio* to 1.5x requires doing less work per emitted token — i.e.
speculative decoding (Workstream 2), a multi-week effort with greedy-parity
risk that is explicitly out of scope for the current baseline.

## Why this is a gate-semantics question, not just "we're slow"

The product claim is: *Krill beats Ollama by 1.5x–3x on the accepted
benchmark matrix, with equivalent inputs.* The question is **which metric
substantiates that claim to a user**:

- A user experiences **end-to-end latency** (`*_wall_ratio`) and
  **responsiveness** (`text_ttft_ratio`). On the current matrix these are
  hard-gated and pass at 1.5x–5x. This is the claim, and it holds.
- `text_decode_ratio` is steady-state tok/s — a **kernel microbenchmark**.
  On the current 32-token matrix it does not drive the user-visible number
  (`text_wall` already captures the win). It only dominates wall time for
  **long** generations, which the current matrix does not exercise.

The codebase already accepts exactly this reasoning for
`text_prefill_ratio`: under `release_candidate` it is **advisory** because
"`text_wall` and `text_ttft` are the user-visible signals and are already
hard-gated; prefill TPS is noisy on short prompts"
(`tools/release_gate.py`, profile comment). `text_decode_ratio` on a
short-output matrix is the same situation.

## Proposal

Under `release_candidate` **only** (strict keeps `text_decode_ratio` hard
at ≥1.5x, unchanged):

1. **Demote the ≥1.5x target to `advisory`** — mirrors the already-accepted
   `text_prefill_ratio` treatment. The aspirational 1.5x is still
   evaluated, reported, and printed; it just does not break the gate.

2. **Add an explicit HARD non-regression floor: `text_decode_ratio ≥ 1.0`.**
   This preserves a real, release-quality guarantee — *Krill must never
   decode slower than Ollama* — even for long outputs. It is robustly met
   (1.13–1.19x across runs, with margin against documented Ollama decode
   variance of 88–95 tok/s). If Krill ever regresses below Ollama parity,
   the gate hard-fails regardless of the advisory target.

This is **not** a silent relaxation:

- The floor and the demotion are encoded in `GATE_PROFILES` /
  `RELEASE_CANDIDATE_HARD_FLOORS` with a documented rationale comment.
- The gate report records `scope.text_decode_ratio` (the demotion + the
  active hard floor) and adds a caveat line, exactly like the
  `memory_ratio` conditional downgrade.
- The terminal summary prints `text_decode_ratio` as
  `ADVISORY (hard floor ≥1.0: PASS)` so a reader cannot miss it.
- `strict` is untouched: anyone wanting the full ≥1.5x rigor runs
  `--profile strict`.

## Re-promotion contract (objective, mandatory)

`text_decode_ratio` re-promotes to **hard ≥1.5x** under `release_candidate`
when **either**:

- (a) Gemma 4-compatible speculative decoding (Workstream 2) lands and
  sustains `text_decode_ratio ≥ 1.5x` with greedy parity, **or**
- (b) the benchmark matrix adds a **long-output decode task** (e.g.
  `--text-max-tokens 512`) where decode genuinely dominates wall time —
  there, decode ratio *is* the user-visible signal and must be hard-gated.

Until then the floor (≥1.0) is the hard guarantee and 1.5x is the tracked
advisory target. This contract is recorded in
`OLLAMA_SPEEDUP_EXECUTION_PLAN.md` §4 and
`RELEASE_READINESS_REMEDIATION.md`.

## What this does and does not claim

- It **does not** claim Krill decodes 1.5x faster than Ollama. It does
  not. Release notes/README must not say so.
- It **does** let the gate exit `0` under `release_candidate` on the
  evidence that actually backs the product claim (wall, TTFT, memory all
  hard-pass at 1.5x+) while keeping a hard floor that Krill is never
  slower than Ollama at decode.
- `strict` remains the uncompromised reference and still exits `1`.

## Decision

Owner must explicitly accept this before it is applied. If accepted, the
concrete patch (gate code + unit tests + doc updates, `strict` unchanged)
ships in the same PR branch. If rejected, the gate stays red on
`text_decode_ratio` and the project remains an honest release-readiness
baseline pending Workstream 2.
