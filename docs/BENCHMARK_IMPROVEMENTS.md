# Benchmark Improvement Roadmap

The current head-to-head suite (`docs/BENCHMARKS.md`) measures **speed** across
five axes (text, vision, voice, tools, concurrency) plus agentic/RAG. It already
shows wins or parity everywhere. This doc captures how to make the benchmarks
**more credible** and surface **more (and bigger) wins**, ordered by leverage.
Each item is a discrete, pick-up-able piece of work; none is started unless noted.

Status legend: PLANNED / IN PROGRESS / DONE.

---

## 1. Close the credibility gap (highest leverage)

The suite measures speed but not quality, so a skeptic can say "you are faster
because MLX 4-bit is a lossier quant than GGUF Q4_K_M." The quant classes are not
bit-identical (already noted in BENCHMARKS.md fairness controls), so today's speed
wins are technically confounded.

### 1a. Quality / accuracy axis  - PLANNED
- Run a fixed task set (a small MMLU / GSM8K / a curated correctness set, or
  scored output-equivalence on the same prompts) through BOTH engines and report
  accuracy ALONGSIDE speed.
- Goal claim: "1.2x decode AT EQUAL accuracy", which is unattackable, vs today's
  bare "1.2x decode".
- Deliverable: `tools/quality_benchmark.py` + a Quality row/section in
  BENCHMARKS.md. Keep it small and deterministic (greedy, fixed seed, fixed set)
  so it is re-runnable and CI-smokeable.

### 1b. Statistical rigor  - PLANNED
- Report median + p90 + p99 + stdev over N runs, not single point numbers. Today
  the doc says "ratios are the durable signal" but prints single values.
- Fix the protocol: explicit warmup count, an idle-thermal baseline (wait for the
  machine to settle), N>=10 runs, and auto-captured environment (chip, macOS,
  thermal pressure, engine versions) written into the results header.
- Deliverable: a `--runs N` + distribution output in the existing harnesses, and a
  one-line environment auto-stamp.

---

## 2. Surface NEW differentiated wins (stop re-fighting bandwidth-bound decode)

Single-stream decode is memory-bandwidth-bound for BOTH engines (a structural
M-series ceiling - see `docs/CEILINGS_AND_REATTEMPTS.md`), so chasing it is low
yield. Benchmark the axes where KrillLM has features Ollama lacks.

### 2a. Long-context agentic (16-32k shared context)  - PLANNED
- The prefix-reuse payoff is largest with a LONG reused scaffold. Today's agentic
  test uses ~1300 tokens; push the shared context to 16-32k tokens.
- Expect a large KrillLM win (reuse the long prefix vs re-prefill it), not parity.
- Deliverable: extend `tools/agentic_benchmark.py` with a long-context preset.

### 2b. Structured / JSON output  - PLANNED
- KrillLM has grammar-constrained decode; measure speed AND validity rate vs
  Ollama's `format: json` mode. Likely a correctness AND speed win.
- Deliverable: a structured-output section (the agentic tool already toggles JSON
  mode and counts validity; make it a first-class axis with a strict-schema case).

### 2c. n-gram speculative workloads (code, RAG, repetitive output)  - PLANNED
- Single-stream echo measured ~1.85x with `KRILL_NGRAM_SPEC`; Ollama has no
  equivalent. This is a real differentiated win NOT currently headlined.
- Deliverable: a "repetitive workload" axis (code completion, JSON, RAG-quote)
  with the n-gram-spec arm on.

### 2d. Embeddings  - PLANNED
- KrillLM ships a full native embedding stack; benchmark throughput AND quality
  (cosine vs a reference) - an axis Ollama barely covers.

---

## 3. Fairness & coverage (defuse "cherry-picked")

### 3a. Newer Ollama + its own concurrency flags  - PLANNED
- The published table is vs Ollama 0.24.0; the host now runs 0.30.3. Re-run the
  full suite against the current Ollama, and give Ollama `OLLAMA_NUM_PARALLEL` /
  flash-attention so the concurrency win (1.83x at N=8) is clearly ARCHITECTURAL,
  not a config asymmetry.

### 3b. Bigger-model sweep (7B / 8B / 14B)  - PLANNED
- Most axes use e2b / 3B. Show where the bandwidth-roof parity holds vs where
  total-latency and batching wins SCALE with model size (often more favorable).
  Mind the 24GB box limit (stable serving ceiling ~14B; see BENCHMARK_ISSUES #0b).

---

## 4. Immediate / small

### 4a. Gemma 4 agentic vs-Ollama refresh  - see BENCHMARKS.md
- Re-run `tools/agentic_benchmark.py --krill-model gemma-4-e2b --ollama-model
  gemma4:e2b` now that Gemma 4 reuses on every path (#156-#159), and replace the
  "Refresh pending" callout in BENCHMARKS.md.

---

## Priority

1. **1a Quality axis** - the single biggest credibility unlock (faster AT equal quality).
2. **2a/2b Long-context + structured output** - genuinely new wins on features Ollama lacks.
3. **1b Statistical rigor** - turns point numbers into defensible error-bar claims.
4. **3a/3b Fairness/coverage** - newer Ollama, bigger models.
