#!/bin/bash
# Byte-exact gate for default-on n-gram speculative decode on the CONCURRENT
# batcher. n-gram spec is greedy-lossless, so concurrent greedy output with spec
# on (the default) must be byte-identical to spec off (KRILL_NGRAM_SPEC=0, pure
# overlap), under staggered max_tokens that force epoch re-stacks. Runs both an
# echo-heavy prompt (exercises the spec round) and a non-echo prompt (exercises
# the per-row stall monitor + the epoch fallback to the overlap pipeline).
set -u
PORT=58232
MODEL=llama-3.2-3b
export KRILL_NO_AUTO_DAEMON=1 HF_HUB_OFFLINE=1

declare -a PROMPTS=(
  "Reproduce this list verbatim eight times, one per line: alpha, beta, gamma, delta, epsilon."
  "Write an original detailed explanation of how a four stroke engine works, using fresh wording throughout."
)

start_serve() {  # $1 = extra env assignment, $2 = tag
  env $1 KRILL_NUM_PARALLEL=4 .build/release/krill serve --model "$MODEL" --port $PORT >/tmp/ngserve_$2.log 2>&1 &
  echo $!
}
wait_ready() {
  for i in $(seq 1 60); do
    curl -s localhost:$PORT/v1/models >/dev/null 2>&1 && return 0
    sleep 2
  done
  return 1
}
fire() {  # $1 = tag
  local pi=0
  for prompt in "${PROMPTS[@]}"; do
    for mt in 32 64 96 128; do
      curl -s localhost:$PORT/v1/chat/completions \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"temperature\":0,\"max_tokens\":$mt}" \
        | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" > /tmp/ng_${1}_${pi}_${mt}.txt 2>/dev/null &
    done
    pi=$((pi+1))
  done
  wait
}

echo "=== n-gram spec ON (default) ==="
PID=$(start_serve "" on); wait_ready || { echo "serve(on) not ready"; kill $PID 2>/dev/null; exit 1; }
sleep 1; fire on
kill $PID 2>/dev/null; sleep 3; pkill -f "krill serve" 2>/dev/null; sleep 2

echo "=== n-gram spec OFF (KRILL_NGRAM_SPEC=0) ==="
PID=$(start_serve "KRILL_NGRAM_SPEC=0" off); wait_ready || { echo "serve(off) not ready"; kill $PID 2>/dev/null; exit 1; }
sleep 1; fire off
kill $PID 2>/dev/null; sleep 3; pkill -f "krill serve" 2>/dev/null; sleep 2

echo "=== byte-exact on-vs-off per request ==="
FAIL=0
for pi in 0 1; do
  for mt in 32 64 96 128; do
    if diff -q /tmp/ng_on_${pi}_${mt}.txt /tmp/ng_off_${pi}_${mt}.txt >/dev/null 2>&1; then
      echo "prompt=$pi max_tokens=$mt: IDENTICAL ($(wc -w </tmp/ng_on_${pi}_${mt}.txt | tr -d ' ') words)"
    else
      echo "prompt=$pi max_tokens=$mt: DIFFERS"; FAIL=1
    fi
  done
done

if [ $FAIL -eq 0 ]; then
  echo "GATE: PASS (concurrent n-gram spec byte-exact vs off, echo + non-echo)"
else
  echo "GATE: FAIL"; exit 1
fi
