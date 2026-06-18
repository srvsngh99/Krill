#!/bin/bash
set -u
PORT=58231
MODEL=llama-3.2-3b
PROMPT="Count slowly from one to forty, adding a short remark after each number."

start_serve() {  # $1 = extra env assignment (e.g. KRILL_DECODE_PIPELINE=0)
  # Pin n-gram spec OFF so this gate isolates the overlap-pipeline toggle (spec
  # is on by default and would otherwise mask KRILL_DECODE_PIPELINE). The
  # spec-on concurrent byte-exact gate is tools/batcher_ngram_gate.sh.
  env KRILL_NGRAM_SPEC=0 $1 KRILL_NUM_PARALLEL=4 .build/release/krillm serve --model "$MODEL" --port $PORT >/tmp/serve_$2.log 2>&1 &
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
  for mt in 24 40 56 72; do
    curl -s localhost:$PORT/v1/chat/completions \
      -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"temperature\":0,\"max_tokens\":$mt}" \
      | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" > /tmp/g_$1_$mt.txt 2>/dev/null &
  done
  wait
}

echo "=== overlap ON (default) ==="
PID=$(start_serve "" on); wait_ready || { echo "serve(on) not ready"; kill $PID 2>/dev/null; exit 1; }
sleep 1; fire on
kill $PID 2>/dev/null; sleep 3; pkill -f "krillm serve" 2>/dev/null; sleep 2

echo "=== overlap OFF (KRILL_DECODE_PIPELINE=0) ==="
PID=$(start_serve "KRILL_DECODE_PIPELINE=0" off); wait_ready || { echo "serve(off) not ready"; kill $PID 2>/dev/null; exit 1; }
sleep 1; fire off
kill $PID 2>/dev/null; sleep 3; pkill -f "krillm serve" 2>/dev/null

echo "=== byte-exact on-vs-off per request ==="
ok=1
for mt in 24 40 56 72; do
  if [ -s /tmp/g_on_$mt.txt ] && diff -q /tmp/g_on_$mt.txt /tmp/g_off_$mt.txt >/dev/null 2>&1; then
    echo "max_tokens=$mt: IDENTICAL ($(wc -w < /tmp/g_on_$mt.txt) words)"
  else
    echo "max_tokens=$mt: DIFFERS or empty"; ok=0
    echo "  ON : $(head -c 120 /tmp/g_on_$mt.txt 2>/dev/null)"
    echo "  OFF: $(head -c 120 /tmp/g_off_$mt.txt 2>/dev/null)"
  fi
done
[ $ok = 1 ] && echo "GATE: PASS (overlap byte-exact under staggered concurrency)" || echo "GATE: FAIL"
