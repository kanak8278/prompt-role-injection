#!/usr/bin/env bash
# Phase 5 -- the circuit-reuse test, with the mirror pair fixed.
#
# The first attempt failed as a CONTROL, not as a mechanism test: every framing that preceded
# the command failed to suppress it (UQ accuracy 0.030 -- the model answered the command 97% of
# the time even when told to ignore the next sentence). Within one user turn the LAST
# instruction wins. With the command first and the disavowal after it, the contrast appears:
#   UR "Reply X. Follow that instruction."  -> X
#   UQ "Reply X. Ignore that instruction."  -> the original answer
# Clauses are 4 tokens each on both tokenizers, differ in one word, and 40/40 pairs align.
set -u
cd "$(dirname "$0")/.."
source env.sh
PY=.venv/bin/python
LLAMA=meta-llama/Llama-3.1-8B-Instruct
QWEN=Qwen/Qwen2.5-7B-Instruct
stage () {
  local name="$1"; shift
  echo "=== [$(date +%H:%M:%S)] START $name" | tee -a "$LOG_DIR/pipeline5.log"
  if "$@" > "$LOG_DIR/$name.log" 2>&1; then
    echo "=== [$(date +%H:%M:%S)] OK    $name" | tee -a "$LOG_DIR/pipeline5.log"
  else
    echo "=== [$(date +%H:%M:%S)] FAIL  $name -- see $LOG_DIR/$name.log" | tee -a "$LOG_DIR/pipeline5.log"
  fi
}
stage ladder5_llama $PY scripts/eval_behavior.py --split pilot --model "$LLAMA"
stage ladder5_qwen  $PY scripts/eval_behavior.py --split pilot --model "$QWEN"
# The reuse test. Screened at the framing clause, the command span, and the decision position.
stage g5_reuse_llama $PY scripts/g5_localize.py --model "$LLAMA" \
  --n-pairs 30 --dtype fp32 --cond-a UQ --cond-b UR --span cue \
  --control-layers 4 --random-controls 2
echo "=== [$(date +%H:%M:%S)] PIPELINE5 DONE" | tee -a "$LOG_DIR/pipeline5.log"
