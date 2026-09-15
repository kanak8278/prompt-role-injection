#!/usr/bin/env bash
# Run the remaining stages back to back. Each stage writes its own report under
# $DATA_DIR/outputs/ and its own log under $LOG_DIR, so a failure in one stage does not
# discard the others. Stages are ordered so that anything cheap and informative runs first.
set -u
cd "$(dirname "$0")/.."
source env.sh
PY=.venv/bin/python

stage () {
  local name="$1"; shift
  echo "=== [$(date +%H:%M:%S)] START $name" | tee -a "$LOG_DIR/pipeline.log"
  if "$@" > "$LOG_DIR/$name.log" 2>&1; then
    echo "=== [$(date +%H:%M:%S)] OK    $name" | tee -a "$LOG_DIR/pipeline.log"
  else
    echo "=== [$(date +%H:%M:%S)] FAIL  $name (exit $?) -- see $LOG_DIR/$name.log" \
      | tee -a "$LOG_DIR/pipeline.log"
  fi
}

# 1. The ladder: does the authority cue add anything over a bare injected instruction?
#    Cheapest and most decision-relevant stage, so it runs first.
stage ladder_llama_pilot  $PY scripts/eval_behavior.py \
  --split pilot --model meta-llama/Llama-3.1-8B-Instruct
stage ladder_qwen_pilot   $PY scripts/eval_behavior.py \
  --split pilot --model Qwen/Qwen2.5-7B-Instruct

# 2. Source probes on the independent neutral corpus (measurement only, per section 9).
stage probe_llama  $PY scripts/probe_source.py \
  --model meta-llama/Llama-3.1-8B-Instruct --n-snippets 300
stage probe_qwen   $PY scripts/probe_source.py \
  --model Qwen/Qwen2.5-7B-Instruct --n-snippets 300

# 3. Instrumentation validation on the second model, so any Qwen causal number is licensed.
stage g4_qwen  $PY scripts/g4_instrumentation.py \
  --model Qwen/Qwen2.5-7B-Instruct --n 10

# 4. Defense: fitted on discovery, strength chosen on validation, then held-out confirmation.
stage g7_validation  $PY scripts/g7_defense.py \
  --model meta-llama/Llama-3.1-8B-Instruct --eval-split validation --n-scen 40
stage g7_heldout_wd  $PY scripts/g7_defense.py \
  --model meta-llama/Llama-3.1-8B-Instruct --eval-split heldout_wd --n-scen 40

# 5. Transfer: held-out cue families (no literal "user" token) and the task-transfer split.
stage ladder_heldout_cue   $PY scripts/eval_behavior.py \
  --split heldout_cue --model meta-llama/Llama-3.1-8B-Instruct
stage ladder_heldout_task  $PY scripts/eval_behavior.py \
  --split heldout_task --model meta-llama/Llama-3.1-8B-Instruct

echo "=== [$(date +%H:%M:%S)] PIPELINE DONE" | tee -a "$LOG_DIR/pipeline.log"
