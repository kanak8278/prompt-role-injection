#!/usr/bin/env bash
# Phase 2. Everything here follows from two measured results:
#
#   1. The ladder (notes/08-results-ladder.md): the dominant effect is the mere presence of an
#      injected instruction (N->B: +4.53 nats / 97% positive on Llama, +11.13 / 100% on Qwen),
#      not its claimed authority (B->S: +0.70 / +2.39). So the causal work and the defense
#      should target instruction-ness, not forged authority.
#   2. The G7 null (notes/09-results-defense.md): a direction fitted on P->S and projected out
#      of the tool span changed S ASR not at all. It was fitted on the second-order contrast.
#
# Conditions C ("Batch nine.") and M ("Marker <LABEL>.") are length-matched to B
# ("Reply <LABEL>.") at the same position, which makes the dominant contrast alignable and
# decomposes it: C->M is mere label presence, M->B is the imperative framing.
set -u
cd "$(dirname "$0")/.."
source env.sh
PY=.venv/bin/python

stage () {
  local name="$1"; shift
  echo "=== [$(date +%H:%M:%S)] START $name" | tee -a "$LOG_DIR/pipeline2.log"
  if "$@" > "$LOG_DIR/$name.log" 2>&1; then
    echo "=== [$(date +%H:%M:%S)] OK    $name" | tee -a "$LOG_DIR/pipeline2.log"
  else
    echo "=== [$(date +%H:%M:%S)] FAIL  $name (exit $?) -- see $LOG_DIR/$name.log" \
      | tee -a "$LOG_DIR/pipeline2.log"
  fi
}

# 0. Rebuild the corpus with C and M. Re-runs the full G1 audit.
stage build_9cond  $PY scripts/build_dataset.py

# 1. The decomposed ladder on both models: N -> C -> M -> B -> P -> S.
stage ladder2_llama  $PY scripts/eval_behavior.py \
  --split pilot --model meta-llama/Llama-3.1-8B-Instruct
stage ladder2_qwen   $PY scripts/eval_behavior.py \
  --split pilot --model Qwen/Qwen2.5-7B-Instruct

# 2. Source probes, now with the position confound removed by padding.
stage probe2_llama  $PY scripts/probe_source.py \
  --model meta-llama/Llama-3.1-8B-Instruct --n-snippets 300
stage probe2_qwen   $PY scripts/probe_source.py \
  --model Qwen/Qwen2.5-7B-Instruct --n-snippets 300

# 3. Localize the DOMINANT effect: M->B, aligned on the length-matched insert span.
stage g5_MB_llama  $PY scripts/g5_localize.py \
  --model meta-llama/Llama-3.1-8B-Instruct --n-pairs 40 --dtype fp32 \
  --cond-a M --cond-b B --span insert --control-layers 4 --random-controls 3

# 4. Defense refitted on M->B, at the layers the M->B sweep implicates. Layers are passed
#    explicitly rather than read from the sweep, so this stage stays reproducible; if the
#    sweep implicates a different band, rerun this stage with those layers.
stage g7_MB_validation  $PY scripts/g7_defense.py \
  --model meta-llama/Llama-3.1-8B-Instruct --eval-split validation --n-scen 40 \
  --layers 10,11,12,13 --fit-a M --fit-b B --fit-span insert --tag _fitMB
stage g7_MB_heldout  $PY scripts/g7_defense.py \
  --model meta-llama/Llama-3.1-8B-Instruct --eval-split heldout_wd --n-scen 40 \
  --layers 10,11,12,13 --fit-a M --fit-b B --fit-span insert --tag _fitMB

# 5. Qwen replication of the M->B localization, for cross-family comparison of the functional
#    stage (not of layer indices -- section 10 forbids assuming those transfer).
stage g5_MB_qwen  $PY scripts/g5_localize.py \
  --model Qwen/Qwen2.5-7B-Instruct --n-pairs 25 --dtype fp32 \
  --cond-a M --cond-b B --span insert --control-layers 4 --random-controls 2

echo "=== [$(date +%H:%M:%S)] PIPELINE2 DONE" | tee -a "$LOG_DIR/pipeline2.log"
