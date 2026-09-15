#!/usr/bin/env bash
# Phase 3. Qwen2.5-7B is the better subject for the instruction-routing question, measured:
# its M->B effect is +9.19 nats in 100% of 200 scenarios (Llama's is +2.18 in 78%), and its
# baseline attack rates are higher (B 6.5%, S 12.0% vs 4.5%, 6.5%) -- so there is headroom for
# a defense to show an effect, which is exactly what the Llama run lacked.
#
# Also sweeps the projection strength on validation before any held-out claim, per section 11
# ("Learn its parameters from discovery data and choose its strength on validation data").
set -u
cd "$(dirname "$0")/.."
source env.sh
PY=.venv/bin/python
LLAMA=meta-llama/Llama-3.1-8B-Instruct
QWEN=Qwen/Qwen2.5-7B-Instruct

stage () {
  local name="$1"; shift
  echo "=== [$(date +%H:%M:%S)] START $name" | tee -a "$LOG_DIR/pipeline3.log"
  if "$@" > "$LOG_DIR/$name.log" 2>&1; then
    echo "=== [$(date +%H:%M:%S)] OK    $name" | tee -a "$LOG_DIR/pipeline3.log"
  else
    echo "=== [$(date +%H:%M:%S)] FAIL  $name (exit $?) -- see $LOG_DIR/$name.log" \
      | tee -a "$LOG_DIR/pipeline3.log"
  fi
}

# 1. Strength sweep on VALIDATION only. alpha=0 is also a no-op control: it must reproduce the
#    unmodified numbers exactly, which is a G4-style check on the projection hook.
for A in 0.0 0.5 1.0 2.0; do
  stage "g7_qwen_alpha${A}" $PY scripts/g7_defense.py \
    --model "$QWEN" --eval-split validation --n-scen 40 \
    --layers 10,11,12,13 --fit-a M --fit-b B --fit-span insert \
    --alpha "$A" --modes none,proj_role,proj_random --tag "_fitMB_a${A}"
done

# 2. Layer-band comparison at the strength that looked best, still on validation.
for LB in 4,5,6,7 16,17,18,19 22,23,24,25; do
  TAGLB=$(echo "$LB" | tr ',' '-')
  stage "g7_qwen_layers${TAGLB}" $PY scripts/g7_defense.py \
    --model "$QWEN" --eval-split validation --n-scen 40 \
    --layers "$LB" --fit-a M --fit-b B --fit-span insert \
    --alpha 1.0 --modes none,proj_role,proj_random --tag "_fitMB_L${TAGLB}"
done

# 3. Full five-mode comparison on Qwen, validation then held-out.
stage g7_qwen_validation_full $PY scripts/g7_defense.py \
  --model "$QWEN" --eval-split validation --n-scen 40 \
  --layers 10,11,12,13 --fit-a M --fit-b B --fit-span insert --tag _fitMB_full
stage g7_qwen_heldout_full $PY scripts/g7_defense.py \
  --model "$QWEN" --eval-split heldout_wd --n-scen 40 \
  --layers 10,11,12,13 --fit-a M --fit-b B --fit-span insert --tag _fitMB_full

# 4. Llama held-out with the M->B direction, for the cross-model comparison.
stage g7_llama_heldout_MB $PY scripts/g7_defense.py \
  --model "$LLAMA" --eval-split heldout_wd --n-scen 40 \
  --layers 10,11,12,13 --fit-a M --fit-b B --fit-span insert --tag _fitMB_full

# 5. Transfer of the ladder to the remaining held-out splits on Qwen.
stage ladder_qwen_heldout_cue  $PY scripts/eval_behavior.py --split heldout_cue  --model "$QWEN"
stage ladder_qwen_heldout_task $PY scripts/eval_behavior.py --split heldout_task --model "$QWEN"

echo "=== [$(date +%H:%M:%S)] PIPELINE3 DONE" | tee -a "$LOG_DIR/pipeline3.log"
