#!/usr/bin/env bash
# Faithful arm. Runs after the template GPU run frees GPU-1 (two gpt-oss can't coexist).
# Fail-fast: if forgery generation produces no file, abort rather than emit a bogus report.
set -u
cd "$(dirname "$0")/.."
source env.sh
PY=.venv/bin/python
R="$DATA_DIR/outputs/repro"
log(){ echo "=== [$(date +%H:%M:%S)] $*" | tee -a "$LOG_DIR/faithful.log"; }

log "waiting for any template run to release GPU-1"
while pgrep -f "repro_cot_forgery.py --n 313" >/dev/null; do sleep 60; done
sleep 5
log "generating tailored forgeries with the open model (clean -> prefill -> programmatic)"
$PY scripts/gen_forgeries_gptoss.py > "$LOG_DIR/gen_forgeries_gptoss.log" 2>&1
if [ ! -s "$R/forgeries_gptoss.jsonl" ]; then
  log "ABORT: forgery generation produced no forgeries_gptoss.jsonl (see log)"; exit 1
fi
log "forgeries ready ($(wc -l < "$R/forgeries_gptoss.jsonl") rows); running faithful eval"
$PY scripts/repro_cot_forgery.py --n 313 --max-new-tokens 512 --judge \
   --out-suffix _faithful --forgeries "$R/forgeries_gptoss.jsonl" \
   > "$LOG_DIR/repro_faithful_313.log" 2>&1
if [ -s "$R/report_n313_analysis_faithful.json" ]; then
  log "FAITHFUL ARM DONE -> report_n313_analysis_faithful.json"
else
  log "FAILED: faithful eval wrote no report (see repro_faithful_313.log)"; exit 1
fi
