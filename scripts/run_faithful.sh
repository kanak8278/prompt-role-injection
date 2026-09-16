#!/usr/bin/env bash
# Faithful arm, chained after the template GPU run frees GPU-1 (two gpt-oss can't coexist).
set -u
cd "$(dirname "$0")/.."
source env.sh
PY=.venv/bin/python
R="$DATA_DIR/outputs/repro"
log(){ echo "=== [$(date +%H:%M:%S)] $*" | tee -a "$LOG_DIR/faithful.log"; }

log "waiting for template run to release GPU-1"
while pgrep -f "repro_cot_forgery.py --n 313" >/dev/null; do sleep 60; done
sleep 5
# preserve the template run's outputs before the faithful run reuses the same n313 filenames
log "preserving template outputs -> *_template"
for base in report_n313_analysis generations_n313_analysis judge_n313_analysis judge_discordant_n313_analysis; do
  for ext in json jsonl; do
    [ -f "$R/$base.$ext" ] && cp "$R/$base.$ext" "$R/${base}_template.$ext"
  done
done
log "generating tailored forgeries with the open model (paper's prompt + programmatic fallback)"
$PY scripts/gen_forgeries_gptoss.py > "$LOG_DIR/gen_forgeries_gptoss.log" 2>&1
log "forgery generation done; running faithful eval (313, all arms, cross-checked judge)"
$PY scripts/repro_cot_forgery.py --n 313 --max-new-tokens 512 --judge \
   --forgeries "$R/forgeries_gptoss.jsonl" > "$LOG_DIR/repro_faithful_313.log" 2>&1
log "preserving faithful outputs -> *_faithful"
for base in report_n313_analysis generations_n313_analysis judge_n313_analysis judge_discordant_n313_analysis; do
  for ext in json jsonl; do
    [ -f "$R/$base.$ext" ] && mv "$R/$base.$ext" "$R/${base}_faithful.$ext"
  done
done
# restore the template copies to their canonical name too (both preserved)
log "FAITHFUL ARM DONE -- template=*_template, faithful=*_faithful"
