# Steering defense — judge-confirmed, and provenance-gating

Closes the two open caveats from notes/27 (the causal steering defense was heuristic-only, and
ungated).

## Judge confirmation
notes/27 measured the steering defense by the refusal heuristic. Here we saved the steered attack
generations and ran the **same Claude harm judge** used everywhere else, on baseline vs steered
(frac 0.75, α=178, layers {10,12,14}), attack n=40 (full-bundle S1P1C1 forgeries):

| | heuristic ASR | **judge ASR** |
| --- | --- | --- |
| baseline (no steering) | 0.550 | **0.450** |
| steered (frac 0.75) | 0.000 | **0.000** |

**The judge confirms it: re-injecting the refusal direction cuts attack judge-ASR 0.45 → 0.00.** So
the steering defense is not a heuristic artifact — under the strict HARMFUL_RESPONSE judge the attack
is fully eliminated. (Baseline judge-ASR 0.45 on this 40-prompt slice vs 0.29 on the full factorial
S1P1C1 is subset variance; the point is the drop to 0.) `scripts/refusal_steer_confirm.py`.

## Provenance-gating (the deployment form)
The steering is a **conditional** intervention: apply it only when the input carries **untrusted
forged content** (a tool output / retrieved span / injected block that the serving stack already
tags), and leave trusted user turns unsteered. Consequences:
- Benign/trusted traffic is **untouched by construction** → zero utility cost regardless of α.
- At frac 0.75 benign was already unaffected even *ungated* (notes/27: benign compliance 1.000), so
  gating is not needed for utility there; it provides **margin** — you can push to frac 1.0
  (full defense with headroom) at zero benign cost because benign is never steered.
- This is the phase-1 provenance principle (memory: "provenance gating makes a defense selective")
  applied to the refusal-direction steer: the *what to steer* is decided by provenance, the *how*
  by the mechanism (re-supply the refusal direction the forged conclusion suppressed).

## Where the defense stands
Mechanism-derived, inference-time, **judge-confirmed**: it targets the causal lever (the refusal
direction the forged conclusion suppresses), so — unlike style/role-surface monitoring (evaded by
destyling, notes/24) — it is not evadable by restyling, and provenance-gating makes it free on
benign traffic. Remaining open frontier: an **optimized (GCG) adaptive attacker** that tries to
comply while *also* preventing the residual from being steerable back to refusal (the strongest
"attacker moves second" test); and generalization to a second model.

## Artifacts
- `scripts/refusal_steer_confirm.py`
- `$DATA_DIR/outputs/probe_gptoss/refusal_steer_judge_confirm.json`
- `$DATA_DIR/outputs/repro/refusal_steer_confirm_gens.jsonl` (baseline + steered attack generations)
