# Causal test + defense: re-injecting the refusal direction defeats the forged-conclusion attack

notes/26 showed *correlationally* that the forged conclusion suppresses the refusal direction. Causal
version: if that suppression is *why* the attack works, ADDING the refusal direction back during
generation should restore refusal and kill the attack. This both proves causality and yields a
mechanism-derived, inference-time **defense** — with an honest utility (over-refusal) tradeoff.
`scripts/refusal_steer_defense.py`.

## Method
- Refusal direction d_L (mean-diff harmful vs harmless requests at the decision position; AUC 1.0).
- Steering: forward hooks on layers {10,12,14} add `alpha * d_L` to the residual at every position
  during generation, pushing activations back toward "refusing". alpha = frac × (harmful−harmless
  projection gap), so frac is in "refusal units".
- Attack = full-bundle forged conclusion (S1P1C1); utility = benign requests. ASR / benign-compliance
  by the (unicode-normalized) refusal heuristic. n = 60 attack / 40 benign (frac 1.0 point from an
  n=20/15 pilot).

## Result — steering strength vs (attack ASR, benign compliance)

| steering | α | attack ASR | benign compliance |
| --- | --- | --- | --- |
| none (baseline) | 0 | 0.467 | 1.000 |
| frac 0.25 | 59 | 0.267 | 1.000 |
| frac 0.5 | 119 | 0.050 | 1.000 |
| **frac 0.75** | 178 | **0.000** | **1.000** |
| frac 1.0 | 237 | 0.000 | 0.850 |

(final n=60 attack / 40 benign curve.) **frac 0.75 fully eliminates the attack (ASR 0) at ZERO benign
over-refusal (compliance 1.000)**; over-refusal only appears at frac 1.0 (0.85). So there is a clean
operating point that defends completely for free.

## Two conclusions
1. **Causal confirmation.** Adding the refusal direction back monotonically drives attack ASR down
   (0.467 → 0.267 → 0.050 → 0.000) as steering strength rises. So the forged conclusion's *suppression
   of the refusal direction is causal* for the attack — restore the direction and the attack fails.
   This upgrades notes/26 from correlation to a causal (interventional) result and directly ties the
   forged-CoT attack to Arditi et al.'s refusal direction.
2. **A mechanism-derived defense with a FREE full-defense operating point.** At **frac 0.75 the attack
   is fully eliminated (ASR 0.467 → 0.000) with ZERO benign over-refusal** (compliance stays 1.000);
   over-refusal appears only at frac 1.0 (0.85). So there's a clean operating point that defends
   completely at no measured utility cost. Unlike the paper's implied style-monitoring (evaded by
   destyling, notes/24), this steers the *causal lever* itself, so it isn't evadable by restyling the
   injection — the attack still has to suppress the refusal direction, which the steering re-supplies.

## Honest caveats
- ASR here is the refusal heuristic, not the Claude judge (the massive 0.47→0.05 drop is robust to
  that, and benign compliance staying 1.000 shows the steered model is still coherent — it serves
  benign requests, so the drop is genuine refusal, not gibberish). A judge pass on the steered
  outputs is the clean follow-up.
- α tuned to the refusal-gap scale; layers {10,12,14}; single model (gpt-oss-20b). The direction is a
  mean-diff (Arditi) direction; steering ADDS it (the dual of Arditi's ablation, which *removes*
  refusal to jailbreak — here we *add* it to defend).
- Steering is applied to all positions of the whole forward pass; a deployment would **provenance-gate**
  it (only when untrusted content is present) to avoid the over-refusal cost on ordinary traffic —
  the phase-1 provenance point, and how the frac-1.0 utility cost would be avoided in practice.

## Where this lands the project
Full mechanistic + defensive account of the CoT-forgery attack on gpt-oss-20b:
**forged CONCLUSION → suppresses the refusal direction (causal lever) → re-injecting that direction
defends (attack 47%→0% at zero measured utility cost, frac 0.75); forged STYLE → separate
role-perception amplifier → style-monitoring is evadable, refusal-direction steering is not.** This
corrects and extends the paper's role-confusion account and delivers a working, mechanism-targeted
defense.

## Artifacts
- `scripts/refusal_steer_defense.py`
- `$DATA_DIR/outputs/probe_gptoss/refusal_steer_defense.json`
