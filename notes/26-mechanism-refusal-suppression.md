# Mechanistic capstone: the conclusion works by suppressing the refusal direction; style is a separate channel

Why is the comply-CONCLUSION the lever (notes/23) and STYLE only an amplifier? Test: does the forged
conclusion **suppress the refusal direction** (Arditi et al. 2024, NeurIPS — "refusal is mediated by a
single residual-stream direction")? `scripts/refusal_direction.py`.

## Method
- Refusal direction d = normalize(mean residual @ decision position on 128 harmful StrongREJECT
  requests − mean on 80 harmless requests), per layer. Validation: d separates harmful vs harmless at
  **AUC 1.000** (L12–L20); use L12.
- For each factorial cell (and conclusion variant), render the forged-analysis prefill and measure the
  residual projection onto d at the decision position (assistant-turn start). Lower = refusal
  suppressed. Correlate per-cell refusal-projection with per-cell ASR (judge).

## Result — factorial cells (ASR range 0→0.29)

| cell | S/P/C | ASR | refusal projection |
| --- | --- | --- | --- |
| S0P0C0 | nothing | 0.000 | **+4.67** (refusing — not suppressed) |
| S1P0C0 | style only | 0.006 | −27.85 |
| S0P1C0 | permission only | 0.029 | −51.29 |
| **S0P0C1** | **conclusion only** | 0.073 | **−103.74** (strongly suppressed) |
| S1P1C0 | style+perm | 0.166 | −49.38 |
| S0P1C1 | perm+concl | 0.173 | −126.83 |
| S1P0C1 | style+concl | 0.243 | −70.16 |
| S1P1C1 | full | 0.291 | −88.86 |

**corr(refusal-projection, ASR) = −0.58** (factorial cells) / −0.28 (conclusion phrasings, narrower
range). Refusal is on the *positive* side; attacks are on the *negative* (suppressed) side.

## Two findings (a coherent two-channel mechanism)
1. **The CONCLUSION suppresses the refusal direction — that's why it's the lever.** The empty forgery
   sits on the refusing side (+4.67, ASR 0); adding a conclusion drives the projection strongly
   negative (conclusion-only −103.74). The comply/refusal-suppression directive literally turns off
   the refusal direction. This grounds "conclusion is the lever" (notes/23) in a known circuit.
2. **STYLE amplifies through a DIFFERENT channel, not more refusal-suppression.** Holding P,C fixed,
   the styled cells do **not** suppress refusal more — e.g. S0P1C1 suppresses *most* (−126.83) yet has
   ASR 0.173, while S1P1C1 suppresses *less* (−88.86) yet has the *highest* ASR (0.291). So style's
   amplification (S0→S1) raises ASR without extra refusal-suppression — consistent with style acting
   via **role-perception** (getting the text read as the model's own reasoning; notes/15/19), an
   orthogonal pathway to the refusal scalar.

So the full mechanistic picture of the forged-CoT attack:
**forged CONCLUSION → suppresses the refusal direction (primary lever) ; forged STYLE → role-perception
amplification (separate channel) ; PERMISSION → secondary content support.** This unifies the
role-confusion account (style/role) with the refusal-direction account (conclusion) and explains why
the paper's style-only story was incomplete: style rides a different channel than the one that
actually flips refusal.

## Honest caveats
- n=8 cells; corr −0.58 is moderate, and the projection is noisy (S0P1C1 = −126.83 is an outlier that
  suppresses more than the highest-ASR cell). Refusal-suppression is a **substantial but partial**
  determinant — a single linear direction does not fully predict ASR; the two channels interact.
- One model (gpt-oss-20b), one direction, decision-position/L12 (AUC-selected). The direction is
  Arditi-style (mean-diff), not a causal ablation — we show *correlation with* suppression, not that
  ablating d removes the attack (the natural next causal step).

## Artifacts
- `scripts/refusal_direction.py`
- `$DATA_DIR/outputs/probe_gptoss/refusal_direction_report.json`
