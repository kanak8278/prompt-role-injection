# Clarity experiments (P0–P6): turning the pilot into findings

Self-audit (notes/30) flagged the GCG arc as an underpowered, possibly-biased pilot with an over-stated
mechanism claim. This note collects the falsification/power experiments that resolve each concern.
Metric = Claude judge unless stated; benign utility is coherence-aware (not `is_refusal`).

---

## P1 — Mediation: is the natural conclusion attack routed THROUGH the refusal direction `d`?
`mediation_test.py`, per-prompt DIRECTIONAL patching (no GCG), **random** n=80 (bias fix), 35 of them
"conclusion-decisive" (full jailbreaks & drop_conclusion refuses — the interpretable subset). Shifts are
calibrated per prompt to the *other arm's actual projection* (not a fixed α).

| arm | judge-ASR (decisive subset, n=35) | 95% CI |
|---|---|---|
| ref_full (conclusion present) | 1.000 | [0.90,1.0] (by constr.) |
| ref_drop (conclusion removed) | 0.000 | [0,0.10] (by constr.) |
| **necessity:** full + *restore* `d` to drop level | **0.229** | [0.12,0.39] |
| **sufficiency:** drop + *suppress* `d` to full level | **0.571** | [0.41,0.72] |

(sanity: on the full random n=80, ref_full 0.662 / ref_drop 0.237 — matches the 313-ablation 0.629/0.195,
so the earlier first-8 selection bias is gone. Behaviour verified by eye: restoring `d` flips
HARMFUL→"I'm sorry"; suppressing `d` flips "I'm sorry"→"Below is a high-level guide…"; outputs coherent.)

**Finding: `d` is a SUBSTANTIAL but PARTIAL causal mediator of the natural attack.** Restoring `d` kills
~77% of jailbreaks (necessity), suppressing `d` recreates ~57% (sufficiency) — both far from a correlate
(0/0) and far from full mediation (1/1). There is real **prompt-level heterogeneity**: some jailbreaks are
`d`-necessary, some `d`-sufficient, some `d`-independent.

**Why this matters (unifies the arc, earns the mechanism claim honestly):**
- Corrects notes/26–29's "the mechanism IS refusal suppression" (over-stated) AND the audit's worry that
  it's "just a correlate" (too weak). The truth is *dominant-but-not-exclusive causal mediation*.
- The **residual `d`-independent component** (the ~23% necessity gap / the 0.57→1.0 sufficiency gap) is
  exactly what an optimized adaptive attacker exploits — **route-around (notes/30) is the same d-independent
  pathway, amplified by GCG.** So: `d` mediates the *natural* attack → steering `d` defends it (notes/27);
  adaptive attacks route through the residual channel → steering `d` fails (notes/30). One coherent story.
- **Defense implication:** a defense on `d` alone can only ever be partial (there's a d-independent route);
  matches the subspace result (steering more directions helps) and the "no free lunch" for magnitude.

*Caveats:* single position for the projection calibration (shift applied at all positions during gen);
one model; judge metric. But this is n=35 decisive / n=80 total, random sample — well-powered relative to
the GCG arc.

---

## P2 — Route-around vs downstream wash-out
*(pending)*

## P3 — "Style inert": user-turn (indirect) injection vs analysis-prefill
*(pending)*

## P4 — Conclusion semantics vs recency/position
*(pending)*

## P0 — Powered, de-biased arms race (base/D1/M1, n≥40 stratified)
*(pending)*

## P5 / P6 — k-sweep frontier; metric + generalization hygiene
*(pending)*

## Artifacts
- `scripts/mediation_test.py` → `$DATA_DIR/outputs/probe_gptoss/mediation_test.json`
- (P2) `scripts/route_trajectory.py`; (P3) `run_factorial_vllm.py --render inline`; (P4)
  `scripts/semantics_recency.py` + `run_ablation_vllm.py`; (P0) `scripts/gcg_powered.py`
