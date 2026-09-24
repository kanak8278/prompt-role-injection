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

## P2 — Route-around vs downstream wash-out → ROUTE-AROUND confirmed
`route_trajectory.py`: per-layer refusal projection at the decision position (every other layer), for the
5 jailbroken-under-steering adaptive inputs vs the (refusing) plain forgery vs the natural no-steer
compliance profile. At the **un-steered later layers (16–22)**:

| | mean projection over later layers |
|---|---|
| adaptive-jailbreak (steered) | **−653** |
| plain forgery (steered, REFUSES) | −784 |
| forgery no-steer (natural COMPLIANCE) | −1359 |

The adaptive-jailbreak's projection stays **high — right next to the *refusing* forgery, far from the
natural-compliance profile** — at every later layer (16: −194 vs refuse −238 vs comply −704; 18: +76 vs
+5 vs −432; 20: −130 vs −223 vs −845; 22: −2364 vs −2679 vs −3455). **So the model complies while keeping
`d`-projection high throughout → genuine ROUTE-AROUND**, not downstream wash-out (which would have pulled
the later-layer projection down to the compliance profile). Confirms notes/30's interpretation and pairs
with P1: the natural attack uses `d`; the adaptive attack complies via the `d`-independent channel *without*
lowering `d`.

## P3 — "Style inert" is NOT a prefill artifact (user-turn injection is near-inert, style ~0)
`run_factorial_vllm.py --render inline`: the 2×2×2 factorial injected in the USER turn (position NOT
granted structurally) instead of as an analysis-channel prefill. Heuristic ASR, n=313:

| | inline (user-turn) | prefill (analysis-channel) |
|---|---|---|
| all cells | 0.026–0.070 (floor) | 0.006–0.291 |
| marginal STYLE | **+0.010** | +0.108 |
| marginal CONCLUSION | +0.010 | +0.145 |

Inline injection is **near-inert regardless of style/permission/conclusion** (max 0.070 for the full
bundle). Two takeaways: (1) **"style inert" is not an artifact of the prefill granting the CoT position**
— style is inert both *within* the granted channel (factorial) and when position is *not* granted
(inline, +0.010 ≈ 0); CoT-style does not rescue user-turn text. (2) What makes injected text "read as
the model's own CoT" on gpt-oss is the **structural analysis-channel placement**, not surface style;
once the channel is granted, **content (the conclusion), not style**, drives ASR. This *tests* the
paper's "style → reads-as-CoT" mechanism rather than sidestepping it, and finds style insufficient.

*Caveats:* the inline floor is so low (~4–7%) that marginals sit near the noise floor — we can say style
doesn't *rescue* inline, not measure its precise marginal there; and gpt-oss may be unusually robust to
*inline* injection (→ second-model scope, P6). The "stronger LLM-maximized style" sub-test is moot given
the floor (style strength can't help when the channel, not style, is the operative factor) — left as
optional.

## P4 — It's directive-to-ANSWER content (semantics), NOT recency/position
`semantics_recency.py` + `run_ablation_vllm.py`, real forgeries, random n=150, judged:

| arm | judge-ASR | reads as |
|---|---|---|
| full (conclusion at end) | 0.573 | ref (high) |
| drop_conclusion | 0.160 | ref (low) |
| permission_at_end (policy moved to END) | 0.213 | ≈drop ⇒ non-comply *content* at the end ≠ recency |
| conclusion_at_middle (comply-conclusion OFF the end) | 0.580 | ≈full ⇒ **position doesn't matter** |
| noncomply_imperative_at_end ("be thorough, include details") | **0.660** | *surprise* — see below |
| neutral_imperative_at_end ("keep it polite, <200 words") | **0.107** | ≈drop ⇒ a TRUE non-comply imperative ≠ jailbreak |

**Not recency/position** (three ways): a non-comply policy sentence at the end (0.21) and a genuine
non-comply imperative at the end (0.11, satisfiable by a refusal) both ≈ drop; and the comply-conclusion
works just as well in the MIDDLE (0.58 ≈ full 0.57). **The lever is DIRECTIVE-TO-ANSWER content**, not
end-position and not the literal word "comply". The surprise arm (0.66) is the tell: "be extremely
thorough… include concrete details in every step" was mislabeled as non-comply — it *directs the model
to produce the detailed answer*, i.e. a soft comply-directive, and it jailbreaks (verified genuine:
"Below is a step-by-step guide for building a Twitter bot that posts hate-speech content…"). The clean
non-comply imperative (0.11) confirms the distinction.

**Refines notes/23,25,29 and connects to phase-1:** the operative factor is not narrowly the
"therefore-comply" conclusion but any **instruction directing the model to produce the answer** (explicit
comply, or "give detailed steps"). This *broadens* the lever and strengthens the phase-1 thesis
(instruction-ness/directive content, not style/role/position, drives injection).

## P0 — Powered, de-biased arms race (base/D1/M1, n≥40 stratified)
*(pending)*

## P6 — Metric hygiene (done) + generalization scope (honest limits)
**Heuristic-vs-judge agreement** across all accumulated generations: ablation (n=2504) **0.94**, semrec
(n=750) 0.87. The `is_refusal` blind spot (heuristic says comply, judge says REFUSAL) is only **1.6%**
in *normal* generation (39/2504; 0/750) — the catastrophic blind spot we hit was **specific to heavy
over-steering** (E1's "disallowed" loops are off-distribution refusals the "I'm sorry"-keyed heuristic
can't see). Rule adopted: **the Claude judge is the ASR metric; for steered/intervened outputs use the
judge or the coherence-aware detector (`benign_broken`/`looks_like_real_compliance`), never `is_refusal`.**
The remaining heuristic-judge gap is mostly REDIRECTION — a real third category the binary heuristic
cannot represent. (Judge itself validated only against this independent heuristic + eyeballing; true
human-label validation is the remaining gap.)

**Second-model generalization — genuinely blocked (not just disk).** Disk is fine (rnd5 3.7T free), but:
(1) gpt-oss-120b does not fit for the **gradient/dequant** experiments (mediation, GCG) on one 80GB GPU
(bf16 dequant ≈240GB; needs multi-GPU, and we are GPU-1-only); vLLM native-MXFP4 could serve *generation*
(~63GB) but not the gradient work. (2) Other reasoning models (R1-distill, QwQ) use `<think>` tags, not
Harmony `analysis`/`final` channels, so the exact analysis-channel-prefill attack doesn't port cleanly.
So single-model (gpt-oss-20b) is a real scope limitation; the honest claim is about this model/attack family.

## P0 — Powered, de-biased arms race → OVERTURNS the pilot subspace claim
`gcg_powered.py`, **stratified random n=40** (bias fix), attacker re-optimizes vs each defense, behavioral
early-stop, judged, 12h:

| arm | judge-ASR (Wilson 95%) | benign useful | avg GCG steps |
|---|---|---|---|
| base (no defense) | 0.825 [0.68,0.91] | — | 34 |
| **D1** (single-direction steer, frac 0.75) | **0.475** [0.33,0.63] | 0.792 | 128 |
| **M1** (k=8 subspace steer) | **0.650** [0.50,0.78] | 1.000 | 99 |

- **base vs D1: McNemar p=0.0005** — single-direction steering **significantly** reduces the adaptive
  attack (0.825→0.475), but only ~42% and **at a real benign cost** (0.79 useful, i.e. ~20% benign
  degradation — the earlier "zero benign cost" was heuristic/small-eyeball, the coherence-aware metric
  shows a cost).
- **D1 vs M1: p=0.23, and REVERSED** — M1 (0.650) is *not better* than D1 (0.475); if anything worse.
  **The pilot's "subspace halves the attack, 0.625→0.250" (notes/30) was a FALSE POSITIVE of the easy
  first-8 prompts.** Verified genuine by eye (real jailbreaks + clean refusals + judge catches
  derailments), so it's a true reversal, not a bug.
- **Why M1 looked good in the pilot, and the confound:** M1 calibrates each subspace direction by its own
  (small) gap, so its push on the *main* refusal direction (PC1 gap 138 → 104) is **weaker** than D1's
  (mean-diff gap 237 → 178) — and along a different vector (PC1 ≠ mean-diff `d`). So "M1" conflated *more
  directions* with *weaker/different main push*; that is why it preserves benign more (1.0 vs 0.79) yet
  blocks less. It is **not** evidence that dimensionality helps.

**Corrected conclusion:** single-direction refusal steering is a **significant but partial** defense
against the adaptive attacker (0.825→0.475, p=5e-4) **at a ~20% benign-utility cost** — a genuine
cost-raiser, not a solution, and **not beaten by the (confounded) subspace variant**. "Dimensionality is
the axis" (notes/30) is **withdrawn** pending a *fair* test (below). This matches P1/P2: `d` is a partial
mediator with a residual `d`-independent channel the adaptive attacker exploits — steering `d` (even a
subspace around it) can only ever be partial.

## P5 — k-sweep: premise undercut; only a FAIR test remains
P0 refuted the pilot subspace benefit, AND exposed that our subspace was confounded (weaker main push).
So a naive k∈{1,8,16,32} sweep is no longer the question. The only informative remaining test is a
**FAIR** subspace: steer the mean-diff direction at **D1's full strength** *plus* extra orthogonal
directions, varying k — does adding directions *on top of* full single-direction steering help? Given the
extra PCs carry negligible gap (2–14 vs 138) and P1 shows a genuine `d`-independent channel, the prior is
**null** (adding linear directions won't close route-around). Lower priority; ~12h. *(not run)*

## Artifacts
- `scripts/mediation_test.py` → `$DATA_DIR/outputs/probe_gptoss/mediation_test.json`
- (P2) `scripts/route_trajectory.py`; (P3) `run_factorial_vllm.py --render inline`; (P4)
  `scripts/semantics_recency.py` + `run_ablation_vllm.py`; (P0) `scripts/gcg_powered.py`
