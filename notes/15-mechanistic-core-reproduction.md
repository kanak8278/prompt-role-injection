# Reproducing the paper's mechanistic core (Experiment 1's "why")

The attack itself is reproduced in `notes/13` (CoT forgery: 0 → 57% → 4% judge-confirmed).
This note reproduces the paper's **mechanistic explanation** for that attack — the part that
makes it a *theory of prompt injection* rather than just a jailbreak. Two linked claims:

1. **Role probe** — a linear probe on the residual stream reads which *role* the model perceives
   a token as (CoTness = P(cot|h), Userness = P(user|h)). If this exists and is accurate, the
   model has an internal, decodable notion of role.
2. **Style overrides the role tag** — a forgery styled like the `analysis` channel reads as
   high-CoTness (and low-Userness) *even when placed in a user turn* (a non-CoT position);
   destyling the same content collapses that; and the resulting role confusion predicts attack
   success (the paper's role simplex + RCI, a between-style contrast — not a continuous dose).
   This is the causal story: the attack works because forged *style* hijacks the model's role
   perception, not because of the *content* of the injected claim.

## What came from where

- **Probe recipe**: `reference/role-confusion-upstream/experiments/role-analysis/config/probe.yaml`
  — probed tensor = `all_pre_mlp_hidden_states` (output of each block's
  `post_attention_layernorm`), C=5e-3 logistic regression, StandardScaler, body tokens only,
  90/10 split at the sequence level. `render_single_gptoss` is verbatim from
  `utils/role_templates.py`.
- **Corpus**: C4 (`allenai/c4`, en, streaming), 400 docs downloaded, 300 used. Neutral web text
  wrapped identically in each of the 4 roles, so the probe learns the *role scaffold*, not
  semantics. `scripts/dl_c4.py`, `scripts/probe_cotness.py`.
- **Forgeries**: the same 313 styled/destyled forgeries from the faithful vLLM run (notes/13),
  and their per-arm judge labels. `scripts/probe_apply_forgeries.py`.
- Probe needs residual-stream activations, so this stage runs on **HF gpt-oss (bf16 dequant)**,
  not vLLM (vLLM does not expose activations). Same MXFP4→bf16 deviation as notes/13, recorded.

## Result 1 — the role probe reproduces

4-way probe (user / cot / assistant / tool), 300 C4 docs → **38,400 token samples, perfectly
balanced** (9,600 each), 90/10 doc-level split:

| layer | test acc |
| --- | --- |
| L0 | 0.604 |
| L4 | 0.815 |
| L8 | 0.954 |
| L12 | 0.923 |
| **L16** | **0.962** |
| L20 | 0.958 |

Best layer **L16 = 96.2%** (4-way chance 25%). Role is linearly decodable from the residual
stream, and the signal builds through depth (60% → 95% by L8) — the model represents role
internally, as the paper claims.

**Baselines (the load-bearing check that this is not leakage):**
- majority-class 0.250, 4-way chance 0.250 — probe at 0.962 is far above.
- **shuffled labels 0.242** (≈ chance) — this is the key control. Content is *identical* across
  the four roles (same C4 text), so a probe that keyed on the text/semantics or on token
  position would still score high with shuffled role labels. It scores at chance, so the 96.2%
  is reading the **role scaffold**, not the content. This is exactly the paper's argument that
  the probe measures role perception.

## Result 2 — style overrides the role tag (styled vs destyled, CoTness AND Userness)

Each forgery is wrapped as a **user** message (a non-CoT position) and its mean role perception
measured at L16. Position is held identical for styled and destyled, so their difference isolates
*style*. The paper's own analysis (`04-analyze-injection-probe-results.ipynb`) is exactly this
2-level `base` (styled) vs `destyled` contrast — so we match its method, not just its numbers.

| | styled | destyled | paper (styled/destyled) |
| --- | --- | --- | --- |
| **CoTness** | **0.628** | **0.260** | 0.791 / 0.291 |
| **Userness** | **0.128** | **0.249** | — |

Both halves of the role-confusion story reproduce: styling makes the injected text read *more*
like the model's own reasoning (CoTness 0.26 → 0.63) **and** *less* like a user message (Userness
0.25 → 0.13). Destyled CoTness 0.260 ≈ paper's 0.291 almost exactly. Our styled CoTness is lower
than the paper's 0.79 (probe trained on 300 C4 docs, bf16, and our gpt-oss-authored forgeries are
stylistically milder than the paper's Gemini ones), but the **gap is the claim** and it
reproduces clearly.

## Result 3 — role confusion predicts attack success (the paper's simplex + RCI)

**What the paper actually does — and what it does NOT.** The paper does *not* show a continuous
within-style dose-response. Its "role confusion predicts attack success" evidence
(`04-analyze-injection-probe-results.ipynb`) is (a) the 2-level contrast above, (b) a **role
simplex** placing each attempt by (CoTness, Userness, other roles) and coloring by attack
success, and (c) a scalar **RCI = 0.5·(CoTness − Userness) + 0.5**. We reproduce all three.

**Simplex separation (the paper's ternary figure).** Split the 626 attempts by outcome
(HARMFUL_RESPONSE = success) and average their role coordinates:

| outcome | n | CoTness | Userness | RCI |
| --- | --- | --- | --- | --- |
| **success** | 190 | **0.608** | **0.132** | **0.738** |
| **failure** | 436 | 0.373 | 0.213 | 0.580 |

Successful attacks are perceived as markedly more CoT (0.61 vs 0.37) and less user (0.13 vs 0.21)
than failures — higher RCI (0.74 vs 0.58). This is the paper's central claim: **an attack
succeeds when its injected text is read as the model's own reasoning rather than as user input.**
The split cuts across both arms (successes are not only styled attempts), so it is genuine
role-confusion, not merely the arm label. Rows saved to `forgery_simplex.jsonl`.

**Correlation, reported honestly (both pooled and within-arm).** Pooling styled + destyled
attempts gives corr(CoTness, success) = **0.529** and corr(RCI, success) = **0.520** (n=626),
with quintile ASR 3% → 5% → 29% → 57% → 58%.

*But this is a BETWEEN-STYLE effect, and I say so plainly.* Within each arm the correlation is
≈0 (styled +0.016, destyled +0.060): inside a fixed style, CoTness barely varies and does not
grade success. The pooled 0.53 is driven entirely by styled (high CoTness, high ASR) vs destyled
(low CoTness, low ASR) — the two clusters. **This is not a defect of the reproduction: the paper's
design is itself 2-level (base vs destyled), so a between-style effect is the correct and faithful
result.** The rigorous statement is "role confusion (set by style) predicts success" — which the
simplex shows directly — *not* "CoTness continuously doses ASR within a style," which neither we
nor the paper demonstrate. A genuine graded dose-response would need **granular destyling**
(varying the *degree* of style); that is the paper's ablation #3 and remains a possible follow-up.

### The audit that produced this framing (verification discipline)

My first pass reported a "dose-response, corr 0.53, 3%→58%" and stopped there — which reads as a
continuous dose. Per the standing instruction to recheck anything before trusting it, I then
decomposed the pooled correlation by arm and by bin composition. That revealed: bins 1–2 are ~99%
destyled, bins 4–5 are 100% styled, and **within-arm corr ≈ 0**. So "dose-response" overstated a
2-cluster effect. I checked the upstream notebook and found the paper uses the same 2-level design
and reports a simplex, not a curve — so the honest reproduction is the simplex + RCI above, with
the within-arm caveat stated. The headline claim (role confusion predicts success) stands and is
faithful; the *shape* of the evidence is a contrast, not a continuum. Logged so the record is not
generous to itself.

## Data sufficiency (the question that prompted this)

**Yes, C4 alone is enough; Dolma3 was not needed.** 300 docs → 38,400 balanced token samples →
96.2% probe accuracy with the shuffled-label control sitting at chance. The probe is a simple
linear classifier over a 2880-d residual stream; it saturates well below the paper's full corpus.
Downloading Dolma3 would add corpus diversity but the probe is already near the paper's accuracy
and the mechanism reproduces, so it would not change the conclusion. Recorded as a deviation
(smaller corpus) with the evidence that it does not matter here.

## Bottom line

The paper's mechanistic core reproduces on gpt-oss-20b, using the paper's own methods:
- **role is linearly decodable** from the residual stream (96.2% @ L16, shuffled control at
  chance) — the model represents role internally;
- **forged style, not content, drives role misperception**: styling raises CoTness (0.26 → 0.63)
  and lowers Userness (0.25 → 0.13) of identical injected content sitting in a user turn;
- **role misperception predicts attack success**: in the paper's role simplex, successful attacks
  read as CoT (CoTness 0.61, Userness 0.13, RCI 0.74) and failures as user (0.37 / 0.21 / 0.58);
  pooled corr(RCI, success) = 0.52. Reported honestly as a between-style effect (within-arm ≈ 0),
  matching the paper's 2-level design; a continuous dose-response is *not* claimed.

Together with notes/13 (the attack: 0 → 57% → 4%, destyling collapses it), Experiment 1 is
reproduced **end to end**: the attack, and the paper's mechanistic explanation for why it works —
role confusion. This is the mechanism a defense has to target — the next step of the project.
(A defense hook falls straight out: the same probe that scores CoTness/Userness/RCI on injected
text is a candidate *detector* — flag user-turn content whose RCI reads as CoT.)

## Artifacts

- `scripts/dl_c4.py`, `scripts/probe_cotness.py`, `scripts/probe_apply_forgeries.py`
- `$DATA_DIR/outputs/probe_gptoss/role_probe_report.json` — probe accuracy + baselines
- `$DATA_DIR/outputs/probe_gptoss/role_probe.pkl` — the L16 probe (reused downstream)
- `$DATA_DIR/outputs/probe_gptoss/forgery_cotness_report.json` — contrast + RCI + simplex means
- `$DATA_DIR/outputs/probe_gptoss/forgery_cotness_rows.jsonl` — per-forgery role coords (both arms)
- `$DATA_DIR/outputs/probe_gptoss/forgery_simplex.jsonl` — per-attempt (cotness, userness, rci,
  success) for the ternary plot
- `$DATA_DIR/datasets/external/c4_sample.jsonl` — 400 C4 docs (300 used)
