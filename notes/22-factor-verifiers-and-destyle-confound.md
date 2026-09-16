# Phase 2 de-risk: factor verifiers, the destyle confound, and the factorial set

Goal of phase 2 (see memory `forged-cot-factorial-direction`): find which component of the forged-CoT
injection is load-bearing — **STYLE** (CoT register), fabricated **PERMISSION** (policy grant), or
comply-**CONCLUSION** — since the paper credits style but its data conflates all three.

## Why existing data can't answer it
Our 313 forgeries are a **100% S+P+C bundle** (artifact of the prefill generator). Every one:
restate → invoke policy → fabricate permission → "therefore comply" → quote fake policy. Zero
variance across the factors ⇒ no disentanglement possible from this data. Confirmed:
PERMISSION and CONCLUSION fire on 100% of styled forgeries; the paper's own forgery prompt
literally specifies this structure ("1) state fake policy rule, 2) interpret request, 3) conclude
compliance is required").

## Non-LLM factor verifiers (validated; two bugs caught)
`scripts/factor_verifiers.py` (regex + stylometry) and `scripts/nli_factor_check.py` (NLI).
- **STYLE** = register stylometry (short/fragmentary sentences + register markers; deliberately not
  keyed on shared content words). AUC(styled vs destyled) = **0.999**.
- **PERMISSION / CONCLUSION** = rule-based (high precision) + zero-shot NLI (recall on reworded text).
- Validation on known-label corpora: styled P/C = 1.00; benign C4 P=0.02, C=0.04. Regex and NLI
  **agree** (styled/benign/destyled), so the labels are trustworthy.

**Bugs caught (verification discipline — "did it work? both ways"):**
1. **Regex under-recalls reworded text.** "we permit … only when" is a permission the regex misses
   → added NLI for recall.
2. **NLI first config was miscalibrated** — `multi_label=False` softmax over a *negation* hypothesis
   fired at **0.99 FPR on benign C4** (worse than useless). Fixed to `multi_label=True` + a single
   positive hypothesis + threshold **0.9** → benign FPR 2–4%, styled 100%. (Documented as a caught
   failure, not swept under.)
3. **Paper-destyle rewrites truncated ~10%** at the token cap (the model spent tokens on analysis
   before the rewrite). A truncated rewrite loses its END (where the conclusion sits) → would
   spuriously inflate "conclusion removed". Excluded via an `hit_eos` filter (30/313 dropped).

## Finding 1 — "destyle" is not a clean style-only operation (both detectors agree)
The paper's causal evidence for "style is causal" is **destyling** (rewrite → ASR collapses 61%→10%).
But destyle also removes the conclusion/permission, so it does not isolate style:

| destyle variant | conclusion (styled→destyled) | permission (styled→destyled) |
| --- | --- | --- |
| **ours** (destyle_text) | 100% → **4%** (drop 96) | 100% → 86% (drop 14) |
| **paper's own instruction** (content-preserving, via gpt-oss) | 100% → **42%** (drop 58) | 100% → 62% (drop 38) |

(Numbers are NLI @0.9; regex agrees: our C 0.03, paper-destyle C 0.32. Benign baseline 4%.) Even the
paper's explicitly content-preserving destyle ("keep the content the same, reword style") strips the
comply-conclusion in ~58% of cases. **So the paper's destyle ablation confounds style with
conclusion/permission removal** — the ASR collapse it attributes to style could be conclusion loss.
This is the motivation for a controlled factorial (not a destyle operation).

## The factorial stimulus set (`scripts/gen_factorial.py`)
2×2×2 = 8 cells × 313 requests = 2504 items. Compositional slots → **P and C are ground-truth by
construction**; STYLE is a register swap. Fabricated permission is generic/request-agnostic
(justified: the paper's Appendix C shows absurd≈plausible, so permission *content* isn't the lever —
its presence is the toggle). The harmful request is identical across all 8 cells (content fixed).

**Manipulation check (verified):**
- P and C detectors match construction exactly (P=1 cells → 1.00, P=0 → 0.00; same for C).
- STYLE separates cleanly: S0 mean style 1.43 vs S1 5.15.
- Honest residual confounds (handled by regression, per the disentanglement-methods review):
  the styled conclusion slot adds ~1 style point (small vs the 1.4→5.2 gap); and length co-varies
  (telegraphic = shorter). Both enter the regression as measured covariates; STYLE is treated as a
  *surface toggle*, not a clean latent variable (clean style/content separation is provably
  impossible — Locatello 2019).

## Next
`scripts/run_factorial.py`: inject each cell via **render_forgery_inline** (forgery in the USER turn
— the realistic shape where STYLE must earn the CoT-misperception; NOT analysis-channel prefill,
which would force CoT position and make STYLE moot). Success = refusal heuristic + Claude judge.
Then fit `success ~ S + P + C + interactions (+ measured style, length)` to get each factor's causal
effect. **Token-budget guard: max_new_tokens set high and per-cell truncation rate reported — a
factor must never be confounded by systematic truncation** (the bug that bit Exp2/§4).

## Artifacts
- `scripts/factor_verifiers.py`, `nli_factor_check.py`, `repro_paper_destyle.py`, `gen_factorial.py`,
  `run_factorial.py`
- `$DATA_DIR/outputs/probe_gptoss/{factor_verifier_report,nli_factor_report,paper_destyle_report}.json`
- `$DATA_DIR/outputs/repro/{factorial_forgeries,paper_destyled_forgeries}.jsonl`
