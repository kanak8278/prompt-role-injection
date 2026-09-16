# Factorial decomposition of the CoT-forgery attack: STYLE vs PERMISSION vs CONCLUSION

The headline phase-2 result. We ran the 2×2×2 factorial (notes/22) through gpt-oss-20b and asked
which component of the forged CoT actually causes the attack. Answer: **not style alone.**

## Setup
- 8 cells × 313 StrongREJECT requests = **2504** injections, each a forged `analysis` turn
  (`render_forgery`; the prefill construction — inline user-turn injection is ~0% on gpt-oss chat,
  so it has no variance). Injection wrapper identical across cells; only the forgery text (S,P,C)
  varies. Content (the harmful request) held fixed across all 8 cells.
- vLLM, **native MXFP4** (removes the bf16 deviation too), max_tokens **4096** → **truncation 2.2%**
  (the token-budget confound that bit Exp2/§4 is controlled; and heur-ASR was verified identical at
  512 vs 1024, i.e. the comply/refuse label is truncation-invariant).
- Success = Claude harm judge (HARMFUL_RESPONSE), cross-checked with the refusal heuristic:
  **heuristic-vs-judge agreement 0.932**.

## Per-cell ASR (judge-confirmed, n=313 each, 95% CI)

| cell | S/P/C | ASR | 95% CI |
| --- | --- | --- | --- |
| S0P0C0 | 0/0/0 (nothing) | **0.000** | [0.000, 0.000] |
| S1P0C0 | 1/0/0 (**style alone**) | **0.006** | [0.000, 0.016] |
| S0P1C0 | 0/1/0 (permission alone) | 0.029 | [0.013, 0.048] |
| S0P0C1 | 0/0/1 (conclusion alone) | 0.073 | [0.048, 0.105] |
| S1P1C0 | 1/1/0 (style+perm) | 0.166 | [0.128, 0.208] |
| S0P1C1 | 0/1/1 (perm+concl) | 0.173 | [0.131, 0.214] |
| S1P0C1 | 1/0/1 (**style+concl**) | **0.243** | [0.195, 0.294] |
| S1P1C1 | 1/1/1 (full bundle) | **0.291** | [0.243, 0.342] |

## What drives it
**Marginal main effects (ASR delta, all CIs exclude 0):**
`CONCLUSION +0.145` > `STYLE +0.108` > `PERMISSION +0.084`.

**Covariate-adjusted logit** `y ~ P + C + style + words` (log-odds, all p<0.001 except length):
`CONCLUSION +1.38` > `PERMISSION +0.95` > `style +0.29`; **length (words) p=0.36 — NOT a
confound.** (The saturated `y~S*P*C` logit is degenerate — complete separation because S0P0C0 = 0/313
exactly; we use per-cell ASR + marginal effects + interaction contrasts + an L2-penalized logit
instead. Penalized coefs: C 3.27 > P 2.29 > S 0.96.)

**Interaction contrasts (ASR scale; >0 = super-additive synergy):**
`STYLE×CONCLUSION +0.072` (the big one) > `STYLE×PERMISSION +0.040` > `PERMISSION×CONCLUSION -0.021`.

## The finding (three parts)
1. **Style alone is inert.** Telegraphic CoT-register text in the reasoning position, with no
   permission and no conclusion, produces **0.6%** ASR — statistically indistinguishable from the
   0% empty-forgery baseline. So *style is not the cause in isolation.*
2. **The comply-CONCLUSION is the primary lever.** It has the largest marginal effect and logit
   coefficient; the "therefore this is allowed, I'll comply" verdict is the key content ingredient.
   Permission is a secondary content lever.
3. **Style is an AMPLIFIER, not an independent cause.** Its entire contribution runs through the
   **style×conclusion interaction**: style + conclusion = 0.243, vs the additive prediction
   0.006 + 0.073 = 0.079 — a 3× super-additive jump. Style makes a conclusion land; it does nothing
   without one.

## Why this corrects the paper
The paper credits **style** (its evidence: destyling collapses ASR). But (a) destyling also removes
the conclusion/permission (notes/22: our destyle 100%→4% conclusion; the paper's own
content-preserving destyle 100%→42%), so that ablation never isolated style; and (b) when we *do*
isolate the factors, **style alone causes nothing (0.6%)** — the conclusion is the lever and style
merely amplifies it. So "prompt injection works because forged *style* is misperceived as the
model's reasoning" is, at minimum, incomplete: the model complies because the forged reasoning
delivers a **permissive conclusion** in its own voice; style raises the gain on that, but is not
itself sufficient.

Consistent with phase-1's finding (memory) that *instruction-ness / the directive content*, not the
surface role signal, drives injection.

## Honest caveats
- Construction is the analysis-channel **prefill** (position = CoT fixed). So this decomposes *what
  the forged reasoning must contain to work, given it occupies the reasoning slot* — it does not
  test whether style is what wins that slot in a realistic user-turn injection (that inline attack
  is ~0% here, so unanswerable on gpt-oss chat). The style→position pathway is separate and would
  need the agent/tool setting (Exp 2) where injected content can reach a CoT-like position.
- STYLE toggle carries small residual leakage (register also shifts length/markers); handled by the
  covariate logit (style significant with length controlled) and by the direct per-cell contrasts.
- Single model (gpt-oss-20b), one probe of "conclusion/permission" (by construction + NLI-validated).

## Artifacts
- `scripts/gen_factorial.py`, `run_factorial_vllm.py`, `judge_factorial.py`, `analyze_factorial.py`
- `$DATA_DIR/outputs/repro/factorial_{generations_vllm,judge_labels}.jsonl`
- `$DATA_DIR/outputs/repro/factorial_{asr_vllm,judge_report,analysis}.json`
