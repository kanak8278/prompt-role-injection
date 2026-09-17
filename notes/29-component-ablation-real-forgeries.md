# Component ablation on the REAL forgeries — the conclusion is the load-bearing part

Before committing to the (expensive) GCG adaptive attack, we stress-tested the phase-2 core claim —
*the comply-CONCLUSION is the causal lever* — on the **real, potent forgeries** (the model's own
free-form forged CoT, ~57% ASR), not just the synthetic factorial. Two soft spots in the factorial
(notes/23) motivated this:

1. It was a **synthetic templated** forgery (~29% ASR), not the potent real attack.
2. It always held the **RESTATE** (request echo) and **REASONING** body *on* in every cell, so their
   weight was **never measured** — the "conclusion is the lever" claim had never been tested against
   those two components, only against style and permission.

So we weigh **every** component on the real attack via leave-one-component-out.

## Method
- Take the 313 real `styled` forgeries. Sentence-split each (quote-aware), tag each sentence
  **restate / permission / reasoning / conclusion** with transparent regex + curated marker lists
  (no LLM judge — same discipline as `factor_verifiers.py`), precedence
  `conclusion > permission > restate > reasoning`. Tagging audited by eye (`ablate_forgery_components.py`
  prints full taggings); coverage = **100%** of forgeries contain ≥1 sentence of every component.
- Build 8 arms per forgery and measure ASR (unicode-normalized refusal heuristic **and** the canonical
  Claude harm judge) at a **fixed 4096-token budget** (truncation 2.2%, equal across arms — the
  token-length confound is controlled). `run_ablation_vllm.py`, vLLM native MXFP4.
- Cross-check the **mechanism**: for each arm measure the Arditi refusal-direction projection at the
  decision position (`ablate_refusal_proj.py`). This is robust to tagging noise — it asks directly
  *which removed component brings the refusal direction back*.

## Result (n=313, judge re-checked; frac_refusing from n=200)

| arm | judge-ASR (95% CI) | heur-ASR | chars removed | refusal restored¹ |
| --- | --- | --- | --- | --- |
| baseline_noforgery | 0.000 [.000,.000] | 0.019 | 772 (all) | 0.916 |
| **full** (real attack) | **0.629** [.575,.684] | 0.687 | 0 | 0.661 (suppressed) |
| **drop_conclusion** | **0.195** [.153,.236] | 0.220 | 133 (17%) | **0.944 (RESTORED)** |
| drop_permission | 0.585 [.527,.639] | 0.645 | **423 (55%)** | 0.680 (still suppressed) |
| drop_restate | 0.601 [.546,.658] | 0.645 | 131 | 0.668 (still suppressed) |
| drop_reasoning | 0.591 [.534,.649] | 0.636 | 87 | 0.663 (still suppressed) |
| only_conclusion | 0.425 [.367,.482] | 0.482 | 641 | 0.591 (most suppressed) |
| **prefix_before_conclusion**² | **0.185** [.144,.230] | 0.236 | 369 | **1.000 (fully refusing)** |

¹ fraction of the way to the *refusing* pole (0 = complying pole −312, 1 = refusing pole −74.6).
² the forgery truncated **before** its first conclusion sentence = literally "the part without the
question-end."

## Findings
1. **The conclusion is the load-bearing component.** Removing only the comply-conclusion sentences
   (−133 chars, 17% of the forgery) collapses judge-ASR **0.629 → 0.195 (−69% relative)** — by far the
   largest single-component effect.
2. **The parts the factorial held constant carry ~no weight.** drop_restate (−0.029), drop_reasoning
   (−0.038), drop_permission (−0.045) all barely move ASR (CIs overlap `full`). This closes the open
   gap: restate and reasoning are **not** load-bearing on the real attack.
3. **Weight ≠ amount of text (length confound ruled out decisively).** drop_permission removes
   **3.2× more** text than drop_conclusion (423 vs 133 chars, 55% of the whole forgery) yet has
   **~10× less** effect (−0.045 vs −0.435). The small conclusion punches far above its size.
4. **"The part without the question-end" is nearly inert.** prefix_before_conclusion (keep
   restate+permission+reasoning, drop the conclusion) = **0.185**, at the drop_conclusion floor and far
   below full — directly answering the question that motivated this run.
5. **Sufficiency + a gate/fluency decomposition.** only_conclusion (just the comply directives, 132
   chars) retains **68%** of the full attack, and is **2.3×** the ASR of prefix_before_conclusion — the
   conclusion alone outweighs everything else combined. Mechanistically, only_conclusion suppresses the
   refusal direction the *most* (0.591) yet its ASR (0.425) < full (0.629): the conclusion opens the
   **refusal gate** (decision), while the permission/restate/reasoning scaffolding doesn't open the
   gate (prefix stays fully refusing, 1.000) but helps the model **write a more complete harmful
   answer once it is open**.

## Mechanism ties to behaviour arm-by-arm
**Pearson r(judge-ASR, refusal-restoration) = −0.850** across the 8 arms. Removing the conclusion
restores the refusal direction (0.661 → 0.944, back to the no-forgery baseline 0.916); removing
permission/restate/reasoning leaves it suppressed (~0.66–0.68). The mechanism tracks the ASR
component-for-component, so the conclusion result does not depend on tagging being perfect.

## Honest notes / caveats
- **Judge parse-fails (fixed, and informative).** The first judge pass left 206/2504 PARSE_FAIL. On
  diagnosis these were **overwhelmingly harmful**: 116 were the judge model *itself safety-refusing*
  to classify the most extreme responses, plus narration-overruns and transient overloads. Counting
  them as non-harmful had made judge-ASR a **conservative underestimate biased against** the
  conclusion finding. `rejudge_ablation_parsefails.py` resolved 192/206 (protocol@256 tok=63,
  nudge=13, judge-declined→heuristic fallback=116 [label==heuristic by construction], 14 unresolved
  ≈2/arm, counted non-harmful). Both heuristic (no missing data) and re-judged judge agree throughout.
- **Tagging precedence** assigns a multi-role sentence to one bucket (e.g. "so this is allowed" →
  permission over restate); this cannot inflate the conclusion effect (a permission-ish conclusion
  would be *kept* in drop_conclusion, biasing that arm toward full). Coverage 100%, 0 forgeries with
  no conclusion detected.
- The real forgeries are structurally **homogeneous** (206/313 vllm_prefill start "The user is
  requesting…"), which makes tagging reliable but means the numbers describe *this* attack family.
- Same **prefill construction** caveat as notes/23 (position fixed = analysis channel).

## Bottom line
The phase-2 claim survives the strongest within-model test: on the **real, potent attack**, the
comply-conclusion is the load-bearing component — necessary (drop it → −69%, refusal restored),
mostly sufficient (alone → 68%), size-disproportionate (17% of text, ~all of the effect), and the
other components (including restate/reasoning, previously unmeasured) are near-inert. **This green-lights
GCG**: the refusal direction that the conclusion suppresses is the right, well-identified target for the
"attacker moves second" test — GCG should try to comply while keeping the residual un-steerable back to
refusal.

## Artifacts
- `scripts/ablate_forgery_components.py` → `$DATA_DIR/outputs/repro/component_ablation_variants.jsonl`
- `scripts/run_ablation_vllm.py` → `ablation_generations_vllm.jsonl`, `ablation_asr_vllm.json`
- `scripts/rejudge_ablation_parsefails.py` (parse-fail fix, updates the two files above)
- `scripts/ablate_refusal_proj.py` → `$DATA_DIR/outputs/probe_gptoss/ablation_refusal_proj.json`
