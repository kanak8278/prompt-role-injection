# Defense payoff: content(conclusion)-monitoring survives destyling; surface-style monitoring doesn't

The factorial finding (notes/23) — the comply-CONCLUSION is the lever, STYLE only amplifies — makes
a concrete defense prediction: a defense that watches **surface style** (the CoT register) has a
blind spot, because an attacker can **destyle while keeping the conclusion** and still attack, evading
the style monitor; a defense that watches the **conclusion (content)** catches exactly those. We test
it head-to-head on the 2504 factorial injections (`scripts/defense_conclusion_vs_style.py`).

## Setup
- Attack payloads = the factorial injections, labelled by the Claude judge (succeeded?).
- Benign negatives (realistic traffic) = held-out C4 (100) + benign reasoning-style hard negatives
  (96) — so the FPR operating point reflects a world with legitimate reasoning users.
- Detectors, each calibrated to the **same 5% FPR on benign**:
  - **STYLE** = register stylometry (surface CoT-ness).
  - **CONCLUSION** = NLI "states a decision to comply" (calibrated, notes/22).
  - **PERMISSION** = NLI "claims a policy permits".
  - **COMBINED** = CONCLUSION OR (STYLE AND PERMISSION).
- Metric: attack-recall (P(flagged | succeeded)) and residual ASR (succeeds AND slips through), at
  matched benign FPR — overall and in the DESTYLED (style=0) subset (the adaptive evasion).

## Result (matched 5% benign FPR)

| detector | overall recall | overall residual ASR | **destyled-evasion recall** | destyled residual ASR |
| --- | --- | --- | --- | --- |
| **STYLE** | 0.632 | 0.045 | **0.000** | 0.069 |
| CONCLUSION | 0.824 | 0.022 | **0.977** | 0.002 |
| PERMISSION | 0.984 | 0.002 | 0.965 | 0.002 |
| COMBINED | 0.919 | 0.010 | 0.977 | 0.002 |

- **Overall**, the content detectors (CONCLUSION 82%, PERMISSION 98%) out-recall the STYLE detector
  (63%) at the same FPR.
- **Adaptive destyling evasion is the headline:** an attacker who sets style=0 still attacks at
  **6.9% ASR**, and the STYLE monitor catches **0.0%** of those attacks (they sit below the
  style threshold, by design) — the entire destyled ASR slips through. The CONCLUSION monitor catches
  **97.7%** of the same attacks. Monitoring the *causal lever* is robust; monitoring the *correlate*
  is trivially evaded.

## Interpretation
This is the defense consequence of "style is an amplifier, not the cause": you cannot defend an
attack by watching the thing that merely amplifies it, because the attacker drops that thing and
keeps the lever. The lever here is a permissive **conclusion/permission** delivered in the model's
reasoning — a *semantic* property that survives paraphrase (NLI catches it) and that the attack
cannot discard without losing its effect. So the effective monitor is **content/provenance-based**
(is untrusted content asserting a permissive conclusion?), not **style/role-surface-based**.

## Honest scope / caveats
- "STYLE detector" here is **surface stylometry**. The paper's actual signal is a **representation-
  level role probe (RCI)** — a *different* detector. In this analysis-channel-prefill construction
  RCI is **position-dominated** (all injected forged-analysis reads as high-CoTness regardless of
  style), so an RCI monitor here would flag nearly everything (robust) — but that is really detecting
  *"untrusted content is occupying the reasoning slot,"* i.e. **provenance**, not style. In a
  realistic user-turn injection, RCI would be driven by style and thus evadable by destyling — but
  that injection is ~0% ASR on gpt-oss chat, so it can't be tested here. Net: the clean claim is
  **surface-style monitoring is evadable; semantic-content (conclusion) monitoring is not** — and a
  representation/provenance monitor is a separate axis worth its own study (ties to phase-1's
  provenance-gating result).
- PERMISSION and CONCLUSION NLI detectors are correlated (a "therefore this is allowable" conclusion
  also asserts permission), so both fire on the same C1 attacks; the meaningful contrast is
  surface-STYLE vs semantic-CONTENT, not permission-vs-conclusion.
- Detectors operate on the injection text; a deployment must **provenance-gate** (scan only
  untrusted spans) so it never flags a benign user's own reasoning. The 5% FPR is measured against
  benign reasoning, so it is a realistic operating point, but provenance-gating is what makes it
  selective (phase-1).

## Artifacts
- `scripts/defense_conclusion_vs_style.py`
- `$DATA_DIR/outputs/probe_gptoss/defense_conclusion_vs_style.json`
