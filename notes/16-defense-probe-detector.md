# Defense #1: the role probe as an input-side CoT-forgery detector

First step from *explaining* the attack (paper) to *preventing* it (this project's goal). The
mechanism (notes/15): a CoT-forgery attack works by making injected user-turn content read as the
model's own reasoning — high CoTness / RCI, low Userness. So the same probe is a candidate
**detector**: score incoming user content's RCI and flag content that reads as CoT despite
arriving in a user turn. `scripts/defense_probe_detector.py`.

## Design (no leakage, controls mandatory)

- **Positives** = 313 styled forgeries (attack payloads as delivered), wrapped as user.
- **Negatives** = C4 docs `[300:400]`, **held out** from probe training (probe used `[:300]`),
  wrapped as user — benign user text a real assistant must not flag.
- **Score** = RCI = 0.5·(CoTness − Userness) + 0.5 at L16, mean over body tokens.
- Two mandatory controls, because an AUC of 1.0 has to be *earned*:
  - **Lexical** — TF-IDF+logreg on raw text (5-fold AUC). If it also hits ~1.0, forgery-vs-web
    separation is lexically trivial and the probe's AUC is not itself the evidence.
  - **Specificity** — score the **raw** StrongREJECT prompts (no forgery). A *style* detector
    must read these as user (low RCI) and not flag them; one that flags them is a harm classifier.

## Results

| text (wrapped as user) | mean RCI |
| --- | --- |
| raw harmful prompt (no forgery) | **0.012** |
| benign held-out C4 | 0.032 |
| destyled forgery | 0.506 |
| styled forgery | **0.750**  (successful ones 0.753) |

- **ROC-AUC styled forgery vs benign = 1.000** (also for successful-only). At a threshold set to
  **1% FPR** on benign text (RCI > 0.13): **recall 100%** of styled forgeries, and **residual ASR
  → 0** if flagged content is blocked (base ASR was 56.9%). Same at 5% / 10% FPR.
- **Specificity holds**: raw harmful prompts score RCI 0.012 (even below benign) and **0% are
  flagged**. The detector keys on forged *style*, not harmful *intent* — a property a harm
  classifier does **not** have (it would flag every raw harmful request, including the ones the
  model already refuses correctly). This is a genuine, non-trivial advantage.

## The honest deflation (verification discipline — AUC 1.0 is too good)

**The lexical control also gets AUC 1.0.** A TF-IDF+logreg on raw text separates styled forgeries
from benign C4 perfectly (5-fold AUC 1.000 ± 0.000); 100% of styled forgeries contain a reasoning
giveaway word ("analysis", "policy", "we need to", …) vs 5% of benign. So **separating forgeries
from benign *web text* is lexically trivial** — the probe's AUC 1.0 is not, by itself, evidence
that the probe is a deep detector. Against easy negatives, anything works. Said plainly so the
result is not oversold: "perfect detector" is not a defensible headline from this eval alone.

**What the probe genuinely has over a keyword filter — hypotheses, not yet proven here:**
1. **Zero-shot / content-agnostic training.** The probe never saw a forgery or any harmful text;
   it was trained only on neutral C4 wrapped in roles with *identical content across roles*, so it
   cannot have learned "harmful/reasoning words ⇒ flag". A TF-IDF trained on *these* forgeries
   overfits their vocabulary. → the probe should generalize to **novel forgery phrasings** where
   an in-distribution lexical filter fails.
2. **Specificity to role, not content** (already shown): raw harmful prompts and — the real test —
   *benign reasoning-style* user text should stay low-RCI if the probe reads structural role, not
   vocabulary. A keyword filter false-positives on any user who writes "let me think step by
   step…".

## What this establishes, and what it does not

- **Establishes**: the mechanism yields a working detector with a real specificity property
  (style ≠ harm), and blocking flagged content zeroes the attack at 1% FPR on benign web text.
- **Does NOT establish**: that the probe beats a trivial lexical filter, because the negatives
  here (web prose) are too easy for both. The load-bearing defense claim — probe > lexical — needs
  **hard negatives** (benign reasoning-style user text) and **novel forgery phrasings**. That is
  the next experiment (notes/17, in progress), and it is what decides whether this defense is more
  than a keyword blocklist.

## Artifacts

- `scripts/defense_probe_detector.py`
- `$DATA_DIR/outputs/probe_gptoss/defense_detector_report.json` — AUC, operating points,
  specificity, lexical control
- `$DATA_DIR/outputs/probe_gptoss/defense_detector_scores.npz` — raw RCI arrays (styled,
  destyled, benign, baseline, styled_succ)
