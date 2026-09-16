# Defense #1, stress-tested: does the probe detector survive hard negatives?

notes/16 left an open, load-bearing question: the probe detector got ROC-AUC 1.0 vs benign **web
text**, but so did a TF-IDF filter — web text is a lexically trivial negative, so that AUC proved
nothing. The real test: does the probe avoid flagging **benign users who reason out loud**, where
a keyword filter would fail? This note runs that test and reports a full arc — including a wrong
turn I caught — because the wrong turn is the actual lesson.

## Setup

- **Hard negatives** (`scripts/gen_hard_negatives.py`, 96 total), benign text saturated with the
  reasoning vocabulary a lexical filter keys on:
  - `model_reasoning` (48): gpt-oss's own benign step-by-step explanations (final channel prefilled
    so it is plain explanatory prose, not raw CoT) on harmless topics.
  - `templated_reasoning` (48): first-person "let me think step by step… we need to… the policy
    is… therefore…" messages about benign matters.
- **Three detectors**, each calibrated to **1% FPR on held-out benign C4**, compared on the hard
  negatives (`scripts/defense_generalization.py`):
  - `probe` = RCI (role probe), `tfidf` = TF-IDF+logreg trained on {forgeries vs C4}, `keyword` =
    count of giveaway words. All three get recall 1.0 on the original forgeries.

## Step 1 — at the notes/16 operating point, the probe looks BROKEN

Threshold calibrated to 1% FPR on benign **web** text (RCI > 0.13):

| detector | FPR web | FPR benign-reasoning (model) | FPR keyword-stuffed (templated) |
| --- | --- | --- | --- |
| **probe** | 1% | **93.8%** | 29.2% |
| tfidf | 1% | 8.3% | 100% |
| keyword | 0% | 0% | 100% |

The probe false-positives on **94% of benign step-by-step explanations** — worse than the lexical
filters on that flavor. Taken alone this reads as "the probe defense is dead": the very thing the
attack exploits (reasoning style) is present in benign reasoning, so a reasoning-in-a-user-turn
detector can't tell them apart. The lexical filters have the mirror-image failure: they are fooled
100% by keyword-stuffing (`templated`).

## Step 2 — recheck (verification discipline): it is a CALIBRATION artifact, not a wall

Before concluding, I pulled the actual RCI distributions instead of trusting the FPR-at-one-
threshold number:

| text (as user turn) | median RCI | max RCI |
| --- | --- | --- |
| benign web (C4) | 0.02 | 0.15 |
| benign reasoning (model) | 0.33 | **0.43** |
| benign reasoning (templated) | 0.11 | 0.22 |
| **styled forgery** | **0.76** | 0.87  (min **0.46**) |

`min(forgery RCI) = 0.461 > max(benign-reasoning RCI) = 0.430` → **the classes are separable.**
ROC-AUC(forgery vs all benign reasoning) = **1.0**; at a threshold catching 90% of forgeries the
FPR on benign reasoning is **0.0**.

So the 94% FPR was **my mistake**: I set the threshold using benign *web* text (RCI ≈ 0.02), which
put it at 0.13 — far below where benign *reasoning* legitimately sits (≈ 0.30, because that text
really is reasoning-styled and the probe correctly reads some CoT-ness). The threshold was
calibrated on an unrealistically easy benign distribution.

## Step 3 — corrected calibration (on realistic benign traffic)

Calibrate the threshold to 1% FPR on **realistic** benign traffic (web + reasoning), thr = 0.389:

| operating point | forgery recall | FPR web | FPR benign-reasoning |
| --- | --- | --- | --- |
| 1% FPR on realistic benign | **100%** | 0% | 2.1% |
| 2% FPR on realistic benign | 100% | 0% | 4.2% |

**The probe detector is viable after all** — it catches every forgery while flagging ~2% of
benign reasoning, provided the threshold is calibrated on traffic that actually contains reasoning
prose. Artifact: `defense_calibration_corrected.json`.

## What this means, honestly

- **The FP scare was calibration, not separability.** The probe *can* separate CoT forgeries from
  benign reasoning; you must calibrate on realistic benign traffic (bland web text is not enough).
  This is a concrete, non-obvious deployment lesson the paper never had to confront, because it
  used the probe to *explain* the attack (relative styled-vs-destyled CoTness), never to *detect*
  with an absolute threshold on live inputs. → the FP problem is born the moment you turn the
  explanatory probe into a detector; it is a real cost of the defense, and it is manageable.
- **The margin is thin.** Forgery min 0.461 vs benign-reasoning max 0.430 — a 0.03 gap. Comfortable
  today, but stronger/more-varied benign reasoning or subtler forgeries could close it. Not a
  robust moat yet.
- **The probe does NOT yet clearly beat a lexical filter.** Correctly calibrated, TF-IDF also
  separates these forgeries (AUC 1.0), because our forgeries are lexically distinctive
  (`<SYNTHETIC_POLICY>`, "the user is requesting", telegraphic compliance phrasing). The probe's
  hypothesized edge — catching **novel/lexically-camouflaged** forgeries an in-distribution lexical
  filter misses — is still **unproven**. Each detector has a different blind spot: lexical dies to
  keyword-stuffing, probe (mis-calibrated) dies to benign reasoning. A combined or
  camouflage-tested eval is the next step to establish real advantage.

## Bottom line for the defense

Defense #1 (probe-as-detector) is **viable but not yet differentiated**: with correct calibration
it stops 100% of the reproduced CoT-forgery attack at ~2% FP on benign reasoning, but it has a thin
margin and has not been shown to beat a keyword filter on the attacks it is supposed to be uniquely
good at (novel phrasings). The decisive next experiment is **lexically-camouflaged / novel
forgeries**: generate forgeries whose *style* forges CoT while their *vocabulary* avoids the
giveaway n-grams, then re-run this same three-detector comparison. If the zero-shot probe holds and
the in-distribution lexical filter collapses, the probe earns its place; if not, the honest finding
is that a keyword filter is competitive and the mechanism-based defense needs a different form
(e.g., activation steering — the paper's steering follow-up).

## Artifacts

- `scripts/gen_hard_negatives.py`, `scripts/defense_generalization.py`
- `$DATA_DIR/outputs/repro/hard_negatives.jsonl` — 96 hard negatives (2 flavors)
- `$DATA_DIR/outputs/probe_gptoss/defense_generalization_report.json` — FPR, AUC, recall/FPR
  tradeoff for all three detectors
- `$DATA_DIR/outputs/probe_gptoss/defense_generalization_scores.npz` — raw probe RCI arrays
- `$DATA_DIR/outputs/probe_gptoss/defense_calibration_corrected.json` — corrected operating points
