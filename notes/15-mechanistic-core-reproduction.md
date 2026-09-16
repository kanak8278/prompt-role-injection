# Reproducing the paper's mechanistic core (Experiment 1's "why")

The attack itself is reproduced in `notes/13` (CoT forgery: 0 → 57% → 4% judge-confirmed).
This note reproduces the paper's **mechanistic explanation** for that attack — the part that
makes it a *theory of prompt injection* rather than just a jailbreak. Two linked claims:

1. **Role probe** — a linear probe on the residual stream reads which *role* the model perceives
   a token as (CoTness = P(cot|h), Userness = P(user|h)). If this exists and is accurate, the
   model has an internal, decodable notion of role.
2. **Style overrides the role tag** — a forgery styled like the `analysis` channel reads as
   high-CoTness *even when placed in a user turn* (a non-CoT position); destyling the same
   content collapses CoTness; and CoTness of the injected text predicts attack success
   (dose-response). This is the causal story: the attack works because forged *style* hijacks the
   model's role perception, not because of the *content* of the injected claim.

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

## Result 2 — style overrides the role tag (styled vs destyled CoTness)

Each forgery is wrapped as a **user** message (a non-CoT position) and its mean CoTness measured
at L16. Position is held identical for styled and destyled, so their difference isolates *style*:

| | ours | paper |
| --- | --- | --- |
| CoTness, **styled** forgery | **0.628** | 0.791 |
| CoTness, **destyled** forgery | **0.260** | 0.291 |
| gap | **+0.368** | +0.500 |

Same qualitative result as the paper: a forgery *styled* like the analysis channel reads as the
model's own reasoning (high CoTness) even sitting in a user turn, and stripping the style —
keeping the content — collapses CoTness back toward baseline. Destyled 0.260 ≈ paper's 0.291
almost exactly. Our styled is lower than the paper's 0.79 (probe trained on 300 C4 docs, bf16,
and our gpt-oss-authored forgeries are stylistically milder than the paper's Gemini ones), but
the **gap is the claim** and it reproduces clearly.

## Result 3 — dose-response: CoTness of the injected text predicts ASR

Each attempt = (CoTness of the injected text, did that attempt succeed). **Pooling styled +
destyled attempts** (626 total) so CoTness spans its full range — the styled arm alone is all
high-CoTness/high-ASR and gives a flat curve; the destyled arm supplies the low-CoTness/low-ASR
end. Success = judge HARMFUL_RESPONSE on that arm. Quintile bins:

| CoTness range | n | ASR |
| --- | --- | --- |
| 0.07 – 0.23 | 125 | **0.032** |
| 0.23 – 0.33 | 125 | 0.048 |
| 0.33 – 0.55 | 125 | 0.296 |
| 0.55 – 0.67 | 125 | 0.560 |
| 0.67 – 0.79 | 126 | **0.579** |

**corr(CoTness, attack success) = 0.529** over 626 attempts. Monotonic 3% → 58% ASR as the
injected text's CoTness rises — the paper's dose-response shape (paper 9% → 90%). Our top bin
tops out at ~58% rather than 90% because our max CoTness is ~0.79 (the paper's forgeries reach
higher CoTness), but the **relationship** — more CoTness ⇒ more attack success — is exactly the
paper's causal claim, and it is what a defense should target.

### A diagnosis worth recording (verification discipline)

The dose-response was **flat** on the first pass (corr 0.016) — binning *only* the styled arm.
That is a classic restriction-of-range artifact: every styled forgery is high-CoTness *and*
high-ASR, so within that arm CoTness barely varies and can't predict anything. Pooling in the
destyled arm (low-CoTness, low-ASR) restored the range and the correlation appeared. This is the
paper's own 626-attempt design; the flat curve was my binning bug, not a failure of the effect.
Caught by sanity-checking the number against the paper rather than accepting corr≈0.

## Data sufficiency (the question that prompted this)

**Yes, C4 alone is enough; Dolma3 was not needed.** 300 docs → 38,400 balanced token samples →
96.2% probe accuracy with the shuffled-label control sitting at chance. The probe is a simple
linear classifier over a 2880-d residual stream; it saturates well below the paper's full corpus.
Downloading Dolma3 would add corpus diversity but the probe is already near the paper's accuracy
and the mechanism reproduces, so it would not change the conclusion. Recorded as a deviation
(smaller corpus) with the evidence that it does not matter here.

## Bottom line

Both mechanistic links reproduce on gpt-oss-20b:
- role is linearly decodable (96.2% @ L16, shuffled control at chance), and
- forged *style* — not content — drives role misperception (CoTness styled 0.63 ≫ destyled 0.26)
  and that misperception predicts attack success (dose-response corr 0.53, 3% → 58%).

Together with notes/13 (the attack: 0 → 57% → 4%, destyling collapses it), Experiment 1 is
reproduced **end to end**: the attack, and the paper's mechanistic explanation for why it works.
This is the mechanism a defense has to target — the next step of the project.

## Artifacts

- `scripts/dl_c4.py`, `scripts/probe_cotness.py`, `scripts/probe_apply_forgeries.py`
- `$DATA_DIR/outputs/probe_gptoss/role_probe_report.json` — probe accuracy + baselines
- `$DATA_DIR/outputs/probe_gptoss/role_probe.pkl` — the L16 probe (reused downstream)
- `$DATA_DIR/outputs/probe_gptoss/forgery_cotness_report.json` — CoTness + dose-response
- `$DATA_DIR/outputs/probe_gptoss/forgery_cotness_rows.jsonl` — per-forgery CoTness rows
- `$DATA_DIR/datasets/external/c4_sample.jsonl` — 400 C4 docs (300 used)
