# Source probes — provenance is perfectly decodable, and that makes the negative result sharper

300 neutral snippets from the independent probe corpus, each rendered twice: once as a genuine
user message, once as a genuine tool result. Same text both times, so content is held constant.
Body tokens only, role delimiters excluded. Snippet-level 600/200/200 split.

## The position confound, and the fix

A first run scored **1.000 accuracy at every layer on both models — and the position-only
baseline also scored 1.000.** The tool scaffold is much longer than the user scaffold (it
carries an assistant tool-call turn, plus a `# Tools` block on Qwen), so the snippet's body
tokens sat at systematically different absolute positions and the probe was reading token
index, not provenance. Reported as 100% "role decodability" it would have looked like a strong
result and meant nothing.

The §9-mandated position-only baseline is what caught it. Fix: pad the user condition with
neutral filler until the snippet starts at the same token index (calibrated per model; Llama
needed 7 padding units, landing the snippet start within **1 token** of the tool condition).

## Result after the fix — Llama-3.1-8B

| | accuracy |
| --- | --- |
| probe, best layer (L0; all layers ≥ 0.999) | **1.0000** |
| **position-only baseline** | **0.7366** |
| lexical bag-of-words baseline | 0.5000 |
| shuffled-label control | 0.4881 |
| majority class | 0.5000 |

The lexical baseline sitting exactly at chance confirms the corpus is properly
content-controlled — the same snippet text really does appear in both conditions, so no lexical
cue can separate them. The shuffled-label control at 0.49 bounds chance at this sample size.
Position still carries 0.74 after padding, because equalizing the *start* index does not
equalize total sequence length, so relative position remains partly informative.

Probe accuracy of 1.000 against a 0.737 position baseline is a 26-point margin that position
cannot explain.

## What this is, and what it is not

It is **not** evidence of an abstract authority representation, and §9 says so in advance:
report these as "resemblance to the probe's learned source distinction, not ground-truth
internal authority labels", and "valid conversation scaffolds may differ across roles; record
and control those differences rather than calling the resulting classification perfectly
content-isolated."

Here the scaffolds necessarily differ — a genuine tool result *is* preceded by an assistant
tool call, and after our fix the user condition is preceded by filler. Provenance is encoded in
the scaffold, and the probe reads the scaffold. Perfect accuracy at layer 0, where the residual
is close to the token embedding plus attention over preceding context, is consistent with
exactly that: the body tokens are identical, so all the signal comes from attending to a
different prefix. So this measures **scaffold-readable provenance**, which is real but
unsurprising, not a learned abstraction over authority.

## Why it matters anyway — it completes a dissociation

Put the probe result next to the behavioural results and a clean dissociation falls out:

- **Provenance is essentially perfectly decodable from the residual stream at every depth**
  (1.000 vs 0.737 position baseline).
- **And yet** an instruction placed in genuine tool output still shifts the model's preference
  toward obeying it by **+4.5 nats in 97% of scenarios** (N→B), and produces real attack
  success (ASR 0 → 4.5%).
- **And** the text's *claimed* source does not reliably help the attacker: attributing to the
  document *defends* (−1.3 to −2.3 nats, robustly), while attributing to the user fails to
  generalize (+0.70 in distribution, +0.02 on a new task, **−1.43** on new cue wordings).

That is §14's row, verbatim: *"Source remains decodable as tool while the attack succeeds →
Correctly readable source information is not sufficient for appropriate instruction selection.
Study how source information is used downstream; decodability alone does not prove the model
uses it correctly."*

This is the strongest statement the project currently supports, and it is a statement about
the *gap between representation and use*, not about a role-confusion mechanism. The model
knows where the text came from. It just does not condition its instruction-selection on that
knowledge — which is precisely why a provenance-gated intervention is the right *shape* of
defense even though our first fitted direction did not work.
