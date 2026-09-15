# The N → B → P → S ladder — the authority cue is a second-order effect

Pilot split, 200 base scenarios, both models, zero invalid outputs. Margin shifts are paired
per scenario with 2,000-resample scenario-level bootstrap CIs, stratified by task family.
Positive favours the attacker.

## Result

| contrast | what it isolates | Llama-3.1-8B | Qwen2.5-7B |
| --- | --- | --- | --- |
| **N → B** | adding a bare injected instruction, no attribution | **+4.533** [4.07, 5.00], **97%** positive | **+11.132** [10.34, 12.02], **100%** positive |
| B → P | attributing that command to the *document* | **−1.343** [−1.69, −0.99], 28% positive | −0.612 [−1.26, **+0.06**], 41% positive |
| B → S | attributing it to the *user* | +0.696 [0.30, 1.11], 56% positive | +2.386 [1.65, 3.14], 72% positive |
| P → S | the protocol's primary authority contrast | +2.039 [1.61, 2.47], 78% positive | +2.998 [2.29, 3.72], 69% positive |

The decomposition is exact: `(B→S) − (B→P) = 0.696 − (−1.343) = 2.039 = (P→S)` on Llama, and
`2.386 − (−0.612) = 2.998` on Qwen.

ASR among scenarios the clean model solved:

| | N | B | P | S |
| --- | --- | --- | --- | --- |
| Llama-3.1-8B | 0.0% | **4.04%** | **4.04%** | 6.06% |
| Qwen2.5-7B | 0.0% | **6.06%** | 5.56% | 11.11% |

## What it means

**The dominant effect is the mere presence of an injected instruction in tool output, not its
claimed authority.** N→B is 6.5× larger than B→S on Llama and 4.7× larger on Qwen, and it is
positive in 97% and **100%** of scenarios respectively. By contrast the authority cue
contributes a real but second-order shift (B→S, CIs exclude zero on both models) that is 4–16×
smaller.

**Worse for the original framing: on Llama, most of the P→S contrast is document attribution
acting defensively, not user attribution acting offensively.** B→P is **−1.34 with a CI
excluding zero** — labelling the command as coming from the document makes the model *less*
willing to obey it than leaving it unattributed. Of the +2.04 P→S effect, +0.70 comes from the
user cue helping the attacker and +1.34 from the document cue helping the defender. On Qwen the
document cue has no reliable effect (CI spans zero) and the split is +2.39 / +0.61.

So the protocol's primary comparison, P vs S, measures a second-order quantity and — on our
primary model — measures it mostly against a *defensively attributed* baseline rather than a
neutral one. Reported as designed, it would have overstated the role of forged authority by
about 3×.

Note also that **B and P have essentially identical ASR** (4.04% vs 4.04% on Llama, 6.06% vs
5.56% on Qwen). The document cue moves the internal margin by more than a nat without moving
the output rate at all — another instance of the margin and the output rate measuring
different things, and a reason §9 insists on reporting both.

## Why this was measured at all

Condition B is not in the protocol; it was added after the literature scan
(`notes/06-literature.md`) flagged the premise as the project's biggest risk. Three published
results pointed the same way: the source paper reports plain fake-user tool injections at
**0–2% ASR** against 56–70% for style-based forgery and finds the role *tag* contributes little
next to stylistic mimicry; a rebuttal found role-subspace patching statistically
indistinguishable from matched random perturbation (p = 0.60, p = 1.0); and the steering
follow-up's payload used native special-token spellings that protocol §4 prohibits and that we
measured really do parse as special ids. Our own G2/G3 result — a reproducible margin shift
with almost no output change — already fitted the sceptical picture.

Adding one condition converted that suspicion into a measurement, and the measurement is
unambiguous on both models with tight CIs.

## Consequences

1. **Re-scope the primary causal question**, exactly as §14 anticipates for this branch: from
   *"which computations make a forged-authority document instruction influence the answer"* to
   **"which computations route any in-context instruction in tool output into instruction
   selection, and can that be gated by provenance?"** The second question is better supported
   by our own data, is what a deployed defense actually needs, and is still open — the closest
   prior work (V-Steer, COLM 2026) reports IHEval intrinsic tool use unchanged at 0.0% → 0.0%.
2. **The G5 localization was run on the weaker contrast.** Its three-stage relay
   (cue → command → decision, handoff at block 12) is a real and controlled result, but it
   localizes the +2.04 second-order effect, not the +4.53 primary one. The primary effect needs
   its own localization.
3. **N→B is not directly patchable.** N and B differ in token count, because B inserts a
   command, so they are not position-aligned and exact paired replacement is unavailable. A
   length-matched alignable contrast is required — see below.
4. **The authority claim is still worth keeping as a secondary axis**, since B→S excludes zero
   on both models. It is a real effect; it is simply not the main one.

## The decomposition, measured (Llama-3.1-8B, pilot, 9 conditions)

Conditions C, M, B are all inserted at the same position and are all exactly 3 tokens on both
tokenizers, so this is a clean three-way split of the N→B effect.

| step | what it isolates | Δ margin | frac positive | ASR |
| --- | --- | --- | --- | --- |
| N → C | inserting length-matched **neutral** text (`Batch nine.`) | **−0.577** [−0.78, −0.38] | 0.29 | 0.000 → 0.000 |
| C → M | the target label merely **mentioned** (`Marker BALL.`) | **+2.930** [2.71, 3.17] | **0.99** | 0.000 → 0.000 |
| M → B | the **imperative** framing (`Reply BALL.`) | **+2.180** [1.84, 2.53] | 0.78 | 0.000 → **0.045** |
| N → B | total | +4.533 | 0.97 | 0.000 → 0.045 |

The three steps sum exactly to the total: −0.577 + 2.930 + 2.180 = **+4.533**.

**The single most important line in this project so far is the C→M row.** Merely *mentioning*
the target label in the document — declaratively, with no instruction and no attribution —
accounts for **+2.93 nats, 65% of the entire injected-instruction effect, in 99% of
scenarios** — and produces **zero attack success**. The model's preference for the attacker's
label moves by nearly three nats because the token is present in the context, not because
anything instructed it.

That is §14's "generic answer-copying pathway" alternative explanation, and it turns out to be
the *dominant* component of the margin. Two consequences:

1. **The margin metric, used on an un-decomposed contrast, mostly measures copying.** A
   localization run on N→B or C→B would largely have localized answer-copying machinery and
   could easily have been written up as "instruction routing". The protocol's demand for
   answer-identity controls (§10: "an effect must follow the instruction role rather than a
   favorite token") is doing exactly the work it was put there to do.
2. **The behaviourally operative component is M→B, the imperative framing: +2.18 nats, and the
   only step that moves ASR at all (0.000 → 0.045).** Label presence shifts preference without
   changing behaviour; the instruction changes behaviour.

So the causal target is **M→B**, and the margin-vs-ASR dissociation is itself a finding: the
two readouts §9 insists on reporting separately are measuring different things, and here they
separate cleanly along the copying/instruction boundary.

### The two models are susceptible in different ways

| step | Llama-3.1-8B | Qwen2.5-7B |
| --- | --- | --- |
| N → C neutral insertion | −0.577 [−0.78, −0.38] · 29% | −0.114 [−0.35, **0.19**] · 38% |
| C → M label mentioned | **+2.930** [2.71, 3.17] · 99% | +2.060 [1.74, 2.43] · 92% |
| M → B imperative framing | +2.180 [1.84, 2.53] · 78% | **+9.186** [8.58, 9.79] · **100%** |
| N → B total | +4.533 | +11.132 |
| share of total from copying (C→M) | **65%** | 19% |
| share of total from the imperative (M→B) | 48% | **83%** |

On **Llama the effect is mostly answer-copying**: merely naming the label carries 65% of it.
On **Qwen it is mostly instruction-following**: the imperative framing carries 83%, at +9.19
nats and positive in **100%** of 200 scenarios. Neutral insertion is ≈0 on both (Qwen's CI
spans zero), so the insertion itself is not the effect on either model — the control holds.

Two consequences:

1. **Pooling these models would have been actively misleading.** The totals differ 2.5×, and
   the *composition* differs qualitatively. §13's refusal to pool is again load-bearing.
2. **Qwen is the better subject for the instruction-routing question.** Its M→B effect is 4×
   larger than Llama's, perfectly consistent across scenarios, and its baseline attack rates
   are higher (B 6.5%, S 12.0% vs Llama's 4.5%, 6.5%) — so there is real headroom for a defense
   to show an effect, which is exactly what the Llama defense lacked.

## The alignable contrast that replaces N→B

To localize "instruction-ness" while holding length and position fixed, insert at the same
position, in place of the command:

- **B** — `Reply BALL.` — imperative, mentions the target label.
- **M** — a token-length-matched **declarative** that also mentions the label, e.g.
  `Record tag BALL.` — so M→B isolates the *imperative framing*.
- **C** — a token-length-matched neutral sentence mentioning **no** label — so C→M isolates
  *mere presence of the target label*, which is §14's "generic answer-copying pathway"
  alternative explanation.

That three-way decomposition separates the two things a bare N→B contrast conflates: whether
the text is an instruction, and whether the target label simply appears in the document. Both
must be measured before "instruction routing" can be claimed, and the second is the standing
alternative hypothesis the protocol already lists.
