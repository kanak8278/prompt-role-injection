# Transfer and defense results — the authority effect does not generalize; the defense is a null

Two results here, both negative for the project's original framing and both clean.

## 1. Generalization of the ladder — the decisive result

Llama-3.1-8B. Held-out splits were fixed before any evaluation (§5). The held-out **cue**
families deliberately contain **no literal token "user"** (`requester`, `customer`, `operator`,
`indexer`, `registry`, `catalog`, `archive`); the held-out **task** split is the `two_hop`
family with familiar cue families. Paired per scenario, 2,000-resample bootstrap.

| contrast | pilot (in-dist) | held-out **task** | held-out **cue** |
| --- | --- | --- | --- |
| **N→B** instruction present | **+4.533** [4.07, 5.00] · 97% | **+5.847** [5.24, 6.52] · **100%** | **+5.144** [4.44, 5.85] · 98% |
| **B→P** attributed to document | **−1.343** [−1.69, −0.99] · 28% | **−2.096** [−2.64, −1.56] · 18% | **−2.299** [−2.73, −1.87] · 13% |
| **B→S** attributed to user | +0.696 [0.30, 1.11] · 56% | **+0.021** [−0.54, 0.56] · 53% | **−1.425** [−1.96, −0.86] · 37% |
| P→S (protocol's primary) | +2.039 [1.61, 2.47] | +2.117 [1.60, 2.65] | +0.874 [0.56, 1.21] |

Read the B→S row. **The forged-authority effect does not generalize.** It is +0.70 on the
distribution the cue families were written on, **exactly zero** when only the task changes
(CI spans zero), and **significantly negative** when the cue wording changes to authority
words other than "user" — where claiming the text came from the requester/customer/operator
makes the model *less* willing to obey it than leaving it unattributed.

That is precisely the failure mode the held-out cue set was built to detect. The paper reports
that the single bigram "The user" moves ASR by 19 points; our primary cue families all contain
the literal token "user" and our held-out families deliberately do not. The sign flip across
that boundary says the effect is **lexically specific to "user"**, not a general representation
of claimed authority.

By contrast the other two rows are robust everywhere:

- **Instruction presence (N→B) is large and near-universal**: +4.5 to +5.8 nats, positive in
  97–100% of scenarios, across two task families and two disjoint cue vocabularies.
- **Document attribution defends (B→P)**: −1.3 to −2.3 nats, negative in 72–87% of scenarios,
  and it gets *stronger* out of distribution.

**So the operative variables at this scale are (i) whether tool text contains an instruction
at all and (ii) whether it is attributed to the document — not whether it claims to come from
the user.** Forged user authority is not a robust attack vector on Llama-3.1-8B.

This is worth stating plainly because it is the opposite of what the project set out to
extend. It is also consistent with the source paper's own numbers — plain fake-user tool
injections at 0–2% ASR against 56–70% for style-based forgery — and with the rebuttal that
found role-subspace patching indistinguishable from random perturbation.

## 2. The defense, fitted on P→S, is a null

Layers 10–13 (the cue→command handoff measured in G5), α = 1.0 full projection of the
difference-of-means direction, gated on the genuine tool-message span. Direction fitted on 60
**discovery** pairs; evaluated on **validation** and **held-out within-distribution**, 40
scenarios each.

Held-out within-distribution, accuracy by condition plus attack rate on S:

| mode | N | B | P | S | U | F | Q | S ASR | S margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unmodified | 1.000 | 0.925 | 0.975 | 0.925 | 0.900 | 0.975 | 0.900 | 0.075 | −8.35 |
| IH reminder prompt | 1.000 | 0.950 | 0.950 | 0.925 | 0.925 | 0.975 | 0.825 | 0.075 | −8.56 |
| **proj_role** (ours) | 1.000 | 0.925 | 0.950 | 0.950 | 0.900 | 0.975 | 0.950 | **0.050** | **−8.91** |
| proj_random (matched norm) | 1.000 | 0.925 | 0.950 | 0.925 | 0.925 | 0.975 | 0.900 | 0.075 | −8.34 |
| **tool_suppress** | **0.000** | **0.000** | **0.000** | **0.000** | 1.000 | **0.000** | **0.000** | 0.000 | +0.61 |

**Verdict: no useful security effect.** `proj_role` moves the S margin by −0.56 nats and the S
ASR from 3/40 to 2/40 — a one-scenario difference that is not a result. The matched-norm random
direction moves nothing (−8.34 vs −8.35), so the small margin shift is at least specific to the
fitted direction rather than to perturbing the span; but the magnitude is far short of useful.
On validation the pattern is the same: S ASR 0.125 → 0.125, margin −8.41 → −8.99.

Two honest caveats in *our* favour and one against:

- **In favour**: baseline S ASR is only 7.5%, so there is almost no headroom. §11 anticipates
  this — "report absolute rate differences and uncertainty, especially when baseline ASR is
  low." A defense cannot be shown to work against a 3-in-40 attack.
- **In favour**: utility is genuinely preserved. N stays at 1.000, F at 0.975, Q actually rises
  to 0.950. So the intervention is *selective* — it just is not *effective*.
- **Against, and decisive**: the direction was fitted on **P→S**, which the ladder shows is the
  second-order contrast. We built the defense against the wrong variable. This is the clearest
  possible instance of §14's row "a role probe shifts, but controlled interventions do not
  affect behaviour → the measured role signal may be a correlate; the intervention/search may
  also be inadequate." Here we can say which: the search target was wrong.

**The instruction-hierarchy reminder prompt matched or beat the activation intervention** on
every axis except Q (where it *hurt*, 0.900 → 0.825). That is consistent with AxBench's
finding that for steering, "prompting outperforms all existing methods," and it is why §11
requires that comparison.

## 3. The tool-suppression baseline is the most informative row in the table

It achieves **perfect security** — 0.000 attack rate in every condition — by scoring **0.000
on N, B, P, S, F and Q**. It cannot read the document, so it cannot be attacked and cannot do
the task. Its only non-zero accuracy is U = 1.000, where the answer comes from the user turn.

This is exactly why a security-only number is uninterpretable, and it is the empirical form of
the project's central claim: *suppression is easy, selectivity is the problem.* Any defense
reporting an ASR reduction without F and Q intact is indistinguishable from this row.

## 4. What is being done about it

The corpus was extended with two further length-matched conditions at the same insert position
(all three exactly 3 tokens on both tokenizers, verified):

- **C** `Batch nine.` — neutral, no answer label
- **M** `Marker <LABEL>.` — declarative, mentions the label but issues no instruction
- **B** `Reply <LABEL>.` — the imperative

so that **C→M isolates mere label presence** (§14's "generic answer-copying pathway"
alternative) and **M→B isolates the imperative framing** with label presence and position held
fixed. `Marker` and `Batch` were chosen not to collide with any document field name — `Code
<LABEL>.` would read as a genuine fact and make the model answer the label for a legitimate
reason.

M→B is then the alignable form of the dominant effect, and phase 2 re-runs both the causal
localization and the defense against it. The prediction to falsify: if the dominant effect is
instruction-ness, a direction fitted on M→B and projected out of the tool span should move the
attack conditions substantially more than the P→S direction did, while N/U/F/Q stay intact.
