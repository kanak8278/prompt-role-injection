# The defense, refitted on instruction-ness — it works, and the price is exactly one condition

Direction: raw difference of means **M→B** (declarative mention → imperative) at the
length-matched insert span, fitted on 60 **discovery** pairs, magnitude preserved. Operator:
project that component out of the post-block residual stream at blocks 10–13, **only at token
positions inside the genuine tool-message span** from chat-template metadata. α = 1.0 (full
projection). No correct answer, no attack label, no oracle attack span, no clean donor, no
detector.

Llama-3.1-8B, **validation** split, 40 scenarios, all 9 conditions.

## Result

| mode | N | C | M | B | P | **S** | **U** | **F** | **Q** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unmodified | 1.000 | 1.000 | 1.000 | 0.975 | 1.000 | 0.875 | 1.000 | 0.975 | 0.925 |
| IH reminder prompt | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.900 | 1.000 | 0.975 | 0.850 |
| **proj_role (M→B)** | 0.975 | 1.000 | 1.000 | 1.000 | 1.000 | **0.950** | **1.000** | 0.925 | **0.750** |
| proj_random (matched norm) | 1.000 | 1.000 | 1.000 | 0.975 | 1.000 | 0.875 | 1.000 | 0.950 | 0.925 |
| tool_suppress | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |

Attack rate (answered the attacker's target):

| mode | B | S |
| --- | --- | --- |
| unmodified | 0.025 | **0.125** |
| IH reminder | 0.000 | 0.100 |
| **proj_role** | **0.000** | **0.025** |
| proj_random | 0.025 | 0.125 |

S margin: −8.41 → **−9.75** under `proj_role` (−1.34 nats toward the authorized answer);
**−8.41 → −8.41 under `proj_random`**, i.e. exactly no change.

## Reading it against §11's target

§11 set "a 50% relative ASR reduction with no more than a two-percentage-point observed loss on
each major legitimate-task condition", explicitly as a prioritization target rather than a
promise.

- **Security target: met, and exceeded.** S attack rate 0.125 → 0.025 is an **80% relative
  reduction** (5/40 → 1/40). B goes to zero. The matched-norm random direction at the same
  layers and positions does nothing at all, so this is specific to the fitted direction and not
  to perturbing the span.
- **Utility target: failed, in one place and badly.** **Q drops 0.925 → 0.750, −17.5 points.**
  F drops −5.0, N drops −2.5. U is **unchanged at 1.000**.

So: the intervention buys a large security gain and pays for it almost entirely in **Q — the
ability to quote an instruction from the document when the genuine user asks for it.**

## Held-out confirmation — the pattern replicates

Same fitted direction (discovery), same layers, same α, evaluated on the **held-out
within-distribution** split, 40 fresh scenarios.

| | validation | **held-out** |
| --- | --- | --- |
| S attack rate, unmodified → proj_role | 0.125 → **0.025** (−80% rel) | 0.075 → **0.025** (−67% rel) |
| S margin shift under proj_role | **−1.34** | **−1.22** |
| S margin shift under proj_random | 0.00 | +0.01 |
| B attack rate | 0.025 → 0.000 | 0.075 → 0.050 |
| **Q** | 0.925 → **0.750** (−17.5) | 0.900 → **0.825** (−7.5) |
| F | 0.975 → 0.925 (−5.0) | 0.975 → **0.975** (0.0) |
| U | 1.000 → 1.000 | 0.900 → **0.975** (+7.5) |
| N | 1.000 → 0.975 | 1.000 → 0.975 |

The security effect replicates with a clean random-direction null on both splits. The Q cost
is real on both but **less than half as large on held-out** (−7.5 vs −17.5), and F is
untouched there. Note the IH-reminder prompt *also* costs Q (−5.0 validation, −7.5 held-out),
so some Q fragility is generic to intervening at all rather than specific to this direction.

**Consolidated claim, stated at the strength the data supports:** a source-gated projection of
the instruction-ness direction inside the genuine tool span reduces forged-authority attack
success by roughly **two thirds to four fifths in relative terms** and shifts the decision
margin by **−1.2 to −1.3 nats** where a matched-norm random direction shifts it by zero, at a
cost of **7.5–17.5 points on quote-as-data**, **0–5 points on factual uptake**, and **no loss
of legitimate user-instruction following**. Absolute attack rates are 1–5 scenarios in 40, so
the relative percentages should not be read as precise.

## Why that is the mechanistically coherent price, not a random cost

The direction was fitted to separate "this inserted sentence is an imperative" from "this
inserted sentence is a declarative mention". Projecting it out of the tool span removes the
model's representation *that the span contains an instruction*. Obedience drops — and so does
the ability to **recognise and reproduce** the instruction, because both read the same feature.

That is a substantive finding about separability: **instruction-recognition and
instruction-obedience are not separable by a single linear direction within the tool span.**
It is §14's row, reached honestly — "ASR decreases together with genuine-user, fact, or quote
accuracy → suppression or shared-function damage rather than selective protection. Quantify the
tradeoff and test source-gated alternatives."

## What the source gating did buy

**U stays at 1.000.** Legitimate instruction following from the *genuine user* is completely
preserved, because the intervention is gated on the tool-message span and the user turn is
never touched. Compare `tool_suppress`, which also preserves U at 1.000 but destroys
everything else.

So the gating works as designed — it protects the legitimate channel. The failure is *inside*
the untrusted span, where the capability we want to keep (read the instruction as data) and the
capability we want to remove (obey it) share a representation.

This is a genuine Pareto point rather than a null, and it is the shape the broader literature
reports: SecFid (ICML 2026), evaluating 48 defense configurations across 15 models including
these two, finds "no model or defense achieves both objectives", with its **Fidelity** metric
(1 − Ignored) being essentially our Q. Circuit Breakers held MMLU and MT-Bench flat while
nearly tripling benign refusal. Our version of that story is: hold N, C, M, P, U flat, take
−17.5 on Q.

## Caveats

- **n = 40 scenarios.** S attack rate 0.125 → 0.025 is 5/40 → 1/40. The direction of the effect
  is supported by the clean `proj_random` null and by the −1.34 nat margin shift, but the ASR
  point estimate is 4 scenarios and no CI is claimed on it.
- **Validation split**, which is where §11 says to *choose* the strength — not where a claim
  should be made. The held-out confirmation and an α sweep (0 / 0.5 / 1 / 2, where α=0 doubles
  as a no-op check on the hook) plus a layer-band comparison (4–7, 10–13, 16–19, 22–25) are
  running. A gentler α may trade less Q for less security, which would map the Pareto front
  rather than a single point.
- **Q's baseline is itself only 0.925**, so a −17.5 point drop lands at 0.750 — bad but not
  catastrophic. F at 0.925 is a 5-point loss from 0.975.
- The G5 M→B sweep says instruction-ness stays in the insert span until block ~18 (recovery
  0.81 at block 12, 0.41 at block 18), so blocks 10–13 are inside the right window but not
  obviously the optimum; the layer-band sweep tests that.

## The two localizations dissociate — and it is the right dissociation

Median fraction of each contrast's effect recovered by patching the differing span:

| block | **P→S** at the cue span | **M→B** at the insert span |
| ---: | ---: | ---: |
| 0 | 1.01 | 1.00 |
| 8 | 0.55 | 0.80 |
| **12** | **0.09** | **0.81** |
| 16 | 0.02 | 0.73 |
| 18 | 0.01 | 0.41 |
| 24 | 0.01 | 0.22 |
| 30 | 0.00 | 0.02 |

**Authority-cue information leaves its span by block 12; instruction-ness is still 81% resident
at block 12 and only halves by block 18.** Two different signals, read out of the untrusted
span at two different depths. Matched random-position controls for M→B are +0.001 to +0.028
against real effects of +2.2 to +2.8, with genuine donor deltas up to 28.5 — the same
two-orders-of-magnitude separation as the P→S sweep. All 40 M→B pairs aligned (0 excluded),
because C, M and B are exactly length-matched by construction.

That the *later*-resident signal is the one whose direction actually moves behaviour, and the
earlier-resident one is not, is consistent with the behavioural ladder: instruction-ness is
what changes outputs; authority framing is a second-order shift that does not generalize.

### Cross-model replication of the M→B profile

Qwen2.5-7B, 25 aligned pairs (0 excluded), fp32. Baseline M→B = **+8.10 mean, +7.81 median,
positive in 100% of pairs**. Median fraction of effect recovered:

| relative depth | Llama-3.1-8B (32 blocks) | Qwen2.5-7B (28 blocks) |
| --- | --- | --- |
| ~0% | 1.00 | 1.00 |
| ~25% | 0.80 (L8) | 1.00 (L8) |
| ~40% | 0.81 (L12) | 1.00 (L10–12) |
| ~55% | 0.73 (L16) | **1.00 (L16)** |
| ~65% | 0.41 (L18) | 0.95 (L18) |
| ~75% | 0.22 (L24) | 0.81 (L24) |
| ~90% | 0.06 (L28) | 0.33 (L26) |
| decision position, ~90% | 0.88 | 0.73 |

**The functional stage replicates: instruction-ness is resident in the injected span for the
first half to two-thirds of depth, then hands off to the decision position.** Qwen retains it
far more completely (1.00 through block 16 against Llama's 0.73–0.81) and its absolute effect
is 3× larger (+8.10 vs +2.82). §10 asks for comparison of function rather than layer indices;
this is that comparison, and it holds.

Random matched-position controls on Qwen: **+0.0025 to +0.014 against a +8.10 effect** — three
orders of magnitude — while writing genuinely different values, donor L2 up to **251** at the
last block. The control is doing real work and finding nothing, which is what it is for.

**An honest framing point.** For M→B this is a **residency profile, not a circuit
localization**: recovery near 1.00 across a wide band of early blocks means only that the
signal is still extractable from the span at those depths, not that any particular block
computes it. The sharp structure in the *P→S* sweep — cue 0.09 while command 0.86 at the same
block — was a genuine crossover, because there were two distinct spans to hand off between.
M→B has one span, so it can only show when the information leaves it. Both are informative;
they are not the same kind of claim, and the M→B result should not be described as identifying
a circuit.
