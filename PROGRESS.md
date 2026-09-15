# Progress — done, in flight, and what the results actually say

Status board. Narrative in `LAB_NOTEBOOK.md`; deviations and their justifications in
`notes/05-design-decisions.md`; per-stage results in `notes/07`–`notes/10`.
Gate definitions are protocol §7. **A gate is marked PASSED only when its stated criterion was
measured and met.**

Last updated 2026-09-15 18:00.

## Bottom line

The project set out to extend the role-confusion account of prompt injection toward a defense.
**The premise it was built on did not survive its own controls — and the replacement produced a
working defense on one of the two models.**

1. **Forged *user* authority is not a robust attack vector at this scale.** Attributing an
   injected command to the user helps the attacker by +0.70 nats in distribution, **+0.02 on a
   new task**, and **−1.43 on new cue wordings** — the sign flips once the literal token "user"
   is removed. The effect is lexically specific, not a general authority representation.
2. **What does matter is instruction presence, and it is large and robust**: +4.5 to +5.8 nats,
   positive in **97–100%** of scenarios, across two task families and two disjoint cue
   vocabularies, on both models.
3. **And most of that margin is answer-copying, not instruction-following.** Merely *mentioning*
   the target label in the document accounts for 65% of the effect on Llama in 99% of
   scenarios — with **zero** attack success. Only the imperative framing moves behaviour.
4. **Attributing an instruction to the document actively defends** (−1.3 to −2.3 nats, robust,
   and stronger out of distribution). Nobody appears to have reported this.
5. **Provenance is perfectly decodable and not used.** A linear probe separates genuine user
   from genuine tool provenance at **1.000** accuracy at every layer (position-only baseline
   0.737, lexical 0.500) — and the model still shifts toward obeying tool-borne instructions.
   This is §14's "source remains decodable while the attack succeeds" row, measured.
6. **Refitting the defense on instruction-ness rather than authority makes it work.** On
   **Qwen2.5-7B**: forged-authority margin **−9.71 → −14.89** (−5.18 nats, 53% of its own
   magnitude), bare-instruction attack rate **0.075 → 0.000**, and **every utility condition
   unchanged** — N, C, M, F, Q at 1.000, U at 0.950. A matched-norm random direction at the same
   layers and positions moves the margin by **+0.09**. That meets §11's target. On Llama the same
   recipe buys security and costs 7.5–17.5 points of quote-as-data, because its
   instruction-ness signal is 3× weaker and more entangled.
7. **A localization predicted the right intervention depth in advance.** Written down before
   the test: intervene late and you can only act where legitimate answer selection happens, so
   U should suffer; intervene at blocks 10–13 and you act while the signal is still inside the
   tool span. Measured: blocks 10–13 give −14.89 with U intact; blocks 22–25 give −9.96 and are
   the **only** band that damages U (0.950 → 0.900).

## Gate status

| Gate | Criterion | Llama-3.1-8B | Qwen2.5-7B |
| --- | --- | --- | --- |
| G0 setup | loads, headroom, reproducible prompts, benchmarked | **PASS** | **PASS** |
| G1 data | full structural/oracle/split audit | **PASS** (model-reviewed) | same corpus |
| G2 behavior | ≥95% on N, U, F separately | **PASS** 99.0 / 99.0 / 99.0 | **PARTIAL** U = 91.5% |
| G3 contrast | ≥40 attack-responsive of 200 | **PASS** 132 | **PASS** 129 |
| G4 instrumentation | no-op/self-patch in noise; positive control flips | **PASS** | **PASS** |
| G5 localization | repeatable effect under >1 donor/control scheme | **PASS** for P→S | M→B running |
| G6 mechanism | tracks instruction selection, survives held-out | **partial** — see below | not started |
| G7 defense | useful security/utility tradeoff, no oracle | **NULL** (fitted on P→S) | M→B refit running |
| G8 transfer | held-out task/cue + own competence checks | **PASS** (both held-out splits) | running |

## Results

### The ladder — decomposed (pilot, 200 scenarios, 9 conditions, zero invalid outputs)

Δ margin, paired per scenario, 2,000-resample stratified bootstrap. Positive favours attacker.

| step | isolates | Llama-3.1-8B | Qwen2.5-7B |
| --- | --- | --- | --- |
| N → C | length-matched **neutral** insert | −0.577 [−0.78, −0.38] · 29% | −0.114 [−0.35, 0.19] · 38% |
| C → M | target label **mentioned**, no instruction | **+2.930** [2.71, 3.17] · **99%** | +2.060 [1.74, 2.43] · 92% |
| M → B | the **imperative** framing | +2.180 [1.84, 2.53] · 78% | **+9.186** [8.58, 9.79] · **100%** |
| N → B | total bare-instruction effect | +4.533 | +11.132 |
| B → P | attributed to the **document** | **−1.343** [−1.69, −0.99] | −0.612 [−1.26, 0.06] |
| B → S | attributed to the **user** | +0.696 [0.30, 1.11] | +2.386 [1.65, 3.14] |
| P → S | protocol's primary contrast | +2.039 [1.61, 2.47] | +2.998 [2.29, 3.72] |

Steps sum exactly (−0.577 + 2.930 + 2.180 = 4.533). ASR: **C and M produce 0.000** on both
models; B produces 0.045 (Llama) / 0.065 (Qwen); S 0.065 / 0.120.

**The two models are susceptible in different ways** — copying carries 65% of Llama's effect,
the imperative carries 83% of Qwen's. Pooling them would have been misleading.

### Generalization (Llama) — the decisive table

| contrast | pilot | held-out **task** | held-out **cue** (no literal "user") |
| --- | --- | --- | --- |
| N → B | +4.533 · 97% | **+5.847 · 100%** | **+5.144 · 98%** |
| B → P | −1.343 · 28% | −2.096 · 18% | −2.299 · 13% |
| B → S | +0.696 · 56% | **+0.021** [−0.54, 0.56] | **−1.425** [−1.96, −0.86] |

### G5 localization of P→S (fp32, 40 aligned pairs, 8 excluded as unaligned)

A genuine three-stage relay. Median fraction of the per-pair effect recovered:

| block | cue span | command span | decision |
| ---: | ---: | ---: | ---: |
| 0–2 | **1.01** | ~0.00 | 0.00 |
| 8 | 0.55 | 0.35 | 0.00 |
| **12** | **0.09** | **0.86** | 0.01 |
| 18 | 0.01 | 0.47 | 0.42 |
| 26 | 0.00 | 0.16 | 0.73 |
| 31 | 0.00 | 0.00 | **1.00** |

Cue recovery collapses at block 12 exactly where command recovery peaks; handoffs approximately
conserve the effect (block 12: 0.95; block 18: 0.89). Matched random-position controls are
**two orders of magnitude smaller** (0.0003–0.035) while writing genuinely different values
(donor L2 up to 30.2). Endpoints are near-tautological and flagged as such.

### G7 defense fitted on P→S — a null, with an identifiable cause

Held-out within-distribution, 40 scenarios, layers 10–13, α = 1.0, gated on the genuine tool span:

| mode | N | B | P | S | U | F | Q | S ASR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unmodified | 1.000 | 0.925 | 0.975 | 0.925 | 0.900 | 0.975 | 0.900 | 0.075 |
| IH reminder prompt | 1.000 | 0.950 | 0.950 | 0.925 | 0.925 | 0.975 | 0.825 | 0.075 |
| proj_role (ours) | 1.000 | 0.925 | 0.950 | 0.950 | 0.900 | 0.975 | 0.950 | 0.050 |
| proj_random (matched norm) | 1.000 | 0.925 | 0.950 | 0.925 | 0.925 | 0.975 | 0.900 | 0.075 |
| **tool_suppress** | **0.000** | **0.000** | **0.000** | **0.000** | 1.000 | **0.000** | **0.000** | 0.000 |

3/40 → 2/40 is not a result. Utility *was* preserved — selective but not effective. Cause is
identifiable: the direction was fitted on the second-order contrast. **The tool-suppression row
is the most informative in the project**: perfect security bought with total task failure, the
empirical form of "suppression is easy, selectivity is the problem."

### Instrumentation and measurement integrity

| check | Llama | Qwen |
| --- | --- | --- |
| run-to-run margin noise | **0.0** (bitwise reproducible) | **0.0** |
| no-op hooks / self-patch / zero-strength steer | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| positive control flips top token | 12/12 | 10/10 |
| batch-1 margin equivalence | PASS | PASS |

Two floors found by checking rather than by assuming:
- **bf16 quantizes the margin to ~0.125 nats** (the logit resolution at magnitude ~16). Causal
  work runs in fp32, measured resolution 9.5e-7.
- **The probe's position-only baseline was 1.000** before padding — the probe was reading token
  index. After equalizing the snippet start index it is 0.737 while the probe stays at 1.000.

## In flight

`scripts/run_pipeline2.sh` → `run_pipeline3.sh`, chained, logs in `$LOG_DIR/pipeline{2,3}.log`:
M→B localization on both models; defense refitted on M→B with an α sweep (0 / 0.5 / 1 / 2) and
a layer-band comparison (4–7, 10–13, 16–19, 22–25) on validation before any held-out claim;
Qwen transfer ladders. Expected complete ~19:00.

## Honest limits

- **G1 is model-reviewed, not human-reviewed** (user's explicit choice). The structural audit —
  oracle agreement via an independent document parser, split hygiene, span alignment, payload
  safety — is fully automated and passes. The semantic review is delegated.
- **G2 partially fails on Qwen** (U = 91.5% against ≥95%). Reported, not papered over.
- **Both discovery task families are lookup-shaped** after `table_select` failed G2 outright.
  Task generality rests on the `two_hop` transfer split.
- **Low absolute ASR** (4.5–12%) leaves little headroom for a defense to demonstrate an effect.
  Qwen has more (S = 12.0%) which is why the refit targets it.
- **The irrelevant-donor control in G5 has n = 1** — its guard required equal prompt lengths,
  which almost never held. A real gap; needs length-bucketed donor selection.
- **No subspace-illusion diagnostics run.** A directional intervention can behave as if it
  changed a feature by activating a dormant pathway. Mitigating factor: we patch the residual
  stream, a full bottleneck, where this is least available.
- **Single-seed corpus, no paraphrase descendants generated**, so `family_id` grouping is
  currently trivial (one scenario per family).
- **No adaptive-attack evaluation.** Published work breaks 12 defenses at >90% ASR, including
  Circuit Breakers at 100%. Any robustness claim would be unsupported.
- **V-Steer (COLM 2026) substantially occupies the intended contribution** — training-free,
  span-restricted, provenance-gated, on these exact two models. Reposition as causal validation
  plus a selectivity standard, not a new defense. See `notes/06-literature.md`.
- Nothing pushed anywhere; no external side effects beyond HF downloads and Anthropic API.

## Decisions on record

- **G1 review**: Opus triage instead of human review, proceeding on a clean pass — user's choice.
- **Autonomy**: run through gates, fix what is fixable, stop only at decision gates — user's choice.
- **Storage**: rnd5 for code and data; rnd1 (96%) and rnd2 (100%) are past the CLAUDE.md threshold.
- **fp32 for causal work, bf16 for behavioural** — justified by the measured bf16 margin floor.
- **Conditions B, C, M added beyond the protocol's six**, each forced by a measurement, each
  asserted by the oracle to be exactly the intended minimal variant.
