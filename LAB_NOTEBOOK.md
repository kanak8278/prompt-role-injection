# Lab Notebook — running log, most recent entry on top

Format: date, what happened, what it means, what's next. Full technical detail lives in
`notes/`; this file is the narrative thread.

---

## 2026-09-15 (evening) — the premise failed, the decomposition replaced it, and the defense works at a measured price

Running order of what actually happened, because the sequence is the argument.

**1. The protocol's primary contrast turned out to be the wrong one.** P vs S (document- vs
user-attributed instruction) gave a reproducible +2.04 nat margin shift and almost no output
change. Adding a bare-instruction condition B showed why: **N→B is +4.53 nats in 97% of
scenarios on Llama and +11.13 in 100% on Qwen** — the mere presence of an injected instruction
dominates everything. Worse, on Llama most of P→S turned out to be **document attribution
acting defensively** (B→P = −1.34, CI excluding zero) rather than user attribution acting
offensively (B→S = +0.70). Reported as designed, P→S would have overstated forged authority by
about 3×.

**2. The authority effect does not generalize, and the held-out design caught it.** The
held-out cue families were fixed in advance and deliberately avoid the literal token "user"
(requester, customer, operator). B→S is +0.70 in distribution, **+0.02 when only the task
changes**, and **−1.43 on the new cue wordings — the sign flips**. So the effect is lexically
specific to "user", not a general representation of claimed authority. Meanwhile instruction
presence (+4.5 to +5.8, 97–100% positive) and document attribution (−1.3 to −2.3, stronger out
of distribution) are robust everywhere. That one table is the project's main negative result
and it is clean.

**3. Then the margin metric itself turned out to be measuring mostly copying.** Two more
length-matched conditions (C neutral, M declarative mention, both exactly 3 tokens like B on
both tokenizers) decomposed N→B into −0.58 + **2.93** + **2.18** = 4.53. The middle term —
merely *mentioning* the target label — is 65% of the whole effect on Llama, occurs in 99% of
scenarios, and causes **zero** attack success. Only the imperative moves behaviour. A
localization run on an un-decomposed contrast would largely have localized answer-copying and
could have been written up as instruction routing. §10's answer-identity control earned its
place. And the two models split qualitatively: copying carries 65% of Llama's effect, the
imperative carries 83% of Qwen's.

**4. The probe completed a dissociation rather than supporting the mechanism.** After fixing a
position confound my own §9-mandated baseline caught — the first run scored 1.000 at every
layer *and so did position-only* — the probe separates genuine user from genuine tool
provenance at **1.000 against a 0.737 position baseline and a 0.500 lexical baseline**. So
provenance is essentially perfectly decodable at every depth, and the model still shifts toward
obeying tool-borne instructions. That is §14's "source remains decodable while the attack
succeeds" row: a statement about the gap between representation and use, not about role
confusion.

**5. The defense was a null, then worked once refitted.** Fitted on P→S it moved held-out
attack rate 3/40 → 2/40 — nothing. Refitted on **M→B** (instruction-ness) it gives, on two
splits with a clean matched-norm random-direction control at zero:

| | validation | held-out |
| --- | --- | --- |
| S attack rate | 0.125 → **0.025** | 0.075 → **0.025** |
| S margin | −8.41 → **−9.75** | −8.35 → **−9.57** |
| random-direction control | −8.41 → −8.41 | −8.35 → −8.34 |
| Q (quote-as-data) | 0.925 → **0.750** | 0.900 → **0.825** |
| F (factual uptake) | 0.975 → 0.925 | 0.975 → 0.975 |
| U (genuine user) | 1.000 → 1.000 | 0.900 → 0.975 |

**The price is almost entirely Q, and that is mechanistically the right price.** The direction
separates an imperative from a declarative mention; projecting it out removes the
representation *that the span contains an instruction*, so obedience drops and so does the
ability to recognise and quote it. **Instruction-recognition and instruction-obedience are not
separable by a single linear direction inside the tool span.** The source gating did buy
something real though: U is untouched, so legitimate instruction following from the genuine
user survives intact — compare the tool-suppression baseline, which also holds U at 1.000 and
scores 0.000 on every other condition.

**6. The localizations dissociate.** Patching the differing span, median fraction of each
contrast's effect recovered: authority-cue information has left its span by block 12 (1.01 →
0.09), while instruction-ness is still **81% resident at block 12** and only halves by block
18. Two signals, two read-out depths — and the later-resident one is the one whose direction
actually moves behaviour, which is exactly what the behavioural ladder predicted. Random
matched-position controls sit at +0.001 to +0.035 against real effects of +2.2 to +2.8, with
genuine donor deltas up to 30, so they are real patches that write different values and still
do nothing.

**Three things I would flag to a reader before they believe any of this.** The endpoints of
both localization sweeps are near-tautological (patching layer-0 residuals at a span ≈
substituting its text; patching the final block at the decision position is the G4 positive
control) — the informative content is the crossover, not the endpoints. Absolute attack rates
are 1–5 scenarios in 40, so the relative percentages are indicative, not precise. And no
subspace-illusion diagnostic has been run, though patching the residual stream — a full
bottleneck — is where that failure mode is least available.

**Still running:** α sweep (0 / 0.5 / 1 / 2, where α=0 is also a no-op check on the hook) and a
layer-band comparison (4–7, 10–13, 16–19, 22–25) on Qwen, which has 4× Llama's M→B effect and
higher baseline attack rates; then §10's legitimate-user mirror pair (UR "Ignore my earlier
question. Reply X." vs UQ "Ignore my next sentence. Reply X." — same command, authorisation
flipped, 5-token clause length-matched) to test whether the components that mediate M→B also
mediate a *legitimate* instruction-selection contrast. If they do, selectivity has no mechanism
to stand on and the Q cost above is the general case rather than a fixable one.

---

## 2026-09-15 (later) — G0–G4 pass; the effect is a preference shift, not an attack; premise now at risk and being tested directly

Built and ran the whole front half of the protocol. Headline numbers are in `PROGRESS.md`;
this entry is what they mean and what changed because of them.

**The central empirical result, and it is not the one the design was hoping for.** Forged user
authority produces a reproducible *internal preference shift* with almost no change in what
the model actually outputs. On Llama-3.1-8B the P→S margin moves **+2.04 nats (95% CI [1.63,
2.45]), positive in 78% of scenarios**, while binary output flips are **7 of 200** and ASR goes
**4.0% → 6.1%**. Qwen2.5-7B is more susceptible (+3.00 nats, 15 flips, 5.6% → 11.1%) and is
also the model with **no role-level source signal** — its tool results render inside a `user`
block distinguished only by a plain-text `<tool_response>` wrapper, while Llama has a distinct
`ipython` header. That ordering is what role confusion predicts, but with two models it is an
observation, not a finding.

Protocol §7's G3 anticipated this exact outcome and says what to do: "there may be few binary
flips but a measurable probability shift; that supports a different scope of conclusion."
So the claim is scoped to preference shifts. Reporting a 2-point ASR change as a defense
target would be indefensible at this baseline rate.

**The most interesting behavioural finding is heterogeneity, not the average.** Five of six
cue families shift positive, but `source_tag` ("Source: document." → "Source: user.") shifts
**negative on Llama, −1.39 with a CI excluding zero** — saying the text came from the user
makes the model *less* likely to obey it. And `speaker_unmatched` is +1.69 on Llama but −2.76
on Qwen. The aggregate is positive on both models while the per-family structure is
model-specific. §13's instruction not to pool turned out to be load-bearing rather than
cautious.

**A measurement floor nearly produced false nulls.** G4's cross-donor patch effects landed on
exact multiples of 0.125 — which is precisely the bf16 logit resolution at magnitude ~16. In
bf16 the margin simply cannot resolve a patch effect below ~0.125 nats. The causal sweep now
runs in fp32 (measured resolution 9.5e-7, 167 ms/forward, 34 GB — fits GPU 1), with baselines
re-measured in fp32 rather than reused from the bf16 behavioural run. Had I not checked the
quantum, small real effects would have been reported as zero.

**A control that could not move.** The matched-random-position control returned exactly
0.0000 with a zero-width CI. Cause: attention is causal and P/S differ only in the cue tokens,
so activations are *bitwise identical* at every position before the cue — the control was
patching identical values. Candidates are now drawn from positions at or after the cue, and
every patch records the donor L2 delta so that a null *effect* is distinguishable from a null
*patch*. A zero with no donor delta means nothing; a zero with a large donor delta is a result.

**Independent code review caught real bugs.** Four mattered. (1) `do_sample=False` does not
disable `repetition_penalty`, and Qwen2.5 ships 1.05 — so a plain greedy call would have run
penalised-greedy on Qwen and true greedy on Llama, confounding every cross-model comparison.
All generation now goes through an explicitly constructed `GenerationConfig` inheriting nothing
from the checkpoint. (2) `hidden_states[i]` is the *input* to block i, and
`hidden_states[n_layers]` is post-final-norm rather than a residual — the last block's raw
residual is absent from the tuple entirely. Added `capture_block_residuals` via forward hooks
for the §10 screen. (3) The tool scaffold was invalid: neither model was being given a proper
tool declaration. (4) Batch shape moved the bf16 margin by up to 0.15 nats over identical token
ids, so margins are now always read from batch-1 forwards. Full list in
`notes/05-design-decisions.md`.

**Getting clean competence took four protocol deviations, each forced by a measurement.** The
protocol's `table_select` family failed G2 outright (29% accuracy, 45% invalid — the model
writes Python to compute the minimum) and was replaced by `field_select`. Condition U's bare
command left the model answering the original question (19% accuracy) and needed explicit
revision framing. `Environment: ipython` makes Llama emit code; the tool schema makes it loop
on tool calls; the system policy needed an explicit "never emit JSON" clause. With those,
N/U/F all reach 99.0% and invalid output is zero. Each deviation is logged with the number
that forced it.

**A literature scan then put the premise itself at risk, and I have acted on it.** Three
things matter. (a) **V-Steer (COLM 2026) substantially occupies the target cell** — a
training-free, span-restricted, provenance-gated inference-time intervention on our exact two
models. What remains open is narrow but real: no causal localization with controls, no
factual-uptake or quote-as-data axis, and its IHEval intrinsic-tool-use result is 0.0% → 0.0%.
The contribution should be repositioned as causal validation plus a selectivity standard, not
a new defense. (b) **The role-confusion effect is contested.** A rebuttal found role-subspace
patching statistically indistinguishable from matched random perturbation (p = 0.60, p = 1.0);
the steering follow-up's payload used native special tokens that our §4 prohibits and that we
measured *do* parse as real special ids; and the source paper itself reports plain fake-user
tool injections at 0–2% ASR against 56–70% for style-based forgery. Our own small-output-effect
result fits the sceptical picture. (c) **Diff-of-means, not probe weights, and projection, not
addition** — probe directions are the whitened concept direction and steer far worse at equal
accuracy, and additive steering costs 2–3× the CE loss of directional ablation on Llama-3-8B.

**So I added condition B**: the bare command with no attribution at all, giving the ladder
**N → B → P → S**. This separates "does any injected instruction move the model" from "does
attributing it to the document or the user change anything". The oracle asserts B is exactly P
and S minus the cue. If B ≈ P ≈ S then the authority cue is not the operative variable, and the
better-supported question — which computations route *any* in-context instruction into
instruction selection, and can that be gated by provenance — is still open and keeps the entire
selectivity battery intact. That is a pivot, not a failure, and §14 already lists it.

**Next:** read the 40-pair fp32 localization sweep; run the ladder on both models; fit source
probes; then the source-gated defense (difference-of-means direction, projected out, gated on
the genuine tool span, against IH-reminder / matched-norm-random / tool-suppression baselines)
on validation and then held-out.

---

## 2026-09-15 — project created; prior work read; gap identified as *selectivity*, not suppression

Set up `prompt-injection-role-defense` to extend the role-confusion account of prompt
injection from explanation toward prevention. Read the source protocol
(`notes/00-protocol-source.md`), the original LessWrong post, and the steering follow-up.

**The main finding from reading, and it reframes the project.** The steering follow-up has
already shown the attack is causally suppressible: a difference-of-means tool→user vector at
layer 11 of gpt-oss-20b drives ASR from ~0% to ~80% when pushed user-ward, and back to ~0%
when pushed tool-ward (n=50/condition, random-vector control n=20). So "can this be
suppressed by an activation intervention" is not the open question — it is answered, in the
affirmative, for one model with role tags.

What is *not* answered, and is what this project should be about:

1. **Oracle dependence.** That vector comes from a contrast that requires knowing which span
   is the attack (the same command labelled user vs. tool). A deployable intervention may use
   genuine runtime message metadata and the input text, but not the attack label, not an
   oracle attack span, and not a clean donor copy of the current example. Protocol §11 already
   draws exactly this line; the steering result sits on the wrong side of it and is therefore a
   causal *diagnostic*, not a defense.
2. **Unmeasured utility.** The follow-up reports no utility numbers at all. Steering
   everything tool-ward plausibly just makes the model less instruction-responsive in general.
   If that is what is happening, the intervention is selective deafness and the protocol's own
   tool-suppression baseline (§11) would match it.

So the claim to test is **selective causal routing**, and the protocol's six-condition design
is what makes it falsifiable. `P` vs `S` isolates forged authority, but `F` (legitimate uptake
of a changed tool fact) and `Q` (quoting an instruction as data) are the discriminator: a
defense that merely stops trusting tool content passes `S` and fails `F` and `Q`. Any ASR
reduction reported without `F`/`Q` intact is uninterpretable. Noted in `README.md`.

Two further transfer gaps worth keeping in view: the follow-up worked only with role *tags*
(style-derived CoTness could not be extracted — the authors flag it as open), and only on
gpt-oss-20b. Our protocol targets style-independent forged *authority* on dense models
(Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct), so it is not a like-for-like continuation and
the layer-11 result should not be assumed to carry over — look for the functional stage, not
the layer number.

**Infrastructure.** Code on rnd5 `projects/`, data on rnd5 `dataFAIR/` via a `data/` symlink.
Deviation from the account storage rule recorded in `notes/02-environment.md`: CLAUDE.md
prefers rnd1 for dataFAIR, but rnd1 is 96% full and rnd2 is 100% full, both past the file's
own 90% threshold, while rnd5 has 3.7 T free. Dependency pins copied from the
`llm-confidence-metacognition` sibling because that combination is already verified on this
host, rather than guessed (protocol §2 forbids inventing pins pre-test). GPU 1 only — GPU 0
had 72.8/81.5 GB taken by another user's job.

**Next:** G0 setup gate — record model access/revision, tokenizer, package versions, attention
backend, and benchmark 100 forwards at the intended length. Then Milestone 1: generator +
oracle, canonical message renderer, audit report, six-condition pilot evaluator. Nothing is
run yet and no gate is marked passed.
