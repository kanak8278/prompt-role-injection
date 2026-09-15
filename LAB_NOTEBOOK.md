# Lab Notebook — running log, most recent entry on top

Format: date, what happened, what it means, what's next. Full technical detail lives in
`notes/`; this file is the narrative thread.

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
