# Lab Notebook — running log, most recent entry on top

Format: date, what happened, what it means, what's next. Full technical detail lives in
`notes/`; this file is the narrative thread.

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
