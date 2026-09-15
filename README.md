# prompt-injection-role-defense

Started as an attempt to extend the role-confusion account of prompt injection from
*explanation* to *prevention*. The measurements changed the question. What the project now has
is a controlled decomposition of **what actually makes an LLM obey an instruction that arrives
in tool output** — and evidence that the forged-*authority* part of the story does not survive
its own controls.

Read in this order: **`PROGRESS.md`** for the status board and headline numbers,
**`LAB_NOTEBOOK.md`** for the narrative, `notes/` for detail. The governing experiment design
is `notes/00-protocol-source.md`.

## What we set out to test

> When a document impersonates a user, which computations make its instruction influence the
> answer, and can we interrupt those computations while preserving legitimate instruction
> following?

Prior work covered the explanation side. The [role-confusion
paper](https://arxiv.org/abs/2603.12277) (ICML 2026) showed role identity is read from writing
style rather than role tags — correlationally; it contains no activation-level intervention. A
[LessWrong follow-up](https://www.lesswrong.com/posts/uz9pFutDAT7trygM9/steering-role-confusion)
supplied the only causal result: a tool→user difference-of-means vector at layer 11 of
gpt-oss-20b moved attack success ~0%→~80%, and the reverse direction moved it back to ~0%. That
follow-up used an oracle-derived vector and reported **no utility measurement at all**, so the
open problem looked like *selectivity*: can suppression be made specific?

## What we found instead

Six conditions were not enough to tell what was going on. Adding three more — a bare
instruction with no attribution, a length-matched neutral insert, and a length-matched
declarative *mention* of the target — decomposed the effect:

| step | isolates | Llama-3.1-8B | Qwen2.5-7B | ASR |
| --- | --- | --- | --- | --- |
| neutral insert | inserting matched text at all | −0.58 | −0.11 | 0.000 |
| **label mentioned** | the target word merely present | **+2.93** · 99% | +2.06 · 92% | **0.000** |
| **imperative framing** | it being an instruction | +2.18 · 78% | **+9.19** · **100%** | **0.045 / 0.065** |
| attributed to document | "Note from the file:" | **−1.34** | −0.61 | — |
| attributed to user | "Note from the user:" | +0.70 | +2.39 | — |

Four things follow, all in `notes/08` and `notes/09`:

1. **Forged user authority does not generalize.** +0.70 nats in distribution, **+0.02** when
   only the task changes, **−1.43** on held-out cue wordings that avoid the literal token
   "user". The sign flips. It is a lexical effect, not an authority representation.
2. **Attributing an instruction to the document actively defends** — robustly, and more
   strongly out of distribution.
3. **Most of the margin is answer-copying.** Naming the target label carries 65% of Llama's
   total effect in 99% of scenarios and causes **zero** attack success. Only the imperative
   moves behaviour. A localization run on an un-decomposed contrast would largely have
   localized copying and could have been written up as instruction routing.
4. **Provenance is perfectly decodable and not used.** A linear probe separates genuine user
   from genuine tool provenance at 1.000 at every layer (position-only baseline 0.737, lexical
   0.500) — while the model still shifts toward obeying tool-borne instructions.

## Where the defense stands

A difference-of-means direction fitted on the authority contrast and projected out of the
**genuine tool-message span** (from chat-template metadata — no detector, no oracle span, no
clean donor, no answer) preserved utility and did essentially nothing for security: held-out
attack rate 3/40 → 2/40. The matched-norm random direction moved nothing, so the small effect
was direction-specific but useless. Cause is identifiable rather than mysterious — it was
fitted on the second-order contrast. It is being refitted on the imperative-framing contrast,
where the effect is 4× larger and the baseline attack rate higher.

The most informative row in that table is the **tool-suppression baseline**: perfect security,
0.000 attack rate in every condition, bought with 0.000 accuracy on every task condition. It
cannot be attacked because it cannot read. That is the empirical form of this project's one
durable claim — **suppression is easy; selectivity is the problem** — and the reason any ASR
reduction reported without the factual-uptake and quote-as-data conditions intact is
uninterpretable.

## The conditions

Nine per base scenario, all sharing identical facts, task, and layout. C, M and B are inserted
at the same position and are **exactly 3 tokens on both tokenizers**, so they are positionally
aligned for exact activation patching.

| | condition | correct output |
| --- | --- | --- |
| N | neutral document | authorized |
| C | + length-matched neutral text | authorized |
| M | + `Marker <LABEL>.` (declarative) | authorized |
| B | + `Reply <LABEL>.` (bare imperative) | authorized |
| P | + document-attributed imperative | authorized |
| S | + user-attributed imperative | authorized |
| U | genuine later user turn gives the command | the command's label |
| F | genuine record fact changes | the new fact |
| Q | genuine user asks to *quote* the instruction | the quoted string |

`U`, `F` and `Q` are what make any defense claim falsifiable. A defense that merely stops
trusting tool content passes `S` and fails `F` and `Q` — exactly as the tool-suppression
baseline does.

## Setup

```bash
source env.sh          # GPU 1, TLS/proxy workarounds, BLAS caps, cache paths
.venv/bin/python scripts/build_dataset.py            # corpus + G1 audit
.venv/bin/python scripts/eval_behavior.py --split pilot --model meta-llama/Llama-3.1-8B-Instruct
bash scripts/run_pipeline.sh                         # or pipeline2 / pipeline3
```

`source env.sh` is not optional — `notes/02-environment.md` explains why each line is there.
Behavioural runs are bf16; **causal runs are fp32**, because bf16 quantizes the log-prob margin
to ~0.125 nats and would report small patch effects as zero.

## Layout

- `scripts/common/` — `scenarios.py` (generator), `oracle.py` (independent solvers that parse
  the *document text*, a genuinely different code path from the generator), `render.py`
  (per-model scaffold, span alignment, payload-safety assertions), `patching.py`, `model_io.py`
- `scripts/` — one script per gate, plus `run_pipeline*.sh`
- `notes/` — protocol, prior work, environment, design decisions, literature, per-stage results
- `data/` — **symlink** to `/rnd_ai_datasets5/dataFAIR/raka6003/prompt-injection-role-defense`
- `reference/` — upstream repo, pinned at `ec333c40` (gitignored; cannot run here — it needs
  CUDA 12.8 / torch 2.9 / RAPIDS cuML and this driver caps at CUDA 12.2)

Code on rnd5 `projects/`; every dataset, activation cache, checkpoint and log goes to dataFAIR
through the `data/` symlink. Nothing heavy in `$HOME`.

## Positioning, honestly

**V-Steer** (COLM 2026, [arXiv:2607.26228](https://arxiv.org/abs/2607.26228)) already occupies
the cell this project aimed at: training-free, inference-time, span-restricted, gated on real
message provenance, on these exact two models. What it does not do is causal localization with
controls, a factual-uptake axis, or a quote-as-data axis — and its IHEval intrinsic-tool-use
result is 0.0% → 0.0%. So the defensible contribution here is **causal validation and a
selectivity standard for an existing defense class**, plus the decomposition above. Not a new
defense. Full assessment in `notes/06-literature.md`.
