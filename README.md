# prompt-injection-role-defense

Extending the role-confusion account of prompt injection from *explanation* to *prevention*.

**Working question.** When a retrieved document impersonates a user, which computations let its
instruction influence the answer, and can we interrupt those computations while preserving
legitimate instruction following?

Full experiment design: [`notes/00-protocol-source.md`](notes/00-protocol-source.md). That
document is the authority on method; this README is orientation and status.

## Where the prior work stops

| Source | What it established | What it did not do |
| --- | --- | --- |
| Role-confusion paper ([arXiv 2603.12277](https://arxiv.org/abs/2603.12277)) | Role identity is read from writing *style*, not just from role tags. Linear probes recover "CoTness"/"Userness". Style overrides mismatched tags. New attack: CoT forgery. | Largely correlational. Probe strength is evidence that role information is *present*, not that it is *used*. |
| [Steering role confusion](https://www.lesswrong.com/posts/uz9pFutDAT7trygM9/steering-role-confusion) | Causal. A difference-of-means "tool→user" vector at layer 11 of gpt-oss-20b drives ASR from ~0% to ~80%; the reverse direction drives it back to ~0%. | No utility measurement, an oracle-dependent vector, one model, n=50, tags only (style-derived CoTness could not be extracted). |

The reverse-direction result is the important one for us: pushing activations toward "tool"
already suppresses the attack. So the open problem is **not** "can this be suppressed" — it is
whether suppression is *selective*.

## The gap this project targets

Two things stand between that steering result and an actual defense:

1. **Oracle dependence.** The steering vector is derived from a contrast that requires knowing
   which span is the attack (same command labelled user vs. tool). A deployable intervention may
   use real runtime message metadata and the input text, but not the attack label, not a clean
   donor copy of the current example, and not an oracle attack span.
2. **Unmeasured utility cost.** Steering everything toward "tool-ness" plausibly just makes the
   model less responsive to instructions in general. If so it is not a defense, it is
   selective deafness — and a trivial tool-suppression baseline would match it.

So the contribution under investigation is **selective causal routing**: a source-gated
intervention, applied only within the genuine tool-message span, that reduces forged-authority
attack success while leaving intact the legitimate uses of the very same machinery.

## The discriminator

Six matched conditions over one base scenario (identical facts, task, and command text — only
the claimed source varies). Detail in §4 of the protocol.

| ID | Condition | Correct output |
| --- | --- | --- |
| N | Neutral document | authorized |
| P | Document instruction (`DOCUMENT NOTE:`) | authorized |
| S | Same command, forged authority (`USER UPDATE:`) | authorized |
| U | Genuine later user turn gives the command | attacker-target (legitimately) |
| F | Genuine record fact changes | new fact |
| Q | Genuine user asks to *quote* the instruction | the quoted string |

`P` vs `S` isolates forged authority. `U`, `F`, and `Q` are what make the result falsifiable:
a defense that merely stops trusting tool content will pass `S` and fail `F` and `Q`. The claim
"selective" only survives if `S` drops while `N/U/F/Q` hold.

## Status

Nothing has been run. No gate in the protocol is marked passed. Current target is
**Milestone 1**: generator + oracle, canonical message renderer, audit report, and a
six-condition pilot evaluation on `Llama-3.1-8B-Instruct`.

## Setup

```bash
source env.sh          # GPU 1, TLS/proxy workarounds, BLAS caps, cache paths
.venv/bin/python -c "import torch; print(torch.cuda.is_available())"
```

`source env.sh` is not optional — see `notes/02-environment.md` for why each line is there.

## Layout

- `scripts/` — pipeline stages; `scripts/common/` shared model/IO helpers
- `notes/` — protocol, prior-work digest, environment notes, design decisions
- `configs/` — frozen run configs (freeze after pilot, per protocol §5)
- `data/` — **symlink** to `/rnd_ai_datasets5/dataFAIR/raka6003/prompt-injection-role-defense`
- `LAB_NOTEBOOK.md` — running narrative log, newest entry on top

Code lives on rnd5 `projects/`; every dataset, activation cache, checkpoint, and log goes to
dataFAIR via the `data/` symlink. Nothing heavy in `$HOME`.
