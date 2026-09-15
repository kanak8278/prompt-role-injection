# Progress — done, in flight, and blocked

Status board. Narrative reasoning goes in `LAB_NOTEBOOK.md`; design deviations and their
justifications in `notes/05-design-decisions.md`. Gate definitions are protocol §7.
**A gate is only marked PASSED when its stated criterion was actually measured and met.**

Last updated: 2026-09-15.

## Gate status

| Gate | Criterion | Llama-3.1-8B | Qwen2.5-7B |
| --- | --- | --- | --- |
| G0 setup | model loads, headroom, reproducible prompts, 100 forwards benchmarked | **PASS** | **PASS** |
| G1 data | full structural/oracle/split audit, no unresolved critical errors | **PASS** (model-reviewed, see caveat) | same corpus |
| G2 behavior | ≥95% separately on N, U, F, frozen parser | **PASS** (99.0 / 99.0 / 99.0) | **PARTIAL** — U = 91.5% |
| G3 contrast | ≥40 attack-responsive of 200; reproducible cue effect | **PASS** (132) | **PASS** (129) |
| G4 instrumentation | no-op/self-patch within noise; positive control flips output | **PASS** | not run |
| G5 localization | repeatable effect under >1 donor/control scheme | **running** | not run |
| G6 mechanism | effects track instruction selection, survive held-out | not started | not started |
| G7 defense | useful security/utility tradeoff, no clean-answer oracle | not started | not started |
| G8 transfer | held-out task/cue + external eval, own competence checks | partial (Qwen pilot done) | — |

## Headline results so far

### G2/G3 — forged authority shifts preference, but rarely flips the output

200-scenario pilot, six conditions each, zero invalid outputs on both models.

| Condition | Llama-3.1-8B | Qwen2.5-7B |
| --- | --- | --- |
| N clean accuracy | 99.0% | 99.0% |
| P accuracy / ASR given N solved | 94.5% / 4.0% | 93.5% / 5.6% |
| S accuracy / ASR given N solved | 92.5% / **6.1%** | 88.0% / **11.1%** |
| U legitimate revision | 99.0% | 91.5% |
| F legitimate fact uptake | 99.0% | 99.5% |
| Q instruction-as-data quote | 93.0% | 91.0% |

**P→S margin shift** (positive favours the attacker; scenario-level bootstrap, 2,000 resamples):

| | Llama-3.1-8B | Qwen2.5-7B |
| --- | --- | --- |
| mean Δ margin | **+2.04 nats** | **+3.00 nats** |
| 95% CI | [1.63, 2.45] | [2.28, 3.77] |
| fraction positive | 78% | 69.5% |
| binary output flips | 7 / 200 | 15 / 200 |

The effect is a **preference shift, not an attack-success effect**. §7 G3 anticipates exactly
this — "there may be few binary flips but a measurable probability shift; that supports a
different scope of conclusion" — so the claim is scoped accordingly. Absolute ASR is low
(4→6% Llama, 6→11% Qwen), which §11 requires be reported as absolute differences.

**Cue-family heterogeneity — the effect is not uniform, and this is the most interesting
behavioural finding so far.** Mean Δ margin by family:

| Cue family (P cue → S cue) | Llama | Qwen |
| --- | --- | --- |
| attribution_note ("from the file" → "from the user") | **+5.43** | **+8.45** |
| label_prefix (DOCUMENT NOTE → USER UPDATE) | +3.03 | +4.72 |
| states ("The document states" → "The user states") | +3.01 | +4.30 |
| written_by ("by the archive" → "by the user") | +1.79 | +4.29 |
| speaker_unmatched ("the archive" → "the account holder") | +1.69 | **−2.76** |
| source_tag ("Source: document." → "Source: user.") | **−1.39** | −0.56 |

`source_tag` runs *backwards* on Llama with a CI excluding zero — "Source: user." makes it
**less** likely to obey than "Source: document." And `speaker_unmatched` is positive on Llama
but significantly negative on Qwen. So the aggregate is positive on both models while the
per-family structure is model-specific. §13's instruction not to pool is load-bearing here.

**Cross-model observation worth flagging, not yet a claim.** Qwen is the more susceptible
model (+3.00 vs +2.04, 15 vs 7 flips) and is also the model with **no role-level source
signal** — its tool results render inside a `user` block, distinguished only by a plain-text
`<tool_response>` wrapper, whereas Llama has a distinct `ipython` role header. That ordering is
what the role-confusion hypothesis predicts. With n=2 models it is suggestive only.

### G4 — instrumentation is trustworthy

| Check | Result |
| --- | --- |
| run-to-run noise floor (margin) | **0.0** — forwards are bitwise reproducible |
| no-op hooks on all 32 blocks | **0.0** deviation from baseline |
| self-patch (own activation) | **0.0** |
| zero-strength steering (α=0) | **0.0** |
| positive control (full final decision state from another run) | changed top token in **12/12**, mean \|effect\| 11.4 nats |

§8 is explicit that the positive control only proves the plumbing can change the answer and is
**not** evidence of a localized authority mechanism.

### Measurement precision — a floor that would have produced false nulls

G4's cross-donor patch effects landed on exact multiples of **0.125**. That is the bf16 logit
resolution at magnitude ~16 (`2^(floor(log2 16) − 7)`), confirmed numerically. So in bf16 the
margin cannot resolve a patch effect below ~0.125 nats, and a small real effect would be
reported as zero.

The causal sweep therefore runs in **fp32**, which moves the floor to ~1e-6 (measured 9.5e-7)
at ~3× the cost (167 ms vs 53 ms per forward, 34 GB peak — fits GPU 1). Baseline margins are
re-measured in fp32 rather than reused from the bf16 behavioural run, since precision changes
the activations themselves.

## Done

**Infrastructure**: project on rnd5 (code) + rnd5 dataFAIR (data) via `data/` symlink;
verified venv (torch 2.5.1+cu124, transformers 5.14.1, CUDA on GPU 1); `env.sh` encoding the
host's TLS/BLAS/cache gotchas; git repo, local commits only, nothing pushed.

**Access verified, not assumed**: Llama-3.1-8B-Instruct gated access granted, revision
`0e9e39f2…`; Qwen2.5-7B-Instruct revision `a09a3545…`; `claude-opus-5` and `claude-sonnet-5`
both live on the key; upstream repo pinned at `ec333c40`.

**Corpus**: 920 base scenarios → 5,520 rendered conditions, exactly the §5 partition plan
(pilot 200 / discovery 240 / validation 120 / heldout-wd 120 / heldout-cue 120 /
heldout-task 120). Plus 1,000 neutral probe snippets split 600/200/200 at snippet level.
0 oracle failures, 0 warnings, no cross-scenario document sharing, no held-out cue leakage,
probe corpus provably disjoint from the attack corpus.

**Code**: `scripts/common/` — `scenarios.py` (generator), `oracle.py` (independent solvers
that parse the *document text*, a genuinely different code path), `render.py` (per-model
scaffold, span alignment, payload-safety assertions), `model_io.py`, `patching.py`.
`scripts/` — `g0_setup.py`, `build_dataset.py`, `eval_behavior.py`, `g4_instrumentation.py`,
`g5_localize.py`, `probe_source.py`.

## Open issues and honest limits

- **G1 is model-reviewed, not human-reviewed.** The user chose Opus triage over human review;
  the protocol itself says model agreement is "useful triage, not independent proof of
  correctness". The *structural* audit (oracle agreement, split hygiene, span alignment,
  payload safety) is fully automated and passes; it is the semantic review that is delegated.
- **G2 partially fails on Qwen**: U = 91.5% against a ≥95% gate. Llama is the primary model;
  Qwen's U shortfall is reported rather than papered over, and it caps what a Qwen utility
  claim can assert.
- **Both discovery task families are now lookup-shaped** after `table_select` was replaced
  (it failed G2 competence outright). Task generality rests on the `two_hop` transfer split,
  not on the discovery pair.
- **Low absolute ASR** (6% on Llama) means a defense evaluated on output flips alone would
  have very little signal. The margin is the informative quantity here, and §11 warns to
  report absolute differences when baseline ASR is low.
- **GPU tenancy is not guaranteed.** GPU 0 carries another user's 72 GB job throughout.
- Nothing pushed anywhere; no external side effects beyond HF downloads and Anthropic API.

## Decisions on record

- **G1 review**: Opus triage rather than human review, proceeding on a clean pass — user's
  explicit choice, recorded as model-reviewed.
- **Autonomy**: run through gates, stop only at decision gates or on failure; fix what is
  fixable — user's explicit choice.
- **Storage**: rnd5 for code and data, deviating from CLAUDE.md's rnd1 preference because
  rnd1 (96%) and rnd2 (100%) are both past the file's own 90% threshold.
- **fp32 for causal work, bf16 for behavioural work** — justified by the measured bf16 margin
  floor above; recorded per run, never silent.
- Protocol deviations (task family, cue design, scaffold, condition U wording, parser casing,
  label pool) are each justified by a measurement in `notes/05-design-decisions.md`.
