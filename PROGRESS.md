# Progress — done, in flight, and blocked

Status board. Narrative reasoning goes in `LAB_NOTEBOOK.md`; this file is just state.
Gate definitions are protocol §7 (`notes/00-protocol-source.md`). **A gate is only marked
PASSED when its stated criterion was actually measured and met.**

Last updated: 2026-09-15.

## Gate status

| Gate | What it requires | Status |
| --- | --- | --- |
| G0 setup | One model loads with headroom; versions + rendered prompts reproducible; 100 forwards benchmarked | **running** |
| G1 data | Full structural/oracle/split audit, no unresolved critical label errors | not started |
| G2 behavior | ≥95% success separately on N, U, F with a frozen parser | not started |
| G3 contrast | ≥40 attack-responsive scenarios of 200; reproducible aggregate cue effect | not started |
| G4 instrumentation | No-op/self-patch agree within noise; positive control flips output | not started |
| G5 localization | Repeatable effect under >1 donor/control scheme | not started |
| G6 mechanism | Effects track instruction selection, survive held-out, alternatives tested | not started |
| G7 defense | Useful security/utility tradeoff, no clean-answer oracle | not started |
| G8 transfer | Qwen + held-out task/cue + external eval, each with own competence check | not started |

## Done

**Infrastructure**
- Project created: code `/rnd_ai_datasets5/projects/raka6003/prompt-injection-role-defense`,
  data `/rnd_ai_datasets5/dataFAIR/raka6003/prompt-injection-role-defense` via `data/` symlink.
  rnd5 chosen because rnd1 is 96% full and rnd2 100% — both past the CLAUDE.md 90% rule.
- venv built and **verified on GPU**: torch 2.5.1+cu124, transformers 5.14.1, sklearn 1.9.0,
  numpy 2.4.6, anthropic 0.120.2; `cuda_available True`, `device_count 1`, H100 PCIe,
  78.7/79.1 GiB free. Pins copied from a host-verified sibling project, not guessed.
- `env.sh` encodes the host's real gotchas (Zscaler TLS bundle, `UV_SYSTEM_CERTS`, BLAS caps,
  GPU pin, cache paths). Secrets stay in `~/.bashrc`; no second copy on disk.
- git initialized, first commit `27024b8`. Local only — nothing pushed, no remote configured.

**Access verified (not assumed)**
- `meta-llama/Llama-3.1-8B-Instruct` — gated access **granted**, weights downloaded,
  revision `0e9e39f249a16976918f6564b8830bc894c89659`.
- `Qwen/Qwen2.5-7B-Instruct` — already in the shared HF cache.
- Anthropic key resolves `claude-opus-5` and `claude-sonnet-5`, both models the protocol names.
- Upstream repo cloned and pinned at `ec333c40fd43fe991e1ebf66765051b6d7e35784`.

**Reading and analysis**
- `notes/00-protocol-source.md` — governing protocol (copied in).
- `notes/01-prior-work.md` — paper (v6, sourced from code where the paper is vague) + steering
  follow-up, with the gap stated precisely.
- `notes/03-upstream-repo.md` — what is reusable, what cannot run here, and the
  delimiter confound in the paper's headline Userness result.
- `notes/04-rendering-findings.md` — measured chat-template behavior and the Qwen boundary escape.

**Code written**
- `scripts/common/model_io.py` — bf16→4bit fallback loader, batch-1 forward with hidden
  states, greedy generation, teacher-forced answer log-prob for the §9 margin.
- `scripts/g0_setup.py` — the G0 gate harness.

## Findings so far

1. **The paper has no activation-level causal result.** No steering, patching, or ablation
   anywhere in it. The LessWrong follow-up supplies the only causal evidence.
2. **Suppression is already solved; selectivity is not.** Tool-ward steering drove ASR to ~0%
   in the follow-up — but from an oracle-derived vector, with zero utility measurement. That
   reframes our contribution from "can we stop it" to "can we stop it without breaking
   legitimate instruction following," which is what conditions F and Q exist to test.
3. **Qwen2.5 renders tool results inside a `user` block.** Protocol §8's warning confirmed by
   measurement: a genuine tool result and a genuine later user turn are two adjacent
   `<|im_start|>user` blocks differing only by a plain-text `<tool_response>` wrapper. Qwen
   therefore offers *no role-header-level* source signal, unlike Llama's expected distinct
   `ipython` role. A substantive asymmetry between our two models — report separately, do not pool.
4. **Untrusted text can forge a real chat turn on Qwen2.5.** `<|im_start|>`/`<|im_end|>` parse
   to real special IDs from plain text, so a document body can open its own user turn
   (measured: 6 `<|im_start|>` where a legitimate 4-message conversation has 4). This is
   delimiter injection, not role confusion, and is out of scope per §4 — which makes the §6
   assertion against it load-bearing rather than a formality.
5. **A live alternative explanation for the paper's headline Userness→ASR curve.** 12 of the
   211 released templates use gpt-oss's own Harmony delimiters, and the reported top-5
   highest-Userness templates are dominated by literal chat delimiters. On a model where those
   spellings parse as real special tokens, such a template forges a turn rather than merely
   sounding user-like. Not a demonstrated error in their work — we have not verified how their
   harness tokenized page text — but it is why we stratify cue families and exclude literal
   delimiters from the primary analysis.

## Decisions taken (and by whom)

- **G1 review**: Opus 5 semantic review used as the gate instead of human review, proceeding
  on a clean pass — *user's explicit choice*. Recorded honestly: the protocol itself says model
  agreement is "useful triage, not independent proof of correctness," so G1 will be reported as
  **model-reviewed, not human-reviewed**.
- **Autonomy**: run through gates, stop only at decision gates or on failure; fix what is
  fixable and keep going — *user's explicit choice*.
- **Storage**: rnd5 for both code and data, deviating from CLAUDE.md's rnd1 preference because
  rnd1/rnd2 are over the file's own capacity threshold.
- **Dependency pins**: copied from `llm-confidence-metacognition`, a host-verified working set,
  because protocol §2 forbids inventing pins before testing compatibility.
- **Upstream repo is reference-only**: it needs CUDA 12.8 / torch 2.9 / RAPIDS cuML and this
  driver caps at CUDA 12.2. Port from it; do not depend on it.

## Known limits of this run

- **G1 is model-reviewed, not human-reviewed** (above). Any label-quality claim inherits that.
- **Whether the phenomenon appears at 8B is unknown.** The paper's deep analysis used
  gpt-oss-20b, which its own authors note has "especially good linear separability for all
  roles." G3 may fail on a dense 8B model. The protocol has a defined failure response and it
  will be followed rather than forced.
- **GPU tenancy is not guaranteed.** GPU 0 carries another user's 72 GB job; GPU 1 has been
  claimed mid-session before on this box. Load path falls back bf16→4-bit, and 4-bit would
  invalidate cross-run activation comparisons, so it is recorded per run.
- Nothing is pushed anywhere. No external side effects beyond HF downloads and Anthropic API
  calls.
