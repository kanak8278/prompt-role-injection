# Prior work — what is established, and precisely where it stops

Compiled 2026-09-15 from arXiv:2603.12277 (**v6, 27 Jun 2026** — read v2 and v6; six versions
exist and v6 materially differs), the project page `role-confusion.github.io`, the released
repo, and the LessWrong follow-up. Every number here is sourced; gaps are marked.

## 1. The paper — "Prompt Injection as Role Confusion"

Charles Ye, Jasmine Cui, Dylan Hadfield-Menell (MIT). ICML 2026. cs.CL/cs.AI/cs.CR.

**Thesis.** LLMs infer role identity from *writing style*, not from the structural role tags
the serving stack uses for authority. Tags are "weak offsets to the dominant style vector."
Therefore tags are not an authentication mechanism, and prompt injection is role confusion.

### Probe method (the part we reproduce)

Paper is vague; the code pins it down. Probed tensor is `all_pre_mlp_hidden_states` —
residual stream after the attention add, **through `post_attention_layernorm`**, i.e. the
normalized MLP input, *not* the block output. Classifier is cuML
`LogisticRegression(penalty='l2', fit_intercept=True, max_iter=5000)`, multinomial over
subsets of {system, user, CoT, assistant, tool}, optional `StandardScaler`. λ by grid search
over 1e-4…1e3; gpt-oss-20b uses `C=5e-3`, no scaling.

Training text is **neutral pretraining text only** — C4 (`en`/validation) + Dolma3
(`dolma3_mix-150B-1025`, rev `3a8349c`), 25/75 mix, seed 123 — 250 base sequences × 1024
tokens, each wrapped in all 5 roles (~1,250 sequences, 1.28 M tokens per model). Identical
content across role classes, so the probe cannot use semantics. **Content tokens only, role
tag tokens excluded.** Split is 90/10 **at the sequence level** (no token leakage).

This is a near-exact match to protocol §9's separate neutral corpus with body-token-only
fitting — our design was arrived at independently and agrees. Two divergences to decide
deliberately: they probe the normalized MLP input, we plan block-level residual; and they use
1024-token sequences, we cap at 1024 with 512 typical.

**CoTness/Userness are probe posterior probabilities** — `CoTness(t) := P(CoT | h_t)` — mean-
aggregated over the tokens of the span of interest, at a single mid layer. Because it is a
multinomial softmax, **the value depends on which role subset the probe was trained on**;
eight subsets exist for gpt-oss-20b and the paper never says which produced which figure.
Any "Userness" we report must state its role subset.

### Models

Probed (4): `openai/gpt-oss-20b` (24 layers), `openai/gpt-oss-120b` (36),
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (52), `Qwen/Qwen3-30B-A3B-Thinking-2507` (48).
Probe layer grid is every 2nd layer if <30 layers else every 4th; mid-layer for gpt-oss-20b is
**layer 12 (inferred from the grid — never stated)**. Deep single-model analysis is gpt-oss-20b
only. Behavioral-only: o4-mini, gpt-5-nano/mini/5; cross-family GLM-4.6 (92% ASR),
Kimi-K2 (84%), Minimax-M2 (74%). Auxiliary/judge model is Gemini-2.5-Pro.

### Headline results

| Experiment | Result |
| --- | --- |
| CoT-forgery chat jailbreak (StrongREJECT, N=313) | ~0–4% baseline → **60% ASR** avg; gpt-oss family and o4-mini >80%; GPT-5 series 17–52% |
| Destyling the forgery | **61% → 10% ASR**, avg −51 pp, consistent across models |
| Agent exfiltration (100 Wikipedia pages × 2 variants) | standard injection 0–2% (gpt-oss-20b 26%) → **CoT forgery 56–70%** |
| Absurd vs. plausible justification | 60% vs 63% — **logic is not what carries the attack** |
| Role scores, 313 attacks, gpt-oss-20b | styled forgery **79.1% CoTness** (genuine CoT 67.7%); destyled 29.1% CoTness / 62.9% Userness |
| Dose-response, 626 attempts | lowest CoTness quantile 9% ASR → highest **90%** |
| 212 templates × 1,000 trials, fixed command | lowest Userness quantile 2% → highest **70%**; Userness β=6.01 (p<.001) surviving control for declared role |
| User-style text force-wrapped in `<tool>` | still **87.6% Userness / 9.3% Toolness** (gpt-oss-20b) |
| Assistant-style text force-wrapped in `<tool>` | still **85.1% Assistantness** |
| v6 granular destyling | syntactic alone 61%→26%; a single bigram ("The user") −19 pp |
| v6 IH-failure prediction (N=160) | Systemness/Userness ratio AUC **.74** vs .52 shuffled control |

### What the paper does *not* do — this is the load-bearing fact for us

**There is no activation-level causal intervention anywhere in the paper.** No steering, no
patching, no directional ablation, no head ablation, no circuit analysis. Probes are read-out
instruments only. All interventions are **input-space**: tag swaps, destyling, template
framing, position, logic ablation.

So the role-confusion → ASR link itself is **observational** — an association between a
measured latent and an outcome, within manipulated input conditions. The paper's language
drifts here and v6 walks it back: v2 said Figure 9 "confirms the causal pathway", v6 says it
"supports the pathway", and "monotonically" became "near-monotonically" in three places. Cite
the causal claim carefully.

The unexcluded alternative: style may drive ASR through a channel *other* than the probe-
measured direction, with the probe reading a correlate. The authors' answer is convergent
validity plus downstream prediction, which they state openly.

### The paper proposes no defense, and argues one class of defense is harmful

It is a diagnosis paper. Destyling is an *attack ablation*, not deployable — a defender cannot
rewrite the attacker's payload without also destroying genuine CoT.

§7.1 argues that memorization-based defense is **actively harmful**: teaching a model to
distrust reasoning-like text forces it to distrust its own real CoT, which is brittle to
iteration, costly in capacity, and destroys CoT faithfulness. Generalized: an anti-exfiltration
defense keyed on "dangerous upload patterns" must also suspect legitimate user commands, which
"shifts the locus of control from human authority to an LLM judgment of what's allowed."

Its prescription is a *requirement*, not a mechanism:

> "Robust defense requires boundaries that survive into representation."
> "Our findings provide a theory for what defenses must solve: role perception is governed by
> attacker-controllable features."

Stated open questions include, verbatim: **"defense evaluation: role probes can test whether
interventions reshape geometry or merely add patterns"**; **"robust boundaries: how can model
design and training achieve clean latent separation?"**; and **"detection: could discrepancies
between the intended role and the probe-measured role flag injection attempts before
generation?"**

### Released artifacts — plan to reproduce, not to download

Repo `role-confusion/prompt-injection-as-role-confusion`, branch `master`, 127 entries, MIT-ish
license. **Not pip-installable** (no `pyproject.toml`/`setup.py`/`requirements.txt`; setup is
an editable `setup_python.sh`). Probes are **GPU-only via RAPIDS cuML + CuPy**, and it targets
CUDA 12.8 / torch 2.9 — we are on driver-capped CUDA 12.2 / torch 2.5.1, so **the repo will
not run here as-is**. Treat it as a reference implementation to read, not a dependency.

`.gitignore` excludes `*.pkl *.csv *.pt *.parquet *.png …`, so **no probe weights, no
activations, no datasets, no generated forgeries are released**, and notebook outputs are
stripped. We must train our own probes regardless. Most useful files to read:
`demo/role-probe-demo.ipynb` (self-contained, 512-token, 150-seq version),
`utils/probes.py`, `utils/loader.py`, `utils/role_assignments.py` (89 KB token→role labeler),
`utils/role_templates.py`, `experiments/role-analysis/02-train-role-probes.ipynb`.

### Discrepancies found in the released material — do not silently inherit

- Template CSV has **211 rows**, paper says 212.
- Released chat-eval notebook samples **N=200**, paper reports 313 / 626.
- README links two role-confusion notebooks by the wrong filenames.
- Appendix E says 82% where main-text Figure 7 says 83% for no-tags CoTness.
- No numeric held-out probe accuracy is published anywhere (computed by code, gitignored).

### Selection caveat from the project page, worth heeding

> "Every LLM we tested had strong linear separation between user and assistant, but `think` is
> less common; **gpt-oss-20b has especially good linear separability for all roles**."

The deep single-model analysis therefore ran on the most favourable model. Our dense 7–8 B
targets may separate worse. A weaker probe on Llama/Qwen is an expected outcome, not a bug —
protocol §9 already says probe strength is neither necessary nor sufficient for patching.

Also undocumented-but-reported there: prepending "Great job!" raises injection success, as does
swearing if earlier genuine user text swore. And an early *negative* result on subconscious
steering (cockroach text did not reduce agent purchase rate).

## 2. The follow-up — "Steering role confusion" (LessWrong)

The only causal work in this line, and the direct predecessor of this project.

**Method.** Difference-of-means over **command tokens** at **layer 11** of gpt-oss-20b, between
the same text labelled user vs. tool → a "toolness→userness" vector, rescaled to the residual
norm, added with coefficient α at layer 11. Effect verified at layer 14 with a multinomial
logistic probe.

**Results.** Steering user-ward: ASR ~0% → **~80%**, monotone in α (n=50/condition). Steering
tool-ward: ASR → **~0%**. Random-vector control n=20.

**What failed, per the author.** Probe-derived vectors (optimized for prediction, not for
behaviour change) produced no significant ASR change; neither did difference-of-means between
styled and de-styled CoT text. **Style-derived CoTness could not be extracted at all** — they
flag it as open.

**Limits.** Role *tags* only, one model, n=50, no code released, and — decisively — **no utility
measurement of any kind**.

## 3. The gap, stated precisely

The tool-ward result means suppression is already demonstrated. Two things still stand between
it and a defense, and they are what this project is for.

1. **Oracle dependence.** That vector is derived from a contrast requiring knowledge of which
   span is the attack (same command labelled user vs. tool). Protocol §11 forbids exactly this
   in a deployed intervention: no attack label, no oracle attack span, no clean donor copy of
   the current example. As it stands the steering result is a causal *diagnostic*.
2. **Unmeasured selectivity.** With no utility numbers, "tool-ward steering suppresses the
   attack" is indistinguishable from "tool-ward steering makes the model less instruction-
   responsive in general." The protocol's own tool-suppression baseline (§11) would reproduce
   the former. The six-condition design settles it: a merely-deafening intervention passes `S`
   and **fails `F` (legitimate uptake of a changed tool fact) and `Q` (quoting an instruction as
   data)**. Any ASR reduction reported without `F`/`Q` intact is uninterpretable.

Note the paper independently asks for precisely this instrument — "role probes can test whether
interventions reshape geometry or merely add patterns" — so probing the intervened model is
part of the evaluation, not a side quest.

**Transfer risk to keep in view.** The follow-up worked on tags, on gpt-oss-20b (the most
linearly separable model tested), targeting CoT forgery. We target forged *user authority* on
dense Llama-3.1-8B / Qwen2.5-7B. This is not a like-for-like continuation, so **layer 11 must
not be assumed to carry over** — search for the functional stage, not the layer index, exactly
as protocol §10 requires ("comparing function rather than head numbers").
