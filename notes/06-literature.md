# Literature scan — what it changes about this project

Compiled 2026-09-15. Only the findings that alter our design or our claims. IDs marked ⚠ were
read by a research agent rather than fetched directly and should be verified before citing.

## 1. The contribution is substantially taken. Reposition.

**V-Steer** — Zeng, Lee, Zhao, Hockenmaier, **COLM 2026**, [arXiv:2607.26228](https://arxiv.org/abs/2607.26228),
code `github.com/cindy2000sh/v-steer`.

It is training-free, inference-time, **span-restricted**, and **gated on real message
provenance rather than a detector** — the exact cell we aimed at. On **Llama-3.1-8B-Instruct
and Qwen2.5-7B-Instruct**, our exact two models. Spans come from role markers, "no extraction
needed". It localizes heads by direct logit attribution, then multiplicatively edits cached
**V tensors** (γ₊=2.5 privileged, γ₋=0.75 non-privileged), lifted to KV-head level under GQA.
It already reports three of our four selectivity axes: MMLU 66.1→57.6 (**−8.5**), IFEval
−2.3, BBH −1.9, IHEval-aligned −2.0, and a forged-authority experiment (authority bias
42.1%→14.9% on Llama-3.1-8B).

**What is genuinely left, and it is narrow:**
- No causal localization with controls. Its own limitations say it is "unclear if identified
  heads form stable and causally meaningful role-priority circuits."
- **Its tool-output result is its weakest**: Llama-3.1-8B IHEval **intrinsic tool use
  0.0% → 0.0%** — the instruction-as-data case does not move at all. That is our condition Q.
- No factual-uptake (F) or quote-as-data (Q) condition.
- A hand-set multiplicative heuristic, not a fitted directional correction.
- No adaptive-attack evaluation.
- −8.5 MMLU is a real cost a tighter intervention could beat.

Also close: **CachePrune** (ACL 2026, [2504.21228](https://arxiv.org/abs/2504.21228)) is
provenance-gated by construction ("we only score and prune with activations encoded from the
text span of input context"); LLaMA3-8B HotpotQA ASR 69.01→15.23%, clean F1 preserved. No
code. No quote-as-data or legitimate-revision evaluation.

**Claim to make instead of "a new defense":** causal validation and a selectivity standard for
an existing defense class. Defensible: (1) first causal localization of forged-authority
routing with matched random-direction, self-patch and random-donor controls; (2) first
four-axis selectivity evaluation of a span-gated representation-level IPI defense, including
the factual-uptake and quote-as-data axes nobody measures; (3) a fitted directional correction
compared head-to-head against V-Steer's multiplicative heuristic on the same models.

Note the idea is *publicly proposed but unexecuted*: the steering follow-up's own discussion
says "if a piece of text with abnormally high 'User-ness' exists inside a tool block, we could
reduce the 'User-ness' via steering." Novelty must live in execution and rigour, not the idea.

## 2. The premise is contested — and our own numbers agree with the sceptics

- **Rebuttal to the steering post**: "Role confusion: sounding like the cause is
  indistinguishable from being it" (Mogford, 29 Jun 2026). Probe-direction steering barely
  moved behaviour (CoTness 0.93→0.25 moved ASR 0.75→0.69). Patching moved behaviour but
  **was not specific**: role subspace vs *random* subspace, Fisher **p = 0.60**; full
  destyled transplant vs matched random perturbation, **p = 1.0**. The styled/destyled gap lies
  ~95% *outside* the probe's role subspace.
- **The steering post's payload violates our threat model.** It used
  `<|im_start|>user … <|im_end|>` — model-native special-token spellings inside untrusted
  text, which protocol §4 prohibits and which we measured *do* parse as real special ids
  (notes/04). Its baseline ASR was ~0%; the attack had to be created by steering.
- **The source paper's own numbers**: plain fake-user tool injections at **0–2% ASR**
  (26% on gpt-oss-20b) versus **56–70%** for style-based CoT forgery. The role *tag*
  contributes little next to stylistic mimicry (destyling: 61%→10%).

**Our measured G2/G3 result is consistent with this**: a reproducible margin shift
(+2.04 nats, CI [1.63, 2.45]) with almost no output change (7/200 flips, ASR 4.0%→6.1%). We
are not seeing a strong attack; we are seeing a preference nudge.

**Design response, implemented**: added **condition B** — the bare command with no attribution
at all — giving the ladder **N → B → P → S**. This separates "does any injected instruction
move the model" from "does attributing it to the document/user change anything". If B ≈ P ≈ S,
the authority cue is not the operative variable and the causal question should be re-scoped to
in-context instruction routing generally, which is both better supported and still open
(V-Steer's intrinsic tool use is 0.0%→0.0%).

## 3. Steering method: use difference-of-means, and prefer ablation over addition

Two findings that would have cost us a whole experiment cycle.

**Probe weights are the wrong direction to steer.** Marks & Tegmark, "The Geometry of Truth",
**COLM 2024**, [2310.06824](https://arxiv.org/abs/2310.06824): Theorem F.1 gives
`θ_LR ∝ Σ⁻¹θ` — the logistic-regression direction *is* the whitened concept direction, so it
deliberately removes interference from correlated features and stops tracking the feature
direction. Empirically, with **near-identical classification accuracy**, normalized indirect
effects were LR .13/.19 versus **mass-mean .77/.90**. Belrose's diff-in-means result
([blog.eleuther.ai/diff-in-means/](https://blog.eleuther.ai/diff-in-means/), Theorem 3) shows
`δ/|δ|` is **worst-case optimal** precisely when the downstream reader is unidentified —
exactly our situation.

Corroborated on our task: ⚠[2608.28648] reports "**Separability is not causal control**… the
IID mass-mean direction gives the strongest linear separation… If linear separability alone
selected causal directions, this should be the best vector to add. **It is not.**" And
⚠[2510.01228] (NeurIPS 2025 workshop) found role-conflict steering vectors "**amplify
instruction following in a role-agnostic way**" — a generic obedience knob, which is precisely
the failure our U/F/Q conditions exist to catch.

**Consequence for §9/§11**: fit the probe for *measurement* only, as §9 already says. Derive
any intervention direction from **raw difference-of-means with magnitude preserved** (not
unit-normalized), and do not whiten.

**Prefer directional ablation/projection to activation addition.** Arditi et al.,
**NeurIPS 2024**, [2406.11717](https://arxiv.org/abs/2406.11717), App. G.3, on-distribution CE
loss for **Llama-3 8B**: baseline 0.195 / directional ablation 0.213 / **activation addition
0.441**. Addition costs ~2–3× the loss because it pushes benign activations off-distribution;
ablation moves harmful activations *toward* the benign manifold. Their method template also
applies directly: diff-of-means per layer **× per post-instruction token position**, selected
by minimizing bypass subject to `kl_score < 0.1` on benign validation and `l < 0.8L`. Their
chosen position for Llama-3 8B was the **fifth-from-last token, layer 12 of 32** — *not* the
last token, so sweep positions rather than assuming the decision position.

## 4. Mandatory controls we were missing

- **Matched-norm random donor.** nnsight's activation-patching guide reports a random donor of
  matched norm moving the logit difference "by up to +3.6 at layer 3, further than the real
  effect at layers 8–11". Our random-position control is *matched on span length and on being
  inside the untrusted span*; a **norm-matched random direction** control is a different and
  necessary one.
- **The subspace-interchange illusion.** Makelov, Lange, Nanda,
  [2311.17030](https://arxiv.org/abs/2311.17030): a subspace intervention can make the output
  behave as if a feature changed "by activating a dormant parallel pathway… instead of
  localizing a variable used by the model, subspace interventions **can create such a
  variable**." They find illusory directions "**even when the MLP weights are replaced by
  random matrices**." Two diagnostics to run: (1) compare the patch against patching only the
  `v_row` component with the causally disconnected part removed — if truly dormant it should
  have no effect; (2) compare the spread of projections onto `v_row` vs `v_null`. Their
  recommendation is to work **in activation bottlenecks, especially the residual stream** —
  which is what we patch, and for a stated reason: the residual stream at the last token is a
  full bottleneck, so no earlier component can activate the direction while skipping the patch.
- **Self-repair means ablation under-reports.** McGrath et al.,
  [2307.15771](https://arxiv.org/abs/2307.15771): at middle layers self-repair restores ~70%
  of an ablated component's logit reduction; "**more important network components are more
  likely to be robust to ablation.**" Rushing & Nanda, **ICML 2024**,
  [2402.15390](https://arxiv.org/abs/2402.15390), add the mechanism that matters for us:
  "possibly 30% of the self-repairing of direct effects can be attributed to just the effect
  of ablations on the LayerNorm normalization factor" — and their footnote says this "applies
  in the same way to RMSNorm… which is used in LLaMA."
- **Backup heads.** Wang et al. IOI, [2211.00593](https://arxiv.org/abs/2211.00593): knocking
  out *all* Name Mover Heads still left the circuit working (5% logit-difference drop). Their
  three criteria — faithfulness, **completeness** (`|F(C∖K) − F(M∖K)|` small for every subset
  K), minimality — are the right validation frame, and they honestly report that greedy search
  broke their own completeness.

## 5. Denoising vs noising are not mirrors, and the asymmetry tells us the circuit shape

Heimersheim & Nanda, [2404.15255](https://arxiv.org/abs/2404.15255), §2.4: for a **serial
(AND)** structure, noising finds all components while denoising finds only the last; for a
**redundant (OR/parallel)** structure, **denoising** finds all while noising finds only the
last. Forged-authority routing is very likely OR-shaped — many heads can carry "this reads
like a user instruction" — so **denoising (clean→corrupt, i.e. P→S) is the right primary
tool** and noising will under-report. We run both directions already; the *asymmetry* should
be reported as evidence about circuit shape rather than averaged away.

Also: "**If you patch component A in layer N, it has seen clean versions of every component in
layers 0 to N−1**" — denoising has a structural blind spot for upstream mediated components.

**Correction to our §13 premise**: I could not find a published statement that gap-normalized
recovery "explodes when the gap is near zero". The real, separately sourced hazards are
(a) dividing by a near-zero *corrupted-run probability* (Zhang & Nanda App. C — "the small
denominator… acts as a large multiplier", producing an "extremely pronounced peak" that was a
normalization artifact); (b) symmetric corruption giving ~50% "recovery" for free; (c) ">100%
recovered" from omitting negative components. Our choice to report **raw unnormalized margin
differences** as the principal score is right regardless — keep it.

**Zhang & Nanda, [2309.16042](https://arxiv.org/abs/2309.16042), is also where the
equal-token-length requirement actually comes from** ("STR replaces the key tokens by similar
ones with equal sequence length"), *not* from Heimersheim & Nanda, which contains no
position-alignment guidance. Cite accordingly.

## 6. GQA: our k/v grids are 8 groups, not 32 heads

Llama-3.1-8B: 32 layers, 32 query heads, **8 KV heads**, head_dim 128 → **4 query heads per KV
head**; query head `h` reads KV head `h // 4`. Qwen2.5-7B: 28 heads, **4 KV heads** (7 query
heads per group). `attn_weights` is computed *after* `repeat_kv`, so the pattern has 32 rows
while K and V physically exist only at 8-head granularity.

So `z`/`q`/`pattern` grids are 32×32; **`k`/`v` grids are 32×8 and each cell is a 4-query-head
group intervention.** Never write "head 17's key matters" — write "KV group 4 (query heads
16–19)". Protocol §8 already requires this distinction; this is its implementation. Notably,
**no paper or blog treats GQA as an activation-patching pitfall** — worth stating ourselves.

RoPE applies to **Q and K only**, before the KV cache is written. Patching `v`/`z` across
positions carries no positional signal; patching post-RoPE `k` from position *p* into *q ≠ p*
silently patches position along with content. Our design patches position-for-position in
equal-length prompts, so the distinction is moot — but it constrains any future cross-position
transplant.

## 7. A named mechanism by which our defense could break condition U

**GCAD / "Prompt–Activation Duality"**, [2605.10664](https://arxiv.org/abs/2605.10664):
"standard residual-stream persona-vector steering repeatedly injects the same perturbation
into **states that future tokens may attend to**. An intervention that appears effective in a
short response can therefore become a cumulative source of degradation in multi-turn dialogue."
They call it **KV-cache contamination**. On our exact models: 10-turn coherence drift
**−26.5 → −0.2** (Llama-3.1-8B) and **−18.6 → −1.9** (Qwen2.5-7B) once fixed.

Condition U inserts a genuine later user turn that reads a KV cache containing our steered
tool span. Design responses: intervene on attention output before the MLP, or edit V tensors
(as V-Steer does — V is never RoPE-rotated), rather than the residual stream; and **measure U
with the tool span in cache, not re-prefilled** — those are different experiments and we
should say which we ran.

## 8. The selectivity framing and its vocabulary already exist — adopt them

- **SecFid** — "Security–Fidelity Tradeoffs", **ICML 2026**,
  [2606.30783](https://arxiv.org/abs/2606.30783). 1,168 instances, 15 base models **including
  Llama-3.1-8B and Qwen2.5-7B**, 48 defense configs. Three-way outcome
  **Executed / Processed / Ignored**, with **Security = 1−Executed** and **Fidelity =
  1−Ignored**. *That fidelity metric is our condition Q.* Result: "no model or defense achieves
  both objectives"; best fidelity 96.5% at 47.8% security. **It evaluates no
  representation-level defense — that is our slot.**
- **"specificity"** is the established term, decomposed into general / control / **robustness**
  specificity — Goyal & Daumé, **EACL 2026**, ⚠[2602.06256]. Their warning: steering "largely
  maintains general and control specificity, [but] consistently fails to preserve robustness
  specificity."
- **SteeringSafety**, **ICML 2026**, ⚠[2509.13450], formalizes **Entanglement** and finds
  "conditional steering enables better effectiveness-entanglement tradeoffs… but does not
  eliminate entanglement." Its untested hypothesis (iii) is our thesis stated as an open
  question: "**Token-localized application (CAA at post-instruction tokens only) should produce
  narrower per-perspective effects than all-position application, plausibly reducing
  entanglement.**" Excellent positioning citation.
- **CAST** (ICLR 2025 Spotlight, [2409.05907](https://arxiv.org/abs/2409.05907)) gates on
  *when* (prompt-level condition), DSAS on *how much*, GAPS on *which dimensions* — **we gate
  on *where***. CAST is explicitly not span-restricted: once the condition fires, "the behavior
  vector is applied in every subsequent forward pass." Its only preservation metric is harmless
  refusal rate — no MMLU, no perplexity — which is an opening.

**The single most transferable methodological warning, from model editing:** ROME
([2202.05262](https://arxiv.org/abs/2202.05262)) — zsRE specificity "is not a sensitive measure
of model damage, since these prompts are sampled from a large space of possible facts, whereas
**bleedover is most likely to occur on related neighboring subjects**." Translation: **a random
MMLU battery will not detect our damage; the specificity set must be near-neighbour —
legitimate, instruction-bearing tool outputs.** That is exactly our U/F/Q conditions, and now
there is a citation for why they are necessary rather than decorative.

Circuit Breakers (**NeurIPS 2024**, [2406.04313](https://arxiv.org/abs/2406.04313)) makes the
same point by counterexample: MT-Bench 8.05→8.00 and OpenLLM 68.8→68.3 (flat), while benign
refusal on 500 WildChat requests went **2.2 → 6.2**. A representation-level defense held MMLU
and MT-Bench flat while nearly tripling over-refusal. **MMLU + MT-Bench alone does not
establish selectivity.**

## 9. Baselines we should actually run

Zero-training and runnable on an 8B today:
- **DefensiveTokens** (ACM AISec '25, [2507.07974](https://arxiv.org/abs/2507.07974)) — 5
  optimized token embeddings, weights untouched, and **pre-trained tokens are released for
  exactly Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct** (`Sizhe-Chen/DefensiveToken`).
  AlpacaEval2 essentially unchanged. **Our best zero-training baseline.**
- **Spotlighting / datamarking** (CAMLIS 2024, [2403.14720](https://arxiv.org/abs/2403.14720))
  — the canonical strawman. Skip base64 encoding at 8B; the paper calls it "very detrimental"
  on weak models.
- **Attention Tracker** (Findings NAACL 2025, [2411.00348](https://arxiv.org/abs/2411.00348))
  — training-free, AUROC 1.00 on Llama-3-8B, but reports **no FPR and no utility cost**.
  Producing those is itself a contribution. Also: **use its head scores to seed our head
  search** rather than sweeping blind.
- **V-Steer** — mandatory, code released.
- **Meta-SecAlign-8B** (`facebook/Meta-SecAlign-8B`) — a free training-time upper bound by
  model swap.
- **An instruction-hierarchy reminder prompt** — already in our §11, and mandatory: AxBench
  (ICML 2025) found "**for steering, prompting outperforms all existing methods.**"

## 10. Adaptive attacks — our §11 budget is too small to be credible

Nasr, Carlini, Sitawarin, Schulhoff et al. (Google DeepMind / OpenAI / Anthropic / ETH),
[2510.09023](https://arxiv.org/abs/2510.09023): "we bypass **12 recent defenses with attack
success rate above 90%** for most; importantly, the majority of defenses originally reported
near-zero attack success rates." **Circuit Breakers — the closest representation-level
precedent — went to 100%.** Median queries-to-break 13–46 against an 800-query budget. Their
utility finding cuts both ways and is worth quoting: "All of the model filtering defenses
substantially harm utility… the prompting-based defenses generally maintain utility well… but
they offer no robustness improvement beyond the undefended model."

Separately, Zhan, Fang, Panchal, Kang, **Findings NAACL 2025**,
[2503.00061](https://arxiv.org/abs/2503.00061), code `uiuc-kang-lab/AdaptiveAttackAgent`, runs
on **Llama-3.1-8B-Instruct**: all 8 IPI defenses exceed 50% ASR under adaptation; a fine-tuned
detector collapsed 61%→10%. Their honest gap — "**we do not assess the impact on normal
cases**" — is one we can fill.

**Response**: keep §11's 50 scenarios × 20 queries but report it as an **ASR-versus-query-budget
curve**, explicitly not a robustness claim, and say so.

## 11. Is the phenomenon present at 7–8B? Yes for injection generally; unproven for the cue

Confirmed susceptibility at our scale: LLaMA3-8B HotpotQA **69.0%** ASR (CachePrune);
Llama-3.1-8B SEP **69.3–95.0%** (⚠DataFilter); Llama-3.1-8B AlpacaFarm 68.3% / SEP 98.1%
(⚠Meta SecAlign); Qwen2.5-7B AgentDojo **25.8%** (Progent reproduction); Llama-3.1-8B
InjecAgent 9% → **87%±10** adaptive (Zhan et al.). Llama-3.1-8B is described as "the strongest
anti-hierarchy case", following system instructions in **0.10** of conflict trials.

So the models are, if anything, *too* susceptible to injection in general. What is **not**
established anywhere is that the *forged-authority cue specifically* adds measurable ASR over a
plain injected instruction at this scale — and no paper trains a role/provenance probe on
Llama-3.1-8B at all. That is simultaneously our opportunity and our main risk, and it is
exactly what condition B now measures.

## 12. Corrections to propagate

- CAA's first author is **Nina Panickssery**; **Wes Gurnee is not an author** (he is on Arditi
  et al.). ActAdd is retitled "Steering Language Models With Activation Engineering".
  Atticus Geiger is **not** an author on the subspace-illusion paper. The Hydra-effect paper
  does not use the phrase "counterfactual importance".
- RepE ([2310.01405](https://arxiv.org/abs/2310.01405)) and Nasr et al. are **preprints**, no
  venue in their arXiv comments. ARGUS lists **no venue** ("CVPR 2026" appears only on
  aggregators). "The Defense Trilemma" (⚠[2604.06436]) is **not** ICLR 2026 and scopes itself
  to input wrappers only, so it does not condemn this approach.
- Do **not** cite Heimersheim & Nanda for the equal-token-length requirement — it is not in
  there. Cite Zhang & Nanda.
- Licensing to resolve before any data freeze: **IHEval is CC BY-NC-ND** (NoDerivatives —
  conflicts with building a modified variant); the flow-separation repo
  `prompt_injections_so_back` has **no license at all**; Attention Tracker is CC BY-NC.
