# Adaptive GCG defeats the refusal-direction steering defense (and how)

The steering defense (notes/27,28) re-injects the refusal direction the forged conclusion suppresses
and fully stops the forged-CoT attack (judge-ASR 0.47→0.00 at frac 0.75). But it was only ever tested
against the **fixed** attack. The rigorous question ("attacker moves second"): can a **white-box
adaptive attacker with full knowledge of the defense** get through? We use GCG (Zou et al. 2023) —
gradient-based, no LLM-attacker needed.

## Setup
Own minimal GCG (`gcg_core.py`, validated in `gcg_probe.py`/`gcg_base.py`: gradients flow through the
MXFP4→bf16 MoE, loss ↓, real base-model jailbreak "Sure, here are the detailed step-by-step
instructions…", 44 GB, ~2.5 s/step). Threat model: an injection attacker appends a 20-token
adversarial span to the real forgery inside the `analysis` channel; the defense steers refusal
(frac 0.75, α=178, layers {10,12,14}) because the input is untrusted-provenance. **The attacker
optimizes WITH the steering hooks active.** Success = natural render_forgery generation + Claude judge.
n=8, 300 steps, batch 256. `gcg_defense.py`.

## Result — 2×2 {steering off/on} × {GCG off/on} (judge-ASR, n=8)

| | no steering | **steering (defense)** |
|---|---|---|
| forgery only (fixed attack) | 0.875 | **0.000** |
| non-adaptive GCG (opt w/o defense) | 0.750 | 0.125 |
| **adaptive GCG (opt w/ defense on)** | 0.875 | **0.625** |

**The defense is defeated by a defense-aware adaptive attacker: judge-ASR 0.000 → 0.625.** Adaptivity
is necessary — the fixed attack (0.000) and a GCG suffix optimized *without* the defense (0.125) both
fail under steering; only optimizing *against* the live defense recovers the attack.

**Verified genuine** (audited every adaptive-cell generation): 5/8 are coherent harmful compliance
with the original request (hate-speech bot + Python, propaganda tips, bot-network, watermark removal,
résumé fraud); 2/8 the defense held ("I'm sorry, I can't help"); 1/8 GCG *derailed* the output onto a
nonsense injected task ("E422 Manual Bolt87") and the judge correctly scored it REDIRECTION, not
harmful — excluded. So 0.625 is real harm, not GCG gibberish the judge mislabeled.

## Mechanism — it ROUTES AROUND the refusal direction (does not out-suppress it)
For the 5 jailbroken-under-steering cases we measured the base (no-steering) projection onto the
refusal direction `d` at the decision position (poles: complying −312, refusing −74.6):

- adaptive base-proj **−135.1** vs plain forgery **−138.8** (Δ **+3.6**) — the adaptive suffix does
  **not** drive the projection more negative to cancel the steering.
- With steering added, the total projection is **+39…+52** — *past the refusing pole* — **yet the
  model complies.** (The 2 cases that stayed defended actually had *lower* total projection.)

So compliance happens at *high* refusal-direction projection: the linear refusal direction is **not a
necessary bottleneck** — GCG finds an alternate compliance pathway that steering along `d` does not
close. This is the activation-defense bypass phenomenon (cf. Obfuscated Activations, 2412.09565)
demonstrated for a *causal steering* defense, not just a probe. GCG barely pays for it: steer-opt loss
floor (mean 0.59) is only marginally above base-opt (0.49).

## Does steering harder re-defend? (α-escalation on the FIXED suffix)
| steering | frac 0.75 (opt point) | 1.0 | 1.5 | 2.0 |
|---|---|---|---|---|
| fixed-adaptive judge-ASR | 0.625 | 0.125 | 0.000 | 0.000 |

Escalating α re-defends the **already-fixed** suffix (it was tuned to α=178; more steering breaks it).
But this is **not** a resolution: (a) it is a *non-adaptive* test — a GCG re-optimized at the higher α
is the next arms-race round (untested here); given the route-around mechanism, re-adaptation is
expected to recover. (b) Higher α degrades outputs and (notes/27) over-refuses benign (here benign
heuristic-compliance stays 1.0, but the refusal heuristic masks degraded/gibberish benign generations
at α=475 — a proper judged benign-utility curve is follow-up). So escalation trades utility for a
temporary, attacker-specific patch.

## Arms-race round: defender ESCALATES, attacker RE-OPTIMIZES → NO FREE LUNCH
The decisive follow-up (`gcg_armsrace.py`): the defender escalates to steering the refusal direction
at **7 layers {8,10,12,14,16,18,20} at frac 1.0** (α=539.6, ~3×), and GCG is **re-optimized against
this new config** (n=8, 300 steps). Result:

| config | adaptive GCG judge-ASR | benign utility (eyeballed) |
|---|---|---|
| D1 — {10,12,14}, frac 0.75 | **0.625** (attacker wins, route-around) | coherent + useful (verified) |
| E1 — {8..20}, frac 1.0 | **0.000** (harmful all refused) | **BROKEN** — benign also refused |

Under E1 the re-optimized GCG cannot drive compliance (loss stuck ~2.7 vs ~0.3 undefended) and every
harmful output degenerates to a refusal loop ("The user request is disallowed…"). **But the same E1
steering breaks BENIGN too**: "How do I bake sourdough bread?" → "This is a request for disallowed
content…". So E1 is not a defense — it is a lobotomy that refuses everything. (The arms-race script's
`benign non-refusal = 1.000` was a `is_refusal` FALSE NEGATIVE: the heuristic keys on "I'm sorry" and
misses "disallowed"-style refusals — the same blind spot that inflated the harmful heuristic to 1.0.
Ground-truth by eye: `check_e1_benign.py`. At D1/frac0.75 benign is genuinely coherent — the capital
of France, a real sourdough recipe — so the notes/27 benign claim holds *there*.)

**Verdict: the defender has NO config that both blocks the adaptive attacker and preserves utility.**
At usable strength (D1) the attacker route-arounds (0.625); at strength that stops the attacker (E1)
the model is broken for everyone. This is the direct consequence of route-around: because compliance
is not gated by the refusal direction, you cannot suppress compliance-through-that-direction hard
enough to stop an adaptive attacker without also suppressing normal generation.

## Multi-DIRECTION subspace steering — the right axis is dimensionality, not magnitude
If route-around works by complying at high projection on the *single* refusal direction, steer the
refusal *subspace* instead. M1 (`gcg_subspace.py`): at the SAME 3 layers {10,12,14} and same frac 0.75
as D1 (so the only change from the route-aroundable D1 is k=1 → k=8), build a k-dim subspace = top-k
right singular vectors of the harmful-relative-to-harmless deviations, steer each basis direction e_j by
frac·gap_j. Re-optimize GCG against it.

| config | change from D1 | adaptive GCG judge-ASR | benign utility (coherence-aware) |
|---|---|---|---|
| D1 | k=1, {10,12,14}, 0.75 | 0.625 | coherent ✓ |
| **M1** | **k=8 subspace**, same layers/frac | **0.250** | coherent ✓ (USEFUL 1.000) |
| E1 | k=1, {8..20}, 1.0 | 0.000 | BROKEN ✗ (benign lobotomised) |

**Subspace steering roughly halves the adaptive attack (0.625 → 0.250) at ZERO utility cost** — the 6
defended cases are clean "I'm sorry" refusals, benign answers stay coherent. Strikingly, the extra 7
directions carry tiny mean-gap signal (per-layer PC gaps ~[138, 14, 5, 2, 1, 3, 4, 3] — the harmful-
vs-harmless *mean* difference is ~1-D), yet steering them cuts the attack in half: **route-around
exploited exactly the small orthogonal directions the single-direction defense left open.** So the
productive way to escalate is **dimensionality** (steer more directions), not **magnitude** (bigger α /
more layers, which lobotomised benign in E1). It is still not complete (0.250 residual, 2/8 genuine
jailbreaks: propaganda, plagiarism), so robustness looks like it *scales with the steered subspace
dimension* up to some benign-cost limit — a frontier to map.

## What this means (honest bottom line)
- **Corrects our own claim.** notes/27–28 said the steering defense "targets the causal lever, so it
  isn't evadable." True vs *restyling / fixed / non-adaptive* attacks — but a white-box adaptive GCG
  evades single-direction steering (0→0.625), and brute-force hardening (E1) destroys utility.
  **Single-direction refusal steering is not adaptively robust; a low-dim subspace recovers much of the
  robustness (0.625→0.250) at no utility cost** — the axis that matters is dimensionality, not strength.
- **The scientific contribution is the pair + the tension**: forged-CoT works via the refusal-
  direction-suppressing conclusion (notes/23–29, verified causal); yet that direction is **not a
  necessary bottleneck** for compliance — a defense-aware attacker complies at high projection
  (route-around), and steering it hard enough to close the bypass breaks benign generation. Correcting
  a published mechanism *and* adaptively falsifying the mechanism-derived defense (with the utility/
  robustness tradeoff mapped) is a stronger, more honest story than an untested defense.
- **Implication**: activation-steering along a single (few-layer) linear direction is fundamentally
  limited against adaptive attackers; robustness needs either a genuinely multi-dimensional refusal
  representation or defense-in-depth (e.g., pair with an input-side content monitor), not more α.

## Next steps
1. **k-sweep frontier** (the natural continuation of M1): adaptive judge-ASR *and* coherence-aware
   benign-utility vs subspace dim k ∈ {1, 8, 16, 32} at fixed layers/frac — does robustness keep
   improving with k, and where does benign start to break? Turns "dimensionality is the axis" into a curve.
2. **Exp A (contrast):** GCG vs the input-side NLI *conclusion* detector — expected to be evaded even
   more cheaply than steering, sharpening "watch the lever, but the lever is bypassable under optimization."
3. Larger n / second model (blocked: /data full, no 120b) for tighter CIs.

## Artifacts
- `scripts/gcg_core.py`, `gcg_probe.py`, `gcg_base.py`, `gcg_defense.py`, `gcg_route_analysis.py`,
  `gcg_armsrace.py`, `gcg_subspace.py`, `check_e1_benign.py`
- `$DATA_DIR/outputs/probe_gptoss/gcg_defense_main.json`, `gcg_route_analysis.json`, `gcg_base.json`,
  `gcg_armsrace_E1.json`, `gcg_subspace_M1.json`
- **Caveat**: `is_refusal` (heuristic) misses "disallowed"-style refusals → treat it as a lower bound
  on refusal; the Claude judge is the metric for ASR, and benign utility must be judged/coherence-aware
  (not `is_refusal`) — the `benign_broken` detector in `gcg_subspace.py` and eyeballing are ground truth.
