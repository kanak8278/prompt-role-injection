# G5 — causal localization of forged-authority routing (Llama-3.1-8B, fp32)

Run: 40 aligned P/S pairs from the **discovery** split, 8 excluded as unaligned (all the
deliberately length-mismatched cue family). fp32, measured margin resolution **9.5e-7**.
Patched tensor: the **post-block residual stream** (`model.model.layers[i]` output). Both
directions, three semantic locations, all 32 blocks. 40 pairs × (2 captures + 32×3×2 patched
forwards + controls) ≈ 9,000 forwards, 18 min wall clock.

Artifacts: `outputs/g5/g5_Llama-3.1-8B-Instruct_fp32.json`, `rows_*.jsonl`, `baselines_*.jsonl`.

## Baseline contrast on these pairs

| | |
| --- | --- |
| mean Δ margin (S − P) | **+2.380** |
| median | +1.453 |
| range | −2.958 … +17.163 |
| fraction positive | 0.80 |
| mean excluding the 2 largest | +1.611 |

The mean is inflated by two scenarios at +16.8 and +17.2, where the forged cue pushed the
margin across zero into attacker-favouring territory. **All recovery fractions below are
computed from per-pair medians**, not from ratios of means, so they are not driven by those
two pairs. §13's instruction to report full distributions rather than averages is doing real
work here.

## The result: a three-stage relay, with sharp complementary handoffs

Median fraction of the per-pair P→S margin effect recovered by patching P's activations into
S at one (layer, location), restricted to pairs with |Δ| > 0.5:

| layer | cue span | command span | decision position |
| ---: | ---: | ---: | ---: |
| 0 | **1.01** | −0.02 | 0.00 |
| 2 | **1.01** | 0.01 | 0.00 |
| 4 | 0.87 | 0.12 | 0.00 |
| 6 | 0.71 | 0.13 | 0.00 |
| 8 | 0.55 | 0.35 | 0.00 |
| 10 | 0.36 | 0.59 | 0.01 |
| **12** | **0.09** | **0.86** | 0.01 |
| 14 | 0.04 | 0.78 | 0.04 |
| 16 | 0.02 | 0.67 | 0.22 |
| 18 | 0.01 | 0.47 | 0.42 |
| 20 | 0.00 | 0.50 | 0.43 |
| 22 | 0.00 | 0.51 | 0.43 |
| 24 | 0.01 | 0.30 | 0.55 |
| 26 | 0.00 | 0.16 | 0.73 |
| 28 | 0.00 | 0.08 | 0.82 |
| 30 | 0.00 | 0.02 | 0.97 |
| 31 | 0.00 | 0.00 | **1.00** |

Read down the columns and the structure is unambiguous:

1. **Blocks 0–11 — the authority signal is at the cue tokens.** Cue patching recovers
   essentially all of the effect early and decays steadily (1.01 → 0.36 by block 10).
2. **Block ~12 is the handoff.** Cue recovery collapses to **0.09** in the same block where
   command recovery **peaks at 0.86**. By block 12 the authority information has left the cue
   positions and is carried by the command positions.
3. **Blocks 12–24 — it is at the command tokens**, decaying 0.86 → 0.30.
4. **Blocks 16–31 — it transfers to the decision position**, rising 0.22 → 1.00.

The handoffs approximately conserve the effect: at block 12, cue + command ≈ 0.95; at block
18, command + decision ≈ 0.89; at block 28, command + decision ≈ 0.90. That conservation is
what makes this a relay rather than three independent measurements.

**This is what protocol §10 names as the stronger target**, verbatim: "a causally supported
path from the authority cue to command processing and then answer selection, while factual
processing remains usable." The first half of that sentence is now measured. The second half
(factual processing intact) is what G7's F and Q conditions test.

**And it is specifically not the known null pattern.** A whole-span residual sweep that decays
monotonically from 1.0 to 0.0 is the null result, not a finding — patch the embeddings and you
reproduce the clean run by construction; patch the last block at a non-final position and no
path to the logits remains. Here there are three locations with distinct, non-monotone,
complementary profiles: one decaying, one peaking mid-stack, one rising. The mid-stack peak in
the command column at block 12 cannot be produced by that artifact.

## Controls

| control | n | mean recovery | 95% CI | mean donor ΔL2 |
| --- | ---: | ---: | --- | ---: |
| random matched positions, cue-length, L0 | 81 | **+0.0003** | [−0.0023, 0.0030] | 0.034 |
| random matched positions, cmd-length, L0 | 81 | −0.0002 | [−0.0028, 0.0021] | 0.028 |
| random matched positions, cue-length, L10 | 81 | **+0.0348** | [−0.0055, 0.0761] | 2.129 |
| random matched positions, cmd-length, L10 | 81 | +0.0219 | [−0.0150, 0.0686] | 1.901 |
| random matched positions, cue-length, L21 | 81 | +0.0068 | [−0.0152, 0.0295] | 7.301 |
| random matched positions, cmd-length, L21 | 81 | +0.0125 | [0.0020, 0.0242] | 6.339 |
| random matched positions, L31 | 81 | +0.0000 | [0.0, 0.0] | 25.9–30.2 |

The random-position controls are **two orders of magnitude smaller** than the real effects
(0.0003–0.035 against +2.0 to +2.5 nats), while having genuinely non-zero donor deltas — up to
**30.2** at block 31. So they are real patches that write substantially different values and
still do not move the margin. That is the control working as intended, and it is why recording
the donor delta mattered: without it, a small number would be indistinguishable from a patch
that wrote nothing.

## What this does NOT establish — read before citing

- **The endpoints are near-tautological.** Cue recovery of 1.01 at block 0 is close to
  substituting the cue *text*, since a layer-0 residual at those positions is little more than
  the token embedding. Decision recovery of 1.00 at block 31 is the G4 positive control in
  another guise — §8 says explicitly that it "validates that the intervention plumbing can
  change the answer; it is not evidence for a localized authority mechanism." **The
  informative content is the crossover structure between the endpoints, not the endpoints.**
- **Sufficiency within the tested receiving context, and dependence under the tested
  replacement.** Nothing here shows a *minimal* or *complete* circuit. Redundancy can hide
  dependence, and self-repair is known to restore ~70% of an ablated component's effect at
  middle layers — with ~30% of that attributable to the normalization factor alone, which
  applies to RMSNorm as used in Llama. Backup-head results (knocking out *all* Name Mover Heads
  cost only 5% of the logit difference) are the standing warning against reading a localized
  effect as the whole mechanism.
- **The irrelevant-donor control is underpowered: n = 1.** Its guard required another
  scenario's prompt to have identical token length, which almost never held. This must be
  rerun with length-bucketed donor selection before the donor control can be claimed. Reported
  as a gap, not quietly dropped.
- **Denoising only, effectively.** Both directions were run and recovery/injection track each
  other closely (e.g. L0:cue +2.455 recovery vs +2.475 injection), but for a redundant/OR-shaped
  circuit denoising finds all components while noising finds only the last. The near-symmetry
  here is therefore weak evidence *against* strong redundancy, not proof of a serial circuit.
- **A subspace-interchange illusion has not been excluded.** A directional intervention can
  make outputs behave as if a feature changed by activating a dormant parallel pathway rather
  than by touching a variable the model uses — and such illusory directions appear even with
  randomized MLP weights. The two published diagnostics (compare against patching only the
  reader-aligned component; compare the spread of projections onto the reader vs null
  subspaces) have **not** been run. Mitigating factor: we patch the residual stream, which is
  a full bottleneck, and that is exactly where this failure mode is least available.
- **Single model, discovery split, 40 pairs.** No held-out confirmation yet, and no Qwen
  replication. Block indices must not be assumed to transfer; §10 requires comparing function
  rather than layer numbers.

## What it changes downstream

The defense should intervene where the authority signal is still inside the tool span and
being read out of it — i.e. **blocks 10–13**, around the cue→command handoff — rather than the
mid-stack guess (14–16) made before this measurement. G7 was re-parameterized accordingly
before it ran.

It also predicts something falsifiable: an intervention applied at blocks ≥ 16 should be able
to suppress the attack only by acting at the decision position, which is where legitimate
answer selection also happens — so it should damage U and F. An intervention at blocks 10–13
restricted to the tool span acts before legitimate instruction selection is engaged, so it has
a route to being selective. That is the experiment G7 runs.
