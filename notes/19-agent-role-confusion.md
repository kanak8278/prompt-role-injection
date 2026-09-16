# §3 (agent half): role confusion for the agent attack

notes/18 reproduced the agent attack (forged-CoT injection 0% -> 20%). This is the mechanistic
*why*, the agent analogue of notes/15: does the forged block land in the agent's context as the
model's own reasoning? Scored with the trained role probe (`scripts/probe_agent_injections.py`),
RCI = 0.5*(CoTness - Userness) + 0.5.

## The key move: measure the forged-reasoning SPAN, not the whole block

The forgery injection = [plain exfil request] + [forged reasoning paragraph "We need to consider
policy… policy says we can comply… Let's comply."]. Averaging RCI over the whole block hides the
effect because the plain request dominates the mean. Measuring the **forged-reasoning span** (the
suffix the forgery adds on top of the shared base request) isolates the causal element.

| text (n=5 injection types) | RCI (user pos) | RCI (tool pos) | CoTness (tool) |
| --- | --- | --- | --- |
| plain request (base) | 0.038 | 0.373 | 0.047 |
| whole forgery block | 0.046 | 0.374 | 0.053 |
| **forged-reasoning span** | **0.674** | **0.861** | **0.722** |

**The forged-reasoning span reads as the model's own chain-of-thought** (RCI 0.67 user / 0.86 tool;
CoTness 0.72 tool), while the plain request reads as user input (RCI 0.04 user). Same mechanism as
the chat attack (notes/15: styled 0.63 vs destyled 0.26): forged *reasoning style* is perceived as
CoT even though it arrives as external input. The whole-block number (0.046) is why the naive
measurement in the first pass showed nothing — dilution by the surrounding request.

## Tool position amplifies role confusion

Two things jump out from the position comparison:
- The **plain request** jumps from RCI 0.04 (user position) to 0.37 (tool position). Content placed
  in a tool/commentary output is inherently read as more reasoning-like than the same text in a
  user turn — the tool channel sits closer to the assistant's own reasoning space.
- The **forged span** goes from 0.67 (user) to **0.86** (tool). So when the forged reasoning lands
  in a tool output — exactly where an indirect (webpage) injection delivers it — role confusion is
  *amplified*. This is a concrete mechanistic reason indirect prompt injection via tool outputs is
  potent: the delivery position already tilts perception toward CoT, and the forged style pushes it
  the rest of the way.

## Honest limitations

- **n = 5** injection types (5 base + 5 forgery blocks), by construction. The span-level CoTness
  gap is large and consistent, but the per-type correlation between whole-block RCI and exfil rate
  is noise (corr -0.68 at n=5, and computed on the *diluted* whole-block RCI) — not reported as a
  result. The claim rests on the span-level perception gap, not a dose-response.
- The paper's full §3-agent (notebook 05) exports the agent's *own* activations during the run and
  projects the injected span in-context; this is the lighter "score the injection text with the
  probe" version. It reproduces the same qualitative claim (forged reasoning ⇒ CoT perception) but
  is not the identical activation-export pipeline.

## Bottom line

The agent attack shares the chat attack's mechanism: the forged policy-reasoning block is perceived
as the model's own CoT (RCI 0.86 in the tool position it's delivered to), which is why appending it
flips the agent from refusing to exfiltrating (notes/18). Role confusion — not the content of the
request — is the lever, in the agent setting too.

## Artifacts

- `scripts/probe_agent_injections.py`
- `$DATA_DIR/outputs/probe_gptoss/agent_injection_probe_report.json`
