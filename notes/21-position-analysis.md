# Position analysis: role perception has positional structure

Reproduces the paper's positional analysis (experiments/position-analysis): the probe's role
reading of a token depends on its **position**, not only its content / role tag.
`scripts/position_analysis.py`. Deviation: C4 instead of the paper's generated conversation data
(same corpus deviation as the probe, notes/15); the positional structure is content-agnostic.

## A. Role binding decays with position

Neutral content wrapped in a single **user** turn (the role tag says "user" throughout); mean
P(role) by position decile over the turn's body tokens:

| position decile → | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **P(user)** | 0.98 | 0.92 | 0.88 | 0.83 | 0.76 | 0.71 | 0.72 | 0.65 | 0.61 | **0.60** |
| **P(cot)** | 0.00 | 0.01 | 0.02 | 0.03 | 0.06 | 0.06 | 0.05 | 0.07 | 0.09 | **0.08** |
| P(assistant) | ~0 | ~0 | ~0 | ~0 | ~0 | ~0 | ~0 | ~0 | ~0 | ~0 |

Even though the role tag is "user" the whole way, **Userness decays 0.98 → 0.60 and CoTness rises
0 → 0.08 across the turn.** The role delimiter binds strongly at the start and its influence weakens
with distance — the paper's core positional claim (role perception is positional, not just a
function of the nearest tag).

**Injection-relevant reading:** content buried **deep** in a long turn (or a long tool output / web
page) is perceived as *less user, more CoT* than content near the delimiter. So *where* a payload
sits matters: the further from the user tag, the more it drifts toward being read as the model's own
reasoning — a positional contribution to role confusion, on top of the style effect.

## B. For a styled forgery span, style dominates position

Same styled forgery span embedded early / middle / late inside a benign user turn:

| position | CoTness | Userness | RCI |
| --- | --- | --- | --- |
| early | 0.616 | 0.137 | 0.740 |
| middle | 0.558 | 0.189 | 0.685 |
| late | 0.609 | 0.171 | 0.719 |

The forgery's CoTness is high and roughly flat across positions (0.56–0.62; a slight middle dip) —
**style already saturates the CoT reading, so position adds little.** Consistent with the paper's
central claim that *style* is the dominant lever for CoT forgery; the positional effect (part A) is
a second-order contributor that matters most for *un-styled* content.

## Together

Two contributors to how injected text is role-perceived: **style** (dominant — a forged `analysis`
block reads as CoT anywhere, part B and notes/15/19) and **position** (secondary — even plain
content drifts user→CoT with distance from the delimiter, part A). Both push injected content away
from "untrusted user/tool input" toward "the model's own reasoning," which is the mechanism the
whole paper is about.

## Artifacts

- `scripts/position_analysis.py`
- `$DATA_DIR/outputs/probe_gptoss/position_analysis_report.json`
