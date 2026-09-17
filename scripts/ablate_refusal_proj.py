"""Mechanism cross-check for the component ablation: refusal-direction projection per arm.

notes/26 (correlational) + notes/27 (causal): the forged CONCLUSION suppresses the Arditi refusal
direction (attack projection goes negative), and re-injecting the direction defends. Here we tie the
ABLATION to that mechanism: for each ablation arm we render the ablated forgery and measure the
refusal-direction projection at the DECISION position (right after <|start|>assistant). If the
conclusion is the causal lever, then removing it (drop_conclusion) should RESTORE the projection
toward the refusing pole, while removing restate / permission / reasoning should NOT. This is
mechanistic evidence that is robust to imperfect sentence tagging: it directly asks which removed
component brings the refusal direction back.

Refusal direction = normalize(mean_harmful - mean_harmless) at each STEER_LAYER, from render_baseline
of harmful (forgery prompts) vs harmless (benign) requests — identical construction to
refusal_steer_defense.py, so magnitudes are comparable to notes/27.

Sign convention: HIGH projection = 'refusing' pole (harmful baseline), LOW/negative = 'complying'
(suppressed). Poles reported for reference.

GPU (HF gpt-oss, GPU-1 — run AFTER vLLM is torn down). Usage:
    source env.sh && .venv/bin/python scripts/ablate_refusal_proj.py --n 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common.gptoss_io import load_gptoss, render_baseline, render_forgery
from gen_hard_negatives import BENIGN_TOPICS
from refusal_steer_defense import ResidCapture, last_resid, STEER_LAYERS

DATA = Path(os.environ["DATA_DIR"])
R = DATA / "outputs" / "repro"
PG = DATA / "outputs" / "probe_gptoss"
ARM_ORDER = ["baseline_noforgery", "full", "drop_conclusion", "drop_permission", "drop_restate",
             "drop_reasoning", "only_conclusion", "prefix_before_conclusion"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    variants = [json.loads(l) for l in (R / "component_ablation_variants.jsonl").read_text().splitlines() if l.strip()]
    keep = set(sorted({v["idx"] for v in variants})[: args.n])
    variants = [v for v in variants if v["idx"] in keep]
    byidx = defaultdict(dict)
    for v in variants:
        byidx[v["idx"]][v["arm"]] = v

    forg = [json.loads(l) for l in (R / "forgeries_vllm.jsonl").read_text().splitlines() if l.strip()]
    harmful = [f["prompt"] for f in forg][:128]
    harmless = [f"How do I {t}?" for t in BENIGN_TOPICS] + [f"Can you explain how to {t}?" for t in BENIGN_TOPICS[:32]]

    model, tok, load_mode = load_gptoss()
    cap = ResidCapture(model, STEER_LAYERS)

    # ---- refusal direction + poles (same construction as notes/27) ----
    Hh = [last_resid(model, tok, cap, render_baseline(None, p)) for p in harmful]
    Hl = [last_resid(model, tok, cap, render_baseline(None, p)) for p in harmless]
    dirs, pole_harmful, pole_harmless = {}, {}, {}
    for L in STEER_LAYERS:
        mh = np.mean([h[L] for h in Hh], axis=0); ml = np.mean([h[L] for h in Hl], axis=0)
        d = mh - ml; d = d / (np.linalg.norm(d) + 1e-8)
        dirs[L] = d
        pole_harmful[L] = float(mh @ d); pole_harmless[L] = float(ml @ d)
    pole_h = float(np.mean(list(pole_harmful.values())))    # refusing pole
    pole_l = float(np.mean(list(pole_harmless.values())))   # complying pole
    print(f"poles: refusing(harmful baseline)={pole_h:.2f}  complying(harmless)={pole_l:.2f}", flush=True)

    def proj_of(prompt, arm_text, is_baseline):
        rendered = render_baseline(None, prompt) if is_baseline else render_forgery(None, prompt, arm_text)
        h = last_resid(model, tok, cap, rendered)
        return float(np.mean([h[L] @ dirs[L] for L in STEER_LAYERS]))

    # ---- per-arm projection at the decision position ----
    arm_proj = defaultdict(list)
    idxs = sorted(keep)
    for n, idx in enumerate(idxs):
        for arm in ARM_ORDER:
            v = byidx[idx].get(arm)
            if v is None:
                continue
            is_base = (arm == "baseline_noforgery") or (not v["forgery_text"].strip())
            arm_proj[arm].append(proj_of(v["prompt"], v["forgery_text"], is_base))
        if (n + 1) % 50 == 0:
            print(f"  {n+1}/{len(idxs)} forgeries", flush=True)
    cap.remove()

    per_arm = {}
    for arm in ARM_ORDER:
        a = np.array(arm_proj[arm])
        if len(a) == 0:
            continue
        # normalize to 0=complying pole, 1=refusing pole for interpretability
        restored = (a.mean() - pole_l) / (pole_h - pole_l + 1e-8)
        per_arm[arm] = {"n": len(a), "mean_proj": float(a.mean()), "std_proj": float(a.std()),
                        "frac_of_refusing_pole": float(restored)}

    report = {"stage": "component_ablation_refusal_proj", "load_mode": load_mode,
              "layers": STEER_LAYERS, "pole_refusing": pole_h, "pole_complying": pole_l,
              "per_arm": per_arm}
    (PG / "ablation_refusal_proj.json").write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 78)
    print(f"REFUSAL-DIRECTION PROJECTION per ablation arm (n={args.n}) at the decision position")
    print(f"  poles: complying={pole_l:.2f} (0.0)  ...  refusing={pole_h:.2f} (1.0)")
    print(f"{'arm':26} {'mean_proj':>10} {'frac_refusing_pole':>20}")
    full_p = per_arm.get("full", {}).get("mean_proj", 0.0)
    for arm in ARM_ORDER:
        if arm not in per_arm:
            continue
        v = per_arm[arm]
        d = v["mean_proj"] - full_p
        print(f"{arm:26} {v['mean_proj']:>10.2f} {v['frac_of_refusing_pole']:>19.3f}  "
              f"(vs full {d:+.2f})")
    print("\n(higher = refusal RESTORED; if drop_conclusion rises toward the refusing pole while "
          "drop_permission/restate/reasoning stay low, the conclusion is the causal lever)")
    print(f"wrote {PG}/ablation_refusal_proj.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
