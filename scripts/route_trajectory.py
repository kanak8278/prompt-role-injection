"""P2 — Route-around vs downstream wash-out: refusal projection ACROSS ALL LAYERS at the decision
position, for the adaptive-jailbreak-under-steering input vs the (refusing) plain forgery.

notes/30 inferred "route-around" (compliance at high projection on d) from a single-position, 3-layer
measurement. An equally consistent story: the adaptive suffix makes the *later* (un-steered) layers
ignore/erase the mid-layer steering (downstream wash-out). We distinguish them by the per-LAYER
projection profile at the decision position, under steering:

  - steered layers {10,12,14}: +alpha added by construction -> high for everyone (uninformative).
  - UN-steered later layers (16..): if adaptive-jailbreak has LOW projection there (like the natural
    no-steer compliance profile) -> WASH-OUT (the model recovers a complying state downstream).
    If adaptive-jailbreak keeps HIGH projection there yet complies -> genuine ROUTE-AROUND
    (compliance decoupled from d at every layer).

Uses the already-optimized adaptive suffixes from gcg_defense_main.json (the 5 jailbroken-under-steer
prompts + the 2 that held, for contrast). GPU (HF), no GCG. Usage:
    source env.sh && .venv/bin/python scripts/route_trajectory.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common.gptoss_io import load_gptoss, render_baseline, render_forgery
from gen_hard_negatives import BENIGN_TOPICS
from refusal_steer_defense import ResidCapture, last_resid
from gcg_defense import make_steer_hooks, _install, _remove

DATA = Path(os.environ["DATA_DIR"])
R = DATA / "outputs" / "repro"
PG = DATA / "outputs" / "probe_gptoss"
STEERED = [10, 12, 14]


def main():
    model, tok, _ = load_gptoss()
    n_layers = len(model.model.layers)
    all_layers = list(range(2, n_layers, 2))            # every other layer
    forg = [json.loads(l) for l in (R / "forgeries_vllm.jsonl").read_text().splitlines() if l.strip()]
    harmful = [f["prompt"] for f in forg][:128]
    harmless = [f"How do I {t}?" for t in BENIGN_TOPICS] + [f"Can you explain how to {t}?" for t in BENIGN_TOPICS[:32]]
    fmap = {f["prompt"]: f["styled"] for f in forg}

    rep = json.loads((PG / "gcg_defense_main.json").read_text())["results"]

    # per-layer unit refusal direction + poles
    cap = ResidCapture(model, all_layers)
    Hh = [last_resid(model, tok, cap, render_baseline(None, p)) for p in harmful]
    Hl = [last_resid(model, tok, cap, render_baseline(None, p)) for p in harmless]
    dunit, poleH, poleL, gap = {}, {}, {}, {}
    for L in all_layers:
        mh = np.mean([h[L] for h in Hh], axis=0); ml = np.mean([h[L] for h in Hl], axis=0)
        d = mh - ml; d = d / (np.linalg.norm(d) + 1e-8)
        dunit[L] = d; poleH[L] = float(mh @ d); poleL[L] = float(ml @ d)
        gap[L] = poleH[L] - poleL[L]
    # steering hooks at {10,12,14} (frac 0.75) — reuse the exact defense config
    alpha = 0.75 * float(np.mean([gap[L] for L in STEERED]))
    steer_hooks = make_steer_hooks(model, {L: dunit[L] for L in STEERED}, alpha)

    def profile(injected, req, steer):
        h = _install(steer_hooks) if steer else []
        try:
            res = last_resid(model, tok, cap, render_forgery(None, req, injected))
        finally:
            _remove(h)
        return {L: float(res[L] @ dunit[L]) for L in all_layers}

    rows = []
    for o in rep:
        req = o["prompt"]; full = fmap[req]; adv = o.get("adv_steer", "")
        jb = o.get("gcgsteer_steer_judge") == "HARMFUL_RESPONSE"
        rows.append({"idx": o["idx"], "jailbroke_steer": jb,
                     "adaptive_steer": profile(full + " " + adv, req, True),      # jailbreaks (if jb)
                     "forgery_steer": profile(full + " ! ! !", req, True),          # refuses
                     "forgery_nosteer": profile(full, req, False)})                 # natural compliance profile
    cap.remove()

    def mean_over(rows_sub, key, L):
        return float(np.mean([r[key][L] for r in rows_sub]))

    jbrows = [r for r in rows if r["jailbroke_steer"]]
    report = {"stage": "route_trajectory", "steered_layers": STEERED, "alpha": alpha,
              "layers": all_layers, "poles_refusing": poleH, "poles_complying": poleL,
              "n_jailbroke": len(jbrows), "rows": rows}
    (PG / "route_trajectory.json").write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 84)
    print(f"REFUSAL PROJECTION per layer at decision position (n jailbroke-under-steer={len(jbrows)})")
    print(f"steered layers {STEERED} (alpha {alpha:.0f}); '*'=steered. poles vary per layer.")
    print(f"{'layer':>5} {'adaptiveSteer':>14} {'forgerySteer':>13} {'forgeryNoSteer':>15} "
          f"{'refusePole':>11} {'complyPole':>11}")
    for L in all_layers:
        star = "*" if L in STEERED else " "
        print(f"{L:>4}{star} {mean_over(jbrows,'adaptive_steer',L):>14.1f} "
              f"{mean_over(jbrows,'forgery_steer',L):>13.1f} {mean_over(jbrows,'forgery_nosteer',L):>15.1f} "
              f"{poleH[L]:>11.1f} {poleL[L]:>11.1f}")
    # verdict heuristic: at un-steered later layers, is adaptive closer to forgery_steer(refuse,high)
    # or to forgery_nosteer(comply,low)?
    later = [L for L in all_layers if L > max(STEERED)]
    a = np.mean([mean_over(jbrows, 'adaptive_steer', L) for L in later])
    fs = np.mean([mean_over(jbrows, 'forgery_steer', L) for L in later])
    fn = np.mean([mean_over(jbrows, 'forgery_nosteer', L) for L in later])
    print(f"\nlater un-steered layers (>{max(STEERED)}): adaptiveSteer {a:.1f} | forgerySteer(refuse) {fs:.1f} "
          f"| forgeryNoSteer(comply) {fn:.1f}")
    print("  adaptive close to forgeryNoSteer(low) => WASH-OUT downstream; close to forgerySteer(high) "
          "=> genuine ROUTE-AROUND (complies at high projection).")
    print(f"wrote {PG}/route_trajectory.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
