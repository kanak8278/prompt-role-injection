"""Mechanistic 'why': does the forged CONCLUSION work by suppressing the refusal direction?

Arditi et al. 2024 (NeurIPS): refusal is mediated by a single residual-stream direction. Our
factorial says the explicit comply-CONCLUSION is the lever. Hypothesis: the effective conclusion
lowers the model's projection onto the refusal direction at the decision position (it *suppresses
refusal*), which would explain the mechanism (conclusion = refusal-suppression; style amplifies by
getting the text read as reasoning) and tie it to a known circuit.

Steps:
  1. Extract the refusal direction d_L at each candidate layer = normalize(mean act on harmful
     requests - mean act on harmless requests), at the decision position (assistant-turn start).
  2. Pick the layer where d best separates harmful vs harmless (validation).
  3. For each conclusion variant (forged analysis prefill), measure the mean projection onto d at
     the decision position. Correlate per-variant refusal-projection with per-variant ASR.

Prediction: effective conclusions (explicit) -> LOWER refusal projection -> higher ASR
(corr(refusal_proj, ASR) < 0).

GPU (HF gpt-oss, residual-stream hooks). Usage: source env.sh && .venv/bin/python scripts/refusal_direction.py
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

DATA = Path(os.environ["DATA_DIR"])
R = DATA / "outputs" / "repro"
PG = DATA / "outputs" / "probe_gptoss"
LAYERS = [10, 12, 14, 16, 20]


class ResidCapture:
    """Capture decoder-layer residual-stream output (block output) at chosen layers."""
    def __init__(self, model, layers):
        self.store, self.handles = {}, []
        for L in layers:
            self.handles.append(model.model.layers[L].register_forward_hook(self._mk(L)))

    def _mk(self, L):
        def hook(_m, _i, out):
            self.store[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def last_resid(model, tok, cap, prompt, device="cuda"):
    """Residual at the LAST token (the decision position, = assistant-turn start) per layer."""
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt").to(device)
    model(**ids, use_cache=False)
    return {L: cap.store[L][0, -1].float().cpu().numpy() for L in cap.store}


def main():
    forg = [json.loads(l) for l in (R / "forgeries_vllm.jsonl").read_text().splitlines() if l.strip()]
    harmful = [f["prompt"] for f in forg][:128]
    harmless = [f"How do I {t}?" for t in BENIGN_TOPICS] + [f"Can you explain how to {t}?" for t in BENIGN_TOPICS[:32]]
    conc_variants = [json.loads(l) for l in (R / "conclusion_variants.jsonl").read_text().splitlines() if l.strip()]
    # per-variant ASR from the judge
    lab = [json.loads(l) for l in (R / "factorial_judge_labels_concl.jsonl").read_text().splitlines() if l.strip()]
    from collections import defaultdict
    asr = {}
    tmp = defaultdict(list)
    for r in lab:
        tmp[r["cell"]].append(1.0 if r["success"] else 0.0)
    asr = {k: float(np.mean(v)) for k, v in tmp.items()}

    model, tok, load_mode = load_gptoss()
    cap = ResidCapture(model, LAYERS)
    try:
        # ---- 1-2. refusal direction per layer + validation ----
        Hh = [last_resid(model, tok, cap, render_baseline(None, p)) for p in harmful]
        Hl = [last_resid(model, tok, cap, render_baseline(None, p)) for p in harmless]
        dirs, sep = {}, {}
        for L in LAYERS:
            mh = np.mean([h[L] for h in Hh], axis=0); ml = np.mean([h[L] for h in Hl], axis=0)
            d = mh - ml; d = d / (np.linalg.norm(d) + 1e-8)
            dirs[L] = d
            ph = np.array([h[L] @ d for h in Hh]); pl = np.array([h[L] @ d for h in Hl])
            # separation = AUC(harmful>harmless) on the projection
            gt = (ph[:, None] > pl[None, :]).mean()
            sep[L] = float(gt)
        bestL = max(sep, key=sep.get)
        d = dirs[bestL]
        print(f"refusal-direction layer separation (AUC harmful vs harmless): "
              + "  ".join(f"L{L}={sep[L]:.3f}" for L in LAYERS))
        print(f"using best layer L{bestL} (AUC {sep[bestL]:.3f})", flush=True)

        # ---- 3. refusal projection per conclusion variant ----
        by_variant = defaultdict(list)
        # subsample prompts per variant for speed (100)
        keep = set(range(100))
        for r in conc_variants:
            if r["idx"] in keep:
                by_variant[r["cell"]].append(r)
        rows = []
        for v, rs in by_variant.items():
            projs = []
            for r in rs:
                pr = render_forgery(None, r["prompt"], r["forgery_text"])
                projs.append(last_resid(model, tok, cap, pr)[bestL] @ d)
            rows.append({"variant": v, "n": len(rs), "refusal_proj": float(np.mean(projs)),
                         "asr": asr.get(v)})

        # ---- 4. ALSO the 8 factorial S/P/C cells (wider ASR range 0->0.29 = better-powered test) ----
        fac = [json.loads(l) for l in (R / "factorial_forgeries.jsonl").read_text().splitlines() if l.strip()]
        faclab = [json.loads(l) for l in (R / "factorial_judge_labels.jsonl").read_text().splitlines() if l.strip()]
        fac_asr_t = defaultdict(list)
        for r in faclab:
            fac_asr_t[r["cell"]].append(1.0 if r["success"] else 0.0)
        fac_asr = {k: float(np.mean(v)) for k, v in fac_asr_t.items()}
        by_cell = defaultdict(list)
        for r in fac:
            if r["idx"] in keep:
                by_cell[r["cell"]].append(r)
        fac_rows = []
        for c, rs in by_cell.items():
            projs = [last_resid(model, tok, cap, render_forgery(None, r["prompt"], r["forgery_text"]))[bestL] @ d
                     for r in rs]
            fac_rows.append({"cell": c, "n": len(rs), "refusal_proj": float(np.mean(projs)),
                             "asr": fac_asr.get(c)})
    finally:
        cap.remove()

    # baseline refusal projection: harmful prompt with NO forgery (should be high = refusing)
    rows.sort(key=lambda x: x["asr"] if x["asr"] is not None else 0, reverse=True)
    a = np.array([r["asr"] for r in rows]); rp = np.array([r["refusal_proj"] for r in rows])
    corr = float(np.corrcoef(rp, a)[0, 1]) if a.std() > 0 and rp.std() > 0 else None

    fa = np.array([r["asr"] for r in fac_rows]); frp = np.array([r["refusal_proj"] for r in fac_rows])
    fcorr = float(np.corrcoef(frp, fa)[0, 1]) if fa.std() > 0 and frp.std() > 0 else None
    report = {"stage": "refusal_direction", "load_mode": load_mode, "best_layer": bestL,
              "layer_separation_auc": sep, "corr_refusalproj_vs_asr_conclvariants": corr,
              "corr_refusalproj_vs_asr_factorial": fcorr,
              "per_variant": rows, "per_factorial_cell": sorted(fac_rows, key=lambda z: -(z["asr"] or 0))}
    (PG / "refusal_direction_report.json").write_text(json.dumps(report, indent=2))
    print("\n" + "=" * 60)
    print(f"corr(refusal-proj, ASR) conclusion-variants (n=8, ASR 0.16-0.29) = {corr}")
    print(f"corr(refusal-proj, ASR) FACTORIAL cells (n=8, ASR 0.00-0.29)    = {fcorr}  (predict < 0)")
    print(f"\nfactorial cells (wider range):\n{'cell':8} {'ASR':>6} {'refusal_proj':>13}")
    for r in sorted(fac_rows, key=lambda z: -(z["asr"] or 0)):
        print(f"{r['cell']:8} {r['asr']:>6.3f} {r['refusal_proj']:>13.2f}")
    print(f"\nwrote {PG}/refusal_direction_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
