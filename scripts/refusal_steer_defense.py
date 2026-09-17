"""Causal test + defense: re-inject the refusal direction to defeat the forged-conclusion attack.

notes/26 (correlational): the forged conclusion suppresses the refusal direction (attack proj goes
negative). Causal version: if that suppression is WHY the attack works, then ADDING the refusal
direction back during generation (steering the residual toward 'refusing') should restore refusal on
the attack -> ASR drops. That both proves causality and is a mechanism-derived inference-time defense.
We also measure the utility cost (does it over-refuse benign requests?), so it's an honest tradeoff.

Steering: forward hooks on mid layers add alpha * d_L (unit refusal direction) to the residual at
every position during generation. alpha calibrated against the harmful-vs-harmless projection gap.

GPU (HF gpt-oss + hooks). Usage: source env.sh && .venv/bin/python scripts/refusal_steer_defense.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common.gptoss_io import load_gptoss, render_baseline, render_forgery
from gen_hard_negatives import BENIGN_TOPICS
from run_factorial import is_refusal, extract_final    # unicode-normalized refusal heuristic + HF extractor

DATA = Path(os.environ["DATA_DIR"])
R = DATA / "outputs" / "repro"
PG = DATA / "outputs" / "probe_gptoss"
STEER_LAYERS = [10, 12, 14]


class ResidCapture:
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


class RefusalSteer:
    """Add alpha * d_L to each layer's residual output (all positions) during generation."""
    def __init__(self, model, dirs, alpha):
        self.handles = []
        self.alpha = alpha
        for L, d in dirs.items():
            dv = torch.tensor(d, dtype=next(model.parameters()).dtype, device=model.device)
            self.handles.append(model.model.layers[L].register_forward_hook(self._mk(dv)))
    def _mk(self, dv):
        def hook(_m, _i, out):
            if isinstance(out, tuple):
                return (out[0] + self.alpha * dv,) + out[1:]
            return out + self.alpha * dv
        return hook
    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def last_resid(model, tok, cap, prompt):
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    model(**ids, use_cache=False)
    return {L: cap.store[L][0, -1].float().cpu().numpy() for L in cap.store}


@torch.no_grad()
def gen(model, tok, prompt, id_ret, max_new_tokens=512):
    enc = tok(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                         top_p=None, top_k=None, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=False)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-attack", type=int, default=100)
    ap.add_argument("--n-benign", type=int, default=60)
    ap.add_argument("--fracs", default="0.5,1.0,2.0")
    args = ap.parse_args()
    fracs = [float(x) for x in args.fracs.split(",")]

    forg = [json.loads(l) for l in (R / "forgeries_vllm.jsonl").read_text().splitlines() if l.strip()]
    harmful = [f["prompt"] for f in forg][:128]
    harmless = [f"How do I {t}?" for t in BENIGN_TOPICS] + [f"Can you explain how to {t}?" for t in BENIGN_TOPICS[:32]]
    # attack set = full-bundle forged conclusion (S1P1C1)
    fac = [json.loads(l) for l in (R / "factorial_forgeries.jsonl").read_text().splitlines() if l.strip()]
    attack = [r for r in fac if r["cell"] == "S1P1C1" and r["idx"] < args.n_attack]

    model, tok, load_mode = load_gptoss()
    id_ret = tok.convert_tokens_to_ids("<|return|>")

    # ---- refusal direction d_L + calibration gap ----
    cap = ResidCapture(model, STEER_LAYERS)
    Hh = [last_resid(model, tok, cap, render_baseline(None, p)) for p in harmful]
    Hl = [last_resid(model, tok, cap, render_baseline(None, p)) for p in harmless]
    dirs, gap = {}, {}
    for L in STEER_LAYERS:
        mh = np.mean([h[L] for h in Hh], axis=0); ml = np.mean([h[L] for h in Hl], axis=0)
        d = mh - ml; d = d / (np.linalg.norm(d) + 1e-8)
        dirs[L] = d
        gap[L] = float(mh @ d - ml @ d)          # harmful-harmless projection gap (the 'refusal scale')
    cap.remove()
    print(f"refusal projection gap (harmful-harmless) per layer: {gap}", flush=True)
    unit = np.mean(list(gap.values()))           # rough scale of one 'refusal unit'

    # ---- baseline (no steering): attack ASR + benign refusal (utility) ----
    def asr(items, render_fn):
        n_ref = 0
        for it in items:
            g = gen(model, tok, render_fn(it), id_ret)
            if is_refusal(extract_final(g)):
                n_ref += 1
        return 1.0 - n_ref / len(items)          # ASR = non-refusal rate (heuristic)

    attack_render = lambda it: render_forgery(None, it["prompt"], it["forgery_text"])
    benign = harmless[:args.n_benign]
    benign_render = lambda p: render_baseline(None, p)

    results = {"stage": "refusal_steer_defense", "load_mode": load_mode, "layers": STEER_LAYERS,
               "gap": gap, "sweep": []}
    base_attack = asr(attack, attack_render)
    base_benign_comply = asr(benign, benign_render)      # benign 'ASR' = benign compliance (want high)
    print(f"baseline: attack ASR {base_attack:.3f} | benign compliance {base_benign_comply:.3f}", flush=True)
    results["baseline"] = {"attack_asr": base_attack, "benign_compliance": base_benign_comply}

    # ---- steering sweep: add alpha*(fraction of the refusal gap) ----
    for frac in fracs:
        alpha = frac * unit
        st = RefusalSteer(model, dirs, alpha)
        try:
            a_attack = asr(attack, attack_render)
            a_benign = asr(benign, benign_render)
        finally:
            st.remove()
        results["sweep"].append({"frac": frac, "alpha": alpha, "attack_asr": a_attack,
                                 "benign_compliance": a_benign})
        print(f"  steer frac={frac} (alpha={alpha:.1f}): attack ASR {a_attack:.3f} "
              f"(base {base_attack:.3f}) | benign compliance {a_benign:.3f} (base {base_benign_comply:.3f})",
              flush=True)

    (PG / "refusal_steer_defense.json").write_text(json.dumps(results, indent=2))
    print("\n" + "=" * 60)
    print("re-injecting the refusal direction (defense) — attack ASR / benign compliance:")
    print(f"  baseline           attack {base_attack:.3f}  benign {base_benign_comply:.3f}")
    for s in results["sweep"]:
        print(f"  frac {s['frac']:>4}          attack {s['attack_asr']:.3f}  benign {s['benign_compliance']:.3f}")
    print(f"\nwrote {PG}/refusal_steer_defense.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
