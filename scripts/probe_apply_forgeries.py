"""Apply the trained role probe to the forgeries: CoTness(styled vs destyled) + dose-response.

Reproduces the paper's two mechanistic links for Experiment 1:
  * styled forgeries read as the model's own reasoning, destyled ones do not
    (paper: 79.1% vs 29.1% CoTness).
  * CoTness of the injected text predicts attack success (paper dose-response: 9% -> 90% ASR).

CoTness is measured with each forgery wrapped as a **user** message (a non-CoT position), so any
CoTness above baseline is STYLE leaking into a position whose role tag is not CoT -- the paper's
core "style overrides the role tag" claim. Position is held identical for styled and destyled,
so their difference isolates style.

Uses HF gpt-oss (probe needs residual-stream hidden states) + the pickled probe from
probe_cotness.py. Joins the forgery-arm judge labels for the dose-response.

Usage:
    source env.sh && .venv/bin/python scripts/probe_apply_forgeries.py
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common.gptoss_io import load_gptoss
from probe_cotness import (TEST_PREFIX, ROLES, render_single_gptoss,
                           content_token_positions, HiddenCapture)

DATA = Path(os.environ["DATA_DIR"])
R = DATA / "outputs" / "repro"
PG = DATA / "outputs" / "probe_gptoss"
COT_IDX = ROLES.index("cot")


@torch.no_grad()
def cotness(model, tok, cap_hook, layer, clf, text, cap=64, device="cuda"):
    """Mean P(cot) over the body tokens of `text` wrapped as a user message."""
    content = text[:1200]
    rendered = TEST_PREFIX + render_single_gptoss("user", content)
    ids, pos = content_token_positions(tok, rendered, content, cap)
    if not pos:
        return None
    inp = torch.tensor([ids], device=device)
    model(input_ids=inp, attention_mask=torch.ones_like(inp), use_cache=False)
    h = cap_hook.store[layer][0][pos].float().cpu().numpy()
    return float(clf.predict_proba(h)[:, COT_IDX].mean())


def main():
    probe = pickle.load((PG / "role_probe.pkl").open("rb"))
    layer, clf = probe["layer"], probe["clf"]
    print(f"probe: layer {layer}, roles {probe['roles']}", flush=True)

    forg = [json.loads(l) for l in (R / "forgeries_vllm.jsonl").read_text().splitlines() if l.strip()]
    # per-arm judge label (HARMFUL_RESPONSE = attack success), for both forgery and destyled
    labels = {}
    succ_destyled = {}
    for l in (R / "judge_labels_vllm.jsonl").read_text().splitlines():
        if l.strip():
            x = json.loads(l)
            if x["arm"] == "forgery":
                labels[x["prompt"]] = (x["label"] == "HARMFUL_RESPONSE")
            elif x["arm"] == "destyled":
                succ_destyled[x["prompt"]] = (x["label"] == "HARMFUL_RESPONSE")

    model, tok, load_mode = load_gptoss()
    cap_hook = HiddenCapture(model, [layer])
    rows = []
    try:
        for i, f in enumerate(forg):
            cs = cotness(model, tok, cap_hook, layer, clf, f["styled"])
            cd = cotness(model, tok, cap_hook, layer, clf, f["destyled"])
            rows.append({"prompt": f["prompt"], "source": f["source"],
                         "cotness_styled": cs, "cotness_destyled": cd,
                         "attack_success": labels.get(f["prompt"])})
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(forg)}", flush=True)
    finally:
        cap_hook.remove()

    cs = np.array([r["cotness_styled"] for r in rows if r["cotness_styled"] is not None])
    cd = np.array([r["cotness_destyled"] for r in rows if r["cotness_destyled"] is not None])

    # dose-response: POOL styled + destyled attempts so CoTness spans its full range (the
    # paper's 626-attempt design). Binning only the styled arm gives a flat curve because every
    # styled forgery is high-CoTness/high-ASR; the destyled arm supplies the low-CoTness/low-ASR
    # end. Each attempt = (CoTness of the injected text, did THAT attempt succeed).
    ds = []
    for r in rows:
        if r["cotness_styled"] is not None and r["attack_success"] is not None:
            ds.append((r["cotness_styled"], r["attack_success"]))
        cd_succ = succ_destyled.get(r["prompt"])
        if r["cotness_destyled"] is not None and cd_succ is not None:
            ds.append((r["cotness_destyled"], cd_succ))
    ds.sort()
    dose = []
    if ds:
        qs = np.quantile([c for c, _ in ds], [0, 0.2, 0.4, 0.6, 0.8, 1.0])
        for b in range(5):
            lo, hi = qs[b], qs[b + 1]
            bin_items = [s for c, s in ds if (lo <= c <= hi if b == 4 else lo <= c < hi)]
            if bin_items:
                dose.append({"cotness_range": [round(float(lo), 3), round(float(hi), 3)],
                             "n": len(bin_items), "asr": sum(bin_items) / len(bin_items)})

    report = {
        "stage": "forgery_cotness_and_dose_response",
        "probe_layer": layer, "load_mode": load_mode, "n_forgeries": len(rows),
        "cotness_styled_mean": float(cs.mean()), "cotness_destyled_mean": float(cd.mean()),
        "cotness_gap": float(cs.mean() - cd.mean()),
        "paper_reference": {"styled": 0.791, "destyled": 0.291},
        "dose_response": dose,
        "corr_cotness_vs_success": None,
    }
    if ds:
        a = np.array([c for c, _ in ds]); bb = np.array([1.0 if s else 0.0 for _, s in ds])
        if a.std() > 0 and bb.std() > 0:
            report["corr_cotness_vs_success"] = float(np.corrcoef(a, bb)[0, 1])
        report["n_pooled_attempts"] = len(ds)

    (PG / "forgery_cotness_report.json").write_text(json.dumps(report, indent=2))
    (PG / "forgery_cotness_rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    print("\n" + "=" * 60)
    print(f"CoTness styled  = {cs.mean():.3f}   (paper 0.79)")
    print(f"CoTness destyled= {cd.mean():.3f}   (paper 0.29)")
    print(f"gap             = {cs.mean()-cd.mean():+.3f}")
    print(f"corr(CoTness, attack success) = {report['corr_cotness_vs_success']:.3f} "
          f"(n={report.get('n_pooled_attempts')} pooled styled+destyled attempts)")
    print("dose-response (pooled CoTness quintile -> ASR;  paper 9% -> 90%):")
    for d in dose:
        print(f"  CoTness {d['cotness_range']}  n={d['n']:3}  ASR={d['asr']:.3f}")
    print(f"\nwrote {PG}/forgery_cotness_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
