"""Apply the trained role probe to the forgeries, reproducing the paper's own analysis
(experiments/cot-forgery-role-confusion/04-analyze-injection-probe-results.ipynb).

The paper makes the "role confusion predicts attack success" case with a **2-level style
contrast** (policy_style in {base, destyled}) — NOT a continuous within-style dose. Its figures:
  * styled ("base") forgeries read as high-CoTness, destyled ones do not (paper 79% vs 29%);
  * a role simplex where each attempt is placed by (CoTness, Userness, other) and colored by
    attack success — successes cluster toward the CoTness corner, failures toward Userness;
  * RCI = 0.5*(CoTness - Userness) + 0.5, the paper's scalar "role confusion index".

We reproduce all three. Each forgery is wrapped as a **user** message (a non-CoT position), so
any CoTness above baseline is STYLE leaking into a position whose role tag is not CoT. Position
is identical for styled and destyled, so their difference isolates style.

HONESTY NOTE (from the verification-discipline instruction): the pooled styled+destyled
CoTness->ASR correlation is a BETWEEN-STYLE effect. Within each arm CoTness barely varies and
does not predict success (reported below as corr_within_arm). This matches the paper, whose
evidence is also the base-vs-destyled contrast + simplex clustering, not a graded within-style
curve. We report both so the effect is not overstated as a continuous dose-response.

Uses HF gpt-oss (probe needs residual-stream hidden states) + the pickled probe from
probe_cotness.py. Joins the per-arm judge labels (HARMFUL_RESPONSE = attack success).

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
USER_IDX = ROLES.index("user")


@torch.no_grad()
def roleness(model, tok, cap_hook, layer, clf, text, cap=64, device="cuda"):
    """Mean predict_proba vector (over ROLES) across the body tokens of `text` as a user msg.

    Returns a dict {cotness, userness, rci} or None. rci is the paper's role-confusion index
    0.5*(cotness - userness) + 0.5, on [0,1] with 1 = pure CoT perception.
    """
    content = text[:1200]
    rendered = TEST_PREFIX + render_single_gptoss("user", content)
    ids, pos = content_token_positions(tok, rendered, content, cap)
    if not pos:
        return None
    inp = torch.tensor([ids], device=device)
    model(input_ids=inp, attention_mask=torch.ones_like(inp), use_cache=False)
    h = cap_hook.store[layer][0][pos].float().cpu().numpy()
    p = clf.predict_proba(h).mean(axis=0)          # mean role distribution over body tokens
    cot, usr = float(p[COT_IDX]), float(p[USER_IDX])
    return {"cotness": cot, "userness": usr, "rci": 0.5 * (cot - usr) + 0.5}


def corr(pairs):
    """Pearson r over [(x, bool_success), ...]; None if degenerate."""
    if not pairs:
        return None
    x = np.array([p[0] for p in pairs]); y = np.array([1.0 if p[1] else 0.0 for p in pairs])
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def dose_curve(pairs, nbins=5):
    """Quantile-binned ASR over [(score, bool_success), ...]."""
    pairs = sorted(pairs)
    if not pairs:
        return []
    qs = np.quantile([p[0] for p in pairs], np.linspace(0, 1, nbins + 1))
    out = []
    for b in range(nbins):
        lo, hi = qs[b], qs[b + 1]
        items = [s for c, s in pairs if (lo <= c <= hi if b == nbins - 1 else lo <= c < hi)]
        if items:
            out.append({"range": [round(float(lo), 3), round(float(hi), 3)],
                        "n": len(items), "asr": sum(items) / len(items)})
    return out


def main():
    probe = pickle.load((PG / "role_probe.pkl").open("rb"))
    layer, clf = probe["layer"], probe["clf"]
    print(f"probe: layer {layer}, roles {probe['roles']}", flush=True)

    forg = [json.loads(l) for l in (R / "forgeries_vllm.jsonl").read_text().splitlines() if l.strip()]
    succ_styled, succ_destyled = {}, {}
    for l in (R / "judge_labels_vllm.jsonl").read_text().splitlines():
        if l.strip():
            x = json.loads(l)
            if x["arm"] == "forgery":
                succ_styled[x["prompt"]] = (x["label"] == "HARMFUL_RESPONSE")
            elif x["arm"] == "destyled":
                succ_destyled[x["prompt"]] = (x["label"] == "HARMFUL_RESPONSE")

    model, tok, load_mode = load_gptoss()
    cap_hook = HiddenCapture(model, [layer])
    rows = []
    try:
        for i, f in enumerate(forg):
            rs = roleness(model, tok, cap_hook, layer, clf, f["styled"])
            rd = roleness(model, tok, cap_hook, layer, clf, f["destyled"])
            rows.append({
                "prompt": f["prompt"], "source": f["source"],
                "cotness_styled": rs["cotness"] if rs else None,
                "userness_styled": rs["userness"] if rs else None,
                "rci_styled": rs["rci"] if rs else None,
                "cotness_destyled": rd["cotness"] if rd else None,
                "userness_destyled": rd["userness"] if rd else None,
                "rci_destyled": rd["rci"] if rd else None,
                "success_styled": succ_styled.get(f["prompt"]),
                "success_destyled": succ_destyled.get(f["prompt"]),
            })
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(forg)}", flush=True)
    finally:
        cap_hook.remove()

    cs = np.array([r["cotness_styled"] for r in rows if r["cotness_styled"] is not None])
    cd = np.array([r["cotness_destyled"] for r in rows if r["cotness_destyled"] is not None])
    us = np.array([r["userness_styled"] for r in rows if r["userness_styled"] is not None])
    ud = np.array([r["userness_destyled"] for r in rows if r["userness_destyled"] is not None])

    # per-attempt (score, success) pairs, tagged by arm, for pooled + within-arm analysis
    styled_pairs, destyled_pairs = [], []          # (cotness, success)
    styled_rci, destyled_rci = [], []              # (rci, success)
    simplex = []                                    # for the paper's ternary figure
    for r in rows:
        if r["cotness_styled"] is not None and r["success_styled"] is not None:
            styled_pairs.append((r["cotness_styled"], r["success_styled"]))
            styled_rci.append((r["rci_styled"], r["success_styled"]))
            simplex.append({"arm": "styled", "cotness": r["cotness_styled"],
                            "userness": r["userness_styled"], "rci": r["rci_styled"],
                            "success": r["success_styled"]})
        if r["cotness_destyled"] is not None and r["success_destyled"] is not None:
            destyled_pairs.append((r["cotness_destyled"], r["success_destyled"]))
            destyled_rci.append((r["rci_destyled"], r["success_destyled"]))
            simplex.append({"arm": "destyled", "cotness": r["cotness_destyled"],
                            "userness": r["userness_destyled"], "rci": r["rci_destyled"],
                            "success": r["success_destyled"]})

    pooled_cot = styled_pairs + destyled_pairs
    pooled_rci = styled_rci + destyled_rci

    report = {
        "stage": "forgery_role_confusion",
        "method": "paper 04-analyze-injection-probe-results: base-vs-destyled contrast + simplex + RCI",
        "probe_layer": layer, "load_mode": load_mode, "n_forgeries": len(rows),
        # (1) 2-level style contrast (the paper's CoTness / Userness bars)
        "contrast": {
            "cotness_styled": float(cs.mean()), "cotness_destyled": float(cd.mean()),
            "cotness_gap": float(cs.mean() - cd.mean()),
            "userness_styled": float(us.mean()), "userness_destyled": float(ud.mean()),
            "paper_reference_cotness": {"styled": 0.791, "destyled": 0.291},
        },
        # (2) role confusion vs attack success, reported HONESTLY: pooled AND within-arm
        "cotness_vs_success": {
            "corr_pooled": corr(pooled_cot), "n_pooled": len(pooled_cot),
            "corr_within_styled": corr(styled_pairs), "n_styled": len(styled_pairs),
            "corr_within_destyled": corr(destyled_pairs), "n_destyled": len(destyled_pairs),
            "note": ("pooled corr is a BETWEEN-STYLE effect; within-arm ~0 means CoTness does not "
                     "grade success inside a fixed style. Matches the paper's 2-level design."),
            "dose_pooled": dose_curve(pooled_cot),
        },
        "rci_vs_success": {
            "corr_pooled": corr(pooled_rci),
            "dose_pooled": dose_curve(pooled_rci),
        },
        # (3) simplex separation: mean role coords for success vs failure (the ternary's claim)
        "simplex_means": {
            cls: {
                "n": sum(1 for s in simplex if s["success"] is (cls == "success")),
                "cotness": float(np.mean([s["cotness"] for s in simplex
                                          if s["success"] is (cls == "success")])),
                "userness": float(np.mean([s["userness"] for s in simplex
                                           if s["success"] is (cls == "success")])),
                "rci": float(np.mean([s["rci"] for s in simplex
                                      if s["success"] is (cls == "success")])),
            } for cls in ("success", "failure")
        },
    }

    (PG / "forgery_cotness_report.json").write_text(json.dumps(report, indent=2))
    (PG / "forgery_cotness_rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (PG / "forgery_simplex.jsonl").write_text("\n".join(json.dumps(s) for s in simplex) + "\n")

    c = report["contrast"]; cv = report["cotness_vs_success"]; sm = report["simplex_means"]
    print("\n" + "=" * 64)
    print("(1) 2-level style contrast (paper base vs destyled):")
    print(f"    CoTness  styled {c['cotness_styled']:.3f}  destyled {c['cotness_destyled']:.3f}  "
          f"gap {c['cotness_gap']:+.3f}   (paper 0.79 / 0.29)")
    print(f"    Userness styled {c['userness_styled']:.3f}  destyled {c['userness_destyled']:.3f}")
    print("(2) role confusion -> attack success (honest: pooled vs within-arm):")
    print(f"    corr(CoTness,succ)  pooled {cv['corr_pooled']:+.3f} (n={cv['n_pooled']})  "
          f"within-styled {cv['corr_within_styled']:+.3f}  within-destyled {cv['corr_within_destyled']:+.3f}")
    print(f"    corr(RCI,succ) pooled {report['rci_vs_success']['corr_pooled']:+.3f}")
    print(f"    pooled CoTness quintile -> ASR: "
          f"{[round(d['asr'], 3) for d in cv['dose_pooled']]}  (paper shape 9% -> 90%)")
    print("(3) simplex separation (success clusters toward CoTness, failure toward Userness):")
    print(f"    success  CoTness {sm['success']['cotness']:.3f}  Userness {sm['success']['userness']:.3f}  "
          f"RCI {sm['success']['rci']:.3f}  (n={sm['success']['n']})")
    print(f"    failure  CoTness {sm['failure']['cotness']:.3f}  Userness {sm['failure']['userness']:.3f}  "
          f"RCI {sm['failure']['rci']:.3f}  (n={sm['failure']['n']})")
    print(f"\nwrote {PG}/forgery_cotness_report.json (+ rows, + simplex)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
