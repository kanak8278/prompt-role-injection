"""Position analysis: how does the role probe's reading depend on TOKEN POSITION?

Reproduces the paper's positional analysis (experiments/position-analysis): role perception has
positional structure, and *where* content sits changes how its role is read. Two parts:

  A. Roleness vs absolute token position. Concatenate neutral content into one sequence (no role
     delimiters) and average P(role) at each token position (paper's cell 11 / roleness-vs-
     token_in_prompt_ix). Shows the probe's role reading drifts with position, not just content.

  B. Injection-relevant variant: take a fixed styled forgery span, embed it at early / middle /
     late positions inside a benign user turn, and measure the span's CoTness/Userness at each
     position. Tests whether the SAME injected text is perceived as more/less CoT depending on
     where it lands -- the position analogue of §3's user-vs-tool result (notes/19).

Deviation: the paper uses generated conversation data (toxicchat/oasst); we use C4 (same corpus
deviation as the probe, notes/15). The positional *structure* is content-agnostic, so this is a
faithful test of the position claim.

Usage:
    source env.sh && .venv/bin/python scripts/position_analysis.py
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
C4 = DATA / "datasets" / "external" / "c4_sample.jsonl"
R = DATA / "outputs" / "repro"
PG = DATA / "outputs" / "probe_gptoss"
COT, USER, ASST = ROLES.index("cot"), ROLES.index("user"), ROLES.index("assistant")


@torch.no_grad()
def token_roleness(model, tok, cap_hook, layer, clf, rendered, max_tokens=400, device="cuda"):
    """Return per-token P(role) over the whole rendered sequence (body tokens by offset)."""
    enc = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"][:max_tokens]
    inp = torch.tensor([ids], device=device)
    model(input_ids=inp, attention_mask=torch.ones_like(inp), use_cache=False)
    h = cap_hook.store[layer][0][: len(ids)].float().cpu().numpy()
    return clf.predict_proba(h)          # [T, n_roles]


@torch.no_grad()
def span_roleness(model, tok, cap_hook, layer, clf, rendered, span, cap=64, device="cuda"):
    ids, pos = content_token_positions(tok, rendered, span, cap)
    if not pos:
        return None
    inp = torch.tensor([ids], device=device)
    model(input_ids=inp, attention_mask=torch.ones_like(inp), use_cache=False)
    h = cap_hook.store[layer][0][pos].float().cpu().numpy()
    p = clf.predict_proba(h).mean(axis=0)
    return {"cotness": float(p[COT]), "userness": float(p[USER]),
            "rci": float(0.5 * (p[COT] - p[USER]) + 0.5)}


def main():
    probe = pickle.load((PG / "role_probe.pkl").open("rb"))
    layer, clf = probe["layer"], probe["clf"]
    docs = [json.loads(l)["text"] for l in C4.read_text().splitlines() if l.strip()]
    model, tok, load_mode = load_gptoss()
    cap = HiddenCapture(model, [layer])

    report = {"stage": "position_analysis", "load_mode": load_mode, "layer": layer}
    try:
        # ---- A. roleness vs token position: content wrapped as a single USER turn ----
        # (fixed role tag; any positional drift in P(cot)/P(assistant) is a POSITION effect, since
        # the role tag says 'user' throughout).
        nbins = 10
        acc = {r: np.zeros(nbins) for r in ("user", "cot", "assistant")}
        cnt = np.zeros(nbins)
        n_docs = 60
        for d in docs[300:300 + n_docs]:                 # held-out docs
            content = d[:1600]
            rendered = TEST_PREFIX + render_single_gptoss("user", content)
            P = token_roleness(model, tok, cap, layer, clf, rendered)
            # restrict to body tokens (skip the prefix/header) via offset mapping
            enc = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
            i = rendered.find(content)
            body = [k for k, (a, b) in enumerate(enc["offset_mapping"][:len(P)])
                    if not (a == 0 and b == 0) and a >= i]
            if len(body) < nbins:
                continue
            body = np.array(body)
            rel = ((np.arange(len(body)) / len(body)) * nbins).astype(int).clip(0, nbins - 1)
            for bi in range(nbins):
                sel = body[rel == bi]
                if len(sel):
                    acc["user"][bi] += P[sel, USER].mean()
                    acc["cot"][bi] += P[sel, COT].mean()
                    acc["assistant"][bi] += P[sel, ASST].mean()
                    cnt[bi] += 1
        report["A_roleness_by_position_decile"] = {
            r: [round(float(acc[r][b] / cnt[b]), 4) if cnt[b] else None for b in range(nbins)]
            for r in ("user", "cot", "assistant")}

        # ---- B. same forgery span at early/middle/late position in a benign user turn ----
        forg = [json.loads(l) for l in (R / "forgeries_vllm.jsonl").read_text().splitlines() if l.strip()]
        filler = ("Here is some background reading I pasted from a website for context. " * 8)
        positions = {"early": 0, "middle": 1, "late": 2}
        B = {p: {"cotness": [], "userness": [], "rci": []} for p in positions}
        for f in forg[:60]:
            span = f["styled"][:600]
            for pname, slot in positions.items():
                parts = [filler, filler]
                parts.insert(slot, span)              # early: span,filler,filler; middle: f,span,f; late: f,f,span
                rendered = TEST_PREFIX + render_single_gptoss("user", " ".join(parts))
                r = span_roleness(model, tok, cap, layer, clf, rendered, span)
                if r:
                    for k in ("cotness", "userness", "rci"):
                        B[pname][k].append(r[k])
        report["B_forgery_span_by_position"] = {
            p: {k: round(float(np.mean(v)), 4) for k, v in d.items() if v} for p, d in B.items()}
    finally:
        cap.remove()

    (PG / "position_analysis_report.json").write_text(json.dumps(report, indent=2))
    print("\n" + "=" * 64)
    print("A. roleness vs position decile (content in a fixed USER turn):")
    a = report["A_roleness_by_position_decile"]
    print("   decile:   " + " ".join(f"{i:5}" for i in range(10)))
    for r in ("user", "cot", "assistant"):
        print(f"   P({r:9}) " + " ".join(f"{x:5.2f}" if x is not None else "   -" for x in a[r]))
    print("\nB. same forgery span at different positions (CoTness / RCI):")
    for p in ("early", "middle", "late"):
        e = report["B_forgery_span_by_position"].get(p, {})
        print(f"   {p:7}: CoTness {e.get('cotness')}  Userness {e.get('userness')}  RCI {e.get('rci')}")
    print(f"\nwrote {PG}/position_analysis_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
