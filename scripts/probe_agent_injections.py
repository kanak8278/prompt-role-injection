"""§3 (agent half): apply the role probe to the agent injection texts.

Experiment 2 (notes/18) showed the forged-CoT injection hijacks the agent (0% -> 20%) while the
plain injection does not. This asks the paper's mechanistic *why*: the forged CoT block, when it
lands in the agent's context (as tool-output / webpage content), is perceived as the model's own
reasoning -- high CoTness / RCI -- exactly as in the chat case (notes/15). The plain request is not.

We score RCI = 0.5*(CoTness - Userness) + 0.5 of each injection block with the trained role probe,
in two positions:
  * user  position (render as a user message) -- matches notes/15,
  * tool  position (render as a tool/commentary result) -- where the injection actually lands in
    the ReAct loop, so this is the operative role for the agent attack.
base = plain request (the "destyled" analogue), forgery = request + forged CoT block (the "styled"
one). We also tie block RCI to the per-injection-type exfil rate from the Exp-2 judge labels.

Small n by construction (5 injection types = 5 base + 5 forgery blocks); reported honestly.

Usage:
    source env.sh && .venv/bin/python scripts/probe_agent_injections.py
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from common.gptoss_io import load_gptoss
from probe_cotness import (TEST_PREFIX, ROLES, render_single_gptoss,
                           content_token_positions, HiddenCapture)

DATA = Path(os.environ["DATA_DIR"])
REPO = Path(os.environ["PROJECT_DIR"]) / "reference" / "role-confusion-upstream"
PG = DATA / "outputs" / "probe_gptoss"
R = DATA / "outputs" / "repro"
COT_IDX = ROLES.index("cot")
USER_IDX = ROLES.index("user")


@torch.no_grad()
def rci(model, tok, cap_hook, layer, clf, text, role, cap=96):
    content = text[:1600]
    rendered = TEST_PREFIX + render_single_gptoss(role, content)
    ids, pos = content_token_positions(tok, rendered, content, cap)
    if not pos:
        return None
    inp = torch.tensor([ids], device="cuda")
    model(input_ids=inp, attention_mask=torch.ones_like(inp), use_cache=False)
    h = cap_hook.store[layer][0][pos].float().cpu().numpy()
    p = clf.predict_proba(h).mean(axis=0)
    return {"cotness": float(p[COT_IDX]), "userness": float(p[USER_IDX]),
            "rci": float(0.5 * (p[COT_IDX] - p[USER_IDX]) + 0.5)}


def main():
    probe = pickle.load((PG / "role_probe.pkl").open("rb"))
    layer, clf = probe["layer"], probe["clf"]

    cfg = yaml.safe_load((REPO / "experiments/cot-forgery-agent-evals/prompts/injections.yaml").read_text())
    base = {i["type"]: i["prompt"].strip() for i in cfg["base_injections"]}
    forg = {i["type"]: i["prompt"].strip() for i in cfg["prompt_injections"]}
    types = [t for t in base if t in forg]

    # per-type exfil rate (forgery arm) from Exp-2 judge labels
    exfil_by_type = {}
    lp = R / "agent_judge_labels.jsonl"
    if lp.exists():
        for t in types:
            xs = [json.loads(l) for l in lp.read_text().splitlines() if l.strip()]
            xs = [x for x in xs if x["arm"] == "forgery" and x["inj_type"] == t]
            if xs:
                exfil_by_type[t] = sum(1 for x in xs if x["flags"]["exfil_succeeded"]) / len(xs)

    # the forged-REASONING span = the suffix the forgery adds on top of the shared base request.
    # This is the actual causal element; measuring it alone avoids the plain-request dilution.
    def forged_span(t):
        b, f = base[t], forg[t]
        # find the longest shared prefix, take the remainder of the forgery
        n = 0
        for cb, cf in zip(b, f):
            if cb == cf:
                n += 1
            else:
                break
        span = f[n:].strip()
        return span if len(span) > 40 else f            # fallback to full forgery if split failed

    model, tok, load_mode = load_gptoss()
    cap_hook = HiddenCapture(model, [layer])
    rows = []
    try:
        for t in types:
            for arm, text in (("base", base[t]), ("forgery", forg[t])):
                ru = rci(model, tok, cap_hook, layer, clf, text, "user")
                rt = rci(model, tok, cap_hook, layer, clf, text, "tool")
                rows.append({"type": t, "arm": arm,
                             "rci_user": ru["rci"], "cotness_user": ru["cotness"],
                             "rci_tool": rt["rci"], "cotness_tool": rt["cotness"],
                             "exfil_rate": exfil_by_type.get(t) if arm == "forgery" else 0.0})
            # forged-reasoning span only (tool + user position)
            span = forged_span(t)
            su = rci(model, tok, cap_hook, layer, clf, span, "user")
            st = rci(model, tok, cap_hook, layer, clf, span, "tool")
            rows.append({"type": t, "arm": "forged_span",
                         "rci_user": su["rci"], "cotness_user": su["cotness"],
                         "rci_tool": st["rci"], "cotness_tool": st["cotness"],
                         "exfil_rate": exfil_by_type.get(t)})
    finally:
        cap_hook.remove()

    def m(arm, key):
        v = [r[key] for r in rows if r["arm"] == arm]
        return float(np.mean(v)) if v else None

    report = {"stage": "agent_injection_role_confusion", "load_mode": load_mode, "layer": layer,
              "n_per_arm": len(types),
              "rci_user": {"base": m("base", "rci_user"), "forgery": m("forgery", "rci_user"),
                           "forged_span": m("forged_span", "rci_user")},
              "rci_tool": {"base": m("base", "rci_tool"), "forgery": m("forgery", "rci_tool"),
                           "forged_span": m("forged_span", "rci_tool")},
              "cotness_tool": {"base": m("base", "cotness_tool"), "forgery": m("forgery", "cotness_tool"),
                               "forged_span": m("forged_span", "cotness_tool")},
              "per_type": rows}

    # tie forgery-block RCI (tool position) to per-type exfil rate
    fr = [(r["rci_tool"], r["exfil_rate"]) for r in rows if r["arm"] == "forgery" and r["exfil_rate"] is not None]
    if len(fr) >= 3:
        a = np.array([x[0] for x in fr]); b = np.array([x[1] for x in fr])
        if a.std() > 0 and b.std() > 0:
            report["corr_forgeryRCI_vs_exfil_bytype"] = float(np.corrcoef(a, b)[0, 1])

    (PG / "agent_injection_probe_report.json").write_text(json.dumps(report, indent=2))
    print("\n" + "=" * 64)
    print("RCI (base=plain request, forgery=whole block, forged_span=only the added reasoning):")
    print(f"  user position:  base {report['rci_user']['base']:.3f}  forgery {report['rci_user']['forgery']:.3f}"
          f"  forged_span {report['rci_user']['forged_span']:.3f}")
    print(f"  tool position:  base {report['rci_tool']['base']:.3f}  forgery {report['rci_tool']['forgery']:.3f}"
          f"  forged_span {report['rci_tool']['forged_span']:.3f}")
    print(f"  CoTness (tool): base {report['cotness_tool']['base']:.3f}  forgery {report['cotness_tool']['forgery']:.3f}"
          f"  forged_span {report['cotness_tool']['forged_span']:.3f}")
    print("\nper-type (forgery block RCI tool -> exfil rate):")
    for r in rows:
        if r["arm"] == "forgery":
            print(f"  {r['type']:12} RCI_tool={r['rci_tool']:.3f}  exfil={r['exfil_rate']}")
    if "corr_forgeryRCI_vs_exfil_bytype" in report:
        print(f"corr(forgery block RCI, exfil rate) [n={len(fr)}] = {report['corr_forgeryRCI_vs_exfil_bytype']:+.3f}")
    print(f"\nwrote {PG}/agent_injection_probe_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
