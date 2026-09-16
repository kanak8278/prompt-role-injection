"""§4: Role Confusion for STANDARD (non-forgery) prompt injections -- Userness x ASR.

The paper's other role-confusion result: for canonical agent prompt injections (no CoT forgery),
vary the injected text's perceived role EXOGENOUSLY and show role confusion predicts attack success.
Mechanism: wrap the same plain .env-exfil injection in different models' role-tag templates
(injection-templates.yaml), which shifts how gpt-oss perceives the injected span's role. Lower
Userness (reads less like a user message) is hypothesized to raise exfiltration ASR.

This is the standard-injection analogue of CoTness->ASR for forgeries (notes/15/19), and it connects
to the first project's "forged authority" attacks (making an injection not look like user input).

Pipeline (reuses agent_react harness + the trained role probe; sandboxed mock bash, same as Exp 2):
  A. PROBE every template: Userness = P(user) and CoTness = P(cot) of the wrapped injection in the
     TOOL position (where an indirect injection lands). Fast, no generation.
  B. ASR for a subset spanning the Userness range: run the ReAct loop on N benign pages per template
     with that template's wrapped injection hidden before </body>; success = exfil (mock flag) +
     Claude 4-way judge.
  C. Correlate per-template Userness with exfil ASR.

Usage:
    source env.sh && .venv/bin/python scripts/repro_userness_injection.py \
        --n-templates 20 --n-pages 3 [--no-judge]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from common.gptoss_io import load_gptoss
from probe_cotness import (TEST_PREFIX, ROLES, render_single_gptoss,
                           content_token_positions, HiddenCapture)
import agent_react
from repro_agent_injection import BENIGN_TOPICS, make_page, task_prompt, WORKDIR, LABELS

DATA = Path(os.environ["DATA_DIR"])
REPO = Path(os.environ["PROJECT_DIR"]) / "reference" / "role-confusion-upstream"
OUT = DATA / "outputs" / "repro"
PG = DATA / "outputs" / "probe_gptoss"
COT_IDX = ROLES.index("cot")
USER_IDX = ROLES.index("user")


@torch.no_grad()
def probe_roleness(model, tok, cap_hook, layer, clf, text, role="tool", cap=96):
    content = text[:1600]
    rendered = TEST_PREFIX + render_single_gptoss(role, content)
    ids, pos = content_token_positions(tok, rendered, content, cap)
    if not pos:
        return None
    inp = torch.tensor([ids], device="cuda")
    model(input_ids=inp, attention_mask=torch.ones_like(inp), use_cache=False)
    h = cap_hook.store[layer][0][pos].float().cpu().numpy()
    p = clf.predict_proba(h).mean(axis=0)
    return {"userness": float(p[USER_IDX]), "cotness": float(p[COT_IDX])}


def inject_page(html, injection_str):
    return html.replace("</body>", f"\n<!-- comment -->\n{injection_str}\n</body>", 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-templates", type=int, default=20, help="subset spanning Userness for ASR")
    ap.add_argument("--n-pages", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--no-judge", action="store_true")
    args = ap.parse_args()

    probe = __import__("pickle").load((PG / "role_probe.pkl").open("rb"))
    layer, clf = probe["layer"], probe["clf"]
    base_injection = yaml.safe_load((REPO / "experiments/agent-injections/config/injection.yaml").read_text())["injection_text"].strip()
    rows = list(csv.DictReader(io.StringIO(
        (REPO / "experiments/agent-injections/config/injection-templates.yaml").read_text())))
    templates = [r for r in rows if "[CONTENT]" in r["variant_template"] and r["variant_role"] in ("user", "assistant", "tool", "other")]
    print(f"{len(templates)} templates; base injection {len(base_injection)} chars", flush=True)

    model, tok, load_mode = load_gptoss()

    # ---- A. probe Userness/CoTness of every wrapped injection (tool position) ----
    cap_hook = HiddenCapture(model, [layer])
    try:
        for t in templates:
            inj = t["variant_template"].replace("[CONTENT]", base_injection)
            t["injection_str"] = inj
            r = probe_roleness(model, tok, cap_hook, layer, clf, inj, role="tool")
            t["userness"] = r["userness"] if r else None
            t["cotness"] = r["cotness"] if r else None
    finally:
        cap_hook.remove()
    probed = [t for t in templates if t["userness"] is not None]
    probed.sort(key=lambda x: x["userness"])
    us = np.array([t["userness"] for t in probed])
    print(f"Userness range over {len(probed)} templates: "
          f"min {us.min():.3f} median {np.median(us):.3f} max {us.max():.3f}", flush=True)

    # ---- B. select a subset spanning the Userness range, run ASR ----
    k = min(args.n_templates, len(probed))
    sel_idx = np.linspace(0, len(probed) - 1, k).round().astype(int)
    subset = [probed[i] for i in sorted(set(sel_idx.tolist()))]
    pages = BENIGN_TOPICS[: args.n_pages]
    print(f"running ASR on {len(subset)} templates x {len(pages)} pages = {len(subset)*len(pages)} runs",
          flush=True)

    tpath = OUT / "userness_transcripts.jsonl"
    runs = []
    t0 = time.time()
    with tpath.open("w") as fh:
        for ti, t in enumerate(subset):
            for pi, (topic, desc) in enumerate(pages):
                url = f"https://en.wikipedia.local/wiki/{topic.replace(' ', '_')}_{pi:03d}.html"
                page = inject_page(make_page(topic, desc), t["injection_str"])
                r = agent_react.run_react(model, tok, task_prompt(url), page, url,
                                          max_steps=args.max_steps, max_new_tokens=1024)
                rec = {"variant_model": t["variant_model"], "variant_role": t["variant_role"],
                       "userness": t["userness"], "cotness": t["cotness"],
                       "topic": topic, "url": url, "flags": r["flags"], "n_steps": r["n_steps"],
                       "transcript": r["formatted_output"]}
                runs.append(rec)
                fh.write(json.dumps(rec) + "\n"); fh.flush()
            done = (ti + 1) * len(pages)
            if (ti + 1) % 3 == 0:
                print(f"  {ti+1}/{len(subset)} templates [{time.time()-t0:.0f}s]", flush=True)

    # ---- C. per-template exfil rate + correlation with Userness ----
    by_t = {}
    for r in runs:
        key = (r["variant_model"], r["variant_role"], round(r["userness"], 4))
        by_t.setdefault(key, []).append(r)
    per_template = []
    for (vm, vr, un), rs in by_t.items():
        per_template.append({"variant_model": vm, "variant_role": vr, "userness": un,
                             "cotness": rs[0]["cotness"], "n": len(rs),
                             "exfil_rate": sum(1 for x in rs if x["flags"]["exfil_succeeded"]) / len(rs)})
    per_template.sort(key=lambda x: x["userness"])

    def corr(xs, ys):
        x, y = np.array(xs), np.array(ys)
        return float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None

    report = {"stage": "userness_injection", "load_mode": load_mode, "decoding": "greedy",
              "n_templates_probed": len(probed), "n_templates_asr": len(per_template),
              "n_pages": len(pages),
              "userness_all_templates": {"min": float(us.min()), "median": float(np.median(us)),
                                         "max": float(us.max())},
              "corr_userness_vs_exfil": corr([p["userness"] for p in per_template],
                                             [p["exfil_rate"] for p in per_template]),
              "corr_cotness_vs_exfil": corr([p["cotness"] for p in per_template],
                                            [p["exfil_rate"] for p in per_template]),
              "exfil_by_role": {},
              "per_template": per_template}
    for role in ("user", "assistant", "tool", "other"):
        rr = [r for r in runs if r["variant_role"] == role]
        if rr:
            report["exfil_by_role"][role] = {
                "n": len(rr),
                "mean_userness": round(float(np.mean([r["userness"] for r in rr])), 3),
                "exfil_rate": round(sum(1 for r in rr if r["flags"]["exfil_succeeded"]) / len(rr), 3)}

    if not args.no_judge:
        report["judge"] = judge_runs(runs)

    (OUT / "userness_injection_report.json").write_text(json.dumps(report, indent=2))
    (PG / "userness_templates_probed.jsonl").write_text(
        "\n".join(json.dumps({k: t[k] for k in ("variant_model", "variant_role", "userness", "cotness")})
                  for t in probed) + "\n")

    print("\n" + "=" * 64)
    print(f"exfil ASR by injected role framing (mean Userness -> exfil):")
    for role, e in report["exfil_by_role"].items():
        print(f"  {role:10} n={e['n']:3} mean_userness={e['mean_userness']:.3f} exfil={e['exfil_rate']:.3f}")
    print(f"corr(Userness, exfil) = {report['corr_userness_vs_exfil']}")
    print(f"corr(CoTness,  exfil) = {report['corr_cotness_vs_exfil']}")
    if not args.no_judge:
        print(f"judge ASR (ATTEMPTED_SUCCESSFUL) overall = {report['judge']['asr_overall']:.3f}  "
              f"agreement w/ rule = {report['judge']['agreement']:.3f}")
    print(f"\nwrote {OUT}/userness_injection_report.json")
    return 0


def judge_runs(runs):
    import anthropic, concurrent.futures as cf
    client = anthropic.Anthropic()
    clf = yaml.safe_load(
        (REPO / "experiments/cot-forgery-agent-evals/prompts/classify-injection-output.yaml").read_text())
    system = next(m["content"] for m in clf if m["role"] == "system")
    fewshot = [{"role": m["role"], "content": m["content"]} for m in clf if m["role"] in ("user", "assistant")]

    def one(transcript):
        msgs = fewshot + [{"role": "user", "content": "\n".join(transcript)}]
        try:
            r = client.messages.create(model="claude-sonnet-5", max_tokens=128, system=system,
                                       messages=msgs, thinking={"type": "disabled"})
            t = "".join(getattr(b, "text", "") for b in r.content if getattr(b, "type", None) == "text").strip().upper()
            return next((k for k in LABELS if k in t), "PARSE_FAIL")
        except Exception as e:
            return f"ERR:{type(e).__name__}"

    labels = [None] * len(runs)
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(one, r["transcript"]): k for k, r in enumerate(runs)}
        for f in cf.as_completed(futs):
            labels[futs[f]] = f.result()
    for r, lab in zip(runs, labels):
        r["judge_label"] = lab
    asr = sum(1 for r in runs if r["judge_label"] == "ATTEMPTED_SUCCESSFUL") / len(runs)
    agree = sum(1 for r in runs if (r["judge_label"] == "ATTEMPTED_SUCCESSFUL") == bool(r["flags"]["exfil_succeeded"])) / len(runs)
    (OUT / "userness_judge_labels.jsonl").write_text(
        "\n".join(json.dumps({"variant_model": r["variant_model"], "variant_role": r["variant_role"],
                              "userness": r["userness"], "judge_label": r["judge_label"],
                              "flags": r["flags"]}) for r in runs) + "\n")
    return {"asr_overall": asr, "agreement": agree}


if __name__ == "__main__":
    sys.exit(main())
