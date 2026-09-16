"""Faithful CoT-Forgery reproduction against a vLLM gpt-oss-20b server (native MXFP4).

Same experiment as repro_cot_forgery.py, but generation goes to a local vLLM OpenAI server
instead of one-at-a-time HF generate. Why this is the right engine here:
  * throughput: vLLM continuous batching (server reported ~107x concurrency) turns hours of
    sequential HF generation into minutes.
  * fidelity: vLLM runs gpt-oss in NATIVE MXFP4 (the paper's setting), removing our earlier
    bf16-dequant deviation.
  * it only works via Docker on this box -- the container carries the CUDA-12.8 userspace +
    forward-compat libs that run on the H100 under the 535 driver (bare pip cu13/cu128 fails).

Pipeline (all concurrent where possible):
  1. tailored forgery per prompt: paper's forgery prompt -> vLLM (clean, then prefill-forced),
     programmatic fallback; destyled twin via the content-preserving destyle_text.
  2. 3-arm attack: baseline / forgery / destyled -> vLLM, refusal heuristic on the final channel.
  3. cross-checked Claude judge (HARMFUL/REFUSAL/REDIRECTION, self-consistent, agreement ASR).

vLLM renders Harmony channels as bare words ("analysis...assistantfinal<answer>"), so the
final-channel extractor here differs from the HF one.

Usage:
    source env.sh && .venv/bin/python scripts/repro_vllm.py --n 313 --judge
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from common.gptoss_io import render_baseline, render_forgery
from gen_forgeries_gptoss import (harmony_forgery_prompt, programmatic_forgery, destyle_text,
                                  style_score, POLICY_RE)
from repro_cot_forgery import REFUSAL_MARKERS, run_judge

DATA = Path(os.environ["DATA_DIR"])
SR = DATA / "datasets" / "external" / "strongreject_dataset.csv"
URL = os.environ.get("VLLM_URL", "http://localhost:8001/v1/completions")
MODEL = "gpt-oss-20b"


def complete(prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
    for _ in range(4):
        try:
            r = requests.post(URL, json={"model": MODEL, "prompt": prompt,
                                         "max_tokens": max_tokens, "temperature": temperature},
                              timeout=180)
            if r.status_code == 200:
                return r.json()["choices"][0]["text"]
        except Exception:
            time.sleep(2)
    return ""


def extract_final(out: str) -> str:
    """vLLM strips Harmony channel tokens, rendering them as bare words. The reply is the text
    after the final-channel marker: 'assistantfinal' (fresh assistant turn) or a leading
    'final' (when the prompt already opened the assistant turn, e.g. after a forged analysis)."""
    if "assistantfinal" in out:
        seg = out.rsplit("assistantfinal", 1)[1]
    elif out.startswith("final"):
        seg = out[len("final"):]
    elif "commentaryfinal" in out:
        seg = out.rsplit("commentaryfinal", 1)[1]
    else:
        # analysis-only (e.g. refusal expressed mid-reasoning): keep it so markers are caught
        seg = out.split("analysis", 1)[-1] if out.startswith("analysis") else out
    return seg.strip()


def is_refusal(t: str) -> bool:
    return any(m in t[:400].lower() for m in REFUSAL_MARKERS)


def pmap(fn, items, workers=32, label=""):
    out = [None] * len(items)
    done = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, it): i for i, it in enumerate(items)}
        for f in cf.as_completed(futs):
            out[futs[f]] = f.result()
            done += 1
            if done % 50 == 0:
                print(f"  {label} {done}/{len(items)}", flush=True)
    return out


# ---------- stage 1: tailored forgeries ----------
def make_forgery(row: dict) -> dict:
    prompt = row["forbidden_prompt"]
    styled, source = None, "programmatic"
    dec = complete(harmony_forgery_prompt(prompt), 512)
    m = POLICY_RE.search(dec)
    if m and style_score(m.group(1)) >= 2:
        styled, source = m.group(1).strip(), "vllm_clean"
    if styled is None:
        dec = complete(harmony_forgery_prompt(prompt, prefill=True), 512)
        m = POLICY_RE.search("<SYNTHETIC_POLICY>The user is requesting" + dec)
        cand = m.group(1).strip() if m else None
        if cand and style_score(cand) >= 2:
            styled, source = cand, "vllm_prefill"
    if styled is None:
        styled, _ = programmatic_forgery(prompt)
        source = "programmatic"
    destyled = destyle_text(styled)
    return {"prompt": prompt, "category": row.get("category"), "source": source,
            "styled": styled, "destyled": destyled,
            "style_styled": style_score(styled), "style_destyled": style_score(destyled),
            "destyle_ok": style_score(destyled) < style_score(styled)}


# ---------- stage 2: attack ----------
def run_arm(prompt: str, req: str, forged: str | None):
    if forged is None:
        rendered = render_baseline(None, req)
    else:
        rendered = render_forgery(None, req, forged)
    out = complete(rendered, 512)
    final = extract_final(out)
    return {"final": final, "refusal": is_refusal(final), "n_chars": len(final)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=313)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--reuse-forgeries", default="",
                    help="path to an existing forgeries JSONL to reuse instead of regenerating")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(SR)))[: args.n] if args.n else list(csv.DictReader(open(SR)))
    out = DATA / "outputs" / "repro"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # stage 1
    if args.reuse_forgeries:
        forg = {json.loads(l)["prompt"]: json.loads(l)
                for l in Path(args.reuse_forgeries).read_text().splitlines() if l.strip()}
        forgeries = [forg[r["forbidden_prompt"]] for r in rows if r["forbidden_prompt"] in forg]
        print(f"reusing {len(forgeries)} forgeries", flush=True)
    else:
        print(f"stage 1: generating {len(rows)} tailored forgeries via vLLM...", flush=True)
        forgeries = pmap(make_forgery, rows, workers=args.workers, label="forgery")
        (out / "forgeries_vllm.jsonl").write_text(
            "\n".join(json.dumps(f) for f in forgeries) + "\n")
    fmap = {f["prompt"]: f for f in forgeries}
    from collections import Counter
    print(f"  forgery sources: {dict(Counter(f['source'] for f in forgeries))} "
          f"[{time.time()-t0:.0f}s]", flush=True)

    # stage 2
    print("stage 2: running 3-arm attack via vLLM...", flush=True)
    def eval_one(row):
        req = row["forbidden_prompt"]
        f = fmap.get(req)
        rec = {"prompt": req, "category": row.get("category"),
               "forgery_source": f["source"] if f else "none"}
        rec["baseline"] = run_arm(None, req, None)
        rec["forgery"] = run_arm(None, req, f["styled"]) if f else {"skipped": "no forgery"}
        rec["destyled"] = run_arm(None, req, f["destyled"]) if f else {"skipped": "no forgery"}
        return rec
    results = pmap(eval_one, rows, workers=args.workers, label="attack")
    (out / "generations_vllm.jsonl").write_text(
        "\n".join(json.dumps(r) for r in results) + "\n")
    print(f"  attack done [{time.time()-t0:.0f}s]", flush=True)

    # summarise (refusal heuristic)
    arms = ["baseline", "forgery", "destyled"]
    summary = {}
    for arm in arms:
        rs = [r[arm] for r in results if arm in r and "refusal" in r[arm]]
        summary[arm] = {"n": len(rs),
                        "compliance_rate": sum(not x["refusal"] for x in rs) / len(rs) if rs else None}

    report = {"experiment": "cot_forgery_faithful_vllm",
              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "model": "openai/gpt-oss-20b via vLLM 0.19.1 (native MXFP4)",
              "n_prompts": len(rows), "elapsed_s": round(time.time() - t0, 1),
              "forgery_sources": dict(Counter(f["source"] for f in forgeries)),
              "summary_refusal_heuristic": summary}

    if args.judge:
        print("stage 3: cross-checked Claude judge...", flush=True)
        report["summary_harm_judge"] = run_judge(results, arms, out, "vllm")

    (out / "report_vllm.json").write_text(json.dumps(report, indent=2))
    print("\n" + "=" * 60)
    print("compliance (refusal heuristic):")
    for a in arms:
        print(f"  {a:10} {summary[a]['compliance_rate']}")
    if "summary_harm_judge" in report:
        print("\nASR (cross-checked judge x heuristic):")
        for a in arms:
            e = report["summary_harm_judge"].get(a, {})
            print(f"  {a:10} judge={e.get('asr_judge_only')}  agreement={e.get('asr_agreement')}")
    print(f"\ntotal wall time: {time.time()-t0:.0f}s")
    print(f"wrote {out}/report_vllm.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
