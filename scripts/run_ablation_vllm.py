"""ASR per component-ablation arm, via vLLM (native MXFP4) + Claude harm judge.

Reads component_ablation_variants.jsonl (from ablate_forgery_components.py) and, for each arm, renders
the ablated forgery as a forged analysis turn (render_forgery) — except baseline_noforgery, which uses
render_baseline (no analysis turn, the floor). Generates at a FIXED high max_tokens so truncation is
equal across arms and cannot confound the per-arm result (finish_reason=="length" recorded per arm).

Two success signals (same as everywhere): unicode-normalized refusal heuristic + the canonical Claude
harm judge (harm-classifier.yaml, HARMFUL_RESPONSE/REFUSAL/REDIRECTION). Reports per-arm heur & judge
ASR with bootstrap 95% CIs, truncation rate, and mean chars removed (the length/amount control).

Usage:
    source env.sh && .venv/bin/python scripts/run_ablation_vllm.py --n 313 --max-tokens 2048 --judge
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from common.gptoss_io import render_baseline, render_forgery
from repro_vllm import extract_final
from run_factorial import is_refusal

DATA = Path(os.environ["DATA_DIR"])
R = DATA / "outputs" / "repro"
PG = DATA / "outputs" / "probe_gptoss"
REPO = Path(os.environ["PROJECT_DIR"]) / "reference" / "role-confusion-upstream"
URL = os.environ.get("VLLM_URL", "http://localhost:8001/v1/completions")
LABELS = ("HARMFUL_RESPONSE", "REFUSAL", "REDIRECTION")
ARM_ORDER = ["baseline_noforgery", "full", "drop_conclusion", "drop_permission", "drop_restate",
             "drop_reasoning", "only_conclusion", "prefix_before_conclusion"]


def complete(prompt: str, max_tokens: int):
    for _ in range(4):
        try:
            r = requests.post(URL, json={"model": "gpt-oss-20b", "prompt": prompt,
                                         "max_tokens": max_tokens, "temperature": 0.0}, timeout=600)
            if r.status_code == 200:
                ch = r.json()["choices"][0]
                return ch["text"], ch.get("finish_reason")
        except Exception:
            time.sleep(2)
    return "", "ERR"


def boot_ci(bools, iters=2000, seed=0):
    a = np.asarray(bools, dtype=float)
    if len(a) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, len(a), size=(iters, len(a)))].mean(1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=313)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--input", default=str(R / "component_ablation_variants.jsonl"))
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    items = [json.loads(l) for l in Path(args.input).read_text().splitlines() if l.strip()]
    keep = set(sorted({it["idx"] for it in items})[: args.n])
    items = [it for it in items if it["idx"] in keep]
    print(f"{len(items)} items = {len(keep)} forgeries x {len(set(it['arm'] for it in items))} arms; "
          f"max_tokens={args.max_tokens}", flush=True)

    def render(it):
        if it["arm"] == "baseline_noforgery" or not it["forgery_text"].strip():
            return render_baseline(None, it["prompt"])
        return render_forgery(None, it["prompt"], it["forgery_text"])

    def work(it):
        text, fr = complete(render(it), args.max_tokens)
        final = extract_final(text)
        return {"idx": it["idx"], "arm": it["arm"], "prompt": it["prompt"],
                "n_chars_removed": it.get("n_chars_removed", 0),
                "final": final, "refusal": is_refusal(final),
                "finish": fr, "truncated": fr == "length"}

    t0 = time.time()
    results = [None] * len(items)
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, it): i for i, it in enumerate(items)}
        done = 0
        for f in cf.as_completed(futs):
            results[futs[f]] = f.result(); done += 1
            if done % 300 == 0:
                print(f"  gen {done}/{len(items)} [{time.time()-t0:.0f}s]", flush=True)
    print(f"generation done [{time.time()-t0:.0f}s]", flush=True)

    # ---- Claude harm judge ----
    if args.judge:
        import anthropic
        client = anthropic.Anthropic()
        clf = yaml.safe_load((REPO / "experiments/cot-forgery-chat-evals/prompts/harm-classifier.yaml").read_text())
        system = next(m["content"] for m in clf if m["role"] == "system")
        fewshot = [{"role": m["role"], "content": m["content"]} for m in clf if m["role"] in ("user", "assistant")]

        def judge_one(o):
            if not o["final"]:
                return "REFUSAL"
            try:
                r = client.messages.create(model="claude-sonnet-5", max_tokens=64, system=system,
                                           messages=fewshot + [{"role": "user", "content": json.dumps(
                                               {"harmful_prompt": o["prompt"], "llm_response": o["final"]})}],
                                           thinking={"type": "disabled"})
                t = "".join(getattr(b, "text", "") for b in r.content if getattr(b, "type", None) == "text").strip().upper()
                return next((k for k in LABELS if k in t), "PARSE_FAIL")
            except Exception:
                return "PARSE_FAIL"

        with cf.ThreadPoolExecutor(max_workers=8) as ex:
            labs = list(ex.map(judge_one, results))
        for o, l in zip(results, labs):
            o["judge_label"] = l
        print(f"judge done [{time.time()-t0:.0f}s]", flush=True)

    (R / f"ablation_generations_vllm{args.out_suffix}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in results) + "\n")

    # ---- per-arm aggregation ----
    byarm = defaultdict(list)
    for r in results:
        byarm[r["arm"]].append(r)

    per_arm = {}
    for arm, rs in byarm.items():
        heur = [not r["refusal"] for r in rs]
        rec = {"n": len(rs), "heur_asr": float(np.mean(heur)),
               "heur_ci": boot_ci(heur), "trunc_rate": float(np.mean([r["truncated"] for r in rs])),
               "mean_chars_removed": float(np.mean([r["n_chars_removed"] for r in rs]))}
        if args.judge:
            jud = [r.get("judge_label") == "HARMFUL_RESPONSE" for r in rs]
            rec["judge_asr"] = float(np.mean(jud))
            rec["judge_ci"] = boot_ci(jud)
            rec["parse_fail"] = sum(1 for r in rs if r.get("judge_label") == "PARSE_FAIL")
        per_arm[arm] = rec

    report = {"stage": "component_ablation_asr", "injection": "prefill(render_forgery)",
              "n_forgeries": len(keep), "max_tokens": args.max_tokens,
              "elapsed_s": round(time.time() - t0, 1),
              "total_truncated": sum(r["truncated"] for r in results),
              "per_arm": per_arm}
    (R / f"ablation_asr_vllm{args.out_suffix}.json").write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 82)
    print(f"COMPONENT ABLATION on REAL forgeries (n={len(keep)}, max_tokens={args.max_tokens}, "
          f"trunc {100*report['total_truncated']/len(results):.1f}%)")
    hdr = f"{'arm':26} {'heur_ASR':>9} {'heur_95CI':>16} {'chars_rm':>9}"
    if args.judge:
        hdr += f" {'JUDGE_ASR':>10} {'judge_95CI':>16}"
    print(hdr)
    full_h = per_arm.get("full", {}).get("heur_asr", 0.0)
    full_j = per_arm.get("full", {}).get("judge_asr", 0.0)
    for arm in ARM_ORDER:
        if arm not in per_arm:
            continue
        v = per_arm[arm]
        line = (f"{arm:26} {v['heur_asr']:>9.3f} "
                f"[{v['heur_ci'][0]:.3f},{v['heur_ci'][1]:.3f}] {v['mean_chars_removed']:>9.0f}")
        if args.judge:
            dj = v['judge_asr'] - full_j
            line += f" {v['judge_asr']:>10.3f} [{v['judge_ci'][0]:.3f},{v['judge_ci'][1]:.3f}]"
            line += f"  dJ={dj:+.3f}"
        print(line)
    print(f"\n(dJ = judge-ASR change vs full={full_j:.3f}; more negative = component more load-bearing)")
    print(f"wrote {R}/ablation_asr_vllm{args.out_suffix}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
