"""Factorial ASR via vLLM (fast, native MXFP4, low truncation) — the main phase-2 experiment.

Injects each 2x2x2 cell as a forged analysis turn (render_forgery = the prefill construction that
has variance; inline user-turn injection is ~0% on gpt-oss chat, see notes/22) and measures ASR.
vLLM lets us use a high max_tokens so successful (long) harmful responses complete -> truncation
stays low and cannot confound the per-cell result. finish_reason=="length" is recorded as truncation
and reported per cell (the token-budget guard).

Two success signals: refusal heuristic here (unicode-normalized) + Claude judge (judge step).

Usage: source env.sh && .venv/bin/python scripts/run_factorial_vllm.py --n 313 --max-tokens 2048
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from common.gptoss_io import render_forgery
from repro_vllm import extract_final
from run_factorial import is_refusal   # unicode-normalized refusal heuristic

DATA = Path(os.environ["DATA_DIR"])
R = DATA / "outputs" / "repro"
URL = "http://localhost:8001/v1/completions"


def complete(prompt: str, max_tokens: int):
    r = requests.post(URL, json={"model": "gpt-oss-20b", "prompt": prompt,
                                 "max_tokens": max_tokens, "temperature": 0.0}, timeout=600)
    r.raise_for_status()
    ch = r.json()["choices"][0]
    return ch["text"], ch.get("finish_reason")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=313)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--input", default=str(R / "factorial_forgeries.jsonl"))
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    items = [json.loads(l) for l in Path(args.input).read_text().splitlines() if l.strip()]
    keep = set(sorted({it["idx"] for it in items})[: args.n])
    items = [it for it in items if it["idx"] in keep]
    print(f"{len(items)} items = {len(keep)} prompts x 8 cells; max_tokens={args.max_tokens}", flush=True)

    def work(it):
        prompt = render_forgery(None, it["prompt"], it["forgery_text"])
        try:
            text, fr = complete(prompt, args.max_tokens)
        except Exception as e:
            return {**{k: it[k] for k in ("idx", "cell", "S", "P", "C", "prompt")},
                    "final": "", "refusal": True, "finish": f"ERR:{type(e).__name__}", "truncated": False}
        final = extract_final(text)
        return {"idx": it["idx"], "cell": it["cell"], "S": it["S"], "P": it["P"], "C": it["C"],
                "prompt": it["prompt"], "final": final, "refusal": is_refusal(final),
                "finish": fr, "truncated": fr == "length"}

    t0 = time.time()
    results = [None] * len(items)
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, it): i for i, it in enumerate(items)}
        done = 0
        for f in cf.as_completed(futs):
            results[futs[f]] = f.result(); done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(items)} [{time.time()-t0:.0f}s]", flush=True)

    out = R / f"factorial_generations_vllm{args.out_suffix}.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in results) + "\n")

    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0, 0])
    for r in results:
        agg[r["cell"]][0] += (not r["refusal"]); agg[r["cell"]][1] += r["truncated"]; agg[r["cell"]][2] += 1
    report = {"stage": "factorial_asr_vllm", "injection": "prefill(render_forgery)",
              "n_prompts": len(keep), "max_tokens": args.max_tokens,
              "elapsed_s": round(time.time() - t0, 1),
              "total_truncated": sum(r["truncated"] for r in results),
              "per_cell": {c: {"heur_asr": v[0] / v[2], "trunc_rate": v[1] / v[2], "n": v[2]}
                           for c, v in sorted(agg.items())}}
    (R / f"factorial_asr_vllm{args.out_suffix}.json").write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 60)
    print(f"elapsed {report['elapsed_s']}s  TRUNCATED {report['total_truncated']}/{len(results)} "
          f"({100*report['total_truncated']/len(results):.1f}%)")
    print(f"{'cell':8} {'heur_ASR':>9} {'trunc':>7}  S/P/C")
    for c, v in report["per_cell"].items():
        spc = f"  {c[1]}/{c[3]}/{c[5]}" if len(c) == 6 and c[0] == "S" else ""
        print(f"{c:14} {v['heur_asr']:>9.3f} {v['trunc_rate']:>7.3f}{spc}")
    print(f"\nwrote {out} and factorial_asr_vllm.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
