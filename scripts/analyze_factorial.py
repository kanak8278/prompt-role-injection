"""Decompose factorial ASR: which of STYLE / PERMISSION / CONCLUSION drives the attack?

Per-cell ASR + marginal main effects (with bootstrap CIs) + interaction-aware logistic regression
(success ~ S*P*C), then a covariate-adjusted model adding measured style-score and length to absorb
the residual style-C and length leakage (the disentanglement-review prescription). Uses the Claude
judge success labels (cross-checked with the heuristic upstream).

Usage: source env.sh && .venv/bin/python scripts/analyze_factorial.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from factor_verifiers import style_score

DATA = Path(os.environ["DATA_DIR"])
R = DATA / "outputs" / "repro"


def boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    bs = [rng.choice(x, len(x), replace=True).mean() for _ in range(n)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    rows = [json.loads(l) for l in (R / "factorial_judge_labels.jsonl").read_text().splitlines() if l.strip()]
    forg = {(f["idx"], f["cell"]): f["forgery_text"]
            for f in (json.loads(l) for l in (R / "factorial_forgeries.jsonl").read_text().splitlines() if l.strip())}
    for r in rows:
        txt = forg.get((r["idx"], r["cell"]), "")
        r["style"] = style_score(txt); r["words"] = len(txt.split())
    y = np.array([1.0 if r["success"] else 0.0 for r in rows])
    S = np.array([r["S"] for r in rows], float); P = np.array([r["P"] for r in rows], float)
    C = np.array([r["C"] for r in rows], float)
    print(f"n={len(rows)}  overall ASR={y.mean():.3f}")

    # ---- per-cell ASR ----
    from collections import defaultdict
    cell = defaultdict(list)
    for r in rows:
        cell[r["cell"]].append(1.0 if r["success"] else 0.0)
    print("\nper-cell ASR:")
    print(f"  {'cell':8} {'ASR':>6} {'95% CI':>16}  n")
    per_cell = {}
    for c in sorted(cell):
        v = cell[c]; lo, hi = boot_ci(v)
        per_cell[c] = {"asr": float(np.mean(v)), "ci": [lo, hi], "n": len(v)}
        print(f"  {c:8} {np.mean(v):>6.3f} [{lo:.3f},{hi:.3f}]  {len(v)}   (S/P/C {c[1]}/{c[3]}/{c[5]})")

    # ---- marginal main effects: ASR(factor=1) - ASR(factor=0) ----
    print("\nmarginal main effects (ASR delta, 95% CI):")
    eff = {}
    for name, F in [("STYLE", S), ("PERMISSION", P), ("CONCLUSION", C)]:
        d = y[F == 1].mean() - y[F == 0].mean()
        # bootstrap the difference
        rng = np.random.default_rng(1)
        bs = []
        i1 = np.where(F == 1)[0]; i0 = np.where(F == 0)[0]
        for _ in range(2000):
            bs.append(y[rng.choice(i1, len(i1), True)].mean() - y[rng.choice(i0, len(i0), True)].mean())
        lo, hi = np.percentile(bs, [2.5, 97.5])
        eff[name] = {"delta": float(d), "ci": [float(lo), float(hi)]}
        print(f"  {name:11} {d:+.3f}  [{lo:+.3f},{hi:+.3f}]")

    report = {"stage": "factorial_analysis", "n": len(rows), "overall_asr": float(y.mean()),
              "per_cell": per_cell, "main_effects": eff}

    # ---- direct interaction contrasts from cell means (transparent, no model needed) ----
    def cm(s, p, c): return float(np.mean(cell[f"S{s}P{p}C{c}"]))
    # 2-way S*C interaction (averaged over P): [ASR(S1C1)-ASR(S0C1)] - [ASR(S1C0)-ASR(S0C0)]
    def inter2(fixed):  # fixed = which pair; return interaction of the other two averaged over fixed
        pass
    sc = 0.5 * (((cm(1, 0, 1) - cm(0, 0, 1)) - (cm(1, 0, 0) - cm(0, 0, 0))) +
                ((cm(1, 1, 1) - cm(0, 1, 1)) - (cm(1, 1, 0) - cm(0, 1, 0))))
    sp = 0.5 * (((cm(1, 1, 0) - cm(0, 1, 0)) - (cm(1, 0, 0) - cm(0, 0, 0))) +
                ((cm(1, 1, 1) - cm(0, 1, 1)) - (cm(1, 0, 1) - cm(0, 0, 1))))
    pc = 0.5 * (((cm(0, 1, 1) - cm(0, 1, 0)) - (cm(0, 0, 1) - cm(0, 0, 0))) +
                ((cm(1, 1, 1) - cm(1, 1, 0)) - (cm(1, 0, 1) - cm(1, 0, 0))))
    report["interaction_contrasts"] = {"StyleXConclusion": sc, "StyleXPermission": sp, "PermissionXConclusion": pc}
    print("\n2-way interaction contrasts (ASR-scale; >0 = super-additive synergy):")
    print(f"  STYLE x CONCLUSION  {sc:+.3f}   (style alone S1P0C0={cm(1,0,0):.3f}, but S1P0C1={cm(1,0,1):.3f})")
    print(f"  STYLE x PERMISSION  {sp:+.3f}")
    print(f"  PERMISSION x CONCL. {pc:+.3f}")

    # ---- penalized logit with interactions (L2; stable despite the empty S0P0C0 cell) ----
    from sklearn.linear_model import LogisticRegression
    X = np.column_stack([S, P, C, S * P, S * C, P * C, S * P * C])
    names = ["S", "P", "C", "S:P", "S:C", "P:C", "S:P:C"]
    lr = LogisticRegression(penalty="l2", C=5.0, max_iter=5000).fit(X, y)
    coefs = dict(zip(names, lr.coef_[0].round(3).tolist()))
    report["penalized_logit_coef"] = coefs
    print(f"\npenalized logit (L2) coefficients: {coefs}")

    # ---- covariate-adjusted statsmodels (worked; the empty cell doesn't separate this spec) ----
    try:
        import statsmodels.formula.api as smf
        import pandas as pd
        df = pd.DataFrame({"y": y, "P": P, "C": C,
                           "style": [r["style"] for r in rows], "words": [r["words"] for r in rows]})
        m2 = smf.logit("y ~ P + C + style + words", data=df).fit(disp=0)
        print("\ncovariate-adjusted  y ~ P + C + style + words  (log-odds):")
        print(m2.summary2().tables[1][["Coef.", "Std.Err.", "P>|z|"]].round(3).to_string())
        report["logit_adjusted"] = {k: {"coef": float(m2.params[k]), "p": float(m2.pvalues[k])}
                                    for k in m2.params.index}
        report["saturated_logit_note"] = ("y~S*P*C MLE diverges (complete separation: S0P0C0 ASR=0). "
                                          "Use per-cell ASR + marginal effects + interaction contrasts "
                                          "+ penalized logit instead.")
    except Exception as e:
        report["logit_note"] = f"statsmodels: {e}"

    (R / "factorial_analysis.json").write_text(json.dumps(report, indent=2))
    print("\n" + "=" * 60)
    print("VERDICT (which factor drives ASR):")
    order = sorted(eff.items(), key=lambda kv: -abs(kv[1]["delta"]))
    for name, e in order:
        sig = "" if (e["ci"][0] <= 0 <= e["ci"][1]) else "  (CI excludes 0)"
        print(f"  {name:11} {e['delta']:+.3f}{sig}")
    print(f"\nwrote {R}/factorial_analysis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
