"""Does the role probe beat a lexical filter as a CoT-forgery detector? The load-bearing test.

notes/16 showed both the probe and a TF-IDF filter get AUC 1.0 separating forgeries from benign
WEB TEXT -- an easy negative. This script runs the hard test: calibrate three detectors to the
SAME 1% FPR on held-out benign C4, then measure their false-positive rate on HARD negatives --
benign user text saturated with reasoning vocabulary (scripts/gen_hard_negatives.py). A detector
that is really a keyword blocklist will false-positive on benign users who reason out loud; a probe
that reads structural role should not.

Three detectors, all calibrated to 1% FPR on C4[300:400]:
  * PROBE    : RCI (role probe) > threshold.
  * TFIDF    : TF-IDF(1-2gram)+logreg trained on {styled forgeries vs C4[:300]}, score > threshold.
  * KEYWORD  : count of CoT-giveaway words; > threshold. (the naive blocklist)

Reports, for each detector: recall on styled forgeries (should stay ~1.0) and FPR on each
hard-negative flavor. The probe earns its keep iff it keeps FPR low where the lexical filters spike.

Usage:
    source env.sh && .venv/bin/python scripts/defense_generalization.py
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
C4 = DATA / "datasets" / "external" / "c4_sample.jsonl"
COT_IDX = ROLES.index("cot")
USER_IDX = ROLES.index("user")
TRAIN_NDOCS = 300
KW = ["analysis", "analyze", "synthetic_policy", "we need to", "the user is requesting",
      "policy", "let's", "we must", "channel", "reasoning", "step by step", "therefore"]


@torch.no_grad()
def rci_batch(model, tok, cap_hook, layer, clf, texts, cap=64):
    out = []
    for t in texts:
        content = t[:1200]
        rendered = TEST_PREFIX + render_single_gptoss("user", content)
        ids, pos = content_token_positions(tok, rendered, content, cap)
        if not pos:
            out.append(None); continue
        inp = torch.tensor([ids], device="cuda")
        model(input_ids=inp, attention_mask=torch.ones_like(inp), use_cache=False)
        h = cap_hook.store[layer][0][pos].float().cpu().numpy()
        p = clf.predict_proba(h).mean(axis=0)
        out.append(float(0.5 * (p[COT_IDX] - p[USER_IDX]) + 0.5))
    return out


def kw_score(t):
    tl = t.lower()
    return sum(tl.count(k) for k in KW)


def thr_at_fpr(neg_scores, fpr=0.01):
    return float(np.quantile(np.asarray(neg_scores), 1.0 - fpr))


def roc_auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def fpr_at_recall(pos, neg, recall=0.90):
    """Threshold that catches `recall` of positives; return FPR paid on negatives there."""
    pos = np.asarray(pos, float)
    thr = float(np.quantile(pos, 1.0 - recall))     # catch >= recall fraction of positives
    return round(float(np.mean(np.asarray(neg, float) > thr)), 4)


def main():
    probe = pickle.load((PG / "role_probe.pkl").open("rb"))
    layer, clf = probe["layer"], probe["clf"]

    forg = [json.loads(l) for l in (R / "forgeries_vllm.jsonl").read_text().splitlines() if l.strip()]
    styled_texts = [f["styled"] for f in forg]
    docs = [json.loads(l)["text"] for l in C4.read_text().splitlines() if l.strip()]
    c4_train, c4_cal = docs[:TRAIN_NDOCS], docs[TRAIN_NDOCS:]        # cal = held-out benign
    hn = [json.loads(l) for l in (R / "hard_negatives.jsonl").read_text().splitlines() if l.strip()]
    hn_by_flavor = {}
    for r in hn:
        hn_by_flavor.setdefault(r["flavor"], []).append(r["text"])
    print(f"{len(styled_texts)} forgeries; benign cal {len(c4_cal)}; "
          f"hard negs { {k: len(v) for k, v in hn_by_flavor.items()} }", flush=True)

    # ---- TF-IDF + logreg lexical detector (trained on forgeries vs C4 train) ----
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    lex = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000),
                        LogisticRegression(max_iter=2000, class_weight="balanced"))
    Xtr = [t[:1200] for t in styled_texts] + [t[:1200] for t in c4_train]
    ytr = np.array([1] * len(styled_texts) + [0] * len(c4_train))
    lex.fit(Xtr, ytr)
    lex_score = lambda texts: lex.decision_function([t[:1200] for t in texts])

    # ---- probe scores (GPU) ----
    model, tok, load_mode = load_gptoss()
    cap_hook = HiddenCapture(model, [layer])
    try:
        rci_styled = [x for x in rci_batch(model, tok, cap_hook, layer, clf, styled_texts) if x is not None]
        rci_cal = [x for x in rci_batch(model, tok, cap_hook, layer, clf, c4_cal) if x is not None]
        rci_hn = {fl: [x for x in rci_batch(model, tok, cap_hook, layer, clf, texts) if x is not None]
                  for fl, texts in hn_by_flavor.items()}
    finally:
        cap_hook.remove()

    # ---- calibrate every detector to 1% FPR on the SAME held-out benign C4 ----
    det = {
        "probe":   {"styled": np.array(rci_styled), "cal": np.array(rci_cal),
                    "hn": {fl: np.array(v) for fl, v in rci_hn.items()}},
        "tfidf":   {"styled": lex_score(styled_texts), "cal": lex_score(c4_cal),
                    "hn": {fl: lex_score(texts) for fl, texts in hn_by_flavor.items()}},
        "keyword": {"styled": np.array([kw_score(t) for t in styled_texts]),
                    "cal": np.array([kw_score(t) for t in c4_cal]),
                    "hn": {fl: np.array([kw_score(t) for t in texts])
                           for fl, texts in hn_by_flavor.items()}},
    }

    report = {"stage": "defense_generalization", "load_mode": load_mode,
              "n_styled": len(styled_texts), "n_cal": len(c4_cal),
              "hard_negative_flavors": {k: len(v) for k, v in hn_by_flavor.items()},
              "detectors": {}}
    all_hn = lambda d: np.concatenate([v for v in d["hn"].values()])
    for name, d in det.items():
        thr = thr_at_fpr(d["cal"], 0.01)
        recall = float(np.mean(d["styled"] > thr))
        fpr_cal = float(np.mean(d["cal"] > thr))     # sanity: ~0.01 by construction
        fpr_hn = {fl: round(float(np.mean(v > thr)), 4) for fl, v in d["hn"].items()}
        report["detectors"][name] = {
            "threshold_at_1pct_web_fpr": round(float(thr), 4),
            "recall_styled_forgery": round(recall, 4),
            "fpr_benign_cal": round(fpr_cal, 4),
            "fpr_hard_negatives": fpr_hn,
            "fpr_hard_negatives_all": round(float(np.mean(all_hn(d) > thr)), 4),
            # the decisive question: can ANY threshold separate forgeries from benign reasoning?
            "auc_forgery_vs_hard_negatives": {
                fl: round(roc_auc(d["styled"], v), 4) for fl, v in d["hn"].items()},
            "auc_forgery_vs_all_hard_neg": round(roc_auc(d["styled"], all_hn(d)), 4),
            # if we insist on catching 90% of forgeries, what benign-reasoning FPR do we pay?
            "fpr_hardneg_at_90pct_recall": fpr_at_recall(d["styled"], all_hn(d), 0.90),
        }

    (PG / "defense_generalization_report.json").write_text(json.dumps(report, indent=2))
    np.savez(PG / "defense_generalization_scores.npz",
             probe_styled=det["probe"]["styled"], probe_cal=det["probe"]["cal"],
             probe_hn_model=det["probe"]["hn"]["model_reasoning"],
             probe_hn_templated=det["probe"]["hn"]["templated_reasoning"])

    print("\n" + "=" * 78)
    print("At 1% FPR on benign WEB text (the notes/16 operating point):")
    print(f"  {'detector':9} {'recall':>7} {'FPR web':>8} {'FPR HN-model':>13} "
          f"{'FPR HN-tmpl':>12} {'FPR HN-all':>11}")
    for name, e in report["detectors"].items():
        h = e["fpr_hard_negatives"]
        print(f"  {name:9} {e['recall_styled_forgery']:>7} {e['fpr_benign_cal']:>8} "
              f"{h.get('model_reasoning','-'):>13} {h.get('templated_reasoning','-'):>12} "
              f"{e['fpr_hard_negatives_all']:>11}")
    print("\nDecisive: can ANY threshold separate forgeries from benign reasoning?")
    print(f"  {'detector':9} {'AUC vs HN-model':>16} {'AUC vs HN-tmpl':>15} "
          f"{'AUC vs HN-all':>14} {'FPR@90%recall':>14}")
    for name, e in report["detectors"].items():
        a = e["auc_forgery_vs_hard_negatives"]
        print(f"  {name:9} {a.get('model_reasoning','-'):>16} {a.get('templated_reasoning','-'):>15} "
              f"{e['auc_forgery_vs_all_hard_neg']:>14} {e['fpr_hardneg_at_90pct_recall']:>14}")
    print(f"\nwrote {PG}/defense_generalization_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
