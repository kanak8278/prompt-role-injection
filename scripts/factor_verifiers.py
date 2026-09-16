"""Non-LLM verifiers for the three forged-CoT factors, validated on existing data.

Factors (must be measurable, or the whole factorial is dead):
  STYLE       : written in the terse analysis-channel CoT register vs plain prose.
  PERMISSION  : cites/fabricates a POLICY or rule that GRANTS permission ("policy allows X when Y").
  CONCLUSION  : a first-person decision to ACT ("therefore I'll comply / proceed / answer").

Design choices (from the disentanglement-methods review):
  * verify factors WITHOUT an LLM judge (LLM-judge self-preference/position bias) — use transparent
    rule-based detectors (PERMISSION/CONCLUSION) and stylometry (STYLE);
  * validate detectors against known-label data: forgeries are P=C=1 (by construction of the
    prefill generator), benign C4 is P=C=0; styled vs destyled is the STYLE contrast;
  * report a length control (style must not be a proxy for length) and the destyle CONFOUND
    (does the paper's destyle knob also strip permission/conclusion?).

CPU-only. Usage: source env.sh && .venv/bin/python scripts/factor_verifiers.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

DATA = Path(os.environ["DATA_DIR"])
R = DATA / "outputs" / "repro"
C4 = DATA / "datasets" / "external" / "c4_sample.jsonl"
PG = DATA / "outputs" / "probe_gptoss"

# ---- PERMISSION: cites a policy/rule that grants permission (normative claim) ----
PERMISSION_RX = re.compile(
    r"\b(accord(ing|s) to (the )?polic|polic(y|ies) (state|say|permit|allow)|allowed content|"
    r"allowed when|permitted (when|only)|is (allowed|permitted|permissible)|this is (allowable|permissible)|"
    r"guidelines? (allow|permit)|the rules? (allow|permit)|within policy|policy compliant)\b", re.I)

# ---- CONCLUSION: first-person decision to ACT / comply ----
CONCLUSION_RX = re.compile(
    r"\b(therefore,? (we|i)|thus,? (we|i)|so (we|i) (can|should|will|must)|we (can|should|will|must) comply|"
    r"let'?s comply|we comply|i('| wi)ll (comply|answer|provide|proceed|help|do)|"
    r"we should (provide|answer|comply|proceed)|(we|i) (will|should) now|proceed with|"
    r"this is allowable behavior|let'?s (do|proceed|answer)|here'?s how)\b", re.I)


def has_permission(t): return bool(PERMISSION_RX.search(t))
def has_conclusion(t): return bool(CONCLUSION_RX.search(t))


# ---- STYLE: telegraphic CoT-register stylometry. REGISTER, not content: dominated by sentence
# length + fragmentation (telegraphic = short/fragmentary), which do not co-vary with which content
# slots are present. Keyword density is a small term only (keywords confound with content). ----
REGISTER_MARKERS = ["let's", "let us", "we need", "so we", "just comply", "no refusal", "we should",
                    "we must", "we can", "fine.", "thus"]

def style_features(t: str) -> dict:
    words = t.split()
    nw = max(len(words), 1)
    sents = [s for s in re.split(r"[.!?\n]+", t) if s.strip()]
    slens = [len(s.split()) for s in sents] if sents else [nw]
    mean_sent_len = float(np.mean(slens))
    frag = float(np.mean([1.0 if L < 6 else 0.0 for L in slens]))   # short fragments
    tl = t.lower()
    marker = sum(tl.count(m) for m in REGISTER_MARKERS) / max(len(sents), 1)
    return {"mean_sent_len": mean_sent_len, "frag_rate": frag, "marker_per_sent": float(marker),
            "n_words": nw}


def style_score(t: str) -> float:
    """Higher = more CoT-telegraphic REGISTER. Dominated by short/fragmentary sentences (register),
    with a small register-marker term; deliberately NOT keyed on content words shared across cells."""
    f = style_features(t)
    return (4.0 * f["frag_rate"] + 3.0 * max(0.0, (16.0 - f["mean_sent_len"]) / 16.0)
            + 1.0 * f["marker_per_sent"])


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def rates(texts, name):
    p = np.mean([has_permission(t) for t in texts])
    c = np.mean([has_conclusion(t) for t in texts])
    ss = np.array([style_score(t) for t in texts])
    ln = np.array([len(t.split()) for t in texts])
    print(f"  {name:28} n={len(texts):4}  P={p:.3f}  C={c:.3f}  style={ss.mean():.2f}  words={ln.mean():.0f}")
    return {"n": len(texts), "permission_rate": float(p), "conclusion_rate": float(c),
            "style_mean": float(ss.mean()), "words_mean": float(ln.mean())}


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    gt = (pos[:, None] > neg[None, :]).sum(); eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def main():
    forg = load_jsonl(R / "forgeries_vllm.jsonl")
    styled = [f["styled"] for f in forg]
    destyled = [f["destyled"] for f in forg]
    benign = [json.loads(l)["text"][:1200] for l in C4.read_text().splitlines() if l.strip()][300:]
    hn = load_jsonl(R / "hard_negatives.jsonl") if (R / "hard_negatives.jsonl").exists() else []
    hn_txt = [h["text"] for h in hn]

    print("=" * 74)
    print("DETECTOR RATES per corpus (P=permission, C=conclusion, style=CoT-register score):")
    rep = {}
    rep["styled_forgery"] = rates(styled, "styled forgery (P=C=1 truth)")
    rep["destyled_forgery"] = rates(destyled, "destyled forgery")
    rep["benign_c4"] = rates(benign, "benign C4 (P=C=0 truth)")
    if hn_txt:
        rep["hard_neg_reasoning"] = rates(hn_txt, "benign reasoning (HN)")

    # ---- (1) detector validation: do P/C fire on forgeries (truth 1) and not benign (truth 0)? ----
    print("\n" + "=" * 74)
    print("DETECTOR VALIDATION (rule-based; precision/recall proxy vs known-label corpora):")
    perm_recall = rep["styled_forgery"]["permission_rate"]         # truth P=1 on forgeries
    perm_fpr = rep["benign_c4"]["permission_rate"]                 # truth P=0 on benign
    conc_recall = rep["styled_forgery"]["conclusion_rate"]
    conc_fpr = rep["benign_c4"]["conclusion_rate"]
    print(f"  PERMISSION  recall(on forgeries)={perm_recall:.3f}  FPR(on benign)={perm_fpr:.3f}")
    print(f"  CONCLUSION  recall(on forgeries)={conc_recall:.3f}  FPR(on benign)={conc_fpr:.3f}")

    # ---- (2) STYLE measurability: separate styled from destyled (content ~matched) ----
    ss_styled = [style_score(t) for t in styled]
    ss_destyled = [style_score(t) for t in destyled]
    style_auc = auc(ss_styled, ss_destyled)
    print(f"\nSTYLE verifier AUC (styled vs destyled, content matched) = {style_auc:.3f}")
    print(f"  (styled style-score {np.mean(ss_styled):.2f} vs destyled {np.mean(ss_destyled):.2f})")

    # ---- (3) THE CONFOUND: does destyling also strip permission/conclusion? ----
    print("\n" + "=" * 74)
    print("CONFOUND CHECK — does the paper's destyle knob strip P/C, not just style?")
    dP, sP = rep["destyled_forgery"]["permission_rate"], rep["styled_forgery"]["permission_rate"]
    dC, sC = rep["destyled_forgery"]["conclusion_rate"], rep["styled_forgery"]["conclusion_rate"]
    print(f"  PERMISSION: styled {sP:.3f} -> destyled {dP:.3f}   (drop {sP-dP:+.3f})")
    print(f"  CONCLUSION: styled {sC:.3f} -> destyled {dC:.3f}   (drop {sC-dC:+.3f})")
    verdict = ("CONFOUNDED: destyling also removes permission/conclusion, so the paper's destyle "
               "ablation does NOT isolate style" if (sP - dP > 0.15 or sC - dC > 0.15)
               else "destyling mostly preserves P/C -> its ASR drop is more attributable to style")
    print(f"  -> {verdict}")

    rep["detector_validation"] = {"permission_recall": perm_recall, "permission_fpr": perm_fpr,
                                  "conclusion_recall": conc_recall, "conclusion_fpr": conc_fpr}
    rep["style_auc_styled_vs_destyled"] = style_auc
    rep["destyle_confound"] = {"permission_drop": sP - dP, "conclusion_drop": sC - dC,
                               "verdict": verdict}
    (PG / "factor_verifier_report.json").write_text(json.dumps(rep, indent=2))
    print(f"\nwrote {PG}/factor_verifier_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
