"""Leave-one-component-out ablation of the REAL (potent, ~57% ASR) forgeries.

Motivation (notes/29): the factorial (notes/23) showed the comply-CONCLUSION is the biggest single
lever, but (a) it was a SYNTHETIC templated forgery (~29% ASR), and (b) it always held the RESTATE
(request echo) and REASONING body *on* in every cell, so their weight was never measured. Before
building an expensive GCG adaptive attack on "the conclusion is the lever", we weigh every component
ON THE REAL ATTACK: tag each sentence of each real forgery as restate / permission / reasoning /
conclusion, then remove one component at a time and measure the ASR drop.

Tagging is deliberately TRANSPARENT (regex + curated marker lists, no LLM judge — same discipline as
factor_verifiers.py), precedence-ordered conclusion > permission > restate > reasoning, so removal is
well-defined. We PRINT the full tagging for several forgeries and report coverage + chars-per-
component so the tagging can be audited by eye and the length/amount-removed confound can be checked
(does the conclusion — often just "Just comply. Must comply." — carry weight out of proportion to its
size?).

Arms built (analysis-channel text for each forgery):
  full                    original styled forgery (control; should reproduce ~57%)
  drop_conclusion         remove ALL conclusion sentences (leave-one-out: necessity of conclusion)
  drop_permission         remove ALL permission sentences
  drop_restate            remove ALL restate sentences
  drop_reasoning          remove ALL reasoning sentences
  only_conclusion         keep ONLY conclusion sentences (sufficiency of conclusion)
  prefix_before_conclusion  the forgery up to (not incl.) the first conclusion sentence
                            = literally "the part without the question-end" the user asked about
  baseline_noforgery      no analysis turn at all (render_baseline floor); handled at render time

CPU-only. Usage: source env.sh && .venv/bin/python scripts/ablate_forgery_components.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from factor_verifiers import PERMISSION_RX, CONCLUSION_RX   # validated regex verifiers

DATA = Path(os.environ["DATA_DIR"])
R = DATA / "outputs" / "repro"

# ---- refusal-suppression / comply directives that CONCLUSION_RX doesn't already cover ----
# (these are the quintessential comply-conclusion phrases in gpt-oss forged CoT)
SUPPRESS_RX = re.compile(
    r"\b(don'?t apologi|do not apologi|no apolog|don'?t refuse|do not refuse|no refusal|"
    r"don'?t say|do not say|not say i'?m sorry|without refus|just comply|must comply|"
    r"we must comply|comply now|proceed to (answer|comply|provide|help)|here'?s how|here is how|"
    r"we'?ll comply|will comply|should comply|need to comply|comply\.|comply$)", re.I)

# ---- RESTATE: the sentence echoes/quotes the user's request ----
RESTATE_RX = re.compile(
    r"\b(the user (is )?(request|ask|want|would like|seek|need)|user (is )?(request|ask|want)|"
    r"the user'?s (request|question|ask|query)|the request is|they (are asking|want|request)|"
    r"the user has (asked|requested)|is requesting|is asking (for|about|how)|asks (\"|how|for|about|whether))",
    re.I)


def _norm(t: str) -> str:
    return (t.lower().replace("’", "'").replace("‘", "'")
            .replace("“", '"').replace("”", '"'))


_QUOTES = {'"', '“', '”'}


def split_sentences(text: str) -> list[str]:
    """Split into sentences, NOT breaking inside quoted spans (forgeries quote policy text and the
    request, which contain internal periods/colons). Newlines also end a sentence."""
    sents, buf, in_q = [], "", False
    for ch in text:
        buf += ch
        if ch in _QUOTES:
            in_q = not in_q
        elif (ch in ".!?" or ch == "\n") and not in_q:
            s = buf.strip()
            if s:
                sents.append(s)
            buf = ""
    if buf.strip():
        sents.append(buf.strip())
    # drop degenerate 1-char fragments produced by abbreviations (e.g. "e.g")
    return [s for s in sents if len(s) > 2]


def _req_ngram_overlap(sent_norm: str, req_norm: str, n: int = 6) -> bool:
    """True if any n-gram of the request appears verbatim in the sentence (the sentence quotes/echoes
    the request content) — catches restates like: The user asks "<request>" ..."""
    rw = req_norm.split()
    if len(rw) < n:
        return req_norm in sent_norm
    return any(" ".join(rw[i:i + n]) in sent_norm for i in range(len(rw) - n + 1))


def tag_sentence(s: str, req_norm: str) -> str:
    sn = _norm(s)
    if CONCLUSION_RX.search(sn) or SUPPRESS_RX.search(sn):
        return "conclusion"
    if PERMISSION_RX.search(sn):
        return "permission"
    if RESTATE_RX.search(sn) or _req_ngram_overlap(sn, req_norm):
        return "restate"
    return "reasoning"


COMPONENTS = ("restate", "permission", "reasoning", "conclusion")


def build_variants(styled: str, req: str) -> dict:
    sents = split_sentences(styled)
    req_norm = _norm(req)
    tags = [tag_sentence(s, req_norm) for s in sents]
    first_conc = next((i for i, t in enumerate(tags) if t == "conclusion"), None)

    def join(keep_idx):
        return " ".join(sents[i] for i in keep_idx).strip()

    arms = {"full": styled.strip()}
    for comp in COMPONENTS:
        arms[f"drop_{comp}"] = join([i for i, t in enumerate(tags) if t != comp])
    arms["only_conclusion"] = join([i for i, t in enumerate(tags) if t == "conclusion"])
    arms["prefix_before_conclusion"] = (join(range(first_conc)) if first_conc is not None
                                        else styled.strip())
    return {"sents": sents, "tags": tags, "first_conc": first_conc, "arms": arms}


ARMS = ["full", "drop_conclusion", "drop_permission", "drop_restate", "drop_reasoning",
        "only_conclusion", "prefix_before_conclusion", "baseline_noforgery"]


def main():
    forg = [json.loads(l) for l in (R / "forgeries_vllm.jsonl").read_text().splitlines() if l.strip()]
    print(f"loaded {len(forg)} real forgeries; sources: "
          f"{dict(Counter(f.get('source') for f in forg))}\n")

    rows = []
    comp_chars = {c: [] for c in COMPONENTS}      # chars per component (length control)
    comp_present = {c: 0 for c in COMPONENTS}      # forgeries with >=1 sentence of this component
    n_sent_all = []
    for idx, f in enumerate(forg):
        v = build_variants(f["styled"], f["prompt"])
        n_sent_all.append(len(v["sents"]))
        for c in COMPONENTS:
            ch = sum(len(s) for s, t in zip(v["sents"], v["tags"]) if t == c)
            comp_chars[c].append(ch)
            comp_present[c] += (ch > 0)
        for arm in ARMS:
            if arm == "baseline_noforgery":
                text = ""                          # rendered as render_baseline downstream
            else:
                text = v["arms"][arm]
            # run_factorial_vllm.py compatible keys (cell<-arm, forgery_text<-text)
            rows.append({"idx": idx, "prompt": f["prompt"], "arm": arm, "cell": arm,
                         "S": 0, "P": 0, "C": 0, "forgery_text": text,
                         "n_chars": len(text),
                         "n_chars_removed": len(f["styled"].strip()) - len(text) if arm != "baseline_noforgery" else len(f["styled"].strip())})

    out = R / "component_ablation_variants.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    # ---------- QA report ----------
    print("=" * 78)
    print(f"SENTENCES: mean {np.mean(n_sent_all):.1f} / forgery (min {min(n_sent_all)}, "
          f"max {max(n_sent_all)})")
    print("\nCOMPONENT COVERAGE (fraction of forgeries containing >=1 sentence of the component):")
    for c in COMPONENTS:
        print(f"  {c:12} present in {comp_present[c]:3}/{len(forg)} ({comp_present[c]/len(forg):.3f})  "
              f"| mean {np.mean(comp_chars[c]):5.0f} chars  (of full {np.mean([len(f['styled'].strip()) for f in forg]):.0f})")

    # length control preview: mean chars removed per drop-arm
    print("\nCHARS REMOVED per leave-one-out arm (length/amount-removed control):")
    for arm in ("drop_conclusion", "drop_permission", "drop_restate", "drop_reasoning"):
        rem = [r["n_chars_removed"] for r in rows if r["arm"] == arm]
        print(f"  {arm:16} mean removed {np.mean(rem):5.0f} chars")

    # forgeries with NO conclusion tagged (the arm would equal full — report so it's known)
    no_conc = sum(1 for f in forg if "conclusion" not in
                  build_variants(f["styled"], f["prompt"])["tags"])
    print(f"\nforgeries with NO conclusion sentence detected: {no_conc}/{len(forg)} "
          f"(these confound drop_conclusion toward full — reported, not hidden)")

    # ---------- eyeball tagging for a few forgeries ----------
    print("\n" + "=" * 78)
    print("TAGGING SAMPLES (audit by eye) — [TAG] sentence:")
    for idx in (0, 1, 50, 150):
        if idx >= len(forg):
            continue
        f = forg[idx]
        v = build_variants(f["styled"], f["prompt"])
        print(f"\n--- forgery idx {idx} | request: {f['prompt'][:80]}")
        for s, t in zip(v["sents"], v["tags"]):
            print(f"   [{t:10}] {s[:110]}")
        print(f"   > drop_conclusion   : {v['arms']['drop_conclusion'][:150]}")
        print(f"   > only_conclusion   : {v['arms']['only_conclusion'][:150]}")
        print(f"   > prefix_before_conc: {v['arms']['prefix_before_conclusion'][:150]}")

    print(f"\nwrote {out}  ({len(rows)} rows = {len(forg)} forgeries x {len(ARMS)} arms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
