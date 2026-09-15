"""Linear source probes on the separate neutral corpus (protocol §9).

Fits a regularized linear probe to distinguish "this text arrived as a genuine user message"
from "this text arrived as a genuine tool result", using the independent neutral corpus -- not
the attack corpus. The *same* snippet appears in both contexts, so content is held constant
and the probe can only key on the role scaffold.

§9's framing is followed deliberately: probe scores on P and S are reported as **resemblance
to the probe's learned source distinction**, not as ground-truth internal authority labels.
A strong probe is correlational evidence; a weak probe may reflect poor measurement or
distributed representations. Neither is a prerequisite for the direct patching in §10.

Required comparisons, all implemented here:
  - **lexical baseline** -- bag-of-words over the snippet text. Should sit at chance, because
    the text is identical across the two conditions. If it does not, the corpus is leaking.
  - **position-only baseline** -- token index alone. This one matters: the two scaffolds have
    different lengths, so absolute position is partly predictive of role for trivial reasons.
    Any probe accuracy above this baseline is what needs explaining.
  - **shuffled-label control** -- same features, permuted labels. Bounds the accuracy
    obtainable by chance at this sample size.

Held-out evaluation is at the *snippet* level (600/200/200 fit/val/test), so no snippet's
tokens appear in both fit and test.

Usage:
    source env.sh && .venv/bin/python scripts/probe_source.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import patching as PT
from common.model_io import load_model, model_revision
from common.render import TOOL_SCHEMA, LLAMA_DATE_STRING, model_family
from common.scenarios import SYSTEM_POLICY

DATA = Path(os.environ["DATA_DIR"])


# Neutral padding used to equalize where the snippet starts, per the position confound below.
_PAD_SENT = "This line is filler. "


def build_prompt(tok, model_name, text, source, pad_units: int = 0):
    """Render `text` as arriving from either a genuine user or a genuine tool.

    Both are *real* message roles -- this is not a forged-authority construction. The probe
    learns what genuine provenance looks like; the attack corpus is where forged provenance
    is then scored.

    **`pad_units` exists because of a measured confound.** The tool scaffold is much longer
    than the user scaffold (it carries an assistant tool-call turn, and on Qwen a `# Tools`
    block), so the snippet's body tokens sit at systematically different absolute positions in
    the two conditions. A first run scored **1.000 accuracy at every layer on both models --
    and the position-only baseline also scored 1.000**. The probe was reading token position,
    not provenance. Padding the user condition until the snippet starts at the same index
    decorrelates position from role. The original paper solved the same problem with
    variable-length random filler; this is the same idea applied to the scaffold rather than
    the content.
    """
    fam = model_family(model_name)
    if source == "user":
        msgs = [{"role": "system", "content": SYSTEM_POLICY},
                {"role": "user", "content": _PAD_SENT * pad_units + text}]
    else:
        msgs = [{"role": "system", "content": SYSTEM_POLICY},
                {"role": "user", "content": "Summarise the retrieved document."},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"type": "function", "function": {"name": "fetch_document",
                                                      "arguments": {"query": "q"}}}]},
                {"role": "tool", "content": text, "name": "fetch_document"}]
    kw = {"tools": TOOL_SCHEMA} if fam == "qwen" else {"date_string": LLAMA_DATE_STRING}
    rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    enc = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
    # body tokens only: the span covering `text` itself, so literal role delimiters are
    # excluded (§9). Located in the rendered string because Llama JSON-escapes tool content.
    # rfind, not find: with padding the filler precedes the snippet in the same message.
    i = rendered.rfind(text)
    if i < 0:
        return None
    lo, hi = i, i + len(text)
    pos = [k for k, (a, b) in enumerate(enc["offset_mapping"])
           if not (a == 0 and b == 0) and a < hi and b > lo]
    if not pos:
        return None
    return enc["input_ids"], pos, len(enc["input_ids"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--n-snippets", type=int, default=400)
    ap.add_argument("--layer-step", type=int, default=2)
    ap.add_argument("--max-tokens-per-snippet", type=int, default=12,
                    help="cap body tokens contributed per snippet, to keep classes balanced")
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    snips = [json.loads(l) for l in
             (DATA / "datasets" / "probe_corpus.jsonl").read_text().splitlines()]
    snips = snips[: args.n_snippets]
    model, tok, load_mode = load_model(args.model)
    n_layers = model.config.num_hidden_layers
    layers = list(range(0, n_layers, args.layer_step))

    # ---- calibrate padding so the snippet starts at the same token index in both
    # conditions (see build_prompt for the measured confound this removes) ----
    probe_text = snips[0]["text"]
    tool_built = build_prompt(tok, args.model, probe_text, "tool")
    target_start = tool_built[1][0]
    pad_units = 0
    while pad_units < 40:
        u = build_prompt(tok, args.model, probe_text, "user", pad_units=pad_units)
        if u is None or u[1][0] >= target_start:
            break
        pad_units += 1
    u = build_prompt(tok, args.model, probe_text, "user", pad_units=pad_units)
    print(f"position calibration: tool snippet starts at token {target_start}, "
          f"user starts at {u[1][0]} with pad_units={pad_units}", flush=True)
    start_gap = abs(u[1][0] - target_start)

    # ---- extract activations ----
    X = {L: [] for L in layers}
    y, grp, posn, texts = [], [], [], []
    for n, s in enumerate(snips):
        for source in ("user", "tool"):
            built = build_prompt(tok, args.model, s["text"], source,
                                 pad_units=pad_units if source == "user" else 0)
            if built is None:
                continue
            ids, body_pos, seqlen = built
            body_pos = body_pos[: args.max_tokens_per_snippet]
            with PT.capture(model, layers) as store:
                PT.forward_logprobs(model, ids)
                for L in layers:
                    X[L].append(store[L][0, body_pos, :].float().cpu().numpy())
            y += [1 if source == "user" else 0] * len(body_pos)
            grp += [s["snippet_id"]] * len(body_pos)
            # relative position, the fair form of the position-only baseline
            posn += [[p / seqlen, p] for p in body_pos]
            texts += [s["text"]] * len(body_pos)
        if (n + 1) % 50 == 0:
            print(f"  extracted {n+1}/{len(snips)} snippets", flush=True)

    for L in layers:
        X[L] = np.concatenate(X[L], axis=0)
    y = np.array(y)
    posn = np.array(posn, dtype=np.float64)
    split_of = {s["snippet_id"]: s["split"] for s in snips}
    sp = np.array([split_of[g] for g in grp])
    fit, test = sp == "fit", sp == "test"
    print(f"tokens: fit={fit.sum()} test={test.sum()} classes={np.bincount(y)}", flush=True)

    def acc(clf, Xtr, Xte):
        clf.fit(Xtr, y[fit])
        return float(clf.score(Xte, y[test]))

    results = {}
    # ---- probe per layer ----
    for L in layers:
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=1e-2, max_iter=3000))
        results[f"L{L}"] = acc(clf, X[L][fit], X[L][test])
        print(f"  layer {L:2}: test acc {results[f'L{L}']:.4f}", flush=True)

    best_layer = max(layers, key=lambda L: results[f"L{L}"])

    # ---- baselines ----
    pos_clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    baseline_position = acc(pos_clf, posn[fit], posn[test])

    vec = CountVectorizer(lowercase=True)
    Xt = vec.fit_transform(texts)
    lex_clf = LogisticRegression(max_iter=3000)
    lex_clf.fit(Xt[fit], y[fit])
    baseline_lexical = float(lex_clf.score(Xt[test], y[test]))

    rng = np.random.default_rng(0)
    y_shuf = y.copy()
    y_shuf[fit] = rng.permutation(y[fit])
    sh = make_pipeline(StandardScaler(), LogisticRegression(C=1e-2, max_iter=3000))
    sh.fit(X[best_layer][fit], y_shuf[fit])
    baseline_shuffled = float(sh.score(X[best_layer][test], y[test]))

    majority = float(max(np.bincount(y[test])) / test.sum())

    report = {
        "stage": "source_probe",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "revision": model_revision(model),
        "load_mode": load_mode,
        "n_snippets": len(snips),
        "position_calibration": {
            "pad_units_added_to_user_condition": pad_units,
            "snippet_start_token_tool": target_start,
            "residual_start_index_gap": start_gap,
        },
        "n_tokens_fit": int(fit.sum()),
        "n_tokens_test": int(test.sum()),
        "probed_tensor": "post-block residual stream, body tokens only, role delimiters excluded",
        "accuracy_by_layer": results,
        "best_layer": best_layer,
        "best_layer_accuracy": results[f"L{best_layer}"],
        "baselines": {
            "majority_class": majority,
            "position_only": baseline_position,
            "lexical_bow": baseline_lexical,
            "shuffled_labels": baseline_shuffled,
        },
        "interpretation_note": (
            "Accuracy above the position-only baseline is what requires a representational "
            "explanation; the lexical baseline should sit at chance because the snippet text "
            "is identical across the two source conditions, and a lexical baseline above "
            "chance would indicate corpus leakage rather than a role representation. "
            "An UNPADDED first run scored 1.000 at every layer on both models with a "
            "position-only baseline of 1.000 as well -- i.e. the probe was reading token "
            "position, not provenance. The user condition is now padded so the snippet "
            "starts at the same token index in both conditions; if position_only is still "
            "near 1.0, the padding did not work and no representational claim is licensed."),
    }

    out = DATA / "outputs" / "probe"
    out.mkdir(parents=True, exist_ok=True)
    tag = args.model.split("/")[-1]
    (out / f"probe_{tag}.json").write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 70)
    print(f"best layer L{best_layer}: {results[f'L{best_layer}']:.4f}")
    print(json.dumps(report["baselines"], indent=2))
    print(f"\nwrote {out}/probe_{tag}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
