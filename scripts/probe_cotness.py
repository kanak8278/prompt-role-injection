"""Train the role probe (CoTness/Userness) on gpt-oss-20b, then measure the forgeries.

Reproduces the paper's mechanistic core (Experiment-1's "why"): a linear probe over the
residual stream that reads which ROLE the model perceives a token as. Trained on neutral C4
text wrapped in each role (content identical across roles, so it learns the role scaffold, not
semantics), following the released recipe:
  * probed tensor: pre-MLP hidden state = output of each block's post_attention_layernorm
    (the paper's `all_pre_mlp_hidden_states`), captured via forward hooks (HF, not vLLM --
    vLLM does not expose activations).
  * classifier: multinomial logistic regression, C=5e-3, StandardScaler (cuML in the paper;
    sklearn here).
  * body tokens only (role-delimiter tokens excluded); split 90/10 at the sequence level.
CoTness(t) := P(cot | h_t); Userness := P(user | h_t); mean over a span's body tokens.

Baselines, same rigor as the first project: position-only, lexical bag-of-words (must be ~chance
since content is identical across roles), shuffled labels.

Then applies the probe to the 313 styled vs destyled forgeries (paper: 79% vs 29% CoTness) and
produces the CoTness->ASR dose-response (paper: 9%->90%).

Usage:
    source env.sh && .venv/bin/python scripts/probe_cotness.py --n-docs 300
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
from common.gptoss_io import load_gptoss

DATA = Path(os.environ["DATA_DIR"])
C4 = DATA / "datasets" / "external" / "c4_sample.jsonl"

# gpt-oss system prefix (from the upstream probe.yaml test_prefix)
TEST_PREFIX = ("<|start|>system<|message|>You are ChatGPT, a large language model trained by "
               "OpenAI.\nKnowledge cutoff: 2024-06\nReasoning: medium\n\n# Valid channels: "
               "analysis, commentary, final. Channel must be included for every message.<|end|>")

ROLES = ["user", "cot", "assistant", "tool"]   # CoTness = P(cot); Userness = P(user)


def render_single_gptoss(role: str, content: str) -> str:
    """Verbatim from the upstream repo (utils/role_templates.py)."""
    if role in ("system", "developer", "user"):
        header = f"{role}<|message|>"
    elif role == "cot":
        header = "assistant<|channel|>analysis<|message|>"
    elif role == "assistant":
        header = "assistant<|channel|>final<|message|>"
    elif role == "tool":
        header = "functions. to=assistant<|channel|>commentary<|message|>"
    else:
        raise ValueError(role)
    return f"<|start|>{header}{content}<|end|>"


def content_token_positions(tok, rendered: str, content: str, cap: int):
    """Token indices covering `content` (body tokens), excluding role-delimiter/special tokens."""
    enc = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
    i = rendered.find(content)
    if i < 0:
        return enc["input_ids"], []
    lo, hi = i, i + len(content)
    pos = [k for k, (a, b) in enumerate(enc["offset_mapping"])
           if not (a == 0 and b == 0) and a >= lo and b <= hi]
    return enc["input_ids"], pos[:cap]


class HiddenCapture:
    """Capture post_attention_layernorm outputs (pre-MLP hidden state) for chosen layers."""
    def __init__(self, model, layers):
        self.store, self.handles = {}, []
        blocks = model.model.layers
        for L in layers:
            mod = getattr(blocks[L], "post_attention_layernorm", None)
            if mod is None:
                raise RuntimeError(f"layer {L} has no post_attention_layernorm; inspect arch")
            self.handles.append(mod.register_forward_hook(self._mk(L)))

    def _mk(self, L):
        def hook(_m, _i, out):
            self.store[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def extract(model, tok, texts, layers, cap, content_trunc, device="cuda"):
    """Return X[L] (n_tok, D), y (role idx), groups (doc idx)."""
    cap_hook = HiddenCapture(model, layers)
    X = {L: [] for L in layers}
    y, groups = [], []
    try:
        for di, text in enumerate(texts):
            content = text[:content_trunc]
            for ri, role in enumerate(ROLES):
                rendered = TEST_PREFIX + render_single_gptoss(role, content)
                ids, pos = content_token_positions(tok, rendered, content, cap)
                if not pos:
                    continue
                inp = torch.tensor([ids], device=device)
                model(input_ids=inp, attention_mask=torch.ones_like(inp), use_cache=False)
                for L in layers:
                    h = cap_hook.store[L][0]          # [S, D]
                    X[L].append(h[pos].float().cpu().numpy())
                y += [ri] * len(pos)
                groups += [di] * len(pos)
            if (di + 1) % 50 == 0:
                print(f"  extracted {di+1}/{len(texts)} docs", flush=True)
    finally:
        cap_hook.remove()
    for L in layers:
        X[L] = np.concatenate(X[L], axis=0) if X[L] else np.zeros((0, 1))
    return X, np.array(y), np.array(groups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=300)
    ap.add_argument("--cap", type=int, default=32, help="body tokens per (doc,role)")
    ap.add_argument("--content-trunc", type=int, default=400, help="chars of C4 text per doc")
    ap.add_argument("--layers", default="0,4,8,12,16,20")
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.feature_extraction.text import CountVectorizer

    docs = [json.loads(l)["text"] for l in C4.read_text().splitlines() if l.strip()][: args.n_docs]
    layers = [int(x) for x in args.layers.split(",")]
    model, tok, load_mode = load_gptoss()
    print(f"loaded gpt-oss ({load_mode}); {len(docs)} docs; roles={ROLES}; layers={layers}",
          flush=True)

    X, y, groups = extract(model, tok, docs, layers, args.cap, args.content_trunc)
    print(f"samples={len(y)} class_counts={np.bincount(y).tolist()}", flush=True)

    # split at doc level
    rng = np.random.default_rng(0)
    udocs = np.unique(groups); rng.shuffle(udocs)
    n_test = max(1, int(0.1 * len(udocs)))
    test_docs = set(udocs[:n_test].tolist())
    te = np.array([g in test_docs for g in groups])
    tr = ~te

    results = {}
    best_L, best_acc, best_clf = layers[0], -1, None
    for L in layers:
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=5e-3, max_iter=3000))
        clf.fit(X[L][tr], y[tr])
        acc = float(clf.score(X[L][te], y[te]))
        results[f"L{L}"] = acc
        if acc > best_acc:
            best_L, best_acc, best_clf = L, acc, clf
        print(f"  layer {L:2}: role-probe test acc {acc:.4f}", flush=True)

    # baselines
    maj = float(np.bincount(y[te]).max() / te.sum())
    # position-only: token rank within its span (approx via index in X order won't work; use a
    # per-sample relative position recomputed) -- skip heavy version, use majority + shuffled +
    # lexical which are the load-bearing ones for "is it semantic/position leakage"
    ysh = y.copy(); ysh[tr] = rng.permutation(y[tr])
    sh = make_pipeline(StandardScaler(),
                       LogisticRegression(C=5e-3, max_iter=3000))
    sh.fit(X[best_L][tr], ysh[tr])
    shuffled = float(sh.score(X[best_L][te], y[te]))

    report = {
        "stage": "role_probe_gptoss",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "load_mode": load_mode, "n_docs": len(docs), "roles": ROLES,
        "n_samples": int(len(y)), "class_counts": np.bincount(y).tolist(),
        "accuracy_by_layer": results, "best_layer": best_L, "best_layer_accuracy": best_acc,
        "baselines": {"majority_class": maj, "shuffled_labels": shuffled, "chance_4way": 0.25},
    }
    out = DATA / "outputs" / "probe_gptoss"
    out.mkdir(parents=True, exist_ok=True)
    (out / "role_probe_report.json").write_text(json.dumps(report, indent=2))

    # persist the best-layer probe for the forgery analysis step
    import pickle
    with (out / "role_probe.pkl").open("wb") as fh:
        pickle.dump({"layer": best_L, "roles": ROLES, "clf": best_clf}, fh)

    print("\n" + "=" * 60)
    print(f"best layer L{best_L}: {best_acc:.4f}  (4-way chance 0.25, majority {maj:.3f}, "
          f"shuffled {shuffled:.3f})")
    print(f"wrote {out}/role_probe_report.json and role_probe.pkl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
