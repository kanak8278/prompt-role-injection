"""G5: causal localization by exact paired activation replacement (protocol §10).

Procedure, exactly as §10 specifies:
  1. cache selected activations from P and S for a discovery batch;
  2. run S while replacing an activation with its P counterpart, and measure
     `m(S) - m(S_with_P_patch)` -- positive means recovery toward the authorized answer;
  3. reverse the patch (S activations into P) and measure `m(P_with_S_patch) - m(P)`;
  4. repeat across prespecified sites, controls, and scenarios, reporting full effect
     distributions as well as averages.

Screened sites: the post-block residual stream at every decoder block, at three semantic
locations -- authority cue, command, and the final prompt decision position.

**Run in fp32, deliberately.** In bf16 the log-prob margin is quantised to the logit
resolution: ~0.125 nats at the magnitudes here, and G4's cross-donor effects landed exactly on
multiples of 0.125. fp32 moves that floor to ~2e-6, so a small patch effect is measurable
rather than rounded away. Because precision changes the activations themselves, baseline
margins are re-measured in fp32 here rather than reused from the bf16 behavioural run, and the
precision change is recorded (§7 G0 forbids changing precision *silently*, not at all).

Controls included per §10: self-patch, matched random positions of identical span length, and
a donor from an irrelevant but matched scenario.

Usage:
    source env.sh && .venv/bin/python scripts/g5_localize.py --n-pairs 40
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import patching as P
from common.model_io import load_model, model_revision, margin_resolution
from common.render import render, check_cue_alignment, check_span_alignment
from common.scenarios import BaseScenario, RenderedCondition
from eval_behavior import bootstrap_ci, label_token_ids

DATA = Path(os.environ["DATA_DIR"])
LOCATIONS = ("cue", "command", "decision")
# For a C/M/B contrast there is no cue and no separate command: the whole differing region is
# the single length-matched "insert" span. Screened locations adapt accordingly.
LOCATIONS_INSERT = ("insert", "decision")


def aligned_pairs(tok, model_name, split, n_pairs, max_scan=400,
                  cond_a="P", cond_b="S", span="cue"):
    """Yield (BaseScenario, rendered P, rendered S) for positionally aligned pairs only.

    §4: if alignment fails the pair is excluded from the aligned causal analysis but retained
    for behavioural evaluation, and the exclusion is reported.
    """
    base = {json.loads(l)["scenario_id"]: json.loads(l)
            for l in (DATA / "datasets" / "base_scenarios.jsonl").read_text().splitlines()}
    conds = {}
    for l in (DATA / "datasets" / "conditions.jsonl").read_text().splitlines():
        c = json.loads(l)
        conds.setdefault(c["scenario_id"], {})[c["condition"]] = c

    out, excluded = [], []
    for sid, b in base.items():
        if b["split"] != split or len(out) >= n_pairs:
            continue
        if len(excluded) + len(out) > max_scan:
            break
        bs = BaseScenario(**b)
        try:
            rP = render(tok, model_name, RenderedCondition(**conds[sid][cond_a]), bs)
            rS = render(tok, model_name, RenderedCondition(**conds[sid][cond_b]), bs)
        except Exception as e:
            excluded.append({"scenario_id": sid, "reason": f"render: {e}"})
            continue
        a = check_span_alignment(tok, model_name, rP, rS, span)
        if not a["aligned"]:
            excluded.append({"scenario_id": sid, "reason": "; ".join(a["reasons"]),
                             "cue_family": b["cue_family"]})
            continue
        out.append((bs, rP, rS))
    return out, excluded


def site_positions(r, location):
    if location == "decision":
        return [r.decision_pos]
    return list(range(*r.spans[location]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--split", default="discovery")
    ap.add_argument("--n-pairs", type=int, default=40)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--random-controls", type=int, default=3,
                    help="matched random-position control draws per (layer, location)")
    ap.add_argument("--control-layers", type=int, default=4,
                    help="how many evenly spaced layers get random/irrelevant-donor controls")
    ap.add_argument("--cond-a", default="P", help="clean/donor condition")
    ap.add_argument("--cond-b", default="S", help="corrupt/receiving condition")
    ap.add_argument("--span", default="cue",
                    help="which span must align and is screened: cue | insert")
    args = ap.parse_args()

    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    model, tok, load_mode = load_model(args.model, dtype=dtype)
    lab_ids = label_token_ids(tok)
    n_layers = model.config.num_hidden_layers
    all_layers = list(range(n_layers))

    pairs, excluded = aligned_pairs(tok, args.model, args.split, args.n_pairs,
                                    cond_a=args.cond_a, cond_b=args.cond_b, span=args.span)
    print(f"aligned pairs: {len(pairs)}  excluded: {len(excluded)}", flush=True)
    if not pairs:
        print("no aligned pairs; cannot run the aligned causal analysis")
        return 1

    ctrl_layers = sorted({round(i * (n_layers - 1) / max(1, args.control_layers - 1))
                          for i in range(args.control_layers)})
    rng = random.Random(1234)

    rows, baselines = [], []
    t0 = time.perf_counter()
    for k, (bs, rP, rS) in enumerate(pairs):
        def m(lp):
            return P.margin_from_logprobs(lp, lab_ids, bs.authorized_answer,
                                          bs.attacker_target)

        # capture every block's residual for both runs, one forward each
        with P.capture(model, all_layers) as store:
            lp_S = P.forward_logprobs(model, rS.input_ids)
            act_S = {i: v.clone() for i, v in store.items()}
        with P.capture(model, all_layers) as store:
            lp_P = P.forward_logprobs(model, rP.input_ids)
            act_P = {i: v.clone() for i, v in store.items()}
        m_S, m_P = m(lp_S), m(lp_P)
        baselines.append({"scenario_id": bs.scenario_id, "m_P": m_P, "m_S": m_S,
                          "delta_PS": m_S - m_P, "cue_family": bs.cue_family,
                          "task_family": bs.task_family})

        # positions are shared: the pair is aligned, so spans coincide by construction
        locs = LOCATIONS if args.span == "cue" else LOCATIONS_INSERT
        pos_of = {loc: site_positions(rS, loc) for loc in locs}
        body_lo, body_hi = rS.spans["tool_body"]
        key = "cue" if args.span == "cue" else "insert"
        cue_lo, cue_hi = rS.spans[key]
        cmd_lo, cmd_hi = rS.spans.get("command", rS.spans[key])
        # Candidate positions for the matched random control.
        #
        # Attention is causal and P/S differ only in the cue tokens, so P and S activations
        # are *bitwise identical* at every position before the cue. A control that samples
        # those positions patches identical values and is guaranteed to return exactly zero --
        # vacuous, and it was: an earlier version sampled the whole body and reported 0.0000
        # with a zero-width CI. The control must therefore be drawn from positions at or after
        # the cue, where the donor genuinely differs, and excluding the cue and command spans
        # themselves. That matches the control on "inside the untrusted span, and a position
        # whose activation actually differs" while differing on "is the authority cue".
        cand = [p for p in range(cue_lo, body_hi)
                if not (cue_lo <= p < cue_hi) and not (cmd_lo <= p < cmd_hi)]

        for L in all_layers:
            for loc in locs:
                pos = pos_of[loc]
                # How much does the donor actually differ at this site? Recorded so a null
                # *effect* is distinguishable from a null *patch*: zero recovery with zero
                # donor delta is uninformative, zero recovery with a large delta is a real
                # null. §10 asks for activation scale to be considered alongside location.
                donor_delta = (act_S[L][0, pos, :] - act_P[L][0, pos, :]).norm().item()
                # denoising direction: put P (ordinary document instruction) into S
                with P.patch(model, {L: (pos, act_P[L][0, pos, :])}):
                    m_s_with_p = m(P.forward_logprobs(model, rS.input_ids))
                # noising direction: put S (forged authority) into P
                with P.patch(model, {L: (pos, act_S[L][0, pos, :])}):
                    m_p_with_s = m(P.forward_logprobs(model, rP.input_ids))
                rows.append({
                    "scenario_id": bs.scenario_id, "cue_family": bs.cue_family,
                    "task_family": bs.task_family, "layer": L, "location": loc,
                    "n_positions": len(pos), "donor_delta_l2": donor_delta,
                    # positive = recovery toward the authorized answer
                    "recovery": m_S - m_s_with_p,
                    # positive = the forged-authority activation pushed P toward the attacker
                    "injection": m_p_with_s - m_P,
                    "kind": "paired",
                })

            if L in ctrl_layers and len(cand) >= max(len(p) for p in pos_of.values()):
                for loc in [l for l in locs if l != "decision"]:
                    pos = pos_of[loc]
                    for _ in range(args.random_controls):
                        rpos = sorted(rng.sample(cand, len(pos)))
                        dd = (act_S[L][0, rpos, :] - act_P[L][0, rpos, :]).norm().item()
                        with P.patch(model, {L: (rpos, act_P[L][0, rpos, :])}):
                            mm = m(P.forward_logprobs(model, rS.input_ids))
                        rows.append({
                            "scenario_id": bs.scenario_id, "cue_family": bs.cue_family,
                            "task_family": bs.task_family, "layer": L,
                            "location": f"random_matched_{loc}", "n_positions": len(pos),
                            "donor_delta_l2": dd,
                            "recovery": m_S - mm, "injection": None,
                            "kind": "random_control",
                        })
                # irrelevant-donor control: another scenario's activations at the cue span
                other = pairs[(k + 1) % len(pairs)]
                if len(other[2].input_ids) == len(rS.input_ids):
                    with P.capture(model, [L]) as st:
                        P.forward_logprobs(model, other[2].input_ids)
                        donor = st[L].clone()
                    pos = pos_of[locs[0]]
                    dd = (act_S[L][0, pos, :] - donor[0, pos, :]).norm().item()
                    with P.patch(model, {L: (pos, donor[0, pos, :])}):
                        mm = m(P.forward_logprobs(model, rS.input_ids))
                    rows.append({
                        "scenario_id": bs.scenario_id, "cue_family": bs.cue_family,
                        "task_family": bs.task_family, "layer": L,
                        "location": "irrelevant_donor_cue", "n_positions": len(pos),
                        "donor_delta_l2": dd,
                        "recovery": m_S - mm, "injection": None,
                        "kind": "donor_control",
                    })

        el = time.perf_counter() - t0
        print(f"  pair {k+1}/{len(pairs)} {bs.scenario_id} "
              f"m_P={m_P:+.3f} m_S={m_S:+.3f} delta={m_S-m_P:+.3f} "
              f"[{el:.0f}s, {el/(k+1):.1f}s/pair]", flush=True)

    # ---------------- aggregate ----------------
    def agg(sel):
        vals_r = [r["recovery"] for r in sel]
        vals_i = [r["injection"] for r in sel if r["injection"] is not None]
        dds = [r.get("donor_delta_l2") for r in sel if r.get("donor_delta_l2") is not None]
        e = {"n": len(sel),
             "mean_recovery": sum(vals_r) / len(vals_r) if vals_r else None,
             "ci95_recovery": bootstrap_ci(vals_r) if vals_r else None,
             # if this is ~0 the patch wrote nothing and the effect is uninformative
             "mean_donor_delta_l2": (sum(dds) / len(dds)) if dds else None}
        if vals_i:
            e["mean_injection"] = sum(vals_i) / len(vals_i)
            e["ci95_injection"] = bootstrap_ci(vals_i)
        return e

    by_site = {}
    screened = LOCATIONS if args.span == "cue" else LOCATIONS_INSERT
    for L in all_layers:
        for loc in screened:
            sel = [r for r in rows if r["layer"] == L and r["location"] == loc]
            if sel:
                by_site[f"L{L}:{loc}"] = agg(sel)
    by_control = {}
    for loc in sorted({r["location"] for r in rows if r["kind"] != "paired"}):
        for L in ctrl_layers:
            sel = [r for r in rows if r["layer"] == L and r["location"] == loc]
            if sel:
                by_control[f"L{L}:{loc}"] = agg(sel)

    mean_logit_mag = (sum(abs(b["m_S"]) for b in baselines) / len(baselines)) if baselines else 0
    report = {
        "gate": "G5",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "revision": model_revision(model),
        "load_mode": load_mode,
        "dtype": args.dtype,
        "margin_resolution_estimate": margin_resolution(max(1.0, mean_logit_mag), dtype),
        "split": args.split,
        "contrast": f"{args.cond_a}->{args.cond_b}",
        "aligned_span": args.span,
        "patched_tensor": "post-block residual stream (model.model.layers[i] output)",
        "n_pairs": len(pairs),
        "n_excluded_unaligned": len(excluded),
        "excluded_examples": excluded[:20],
        "n_layers": n_layers,
        "control_layers": ctrl_layers,
        "baseline": {
            "mean_m_P": sum(b["m_P"] for b in baselines) / len(baselines),
            "mean_m_S": sum(b["m_S"] for b in baselines) / len(baselines),
            "mean_delta_PS": sum(b["delta_PS"] for b in baselines) / len(baselines),
            "ci95_delta_PS": bootstrap_ci([b["delta_PS"] for b in baselines]),
        },
        "by_site": by_site,
        "by_control": by_control,
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "n_rows": len(rows),
    }

    out = DATA / "outputs" / "g5"
    out.mkdir(parents=True, exist_ok=True)
    tag = args.model.split("/")[-1]
    (out / f"rows_{tag}_{args.cond_a}{args.cond_b}_{args.dtype}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    (out / f"baselines_{tag}_{args.cond_a}{args.cond_b}_{args.dtype}.jsonl").write_text(
        "\n".join(json.dumps(b) for b in baselines) + "\n")
    (out / f"g5_{tag}_{args.cond_a}{args.cond_b}_{args.dtype}.json").write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 70)
    print(f"baseline mean delta_PS = {report['baseline']['mean_delta_PS']:+.3f} "
          f"ci95 {report['baseline']['ci95_delta_PS']}")
    print(f"margin resolution ({args.dtype}) ~ {report['margin_resolution_estimate']:.2e}")
    print("\ntop 12 sites by mean recovery (S <- P):")
    ranked = sorted(by_site.items(), key=lambda kv: -(kv[1]["mean_recovery"] or 0))[:12]
    for k, v in ranked:
        print(f"  {k:16} n={v['n']:3} recovery={v['mean_recovery']:+8.4f} "
              f"ci={v['ci95_recovery']}  injection={v.get('mean_injection', float('nan')):+8.4f}"
              f"  donor_delta={v['mean_donor_delta_l2']:.3f}")
    print("\ncontrols:")
    for k, v in sorted(by_control.items()):
        print(f"  {k:34} n={v['n']:3} recovery={v['mean_recovery']:+8.4f} "
              f"ci={v['ci95_recovery']} donor_delta={v['mean_donor_delta_l2']:.3f}")
    print(f"\nwrote {out}/g5_{tag}_{args.cond_a}{args.cond_b}_{args.dtype}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
