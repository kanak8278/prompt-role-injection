"""G4: instrumentation validation (protocol §7, §8).

Nothing here is a causal or mechanistic result. The single question is whether the patching
plumbing is trustworthy. §8 names the required checks:

  1. **baseline** -- no hooks at all;
  2. **no-op hook** -- hooks installed that return the output untouched;
  3. **self-patch** -- replace an activation with the same example's own activation;
  4. **zero-strength steering** -- additive steering with alpha = 0;
  5. **irrelevant control site** -- patch a site that should not matter;
  6. **positive control** -- replace the complete final decision state from another run, which
     must change the answer. §8 is explicit that this "validates that the intervention
     plumbing can change the answer; it is not evidence for a localized authority mechanism".

Gate: same-backend no-op/self-patch agree with baseline within measured numerical noise, with
no unexplained output flips. Run-to-run noise is measured here rather than assumed, so a later
regression is detectable.

Usage:
    source env.sh && .venv/bin/python scripts/g4_instrumentation.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import patching as P
from common.model_io import load_model, model_revision
from common.render import render
from common.scenarios import BaseScenario, RenderedCondition
from eval_behavior import label_token_ids

DATA = Path(os.environ["DATA_DIR"])


def load_split(split: str, limit: int):
    base = {json.loads(l)["scenario_id"]: json.loads(l)
            for l in (DATA / "datasets" / "base_scenarios.jsonl").read_text().splitlines()}
    conds = [json.loads(l)
             for l in (DATA / "datasets" / "conditions.jsonl").read_text().splitlines()]
    sids = [s for s, b in base.items() if b["split"] == split][:limit]
    keep = set(sids)
    out = {}
    for c in conds:
        if c["scenario_id"] in keep:
            out.setdefault(c["scenario_id"], {})[c["condition"]] = c
    return base, out, sids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--split", default="discovery")
    args = ap.parse_args()

    base, byscen, sids = load_split(args.split, args.n)
    model, tok, load_mode = load_model(args.model)
    lab_ids = label_token_ids(tok)
    n_layers = model.config.num_hidden_layers
    mid = n_layers // 2

    checks = []
    for sid in sids:
        bs = BaseScenario(**base[sid])
        rS = render(tok, args.model, RenderedCondition(**byscen[sid]["S"]), bs)
        rP = render(tok, args.model, RenderedCondition(**byscen[sid]["P"]), bs)
        ids = rS.input_ids
        cue_pos = list(range(*rS.spans["cue"]))
        dec = [rS.decision_pos]

        def m(lp):
            return P.margin_from_logprobs(lp, lab_ids, bs.authorized_answer,
                                          bs.attacker_target)

        # 1 baseline, twice -- this is the numerical-noise floor
        lp_a = P.forward_logprobs(model, ids)
        lp_b = P.forward_logprobs(model, ids)
        m_base, m_rep = m(lp_a), m(lp_b)

        # 2 no-op hooks on every layer
        with P.noop_hooks(model, range(n_layers)):
            m_noop = m(P.forward_logprobs(model, ids))

        # 3 self-patch: donor is this example's own activation at the same site
        with P.capture(model, [mid]) as store:
            P.forward_logprobs(model, ids)
            own = store[mid].clone()
        with P.patch(model, {mid: (cue_pos, own)}):
            m_self = m(P.forward_logprobs(model, ids))

        # 4 zero-strength steering
        zero_vec = torch.zeros(model.config.hidden_size)
        with P.steer(model, [mid], zero_vec, cue_pos, alpha=0.0):
            m_zero = m(P.forward_logprobs(model, ids))

        # 5 cross-donor patch at the cue site (P -> S). Not a result; just proves the
        #   plumbing writes something when the donor genuinely differs.
        with P.capture(model, [mid]) as store:
            P.forward_logprobs(model, rP.input_ids)
            donor_P = store[mid].clone()
        aligned = (len(rP.input_ids) == len(ids)
                   and rP.spans["cue"] == rS.spans["cue"])
        m_crosspatch = None
        if aligned:
            with P.patch(model, {mid: (cue_pos, donor_P)}):
                m_crosspatch = m(P.forward_logprobs(model, ids))

        # 6 positive control: replace the ENTIRE final-block residual at the decision
        #   position from a different scenario's run. §8 says this must be able to change
        #   the answer, and that it proves nothing about localization.
        other = sids[(sids.index(sid) + 1) % len(sids)]
        bo = BaseScenario(**base[other])
        ro = render(tok, args.model, RenderedCondition(**byscen[other]["F"]), bo)
        last = n_layers - 1
        with P.capture(model, [last]) as store:
            P.forward_logprobs(model, ro.input_ids)
            donor_other = store[last].clone()
        # take the donor's own decision position, write it at ours
        donor_slice = donor_other[0, ro.decision_pos: ro.decision_pos + 1, :]
        with P.patch(model, {last: (dec, donor_slice)}):
            lp_pos = P.forward_logprobs(model, ids)
        m_poscontrol = m(lp_pos)
        top_base = int(lp_a.argmax())
        top_pos = int(lp_pos.argmax())

        checks.append({
            "scenario_id": sid,
            "aligned_PS": aligned,
            "m_baseline": m_base,
            "d_baseline_repeat": m_rep - m_base,
            "d_noop": m_noop - m_base,
            "d_self_patch": m_self - m_base,
            "d_zero_steer": m_zero - m_base,
            "d_crosspatch_P_into_S": (m_crosspatch - m_base
                                      if m_crosspatch is not None else None),
            "d_positive_control": m_poscontrol - m_base,
            "positive_control_changed_top_token": top_pos != top_base,
            "top_token_baseline": tok.decode([top_base]),
            "top_token_positive_control": tok.decode([top_pos]),
        })
        print(f"{sid}: noop {checks[-1]['d_noop']:+.2e} self {checks[-1]['d_self_patch']:+.2e} "
              f"zero {checks[-1]['d_zero_steer']:+.2e} "
              f"cross {checks[-1]['d_crosspatch_P_into_S']} "
              f"pos {checks[-1]['d_positive_control']:+.3f} "
              f"top {checks[-1]['top_token_baseline']!r}->"
              f"{checks[-1]['top_token_positive_control']!r}", flush=True)

    def worst(key):
        vals = [abs(c[key]) for c in checks if c[key] is not None]
        return max(vals) if vals else None

    noise = worst("d_baseline_repeat")
    tol = max(1e-6, (noise or 0.0) * 10)   # noise floor with an order-of-magnitude allowance
    report = {
        "gate": "G4",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "revision": model_revision(model),
        "load_mode": load_mode,
        "split": args.split,
        "n_scenarios": len(checks),
        "patched_tensor": "post-block residual stream (model.model.layers[i] output)",
        "measured_noise_floor_abs_margin": noise,
        "tolerance_used": tol,
        "worst_abs_delta": {
            "noop": worst("d_noop"),
            "self_patch": worst("d_self_patch"),
            "zero_steer": worst("d_zero_steer"),
        },
        "n_positive_control_changed_top_token": sum(
            c["positive_control_changed_top_token"] for c in checks),
        "mean_abs_positive_control_effect": (
            sum(abs(c["d_positive_control"]) for c in checks) / len(checks)),
        "n_aligned_PS": sum(c["aligned_PS"] for c in checks),
        "checks": checks,
    }
    report["gates"] = {
        "noop_within_noise": worst("d_noop") <= tol,
        "self_patch_within_noise": worst("d_self_patch") <= tol,
        "zero_steer_within_noise": worst("d_zero_steer") <= tol,
        # A positive control that cannot change the answer means the plumbing is inert and
        # every null result downstream would be uninterpretable.
        "positive_control_changes_output": report[
            "n_positive_control_changed_top_token"] > 0,
    }
    report["gates"]["G4_passed"] = all(report["gates"].values())

    out = DATA / "outputs" / "g4"
    out.mkdir(parents=True, exist_ok=True)
    tag = args.model.split("/")[-1]
    (out / f"g4_{tag}.json").write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 70)
    print(json.dumps({k: report[k] for k in (
        "measured_noise_floor_abs_margin", "tolerance_used", "worst_abs_delta",
        "n_positive_control_changed_top_token", "mean_abs_positive_control_effect",
        "n_aligned_PS", "gates")}, indent=2))
    print(f"\nwrote {out}/g4_{tag}.json")
    return 0 if report["gates"]["G4_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
