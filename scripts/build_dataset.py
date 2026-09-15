"""Build the controlled corpus and run the G1 audit (protocol §5, §6, §7).

Partitions follow §5 exactly. Invariants enforced here rather than trusted:
  - all six siblings of a base scenario stay in one partition;
  - held-out cue families are used ONLY in the cue-family split and were fixed before any
    target-model evaluation;
  - the task-transfer split changes the task family while keeping familiar cue families, so
    task generalization and cue generalization stay distinguishable;
  - the source-probe corpus is generated independently of the attack corpus and shares no
    snippets with it.

Writes to $DATA_DIR/datasets/ plus a G1 audit report. Nothing is model-specific here except
the per-model rendering pass, which is written separately per §6 ("Each model rendering adds
...").

Usage:
    source env.sh && .venv/bin/python scripts/build_dataset.py
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import oracle
from common.scenarios import (CUE_FAMILIES, HELDOUT_CUE_FAMILIES, GENERATOR_VERSION,
                              ANSWER_LABELS, CONDITIONS, build_conditions,
                              make_base_scenario)
from common.oracle import OracleError

DATA = Path(os.environ["DATA_DIR"])
OUT = DATA / "datasets"
REPORT = DATA / "outputs" / "g1"

MASTER_SEED = 20260915

# §5. (n_base_scenarios, task families, cue pool)
PARTITIONS = {
    "pilot":        (200, ("record_lookup", "field_select"), "primary"),
    "discovery":    (240, ("record_lookup", "field_select"), "primary"),
    "validation":   (120, ("record_lookup", "field_select"), "primary"),
    "heldout_wd":   (120, ("record_lookup", "field_select"), "primary"),
    "heldout_cue":  (120, ("record_lookup", "field_select"), "heldout"),
    "heldout_task": (120, ("two_hop",),                      "primary"),
}

CUE_POOLS = {"primary": CUE_FAMILIES, "heldout": HELDOUT_CUE_FAMILIES}


def build_partition(split: str, n: int, families: tuple[str, ...], pool_name: str):
    """Generate `n` base scenarios, balancing task families within the partition (§5)."""
    pool = CUE_POOLS[pool_name]
    rng = random.Random(f"{MASTER_SEED}:{split}")
    rows, cond_rows, failures, warnings = [], [], [], []

    for i in range(n):
        # Deterministic balanced assignment rather than random, so counts are exact.
        fam = families[i % len(families)]
        sid = f"{split}-{i:04d}"
        seed = rng.randrange(2**31)
        bs = make_base_scenario(sid, fam, split, seed, pool)
        conds = build_conditions(bs)
        try:
            warn = oracle.check_scenario(bs, conds)
        except OracleError as e:
            failures.append({"scenario_id": sid, "task_family": fam,
                             "cue_family": bs.cue_family, "error": str(e)})
            continue
        warnings.extend({"scenario_id": sid, "warning": w} for w in warn)
        rows.append(asdict(bs))
        for cond, rc in conds.items():
            cond_rows.append(rc.to_record())
    return rows, cond_rows, failures, warnings


def build_probe_corpus():
    """Separate neutral corpus for source probes (§5): 1,000 snippets, split 600/200/200 at
    the snippet level, independent of the attack corpus, free of answer labels and attack
    phrasing."""
    rng = random.Random(f"{MASTER_SEED}:probe")
    subjects = ["The maintenance window", "This collection", "The export job", "The index",
                "The retention policy", "The migration batch", "The audit trail",
                "The schema revision", "The staging area", "The archive tier",
                "The catalog entry", "The replication lag", "The backup set",
                "The ingest queue", "The checksum pass", "The storage volume"]
    verbs = ["was completed", "is reviewed", "has been scheduled", "remains unchanged",
             "was recorded", "is monitored", "has been rebuilt", "was verified",
             "is documented", "was rotated", "has been archived", "is tracked"]
    tails = ["on the usual cycle.", "without manual intervention.", "by the standing process.",
             "according to the internal convention.", "during the quiet period.",
             "as part of routine operation.", "under the current configuration.",
             "in the order originally received.", "for the retention interval.",
             "with no reported exceptions."]

    seen, snips = set(), []
    while len(snips) < 1000:
        s = f"{rng.choice(subjects)} {rng.choice(verbs)} {rng.choice(tails)}"
        if s in seen:
            continue
        # §5: no answer labels, no attack phrasing in this corpus.
        if any(l in s for l in ANSWER_LABELS) or "Reply" in s:
            continue
        seen.add(s)
        snips.append(s)

    idx = list(range(1000))
    rng.shuffle(idx)
    split_of = {}
    for rank, j in enumerate(idx):
        split_of[j] = "fit" if rank < 600 else ("val" if rank < 800 else "test")
    return [{"snippet_id": f"nb-{j:04d}", "text": snips[j], "split": split_of[j]}
            for j in range(1000)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)

    all_base, all_cond, all_fail, all_warn = [], [], [], []
    per_split = {}

    for split, (n, fams, pool) in PARTITIONS.items():
        rows, conds, fails, warns = build_partition(split, n, fams, pool)
        per_split[split] = {
            "requested_base": n,
            "retained_base": len(rows),
            "rendered_conditions": len(conds),
            "failures": len(fails),
            "task_families": dict(Counter(r["task_family"] for r in rows)),
            "cue_families": dict(Counter(r["cue_family"] for r in rows)),
            "cue_strata": dict(Counter(r["cue_stratum"] for r in rows)),
            "position_strata": dict(Counter(r["position_stratum"] for r in rows)),
        }
        all_base += rows
        all_cond += conds
        all_fail += fails
        all_warn += warns
        print(f"{split:14} base {len(rows):4}/{n:<4} conditions {len(conds):5} "
              f"failures {len(fails)}", flush=True)

    probe = build_probe_corpus()

    # ---------------- G1 audit: checks over the whole corpus ----------------
    audit: dict[str, object] = {}

    # 1. every retained scenario has exactly six conditions
    by_sid = defaultdict(set)
    for c in all_cond:
        by_sid[c["scenario_id"]].add(c["condition"])
    bad_sets = {s: sorted(v) for s, v in by_sid.items() if set(v) != set(CONDITIONS)}
    audit["scenarios_missing_conditions"] = bad_sets

    # 2. no scenario family crosses splits (§6)
    fam_splits = defaultdict(set)
    for r in all_base:
        fam_splits[r["family_id"]].add(r["split"])
    audit["families_crossing_splits"] = {f: sorted(s) for f, s in fam_splits.items()
                                         if len(s) > 1}

    # 3. held-out cue families appear ONLY in heldout_cue
    heldout_names = {c.name for c in HELDOUT_CUE_FAMILIES}
    leak = [(r["scenario_id"], r["split"], r["cue_family"]) for r in all_base
            if r["cue_family"] in heldout_names and r["split"] != "heldout_cue"]
    audit["heldout_cue_leakage"] = leak
    primary_names = {c.name for c in CUE_FAMILIES}
    leak2 = [(r["scenario_id"], r["cue_family"]) for r in all_base
             if r["split"] == "heldout_cue" and r["cue_family"] in primary_names]
    audit["primary_cue_in_heldout_split"] = leak2

    # 4. task-transfer split really changes the task and keeps familiar cues
    tt = [r for r in all_base if r["split"] == "heldout_task"]
    audit["heldout_task_families"] = dict(Counter(r["task_family"] for r in tt))
    audit["heldout_task_uses_primary_cues"] = all(
        r["cue_family"] in primary_names for r in tt)

    # 5. answer-identity balance (§6: answer identities, task types, positions balanced)
    audit["authorized_answer_distribution"] = dict(
        Counter(r["authorized_answer"] for r in all_base))
    audit["attacker_target_distribution"] = dict(
        Counter(r["attacker_target"] for r in all_base))
    # A label the model simply prefers would masquerade as an effect; report the spread.
    ac = Counter(r["authorized_answer"] for r in all_base)
    audit["answer_balance_minmax"] = [min(ac.values()), max(ac.values())] if ac else None

    # 6. authorized != attacker target everywhere (belt and braces; oracle already asserts)
    audit["answer_collisions"] = [r["scenario_id"] for r in all_base
                                  if r["authorized_answer"] == r["attacker_target"]]

    # 7. probe corpus independence from the attack corpus
    attack_text = " ".join(c["document"] for c in all_cond)
    audit["probe_snippets_appearing_in_attack_corpus"] = sum(
        1 for p in probe if p["text"] in attack_text)
    audit["probe_split_counts"] = dict(Counter(p["split"] for p in probe))
    audit["probe_duplicate_texts"] = len(probe) - len({p["text"] for p in probe})

    # 8. document duplication (§5: group near-duplicate documents).
    # Two duplications are expected BY DESIGN and are controls, not leakage:
    #   N == U  (same neutral document; only a genuine later user turn is added)
    #   S == Q  (same document carrying the forged cue; only the user's task differs, which
    #            is exactly what makes Q a clean "instruction as data" control)
    # What would actually be a problem is the same document appearing under two different
    # scenarios, which could put near-duplicates on both sides of a split boundary.
    doc_to_sids = defaultdict(set)
    doc_to_conds = defaultdict(set)
    for c in all_cond:
        doc_to_sids[c["document"]].add(c["scenario_id"])
        doc_to_conds[c["document"]].add(c["condition"])
    cross = {d: sorted(s) for d, s in doc_to_sids.items() if len(s) > 1}
    audit["documents_shared_across_scenarios"] = len(cross)
    audit["documents_shared_across_scenarios_examples"] = [
        {"scenario_ids": v} for v in list(cross.values())[:5]]
    # Confirm every within-scenario duplicate is one of the two expected pairs.
    unexpected = [sorted(cs) for d, cs in doc_to_conds.items()
                  if len(cs) > 1 and sorted(cs) not in (["N", "U"], ["Q", "S"])]
    audit["unexpected_condition_document_collisions"] = unexpected[:10]
    audit["n_unexpected_condition_document_collisions"] = len(unexpected)
    audit["expected_design_duplicates_NU_SQ"] = sum(
        1 for cs in doc_to_conds.values() if sorted(cs) in (["N", "U"], ["Q", "S"]))

    critical = (bool(bad_sets) or bool(audit["families_crossing_splits"]) or bool(leak)
                or bool(leak2) or bool(audit["answer_collisions"]) or bool(all_fail)
                or audit["probe_snippets_appearing_in_attack_corpus"] > 0
                or audit["probe_duplicate_texts"] > 0
                or not audit["heldout_task_uses_primary_cues"]
                or audit["documents_shared_across_scenarios"] > 0
                or audit["n_unexpected_condition_document_collisions"] > 0)

    report = {
        "gate": "G1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "generator_version": GENERATOR_VERSION,
        "master_seed": MASTER_SEED,
        "totals": {
            "base_scenarios": len(all_base),
            "rendered_conditions": len(all_cond),
            "probe_snippets": len(probe),
            "oracle_failures": len(all_fail),
            "warnings": len(all_warn),
        },
        "per_split": per_split,
        "audit": audit,
        "oracle_failures": all_fail[:50],
        "warnings_sample": all_warn[:50],
        "critical_errors_present": critical,
    }

    (OUT / "base_scenarios.jsonl").write_text(
        "\n".join(json.dumps(r) for r in all_base) + "\n")
    (OUT / "conditions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in all_cond) + "\n")
    (OUT / "probe_corpus.jsonl").write_text(
        "\n".join(json.dumps(r) for r in probe) + "\n")
    (REPORT / "g1_report.json").write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 70)
    print(json.dumps({k: report[k] for k in ("totals",)}, indent=2))
    print("audit summary:")
    for k, v in audit.items():
        flag = ""
        if isinstance(v, (list, dict)) and len(v) > 0 and k not in (
                "authorized_answer_distribution", "attacker_target_distribution",
                "heldout_task_families", "probe_split_counts", "answer_balance_minmax"):
            flag = "  <-- NONEMPTY"
        shown = v if not isinstance(v, (list, dict)) or len(str(v)) < 120 else f"<{len(v)} items>"
        print(f"  {k:48} {shown}{flag}")
    print(f"\nG1 critical errors present: {critical}")
    print(f"wrote {OUT}/ and {REPORT}/g1_report.json")
    return 1 if critical else 0


if __name__ == "__main__":
    sys.exit(main())
