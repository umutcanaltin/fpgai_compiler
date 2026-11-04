"""R3f: controlled C2 finite action-space coverage on six variable-size graphs.

Six source-compiler cohorts, eight fixed typed actions each (48 total):
three MatMul chain depths and three bounded fanout degrees. The actual
existing compiler transition decides source admission and bounded E0
certification; blind vs masked comparison uses identical action inputs.
No trained RL, vendor HLS, executable GCC, or physical resource claims.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import re

from fpgai_research.experiments.cst_chain_campaign import make_chain
from fpgai_research.experiments.cst_fanout_models import sample_fanout
from fpgai_research.experiments.cst_r3d_c1c2_finite_contract_ablation import (
    _git_revision, structural_signature,
)
from fpgai_research.model_ir.decision_interface import DecisionAction, create_state
from fpgai_research.model_ir.fanout_decision_interface import create_fanout_state
from fpgai_research.model_ir.unified_decision_interface import (
    unified_action_mask, unified_graph_observation, unified_state_hash,
    unified_transition,
)

SCHEMA = "fpgai-research/R3f-C2-finite-variable-graph-mask-coverage/v1"
COHORTS = (
    ("chain_d2", 2), ("chain_d3", 3), ("chain_d4", 4),
    ("fanout_r2", 2), ("fanout_r3", 3), ("fanout_r4", 4),
)
CHAIN_SHAPES = {
    2: (3,4,2),
    3: (3,5,4,2),
    4: (2,3,4,3,2),
}
STATUSES = ("BOUNDED_CERTIFIED_E0", "UNKNOWN_E0", "REJECTED")


def _case(name: str, width: int):
    if name.startswith("chain_"):
        state = create_state(
            make_chain(width, CHAIN_SHAPES[width]), exchange_policy="row_chain")
        actions = (
            DecisionAction("operator","mm0","pes",2),
            DecisionAction("operator","mm0","pes",0),
            DecisionAction("exchange","model","exchange_policy","whole_tensor"),
            DecisionAction("exchange","model","row_block",2),
            DecisionAction("operator","mm0","simd",2),
            DecisionAction("operator","mm0","pes",1),
            DecisionAction("operator","mm0","pes",3),
            DecisionAction("operator","absent_operator","pes",2),
        )
    else:
        state = create_fanout_state(
            sample_fanout(width, m=2,k=2,n=3,q=4), layout="disjoint")
        actions = (
            DecisionAction("operator","reader0","pes",2),
            DecisionAction("operator",f"reader{width-1}","pes",2),
            DecisionAction("operator","producer","pes",2),
            DecisionAction("operator","reader0","pes",0),
            DecisionAction("memory","model","layout","per_element_optimal"),
            DecisionAction("memory","model","layout","disjoint"),
            DecisionAction("operator","reader0","simd",2),
            DecisionAction("operator","reader_undefined","pes",2),
        )
    return state, actions


def evaluate(name: str, width: int) -> dict:
    if (name, width) not in COHORTS:
        raise ValueError("R3f: unsupported or undeclared graph cohort")
    state, actions = _case(name, width)
    original = unified_state_hash(state)
    graph = unified_graph_observation(state)
    sig = structural_signature(graph)
    mask = unified_action_mask(state, actions)
    if (mask["candidate_count"] != 8
            or mask["parent_state_sha256"] != original
            or mask["domain"] != graph["domain"]):
        raise ValueError("R3f: actual compiler candidate mask/source state mismatch")
    # Separate real compiler replay of the *identical* action list, rather
    # than a cheap proxy used to confirm its own implementation.
    replay = tuple(unified_transition(state, action) for action in actions)
    if (tuple(result.status for result in replay) != mask["statuses"]
            or [result.as_receipt() for result in replay] != mask["rows"]
            or unified_state_hash(state) != original):
        raise ValueError("R3f: mask diverged from direct independent replay calls")
    statuses = mask["statuses"]
    if any(s not in STATUSES for s in statuses):
        raise ValueError("R3f: silently promoted unsupported compiler status")
    # Strong restricted negative controls: these are syntactically invalid,
    # or exact no-ops, independent of target optimization objective.
    negative_indices = (1, 5, 7) if name.startswith("chain_") else (2, 3, 5, 6, 7)
    if any(statuses[i] != "REJECTED" for i in negative_indices):
        raise ValueError("R3f: expected invalid source action was not rejected")
    counts = Counter(statuses)
    if sum(counts.values()) != 8 or (
            counts["BOUNDED_CERTIFIED_E0"] +
            counts["UNKNOWN_E0"] +
            counts["REJECTED"] != 8):
        raise ValueError("R3f: masked complete action population corrupted")
    if sum(mask["e0_certified"]) != counts["BOUNDED_CERTIFIED_E0"]:
        raise ValueError("R3f: E0 mask falsely promoted UNKNOWN/source-only")
    return {
        "cohort": name, "graph_parameter": width,
        "supported_domain": graph["domain"],
        "operator_count": len(state.model.operators),
        "tensor_count": sum(n.get("type") == "tensor" for n in graph["nodes"]),
        "graph_feature_digest": sig["digest"],
        "source_state_sha256": original,
        "candidate_count": 8,
        "statuses": list(statuses),
        "candidate_actions": [asdict(a) for a in actions],
        "source_cpp_sha256": [row["cpp_sha256"] for row in mask["rows"]],
        "certified": counts["BOUNDED_CERTIFIED_E0"],
        "source_admitted_but_e0_unknown": counts["UNKNOWN_E0"],
        "rejected": counts["REJECTED"],
        "mask_compiler_transitions": 8,
        "blind_replay_compiler_transitions": 8,
        "e0_certified_proposal_fraction": counts["BOUNDED_CERTIFIED_E0"] / 8,
        "invalid_or_uncertified_blind_candidates": (
            counts["REJECTED"] + counts["UNKNOWN_E0"]),
    }


def run() -> dict:
    rows = [evaluate(name, width) for name, width in COHORTS]
    total = sum(row["candidate_count"] for row in rows)
    counts = {k: sum(row[k] for row in rows) for k in
              ("certified", "source_admitted_but_e0_unknown", "rejected")}
    if total != 48 or sum(counts.values()) != total:
        raise AssertionError("R3f: source proposal denominator incomplete")
    return {
        "schema": SCHEMA,
        "prespecified_graphs": len(rows),
        "prespecified_candidates_per_graph": 8,
        "total_candidate_population": total,
        "total_e0_certified": counts["certified"],
        "total_source_admitted_but_e0_unknown":
            counts["source_admitted_but_e0_unknown"],
        "total_rejected": counts["rejected"],
        "mask_compiler_transition_calls": total,
        "blind_replay_compiler_transition_calls": total,
        "rows": rows,
        "native_vitis_calls": 0, "source_gcc_execution_calls": 0,
        "original_vivado_calls": 0, "trained_agents": 0,
        "limits": [
            "Sequential 2–4 MatMul chain or 2–4 reader MatMul fanout, not arbitrary graphs",
            "Complete typed source states, not partially constructed action histories",
            "48 hand-cataloged candidate actions, not an exhaustive hardware design space",
            "Masks and blind feedback both call the same actual source compiler",
            "Mask construction has a full 48-transition compiler cost",
            "A legal E0 source action may fail native HLS timing, resource, precision or physical DRC",
            "No learned agent, HLS reward, sample efficiency or general FPGA performance improvement",
        ],
    }


def create_only(output: str | Path, result: dict, *, revision: str) -> None:
    path = Path(output)
    if path.exists() or path.is_symlink():
        raise FileExistsError("R3f: source action campaign output already exists")
    if (result.get("schema") != SCHEMA
            or not isinstance(revision, str)
            or re.fullmatch(r"[0-9a-f]{40}", revision) is None):
        raise ValueError("R3f: source SHA or campaign schema invalid")
    path.mkdir(parents=True, exist_ok=False)
    (path / "r3f_c2_source_coverage.json").write_text(
        json.dumps({**result, "source_revision": revision},
                   sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")


def audit(study: str | Path) -> dict:
    root = Path(study)
    file = root / "r3f_c2_source_coverage.json"
    if (root.is_symlink() or not root.is_dir()
            or file.is_symlink() or not file.is_file()):
        raise ValueError("R3f: missing or linked source study")
    data = json.loads(file.read_text())
    rows = data.get("rows")
    if (data.get("schema") != SCHEMA
            or re.fullmatch(r"[0-9a-f]{40}", data.get("source_revision") or "") is None
            or data.get("prespecified_graphs") != 6
            or data.get("prespecified_candidates_per_graph") != 8
            or data.get("total_candidate_population") != 48
            or not isinstance(rows, list) or len(rows) != len(COHORTS)
            or any(data.get(x) != 0 for x in
                   ("native_vitis_calls","source_gcc_execution_calls",
                    "original_vivado_calls","trained_agents"))):
        raise ValueError("R3f: original source cohort signature/count invalid")
    for row, (name, width) in zip(rows, COHORTS):
        if (row.get("cohort") != name or row.get("graph_parameter") != width
                or row.get("candidate_count") != 8
                or not isinstance(row.get("statuses"), list)
                or len(row["statuses"]) != 8
                or not isinstance(row.get("candidate_actions"), list)
                or len(row["candidate_actions"]) != 8
                or row.get("mask_compiler_transitions") != 8
                or row.get("blind_replay_compiler_transitions") != 8
                or any(s not in STATUSES for s in row["statuses"])
                or not isinstance(row.get("graph_feature_digest"), str)
                or re.fullmatch(r"[a-f0-9]{64}", row["graph_feature_digest"]) is None):
            raise ValueError("R3f: corrupted saved action population")
        counted = Counter(row["statuses"])
        if (row.get("certified") != counted["BOUNDED_CERTIFIED_E0"]
                or row.get("source_admitted_but_e0_unknown") != counted["UNKNOWN_E0"]
                or row.get("rejected") != counted["REJECTED"]):
            raise ValueError("R3f: saved E0 population differs from actual statuses")
    for total, field in (
        ("certified","total_e0_certified"),
        ("source_admitted_but_e0_unknown","total_source_admitted_but_e0_unknown"),
        ("rejected","total_rejected"),
    ):
        if sum(row[total] for row in rows) != data.get(field):
            raise ValueError("R3f: overall masked source denominator mismatch")
    if (data.get("mask_compiler_transition_calls") != 48
            or data.get("blind_replay_compiler_transition_calls") != 48):
        raise ValueError("R3f: screen cost or blind baseline missing")
    return {
        "schema": SCHEMA,
        "original_source_revision": data["source_revision"],
        "graph_population": 6, "candidate_population": 48,
        "certified": data["total_e0_certified"],
        "unknown_e0": data["total_source_admitted_but_e0_unknown"],
        "rejected": data["total_rejected"],
        "read_only_receipt_sha256": sha256(file.read_bytes()).hexdigest(),
        "new_compiler_or_vendor_calls": 0,
        "trained_agents": 0,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output")
    mode.add_argument("--study")
    args = parser.parse_args(argv)
    if args.output:
        revision = _git_revision()
        result = run()
        create_only(args.output, result, revision=revision)
        print(json.dumps({
            "schema": SCHEMA, "source_revision": revision,
            "graph_population": 6, "candidate_population": 48,
            "certified": result["total_e0_certified"],
            "unknown_e0": result["total_source_admitted_but_e0_unknown"],
            "rejected": result["total_rejected"],
            "native_vendor_calls": 0, "trained_agents": 0,
        }, indent=2, sort_keys=True))
    else:
        print(json.dumps(audit(args.study), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
