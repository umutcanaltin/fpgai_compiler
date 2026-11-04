"""R2n finite policy-neutral C3 original-GCC source-qualified action study.

Three static row-chain depths each receive the SAME three candidate kinds:
a supported Phi edit, a malformed Phi edit, and a source-admissible
non-row-chain plan with no verified mapped-value certificate.

Only the first E0-bounded action is eligible for the independently
executed GCC host-numeric and logical value-lifetime checks. The study
records ALL outcomes. No training, RL reward, Vitis or Vivado calls.
"""
from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
import subprocess

import numpy as np

from fpgai_research.experiments.cst_chain_campaign import make_chain
from fpgai_research.model_ir.decision_interface import (
    DecisionAction, create_state, source_executed_action_mask,
)

SCHEMA = "fpgai-research/cst-R2n-original-GCC-certified-C3-decisions/v1"
CATALOG = (
    (2, 3, (3, 5, 2)),
    (3, 3, (3, 5, 4, 2)),
    (4, 3, (3, 5, 4, 3, 2)),
)
ACTION_ROLES = (
    "mapped_Phi_edit",
    "invalid_Phi_edit",
    "source_only_unverified_exchange",
)


def run(*, output: str | Path, graphs: int = 3) -> dict:
    target = Path(output)
    if target.exists() or target.is_symlink():
        raise FileExistsError("R2n original GCC study export is create-only")
    if type(graphs) is not int or not 1 <= graphs <= len(CATALOG):
        raise ValueError("R2n graph count must be 1 through 3")
    # Verify original source identity BEFORE any GCC process can run.
    repository = Path(__file__).resolve().parents[3]
    tracked = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain",
         "--untracked-files=no"],
        capture_output=True, text=True, check=True, timeout=10)
    if tracked.stdout.strip():
        raise RuntimeError("R2n source study requires clean tracked Git revision")
    revision = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True, timeout=10)
    from re import fullmatch
    if fullmatch(r"[0-9a-f]{40}", revision.stdout.strip()) is None:
        raise ValueError("R2n original GCC source revision must be a full commit SHA")
    records = []
    for depth, rows, widths in CATALOG[:graphs]:
        state = create_state(make_chain(rows, widths),
                             exchange_policy="row_chain")
        first = state.model.operators[0].name
        actions = (
            DecisionAction("operator", first, "pes", 2),
            DecisionAction("operator", first, "pes", 0),
            DecisionAction("exchange", "model", "exchange_policy",
                           "whole_tensor"),
        )
        input_data = np.linspace(
            -.125, .125, rows * widths[0], dtype=np.float32).reshape(
                rows, widths[0])
        outcome = source_executed_action_mask(
            state, actions, runtime_input=input_data,
            max_e1_executions=len(actions))
        if (outcome["native_vitis_calls"] != 0
                or outcome["rl_policies_trained"] != 0
                or outcome["physical_feasibility_checked"]):
            raise ValueError("R2n cannot claim native or trained evidence")
        if len(outcome["rows"]) != len(actions):
            raise ValueError("R2n candidate evidence unexpectedly incomplete")
        records.append({
            "graph_depth": depth,
            "tensor_nodes": len(state.model.tensors),
            "fixed_input_sha256": sha256(input_data.tobytes()).hexdigest(),
            "original_state_sha256": outcome["parent_state_sha256"],
            "roles": list(ACTION_ROLES),
            "actions": outcome["rows"],
            "e1_attempts": outcome["e1_executions_attempted"],
            "native_vitis_calls": 0,
            "rl_policies_trained": 0,
        })
    counts = {}
    for r in records:
        for case in r["actions"]:
            status = case["status"]
            counts[status] = counts.get(status, 0) + 1
    result = {
        "schema": SCHEMA,
        "source_revision": revision.stdout.strip(),
        "graph_depths": [r["graph_depth"] for r in records],
        "graphs": len(records),
        "all_candidate_actions": sum(len(r["actions"]) for r in records),
        "status_denominators": counts,
        "e1_host_executions_attempted": sum(
            r["e1_attempts"] for r in records),
        "e1_verified_actions": counts.get("E1_GCC_SOURCE_QUALIFIED", 0),
        "source_only_uncertified_actions": counts.get(
            "UNKNOWN_E0_CERTIFICATE", 0),
        "rejected_actions": counts.get("REJECTED", 0),
        "failed_e1_actions": counts.get(
            "FAILED_E1_SOURCE_EXECUTION", 0),
        "native_vitis_calls": 0,
        "new_vivado_calls": 0,
        "rl_policies_trained": 0,
        "evidence": "bounded E0 compiler + selected original GCC E1 source-only",
        "limits": [
            "Three fixed small single-reader MatMul chains only",
            "One deterministic host-float input per qualified complete action",
            "No trained policy, RL reward, learning sample efficiency or transfer",
            "No general fan-out/graph-wide policy semantic certification",
            "Not an E2/E3 native fixed-point Vitis or E4 routed timing result",
            "SOURCE-admissible may remain E0 UNKNOWN and is not E1 certified",
        ],
    }
    # No source/native study files are written before all candidates have been
    # classified; only a NEW read-only result directory is created here.
    target.mkdir(parents=True)
    (target / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    (target / "case_evidence.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n")
    return result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--graphs", type=int, default=3)
    args = p.parse_args(argv)
    print(json.dumps(run(output=args.output, graphs=args.graphs), indent=2))


if __name__ == "__main__":
    main()
