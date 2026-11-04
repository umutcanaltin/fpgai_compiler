"""R2o original GCC C3: complete Phi-mapped 2..4-reader decision study.

Prespecified 3 fanout shapes x four policy-neutral actions. Real R2h
generated source, source-address observations and NumPy checks for each
E0-admissible candidate; original source, manifests, full GCC stdout
and case receipts are create-only. No RL training or Vitis/Vivado.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from fpgai_research.experiments.cst_fanout_models import sample_fanout
from fpgai_research.experiments.cst_r2b_source_campaign import _input
from fpgai_research.model_ir.decision_interface import DecisionAction
from fpgai_research.model_ir.fanout_decision_interface import (
    create_fanout_state, fanout_certificate, source_qualified_multi_reader_mask,
)

SCHEMA = "fpgai-research/cst-R2o-original-multi-reader-C3-e1/v1"
CATALOG = (
    (2, (2, 2, 3, 4)),
    (3, (2, 3, 5, 4)),
    (4, (3, 3, 4, 4)),
)
ACTION_ROLES = (
    "first_reader_Phi_pes2",
    "last_reader_Phi_pes2",
    "illegal_producer_PE",
    "certified_source_slot_aliasing",
)


def run(*, output: str | Path, graphs: int = 3) -> dict:
    root = Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("R2o original GCC source study is create-only")
    if type(graphs) is not int or not 1 <= graphs <= len(CATALOG):
        raise ValueError("R2o graph count must be between 1 and 3")
    repo = Path(__file__).resolve().parents[3]
    check = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain",
         "--untracked-files=no"], capture_output=True,
        text=True, check=True, timeout=10)
    if check.stdout.strip():
        raise RuntimeError("R2o original study requires clean tracked source")
    rev = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    if re.fullmatch(r"[a-f0-9]{40}", rev) is None:
        raise ValueError("R2o original source revision must be full Git SHA")

    root.mkdir(parents=True)
    cases = []
    for i, (readers, shape) in enumerate(CATALOG[:graphs]):
        folder = root / f"graph_{i:03d}"
        folder.mkdir()
        m, k, n, q = shape
        model = sample_fanout(readers, m=m, k=k, n=n, q=q)
        initial = create_fanout_state(model, layout="disjoint")
        baseline = fanout_certificate(initial)
        original_input = np.asarray(_input(m, k), dtype=np.float32)
        actions = (
            DecisionAction("operator", "reader0", "pes", 2),
            DecisionAction("operator", f"reader{readers-1}", "pes", 2),
            DecisionAction("operator", "producer", "pes", 2),
            DecisionAction("memory", "model", "layout", "per_element_optimal"),
        )
        try:
            result = source_qualified_multi_reader_mask(
                initial, actions, runtime_input=original_input,
                max_e1_executions=3, evidence_dir=folder)
        except Exception as exc:
            (folder / "FAILED.json").write_text(json.dumps({
                "case": i, "source_revision": rev,
                "status": "FAILED_ORIGINAL_R2o_GCC",
                "exception_type": type(exc).__name__,
                "error": str(exc)[:1000],
            }, indent=2, sort_keys=True) + "\n")
            raise
        cases.append({
            "graph_case": f"graph_{i:03d}",
            "fanout_readers": readers,
            "shape": list(shape),
            "original_source_revision": rev,
            "initial_source_certificate_sha256":
                baseline["certificate_sha256"],
            "input_float32_sha256": sha256(original_input.tobytes()).hexdigest(),
            "action_roles": list(ACTION_ROLES),
            "evidence": result,
        })
        # Save case receipts as soon as the original source work finishes.
        with (folder / "receipt.json").open("x", encoding="utf-8") as out:
            json.dump(cases[-1], out, indent=2, sort_keys=True)
            out.write("\n")
    summary_counts = {}
    for case in cases:
        for value in case["evidence"]["statuses"]:
            summary_counts[value] = summary_counts.get(value, 0) + 1
    summary = {
        "schema": SCHEMA,
        "original_source_revision": rev,
        "graph_count": len(cases),
        "prespecified_graphs": [
            {"fanout_readers": r, "shape": list(d)}
            for r, d in CATALOG[:graphs]
        ],
        "actions_per_graph": len(ACTION_ROLES),
        "original_action_denominator": len(cases)*len(ACTION_ROLES),
        "status_denominators": summary_counts,
        "original_GCC_source_qualification_attempts": sum(
            c["evidence"]["gcc_e1_attempts"] for c in cases),
        "original_GCC_qualified_actions":
            summary_counts.get("E1_GCC_MULTI_READER_QUALIFIED", 0),
        "rejected_actions": summary_counts.get("REJECTED", 0),
        "unknown_actions": summary_counts.get("UNKNOWN_E1_NOT_RUN", 0),
        "failed_e1_actions":
            summary_counts.get("FAILED_E1_SOURCE_EXECUTION", 0),
        "original_Vitis_calls": 0,
        "original_Vivado_calls": 0,
        "trained_RL_policies": 0,
        "evidence": "bounded original R2h GCC actual multi-reader source E1",
        "limits": [
            "Specific 2..4 MatMul reader fanout, left Add and Relu only",
            "Only supported first/last reader PE widths and layout edits",
            "One fixed host-float input per graph, not universal equivalence",
            "Original source GCC trace and NumPy, not HLS cycle or BRAM",
            "No original fixed-point Vitis, clean-DRC E4 or trained RL",
        ],
    }
    with (root / "summary.json").open("x", encoding="utf-8") as out:
        json.dump(summary, out, indent=2, sort_keys=True)
        out.write("\n")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--graphs", type=int, default=3)
    args = parser.parse_args(argv)
    print(json.dumps(run(output=args.output, graphs=args.graphs),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
