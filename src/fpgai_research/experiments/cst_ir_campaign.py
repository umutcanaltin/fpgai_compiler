"""Reproducible, CPU-only first-paper CST-IR baseline/alternative campaign.

Preserves original native Vitis receipts. Never launches Vitis, RL, or writes
outside a new create-only output directory. Numerical host testing is optional
and requires g++; absence is reported rather than faked. All findings are
E0/E1, not an HLS throughput or physical timing claim.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import subprocess

from fpgai_research.architecture.resolver import resolve
from fpgai_research.compiler.composition import compile_composed_architecture
from fpgai_research.model_ir.contracts import (
    ArchitectureRequest, FunctionalModel, Operator, Tensor,
)
from fpgai_research.model_ir.semantics import architecture_fingerprint
from fpgai_research.validation.host import validate_host


def _model(rows: int, inner: int, middle: int, output: int) -> FunctionalModel:
    # Deterministic finite exact source weights; these do not encode a
    # pretrained network or trained DRL policy.
    w1 = tuple(0.025 * ((i % 7) - 3) for i in range(inner * middle))
    w2 = tuple(0.025 * ((i % 5) - 2) for i in range(middle * output))
    return FunctionalModel(
        tensors=(Tensor("x", (rows, inner)),
                 Tensor("w1", (inner, middle), data=w1),
                 Tensor("h", (rows, middle)),
                 Tensor("w2", (middle, output), data=w2),
                 Tensor("y", (rows, output))),
        operators=(Operator("first", "MatMul", ("x", "w1"), ("h",)),
                   Operator("second", "MatMul", ("h", "w2"), ("y",))),
        inputs=("x",), outputs=("y",),
    )


def _arch(model: FunctionalModel, mode: str):
    base = ArchitectureRequest()
    if mode == "uniform":
        return resolve(model, base)
    if mode == "operator_local":
        return resolve(model, base, {
            "first": replace(base, pes=2, simd=3, serialization=2),
            "second": replace(base, pes=3, simd=2, serialization=3),
        })
    raise ValueError("Unknown architecture mode")


def _revision():
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=5)
        return proc.stdout.strip() if proc.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def run(*, output: str | Path, max_cases: int = 96, host_check_pairs: int = 4) -> dict:
    """Generate a matched, immutable E0/E1 CSV/JSON study; zero vendor calls."""
    if type(max_cases) is not int or max_cases < 2 or max_cases > 1000:
        raise ValueError("max_cases must be 2..1000")
    if type(host_check_pairs) is not int or host_check_pairs < 0 or host_check_pairs > max_cases // 2:
        raise ValueError("Invalid host_check_pairs")
    dest = Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Create-only experiment output already exists")
    tasks = [
        (rows, inner, middle, outputs, mode)
        for rows in (2, 3, 5)
        for inner in (3, 4)
        for middle in (3, 5, 7)
        for outputs in (2, 3)
        for mode in ("uniform", "operator_local")
    ]
    rows_out = []
    pairs = 0
    for rows, inner, middle, outputs, mode in tasks:
        if len(rows_out) + 2 > max_cases:
            break
        graph = _model(rows, inner, middle, outputs)
        arch = _arch(graph, mode)
        architecture_id = architecture_fingerprint(arch)
        for layout in ("row_major", "column_major"):
            layout_request = {} if layout == "row_major" else {"h": "column_major"}
            compiled = compile_composed_architecture(
                arch, intermediate_layouts=layout_request)
            checked = compiled.evidence["operator_to_boundary_exchange_witness"]
            if any(item["status"] != "BOUNDED_PHI_EXCHANGE_VERIFIED"
                   for item in checked["relations"]):
                raise AssertionError("Campaign has an unverified Phi boundary")
            row = {
                "shape": f"{rows}x{inner}x{middle}x{outputs}",
                "mapping_mode": mode, "physical_layout": layout,
                "functional_sha256": compiled.emission.manifest["functional_sha256"],
                "effective_architecture_sha256": architecture_id,
                "cpp_sha256": compiled.emission.manifest["cpp_sha256"],
                "boundary_sha256": compiled.evidence["boundary_sha256"],
                "exchange_relation_sha256": checked["relations"][0]["event_trace_sha256"],
                "source_emitted": True,
                "source_layout_materialized": compiled.evidence["emitter_layouts"] == layout_request,
                "host_checked": False, "host_numeric_pass": None,
                "host_max_abs_error": None,
                "vitis_calls": 0, "physical_timing_claim": False,
            }
            if pairs < host_check_pairs:
                import numpy as np
                x = np.linspace(-0.125, 0.125, rows * inner,
                                dtype=np.float32).reshape(rows, inner)
                tested = validate_host(graph, compiled.emission, x)
                row["host_checked"] = True
                row["host_numeric_pass"] = tested["accepted"]
                row["host_max_abs_error"] = tested["max_absolute_error"]
                if not tested["accepted"]:
                    raise AssertionError("Generated C++ failed actual numerical host test")
            rows_out.append(row)
        before, after = rows_out[-2:]
        if (before["functional_sha256"] != after["functional_sha256"]
                or before["effective_architecture_sha256"] != after["effective_architecture_sha256"]
                or before["cpp_sha256"] == after["cpp_sha256"]):
            raise AssertionError("Matched source-change pair invariant violated")
        pairs += 1
    dest.mkdir(parents=True)
    headers = list(rows_out[0])
    with (dest / "cases.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, headers)
        writer.writeheader()
        writer.writerows(rows_out)
    # Separate the number of mathematical models from architecture choices.
    # architecture_fingerprint includes Functional IR identity, so raw count
    # across networks is NOT the number of architectures of one model.
    graphs = {}
    for case in rows_out:
        graphs.setdefault(case["functional_sha256"], set()).add(
            case["effective_architecture_sha256"])
    summary = {
        "schema": "fpgai-research/compositional-ir-cpu-campaign/v1",
        "source_revision": _revision(),
        "cases": len(rows_out), "matched_pairs": pairs,
        "host_checked_cases": sum(x["host_checked"] for x in rows_out),
        "host_numeric_passes": sum(x["host_numeric_pass"] is True for x in rows_out),
        "distinct_functional_graphs": len(graphs),
        "graphs_with_at_least_two_architecture_mappings": sum(
            len(options) >= 2 for options in graphs.values()),
        "max_effective_mappings_per_functional_graph": max(
            len(options) for options in graphs.values()),
        "mapping_fingerprints_include_functional_graph_identity": True,
        "distinct_effective_architectures": len(set(
            x["effective_architecture_sha256"] for x in rows_out)),
        "distinct_generated_cpp": len(set(x["cpp_sha256"] for x in rows_out)),
        "all_alternative_layouts_materialized": all(
            x["source_layout_materialized"] for x in rows_out),
        "vitis_calls": 0,
        "evidence": "E0-typed-Phi-boundaries-plus-E1-generated-source",
        "not_evidence_of": ["Vitis-CSim", "HLS-synthesis", "routed-clock",
                            "RTL-equivalence", "physical-layout", "RL-learning"],
        "limits": "Restricted same-compiler baseline; alternate flat-array layout is not itself a novelty claim",
    }
    (dest / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--max-cases", type=int, default=96)
    p.add_argument("--host-check-pairs", type=int, default=4)
    args = p.parse_args(argv)
    print(json.dumps(run(output=args.output, max_cases=args.max_cases,
                         host_check_pairs=args.host_check_pairs), indent=2))


if __name__ == "__main__":
    main()
