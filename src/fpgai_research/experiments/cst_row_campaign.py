"""Small CPU and GCC E0/E1 experiment: whole-graph vs row-epoch MatMul.

No Vitis calls, no RL import and no physical BRAM claims. A reduced static
source array length is a concrete source declaration, not proof of BRAM cost.
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import subprocess

from fpgai_research.architecture.resolver import resolve
from fpgai_research.compiler.composition import compile_composed_architecture
from fpgai_research.compiler.row_interleaved import compile_row_interleaved
from fpgai_research.model_ir.contracts import (
    ArchitectureRequest, FunctionalModel, Operator, Tensor,
)
from fpgai_research.validation.host import validate_host


def model_for(rows, inner, middle, outer):
    w1 = tuple(0.025*(i%7-3) for i in range(inner*middle))
    w2 = tuple(0.025*(i%5-2) for i in range(middle*outer))
    return FunctionalModel(
        (Tensor("x", (rows, inner)),
         Tensor("w1", (inner, middle), data=w1),
         Tensor("h", (rows, middle)),
         Tensor("w2", (middle, outer), data=w2),
         Tensor("y", (rows, outer))),
        (Operator("a", "MatMul", ("x", "w1"), ("h",)),
         Operator("b", "MatMul", ("h", "w2"), ("y",))),
        ("x",), ("y",),
    )


def run(output, *, max_pairs=18, host_pairs=4):
    if type(max_pairs) is not int or not 1 <= max_pairs <= 36:
        raise ValueError("max_pairs must be 1..36")
    if type(host_pairs) is not int or not 0 <= host_pairs <= max_pairs:
        raise ValueError("host_pairs outside 0..max_pairs")
    dest = Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Create-only source study output already exists")
    configurations = [
        (rows, middle, mode)
        for rows in (2, 3, 5)
        for middle in (3, 5, 7)
        for mode in ("uniform", "heterogeneous")
    ][:max_pairs]
    records = []
    for case_index, (rows, middle, mode) in enumerate(configurations):
        graph = model_for(rows, 3, middle, 2)
        a = ArchitectureRequest()
        architecture = resolve(
            graph, a,
            {} if mode == "uniform" else {
                "a": replace(a, pes=2, parallelism=2,
                             simd=2, serialization=2),
                "b": replace(a, pes=3, parallelism=2,
                             simd=3, serialization=2),
            })
        baseline = compile_composed_architecture(architecture)
        interleaved = compile_row_interleaved(architecture)
        if baseline.emission.manifest["functional_sha256"] != interleaved.emission.manifest["functional_sha256"]:
            raise AssertionError("Unmatched Functional Model IR source")
        if baseline.emission.manifest["cpp_sha256"] == interleaved.emission.manifest["cpp_sha256"]:
            raise AssertionError("Expected a truly different generated HLS schedule")
        row = {
            "shape": f"{rows}x3x{middle}x2", "mapping_mode": mode,
            "baseline_cpp_sha256": baseline.emission.manifest["cpp_sha256"],
            "row_interleaved_cpp_sha256": interleaved.emission.manifest["cpp_sha256"],
            "baseline_declared_intermediate_elements": rows*middle,
            "interleaved_declared_scratch_elements": middle,
            "source_scratch_reduction_ratio": rows,
            "host_executed": False, "baseline_numeric_pass": None,
            "interleaved_numeric_pass": None, "max_abs_error": None,
            "original_vitis_calls": 0,
        }
        if case_index < host_pairs:
            import numpy as np
            x = np.linspace(-0.1, 0.1, rows*3, dtype=np.float32).reshape(rows,3)
            b = validate_host(graph, baseline.emission, x)
            v = validate_host(graph, interleaved.emission, x)
            row["host_executed"] = True
            row["baseline_numeric_pass"] = b["accepted"]
            row["interleaved_numeric_pass"] = v["accepted"]
            row["max_abs_error"] = v["max_absolute_error"]
            if not (b["accepted"] and v["accepted"]):
                raise AssertionError("Native GCC source numeric qualification failed")
        records.append(row)
    dest.mkdir(parents=True)
    with (dest / "cases.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    rev = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True, timeout=5)
    summary = {
        "schema": "fpgai-research/row-lifetime-cpu-campaign/v1",
        "source_revision": rev.stdout.strip() if rev.returncode == 0 else None,
        "paired_cases": len(records), "host_checked_pairs": host_pairs,
        "host_both_qualified": sum(x["host_executed"] and
                                  x["baseline_numeric_pass"] is True and
                                  x["interleaved_numeric_pass"] is True for x in records),
        "all_generated_sources_distinct": all(x["baseline_cpp_sha256"] !=
                                              x["row_interleaved_cpp_sha256"]
                                              for x in records),
        "max_source_scratch_reduction_ratio": max(
            x["source_scratch_reduction_ratio"] for x in records),
        "original_vitis_calls": 0,
        "evidence": "E0-typed-row-lifetime-plus-E1-source-and-optional-GCC-host",
        "not_evidence_of": ["synthesized BRAM reduction", "timing closure",
                            "physical throughput", "streaming FIFO", "RL improvement"],
    }
    (dest / "summary.json").write_text(json.dumps(summary, indent=2,
                                                sort_keys=True)+"\n")
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--max-pairs", type=int, default=18)
    p.add_argument("--host-pairs", type=int, default=4)
    args = p.parse_args(argv)
    print(json.dumps(run(args.output, max_pairs=args.max_pairs,
                         host_pairs=args.host_pairs), indent=2))


if __name__ == "__main__":
    main()
