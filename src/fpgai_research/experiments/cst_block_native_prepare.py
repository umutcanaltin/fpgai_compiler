"""Prospective original-Vitis row-block schedule sweep: source-only preparation.

Three matched actual compiler lowerings for each 3-/4-operator graph:
block one row, block two rows, and block all rows. This is a *new*
experiment family, not the original whole/column/row triplet.

The original Vitis runner/auditor in cst_native_campaign can execute and
audit these packages without change. Results remain unmeasured until
explicit original Vitis invocation on a licensed machine.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import json

import numpy as np

from fpgai_research.architecture.resolver import resolve
from fpgai_research.compiler.row_chain import compile_row_chain
from fpgai_research.experiments.cst_chain_campaign import make_chain
from fpgai_research.experiments.cst_native_campaign import (
    SCHEMA, file_hash, revision,
)
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.toolchains.vitis_hls.package import write_vitis_package


def prepare(*, output: str | Path, max_cases: int = 6,
            part: str = "xck26-sfvc784-2LV-c",
            clock_ns: float = 32.0,
            clock_uncertainty_ns: float = 8.64,
            tolerance: float = 0.05, shape_catalog: str = "pilot") -> dict:
    import math
    if shape_catalog not in ("pilot", "paper"):
        raise ValueError("shape_catalog must be pilot or paper")
    if type(max_cases) is not int or max_cases not in (3, 6, 9, 12):
        raise ValueError("Study requires 3, 6, 9 or 12 exact matched cases")
    if not all(math.isfinite(x) for x in (
            clock_ns, clock_uncertainty_ns, tolerance)):
        raise ValueError("Finite target clock and tolerance required")
    if not (0 <= clock_uncertainty_ns < clock_ns and tolerance > 0):
        raise ValueError("Invalid native clock/tolerance constraints")
    dest = Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Source study must be create-only")
    pilot_configs = [
        (3, (3,5,4,2), False),
        (5, (3,5,4,3,2), True),
        (4, (4,3,7,2), True),
        (6, (4,5,3,2), False),
    ]
    # Prespecified medium-size exploratory schedules; source validation and
    # actual Vitis measurement are different levels of evidence.
    paper_configs = [
        (3, (3,5,4,2), False),
        (5, (3,5,4,3,2), True),
        (8, (16,32,32,16), True),
        (16, (32,32,64,16), False),
    ]
    configs = (paper_configs if shape_catalog == "paper" else pilot_configs)[:max_cases//3]
    proposed = []
    for group, (rows, widths, hetero) in enumerate(configs):
        graph = make_chain(rows, widths)
        base = ArchitectureRequest()
        overrides = ({
            op.name: replace(base, pes=(i%3)+1, simd=(i%3)+1,
                             serialization=(i%2)+1, parallelism=(i%2)+1)
            for i, op in enumerate(graph.operators)
        } if hetero else {})
        arch = resolve(graph, base, overrides)
        x = np.linspace(-0.125, 0.125, rows*widths[0],
                        dtype=np.float32).reshape(rows, widths[0])
        for block in (1, 2, rows):
            compiled = compile_row_chain(arch, row_block=block)
            proof = compiled.evidence["schedule_witness"]
            if proof["status"] != "VERIFIED":
                raise AssertionError("Native case has unqualified schedule")
            proposed.append((group, rows, widths, hetero, block,
                             graph, x, compiled))
    # No Vitis/RTL work here.
    dest.mkdir(parents=True)
    records = []
    for i, (group, rows, widths, hetero, block, graph, x, compiled) in enumerate(proposed):
        case_name = f"case_{i:03d}"
        case = dest/case_name
        pkg = write_vitis_package(
            str(case), graph, compiled.emission, x,
            part=part, clock_ns=clock_ns,
            tolerance=tolerance,
            clock_uncertainty_ns=clock_uncertainty_ns,
        )
        records.append({
            "case": case_name, "group": group,
            "mapping_mode": "heterogeneous" if hetero else "uniform",
            "variant": f"row_block_{block}",
            "row_block": block,
            "model_shape": list(graph.tensor(graph.inputs[0]).shape),
            "operator_count": len(graph.operators),
            "functional_sha256": pkg["functional_sha256"],
            "cpp_sha256": pkg["cpp_sha256"],
            "testbench_sha256": pkg["testbench_sha256"],
            "tcl_sha256": file_hash(case/"run_hls.tcl"),
            "edge_contract_sha256": compiled.evidence["contract_sha256"],
            "schedule_sha256": compiled.evidence["schedule_witness"]["schedule_sha256"],
            "declared_scratch_elements": compiled.emission.manifest[
                "source_intermediate_scratch_elements"],
            "requested_clock_ns": clock_ns,
            "clock_uncertainty_ns": clock_uncertainty_ns,
            "numeric_tolerance": tolerance,
            "original_vitis_run": False,
            "expected_evidence": "E1-generated-source-only",
        })
    for group in range(max_cases//3):
        subset = [x for x in records if x["group"] == group]
        if len(subset) != 3 or len({x["functional_sha256"] for x in subset}) != 1:
            raise AssertionError("Unmatched HLS schedule study")
        if len({x["cpp_sha256"] for x in subset}) != 3:
            raise AssertionError("Block schedules did not lower to distinct source")
        if not (subset[0]["declared_scratch_elements"]
                < subset[1]["declared_scratch_elements"]
                < subset[2]["declared_scratch_elements"]):
            raise AssertionError("Block sizes must map to monotone source scratch")
    manifest = {
        "schema": SCHEMA,
        "study_family": "row_block_schedule_v1",
        "shape_catalog": shape_catalog,
        "revision": revision(),
        "original_native_tool_executed_by_preparation": False,
        "part": part, "clock_ns": clock_ns,
        "clock_uncertainty_ns": clock_uncertainty_ns,
        "tolerance": tolerance, "cases": records,
        "warnings": [
            "This is matched row-block source, not native hardware measurement",
            "Separate from RL and earlier whole/column/row experiment",
            "HLS clock target does not establish physical Fmax",
            "Scratch array size is not measured BRAM",
        ],
    }
    (dest/"compiler_ir_native_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)+"\n")
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--max-cases", type=int, default=6)
    p.add_argument("--shape-catalog", choices=("pilot", "paper"), default="pilot")
    p.add_argument("--part", default="xck26-sfvc784-2LV-c")
    p.add_argument("--clock-ns", type=float, default=32.0)
    p.add_argument("--clock-uncertainty-ns", type=float, default=8.64)
    p.add_argument("--tolerance", type=float, default=0.05)
    a = p.parse_args(argv)
    print(json.dumps(prepare(output=a.output, max_cases=a.max_cases,
                             part=a.part, clock_ns=a.clock_ns,
                             clock_uncertainty_ns=a.clock_uncertainty_ns,
                             tolerance=a.tolerance,
                             shape_catalog=a.shape_catalog), indent=2))


if __name__ == "__main__":
    main()
