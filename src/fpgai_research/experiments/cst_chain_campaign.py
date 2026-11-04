"""Deterministic compiler IR row-chain E0/E1 study on 3- and 4-layer models.

Runs no original Vitis, RL or other vendor tools. The output directory is
create-only; every source fingerprint, numerical host receipt, and row-chain
event trace is collected from a real generated compiler artifact.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import subprocess

import numpy as np

from fpgai_research.architecture.resolver import resolve
from fpgai_research.compiler.row_chain import compile_row_chain
from fpgai_research.codegen.hls.emitter import emit_hls
from fpgai_research.model_ir.contracts import ArchitectureRequest, FunctionalModel, Operator, Tensor
from fpgai_research.model_ir.lowering import lower
from fpgai_research.validation.host import validate_host
from fpgai_research.validation.cpp_chain_trace import trace_chain_epochs
from fpgai_research.validation.cpp_phi_trace import trace_cpp_phi


def make_chain(rows: int, widths: tuple[int, ...]) -> FunctionalModel:
    if type(rows) is not int or rows < 2 or not 3 <= len(widths) <= 5:
        raise ValueError("Chain CPU campaign supports 3..5 logical widths and >=2 rows")
    if any(type(n) is not int or n < 2 for n in widths):
        raise ValueError("All chain widths must be >=2")
    tensors = [Tensor("x", (rows, widths[0]))]
    operators = []
    previous = "x"
    for index, (k, n) in enumerate(zip(widths[:-1], widths[1:])):
        wname = f"w{index}"
        output = "y" if index == len(widths)-2 else f"h{index}"
        tensors.extend((
            Tensor(wname, (k, n), data=tuple(0.025*((j%11)-5) for j in range(k*n))),
            Tensor(output, (rows, n)),
        ))
        operators.append(Operator(f"mm{index}", "MatMul",
                                  (previous, wname), (output,)))
        previous = output
    return FunctionalModel(tuple(tensors), tuple(operators), ("x",), ("y",))


def _architecture(model, mode: str):
    base = ArchitectureRequest()
    if mode == "uniform":
        return resolve(model, base)
    if mode == "heterogeneous":
        choices = {
            op.name: replace(
                base,
                pes=index+1,
                simd=index+2,
                serialization=(index%2)+1,
                parallelism=(index%2)+1,
            ) for index, op in enumerate(model.operators)
        }
        return resolve(model, base, choices)
    raise ValueError("Unknown architecture mode")


def run(*, output: str | Path, max_cases: int = 12,
        host_cases: int = 4, full_trace_cases: int = 2) -> dict:
    if type(max_cases) is not int or not 1 <= max_cases <= 36:
        raise ValueError("max_cases must be 1..36")
    if any(type(v) is not int or v < 0 for v in (host_cases, full_trace_cases)):
        raise ValueError("Host and trace budgets must be nonnegative integers")
    if host_cases > max_cases or full_trace_cases > max_cases:
        raise ValueError("Host and trace budgets exceed campaign bound")
    dest = Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Create-only row-chain study already exists")
    catalog = [
        (rows, widths, mode)
        for rows in (2, 3, 5)
        for widths in ((3, 5, 4, 2), (3, 5, 4, 3, 2),
                       (4, 3, 7, 2), (4, 5, 3, 2))
        for mode in ("uniform", "heterogeneous")
    ][:max_cases]
    records = []
    for case_id, (rows, widths, mode) in enumerate(catalog):
        model = make_chain(rows, widths)
        arch = _architecture(model, mode)
        normal = emit_hls(lower(arch))
        chained = compile_row_chain(arch)
        if normal.manifest["functional_sha256"] != chained.emission.manifest["functional_sha256"]:
            raise AssertionError("Mismatched Functional Model IR across chain architectures")
        if normal.manifest["cpp_sha256"] == chained.emission.manifest["cpp_sha256"]:
            raise AssertionError("Row-chain must actually lower to distinct source")
        proof = chained.evidence["independent_liveness_witness"]
        if proof["status"] != "EXHAUSTIVE_BOUNDED_LIVENESS_VERIFIED":
            raise AssertionError("No valid independent cross-operator value lifetime")
        row = {
            "case": f"case_{case_id:03d}",
            "rows": rows,
            "operator_count": len(model.operators),
            "widths": "x".join(map(str, widths)),
            "mapping_mode": mode,
            "functional_sha256": normal.manifest["functional_sha256"],
            "whole_tensor_cpp_sha256": normal.manifest["cpp_sha256"],
            "row_chain_cpp_sha256": chained.emission.manifest["cpp_sha256"],
            "source_scratch_elements": chained.ir.physical_scratch_elements,
            "whole_intermediate_elements": chained.ir.full_intermediate_elements,
            "source_buffer_ratio": rows,
            "independent_liveness_pass": proof["all_live_values_released_each_row"],
            "liveness_event_sha256": proof["event_sha256"],
            "host_checked": False,
            "host_numeric_pass": None,
            "host_max_abs_error": None,
            "row_epoch_checked": False,
            "phi_trace_checked": False,
            "original_vitis_calls": 0,
        }
        if case_id < host_cases:
            x = np.linspace(-0.125, 0.125, rows*widths[0],
                            dtype=np.float32).reshape(rows, widths[0])
            control = validate_host(model, normal, x)
            actual = validate_host(model, chained.emission, x)
            row["host_checked"] = True
            row["host_numeric_pass"] = control["accepted"] and actual["accepted"]
            row["host_max_abs_error"] = actual["max_absolute_error"]
            if not row["host_numeric_pass"]:
                raise AssertionError("Compiled row-chain source disagrees with reference")
        if case_id < full_trace_cases:
            event = trace_chain_epochs(chained.ir, chained.emission)
            phi = trace_cpp_phi(chained.ir.composed.lowered,
                                emission=chained.emission)
            row["row_epoch_checked"] = event["producer_ready_before_consumer_finished"]
            row["phi_trace_checked"] = phi["source_phi_agreement"]
            if not (row["row_epoch_checked"] and row["phi_trace_checked"]):
                raise AssertionError("Failed actual C++ temporal/Phi evidence")
        records.append(row)
    dest.mkdir(parents=True)
    with (dest/"cases.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    revision = subprocess.run(["git", "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=5)
    summary = {
        "schema": "fpgai-research/compositional-row-chain-cpu/v1",
        "source_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "cases": len(records),
        "chains_with_three_operators": sum(r["operator_count"] == 3 for r in records),
        "chains_with_four_operators": sum(r["operator_count"] == 4 for r in records),
        "host_checked": sum(r["host_checked"] for r in records),
        "host_numeric_passes": sum(r["host_numeric_pass"] is True for r in records),
        "phi_and_row_epoch_traces": sum(r["phi_trace_checked"] and r["row_epoch_checked"]
                                         for r in records),
        "max_source_buffer_ratio": max(r["source_buffer_ratio"] for r in records),
        "independent_liveness_verified_cases": sum(
            r["independent_liveness_pass"] for r in records),
        "vitis_calls": 0,
        "evidence": "E0-typed-row-chain-E1-source-optional-GCC",
        "not_evidence_of": ["RTL schedule", "physical BRAM", "Fmax",
                            "hardware speedup", "prior-art scientific novelty"],
    }
    (dest/"summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n")
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--max-cases", type=int, default=12)
    p.add_argument("--host-cases", type=int, default=4)
    p.add_argument("--full-trace-cases", type=int, default=2)
    a = p.parse_args(argv)
    print(json.dumps(run(output=a.output, max_cases=a.max_cases,
                         host_cases=a.host_cases,
                         full_trace_cases=a.full_trace_cases),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
