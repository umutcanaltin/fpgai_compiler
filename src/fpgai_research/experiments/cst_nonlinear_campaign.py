"""Controlled E1 study of *nonlinear* sequential CST Model IR lowering.

Static synthetic (NOT trained) Flatten→Gemm(bias)→ReLU→Gemm(bias) is used
to isolate operator-semantic retention across a bounded B-row schedule.
Every input/weight, graph and source hash is deterministic. Exact per-case
results and deliberately unexecuted host trials remain visible. This
does not substitute for the separate original-trained-MNIST experiment.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import csv
import json
from pathlib import Path
import subprocess

import numpy as np

from fpgai_research.architecture.resolver import resolve
from fpgai_research.compiler.row_chain import compile_row_chain
from fpgai_research.model_ir.contracts import ArchitectureRequest, FunctionalModel, Operator, Tensor
from fpgai_research.model_ir.semantics import functional_fingerprint
from fpgai_research.validation.host import validate_host
from fpgai_research.validation.reference import evaluate
from fpgai_research.validation.cpp_chain_trace import trace_chain_epochs


def synthetic_nonlinear_model(rows: int, *, trans_b: int) -> FunctionalModel:
    if type(rows) is not int or rows < 2:
        raise ValueError("Synthetic nonlinear model needs >=2 rows")
    if type(trans_b) is not int or trans_b not in (0, 1):
        raise ValueError("Gemm transB must be exactly 0 or 1")
    first = np.array([
        [.20, -.12, .04, -.19, .25, .31],
        [-.07, .13, -.27, .35, .02, -.11],
        [.21, -.16, .06, -.04, .17, -.33],
        [-.18, .28, .09, -.23, -.14, .05]], dtype=np.float32)
    second = np.array([
        [.13, -.23, .32, .10],
        [-.15, .29, -.09, .26]], dtype=np.float32)
    if not trans_b:
        first = first.T.copy()
        second = second.T.copy()
    attrs = (("transB",1),) if trans_b else ()
    return FunctionalModel(
        tensors=(
            Tensor("image", (rows,1,2,3)),
            Tensor("flat", (rows,6)),
            Tensor("w0", first.shape, data=tuple(first.flat)),
            Tensor("bias0", (4,), data=(.07,-.15,.13,-.04)),
            Tensor("linear0", (rows,4)),
            Tensor("activation", (rows,4)),
            Tensor("w1", second.shape, data=tuple(second.flat)),
            Tensor("bias1", (2,), data=(.11,-.08)),
            Tensor("output", (rows,2)),
        ),
        operators=(
            Operator("flatten","Flatten",("image",),("flat",),
                     attributes=(("axis",1),)),
            Operator("gemm0","Gemm",("flat","w0","bias0"),("linear0",),
                     attributes=attrs),
            Operator("relu0","Relu",("linear0",),("activation",)),
            Operator("gemm1","Gemm",("activation","w1","bias1"),
                     ("output",),attributes=attrs),
        ),
        inputs=("image",), outputs=("output",),
    )


def run(*, output: str | Path, host_cases: int = 6,
        trace_cases: int = 3) -> dict:
    if type(host_cases) is not int or not 0 <= host_cases <= 16:
        raise ValueError("host_cases must be 0..16")
    if type(trace_cases) is not int or not 0 <= trace_cases <= host_cases:
        raise ValueError("trace_cases must be 0..host_cases")
    target = Path(output)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Create-only nonlinear CST experiment already exists")
    records = []
    checked = traces = 0
    for rows in (2,3,5):
        for trans_b in (0,1):
            graph = synthetic_nonlinear_model(rows, trans_b=trans_b)
            default = ArchitectureRequest()
            arch = resolve(graph, default, {
                "gemm0": replace(default, pes=2, simd=2),
                "gemm1": replace(default, pes=2, simd=2),
            })
            runtime = np.linspace(-.9,.8,rows*6,dtype=np.float32).reshape(rows,1,2,3)
            intermediates = evaluate(graph, {"image":runtime}, return_all=True)
            pre_relu = intermediates["linear0"]
            if not (np.any(pre_relu < 0) and np.any(pre_relu > 0)):
                raise AssertionError("Prespecified nonlinear witness is not activated")
            for block in sorted({1,min(2,rows),rows}):
                produced = compile_row_chain(arch,row_block=block)
                emission = produced.emission
                if produced.evidence["schedule_witness"]["status"] != "VERIFIED":
                    raise AssertionError("Unsupported row schedule was emitted")
                item = {
                    "rows": rows, "trans_b": trans_b, "block": block,
                    "functional_sha256": functional_fingerprint(graph),
                    "cpp_sha256": emission.manifest["cpp_sha256"],
                    "source_scratch_elements": emission.manifest[
                        "source_intermediate_scratch_elements"],
                    "reference_nonlinearity_observed": True,
                    "constant_bias_preserved": ("bias1" in emission.cpp),
                    "actual_relu_lowered": "if (value < data_t(0))" in emission.cpp,
                    "gcc_host_checked": False, "gcc_host_numeric_accepted": None,
                    "gcc_host_max_absolute_error": None,
                    "executed_row_event_checked": False,
                    "original_vitis_calls": 0,
                }
                if len(records) < host_cases:
                    receipt = validate_host(graph, emission, runtime,
                                            atol=1e-5, rtol=1e-4)
                    item["gcc_host_checked"] = True
                    item["gcc_host_numeric_accepted"] = receipt["accepted"]
                    item["gcc_host_max_absolute_error"] = receipt["max_absolute_error"]
                    checked += 1
                    if not receipt["accepted"]:
                        raise AssertionError("Nonlinear source disagrees with independent NumPy reference")
                    if traces < trace_cases:
                        event = trace_chain_epochs(produced.ir, emission)
                        item["executed_row_event_checked"] = event[
                            "producer_ready_before_consumer_finished"]
                        if not item["executed_row_event_checked"]:
                            raise AssertionError("Nonlinear source event ordering failed")
                        traces += 1
                records.append(item)
    if len(records) != 16:
        raise AssertionError("Nonlinear trial matrix drifted from prespecified 16 cases")
    for rows in (2,3,5):
        for trans_b in (0,1):
            matches=[r for r in records if r["rows"]==rows and r["trans_b"]==trans_b]
            if len({r["functional_sha256"] for r in matches}) != 1:
                raise AssertionError("Nonlinear matched schedule graph changed")
            if len({r["cpp_sha256"] for r in matches}) != len(matches):
                raise AssertionError("Nonlinear B schedules must change actual generated source")
    revision = subprocess.run(["git","rev-parse","HEAD"],
                              capture_output=True,text=True,timeout=5)
    summary={
        "schema":"fpgai-research/nonlinear-cst-host-campaign/v1",
        "source_revision":revision.stdout.strip() if revision.returncode==0 else None,
        "generated_cases":len(records),
        "fixed_functional_graphs":6,
        "rows":[2,3,5],
        "full_semantic_operator_chain":["Flatten","Gemm","Relu","Gemm"],
        "numerical_oracle":"Independent Functional Model IR NumPy float32 evaluator",
        "selected_host_checks":checked,
        "host_numerical_passes":sum(r["gcc_host_numeric_accepted"] is True for r in records),
        "executed_row_event_passes":sum(r["executed_row_event_checked"] for r in records),
        "trained_weights_used":False,
        "original_vitis_calls":0,
        "not_evidence_of":["real trained MNIST classification accuracy",
                           "fixed-point Vitis CSim", "CSynth resources",
                           "hardware Fmax", "novelty against external compilers"],
    }
    target.mkdir(parents=True)
    with (target/"cases.csv").open("w",newline="") as out:
        writer=csv.DictWriter(out,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (target/"summary.json").write_text(
        json.dumps(summary,indent=2,sort_keys=True)+"\n")
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True)
    p.add_argument("--host-cases",type=int,default=6)
    p.add_argument("--trace-cases",type=int,default=3)
    a=p.parse_args(argv)
    print(json.dumps(run(output=a.output,host_cases=a.host_cases,
                         trace_cases=a.trace_cases),indent=2))


if __name__=="__main__":
    main()
