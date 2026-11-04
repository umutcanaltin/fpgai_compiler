"""Small deterministic paper experiment: invariance vs operator-local specialization.

This is an actual two-operator computation, not a synthetic resource estimate.
No vendor tool or FPGA execution is launched.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.emitter import emit_hls
from fpgai_research.model_ir.artifacts import compare_ir_artifacts, write_ir_artifact
from fpgai_research.model_ir.contracts import ArchitectureRequest, FunctionalModel, Operator, Tensor
from fpgai_research.model_ir.lowering import lower
from fpgai_research.validation.reference import evaluate
from fpgai_research.validation.host import validate_host


def _study_fixture(seed: int):
    rng = np.random.default_rng(seed)
    w0 = tuple(map(float, rng.uniform(-.25, .25, 16)))
    w1 = tuple(map(float, rng.uniform(-.25, .25, 12)))
    model = FunctionalModel(
        (Tensor("input", (1, 4)), Tensor("w0", (4, 4), data=w0),
         Tensor("hidden", (1, 4)), Tensor("w1", (4, 3), data=w1),
         Tensor("output", (1, 3))),
        (Operator("layer0", "MatMul", ("input", "w0"), ("hidden",)),
         Operator("layer1", "Gemm", ("hidden", "w1"), ("output",))),
        ("input",), ("output",),
    )
    x = rng.uniform(-.5, .5, size=(1, 4)).astype(np.float32)
    return model, x


def run_refinement_ablation(output_directory, *, seed=2026, host_cpp=False) -> dict:
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an unsigned 32-bit integer")
    if type(host_cpp) is not bool:
        raise ValueError("host_cpp must be Boolean")
    directory = Path(output_directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Refusing to replace existing Model IR study artifacts")
    model, values = _study_fixture(seed)
    baseline = ArchitectureRequest()
    specialized = ArchitectureRequest(pes=2, simd=2, serialization=2, tile=(1, 3, 3))
    results = {}
    evidence = {}
    for label, override in (("baseline", {}), ("operator_local", {"layer0": specialized})):
        architecture = resolve(model, baseline, override)
        lowered = lower(architecture)
        # Intentionally preserve the exact operator graph across both versions.
        emitted = emit_hls(lowered)
        if host_cpp:
            host = validate_host(model, emitted, values)
            if not host["accepted"]:
                raise AssertionError(f"Generated C++ numerical mismatch in {label}")
            evidence[label] = {
                "accepted": bool(host["accepted"]),
                "rmse": float(host["rmse"]),
                "max_absolute_error": float(host["max_absolute_error"]),
                "cpp_sha256": host["cpp_sha256"],
            }
        results[label] = lowered
    directory.mkdir(parents=True, exist_ok=True)
    base_receipt = write_ir_artifact(directory / "baseline_ir.json", results["baseline"])
    local_receipt = write_ir_artifact(directory / "operator_local_ir.json", results["operator_local"])
    compared = compare_ir_artifacts(base_receipt, local_receipt)
    expected = evaluate(model, {"input": values})["output"]
    receipt = {
        "schema": "fpgai-research/refinement-ablation/v1",
        "seed": seed,
        "functional_sha256": compared["functional_sha256"],
        "inputs_sha256": sha256(values.tobytes()).hexdigest(),
        "reference_output_sha256": sha256(expected.tobytes()).hexdigest(),
        "comparison": compared,
        "host_cpp": evidence if host_cpp else None,
        "evidence": "E0/E1" + ("+GCC-host-float" if host_cpp else ""),
        "vendor_tools_executed": False,
        "not_fpga_qor": True,
        "limitations": "Finite deterministic ablation; no Vitis fixed-point, achieved II or physical resource evidence",
    }
    (directory / "study.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt
