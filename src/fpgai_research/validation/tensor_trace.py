"""Independent per-operator numerical conformance against *executed* HLS C++.

GCC executes generated source under FPGAI_HOST_FLOAT and FPGAI_TRACE_TENSORS.
This verifies functional preservation at each operator boundary, rather than
only the final model output. It is neither Vitis fixed-point simulation nor
proof of HLS resource or schedule realization.
"""
from __future__ import annotations

from hashlib import sha256
from math import isfinite, prod
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

import numpy as np

from fpgai_research.codegen.hls.emitter import emit_hls
from fpgai_research.model_ir.contracts import LoweredModel
from fpgai_research.model_ir.semantics import verify_refinement
from fpgai_research.validation.reference import evaluate


def _float_literal(value: float) -> str:
    # Nine decimal significant digits round-trip a finite binary32 value.
    text = f"{float(value):.9g}"
    if not any(ch in text for ch in ".eE"):
        text += ".0"
    return text + "f"


def check_tensor_trace(
    model: LoweredModel,
    runtime_input: np.ndarray,
    *,
    max_values: int = 20_000,
    absolute_tolerance: float = 1e-5,
    relative_tolerance: float = 1e-4,
    timeout_seconds: int = 45,
    compiler: str = "g++",
) -> dict:
    if type(max_values) is not int or max_values < 1:
        raise ValueError("max_values must be a positive integer")
    if type(timeout_seconds) is not int or timeout_seconds < 1:
        raise ValueError("timeout_seconds must be positive")
    if not (isfinite(absolute_tolerance) and isfinite(relative_tolerance)
            and absolute_tolerance >= 0 and relative_tolerance >= 0):
        raise ValueError("Numeric tolerances must be finite and nonnegative")
    verify_refinement(model)
    functional = model.architectural.functional
    if len(functional.inputs) != 1 or len(functional.outputs) != 1:
        raise ValueError("Tensor tracing requires one model input and output")
    x = np.asarray(runtime_input, dtype=np.float32)
    expected_shape = functional.tensor(functional.inputs[0]).shape
    if x.shape != expected_shape or not np.all(np.isfinite(x)):
        raise ValueError("Expected a finite, exactly shaped float32 sample")
    expected = evaluate(functional, {functional.inputs[0]: x}, return_all=True)
    quantities = [prod(functional.tensor(op.outputs[0]).shape) for op in functional.operators]
    total = sum(quantities)
    if not 0 < total <= max_values:
        raise ValueError(f"Operator trace size {total} exceeds requested bound {max_values}")
    executable = shutil.which(compiler)
    if executable is None:
        raise FileNotFoundError(f"C++ compiler unavailable: {compiler}")
    emission = emit_hls(model)
    initializers = ", ".join(_float_literal(value) for value in x.flat)
    output_size = prod(functional.tensor(functional.outputs[0]).shape)
    runner = (f'\nint main() {{ float x[{x.size}] = {{{initializers}}}; '
              f'float y[{output_size}] = {{}}; fpgai_model(x, y); return 0; }}\n')
    with TemporaryDirectory(prefix="fpgai_tensor_trace_") as temp:
        source = Path(temp) / "trace.cpp"
        binary = Path(temp) / "trace"
        source.write_text(emission.cpp + runner)
        build = subprocess.run([executable, "-std=c++17", "-O0", "-Wno-unknown-pragmas",
                                "-DFPGAI_HOST_FLOAT", "-DFPGAI_TRACE_TENSORS",
                                str(source), "-o", str(binary)],
                               shell=False, text=True, capture_output=True,
                               timeout=timeout_seconds, check=False)
        if build.returncode:
            raise RuntimeError("Host C++ tensor trace failed to compile: " + build.stderr[-4000:])
        run = subprocess.run([str(binary)], shell=False, text=True, capture_output=True,
                             timeout=timeout_seconds, check=False)
        if run.returncode:
            raise RuntimeError("Host C++ tensor trace failed to execute: " + run.stderr[-2000:])
        output = run.stdout
    actual = [np.empty(n, np.float64) for n in quantities]
    observed = [set() for _ in quantities]
    for line in output.splitlines():
        parts = line.split()
        if len(parts) != 4 or parts[0] != "FPGAI_TENSOR":
            raise ValueError("Unexpected C++ tensor trace output")
        try:
            op_idx, element_idx = int(parts[1]), int(parts[2])
            value = float(parts[3])
        except ValueError as exc:
            raise ValueError("Malformed tensor trace coordinate/value") from exc
        if not (0 <= op_idx < len(actual)) or not (0 <= element_idx < len(actual[op_idx])):
            raise ValueError("Out-of-domain C++ tensor trace")
        if element_idx in observed[op_idx]:
            raise ValueError("Duplicate C++ tensor observation")
        if not isfinite(value):
            raise ValueError("Nonfinite C++ tensor observation")
        observed[op_idx].add(element_idx)
        actual[op_idx][element_idx] = value
    if sum(map(len, observed)) != total:
        raise ValueError("Missing operator tensor observations")
    per_operator = []
    for ordinal, (op, values) in enumerate(zip(functional.operators, actual)):
        goal = np.asarray(expected[op.outputs[0]], dtype=np.float64).ravel()
        delta = values - goal
        pass_check = bool(np.allclose(values, goal, rtol=relative_tolerance,
                                       atol=absolute_tolerance))
        per_operator.append({"operator": op.name, "kind": op.kind,
                             "element_count": len(values), "accepted": pass_check,
                             "max_absolute_error": float(np.max(np.abs(delta))),
                             "rmse": float(np.sqrt(np.mean(delta * delta))),
                             "reference_sha256": sha256(goal.tobytes()).hexdigest()})
    return {
        "schema": "fpgai-research/operator-numerical-trace/v1",
        "evidence": "E1-executed-GCC-host-float-intermediate-tensors",
        "cpp_sha256": emission.manifest["cpp_sha256"],
        "input_sha256": sha256(x.tobytes()).hexdigest(),
        "raw_trace_sha256": sha256(output.encode("utf-8")).hexdigest(),
        "operator_count": len(per_operator), "total_values": total,
        "accepted": all(record["accepted"] for record in per_operator),
        "operators": per_operator,
        "native_vitis_executed": False, "fixed_point_bit_exact": False,
        "limitations": "GCC float-mode operator values are not Vitis ap_fixed numerical, RTL or FPGA evidence",
    }
