"""Original host GCC validation of the prospective row-retained fanout arm.

Verifies ALL intermediate and final NumPy results, full source event
order, Phi issue order, source slot alias legality, and actual-address
witnesses. Results are E1 host only; no Vitis/physical implications.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

import numpy as np

from fpgai_research.codegen.hls.row_retained_fanout_cpp import (
    SCHEMA as EMITTER_SCHEMA, row_retained_plan,
)
from fpgai_research.experiments.cst_r2f_fanout_campaign import (
    project_source_value_events,
)
from fpgai_research.model_ir.element_scratch_allocation import verify_allocation
from fpgai_research.model_ir.fanout_contract import (
    admit_fanout, disjoint_fanout_offsets,
    verify_fanout_events, verify_fanout_issue,
)
from fpgai_research.model_ir.semantics import (
    architecture_fingerprint, functional_fingerprint,
)
from fpgai_research.validation.r2b_interleaved_cpp import _float_literal
from fpgai_research.validation.r2h_allocated_fanout_cpp import (
    decode_address_witnesses,
)
from fpgai_research.validation.reference import evaluate

SCHEMA = "fpgai-research/first-paper-row-retained-original-GCC-E1/v1"


def verify_row_retained_gcc(
    arch, emission, *, runtime_input, cxx: str = "g++",
    timeout_seconds: int = 120, original_trace_output: str | Path | None = None,
    atol: float = 1e-5, rtol: float = 1e-4,
) -> dict:
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 240:
        raise ValueError("Bounded row-retained GCC timeout required")
    if original_trace_output is not None:
        dest = Path(original_trace_output)
        if dest.exists() or dest.is_symlink() or not dest.parent.is_dir():
            raise FileExistsError("Row-retained GCC trace must be create-only")
    compiler = shutil.which(cxx)
    if compiler is None:
        raise FileNotFoundError("Real C++ compiler required for E1 experiment")

    spec = admit_fanout(arch)
    plan = row_retained_plan(arch)
    m, k, n, q = spec.m, spec.k, spec.n, spec.q
    sha = sha256(emission.cpp.encode()).hexdigest()
    manifest = emission.manifest
    expected_slot_rows = tuple(sorted(
        (tensor, i, address)
        for (tensor, i), address in plan["source_slot_map"].items()))
    if (
        manifest.get("schema") != EMITTER_SCHEMA
        or manifest.get("source_cpp_sha256") != sha
        or manifest.get("functional_sha256")
            != functional_fingerprint(arch.functional)
        or manifest.get("architecture_sha256")
            != architecture_fingerprint(arch)
        or manifest.get("source_slot_map_sha256")
            != plan["source_slot_map_sha256"]
        or tuple(map(tuple, manifest.get("source_slot_map", ())))
            != expected_slot_rows
        or manifest.get("source_event_order_sha256")
            != plan["source_event_sha256"]
        or manifest.get("source_scratch_array_elements")
            != plan["source_scratch_array_elements"]
        or manifest.get("source_H_row_buffer_elements") != n
    ):
        raise ValueError("Row-retained C++ source/contract SHA mismatch")
    x = np.asarray(runtime_input, dtype=np.float32)
    if x.shape != (m, k) or not np.all(np.isfinite(x)):
        raise ValueError("Finite float32 input of exact matrix shape required")
    original = evaluate(arch.functional, {spec.input: x}, return_all=True)
    input_literals = ",".join(_float_literal(float(v)) for v in x.flat)
    runner = (
        "\nint main() {\n"
        f"  float input[{m*k}]={{{input_literals}}};\n"
        f"  float output[{m*q}]={{}};\n"
        "  fpgai_model(input,output);\n"
        f'  for(int i=0;i<{m*q};++i) '
        'std::printf("FPGAI_R2G OUTPUT %d %.9g\\n",i,double(output[i]));\n'
        "  return 0;\n}\n"
    )
    with TemporaryDirectory(prefix="first_paper_row_retained_gcc_") as temp:
        cpp_path = Path(temp) / "source.cpp"
        bin_path = Path(temp) / "run"
        cpp_path.write_text(emission.cpp + runner)
        build = subprocess.run([
            compiler, "-std=c++17", "-O0", "-ffp-contract=off",
            "-DFPGAI_HOST_FLOAT", "-DFPGAI_TRACE_R2G",
            str(cpp_path), "-o", str(bin_path),
        ], capture_output=True, text=True, timeout=timeout_seconds)
        if build.returncode:
            raise RuntimeError("Row-retained GCC compilation failure: " +
                               build.stderr[-4000:])
        process = subprocess.run([str(bin_path)], capture_output=True,
                                 text=True, timeout=timeout_seconds)
        if process.returncode:
            raise RuntimeError("Row-retained GCC execution failure: " +
                               process.stderr[-4000:])

    events, values, outputs, addresses = decode_address_witnesses(
        process.stdout, slot_map=plan["source_slot_map"],
        capacity=plan["source_scratch_array_elements"],
    )
    if events != plan["source_events"]:
        raise AssertionError("Original executed source order differs from certified row schedule")
    logical = verify_fanout_events(arch, disjoint_fanout_offsets(spec), events)
    issue = verify_fanout_issue(arch, events)
    allocated = verify_allocation(
        project_source_value_events(events), plan["source_slot_map"],
        unit_slots=plan["source_scratch_array_elements"],
    )
    if (
        logical["status"] != "VERIFIED_E0"
        or issue["status"] != "VERIFIED_E0"
        or allocated["status"] != "EXACT_SOURCE_SLOT_REPLAY_VERIFIED"
    ):
        raise AssertionError("Original generated source violated a contract")

    errors = {}
    expected_keys = set()
    for tensor in spec.intermediates:
        reference = np.asarray(original[tensor], dtype=np.float32).reshape(-1)
        keys = [(tensor, index) for index in range(reference.size)]
        expected_keys.update(keys)
        actual = np.asarray([values.get(key, np.nan) for key in keys],
                            dtype=np.float64)
        if not np.allclose(actual, reference, atol=atol, rtol=rtol):
            raise AssertionError(
                "Original row-retained intermediate numerical mismatch: " + tensor)
        errors[tensor] = float(np.max(np.abs(actual - reference)))
    if set(values) != expected_keys or set(outputs) != set(range(m*q)):
        raise AssertionError("Original generated source omitted/duplicated values")
    output = np.asarray([outputs[i] for i in range(m*q)], dtype=np.float64)
    reference = np.asarray(
        original[arch.functional.outputs[0]], dtype=np.float32).reshape(-1)
    if not np.allclose(output, reference, atol=atol, rtol=rtol):
        raise AssertionError("Original row-retained final numerical mismatch")
    if original_trace_output is not None:
        with Path(original_trace_output).open("x", encoding="utf-8") as file:
            file.write(process.stdout)
    return {
        "schema": SCHEMA,
        "status": "ORIGINAL_E1_ROW_RETAINED_HOST_CPP_NUMERIC_AND_SOURCE_CERTIFIED",
        "functional_sha256": manifest["functional_sha256"],
        "architecture_sha256": manifest["architecture_sha256"],
        "source_cpp_sha256": sha,
        "runtime_input_sha256": sha256(x.tobytes()).hexdigest(),
        "runtime_stdout_sha256": sha256(process.stdout.encode()).hexdigest(),
        "original_source_event_sha256": logical["source_event_sha256"],
        "source_event_count": len(events),
        "original_address_witnesses": addresses,
        "all_intermediate_max_absolute_error": max(errors.values()),
        "final_max_absolute_error": float(np.max(np.abs(output-reference))),
        "H_buffer_source_elements": n,
        "H_full_source_elements": m*n,
        "source_saved_scratch_elements": (m-1)*n,
        "total_mutable_array_elements":
            manifest["source_all_declared_mutable_array_elements"],
        "original_GCC_compilations": 1,
        "original_GCC_executions": 1,
        "original_Vitis_calls": 0,
        "original_Vivado_calls": 0,
        "HLS_BRAM_result": None,
        "physical_FPGA_result": None,
        "limitations": [
            "Host float32 C++ testing of bounded static MatMul fanout",
            "Printed address witness is source-local, not binary memory instrumentation",
            "Cannot infer native HLS II/timing, BRAM, bank ports or real concurrency",
        ],
    }
