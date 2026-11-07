"""R4k real-host E1 verification of matched row-schedule storage layouts.

Checks actual emitted C++ for each arm with g++, the R4a certified
source event order, each adjacent source address witness, and every
intermediate plus final value versus independent Functional Model IR.
No Vitis, Vivado or physical claims. The original E1 arms are immutable.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

import numpy as np

from fpgai_research.codegen.hls.row_retained_fanout_cpp import (
    emit_row_retained_fanout_cpp, row_retained_plan,
)
from fpgai_research.codegen.hls.r4k_full_storage_row_schedule_cpp import (
    SCHEMA as FULL_SCHEMA, emit_full_storage_row_schedule_cpp,
    full_storage_row_schedule_plan,
)
from fpgai_research.experiments.cst_r2f_fanout_campaign import (
    project_source_value_events,
)
from fpgai_research.model_ir.element_scratch_allocation import verify_allocation
from fpgai_research.model_ir.fanout_contract import (
    admit_fanout, disjoint_fanout_offsets,
    verify_fanout_events, verify_fanout_issue,
)
from fpgai_research.validation.r2b_interleaved_cpp import _float_literal
from fpgai_research.validation.r2h_allocated_fanout_cpp import (
    decode_address_witnesses,
)
from fpgai_research.validation.reference import evaluate

SCHEMA = "fpgai-research/R4k-row-schedule-storage-controlled-original-E1/v1"


def _execute(arch, emission, *, runtime_input, plan: dict, original: dict,
             cxx: str, timeout_seconds: int, trace_output: Path | None) -> dict:
    spec = admit_fanout(arch)
    m, k, n, q = spec.m, spec.k, spec.n, spec.q
    digest = sha256(emission.cpp.encode()).hexdigest()
    if (digest != emission.manifest.get("source_cpp_sha256")
            or emission.manifest.get("source_event_order_sha256")
               != plan["source_event_sha256"]
            or emission.manifest.get("source_scratch_array_elements")
               != plan["source_scratch_array_elements"]
            or emission.manifest.get("source_slot_map_sha256")
               != plan["source_slot_map_sha256"]
            or tuple(map(tuple, emission.manifest.get("source_slot_map", ())))
               != tuple(sorted((t, i, a)
                    for (t, i), a in plan["source_slot_map"].items()))):
        raise ValueError("R4k original E1 emission source/slot/schedule mismatch")
    x = np.asarray(runtime_input, dtype=np.float32)
    if x.shape != (m, k) or not np.all(np.isfinite(x)):
        raise ValueError("R4k invalid host sample")
    runner = (
        "\nint main() {\n"
        f"  float input[{m*k}]={{{','.join(_float_literal(float(v)) for v in x.flat)}}};\n"
        f"  float output[{m*q}]={{}};\n"
        "  fpgai_model(input,output);\n"
        f'  for(int i=0;i<{m*q};++i) '
        'std::printf("FPGAI_R2G OUTPUT %d %.9g\\n",i,double(output[i]));\n'
        "  return 0;\n}\n"
    )
    with TemporaryDirectory(prefix="R4k_matched_source_E1_") as tmp:
        file = Path(tmp) / "model.cpp"
        exe = Path(tmp) / "source_run"
        file.write_text(emission.cpp + runner)
        compiled = subprocess.run(
            [cxx, "-std=c++17", "-O0", "-ffp-contract=off",
             "-DFPGAI_HOST_FLOAT", "-DFPGAI_TRACE_R2G",
             str(file), "-o", str(exe)],
            capture_output=True, text=True, timeout=timeout_seconds,
        )
        if compiled.returncode != 0:
            raise RuntimeError(
                "R4k real-host C++ compilation failed: " + compiled.stderr[-4000:])
        executed = subprocess.run(
            [str(exe)], capture_output=True, text=True,
            timeout=timeout_seconds,
        )
        if executed.returncode != 0:
            raise RuntimeError(
                "R4k real-host C++ execution failed: " + executed.stderr[-4000:])
    events, values, outputs, addresses = decode_address_witnesses(
        executed.stdout, slot_map=plan["source_slot_map"],
        capacity=plan["source_scratch_array_elements"],
    )
    expected_events = row_retained_plan(arch)["source_events"]
    if events != expected_events:
        raise AssertionError("R4k actual source issue order differed")
    proof = verify_fanout_events(
        arch, disjoint_fanout_offsets(spec), events)
    issue = verify_fanout_issue(arch, events)
    replay = verify_allocation(
        project_source_value_events(events), plan["source_slot_map"],
        unit_slots=plan["source_scratch_array_elements"],
    )
    if (proof["status"] != "VERIFIED_E0"
            or issue["status"] != "VERIFIED_E0"
            or replay["status"] != "EXACT_SOURCE_SLOT_REPLAY_VERIFIED"):
        raise AssertionError("R4k actual host source violated E0 ownership or issue")
    required = set()
    errors = {}
    for tensor in spec.intermediates:
        wanted = np.asarray(original[tensor], dtype=np.float32).reshape(-1)
        keys = [(tensor, i) for i in range(wanted.size)]
        required.update(keys)
        actual = np.asarray([values.get(item, float("nan")) for item in keys])
        if not np.allclose(actual, wanted, atol=1e-5, rtol=1e-4):
            raise AssertionError(
                "R4k intermediate numerical mismatch: " + tensor)
        errors[tensor] = float(np.max(np.abs(actual - wanted)))
    wanted = np.asarray(
        original[arch.functional.outputs[0]], dtype=np.float32).reshape(-1)
    output = np.asarray([outputs.get(i, float("nan"))
                         for i in range(m*q)])
    if (set(values) != required or set(outputs) != set(range(m*q))
            or not np.allclose(output, wanted, atol=1e-5, rtol=1e-4)):
        raise AssertionError("R4k missing or numerically wrong model outputs")
    if trace_output is not None:
        if trace_output.exists() or trace_output.is_symlink():
            raise FileExistsError("R4k original GCC trace must be create-only")
        if not trace_output.parent.is_dir():
            raise ValueError("R4k GCC trace destination parent is missing")
        with trace_output.open("x", encoding="utf-8") as f:
            f.write(executed.stdout)
    plain_trace = "\n".join(
        line for line in executed.stdout.splitlines()
        if not line.startswith("FPGAI_SLOT ")
    ) + "\n"
    return {
        "source_cpp_sha256": digest,
        "host_stdout_sha256": sha256(executed.stdout.encode()).hexdigest(),
        "logical_trace_without_addresses_sha256":
            sha256(plain_trace.encode()).hexdigest(),
        "events": events,
        "values": values,
        "outputs": outputs,
        "address_witnesses": addresses,
        "max_intermediate_error": max(errors.values()),
        "max_final_error": float(np.max(np.abs(output - wanted))),
    }


def verify_matched_gcc(arch, *, runtime_input,
                       trace_dir: str | Path | None = None,
                       cxx: str = "g++", timeout_seconds: int = 120) -> dict:
    """Compile/execute BOTH matched source arms; fail closed on any drift."""
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 240:
        raise ValueError("R4k bounded GCC timeout required")
    if shutil.which(cxx) is None:
        raise FileNotFoundError("R4k real host g++ required")
    spec = admit_fanout(arch)
    x = np.asarray(runtime_input, dtype=np.float32)
    if x.shape != (spec.m, spec.k) or not np.all(np.isfinite(x)):
        raise ValueError("R4k original input must match bounded fanout model")
    reference = evaluate(
        arch.functional, {spec.input: x}, return_all=True,
    )
    row_emission = emit_row_retained_fanout_cpp(arch)
    full_emission = emit_full_storage_row_schedule_cpp(arch)
    if (full_emission.manifest.get("schema") != FULL_SCHEMA
            or full_emission.manifest.get("base_row_retained_cpp_sha256")
               != row_emission.manifest["source_cpp_sha256"]):
        raise ValueError("R4k failed source lineage from row emitter")
    root = Path(trace_dir) if trace_dir is not None else None
    if root is not None and (not root.is_dir() or root.is_symlink()
                             or any((root / (arm + ".host_stdout.txt")).exists()
                                    for arm in ("full_storage", "row_storage"))):
        raise FileExistsError("R4k trace directory must preexist and be unused")
    arms = {
        "full_storage": (
            full_emission, full_storage_row_schedule_plan(arch)),
        "row_storage": (row_emission, row_retained_plan(arch)),
    }
    reports = {}
    for arm, (emission, plan) in arms.items():
        reports[arm] = _execute(
            arch, emission, runtime_input=x, plan=plan, original=reference,
            cxx=shutil.which(cxx), timeout_seconds=timeout_seconds,
            trace_output=(root / (arm + ".host_stdout.txt")
                          if root is not None else None),
        )
    a, b = reports["full_storage"], reports["row_storage"]
    if (a["events"] != b["events"]
            or a["values"] != b["values"]
            or a["outputs"] != b["outputs"]
            or a["logical_trace_without_addresses_sha256"]
               != b["logical_trace_without_addresses_sha256"]):
        raise AssertionError(
            "R4k storage-only source comparison altered source issue/value trace")
    return {
        "schema": SCHEMA,
        "status": "MATCHED_ROW_SOURCE_SCHEDULE_REAL_HOST_E1_ACCEPTED",
        "source_schedule_sha256": row_retained_plan(arch)["source_event_sha256"],
        "full_source_sha256": full_emission.manifest["source_cpp_sha256"],
        "row_source_sha256": row_emission.manifest["source_cpp_sha256"],
        "original_input_sha256": sha256(x.tobytes()).hexdigest(),
        "source_logic_trace_without_addresses_sha256":
            a["logical_trace_without_addresses_sha256"],
        "all_intermediates_numerically_equal": True,
        "all_final_outputs_numerically_equal": True,
        "full_source_scratch_elements":
            full_emission.manifest["source_scratch_array_elements"],
        "row_source_scratch_elements":
            row_emission.manifest["source_scratch_array_elements"],
        "source_scratch_elements_saved":
            (full_emission.manifest["source_scratch_array_elements"]
             - row_emission.manifest["source_scratch_array_elements"]),
        "original_host_gcc_compilations": 2,
        "original_host_gcc_executions": 2,
        "new_native_Vitis_calls": 0,
        "new_native_Vivado_calls": 0,
        "physical_E4": None,
        "no_hardware_QoR_claim": True,
    }
