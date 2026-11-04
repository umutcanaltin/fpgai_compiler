"""Execute generated row-interleaved source and verify actual row event ordering.

E1 independent host-C++ evidence only. The emitter prints row events under
explicit FPGAI_TRACE_ROWS; this verifier does not trust emitter manifests as
evidence that the events actually occurred. It compiles and executes C++.
"""
from __future__ import annotations

from hashlib import sha256
from math import prod
from pathlib import Path
from tempfile import TemporaryDirectory
import re
import shutil
import subprocess

from fpgai_research.model_ir.row_lifetime import RowInterleavedModelIR
from fpgai_research.model_ir.semantics import functional_fingerprint


_EVENT = re.compile(r"FPGAI_ROW_(READY|CONSUMED) ([0-9]+)\Z")


def trace_row_epochs(ir: RowInterleavedModelIR, emission, *,
                     cxx: str = "g++", timeout_seconds: int = 45) -> dict:
    """Check producer-ready and consumer-finished order from executed C++."""
    if not isinstance(ir, RowInterleavedModelIR):
        raise TypeError("Expected typed RowInterleavedModelIR")
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 180:
        raise ValueError("Invalid host trace timeout")
    if not emission.manifest.get("row_lifetime_contract"):
        raise ValueError("Row-lifetime source manifest required")
    if sha256(emission.cpp.encode()).hexdigest() != emission.manifest.get("cpp_sha256"):
        raise ValueError("Unbound generated C++ source")
    model = ir.base.lowered.architectural.functional
    if emission.manifest.get("functional_sha256") != functional_fingerprint(model):
        raise ValueError("Generated source has different Functional Model IR")
    compiler = shutil.which(cxx)
    if not compiler:
        raise FileNotFoundError("C++ compiler unavailable")
    if len(model.inputs) != 1 or len(model.outputs) != 1:
        raise ValueError("Host trace requires one input and one output")
    inputs = prod(model.tensor(model.inputs[0]).shape)
    outputs = prod(model.tensor(model.outputs[0]).shape)
    runner = (f"\nint main() {{ float a[{inputs}] = {{}}; "
              f"float y[{outputs}] = {{}}; fpgai_model(a,y); return 0; }}\n")
    with TemporaryDirectory(prefix="fpgai_cst_row_trace_") as d:
        cpp = Path(d)/"row.cpp"
        exe = Path(d)/"row"
        cpp.write_text(emission.cpp + runner)
        build = subprocess.run(
            [compiler, "-std=c++17", "-O0", "-DFPGAI_HOST_FLOAT",
             "-DFPGAI_TRACE_ROWS", "-Wno-unknown-pragmas",
             str(cpp), "-o", str(exe)],
            capture_output=True, text=True, timeout=timeout_seconds)
        if build.returncode:
            raise RuntimeError("Generated row trace C++ did not compile: "
                               + build.stderr[-2000:])
        completed = subprocess.run(
            [str(exe)], capture_output=True, text=True,
            timeout=timeout_seconds)
        if completed.returncode:
            raise RuntimeError("Generated row trace executable failed")
    events = []
    for line in completed.stdout.splitlines():
        match = _EVENT.fullmatch(line)
        if not match:
            raise AssertionError("Unknown C++ row trace event: " + line[:120])
        events.append((match.group(1), int(match.group(2))))
    rows = ir.row_contract.shape[0]
    expected = [event for row in range(rows)
                for event in (("READY", row), ("CONSUMED", row))]
    if events != expected:
        mismatch = next((i for i, (a, b) in enumerate(zip(events, expected))
                         if a != b), min(len(events), len(expected)))
        raise AssertionError(
            "ROW_LIFETIME_ORDER_VIOLATION: "
            f"first mismatch at event {mismatch}; "
            f"observed={events[mismatch:mismatch+3]}, "
            f"expected={expected[mismatch:mismatch+3]}")
    return {
        "schema": "fpgai-research/row-lifetime-executed-cpp-events/v1",
        "cpp_sha256": emission.manifest["cpp_sha256"],
        "functional_sha256": emission.manifest["functional_sha256"],
        "row_count": rows,
        "actual_events": len(events),
        "row_ready_before_consumed": True,
        "scratch_reused_only_after_consumption": True,
        "trace_sha256": sha256(completed.stdout.encode()).hexdigest(),
        "evidence": "E1-executed-host-C++-temporal-order-only",
        "fixed_point_bit_exact": False,
        "native_vitis_executed": False,
        "achieved_scheduling_cycles_proven": False,
    }
