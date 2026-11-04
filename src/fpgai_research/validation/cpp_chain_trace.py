"""Independent executed-C++ event verifier for multi-operator row-chain IR.

Checks source-emitted row-epoch read-after-production and scratch lifetime
boundaries for every internal tensor using actually compiled host C++ logs.
Not an FPGA cycle schedule, RTL equivalence or source proof in general.
"""
from __future__ import annotations

from hashlib import sha256
from math import prod
from pathlib import Path
from tempfile import TemporaryDirectory
import re
import shutil
import subprocess

from fpgai_research.model_ir.row_chain import RowChainIR
from fpgai_research.model_ir.temporal_schedule import blocked_schedule
from fpgai_research.model_ir.semantics import functional_fingerprint

RECORD = re.compile(r"FPGAI_CHAIN_ROW_(READY|CONSUMED) ([0-9]+) ([0-9]+)\Z")


def trace_chain_epochs(ir: RowChainIR, emission, *,
                       cxx: str = "g++", timeout_seconds: int = 60) -> dict:
    if not isinstance(ir, RowChainIR):
        raise TypeError("Expected typed RowChainIR")
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 180:
        raise ValueError("Invalid source execution timeout")
    graph = ir.composed.lowered.architectural.functional
    if emission.manifest.get("functional_sha256") != functional_fingerprint(graph):
        raise ValueError("Executable C++ does not match the Functional IR")
    if emission.manifest.get("cpp_sha256") != sha256(emission.cpp.encode()).hexdigest():
        raise ValueError("C++ SHA differs from emission manifest")
    if len(graph.inputs) != 1 or len(graph.outputs) != 1:
        raise ValueError("Row-chain source needs exactly one graph input/output")
    compiler = shutil.which(cxx)
    if not compiler:
        raise FileNotFoundError("Required C++ host compiler missing")
    count_in = prod(graph.tensor(graph.inputs[0]).shape)
    count_out = prod(graph.tensor(graph.outputs[0]).shape)
    runner = (f"\nint main() {{ float a[{count_in}] = {{}}; "
              f"float y[{count_out}] = {{}}; fpgai_model(a,y); return 0; }}\n")
    with TemporaryDirectory(prefix="fpgai_chain_trace_") as folder:
        source = Path(folder)/"run.cpp"
        binary = Path(folder)/"run"
        source.write_text(emission.cpp + runner)
        built = subprocess.run([
            compiler, "-std=c++17", "-O0", "-DFPGAI_HOST_FLOAT",
            "-DFPGAI_TRACE_ROWS", "-Wno-unknown-pragmas",
            str(source), "-o", str(binary),
        ], capture_output=True, text=True, timeout=timeout_seconds)
        if built.returncode:
            raise RuntimeError("Source chain trace C++ build failed: " + built.stderr[-2400:])
        ran = subprocess.run([str(binary)], capture_output=True, text=True,
                             timeout=timeout_seconds)
        if ran.returncode:
            raise RuntimeError("Row chain host executable failed")
    events = []
    for line in ran.stdout.splitlines():
        match = RECORD.fullmatch(line)
        if not match:
            raise AssertionError("Unknown or malformed row-chain trace output: " + line[:120])
        events.append((match.group(1), int(match.group(2)), int(match.group(3))))
    row_block = emission.manifest.get("row_block", 1)
    if type(row_block) is not int or not 1 <= row_block <= ir.row_count:
        raise ValueError("Invalid emitted row-block contract")
    expected = []
    for event in blocked_schedule(ir.row_count, len(ir.chain), row_block):
        row, stage = event.row, event.stage
        if stage < len(ir.chain)-1:
            expected.append(("READY", stage, row))
        if stage > 0:
            expected.append(("CONSUMED", stage-1, row))
    if events != expected:
        mismatch = next((i for i, (actual, want) in enumerate(zip(events, expected))
                         if actual != want), min(len(events), len(expected)))
        raise AssertionError(
            "CST_CHAIN_LIFETIME_VIOLATION: "
            f"event {mismatch}: got {events[mismatch:mismatch+3]}, "
            f"expected {expected[mismatch:mismatch+3]}"
        )
    return {
        "schema": "fpgai-research/actual-cxx-row-chain-epochs/v1",
        "cpp_sha256": emission.manifest["cpp_sha256"],
        "graph_sha256": emission.manifest["functional_sha256"],
        "checked_rows": ir.row_count,
        "checked_edges": len(ir.chain)-1,
        "emitted_row_block": row_block,
        "actual_event_count": len(events),
        "producer_ready_before_consumer_finished": True,
        "each_scratch_reused_only_across_disjoint_row_epochs": True,
        "host_trace_sha256": sha256(ran.stdout.encode()).hexdigest(),
        "evidence": "E1-executed-host-C++-row-chain-only",
        "vitis_executed": False,
        "physical_scheduling_proven": False,
    }
