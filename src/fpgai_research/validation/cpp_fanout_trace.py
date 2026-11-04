"""Actually execute restricted generated diamond C++ and replay its read trace.

Unlike E0 producer-acknowledgment assertions, source lines are emitted by
compiled C++ at the concrete scratch reads/writes. This checks emitted
execution order, alias release and source hash. It does NOT prove true
hardware ports, original Vitis scheduling or Phi-conditioned concurrency.
"""
from __future__ import annotations

from hashlib import sha256
from math import prod
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

from fpgai_research.codegen.hls.fanout_diamond import (
    SCHEMA as SOURCE_SCHEMA, source_order_events,
)
from fpgai_research.model_ir.contracts import FunctionalModel, ArchitecturalModel
from fpgai_research.model_ir.fanout_lifetime import (
    LifetimeEvent, verify_fanout_events,
)
from fpgai_research.model_ir.semantics import functional_fingerprint

SCHEMA = "fpgai-research/cst-actual-cpp-fanout-read-write-e1/v1"


def trace_fanout_cpp(
    model: FunctionalModel,
    emission,
    *,
    cxx: str = "g++",
    max_events: int = 20_000,
    timeout_seconds: int = 60,
    architecture: ArchitecturalModel | None = None,
) -> dict:
    if (type(max_events) is not int or max_events < 1 or
            type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 180):
        raise ValueError("Positive bounded event and compiler timeout required")
    if (emission.manifest.get("schema") != SOURCE_SCHEMA or
            emission.manifest.get("functional_sha256") != functional_fingerprint(model) or
            emission.manifest.get("cpp_sha256") != sha256(emission.cpp.encode()).hexdigest()):
        raise ValueError("Source manifest or Functional IR hash mismatch")
    mapped=emission.manifest.get("mapped_phi_enabled",False)
    if mapped:
        from fpgai_research.model_ir.fanout_phi_source import (
            mapped_source_order_events, phi_producer_iteration_order,
        )
        from fpgai_research.model_ir.semantics import architecture_fingerprint
        if (not isinstance(architecture,ArchitecturalModel) or
                architecture.functional != model or
                architecture_fingerprint(architecture) !=
                emission.manifest.get("architecture_sha256")):
            raise ValueError("Mapped C++ requires the exact resolved Phi architecture")
        expected=mapped_source_order_events(architecture,max_events=max_events)
        expected_iterations=phi_producer_iteration_order(
            architecture,max_iterations=max_events)
    else:
        if architecture is not None:
            raise ValueError("Unmapped source must not claim Phi verification")
        expected=source_order_events(model)
        expected_iterations=()
    if len(expected)+len(expected_iterations) > max_events:
        raise ValueError("Source event trace exceeds declared finite bound")
    compiler = shutil.which(cxx)
    if compiler is None:
        raise FileNotFoundError("Actual host C++ compiler unavailable")
    names = emission.manifest.get("intermediate_offsets")
    if type(names) is not dict:
        raise ValueError("Missing emitted physical address plan")
    count_in = prod(model.tensor(model.inputs[0]).shape)
    count_out = prod(model.tensor(model.outputs[0]).shape)
    runner = (f"\nint main() {{ float a[{count_in}]={{}}; "
              f"float b[{count_out}]={{}}; "
              "fpgai_model(a,b); return 0; }\n")
    with TemporaryDirectory(prefix="fpgai_diamond_trace_") as directory:
        path = Path(directory)
        source, binary = path / "source.cpp", path / "source"
        source.write_text(emission.cpp + runner)
        built = subprocess.run([
            compiler, "-std=c++17", "-O0", "-Wno-unknown-pragmas",
            "-DFPGAI_HOST_FLOAT", "-DFPGAI_TRACE_FANOUT",
            *([] if not mapped else ["-DFPGAI_TRACE_ITERATIONS"]),
            str(source), "-o", str(binary),
        ], capture_output=True, text=True, timeout=timeout_seconds)
        if built.returncode:
            raise RuntimeError("Generated diamond C++ compilation failed: " +
                               built.stderr[-4000:])
        process = subprocess.run([str(binary)], capture_output=True,
                                 text=True, timeout=timeout_seconds)
        if process.returncode:
            raise RuntimeError("Generated diamond C++ execution failed: " +
                               process.stderr[-4000:])
    actual = []
    contributions = []
    partial = {}
    committed = set()
    from fpgai_research.architecture.resolver import coordinate
    for i, line in enumerate(process.stdout.splitlines()):
        parts = line.split()
        if parts and parts[0] == "FPGAI_ITER":
            if not mapped or len(parts)!=9:
                raise AssertionError(f"Unexpected or malformed mapped contribution at line {i}")
            try:
                stage,mi,ni,kk,replica,pe,lane,fold=map(int,parts[1:])
            except ValueError as exc:
                raise AssertionError("Noninteger source Phi contribution") from exc
            if stage != 0:
                raise AssertionError("Wrong producer operator ordinal in source Phi trace")
            (m,k,n)=(
                model.tensor(model.inputs[0]).shape[0],
                model.tensor(model.inputs[0]).shape[1],
                model.tensor(model.operators[0].outputs[0]).shape[1])
            if not (0<=mi<m and 0<=ni<n and 0<=kk<k):
                raise AssertionError("Mapped producer reduction index outside Functional graph")
            signature=coordinate(architecture.points[0],(mi,ni,kk))
            if (replica,pe,lane,fold) != (
                    signature["replica"],signature["pe"],signature["lane"],
                    signature["reuse_step"][2]):
                raise AssertionError("SOURCE_PHI_OWNER_MISMATCH")
            key=(mi,ni)
            used=partial.setdefault(key,set())
            if kk in used or key in committed:
                raise AssertionError("Duplicate or late mapped reduction contribution")
            used.add(kk)
            contributions.append((mi,ni,kk,replica,pe,lane,fold))
            continue
        if (len(parts) not in (4,5) or parts[0] != "FPGAI_FANOUT" or
                parts[1] not in ("COMMIT","CONSUME") or
                (parts[1] == "COMMIT" and len(parts) != 4) or
                (parts[1] == "CONSUME" and len(parts) != 5)):
            raise AssertionError(f"Malformed source trace at event {i}: {line[:100]}")
        try:
            elem = int(parts[3])
        except ValueError as exc:
            raise AssertionError("Noninteger emitted tensor index") from exc
        if mapped and parts[1]=="COMMIT" and parts[2]==model.operators[0].outputs[0]:
            m,k=model.tensor(model.inputs[0]).shape
            n=model.tensor(model.operators[0].outputs[0]).shape[1]
            key=(elem//n,elem%n)
            if key in committed or partial.get(key,set())!=set(range(k)):
                raise AssertionError("SOURCE_PREMATURE_PHI_REDUCTION_COMMIT")
            committed.add(key)
        actual.append(LifetimeEvent(parts[1], parts[2], elem,
                                    parts[4] if len(parts) == 5 else ""))
    if mapped:
        if tuple(contributions)!=expected_iterations:
            raise AssertionError("SOURCE_PHI_ITERATION_ORDER_MISMATCH")
        m,k=model.tensor(model.inputs[0]).shape
        n=model.tensor(model.operators[0].outputs[0]).shape[1]
        if len(committed)!=m*n:
            raise AssertionError("Source lost mapped MatMul output commits")
    actual = tuple(actual)
    if actual != expected:
        index = next((i for i, (want, seen) in enumerate(zip(expected, actual))
                      if want != seen), min(len(expected), len(actual)))
        raise AssertionError(
            "SOURCE_FANOUT_EVENT_MISMATCH at index " + str(index) +
            f" expected={expected[index:index+2]} observed={actual[index:index+2]}")
    replay = verify_fanout_events(
        model, names, actual, max_events=max_events)
    if replay["status"] != "VERIFIED":
        raise AssertionError("SOURCE_FANOUT_LIFETIME_VIOLATION: "+
                             str(replay.get("reason")))
    if replay["event_sha256"] != emission.manifest["source_e0_event_sha256"]:
        raise AssertionError("Actual source event digest differs from submitted E0 plan")
    from hashlib import sha256 as _sha256
    import json
    iteration_sha = (
        _sha256(json.dumps(contributions,separators=(",",":")).encode()).hexdigest()
        if mapped else None)
    if mapped and iteration_sha != emission.manifest.get("producer_phi_iteration_sha256"):
        raise AssertionError("Source Phi reduction trace differs from architectural receipt")
    return {
        "schema": SCHEMA,
        "status": "SOURCE_FANOUT_LIFETIME_VERIFIED",
        "functional_sha256": functional_fingerprint(model),
        "cpp_sha256": emission.manifest["cpp_sha256"],
        "observed_events": len(actual),
        "mapped_phi_verified": mapped,
        "phi_reduction_iterations_verified":len(contributions),
        "phi_iteration_sha256":iteration_sha,
        "multi_reader_values": replay["multi_reader_values"],
        "physical_source_scratch_elements": emission.manifest["scratch_elements"],
        "observed_event_sha256": replay["event_sha256"],
        "raw_source_log_sha256": sha256(process.stdout.encode()).hexdigest(),
        "evidence": "E1-actual-executed-GCC-host-source-events-only",
        "native_vitis_calls": 0,
        "limitations": [
            "C++ instrumentation and trace cannot prove Vitis/RTL scheduling",
            "Restricted two-reader diamond; optional producer Phi, scalar consumers",
            "Static scratch alias is not physical BRAM/port allocation",
        ],
    }
