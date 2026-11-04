"""Original GCC E1 source-executed CST R2b READ/COMMIT and NumPy audit.

This verifier parses actual C++ printed source events, checks their
complete typed sequence against the independently emitted E0 contract,
re-verifies address safety and compares the resulting numerical model
outputs with an independently evaluated Functional IR. It does not
establish real HLS execution cycles, physical PE/SIMD parallelism,
on-chip RAM-bank/port legality, or published-peer firstness.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

import numpy as np

from fpgai_research.codegen.hls.r2b_interleaved_branch import SCHEMA as EMISSION_SCHEMA
from fpgai_research.model_ir.branch_matmul_contract import (
    BranchEvent, _hash, admit_branch, validate_branch_events,
)
from fpgai_research.model_ir.interleaved_branch_schedule import (
    interleaved_branch_events, release_positions,
)
from fpgai_research.model_ir.semantics import (
    architecture_fingerprint, functional_fingerprint,
)
from fpgai_research.validation.reference import evaluate

SCHEMA = "fpgai-research/cst-r2b-executed-source-and-independent-numeric-e1/v1"


def _float_literal(value: float) -> str:
    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError("Only finite original input samples are permitted")
    value_string = f"{numeric:.9g}"
    return (value_string if "." in value_string or "e" in value_string
            else value_string+".0")+"f"


def _decode_trace(raw: str, *, n_outputs: int,
                  max_events: int) -> tuple[tuple[BranchEvent,...], np.ndarray]:
    """Independent strict event parser, no use of the generator's plan."""
    events, output = [], {}
    if len(raw.encode()) > 8*1024*1024:
        raise ValueError("Unbounded emitted C++ event stdout")
    for lineno, line in enumerate(raw.splitlines(), 1):
        parts = line.split()
        if not parts or parts[0] != "FPGAI_R2B":
            raise AssertionError(f"Unexpected source trace line {lineno}: {line[:90]}")
        if parts[1:2] == ["COMMIT"]:
            if len(parts) != 4:
                raise AssertionError("Invalid source COMMIT record")
            try:
                events.append(BranchEvent("COMMIT", parts[2], int(parts[3])))
            except ValueError as exc:
                raise AssertionError("Noninteger source COMMIT index") from exc
        elif parts[1:2] == ["READ"]:
            if len(parts) != 10:
                raise AssertionError("Invalid source READ record")
            try:
                element = int(parts[3])
                dest = int(parts[5])
                owner_numbers = tuple(map(int, parts[6:]))
            except ValueError as exc:
                raise AssertionError("Noninteger source READ witness") from exc
            if owner_numbers == (-1,-1,-1,-1):
                owner = None
            elif len(owner_numbers) == 4 and all(x>=0 for x in owner_numbers):
                owner = owner_numbers
            else:
                raise AssertionError("Invalid complete or absent source Phi owner")
            events.append(BranchEvent("READ", parts[2], element,
                                      parts[4], dest, owner))
        elif parts[1:2] == ["OUTPUT"]:
            if len(parts) != 4:
                raise AssertionError("Invalid source OUTPUT record")
            try:
                index,value=int(parts[2]),float(parts[3])
            except ValueError as exc:
                raise AssertionError("Malformed source numeric output") from exc
            if index in output or not 0 <= index < n_outputs or not isfinite(value):
                raise AssertionError("Duplicate/bad original numeric output")
            output[index]=value
        else:
            raise AssertionError(f"Unknown original source event at line {lineno}")
        if len(events)>max_events:
            raise ValueError("Original source emitted more events than agreed limit")
    if set(output)!=set(range(n_outputs)):
        raise AssertionError("Original C++ source output cardinality mismatch")
    return tuple(events), np.asarray([output[i] for i in range(n_outputs)],
                                      dtype=np.float64)


def verify_r2b_gcc(arch, emission, *, offsets: dict[str,int],
                   runtime_input, cxx: str = "g++",
                   max_events: int = 100_000,
                   timeout_seconds: int = 90,
                   atol: float = 1e-5, rtol: float = 1e-4,
                   original_trace_output: str | Path | None = None) -> dict:
    if type(max_events) is not int or not 1<=max_events<=100_000:
        raise ValueError("A bounded original source event budget is required")
    if type(timeout_seconds) is not int or not 1<=timeout_seconds<=180:
        raise ValueError("A bounded original GCC source timeout is required")
    if original_trace_output is not None:
        path = Path(original_trace_output)
        if path.exists() or path.is_symlink():
            raise FileExistsError("Original GCC trace must be create-only")
        if not path.parent.is_dir():
            raise ValueError("Original GCC trace parent must exist")
    compiler = shutil.which(cxx)
    if compiler is None:
        raise FileNotFoundError("Independent real GCC compiler g++ unavailable")
    model = arch.functional
    (m,k,n,q),(h,a,b,g),_weights,_ops = admit_branch(model)
    sha = sha256(emission.cpp.encode()).hexdigest()
    manifest = emission.manifest
    if (manifest.get("schema") != EMISSION_SCHEMA
            or manifest.get("cpp_sha256") != sha
            or manifest.get("functional_sha256") != functional_fingerprint(model)
            or manifest.get("architecture_sha256") != architecture_fingerprint(arch)
            or manifest.get("scratch_offsets") != dict(sorted(offsets.items()))):
        raise ValueError("R2b source/architecture/offset identity mismatch")
    proposed = interleaved_branch_events(arch,max_events=max_events)
    from fpgai_research.model_ir.independent_phi_issue_reference import validate_phi_issue_conformance
    phi_proof=validate_phi_issue_conformance(arch,proposed.events)
    if phi_proof["status"]!="VERIFIED_E0":
        raise AssertionError("Independent Phi source issue specification rejects generator")
    if manifest.get("planned_source_event_sha256") != _hash(
            [asdict(e) for e in proposed.events]):
        raise ValueError("R2b emission plan event digest does not match original IR")
    x = np.asarray(runtime_input,dtype=np.float32)
    if x.shape!=(m,k) or not np.all(np.isfinite(x)):
        raise ValueError("Original source runtime input has wrong shape or nonfinite data")
    expected = np.asarray(evaluate(model,{model.inputs[0]:x})[
        model.outputs[0]],dtype=np.float32).reshape(-1).astype(np.float64)
    lit = ",".join(_float_literal(v) for v in x.flat)
    runner = (
        "\nint main() {\n"
        f"  float input[{m*k}]={{{lit}}};\n"
        f"  float output[{m*q}]={{}};\n"
        "  fpgai_model(input, output);\n"
        f'  for(int i=0;i<{m*q};++i) '
        'std::printf("FPGAI_R2B OUTPUT %d %.9g\\n",i,output[i]);\n'
        "  return 0;\n}\n"
    )
    with TemporaryDirectory(prefix="fpgai_r2b_original_source_") as location:
        src=Path(location)/"original_r2b.cpp"
        executable=Path(location)/"original_r2b_exe"
        src.write_text(emission.cpp+runner)
        build=subprocess.run(
            [compiler,"-std=c++17","-O0","-ffp-contract=off",
             "-DFPGAI_HOST_FLOAT","-DFPGAI_TRACE_R2B",
             str(src),"-o",str(executable)],
            capture_output=True,text=True,timeout=timeout_seconds)
        if build.returncode:
            raise RuntimeError("Original emitted R2b GCC compile failure:\n"+
                               build.stderr[-4000:])
        finished=subprocess.run([str(executable)],capture_output=True,
                                text=True,timeout=timeout_seconds)
        if finished.returncode:
            raise RuntimeError("Original emitted R2b GCC run failure:\n"+
                               finished.stderr[-4000:])
    actual, output = _decode_trace(
        finished.stdout,n_outputs=m*q,max_events=max_events)
    # This equality is a strict byte-reconstructed E1 contract, separate
    # from the independently implemented NumPy FunctionalModel evaluator.
    observed_spec=validate_phi_issue_conformance(arch,actual)
    if observed_spec["status"]!="VERIFIED_E0":
        raise AssertionError("Independent Phi source issue proof rejects GCC trace")
    if actual != proposed.events:
        mismatch=next(
            (i for i,(x,y) in enumerate(zip(actual,proposed.events)) if x!=y),
            min(len(actual),len(proposed.events)))
        raise AssertionError(
            "SOURCE_EVENT_SEQUENCE_MISMATCH at "+
            str(mismatch)+": actual="+str(actual[mismatch:mismatch+1])+
            " expected="+str(proposed.events[mismatch:mismatch+1]))
    checked = validate_branch_events(arch,offsets,actual,max_events=max_events)
    if checked["status"]!="VERIFIED_E0":
        raise AssertionError("E1 original source was unsafe: "+
                             str(checked.get("counterexample")))
    if not np.all(np.isfinite(output)):
        raise AssertionError("Nonfinite E1 source output")
    numeric_accepted=bool(np.allclose(
        output,expected,atol=atol,rtol=rtol))
    if not numeric_accepted:
        raise AssertionError(
            "INDEPENDENT_NUMPY_SOURCE_NUMERIC_MISMATCH: "+
            str(float(np.max(np.abs(output-expected)))))
    positions={}
    # Calculate actual last-read ordinals independently of the E0
    # certificate hash and architecture Phi identity.
    for index,event in enumerate(actual):
        if event.kind=="READ" and event.tensor==h:
            positions[event.element]=index
    observed_last=tuple(sorted(positions.items()))
    if observed_last!=release_positions(proposed,h):
        raise AssertionError("SOURCE_OBSERVED_H_LAST_USE_MISMATCH")
    # Stronger discriminator: count *only H source READ operations*.
    # Shifting an unrelated COMMIT must not be misrepresented as a
    # changed last consumption or changed source memory lifetime.
    final_h_reads={}
    h_read_rank=0
    for event in actual:
        if event.kind=="READ" and event.tensor==h:
            final_h_reads[event.element]=(
                h_read_rank,event.consumer,event.output_element)
            h_read_rank+=1
    observed_last_read=tuple(sorted(
        (element,*identity) for element,identity in final_h_reads.items()))
    if len(observed_last_read)!=m*n or h_read_rank!=2*m*n*q:
        raise AssertionError("INCOMPLETE_SOURCE_H_READ_RANKS")
    if original_trace_output is not None:
        # Only after all event and independent numerical checks succeeded.
        # Immutable stdout bytes can then be independently re-parsed.
        with Path(original_trace_output).open("x",encoding="utf-8") as stream:
            stream.write(finished.stdout)
    return {
        "schema":SCHEMA,
        "status":"E1_ORIGINAL_GCC_SOURCE_EVENT_AND_NUMERIC_VERIFIED",
        "source_cpp_sha256":sha,
        "functional_sha256":manifest["functional_sha256"],
        "architecture_sha256":manifest["architecture_sha256"],
        "runtime_input_sha256":sha256(x.tobytes()).hexdigest(),
        "raw_original_gcc_stdout_sha256":sha256(finished.stdout.encode()).hexdigest(),
        "observed_event_sha256":_hash([asdict(e) for e in actual]),
        "observed_h_last_use_sha256":_hash(observed_last),
        "observed_h_last_use_indices":list(observed_last),
        "observed_h_final_read_rank_and_owner_sha256":_hash(observed_last_read),
        "observed_h_final_read_rank_and_owner":list(observed_last_read),
        "original_source_h_read_only_events":h_read_rank,
        "original_source_intermediate_events":len(actual),
        "original_source_h_reads":sum(
            e.kind=="READ" and e.tensor==h for e in actual),
        "original_source_commits":sum(e.kind=="COMMIT" for e in actual),
        "original_source_output_elements":len(output),
        "max_absolute_numeric_error":float(np.max(np.abs(output-expected))),
        "numeric_atol":atol,"numeric_rtol":rtol,
        "original_gcc_compilations":1,
        "original_gcc_executions":1,
        "original_vitis_calls":0,
        "original_vivado_calls":0,
        "evidence_level":"E1-actual-original-GCC-compiled-C++-source-events-and-NUMPy",
        "limitations":[
            "Host-float C++ source order is not FPGA clock-cycle scheduling",
            "This E1 code emits sequential PE-labelled reads, not PE-parallel circuits",
            "No native Vitis CSim/CSynth or real RAM/port proof",
            "Only fixed static three-MatMul reader diamond, not arbitrary graph",
            "Independent NumPy oracle validates outputs, not bit-exact ap_fixed arithmetic",
        ],
    }
