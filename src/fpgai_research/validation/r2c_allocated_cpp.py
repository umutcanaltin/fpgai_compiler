"""Independent real GCC source verification for CST R2c per-value scratch.

The parser reads separately reported runtime slot/index witnesses and
the actual typed READ/COMMIT order, verifies exact source-address
liveness by event replay, checks source/code/address SHA lineage,
and compares original GCC output against an independently evaluated
NumPy Functional Model IR. All original source paths are create-only.

Does not prove physical FPGA RAM/bank/port legality, hardware-clock
liveness, or savings in synthesized resources.
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

from fpgai_research.codegen.hls.r2c_allocated_branch import SCHEMA as EMIT_SCHEMA
from fpgai_research.model_ir.branch_matmul_contract import admit_branch,_hash
from fpgai_research.model_ir.element_scratch_allocation import (
    allocate_minimum_slots,verify_allocation,
)
from fpgai_research.model_ir.interleaved_branch_schedule import interleaved_branch_events
from fpgai_research.model_ir.semantics import architecture_fingerprint,functional_fingerprint
from fpgai_research.validation.r2b_interleaved_cpp import _decode_trace,_float_literal
from fpgai_research.validation.reference import evaluate


SCHEMA="fpgai-research/cst-R2c-original-GCC-allocated-source-E1/v1"


def decode_allocated_stdout(raw:str,*,n_outputs:int,max_events:int=4096):
    """Independently parse every alternating source operation/slot witness.

    Format is strict: before numerical outputs, exactly one dynamic
    FPGAI_SLOT witness and one typed FPGAI_R2B witness per scratch op.
    A dropped, reordered, duplicated or forged slot fails closed.
    """
    if len(raw.encode())>8*1024*1024:
        raise ValueError("Unbounded original GCC source trace")
    lines=raw.splitlines()
    legacy=[]
    runtime_slots=[]
    cursor=0
    in_outputs=False
    while cursor<len(lines):
        row=lines[cursor]
        if row.startswith("FPGAI_R2B OUTPUT "):
            in_outputs=True
            legacy.append(row)
            cursor+=1
            continue
        if in_outputs:
            raise AssertionError("Scratch event after numerical output")
        parts=row.split()
        if (len(parts)!=5 or parts[0]!="FPGAI_SLOT"
                or parts[1] not in ("COMMIT","READ")):
            raise AssertionError("Missing/malformed original dynamic scratch slot witness")
        try:
            idx,slot=int(parts[3]),int(parts[4])
        except ValueError as exc:
            raise AssertionError("Noninteger original physical source slot") from exc
        if idx<0 or slot<0:
            raise AssertionError("Negative original source index/slot")
        if cursor+1>=len(lines) or not lines[cursor+1].startswith(
                "FPGAI_R2B "):
            raise AssertionError("Missing typed source event after memory operation")
        witness=lines[cursor+1].split()
        if (len(witness)<4 or witness[1]!=parts[1] or
                witness[2]!=parts[2]):
            raise AssertionError("Runtime slot operation and typed event disagree")
        try:
            typed_index=int(witness[3])
        except ValueError as exc:
            raise AssertionError("Invalid typed source event element") from exc
        if typed_index!=idx:
            raise AssertionError("Same C++ value-index symbol disagrees with READ witness")
        runtime_slots.append((parts[1],parts[2],idx,slot))
        legacy.append(lines[cursor+1])
        cursor+=2
        if len(runtime_slots)>max_events:
            raise ValueError("Original source event bound exceeded")
    parsed,output=_decode_trace(
        "\n".join(legacy)+"\n",n_outputs=n_outputs,max_events=max_events)
    if len(parsed)!=len(runtime_slots):
        raise AssertionError("Physical slot witnesses/source event count mismatch")
    for event,slot in zip(parsed,runtime_slots):
        if (event.kind,event.tensor,event.element)!=slot[:3]:
            raise AssertionError("Original runtime slot witness disagrees with tensor event")
    return parsed,tuple(runtime_slots),output


def verify_r2c_gcc(arch,emission,*,runtime_input,
                   cxx:str="g++",timeout_seconds:int=90,
                   max_events:int=4096,atol:float=1e-5,
                   rtol:float=1e-4,
                   original_trace_output:str|Path|None=None)->dict:
    if (type(max_events) is not int or not 1<=max_events<=4096 or
            type(timeout_seconds) is not int or not 1<=timeout_seconds<=180):
        raise ValueError("Bounded original R2c GCC limits required")
    if original_trace_output is not None:
        dest=Path(original_trace_output)
        if dest.exists() or dest.is_symlink():
            raise FileExistsError("Original GCC trace must be create-only")
        if not dest.parent.is_dir():
            raise ValueError("Original trace parent missing")
    compiler=shutil.which(cxx)
    if compiler is None:
        raise FileNotFoundError("g++ original compiler required")
    model=arch.functional
    (m,k,n,q),(h,a,b,g),_,_=admit_branch(model)
    manifest=emission.manifest
    cpp_sha=sha256(emission.cpp.encode()).hexdigest()
    if (manifest.get("schema")!=EMIT_SCHEMA or
            manifest.get("cpp_sha256")!=cpp_sha or
            manifest.get("functional_sha256")!=functional_fingerprint(model) or
            manifest.get("architecture_sha256")!=architecture_fingerprint(arch)):
        raise ValueError("R2c emitted source/functional/Phi/CPP identity mismatch")
    plan=interleaved_branch_events(arch,max_events=max_events)
    from fpgai_research.model_ir.independent_phi_issue_reference import validate_phi_issue_conformance
    phi_proof=validate_phi_issue_conformance(arch,plan.events)
    if phi_proof["status"]!="VERIFIED_E0":
        raise AssertionError("Independent Phi source issue specification rejects generator")
    expected_sha=_hash([asdict(e) for e in plan.events])
    if manifest.get("source_event_sha256")!=expected_sha:
        raise ValueError("R2c finite source event identity corrupted")
    triples=manifest.get("source_value_slots")
    if not isinstance(triples,list):
        raise ValueError("Expected complete original source value-address table")
    try:
        tuples=tuple((name,int(index),int(slot)) for name,index,slot in triples)
        if (len(tuples)!=len(set((name,index) for name,index,slot in tuples))
                or any(type(index) is not int or type(slot) is not int
                       for name,index,slot in triples)):
            raise ValueError("Duplicate or noninteger original source value allocation")
    except (ValueError,TypeError,KeyError) as exc:
        raise ValueError("Invalid source address table") from exc
    addresses={(name,index):slot for name,index,slot in tuples}
    allocated=manifest.get("source_scratch_unit_slots")
    if type(allocated) is not int or not 0<allocated<=4096:
        raise ValueError("Invalid original source scratch allocation capacity")
    verify_allocation(plan.events,addresses,unit_slots=allocated)
    order=tuple(sorted(tuples))
    allocation_hash=sha256(json.dumps(
        order,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    if (allocation_hash!=manifest.get("source_allocation_sha256")
            or manifest.get("source_events")!=len(plan.events)
            or manifest.get("total_declared_mutable_source_array_elements")!=
                allocated+4*m*q):
        raise ValueError("R2c original source scratch/accumulator lineage mismatch")
    optimal=allocate_minimum_slots(plan.events,max_events=max_events)
    if (manifest.get("source_minimum_unit_slots")!=optimal.unit_slots or
            manifest.get("source_peak_live_intermediates")!=
                optimal.peak_simultaneous_live_values):
        raise ValueError("R2c original exact optimality receipt corrupted")
    x=np.asarray(runtime_input,dtype=np.float32)
    if x.shape!=(m,k) or not np.all(np.isfinite(x)):
        raise ValueError("Wrong original source inputs or nonfinite values")
    expected=np.asarray(evaluate(
        model,{model.inputs[0]:x})[model.outputs[0]],
        dtype=np.float32).reshape(-1).astype(np.float64)
    initializer=",".join(_float_literal(v) for v in x.flat)
    runner=(
        "\nint main() {\n"
        f"  float input[{m*k}]={{{initializer}}};\n"
        f"  float output[{m*q}]={{}};\n"
        "  fpgai_model(input,output);\n"
        f'  for(int i=0;i<{m*q};++i) '
        'std::printf("FPGAI_R2B OUTPUT %d %.9g\\n",i,output[i]);\n'
        "  return 0;\n}\n"
    )
    with TemporaryDirectory(prefix="cst_r2c_original_GCC_") as directory:
        src=Path(directory)/"source.cpp"
        exe=Path(directory)/"run"
        src.write_text(emission.cpp+runner)
        result=subprocess.run(
            [compiler,"-std=c++17","-O0","-ffp-contract=off",
             "-DFPGAI_HOST_FLOAT","-DFPGAI_TRACE_R2C",
             str(src),"-o",str(exe)],capture_output=True,text=True,
            timeout=timeout_seconds)
        if result.returncode:
            raise RuntimeError("Original R2c GCC compile failed: "+result.stderr[-3500:])
        run=subprocess.run([str(exe)],capture_output=True,text=True,
                           timeout=timeout_seconds)
        if run.returncode:
            raise RuntimeError("Original R2c GCC source execution failed: "+run.stderr[-3500:])
    observed,slot_witnesses,outputs=decode_allocated_stdout(
        run.stdout,n_outputs=m*q,max_events=max_events)
    observed_spec=validate_phi_issue_conformance(arch,observed)
    if observed_spec["status"]!="VERIFIED_E0":
        raise AssertionError("Independent Phi source issue proof rejects GCC trace")
    if observed!=plan.events:
        mismatch=next((i for i,(left,right) in enumerate(zip(
            observed,plan.events)) if left!=right),
            min(len(observed),len(plan.events)))
        raise AssertionError("R2C_ORIGINAL_SOURCE_EVENT_MISMATCH index="+str(mismatch))
    for operation,event in zip(slot_witnesses,observed):
        if operation[3]!=addresses[(event.tensor,event.element)]:
            raise AssertionError(
                "R2C_RUNTIME_SCRATCH_SLOT_DISAGREES_WITH_SOURCE_CERTIFICATE: "+
                str(operation))
    # Runtime source event identities independently replay the address
    # allocation certificate rather than trusting a manifest SHA.
    replay=verify_allocation(observed,addresses,unit_slots=allocated)
    if replay["status"]!="EXACT_SOURCE_SLOT_REPLAY_VERIFIED":
        raise AssertionError("Original runtime source scratch alias audit failed")
    accepted=bool(np.allclose(outputs,expected,atol=atol,rtol=rtol))
    if not accepted:
        raise AssertionError("R2C_INDEPENDENT_NUMPY_NUMERICAL_MISMATCH "+
                             str(float(np.max(np.abs(outputs-expected)))))
    if original_trace_output is not None:
        with Path(original_trace_output).open("x",encoding="utf-8") as stream:
            stream.write(run.stdout)
    return {
        "schema":SCHEMA,
        "status":"ORIGINAL_GCC_SOURCE_NUMERIC_AND_SCRATCH_SAFETY_VERIFIED_E1",
        "functional_sha256":manifest["functional_sha256"],
        "architecture_sha256":manifest["architecture_sha256"],
        "cpp_sha256":cpp_sha,
        "original_input_sha256":sha256(x.tobytes()).hexdigest(),
        "original_GCC_stdout_sha256":sha256(run.stdout.encode()).hexdigest(),
        "original_source_event_sha256":_hash([asdict(ev) for ev in observed]),
        "actual_scratch_witness_sha256":sha256(json.dumps(
            slot_witnesses,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
        "original_source_read_events":sum(ev.kind=="READ" for ev in observed),
        "original_source_commit_events":sum(ev.kind=="COMMIT" for ev in observed),
        "original_source_unit_scratch_slots":allocated,
        "original_optimal_unit_scratch_slots":optimal.unit_slots,
        "original_total_declared_mutable_source_array_elements":allocated+4*m*q,
        "source_optimality_proven_for_fixed_event_order":
            allocated==optimal.unit_slots,
        "original_float32_numeric_max_abs_error":
            float(np.max(np.abs(outputs-expected))),
        "original_numeric_atol":atol,"original_numeric_rtol":rtol,
        "original_GCC_compilations":1,"original_GCC_executions":1,
        "original_Vitis_calls":0,"original_Vivado_calls":0,
        "evidence":"E1-GCC-real-per-element-source-address-and-numeric",
        "limitations":[
            "Declared mutable ARRAY elements count scratch and four C++ arrays; excludes C++ scalar locals, input/output and read-only weights",
            "Source-address safety is not hardware-cycle liveness, bank/port legality, physical BRAM, Fmax or power",
            "Only original fixed three-MatMul two-reader source diamond; not arbitrary DAG",
            "No native Vitis CSim or synthesis and no external compiler novelty claims",
        ],
    }
