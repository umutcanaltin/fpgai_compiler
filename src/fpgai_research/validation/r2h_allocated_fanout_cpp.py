"""R2h E1 original paired allocated-fanout GCC source verifier.

The disjoint and per-element optimal source arms use exactly the same
runtime C++ ordinal reader algorithm. The actual memory subscript is
printed adjacent to the corresponding source READ/COMMIT. This strict
validator cross-checks every address witness against independently
reconstructed value lifetimes and additionally validates every
intermediate output numerically against the FunctionalModel.

A runtime-source address witness is NOT physical FPGA BRAM, timing,
RAM port arbitration, or formal equivalence for arbitrary inputs.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil
import subprocess

import numpy as np

from fpgai_research.codegen.hls.r2h_allocated_fanout_cpp import (
    SCHEMA as EMITTER_SCHEMA,source_layout,
)
from fpgai_research.model_ir.element_scratch_allocation import (
    verify_allocation,
)
from fpgai_research.model_ir.fanout_contract import (
    verify_fanout_events,verify_fanout_issue,disjoint_fanout_offsets,
)
from fpgai_research.model_ir.semantics import (
    architecture_fingerprint,functional_fingerprint,
)
from fpgai_research.experiments.cst_r2f_fanout_campaign import (
    project_source_value_events,
)
from fpgai_research.validation.r2g_fanout_cpp import decode_stdout
from fpgai_research.validation.r2b_interleaved_cpp import _float_literal
from fpgai_research.validation.reference import evaluate

SCHEMA="fpgai-research/cst-R2h-GCC-dynamic-allocated-fanout-e1/v1"


def decode_address_witnesses(raw:str,*,slot_map:dict,
                             capacity:int,max_events:int=100_000):
    """Strict adjacency and slot equality for every source scratch event.

    The C++ variable printed by FPGAI_SLOT is the variable used by
    scratch[...], not an independently constructed Python label.
    """
    if type(slot_map) is not dict or type(capacity) is not int or capacity<1:
        raise ValueError("Missing certified address assignment")
    clean=[]
    seen=[]
    pending=None
    for line_number,line in enumerate(raw.splitlines(),1):
        fields=line.split()
        if not fields:
            raise ValueError("Unexpected blank original slot trace")
        if pending is not None:
            kind,tensor,index,address=pending
            if (len(fields)<4 or fields[:3]!=[
                    "FPGAI_R2G",kind,tensor] or fields[3]!=str(index)):
                raise ValueError(
                    "R2H_ADDRESS_WITNESS_NOT_ADJACENT_TO_ACTUAL_SOURCE_EVENT "+
                    str(line_number))
            clean.append(line)
            seen.append(pending)
            pending=None
        elif fields[0]=="FPGAI_SLOT":
            if (len(fields)!=5 or fields[1] not in ("READ","COMMIT")):
                raise ValueError("R2H_INVALID_SOURCE_ADDRESS_WITNESS")
            try:
                index=int(fields[3])
                address=int(fields[4])
            except ValueError as exc:
                raise ValueError("R2H_NONINTEGER_SOURCE_ADDRESS") from exc
            key=(fields[2],index)
            if (key not in slot_map or
                    not 0<=address<capacity or
                    slot_map[key]!=address):
                raise ValueError(
                    "R2H_RUNTIME_SOURCE_ADDRESS_DIFFERS_FROM_CERTIFIED_VALUE")
            pending=(fields[1],fields[2],index,address)
        else:
            if fields[0]!="FPGAI_R2G":
                raise ValueError("R2H_UNKNOWN_SOURCE_LOG")
            if fields[1] in ("READ","COMMIT"):
                raise ValueError("R2H_TYPED_SOURCE_EVENT_WITHOUT_ACTUAL_ADDRESS_WITNESS")
            clean.append(line)
        if len(seen)>max_events:
            raise ValueError("R2H_SOURCE_ADDRESS_EVENT_BUDGET_EXCEEDED")
    if pending is not None:
        raise ValueError("R2H_TRUNCATED_SOURCE_ADDRESS_WITNESS")
    clean_raw="\n".join(clean)+"\n"
    events,values,outputs=decode_stdout(clean_raw,max_events=max_events)
    actual=tuple((e.kind,e.tensor,e.element,slot_map[(e.tensor,e.element)])
                 for e in events if e.kind in ("READ","COMMIT"))
    if actual!=tuple(seen):
        raise ValueError("R2H_ACTUAL_SOURCE_EVENT_ADDRESS_SEQUENCE_MISMATCH")
    return events,values,outputs,len(seen)


def verify_allocated_gcc(arch,emission,*,runtime_input,
                         cxx:str="g++",timeout_seconds:int=90,
                         original_trace_output:str|Path|None=None,
                         atol:float=1e-5,rtol:float=1e-4)->dict:
    if type(timeout_seconds) is not int or not 1<=timeout_seconds<=180:
        raise ValueError("R2h bounded original GCC budget required")
    if original_trace_output is not None:
        destination=Path(original_trace_output)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("R2h original GCC trace must be create-only")
        if not destination.parent.is_dir():
            raise ValueError("Original R2h evidence parent missing")
    compiler=shutil.which(cxx)
    if compiler is None:
        raise FileNotFoundError("Original real GCC C++ compiler required")
    manifest=emission.manifest
    layout=manifest.get("source_layout")
    spec,assigned,slots,optimal,planned=source_layout(arch,layout=layout)
    m,k,n,q=spec.m,spec.k,spec.n,spec.q
    source_sha=sha256(emission.cpp.encode()).hexdigest()
    if (manifest.get("schema")!=EMITTER_SCHEMA
            or manifest.get("source_cpp_sha256")!=source_sha
            or manifest.get("functional_sha256")!=functional_fingerprint(arch.functional)
            or manifest.get("architecture_sha256")!=architecture_fingerprint(arch)
            or manifest.get("source_scratch_array_elements")!=slots
            or manifest.get("source_optimal_unit_slots_for_fixed_order")!=optimal.unit_slots
            or manifest.get("source_event_order_sha256")!=optimal.schedule_sha256
            or tuple(map(tuple,manifest.get("source_slot_map",())))!=tuple(
                sorted((tensor,i,slot) for (tensor,i),slot in assigned.items()))
            or manifest.get("source_all_declared_mutable_array_elements")!=
                slots+2*len(spec.readers)*m*q):
        raise ValueError("R2h source/architecture/allocator SHA lineage mismatch")
    x=np.asarray(runtime_input,dtype=np.float32)
    if x.shape!=(m,k) or not np.all(np.isfinite(x)):
        raise ValueError("Original R2h GCC input must be finite float32 matrix")
    original=evaluate(arch.functional,{spec.input:x},return_all=True)
    runner=(
        "\nint main() {\n"
        f"  float input[{m*k}]={{{','.join(_float_literal(float(v)) for v in x.flat)}}};\n"
        f"  float output[{m*q}]={{}};\n"
        "  fpgai_model(input,output);\n"
        f'  for(int i=0;i<{m*q};++i) '
        'std::printf("FPGAI_R2G OUTPUT %d %.9g\\n",i,double(output[i]));\n'
        "  return 0;\n}\n"
    )
    with TemporaryDirectory(prefix="cst_R2h_original_GCC_") as tmp:
        src=Path(tmp)/"allocated_fanout.cpp"
        exe=Path(tmp)/"run"
        src.write_text(emission.cpp+runner)
        compiled=subprocess.run([
            compiler,"-std=c++17","-O0","-ffp-contract=off",
            "-DFPGAI_HOST_FLOAT","-DFPGAI_TRACE_R2G",
            str(src),"-o",str(exe),
        ],capture_output=True,text=True,timeout=timeout_seconds)
        if compiled.returncode:
            raise RuntimeError("Original R2h GCC compile failed: "+compiled.stderr[-4000:])
        completed=subprocess.run([str(exe)],capture_output=True,
                                 text=True,timeout=timeout_seconds)
        if completed.returncode:
            raise RuntimeError("Original R2h GCC execution failed: "+completed.stderr[-4000:])
    events,values,outputs,address_count=decode_address_witnesses(
        completed.stdout,slot_map=assigned,capacity=slots)
    # Verify graph source schedule, then replay the actual observed memory
    # events with the allocated capacity rather than a false disjoint base.
    disjoint=disjoint_fanout_offsets(spec)
    safety=verify_fanout_events(arch,disjoint,events)
    order=verify_fanout_issue(arch,events)
    if safety["status"]!="VERIFIED_E0" or order["status"]!="VERIFIED_E0":
        raise AssertionError("R2H_SOURCE_REDUCTION_OR_PHI_EVENT_MISMATCH")
    actual_projected=project_source_value_events(events)
    alias=verify_allocation(actual_projected,assigned,unit_slots=slots)
    if alias["status"]!="EXACT_SOURCE_SLOT_REPLAY_VERIFIED":
        raise AssertionError("R2H_ACTUAL_SCRATCH_ALIAS_REPLAY_INVALID")
    errors={}
    for tensor in spec.intermediates:
        reference=np.asarray(original[tensor],dtype=np.float32).reshape(-1)
        actual=np.asarray([values[(tensor,i)] for i in range(reference.size)])
        if not np.allclose(actual,reference,atol=atol,rtol=rtol):
            raise AssertionError(
                "R2H_INDEPENDENT_INTERMEDIATE_NUMERIC_MISMATCH "+tensor)
        errors[tensor]=float(np.max(np.abs(actual-reference)))
    expected_population=sum(
        np.asarray(original[tensor]).size for tensor in spec.intermediates)
    if len(values)!=expected_population:
        raise AssertionError("R2H_UNEXPECTED_INTERMEDIATE_VALUE_POPULATION")
    if set(outputs)!=set(range(m*q)):
        raise AssertionError("R2H_INCOMPLETE_FINAL_OUTPUT")
    final=np.asarray([outputs[i] for i in range(m*q)])
    reference=np.asarray(original[arch.functional.outputs[0]],
                         dtype=np.float32).reshape(-1)
    if not np.allclose(final,reference,rtol=rtol,atol=atol):
        raise AssertionError("R2H_FINAL_NUMERIC_MISMATCH")
    if original_trace_output is not None:
        with Path(original_trace_output).open("x",encoding="utf-8") as out:
            out.write(completed.stdout)
    return {
        "schema":SCHEMA,
        "status":"R2H_ACTUAL_SOURCE_SCRATCH_ALIAS_AND_NUMPY_VERIFIED",
        "source_layout":layout,
        "functional_sha256":manifest["functional_sha256"],
        "architecture_sha256":manifest["architecture_sha256"],
        "source_cpp_sha256":source_sha,
        "original_input_sha256":sha256(x.tobytes()).hexdigest(),
        "original_stdout_sha256":sha256(completed.stdout.encode()).hexdigest(),
        "original_runtime_source_event_sha256":safety["source_event_sha256"],
        "source_slot_map_sha256":manifest["source_slot_map_sha256"],
        "original_actual_address_witness_count":address_count,
        "source_scratch_unit_slots":slots,
        "exact_fixed_order_optimal_unit_slots":optimal.unit_slots,
        "source_all_declared_mutable_array_elements":slots+2*len(spec.readers)*m*q,
        "original_all_intermediate_numeric_max_absolute_error":max(errors.values()),
        "original_final_numeric_max_absolute_error":
            float(np.max(np.abs(final-reference))),
        "original_GCC_compilations":1,"original_GCC_executions":1,
        "original_Vitis_calls":0,"original_Vivado_calls":0,
        "actual_optimized_source_memory_savings_if_paired":layout=="per_element_optimal",
        "FPGA_hardware_BRAM_savings_proven":False,
        "limitations":[
            "Correct fixed-order source unit-scratch aliasing is not physical BRAM measurement",
            "C++ memory includes separate accumulator and counter arrays",
            "No independent native HLS timing, dual ports, placement or clock schedule",
        ],
    }
