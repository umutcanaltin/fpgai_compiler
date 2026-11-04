"""R2d GCC E1: dynamically computed source READ indices vs independent Phi spec.

The READ index and output index are calculated in a runtime C++ loop
from a rank decoder; the source event expectation is a separate
closed-form Python rank/owner specification, NOT reconstructed by the
C++ generator's own event-list algorithm. The checker also performs
independent BranchEvent reduction/alias safety and NumPy output tests.

No original R2b/R2c source is modified. Host GCC ordering is not
scheduled FPGA hardware cycles or proof of generated RTL correctness.
"""
from __future__ import annotations

from hashlib import sha256
from math import isfinite
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

import numpy as np

from fpgai_research.codegen.hls.r2d_loop_driven_branch import SCHEMA as EMIT_SCHEMA
from fpgai_research.model_ir.branch_matmul_contract import (
    admit_branch, validate_branch_events,
)
from fpgai_research.model_ir.independent_phi_issue_reference import (
    validate_phi_issue_conformance,
)
from fpgai_research.model_ir.semantics import functional_fingerprint, architecture_fingerprint
from fpgai_research.validation.r2b_interleaved_cpp import _decode_trace,_float_literal
from fpgai_research.validation.reference import evaluate


SCHEMA="fpgai-research/cst-R2d-GCC-dynamic-source-index-independent-issue-e1/v1"


def verify_loop_driven_gcc(arch,emission,*,offsets:dict[str,int],
                            runtime_input,cxx:str="g++",timeout_seconds:int=90,
                            original_trace_output:str|Path|None=None,
                            atol:float=1e-5,rtol:float=1e-4)->dict:
    if type(timeout_seconds) is not int or not 1<=timeout_seconds<=180:
        raise ValueError("R2d requires finite GCC runtime limit")
    if original_trace_output is not None:
        path=Path(original_trace_output)
        if path.exists() or path.is_symlink():
            raise FileExistsError("Original R2d GCC source trace must be create-only")
        if not path.parent.is_dir():
            raise ValueError("Source trace parent folder absent")
    compiler=shutil.which(cxx)
    if compiler is None:
        raise FileNotFoundError("C++ host compiler g++ is required")
    model=arch.functional
    (m,k,n,q),(h,a,b,g),_weights,ops=admit_branch(model)
    manifest=emission.manifest
    cpp_sha=sha256(emission.cpp.encode()).hexdigest()
    if (manifest.get("schema")!=EMIT_SCHEMA or
            manifest.get("cpp_sha256")!=cpp_sha or
            manifest.get("functional_sha256")!=functional_fingerprint(model) or
            manifest.get("architecture_sha256")!=architecture_fingerprint(arch) or
            manifest.get("scratch_offsets")!=dict(sorted(offsets.items()))):
        raise ValueError("R2d original emitted source, mapping or scratch lineage mismatch")
    x=np.asarray(runtime_input,dtype=np.float32)
    if x.shape!=(m,k) or not np.all(np.isfinite(x)):
        raise ValueError("Original R2d input must be finite float32 MatMul shape")
    ref=np.asarray(
        evaluate(model,{model.inputs[0]:x})[model.outputs[0]],
        dtype=np.float32).reshape(-1).astype(np.float64)
    lit=",".join(_float_literal(float(value)) for value in x.flat)
    runner=(
        "\nint main() {\n"
        f"  float input[{m*k}]={{{lit}}};\n"
        f"  float output[{m*q}]={{}};\n"
        "  fpgai_model(input,output);\n"
        f"  for(int i=0;i<{m*q};++i) "
        'std::printf("FPGAI_R2B OUTPUT %d %.9g\\n",i,output[i]);\n'
        "  return 0;\n}\n"
    )
    with TemporaryDirectory(prefix="r2d_original_gcc_runtime_") as tmp:
        source=Path(tmp)/"runtime_loop.cpp"
        exe=Path(tmp)/"runtime_loop"
        source.write_text(emission.cpp+runner)
        compiled=subprocess.run(
            [compiler,"-std=c++17","-O0","-ffp-contract=off",
             "-DFPGAI_HOST_FLOAT","-DFPGAI_TRACE_R2D",
             str(source),"-o",str(exe)],
            capture_output=True,text=True,timeout=timeout_seconds)
        if compiled.returncode:
            raise RuntimeError("Original R2d runtime C++ compile failure: "+
                               compiled.stderr[-4000:])
        executed=subprocess.run(
            [str(exe)],capture_output=True,text=True,timeout=timeout_seconds)
        if executed.returncode:
            raise RuntimeError("Original R2d runtime C++ execution failure: "+
                               executed.stderr[-4000:])
    events,output=_decode_trace(
        executed.stdout,n_outputs=m*q,max_events=100_000)
    # This is NOT equality with interleaved_branch_events or a prewritten
    # generator list. The source loop's dynamic READ indices must satisfy
    # a separately implemented mathematical rank/owner oracle.
    phi=validate_phi_issue_conformance(arch,events)
    if phi["status"]!="VERIFIED_E0":
        raise AssertionError(
            "INDEPENDENT_RUNTIME_PHI_ISSUE_ORDER_MISMATCH: "+str(phi))
    safety=validate_branch_events(arch,offsets,events)
    if safety["status"]!="VERIFIED_E0":
        raise AssertionError(
            "INDEPENDENT_RUNTIME_REDUCTION_OR_ALIAS_VIOLATION: "+str(safety))
    if not np.allclose(output,ref,atol=atol,rtol=rtol):
        raise AssertionError(
            "R2D_INDEPENDENT_NUMPY_NUMERIC_MISMATCH: "+
            str(float(np.max(np.abs(output-ref)))))
    if original_trace_output is not None:
        with Path(original_trace_output).open("x",encoding="utf-8") as stream:
            stream.write(executed.stdout)
    source_indices=[]
    h_read_only_last={}
    for index,event in enumerate(events):
        if event.kind=="READ" and event.tensor==h:
            source_indices.append((event.consumer,event.output_element,event.element))
            h_read_only_last[event.element]=(len(source_indices)-1,event.consumer)
    return {
        "schema":SCHEMA,
        "status":"E1_GCC_RUNTIME_SUBSCRIPT_PHI_ORDER_AND_NUMPY_VERIFIED",
        "source_cpp_sha256":cpp_sha,
        "functional_sha256":manifest["functional_sha256"],
        "architecture_sha256":manifest["architecture_sha256"],
        "original_runtime_input_sha256":sha256(x.tobytes()).hexdigest(),
        "original_GCC_trace_sha256":sha256(executed.stdout.encode()).hexdigest(),
        "original_trace_event_count":len(events),
        "actual_H_read_count":len(source_indices),
        "actual_H_final_reader_ranks":[
            [element,rank,consumer] for element,(rank,consumer)
            in sorted(h_read_only_last.items())],
        "actual_runtime_source_H_issue_signature_SHA256":
            phi["observed_issue_signature_sha256"],
        "independent_phi_source_rule":"VERIFIED_E0_CLOSED_FORM",
        "independent_branch_reduction_alias":"VERIFIED_E0",
        "original_numerical_max_absolute_error":
            float(np.max(np.abs(output-ref))),
        "original_GCC_compilations":1,"original_GCC_executions":1,
        "original_native_Vitis_calls":0,"original_native_Vivado_calls":0,
        "runtime_subscript_computed_by_CPP_loop":True,
        "source_order_witness_property":"actual-C++-runtime-subscript-vs-independent-Python-rank",
        "hardware_execution_scheduling_proven":False,
        "source_order_generated_event_replay_dependency":False,
        "limitations":[
            "Host source runtime index and Python spec both describe one serialized source issue policy",
            "The C++ decoder is source-level, not FPGA cycles/PE parallelism",
            "No independent H producer CONTRIBUTE events or Vitis HLS/RTL validation",
            "Only a bounded five-op diamond with one static graph and source array offsets",
        ],
    }
