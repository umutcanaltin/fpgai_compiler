"""Original GCC E1 validation of R2e loop-driven H producer reductions.

Independently parse live producer contribution and commit witnesses,
compare all k reduction coordinates with the mathematical MatMul
definition, check all reader Phi/source alias events and NumPy output.
No equality to a generator's precomputed producer event list.

This is not full semantic proof of arbitrary C++, RTL cycles or
bit-exact fixed-point arithmetic.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

import numpy as np

from fpgai_research.codegen.hls.r2e_producer_traced_branch import SCHEMA as SOURCE_SCHEMA
from fpgai_research.model_ir.branch_matmul_contract import (
    admit_branch,validate_branch_events,
)
from fpgai_research.model_ir.independent_phi_issue_reference import validate_phi_issue_conformance
from fpgai_research.model_ir.semantics import architecture_fingerprint,functional_fingerprint
from fpgai_research.validation.r2b_interleaved_cpp import _decode_trace,_float_literal
from fpgai_research.validation.r2e_independent_producer_oracle import inspect_producer_stdout
from fpgai_research.validation.reference import evaluate

SCHEMA="fpgai-research/cst-R2e-original-GCC-producer-matmul-completeness-e1/v1"


def verify_producer_traced_gcc(arch,emission,*,offsets:dict[str,int],
                               runtime_input,cxx:str="g++",
                               timeout_seconds:int=90,max_events:int=100_000,
                               atol:float=1e-5,rtol:float=1e-4,
                               original_trace_output:str|Path|None=None)->dict:
    if (type(timeout_seconds) is not int or not 1<=timeout_seconds<=180
            or type(max_events) is not int or not 1<=max_events<=100_000):
        raise ValueError("R2e source and vendor-free runtime require strict finite bounds")
    if original_trace_output is not None:
        dest=Path(original_trace_output)
        if dest.exists() or dest.is_symlink():
            raise FileExistsError("Original R2e GCC trace is create-only")
        if not dest.parent.is_dir():
            raise ValueError("Original R2e evidence parent missing")
    compiler=shutil.which(cxx)
    if compiler is None:
        raise FileNotFoundError("Original C++ g++ tool required")
    model=arch.functional
    (m,k,n,q),(h,a,b,g),weight_names,_=admit_branch(model)
    manifest=emission.manifest
    cpp_sha=sha256(emission.cpp.encode()).hexdigest()
    if (manifest.get("schema")!=SOURCE_SCHEMA or
            manifest.get("cpp_sha256")!=cpp_sha or
            manifest.get("functional_sha256")!=functional_fingerprint(model) or
            manifest.get("architecture_sha256")!=architecture_fingerprint(arch) or
            manifest.get("scratch_offsets")!=dict(sorted(offsets.items())) or
            manifest.get("expected_producer_H_outputs")!=m*n or
            manifest.get("expected_producer_contributions")!=m*k*n or
            manifest.get("expected_intermediate_H_numeric_values")!=m*n):
        raise ValueError("R2e original C++/model/Phi/input geometry lineage mismatch")
    x=np.asarray(runtime_input,dtype=np.float32)
    if x.shape!=(m,k) or not np.all(np.isfinite(x)):
        raise ValueError("R2e expected finite source input matrix")
    desired=np.asarray(evaluate(
        model,{model.inputs[0]:x})[model.outputs[0]],
        dtype=np.float32).reshape(-1).astype(np.float64)
    runner=(
        "\nint main() {\n"
        f"  float input[{m*k}]={{{','.join(_float_literal(v) for v in x.flat)}}};\n"
        f"  float output[{m*q}]={{}};\n"
        "  fpgai_model(input,output);\n"
        f"  for(int i=0;i<{m*q};++i) "
        'std::printf("FPGAI_R2B OUTPUT %d %.9g\\n",i,output[i]);\n'
        "  return 0;\n}\n"
    )
    with TemporaryDirectory(prefix="cst_r2e_original_GCC_") as folder:
        src=Path(folder)/"producer_contribution_runtime.cpp"
        exe=Path(folder)/"run"
        src.write_text(emission.cpp+runner)
        compile_result=subprocess.run(
            [compiler,"-std=c++17","-O0","-ffp-contract=off",
             "-DFPGAI_HOST_FLOAT","-DFPGAI_TRACE_R2D",
             str(src),"-o",str(exe)],
            capture_output=True,text=True,timeout=timeout_seconds)
        if compile_result.returncode:
            raise RuntimeError("R2e original C++ compilation failed: "+
                               compile_result.stderr[-4000:])
        executed=subprocess.run(
            [str(exe)],capture_output=True,text=True,timeout=timeout_seconds)
        if executed.returncode:
            raise RuntimeError("R2e original C++ execution failed: "+
                               executed.stderr[-4000:])
    producer=inspect_producer_stdout(
        executed.stdout,m=m,k=k,n=n,h=h,max_events=max_events)
    events,actual=_decode_trace(producer["typed_branch_stdout"],
                                n_outputs=m*q,max_events=max_events)
    # The independent H reference is evaluated BEFORE the final Add/ReLU
    # can hide incorrect producer values (for example by ReLU clamping).
    w0=np.asarray(model.tensor(weight_names[0]).data,
                  dtype=np.float32).reshape(k,n)
    expected_h=np.matmul(x,w0).astype(np.float64).reshape(-1)
    actual_h=np.asarray(producer["runtime_H_values"],dtype=np.float64)
    if (actual_h.shape!=(m*n,) or not np.all(np.isfinite(actual_h))
            or not np.allclose(actual_h,expected_h,atol=atol,rtol=rtol)):
        raise AssertionError(
            "R2E_INDEPENDENT_PRODUCER_H_NUMERIC_MISMATCH: "+
            str(float(np.max(np.abs(actual_h-expected_h)))))
    issue=validate_phi_issue_conformance(arch,events)
    if issue["status"]!="VERIFIED_E0":
        raise AssertionError("R2E_INDEPENDENT_PHI_ISSUE_ORACLE_REJECTED")
    safety=validate_branch_events(arch,offsets,events,max_events=max_events)
    if safety["status"]!="VERIFIED_E0":
        raise AssertionError("R2E_CONSUMER_REDUCTION_OR_ALIAS_VIOLATION: "+
                             str(safety.get("counterexample")))
    if not np.allclose(actual,desired,atol=atol,rtol=rtol):
        raise AssertionError(
            "R2E_INDEPENDENT_NUMPY_NUMERIC_MISMATCH: "+
            str(float(np.max(np.abs(actual-desired)))))
    if original_trace_output is not None:
        with Path(original_trace_output).open("x",encoding="utf-8") as stream:
            stream.write(executed.stdout)
    return {
        "schema":SCHEMA,
        "status":"ORIGINAL_R2E_GCC_PRODUCER_CONTRIBUTIONS_AND_NUMPY_VERIFIED_E1",
        "source_cpp_sha256":cpp_sha,
        "functional_sha256":manifest["functional_sha256"],
        "architecture_sha256":manifest["architecture_sha256"],
        "original_input_sha256":sha256(x.tobytes()).hexdigest(),
        "original_full_stdout_sha256":sha256(executed.stdout.encode()).hexdigest(),
        "original_producer_trace_sha256":producer["producer_trace_sha256"],
        "runtime_H_values_sha256":producer["producer_H_values_sha256"],
        "original_intermediate_H_numeric_max_abs_error":
            float(np.max(np.abs(actual_h-expected_h))),
        "independent_complete_H_producer_contributions":producer["observed_contributions"],
        "independent_complete_H_producer_commits":producer["producer_commits"],
        "independent_H_store_witnesses":producer["H_typed_source_commits"],
        "independent_reader_Phi_issue_signature_sha256":
            issue["observed_issue_signature_sha256"],
        "original_source_reduction_alias_verified":True,
        "original_numeric_max_absolute_error":
            float(np.max(np.abs(actual-desired))),
        "original_GCC_compilations":1,"original_GCC_executions":1,
        "original_Vitis_calls":0,"original_Vivado_calls":0,
        "E0_producer_completeness_witnessed":True,
        "E1_actual_source_producer_trace_and_numeric_verified":True,
        "FPGA_cycle_producer_completion_proven":False,
        "source_claim":"bounded-runtime-producer-COMPLETE-before-H-COMMIT-plus-Float32-numeric",
        "limitations":[
            "Runtime multiply terms and intermediate H values checked against independent NumPy, not a formal proof of arbitrary C++ arithmetic",
            "Actual H producer loop source is serialized, not concurrent hardware schedule",
            "Only admitted five-operator MatMul diamond; no arbitrary multi-operator producer",
            "No native Vitis, RTL, RAM bank/port, physical QoR or external compiler novelty evidence",
        ],
    }
