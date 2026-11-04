"""E1 independent numerical refinement of every executed NCHW graph boundary.

Runs the original emitted HLS C++ with FPGAI_TRACE_TENSORS to read every
intermediate value *after* its operator has run. Compares every observed
tensor, not just the final output, against the independent NumPy graph
interpreter. These are post-operator snapshots; they do NOT prove actual
individual memory WRITE/COMMIT or consumer READ scheduling.
"""
from __future__ import annotations

from hashlib import sha256
from math import prod
from pathlib import Path
import json
import shutil
import subprocess
from tempfile import TemporaryDirectory

import numpy as np

from fpgai_research.model_ir.nchw_graph_contract import admit_nchw_graph
from fpgai_research.model_ir.semantics import (
    architecture_fingerprint,functional_fingerprint,
)
from fpgai_research.validation.reference import evaluate
from fpgai_research.validation.host import _float_literal

SCHEMA="fpgai-research/cst-NCHW-GCC-intermediate-snapshots-E1/v1"


def attest_nchw_intermediates(
    arch,emission,runtime_input,*,rtol:float=1e-4,atol:float=1e-5,
    max_elements:int=100_000,timeout_seconds:int=90,cxx:str="g++",
):
    model,_=admit_nchw_graph(arch)
    manifest=emission.manifest
    if (manifest.get("functional_sha256")!=functional_fingerprint(model)
            or manifest.get("cpp_sha256")!=sha256(emission.cpp.encode()).hexdigest()
            or len(manifest.get("operators",()))!=len(model.operators)):
        raise ValueError("Functional/source digest or operator count mismatch")
    for op,point,receipt in zip(model.operators,arch.points,manifest["operators"]):
        if (receipt.get("operator")!=op.name or receipt.get("kind")!=op.kind
                or receipt.get("effective")!=vars(point.effective)):
            raise ValueError("Original C++ operator Phi differs from resolved graph")
    if (type(max_elements) is not int or max_elements<1 or
            type(timeout_seconds) is not int or not 1<=timeout_seconds<=180):
        raise ValueError("Bounded trace elements and timeout required")
    expected_count=sum(prod(model.tensor(op.outputs[0]).shape)
                       for op in model.operators)
    if expected_count>max_elements:
        return {"schema":SCHEMA,"status":"UNKNOWN_ELEMENT_BUDGET",
                "required_elements":expected_count,"max_elements":max_elements,
                "native_vitis_calls":0}
    compiler=shutil.which(cxx)
    if compiler is None:
        raise FileNotFoundError("Actual GCC required")
    input_tensor=np.asarray(runtime_input,dtype=np.float32)
    if input_tensor.shape!=model.tensor(model.inputs[0]).shape:
        raise ValueError("Runtime input shape mismatch")
    if not np.all(np.isfinite(input_tensor)):
        raise ValueError("Nonfinite input is unsupported")
    expected=evaluate(model,{model.inputs[0]:input_tensor},return_all=True)
    output_shape=model.tensor(model.outputs[0]).shape
    output_count=prod(output_shape)
    literals=",".join(_float_literal(x) for x in input_tensor.flat)
    runner=(
        f"\nint main() {{ float input[{input_tensor.size}]={{{literals}}}; "
        f"float output[{output_count}]={{}}; fpgai_model(input,output); "
        f"for(int i=0;i<{output_count};++i) "
        'std::printf("FPGAI_FINAL %d %.9g\\n",i,double(output[i])); '
        "return 0; }\n"
    )
    with TemporaryDirectory(prefix="fpgai_nchw_snapshot_") as directory:
        path=Path(directory)/"source.cpp"
        binary=Path(directory)/"run"
        path.write_text(emission.cpp+runner)
        compile_result=subprocess.run([
            compiler,"-std=c++17","-O0","-Wno-unknown-pragmas",
            "-DFPGAI_HOST_FLOAT","-DFPGAI_TRACE_TENSORS",
            str(path),"-o",str(binary)
        ],capture_output=True,text=True,timeout=timeout_seconds)
        if compile_result.returncode:
            raise RuntimeError("Actual NCHW HLS-source GCC build failed: "+
                               compile_result.stderr[-5000:])
        executed=subprocess.run([str(binary)],capture_output=True,text=True,
                                timeout=timeout_seconds)
        if executed.returncode:
            raise RuntimeError("Actual generated C++ execution failed: "+
                               executed.stderr[-5000:])
    actual={op.outputs[0]:[] for op in model.operators}
    seen_final=[]
    stage=0
    cursor=0
    for line_number,line in enumerate(executed.stdout.splitlines()):
        fields=line.split()
        if len(fields)!=4 or fields[0]!="FPGAI_TENSOR":
            if len(fields)==3 and fields[0]=="FPGAI_FINAL":
                if stage!=len(model.operators) or int(fields[1])!=len(seen_final):
                    raise AssertionError("Out-of-order external final output")
                seen_final.append(float(fields[2]))
                continue
            raise AssertionError("Unexpected GCC tensor trace row "+str(line_number))
        try:
            ordinal,index=int(fields[1]),int(fields[2])
            value=float(fields[3])
        except ValueError as exc:
            raise AssertionError("Invalid original C++ tensor trace") from exc
        if stage>=len(model.operators) or ordinal!=stage or index!=cursor:
            raise AssertionError("Duplicate, reordered, or missing tensor snapshot")
        name=model.operators[stage].outputs[0]
        if not np.isfinite(value):
            raise AssertionError("Nonfinite intermediate observed from original C++")
        actual[name].append(value)
        cursor+=1
        if cursor==prod(model.tensor(name).shape):
            stage+=1
            cursor=0
    if stage!=len(model.operators) or cursor or len(seen_final)!=output_count:
        raise AssertionError("Incomplete intermediate or final source evidence")
    summaries=[]
    for op in model.operators:
        name=op.outputs[0]
        numbers=np.asarray(actual[name],dtype=np.float64).reshape(
            model.tensor(name).shape)
        oracle=expected[name].astype(np.float64)
        if not np.allclose(numbers,oracle,rtol=rtol,atol=atol):
            index=int(np.argmax(np.abs(numbers-oracle)))
            raise AssertionError(
                "SOURCE_NCHW_INTERMEDIATE_NUMERICAL_MISMATCH "+
                f"operator={op.name} tensor={name} element={index} "+
                f"source={numbers.flat[index]:.9g} reference={oracle.flat[index]:.9g}"
            )
        summaries.append({
            "operator":op.name,"tensor":name,"kind":op.kind,
            "elements":numbers.size,
            "host_float_max_abs_error":float(np.max(np.abs(numbers-oracle))),
            "observed_tensor_sha256":sha256(
                json.dumps(numbers.ravel().tolist(),separators=(",",":")).encode()
            ).hexdigest(),
        })
    final=np.asarray(seen_final,dtype=np.float64)
    if not np.allclose(final,np.asarray(actual[model.outputs[0]],dtype=np.float64),
                       rtol=rtol,atol=atol):
        raise AssertionError("External C++ output and last source tensor diverge")
    return {
        "schema":SCHEMA,"status":"ACTUAL_NCHW_INTERMEDIATE_VALUES_VERIFIED_E1",
        "functional_sha256":manifest["functional_sha256"],
        "architecture_sha256":architecture_fingerprint(arch),
        "source_cpp_sha256":manifest["cpp_sha256"],
        "input_sha256":sha256(input_tensor.tobytes()).hexdigest(),
        "observed_raw_trace_sha256":sha256(executed.stdout.encode()).hexdigest(),
        "operator_snapshots":len(summaries),
        "observed_tensor_elements":expected_count,
        "individual_operator_receipts":summaries,
        "native_vitis_calls":0,
        "evidence":"E1-original-GCC-post-operator-tensor-snapshot-vs-NumPy",
        "limitations":[
            "Post-operator values are observed, NOT every actual READ or WRITE event",
            "Does not prove global source-order last-use or materialized scratch aliases",
            "HOST_FLOAT GCC is not original fixed-point Vitis CSim/CSynth or RTL",
        ],
    }
