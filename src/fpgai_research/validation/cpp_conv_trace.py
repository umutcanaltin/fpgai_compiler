"""Original GCC E1 Conv READ-index and Phi ownership trace verifier.

Opt-in instrumented standard HLS code emits each actual per-kernel
input load address through the SAME local variable used by the C++ load.
The observed multiset is checked against independent NumPy sliding-window
source semantics and resolved consumer Phi. This is NOT a full temporal
C++ read/commit proof, original Vitis CSim, RTL or memory port certificate.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
from math import prod
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

from fpgai_research.architecture.resolver import coordinate
from fpgai_research.model_ir.conv_boundary import (
    ConvValueEvent, admissible_conv_diamond,conv_logical_events,
)
from fpgai_research.model_ir.semantics import functional_fingerprint
from fpgai_research.validation.conv_access_oracle import independent_conv_access

SCHEMA="fpgai-research/cst-original-gcc-conv-read-trace-e1/v1"


def trace_conv_cpp(arch,emission,*,timeout_seconds:int=90,cxx="g++"):
    admissible_conv_diamond(arch)
    model=arch.functional
    manifest=emission.manifest
    if (not manifest.get("conv_source_read_trace_opt_in") or
            manifest.get("functional_sha256")!=functional_fingerprint(model) or
            manifest.get("cpp_sha256")!=sha256(emission.cpp.encode()).hexdigest()):
        raise ValueError("Conv source trace needs original opt-in source and matching math hash")
    if len(manifest["operators"])!=len(model.operators):
        raise ValueError("Missing actual operator architecture receipts")
    for op,point,receipt in zip(model.operators,arch.points,manifest["operators"]):
        if (receipt["operator"]!=op.name or receipt["kind"]!=op.kind or
                receipt["effective"]!=vars(point.effective)):
            raise ValueError("Original Conv source mapping differs from resolved Phi")
    if (type(timeout_seconds) is not int or not 1<=timeout_seconds<=180):
        raise ValueError("Bounded C++ execution timeout required")
    compiler=shutil.which(cxx)
    if compiler is None:
        raise FileNotFoundError("Actual host GCC compiler required")
    x=model.tensor(model.inputs[0])
    y=model.tensor(model.outputs[0])
    runner=(f"\nint main() {{ float x[{prod(x.shape)}]={{}}; "
            f"float y[{prod(y.shape)}]={{}}; fpgai_model(x,y); return 0; }}\n")
    with TemporaryDirectory(prefix="fpgai_conv_read_trace_") as folder:
        src=Path(folder)/"conv.cpp"
        binary=Path(folder)/"conv_bin"
        src.write_text(emission.cpp+runner)
        build=subprocess.run([compiler,"-std=c++17","-O0","-Wno-unknown-pragmas",
            "-DFPGAI_HOST_FLOAT","-DFPGAI_TRACE_CONV_SOURCE",
            str(src),"-o",str(binary)],
            capture_output=True,text=True,timeout=timeout_seconds)
        if build.returncode:
            raise RuntimeError("Source Conv GCC build failed: "+build.stderr[-4500:])
        run=subprocess.run([str(binary)],capture_output=True,text=True,
                           timeout=timeout_seconds)
        if run.returncode:
            raise RuntimeError("Source Conv GCC execution failed: "+run.stderr[-4500:])
    op=model.operators[2]
    ordinal=2
    lhs=model.tensor(op.inputs[0])
    out=model.tensor(op.outputs[0])
    weight=model.tensor(op.inputs[1])
    expected=conv_logical_events(arch)
    planned=Counter(
        (e.output_element,e.element,e.reduction_k,e.phi_owner)
        for e in expected if e.kind=="READ" and e.consumer==op.name)
    observed=Counter()
    actual_events=[]
    for i,line in enumerate(run.stdout.splitlines()):
        bits=line.split()
        if len(bits)!=9 or bits[0]!="FPGAI_CONV_READ":
            raise AssertionError(f"Unexpected Conv source trace line {i}: {line[:120]}")
        try:
            values=tuple(map(int,bits[1:]))
        except ValueError as exc:
            raise AssertionError("Source Conv trace has invalid integer coordinates") from exc
        stage,out_idx,source_idx,red,replica,pe,lane,fold=values
        if stage!=ordinal:
            raise AssertionError("Wrong source Conv operator ordinal")
        if not(0<=source_idx<prod(lhs.shape) and
               0<=out_idx<prod(out.shape) and 0<=red<prod(weight.shape[1:])):
            raise AssertionError("Out-of-range actual emitted Conv access")
        coords=[]
        tmp=out_idx
        for dim in reversed(out.shape):
            coords.append(tmp%dim)
            tmp//=dim
        batch,oc,oh,ow=reversed(coords)
        ic=red//(weight.shape[2]*weight.shape[3])
        rem=red%(weight.shape[2]*weight.shape[3])
        ky,kx=divmod(rem,weight.shape[3])
        phi=coordinate(arch.points[ordinal],(batch,oc,oh,ow,ic,ky,kx))
        owner=(phi["replica"],phi["pe"],phi["lane"],phi["reuse_step"][4])
        if owner!=(replica,pe,lane,fold):
            raise AssertionError("ACTUAL_CONV_PHI_OWNER_MISMATCH")
        signature=(out_idx,source_idx,red,owner)
        observed[signature]+=1
        actual_events.append(ConvValueEvent(
            "READ",op.inputs[0],source_idx,op.name,out_idx,0,red,owner))
    if observed!=planned:
        missing=planned-observed
        unexpected=observed-planned
        raise AssertionError("ACTUAL_CONV_SOURCE_READ_MISMATCH "+
                             str({"missing":next(iter(missing),None),
                                  "unexpected":next(iter(unexpected),None)}))
    other=tuple(e for e in expected
                if not(e.kind=="READ" and e.consumer==op.name))
    semantic=independent_conv_access(arch,other+tuple(actual_events))
    if semantic["status"]!="VERIFIED_INDEPENDENT_CONV_READS_E0":
        raise AssertionError("ACTUAL_CONV_INDEPENDENT_NUMPY_READ_REJECTED")
    return {
        "schema":SCHEMA,"status":"ACTUAL_CONV_KERNEL_LOADS_VERIFIED_E1",
        "source_cpp_sha256":manifest["cpp_sha256"],
        "source_read_count":sum(observed.values()),
        "independent_numpy_multiset_sha256":semantic["multiset_sha256"],
        "raw_gcc_trace_sha256":sha256(run.stdout.encode()).hexdigest(),
        "mapped_consumer":op.name,"mapped_producer":model.operators[0].name,
        "native_vitis_calls":0,
        "evidence":"Original GCC actual Conv per-kernel source LOAD indices and resolved Phi",
        "limitations":[
            "Only Conv reader kernel loads attested, not all graph scalar loads/COMMIT order",
            "Source-code instrumentation only under FPGAI_HOST_FLOAT, no original Vitis",
            "No hardware cycle, inferred RAM port, utilization, timing or physical qualification",
        ],
    }
