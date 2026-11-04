"""Original GCC Conv-kernel load attestation for general bounded NCHW DAGs.

Every emitted Conv whose source is an earlier produced tensor is traced
through the *same C++ local index variable actually used for its load*.
All observed Conv READ tuples must match E0 ownership and the independent
NumPy graph access multiset. Elementwise, pool and sink reads are logical
E0 only; stock HLS emission uses its own arrays (no source-storage reuse).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from hashlib import sha256
from math import prod
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

from fpgai_research.architecture.resolver import coordinate
from fpgai_research.model_ir.nchw_graph_contract import (
    admit_nchw_graph,nchw_graph_events,
)
from fpgai_research.validation.nchw_access_oracle import independent_nchw_access


def trace_nchw_conv_sources(arch,emission,*,cxx="g++",timeout_seconds=90):
    model,produced=admit_nchw_graph(arch)
    manifest=emission.manifest
    if (not manifest.get("conv_source_read_trace_opt_in") or
            manifest.get("cpp_sha256")!=sha256(emission.cpp.encode()).hexdigest()):
        raise ValueError("Must use unchanged opt-in Conv READ source")
    if len(manifest.get("operators",()))!=len(model.operators):
        raise ValueError("Missing original generated operator receipts")
    for op,p,receipt in zip(model.operators,arch.points,manifest["operators"]):
        if (receipt["operator"]!=op.name or receipt["kind"]!=op.kind or
                receipt["effective"]!=vars(p.effective)):
            raise ValueError("Mismatch between original HLS source Phi and graph")
    if type(timeout_seconds) is not int or not 1<=timeout_seconds<=180:
        raise ValueError("Finite GCC execution timeout required")
    compiler=shutil.which(cxx)
    if compiler is None:
        raise FileNotFoundError("Real host GCC required")
    x=model.tensor(model.inputs[0]).shape
    y=model.tensor(model.outputs[0]).shape
    runner=f"\nint main() {{float x[{prod(x)}]={{}}; float y[{prod(y)}]={{}}; fpgai_model(x,y); return 0;}}\n"
    with TemporaryDirectory(prefix="fpgai_nchw_multi_conv_") as folder:
        file=Path(folder)/"generated.cpp"
        binary=Path(folder)/"generated"
        file.write_text(emission.cpp+runner)
        build=subprocess.run([compiler,"-std=c++17","-O0","-Wno-unknown-pragmas",
            "-DFPGAI_HOST_FLOAT","-DFPGAI_TRACE_CONV_SOURCE",
            str(file),"-o",str(binary)],capture_output=True,text=True,
            timeout=timeout_seconds)
        if build.returncode:
            raise RuntimeError("Actual Conv graph source C++ build failed: "+build.stderr[-4500:])
        executed=subprocess.run([str(binary)],capture_output=True,text=True,
                                timeout=timeout_seconds)
        if executed.returncode:
            raise RuntimeError("Actual Conv graph execution failed: "+executed.stderr[-4500:])
    logical=nchw_graph_events(arch)
    owner_expected=Counter()
    for e in logical:
        if e.kind=="READ" and e.consumer in {op.name for op in model.operators
                                              if op.kind=="Conv"}:
            ordinal=next(i for i,o in enumerate(model.operators) if o.name==e.consumer)
            owner_expected[(ordinal,e.output_element,e.element,e.reduction_k,
                            *e.phi_owner)]+=1
    observed=Counter()
    actual_conv_events=[]
    for line_no,line in enumerate(executed.stdout.splitlines()):
        parts=line.split()
        if len(parts)!=9 or parts[0]!="FPGAI_CONV_READ":
            raise AssertionError("Unexpected original Conv source line "+str(line_no))
        try:
            ordinal,output_index,src_index,red,replica,pe,lane,fold=map(int,parts[1:])
        except ValueError as exc:
            raise AssertionError("Nonintegral Conv read source event") from exc
        if not 0<=ordinal<len(model.operators) or model.operators[ordinal].kind!="Conv":
            raise AssertionError("Unexpected non-Conv source stage")
        op=model.operators[ordinal]
        if op.inputs[0] not in produced:
            raise AssertionError("Source traced an external runtime input")
        xshape=model.tensor(op.inputs[0]).shape
        outshape=model.tensor(op.outputs[0]).shape
        wshape=model.tensor(op.inputs[1]).shape
        if not (0<=output_index<prod(outshape) and 0<=src_index<prod(xshape)
                and 0<=red<prod(wshape[1:])):
            raise AssertionError("Out-of-range actual emitted NCHW Conv source address")
        from numpy import unravel_index
        n,oc,oy,ox=unravel_index(output_index,outshape)
        ic=red//(wshape[2]*wshape[3])
        ky,kx=divmod(red%(wshape[2]*wshape[3]),wshape[3])
        phi=coordinate(arch.points[ordinal],(n,oc,oy,ox,ic,ky,kx))
        owner=(phi["replica"],phi["pe"],phi["lane"],phi["reuse_step"][4])
        if (replica,pe,lane,fold)!=owner:
            raise AssertionError("ACTUAL_MULTI_CONV_PHI_OWNER_INVALID")
        observed[(ordinal,output_index,src_index,red,replica,pe,lane,fold)]+=1
        from fpgai_research.model_ir.conv_boundary import ConvValueEvent
        actual_conv_events.append(ConvValueEvent(
            "READ",op.inputs[0],src_index,op.name,output_index,0,red,owner))
    if observed!=owner_expected:
        missing=owner_expected-observed
        unexpected=observed-owner_expected
        raise AssertionError("ACTUAL_NCHW_CONV_LOAD_MAP_MISMATCH "+
            str({"missing":next(iter(missing),None),
                 "unexpected":next(iter(unexpected),None)}))
    # Substitute ACTUAL executed Conv per-tap accesses into a logical
    # multiset with the other operators' planned E0 accesses; this does
    # NOT claim scalar READ/COMMIT events came from the original source.
    conv_consumer_names={o.name for o in model.operators if o.kind=="Conv"}
    other=tuple(e for e in logical
                if not(e.kind=="READ" and e.consumer in conv_consumer_names))
    independent=independent_nchw_access(arch,other+tuple(actual_conv_events))
    if independent["status"]!="VERIFIED_INDEPENDENT_NCHW_ACCESS_E0":
        raise AssertionError("INDEPENDENT_NUMPY_REJECTS_ORIGINAL_CONV_LOAD")
    return {"schema":"fpgai-research/NCHW-general-DAG-GCC-Conv-load-attestation-E1/v1",
            "status":"ACTUAL_CONV_SOURCE_READS_VERIFIED_E1",
            "source_cpp_sha256":manifest["cpp_sha256"],
            "actual_conv_reader_taps":sum(observed.values()),
            "source_covered_conv_operators":len({x[0] for x in observed}),
            "observed_conv_trace_sha256":sha256(executed.stdout.encode()).hexdigest(),
            "independent_source_map_sha256":independent["access_multiset_sha256"],
            "native_vitis_calls":0,
            "limitations":[
                "Only Conv intermediate input reads are observed from real GCC source",
                "Other scalar/Concat/Pool READ and all COMMITs remain E0 modeled",
                "Source scratch reuse is NOT materialized in the canonical HLS emitter",
                "No RTL, original Vitis fixed-point/CSynth, physical port/clock claim",
            ]}
