"""Independent NumPy 4D NCHW source-index and reduction multiplicity oracle.

Uses NumPy padded index arrays and sliding-window indexing, not the E0
compiler's Conv source read-map helper. This checks which element is read,
NOT MatMul/Conv source scheduling, Phi lane ownership or FPGA hardware.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from math import prod
import numpy as np

SCHEMA="fpgai-research/cst-independent-NCHW-conv-access-oracle/v1"


def _digest(rows):
    return sha256(json.dumps(rows,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def independent_conv_access(arch,events):
    model=arch.functional
    expected=Counter()
    produced={op.outputs[0] for op in model.operators}
    for op in model.operators:
        outshape=model.tensor(op.outputs[0]).shape
        if op.kind=="Conv":
            if op.inputs[0] not in produced:
                continue
            xshape=model.tensor(op.inputs[0]).shape
            weight=model.tensor(op.inputs[1]).shape
            pads=tuple(op.attr("pads",(0,0,0,0)))
            strides=tuple(op.attr("strides",(1,1)))
            # Unique nonnegative input ids; out-of-input padding is -1.
            raw=np.arange(prod(xshape),dtype=np.int64).reshape(xshape)
            padded=np.pad(raw,((0,0),(0,0),(pads[0],pads[2]),
                               (pads[1],pads[3])),constant_values=-1)
            kh,kw=weight[2:]
            from numpy.lib.stride_tricks import sliding_window_view
            windows=sliding_window_view(padded,(kh,kw),axis=(2,3))
            windows=windows[:,:,::strides[0],::strides[1],:,:]
            if windows.shape[2:4]!=outshape[2:]:
                raise ValueError("Independent window geometry disagrees with output shape")
            for b in range(outshape[0]):
                for oc in range(outshape[1]):
                    for oy in range(outshape[2]):
                        for ox in range(outshape[3]):
                            target=np.ravel_multi_index((b,oc,oy,ox),outshape)
                            for ic in range(xshape[1]):
                                for ky in range(kh):
                                    for kx in range(kw):
                                        element=int(windows[b,ic,oy,ox,ky,kx])
                                        if element>=0:
                                            red=(ic*kh+ky)*kw+kx
                                            expected[(op.inputs[0],element,op.name,
                                                      int(target),0,red)]+=1
        elif op.kind in ("Relu","Add"):
            for port,name in enumerate(op.inputs):
                if name in produced:
                    for i in range(prod(outshape)):
                        expected[(name,i,op.name,i,port,-1)]+=1
        else:
            raise ValueError("Independent Conv oracle has no operator semantics")
    sink=model.outputs[0]
    for i in range(prod(model.tensor(sink).shape)):
        expected[(sink,i,"EXTERNAL",i,0,-1)]+=1
    observed=Counter(
        (e.tensor,e.element,e.consumer,e.output_element,e.input_port,e.reduction_k)
        for e in events if e.kind=="READ")
    if observed!=expected:
        missing=expected-observed
        unexpected=observed-expected
        return {"schema":SCHEMA,"status":"REJECTED_INDEPENDENT_CONV_READ_MAP",
                "missing":next(iter(sorted(missing)),None),
                "unexpected":next(iter(sorted(unexpected)),None),
                "needed_reads":sum(expected.values()),
                "actual_reads":sum(observed.values()),
                "original_vitis_calls":0}
    return {"schema":SCHEMA,"status":"VERIFIED_INDEPENDENT_CONV_READS_E0",
            "read_count":sum(expected.values()),
            "multiset_sha256":_digest(sorted((list(k),v) for k,v in expected.items())),
            "numpy_reference":"independent-padded-NCHW-sliding-window-index-grid",
            "original_vitis_calls":0,
            "limitations":[
                "Only independent logical source access, not Phi owner/ordering",
                "No executed source read trace, original Vitis or hardware bank proof",
            ]}
