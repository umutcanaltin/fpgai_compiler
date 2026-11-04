"""Independent, input-sensitive NumPy oracle for actual NCHW C++ READ/WRITE counts.

The C++ code's ReLU ternary reads positive inputs a second time and
MaxPool's seed read duplicates its (ky=0,kx=0) access. Neither belongs
to the minimal mathematical E0 relation. Do not use compiler source
access-map helper or a replay of the generator's planned event list.
"""
from __future__ import annotations
from collections import Counter
from math import prod
import numpy as np

from fpgai_research.model_ir.nchw_graph_contract import admit_nchw_graph
from fpgai_research.validation.reference import evaluate


def expected_actual_nchw_counts(arch,source_input):
    model,produced=admit_nchw_graph(arch)
    x=np.asarray(source_input,dtype=np.float32)
    if x.shape!=model.tensor(model.inputs[0]).shape or not np.all(np.isfinite(x)):
        raise ValueError("Finite NCHW runtime input with exact shape required")
    reference=evaluate(model,{model.inputs[0]:x},return_all=True)
    from numpy.lib.stride_tricks import sliding_window_view
    expected=Counter()
    for ordinal,op in enumerate(model.operators):
        outshape=model.tensor(op.outputs[0]).shape
        if op.kind=="Conv" and op.inputs[0] in produced:
            name=op.inputs[0]
            shape=model.tensor(name).shape
            kernel=model.tensor(op.inputs[1]).shape
            sy,sx=op.attr("strides",(1,1))
            p=op.attr("pads",(0,0,0,0))
            indexes=np.arange(prod(shape),dtype=np.int64).reshape(shape)
            padded=np.pad(indexes,((0,0),(0,0),(p[0],p[2]),(p[1],p[3])),
                          constant_values=-1)
            windows=sliding_window_view(padded,kernel[2:],axis=(2,3))
            windows=windows[:,:,::sy,::sx,:,:]
            if windows.shape[2:4]!=outshape[2:]:
                raise ValueError("Independent Conv output geometry mismatch")
            for n in range(outshape[0]):
                for oc in range(outshape[1]):
                    for oy in range(outshape[2]):
                        for ox in range(outshape[3]):
                            for ic in range(shape[1]):
                                for idx in windows[n,ic,oy,ox].flat:
                                    if idx>=0:
                                        expected[("R",ordinal,name,int(idx))]+=1
        elif op.kind=="Relu" and op.inputs[0] in produced:
            name=op.inputs[0]
            for idx,value in enumerate(reference[name].flat):
                expected[("R",ordinal,name,idx)]+=1+int(float(value)>0)
        elif op.kind=="Add":
            for name in op.inputs:
                if name not in produced:
                    continue
                shape=model.tensor(name).shape
                ids=np.arange(prod(shape),dtype=np.int64).reshape(shape)
                for idx in np.broadcast_to(ids,outshape).flat:
                    expected[("R",ordinal,name,int(idx))]+=1
        elif op.kind=="Concat":
            for name in op.inputs:
                if name in produced:
                    for idx in range(prod(model.tensor(name).shape)):
                        expected[("R",ordinal,name,idx)]+=1
        elif op.kind=="MaxPool" and op.inputs[0] in produced:
            name=op.inputs[0]
            shape=model.tensor(name).shape
            ids=np.arange(prod(shape),dtype=np.int64).reshape(shape)
            windows=sliding_window_view(ids,(2,2),axis=(2,3))
            windows=windows[:,:,::2,::2,:,:]
            if windows.shape[2:4]!=outshape[2:]:
                raise ValueError("Independent MaxPool geometry mismatch")
            for block in windows.reshape(-1,4):
                expected[("R",ordinal,name,int(block[0]))]+=1 # C++ mx seed
                for idx in block:
                    expected[("R",ordinal,name,int(idx))]+=1
        for idx in range(prod(outshape)):
            expected[("W",ordinal,op.outputs[0],idx)]+=1
    return expected,reference
