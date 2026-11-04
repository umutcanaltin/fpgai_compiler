"""Independent NumPy index-array oracle for mixed static NCHW DAG reads.

Expected source element/port/reduction identities come from NumPy-padded
sliding windows, broadcasts and concatenated identity grids, not from the
production NCHW compiler access helper. Logical multiset only; no Phi
ownership or GCC temporal correctness is claimed here.
"""
from __future__ import annotations
from collections import Counter
from hashlib import sha256
from math import prod
import json
import numpy as np


def independent_nchw_access(arch,events,*,max_events:int=100_000):
    model=arch.functional
    model.validate()
    if type(max_events) is not int or max_events<1 or len(events)>max_events:
        raise ValueError("Finite source-access oracle budget exceeded")
    produced={x.outputs[0] for x in model.operators}
    expected=Counter()
    for op in model.operators:
        outshape=model.tensor(op.outputs[0]).shape
        if len(outshape)!=4:
            raise ValueError("Oracle requires NCHW intermediates")
        total=prod(outshape)
        if op.kind=="Conv":
            name=op.inputs[0]
            if name not in produced:
                continue
            sx=model.tensor(name).shape
            sw=model.tensor(op.inputs[1]).shape
            ids=np.arange(prod(sx),dtype=np.int64).reshape(sx)
            pads=tuple(op.attr("pads",(0,0,0,0)))
            stride=tuple(op.attr("strides",(1,1)))
            padded=np.pad(ids,((0,0),(0,0),(pads[0],pads[2]),
                               (pads[1],pads[3])),constant_values=-1)
            from numpy.lib.stride_tricks import sliding_window_view
            windows=sliding_window_view(padded,sw[2:],axis=(2,3))
            windows=windows[:,:,::stride[0],::stride[1],:,:]
            if windows.shape[2:4]!=outshape[2:]:
                raise ValueError("Conv NumPy output geometry disagrees")
            for outidx in range(total):
                n,oc,oy,ox=np.unravel_index(outidx,outshape)
                for ic in range(sx[1]):
                    for ky in range(sw[2]):
                        for kx in range(sw[3]):
                            source=int(windows[n,ic,oy,ox,ky,kx])
                            if source!=-1:
                                red=(ic*sw[2]+ky)*sw[3]+kx
                                expected[(name,source,op.name,outidx,0,red)]+=1
        elif op.kind=="Relu":
            name=op.inputs[0]
            if name in produced:
                for i in range(total):
                    expected[(name,i,op.name,i,0,-1)]+=1
        elif op.kind=="Add":
            for port,name in enumerate(op.inputs):
                if name not in produced:
                    continue
                shape=model.tensor(name).shape
                grid=np.arange(prod(shape),dtype=np.int64).reshape(shape)
                mapped=np.broadcast_to(grid,outshape)
                for i in range(total):
                    expected[(name,int(mapped.flat[i]),op.name,i,port,-1)]+=1
        elif op.kind=="Concat":
            grids=[np.arange(prod(model.tensor(name).shape),dtype=np.int64)
                   .reshape(model.tensor(name).shape)
                   for name in op.inputs]
            port_grids=[np.full(model.tensor(name).shape,port,dtype=np.int64)
                        for port,name in enumerate(op.inputs)]
            index_map=np.concatenate(grids,axis=1)
            port_map=np.concatenate(port_grids,axis=1)
            if index_map.shape!=outshape:
                raise ValueError("Concat NumPy output shape mismatch")
            for i in range(total):
                port=int(port_map.flat[i])
                name=op.inputs[port]
                if name in produced:
                    expected[(name,int(index_map.flat[i]),op.name,i,port,-1)]+=1
        elif op.kind=="MaxPool":
            name=op.inputs[0]
            if name not in produced:
                continue
            shape=model.tensor(name).shape
            ids=np.arange(prod(shape),dtype=np.int64).reshape(shape)
            from numpy.lib.stride_tricks import sliding_window_view
            windows=sliding_window_view(ids,(2,2),axis=(2,3))[:,:,::2,::2,:,:]
            if windows.shape[2:4]!=outshape[2:]:
                raise ValueError("Pool NumPy output shape mismatch")
            for i in range(total):
                n,c,y,x=np.unravel_index(i,outshape)
                for ky in range(2):
                    for kx in range(2):
                        expected[(name,int(windows[n,c,y,x,ky,kx]),
                                  op.name,i,0,ky*2+kx)]+=1
        else:
            raise ValueError("Oracle has no NCHW semantics for "+op.kind)
    sink=model.outputs[0]
    for i in range(prod(model.tensor(sink).shape)):
        expected[(sink,i,"EXTERNAL",i,0,-1)]+=1
    actual=Counter((x.tensor,x.element,x.consumer,x.output_element,
                    x.input_port,x.reduction_k)
                   for x in events if x.kind=="READ")
    if actual!=expected:
        missing=expected-actual
        unexpected=actual-expected
        return {"schema":"fpgai-research/independent-NCHW-graph-read-oracle/v1",
                "status":"REJECTED_INDEPENDENT_NCHW_ACCESS",
                "expected_reads":sum(expected.values()),
                "actual_reads":sum(actual.values()),
                "first_missing":next(iter(sorted(missing)),None),
                "first_unexpected":next(iter(sorted(unexpected)),None),
                "original_vitis_calls":0}
    rows=sorted((list(k),v) for k,v in expected.items())
    signature=sha256(json.dumps(rows,sort_keys=True,
               separators=(",",":")).encode()).hexdigest()
    return {"schema":"fpgai-research/independent-NCHW-graph-read-oracle/v1",
            "status":"VERIFIED_INDEPENDENT_NCHW_ACCESS_E0",
            "reads":sum(expected.values()),
            "access_multiset_sha256":signature,
            "original_vitis_calls":0,
            "evidence":"E0-NumPy-slide-window-broadcast-concat-pool-identity-grid-only"}
