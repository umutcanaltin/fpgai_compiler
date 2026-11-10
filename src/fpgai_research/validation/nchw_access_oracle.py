"""Independent NumPy index-array oracle for mixed static NCHW DAG reads.

Expected source element/port/reduction identities come from NumPy-padded
sliding windows, broadcasts and concatenated identity grids, not from the
production NCHW compiler access helper. Logical multiset plus an independently calculated Conv output/channel/reduction
Phi ownership reference. Does not import resolver.coordinate or the E0
source-event generator. Source loop timing, scalar ownership, hardware cycles
and actual bank/port correctness remain outside this E0 oracle.
"""
from __future__ import annotations
from collections import Counter
from hashlib import sha256
from math import prod
import json
import numpy as np


def _independent_conv_owner(point, out_channel:int, input_channel:int):
    """Closed-form Conv Phi owner, independently of the resolver and emitter."""
    if point.kind!="Conv" or len(point.iteration_shape)!=7:
        raise ValueError("Independent owner oracle expects resolved Conv")
    a=point.effective
    shape=point.iteration_shape
    if (min(a.parallelism,a.pes,a.simd,a.serialization)<1 or
            (a.tile and len(a.tile)!=7)):
        raise ValueError("Nonpositive or malformed Conv source mapping")
    if not (0<=out_channel<shape[1] and 0<=input_channel<shape[4]):
        raise ValueError("Conv source channel index out of range")
    tile_out=a.tile[1] if a.tile else shape[1]
    tile_reduce=a.tile[4] if a.tile else shape[4]
    if min(tile_out,tile_reduce)<1:
        raise ValueError("Invalid source Conv tile extent")
    local_output=out_channel%tile_out
    lanes_per_group=a.parallelism*a.pes
    return (
        (local_output//a.pes)%a.parallelism,
        local_output%a.pes,
        (input_channel%tile_reduce)%a.simd,
        (local_output//lanes_per_group)%a.serialization,
    )


def independent_nchw_access(arch,events,*,max_events:int=100_000):
    model=arch.functional
    model.validate()
    if type(max_events) is not int or max_events<1 or len(events)>max_events:
        raise ValueError("Finite source-access oracle budget exceeded")
    produced={x.outputs[0] for x in model.operators}
    expected=Counter()
    expected_owner=Counter()
    conv_names={op.name for op in model.operators if op.kind=="Conv"}
    for ordinal,op in enumerate(model.operators):
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
                                key=(name,source,op.name,outidx,0,red)
                                expected[key]+=1
                                expected_owner[(*key,_independent_conv_owner(
                                    arch.points[ordinal],oc,ic))]+=1
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
                "reason":"NCHW_SOURCE_ACCESS_MISMATCH",
                "expected_reads":sum(expected.values()),
                "actual_reads":sum(actual.values()),
                "first_missing":next(iter(sorted(missing)),None),
                "first_unexpected":next(iter(sorted(unexpected)),None),
                "original_vitis_calls":0}
    observed_owner=Counter(
        (x.tensor,x.element,x.consumer,x.output_element,x.input_port,
         x.reduction_k,x.phi_owner)
        for x in events if x.kind=="READ" and x.consumer in conv_names)
    if observed_owner!=expected_owner:
        missing=expected_owner-observed_owner
        unexpected=observed_owner-expected_owner
        return {"schema":"fpgai-research/independent-NCHW-graph-read-oracle/v1",
                "status":"REJECTED_INDEPENDENT_NCHW_ACCESS",
                "reason":"INDEPENDENT_CONV_PHI_OWNER_MISMATCH",
                "expected_conv_owner_reads":sum(expected_owner.values()),
                "observed_conv_owner_reads":sum(observed_owner.values()),
                "first_missing_owner":next(iter(missing),None),
                "first_unexpected_owner":next(iter(unexpected),None),
                "original_vitis_calls":0}
    owner_rows=sorted((list(key),count) for key,count in expected_owner.items())
    owner_digest=sha256(json.dumps(owner_rows,sort_keys=True,
                 separators=(",",":"),allow_nan=False).encode()).hexdigest()
    rows=sorted((list(k),v) for k,v in expected.items())
    signature=sha256(json.dumps(rows,sort_keys=True,
               separators=(",",":")).encode()).hexdigest()
    return {"schema":"fpgai-research/independent-NCHW-graph-read-oracle/v1",
            "status":"VERIFIED_INDEPENDENT_NCHW_ACCESS_E0",
            "reads":sum(expected.values()),
            "access_multiset_sha256":signature,
            "phi_owner_checked_conv_reads":sum(expected_owner.values()),
            "phi_owner_multiset_sha256":owner_digest,
            "phi_owner_validation":"VERIFIED_INDEPENDENT_CONV_OWNER_E0",
            "oracle_independent_of_resolver_coordinate":True,
            "original_vitis_calls":0,
            "evidence":"E0-NumPy-access-and-closed-form-Conv-owner-only",
            "limitations":[
                "Independent logical Conv owner, not source-issue cycle order",
                "Other scalar, pooling and concatenation owners not verified",
                "No physical PE, memory bank, HLS or RTL qualification",
            ]}
