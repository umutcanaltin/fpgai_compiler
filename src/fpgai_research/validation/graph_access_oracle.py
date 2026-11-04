"""Independent graph input-access oracle based on NumPy index arrays.

Not derived from `graph_access_maps.scalar_output_accesses`, graph source
event scheduling, or the production typed lifetime verifier. It validates
which logical input element each operator-output/reduction must read, not
the source-loop order, Phi reduction owner, hardware clocks or BRAM ports.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from math import prod

import numpy as np

SCHEMA="fpgai-research/cst-independent-numpy-graph-access-e0/v1"


def _ident(value):
    return sha256(json.dumps(value,sort_keys=True,
                             separators=(",",":")).encode()).hexdigest()


def independent_numpy_access_oracle(arch,events,*,max_events:int=100_000):
    """Check full graph READ owner/port/address multisets independently.

    Works on admitted static rank-two MatMul/Relu/Add/Concat graphs. A
    source-op register is required to distinguish one port from another.
    An external sink reads every output element once.
    """
    model=arch.functional
    model.validate()
    if (type(max_events) is not int or max_events<1 or
            len(events)>max_events):
        raise ValueError("Finite NumPy access-oracle budget required")
    produced={op.outputs[0] for op in model.operators}
    expected=Counter()
    for op in model.operators:
        outshape=model.tensor(op.outputs[0]).shape
        if len(outshape)!=2 or op.kind not in ("MatMul","Relu","Add","Concat"):
            raise ValueError("Graph access oracle has no supported semantics")
        if op.kind=="MatMul":
            if op.inputs[0] in produced:
                ins=model.tensor(op.inputs[0]).shape
                m,k=ins
                rows,width=outshape
                if m!=rows:
                    raise ValueError("MatMul source row mismatch")
                for row in range(rows):
                    for column in range(width):
                        output_index=row*width+column
                        for reduction in range(k):
                            expected[(op.inputs[0],row*k+reduction,
                                      op.name,output_index,0,reduction)]+=1
        elif op.kind=="Concat":
            arrays=[]
            for port,name in enumerate(op.inputs):
                shape=model.tensor(name).shape
                if len(shape)!=2:
                    raise ValueError("Only 2D Concat supported")
                ids=np.arange(prod(shape),dtype=np.int64).reshape(shape)
                ports=np.full(shape,port,dtype=np.int64)
                arrays.append((ids,ports))
            all_indices=np.concatenate([x[0] for x in arrays],axis=1)
            all_ports=np.concatenate([x[1] for x in arrays],axis=1)
            if all_indices.shape!=outshape:
                raise ValueError("Concat output shape mismatch")
            for i in range(prod(outshape)):
                port=int(all_ports.flat[i])
                name=op.inputs[port]
                if name in produced:
                    expected[(name,int(all_indices.flat[i]),op.name,i,port,-1)]+=1
        else:
            for port,name in enumerate(op.inputs):
                if name not in produced:
                    continue
                shape=model.tensor(name).shape
                indices=np.arange(prod(shape),dtype=np.int64).reshape(shape)
                if op.kind=="Relu":
                    if shape!=outshape:
                        raise ValueError("Relu expected exact tensor shape")
                    mapping=indices
                else:
                    mapping=np.broadcast_to(indices,outshape)
                for i in range(prod(outshape)):
                    expected[(name,int(mapping.flat[i]),op.name,i,port,-1)]+=1
    output=model.outputs[0]
    for i in range(prod(model.tensor(output).shape)):
        expected[(output,i,"EXTERNAL",i,0,-1)]+=1
    observed=Counter()
    for event in events:
        if event.kind!="READ":
            continue
        observed[(event.tensor,event.element,event.consumer,
                  event.output_element,event.input_port,event.reduction_k)]+=1
    if expected!=observed:
        missing=expected-observed
        unexpected=observed-expected
        def first(counter):
            if not counter:
                return None
            key,count=next(iter(sorted(counter.items())))
            return {"tensor":key[0],"element":key[1],
                    "consumer":key[2],"output_element":key[3],
                    "input_port":key[4],"reduction_k":key[5],
                    "multiplicity":count}
        return {
            "schema":SCHEMA,"status":"REJECTED_INDEPENDENT_ACCESS_MAP",
            "expected_reads":sum(expected.values()),
            "observed_reads":sum(observed.values()),
            "first_missing":first(missing),
            "first_unexpected":first(unexpected),
            "original_vitis_calls":0,
        }
    rows=sorted((list(key),count) for key,count in expected.items())
    return {
        "schema":SCHEMA,"status":"VERIFIED_INDEPENDENT_ACCESS_E0",
        "read_obligations":sum(expected.values()),
        "access_multiset_sha256":_ident(rows),
        "numpy_semantics":"independent-rank2-index-grid-broadcast-concatenate",
        "original_vitis_calls":0,
        "limitations":[
            "Independent NumPy graph access multiset, not proof of event order",
            "Phi owner, source lifetime, C++ execution and hardware ports require separate checks",
            "Only bounded 2D MatMul/Relu/Add/Concat subset, not arbitrary ONNX",
        ],
    }
