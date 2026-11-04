"""First-class bounded graph tensor edge/refinement witness objects.

Every edge records per-value COMMIT, every consumer-specific mapped READ,
global release after *all* readers, and concrete source-scratch address.
The witness is checkable against the full E0 event verifier and original
GCC source audit. This is a source-level certificate, not a hardware proof.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass
from hashlib import sha256
import json

from fpgai_research.model_ir.graph_phi_contract import (
    OUT_SINK,graph_source_events,verify_graph_events,admit_graph,
)
from fpgai_research.model_ir.matmul_phi_order import mapped_matmul_order


@dataclass(frozen=True)
class GraphValueWitness:
    element: int
    source_address: int
    ready_at_event: int
    release_after_event: int
    consumer_reads: tuple[int,...]
    phi_read_owners: tuple[tuple[int,int,int,int] | None,...]


@dataclass(frozen=True)
class GraphEdgeWitness:
    tensor: str
    producer: str
    consumer: str
    input_port: int
    producer_phi_projection_sha256: str
    consumer_phi_projection_sha256: str
    source_events_sha256: str
    values: tuple[GraphValueWitness,...]


def _sha(item):
    return sha256(json.dumps(item,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def edge_witnesses(arch,offsets,*,events=None,max_events:int=100_000):
    """Construct only after the full same-event E0 relation is VERIFIED."""
    model,produced=admit_graph(arch)
    if events is None:
        events,_=graph_source_events(arch,max_events=max_events)
    check=verify_graph_events(arch,offsets,events,max_events=max_events)
    if check["status"]!="VERIFIED_E0":
        raise ValueError("NO_GRAPH_EDGE_CERTIFICATE_FOR_INVALID_EVENTS "+
                         check["reason"]+" "+str(check.get("counterexample")))
    producer={op.outputs[0]:(i,op) for i,op in enumerate(model.operators)}
    operators={op.name:(i,op) for i,op in enumerate(model.operators)}
    ready={}
    all_reads={}
    consumer_reads={}
    for i,event in enumerate(events):
        key=(event.tensor,event.element)
        if event.kind=="COMMIT":
            ready[key]=i
        else:
            all_reads.setdefault(key,[]).append(i)
            consumer_reads.setdefault((event.tensor,event.consumer,
                                       event.input_port,event.element),[]).append(i)
    projections={}
    for i,op in enumerate(model.operators):
        point=arch.points[i]
        projection=(mapped_matmul_order(point,max_iterations=max_events)
                    if op.kind=="MatMul" else (
                        "scalar",op.kind,op.name,point.iteration_shape,
                        asdict(point.effective)))
        projections[op.name]=_sha(projection)
    projections[OUT_SINK]=_sha(("external_output_sink",model.outputs[0]))
    result=[]
    for op in model.operators:
        name=op.outputs[0]
        consumers=[(reader.name,port) for reader in model.operators
                   for port,input_name in enumerate(reader.inputs)
                   if input_name==name]
        if name in model.outputs:
            consumers.append((OUT_SINK,0))
        for consumer,port in consumers:
            count=model.tensor(name).shape[0]*model.tensor(name).shape[1]
            values=[]
            for element in range(count):
                r=consumer_reads.get((name,consumer,port,element),[])
                if not r:
                    raise AssertionError("Complete graph verifier missed required consumer read")
                values.append(GraphValueWitness(
                    element=element,source_address=offsets[name]+element,
                    ready_at_event=ready[(name,element)],
                    release_after_event=max(all_reads[(name,element)]),
                    consumer_reads=tuple(r),
                    phi_read_owners=tuple(events[j].phi_owner for j in r),
                ))
            result.append(GraphEdgeWitness(
                tensor=name,producer=op.name,consumer=consumer,input_port=port,
                producer_phi_projection_sha256=projections[op.name],
                consumer_phi_projection_sha256=projections[consumer],
                source_events_sha256=check["source_event_sha256"],
                values=tuple(values),
            ))
    return tuple(result)


def edge_witness_receipt(arch,offsets,*,events=None,max_events:int=100_000):
    witnesses=edge_witnesses(arch,offsets,events=events,max_events=max_events)
    serial=[asdict(x) for x in witnesses]
    return {
        "schema":"fpgai-research/cst-graph-first-class-edge-witness-e0/v1",
        "status":"VERIFIED_E0",
        "edges":len(witnesses),
        "value_edges":sum(len(x.values) for x in witnesses),
        "source_witness_sha256":_sha(serial),
        "source_event_sha256":witnesses[0].source_events_sha256,
        "witnesses":serial,
        "native_vitis_calls":0,
        "evidence":"E0-checker-accepted-composable-tensor-edge-value-witnesses",
        "limitations":[
            "Finite fixed sequential source-event schedule, not a hardware clock",
            "Source scratch address is not qualified bank/BRAM port placement",
            "No proof of arbitrary C++ or original Vitis/RTL refinement",
        ],
    }
