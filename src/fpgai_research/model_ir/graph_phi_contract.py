"""L6a bounded graph-wide source-order tensor edge certificate.

Unlike the earlier five-operator diamond, this is a *topologically typed*
static rank-two DAG over MatMul(constant RHS), Relu and same-shape Add.
All MatMul operators keep their independently chosen Phi and all producer
values retain per-element, multi-consumer, per-reduction READ obligations.
The last graph output has an explicit external sink read. Every tensor
address is the actual element's contiguous scratch base plus flat index.

Source order and concrete memory aliasing are E0 logical/source obligations,
not Vitis schedule, hardware ports, arbitrary ONNX or formal RTL semantics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import prod
import re

from fpgai_research.architecture.resolver import coordinate
from fpgai_research.model_ir.contracts import ArchitecturalModel
from fpgai_research.model_ir.lowering import lower
from fpgai_research.model_ir.matmul_phi_order import mapped_matmul_order
from fpgai_research.model_ir.graph_access_maps import (
    scalar_output_accesses, validate_scalar_access_shape,
)
from fpgai_research.model_ir.semantics import (
    architecture_fingerprint,functional_fingerprint,verify_refinement,
)

SCHEMA="fpgai-research/cst-graph-phi-boundaries-e0/v1"
OUT_SINK="EXTERNAL"


@dataclass(frozen=True)
class GraphEvent:
    kind: str                 # COMMIT or READ
    tensor: str
    element: int
    consumer: str = ""
    output_element: int = -1
    input_port: int = -1
    reduction_k: int = -1
    phi_owner: tuple[int,int,int,int] | None = None


def _hash(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(",",":"),
                             allow_nan=False).encode()).hexdigest()


def admit_graph(arch:ArchitecturalModel,*,max_operators:int=12):
    if not isinstance(arch,ArchitecturalModel):
        raise TypeError("Resolved ArchitecturalModel required")
    model=arch.functional
    model.validate()
    if (type(max_operators) is not int or max_operators<1 or
            not 2<=len(model.operators)<=max_operators or
            len(model.inputs)!=1 or len(model.outputs)!=1 or
            model.operators[-1].outputs!=model.outputs):
        raise ValueError("Bounded topological DAG requires one input and final output")
    if any(not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*",name)
           for name in (
               *(t.name for t in model.tensors),
               *(op.name for op in model.operators),
           )):
        raise ValueError("Safe identifiers required for source trace")
    verify_refinement(lower(arch))
    formats={(p.effective.precision_bits,p.effective.integer_bits) for p in arch.points}
    if len(formats)!=1:
        raise ValueError("Mixed numerical formats require explicit casts")
    produced={op.outputs[0] for op in model.operators}
    if any(len(t.shape)!=2 for t in model.tensors):
        raise ValueError("Only static rank-two tensors supported")
    if any(t.data is not None for t in model.tensors
           if t.name not in {op.inputs[1] for op in model.operators if op.kind=="MatMul"}):
        raise ValueError("Only MatMul RHS constant tensors supported")
    if len({op.inputs[1] for op in model.operators if op.kind=="MatMul"}) != sum(
            op.kind=="MatMul" for op in model.operators):
        # Distinct constant names simplify one-to-one source provenance.
        raise ValueError("Each MatMul must use an independent constant tensor")
    for index,op in enumerate(model.operators):
        point=arch.points[index]
        if point.operator!=op.name or point.kind!=op.kind:
            raise ValueError("Operator resolution/order identity mismatch")
        if op.attributes and not (
                op.kind=="Concat" and set(dict(op.attributes))=={"axis"}
                and op.attr("axis")==1):
            raise ValueError("Attributes not implemented in bounded graph source")
        a=point.effective
        if op.kind=="MatMul":
            if (len(op.inputs)!=2 or model.tensor(op.inputs[1]).data is None
                    or op.inputs[0] not in produced|set(model.inputs)):
                raise ValueError("MatMul requires runtime/produced lhs and constant RHS")
        elif op.kind in ("Relu","Add","Concat"):
            if (len(set(op.inputs))!=len(op.inputs) or
                    any(n not in produced|set(model.inputs) for n in op.inputs)):
                raise ValueError("Scalar operator inputs must be distinct runtime/produced tensors")
            validate_scalar_access_shape(model,op)
            if ((a.parallelism,a.pes,a.simd,a.serialization)!=(1,1,1,1)
                    or a.tile or a.pipeline_ii is not None):
                raise ValueError("Unlowered scalar Phi/tile/pipeline mapping rejected")
        else:
            raise ValueError("Unsupported graph operator (no fallback scalar lowering)")
    # Every intermediate has at least one declared reader or is the external sink.
    for name in produced:
        if name not in model.outputs and not any(name in op.inputs for op in model.operators):
            raise ValueError("Dead intermediate has no release contract")
    return model,produced


def graph_sizes(arch):
    model,produced=admit_graph(arch)
    return {name:prod(model.tensor(name).shape) for name in sorted(produced)}


def graph_disjoint_offsets(arch):
    sizes=graph_sizes(arch)
    offsets={}
    cursor=0
    for op in arch.functional.operators:
        name=op.outputs[0]
        offsets[name]=cursor
        cursor+=sizes[name]
    return offsets


def _phi_owner(point,mi,ni,kk):
    owner=coordinate(point,(mi,ni,kk))
    return (owner["replica"],owner["pe"],owner["lane"],owner["reuse_step"][2])


def graph_source_events(arch:ArchitecturalModel,*,max_events:int=100_000):
    model,produced=admit_graph(arch)
    if type(max_events) is not int or max_events<1:
        raise ValueError("Positive source-event count budget required")
    events=[]
    iterations={}
    for ordinal,(op,point) in enumerate(zip(model.operators,arch.points)):
        out=op.outputs[0]
        rows,width=model.tensor(out).shape
        if op.kind=="MatMul":
            k=model.tensor(op.inputs[0]).shape[1]
            seq=mapped_matmul_order(point,max_iterations=max_events)
            iterations[op.name]=seq
            partial={}
            committed=set()
            for mi,ni,kk,*owner in seq:
                key=(mi,ni)
                if key in committed:
                    raise AssertionError("MatMul contribution after COMMIT")
                if op.inputs[0] in produced:
                    events.append(GraphEvent("READ",op.inputs[0],mi*k+kk,
                                             op.name,mi*width+ni,0,kk,tuple(owner)))
                seen=partial.setdefault(key,set())
                if kk in seen:
                    raise AssertionError("Duplicate MatMul reduction")
                seen.add(kk)
                if len(seen)==k:
                    committed.add(key)
                    events.append(GraphEvent("COMMIT",out,mi*width+ni))
            if len(committed)!=rows*width:
                raise AssertionError("Unfinished MatMul output reductions")
        else:
            for idx in range(rows*width):
                for port,name,source_idx in scalar_output_accesses(model,op,idx):
                    if name in produced:
                        events.append(GraphEvent("READ",name,source_idx,op.name,idx,port))
                events.append(GraphEvent("COMMIT",out,idx))
        if len(events)>max_events:
            raise ValueError("Graph source event count exceeds finite bound")
    sink=model.outputs[0]
    for idx in range(prod(model.tensor(sink).shape)):
        events.append(GraphEvent("READ",sink,idx,OUT_SINK,idx,0))
    if len(events)>max_events:
        raise ValueError("Graph source event count exceeds finite bound")
    return tuple(events),iterations


def _required_reads(arch):
    model,produced=admit_graph(arch)
    required={(name,i):set()
              for name in produced
              for i in range(prod(model.tensor(name).shape))}
    for op in model.operators:
        rows,out_width=model.tensor(op.outputs[0]).shape
        if op.kind=="MatMul":
            k=model.tensor(op.inputs[0]).shape[1]
            name=op.inputs[0]
            if name in produced:
                for mi in range(rows):
                    for ni in range(out_width):
                        for kk in range(k):
                            required[(name,mi*k+kk)].add((op.name,mi*out_width+ni,0,kk))
        else:
            for idx in range(rows*out_width):
                for port,name,source_idx in scalar_output_accesses(model,op,idx):
                    if name in produced:
                        required[(name,source_idx)].add((op.name,idx,port,-1))
    for idx in range(prod(model.tensor(model.outputs[0]).shape)):
        required[(model.outputs[0],idx)].add((OUT_SINK,idx,0,-1))
    return required


def verify_graph_events(arch:ArchitecturalModel, offsets:dict[str,int],
                        events:tuple[GraphEvent,...],*,max_events:int=100_000):
    model,produced=admit_graph(arch)
    sizes={name:prod(model.tensor(name).shape) for name in produced}
    if (type(offsets) is not dict or set(offsets)!=produced or
            any(type(x) is not int or x<0 for x in offsets.values())):
        raise ValueError("Exact nonnegative intermediate scratch base map required")
    if (type(events) is not tuple or
            any(not isinstance(e,GraphEvent) for e in events) or
            type(max_events) is not int or max_events<1 or
            len(events)>max_events):
        raise ValueError("Bounded immutable typed events required")
    required=_required_reads(arch)
    op_by_name={op.name:(idx,op) for idx,op in enumerate(model.operators)}
    ready=set()
    read={}
    live={}
    releases={}
    def reject(reason,i,e,**context):
        return {"schema":SCHEMA,"status":"REJECTED",
                "reason":reason,"event_index":i,
                "counterexample":{"event":asdict(e),**context},
                "native_vitis_calls":0}
    for i,event in enumerate(events):
        if (event.tensor not in produced or type(event.element) is not int
                or not 0<=event.element<sizes[event.tensor]):
            return reject("UNKNOWN_TENSOR_OR_ELEMENT",i,event)
        key=(event.tensor,event.element)
        addr=offsets[event.tensor]+event.element
        if event.kind=="COMMIT":
            if (event.consumer or event.output_element!=-1 or
                    event.input_port!=-1 or event.reduction_k!=-1 or
                    event.phi_owner is not None):
                return reject("MALFORMED_COMMIT",i,event)
            if key in ready:
                return reject("DUPLICATE_PRODUCER",i,event)
            if addr in live:
                return reject("OVERWRITE_BEFORE_LAST_USE",i,event,
                              address=addr,live_value=list(live[addr]))
            ready.add(key)
            live[addr]=key
        elif event.kind=="READ":
            token=(event.consumer,event.output_element,
                   event.input_port,event.reduction_k)
            if token not in required[key]:
                return reject("UNKNOWN_READ_MAPPING",i,event)
            if key not in ready or live.get(addr)!=key:
                return reject("READ_BEFORE_COMMIT_OR_STALE",i,event,address=addr)
            if event.consumer==OUT_SINK:
                if event.phi_owner is not None:
                    return reject("SPURIOUS_SINK_PHI",i,event)
            else:
                ordinal,op=op_by_name[event.consumer]
                if op.kind=="MatMul":
                    rows,out_width=model.tensor(op.outputs[0]).shape
                    row,ni=divmod(event.output_element,out_width)
                    k=model.tensor(op.inputs[0]).shape[1]
                    mi,kk=divmod(event.element,k)
                    if row!=mi or kk!=event.reduction_k:
                        return reject("WRONG_MATMUL_READ_COORDINATE",i,event)
                    expected=_phi_owner(arch.points[ordinal],mi,ni,kk)
                    if event.phi_owner!=expected:
                        return reject("WRONG_PHI_CONSUMER_OWNER",i,event,
                                      required_owner=list(expected))
                elif event.phi_owner is not None:
                    return reject("UNSUPPORTED_SCALAR_PHI_OWNER",i,event)
            completed=read.setdefault(key,set())
            if token in completed:
                return reject("DUPLICATE_CONTRIBUTION_READ",i,event)
            completed.add(token)
            if len(completed)==len(required[key]):
                releases[key]=i
                del live[addr]
        else:
            return reject("UNKNOWN_EVENT_KIND",i,event)
    if (ready!=set(required) or live or
            any(read.get(key,set())!=need for key,need in required.items())):
        return {
            "schema":SCHEMA,"status":"REJECTED","reason":"INCOMPLETE_GRAPH_RELEASE",
            "counterexample":{"ready_values":len(ready),
                              "required_values":len(required),
                              "released_values":len(releases),
                              "live_addresses":len(live)},
            "native_vitis_calls":0}
    return {
        "schema":SCHEMA,"status":"VERIFIED_E0",
        "functional_sha256":functional_fingerprint(model),
        "architecture_sha256":architecture_fingerprint(arch),
        "source_event_sha256":_hash([asdict(e) for e in events]),
        "last_use_sha256":_hash(sorted((name,i,t)
                                       for (name,i),t in releases.items())),
        "commits":len(required),
        "reads":sum(len(v) for v in required.values()),
        "multi_reader_values":sum(any(len(v)>1 for v in (required[key],))
                                   for key in required),
        "operator_count":len(model.operators),
        "edge_count":sum(n in produced for op in model.operators
                         for n in op.inputs),
        "source_array_elements":max(offsets[name]+sizes[name] for name in sizes),
        "native_vitis_calls":0,
        "evidence":"E0-static-topological-Phi-per-contribution-source-order-graph",
        "limitations":[
            "Only static rank-two MatMul(constant RHS), scalar Relu and equal-shape Add",
            "Sequential original source-loop order, NOT achieved HLS hardware cycles",
            "Bounded source-address alias relation, no bank/port/RTL qualification",
        ],
    }


def graph_contract(arch,offsets,*,max_events:int=100_000):
    events,iterations=graph_source_events(arch,max_events=max_events)
    result=verify_graph_events(arch,offsets,events,max_events=max_events)
    if result["status"]!="VERIFIED_E0":
        raise ValueError("GRAPH_BOUNDARY_UNSAFE "+result["reason"]+
                         " "+str(result.get("counterexample")))
    # An independently calculated source-loop ranking detects a common-mode
    # issue-order error that the value/lifetime replay alone cannot detect.
    # It covers only observable MatMul reads from produced intermediates.
    from fpgai_research.model_ir.independent_phi_issue_reference import (
        validate_graph_phi_issue_conformance,
    )
    issue = validate_graph_phi_issue_conformance(arch, events, max_reads=max_events)
    if issue["status"] == "REJECTED":
        raise ValueError("GRAPH_PHI_SOURCE_ISSUE_MISMATCH " +
                         str(issue["operator"]) + " " + str(issue["first_read_index"]))
    result["independent_phi_issue_status"] = issue["status"]
    result["independent_phi_issue_checked_reads"] = issue["checked_reads"]
    result["independent_phi_issue_checked_operators"] = issue["checked_operators"]
    result["independent_phi_issue_signature_sha256"] = issue["observed_issue_signature_sha256"]
    result["operator_phi_iterations"]={name:len(seq)
                                      for name,seq in iterations.items()}
    result["operator_phi_sha256"]={name:_hash(seq)
                                  for name,seq in iterations.items()}
    return result
