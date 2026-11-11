"""L6a generalized bounded NCHW graph CST tensor access/lifetime contract.

Composes validated 4D Conv/Relu/Add/Concat/MaxPool operators in arbitrary
topological order (one runtime input and final output), not just one fixed
diamond. All Conv input taps carry consumer-local resolved Phi. This E0
logical value schedule is NOT a cycle/RTL/physical implementation proof;
the stock HLS emitter still uses its own intermediate arrays.
"""
from __future__ import annotations

from collections import Counter,defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import prod
import re

from fpgai_research.architecture.resolver import coordinate
from fpgai_research.model_ir.contracts import ArchitecturalModel
from fpgai_research.model_ir.conv_boundary import (
    ConvValueEvent,_read_sources as _conv_read_sources,
    _flat,_unflat,
)
from fpgai_research.model_ir.lowering import lower
from fpgai_research.model_ir.semantics import (
    architecture_fingerprint,functional_fingerprint,verify_refinement,
)

SCHEMA="fpgai-research/cst-general-NCHW-graph-E0/v1"
KINDS=frozenset(("Conv","Relu","Add","Concat","MaxPool"))


def _hash(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(",",":"),
                             allow_nan=False).encode()).hexdigest()


def admit_nchw_graph(arch:ArchitecturalModel,*,max_operators:int=12):
    if not isinstance(arch,ArchitecturalModel):
        raise TypeError("Resolved ArchitecturalModel required")
    model=arch.functional
    model.validate()
    if (type(max_operators) is not int or not 2<=max_operators<=64 or
            not 2<=len(model.operators)<=max_operators or
            len(model.inputs)!=1 or len(model.outputs)!=1 or
            model.operators[-1].outputs!=model.outputs):
        raise ValueError("Only bounded, topological single-input/output NCHW DAGs")
    if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*",name)
           for name in (*[x.name for x in model.tensors],
                        *[x.name for x in model.operators])):
        raise ValueError("Safe graph tensor/operator labels required")
    verify_refinement(lower(arch))
    if len({(p.effective.precision_bits,p.effective.integer_bits)
            for p in arch.points})!=1:
        raise ValueError("Mixed numeric formats require explicit casts")
    produced={op.outputs[0] for op in model.operators}
    if (len(model.tensor(model.inputs[0]).shape)!=4 or
            any(len(model.tensor(name).shape)!=4 for name in produced)):
        raise ValueError("Runtime source and every intermediate must be NCHW rank four")
    if model.tensor(model.inputs[0]).data is not None:
        raise ValueError("Runtime input cannot be constant")
    for ordinal,(op,point) in enumerate(zip(model.operators,arch.points)):
        if point.operator!=op.name or point.kind!=op.kind:
            raise ValueError("Incorrect operator-local resolved Phi identity")
        if op.kind not in KINDS:
            raise ValueError("Unsupported NCHW graph operator: "+op.kind)
        if op.kind=="Conv":
            if (len(op.inputs)!=2 or op.inputs[0] not in produced|set(model.inputs)
                    or len(model.tensor(op.inputs[0]).shape)!=4 or
                    len(model.tensor(op.inputs[1]).shape)!=4 or
                    model.tensor(op.inputs[1]).data is None):
                raise ValueError("Only runtime LHS and unique constant Conv RHS supported")
            if (int(op.attr("group",1))!=1 or
                    tuple(op.attr("dilations",(1,1)))!=(1,1) or
                    op.attr("auto_pad","NOTSET")!="NOTSET"):
                raise ValueError("Grouped/dilated/automatic-padding Conv not implemented")
            if set(dict(op.attributes))-{"strides","pads","group","dilations","kernel_shape"}:
                raise ValueError("Unsupported Conv attribute")
        else:
            if any(model.tensor(name).data is not None or
                   len(model.tensor(name).shape)!=4
                   for name in op.inputs):
                raise ValueError("NCHW elementwise/concat/pool inputs must be runtime values")
            if op.kind=="Relu":
                if len(op.inputs)!=1 or op.attributes:
                    raise ValueError("Only ordinary scalar NCHW Relu")
            elif op.kind=="Add":
                if len(op.inputs)!=2 or op.attributes:
                    raise ValueError("Only ordinary ONNX rank-four broadcast Add")
            elif op.kind=="Concat":
                if (len(op.inputs)<2 or op.attr("axis",1)!=1 or
                        set(dict(op.attributes))-{"axis"}):
                    raise ValueError("Only NCHW channel-axis Concat")
            elif op.kind=="MaxPool":
                if (len(op.inputs)!=1 or
                        set(dict(op.attributes))-{"kernel_shape","strides","pads"} or
                        tuple(op.attr("kernel_shape",()))!=(2,2) or
                        tuple(op.attr("strides",(2,2)))!=(2,2) or
                        tuple(op.attr("pads",(0,0,0,0)))!=(0,0,0,0)):
                    raise ValueError("Only unpadded 2x2 stride2 MaxPool")
            a=point.effective
            if ((a.parallelism,a.pes,a.simd,a.serialization)!=(1,1,1,1)
                    or a.tile or a.pipeline_ii is not None):
                raise ValueError("Unlowered non-Conv hardware Phi not implemented")
    constants={x.name for x in model.tensors if x.data is not None}
    if constants!={op.inputs[1] for op in model.operators if op.kind=="Conv"}:
        raise ValueError("Only unique Conv kernel initializers may be constant")
    if any(not any(name in op.inputs for op in model.operators)
           for name in produced-set(model.outputs)):
        raise ValueError("Dead intermediate lacks last-reader obligation")
    return model,produced


def _source_reads(model,op,out_index):
    """Exact per-output element access map with optional Conv Phi coordinate."""
    if op.kind=="Conv":
        yield from _conv_read_sources(model,op,out_index)
        return
    outshape=model.tensor(op.outputs[0]).shape
    n,c,h,w=_unflat(outshape,out_index)
    if op.kind=="Relu":
        yield (0,op.inputs[0],out_index,-1,None)
    elif op.kind=="Add":
        for port,name in enumerate(op.inputs):
            shape=model.tensor(name).shape
            pos=(n,c,h,w)
            mapped=tuple(0 if d==1 else idx for d,idx in zip(shape,pos))
            yield (port,name,_flat(shape,mapped),-1,None)
    elif op.kind=="Concat":
        start=0
        for port,name in enumerate(op.inputs):
            shape=model.tensor(name).shape
            if start<=c<start+shape[1]:
                yield (port,name,_flat(shape,(n,c-start,h,w)),-1,None)
                return
            start+=shape[1]
        raise AssertionError("Concat output channel has no source port")
    elif op.kind=="MaxPool":
        shape=model.tensor(op.inputs[0]).shape
        for ky in range(2):
            for kx in range(2):
                yield (0,op.inputs[0],_flat(shape,(n,c,2*h+ky,2*w+kx)),
                       ky*2+kx,None)
    else:
        raise ValueError("No admitted NCHW graph source access map")


def _owner(point,coords):
    if coords is None:
        return None
    result=coordinate(point,coords)
    return (result["replica"],result["pe"],result["lane"],result["reuse_step"][4])


def nchw_graph_events(arch,*,max_events:int=100_000):
    model,produced=admit_nchw_graph(arch)
    if type(max_events) is not int or max_events<1:
        raise ValueError("Positive finite graph E0 budget required")
    events=[]
    for ordinal,op in enumerate(model.operators):
        for output_index in range(prod(model.tensor(op.outputs[0]).shape)):
            for port,name,element,red,coords in _source_reads(model,op,output_index):
                if name in produced:
                    events.append(ConvValueEvent("READ",name,element,op.name,
                                                 output_index,port,red,
                                                 _owner(arch.points[ordinal],coords)))
            events.append(ConvValueEvent("COMMIT",op.outputs[0],output_index))
            if len(events)>max_events:
                raise ValueError("Graph NCHW event budget exceeded")
    for idx in range(prod(model.tensor(model.outputs[0]).shape)):
        events.append(ConvValueEvent("READ",model.outputs[0],idx,"EXTERNAL",idx,0))
    if len(events)>max_events:
        raise ValueError("Graph NCHW sink event budget exceeded")
    return tuple(events)


def disjoint_nchw_offsets(arch):
    model,_=admit_nchw_graph(arch)
    offsets={}
    current=0
    for op in model.operators:
        offsets[op.outputs[0]]=current
        current+=prod(model.tensor(op.outputs[0]).shape)
    return offsets


def verify_nchw_graph(arch,offsets,events,*,max_events:int=100_000):
    model,produced=admit_nchw_graph(arch)
    sizes={name:prod(model.tensor(name).shape) for name in produced}
    if (type(offsets) is not dict or set(offsets)!=produced or
            any(type(x) is not int or x<0 for x in offsets.values())):
        raise ValueError("Exact nonnegative NCHW scratch bases required")
    if (type(events) is not tuple or type(max_events) is not int or max_events<1
            or len(events)>max_events or
            any(not isinstance(x,ConvValueEvent) for x in events)):
        raise ValueError("Bounded immutable NCHW source event sequence required")
    expected=Counter()
    required_by_value=Counter()
    required_by_output=Counter()
    output_by_consumer={op.name:op.outputs[0] for op in model.operators}
    for ordinal,op in enumerate(model.operators):
        for out_index in range(sizes[op.outputs[0]]):
            for port,name,source_idx,red,coords in _source_reads(model,op,out_index):
                if name in produced:
                    e=ConvValueEvent("READ",name,source_idx,op.name,out_index,
                                     port,red,_owner(arch.points[ordinal],coords))
                    expected[e]+=1
                    required_by_value[(name,source_idx)]+=1
                    # A downstream output cannot be committed before its
                    # declared produced-input reads have occurred. External
                    # runtime input taps are not observed in this E0 trace.
                    required_by_output[(op.outputs[0],out_index)]+=1
    for out_index in range(sizes[model.outputs[0]]):
        e=ConvValueEvent("READ",model.outputs[0],out_index,
                         "EXTERNAL",out_index,0)
        expected[e]+=1
        required_by_value[(e.tensor,e.element)]+=1
    ready=set()
    active={}
    observed=Counter()
    read_counts=Counter()
    observed_by_output=Counter()
    releases={}
    def reject(code,index,event,**more):
        return {"schema":SCHEMA,"status":"REJECTED","reason":code,
                "event_index":index,
                "counterexample":{"event":asdict(event),**more},
                "native_vitis_calls":0}
    for index,event in enumerate(events):
        if (event.tensor not in produced or type(event.element) is not int
                or not 0<=event.element<sizes[event.tensor]):
            return reject("UNKNOWN_NCHW_VALUE",index,event)
        key=(event.tensor,event.element)
        address=offsets[event.tensor]+event.element
        if event.kind=="COMMIT":
            if (event.consumer or event.output_element!=-1 or
                    event.input_port!=-1 or event.reduction_k!=-1 or
                    event.phi_owner is not None):
                return reject("INVALID_COMMIT",index,event)
            if key in ready:
                return reject("DUPLICATE_COMMIT",index,event)
            if observed_by_output[key]!=required_by_output[key]:
                return reject("COMMIT_BEFORE_MODELED_INPUT_READS",index,event,
                              required_reads=required_by_output[key],
                              observed_reads=observed_by_output[key])
            if address in active:
                return reject("OVERWRITE_BEFORE_GLOBAL_LAST_READ",index,event,
                              address=address,live_value=list(active[address]))
            active[address]=key
            ready.add(key)
        elif event.kind=="READ":
            if event not in expected:
                return reject("INCORRECT_NCHW_SOURCE_OR_PHI",index,event)
            if key not in ready or active.get(address)!=key:
                return reject("READ_BEFORE_COMMIT_OR_STALE",index,event,address=address)
            observed[event]+=1
            if observed[event]>expected[event]:
                return reject("DUPLICATE_READ_TOKEN",index,event)
            if event.consumer!="EXTERNAL" and event.consumer in output_by_consumer:
                target=(output_by_consumer[event.consumer],event.output_element)
                if target in ready:
                    return reject("READ_AFTER_CONSUMER_OUTPUT_COMMIT",
                                  index,event,target=list(target))
                observed_by_output[target]+=1
            read_counts[key]+=1
            if read_counts[key]==required_by_value[key]:
                releases[key]=index
                del active[address]
        else:
            return reject("INVALID_EVENT_KIND",index,event)
    if (observed!=expected or active or
            len(ready)!=sum(sizes.values()) or
            len(releases)!=len(ready)):
        return {"schema":SCHEMA,"status":"REJECTED",
                "reason":"INCOMPLETE_GRAPH_CONSUMPTION",
                "counterexample":{"observed_reads":sum(observed.values()),
                                  "required_reads":sum(expected.values()),
                                  "unreleased_values":len(active)},
                "native_vitis_calls":0}
    return {"schema":SCHEMA,"status":"VERIFIED_E0",
            "functional_sha256":functional_fingerprint(model),
            "architecture_sha256":architecture_fingerprint(arch),
            "event_sha256":_hash([asdict(x) for x in events]),
            "last_use_sha256":_hash(sorted((t,i,last) for (t,i),last in releases.items())),
            "operators":len(model.operators),
            "operator_kinds":sorted({x.kind for x in model.operators}),
            "edges":sum(name in produced for op in model.operators for name in op.inputs),
            "commits":len(ready),"reads":sum(expected.values()),
            "modeled_produced_input_read_obligations":sum(required_by_output.values()),
            "source_scratch_elements":max(offsets[t]+sizes[t] for t in sizes),
            "native_vitis_calls":0,
            "evidence":"E0-general-topological-NCHW-source-read-Phi-last-use",
            "limitations":[
                "Mathematical producer-major source order, NOT the native HLS Conv loop order",
                "Stock HLS emitter uses independent intermediate arrays, NOT planned scratch bases",
                "No per-read source execution proof for all scalar/Conv graph operators",
                "COMMIT-before-input obligation covers only modeled produced-input reads, not external runtime taps",
                "No actual Vitis CSim/CSynth/RTL or bank/clock qualification",
            ]}


def nchw_graph_contract(arch,offsets,*,max_events:int=100_000):
    events=nchw_graph_events(arch,max_events=max_events)
    result=verify_nchw_graph(arch,offsets,events,max_events=max_events)
    if result["status"]!="VERIFIED_E0":
        raise ValueError("NCHW_GRAPH_INVALID "+result["reason"])
    # This reference calculates Conv owners independently from the compiler's
    # resolver.coordinate and E0 source-event generator. It is an additional
    # finite source obligation, not an independent full graph semantics proof.
    from fpgai_research.validation.nchw_access_oracle import (
        independent_nchw_access,
    )
    independent=independent_nchw_access(arch,events,max_events=max_events)
    if independent["status"]!="VERIFIED_INDEPENDENT_NCHW_ACCESS_E0":
        raise ValueError("NCHW_INDEPENDENT_ACCESS_OR_PHI_INVALID "+
                         independent.get("reason","UNKNOWN"))
    result["independent_access_multiset_sha256"]=independent[
        "access_multiset_sha256"]
    result["independent_conv_phi_owner_multiset_sha256"]=independent[
        "phi_owner_multiset_sha256"]
    result["independent_conv_phi_owner_checked_reads"]=independent[
        "phi_owner_checked_conv_reads"]
    result["independent_conv_phi_owner_status"]=independent[
        "phi_owner_validation"]
    return result


@dataclass(frozen=True)
class NCHWValueWitness:
    element: int
    source_address: int
    commit_event: int
    global_last_read_event: int
    consumer_read_events: tuple[int,...]
    mapped_read_owners: tuple[tuple[int,int,int,int]|None,...]


@dataclass(frozen=True)
class NCHWEdgeWitness:
    source_tensor: str
    producer: str
    consumer: str
    input_port: int
    values: tuple[NCHWValueWitness,...]


def nchw_edge_witnesses(arch,offsets,*,events=None,max_events:int=100_000):
    model,produced=admit_nchw_graph(arch)
    if events is None:
        events=nchw_graph_events(arch,max_events=max_events)
    verdict=verify_nchw_graph(arch,offsets,events,max_events=max_events)
    if verdict["status"]!="VERIFIED_E0":
        raise ValueError("Cannot certify invalid NCHW graph: "+verdict["reason"])
    all_reads=defaultdict(list)
    by_consumer=defaultdict(list)
    commits={}
    for i,e in enumerate(events):
        if e.kind=="COMMIT":
            commits[(e.tensor,e.element)]=i
        else:
            all_reads[(e.tensor,e.element)].append(i)
            by_consumer[(e.tensor,e.consumer,e.input_port,e.element)].append(i)
    result=[]
    for op in model.operators:
        name=op.outputs[0]
        readers=[(c.name,port) for c in model.operators
                 for port,input_name in enumerate(c.inputs) if input_name==name]
        if name==model.outputs[0]:
            readers.append(("EXTERNAL",0))
        for consumer,port in readers:
            values=[]
            for element in range(prod(model.tensor(name).shape)):
                actual=by_consumer.get((name,consumer,port,element),[])
                if not actual:
                    continue  # E.g., untouched padded positions or channel slices.
                key=(name,element)
                values.append(NCHWValueWitness(
                    element,offsets[name]+element,commits[key],
                    max(all_reads[key]),tuple(actual),
                    tuple(events[j].phi_owner for j in actual)))
            result.append(NCHWEdgeWitness(name,op.name,consumer,port,tuple(values)))
    return tuple(result)


def nchw_witness_receipt(arch,offsets,*,max_events:int=100_000):
    witnesses=nchw_edge_witnesses(arch,offsets,max_events=max_events)
    return {"schema":"fpgai-research/cst-NCHW-graph-value-edge-witness-E0/v1",
            "status":"VERIFIED_E0",
            "edge_count":len(witnesses),
            "value_edges":sum(len(x.values) for x in witnesses),
            "witness_sha256":_hash([asdict(x) for x in witnesses]),
            "native_vitis_calls":0,
            "evidence":"E0-global-last-reader-across-arbitrary-NCHW-topology",
            "limitations":["Source-index contract, not an emitted storage or HLS schedule proof"]}
