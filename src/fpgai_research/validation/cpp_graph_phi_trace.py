"""Independent E1 executed GCC trace for bounded graph-wide Phi lifetime IR.

All MatMul contributions must match their local Phi and precede an actual
COMMIT; each intermediate scratch load and sink output consumption must
equal the declared graph read/access map and source-level last use.
No claim about Vitis cycle scheduling, physical memory banks or RTL.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

from fpgai_research.architecture.resolver import coordinate
from fpgai_research.codegen.hls.graph_phi import SCHEMA as EMIT_SCHEMA
from fpgai_research.model_ir.graph_phi_contract import (
    GraphEvent,OUT_SINK,admit_graph,graph_source_events,
    verify_graph_events,_hash,
)
from fpgai_research.model_ir.semantics import (
    architecture_fingerprint,functional_fingerprint,
)

SCHEMA="fpgai-research/cst-graph-Phi-actual-GCC-boundary-e1/v1"


def trace_graph_cpp(arch,emission,*,max_events:int=100_000,
                    timeout_seconds:int=90,cxx:str="g++"):
    model,produced=admit_graph(arch)
    manifest=emission.manifest
    if (manifest.get("schema")!=EMIT_SCHEMA or
            manifest.get("functional_sha256")!=functional_fingerprint(model) or
            manifest.get("architecture_sha256")!=architecture_fingerprint(arch) or
            manifest.get("cpp_sha256")!=sha256(emission.cpp.encode()).hexdigest()):
        raise ValueError("Source graph/architecture hash mismatch")
    events,iteration_map=graph_source_events(arch,max_events=max_events)
    if len(events)+sum(len(x) for x in iteration_map.values())>max_events:
        raise ValueError("Source trace exceeds bound")
    if (type(timeout_seconds) is not int or not 1<=timeout_seconds<=180):
        raise ValueError("Positive bounded source execution timeout required")
    compiler=shutil.which(cxx)
    if compiler is None:
        raise FileNotFoundError("Actual host C++ compiler required")
    x=model.tensor(model.inputs[0])
    y=model.tensor(model.outputs[0])
    count_in=x.shape[0]*x.shape[1]
    count_out=y.shape[0]*y.shape[1]
    runner=(f"\nint main() {{ float x[{count_in}]={{}}; "
            f"float y[{count_out}]={{}}; fpgai_model(x,y); return 0; }}\n")
    with TemporaryDirectory(prefix="fpgai_graph_phi_trace_") as folder:
        src=Path(folder)/"graph.cpp"
        binary=Path(folder)/"graph"
        src.write_text(emission.cpp+runner)
        compiled=subprocess.run([
            compiler,"-std=c++17","-O0","-Wno-unknown-pragmas",
            "-DFPGAI_HOST_FLOAT","-DFPGAI_TRACE_ITERATIONS",
            "-DFPGAI_TRACE_FANOUT","-DFPGAI_TRACE_BRANCH",
            str(src),"-o",str(binary)],
            capture_output=True,text=True,timeout=timeout_seconds)
        if compiled.returncode:
            raise RuntimeError("Graph source does not compile: "+compiled.stderr[-5000:])
        run=subprocess.run([str(binary)],capture_output=True,text=True,
                           timeout=timeout_seconds)
        if run.returncode:
            raise RuntimeError("Graph GCC execution failed: "+run.stderr[-5000:])
    actual=[]
    reductions={name:[] for name in iteration_map}
    used={}
    committed=set()
    awaiting_read=None
    ops={op.name:(i,op) for i,op in enumerate(model.operators)}
    for line_index,line in enumerate(run.stdout.splitlines()):
        parts=line.split()
        if parts[:1]==["FPGAI_ITER"]:
            if awaiting_read is not None or len(parts)!=9:
                raise AssertionError("Unmatched MatMul reduction read")
            try:
                ordinal,mi,ni,kk,replica,pe,lane,fold=map(int,parts[1:])
            except ValueError as exc:
                raise AssertionError("Malformed source MatMul Phi reduction") from exc
            if not 0<=ordinal<len(model.operators):
                raise AssertionError("Unknown source operator ordinal")
            op=model.operators[ordinal]
            if op.kind!="MatMul":
                raise AssertionError("Non-MatMul emitted reduction")
            rows,width=model.tensor(op.outputs[0]).shape
            k=model.tensor(op.inputs[0]).shape[1]
            if not(0<=mi<rows and 0<=ni<width and 0<=kk<k):
                raise AssertionError("Out of range C++ MatMul reduction")
            phi=coordinate(arch.points[ordinal],(mi,ni,kk))
            owner=(phi["replica"],phi["pe"],phi["lane"],phi["reuse_step"][2])
            if (replica,pe,lane,fold)!=owner:
                raise AssertionError("SOURCE_GRAPH_PHI_OWNER_MISMATCH")
            key=(op.name,mi,ni)
            known=used.setdefault(key,set())
            if kk in known or key in committed:
                raise AssertionError("Duplicate or post-commit MatMul reduction")
            known.add(kk)
            reductions[op.name].append((mi,ni,kk,replica,pe,lane,fold))
            if op.inputs[0] in produced:
                awaiting_read=(op.inputs[0],mi*k+kk,op.name,
                               mi*width+ni,0,kk,owner)
            continue
        if parts[:2]==["FPGAI_BRANCH","READ"]:
            if len(parts) not in (6,10):
                raise AssertionError("Malformed C++ graph scratch READ")
            try:
                element,out_index=int(parts[3]),int(parts[5])
                owner=tuple(map(int,parts[6:])) if len(parts)==10 else None
            except ValueError as exc:
                raise AssertionError("Invalid graph READ address or owner") from exc
            tensor,consumer=parts[2],parts[4]
            if consumer==OUT_SINK:
                port,reduction=0,-1
            else:
                if consumer not in ops or tensor not in ops[consumer][1].inputs:
                    raise AssertionError("Invalid graph source read input binding")
                ordinal,op=ops[consumer]
                port=op.inputs.index(tensor)
                reduction=(element%model.tensor(tensor).shape[1]
                           if op.kind=="MatMul" else -1)
            entry=GraphEvent("READ",tensor,element,consumer,out_index,
                             port,reduction,owner)
            if awaiting_read is not None:
                if (entry.tensor,entry.element,entry.consumer,
                        entry.output_element,entry.input_port,
                        entry.reduction_k,entry.phi_owner)!=awaiting_read:
                    raise AssertionError("SOURCE_GRAPH_REDUCTION_READ_MISMATCH")
                awaiting_read=None
            elif owner is not None or (
                    consumer in ops and ops[consumer][1].kind=="MatMul"):
                raise AssertionError("Spurious or unpaired source MatMul scratch read")
            actual.append(entry)
            continue
        if parts[:2]==["FPGAI_FANOUT","COMMIT"]:
            if len(parts)!=4 or awaiting_read is not None:
                raise AssertionError("Malformed or premature graph COMMIT")
            try:
                element=int(parts[3])
            except ValueError as exc:
                raise AssertionError("Invalid emitted graph commit element") from exc
            tensor=parts[2]
            event=GraphEvent("COMMIT",tensor,element)
            op=next((o for o in model.operators if o.outputs[0]==tensor),None)
            if op is not None and op.kind=="MatMul":
                rows,width=model.tensor(tensor).shape
                row,ni=divmod(element,width)
                k=model.tensor(op.inputs[0]).shape[1]
                key=(op.name,row,ni)
                if key in committed or used.get(key,set())!=set(range(k)):
                    raise AssertionError("SOURCE_GRAPH_REDUCTION_INCOMPLETE_AT_COMMIT")
                committed.add(key)
            actual.append(event)
            continue
        raise AssertionError(f"Unknown source trace at line {line_index}: {line[:120]}")
    if awaiting_read is not None:
        raise AssertionError("Unpaired last original C++ contribution")
    for name,seq in iteration_map.items():
        if tuple(reductions[name])!=seq:
            raise AssertionError("Source MatMul iteration order changed: "+name)
        if (_hash(reductions[name])!=manifest["operator_phi_sha256"][name] or
                len(seq)!=manifest["operator_phi_iterations"][name]):
            raise AssertionError("Source Phi signature does not match manifest")
    if tuple(actual)!=events:
        i=next((i for i,(x,y) in enumerate(zip(events,actual)) if x!=y),
               min(len(events),len(actual)))
        raise AssertionError("SOURCE_GRAPH_EVENT_MISMATCH "+str(i)+
                             f": expected={events[i:i+1]} observed={tuple(actual)[i:i+1]}")
    # Validate independently ranked source READs against actual GCC events,
    # then bind their digest and coverage to the generated source manifest.
    from fpgai_research.model_ir.independent_phi_issue_reference import (
        validate_graph_phi_issue_conformance,
    )
    issue = validate_graph_phi_issue_conformance(
        arch, tuple(actual), max_reads=max_events)
    if issue["status"] == "REJECTED":
        raise AssertionError(
            "SOURCE_GRAPH_INDEPENDENT_PHI_ISSUE_REJECTED " +
            str(issue["operator"]) + " " + str(issue["first_read_index"]))
    if (issue["status"] != manifest.get("independent_phi_issue_status")
            or issue["checked_reads"] != manifest.get("independent_phi_issue_checked_reads")
            or issue["checked_operators"] != manifest.get("independent_phi_issue_checked_operators")
            or issue["observed_issue_signature_sha256"] != manifest.get("independent_phi_issue_signature_sha256")):
        raise AssertionError("SOURCE_GRAPH_INDEPENDENT_PHI_ISSUE_RECEIPT_MISMATCH")
    replay=verify_graph_events(
        arch,manifest["offsets"],tuple(actual),max_events=max_events)
    from fpgai_research.validation.graph_access_oracle import (
        independent_numpy_access_oracle,
    )
    independent=independent_numpy_access_oracle(
        arch,tuple(actual),max_events=max_events)
    if independent["status"]!="VERIFIED_INDEPENDENT_ACCESS_E0":
        raise AssertionError("SOURCE_GRAPH_INDEPENDENT_SEMANTICS_REJECTED "+str(independent))
    if replay["status"]!="VERIFIED_E0":
        raise AssertionError("SOURCE_GRAPH_LIFETIME_VIOLATION: "+replay["reason"])
    if (replay["source_event_sha256"]!=manifest["source_event_sha256"] or
            replay["last_use_sha256"]!=manifest["last_use_sha256"]):
        raise AssertionError("Source graph logical evidence digest mismatch")
    from fpgai_research.model_ir.graph_edge_witness import edge_witness_receipt
    witness=edge_witness_receipt(arch,manifest["offsets"],events=tuple(actual),
                                 max_events=max_events)
    if (witness["source_witness_sha256"]!=manifest["edge_witness_sha256"] or
            witness["edges"]!=manifest["edge_witnesses"]):
        raise AssertionError("SOURCE_GRAPH_EDGE_WITNESS_MISMATCH")
    return {
        "schema":SCHEMA,"status":"EXECUTED_GRAPH_BOUNDARIES_VERIFIED",
        "functional_sha256":manifest["functional_sha256"],
        "architecture_sha256":manifest["architecture_sha256"],
        "cpp_sha256":manifest["cpp_sha256"],
        "source_event_sha256":replay["source_event_sha256"],
        "last_use_sha256":replay["last_use_sha256"],
        "independent_access_sha256":independent["access_multiset_sha256"],
        "independent_access_reads":independent["read_obligations"],
        "edge_witness_sha256":witness["source_witness_sha256"],
        "edge_witnesses":witness["edges"],
        "independent_phi_issue_status":issue["status"],
        "independent_phi_issue_checked_reads":issue["checked_reads"],
        "independent_phi_issue_checked_operators":issue["checked_operators"],
        "independent_phi_issue_signature_sha256":issue["observed_issue_signature_sha256"],
        "source_array_elements":replay["source_array_elements"],
        "commits":replay["commits"],"reads":replay["reads"],
        "multi_reader_values":replay["multi_reader_values"],
        "operator_phi_iterations":{k:len(v) for k,v in reductions.items()},
        "actual_log_sha256":sha256(run.stdout.encode()).hexdigest(),
        "operator_count":replay["operator_count"],
        "edge_count":replay["edge_count"],
        "native_vitis_calls":0,
        "evidence":"E1-original-compiled-executed-GCC-Phi-graph-source-events",
        "limitations":[
            "Executed GCC host-float, not original fixed-point Vitis CSim",
            "Bounded static rank-two graph, not arbitrary ONNX/streaming/Conv",
            "No hardware cycle/port, DSP/BRAM, throughput, RTL, or priority proof",
        ],
    }
