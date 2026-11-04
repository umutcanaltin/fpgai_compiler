"""Independent GCC execution audit of mapped MatMul -> (MatMul,MatMul).

Observes each original source reduction, actual h scratch load and commit,
then verifies complete per-output reductions, Phi owners, all reads and
last-use storage safety. No hardware-cycle or RTL claim.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

from fpgai_research.architecture.resolver import coordinate
from fpgai_research.codegen.hls.branch_matmul import SCHEMA as SOURCE_SCHEMA
from fpgai_research.model_ir.branch_matmul_contract import (
    BranchEvent, _hash, admit_branch, branch_matmul_source_events,
    validate_branch_events,
)
from fpgai_research.model_ir.contracts import ArchitecturalModel
from fpgai_research.model_ir.semantics import (
    architecture_fingerprint, functional_fingerprint,
)

SCHEMA="fpgai-research/cst-multi-matmul-branch-executed-gcc-e1/v1"


def trace_branch_cpp(arch:ArchitecturalModel,emission,*,cxx:str="g++",
                     max_events:int=100_000,timeout_seconds:int=90)->dict:
    (m,k,n,q),(h,a,b,g),weights,ops=admit_branch(arch.functional)
    manifest=emission.manifest
    if (manifest.get("schema")!=SOURCE_SCHEMA or
            manifest.get("cpp_sha256")!=sha256(emission.cpp.encode()).hexdigest() or
            manifest.get("functional_sha256")!=functional_fingerprint(arch.functional) or
            manifest.get("architecture_sha256")!=architecture_fingerprint(arch)):
        raise ValueError("Generated branch source/architecture identity mismatch")
    expected,iterations=branch_matmul_source_events(arch,max_events=max_events)
    if len(expected)+sum(len(i) for i in iterations.values())>max_events:
        raise ValueError("Source verification event bound exceeded")
    if type(timeout_seconds) is not int or not 1<=timeout_seconds<=180:
        raise ValueError("Bounded GCC source execution timeout required")
    compiler=shutil.which(cxx)
    if compiler is None:
        raise FileNotFoundError("Original host GCC compiler unavailable")
    runner=(f"\nint main() {{ float x[{m*k}]={{}}; float y[{m*q}]={{}}; "
            "fpgai_model(x,y); return 0; }\n")
    with TemporaryDirectory(prefix="fpgai_branch_phi_") as directory:
        path=Path(directory)
        source=path/"source.cpp"
        binary=path/"runner"
        source.write_text(emission.cpp+runner)
        built=subprocess.run([compiler,"-std=c++17","-O0",
                              "-Wno-unknown-pragmas",
                              "-DFPGAI_HOST_FLOAT",
                              "-DFPGAI_TRACE_ITERATIONS",
                              "-DFPGAI_TRACE_FANOUT",
                              "-DFPGAI_TRACE_BRANCH",
                              str(source),"-o",str(binary)],
                             capture_output=True,text=True,timeout=timeout_seconds)
        if built.returncode:
            raise RuntimeError("Mapped source GCC compilation failed: "+built.stderr[-4000:])
        run=subprocess.run([str(binary)],capture_output=True,text=True,timeout=timeout_seconds)
        if run.returncode:
            raise RuntimeError("Mapped source GCC execution failed: "+run.stderr[-4000:])
    source_events=[]
    seen_iterations={op.name:[] for op in ops[:3]}
    contribution_sets={}
    commits=set()
    pending=None
    for i,line in enumerate(run.stdout.splitlines()):
        parts=line.split()
        if parts[:1]==["FPGAI_ITER"]:
            if pending is not None or len(parts)!=9:
                raise AssertionError("Unmatched or malformed mapped source read")
            try:
                stage,mi,ni,kk,replica,pe,lane,fold=map(int,parts[1:])
            except ValueError as exc:
                raise AssertionError("Invalid mapped source contribution integers") from exc
            if not 0<=stage<3:
                raise AssertionError("Unrecognized source MatMul stage ordinal")
            out_width=n if stage==0 else q
            reduction_length=k if stage==0 else n
            if not (0<=mi<m and 0<=ni<out_width and 0<=kk<reduction_length):
                raise AssertionError("Out-of-bounds source MatMul contribution")
            phi=coordinate(arch.points[stage],(mi,ni,kk))
            if (replica,pe,lane,fold)!=(
                    phi["replica"],phi["pe"],phi["lane"],phi["reuse_step"][2]):
                raise AssertionError("SOURCE_PHI_OWNER_MISMATCH")
            key=(stage,mi,ni)
            done=contribution_sets.setdefault(key,set())
            if kk in done or key in commits:
                raise AssertionError("Duplicate or post-commit source reduction")
            done.add(kk)
            seen_iterations[ops[stage].name].append(
                (mi,ni,kk,replica,pe,lane,fold))
            if stage>0:
                pending=(stage,mi,ni,kk)
            continue
        if parts[:2]==["FPGAI_BRANCH","READ"]:
            if len(parts) not in (6,10):
                raise AssertionError("Malformed original source READ event")
            try:
                element,output_element=int(parts[3]),int(parts[5])
                observed_owner=(tuple(map(int,parts[6:])) if len(parts)==10
                                else None)
            except ValueError as exc:
                raise AssertionError("Malformed source read element/owner") from exc
            event=BranchEvent("READ",parts[2],element,parts[4],output_element,
                              observed_owner)
            if pending is not None:
                stage,mi,ni,kk=pending
                if (event.tensor!=h or event.consumer!=ops[stage].name or
                        event.element!=mi*n+kk or
                        event.output_element!=mi*q+ni or
                        event.phi_owner!=(
                            seen_iterations[ops[stage].name][-1][3],
                            seen_iterations[ops[stage].name][-1][4],
                            seen_iterations[ops[stage].name][-1][5],
                            seen_iterations[ops[stage].name][-1][6])):
                    raise AssertionError("SOURCE_REDUCTION_READ_OWNER_MISMATCH")
                pending=None
            elif event.tensor==h:
                raise AssertionError("Unpaired shared-tensor source read")
            source_events.append(event)
            continue
        if parts[:2]==["FPGAI_FANOUT","COMMIT"]:
            if len(parts)!=4 or pending is not None:
                raise AssertionError("Malformed or premature source COMMIT")
            try:
                element=int(parts[3])
            except ValueError as exc:
                raise AssertionError("Invalid source commit index") from exc
            event=BranchEvent("COMMIT",parts[2],element)
            if event.tensor in (h,a,b):
                stage=(h,a,b).index(event.tensor)
                width=n if stage==0 else q
                key=(stage,element//width,element%width)
                length=k if stage==0 else n
                if key in commits or contribution_sets.get(key,set())!=set(range(length)):
                    raise AssertionError("SOURCE_PREMATURE_REDUCTION_COMMIT")
                commits.add(key)
            source_events.append(event)
            continue
        raise AssertionError(f"Unknown original GCC source trace at {i}: {line[:140]}")
    if pending is not None:
        raise AssertionError("Last source MatMul reduction had no matched physical load")
    for op in ops[:3]:
        if tuple(seen_iterations[op.name])!=iterations[op.name]:
            raise AssertionError("SOURCE_OPERATOR_PHI_ORDER_MISMATCH "+op.name)
    if len(commits)!=m*n+2*m*q:
        raise AssertionError("Source did not commit all mapped MatMul outputs")
    actual=tuple(source_events)
    if actual!=expected:
        loc=next((i for i,(x,y) in enumerate(zip(expected,actual)) if x!=y),
                 min(len(expected),len(actual)))
        raise AssertionError(f"SOURCE_BRANCH_EVENT_MISMATCH at {loc}: "+
                             f"want={expected[loc:loc+2]} got={actual[loc:loc+2]}")
    checked=validate_branch_events(arch,manifest["scratch_offsets"],actual,
                                   max_events=max_events)
    if checked["status"]!="VERIFIED_E0":
        raise AssertionError("SOURCE_BRANCH_LIFETIME_FAILED: "+checked["reason"])
    if (checked["source_order_event_sha256"]!=manifest["logical_event_sha256"] or
            checked["last_use_sha256"]!=manifest["last_use_sha256"]):
        raise AssertionError("Source event/last-use certificate digest mismatch")
    for name,seq in iterations.items():
        if (_hash(seen_iterations[name])!=manifest["phi_iteration_sha256"].get(name) or
                len(seq)!=manifest["phi_iterations"].get(name)):
            raise AssertionError("Source per-operator Phi owner certificate mismatch")
    return {
        "schema":SCHEMA,"status":"EXECUTED_BRANCH_PHI_LIFETIMES_VERIFIED",
        "functional_sha256":manifest["functional_sha256"],
        "architecture_sha256":manifest["architecture_sha256"],
        "cpp_sha256":manifest["cpp_sha256"],
        "phi_iterations_verified":{name:len(seq) for name,seq in seen_iterations.items()},
        "phi_iteration_sha256":{name:_hash(seq) for name,seq in seen_iterations.items()},
        "h_mapped_reads_verified":checked["actual_required_h_reads"],
        "multi_reader_values":checked["mapped_multi_reader_values"],
        "source_event_sha256":checked["source_order_event_sha256"],
        "last_use_sha256":checked["last_use_sha256"],
        "observed_events":len(actual),
        "source_scratch_slots":checked["source_scratch_slots"],
        "raw_source_trace_sha256":sha256(run.stdout.encode()).hexdigest(),
        "original_vitis_calls":0,
        "evidence":"E1-executed-GCC-per-reduction-Phi-and-branch-last-use",
        "limitations":[
            "Host C++ source order is not HLS hardware cycle/port arbitration",
            "Only static three-MatMul five-op diamond, not arbitrary DAG",
            "No original Vitis CSim/CSynth or fixed-point equivalence claim",
        ],
    }
