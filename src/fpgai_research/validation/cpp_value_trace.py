"""Independent actual-C++ tensor value lifetime and Phi reduction trace check.

For supported sequential CST row chains only: compile the REAL generated C++
with host tracing, then verify every output logical element is committed after
all K contributions, every mapped contribution matches the resolved Phi, and
every downstream read refers to a live ready element at the correct row-slot
address. Its last read must release the slot before reuse.

Does not infer HLS cycles, bank port conflicts, concurrent RTL availability
or fixed-point bit-exactness. No arbitrary DAG fan-out is supported here.
"""
from __future__ import annotations

from hashlib import sha256
from math import prod
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

from fpgai_research.architecture.resolver import coordinate
from fpgai_research.codegen.hls.row_chain import emit_row_chain_hls
from fpgai_research.model_ir.row_chain import RowChainIR
from fpgai_research.model_ir.semantics import functional_fingerprint


def trace_cpp_values(ir: RowChainIR, *, emission=None, max_events: int = 50_000,
                     cxx: str = "g++") -> dict:
    if not isinstance(ir, RowChainIR):
        raise TypeError("Expected typed sequential RowChainIR")
    if type(max_events) is not int or max_events < 1:
        raise ValueError("Positive finite source event budget required")
    binary=shutil.which(cxx)
    if not binary:
        raise FileNotFoundError("Host C++ compiler not found")
    if emission is None:
        emission=emit_row_chain_hls(ir)
    f=ir.composed.lowered.architectural.functional
    if emission.manifest.get("functional_sha256") != functional_fingerprint(f):
        raise ValueError("Generated C++ does not match Functional IR")
    b=emission.manifest.get("row_block",1)
    if type(b) is not int or not 1<=b<=ir.row_count:
        raise ValueError("Invalid emitted row-block source contract")
    ops=f.operators
    points=ir.composed.lowered.architectural.points
    expected_iterations=sum(prod(p.iteration_shape) for p in points
                            if p.kind in ("MatMul","Gemm"))
    expected_ready=sum(prod(f.tensor(op.outputs[0]).shape) for op in ops)
    expected_read=sum(
        prod(p.iteration_shape) if p.kind in ("MatMul","Gemm")
        else prod(f.tensor(op.outputs[0]).shape)
        for op,p in zip(ops,points))
    if expected_iterations+expected_ready+expected_read>max_events:
        raise ValueError("Exact source element trace exceeds bounded budget")
    input_size=prod(f.tensor(f.inputs[0]).shape)
    output_size=prod(f.tensor(f.outputs[0]).shape)
    main=(f"\nint main() {{float input[{input_size}]={{}}; "
          f"float output[{output_size}]={{}}; "
          "fpgai_model(input,output); return 0;}\n")
    with TemporaryDirectory(prefix="fpgai_value_trace_") as dir:
        source=Path(dir)/"trace.cpp"
        exe=Path(dir)/"trace"
        source.write_text(emission.cpp+main)
        command=[binary,"-std=c++17","-O0","-Wno-unknown-pragmas",
                 "-DFPGAI_HOST_FLOAT","-DFPGAI_TRACE_ITERATIONS",
                 "-DFPGAI_TRACE_TENSOR_VALUES",str(source),"-o",str(exe)]
        compiled=subprocess.run(command,capture_output=True,text=True,timeout=60)
        if compiled.returncode:
            raise RuntimeError("Real generated C++ trace failed to compile: "+
                               compiled.stderr[:3000])
        actual=subprocess.run([str(exe)],capture_output=True,text=True,timeout=60)
        if actual.returncode:
            raise RuntimeError("Real generated C++ trace failed to execute: "+
                               actual.stderr[:3000])
        stdout=actual.stdout
    raw=stdout.splitlines()
    if len(raw)!=expected_iterations+expected_ready+expected_read:
        raise AssertionError(
            "Actual C++ value/iteration event cardinality mismatch: "+
            f"{len(raw)} != {expected_iterations+expected_ready+expected_read}")
    # Each complete row/tensor element is a distinct logical value.
    contributions={}
    committed=set()
    source_live={}  # (stage, slot_offset) -> [row, remaining readers]
    reads_seen=0
    last_releases=0
    phi_verified=0
    for idx,line in enumerate(raw):
        columns=line.split()
        if not columns:
            raise AssertionError("Unexpected empty source trace event")
        tag=columns[0]
        if tag=="FPGAI_ITER":
            if len(columns)!=9:
                raise AssertionError("Malformed mapped-iteration trace")
            stage,row,outcol,k,replica,pe,lane,fold=map(int,columns[1:])
            if not 0<=stage<len(ops) or ops[stage].kind not in ("MatMul","Gemm"):
                raise AssertionError("Unexpected compute iteration stage")
            point=points[stage]
            if (not 0<=row<point.iteration_shape[0]
                    or not 0<=outcol<point.iteration_shape[1]
                    or not 0<=k<point.iteration_shape[2]):
                raise AssertionError("Compute iteration outside declared graph")
            sig=coordinate(point,(row,outcol,k))
            if (replica,pe,lane,fold)!=(
                    sig["replica"],sig["pe"],sig["lane"],sig["reuse_step"][2]):
                raise AssertionError("Actual C++ iteration differs from operator Phi")
            key=(stage,row,outcol)
            if key in committed:
                raise AssertionError("Source executed reduction after logical value commit")
            contributions.setdefault(key,set())
            if k in contributions[key]:
                raise AssertionError("Duplicate reduction contribution")
            contributions[key].add(k)
            phi_verified+=1
        elif tag=="FPGAI_VALUE_READ":
            if len(columns)!=4:
                raise AssertionError("Malformed tensor read event")
            stage,row,col=map(int,columns[1:])
            if not 0<=stage<len(ops) or not 0<=row<ir.row_count:
                raise AssertionError("Tensor read has unsupported stage/row")
            if stage:
                src_width=f.tensor(ops[stage-1].outputs[0]).shape[1]
                if not 0<=col<src_width:
                    raise AssertionError("Tensor read index out of bounds")
                slot=(row%b)*src_width+col
                key=(stage-1,slot)
                value=source_live.get(key)
                if value is None or value[0]!=row or value[1]<1:
                    raise AssertionError(
                        f"READ_BEFORE_READY_OR_STALE_ALIAS event={idx} "
                        f"tensor={ops[stage-1].outputs[0]} row={row} col={col}")
                value[1]-=1
                if value[1]==0:
                    del source_live[key]
                    last_releases+=1
            else:
                input_width=prod(f.tensor(f.inputs[0]).shape[1:])
                if not 0<=col<input_width:
                    raise AssertionError("Runtime input read out of bounds")
            reads_seen+=1
        elif tag=="FPGAI_VALUE_READY":
            if len(columns)!=4:
                raise AssertionError("Malformed element-ready event")
            stage,row,col=map(int,columns[1:])
            if not 0<=stage<len(ops) or not 0<=row<ir.row_count:
                raise AssertionError("Tensor ready outside stage/row")
            width=f.tensor(ops[stage].outputs[0]).shape[1]
            if not 0<=col<width:
                raise AssertionError("Ready output column out of bounds")
            key=(stage,row,col)
            if key in committed:
                raise AssertionError("Duplicate logical output element commit")
            if ops[stage].kind in ("MatMul","Gemm"):
                reduction=points[stage].iteration_shape[2]
                if contributions.get(key,set())!=set(range(reduction)):
                    raise AssertionError(
                        f"PREMATURE_REDUCTION_COMMIT stage={stage} row={row} "
                        f"col={col} completed={len(contributions.get(key,set()))} "
                        f"required={reduction}")
            committed.add(key)
            if stage<len(ops)-1:
                slot=(row%b)*width+col
                address=(stage,slot)
                if address in source_live:
                    raise AssertionError(
                        f"OVERWRITE_UNCONSUMED_VALUE stage={stage} "
                        f"row={row} col={col} address={slot}")
                downstream=ops[stage+1]
                needed=(f.tensor(downstream.outputs[0]).shape[1]
                        if downstream.kind in ("MatMul","Gemm") else 1)
                source_live[address]=[row,needed]
        else:
            raise AssertionError("Unexpected generated source trace: "+line[:120])
    if len(committed)!=expected_ready or phi_verified!=expected_iterations:
        raise AssertionError("Incomplete emitted output or Phi refinement")
    if reads_seen!=expected_read or source_live:
        raise AssertionError("Incomplete consumer reads or unreleased source storage")
    expected_releases=sum(prod(f.tensor(op.outputs[0]).shape)
                          for op in ops[:-1])
    if last_releases!=expected_releases:
        raise AssertionError("Not all intermediate tensor elements were fully consumed")
    return {
        "schema":"fpgai-research/source-mapped-value-lifetime-trace/v1",
        "cpp_sha256":emission.manifest["cpp_sha256"],
        "functional_sha256":functional_fingerprint(f),
        "row_block":b,
        "mapped_reduction_iterations_verified":phi_verified,
        "element_commits_verified":len(committed),
        "downstream_and_input_reads_verified":reads_seen,
        "last_reader_releases_verified":last_releases,
        "source_trace_sha256":sha256(stdout.encode()).hexdigest(),
        "status":"SOURCE_VALUE_LIFETIMES_VERIFIED",
        "native_vitis_calls":0,
        "evidence":"E1-executed-C++-mapped-reduction-and-value-lifetime",
        "limits":[
            "Only sequential single-reader static row-chain graphs",
            "Phi coordinates are spatial ownership, not original HLS cycles",
            "The trace instruments and checks generated GCC-host source only",
            "No fan-out, physical bank ports, fixed-point CSim or RTL proof",
        ],
    }
