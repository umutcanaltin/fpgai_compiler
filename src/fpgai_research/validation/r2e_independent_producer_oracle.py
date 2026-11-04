"""Independent bounded source trace oracle for full H-producer MatMul reductions.

Unlike the R2d branch READ/COMMIT-only contract, this verifier REQUIRES
one runtime CONTRIBUTE witness for every X[row,kk]*W0[kk,col] term
before H[row,col] COMMIT, including correct input and weight indices.

This module has no dependency on compiler source emission, mapped
iteration generation, precomputed producer events or C++ instrumentation.
The witness is checked against a separate closed-form MatMul definition.
Numeric equality is a SEPARATE obligation: a logged contribution alone
does not prove the source multiplied the correct operand.
"""
from __future__ import annotations

from hashlib import sha256
from math import isfinite
import json

SCHEMA="fpgai-research/cst-R2e-independent-producer-contributions-e0/v1"


def inspect_producer_stdout(raw:str, *, m:int,k:int,n:int,
                            h:str="h", max_events:int=100_000)->dict:
    if (any(type(x) is not int or not 1<=x<=256 for x in (m,k,n))
            or type(max_events) is not int or max_events<1
            or type(h) is not str or not h.isidentifier()):
        raise ValueError("Finite, typed producer-reference dimensions required")
    if len(raw.encode())>8*1024*1024:
        raise ValueError("Unbounded original runtime producer trace")
    lines=raw.splitlines()
    filtered=[]
    contributions={}
    completed=set()
    consumed_commits=set()
    pending_commit=None
    expected_value=None
    source_values={}
    producer_records=[]
    def fail(reason,line,record):
        raise ValueError(reason+" line="+str(line)+" record="+record[:160])
    for line_number,line in enumerate(lines,1):
        pieces=line.split()
        if expected_value is not None and pieces[:2]!=["FPGAI_PROD","VALUE"]:
            fail("PRODUCER_H_VALUE_NOT_ADJACENT_TO_SOURCE_COMMIT",
                 line_number,line)
        if pieces[:2]==["FPGAI_PROD","CONTRIBUTE"]:
            if len(pieces)!=6:
                fail("INVALID_PRODUCER_CONTRIBUTE_WITNESS",line_number,line)
            if pending_commit is not None:
                fail("PRODUCER_COMMIT_WITHOUT_TYPED_H_COMMIT",line_number,line)
            try:
                out_idx,kk,input_idx,weight_idx=map(int,pieces[2:])
            except ValueError:
                fail("NONINTEGER_PRODUCER_CONTRIBUTE_WITNESS",line_number,line)
            if not (0<=out_idx<m*n and 0<=kk<k):
                fail("OUT_OF_RANGE_PRODUCER_CONTRIBUTE",line_number,line)
            if out_idx in completed:
                fail("PRODUCER_CONTRIBUTE_AFTER_COMMIT",line_number,line)
            row,col=divmod(out_idx,n)
            if input_idx!=row*k+kk or weight_idx!=kk*n+col:
                fail("PRODUCER_WRONG_INPUT_OR_WEIGHT_SUBSCRIPT",line_number,line)
            done=contributions.setdefault(out_idx,set())
            if kk in done:
                fail("PRODUCER_DUPLICATE_REDUCTION_TERM",line_number,line)
            done.add(kk)
            producer_records.append(("CONTRIBUTE",out_idx,kk,input_idx,weight_idx))
        elif pieces[:2]==["FPGAI_PROD","COMMIT"]:
            if len(pieces)!=3:
                fail("INVALID_PRODUCER_COMMIT_WITNESS",line_number,line)
            if pending_commit is not None:
                fail("PRODUCER_COMMIT_WITHOUT_TYPED_H_COMMIT",line_number,line)
            try:
                out_idx=int(pieces[2])
            except ValueError:
                fail("NONINTEGER_PRODUCER_COMMIT_WITNESS",line_number,line)
            if not 0<=out_idx<m*n or out_idx in completed:
                fail("DUPLICATE_OR_INVALID_PRODUCER_COMMIT",line_number,line)
            missing=set(range(k))-contributions.get(out_idx,set())
            if missing:
                fail("PREMATURE_H_PRODUCER_COMMIT_MISSING_TERMS="+repr(sorted(missing)),
                     line_number,line)
            completed.add(out_idx)
            pending_commit=out_idx
            producer_records.append(("COMMIT",out_idx))
        elif pieces[:2]==["FPGAI_PROD","VALUE"]:
            if len(pieces)!=4 or expected_value is None:
                fail("INVALID_OR_UNEXPECTED_H_PRODUCER_VALUE",line_number,line)
            try:
                out_idx=int(pieces[2])
                measured=float(pieces[3])
            except ValueError:
                fail("INVALID_OR_NONNUMERIC_H_PRODUCER_VALUE",line_number,line)
            if (out_idx!=expected_value or out_idx in source_values or
                    not isfinite(measured)):
                fail("INCORRECT_H_PRODUCER_VALUE_WITNESS",line_number,line)
            source_values[out_idx]=measured
            producer_records.append(("VALUE",out_idx,measured))
            expected_value=None
        else:
            if pending_commit is not None:
                if (len(pieces)!=4 or pieces[:3]!=["FPGAI_R2B","COMMIT",h]
                        or pieces[3]!=str(pending_commit)):
                    fail("PRODUCER_COMMIT_MISSING_ADJACENT_SOURCE_STORE_WITNESS",
                         line_number,line)
                consumed_commits.add(pending_commit)
                expected_value=pending_commit
                pending_commit=None
            elif pieces[:3]==["FPGAI_R2B","COMMIT",h]:
                fail("H_SOURCE_COMMIT_WITHOUT_PRODUCER_COMPLETENESS",line_number,line)
            if pieces and pieces[0]!="FPGAI_R2B":
                fail("UNRECOGNIZED_SOURCE_EVENT",line_number,line)
            filtered.append(line)
        if len(producer_records)>max_events:
            raise ValueError("Bounded independent producer-event budget exceeded")
    if pending_commit is not None:
        raise ValueError("PRODUCER_COMMIT_WITHOUT_TYPED_H_COMMIT_AT_END")
    if expected_value is not None:
        raise ValueError("MISSING_ADJACENT_PRODUCER_H_VALUE_AT_END")
    if (len(completed)!=m*n or len(consumed_commits)!=m*n or
            set(source_values)!=set(range(m*n)) or
            set(contributions)!=set(range(m*n)) or
            any(contributions[out]!=set(range(k)) for out in range(m*n))):
        raise ValueError("INCOMPLETE_PRODUCER_CONTRIBUTIONS_OR_COMMITS")
    return {
        "schema":SCHEMA,
        "status":"INDEPENDENT_COMPLETE_PRODUCER_TRACE_E0",
        "H_elements":m*n,
        "required_contributions":m*k*n,
        "observed_contributions":sum(len(items) for items in contributions.values()),
        "producer_commits":len(completed),
        "H_typed_source_commits":len(consumed_commits),
        "runtime_H_values":[source_values[i] for i in range(m*n)],
        "producer_H_values_sha256":sha256(json.dumps(
            [source_values[i] for i in range(m*n)],
            separators=(",",":"),allow_nan=False).encode()).hexdigest(),
        "producer_trace_sha256":sha256(json.dumps(producer_records,
            separators=(",",":")).encode()).hexdigest(),
        "typed_branch_stdout":"\n".join(filtered)+"\n",
        "original_gcc_calls":0,"original_native_vendor_calls":0,
        "limitations":[
            "Source-runtime event witness does not independently establish actual multiplication values",
            "Separate independent NumPy comparison of H values and final outputs required for arithmetic correctness",
            "No FPGA hardware-cycle completion, RAM ports, or arbitrary DAG producer semantics",
        ],
    }
