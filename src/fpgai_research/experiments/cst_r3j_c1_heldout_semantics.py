"""R3j C1: held-out bounded shape/architecture differential source semantics.

Twenty-four predeclared source variants (three 2..4 reader fanout degrees
× four shapes excluded from R3e/R3g × two distinct operator-local Phi
mapping policies). Each variant tests one valid source trace and eight
independently specified adverse alterations; two references separately
evaluate source-value safety and Phi source issue-order conformance.

This is source-only bounded falsification and generalization *within*
the admitted MatMul fanout family. No arbitrary graph, Conv network,
native HLS, physical FPGA or trained agent is implied.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import re

from fpgai_research.architecture.resolver import resolve
from fpgai_research.experiments.cst_fanout_models import sample_fanout
from fpgai_research.experiments.cst_r3d_c1c2_finite_contract_ablation import _git_revision
from fpgai_research.experiments.cst_r3g_c1_independent_event_semantics import reference_event_oracle
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.fanout_contract import (
    admit_fanout, disjoint_fanout_offsets, generate_fanout_events,
    verify_fanout_events, verify_fanout_issue,
)

SCHEMA = "fpgai-research/R3j-C1-heldout-source-semantics-v1"
SHAPES = ((1,2,2,2),(2,1,2,3),(1,2,3,2),(2,2,3,3))
READER_COUNTS = (2,3,4)
POLICIES = ("scalar","mixed_phi")
CASES = tuple((r,*shape,policy)
              for r in READER_COUNTS
              for shape in SHAPES
              for policy in POLICIES)
MUTANTS = (
    "safe_wrong_issue_order",
    "early_h_read", "early_h_commit", "wrong_phi_owner",
    "missing_final_read", "duplicate_producer_term",
    "overlapping_reader_scratch", "unknown_final_tensor",
)


def _source(config:tuple):
    if config not in CASES:
        raise ValueError("R3j: undeclared source family/shape/policy")
    readers,m,k,n,q,policy=config
    model=sample_fanout(readers,m=m,k=k,n=n,q=q)
    default=ArchitectureRequest()
    changes={}
    if policy=="mixed_phi":
        changes["reader0"]=replace(default,pes=2)
        changes["reader1"]=replace(default,simd=2)
        if readers>=3:
            changes["reader2"]=replace(default,pes=2,simd=2)
    arch=resolve(model,default,changes)
    contract=admit_fanout(arch)
    events=generate_fanout_events(arch)
    offsets=disjoint_fanout_offsets(contract)
    return arch,contract,events,offsets


def variants(config:tuple):
    arch,spec,original,offsets=_source(config)
    valid=[("canonical",original,dict(offsets),True,True)]
    h_reads=[i for i,e in enumerate(original)
             if e.kind=="READ" and e.tensor==spec.h]
    h_commit=next(i for i,e in enumerate(original)
                  if e.kind=="COMMIT" and e.tensor==spec.h)
    assert len(h_reads)>=2 and original[h_reads[0]].operation!=original[h_reads[1]].operation

    swapped=list(original)
    a,b=h_reads[0],h_reads[1]
    swapped[a],swapped[b]=swapped[b],swapped[a]
    valid.append(("safe_wrong_issue_order",tuple(swapped),dict(offsets),True,False))

    premature=list(original)
    early=premature.pop(h_reads[0])
    premature.insert(0,early)
    valid.append(("early_h_read",tuple(premature),dict(offsets),False,None))

    premature=list(original)
    early=premature.pop(h_commit)
    premature.insert(0,early)
    valid.append(("early_h_commit",tuple(premature),dict(offsets),False,None))

    wrong=list(original)
    wrong[h_reads[0]]=replace(wrong[h_reads[0]],phi_owner=(99,99,99,99))
    valid.append(("wrong_phi_owner",tuple(wrong),dict(offsets),False,False))
    valid.append(("missing_final_read",original[:-1],dict(offsets),False,None))
    # Insert duplicate first producer contribution after itself.
    valid.append(("duplicate_producer_term",
                  (original[0],)+original,dict(offsets),False,None))
    # The first two reader results are simultaneously live before merge0,
    # hence they cannot share one scratch starting address.
    conflicting=dict(offsets)
    conflicting[spec.reader_values[1]]=conflicting[spec.reader_values[0]]
    valid.append(("overlapping_reader_scratch",original,conflicting,False,None))
    changed=list(original)
    changed[-1]=replace(changed[-1],tensor="source_missing_value")
    valid.append(("unknown_final_tensor",tuple(changed),dict(offsets),False,None))
    if len(valid)!=1+len(MUTANTS):
        raise AssertionError("R3j fixed mutation population changed")
    return arch,spec,tuple(valid)


def evaluate(config:tuple)->dict:
    arch,spec,variants_=variants(config)
    outcomes=[]
    incorrect=[]
    classifications={"safe_and_issue":0,"safe_wrong_issue":0,
                     "unsafe":0}
    for name,events,offsets,expected_safe,expected_issue in variants_:
        oracle_safe,oracle_issue=reference_event_oracle(arch,events,offsets)
        compiler_safe=(verify_fanout_events(arch,offsets,events)
                       ["status"]=="VERIFIED_E0")
        compiler_issue=(verify_fanout_issue(arch,events)
                        ["status"]=="VERIFIED_E0")
        if (oracle_safe!=compiler_safe or oracle_issue!=compiler_issue
                or oracle_safe is not expected_safe
                or (expected_issue is not None
                    and oracle_issue is not expected_issue)):
            incorrect.append({
                "name":name,"oracle":[oracle_safe,oracle_issue],
                "compiler":[compiler_safe,compiler_issue],
                "expected":[expected_safe,expected_issue],
            })
        label=("safe_and_issue" if oracle_safe and oracle_issue
               else "safe_wrong_issue" if oracle_safe
               else "unsafe")
        classifications[label]+=1
        outcomes.append({
            "name":name, "safe":oracle_safe, "phi_issue_conformant":oracle_issue,
            "source_trace_sha256":sha256(json.dumps(
                [e.__dict__ for e in events],sort_keys=True,
                separators=(",",":"),allow_nan=False).encode()).hexdigest(),
            "source_offsets_sha256":sha256(json.dumps(
                sorted(offsets.items()),sort_keys=True,
                separators=(",",":"),allow_nan=False).encode()).hexdigest(),
        })
    if incorrect:
        raise AssertionError("R3j independent oracle/compiler mismatch: "+
                             repr(incorrect[:4]))
    if (classifications !=
            {"safe_and_issue":1,"safe_wrong_issue":1,"unsafe":len(MUTANTS)-1}):
        raise AssertionError("R3j positive/negative source population missing")
    return {
        "case":[*config],
        "declared_topology":"MatMul producer, independent readers, Add chain, Relu",
        "readers":len(spec.readers),
        "source_event_count":len(variants_[0][1]),
        "variant_population":len(outcomes),
        "semantic_oracle_disagreements":0,
        "expected_semantic_violations":len(MUTANTS)-1,
        "safe_but_phi_issue_nonconformant":1,
        "valid_source_trace":1,
        "classification_counts":classifications,
        "outcomes":outcomes,
        "source_only":True,
    }


def run()->dict:
    rows=[evaluate(c) for c in CASES]
    cases=len(CASES)
    total=sum(r["variant_population"] for r in rows)
    if cases!=24 or total!=216:
        raise AssertionError("R3j declared cohort unexpectedly changed")
    return {
        "schema":SCHEMA,
        "prespecified_source_cases":cases,
        "prespecified_shapes":[list(s) for s in SHAPES],
        "prespecified_reader_counts":list(READER_COUNTS),
        "prespecified_policies":list(POLICIES),
        "variants_per_case":1+len(MUTANTS),
        "total_trace_variants":total,
        "matched_independent_oracle":total,
        "independent_oracle_disagreements":0,
        "safe_wrong_phi_issue_control_count":cases,
        "unsafe_source_mutants_count":cases*(len(MUTANTS)-1),
        "rows":rows,
        "original_gcc_calls":0,"original_vitis_calls":0,
        "original_vivado_calls":0,"trained_agents":0,
        "scope_limits":[
            "Held-out only in shape and operator-local Phi of the SAME fixed MatMul fanout family",
            "Both interpreters share the admitted graph topology and contract assumptions",
            "No independent external compiler baseline, numeric fixed-point proof or physical FPGA evidence",
            "No all-graph or all-mutation correctness/completeness theorem",
        ],
    }


def create_only(report:dict,*,output:str|Path,source_revision:str)->None:
    path=Path(output)
    if path.exists() or path.is_symlink():
        raise FileExistsError("R3j source experiment is strictly create-only")
    if (report.get("schema")!=SCHEMA or
            re.fullmatch(r"[a-f0-9]{40}",source_revision) is None):
        raise ValueError("R3j study missing schema or exact source revision")
    path.mkdir(parents=True,exist_ok=False)
    (path/"r3j_heldout_source_semantics.json").write_text(
        json.dumps({**report,"source_revision":source_revision},
                   indent=2,sort_keys=True,allow_nan=False)+"\n",
        encoding="utf-8")


def audit(root:str|Path)->dict:
    folder=Path(root)
    receipt=folder/"r3j_heldout_source_semantics.json"
    if folder.is_symlink() or not folder.is_dir() or receipt.is_symlink() or not receipt.is_file():
        raise ValueError("R3j receipt absent/symlinked")
    raw=json.loads(receipt.read_text())
    rows=raw.get("rows")
    if (raw.get("schema")!=SCHEMA
            or re.fullmatch(r"[a-f0-9]{40}",raw.get("source_revision") or "") is None
            or raw.get("prespecified_source_cases")!=len(CASES)
            or raw.get("variants_per_case")!=1+len(MUTANTS)
            or raw.get("total_trace_variants")!=len(CASES)*(1+len(MUTANTS))
            or raw.get("matched_independent_oracle")!=raw["total_trace_variants"]
            or raw.get("independent_oracle_disagreements")!=0
            or raw.get("prespecified_shapes")!=[list(s) for s in SHAPES]
            or raw.get("prespecified_reader_counts")!=list(READER_COUNTS)
            or raw.get("prespecified_policies")!=list(POLICIES)
            or any(raw.get(field)!=0 for field in
                   ("original_gcc_calls","original_vitis_calls",
                    "original_vivado_calls","trained_agents"))
            or not isinstance(rows,list) or len(rows)!=len(CASES)):
        raise ValueError("R3j source attestation or total denominator altered")
    for case,row in zip(CASES,rows):
        out=row.get("outcomes")
        if (row.get("case")!=list(case)
                or row.get("variant_population")!=1+len(MUTANTS)
                or row.get("semantic_oracle_disagreements")!=0
                or row.get("valid_source_trace")!=1
                or row.get("safe_but_phi_issue_nonconformant")!=1
                or row.get("expected_semantic_violations")!=len(MUTANTS)-1
                or row.get("source_only") is not True
                or not isinstance(out,list) or len(out)!=1+len(MUTANTS)
                or [v.get("name") for v in out]!=["canonical",*MUTANTS]
                or out[0].get("safe") is not True
                or out[0].get("phi_issue_conformant") is not True
                or out[1].get("safe") is not True
                or out[1].get("phi_issue_conformant") is not False
                or any(v.get("safe") is not False for v in out[2:])
                or any(re.fullmatch(r"[a-f0-9]{64}",v.get("source_trace_sha256") or "") is None for v in out)
                or any(re.fullmatch(r"[a-f0-9]{64}",v.get("source_offsets_sha256") or "") is None for v in out)
                or out[0]["source_offsets_sha256"]==out[7]["source_offsets_sha256"]):
            raise ValueError("R3j original variant population/typed status corrupted")
    return {
        "schema":SCHEMA,
        "original_source_revision":raw["source_revision"],
        "source_cases":len(CASES),
        "tested_trace_variants":raw["total_trace_variants"],
        "independent_oracle_disagreements":0,
        "read_only_receipt_sha256":sha256(receipt.read_bytes()).hexdigest(),
        "original_compiler_or_vendor_calls_during_audit":0,
    }


def main(argv:list[str]|None=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output")
    mode.add_argument("--study")
    args=parser.parse_args(argv)
    if args.study:
        print(json.dumps(audit(args.study),indent=2,sort_keys=True))
        return
    source_revision=_git_revision()
    report=run()
    create_only(report,output=args.output,source_revision=source_revision)
    print(json.dumps({
        "schema":SCHEMA,"source_revision":source_revision,
        "source_cases":report["prespecified_source_cases"],
        "tested_trace_variants":report["total_trace_variants"],
        "oracle_disagreements":report["independent_oracle_disagreements"],
        "new_vendor_calls":0,
    },indent=2,sort_keys=True))


if __name__=="__main__":
    main()
