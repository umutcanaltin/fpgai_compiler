"""CST C2 original finite certificate mutation-adequacy matrix (E0 only).

Twelve static graph/Phi cases × seven prespecified fault classes.
No source code recompilation, GCC, HLS, Vivado or external comparator.
REPORT surviving mutants; do not silently drop failed controls or
declare independence for two checkers which share an event generator.
Source-level general safety and Phi-policy conformance are different.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import csv
import json
from pathlib import Path
import subprocess

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.branch_matmul import disjoint_branch_offsets
from fpgai_research.experiments.cst_branch_models import sample_branch
from fpgai_research.experiments.cst_r2b_source_campaign import SHAPES,POLICIES
from fpgai_research.model_ir.branch_matmul_contract import (
    BranchEvent,validate_branch_events,
)
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.independent_phi_issue_reference import (
    validate_phi_issue_conformance,
)
from fpgai_research.model_ir.interleaved_branch_schedule import interleaved_branch_events

SCHEMA="fpgai-research/cst-C2-finite-mutation-adequacy-e0/v1"
FAULTS=(
    "premature_A_reduction_commit",
    "premature_B_reduction_commit",
    "premature_Add_commit",
    "live_H_slot_alias",
    "wrong_Phi_read_owner",
    "duplicate_H_read",
    "missing_H_read",
    "nonconforming_safe_Phi_issue_order",
)


def _mutate(events,offsets,kind,*,h,a,b,g,ops):
    source=list(events)
    offsets=dict(offsets)
    def move_commit(tensor,at):
        event=next(e for e in source if e.kind=="COMMIT"
                   and e.tensor==tensor and e.element==0)
        source.remove(event)
        where=next(i for i,e in enumerate(source) if at(e))
        source.insert(where+1,event)
    if kind=="premature_A_reduction_commit":
        move_commit(a,lambda e: e.kind=="READ" and e.tensor==h
                    and e.consumer==ops[1].name and e.output_element==0)
    elif kind=="premature_B_reduction_commit":
        move_commit(b,lambda e: e.kind=="READ" and e.tensor==h
                    and e.consumer==ops[2].name and e.output_element==0)
    elif kind=="premature_Add_commit":
        move_commit(g,lambda e: e.kind=="READ" and e.tensor==a
                    and e.element==0)
    elif kind=="live_H_slot_alias":
        offsets[a]=offsets[h]
    elif kind=="wrong_Phi_read_owner":
        i=next(i for i,e in enumerate(source) if e.kind=="READ" and e.tensor==h)
        event=source[i]
        owner=event.phi_owner
        source[i]=replace(event,phi_owner=(owner[0]+1,*owner[1:]))
    elif kind=="duplicate_H_read":
        i=next(i for i,e in enumerate(source) if e.kind=="READ" and e.tensor==h)
        source.insert(i+1,source[i])
    elif kind=="missing_H_read":
        i=next(i for i,e in enumerate(source) if e.kind=="READ" and e.tensor==h)
        source.pop(i)
    elif kind=="nonconforming_safe_Phi_issue_order":
        i=next(i for i,e in enumerate(source) if e.kind=="READ" and e.tensor==h)
        if source[i+1].kind!="READ" or source[i+1].tensor!=h:
            raise AssertionError("First two H READs must be adjacent in R2b policy")
        source[i],source[i+1]=source[i+1],source[i]
    else:
        raise ValueError("Unknown prespecified C2 mutant class")
    return tuple(source),offsets


def run(*,output:str|Path|None=None)->dict:
    target=Path(output) if output is not None else None
    if target is not None and (target.exists() or target.is_symlink()):
        raise FileExistsError("C2 mutation-adequacy report is create-only")
    rows=[]
    for dimensions in SHAPES:
        m,k,n,q=dimensions
        model=sample_branch(*dimensions)
        base=ArchitectureRequest()
        for policy,phi_a,phi_b in POLICIES:
            overrides={
                model.operators[stage].name:replace(base,**options)
                for stage,options in ((1,phi_a),(2,phi_b)) if options
            }
            arch=resolve(model,base,overrides)
            offsets=disjoint_branch_offsets(arch)
            events=interleaved_branch_events(arch).events
            baseline=validate_branch_events(arch,offsets,events)
            baseline_phi=validate_phi_issue_conformance(arch,events)
            if (baseline["status"]!="VERIFIED_E0" or
                    baseline_phi["status"]!="VERIFIED_E0"):
                raise ValueError("E0 original baseline certificate invalid; do not label mutants")
            for fault in FAULTS:
                changed,mutation_offsets=_mutate(
                    events,offsets,fault,
                    h=model.operators[0].outputs[0],
                    a=model.operators[1].outputs[0],
                    b=model.operators[2].outputs[0],
                    g=model.operators[3].outputs[0],
                    ops=model.operators)
                generic=validate_branch_events(arch,mutation_offsets,changed)
                phi=validate_phi_issue_conformance(arch,changed)
                generic_kill=generic["status"]!="VERIFIED_E0"
                phi_kill=phi["status"]!="VERIFIED_E0"
                rows.append({
                    "shape":json.dumps(dimensions),"policy":policy,
                    "fault_class":fault,
                    "generic_safety_killed":generic_kill,
                    "generic_safety_status":generic["status"],
                    "generic_safety_reason":generic.get("reason") or "NONE",
                    "independent_phi_killed":phi_kill,
                    "independent_phi_status":phi["status"],
                    "independent_phi_reason":phi.get("reason") or "NONE",
                    "combined_E0_killed":generic_kill or phi_kill,
                    "E1_GCC_executed":False,"original_vendor_calls":0,
                })
    counters={}
    for fault in FAULTS:
        cohort=[row for row in rows if row["fault_class"]==fault]
        counters[fault]={
            "prespecified_mutants":len(cohort),
            "generic_safety_killed":sum(row["generic_safety_killed"] for row in cohort),
            "independent_phi_killed":sum(row["independent_phi_killed"] for row in cohort),
            "combined_E0_killed":sum(row["combined_E0_killed"] for row in cohort),
            "combined_E0_survivors":sum(not row["combined_E0_killed"] for row in cohort),
        }
    revision=subprocess.run(
        ["git","rev-parse","HEAD"],capture_output=True,text=True,
        check=True,timeout=5).stdout.strip()
    summary={
        "schema":SCHEMA,"source_revision":revision,
        "prespecified_graphs":len(SHAPES),
        "prespecified_Phi_policies":len(POLICIES),
        "prespecified_fault_classes":len(FAULTS),
        "prespecified_mutants":len(rows),
        "combined_E0_mutants_killed":sum(r["combined_E0_killed"] for r in rows),
        "combined_E0_mutants_survived":sum(not r["combined_E0_killed"] for r in rows),
        "class_specific_adequacy":counters,
        "original_gcc_calls":0,"original_vitis_calls":0,"original_vivado_calls":0,
        "E1_observed_fault_detection_established":False,
        "hardware_fault_detection_established":False,
        "external_novelty_established":False,
        "limitations":[
            "Prespecified 8 small fault classes, not a complete verifier correctness theorem",
            "The generic safety checker does not enforce the Phi PE-group issue policy",
            "Independent Phi source-order reference checks exact bounded rule; not hardware clocks",
            "Generator and compiler share a graph fixture and data types",
            "No actual compiled source mutant/NumPy E1 fault matrix or vendor hardware qualification",
        ],
    }
    if target is not None:
        target.mkdir(parents=True)
        with (target/"cases.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
        (target/"summary.json").write_text(
            json.dumps(summary,indent=2,sort_keys=True)+"\n")
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",help="Optional new create-only CPU-only evidence directory")
    args=parser.parse_args(argv)
    print(json.dumps(run(output=args.output),indent=2,sort_keys=True))


if __name__=="__main__":
    main()
