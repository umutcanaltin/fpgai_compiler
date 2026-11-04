"""Prespecified controlled mutation ablations of the L5e/L5f edge relation.

Compare four ABSTRACTION LEVELS on identical finite case streams:
metadata-only, naive first-read ACK, full logical read cardinality without
Phi owner, and the complete source-loop Phi+last-use relation.
The full checker is the specified finite reference, not an external RTL
oracle. The outcomes only establish which obligations each proxy misses.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
import subprocess

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.branch_matmul import disjoint_branch_offsets
from fpgai_research.experiments.cst_branch_models import sample_branch
from fpgai_research.model_ir.branch_ablation import MODES,ablation_admission
from fpgai_research.model_ir.branch_matmul_contract import (
    branch_matmul_source_events,
)
from fpgai_research.model_ir.contracts import ArchitectureRequest


def probe_catalog():
    model=sample_branch(2,2,3,2)
    default=ArchitectureRequest()
    arch=resolve(model,default,{
        model.operators[1].name:replace(default,pes=2,simd=2),
        model.operators[2].name:replace(default,pes=2,serialization=2),
    })
    offsets=disjoint_branch_offsets(arch)
    events,_=branch_matmul_source_events(arch)
    h,a,b,g=[model.operators[i].outputs[0] for i in range(4)]
    i=next(i for i,x in enumerate(events) if x.kind=="READ"
           and x.tensor==h and x.consumer==model.operators[2].name
           and x.output_element%2==1)
    e=events[i]
    committed_index=next(j for j,x in enumerate(events)
                         if x.kind=="COMMIT" and x.tensor==h)
    first_read=next(x for x in events if x.kind=="READ" and x.tensor==h)
    return arch,(
        ("legal_disjoint",offsets,events,True),
        ("legal_after_all_readers",{**offsets,g:0},events,True),
        ("premature_reuse_before_second_reader",{**offsets,b:0},events,False),
        ("duplicated_mapped_read",offsets,events[:i+1]+(e,)+events[i+1:],False),
        ("missing_second_output_read",offsets,events[:i]+events[i+1:],False),
        ("forged_phi_owner",offsets,events[:i]+(replace(e,phi_owner=(99,0,0,0)),)+events[i+1:],False),
        ("read_before_source_commit",offsets,(first_read,)+events,False),
    )


def run(*,output:str|Path):
    root=Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("Controlled E0 ablation output is create-only")
    arch,cases=probe_catalog()
    rows=[]
    for case,offsets,events,expected_legal in cases:
        outcome={mode:ablation_admission(arch,offsets,events,mode=mode)
                 for mode in MODES}
        if outcome["full_phi_and_last_use"]!=expected_legal:
            raise AssertionError("Prespecified full-spec E0 result changed: "+case)
        for mode in MODES:
            rows.append({
                "case":case,"mode":mode,"expected_under_full_spec":expected_legal,
                "accepted":outcome[mode],
                "false_accept_relative_to_full":outcome[mode] and not expected_legal,
                "false_reject_relative_to_full":not outcome[mode] and expected_legal,
                "original_vitis_calls":0,
            })
    summary={
        "schema":"fpgai-research/cst-phi-branch-ablation-e0/v1",
        "source_revision":subprocess.run(["git","rev-parse","HEAD"],check=True,
            capture_output=True,text=True,timeout=5).stdout.strip(),
        "cases":len(cases),"comparators":list(MODES),
        "rows":len(rows),"full_spec_accepted":sum(x[3] for x in cases),
        "full_spec_rejected":sum(not x[3] for x in cases),
        "mode_results":{mode:{
            "accepted":sum(row["accepted"] for row in rows if row["mode"]==mode),
            "false_accepts_relative_to_full":sum(row["false_accept_relative_to_full"]
                for row in rows if row["mode"]==mode),
            "false_rejects_relative_to_full":sum(row["false_reject_relative_to_full"]
                for row in rows if row["mode"]==mode),
        } for mode in MODES},
        "original_vitis_calls":0,
        "evidence":"E0-controlled-synthetic-contract-mutation-ablation-only",
        "limitations":[
            "Full typed bounded E0 checker IS the reference; NOT an independent RTL oracle",
            "Naive first-read ACK proxy deliberately unsound; not a peer compiler baseline",
            "No trained models, GCC on invalid mutations, original Vitis or QoR",
        ],
    }
    root.mkdir(parents=True)
    with (root/"cases.csv").open("w",newline="") as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    (root/"summary.json").write_text(json.dumps(summary,sort_keys=True,indent=2)+"\n")
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    print(json.dumps(run(output=parser.parse_args(argv).output),indent=2))


if __name__=="__main__":
    main()
