"""Scientific R2: same Functional IR, changed Phi, actual owner obligations.

This study measures only bounded E0 mapped logical event equivalence, not
source cycle timing, HLS QoR or learned-policy improvement. Mutations and
mapping choices are fixed before observation and each failure is retained.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import subprocess

from fpgai_research.architecture.resolver import resolve
from fpgai_research.experiments.cst_chain_campaign import make_chain
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.mapped_event_schedule import (
    canonical_value_events, verify_value_events,
)

def run(*,output:str|Path)->dict:
    dest=Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Create-only Phi-conditioned study exists")
    graph=make_chain(3,(3,5,4,2))
    request=ArchitectureRequest()
    archs=[
        ("baseline",resolve(graph,request)),
        ("producer_pe2",resolve(graph,request,{"mm0":replace(request,pes=2)})),
        ("consumer_simd2",resolve(graph,request,{"mm1":replace(request,simd=2)})),
    ]
    base_events=canonical_value_events(archs[0][1],row_block=2)
    records=[]
    for name,arch in archs:
        events=canonical_value_events(arch,row_block=2)
        accepted=verify_value_events(arch,events,row_block=2)
        if accepted["status"]!="VERIFIED":
            raise AssertionError("Valid Phi-mapped event study failed")
        unchanged=verify_value_events(arch,base_events,row_block=2)
        if name!="baseline" and unchanged["status"]!="REJECTED":
            raise AssertionError("Stale Phi event ownership must fail")
        first=next(i for i,e in enumerate(events) if e.kind=="CONTRIBUTE")
        missing=verify_value_events(
            arch,events[:first]+events[first+1:],row_block=2)
        if missing["reason"]!="PREMATURE_REDUCTION_COMMIT":
            raise AssertionError("Missing reduction should have concrete proof failure")
        earliest_read=next(e for e in events if e.kind=="READ")
        early=verify_value_events(arch,(earliest_read,)+events,row_block=2)
        if early["reason"]!="READ_BEFORE_READY":
            raise AssertionError("Early read was not rejected")
        records.append({
            "mapping":name,
            "functional_shape":"3x3→3x5→3x4→3x2",
            "positive_status":accepted["status"],
            "positive_event_count":accepted["events_checked"],
            "stale_original_phi_status":unchanged["status"],
            "stale_original_phi_reason":unchanged.get("reason"),
            "missing_reduction_reason":missing["reason"],
            "early_read_reason":early["reason"],
            "native_vitis_calls":0,
        })
    revision=subprocess.run(["git","rev-parse","HEAD"],text=True,
                            capture_output=True,timeout=5)
    summary={
        "schema":"fpgai-research/cst-phi-conditioned-causality/v1",
        "source_revision":revision.stdout.strip() if revision.returncode==0 else None,
        "candidate_mappings":len(records),
        "canonical_verified":sum(r["positive_status"]=="VERIFIED" for r in records),
        "stale_mapping_rejections":sum(
            r["stale_original_phi_status"]=="REJECTED"
            for r in records if r["mapping"]!="baseline"),
        "missing_reduction_rejections":sum(
            r["missing_reduction_reason"]=="PREMATURE_REDUCTION_COMMIT"
            for r in records),
        "early_read_rejections":sum(r["early_read_reason"]=="READ_BEFORE_READY"
                                  for r in records),
        "vitis_calls":0,
        "limits":["E0 explicit logical events; not emitted cycle schedule",
                  "Single-reader sequential row chains only",
                  "Phi owner mismatches require owner-labeled candidate events",
                  "No RL policy or original HLS QoR executed"],
    }
    dest.mkdir(parents=True)
    with (dest/"cases.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)
    (dest/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    return summary

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True)
    a=p.parse_args(argv)
    print(json.dumps(run(output=a.output),indent=2))

if __name__=="__main__":
    main()
