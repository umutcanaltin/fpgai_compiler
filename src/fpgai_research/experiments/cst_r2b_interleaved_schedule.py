"""Prospective R2b E0: fixed Phi-aware consumer issue algorithm and last-read deltas.

This campaign is NOT the historical R2 GCC study, an executed C++
source test, or a physically implemented HLS schedule. It tests a
different, prospectively specified E0 source-order candidate and
retains all no-op/negative effects. Run in a NEW create-only folder.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import subprocess

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.branch_matmul import disjoint_branch_offsets
from fpgai_research.experiments.cst_branch_models import sample_branch
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.interleaved_branch_schedule import (
    interleaved_branch_events,prove_interleaved_source_order,release_positions,
)

SCHEMA="fpgai-research/cst-prospective-r2b-interleaved-logical-e0/v1"
# Last output q=4 is intentional: PE-grouped contributor issue can change
# the order at which each source element's LAST logical consumer READ occurs.
SHAPES=((2,2,3,4),(2,3,5,4),(3,3,4,4))
POLICIES=(
    ("scalar",{},{}),
    ("reader0_pes2",{"pes":2},{}),
    ("reader1_pes2",{},{"pes":2}),
    ("both_pes2",{"pes":2},{"pes":2}),
)


def run(*,output:str|Path)->dict:
    dest=Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Original R2b experimental output is create-only")
    cases=[]
    for dimensions in SHAPES:
        model=sample_branch(*dimensions)
        baseline=None
        for name,phi0,phi1 in POLICIES:
            base=ArchitectureRequest()
            overrides={
                model.operators[i].name:replace(base,**override)
                for i,override in ((1,phi0),(2,phi1))
                if override
            }
            arch=resolve(model,base,overrides)
            offsets=disjoint_branch_offsets(arch,reuse_after_last_read=False)
            checked=prove_interleaved_source_order(arch,offsets)
            sched=interleaved_branch_events(arch)
            last=release_positions(sched,model.operators[0].outputs[0])
            if checked["h_last_read_ordinals"]!=list(last):
                raise AssertionError("Independent last-read order does not match E0 proof")
            if baseline is None:
                baseline=last
            changed=sum(a[1]!=b[1] for a,b in zip(last,baseline))
            cases.append({
                "shape":json.dumps(dimensions),"policy":name,
                "functional_sha256":checked["functional_sha256"],
                "architecture_sha256":checked["architecture_sha256"],
                "logical_source_event_sha256":checked["source_order_event_sha256"],
                "mapped_h_last_read_elements_changed_vs_scalar":changed,
                "mapped_h_logical_values":checked["mapped_multi_reader_values"],
                "logical_scratch_slots":checked["source_scratch_slots"],
                "certificate_verified_e0":checked["status"]=="VERIFIED_E0",
                "source_executed_e1":False,
                "original_gcc_calls":0,
                "original_vendor_calls":0,
            })
    summary={
        "schema":SCHEMA,
        "source_revision":subprocess.run(
            ["git","rev-parse","HEAD"],capture_output=True,
            text=True,check=True,timeout=5).stdout.strip(),
        "graphs":len(SHAPES),"policies_per_graph":len(POLICIES),
        "prespecified_cases":len(cases),
        "verified_e0_cases":sum(c["certificate_verified_e0"] for c in cases),
        "nonbaseline_phi_interventions":len(SHAPES)*(len(POLICIES)-1),
        "logical_h_last_use_changed_interventions":sum(
            c["policy"]!="scalar" and
            c["mapped_h_last_read_elements_changed_vs_scalar"]>0
            for c in cases),
        "logical_h_last_use_unchanged_interventions":sum(
            c["policy"]!="scalar" and
            c["mapped_h_last_read_elements_changed_vs_scalar"]==0
            for c in cases),
        "source_executed_lifetime_change_established":False,
        "actual_source_memory_savings_established":False,
        "hardware_or_RL_advantage_established":False,
        "external_compiler_novelty_established":False,
        "original_gcc_calls":0,"original_vendor_calls":0,
        "evidence":"E0-prospective-fixed-Phi-issue-policy-not-original-source-E1",
        "limitations":[
            "A computed issue schedule is not proof generated C++ executes it",
            "Producer completes before both reader stages; no producer-reader streaming",
            "Restricted static five-operator MatMul diamond with complete assignments",
            "No independent actual emitted C++ trace, numerical oracle, bank ports or physical RTL",
            "No test of RL learning performance, external published compiler parity or firstness",
        ],
    }
    dest.mkdir(parents=True)
    with (dest/"cases.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(cases[0]))
        w.writeheader();w.writerows(cases)
    (dest/"summary.json").write_text(
        json.dumps(summary,indent=2,sort_keys=True)+"\n")
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    print(json.dumps(run(output=args.output),sort_keys=True,indent=2))


if __name__=="__main__":
    main()
