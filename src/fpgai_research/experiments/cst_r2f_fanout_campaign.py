"""R2f original E0 2..4 fanout MatMul source-event certificate campaign.

Prospective fixed 36-case population (3 fanouts × 3 shapes × 4
reader Phi policies). No C++/GCC, native Vitis/Vivado or RL calls.

Events include producer CONTRIBUTE and COMMIT, independently mapped
consumer READ/COMMIT, Add-chain READ/COMMIT and final READ.
A separate full read/commit-only projection is used with the
existing exact unit-slot lifetime allocator. This is a LOGICAL
source-event capacity bound, not actual source-C++ memory, RAM ports
or FPGA BRAM. Report Phi-induced ties, savings AND regressions.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import subprocess

from fpgai_research.architecture.resolver import resolve
from fpgai_research.experiments.cst_fanout_models import sample_fanout
from fpgai_research.model_ir.branch_matmul_contract import BranchEvent
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.element_scratch_allocation import (
    allocate_minimum_slots,verify_allocation,
)
from fpgai_research.model_ir.fanout_contract import (
    admit_fanout,disjoint_fanout_offsets,generate_fanout_events,
    verify_fanout_events,verify_fanout_issue,
)

SCHEMA="fpgai-research/cst-R2f-36-case-source-fanout-E0/v1"
SHAPES=((2,2,3,4),(2,3,5,4),(3,3,4,4))
FANOUTS=(2,3,4)
POLICY_NAMES=("scalar","reader0_pes2","last_reader_pes2","all_pes2")


def policy_widths(readers:int,policy:str)->tuple[int,...]:
    if readers not in FANOUTS or policy not in POLICY_NAMES:
        raise ValueError("Only prespecified R2f fanout and source Phi policies")
    if policy=="scalar":
        return (1,)*readers
    if policy=="reader0_pes2":
        return (2,)+(1,)*(readers-1)
    if policy=="last_reader_pes2":
        return (1,)*(readers-1)+(2,)
    return (2,)*readers


def project_source_value_events(events):
    """Only COMMIT/READ touch logical scratch; exclude producer computations."""
    return tuple(BranchEvent(e.kind,e.tensor,e.element,
                             e.operation if e.kind=="READ" else "",
                             e.output_element if e.kind=="READ" else -1,
                             e.phi_owner if e.kind=="READ" else None)
                 for e in events if e.kind in ("COMMIT","READ"))


def run(*,output:str|Path|None=None)->dict:
    root=Path(output) if output is not None else None
    if root is not None and (root.exists() or root.is_symlink()):
        raise FileExistsError("Original R2f E0 evidence is create-only")
    # Original audit receipts must refer to the exact tracked source
    # that was actually used. Do not silently use a dirty checkout.
    dirty=subprocess.run(
        ["git","status","--porcelain","--untracked-files=no"],
        capture_output=True,text=True,check=True,timeout=5).stdout
    if dirty.strip():
        raise RuntimeError("Original R2f E0 requires clean tracked source")
    rev=subprocess.run(["git","rev-parse","HEAD"],capture_output=True,
        text=True,check=True,timeout=5).stdout.strip()
    if len(rev)!=40:
        raise RuntimeError("Full original source SHA required")
    rows=[]
    for readers in FANOUTS:
        for dims in SHAPES:
            m,k,n,q=dims
            model=sample_fanout(readers,m=m,k=k,n=n,q=q)
            for policy in POLICY_NAMES:
                requested=policy_widths(readers,policy)
                base=ArchitectureRequest()
                arch=resolve(model,base,{
                    f"reader{i}":replace(base,pes=width)
                    for i,width in enumerate(requested) if width!=1
                })
                spec=admit_fanout(arch)
                events=generate_fanout_events(arch)
                offsets=disjoint_fanout_offsets(spec)
                source=verify_fanout_events(arch,offsets,events)
                order=verify_fanout_issue(arch,events)
                if source["status"]!="VERIFIED_E0" or order["status"]!="VERIFIED_E0":
                    raise AssertionError(
                        "Original R2f E0 graph/Phi completeness failed: "+
                        repr((source,order)))
                scratch_events=project_source_value_events(events)
                allocation=allocate_minimum_slots(scratch_events)
                alias=verify_allocation(
                    scratch_events,allocation.slot_map(),
                    unit_slots=allocation.unit_slots)
                if (alias["status"]!="EXACT_SOURCE_SLOT_REPLAY_VERIFIED" or
                        allocation.unit_slots!=allocation.peak_simultaneous_live_values):
                    raise AssertionError("Original R2f unit-slot source bound failed")
                h_last={}
                h_rank=0
                for ev in events:
                    if ev.kind=="READ" and ev.tensor==spec.h:
                        h_last[ev.element]=(h_rank,ev.operation)
                        h_rank+=1
                rows.append({
                    "case":f"case_{len(rows):03d}",
                    "fanout":readers,
                    "shape":json.dumps(dims),
                    "policy":policy,
                    "source_revision":rev,
                    "functional_sha256":source["functional_sha256"],
                    "architecture_sha256":source["architecture_sha256"],
                    "source_event_sha256":source["source_event_sha256"],
                    "source_scratch_schedule_sha256":allocation.schedule_sha256,
                    "producer_contributions":source["producer_contributions"],
                    "H_reads":source["actual_H_reads"],
                    "H_last_read_rank_and_reader":json.dumps(
                        sorted((element,*pair) for element,pair in h_last.items())),
                    "logical_intermediate_values":source["all_intermediate_value_count"],
                    "whole_tensor_disjoint_unit_slots":source["source_scratch_slots"],
                    "exact_fixed_order_minimum_unit_slots":allocation.unit_slots,
                    "source_peak_live_unit_values":allocation.peak_simultaneous_live_values,
                    "source_readiness_verified":True,
                    "independent_Phi_issue_verified":True,
                    "logical_scalar_alias_replay_verified":True,
                    "source_C++_or_original_GCC_executed":False,
                    "native_vendor_calls":0,
                })
    if len(rows)!=len(FANOUTS)*len(SHAPES)*len(POLICY_NAMES):
        raise AssertionError("Incomplete prespecified R2f source campaign")
    changed_counts=changed_rank=0
    saved=tied=regressed=0
    for fanout in FANOUTS:
        for dims in SHAPES:
            subset=[r for r in rows if r["fanout"]==fanout
                    and json.loads(r["shape"])==list(dims)]
            original=next(r for r in subset if r["policy"]=="scalar")
            for row in subset:
                if row["policy"]=="scalar":
                    continue
                delta=(original["exact_fixed_order_minimum_unit_slots"]
                       -row["exact_fixed_order_minimum_unit_slots"])
                changed_counts+=delta!=0
                changed_rank+=(
                    row["H_last_read_rank_and_reader"]!=
                    original["H_last_read_rank_and_reader"])
                saved+=delta>0
                tied+=delta==0
                regressed+=delta<0
    summary={
        "schema":SCHEMA,"source_revision":rev,
        "prespecified_fanout_degrees":list(FANOUTS),
        "prespecified_shapes":[list(x) for x in SHAPES],
        "prespecified_policies":list(POLICY_NAMES),
        "prespecified_cases":len(rows),
        "all_producer_reader_Add_and_final_E0_certificates_verified":len(rows),
        "independent_Phi_issue_certificates_verified":len(rows),
        "exact_fixed_order_unit_slot_certificates_verified":len(rows),
        "unique_graph_Phi_interventions":len(FANOUTS)*len(SHAPES)*(len(POLICY_NAMES)-1),
        "changed_H_last_reader_rank_interventions":changed_rank,
        "changed_unit_slot_capacity_interventions":changed_counts,
        "Phi_interventions_reduced_logical_unit_slots":saved,
        "Phi_interventions_tied_logical_unit_slots":tied,
        "Phi_interventions_regressed_logical_unit_slots":regressed,
        "original_GCC_executions":0,
        "original_Vitis_calls":0,"original_Vivado_calls":0,
        "source_code_generation_or_executable_validation_established":False,
        "hardware_scratch_or_bram_savings_established":False,
        "external_firstness_or_trained_RL_established":False,
        "evidence":"E0-bounded-2-4-fanout-source-value-lifetime-only",
        "limitations":[
            "All graphs are one MatMul producer, 2..4 MatMul readers, a fixed Add chain and Relu",
            "Only independent PE-group source issue with scalar producer/Add/Relu",
            "Unit-slot optimality is standard interval coloring for one fixed logical event sequence",
            "CONTRIBUTE witnesses are logical model events, not actual executed source arithmetic",
            "No GCC, native Vitis/Vivado, physical FPGA BRAM, bank port/clock or external prior-art parity",
        ],
    }
    if root is not None:
        root.mkdir(parents=True)
        with (root/"cases.csv").open("w",newline="") as fh:
            writer=csv.DictWriter(fh,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (root/"summary.json").write_text(
            json.dumps(summary,indent=2,sort_keys=True)+"\n")
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",help="NEW create-only E0 report folder, no GCC/vendor calls")
    a=p.parse_args(argv)
    print(json.dumps(run(output=a.output),indent=2,sort_keys=True))


if __name__=="__main__":
    main()
