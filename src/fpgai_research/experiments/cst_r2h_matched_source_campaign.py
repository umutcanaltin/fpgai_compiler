"""Prospective 36 matched fanout source-memory pairs; 72 actual GCC programs.

Both arms execute the SAME runtime-ordinal C++ source algorithm,
functional graph, input, reader mapping and typed event order. One
uses disjoint scratch slots; the other materializes E0-certified
per-element optimal unit-scratch slots. Read/commit address witnesses,
all intermediate numeric outputs and exact array counts are checked.

This is NOT hardware BRAM/port/Fmax evidence and is NOT a proof that
the source allocator is a new algorithm. No historical R2f/R2g GCC or
Vitis/Vivado experiment is overwritten.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import subprocess

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.r2h_allocated_fanout_cpp import (
    LAYOUTS,emit_allocated_fanout_cpp,source_layout,
)
from fpgai_research.experiments.cst_fanout_models import sample_fanout
from fpgai_research.experiments.cst_r2b_source_campaign import _input
from fpgai_research.experiments.cst_r2f_fanout_campaign import (
    FANOUTS,SHAPES,POLICY_NAMES,policy_widths,
)
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.validation.r2h_allocated_fanout_cpp import (
    verify_allocated_gcc,
)

SCHEMA="fpgai-research/cst-R2h-matched-36-fanout-source-72-GCC-e1/v1"


def run(*,output:str|Path)->dict:
    root=Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("R2h original source cohort strictly create-only")
    dirty=subprocess.run(["git","status","--porcelain","--untracked-files=no"],
        capture_output=True,text=True,check=True,timeout=5).stdout
    if dirty.strip():
        raise RuntimeError("Original R2h experiment requires clean tracked source")
    revision=subprocess.run(["git","rev-parse","HEAD"],capture_output=True,
        text=True,check=True,timeout=5).stdout.strip()
    if len(revision)!=40:
        raise ValueError("Original R2h requires full commit SHA")
    root.mkdir(parents=True)
    rows=[]
    for fanout in FANOUTS:
        for dims in SHAPES:
            m,k,n,q=dims
            model=sample_fanout(fanout,m=m,k=k,n=n,q=q)
            input_array=_input(m,k)
            for policy in POLICY_NAMES:
                case=f"case_{len(rows):03d}"
                folder=root/case
                folder.mkdir()
                base=ArchitectureRequest()
                arch=resolve(model,base,{
                    f"reader{i}":replace(base,pes=width)
                    for i,width in enumerate(policy_widths(fanout,policy))
                    if width!=1
                })
                arms={}
                for layout in LAYOUTS:
                    emission=emit_allocated_fanout_cpp(arch,layout=layout)
                    source_path=folder/(layout+"_source.cpp")
                    trace_path=folder/(layout+"_original_stdout.txt")
                    source_path.write_text(emission.cpp)
                    (folder/(layout+"_manifest.json")).write_text(
                        json.dumps(emission.manifest,indent=2,sort_keys=True)+"\n")
                    try:
                        receipt=verify_allocated_gcc(
                            arch,emission,runtime_input=input_array,
                            original_trace_output=trace_path)
                    except Exception as exc:
                        (folder/"FAILED.json").write_text(json.dumps({
                            "case":case,"layout":layout,
                            "status":"FAILED_ORIGINAL_R2h_SOURCE_EXECUTION",
                            "exception_type":type(exc).__name__,
                            "exception_message":str(exc)[:3000],
                            "original_source_cpp_sha256":
                                sha256(source_path.read_bytes()).hexdigest(),
                        },sort_keys=True,indent=2)+"\n")
                        raise
                    (folder/(layout+"_verified.json")).write_text(
                        json.dumps(receipt,indent=2,sort_keys=True)+"\n")
                    arms[layout]=receipt
                baseline=arms["disjoint"]
                optimized=arms["per_element_optimal"]
                if (baseline["original_runtime_source_event_sha256"]!=
                        optimized["original_runtime_source_event_sha256"]):
                    (folder/"FAILED.json").write_text(
                        '{"reason":"MATCHED_RUNTIME_EVENTS_DIFFER"}\n')
                    raise AssertionError("Matched R2h typed GCC event sequence differs")
                saved=(baseline["source_scratch_unit_slots"]-
                       optimized["source_scratch_unit_slots"])
                total_saved=(baseline["source_all_declared_mutable_array_elements"]-
                             optimized["source_all_declared_mutable_array_elements"])
                if saved!=total_saved:
                    (folder/"FAILED.json").write_text(
                        '{"reason":"SOURCE_ARRAY_COUNT_INCONSISTENT"}\n')
                    raise AssertionError("R2h source array accounting differs across arms")
                rows.append({
                    "case":case,"fanout":fanout,"shape":json.dumps(dims),
                    "policy":policy,"source_revision":revision,
                    "functional_sha256":baseline["functional_sha256"],
                    "architecture_sha256":baseline["architecture_sha256"],
                    "original_input_sha256":baseline["original_input_sha256"],
                    "original_runtime_source_event_sha256":
                        baseline["original_runtime_source_event_sha256"],
                    "baseline_source_sha256":baseline["source_cpp_sha256"],
                    "optimized_source_sha256":optimized["source_cpp_sha256"],
                    "baseline_original_stdout_sha256":
                        baseline["original_stdout_sha256"],
                    "optimized_original_stdout_sha256":
                        optimized["original_stdout_sha256"],
                    "baseline_scratch_unit_slots":baseline["source_scratch_unit_slots"],
                    "optimized_scratch_unit_slots":optimized["source_scratch_unit_slots"],
                    "source_scratch_unit_slots_saved":saved,
                    "baseline_all_declared_mutable_array_elements":
                        baseline["source_all_declared_mutable_array_elements"],
                    "optimized_all_declared_mutable_array_elements":
                        optimized["source_all_declared_mutable_array_elements"],
                    "all_declared_mutable_array_elements_saved":total_saved,
                    "baseline_address_witnesses":
                        baseline["original_actual_address_witness_count"],
                    "optimized_address_witnesses":
                        optimized["original_actual_address_witness_count"],
                    "original_numeric_and_event_passes":True,
                    "actual_HLS_or_BRAM_savings_measured":False,
                    "original_GCC_compilations":2,
                    "original_GCC_executions":2,
                    "original_Vitis_or_Vivado_calls":0,
                })
    if len(rows)!=len(FANOUTS)*len(SHAPES)*len(POLICY_NAMES):
        raise AssertionError("R2h intended original case denominator lost")
    phi_changed=phi_saved=phi_tied=phi_regressed=0
    for fanout in FANOUTS:
        for dims in SHAPES:
            subset=[r for r in rows if r["fanout"]==fanout
                    and json.loads(r["shape"])==list(dims)]
            scalar=next(r for r in subset if r["policy"]=="scalar")
            for row in subset:
                if row["policy"]=="scalar":
                    continue
                delta=(scalar["optimized_scratch_unit_slots"]-
                       row["optimized_scratch_unit_slots"])
                phi_changed+=delta!=0
                phi_saved+=delta>0
                phi_tied+=delta==0
                phi_regressed+=delta<0
    summary={
        "schema":SCHEMA,"source_revision":revision,
        "prespecified_graphs":len(FANOUTS)*len(SHAPES),
        "prespecified_Phi_policies_per_graph":len(POLICY_NAMES),
        "prespecified_original_matched_pairs":len(rows),
        "original_GCC_compilations":2*len(rows),
        "original_GCC_executions":2*len(rows),
        "original_all_matched_source_Phi_numeric_address_passes":len(rows),
        "source_scratch_reduced_matched_pairs":
            sum(r["source_scratch_unit_slots_saved"]>0 for r in rows),
        "source_scratch_tied_matched_pairs":
            sum(r["source_scratch_unit_slots_saved"]==0 for r in rows),
        "source_scratch_regressed_matched_pairs":
            sum(r["source_scratch_unit_slots_saved"]<0 for r in rows),
        "original_unique_Phi_interventions":len(FANOUTS)*len(SHAPES)*(
            len(POLICY_NAMES)-1),
        "original_Phi_changed_exact_unit_slots":phi_changed,
        "original_Phi_reduced_exact_unit_slots":phi_saved,
        "original_Phi_tied_exact_unit_slots":phi_tied,
        "original_Phi_regressed_exact_unit_slots":phi_regressed,
        "source_alternative_temporary_arrays_accounted":True,
        "actual_optimized_source_slot_materialization_tested":True,
        "native_Vitis_calls":0,"native_Vivado_calls":0,
        "physical_BRAM_or_clock_savings_established":False,
        "external_novelty_or_trained_RL_established":False,
        "limitations":[
            "Serial GCC C++ actual scalar-unit scratch arrays, not measured FPGA BRAM",
            "Same fixed E0 source-order algorithm and 2..4 MatMul fanout/left Add chain",
            "Source accumulators and counters counted; scalar locals excluded",
            "No bank/port, HLS cycle, bit-exact ap_fixed, clean-DRC routed parity or external compiler firstness",
        ],
    }
    with (root/"cases.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    (root/"summary.json").write_text(
        json.dumps(summary,indent=2,sort_keys=True)+"\n")
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    print(json.dumps(run(output=args.output),indent=2,sort_keys=True))


if __name__=="__main__":
    main()
