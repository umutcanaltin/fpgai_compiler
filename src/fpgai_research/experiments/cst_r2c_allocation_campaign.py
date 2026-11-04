"""Prospective original CST R2c real-source scratch allocation experiment.

Twelve FIXED paired original GCC cases: 3 graph shapes × 4 reader-Phi
policies, with a source-pinned best existing whole-tensor last-reader
reuse baseline versus exact per-value lifetime-based allocation.
Both C++ implementations run numerical and original source event/slot
audits against the same fixed graph, weights and input.

Scientific outcomes: SOURCE scratch/total explicit mutable temporary
counts only; not physical BRAM, RAM-port legality, II, frequency or QoR.
All failing/original cases remain immutable; zero native vendor calls.
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
from fpgai_research.codegen.hls.branch_matmul import disjoint_branch_offsets
from fpgai_research.codegen.hls.r2b_interleaved_branch import emit_r2b_interleaved_cpp
from fpgai_research.codegen.hls.r2c_allocated_branch import emit_r2c_allocated_cpp
from fpgai_research.experiments.cst_branch_models import sample_branch
from fpgai_research.experiments.cst_r2b_source_campaign import SHAPES,POLICIES,_input
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.validation.r2b_interleaved_cpp import verify_r2b_gcc
from fpgai_research.validation.r2c_allocated_cpp import verify_r2c_gcc

SCHEMA="fpgai-research/cst-R2c-original-matched-GCC-source-allocation/v1"


def _sha(path:Path)->str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Missing original source campaign file")
    return sha256(path.read_bytes()).hexdigest()


def run(*,output:str|Path)->dict:
    root=Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("Original R2c source campaign must be create-only")
    clean=subprocess.run(
        ["git","status","--porcelain","--untracked-files=no"],
        check=True,capture_output=True,text=True,timeout=5)
    if clean.stdout.strip():
        raise RuntimeError("Original R2c source evidence requires a clean tracked checkout")
    revision=subprocess.run(["git","rev-parse","HEAD"],check=True,
                            capture_output=True,text=True,timeout=5).stdout.strip()
    if len(revision)!=40:
        raise RuntimeError("Original full source revision required")
    root.mkdir(parents=True)
    rows=[]
    for shape in SHAPES:
        m,k,n,q=shape
        model=sample_branch(*shape)
        x=_input(m,k)
        base=ArchitectureRequest()
        for policy,phi0,phi1 in POLICIES:
            case=f"case_{len(rows):03d}"
            folder=root/case
            folder.mkdir()
            overrides={
                model.operators[stage].name:replace(base,**options)
                for stage,options in ((1,phi0),(2,phi1))
                if options
            }
            arch=resolve(model,base,overrides)
            old_offsets=disjoint_branch_offsets(
                arch,reuse_after_last_read=True)
            old=emit_r2b_interleaved_cpp(arch,offsets=old_offsets)
            new=emit_r2c_allocated_cpp(arch)
            (folder/"baseline_source.cpp").write_text(old.cpp)
            (folder/"allocated_source.cpp").write_text(new.cpp)
            (folder/"case_inputs.json").write_text(json.dumps({
                "case":case,"shape":shape,"policy":policy,
                "source_revision":revision,
                "numeric_input_float32":x.reshape(-1).astype(float).tolist(),
                "numeric_input_sha256":sha256(x.tobytes()).hexdigest(),
                "baseline_source_sha256":_sha(folder/"baseline_source.cpp"),
                "allocated_source_sha256":_sha(folder/"allocated_source.cpp"),
                "baseline_manifest":old.manifest,
                "allocated_manifest":new.manifest,
                "status_before_original_GCC":"SOURCE_AND_NUMERIC_INPUTS_PINNED",
            },sort_keys=True,indent=2)+"\n")
            try:
                old_check=verify_r2b_gcc(
                    arch,old,offsets=old_offsets,runtime_input=x,
                    original_trace_output=folder/"baseline_original_stdout.txt")
                new_check=verify_r2c_gcc(
                    arch,new,runtime_input=x,
                    original_trace_output=folder/"allocated_original_stdout.txt")
                if (old_check["status"]!="E1_ORIGINAL_GCC_SOURCE_EVENT_AND_NUMERIC_VERIFIED" or
                        new_check["status"]!="ORIGINAL_GCC_SOURCE_NUMERIC_AND_SCRATCH_SAFETY_VERIFIED_E1" or
                        old_check["observed_event_sha256"]!=new_check["original_source_event_sha256"] or
                        old_check["functional_sha256"]!=new_check["functional_sha256"] or
                        old_check["architecture_sha256"]!=new_check["architecture_sha256"]):
                    raise AssertionError("Original source-matched schedule/numeric pair differs")
                old_slots=old.manifest["scratch_slots"]
                new_slots=new.manifest["source_scratch_unit_slots"]
                if not 0<new_slots<=old_slots:
                    raise AssertionError("Optimal source allocation exceeds existing reuse baseline")
            except Exception as exc:
                (folder/"FAILED.json").write_text(json.dumps({
                    "status":"FAILED_ORIGINAL_R2C_SOURCE_CASE",
                    "type":type(exc).__name__,
                    "error":str(exc)[:2500],
                    "original_allocated_source_sha256":_sha(folder/"allocated_source.cpp"),
                    "original_baseline_source_sha256":_sha(folder/"baseline_source.cpp"),
                },sort_keys=True,indent=2)+"\n")
                raise
            (folder/"baseline_verified_e1.json").write_text(
                json.dumps(old_check,sort_keys=True,indent=2)+"\n")
            (folder/"allocated_verified_e1.json").write_text(
                json.dumps(new_check,sort_keys=True,indent=2)+"\n")
            rows.append({
                "case":case,
                "shape":json.dumps(shape),
                "policy":policy,"source_revision":revision,
                "functional_sha256":new_check["functional_sha256"],
                "architecture_sha256":new_check["architecture_sha256"],
                "numeric_input_sha256":new_check["original_input_sha256"],
                "baseline_source_sha256":_sha(folder/"baseline_source.cpp"),
                "allocated_source_sha256":_sha(folder/"allocated_source.cpp"),
                "baseline_trace_sha256":_sha(folder/"baseline_original_stdout.txt"),
                "allocated_trace_sha256":_sha(folder/"allocated_original_stdout.txt"),
                "actual_original_source_event_sha256":new_check["original_source_event_sha256"],
                "allocated_value_address_sha256":new.manifest["source_allocation_sha256"],
                "baseline_whole_tensor_scratch_slots":old_slots,
                "allocated_per_element_scratch_slots":new_slots,
                "independent_minimum_peak_live_slots":new.manifest[
                    "source_peak_live_intermediates"],
                "source_scratch_slots_saved":old_slots-new_slots,
                "baseline_declared_mutable_array_elements":old_slots+4*m*q,
                "allocated_declared_mutable_array_elements":new_slots+4*m*q,
                "declared_mutable_array_elements_saved":old_slots-new_slots,
                "source_numeric_matches_independent_numpy":True,
                "source_event_sequence_unchanged_between_layouts":True,
                "dynamic_memory_slot_witness_verified":True,
                "minimum_unit_slot_allocation_e0":new.manifest[
                    "source_allocation_is_optimal_unit_slots"],
                "original_GCC_compilations":2,
                "original_GCC_executions":2,
                "original_native_tool_calls":0,
            })
    for shape in SHAPES:
        subset=[r for r in rows if json.loads(r["shape"])==list(shape)]
        if len(subset)!=len(POLICIES):
            raise AssertionError("Missing fixed shape/mapping source cases")
    changed_capacity=0
    changed_assignment=0
    for shape in SHAPES:
        subset=[r for r in rows if json.loads(r["shape"])==list(shape)]
        scalar=next(r for r in subset if r["policy"]=="scalar")
        for row in subset:
            if row["policy"]=="scalar":
                continue
            changed_capacity+=(
                row["allocated_per_element_scratch_slots"]!=
                scalar["allocated_per_element_scratch_slots"])
            changed_assignment+=(
                row["allocated_value_address_sha256"]!=
                scalar["allocated_value_address_sha256"])
    summary={
        "schema":SCHEMA,
        "source_revision":revision,
        "prespecified_graphs":len(SHAPES),
        "source_Phi_policies_per_graph":len(POLICIES),
        "prespecified_original_matched_pairs":len(rows),
        "original_GCC_compilations":2*len(rows),
        "original_GCC_executions":2*len(rows),
        "original_numeric_and_typed_trace_pair_passes":len(rows),
        "original_slot_witness_pair_passes":len(rows),
        "source_scratch_unit_slots_saved_in_pairs":sum(
            r["source_scratch_slots_saved"]>0 for r in rows),
        "source_scratch_unit_slots_tied_in_pairs":sum(
            r["source_scratch_slots_saved"]==0 for r in rows),
        "source_scratch_unit_slots_regressed_in_pairs":sum(
            r["source_scratch_slots_saved"]<0 for r in rows),
        "prespecified_unique_Phi_counterfactuals":len(SHAPES)*(len(POLICIES)-1),
        "mapping_interventions_changed_optimal_scratch_capacity":changed_capacity,
        "mapping_interventions_changed_value_address_assignment":changed_assignment,
        "min_unit_slots_independently_witnessed_by_peak_live_counts":len(rows),
        "actual_native_HLS_BRAM_savings_established":False,
        "source_declared_mutable_array_elements_counted":True,
        "source_memory_optimality_scope":"fixed total source order, identical scalar unit slots, no physical bank-port constraints",
        "original_Vitis_calls":0,"original_Vivado_calls":0,
        "external_compiler_firstness_established":False,
        "trained_RL_advantage_established":False,
        "evidence":"E0-finite-interval-and-peak-lower-bound-plus-E1-real-GCC-dynamic-source-addresses-and-NumPy",
        "limitations":[
            "Results quantify C++ static scratch allocation, not physical FPGA memory cells",
            "Declared mutable arrays include separate accumulators/Add buffers; scalar C++ temporaries excluded from count",
            "All source cases use the same 3 fixed small MatMul diamonds with one input per shape",
            "Phi-dependent capacity change may be zero despite changed address assignment; report exactly",
            "No Vitis CSim/CSynth or Vivado and no independent exact published-artifact parity",
        ],
    }
    with (root/"cases.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    (root/"summary.json").write_text(
        json.dumps(summary,sort_keys=True,indent=2)+"\n")
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    print(json.dumps(run(output=args.output),indent=2,sort_keys=True))


if __name__=="__main__":
    main()
