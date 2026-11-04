"""Original prospective R2b E1: two-reader Phi, actual GCC C++ execution.

Fixed schedule/functional graph/weights/input and each of two source
storage policies. An actual source READ/COMMIT trace, rather than a
manifest Phi hash, determines each element's last-use ordinal. The
independent NumPy oracle checks the numerical output. The COMPLETE
24-case denominator includes all original GCC sources and raw logs.

This is not Vitis HLS, physical PE parallelism, FPGA port scheduling,
physical QoR or research novelty over an external compiler. The prior
R2 fixed-serial study and its negative 0/18 evidence are untouched.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import numpy as np

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.branch_matmul import disjoint_branch_offsets
from fpgai_research.codegen.hls.r2b_interleaved_branch import emit_r2b_interleaved_cpp
from fpgai_research.experiments.cst_branch_models import sample_branch
from fpgai_research.experiments.cst_r2b_interleaved_schedule import SHAPES,POLICIES
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.validation.r2b_interleaved_cpp import verify_r2b_gcc

SCHEMA="fpgai-research/cst-prospective-original-interleaved-GCC-R2b-E1/v1"
LAYOUTS=("disjoint","last_reader_reuse")


def _input(m:int,k:int) -> np.ndarray:
    """Deterministic, signed, fixed input per graph; no training assets."""
    return np.asarray([((i*5+3)%17-8)/6.0 for i in range(m*k)],
                      dtype=np.float32).reshape(m,k)


def _digest(path:Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Original case file missing or symlinked")
    return sha256(path.read_bytes()).hexdigest()


def run(*,output:str|Path)->dict:
    dest=Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Original R2b source campaign is strictly create-only")
    tracked=subprocess.run(
        ["git","status","--porcelain","--untracked-files=no"],
        capture_output=True,text=True,check=True,timeout=5).stdout.strip()
    if tracked:
        raise RuntimeError("Original source experiment requires clean tracked checkout")
    revision=subprocess.run(["git","rev-parse","HEAD"],capture_output=True,
                            text=True,check=True,timeout=5).stdout.strip()
    if len(revision)!=40:
        raise RuntimeError("Full source SHA required for original GCC campaign")
    dest.mkdir(parents=True)
    cases=[]
    for shape in SHAPES:
        m,k,n,q=shape
        model=sample_branch(*shape)
        x=_input(m,k)
        base=ArchitectureRequest()
        for policy,phi0,phi1 in POLICIES:
            overrides={
                model.operators[i].name:replace(base,**options)
                for i,options in ((1,phi0),(2,phi1)) if options
            }
            arch=resolve(model,base,overrides)
            for layout in LAYOUTS:
                case_id=f"case_{len(cases):03d}"
                folder=dest/case_id
                folder.mkdir()
                offsets=disjoint_branch_offsets(
                    arch,reuse_after_last_read=layout=="last_reader_reuse")
                emission=emit_r2b_interleaved_cpp(arch,offsets=offsets)
                original_cpp=folder/"source.cpp"
                original_cpp.write_text(emission.cpp)
                original_meta={
                    "source_revision":revision,
                    "case":case_id,
                    "shape":shape,"policy":policy,"layout":layout,
                    "source_cpp_sha256":_digest(original_cpp),
                    "source_manifest":emission.manifest,
                    "input_float32_values":x.reshape(-1).astype(float).tolist(),
                    "input_float32_sha256":sha256(x.tobytes()).hexdigest(),
                    "original_case_preparation_status":"SOURCE_AND_INPUT_PINNED_BEFORE_GCC",
                }
                (folder/"original_case.json").write_text(
                    json.dumps(original_meta,sort_keys=True,indent=2)+"\n")
                try:
                    attested=verify_r2b_gcc(
                        arch,emission,offsets=offsets,
                        runtime_input=x,
                        original_trace_output=folder/"original_gcc_trace.txt")
                except Exception as exc:
                    # Never discard a failed original source artifact.
                    (folder/"FAILED.json").write_text(json.dumps({
                        "status":"FAILED_ORIGINAL_E1",
                        "exception_type":type(exc).__name__,
                        "reason":str(exc)[:2000],
                        "source_cpp_sha256":_digest(original_cpp),
                    },indent=2)+"\n")
                    raise
                (folder/"verified_e1.json").write_text(
                    json.dumps(attested,sort_keys=True,indent=2)+"\n")
                cases.append({
                    "case":case_id,
                    "shape":json.dumps(shape),
                    "policy":policy,"layout":layout,
                    "source_revision":revision,
                    "functional_sha256":attested["functional_sha256"],
                    "architecture_sha256":attested["architecture_sha256"],
                    "input_sha256":attested["runtime_input_sha256"],
                    "source_cpp_sha256":attested["source_cpp_sha256"],
                    "original_gcc_stdout_sha256":_digest(
                        folder/"original_gcc_trace.txt"),
                    "actual_source_event_sha256":attested["observed_event_sha256"],
                    "actual_source_h_last_use_sha256":attested[
                        "observed_h_last_use_sha256"],
                    "actual_source_h_last_use_indices":json.dumps(
                        attested["observed_h_last_use_indices"]),
                    "actual_source_h_final_read_only_sha256":attested[
                        "observed_h_final_read_rank_and_owner_sha256"],
                    "actual_source_h_final_read_only_positions":json.dumps(
                        attested["observed_h_final_read_rank_and_owner"]),
                    "changed_h_last_read_elements_vs_scalar":0,
                    "changed_h_final_read_only_rank_or_owner_vs_scalar":0,
                    "changed_h_final_reader_identity_vs_scalar":0,
                    "original_actual_source_h_reads":attested["original_source_h_reads"],
                    "original_source_intermediate_events":attested[
                        "original_source_intermediate_events"],
                    "original_numeric_max_abs_error":attested[
                        "max_absolute_numeric_error"],
                    "source_scratch_slots":emission.manifest["scratch_slots"],
                    "source_executed_e1":True,
                    "original_gcc_compilations":1,
                    "original_gcc_executions":1,
                    "original_native_vitis_calls":0,
                    "original_native_vivado_calls":0,
                })
    # A causal contrast is admitted only against the SAME input/graph/
    # layout/source generator and fixed reader issue algorithm.
    lookup={(c["shape"],c["layout"],c["policy"]):c for c in cases}
    for c in cases:
        baseline=lookup[(c["shape"],c["layout"],"scalar")]
        if (c["functional_sha256"]!=baseline["functional_sha256"] or
                c["input_sha256"]!=baseline["input_sha256"]):
            raise AssertionError("Original source causal graph/input were changed")
        before=json.loads(baseline["actual_source_h_last_use_indices"])
        after=json.loads(c["actual_source_h_last_use_indices"])
        if [x[0] for x in before]!=[x[0] for x in after]:
            raise AssertionError("Observed actual H element population changed")
        c["changed_h_last_read_elements_vs_scalar"]=sum(
            a[1]!=b[1] for a,b in zip(before,after))
        # A changed global event ordinal can be a bookkeeping
        # artifact: require an actual H-only READ rank or final
        # reader identity difference for the stronger claim.
        earlier=json.loads(baseline["actual_source_h_final_read_only_positions"])
        later=json.loads(c["actual_source_h_final_read_only_positions"])
        if [x[0] for x in earlier]!=[x[0] for x in later]:
            raise AssertionError("Original last H READ element identity changed")
        c["changed_h_final_read_only_rank_or_owner_vs_scalar"]=sum(
            a[1:]!=b[1:] for a,b in zip(earlier,later))
        c["changed_h_final_reader_identity_vs_scalar"]=sum(
            a[2]!=b[2] for a,b in zip(earlier,later))
        other_layout=lookup[(c["shape"],
                             "disjoint" if c["layout"]=="last_reader_reuse"
                             else "last_reader_reuse",c["policy"])]
        if (c["actual_source_event_sha256"]!=other_layout[
                "actual_source_event_sha256"] or
                c["actual_source_h_last_use_sha256"]!=other_layout[
                    "actual_source_h_last_use_sha256"] or
                c["actual_source_h_final_read_only_sha256"]!=other_layout[
                    "actual_source_h_final_read_only_sha256"]):
            raise AssertionError("Scratch policy altered executed event semantics")
        if c["layout"]=="disjoint" and not (
                c["source_scratch_slots"]>other_layout["source_scratch_slots"]):
            raise AssertionError("CST expected source-scratch capacity control absent")
    nonbaseline=[c for c in cases if c["policy"]!="scalar"]
    summary={
        "schema":SCHEMA,
        "source_revision":revision,
        "graphs":len(SHAPES),
        "mapping_policies":len(POLICIES),
        "source_layout_policies":len(LAYOUTS),
        "prespecified_cases":len(cases),
        "original_gcc_compilations":len(cases),
        "original_gcc_executions":len(cases),
        "original_numeric_and_event_passes":sum(c["source_executed_e1"] for c in cases),
        "original_controlled_phi_contrasts":len(nonbaseline),
        "changed_original_executed_h_last_read_contrasts":sum(
            c["changed_h_last_read_elements_vs_scalar"]>0 for c in nonbaseline),
        "unchanged_original_executed_h_last_read_contrasts":sum(
            c["changed_h_last_read_elements_vs_scalar"]==0 for c in nonbaseline),
        "source_read_only_rank_or_owner_changed_contrasts":sum(
            c["changed_h_final_read_only_rank_or_owner_vs_scalar"]>0
            for c in nonbaseline),
        "source_read_only_rank_or_owner_unchanged_contrasts":sum(
            c["changed_h_final_read_only_rank_or_owner_vs_scalar"]==0
            for c in nonbaseline),
        "source_final_reader_identity_changed_contrasts":sum(
            c["changed_h_final_reader_identity_vs_scalar"]>0
            for c in nonbaseline),
        "source_storage_pairs":len(cases)//2,
        "source_scratch_saving_layout_pairs":len(cases)//2,
        "mapping_conditioned_source_capacity_difference_established":False,
        "per_cycle_hardware_execution_difference_established":False,
        "source_executed_lifetime_change_established":any(
            c["changed_h_final_read_only_rank_or_owner_vs_scalar"]>0
            for c in nonbaseline),
        "global_event_index_only_effects_do_not_qualify":True,
        "external_compiler_firstness_established":False,
        "original_vitis_calls":0,"original_vivado_calls":0,
        "evidence":"E1-original-GCC-executed-interleaved-source-and-independent-NumPy",
        "limitations":[
            "E1 C++ serialized source order is not FPGA cycles, real PE hardware or RAM-port proof",
            "Reader PE issue-rule is a new prospective fixed algorithm, not original serial scheduler",
            "All 24 cases are three fixed five-op branch graphs, not an arbitrary DAG",
            "Source storage disjoint/shared comparison is not Phi-dependent physical capacity",
            "Global event-index changes alone do NOT qualify; use H-only final READ rank/owner",
            "No fixed-point HLS, Vivado, trained RL or externally reproduced prior-art parity",
            "A controlled source-last-use effect is not proof of new algorithmic firstness",
        ],
    }
    with (dest/"cases.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(cases[0]))
        writer.writeheader();writer.writerows(cases)
    (dest/"summary.json").write_text(
        json.dumps(summary,sort_keys=True,indent=2)+"\n")
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    print(json.dumps(run(output=args.output),indent=2,sort_keys=True))


if __name__=="__main__":
    main()
