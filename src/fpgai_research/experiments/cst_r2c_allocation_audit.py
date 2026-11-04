"""Independent read-only original R2c source/slot/NumPy audit.

Re-extract actual slot witnesses and typed events from 12 saved original
source-matched GCC pairs; independently reconstruct the model, Phi,
E0 exact interval packing and numerical reference; check all raw
case/summary counters with no GCC recompilations or vendor tool calls.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict,replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.branch_matmul import disjoint_branch_offsets
from fpgai_research.experiments.cst_branch_models import sample_branch
from fpgai_research.experiments.cst_r2b_source_campaign import SHAPES,POLICIES,_input
from fpgai_research.experiments.cst_r2c_allocation_campaign import SCHEMA
from fpgai_research.model_ir.branch_matmul_contract import _hash
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.element_scratch_allocation import (
    allocate_minimum_slots,verify_allocation,
)
from fpgai_research.model_ir.interleaved_branch_schedule import interleaved_branch_events
from fpgai_research.model_ir.semantics import (
    functional_fingerprint,architecture_fingerprint,
)
from fpgai_research.validation.r2b_interleaved_cpp import _decode_trace
from fpgai_research.validation.r2c_allocated_cpp import decode_allocated_stdout
from fpgai_research.validation.reference import evaluate

AUDIT_SCHEMA="fpgai-research/cst-R2c-original-readonly-pair-audit/v1"


def _read(path:Path)->bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size>8*1024*1024:
        raise ValueError("Missing/oversized/symlinked original R2c source file: "+str(path))
    return path.read_bytes()


def audit_original(study:str|Path)->dict:
    root=Path(study)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Expected original R2c source study directory")
    summary=json.loads(_read(root/"summary.json"))
    rows=list(csv.DictReader(_read(root/"cases.csv").decode().splitlines()))
    if (summary.get("schema")!=SCHEMA or
            summary.get("prespecified_original_matched_pairs")!=12
            or len(rows)!=12 or
            summary.get("original_GCC_executions")!=24 or
            summary.get("original_GCC_compilations")!=24):
        raise ValueError("Original R2c source population/denominator mismatch")
    baseline_sha=set()
    results=[]
    for i,row in enumerate(rows):
        folder=root/f"case_{i:03d}"
        if (row.get("case")!=f"case_{i:03d}" or
                folder.is_symlink() or not folder.is_dir() or
                (folder/"FAILED.json").exists()):
            raise ValueError("Missing, reordered or failed original R2c case")
        case=json.loads(_read(folder/"case_inputs.json"))
        old_proof=json.loads(_read(folder/"baseline_verified_e1.json"))
        new_proof=json.loads(_read(folder/"allocated_verified_e1.json"))
        original_baseline=_read(folder/"baseline_source.cpp")
        original_allocated=_read(folder/"allocated_source.cpp")
        baseline_raw=_read(folder/"baseline_original_stdout.txt")
        allocated_raw=_read(folder/"allocated_original_stdout.txt")
        shape=tuple(json.loads(row["shape"]))
        expected_shape=SHAPES[i//len(POLICIES)]
        policy,phi0,phi1=POLICIES[i%len(POLICIES)]
        if (shape!=expected_shape or
                row["policy"]!=policy or
                tuple(case["shape"])!=shape or
                case["policy"]!=policy or
                case["case"]!=row["case"] or
                case["source_revision"]!=summary["source_revision"] or
                row["source_revision"]!=summary["source_revision"]):
            raise ValueError("Original R2c model/Phi/source identity changed")
        m,k,n,q=shape
        model=sample_branch(*shape)
        base=ArchitectureRequest()
        overrides={
            model.operators[stage].name:replace(base,**options)
            for stage,options in ((1,phi0),(2,phi1)) if options
        }
        arch=resolve(model,base,overrides)
        functional=functional_fingerprint(model)
        arch_sha=architecture_fingerprint(arch)
        expected_input=_input(m,k)
        saved_input=np.asarray(case["numeric_input_float32"],dtype=np.float32)
        if (saved_input.size!=m*k or
                not np.array_equal(saved_input.reshape((m,k)),expected_input) or
                sha256(saved_input.tobytes()).hexdigest()!=case["numeric_input_sha256"] or
                row["numeric_input_sha256"]!=case["numeric_input_sha256"]):
            raise ValueError("Original numerical input source identity changed")
        expected_output=np.asarray(evaluate(
            model,{model.inputs[0]:saved_input.reshape((m,k))})[
                model.outputs[0]],dtype=np.float32).reshape(-1).astype(np.float64)
        old_source_hash=sha256(original_baseline).hexdigest()
        new_source_hash=sha256(original_allocated).hexdigest()
        old_trace_hash=sha256(baseline_raw).hexdigest()
        new_trace_hash=sha256(allocated_raw).hexdigest()
        if (old_source_hash!=case["baseline_source_sha256"] or
                new_source_hash!=case["allocated_source_sha256"] or
                old_source_hash!=row["baseline_source_sha256"] or
                new_source_hash!=row["allocated_source_sha256"] or
                old_trace_hash!=row["baseline_trace_sha256"] or
                new_trace_hash!=row["allocated_trace_sha256"] or
                old_proof.get("raw_original_gcc_stdout_sha256")!=old_trace_hash or
                new_proof.get("original_GCC_stdout_sha256")!=new_trace_hash):
            raise ValueError("Original R2c source/trace SHA-256 provenance mismatch")
        old_manifest=case["baseline_manifest"]
        new_manifest=case["allocated_manifest"]
        if (old_manifest["cpp_sha256"]!=old_source_hash or
                new_manifest["cpp_sha256"]!=new_source_hash or
                old_manifest["functional_sha256"]!=functional or
                new_manifest["functional_sha256"]!=functional or
                old_manifest["architecture_sha256"]!=arch_sha or
                new_manifest["architecture_sha256"]!=arch_sha or
                row["functional_sha256"]!=functional or
                row["architecture_sha256"]!=arch_sha or
                old_proof["functional_sha256"]!=functional or
                new_proof["functional_sha256"]!=functional or
                new_proof["architecture_sha256"]!=arch_sha):
            raise ValueError("Original R2c compiler model/source receipt mismatch")
        plan=interleaved_branch_events(arch,max_events=4096)
        e0=allocate_minimum_slots(plan.events,max_events=4096)
        old_events,old_output=_decode_trace(baseline_raw.decode(),n_outputs=m*q,
                                            max_events=4096)
        new_events,slots,new_output=decode_allocated_stdout(
            allocated_raw.decode(),n_outputs=m*q,max_events=4096)
        if (old_events!=plan.events or new_events!=plan.events or
                not np.allclose(old_output,expected_output,rtol=1e-4,atol=1e-5) or
                not np.allclose(new_output,expected_output,rtol=1e-4,atol=1e-5) or
                not np.allclose(new_output,old_output,rtol=1e-4,atol=1e-5)):
            raise ValueError("Original R2c read-order or independently observed numeric mismatch")
        if (new_proof["original_source_event_sha256"]!=_hash([
                asdict(ev) for ev in new_events]) or
                row["actual_original_source_event_sha256"]!=
                    new_proof["original_source_event_sha256"]):
            raise ValueError("Original R2c actual source event SHA mismatch")
        assigned={
            (str(tensor),int(index)):int(slot)
            for tensor,index,slot in new_manifest["source_value_slots"]
        }
        if len(assigned)!=len(new_manifest["source_value_slots"]):
            raise ValueError("Duplicate original value/slot identity")
        total=new_manifest["source_scratch_unit_slots"]
        check=verify_allocation(new_events,assigned,unit_slots=total)
        if (check["peak_simultaneous_live_values"]!=e0.unit_slots or
                total!=e0.unit_slots or
                new_manifest["source_minimum_unit_slots"]!=e0.unit_slots or
                not new_manifest["source_allocation_is_optimal_unit_slots"]):
            raise ValueError("Original R2c minimal per-element source allocation unqualified")
        for event,witness in zip(new_events,slots):
            if witness!=(
                    event.kind,event.tensor,event.element,
                    assigned[(event.tensor,event.element)]):
                raise ValueError("Runtime GCC source slot does not match certified address")
        allocation_hash=sha256(json.dumps(
            tuple(sorted((name,idx,slot)
                for (name,idx),slot in assigned.items())),
            separators=(",",":"),allow_nan=False).encode()).hexdigest()
        if (allocation_hash!=new_manifest["source_allocation_sha256"] or
                row["allocated_value_address_sha256"]!=allocation_hash):
            raise ValueError("Original per-value source allocation digest mismatch")
        old_offsets=disjoint_branch_offsets(
            arch,reuse_after_last_read=True)
        if old_manifest.get("scratch_offsets")!=dict(sorted(old_offsets.items())):
            raise ValueError("Original baseline reused tensor offsets mismatch")
        old_slots=old_manifest["scratch_slots"]
        if (row["baseline_whole_tensor_scratch_slots"]!=str(old_slots) or
                row["allocated_per_element_scratch_slots"]!=str(total) or
                row["independent_minimum_peak_live_slots"]!=str(e0.unit_slots) or
                row["source_scratch_slots_saved"]!=str(old_slots-total) or
                row["baseline_declared_mutable_array_elements"]!=str(old_slots+4*m*q) or
                row["allocated_declared_mutable_array_elements"]!=str(total+4*m*q) or
                row["declared_mutable_array_elements_saved"]!=str(old_slots-total)):
            raise ValueError("Original R2c source scratch and declared array accounting mismatch")
        if (old_proof.get("original_gcc_compilations")!=1 or
                new_proof.get("original_GCC_compilations")!=1 or
                old_proof.get("original_gcc_executions")!=1 or
                new_proof.get("original_GCC_executions")!=1):
            raise ValueError("Original R2c paired source execution receipts incomplete")
        results.append({
            "case":row["case"],"shape":list(shape),"policy":policy,
            "old_scratch_unit_slots":old_slots,
            "new_scratch_unit_slots":total,
            "source_scratch_saved":old_slots-total,
            "old_explicit_temporary_slots":old_slots+4*m*q,
            "new_explicit_temporary_slots":total+4*m*q,
            "original_C++_numeric_qualified":True,
            "dynamic_GCC_source_address_qualified":True,
            "fixed_event_order_exact_optimum_qualified":True,
            "value_slot_address_sha256":allocation_hash,
        })
        baseline_sha.add(arch_sha)
    if len(results)!=12 or len(baseline_sha)!=12:
        raise ValueError("Original R2c source case identity/denominator incomplete")
    saved=sum(r["source_scratch_saved"]>0 for r in results)
    tied=sum(r["source_scratch_saved"]==0 for r in results)
    regress=sum(r["source_scratch_saved"]<0 for r in results)
    changed_capacity=0
    changed_assignment=0
    for shape in SHAPES:
        candidates=[r for r in results if tuple(r["shape"])==shape]
        scalar=next(r for r in candidates if r["policy"]=="scalar")
        for row in candidates:
            if row["policy"]=="scalar":
                continue
            changed_capacity+=row["new_scratch_unit_slots"]!=scalar[
                "new_scratch_unit_slots"]
            changed_assignment+=row["value_slot_address_sha256"]!=scalar[
                "value_slot_address_sha256"]
    if (summary["source_scratch_unit_slots_saved_in_pairs"]!=saved or
            summary["source_scratch_unit_slots_tied_in_pairs"]!=tied or
            summary["source_scratch_unit_slots_regressed_in_pairs"]!=regress or
            summary["mapping_interventions_changed_optimal_scratch_capacity"]!=changed_capacity or
            summary["mapping_interventions_changed_value_address_assignment"]!=changed_assignment or
            summary["original_numeric_and_typed_trace_pair_passes"]!=12 or
            summary["original_slot_witness_pair_passes"]!=12):
        raise ValueError("Original R2c summary conflicts with independently audited cases")
    return {
        "schema":AUDIT_SCHEMA,
        "source_revision":summary["source_revision"],
        "original_source_matched_pairs_readonly_qualified":len(results),
        "original_GCC_case_sources_readonly_qualified":2*len(results),
        "original_GCC_executions_this_audit":0,
        "original_Vitis_or_Vivado_calls_this_audit":0,
        "original_source_scratch_saving_pairs":saved,
        "original_source_scratch_tied_pairs":tied,
        "original_source_scratch_regression_pairs":regress,
        "original_unique_Phi_interventions":9,
        "original_mapping_changed_optimal_scratch_capacity":changed_capacity,
        "original_mapping_changed_exact_value_address_assignment":changed_assignment,
        "all_executed_source_address_witnesses_verified":True,
        "all_original_outputs_independently_recomputed":True,
        "all_original_exact_source_minimality_bounds_verified":True,
        "all_five_declared_mutable_source_arrays_included":True,
        "hardware_BRAM_or_clock_performance_qualified":False,
        "external_compiler_firstness_qualified":False,
        "original_pairs":results,
        "limitations":[
            "CPU source temporary sizes are not synthesized FPGA BRAM resource counts",
            "Exact optimality is only for the fixed serialized event order and unit scalar values",
            "All five declared mutable C++ array elements counted; scalar locals excluded; no inferred RAM-port legality",
            "No external published peer/compiler or native HLS was executed by this audit",
        ],
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study",required=True)
    p.add_argument("--output",help="Optional new create-only audited summary directory")
    args=p.parse_args(argv)
    destination=Path(args.output) if args.output else None
    if destination is not None and (destination.exists() or destination.is_symlink()):
        raise FileExistsError("Read-only R2c audit output must be create-only")
    result=audit_original(args.study)
    if destination is not None:
        destination.mkdir(parents=True)
        (destination/"r2c_audit.json").write_text(json.dumps(
            result,sort_keys=True,indent=2)+"\n")
    print(json.dumps(result,sort_keys=True,indent=2))


if __name__=="__main__":
    main()
