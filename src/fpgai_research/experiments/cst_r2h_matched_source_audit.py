"""R2h read-only independent paired GCC source/trace/alias/numeric audit.

No source GCC recompilation, no Vitis/Vivado calls. Reconstruct each
(graph,Phi) independently; validate 72 saved actual trace streams,
source/input/manifest receipts, complete address witnesses, exact
source-value lifetimes and all intermediates numerically against NumPy.
Do NOT infer FPGA BRAM or hardware-port savings from C++ scalar slots.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.r2h_allocated_fanout_cpp import (
    LAYOUTS,SCHEMA as EMIT_SCHEMA,source_layout,
)
from fpgai_research.experiments.cst_fanout_models import sample_fanout
from fpgai_research.experiments.cst_r2b_source_campaign import _input
from fpgai_research.experiments.cst_r2f_fanout_campaign import (
    FANOUTS,SHAPES,POLICY_NAMES,policy_widths,
    project_source_value_events,
)
from fpgai_research.experiments.cst_r2h_matched_source_campaign import SCHEMA
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.element_scratch_allocation import verify_allocation
from fpgai_research.model_ir.fanout_contract import (
    verify_fanout_events,verify_fanout_issue,disjoint_fanout_offsets,
)
from fpgai_research.model_ir.semantics import (
    functional_fingerprint,architecture_fingerprint,
)
from fpgai_research.validation.r2h_allocated_fanout_cpp import decode_address_witnesses
from fpgai_research.validation.reference import evaluate

AUDIT_SCHEMA="fpgai-research/cst-R2h-matched-source-readonly-audit/v1"


def _read(path:Path)->bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size>8*1024*1024:
        raise ValueError("R2h original source artifact missing/symlinked/oversized")
    return path.read_bytes()


def audit_original(study:str|Path)->dict:
    root=Path(study)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("R2h original study folder missing")
    summary=json.loads(_read(root/"summary.json"))
    rows=list(csv.DictReader(_read(root/"cases.csv").decode().splitlines()))
    expected=len(FANOUTS)*len(SHAPES)*len(POLICY_NAMES)
    if (summary.get("schema")!=SCHEMA or len(rows)!=expected or
            summary.get("prespecified_original_matched_pairs")!=expected
            or summary.get("original_GCC_compilations")!=2*expected or
            summary.get("original_GCC_executions")!=2*expected):
        raise ValueError("R2h incomplete or forged original denominator")
    results=[]
    for j,row in enumerate(rows):
        fanout=FANOUTS[j//(len(SHAPES)*len(POLICY_NAMES))]
        within=j%(len(SHAPES)*len(POLICY_NAMES))
        dims=SHAPES[within//len(POLICY_NAMES)]
        policy=POLICY_NAMES[within%len(POLICY_NAMES)]
        case=f"case_{j:03d}"
        folder=root/case
        if (row.get("case")!=case or int(row["fanout"])!=fanout or
                tuple(json.loads(row["shape"]))!=dims or row["policy"]!=policy or
                row["source_revision"]!=summary["source_revision"] or
                folder.is_symlink() or not folder.is_dir() or
                (folder/"FAILED.json").exists()):
            raise ValueError("R2h failed, reordered or missing original pair")
        m,k,n,q=dims
        model=sample_fanout(fanout,m=m,k=k,n=n,q=q)
        base=ArchitectureRequest()
        arch=resolve(model,base,{
            f"reader{i}":replace(base,pes=width)
            for i,width in enumerate(policy_widths(fanout,policy))
            if width!=1
        })
        x=_input(m,k)
        fx=functional_fingerprint(model)
        ax=architecture_fingerprint(arch)
        reference=evaluate(model,{model.inputs[0]:x},return_all=True)
        snapshots={}
        for layout in LAYOUTS:
            manifest=json.loads(_read(folder/(layout+"_manifest.json")))
            receipt=json.loads(_read(folder/(layout+"_verified.json")))
            source=_read(folder/(layout+"_source.cpp"))
            raw=_read(folder/(layout+"_original_stdout.txt"))
            spec,assigned,slots,optimum,_=source_layout(arch,layout=layout)
            src_sha=sha256(source).hexdigest()
            trace_sha=sha256(raw).hexdigest()
            if (manifest.get("schema")!=EMIT_SCHEMA or
                    manifest.get("source_layout")!=layout or
                    manifest.get("source_cpp_sha256")!=src_sha or
                    receipt.get("source_cpp_sha256")!=src_sha or
                    row[("baseline" if layout=="disjoint" else "optimized")+
                        "_source_sha256"]!=src_sha or
                    row[("baseline" if layout=="disjoint" else "optimized")+
                        "_original_stdout_sha256"]!=trace_sha or
                    receipt.get("original_stdout_sha256")!=trace_sha or
                    manifest.get("functional_sha256")!=fx or
                    manifest.get("architecture_sha256")!=ax or
                    receipt.get("functional_sha256")!=fx or
                    receipt.get("architecture_sha256")!=ax or
                    manifest.get("source_scratch_array_elements")!=slots or
                    tuple(map(tuple,manifest.get("source_slot_map",())))!=tuple(
                        sorted((tensor,i,v) for (tensor,i),v in assigned.items()))):
                raise ValueError("R2h original source/trace/slot lineage SHA mismatch")
            events,values,outputs,address_count=decode_address_witnesses(
                raw.decode(),slot_map=assigned,capacity=slots)
            safety=verify_fanout_events(
                arch,disjoint_fanout_offsets(spec),events)
            issue=verify_fanout_issue(arch,events)
            if safety["status"]!="VERIFIED_E0" or issue["status"]!="VERIFIED_E0":
                raise ValueError("R2h original source Phi/reduction safety mismatch")
            verify_allocation(
                project_source_value_events(events),assigned,unit_slots=slots)
            numerical_max=0.
            expected_values=sum(np.asarray(reference[t]).size
                                for t in spec.intermediates)
            if len(values)!=expected_values or set(outputs)!=set(range(m*q)):
                raise ValueError("R2h original intermediate/output population mismatch")
            for tensor in spec.intermediates:
                ref=np.asarray(reference[tensor],dtype=np.float32).reshape(-1)
                got=np.asarray([values[(tensor,i)] for i in range(ref.size)])
                if not np.allclose(got,ref,atol=1e-5,rtol=1e-4):
                    raise ValueError("R2h independently audited intermediate numeric mismatch")
                numerical_max=max(numerical_max,float(np.max(np.abs(got-ref))))
            final=np.asarray([outputs[i] for i in range(m*q)])
            final_ref=np.asarray(reference[model.outputs[0]],
                                 dtype=np.float32).reshape(-1)
            if not np.allclose(final,final_ref,atol=1e-5,rtol=1e-4):
                raise ValueError("R2h independently audited final numeric mismatch")
            count=slots+2*fanout*m*q
            arm="baseline" if layout=="disjoint" else "optimized"
            if (receipt.get("status")!="R2H_ACTUAL_SOURCE_SCRATCH_ALIAS_AND_NUMPY_VERIFIED"
                    or receipt.get("source_layout")!=layout
                    or receipt.get("source_scratch_unit_slots")!=slots
                    or receipt.get("exact_fixed_order_optimal_unit_slots")!=optimum.unit_slots
                    or receipt.get("original_actual_address_witness_count")!=address_count
                    or receipt.get("source_all_declared_mutable_array_elements")!=count
                    or row[arm+"_scratch_unit_slots"]!=str(slots)
                    or row[arm+"_all_declared_mutable_array_elements"]!=str(count)
                    or row[arm+"_address_witnesses"]!=str(address_count)
                    or receipt.get("original_GCC_compilations")!=1
                    or receipt.get("original_GCC_executions")!=1
                    or receipt.get("original_runtime_source_event_sha256")!=
                       safety["source_event_sha256"]
                    or abs(float(receipt[
                        "original_all_intermediate_numeric_max_absolute_error"])-numerical_max)>1e-7):
                raise ValueError("R2h original executed source case receipt inconsistent")
            snapshots[layout]={
                "slot_count":slots,
                "all_declared_mutable_array_elements":count,
                "address_witnesses":address_count,
                "source_event_sha256":safety["source_event_sha256"],
                "max_intermediate_numeric_error":numerical_max,
                "source_cpp_sha256":src_sha,
                "original_stdout_sha256":trace_sha,
            }
        baseline=snapshots["disjoint"]
        optimal=snapshots["per_element_optimal"]
        saved=baseline["slot_count"]-optimal["slot_count"]
        if (baseline["source_event_sha256"]!=optimal["source_event_sha256"]
                or saved!=baseline["all_declared_mutable_array_elements"]-
                          optimal["all_declared_mutable_array_elements"]
                or str(saved)!=row["source_scratch_unit_slots_saved"]
                or str(saved)!=row["all_declared_mutable_array_elements_saved"]
                or row["original_runtime_source_event_sha256"]!=
                   baseline["source_event_sha256"]
                or row["functional_sha256"]!=fx
                or row["architecture_sha256"]!=ax
                or row["original_input_sha256"]!=sha256(x.tobytes()).hexdigest()
                or row["original_numeric_and_event_passes"]!="True"
                or row["actual_HLS_or_BRAM_savings_measured"]!="False"):
            raise ValueError("R2h original matched-pair accounting or event equivalence invalid")
        results.append({
            "case":case,"fanout":fanout,
            "shape":list(dims),"policy":policy,
            "original_disjoint_unit_slots":baseline["slot_count"],
            "original_certified_optimal_unit_slots":optimal["slot_count"],
            "original_source_slot_savings":saved,
            "actual_source_order_match":True,
            "every_actual_source_address_verified":True,
            "all_intermediate_numerics_verified":True,
            "source_declaration_count_includes_accumulators_and_counters":True,
        })
    changed=saved_phi=tied_phi=regressed_phi=0
    for fanout in FANOUTS:
        for dims in SHAPES:
            subset=[row for row in results if row["fanout"]==fanout
                    and row["shape"]==list(dims)]
            scalar=next(row for row in subset if row["policy"]=="scalar")
            for item in subset:
                if item["policy"]=="scalar":
                    continue
                delta=(scalar["original_certified_optimal_unit_slots"]-
                       item["original_certified_optimal_unit_slots"])
                changed+=delta!=0
                saved_phi+=delta>0
                tied_phi+=delta==0
                regressed_phi+=delta<0
    if (summary.get("original_all_matched_source_Phi_numeric_address_passes")!=expected
            or summary.get("source_scratch_reduced_matched_pairs")!=sum(
                r["original_source_slot_savings"]>0 for r in results)
            or summary.get("source_scratch_tied_matched_pairs")!=sum(
                r["original_source_slot_savings"]==0 for r in results)
            or summary.get("source_scratch_regressed_matched_pairs")!=sum(
                r["original_source_slot_savings"]<0 for r in results)
            or summary.get("original_unique_Phi_interventions")!=27
            or summary.get("original_Phi_changed_exact_unit_slots")!=changed
            or summary.get("original_Phi_reduced_exact_unit_slots")!=saved_phi
            or summary.get("original_Phi_tied_exact_unit_slots")!=tied_phi
            or summary.get("original_Phi_regressed_exact_unit_slots")!=regressed_phi):
        raise ValueError("R2h original cohort totals contradict every source pair")
    return {
        "schema":AUDIT_SCHEMA,
        "source_revision":summary["source_revision"],
        "original_matched_pairs_readonly_verified":expected,
        "original_source_programs_readonly_verified":2*expected,
        "all_actual_runtime_source_addresses_independently_replayed":True,
        "all_intermediate_and_final_numerics_independently_recomputed":True,
        "source_minimal_scalar_unit_slots_certified_for_fixed_schedule":True,
        "original_source_layout_savings_cases":summary["source_scratch_reduced_matched_pairs"],
        "original_Phi_edits_reducing_optimized_source_slots":saved_phi,
        "original_Phi_edits_tied_optimized_source_slots":tied_phi,
        "original_Phi_edits_regressing_optimized_source_slots":regressed_phi,
        "original_GCC_compilations_or_executions_during_audit":0,
        "original_Vitis_Vivado_calls_during_audit":0,
        "physical_BRAM_clock_or_RAM_ports_qualified":False,
        "external_firstness_qualified":False,
        "cases":results,
        "limitations":[
            "Auditor verifies saved original source/trace SHA but does not independently recompile C++",
            "Actual source scalar-slot savings are not FPGA RAM primitives or HLS synthesis results",
            "Fixed 2..4 MatMul fanout, fixed serialized source order and one deterministic input per graph",
        ],
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study",required=True)
    p.add_argument("--output",help="Optional NEW create-only audit report folder")
    a=p.parse_args(argv)
    destination=Path(a.output) if a.output else None
    if destination is not None and (destination.exists() or destination.is_symlink()):
        raise FileExistsError("R2h read-only audit folder must be create-only")
    report=audit_original(a.study)
    if destination is not None:
        destination.mkdir(parents=True)
        (destination/"r2h_audit.json").write_text(
            json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
