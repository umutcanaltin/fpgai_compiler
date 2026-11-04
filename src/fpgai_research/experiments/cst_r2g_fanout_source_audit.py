"""Independent read-only audit of all original R2g 36 host GCC source cases.

Reconstructs models and mapping choices independently of generated C++,
validates every saved case source/input/stdout SHA and receipt, reparses
actual dynamic fanout source events (not the generator's prepared list),
checks producer reduction and every consumer Add edge, independently
rechecks Phi issue and all intermediate + final NumPy values.
No GCC recompilation or vendor tool calls during this audit.
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
from fpgai_research.experiments.cst_fanout_models import sample_fanout
from fpgai_research.experiments.cst_r2b_source_campaign import _input
from fpgai_research.experiments.cst_r2f_fanout_campaign import (
    FANOUTS,SHAPES,POLICY_NAMES,policy_widths,
)
from fpgai_research.experiments.cst_r2g_fanout_source_campaign import SCHEMA
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.fanout_contract import (
    admit_fanout,disjoint_fanout_offsets,verify_fanout_events,
    verify_fanout_issue,
)
from fpgai_research.model_ir.semantics import functional_fingerprint,architecture_fingerprint
from fpgai_research.validation.r2g_fanout_cpp import decode_stdout
from fpgai_research.validation.reference import evaluate

AUDIT_SCHEMA="fpgai-research/cst-R2g-36-case-original-GCC-readonly-audit/v1"


def _read(path:Path)->bytes:
    if (path.is_symlink() or not path.is_file()
            or path.stat().st_size>8*1024*1024):
        raise ValueError("Original R2g audit file missing/symlinked/oversized: "+str(path))
    return path.read_bytes()


def audit_original(study:str|Path)->dict:
    root=Path(study)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Original R2g source study directory missing")
    summary=json.loads(_read(root/"summary.json"))
    rows=list(csv.DictReader(_read(root/"cases.csv").decode().splitlines()))
    n_cases=len(FANOUTS)*len(SHAPES)*len(POLICY_NAMES)
    if (summary.get("schema")!=SCHEMA or len(rows)!=n_cases
            or summary.get("prespecified_original_cases")!=n_cases
            or summary.get("original_GCC_compilations")!=n_cases
            or summary.get("original_GCC_executions")!=n_cases):
        raise ValueError("Original R2g source study population/denominator changed")
    results=[]
    for i,row in enumerate(rows):
        expected_fanout=FANOUTS[i//(len(SHAPES)*len(POLICY_NAMES))]
        within=i%(len(SHAPES)*len(POLICY_NAMES))
        shape=SHAPES[within//len(POLICY_NAMES)]
        policy=POLICY_NAMES[within%len(POLICY_NAMES)]
        case=f"case_{i:03d}"
        folder=root/case
        if (row.get("case")!=case or folder.is_symlink()
                or not folder.is_dir() or (folder/"FAILED.json").exists()
                or int(row["fanout"])!=expected_fanout
                or tuple(json.loads(row["shape"]))!=shape
                or row["policy"]!=policy):
            raise ValueError("Original R2g failed/missing/reordered case")
        original=json.loads(_read(folder/"case.json"))
        receipt=json.loads(_read(folder/"verified.json"))
        cpp=_read(folder/"source.cpp")
        raw=_read(folder/"original_stdout.txt")
        if (original.get("case")!=case
                or original.get("fanout")!=expected_fanout
                or tuple(original["shape"])!=shape
                or original["policy"]!=policy
                or original["source_revision"]!=summary["source_revision"]
                or row["source_revision"]!=summary["source_revision"]):
            raise ValueError("Original R2g source model or Phi scope mismatch")
        m,k,n,q=shape
        model=sample_fanout(expected_fanout,m=m,k=k,n=n,q=q)
        base=ArchitectureRequest()
        arch=resolve(model,base,{
            f"reader{j}":replace(base,pes=width)
            for j,width in enumerate(policy_widths(expected_fanout,policy))
            if width!=1
        })
        spec=admit_fanout(arch)
        x=np.asarray(original["input_float32_values"],dtype=np.float32)
        if (x.size!=m*k or
                not np.array_equal(x.reshape(m,k),_input(m,k))
                or sha256(x.tobytes()).hexdigest()!=original["input_sha256"]
                or row["original_input_sha256"]!=original["input_sha256"]):
            raise ValueError("Original R2g input source identity changed")
        fhash=functional_fingerprint(model)
        ahash=architecture_fingerprint(arch)
        cpphash=sha256(cpp).hexdigest()
        tracehash=sha256(raw).hexdigest()
        manifest=original["source_manifest"]
        if (cpphash!=original["source_cpp_sha256"]
                or cpphash!=row["source_cpp_sha256"]
                or cpphash!=receipt["source_cpp_sha256"]
                or cpphash!=manifest["source_cpp_sha256"]
                or tracehash!=row["original_stdout_sha256"]
                or tracehash!=receipt["original_stdout_sha256"]
                or fhash!=row["functional_sha256"]
                or ahash!=row["architecture_sha256"]
                or fhash!=manifest["functional_sha256"]
                or ahash!=manifest["architecture_sha256"]
                or fhash!=receipt["functional_sha256"]
                or ahash!=receipt["architecture_sha256"]
                or manifest["source_offsets"]!=dict(sorted(
                    disjoint_fanout_offsets(spec).items()))):
            raise ValueError("Original R2g source/trace/functional SHA lineage invalid")
        events,values,outputs=decode_stdout(raw.decode())
        safety=verify_fanout_events(
            arch,disjoint_fanout_offsets(spec),events)
        phi=verify_fanout_issue(arch,events)
        if safety["status"]!="VERIFIED_E0" or phi["status"]!="VERIFIED_E0":
            raise ValueError("Original R2g source producer/read/Phi/alias event evidence rejected")
        numerical=evaluate(model,{spec.input:x.reshape(m,k)},return_all=True)
        expected_values={
            (tensor,j):float(value)
            for tensor in spec.intermediates
            for j,value in enumerate(np.asarray(
                numerical[tensor],dtype=np.float32).reshape(-1))}
        if set(values)!=set(expected_values):
            raise ValueError("Original R2g intermediate numeric witness population changed")
        maximum=0.
        for (tensor,j),wanted in expected_values.items():
            actual=values[(tensor,j)]
            if not np.isclose(actual,wanted,rtol=1e-4,atol=1e-5):
                raise ValueError("Original R2g independent intermediate NumPy mismatch")
            maximum=max(maximum,abs(actual-wanted))
        target=np.asarray(numerical[model.outputs[0]],dtype=np.float32).reshape(-1)
        if set(outputs)!=set(range(m*q)):
            raise ValueError("Original R2g incomplete final output")
        actual_out=np.asarray([outputs[j] for j in range(m*q)])
        if not np.allclose(actual_out,target,atol=1e-5,rtol=1e-4):
            raise ValueError("Original R2g final NumPy output mismatch")
        output_error=float(np.max(np.abs(actual_out-target)))
        if (receipt["status"]!="ORIGINAL_GCC_RUNTIME_2_TO_4_FANOUT_E1_VERIFIED"
                or receipt["runtime_event_sha256"]!=safety["source_event_sha256"]
                or receipt["original_H_reader_reads"]!=safety["actual_H_reads"]
                or receipt["original_producer_contributions"]!=
                    safety["producer_contributions"]
                or row["runtime_event_sha256"]!=safety["source_event_sha256"]
                or int(row["reader_H_reads"])!=safety["actual_H_reads"]
                or int(row["producer_contributions"])!=safety["producer_contributions"]
                or int(row["intermediate_numeric_values"])!=len(values)
                or not np.isclose(float(row["max_intermediate_abs_error"]),
                                  maximum,rtol=0,atol=1e-7)
                or not np.isclose(float(row["max_final_abs_error"]),
                                  output_error,rtol=0,atol=1e-7)
                or receipt["original_GCC_compilations"]!=1
                or receipt["original_GCC_executions"]!=1
                or row["all_source_oracles_passed"]!="True"
                or row["actual_per_element_optimized_storage_executed"]!="False"):
            raise ValueError("Original R2g case-level evidence receipt mismatch")
        results.append({
            "case":case,"fanout":expected_fanout,
            "shape":list(shape),"policy":policy,
            "producer_contributions":safety["producer_contributions"],
            "H_reads":safety["actual_H_reads"],
            "intermediate_value_witnesses":len(values),
            "independently_recomputed_intermediate_max_error":maximum,
            "independently_recomputed_final_max_error":output_error,
            "source_cpp_sha256":cpphash,
            "original_stdout_sha256":tracehash,
            "independent_source_and_issue_verified":True,
        })
    if (len(results)!=n_cases
            or summary["independent_source_event_Phi_and_alias_passes"]!=n_cases
            or summary["all_intermediate_and_final_NumPy_numeric_passes"]!=n_cases
            or summary["runtime_dynamic_subscript_cases"]!=n_cases
            or summary["producer_CONTRIBUTE_readiness_cases"]!=n_cases
            or summary["source_disjoint_scratch_layout_cases"]!=n_cases):
        raise ValueError("Original R2g aggregate cohort contradicts cases")
    return {
        "schema":AUDIT_SCHEMA,
        "source_revision":summary["source_revision"],
        "original_cases_readonly_verified":len(results),
        "independent_dynamic_Phi_and_producer_reader_Add_checks":len(results),
        "all_intermediate_numerics_and_final_NumPy_verified":len(results),
        "original_GCC_compilations_or_executions_during_audit":0,
        "native_Vitis_or_Vivado_calls_during_audit":0,
        "source_value_layout":"DISJOINT_ONLY_NOT_UNIT_SLOT_OPTIMIZED",
        "FPGA_hardware_BRAM_or_cycle_qualification":False,
        "scientific_firstness_qualification":False,
        "cases":results,
        "limitations":[
            "Saved original GCC source/trace SHA audit is not a new independent recompilation",
            "Fixed 2..4 MatMul fanout and left-associated Add chains, no arbitrary DAG",
            "Source-level disjoint scratch memory; unit-slot optimizer not materially executed",
            "No native HLS timing/BRAM, physical placement, RTL or peer compiler parity",
        ],
    }


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",required=True)
    parser.add_argument("--output",help="Optional new create-only audit folder")
    args=parser.parse_args(argv)
    destination=Path(args.output) if args.output else None
    if destination is not None and (destination.exists() or destination.is_symlink()):
        raise FileExistsError("R2g read-only audit output is create-only")
    result=audit_original(args.study)
    if destination is not None:
        destination.mkdir(parents=True)
        (destination/"r2g_audit.json").write_text(
            json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps(result,sort_keys=True,indent=2))


if __name__=="__main__":
    main()
