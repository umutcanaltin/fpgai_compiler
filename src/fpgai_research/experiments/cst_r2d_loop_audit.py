"""Read-only independent audit of original R2d runtime-index C++ traces.

The independent Phi-order reference is not the emitter's event generator.
Rechecks original source/trace/input SHA, exact dynamic READ sequence,
reduction/alias legality and independently evaluated NumPy outputs
without any new GCC/Vitis/Vivado executions. This audit does not
recompile the source, prove FPGA time, or replace original evidence.
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
from fpgai_research.codegen.hls.branch_matmul import disjoint_branch_offsets
from fpgai_research.experiments.cst_branch_models import sample_branch
from fpgai_research.experiments.cst_r2b_source_campaign import SHAPES,POLICIES,_input
from fpgai_research.experiments.cst_r2d_loop_campaign import SCHEMA
from fpgai_research.model_ir.branch_matmul_contract import validate_branch_events
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.independent_phi_issue_reference import (
    validate_phi_issue_conformance,
)
from fpgai_research.model_ir.semantics import architecture_fingerprint,functional_fingerprint
from fpgai_research.validation.r2b_interleaved_cpp import _decode_trace
from fpgai_research.validation.reference import evaluate

AUDIT_SCHEMA="fpgai-research/cst-R2d-original-independent-GCC-readonly-audit/v1"


def _read(path:Path)->bytes:
    if (path.is_symlink() or not path.is_file()
            or path.stat().st_size>8*1024*1024):
        raise ValueError("Original R2d source/trace file missing or oversized: "+str(path))
    return path.read_bytes()


def audit_original(study:str|Path)->dict:
    root=Path(study)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Expected original R2d study directory")
    summary=json.loads(_read(root/"summary.json"))
    rows=list(csv.DictReader(_read(root/"cases.csv").decode().splitlines()))
    if (summary.get("schema")!=SCHEMA or
            summary.get("prespecified_cases")!=12 or len(rows)!=12 or
            summary.get("original_GCC_compilations")!=12 or
            summary.get("original_GCC_executions")!=12):
        raise ValueError("R2d original case denominator/status mismatch")
    output=[]
    for index,row in enumerate(rows):
        case_id=f"case_{index:03d}"
        folder=root/case_id
        if (row["case"]!=case_id or folder.is_symlink() or
                not folder.is_dir() or (folder/"FAILED.json").exists()):
            raise ValueError("Missing, reordered or failed original R2d case")
        original=json.loads(_read(folder/"case.json"))
        receipt=json.loads(_read(folder/"verified.json"))
        source=_read(folder/"source.cpp")
        raw=_read(folder/"original_gcc_trace.txt")
        dims=SHAPES[index//len(POLICIES)]
        policy,phi0,phi1=POLICIES[index%len(POLICIES)]
        if (tuple(json.loads(row["shape"]))!=dims or
                row["policy"]!=policy or
                tuple(original["shape"])!=dims or
                original["policy"]!=policy or
                original["source_revision"]!=summary["source_revision"] or
                row["source_revision"]!=summary["source_revision"]):
            raise ValueError("Original R2d source graph/Phi identity mismatch")
        m,k,n,q=dims
        model=sample_branch(*dims)
        base=ArchitectureRequest()
        edits={
            model.operators[stage].name:replace(base,**args)
            for stage,args in ((1,phi0),(2,phi1)) if args
        }
        arch=resolve(model,base,edits)
        functional=functional_fingerprint(model)
        architecture=architecture_fingerprint(arch)
        x=np.asarray(original["input_float32"],dtype=np.float32)
        if (x.size!=m*k or
                sha256(x.tobytes()).hexdigest()!=original["input_sha256"] or
                row["input_sha256"]!=original["input_sha256"] or
                not np.array_equal(x.reshape((m,k)),_input(m,k))):
            raise ValueError("Original R2d input bytes differ from prespecified study")
        cpp_sha=sha256(source).hexdigest()
        stdout_sha=sha256(raw).hexdigest()
        manifest=original["source_manifest"]
        if (cpp_sha!=original["original_cpp_sha256"] or
                cpp_sha!=row["cpp_sha256"] or
                cpp_sha!=receipt["source_cpp_sha256"] or
                manifest["cpp_sha256"]!=cpp_sha or
                stdout_sha!=row["original_stdout_sha256"] or
                stdout_sha!=receipt["original_GCC_trace_sha256"] or
                row["functional_sha256"]!=functional or
                row["architecture_sha256"]!=architecture or
                manifest["functional_sha256"]!=functional or
                manifest["architecture_sha256"]!=architecture or
                receipt["functional_sha256"]!=functional or
                receipt["architecture_sha256"]!=architecture):
            raise ValueError("Original R2d source/trace/semantic SHA chain invalid")
        events,observed=_decode_trace(raw.decode(),n_outputs=m*q,
                                      max_events=100_000)
        phi=validate_phi_issue_conformance(arch,events)
        offsets=disjoint_branch_offsets(arch,reuse_after_last_read=True)
        safety=validate_branch_events(arch,offsets,events)
        if phi["status"]!="VERIFIED_E0" or safety["status"]!="VERIFIED_E0":
            raise ValueError("Original R2d GCC stdout fails independent mapping/source safety")
        expected=np.asarray(evaluate(
            model,{model.inputs[0]:x.reshape((m,k))})[
                model.outputs[0]],dtype=np.float32).reshape(-1).astype(np.float64)
        error=float(np.max(np.abs(observed-expected)))
        if (not np.allclose(observed,expected,atol=1e-5,rtol=1e-4)
                or abs(error-float(row["numPy_absolute_error"]))>1e-6
                or abs(error-float(receipt["original_numerical_max_absolute_error"]))>1e-6):
            raise ValueError("Original R2d output differs from independent NumPy model")
        actual_H=sum(ev.kind=="READ" and ev.tensor==model.operators[0].outputs[0]
                     for ev in events)
        if (actual_H!=2*m*n*q or row["original_H_reads"]!=str(actual_H) or
                row["actual_Phi_source_issue_sha256"]!=
                    phi["observed_issue_signature_sha256"] or
                receipt["actual_runtime_source_H_issue_signature_SHA256"]!=
                    phi["observed_issue_signature_sha256"] or
                receipt.get("status")!=
                    "E1_GCC_RUNTIME_SUBSCRIPT_PHI_ORDER_AND_NUMPY_VERIFIED" or
                not receipt.get("runtime_subscript_computed_by_CPP_loop") or
                receipt.get("source_order_generated_event_replay_dependency") is not False or
                receipt.get("original_GCC_executions")!=1):
            raise ValueError("Original R2d GCC runtime Phi receipt inconsistency")
        output.append({
            "case":case_id,"shape":list(dims),"policy":policy,
            "actual_H_reads":actual_H,
            "actual_dynamic_issue_sha256":phi["observed_issue_signature_sha256"],
            "original_cpp_sha256":cpp_sha,
            "original_stdout_sha256":stdout_sha,
            "read_only_independent_numeric_error":error,
            "independent_rule_checked":True,
        })
    if len(output)!=12 or summary.get("independent_runtime_Phi_issue_passes")!=12 \
            or summary.get("original_numeric_passes")!=12 \
            or summary.get("source_reduction_alias_passes")!=12 \
            or summary.get("runtime_index_loop_cases")!=12:
        raise ValueError("Original R2d summary is inconsistent with raw evidence")
    return {
        "schema":AUDIT_SCHEMA,"source_revision":summary["source_revision"],
        "original_cases_reverified":len(output),
        "original_runtime_loop_Phi_issue_passes":len(output),
        "original_reduction_alias_passes":len(output),
        "original_independent_numeric_passes":len(output),
        "original_GCC_compilations_or_executions_this_audit":0,
        "original_native_vendor_calls_this_audit":0,
        "independent_spec_not_generator_event_replay":True,
        "FPGA_hardware_schedule_or_resource_proven":False,
        "external_compiler_novelty_proven":False,
        "cases":output,
        "limitations":[
            "Static saved-GCC trace recheck does not independently recompile the source",
            "Fixed five-op diamond and one input per graph, not arbitrary DAG or ONNX models",
            "Source runtime subscript witnessing is not hardware-cycle source-level translation validation",
        ],
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study",required=True)
    p.add_argument("--output",help="Optional NEW create-only report directory")
    a=p.parse_args(argv)
    dest=Path(a.output) if a.output else None
    if dest is not None and (dest.exists() or dest.is_symlink()):
        raise FileExistsError("R2d read-only audit report is create-only")
    result=audit_original(a.study)
    if dest is not None:
        dest.mkdir(parents=True)
        (dest/"r2d_audit.json").write_text(
            json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps(result,sort_keys=True,indent=2))


if __name__=="__main__":
    main()
