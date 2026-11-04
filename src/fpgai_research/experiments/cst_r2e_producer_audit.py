"""Read-only independent audit of prospective R2e original GCC producer study.

Independently reconstruct each original graph, input, mapping, original
producer reduction contributions and typed H COMMIT order, source/trace
SHA-256, reader Phi issue/order/alias legality and numerical output
against the NumPy Functional Model. No new GCC or vendor executions.
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
from fpgai_research.codegen.hls.r2e_producer_traced_branch import SCHEMA as SOURCE_SCHEMA
from fpgai_research.experiments.cst_branch_models import sample_branch
from fpgai_research.experiments.cst_r2b_source_campaign import SHAPES,POLICIES,_input
from fpgai_research.experiments.cst_r2e_producer_campaign import SCHEMA
from fpgai_research.model_ir.branch_matmul_contract import validate_branch_events
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.independent_phi_issue_reference import validate_phi_issue_conformance
from fpgai_research.model_ir.semantics import functional_fingerprint,architecture_fingerprint
from fpgai_research.validation.r2b_interleaved_cpp import _decode_trace
from fpgai_research.validation.r2e_independent_producer_oracle import inspect_producer_stdout
from fpgai_research.validation.reference import evaluate

AUDIT_SCHEMA="fpgai-research/cst-R2e-original-producer-GCC-readonly-audit/v1"


def _read(path:Path)->bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size>8*1024*1024:
        raise ValueError("Original R2e source evidence missing/oversized: "+str(path))
    return path.read_bytes()


def audit_original(study:str|Path)->dict:
    root=Path(study)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Expected original R2e source study directory")
    summary=json.loads(_read(root/"summary.json"))
    rows=list(csv.DictReader(_read(root/"cases.csv").decode().splitlines()))
    if (summary.get("schema")!=SCHEMA or
            summary.get("prespecified_original_cases")!=12 or len(rows)!=12 or
            summary.get("original_GCC_compilations")!=12 or
            summary.get("original_GCC_executions")!=12):
        raise ValueError("Original R2e cohort denominator incomplete")
    results=[]
    for i,row in enumerate(rows):
        case_id=f"case_{i:03d}"
        folder=root/case_id
        if (row.get("case")!=case_id or folder.is_symlink()
                or not folder.is_dir() or (folder/"FAILED.json").exists()):
            raise ValueError("Original R2e failed/reordered/missing case")
        original=json.loads(_read(folder/"case.json"))
        receipt=json.loads(_read(folder/"verified.json"))
        source=_read(folder/"source.cpp")
        stdout=_read(folder/"original_runtime_trace.txt")
        dims=SHAPES[i//len(POLICIES)]
        policy,p0,p1=POLICIES[i%len(POLICIES)]
        if (tuple(json.loads(row["shape"]))!=dims or row["policy"]!=policy
                or tuple(original["shape"])!=dims or original["policy"]!=policy
                or original["case"]!=case_id
                or row["source_revision"]!=summary["source_revision"]
                or original["source_revision"]!=summary["source_revision"]):
            raise ValueError("Original R2e source graph/Phi case mismatch")
        m,k,n,q=dims
        model=sample_branch(*dims)
        base=ArchitectureRequest()
        edits={
            model.operators[stage].name:replace(base,**mapping)
            for stage,mapping in ((1,p0),(2,p1)) if mapping
        }
        arch=resolve(model,base,edits)
        fx=functional_fingerprint(model)
        ax=architecture_fingerprint(arch)
        x=np.asarray(original["input_float32_values"],dtype=np.float32)
        if (x.size!=m*k or not np.array_equal(x.reshape((m,k)),_input(m,k))
                or sha256(x.tobytes()).hexdigest()!=original["input_float32_sha256"]
                or row["input_sha256"]!=original["input_float32_sha256"]):
            raise ValueError("Original R2e input/original numeric seed mismatch")
        src_sha=sha256(source).hexdigest()
        log_sha=sha256(stdout).hexdigest()
        manifest=original["source_manifest"]
        if (src_sha!=original["original_source_sha256"]
                or src_sha!=row["source_cpp_sha256"]
                or src_sha!=receipt["source_cpp_sha256"]
                or src_sha!=manifest["cpp_sha256"]
                or manifest["schema"]!=SOURCE_SCHEMA
                or log_sha!=row["original_trace_sha256"]
                or log_sha!=receipt["original_full_stdout_sha256"]
                or row["functional_sha256"]!=fx
                or row["architecture_sha256"]!=ax
                or receipt["functional_sha256"]!=fx
                or receipt["architecture_sha256"]!=ax
                or manifest["functional_sha256"]!=fx
                or manifest["architecture_sha256"]!=ax):
            raise ValueError("Original R2e source/trace SHA or semantic identity mismatch")
        tensor_h=model.operators[0].outputs[0]
        producer=inspect_producer_stdout(
            stdout.decode(),m=m,k=k,n=n,h=tensor_h)
        events,actual=_decode_trace(producer["typed_branch_stdout"],
                                    n_outputs=m*q,max_events=100_000)
        # SECONDARY independent arithmetic: explicit closed-form scalar
        # MatMul sum, separate from the source verifier's NumPy matmul.
        w0=np.asarray(
            model.tensor(model.operators[0].inputs[1]).data,
            dtype=np.float32).reshape(k,n)
        producer_h_expected=np.asarray([
            sum(float(x[row*k+kk])*float(w0[kk,col])
                for kk in range(k))
            for row in range(m) for col in range(n)
        ],dtype=np.float64)
        producer_h_actual=np.asarray(
            producer["runtime_H_values"],dtype=np.float64)
        producer_h_error=float(np.max(np.abs(
            producer_h_actual-producer_h_expected)))
        if (producer_h_actual.shape!=(m*n,) or
                not np.all(np.isfinite(producer_h_actual)) or
                not np.allclose(producer_h_actual,producer_h_expected,
                                atol=1e-5,rtol=1e-4)):
            raise ValueError("Original R2e independently reconstructed H values mismatch")
        phi=validate_phi_issue_conformance(arch,events)
        offsets=disjoint_branch_offsets(arch,reuse_after_last_read=True)
        safety=validate_branch_events(arch,offsets,events)
        desired=np.asarray(evaluate(
            model,{model.inputs[0]:x.reshape((m,k))})[
                model.outputs[0]],dtype=np.float32).reshape(-1).astype(np.float64)
        error=float(np.max(np.abs(actual-desired)))
        if (phi["status"]!="VERIFIED_E0" or safety["status"]!="VERIFIED_E0"
                or not np.allclose(actual,desired,rtol=1e-4,atol=1e-5)):
            raise ValueError("Original R2e source producer/consumer/numeric audit rejected")
        if (producer["producer_trace_sha256"]!=row["producer_event_sha256"]
                or producer["producer_trace_sha256"]!=receipt["original_producer_trace_sha256"]
                or producer["producer_H_values_sha256"]!=row["runtime_H_values_sha256"]
                or producer["producer_H_values_sha256"]!=receipt["runtime_H_values_sha256"]
                or abs(producer_h_error-float(
                    row["original_H_numeric_max_abs_error"]))>1e-6
                or abs(producer_h_error-float(
                    receipt["original_intermediate_H_numeric_max_abs_error"]))>1e-6
                or producer["observed_contributions"]!=m*k*n
                or row["actual_producer_contributions"]!=str(m*k*n)
                or row["actual_complete_H_producer_commits"]!=str(m*n)
                or receipt["independent_complete_H_producer_commits"]!=m*n
                or abs(error-float(row["original_numerical_max_abs_error"]))>1e-6
                or abs(error-float(receipt["original_numeric_max_absolute_error"]))>1e-6
                or receipt["status"]!=
                  "ORIGINAL_R2E_GCC_PRODUCER_CONTRIBUTIONS_AND_NUMPY_VERIFIED_E1"
                or receipt["original_GCC_executions"]!=1
                or row["source_producer_complete"]!="True"):
            raise ValueError("Original R2e producer completeness receipt/count mismatch")
        results.append({
            "case":case_id,"shape":list(dims),"policy":policy,
            "original_source_sha256":src_sha,
            "original_stdout_sha256":log_sha,
            "producer_contributions":m*k*n,
            "producer_commits":m*n,
            "independent_producer_complete":True,
            "independent_Phi_source_issue_complete":True,
            "independent_consumer_reductions_complete":True,
            "independent_Numpy_error":error,
            "independent_intermediate_H_numeric_error":producer_h_error,
            "independent_intermediate_H_numeric_qualified":True,
        })
    if (len(results)!=12
            or summary.get("original_producer_H_commit_complete_cases")!=12
            or summary.get("original_numeric_and_consumer_source_passes")!=12
            or summary.get("original_intermediate_H_numeric_passes")!=12
            or summary.get("original_complete_producer_contributions")!=sum(
                r["producer_contributions"] for r in results)):
        raise ValueError("Original R2e aggregate summary contradicts original case evidence")
    return {
        "schema":AUDIT_SCHEMA,
        "source_revision":summary["source_revision"],
        "original_source_cases_readonly_verified":len(results),
        "complete_H_producer_and_typed_COMMIT_cases":len(results),
        "independent_Phi_reader_issue_passes":len(results),
        "independent_consumer_reduction_alias_passes":len(results),
        "independent_NumPy_numeric_passes":len(results),
        "independent_intermediate_H_numeric_passes":len(results),
        "original_total_producer_contributions":sum(
            row["producer_contributions"] for row in results),
        "original_GCC_compilations_or_executions_during_audit":0,
        "original_Vitis_or_Vivado_calls_during_audit":0,
        "FPGA_hardware_producer_completeness_proven":False,
        "arbitrary_graph_producer_completion_proven":False,
        "external_compiler_firstness_proven":False,
        "cases":results,
        "limitations":[
            "Read-only original C++ trace auditing is not a fresh independent recompilation",
            "Runtime contribution witness and NumPy equality do not prove all arbitrary C++ inputs",
            "Only fixed serial producer in three MatMul diamond shapes, not arbitrary DAG",
            "No hardware timing, on-chip RAM ports, physical BRAM, power or external comparison",
        ],
    }


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",required=True)
    parser.add_argument("--output",help="Optional new create-only audit output directory")
    args=parser.parse_args(argv)
    destination=Path(args.output) if args.output else None
    if destination is not None and (destination.exists() or destination.is_symlink()):
        raise FileExistsError("R2e audit output must be create-only")
    report=audit_original(args.study)
    if destination is not None:
        destination.mkdir(parents=True)
        (destination/"r2e_audit.json").write_text(
            json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(json.dumps(report,sort_keys=True,indent=2))


if __name__=="__main__":
    main()
