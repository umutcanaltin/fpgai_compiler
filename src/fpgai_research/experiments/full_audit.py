"""Verify every E0/E1 stage of a prepared progressive Model IR study.

Fails closed when saved IR, HLS source, reference outputs, or host receipts
disagree. Native Vitis/Vivado evidence is never assumed or simulated.
"""
from __future__ import annotations
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import re

from fpgai_research.experiments.campaign import DIMENSIONS
from fpgai_research.experiments.full_pipeline import parse_plan
from fpgai_research.experiments.results import summarize_case
from fpgai_research.model_ir.artifacts import read_ir_artifact

def audit_full_experiment(directory, *, require_host=True, require_all_axes=False):
    if type(require_host) is not bool or type(require_all_axes) is not bool:
        raise ValueError("Audit flags must be boolean")
    root=Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Full experiment must be a regular directory")
    manifest=json.loads((root/"full_experiment.json").read_text())
    index=json.loads((root/"campaign.json").read_text())
    if manifest.get("schema")!="fpgai-research/full-progressive-experiment/v1" or index.get("schema")!="fpgai-research/paper-campaign/v1":
        raise ValueError("Unexpected experiment or campaign schema")
    planfile=root/"experiment_plan.json"
    if not planfile.is_file() or sha256(planfile.read_bytes()).hexdigest()!=manifest.get("plan_sha256"):
        raise ValueError("Exact architecture plan bytes missing or changed")
    plan=parse_plan(json.loads(planfile.read_text()))
    if plan["operator"]!=manifest.get("operator") or plan["operator"]!=index.get("operator"):
        raise ValueError("Exact Model IR operator identity disagrees with saved plan")
    if (index.get("part")!=plan["target_part"] or
        index.get("clock_ns")!=plan["clock_ns"] or
        index.get("tolerance")!=plan["tolerance"]):
        raise ValueError("Campaign FPGA part/clock/numerical tolerance disagree with saved plan")
    planned=plan["points"]
    rows=index.get("cases")
    if not isinstance(rows,list) or not rows or len(rows)!=manifest.get("requested_cases") or len(rows)!=len(planned):
        raise ValueError("Missing or inconsistent experiment case inventory")
    if len(manifest.get("cases",[]))!=len(rows):
        raise ValueError("Scientific case receipts do not cover the campaign")
    reference_file=root/"reference"/"reference_evaluation.json"
    if (not reference_file.is_file() or
        sha256(reference_file.read_bytes()).hexdigest()!=manifest.get("reference_receipt_sha256")):
        raise ValueError("Exact E0 reference receipt changed")
    reference=json.loads(reference_file.read_text())
    outputs=root/"reference"/"reference_outputs.npy"
    if not outputs.is_file() or sha256(outputs.read_bytes()).hexdigest()!=reference.get("outputs_sha256"):
        raise ValueError("Reference output digest differs from recorded E0 output")
    if (reference.get("schema")!="fpgai-research/reference-evaluation/v1"
        or reference!=manifest.get("reference_evaluation")
        or reference.get("model_sha256")!=manifest.get("model_sha256")
        or reference.get("inputs_sha256")!=manifest.get("inputs_sha256")
        or reference.get("labels_sha256")!=manifest.get("labels_sha256")):
        raise ValueError("Source reference receipt and experiment disagree")
    if (root/"inputs").exists() or (root/"inputs").is_symlink():
        from fpgai_research.experiments.input_archive import verify_case_provenance
        verify_case_provenance(root)
    from fpgai_research.experiments.representation_evidence import analyze_representation
    scientific=root/"representation_ablation.json"
    if not scientific.is_file() or sha256(scientific.read_bytes()).hexdigest()!=manifest.get("representation_evidence_sha256"):
        raise ValueError("Scientific representation ablation receipt changed")
    recomputed=analyze_representation(root,rows)
    if recomputed!=json.loads(scientific.read_text()):
        raise ValueError("Scientific representation mapping analysis cannot be reproduced")
    cases=[]
    seen=set()
    axes=set()
    for row,source,(planned_axis,request) in zip(rows,manifest["cases"],planned):
        expected={**asdict(request),"tile":list(request.tile)}
        if row.get("dimension")!=planned_axis or row.get("requested")!=expected:
            raise ValueError("Case axis or architecture disagrees with immutable experiment plan")
        name=row.get("case")
        if not isinstance(name,str) or not re.fullmatch(r"case_[0-9]{4,8}",name) or name in seen or source.get("case")!=name:
            raise ValueError("Unsafe, duplicate or reordered case identity")
        seen.add(name)
        case=root/name
        if case.is_symlink():
            raise ValueError("Case symlink not accepted")
        if row.get("status")=="READY" and source.get("status")!="E1-PREPARED":
            raise ValueError("Prepared case status differs from scientific inventory")
        if row.get("status")=="BLOCKED":
            if not row.get("reason") or source.get("status")!="BLOCKED":
                raise ValueError("Blocked architecture lacks an explicit reason")
            cases.append({"case":name,"status":"BLOCKED","reason":row["reason"]})
            continue
        if row.get("status")!="READY":
            raise ValueError("Unexpected campaign case status")
        axes.add(row.get("dimension"))
        manifest_file=case/"manifest.json"
        if manifest_file.is_symlink() or not manifest_file.is_file():
            raise ValueError("Prepared Vitis case manifest is absent or linked")
        vitis_manifest=json.loads(manifest_file.read_text())
        if (vitis_manifest.get("part")!=plan["target_part"]
            or vitis_manifest.get("clock_ns")!=plan["clock_ns"]
            or vitis_manifest.get("numerical_tolerance")!=plan["tolerance"]):
            raise ValueError("Prepared HLS case target/clock/tolerance differ from architecture plan")
        artifact=read_ir_artifact(case/"progressive_ir.json")
        if artifact.get("functional_sha256")!=manifest.get("functional_sha256") or source.get("stage_sha256")!=artifact.get("stage_sha256"):
            raise ValueError("Three-stage Model IR artifact/functional identity mismatch")
        summary=summarize_case(str(case))
        if row.get("cpp_sha256")!=summary["source_sha256"] or source.get("cpp_sha256")!=summary["source_sha256"]:
            raise ValueError("Generated HLS source disagrees with scientific manifest")
        host_file=case/"host_validation_samples.json"
        if (host_file.is_symlink() or not host_file.is_file() or
            sha256(host_file.read_bytes()).hexdigest()!=source.get("host_receipt_sha256")):
            raise ValueError("GCC host sample evidence digest mismatch")
        host=json.loads(host_file.read_text())
        if host.get("schema")!="fpgai-research/host-sample-list/v1" or host.get("source_sha256")!=summary["source_sha256"]:
            raise ValueError("Host C++ validation source attribution mismatch")
        receipts=host.get("results")
        if not isinstance(receipts,list) or any(
            item.get("accepted") is not True or
            item.get("cpp_sha256")!=summary["source_sha256"] or
            item.get("sample_index")!=i
            for i,item in enumerate(receipts)):
            raise ValueError("Host numerical validation has incomplete, failed or reordered results")
        if len(receipts)!=source.get("host_samples_passed"):
            raise ValueError("Host numeric count does not match experiment receipt")
        expected_samples=manifest.get("sample_value_sha256")
        if (not isinstance(expected_samples,list) or
            len(expected_samples)!=manifest.get("host_samples_requested") or
            len(receipts)!=len(expected_samples) or
            any(item.get("input_sha256")!=digest for item,digest in zip(receipts,expected_samples))):
            raise ValueError("GCC results disagree with prepared input sample fingerprints")
        trace_file=case/"tensor_trace.json"
        if manifest.get("intermediate_trace_requested") and not trace_file.is_file():
            raise ValueError("Required intermediate tensor trace is missing")
        if trace_file.is_file():
            if (trace_file.is_symlink() or
                sha256(trace_file.read_bytes()).hexdigest()!=source.get("trace_receipt_sha256")):
                raise ValueError("Intermediate tensor trace evidence digest mismatch")
            trace=json.loads(trace_file.read_text())
            if not expected_samples or trace.get("input_sha256")!=expected_samples[0]:
                raise ValueError("Intermediate trace input differs from prepared sample")
            if (trace.get("schema")!="fpgai-research/operator-numerical-trace/v1"
                or trace.get("accepted") is not True
                or trace.get("cpp_sha256")!=summary["source_sha256"]):
                raise ValueError("Intermediate tensor trace is invalid or from a different source")
            if source.get("intermediate_tensor_trace_passed") is not True:
                raise ValueError("Intermediate operator trace not in scientific receipt")
        cases.append({"case":name,"status":"READY","axis":row.get("dimension"),
            "functional_sha256":artifact["functional_sha256"],
            "source_sha256":summary["source_sha256"],
            "host_samples_passed":len(receipts),"evidence":summary["evidence"],
            "csim_passed":summary.get("csim_passed"),
            "timing_met":((summary.get("vivado_metrics") or {}).get("timing") or {}).get("timing_met")})
    legal=[row for row in cases if row["status"]=="READY"]
    missing_axes=sorted(set(DIMENSIONS)-axes)
    baseline="baseline" in axes
    ir_valid=bool(legal) and all(row["functional_sha256"]==manifest["functional_sha256"] for row in legal)
    host_verified=all(row["host_samples_passed"]>0 for row in legal)
    # Strict readiness covers every requested point; subset readiness
    # preserves scientifically rejected configurations without suppressing
    # native experiments for separate valid circuits.
    legal_subset_ready=ir_valid and baseline and (host_verified or not require_host)
    ready=legal_subset_ready and len(legal)==len(cases)
    if require_all_axes:
        ready=ready and baseline and not missing_axes
        legal_subset_ready=legal_subset_ready and baseline and not missing_axes
    physical_complete=bool(legal) and len(legal)==len(cases) and baseline and not missing_axes and all(
        row["evidence"]=="E4-Vivado-implementation" and
        row["csim_passed"] is True and row["timing_met"] is True for row in legal)
    return {"schema":"fpgai-research/full-progressive-experiment-audit/v1",
            "experiment_source_verified":ir_valid,
            "e0_reference_verified":True,
            "all_case_ir_artifacts_verified":True,
            "host_numerics_passed_for_all_legal_cases":host_verified,
            "prepared_for_native_experiments":ready,
            "legal_subset_prepared_for_native_execution":legal_subset_ready,
            "infeasible_architecture_case_count":len(cases)-len(legal),
            "seven_axis_coverage_complete":baseline and not missing_axes,
            "baseline_present":baseline,"missing_axes":missing_axes,
            "native_paper_evidence_complete":physical_complete,
            "case_count":len(cases),"cases":cases,
            "limitations":"GCC host float is E1 only; licensed native Vitis/Vivado evidence remains separate"}
