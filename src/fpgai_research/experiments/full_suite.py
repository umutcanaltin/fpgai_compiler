"""Atomic, multi-model E0/E1 paper suite for progressive Model IR studies.

Each supplied model receives its own complete three-stage experiment.
Missing trained assets are blockers, never silently replaced by fake models.
No vendor tools are executed by this module.
"""
from __future__ import annotations
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from fpgai_research.experiments.full_pipeline import prepare_full_experiment
from fpgai_research.experiments.full_audit import audit_full_experiment
from fpgai_research.frontend.onnx import load_onnx
from fpgai_research.architecture.resolver import ArchitectureError

SCHEMA="fpgai-research/full-paper-suite/v1"

def _source(root, name):
    if not isinstance(name,str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("Suite input paths must be relative to the suite manifest")
    path=root/name
    cursor=root
    for component in Path(name).parts:
        cursor=cursor/component
        if cursor.is_symlink():
            raise FileNotFoundError("Suite input missing, linked or outside manifest root: "+name)
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise FileNotFoundError("Suite input missing, linked or outside manifest root: "+name)
    return path

def validate_suite(spec):
    if not isinstance(spec,dict) or set(spec)-{"schema","experiments","notes"} or spec.get("schema")!=SCHEMA:
        raise ValueError("Unsupported research suite schema or fields")
    items=spec.get("experiments")
    if not isinstance(items,list) or not 1<=len(items)<=16:
        raise ValueError("Research suite needs 1..16 experiments")
    seen=set()
    for item in items:
        if not isinstance(item,dict) or set(item)-{"id","family","onnx","input","labels","plan","operator"}:
            raise ValueError("Unexpected study declaration fields")
        name=item.get("id")
        if not isinstance(name,str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,40}",name) or name in seen:
            raise ValueError("Invalid or duplicate suite experiment identifier")
        seen.add(name)
        if item.get("family") not in ("mlp","cnn","residual","attention","detector","other"):
            raise ValueError("Unsupported suite family")
        if not all(isinstance(item.get(key),str) for key in ("onnx","input")):
            raise ValueError("Each experiment needs exact ONNX and input paths")
        if "plan" in item and not isinstance(item["plan"],str):
            raise ValueError("Explicit architecture plan path must be a string")
        if "operator" in item and not isinstance(item["operator"],str):
            raise ValueError("Explicit operator identity must be a string")
    return items

def prepare_suite(*,suite_path,directory,host_samples=1,archive_inputs=False,
                  allow_missing=False,require_all_legal=True,
                  crosscheck_onnxruntime=False,trace_intermediates=False,
                  allow_unsupported=False,architecture_profile="bounded"):
    if any(type(flag) is not bool for flag in
           (archive_inputs,allow_missing,require_all_legal,crosscheck_onnxruntime,trace_intermediates,allow_unsupported)):
        raise ValueError("Suite policy flags must be booleans")
    if architecture_profile not in ("bounded", "comprehensive"):
        raise ValueError("Unknown scientific architecture sampling profile")
    definition=Path(suite_path)
    if definition.is_symlink() or not definition.is_file():
        raise ValueError("Suite definition requires a regular JSON file")
    rows=validate_suite(json.loads(definition.read_text()))
    source_root=definition.parent.resolve()
    destination=Path(directory)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Suite destination already exists")
    with TemporaryDirectory(prefix=".fpgai-suite-",dir=destination.parent) as folder:
        root=Path(folder)
        outcomes=[]
        for row in rows:
            name=row["id"]
            try:
                onnx=_source(source_root,row["onnx"])
                inputs=_source(source_root,row["input"])
                plan=_source(source_root,row["plan"]) if row.get("plan") else None
                labels=_source(source_root,row["labels"]) if row.get("labels") else None
            except FileNotFoundError as exc:
                # Missing external datasets may be recorded as blocked.
                # An invalid or escaping manifest path is not a missing asset
                # and must never be accepted by --allow-missing.
                if not allow_missing:
                    raise
                outcomes.append({"id":name,"family":row["family"],
                    "status":"BLOCKED_ASSETS","reason":str(exc),
                    "native_evidence_complete":False})
                continue
            # Reject unsupported ONNX import at the import boundary only.
            # Searching for the word "unsupported" around a complete pipeline
            # incorrectly hides emitter/host/report failures that reuse it.
            try:
                functional=load_onnx(str(onnx))
            except ValueError as exc:
                detail=str(exc)
                unsupported=(isinstance(exc,ArchitectureError) or
                    any(term in detail.lower() for term in
                        ("unsupported","not supported","only static")))
                if not allow_unsupported or not unsupported:
                    raise
                outcomes.append({"id":name,"family":row["family"],
                    "status":"BLOCKED_UNSUPPORTED","reason":detail,
                    "native_evidence_complete":False})
                continue

            if plan is None:
                from fpgai_research.experiments.plan_factory import (
                    make_seven_axis_plan, make_comprehensive_seven_axis_plan,
                )
                try:
                    build_plan = (
                        make_comprehensive_seven_axis_plan
                        if architecture_profile == "comprehensive"
                        else make_seven_axis_plan
                    )
                    settings=build_plan(functional,row.get("operator"))
                except ValueError as exc:
                    if not allow_unsupported or str(exc)!="No HLS compute operator eligible for seven-axis study":
                        raise
                    outcomes.append({"id":name,"family":row["family"],
                        "status":"BLOCKED_UNSUPPORTED","reason":str(exc),
                        "native_evidence_complete":False})
                    continue
                (root/f"{name}_generated_plan.json").write_text(
                    json.dumps(settings,sort_keys=True,indent=2)+"\n")
            else:
                settings=json.loads(plan.read_text())
            # An unexpected error after import is a failed scientific run,
            # never a valid BLOCKED_UNSUPPORTED result.
            output=prepare_full_experiment(model=functional,
                model_path=onnx,inputs_path=inputs,labels_path=labels,
                plan=settings,directory=root/name,
                host_samples=host_samples,archive_inputs=archive_inputs,
                require_all_legal=require_all_legal,
                crosscheck_onnxruntime=crosscheck_onnxruntime,
                trace_intermediates=trace_intermediates)
            audit=audit_full_experiment(root/name,require_host=host_samples>0,
                                        require_all_axes=False)
            if not audit["legal_subset_prepared_for_native_execution"]:
                raise ValueError("No source-attested legal architecture cases available")
            outcomes.append({"id":name,"family":row["family"],
                "status":"E1-PREPARED","ready_cases":output["ready_cases"],
                "model_sha256":output["model_sha256"],
                "functional_sha256":output["functional_sha256"],
                "seven_axis_coverage_complete":audit["seven_axis_coverage_complete"],
                "blocked_architecture_cases":output["blocked_cases"],
                "native_evidence_complete":audit["native_paper_evidence_complete"]})
        result={"schema":SCHEMA,"requested_models":len(rows),
            "prepared_models":sum(x["status"]=="E1-PREPARED" for x in outcomes),
            "blocked_models":sum(x["status"].startswith("BLOCKED_") for x in outcomes),
            "all_models_ready":all(x["status"]=="E1-PREPARED" for x in outcomes),
            "vitis_executed":False,"vivado_executed":False,
            "evidence":"E0/E1-source-preparation-only","studies":outcomes}
        (root/"suite_results.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("Suite destination appeared during preparation")
        root.rename(destination)
    return result


def audit_suite(directory, *, require_host=True, require_all_axes=False):
    """Revalidate every suite member after preparation or real vendor runs."""
    if type(require_host) is not bool or type(require_all_axes) is not bool:
        raise ValueError("Suite audit flags must be boolean")
    root=Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Suite must be a regular directory")
    manifest=json.loads((root/"suite_results.json").read_text())
    if manifest.get("schema")!=SCHEMA:
        raise ValueError("Unsupported full paper suite schema")
    declared=manifest.get("studies")
    requested=manifest.get("requested_models")
    if (type(requested) is not int or not 1<=requested<=16 or
        not isinstance(declared,list) or len(declared)!=requested or
        any(not isinstance(row,dict) for row in declared)):
        raise ValueError("Incomplete, empty or invalid suite study inventory")
    statuses=[row.get("status") for row in declared]
    expected_prepared=sum(status=="E1-PREPARED" for status in statuses)
    expected_blocked=sum(status in ("BLOCKED_ASSETS","BLOCKED_UNSUPPORTED","BLOCKED_VALIDATION")
                         for status in statuses)
    if (expected_prepared+expected_blocked!=requested or
        type(manifest.get("prepared_models")) is not int or
        type(manifest.get("blocked_models")) is not int or
        manifest["prepared_models"]!=expected_prepared or
        manifest["blocked_models"]!=expected_blocked or
        type(manifest.get("all_models_ready")) is not bool or
        manifest["all_models_ready"]!=(expected_prepared==requested) or
        manifest.get("vitis_executed") is not False or
        manifest.get("vivado_executed") is not False or
        manifest.get("evidence")!="E0/E1-source-preparation-only"):
        raise ValueError("Suite count, preparation status or evidence label was altered")
    results=[]
    seen=set()
    for study in declared:
        name=study.get("id")
        if not isinstance(name,str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,40}",name) or name in seen:
            raise ValueError("Unsafe or duplicate suite experiment identity")
        seen.add(name)
        if study.get("status") in ("BLOCKED_ASSETS","BLOCKED_UNSUPPORTED","BLOCKED_VALIDATION"):
            if (not isinstance(study.get("reason"),str) or not study["reason"] or
                study.get("native_evidence_complete") is not False):
                raise ValueError("Blocked study lacks an explicit reason or clean native status")
            results.append({"id":name,"status":study["status"],"reason":study["reason"]})
            continue
        if study.get("status")!="E1-PREPARED":
            raise ValueError("Unknown suite study status")
        if type(study.get("ready_cases")) is not int or study["ready_cases"]<1:
            raise ValueError("Prepared suite study has no legal architecture points")
        if study.get("native_evidence_complete") is not False:
            raise ValueError("Preparation receipt cannot assert native completion")
        receipt=audit_full_experiment(root/name,require_host=require_host,
                                      require_all_axes=require_all_axes)
        prepared_manifest=json.loads((root/name/"full_experiment.json").read_text())
        if study.get("model_sha256")!=prepared_manifest.get("model_sha256"):
            raise ValueError("Suite model fingerprint disagrees with prepared experiment")
        ready=sum(x["status"]=="READY" for x in receipt["cases"])
        if ready!=study.get("ready_cases"):
            raise ValueError("Suite prepared-case count changed")
        if any(x["functional_sha256"]!=study.get("functional_sha256")
               for x in receipt["cases"] if x["status"]=="READY"):
            raise ValueError("Suite Functional Model IR identity mismatch")
        results.append({"id":name,"status":"E1-PREPARED",
            "prepared_for_native_experiments":receipt["prepared_for_native_experiments"],
            "legal_subset_prepared_for_native_execution":receipt.get(
                "legal_subset_prepared_for_native_execution",
                receipt["prepared_for_native_experiments"]),
            "infeasible_architecture_case_count":sum(
                case["status"]=="BLOCKED" for case in receipt["cases"]),
            "seven_axis_coverage_complete":receipt["seven_axis_coverage_complete"],
            "missing_axes":receipt["missing_axes"],
            "native_paper_evidence_complete":receipt["native_paper_evidence_complete"]})
    prepared=[row for row in results if row["status"]=="E1-PREPARED"]
    return {"schema":"fpgai-research/full-paper-suite-audit/v1",
        "requested_models":len(results),"blocked_models":len(results)-len(prepared),
        "all_models_source_ready":len(prepared)==len(results) and all(
            item["prepared_for_native_experiments"] for item in prepared),
        "all_prepared_models_have_legal_executable_subsets":bool(prepared) and all(
            item["legal_subset_prepared_for_native_execution"] for item in prepared),
        "all_native_paper_experiments_complete":len(prepared)==len(results) and all(
            item["prepared_for_native_experiments"] and
            item["native_paper_evidence_complete"] for item in prepared),
        "studies":results,
        "evidence":"E0/E1-audit, with native stages accepted only from attested source receipts"}
