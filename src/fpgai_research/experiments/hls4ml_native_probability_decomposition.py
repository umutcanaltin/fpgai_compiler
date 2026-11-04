"""Explain raw hls4ml probability errors without changing native E2 acceptance.

Read original source-matched 16x10 hls4ml RF1/RF2 CSim outputs only
AFTER independently replay-auditing the two real Vitis invocation logs,
original cloned C++ sources and numeric result files.

For each row, raw_result - reference = probability_mass_term +
distribution_shape_term, where probability_mass_term=(row_sum-1)*ref,
and distribution_shape_term=row_sum*(raw_result/row_sum - ref).
This decomposition is algebraic (when row_sum>0), NOT a verified root
cause or a source-level fix. NEVER use posthoc normalization for E2.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from fpgai_research.experiments.hls4ml_native_pair_reaudit import audit_native_pair
from fpgai_research.experiments.hls4ml_csim_native_observation import (
    discover_selected_csim_results,
)
from fpgai_research.experiments.hls4ml_csim_numeric import numeric_matrix

SCHEMA="fpgai-research/hls4ml-native-raw-probability-decomposition/v1"
ABS_LIMIT=0.05
SUM_LIMIT=0.01


def _validate_arrays(reference,prediction):
    r=np.asarray(reference,dtype=np.float64)
    y=np.asarray(prediction,dtype=np.float64)
    if (r.shape!=(16,10) or y.shape!=(16,10)
            or not np.all(np.isfinite(r)) or not np.all(np.isfinite(y))):
        raise ValueError("Require exact finite aligned native 16x10 outputs")
    if (np.min(r)<0 or np.max(r)>1
            or np.max(np.abs(np.sum(r,axis=1)-1))>SUM_LIMIT):
        raise ValueError("Reference is not a valid normalized probability matrix")
    return r,y


def decompose_probability_residuals(reference,prediction):
    """Private, per-image algebra; absolute raw policy always preserved."""
    r,y=_validate_arrays(reference,prediction)
    sums=np.sum(y,axis=1)
    residual=y-r
    if not np.all(np.isfinite(sums)):
        raise ValueError("Nonfinite native output sum")
    raw_max=float(np.max(np.abs(residual)))
    raw_top1=int(np.count_nonzero(np.argmax(r,axis=1)!=np.argmax(y,axis=1)))
    positive=bool(np.all(sums>0))
    records=[]
    for i in range(16):
        item={
            "sample_index":i,
            "native_row_sum":float(sums[i]),
            "probability_mass_error":float(sums[i]-1),
            "max_absolute_raw_probability_error":float(np.max(np.abs(residual[i]))),
            "mean_absolute_raw_probability_error":float(np.mean(np.abs(residual[i]))),
            "top1_disagreement":bool(np.argmax(r[i])!=np.argmax(y[i])),
            "native_probability_range_violation_count":int(
                np.count_nonzero((y[i]<0)|(y[i]>1))),
            "normalization_counterfactual_only":True,
            "maximum_algebraic_decomposition_residual":None,
            "maximum_abs_mass_term":None,
            "maximum_abs_distribution_term":None,
            "max_abs_posthoc_normalized_error":None,
        }
        if sums[i]>0:
            mass=(sums[i]-1)*r[i]
            shape=y[i]-sums[i]*r[i]
            normal=y[i]/sums[i]
            reconstructed=mass+shape
            reconstruction_error=float(np.max(np.abs(reconstructed-residual[i])))
            if reconstruction_error>1e-10:
                raise ValueError("Raw residual algebra failed numerical audit")
            item.update({
                "maximum_algebraic_decomposition_residual":reconstruction_error,
                "maximum_abs_mass_term":float(np.max(np.abs(mass))),
                "maximum_abs_distribution_term":float(np.max(np.abs(shape))),
                "max_abs_posthoc_normalized_error":float(
                    np.max(np.abs(normal-r[i]))),
            })
        records.append(item)
    accepted=bool(raw_top1==0 and raw_max<=ABS_LIMIT)
    semantics=bool(
        np.max(np.abs(sums-1))<=SUM_LIMIT
        and all(x["native_probability_range_violation_count"]==0
                for x in records))
    worst=sorted(records,key=lambda row:(
        -row["max_absolute_raw_probability_error"],row["sample_index"]))[:5]
    return {
        "sample_count":16,"class_count":10,
        "pre_registered_raw_max_error_limit":ABS_LIMIT,
        "pre_registered_probability_sum_limit":SUM_LIMIT,
        "max_absolute_raw_probability_error":raw_max,
        "raw_top1_disagreements":raw_top1,
        "max_absolute_probability_row_sum_error":float(
            np.max(np.abs(sums-1))),
        "raw_selected_numeric_policy_passed":accepted,
        "raw_probability_semantics_valid":semantics,
        "raw_numerical_and_probability_semantics_passed":bool(
            accepted and semantics),
        "all_native_row_sums_positive":positive,
        "all_rows":records,
        "worst_five_rows":worst,
        "native_output_can_be_replaced_by_normalized_counterfactual":False,
        "algebraic_decomposition_proves_error_cause":False,
        "qualified_external_numeric_E2_by_this_analysis":False,
    }


def analyze_native_probability_failure(stage_root,global_default,native_run):
    # This is an actual native replay audit: not a trust-me user-provided
    # source path. Original tool logs, CSim run receipt and both native
    # numeric observations are verified before reading input arrays.
    attested=audit_native_pair(stage_root,global_default,native_run)
    if (attested.get("real_native_csim_logs_fingerprint_rechecked") is not True
            or attested.get("real_native_csim_log_count")!=2
            or attested.get("native_csim_observed_in_source_attributed_logs") is not True
            or attested.get("qualified_external_numeric_e2") is not False):
        raise ValueError("Original native pair replay audit did not qualify")
    root=Path(stage_root)
    ref_path=root/"csim_rf1"/"project"/"tb_data"/"tb_output_predictions.dat"
    if (ref_path.is_symlink() or not ref_path.is_file()
            or not 0<ref_path.stat().st_size<=4*1024*1024):
        raise ValueError("Missing original source-verified ONNX probabilities")
    ref_raw=ref_path.read_bytes()
    reference=numeric_matrix(ref_raw.decode("utf-8"))
    variants={}
    for rf,key in ((1,"rf1"),(2,"rf2")):
        file=discover_selected_csim_results(root/("csim_rf"+str(rf)))
        raw=file.read_bytes()
        if sha256(raw).hexdigest()!=attested[key+"_csim_result_sha256"]:
            raise ValueError("Native CSim result mutated after independent replay audit")
        analyzed=decompose_probability_residuals(
            reference,numeric_matrix(raw.decode("utf-8")))
        if (analyzed["raw_selected_numeric_policy_passed"]
                is not attested[key+"_selected_policy_passed"]):
            raise ValueError("Decomposition does not match original numeric acceptance")
        variants[key]={
            "native_result_sha256":sha256(raw).hexdigest(),
            "decomposition":analyzed,
        }
    return {
        "schema":SCHEMA,
        "model_kind":"mlp",
        "original_native_replay_receipt_sha256":
            attested["native_run_receipt_sha256"],
        "reference_testbench_sha256":sha256(ref_raw).hexdigest(),
        "real_native_logs_and_original_generated_sources_reaudited":True,
        "source_config_isolated":attested["source_config_isolated"],
        "preconversion_request_isolated":attested[
            "preconversion_request_isolated"],
        "additional_postconversion_rf_keys":attested[
            "additional_postconversion_rf_keys"],
        "variants":variants,
        "vendor_tools_executed_by_analysis":False,
        "native_E2_qualification_by_this_analysis":False,
        "hardware_qor_comparison_eligible":False,
        "root_cause_identified":False,
        "limitations":(
            "Residual mass/shape algebra diagnoses raw numerical behavior, "
            "not the first divergent native operator. No posthoc row "
            "renormalization qualifies E2. Pre-registered absolute "
            "probability error limit 0.05 and original source settings "
            "remain unchanged. Original Vitis CSim was real and its "
            "fingerprints were independently reaudited; no Vitis/Vivado "
            "runs were performed by this analysis. hls4ml reuse RF "
            "is not isolated where backend-added settings differ."),
    }


def export_native_probability_failure(stage_root,global_default,native_run,output):
    dest=Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Use a new immutable private numeric diagnosis directory")
    report=analyze_native_probability_failure(
        stage_root,global_default,native_run)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix=".probability-mass-",dir=dest.parent) as t:
        folder=Path(t)/"report"
        folder.mkdir()
        (folder/"private_probability_decomposition.json").write_text(
            json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
        lines=[
            "# Source-replayed hls4ml RF1/RF2 raw probability decomposition","",
            "The 16 original CSim probability rows and native logfile/source "
            "fingerprints were rechecked. This private output is NOT a "
            "qualified cross-compiler native E2 or routed hardware result.",
            "",
            "Raw acceptance remains maximum absolute probability error "
            "<= 0.05 and zero top-1 disagreements. Probability row sums "
            "must separately meet the pre-registered 0.01 semantics bound.",
            "",
            "| Variant | Raw max abs error | Max row-sum error | Top-1 disagreements | Raw policy | Probability semantics |",
            "|---|---:|---:|---:|---|---|",
        ]
        for label,item in report["variants"].items():
            x=item["decomposition"]
            lines.append(
                f"| {label} | {x['max_absolute_raw_probability_error']:.10g} | "
                f"{x['max_absolute_probability_row_sum_error']:.10g} | "
                f"{x['raw_top1_disagreements']} | "
                f"{x['raw_selected_numeric_policy_passed']} | "
                f"{x['raw_probability_semantics_valid']} |")
        lines.extend(["",
            "## Worst five original input rows per RF variant","",
            "Mass term = (row sum - 1) * reference probability. "
            "Shape term = observed - (row sum * reference). "
            "Their exact sum reconstructs the original raw residual. "
            "These are descriptive algebra terms, NOT proof that "
            "Softmax or a specific operator caused the discrepancy.",
            "",
            "| Variant | Sample | Raw max abs error | Row-sum error | Max mass term | Max shape term | Posthoc normalized max error (not accepted) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for label,item in report["variants"].items():
            for row in item["decomposition"]["worst_five_rows"]:
                def fmt(v):
                    return "—" if v is None else f"{v:.10g}"
                lines.append(
                    f"| {label} | {row['sample_index']} | "
                    f"{fmt(row['max_absolute_raw_probability_error'])} | "
                    f"{fmt(row['probability_mass_error'])} | "
                    f"{fmt(row['maximum_abs_mass_term'])} | "
                    f"{fmt(row['maximum_abs_distribution_term'])} | "
                    f"{fmt(row['max_abs_posthoc_normalized_error'])} |")
        lines.extend(["",
            "**No root cause established. No altered testbench, native "
            "source, reference, threshold or vendor evidence.** "
            "A posthoc normalized value is diagnostic only, never "
            "a replacement for real CSim output.",""])
        (folder/"summary.md").write_text("\n".join(lines))
        if dest.exists() or dest.is_symlink():
            raise FileExistsError("Native numerical forensics destination appeared")
        folder.rename(dest)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage-root",required=True)
    p.add_argument("--global-default",required=True)
    p.add_argument("--native-run",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args(argv)
    result=export_native_probability_failure(
        a.stage_root,a.global_default,a.native_run,a.output)
    print(json.dumps({
        "schema":SCHEMA,
        "native_source_and_logs_reaudited":
            result["real_native_logs_and_original_generated_sources_reaudited"],
        "raw_numerics":{
            k:{
                "max_abs":v["decomposition"]["max_absolute_raw_probability_error"],
                "max_row_sum_error":
                    v["decomposition"]["max_absolute_probability_row_sum_error"],
                "raw_policy_passed":
                    v["decomposition"]["raw_selected_numeric_policy_passed"],
            } for k,v in result["variants"].items()},
        "vendor_tools_executed_by_analysis":False,
    },sort_keys=True,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
