"""Independent read-only replay of successful 4096-entry hls4ml native CSim.

This checks original 18/6 lineage, exactly one new table-size source edit,
both Vitis native receipt and original log bytes, raw CSim output, the
original selected 16x10 reference and all frozen numerical criteria.

A selected 16-image numeric acceptance does NOT qualify model-wide E2,
full 10000-image fixed-point accuracy, cross-compiler fairness or QoR.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from fpgai_research.experiments.hls4ml_softmax_inverse_table_size import (
    _check, _hash, _regular, RUN_SCHEMA, FROZEN_ABS_LIMIT,
    FROZEN_SUM_LIMIT,
)
from fpgai_research.experiments.hls4ml_csim_native_observation import (
    discover_selected_csim_results,
)
from fpgai_research.experiments.hls4ml_csim_numeric import (
    numeric_matrix, compare_numeric_outputs,
)

SCHEMA="fpgai-research/hls4ml-native-4096-selected-numeric-replay/v1"


def _native_record(path):
    source=_regular(path,2*1024*1024)
    record=json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(record,dict):
        raise ValueError("Native receipt must be a JSON object")
    return record,sha256(source.read_bytes()).hexdigest()


def _same_float(actual,reported,field):
    if (type(reported) not in (float,int)
            or not np.isfinite(reported)
            or abs(float(actual)-float(reported))>1e-9):
        raise ValueError("Native numerical field disagrees with raw data: "+field)


def audit_native_4096(stage, native_run):
    staged=Path(stage)
    completed=Path(native_run)
    if (completed.is_symlink() or not completed.is_dir()):
        raise ValueError("Native result must be an ordinary private directory")
    prepared=_check(staged)  # also attests prior 18/6 original CSim lineage
    record,receipt_sha=_native_record(completed/"native_result.json")
    log_path=_regular(completed/"native_vitis_csim.log",16*1024*1024)
    log_data=log_path.read_bytes()
    if (record.get("schema")!=RUN_SCHEMA
            or record.get("native_vitis_csim_attempted") is not True
            or record.get("native_vitis_csim_completed") is not True
            or record.get("source_rechecked_after_vitis") is not True
            or record.get("native_e2_qualified_by_driver") is not False
            or record.get("external_hardware_qor_eligible") is not False
            or record.get("vendor_hls_synthesis_executed") is not False
            or record.get("vivado_executed") is not False
            or record.get("stage_receipt_sha256")!=_hash(
                staged/"experiment.json",2*1024*1024)
            or record.get("native_log_sha256")!=sha256(log_data).hexdigest()
            or b"CSim done with 0 errors" not in log_data):
        raise ValueError("Original 4096-entry native Vitis CSim receipt/log did not attest")
    measurements=record.get("numerical_observation")
    if (not isinstance(measurements,dict)
            or measurements.get("selected_samples")!=16
            or measurements.get("classes")!=10
            or measurements.get("full_10000_image_fpga_accuracy_measured") is not False
            or measurements.get("qualified_vitis_csim_e2") is not False
            or measurements.get("posthoc_normalization_performed") is not False
            or measurements.get("source_vs_csim_selected_numerical_observation_only")
                is not True
            or measurements.get("selected_numeric_policy_passed") is not True
            or measurements.get("selected_raw_and_probability_semantics_passed")
                is not True
            or measurements.get("out_of_range_probability_count")!=0):
        raise ValueError("Native selected numerical result is incomplete or not accepted")
    candidate_file=discover_selected_csim_results(staged)
    candidate_raw=_regular(candidate_file).read_bytes()
    if sha256(candidate_raw).hexdigest()!=measurements.get("native_output_sha256"):
        raise ValueError("Native CSim raw predictions mutated after original receipt")
    prediction=numeric_matrix(candidate_raw.decode("utf-8"))
    ref_path=_regular(staged/"project"/"tb_data"/"tb_output_predictions.dat")
    reference=numeric_matrix(ref_path.read_text())
    parent=Path(prepared["parent_stage"])
    original_file=discover_selected_csim_results(parent)
    prior=numeric_matrix(_regular(original_file).read_text())
    if (np.min(reference)<0 or np.max(reference)>1
            or np.max(np.abs(reference.sum(axis=1)-1))>1e-2):
        raise ValueError("Frozen original reference probabilities are not normalized")
    fresh=compare_numeric_outputs(
        reference,prediction,absolute_limit=FROZEN_ABS_LIMIT)
    sums=prediction.sum(axis=1)
    max_sum=float(np.max(np.abs(sums-1)))
    minimum=float(np.min(sums))
    maximum=float(np.max(sums))
    invalid=int(np.count_nonzero((prediction<0)|(prediction>1)))
    delta=float(np.max(np.abs(prediction-prior)))
    for field,value in (
        ("selected_probability_max_absolute_difference",
         fresh["selected_probability_max_absolute_difference"]),
        ("selected_probability_mean_absolute_difference",
         fresh["selected_probability_mean_absolute_difference"]),
        ("selected_policy_max_absolute_difference",FROZEN_ABS_LIMIT),
        ("max_abs_probability_row_sum_error",max_sum),
        ("minimum_probability_row_sum",minimum),
        ("maximum_probability_row_sum",maximum),
        ("new_vs_18_6_max_abs_output_delta",delta),
    ):
        _same_float(value,measurements.get(field),field)
    if (measurements["selected_top1_disagreements"]!=fresh[
            "selected_top1_disagreements"]
            or measurements["out_of_range_probability_count"]!=invalid):
        raise ValueError("Native classification/range metrics changed")
    accepted=bool(
        fresh["selected_numeric_policy_passed"]
        and max_sum<=FROZEN_SUM_LIMIT
        and invalid==0
    )
    if not accepted:
        raise ValueError("Actual native output fails frozen selected numerical policy")
    return {
        "schema":SCHEMA,
        "model_kind":"mlp",
        "source_stage_receipt_sha256":_hash(
            staged/"experiment.json",2*1024*1024),
        "native_result_receipt_sha256":receipt_sha,
        "native_vitis_log_sha256":sha256(log_data).hexdigest(),
        "actual_160_native_output_sha256":sha256(candidate_raw).hexdigest(),
        "original_16x10_reference_sha256":sha256(
            ref_path.read_bytes()).hexdigest(),
        "prior_18_6_native_output_sha256":_hash(original_file),
        "source_and_prior_native_lineage_rechecked":True,
        "new_native_log_and_output_rechecked":True,
        "selected_sample_count":16,"class_count":10,
        "max_abs_probability_error":fresh[
            "selected_probability_max_absolute_difference"],
        "mean_abs_probability_error":fresh[
            "selected_probability_mean_absolute_difference"],
        "max_abs_row_sum_error":max_sum,
        "minimum_row_sum":minimum,"maximum_row_sum":maximum,
        "top1_disagreements":fresh["selected_top1_disagreements"],
        "out_of_range_probability_count":invalid,
        "raw_output_delta_vs_18_6":delta,
        "frozen_abs_error_limit":FROZEN_ABS_LIMIT,
        "frozen_row_sum_limit":FROZEN_SUM_LIMIT,
        "qualified_selected_16_image_native_numeric_acceptance":True,
        "qualified_full_model_external_numeric_e2":False,
        "qualified_full_10000_image_fixed_point":False,
        "qualified_hardware_qor_comparison":False,
        "posthoc_normalization_used":False,
        "new_vendor_tools_executed_by_audit":False,
        "limitations":(
            "Attests existing source-bound original 16-image native CSim "
            "numerical acceptance only. Independent ONNX-to-weights "
            "lineage, broad model fixed-point evaluation, physical "
            "routed results, equivalently configured FPGAI and "
            "hls4ml competitors, and full paper qualification "
            "remain separate. No new vendor execution."),
    }


def export_native_4096_audit(stage,native_run,output):
    dest=Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("New immutable 4096 replay directory required")
    report=audit_native_4096(stage,native_run)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix=".4096-native-replay-",dir=dest.parent) as tmp:
        draft=Path(tmp)/"report"
        draft.mkdir()
        (draft/"selected_16_native_replay.json").write_text(
            json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
        summary=[
            "# Original hls4ml 4096-entry stable CSim: independent replay",
            "",
            "The original source, prior 18/6 Vitis CSim evidence, "
            "new Vitis log and all 160 raw prediction values "
            "were independently rechecked without running tools.",
            "",
            f"Selected raw max error: {report['max_abs_probability_error']:.10g}",
            f"Maximum row-sum error: {report['max_abs_row_sum_error']:.10g}",
            f"Top-1 disagreements: {report['top1_disagreements']}",
            "",
            "Selected 16-image native numeric acceptance: QUALIFIED.",
            "Full model external numeric E2: NOT QUALIFIED.",
            "10k-image fixed-point evaluation: NOT MEASURED.",
            "External routed hardware QoR: NOT QUALIFIED.",
            "",
            "No normalization, paper-score promotion, HLS synth or Vivado.",
            "",
        ]
        (draft/"summary.md").write_text("\n".join(summary))
        if dest.exists() or dest.is_symlink():
            raise FileExistsError("Replay destination appeared during publication")
        draft.rename(dest)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",required=True)
    parser.add_argument("--native-run",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    result=export_native_4096_audit(
        args.stage,args.native_run,args.output)
    print(json.dumps({
        "schema":result["schema"],
        "qualified_selected_16_image_native_numeric_acceptance":
            result["qualified_selected_16_image_native_numeric_acceptance"],
        "qualified_full_model_external_numeric_e2":False,
        "raw_max_error":result["max_abs_probability_error"],
        "row_sum_error":result["max_abs_row_sum_error"],
        "top1_disagreements":result["top1_disagreements"],
        "vendor_tools_executed":False,
    },indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
