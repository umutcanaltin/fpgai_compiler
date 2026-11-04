"""Fail-closed, exploratory FPGAI-vs-hls4ml numerical comparison on identical samples.

No Vitis/Vivado execution or native-source attestation is performed here.
Only independently qualified E2 and E3/E4 receipts can support a paper-level
claim of native numerical acceptance or hardware superiority. Never choose
a threshold after observing results or silently normalize native outputs.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

SCHEMA="fpgai-research/matched-compiler-numeric-manifest/v1"
REPORT_SCHEMA="fpgai-research/matched-compiler-numeric-comparison/v1"
PRECISION_TOLERANCE=0.05
ROW_SUM_TOLERANCE=0.01
MAX_FILE_BYTES=128*1024*1024
KINDS=("mlp","cnn")
_DIGEST_CHARS=frozenset("0123456789abcdef")


def _digest(value):
    return isinstance(value,str) and len(value)==64 and set(value)<=_DIGEST_CHARS


def _load_json(path):
    p=Path(path)
    if p.is_symlink() or not p.is_file() or not 0<p.stat().st_size<=1024*1024:
        raise ValueError("Expected bounded ordinary matched-numeric manifest JSON")
    v=json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(v,dict) or v.get("schema")!=SCHEMA:
        raise ValueError("Unknown matched-numeric manifest schema")
    return v


def _load_numpy(parent,spec,name):
    if (not isinstance(spec,dict) or set(spec)!={"file","sha256"}
            or not _digest(spec["sha256"])):
        raise ValueError(f"{name} has no exact source artifact fingerprint")
    rel=Path(spec["file"]) if isinstance(spec["file"],str) else Path("/")
    if (rel.is_absolute() or ".." in rel.parts or rel.suffix!=".npy"
            or len(rel.parts)!=1):
        raise ValueError(f"Unsafe {name} relative array file")
    f=parent/rel
    if (f.is_symlink() or not f.is_file()
            or not 0<f.stat().st_size<=MAX_FILE_BYTES):
        raise ValueError(f"Missing safe {name} array")
    raw=f.read_bytes()
    if sha256(raw).hexdigest()!=spec["sha256"]:
        raise ValueError(f"{name} array fingerprint mismatch")
    arr=np.load(f,allow_pickle=False)
    if arr.dtype.kind not in "iuf" or not np.all(np.isfinite(arr)):
        raise ValueError(f"Invalid {name} values")
    return np.asarray(arr), spec["sha256"]


def _validate_probabilities(a,name,n):
    if a.shape!=(n,10) or a.dtype.kind!="f":
        raise ValueError(f"{name} must have n x 10 float probabilities")
    # A native CSim output outside [0,1] is a NEGATIVE numerical
    # observation. Preserve the raw values and report the violation;
    # never clip, normalize or silently drop a failing competitor.
    return np.asarray(a,dtype=np.float64)


def _score(ref,observed,truth):
    absolute=np.abs(ref-observed)
    sums=observed.sum(axis=1)
    top=np.argmax(observed,axis=1)
    reference_top=np.argmax(ref,axis=1)
    sum_error=float(np.max(np.abs(sums-1)))
    invalid_entries=int(np.count_nonzero((observed<0)|(observed>1)))
    prob_valid=bool(sum_error<=ROW_SUM_TOLERANCE and invalid_entries==0)
    max_err=float(np.max(absolute))
    top1_different=int(np.sum(top!=reference_top))
    raw_accepted=bool(top1_different==0 and max_err<=PRECISION_TOLERANCE)
    can_normalize=bool(np.all(sums>0))
    return {
        "samples":int(len(ref)),
        "top1_disagreements_vs_reference":top1_different,
        "top1_accuracy_vs_labels":float(np.mean(top==truth)),
        "max_absolute_probability_error":max_err,
        "mean_absolute_probability_error":float(np.mean(absolute)),
        "rmse_probability_error":float(np.sqrt(np.mean(np.square(absolute)))),
        "maximum_absolute_probability_row_sum_error":sum_error,
        "raw_probability_semantics_valid":prob_valid,
        "out_of_range_probability_values":invalid_entries,
        "original_raw_numeric_policy_passed":raw_accepted,
        "strict_raw_probability_qualification":
            bool(raw_accepted and prob_valid),
        "posthoc_normalization_diagnostic_only":True,
        "posthoc_max_error_if_renormalized":
            float(np.max(np.abs(observed/sums[:,None]-ref)))
            if can_normalize else None,
        "posthoc_can_qualify_numeric_e2":False,
    }


def compare_arrays(reference, fpgai, hls4ml, labels, *, cohort_min=16):
    r=np.asarray(reference)
    a=np.asarray(fpgai)
    b=np.asarray(hls4ml)
    truth=np.asarray(labels)
    if (r.ndim!=2 or len(r)<cohort_min or len(r)>10000
            or not isinstance(cohort_min,int) or cohort_min<1):
        raise ValueError("Invalid matched sample count")
    n=len(r)
    if (truth.shape!=(n,) or truth.dtype.kind not in "iu"
            or np.min(truth)<0 or np.max(truth)>9):
        raise ValueError("Missing matching exact class labels")
    r=_validate_probabilities(r,"ONNX reference",n)
    a=_validate_probabilities(a,"FPGAI",n)
    b=_validate_probabilities(b,"hls4ml",n)
    if np.max(np.abs(np.sum(r,axis=1)-1))>ROW_SUM_TOLERANCE:
        raise ValueError("Reference model output not normalized")
    # Per-image (not per-class) errors are paired, not independent samples.
    abs_a=np.mean(np.abs(a-r),axis=1)
    abs_b=np.mean(np.abs(b-r),axis=1)
    delta=abs_b-abs_a  # positive means FPGAI has lower average error
    aa=_score(r,a,truth)
    bb=_score(r,b,truth)
    strictly_better=bool(
        aa["mean_absolute_probability_error"] <
        bb["mean_absolute_probability_error"] and
        aa["max_absolute_probability_error"] <
        bb["max_absolute_probability_error"] and
        aa["top1_accuracy_vs_labels"]>=bb["top1_accuracy_vs_labels"] and
        aa["original_raw_numeric_policy_passed"] and
        bb["original_raw_numeric_policy_passed"] and
        aa["raw_probability_semantics_valid"] and
        bb["raw_probability_semantics_valid"]
    )
    return {
        "schema":REPORT_SCHEMA,
        "matched_samples":n,"classes":10,
        "reference_top1_accuracy":float(np.mean(np.argmax(r,axis=1)==truth)),
        "fpgaI":aa,"hls4ml":bb,
        "paired_mean_per_image_absolute_error_improvement_hls_minus_fpgai":
            float(np.mean(delta)),
        "images_fpgai_strictly_lower_mae":int(np.sum(abs_a<abs_b)),
        "images_hls4ml_strictly_lower_mae":int(np.sum(abs_b<abs_a)),
        "images_identical_mae":int(np.sum(abs_b==abs_a)),
        "selected_cohort_both_numerical_policies_passed":
            bool(aa["original_raw_numeric_policy_passed"]
                 and bb["original_raw_numeric_policy_passed"]),
        "selected_cohort_both_outputs_valid_probabilities":
            bool(aa["raw_probability_semantics_valid"]
                 and bb["raw_probability_semantics_valid"]),
        "selected_cohort_strict_fpgai_numeric_dominance_observed":
            strictly_better,
        "official_10000_image_cohort":n==10000,
        "statistical_population_superiority_proven":False,
        "native_source_bound_e2_qualified_by_this_analysis":False,
        "hardware_qor_or_cross_compiler_superiority_proven":False,
        "vendor_tools_executed_by_analysis":False,
    }


def compare_manifest(manifest_file):
    source=Path(manifest_file)
    info=_load_json(source)
    if (info.get("model_kind") not in KINDS
            or not _digest(info.get("original_onnx_sha256"))
            or not _digest(info.get("ordered_input_cohort_sha256"))
            or info.get("output_semantics")!="post_softmax_probability"
            or info.get("native_execution_independently_attested") is not False
            or info.get("external_hardware_qor_claimed") is not False
            or set(info.get("arrays",{}))!={"reference","fpgai","hls4ml","labels"}):
        raise ValueError("Incomplete source/context/semantic qualification of comparator")
    parent=source.parent
    # Do not trust a model/cohort hash written in metadata alone:
    # hash the real original ONNX and ordered raw input tensors in
    # the same private immutable bundle.
    source_name=info.get("original_onnx_file")
    if (not isinstance(source_name,str) or
            source_name!="original_model.onnx"):
        raise ValueError("Matched comparator requires a pinned original ONNX source")
    source_model=parent/source_name
    if (source_model.is_symlink() or not source_model.is_file()
            or not 0<source_model.stat().st_size<=MAX_FILE_BYTES
            or sha256(source_model.read_bytes()).hexdigest()
               !=info["original_onnx_sha256"]):
        raise ValueError("Original ONNX model source fingerprint mismatch")
    cohort,cohort_hash=_load_numpy(
        parent,info.get("ordered_input_cohort"),"ordered_input_cohort")
    if (cohort.ndim<2 or cohort.dtype!=np.float32
            or cohort_hash!=info["ordered_input_cohort_sha256"]):
        raise ValueError("Ordered source cohort bytes or precision do not match")
    observations={}
    digests={}
    for name,spec in info["arrays"].items():
        observations[name],digests[name]=_load_numpy(parent,spec,name)
    # Two truly identical compiler outputs are scientifically meaningful,
    # so never reject a comparison merely because their digests match.
    if len(cohort)!=len(observations["reference"]):
        raise ValueError("Ordered input cohort count mismatches numeric predictions")
    report=compare_arrays(observations["reference"],observations["fpgai"],
                          observations["hls4ml"],observations["labels"])
    report.update({
        "source_manifest_sha256":sha256(source.read_bytes()).hexdigest(),
        "original_onnx_sha256":info["original_onnx_sha256"],
        "ordered_input_cohort_sha256":info["ordered_input_cohort_sha256"],
        "source_array_sha256":digests,
        "actual_original_onnx_and_ordered_input_bytes_verified":True,
        "model_kind":info["model_kind"],
        "numerical_outcome_only_not_independent_native_e2_attestation":True,
        "matched_tool_versions_or_fpga_part_not_proven_by_manifest":True,
        "original_trained_weights_in_native_sources_independently_verified":False,
    })
    return report


def export_matched_numeric(manifest_file,output):
    target=Path(output)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Matched numerical comparison output must be new")
    report=compare_manifest(manifest_file)
    target.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix=".matched-numeric-",dir=target.parent) as tmp:
        folder=Path(tmp)/"report"
        folder.mkdir()
        (folder/"numeric_comparison.json").write_text(
            json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n",
            encoding="utf-8")
        if target.exists() or target.is_symlink():
            raise FileExistsError("Matched numerical comparison destination appeared")
        folder.rename(target)
    return report
