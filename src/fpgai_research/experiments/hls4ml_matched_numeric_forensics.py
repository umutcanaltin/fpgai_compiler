"""Source-bound matched default/tuned hls4ml numerical forensics, without Vitis.

The selected numerical limit remains 0.05. Source-only evidence or an
observed CSim output cannot establish achieved II, native E2 attestation,
physical QoR, or that ReuseFactor itself caused a numerical regression.
"""
from __future__ import annotations
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import numpy as np

from fpgai_research.experiments.hls4ml_csim_native_observation import discover_selected_csim_results
from fpgai_research.experiments.hls4ml_csim_numeric import collect_csim_numeric, numeric_matrix
from fpgai_research.experiments.hls4ml_csim_pair import collect_csim_stage_pair

SCHEMA = "fpgai-research/hls4ml-matched-csim-forensics/v1"


def compare_matrices(reference, default, tuned):
    arrays = [np.asarray(x, dtype=np.float64) for x in (reference, default, tuned)]
    if any(x.shape != (16, 10) or not np.all(np.isfinite(x)) for x in arrays):
        raise ValueError("Expected finite 16-by-10 matched matrices")
    r, d, t = arrays
    if np.min(r) < 0 or np.max(r) > 1 or np.max(np.abs(r.sum(axis=1)-1)) > 0.01:
        raise ValueError("Invalid ONNX probability reference")
    expected = np.argmax(r, axis=1)
    derror = np.abs(d-r)
    terror = np.abs(t-r)
    def observed(x, differences):
        uniform = np.ptp(x, axis=1) <= 0.01
        row_sums = np.sum(x, axis=1)
        # A counterfactual row renormalization is diagnostic ONLY. Never
        # substitute it for the source-bound native CSim output or relax
        # the pre-existing raw-output max-error acceptance limit.
        can_renormalize = bool(np.all(np.isfinite(row_sums) & (row_sums > 0)))
        normalized_error = (
            float(np.max(np.abs((x / row_sums[:, None]) - r)))
            if can_renormalize else None
        )
        return {
            "top1_disagreements": int(np.sum(np.argmax(x, axis=1) != expected)),
            "max_absolute_probability_error": float(np.max(differences)),
            "mean_absolute_probability_error": float(np.mean(differences)),
            "near_uniform_output_count": int(np.sum(uniform)),
            "near_uniform_with_reference_confidence_at_least_0_8":
                int(np.sum(uniform & (np.max(r, axis=1) >= 0.8))),
            "max_row_probability_sum_error":
                float(np.max(np.abs(row_sums-1))),
            "minimum_observed_probability_row_sum": float(np.min(row_sums)),
            "maximum_observed_probability_row_sum": float(np.max(row_sums)),
            "posthoc_row_renormalization_diagnostic_only": True,
            "posthoc_row_renormalization_all_sums_positive": can_renormalize,
            "max_absolute_probability_error_after_posthoc_row_renormalization":
                normalized_error,
            "posthoc_row_renormalization_can_qualify_e2": False,
        }
    return {
        "selected_samples": 16,
        "classes": 10,
        "default": observed(d, derror),
        "tuned": observed(t, terror),
        "default_vs_tuned_top1_disagreements":
            int(np.sum(np.argmax(d, axis=1) != np.argmax(t, axis=1))),
        "default_vs_tuned_all_160_values_exactly_equal":
            bool(np.array_equal(d, t)),
        "default_vs_tuned_max_absolute_output_difference":
            float(np.max(np.abs(d-t))),
        "tuned_worse_max_error_image_count":
            int(np.sum(np.max(terror, axis=1) > np.max(derror, axis=1))),
        "tuned_disagreement_sample_indices":
            np.flatnonzero(np.argmax(t, axis=1) != expected).tolist(),
        "tuned_worst_three_sample_indices":
            np.argsort(np.max(terror, axis=1))[::-1][:3].tolist(),
        "operator_level_root_cause_identified": False,
        "hardware_qor_or_achieved_ii_measured": False,
    }


def export_forensics(default_stage, tuned_stage, output):
    dest = Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Forensics output must be a new private directory")
    pair = collect_csim_stage_pair(default_stage, tuned_stage)
    dpath = discover_selected_csim_results(default_stage)
    tpath = discover_selected_csim_results(tuned_stage)
    default = collect_csim_numeric(default_stage, dpath)
    tuned = collect_csim_numeric(tuned_stage, tpath)
    if (default["model_kind"] != pair["model_kind"]
            or tuned["model_kind"] != pair["model_kind"]
            or default["selected_inputs_sha256"] != tuned["selected_inputs_sha256"]):
        raise ValueError("Numerical CSim sources or selected cohort do not match")
    source = Path(default_stage) / "project" / "tb_data" / "tb_output_predictions.dat"
    metrics = compare_matrices(
        numeric_matrix(source.read_text(encoding="utf-8")),
        numeric_matrix(dpath.read_text(encoding="utf-8")),
        numeric_matrix(tpath.read_text(encoding="utf-8")))
    report = {
        "schema": SCHEMA,
        "evidence": "E2-MATCHED-NUMERICAL-OBSERVATION-ONLY",
        "model_kind": pair["model_kind"],
        "matched_inputs_and_onnx_reference": True,
        "default_result_sha256": default["csim_result_file_sha256"],
        "tuned_result_sha256": tuned["csim_result_file_sha256"],
        "default_selected_policy_passed":
            default["measurements"]["selected_numeric_policy_passed"],
        "tuned_selected_policy_passed":
            tuned["measurements"]["selected_numeric_policy_passed"],
        "exploratory_max_abs_probability_error_limit": 0.05,
        "metrics": metrics,
        "independently_verified_native_vitis_e2": False,
        "hardware_comparison_eligible": False,
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".hls4ml-numeric-forensics-", dir=dest.parent) as tmp:
        staged = Path(tmp) / "result"
        staged.mkdir()
        (staged/"forensics_private.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n",
            encoding="utf-8")
        if dest.exists() or dest.is_symlink():
            raise FileExistsError("Forensics result appeared concurrently")
        staged.rename(dest)
    return report
