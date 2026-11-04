"""Source-pinned paired FPGAI/hls4ml numerical statistics (not native attestation).

This is a second-level analysis of an already verified immutable matched
numeric manifest. It cannot prove original-to-native weight lineage, Vitis
execution, hardware QoR or cross-compiler superiority.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


def exact_mcnemar_pvalue(a_only: int, b_only: int) -> float:
    """Exact two-sided binomial p-value for paired correctness discordances."""
    if min(a_only, b_only) < 0:
        raise ValueError("Counts must be nonnegative")
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    logs = [
        math.lgamma(n + 1) - math.lgamma(i + 1) -
        math.lgamma(n - i + 1) - n * math.log(2)
        for i in range(k + 1)
    ]
    mx = max(logs)
    return min(1.0, 2.0 * math.exp(mx) * math.fsum(math.exp(x - mx) for x in logs))


def paired_numeric_science(reference, fpgai, hls4ml, labels,
                           *, seed=20261027, resamples=2000):
    """Per-image paired diagnostics; bootstrap uncertainty is exploratory."""
    ref, a, b = (np.asarray(v) for v in (reference, fpgai, hls4ml))
    truth = np.asarray(labels)
    if (ref.ndim != 2 or ref.shape[1] != 10 or not 1 <= len(ref) <= 10000
            or a.shape != ref.shape or b.shape != ref.shape
            or truth.shape != (len(ref),)
            or any(v.dtype.kind != "f" or not np.isfinite(v).all()
                   for v in (ref, a, b))
            or truth.dtype.kind not in "iu" or (truth < 0).any()
            or (truth > 9).any() or type(seed) is not int
            or type(resamples) is not int or not 1 <= resamples <= 10000):
        raise ValueError("Requires ordered finite float Nx10 rows and integer labels 0..9")
    if np.max(np.abs(ref.sum(axis=1) - 1.0)) > 0.01:
        raise ValueError("Reference probabilities not normalized")
    delta = np.abs(b - ref).mean(axis=1) - np.abs(a - ref).mean(axis=1)
    ref_top, a_top, b_top = (np.argmax(x, axis=1) for x in (ref, a, b))
    ac = a_top == truth
    bc = b_top == truth
    a_only = int(np.count_nonzero(ac & ~bc))
    b_only = int(np.count_nonzero(bc & ~ac))
    rng = np.random.default_rng(seed)
    boot = np.empty(resamples, dtype=np.float64)
    for i in range(resamples):
        boot[i] = np.mean(delta[rng.integers(0, len(delta), size=len(delta))])
    return {
        "schema": "fpgai-research/paired-numeric-science/v1",
        "samples": int(len(ref)),
        "seed": seed,
        "bootstrap_resamples": resamples,
        "reference_top1_accuracy": float(np.mean(ref_top == truth)),
        "fpgai_top1_accuracy": float(np.mean(ac)),
        "hls4ml_top1_accuracy": float(np.mean(bc)),
        "fpgai_disagreements_vs_reference": int(np.count_nonzero(a_top != ref_top)),
        "hls4ml_disagreements_vs_reference": int(np.count_nonzero(b_top != ref_top)),
        "accuracy_difference_fpgai_minus_hls4ml": float(np.mean(ac) - np.mean(bc)),
        "fpgai_only_correct": a_only,
        "hls4ml_only_correct": b_only,
        "mcnemar_exact_two_sided_p_exploratory": exact_mcnemar_pvalue(a_only, b_only),
        "per_image_mean_mae_improvement_hls_minus_fpgai": float(np.mean(delta)),
        "paired_image_bootstrap_95pct_interval": [
            float(x) for x in np.percentile(boot, [2.5, 97.5])
        ],
        "fpgai_lower_mae_images": int(np.count_nonzero(delta > 0)),
        "hls4ml_lower_mae_images": int(np.count_nonzero(delta < 0)),
        "equal_mae_images": int(np.count_nonzero(delta == 0)),
        "fpgai_raw_max_abs_probability_error": float(np.max(np.abs(a - ref))),
        "hls4ml_raw_max_abs_probability_error": float(np.max(np.abs(b - ref))),
        "fpgai_max_probability_row_sum_error": float(np.max(np.abs(a.sum(axis=1) - 1))),
        "hls4ml_max_probability_row_sum_error": float(np.max(np.abs(b.sum(axis=1) - 1))),
        "fpgai_out_of_range_values": int(np.count_nonzero((a < 0) | (a > 1))),
        "hls4ml_out_of_range_values": int(np.count_nonzero((b < 0) | (b > 1))),
        "image_bootstrap_does_not_imply_population_superiority": True,
        "native_e2_proven_by_analysis": False,
        "hardware_qor_or_superiority_proven": False,
        "vendor_tools_executed": False,
    }


def compare_pinned_manifest(manifest):
    """Independently reuse the existing fail-closed source-and-cohort verifier."""
    from fpgai_research.experiments.matched_compiler_numerics import (
        compare_manifest, _load_json, _load_numpy,
    )

    source = Path(manifest)
    baseline = compare_manifest(source)
    info = _load_json(source)
    arrays = {
        key: _load_numpy(source.parent, spec, key)[0]
        for key, spec in info["arrays"].items()
    }
    report = paired_numeric_science(
        arrays["reference"], arrays["fpgai"], arrays["hls4ml"], arrays["labels"]
    )
    if report["samples"] != baseline["matched_samples"]:
        raise ValueError("Unmatched source cohort")
    report["source_manifest_sha256"] = sha256(source.read_bytes()).hexdigest()
    report["original_onnx_sha256"] = baseline["original_onnx_sha256"]
    report["ordered_input_cohort_sha256"] = baseline["ordered_input_cohort_sha256"]
    report["original_matched_report_sha256"] = sha256(
        json.dumps(baseline, sort_keys=True).encode()
    ).hexdigest()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True, help="New immutable private folder")
    args = parser.parse_args(argv)
    target = Path(args.output)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Refusing to modify an existing report")
    report = compare_pinned_manifest(args.manifest)
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".paired-science-", dir=target.parent) as temp:
        stage = Path(temp) / "report"
        stage.mkdir()
        (stage / "paired_numeric_science.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if target.exists() or target.is_symlink():
            raise FileExistsError("Destination appeared during export")
        stage.rename(target)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
