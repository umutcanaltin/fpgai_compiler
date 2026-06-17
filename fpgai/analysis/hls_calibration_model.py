"""Operator-wise HLS calibration model for FPGAI estimates."""

from __future__ import annotations

from collections import defaultdict
from statistics import median
from typing import Any

RESOURCE_METRICS = ("lut", "ff", "dsp", "bram", "latency_cycles")


def fit_calibration_model(dataset: dict[str, Any]) -> dict[str, Any]:
    """Fit median actual/estimated scale factors per operator and metric."""
    ratios: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    warnings: list[dict[str, Any]] = []

    for sample in dataset.get("samples", []):
        op = sample.get("operator", "Unknown")
        estimated = sample.get("estimated", {}) or {}
        actual = sample.get("hls_actual", {}) or {}
        for metric in RESOURCE_METRICS:
            est = _to_float(estimated.get(metric, 0.0))
            act = _to_float(actual.get(metric, 0.0))
            if est > 0:
                ratios[op][metric].append(act / est)
            elif act > 0:
                warnings.append(
                    {
                        "operator": op,
                        "layer_name": sample.get("layer_name"),
                        "metric": metric,
                        "warning": "zero_estimate_nonzero_actual",
                    }
                )

    operators: dict[str, dict[str, float]] = {}
    global_ratios: dict[str, list[float]] = defaultdict(list)
    for op, metric_ratios in ratios.items():
        operators[op] = {}
        for metric in RESOURCE_METRICS:
            values = metric_ratios.get(metric, [])
            if values:
                operators[op][metric] = float(median(values))
                global_ratios[metric].extend(values)
            else:
                operators[op][metric] = 1.0

    global_scale = {
        metric: float(median(values)) if values else 1.0
        for metric, values in global_ratios.items()
    }
    for metric in RESOURCE_METRICS:
        global_scale.setdefault(metric, 1.0)

    return {
        "schema_version": 1,
        "method": "operator_median_scale_factor",
        "metrics": list(RESOURCE_METRICS),
        "operators": operators,
        "global": global_scale,
        "warnings": warnings,
    }


def apply_calibration_model(dataset: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the dataset with calibrated estimates added."""
    output = dict(dataset)
    output_samples = []
    for sample in dataset.get("samples", []):
        op = sample.get("operator", "Unknown")
        scales = model.get("operators", {}).get(op) or model.get("global", {}) or {}
        estimated = sample.get("estimated", {}) or {}
        calibrated = {}
        warnings = list(sample.get("warnings", []))
        for metric in RESOURCE_METRICS:
            est = _to_float(estimated.get(metric, 0.0))
            scale = _to_float(scales.get(metric, 1.0)) or 1.0
            calibrated[metric] = est * scale
            actual = _to_float((sample.get("hls_actual", {}) or {}).get(metric, 0.0))
            if est == 0 and actual > 0:
                warnings.append({"metric": metric, "warning": "zero_estimate_nonzero_actual"})
        new_sample = dict(sample)
        new_sample["calibrated_estimate"] = calibrated
        if warnings:
            new_sample["warnings"] = warnings
        output_samples.append(new_sample)
    output["samples"] = output_samples
    return output


def mean_absolute_percentage_error(
    samples: list[dict[str, Any]],
    metric: str,
    *,
    calibrated: bool = False,
) -> float:
    """Compute MAPE in percent. Samples with actual == 0 are ignored."""
    errors = []
    estimate_key = "calibrated_estimate" if calibrated else "estimated"
    for sample in samples:
        actual = _to_float((sample.get("hls_actual", {}) or {}).get(metric, 0.0))
        predicted = _to_float((sample.get(estimate_key, {}) or {}).get(metric, 0.0))
        if actual == 0:
            continue
        errors.append(abs(predicted - actual) / abs(actual) * 100.0)
    return float(sum(errors) / len(errors)) if errors else 0.0


def summarize_error(dataset_with_calibration: dict[str, Any]) -> dict[str, dict[str, float]]:
    samples = dataset_with_calibration.get("samples", [])
    return {
        "raw_mean_absolute_percentage_error": {
            metric: mean_absolute_percentage_error(samples, metric, calibrated=False)
            for metric in RESOURCE_METRICS
        },
        "calibrated_mean_absolute_percentage_error": {
            metric: mean_absolute_percentage_error(samples, metric, calibrated=True)
            for metric in RESOURCE_METRICS
        },
    }


def _to_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return 0.0
