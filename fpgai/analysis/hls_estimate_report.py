"""Write JSON and text reports comparing FPGAI estimates with HLS results."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from .hls_calibration_model import (
    RESOURCE_METRICS,
    apply_calibration_model,
    mean_absolute_percentage_error,
)

DISPLAY_NAMES = {
    "lut": "LUT",
    "ff": "FF",
    "dsp": "DSP",
    "bram": "BRAM",
    "latency_cycles": "Latency cycles",
}


def write_estimate_vs_hls_report(
    dataset: dict[str, Any],
    model: dict[str, Any],
    output_json_path: str | Path,
    output_summary_path: str | Path,
) -> dict[str, Any]:
    """Write estimate_vs_hls.json and summary.txt."""
    calibrated_dataset = apply_calibration_model(dataset, model)
    samples = calibrated_dataset.get("samples", [])

    raw_mape = {
        metric: mean_absolute_percentage_error(samples, metric, calibrated=False)
        for metric in RESOURCE_METRICS
    }
    calibrated_mape = {
        metric: mean_absolute_percentage_error(samples, metric, calibrated=True)
        for metric in RESOURCE_METRICS
    }
    improved = {
        metric: calibrated_mape[metric] < raw_mape[metric]
        if raw_mape[metric] != calibrated_mape[metric]
        else None
        for metric in RESOURCE_METRICS
    }

    report = {
        "schema_version": 1,
        "summary": {
            "sample_count": len(samples),
            "operators": sorted({sample.get("operator", "Unknown") for sample in samples}),
            "raw_mean_absolute_percentage_error": raw_mape,
            "calibrated_mean_absolute_percentage_error": calibrated_mape,
            "improved": improved,
        },
        "samples": samples,
        "model": model,
    }

    output_json_path = Path(output_json_path)
    output_summary_path = Path(output_summary_path)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    output_summary_path.write_text(_format_summary(report))
    return report


def _format_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "FPGAI HLS Calibration Summary",
        "=============================",
        "",
        f"Samples: {summary.get('sample_count', 0)}",
        "Operators: " + ", ".join(summary.get("operators", [])),
        "",
        "Raw MAPE:",
    ]
    raw = summary.get("raw_mean_absolute_percentage_error", {})
    calibrated = summary.get("calibrated_mean_absolute_percentage_error", {})
    improved = summary.get("improved", {})
    for metric in RESOURCE_METRICS:
        lines.append(f"  {DISPLAY_NAMES[metric]}: {raw.get(metric, 0.0):.2f}%")
    lines.extend(["", "Calibrated MAPE:"])
    for metric in RESOURCE_METRICS:
        lines.append(f"  {DISPLAY_NAMES[metric]}: {calibrated.get(metric, 0.0):.2f}%")
    lines.extend(["", "Improved metrics:"])
    for metric in RESOURCE_METRICS:
        value = improved.get(metric)
        status = "yes" if value is True else "unchanged" if value is None else "no"
        lines.append(f"  {DISPLAY_NAMES[metric]}: {status}")
    lines.append("")
    return "\n".join(lines)
