"""Analysis utilities for FPGAI."""

from .hls_calibration_dataset import (
    CalibrationSample,
    ResourceEstimate,
    build_calibration_dataset,
    parse_hls_csynth_report,
)
from .hls_calibration_model import (
    apply_calibration_model,
    fit_calibration_model,
    mean_absolute_percentage_error,
)
from .hls_estimate_report import write_estimate_vs_hls_report

__all__ = [
    "CalibrationSample",
    "ResourceEstimate",
    "build_calibration_dataset",
    "parse_hls_csynth_report",
    "apply_calibration_model",
    "fit_calibration_model",
    "mean_absolute_percentage_error",
    "write_estimate_vs_hls_report",
]
