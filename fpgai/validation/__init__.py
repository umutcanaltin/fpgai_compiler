"""Validation helpers for FPGAI-generated artifacts.

This package contains reusable correctness and comparison utilities that were
historically kept under ``scripts/``. Public workflows should call these helpers
through the FPGAI CLI rather than invoking scripts directly.
"""

from .onnx_compare import (
    ComparisonStats,
    compare_arrays,
    compare_host_vs_onnx,
    compare_vitis_vs_host_vs_onnx,
)

__all__ = [
    "ComparisonStats",
    "compare_arrays",
    "compare_host_vs_onnx",
    "compare_vitis_vs_host_vs_onnx",
]
