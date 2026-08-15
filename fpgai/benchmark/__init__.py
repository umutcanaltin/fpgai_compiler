"""Benchmark execution, reference comparison, validation summaries, and artifacts."""

from __future__ import annotations

from typing import Any


def emit_validation_summary_artifacts(*args: Any, **kwargs: Any) -> Any:
    from .verification import emit_validation_summary_artifacts as _impl
    return _impl(*args, **kwargs)


def emit_experiment_artifact_reports(*args: Any, **kwargs: Any) -> Any:
    from .experiment_artifacts import emit_experiment_artifact_reports as _impl
    return _impl(*args, **kwargs)


def generate_benchmark_plot_artifacts(*args: Any, **kwargs: Any) -> Any:
    from .plots import generate_benchmark_plot_artifacts as _impl
    return _impl(*args, **kwargs)


def generate_experiment_setup_artifacts(*args: Any, **kwargs: Any) -> Any:
    from .experiment_setup import generate_experiment_setup_artifacts as _impl
    return _impl(*args, **kwargs)


__all__ = [
    "emit_validation_summary_artifacts",
    "emit_experiment_artifact_reports",
    "generate_benchmark_plot_artifacts",
    "generate_experiment_setup_artifacts",
]
