"""Paper verification, artifact, setup, and plot helpers for FPGAI.

The package keeps imports lazy so `python -m fpgai.paper.<module>` can execute
without pre-loading the target module from this package initializer.
"""

from __future__ import annotations

from typing import Any


def emit_paper_verification_artifacts(*args: Any, **kwargs: Any) -> Any:
    from .verification import emit_paper_verification_artifacts as _impl

    return _impl(*args, **kwargs)


def emit_experiment_artifact_reports(*args: Any, **kwargs: Any) -> Any:
    from .experiment_artifacts import emit_experiment_artifact_reports as _impl

    return _impl(*args, **kwargs)


def generate_paper_plot_artifacts(*args: Any, **kwargs: Any) -> Any:
    from .plots import generate_paper_plot_artifacts as _impl

    return _impl(*args, **kwargs)


def generate_experiment_setup_artifacts(*args: Any, **kwargs: Any) -> Any:
    from .experiment_setup import generate_experiment_setup_artifacts as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "emit_paper_verification_artifacts",
    "emit_experiment_artifact_reports",
    "generate_paper_plot_artifacts",
    "generate_experiment_setup_artifacts",
]
