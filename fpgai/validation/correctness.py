"""Public correctness-validation workflow helpers for FPGAI configs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from fpgai.benchmark.pipeline import BenchmarkResult
from fpgai.config.loader import load_config


@dataclass(frozen=True)
class CorrectnessValidation:
    """Result of a public correctness-validation request."""

    config_path: Path
    pipeline_mode: str
    requested: bool
    executed: bool
    passed: bool
    reason: str | None = None
    build_dir: Path | None = None
    bench_dir: Path | None = None
    metrics_json: Path | None = None
    summary_txt: Path | None = None

    @classmethod
    def skipped(cls, config_path: str | Path, pipeline_mode: str, reason: str) -> "CorrectnessValidation":
        return cls(
            config_path=Path(config_path),
            pipeline_mode=pipeline_mode,
            requested=True,
            executed=False,
            passed=False,
            reason=reason,
        )

    @classmethod
    def from_benchmark(
        cls,
        config_path: str | Path,
        pipeline_mode: str,
        benchmark: "BenchmarkResult",
    ) -> "CorrectnessValidation":
        return cls(
            config_path=Path(config_path),
            pipeline_mode=pipeline_mode,
            requested=True,
            executed=True,
            passed=bool(benchmark.passed),
            build_dir=benchmark.build_dir,
            bench_dir=benchmark.bench_dir,
            metrics_json=benchmark.metrics_json,
            summary_txt=benchmark.summary_txt,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_path": str(self.config_path),
            "pipeline_mode": self.pipeline_mode,
            "requested": self.requested,
            "executed": self.executed,
            "passed": self.passed,
            "reason": self.reason,
            "build_dir": None if self.build_dir is None else str(self.build_dir),
            "bench_dir": None if self.bench_dir is None else str(self.bench_dir),
            "metrics_json": None if self.metrics_json is None else str(self.metrics_json),
            "summary_txt": None if self.summary_txt is None else str(self.summary_txt),
        }


def _pipeline_mode(config_path: str | Path) -> str:
    cfg = load_config(config_path)
    mode = getattr(getattr(cfg, "pipeline", None), "mode", None)
    if mode is None:
        raw = getattr(cfg, "raw", {}) or {}
        mode = ((raw.get("pipeline") or {}).get("mode"))
    return str(mode or "unknown")


def validate_correctness(config_path: str | Path) -> CorrectnessValidation:
    """Run the compiler's numeric correctness workflow for inference or training.

    Inference compares generated execution outputs with the configured software
    reference.  On-device training validates the generated training execution
    against the compiler's reference/capture artifacts, including gradients and
    parameter-update state when those captures are requested.
    """
    path = Path(config_path)
    mode = _pipeline_mode(path)

    if mode == "inference":
        from fpgai.benchmark.pipeline import run_compile_correctness_benchmark
        benchmark = run_compile_correctness_benchmark(config_path=path)
    elif mode == "training_on_device":
        from fpgai.benchmark.pipeline import run_compile_training_benchmark
        benchmark = run_compile_training_benchmark(config_path=path)
    else:
        return CorrectnessValidation.skipped(
            path,
            mode,
            f"numeric correctness validation does not support pipeline.mode={mode}",
        )
    return CorrectnessValidation.from_benchmark(path, mode, benchmark)
