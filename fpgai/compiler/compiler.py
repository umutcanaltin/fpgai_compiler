from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Optional


# Keep this module import-safe in test environments where the legacy engine's
# historical top-level imports are not on PYTHONPATH. The actual import happens
# lazily inside compile(), and tests can monkeyupdate this symbol directly.
fpgai_engine: Optional[Callable[..., Any]] = None


@dataclass
class CompileArtifact:
    # In v1 we store paths because the engine writes generated artifacts.
    out_dir: str
    success: bool = True
    meta: Optional[dict] = None


def _get_fpgai_engine() -> Callable[..., Any]:
    global fpgai_engine

    if fpgai_engine is not None:
        return fpgai_engine

    from fpgai.engine.legacy_engine import fpgai_engine as imported_engine

    fpgai_engine = imported_engine
    return imported_engine


def _training_learning_rate(cfg: Any) -> float:
    training = cfg.raw.get("training", {}) or {}

    if not training.get("enabled", False):
        return 0.1

    optimizer = training.get("optimizer", {}) or {}
    if isinstance(optimizer, dict):
        return float(optimizer.get("learning_rate", 0.1))

    return float(getattr(optimizer, "learning_rate", 0.1))


def _load_codegen_meta(out_dir: str | Path) -> dict[str, Any]:
    """Load HLS codegen metadata into the high-level compile artifact."""
    output_dir = Path(out_dir)
    meta_path = output_dir / "hls" / "codegen_meta.json"

    if not meta_path.exists():
        return {
            "codegen_meta_present": False,
            "codegen_meta_path": str(meta_path),
        }

    try:
        codegen_meta = json.loads(
            meta_path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        return {
            "codegen_meta_present": False,
            "codegen_meta_path": str(meta_path),
            "codegen_meta_error": str(exc),
        }

    artifact_meta: dict[str, Any] = {
        "codegen_meta_present": True,
        "codegen_meta_path": str(meta_path),
        "codegen_meta": codegen_meta,
    }

    for key in (
        "tiling_analysis",
        "tiling_resource_estimate",
        "tiling_performance_estimate",
        "tiling_sweep",
        "tiling_reports_error",
    ):
        if key in codegen_meta:
            artifact_meta[key] = codegen_meta[key]

    return artifact_meta


class Compiler:
    def __init__(self, cfg: Any):
        self.cfg = cfg

    @classmethod
    def from_cfg(cls, cfg: Any) -> "Compiler":
        return cls(cfg)

    def compile(
        self,
        *,
        input_data,
        first_layer_shape,
        output_shape,
        onnx_path: str,
        verbose: bool = False,
    ) -> CompileArtifact:
        # Map YAML → legacy engine args.
        precision = "float"
        if self.cfg.numerics.kind == "fixed":
            precision = self.cfg.numerics.activation

        engine = _get_fpgai_engine()

        _ = engine(
            input_data=input_data,
            first_layer_shape=first_layer_shape,
            output_shape=output_shape,
            learning_rate=_training_learning_rate(self.cfg),
            mode=self.cfg.pipeline.mode,
            onnx_file_name=onnx_path,
            precision=precision,
            verbose=verbose,
        )

        out_dir = self.cfg.raw.get(
            "project",
            {},
        ).get(
            "out_dir",
            "generated_files",
        )

        return CompileArtifact(
            out_dir=out_dir,
            success=True,
            meta=_load_codegen_meta(out_dir),
        )
