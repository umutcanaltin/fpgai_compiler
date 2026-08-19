from __future__ import annotations

from pathlib import Path

from .bridge import MLIRBridgeError, import_fpgai_mlir
from .routes import detect_mlir_dialect, framework_mlir_route
from .stablehlo import StableHLOImportError, import_stablehlo_mlir


class MLIRImportError(ValueError):
    pass


def _read(source: str | Path) -> str:
    if isinstance(source, Path):
        return source.read_text(encoding="utf-8")
    value = str(source)
    if "\n" not in value and len(value) < 4096:
        path = Path(value)
        if path.exists():
            return path.read_text(encoding="utf-8")
    return value


def import_mlir_program(
    source: str | Path,
    *,
    source_framework: str | None = None,
    pipeline_mode: str = "inference",
    target_board: str | None = None,
):
    text = _read(source)
    dialect = detect_mlir_dialect(text)
    if source_framework:
        route = framework_mlir_route(source_framework)
        if not route.accepted_by_fpgai and dialect not in {"stablehlo", "fpgai_bridge", "fpgai_native"}:
            raise MLIRImportError(
                f"MLIRIMPORT001: {source_framework} route emits {route.preferred_dialect!r}; "
                "perform upstream legalization to StableHLO or FPGAI-supported MLIR before import"
            )
    try:
        if dialect == "fpgai_bridge":
            return import_fpgai_mlir(text)
        if dialect == "stablehlo":
            return import_stablehlo_mlir(
                text,
                pipeline_mode=pipeline_mode,
                target_board=target_board,
                source_framework=source_framework,
            )
    except (MLIRBridgeError, StableHLOImportError) as exc:
        raise MLIRImportError(str(exc)) from exc
    if dialect == "tf":
        raise MLIRImportError("MLIRIMPORT002: TensorFlow-dialect MLIR requires upstream legalization to StableHLO")
    if dialect == "torch":
        raise MLIRImportError("MLIRIMPORT003: Torch-dialect MLIR requires torch-mlir/StableHLO legalization before FPGAI import")
    if dialect == "fpgai_native":
        raise MLIRImportError("MLIRIMPORT004: native FPGAI dialect parsing requires the optional native MLIR runtime; use bridge MLIR today")
    raise MLIRImportError("MLIRIMPORT005: unable to detect a supported MLIR dialect")
