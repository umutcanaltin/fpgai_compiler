from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Callable, Dict, Iterable

from fpgai.ir.graph import Graph

FrontendImporter = Callable[..., Graph]


@dataclass(frozen=True)
class FrontendSpec:
    name: str
    importer: FrontendImporter
    extensions: tuple[str, ...] = ()
    description: str = ""
    provider: str = "fpgai"
    version: str = "1"


_FRONTENDS: Dict[str, FrontendSpec] = {}
_DISCOVERED = False


def register_frontend(spec: FrontendSpec, *, replace: bool = False) -> None:
    key = str(spec.name).strip().lower()
    if not key:
        raise ValueError("FRONTEND001: frontend name must be non-empty")
    if key in _FRONTENDS and not replace:
        raise ValueError(f"FRONTEND002: frontend {key!r} is already registered")
    _FRONTENDS[key] = spec


def _register_builtins() -> None:
    if "onnx" not in _FRONTENDS:
        def _onnx(source, **kwargs):
            from fpgai.frontend.onnx import import_onnx
            return import_onnx(source, canonicalize=True, infer_shapes=True, **{k: v for k, v in kwargs.items() if k in {"external_operator_context", "shape_overrides"} and v is not None})
        register_frontend(FrontendSpec("onnx", _onnx, (".onnx",), "ONNX model importer"))

    if "mlir" not in _FRONTENDS:
        def _mlir(source, **kwargs):
            from fpgai.frontend.mlir import import_mlir_program
            return import_mlir_program(
                source,
                source_framework=kwargs.get("source_framework"),
                pipeline_mode=kwargs.get("pipeline_mode", "inference"),
                target_board=kwargs.get("target_board"),
            )
        register_frontend(FrontendSpec("mlir", _mlir, (".mlir",), "Auto-detected MLIR dialect importer"))

    if "stablehlo" not in _FRONTENDS:
        def _stablehlo(source, **kwargs):
            from fpgai.frontend.mlir.stablehlo import import_stablehlo_mlir
            return import_stablehlo_mlir(
                source,
                source_framework=kwargs.get("source_framework"),
                pipeline_mode=kwargs.get("pipeline_mode", "inference"),
                target_board=kwargs.get("target_board"),
            )
        register_frontend(FrontendSpec("stablehlo", _stablehlo, (".mlir",), "StableHLO textual importer"))


def discover_frontends() -> None:
    global _DISCOVERED
    _register_builtins()
    if _DISCOVERED:
        return
    _DISCOVERED = True
    try:
        eps = metadata.entry_points()
        selected = eps.select(group="fpgai.frontends") if hasattr(eps, "select") else eps.get("fpgai.frontends", ())
    except Exception:
        selected = ()
    for ep in selected:
        provider = ep.load()
        value = provider() if callable(provider) and not isinstance(provider, FrontendSpec) else provider
        specs: Iterable[FrontendSpec]
        if isinstance(value, FrontendSpec):
            specs = (value,)
        else:
            specs = tuple(value)
        for spec in specs:
            register_frontend(spec)


def frontend_registry() -> Dict[str, FrontendSpec]:
    discover_frontends()
    return dict(_FRONTENDS)


def infer_frontend_format(source: str | Path, format_hint: str | None = None) -> str:
    if format_hint:
        key = str(format_hint).strip().lower()
        aliases = {"jax": "stablehlo", "pytorch": "stablehlo", "torch": "stablehlo"}
        return aliases.get(key, key)
    suffix = Path(str(source)).suffix.lower()
    if suffix == ".onnx":
        return "onnx"
    if suffix == ".mlir":
        return "mlir"
    raise ValueError(f"FRONTEND003: cannot infer frontend for {source!s}; set model.format")



def source_framework_route(framework: str | None, format_name: str | None = None) -> dict:
    """Return an explicit, claim-safe source-framework ingress contract."""
    fw = str(framework or "unknown").strip().lower()
    fmt = str(format_name or "").strip().lower()
    if fw in {"jax", "pytorch", "tensorflow"}:
        from fpgai.frontend.mlir.routes import framework_mlir_route
        route = framework_mlir_route(fw).to_dict()
        route["selected_format"] = fmt or None
        route["selected_path_accepted"] = (fmt in set(route.get("accepted_formats", ()))) if fmt else None
        if fw == "tensorflow" and fmt in {"stablehlo", "onnx"}:
            route["selected_path_accepted"] = True
            route["legalization_applied_upstream"] = True
        elif fw == "pytorch" and fmt in {"stablehlo", "onnx"}:
            route["legalization_applied_upstream"] = True
        elif fw == "jax" and fmt in {"stablehlo", "mlir"}:
            route["legalization_applied_upstream"] = False
        return route
    if fw == "onnx" or (fw == "unknown" and fmt == "onnx"):
        return {
            "framework": "onnx",
            "producer": "ONNX producer/exporter",
            "preferred_dialect": "onnx",
            "accepted_by_fpgai": True,
            "maturity": "supported_subset",
            "accepted_formats": ["onnx"],
            "legalization_required": False,
            "selected_format": fmt or "onnx",
            "selected_path_accepted": fmt in {"", "onnx"},
            "notes": ["Direct ONNX frontend; unsupported operators fail explicitly."],
        }
    return {
        "framework": fw, "selected_format": fmt or None,
        "accepted_by_fpgai": None, "maturity": "unclassified",
        "selected_path_accepted": None, "notes": [],
    }

def import_model_source(
    source: str | Path,
    *,
    format_hint: str | None = None,
    source_framework: str | None = None,
    pipeline_mode: str = "inference",
    target_board: str | None = None,
    **frontend_options,
) -> Graph:
    discover_frontends()
    key = infer_frontend_format(source, format_hint)
    spec = _FRONTENDS.get(key)
    if spec is None:
        available = ", ".join(sorted(_FRONTENDS))
        raise ValueError(f"FRONTEND004: unsupported model.format {key!r}; registered frontends: {available}")
    graph = spec.importer(
        source,
        source_framework=source_framework,
        pipeline_mode=pipeline_mode,
        target_board=target_board,
        **frontend_options,
    )
    graph.metadata.setdefault("source", {})
    graph.metadata["source"].update({
        "format": key,
        "framework": source_framework,
        "path": str(source),
        "frontend_provider": spec.provider,
        "frontend_version": spec.version,
        "ingress_route": source_framework_route(source_framework, key),
    })
    return graph
