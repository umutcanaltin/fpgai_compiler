from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

_SCHEMA = "fpgai.reference-architecture/v1"
_ALLOWED_FAMILIES = {"llm_like", "yolo_like", "single_stage_detection"}


@dataclass(frozen=True)
class ReferenceArchitecture:
    schema: str
    name: str
    family: str
    description: str
    compiler_special_case: bool
    graph_source: str
    features: tuple[str, ...]
    benchmark_label: str
    parameters: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "name": self.name,
            "family": self.family,
            "description": self.description,
            "compiler_special_case": self.compiler_special_case,
            "graph_source": self.graph_source,
            "features": list(self.features),
            "benchmark_label": self.benchmark_label,
            "parameters": dict(self.parameters),
        }


def load_reference_architecture(path: str | Path) -> ReferenceArchitecture:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("REFARCH001: reference architecture must be a mapping")
    if data.get("schema") != _SCHEMA:
        raise ValueError(f"REFARCH002: unsupported reference architecture schema {data.get('schema')!r}")
    family = str(data.get("family") or "").strip().lower()
    if family not in _ALLOWED_FAMILIES:
        raise ValueError(f"REFARCH003: unsupported reference architecture family {family!r}")
    if bool(data.get("compiler_special_case", True)):
        raise ValueError("REFARCH004: reference architectures must explicitly disable compiler special-casing")
    graph_source = str(data.get("graph_source") or "").strip().lower()
    if graph_source not in {"maintained_generic_graph", "imported_graph"}:
        raise ValueError("REFARCH005: graph_source must be maintained_generic_graph or imported_graph")
    features = tuple(str(x) for x in (data.get("features") or ()))
    if not features:
        raise ValueError("REFARCH006: reference architecture requires at least one feature")
    benchmark_label = str(data.get("benchmark_label") or "").strip()
    if not benchmark_label:
        raise ValueError("REFARCH007: benchmark_label is required")
    return ReferenceArchitecture(
        schema=_SCHEMA,
        name=str(data.get("name") or "").strip(),
        family=family,
        description=str(data.get("description") or "").strip(),
        compiler_special_case=False,
        graph_source=graph_source,
        features=features,
        benchmark_label=benchmark_label,
        parameters=dict(data.get("parameters") or {}),
    )


__all__ = ["ReferenceArchitecture", "load_reference_architecture"]
