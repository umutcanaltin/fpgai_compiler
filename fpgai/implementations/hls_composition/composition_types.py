from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from fpgai.implementations.implementation_contract import ImplementationContract


@dataclass(frozen=True)
class ExternalNodeBinding:
    node_name: str
    op_type: str
    operator_id: str
    operator_package_id: str
    operator_package_version: str
    operator_manifest_hash: str
    contract: ImplementationContract
    attributes: Mapping[str, Any]
    input_tensor: str
    output_tensor: str
    input_words: int
    output_words: int
    wrapper_symbol: str
    conversion_buffers: bool = False
    input_tensors: tuple[str, ...] = ()
    output_tensors: tuple[str, ...] = ()
    port_words: int | None = None
    input_port_words: Mapping[str, int] = field(default_factory=dict)
    output_port_words: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))
        object.__setattr__(self, "input_port_words", MappingProxyType(dict(self.input_port_words)))
        object.__setattr__(self, "output_port_words", MappingProxyType(dict(self.output_port_words)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node_name,
            "op_type": self.op_type,
            "operator_id": self.operator_id,
            "operator_package": {
                "package_id": self.operator_package_id,
                "version": self.operator_package_version,
                "manifest_hash": self.operator_manifest_hash,
            },
            "implementation": self.contract.to_dict(),
            "input_tensor": self.input_tensor,
            "output_tensor": self.output_tensor,
            "input_words": self.input_words,
            "output_words": self.output_words,
            "wrapper_symbol": self.wrapper_symbol,
            "conversion_buffers": self.conversion_buffers,
            "input_tensors": list(self.input_tensors or (self.input_tensor,)),
            "output_tensors": list(self.output_tensors or (self.output_tensor,)),
            "port_words": self.port_words,
            "input_port_words": dict(self.input_port_words),
            "output_port_words": dict(self.output_port_words),
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True)
class HLSCompositionPlan:
    bindings: tuple[ExternalNodeBinding, ...] = ()
    selection_reports: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    graph_mode: str = "sequential_mixed_graph"

    def __post_init__(self) -> None:
        names = [item.node_name for item in self.bindings]
        if len(names) != len(set(names)):
            raise ValueError("HLSCOMP001: duplicate external node binding")
        object.__setattr__(self, "selection_reports", MappingProxyType(dict(self.selection_reports)))

    def binding_for_node(self, node_name: str) -> ExternalNodeBinding | None:
        return next((item for item in self.bindings if item.node_name == node_name), None)

    @property
    def used_package_ids(self) -> tuple[str, ...]:
        values: list[str] = []
        for item in self.bindings:
            for package_id in (item.operator_package_id, item.contract.package_id):
                if package_id and package_id not in values:
                    values.append(package_id)
        return tuple(values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "fpgai.mixed-hls-composition/v1",
            "graph_mode": self.graph_mode,
            "nodes": [item.to_dict() for item in self.bindings],
            "selection_reports": {key: dict(value) for key, value in self.selection_reports.items()},
            "usage": {"platform_scope": "research", "production_path": "morfics"},
        }


@dataclass(frozen=True)
class StagedExternalSources:
    sources: tuple[Path, ...]
    headers: tuple[Path, ...]
    include_dirs: tuple[Path, ...]
