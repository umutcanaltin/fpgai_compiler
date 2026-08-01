from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from fpgai.contracts.package_manifest import PackageManifest, load_package_manifest
from fpgai.contracts.package_types import ImplementationLanguage, ValidationLevel

from .implementation_errors import ImplementationIssue
from .implementation_schema import ImplementationMetrics, InterfaceRequirement

_ID_RE = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)+$")
_VALID_BACKENDS = {"vitis_hls", "vhdl", "verilog", "systemverilog", "simulator", "vivado"}
_VALID_POLICIES = {"explicit_only", "validated_only", "latency", "throughput", "area", "power", "balanced"}
_VALIDATION_RANK = {level.value: index for index, level in enumerate(ValidationLevel)}


@dataclass(frozen=True)
class TrainingImplementationCapabilities:
    forward: bool = False
    backward_input: bool = False
    parameter_gradients: bool = False
    bias_gradients: bool = False
    optimizer_update: bool = False

    def to_dict(self) -> dict[str, bool]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class ImplementationContract:
    package_id: str
    version: str
    operator_id: str
    language: str
    backend: str
    top: str
    sources: tuple[str, ...]
    headers: tuple[str, ...] = ()
    source_order: tuple[str, ...] = ()
    inference: bool = True
    training: TrainingImplementationCapabilities = field(default_factory=TrainingImplementationCapabilities)
    boards: tuple[str, ...] = ()
    toolchains: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    precisions: tuple[str, ...] = ()
    interfaces: tuple[InterfaceRequirement, ...] = ()
    weight_storage: tuple[str, ...] = ()
    activation_storage: tuple[str, ...] = ()
    validation_level: str = "unvalidated"
    metrics: ImplementationMetrics = field(default_factory=ImplementationMetrics)
    license_category: str = "research_only"
    package_root: Path | None = None
    manifest_hash: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "toolchains", MappingProxyType({key: tuple(value) for key, value in self.toolchains.items()}))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        errors = [issue for issue in validate_implementation_contract(self) if issue.severity == "error"]
        if errors:
            raise ValueError("; ".join(f"{issue.code}: {issue.message}" for issue in errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "fpgai.implementation-contract/v1",
            "package_id": self.package_id,
            "version": self.version,
            "operator_id": self.operator_id,
            "language": self.language,
            "backend": self.backend,
            "top": self.top,
            "sources": list(self.sources),
            "headers": list(self.headers),
            "source_order": list(self.source_order),
            "capabilities": {"inference": self.inference, "training": self.training.to_dict()},
            "compatibility": {
                "boards": list(self.boards),
                "toolchains": {key: list(value) for key, value in self.toolchains.items()},
                "precisions": list(self.precisions),
            },
            "interfaces": [item.to_dict() for item in self.interfaces],
            "memory": {
                "weight_storage": list(self.weight_storage),
                "activation_storage": list(self.activation_storage),
            },
            "validation_level": self.validation_level,
            "metrics": self.metrics.to_dict(),
            "license_category": self.license_category,
            "manifest_hash": self.manifest_hash,
            "metadata": dict(self.metadata),
            "usage": {"platform_scope": "research", "production_path": "morfics"},
        }


def validate_implementation_contract(contract: ImplementationContract) -> tuple[ImplementationIssue, ...]:
    issues: list[ImplementationIssue] = []
    if not _ID_RE.fullmatch(contract.package_id):
        issues.append(ImplementationIssue("IMPL001", "package_id", "Package ID must be namespace-qualified and lowercase"))
    if not _ID_RE.fullmatch(contract.operator_id):
        issues.append(ImplementationIssue("IMPL002", "operator_id", "Implemented operator ID must be namespace-qualified and lowercase"))
    if contract.language not in {item.value for item in ImplementationLanguage}:
        issues.append(ImplementationIssue("IMPL003", "language", "Unsupported implementation language"))
    if contract.backend not in _VALID_BACKENDS:
        issues.append(ImplementationIssue("IMPL004", "backend", "Unsupported backend"))
    if not contract.top.strip():
        issues.append(ImplementationIssue("IMPL005", "top", "Top-level symbol is required"))
    if not contract.sources:
        issues.append(ImplementationIssue("IMPL006", "sources", "At least one implementation source is required"))
    if contract.validation_level not in _VALIDATION_RANK:
        issues.append(ImplementationIssue("IMPL007", "validation_level", "Unsupported validation level"))
    if contract.training.backward_input and not contract.training.forward:
        issues.append(ImplementationIssue("IMPL008", "training.backward_input", "Backward input requires training forward"))
    if contract.training.parameter_gradients and not contract.training.forward:
        issues.append(ImplementationIssue("IMPL009", "training.parameter_gradients", "Parameter gradients require training forward"))
    if contract.training.bias_gradients and not contract.training.parameter_gradients:
        issues.append(ImplementationIssue("IMPL010", "training.bias_gradients", "Bias gradients require parameter gradients"))
    if contract.training.optimizer_update and not contract.training.parameter_gradients:
        issues.append(ImplementationIssue("IMPL011", "training.optimizer_update", "Optimizer update requires parameter gradients"))
    names = [item.name for item in contract.interfaces]
    if len(names) != len(set(names)):
        issues.append(ImplementationIssue("IMPL012", "interfaces", "Interface names must be unique"))
    return tuple(issues)


def _tuple(value: Any) -> tuple[str, ...]:
    return tuple(str(item) for item in value) if isinstance(value, list) else ()


def _interface_contracts(value: Any) -> tuple[InterfaceRequirement, ...]:
    if not isinstance(value, Mapping):
        return ()
    result = []
    for name, raw in value.items():
        if not isinstance(raw, Mapping):
            continue
        result.append(
            InterfaceRequirement(
                name=str(name),
                direction=str(raw.get("direction", "")),
                protocol=str(raw.get("protocol", "")),
                data_type=str(raw["data_type"]) if raw.get("data_type") is not None else None,
                data_width=int(raw["data_width"]) if raw.get("data_width") is not None else None,
                layout=str(raw["layout"]) if raw.get("layout") is not None else None,
            )
        )
    return tuple(result)


def implementation_contract_from_manifest(package_root: str | Path, *, manifest_hash: str = "") -> ImplementationContract:
    manifest: PackageManifest = load_package_manifest(package_root)
    if manifest.asset_type != "implementation":
        raise ValueError("Package is not an implementation asset")
    raw = manifest.raw
    package = dict(raw.get("package", {}))
    entrypoint = dict(dict(raw.get("entrypoints", {})).get("implementation", {}))
    implementation = dict(raw.get("implementation", {}))
    capabilities = dict(raw.get("capabilities", {}))
    training_raw = dict(capabilities.get("training", {}))
    compatibility = dict(raw.get("compatibility", {}))
    toolchains = {}
    for item in compatibility.get("toolchains", []) or []:
        if isinstance(item, Mapping) and item.get("name"):
            toolchains[str(item["name"])] = _tuple(item.get("versions", []))
    memory = dict(raw.get("memory", {}))
    metrics_raw = dict(raw.get("metrics", {}))
    validation = dict(raw.get("validation", {}))
    license_cfg = dict(raw.get("license", {}))
    operator_id = str(implementation.get("operator_id") or implementation.get("implements") or "")
    return ImplementationContract(
        package_id=manifest.package_id,
        version=manifest.version,
        operator_id=operator_id,
        language=str(entrypoint.get("language", "")),
        backend=str(implementation.get("backend") or entrypoint.get("backend") or entrypoint.get("language", "")),
        top=str(entrypoint.get("top", "")),
        sources=_tuple(entrypoint.get("sources", [])),
        headers=_tuple(entrypoint.get("headers", [])),
        source_order=_tuple(entrypoint.get("source_order", [])),
        inference=bool(capabilities.get("inference", False)),
        training=TrainingImplementationCapabilities(
            forward=bool(training_raw.get("forward", False)),
            backward_input=bool(training_raw.get("backward_input", False)),
            parameter_gradients=bool(training_raw.get("parameter_gradients", False)),
            bias_gradients=bool(training_raw.get("bias_gradients", False)),
            optimizer_update=bool(training_raw.get("optimizer_update", False)),
        ),
        boards=_tuple(compatibility.get("boards", [])),
        toolchains=toolchains,
        precisions=_tuple(compatibility.get("precisions", capabilities.get("precisions", []))),
        interfaces=_interface_contracts(raw.get("interfaces", {})),
        weight_storage=_tuple(memory.get("supported_weight_storage", memory.get("weight_storage", []))),
        activation_storage=_tuple(memory.get("supported_activation_storage", memory.get("activation_storage", []))),
        validation_level=str(validation.get("declared_level", "unvalidated")),
        metrics=ImplementationMetrics(**{key: value for key, value in metrics_raw.items() if key in ImplementationMetrics.__dataclass_fields__}),
        license_category=str(license_cfg.get("category", "research_only")),
        package_root=manifest.package_root,
        manifest_hash=manifest_hash,
        metadata={
            "name": str(package.get("name", manifest.package_id)),
            "description": str(package.get("description", "")),
            "integration": dict(raw.get("integration", {})) if isinstance(raw.get("integration", {}), Mapping) else {},
        },
    )


def validation_rank(level: str) -> int:
    return _VALIDATION_RANK.get(level, -1)


def valid_selection_policy(value: str) -> bool:
    return value in _VALID_POLICIES
