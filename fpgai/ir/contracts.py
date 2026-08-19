from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple


@dataclass
class MemoryContract:
    storage: str = "unspecified"
    residency: str = "unspecified"
    lifetime: str = "graph"
    banking: Optional[str] = None
    alignment_bytes: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TransportContract:
    protocol: str = "unspecified"
    ready_valid: bool = False
    packet_words: Optional[int] = None
    packing: Optional[str] = None
    variable_length: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingTensorContract:
    role: str = "activation"
    requires_gradient: bool = False
    optimizer_state: bool = False
    checkpoint: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StatefulTensorContract:
    kind: str = "stateless"
    mutable: bool = False
    persistent_across_invocations: bool = False
    update_policy: str = "none"
    sequence_axis: Optional[int] = None
    capacity: Optional[int] = None
    overflow_policy: str = "saturate"
    owner: Optional[str] = None
    state_group: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TensorSemantics:
    memory: MemoryContract = field(default_factory=MemoryContract)
    transport: TransportContract = field(default_factory=TransportContract)
    training: TrainingTensorContract = field(default_factory=TrainingTensorContract)
    state: StatefulTensorContract = field(default_factory=StatefulTensorContract)
    tags: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory": self.memory.to_dict(),
            "transport": self.transport.to_dict(),
            "training": self.training.to_dict(),
            "state": self.state.to_dict(),
            "tags": list(self.tags),
        }


@dataclass
class ImplementationCandidate:
    backend: str
    implementation_id: Optional[str] = None
    package_id: Optional[str] = None
    status: str = "candidate"
    constraints: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OpSemantics:
    implementation_candidates: Tuple[ImplementationCandidate, ...] = ()
    selected_backend: Optional[str] = None
    selected_implementation_id: Optional[str] = None
    buffering: Dict[str, Any] = field(default_factory=dict)
    schedule: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    resource_constraints: Dict[str, Any] = field(default_factory=dict)
    tags: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "implementation_candidates": [item.to_dict() for item in self.implementation_candidates],
            "selected_backend": self.selected_backend,
            "selected_implementation_id": self.selected_implementation_id,
            "buffering": dict(self.buffering),
            "schedule": dict(self.schedule),
            "training": dict(self.training),
            "resource_constraints": dict(self.resource_constraints),
            "tags": list(self.tags),
        }


@dataclass
class GraphSemantics:
    pipeline_mode: str = "inference"
    target_board: Optional[str] = None
    runtime_contract: Dict[str, Any] = field(default_factory=dict)
    resource_constraints: Dict[str, Any] = field(default_factory=dict)
    source_ir: str = "fpgai"
    source_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def semantics_to_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Unsupported IR semantics value: {type(value).__name__}")
