from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class LayerDescriptor:
    node_name: str
    op_type: str
    inputs: List[str]
    outputs: List[str]

    input_shapes: List[Tuple[int, ...]] = field(default_factory=list)
    output_shapes: List[Tuple[int, ...]] = field(default_factory=list)

    param_names: List[str] = field(default_factory=list)
    param_bytes: int = 0
    activation_bytes_in: int = 0
    activation_bytes_out: int = 0

    macs: int = 0
    attrs: Dict[str, Any] = field(default_factory=dict)

    compute_hint: str = "unknown"   # compute_bound / memory_bound / balanced / unknown
    backend_kernel: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LayerPlan:
    node_name: str
    op_type: str

    precision_mode: str = "float32"
    act_bits: Optional[int] = None
    weight_bits: Optional[int] = None

    tile: Dict[str, int] = field(default_factory=dict)
    unroll: Dict[str, int] = field(default_factory=dict)

    pipeline_ii: Optional[int] = 1
    weight_mode: str = "embedded"   # embedded / stream / ddr
    activation_mode: str = "stream" # stream / buffer / ddr
    buffering: str = "single"       # single / double

    backend_kernel: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CompilePlan:
    target_board: str = "unknown"
    target_part: str = "unknown"
    clock_mhz: float = 200.0

    execution_order: List[str] = field(default_factory=list)
    layer_plans: List[LayerPlan] = field(default_factory=list)

    global_resource_budget: Dict[str, Any] = field(default_factory=dict)
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_board": self.target_board,
            "target_part": self.target_part,
            "clock_mhz": self.clock_mhz,
            "execution_order": self.execution_order,
            "layer_plans": [lp.to_dict() for lp in self.layer_plans],
            "global_resource_budget": self.global_resource_budget,
            "notes": self.notes,
        }


@dataclass
class TensorPlacement:
    tensor_name: str
    kind: str                    # input / output / weight / activation / temp
    region: str                  # BRAM / URAM / LUTRAM / DDR / HOST
    layout: str = "raw"          # raw / packed / tiled
    offset: Optional[int] = None
    size_bytes: int = 0
    double_buffer: bool = False
    producer: Optional[str] = None
    consumer: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryPlan:
    placements: List[TensorPlacement] = field(default_factory=list)
    total_bytes_by_region: Dict[str, int] = field(default_factory=dict)
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "placements": [p.to_dict() for p in self.placements],
            "total_bytes_by_region": self.total_bytes_by_region,
            "notes": self.notes,
        }


@dataclass
class CommunicationEdge:
    tensor_name: str
    direction: str               # PS_TO_PL / PL_TO_PS / PL_TO_PL
    encoding: str = "raw"        # raw / bitpack / rle
    packed_bits: Optional[int] = None
    axi_word_bits: int = 32
    burst_len: int = 16
    unpack_in_pl: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CommunicationPlan:
    edges: List[CommunicationEdge] = field(default_factory=list)
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edges": [e.to_dict() for e in self.edges],
            "notes": self.notes,
        }