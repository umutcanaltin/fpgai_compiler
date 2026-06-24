from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


def _positive_int(value: Any, default: int = 1) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return max(1, int(default))


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def _stable_signature(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    compute_hint: str = "unknown"
    backend_kernel: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PrecisionPlan:
    mode: str = "float32"
    activation_bits: Optional[int] = None
    weight_bits: Optional[int] = None
    bias_bits: Optional[int] = None
    accumulator_bits: Optional[int] = None
    activation_int_bits: Optional[int] = None
    weight_int_bits: Optional[int] = None
    bias_int_bits: Optional[int] = None
    accumulator_int_bits: Optional[int] = None

    def __post_init__(self) -> None:
        for name in (
            "activation_bits",
            "weight_bits",
            "bias_bits",
            "accumulator_bits",
            "activation_int_bits",
            "weight_int_bits",
            "bias_int_bits",
            "accumulator_int_bits",
        ):
            value = getattr(self, name)
            if value is not None and int(value) <= 0:
                raise ValueError(f"{name} must be positive when provided")

        pairs = (
            ("activation_int_bits", "activation_bits"),
            ("weight_int_bits", "weight_bits"),
            ("bias_int_bits", "bias_bits"),
            ("accumulator_int_bits", "accumulator_bits"),
        )
        for int_name, total_name in pairs:
            int_bits = getattr(self, int_name)
            total_bits = getattr(self, total_name)
            if (
                int_bits is not None
                and total_bits is not None
                and int(int_bits) > int(total_bits)
            ):
                raise ValueError(f"{int_name} cannot exceed {total_name}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PipelinePlan:
    ii: int = 1
    style: str = "balanced"
    scope: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "ii", _positive_int(self.ii))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ParallelismPlan:
    pe: int = 1
    simd: int = 1
    unroll: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pe", _positive_int(self.pe))
        object.__setattr__(self, "simd", _positive_int(self.simd))
        object.__setattr__(
            self,
            "unroll",
            {
                str(key): _positive_int(value)
                for key, value in self.unroll.items()
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pe": self.pe,
            "simd": self.simd,
            "unroll": dict(self.unroll),
        }


@dataclass(frozen=True)
class PartitionPlan:
    factor: int = 1
    mode: str = "none"
    targets: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "factor", _positive_int(self.factor))
        object.__setattr__(
            self,
            "targets",
            {
                str(key): _positive_int(value)
                for key, value in self.targets.items()
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "factor": self.factor,
            "mode": self.mode,
            "targets": dict(self.targets),
        }


@dataclass(frozen=True)
class TilingPlan:
    sizes: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sizes",
            {
                str(key): _positive_int(value)
                for key, value in self.sizes.items()
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"sizes": dict(self.sizes)}


@dataclass(frozen=True)
class BufferingPlan:
    mode: str = "single"
    double_buffer: bool = False

    def __post_init__(self) -> None:
        normalized = str(self.mode).strip().lower()
        if normalized not in {"single", "double"}:
            raise ValueError("buffering mode must be 'single' or 'double'")
        object.__setattr__(self, "mode", normalized)
        object.__setattr__(
            self,
            "double_buffer",
            normalized == "double",
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LayerMemoryPlan:
    weight_mode: str = "embedded"
    activation_mode: str = "stream"
    weight_region: Optional[str] = None
    activation_region: Optional[str] = None
    gradient_region: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArchitecturePlan:
    precision: PrecisionPlan = field(default_factory=PrecisionPlan)
    pipeline: PipelinePlan = field(default_factory=PipelinePlan)
    parallelism: ParallelismPlan = field(
        default_factory=ParallelismPlan
    )
    partitioning: PartitionPlan = field(
        default_factory=PartitionPlan
    )
    tiling: TilingPlan = field(default_factory=TilingPlan)
    buffering: BufferingPlan = field(
        default_factory=BufferingPlan
    )
    memory: LayerMemoryPlan = field(
        default_factory=LayerMemoryPlan
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "precision": self.precision.to_dict(),
            "pipeline": self.pipeline.to_dict(),
            "parallelism": self.parallelism.to_dict(),
            "partitioning": self.partitioning.to_dict(),
            "tiling": self.tiling.to_dict(),
            "buffering": self.buffering.to_dict(),
            "memory": self.memory.to_dict(),
        }

    @property
    def signature(self) -> str:
        return _stable_signature(self.to_dict())


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
    weight_mode: str = "embedded"
    activation_mode: str = "stream"
    buffering: str = "single"
    backend_kernel: Optional[str] = None
    notes: Dict[str, Any] = field(default_factory=dict)
    architecture: Optional[ArchitecturePlan] = None

    def __post_init__(self) -> None:
        self.act_bits = _optional_int(self.act_bits)
        self.weight_bits = _optional_int(self.weight_bits)
        self.tile = {
            str(key): _positive_int(value)
            for key, value in self.tile.items()
        }
        self.unroll = {
            str(key): _positive_int(value)
            for key, value in self.unroll.items()
        }
        self.pipeline_ii = _positive_int(self.pipeline_ii or 1)

        if self.architecture is None:
            pe = self.unroll.get(
                "out",
                self.unroll.get("oc", 1),
            )
            simd = self.unroll.get(
                "in",
                self.unroll.get("ic", 1),
            )
            partition_factor = _positive_int(
                self.notes.get("partition_factor", 1)
            )
            partition_mode = str(
                self.notes.get("partition_mode", "none")
            )

            self.architecture = ArchitecturePlan(
                precision=PrecisionPlan(
                    mode=self.precision_mode,
                    activation_bits=self.act_bits,
                    weight_bits=self.weight_bits,
                    bias_bits=_optional_int(
                        self.notes.get("requested_bias_bits")
                    ),
                    accumulator_bits=_optional_int(
                        self.notes.get("requested_accum_bits")
                    ),
                    activation_int_bits=_optional_int(
                        self.notes.get("requested_act_int_bits")
                    ),
                    weight_int_bits=_optional_int(
                        self.notes.get("requested_weight_int_bits")
                    ),
                    bias_int_bits=_optional_int(
                        self.notes.get("requested_bias_int_bits")
                    ),
                    accumulator_int_bits=_optional_int(
                        self.notes.get("requested_accum_int_bits")
                    ),
                ),
                pipeline=PipelinePlan(
                    ii=self.pipeline_ii,
                    style=str(
                        self.notes.get(
                            "pipeline_style",
                            "balanced",
                        )
                    ),
                ),
                parallelism=ParallelismPlan(
                    pe=pe,
                    simd=simd,
                    unroll=self.unroll,
                ),
                partitioning=PartitionPlan(
                    factor=partition_factor,
                    mode=partition_mode,
                ),
                tiling=TilingPlan(sizes=self.tile),
                buffering=BufferingPlan(mode=self.buffering),
                memory=LayerMemoryPlan(
                    weight_mode=self.weight_mode,
                    activation_mode=self.activation_mode,
                ),
            )

    @property
    def architecture_signature(self) -> str:
        assert self.architecture is not None
        payload = {
            "op_type": self.op_type,
            "architecture": self.architecture.to_dict(),
        }
        return _stable_signature(payload)

    def to_dict(self) -> Dict[str, Any]:
        assert self.architecture is not None
        return {
            "node_name": self.node_name,
            "op_type": self.op_type,
            "precision_mode": self.precision_mode,
            "act_bits": self.act_bits,
            "weight_bits": self.weight_bits,
            "tile": dict(self.tile),
            "unroll": dict(self.unroll),
            "pipeline_ii": self.pipeline_ii,
            "weight_mode": self.weight_mode,
            "activation_mode": self.activation_mode,
            "buffering": self.buffering,
            "backend_kernel": self.backend_kernel,
            "notes": dict(self.notes),
            "architecture": self.architecture.to_dict(),
            "architecture_signature": self.architecture_signature,
        }


@dataclass
class CompilePlan:
    target_board: str = "unknown"
    target_part: str = "unknown"
    clock_mhz: float = 200.0
    execution_order: List[str] = field(default_factory=list)
    layer_plans: List[LayerPlan] = field(default_factory=list)
    global_resource_budget: Dict[str, Any] = field(default_factory=dict)
    notes: Dict[str, Any] = field(default_factory=dict)

    @property
    def architecture_signature(self) -> str:
        payload = {
            "target_board": self.target_board,
            "target_part": self.target_part,
            "clock_mhz": float(self.clock_mhz),
            "layers": [
                {
                    "op_type": layer.op_type,
                    "architecture_signature": (
                        layer.architecture_signature
                    ),
                }
                for layer in self.layer_plans
            ],
        }
        return _stable_signature(payload)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_board": self.target_board,
            "target_part": self.target_part,
            "clock_mhz": self.clock_mhz,
            "execution_order": list(self.execution_order),
            "layer_plans": [
                layer_plan.to_dict()
                for layer_plan in self.layer_plans
            ],
            "global_resource_budget": dict(
                self.global_resource_budget
            ),
            "notes": dict(self.notes),
            "architecture_signature": self.architecture_signature,
        }


@dataclass
class TensorPlacement:
    tensor_name: str
    kind: str
    region: str
    layout: str = "raw"
    offset: Optional[int] = None
    size_bytes: int = 0
    double_buffer: bool = False
    producer: Optional[str] = None
    consumer: Optional[str] = None
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryPlan:
    placements: List[TensorPlacement] = field(default_factory=list)
    total_bytes_by_region: Dict[str, int] = field(default_factory=dict)
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "placements": [
                placement.to_dict()
                for placement in self.placements
            ],
            "total_bytes_by_region": dict(
                self.total_bytes_by_region
            ),
            "notes": dict(self.notes),
        }


@dataclass
class CommunicationEdge:
    tensor_name: str
    direction: str
    encoding: str = "raw"
    packed_bits: Optional[int] = None
    axi_word_bits: int = 32
    burst_len: int = 16
    unpack_in_pl: bool = False
    size_bytes: int = 0
    transfer_bytes: int = 0
    precision_bits: Optional[int] = None
    compression_enabled: bool = False
    codec: str = "raw"
    source: str = "HOST"
    destination: str = "PL"
    implemented_in_hls: bool = False
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CommunicationPlan:
    edges: List[CommunicationEdge] = field(default_factory=list)
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edges": [
                edge.to_dict()
                for edge in self.edges
            ],
            "notes": dict(self.notes),
        }
