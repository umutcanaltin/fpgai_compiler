from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping

from fpgai.capabilities.capabilities import capability_for


@dataclass(frozen=True)
class LayerKnobSupport:
    precision: str
    pipelining: str
    parallelization: str
    tiling: str
    weight_storage: str
    activation_storage: str
    data_movement: str
    training: str
    notes: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "precision": self.precision,
            "pipelining": self.pipelining,
            "parallelization": self.parallelization,
            "tiling": self.tiling,
            "weight_storage": self.weight_storage,
            "activation_storage": self.activation_storage,
            "data_movement": self.data_movement,
            "training": self.training,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class LayerBackendCapability:
    op_type: str
    category: str
    has_weights: bool
    has_activation_output: bool
    inference_status: str
    training_status: str
    inference_detail: str
    training_detail: str
    knobs: LayerKnobSupport

    @property
    def inference_supported(self) -> bool:
        return self.inference_status in {"supported", "limited"}

    @property
    def training_supported(self) -> bool:
        return self.training_status in {"supported", "limited"}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op_type": self.op_type,
            "category": self.category,
            "has_weights": self.has_weights,
            "has_activation_output": self.has_activation_output,
            "inference": {
                "status": self.inference_status,
                "supported": self.inference_supported,
                "detail": self.inference_detail,
            },
            "training": {
                "status": self.training_status,
                "supported": self.training_supported,
                "detail": self.training_detail,
            },
            "knobs": self.knobs.to_dict(),
        }


def _knobs(
    *,
    has_weights: bool,
    category: str,
    training_status: str,
    tiling: str = "supported_or_limited_by_shape",
    parallelization: str = "supported_or_limited_by_backend",
    notes: Iterable[str] = (),
) -> LayerKnobSupport:
    if has_weights:
        weight_storage = "applies_to_parameter_tensors"
    else:
        weight_storage = "not_applicable_no_weight_tensors"

    if category in {"reshape", "elementwise", "activation", "pooling"}:
        tiling_status = tiling
    else:
        tiling_status = tiling

    return LayerKnobSupport(
        precision="applies_to_compute_and_activation_types",
        pipelining="applies_to_generated_loops_or_rejects_if_no_loop",
        parallelization=parallelization,
        tiling=tiling_status,
        weight_storage=weight_storage,
        activation_storage="applies_to_output_activation_buffers",
        data_movement="applies_to_input_output_edges_and_parameter_edges_when_present",
        training=(
            "applies_to_forward_backward_update"
            if training_status in {"supported", "limited"}
            else "rejects_clear_reason_until_backward_is_implemented"
        ),
        notes=tuple(notes),
    )


_LAYER_METADATA: Mapping[str, Dict[str, Any]] = {
    "Dense": {"category": "linear", "has_weights": True},
    "Linear": {"category": "linear", "has_weights": True, "alias_of": "Dense"},
    "Conv": {"category": "convolution", "has_weights": True, "notes": ("Convolution shape/group validation must reject unsupported shapes before HLS.",)},
    "Conv2D": {"category": "convolution", "has_weights": True, "alias_of": "Conv"},
    "DepthwiseConv2D": {"category": "convolution", "has_weights": True, "alias_of": "Conv", "notes": ("DepthwiseConv2D must lower to grouped Conv semantics or reject if groups/depth multiplier are unsupported.",)},
    "PointwiseConv2D": {"category": "convolution", "has_weights": True, "alias_of": "Conv", "notes": ("PointwiseConv2D lowers to 1x1 Conv semantics.",)},
    "MaxPool": {"category": "pooling", "has_weights": False, "tiling": "applies_to_activation_tiles_only"},
    "AvgPool": {"category": "pooling", "has_weights": False, "tiling": "applies_to_activation_tiles_only"},
    "AveragePool": {"category": "pooling", "has_weights": False, "tiling": "applies_to_activation_tiles_only", "alias_of": "AvgPool"},
    "GlobalAveragePool": {"category": "pooling", "has_weights": False, "tiling": "applies_to_activation_tiles_only"},
    "BatchNormalization": {"category": "normalization", "has_weights": True, "notes": ("BatchNorm parameters are treated as parameter tensors for memory/import/export contracts.",)},
    "BatchNorm": {"category": "normalization", "has_weights": True, "alias_of": "BatchNormalization", "notes": ("Alias of BatchNormalization.",)},
    "Relu": {"category": "activation", "has_weights": False, "tiling": "applies_to_activation_tiles_only"},
    "LeakyRelu": {"category": "activation", "has_weights": False, "tiling": "applies_to_activation_tiles_only"},
    "Sigmoid": {"category": "activation", "has_weights": False, "tiling": "applies_to_activation_tiles_only"},
    "SiLU": {"category": "activation", "has_weights": False, "tiling": "applies_to_activation_tiles_only", "notes": ("Elementwise SiLU/Swish activation used by gated MLPs and modern CNN/LLM graphs.",)},
    "Softmax": {"category": "activation", "has_weights": False, "tiling": "axis_reduction", "notes": ("Static arbitrary-axis Softmax is supported in inference/training HLS, including DFL-style middle-axis reductions.",)},
    "Flatten": {"category": "reshape", "has_weights": False, "tiling": "not_required_or_linear_copy_only"},
    "Reshape": {"category": "reshape", "has_weights": False, "tiling": "not_required_or_linear_copy_only"},
    "Add": {"category": "elementwise", "has_weights": False, "tiling": "limited_by_sequential_graph_backend", "notes": ("General branched Add requires graph scheduling support.",)},
    "Mul": {"category": "elementwise", "has_weights": False, "tiling": "supported_at_ir_level", "notes": ("IR/import support is first-class; backend lowering is implementation-dependent.",)},
    "MatMul": {"category": "linear", "has_weights": False, "tiling": "supported_at_ir_level", "notes": ("General tensor MatMul is preserved in IR and is not forced into Dense unless the ONNX fusion pattern proves parameterized linear semantics.",)},
    "Transpose": {"category": "reshape", "has_weights": False, "tiling": "layout_transform", "notes": ("Explicit layout transform for attention and tensor algebra graphs.",)},
    "LayerNormalization": {"category": "normalization", "has_weights": True, "tiling": "limited_by_reduction_backend", "notes": ("Static last-axis HLS lowering is available; alternative backends remain implementation-selectable.",)},
    "RMSNorm": {"category": "normalization", "has_weights": True, "tiling": "limited_by_reduction_backend", "notes": ("Static last-axis HLS lowering is available and StableHLO decomposition canonicalization is supported.",)},
    "CausalMask": {"category": "elementwise", "has_weights": False, "tiling": "attention_matrix_tiles", "notes": ("Represents causal attention masking explicitly in FPGAI IR; current HLS lowering supports static square score matrices.",)},
    "RotaryEmbedding": {"category": "position_encoding", "has_weights": False, "tiling": "sequence_head_tiles", "notes": ("Pairwise rotary position encoding with compiler-provided cosine/sine tables and static or runtime integer position offsets.",)},
    "MultiHeadAttention": {"category": "attention", "has_weights": False, "tiling": "head_serialized_attention", "notes": ("Serialized HLS supports full-sequence attention and valid-length-aware cached attention over bounded K/V capacity, including GQA head remapping through num_kv_heads; parallel alternatives remain implementation-selectable.",)},
    "GroupQueryAttention": {"category": "attention", "has_weights": False, "tiling": "head_serialized_attention", "notes": ("Generic explicit-cache GQA import/lowering for ONNX Runtime contrib exports; bounded static cache extents are resolved by tensor shapes/YAML overrides.",)},
    "KVCacheUpdate": {"category": "state", "has_weights": False, "tiling": "sequence_state", "notes": ("Persistent BRAM/URAM append and explicit external DDR/host m_axi state ports are implemented by the DAG HLS state backend; board-runtime cursor/reset orchestration remains implementation-selectable.",)},
    "PersistentStateRead": {"category": "state", "has_weights": False, "tiling": "sequence_state", "notes": ("Reads generic dedicated persistent state into the graph without model-specific cache logic.",)},
    "PersistentStateLength": {"category": "state", "has_weights": False, "tiling": "not_required", "notes": ("Exposes the persistent append cursor as a typed integer tensor for decode position and masking logic.",)},
    "PersistentStateReset": {"category": "state", "has_weights": False, "tiling": "sequence_state", "notes": ("Runtime-flag controlled reset of on-chip persistent state and cursor.",)},
    "TransformerBlock": {"category": "composite", "has_weights": True, "tiling": "expanded_before_backend", "notes": ("Composite layer abstraction expanded into ordinary FPGAI IR operators before backend lowering; not a model-specific backend path.",)},
    "GatedMLP": {"category": "composite", "has_weights": True, "tiling": "expanded_before_backend", "notes": ("Composite gated/SwiGLU MLP expanded into ordinary MatMul/SiLU/Mul operators before backend lowering.",)},
    "Sub": {"category": "elementwise", "has_weights": False, "tiling": "supported_at_ir_level"},
    "Div": {"category": "elementwise", "has_weights": False, "tiling": "supported_at_ir_level"},
    "Exp": {"category": "activation", "has_weights": False, "tiling": "supported_at_ir_level"},
    "Rsqrt": {"category": "activation", "has_weights": False, "tiling": "supported_at_ir_level"},
    "Sqrt": {"category": "activation", "has_weights": False, "tiling": "supported_at_ir_level"},
    "ReduceMax": {"category": "reduction", "has_weights": False, "tiling": "limited_by_reduction_axis"},
    "ReduceSum": {"category": "reduction", "has_weights": False, "tiling": "axis_reduction", "notes": ("Static arbitrary-axis HLS forward/backward reduction is implemented; useful for decomposed distribution-expectation heads.",)},
    "Broadcast": {"category": "reshape", "has_weights": False, "tiling": "layout_transform"},
    "Concat": {"category": "tensor", "has_weights": False, "tiling": "axis_partition", "notes": ("Static-axis arbitrary fan-in concatenation is implemented directly by segmented HLS copies.",)},
    "Split": {"category": "tensor", "has_weights": False, "tiling": "axis_partition", "notes": ("Frontend canonicalization lowers static Split into ordinary Slice operators; dynamic split sizes remain explicit gaps.",)},
    "Slice": {"category": "tensor", "has_weights": False, "tiling": "axis_partition", "notes": ("Static one-axis unit-step slice has HLS forward/backward support; general dynamic slicing remains implementation-dependent.",)},
    "Resize": {"category": "tensor", "has_weights": False, "tiling": "spatial_tiles", "notes": ("Static NCHW nearest-neighbor resize implements ONNX coordinate/nearest rounding modes used by exported vision graphs; linear/cubic remain implementation-selectable.",)},
    "Gather": {"category": "tensor", "has_weights": True, "tiling": "row_lookup", "notes": ("Axis-0 row gather/embedding lookup has typed integer-index HLS ingress; general gather axes/ranks remain implementation-dependent.",)},
    "Identity": {"category": "reshape", "has_weights": False, "tiling": "not_required_or_linear_copy_only"},
    "Cast": {"category": "reshape", "has_weights": False, "tiling": "not_required_or_linear_copy_only", "notes": ("Static typed conversion; unsupported dtype combinations reject explicitly.",)},
    "Squeeze": {"category": "reshape", "has_weights": False, "tiling": "not_required_or_linear_copy_only"},
    "Unsqueeze": {"category": "reshape", "has_weights": False, "tiling": "not_required_or_linear_copy_only"},
    "Sub": {"category": "elementwise", "has_weights": False, "tiling": "elementwise"},
    "Div": {"category": "elementwise", "has_weights": False, "tiling": "elementwise"},
    "Sqrt": {"category": "elementwise", "has_weights": False, "tiling": "elementwise"},
    "Pow": {"category": "elementwise", "has_weights": False, "tiling": "elementwise", "notes": ("Current HLS profile supports constant exponent 2.",)},
    "ReduceMean": {"category": "reduction", "has_weights": False, "tiling": "last_axis_reduction", "notes": ("Current HLS profile supports static last-axis reduction.",)},
}


def get_layer_capability(op_type: str, *, pipeline_mode: str = "inference") -> LayerBackendCapability:
    op = str(op_type)
    meta = dict(_LAYER_METADATA.get(op, {}))
    category = str(meta.get("category", "unsupported"))
    has_weights = bool(meta.get("has_weights", False))

    inference = capability_for(op, "inference")
    training = capability_for(op, "training_on_device")

    if pipeline_mode == "training_on_device":
        effective_training = training
    else:
        effective_training = training

    knobs = _knobs(
        has_weights=has_weights,
        category=category,
        training_status=effective_training.status,
        tiling=str(meta.get("tiling", "supported_or_limited_by_shape")),
        notes=tuple(str(x) for x in meta.get("notes", ()) or ()),
    )

    return LayerBackendCapability(
        op_type=op,
        category=category,
        has_weights=has_weights,
        has_activation_output=category != "unsupported",
        inference_status=inference.status,
        training_status=effective_training.status,
        inference_detail=inference.detail,
        training_detail=effective_training.detail,
        knobs=knobs,
    )


def layer_registry(*, pipeline_mode: str = "inference") -> Dict[str, Dict[str, Any]]:
    return {
        op_type: get_layer_capability(op_type, pipeline_mode=pipeline_mode).to_dict()
        for op_type in sorted(_LAYER_METADATA)
    }


def supported_layer_types(*, pipeline_mode: str = "inference") -> list[str]:
    key = "training" if pipeline_mode == "training_on_device" else "inference"
    return [
        op_type
        for op_type, capability in layer_registry(pipeline_mode=pipeline_mode).items()
        if capability[key]["supported"]
    ]
