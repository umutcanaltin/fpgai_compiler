from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping

from fpgai.compiler.capabilities import capability_for


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
    "Softmax": {"category": "activation", "has_weights": False, "tiling": "limited_by_vector_reduction_backend"},
    "Flatten": {"category": "reshape", "has_weights": False, "tiling": "not_required_or_linear_copy_only"},
    "Reshape": {"category": "reshape", "has_weights": False, "tiling": "not_required_or_linear_copy_only"},
    "Add": {"category": "elementwise", "has_weights": False, "tiling": "limited_by_sequential_graph_backend", "notes": ("General branched Add requires graph scheduling support.",)},
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
