from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from fpgai.engine.training import OP_TRAINING_CAPS


@dataclass(frozen=True)
class OperatorCapability:
    status: str
    detail: str = ""

    @property
    def supported(self) -> bool:
        return self.status in {"supported", "limited"}

    def to_dict(self) -> Dict[str, object]:
        return {
            "status": self.status,
            "supported": self.supported,
            "detail": self.detail,
        }


INFERENCE_HLS_CAPABILITIES: Dict[str, OperatorCapability] = {
    "Dense": OperatorCapability("supported"),
    "Linear": OperatorCapability("supported", "Alias of Dense/linear layer in the HLS emitter."),
    "Conv": OperatorCapability(
        "supported",
        "Convolution forward path is implemented by the HLS emitter for compiler-normalized Conv layers; unsupported shapes must reject in shape/planning validation.",
    ),
    "Conv2D": OperatorCapability(
        "supported",
        "Alias of compiler-normalized Conv/2D convolution backend.",
    ),
    "DepthwiseConv2D": OperatorCapability(
        "supported",
        "Depthwise convolution is a grouped-convolution specialization and must lower through the convolution backend or reject with shape/group reason.",
    ),
    "PointwiseConv2D": OperatorCapability(
        "supported",
        "Pointwise convolution is a 1x1 convolution specialization and lowers through the convolution backend.",
    ),
    "MaxPool": OperatorCapability("supported"),
    "AvgPool": OperatorCapability("supported"),
    "AveragePool": OperatorCapability("supported", "Alias of AvgPool."),
    "GlobalAveragePool": OperatorCapability("supported"),
    "BatchNormalization": OperatorCapability("supported"),
    "BatchNorm": OperatorCapability("supported", "Alias of BatchNormalization."),
    "Relu": OperatorCapability("supported"),
    "LeakyRelu": OperatorCapability("supported"),
    "Sigmoid": OperatorCapability("supported"),
    "SiLU": OperatorCapability("supported", "Elementwise SiLU/Swish is implemented by the shared DAG HLS emitter."),
    "Softmax": OperatorCapability("supported", "Static arbitrary-axis Softmax is implemented for inference and training HLS, including YOLO DFL-style middle-axis reductions."),
    "MatMul": OperatorCapability("supported", "Tiled static rank-2 or batch-1 rank-3 MatMul is implemented by the DAG HLS emitter."),
    "Transpose": OperatorCapability("limited", "Static last-two-dimension transpose used by attention is implemented by the DAG HLS emitter."),
    "Mul": OperatorCapability("supported", "Scalar scaling and equal-shaped tensor elementwise multiplication are implemented by the DAG HLS emitter."),
    "LayerNormalization": OperatorCapability("limited", "Static last-axis LayerNormalization is implemented by the DAG HLS emitter."),
    "RMSNorm": OperatorCapability("limited", "Static last-axis RMSNorm is implemented by the DAG HLS emitter."),
    "CausalMask": OperatorCapability("limited", "Static square causal attention masking is implemented by the DAG HLS emitter."),
    "RotaryEmbedding": OperatorCapability("limited", "Pairwise RoPE supports compiler-provided cosine/sine tables with either static position_offset or a one-element runtime integer position input."),
    "MultiHeadAttention": OperatorCapability("limited", "Serialized HLS attention supports equal-length full-sequence execution and valid-length-aware cached attention, including grouped-query attention (GQA) with num_heads divisible by num_kv_heads; full training parity and external-memory cache execution remain limited."),
    "GroupQueryAttention": OperatorCapability("limited", "Serialized batch-1 explicit-cache GQA is implemented for static bounded cache tensors, including ORT contrib past/present cache and optional fused RoPE semantics. Training should use decomposed attention operators."),
    "KVCacheUpdate": OperatorCapability("limited", "Persistent append is implemented for on-chip BRAM/URAM state and explicit external DDR/host m_axi state ports in the DAG HLS backend; full board-runtime cursor/reset orchestration and training parity remain limited."),
    "PersistentStateRead": OperatorCapability("limited", "Reads a dedicated on-chip persistent BRAM/URAM state tensor into the graph; DDR-backed state remains external-backend work."),
    "PersistentStateLength": OperatorCapability("limited", "Exposes the current persistent-state append cursor as a one-element integer tensor for decode position/mask logic."),
    "PersistentStateReset": OperatorCapability("limited", "Resets dedicated on-chip persistent state and its cursor when a one-element runtime reset flag is asserted."),
    "GatedMLP": OperatorCapability("limited", "Composite gated/SwiGLU MLP layer expands to MatMul, SiLU, Mul, and MatMul before backend lowering."),
    "TransformerBlock": OperatorCapability("limited", "Compiler-level composite layer that expands to ordinary FPGAI IR operators before HLS/VHDL implementation selection; backend support depends on expanded operators."),
    "Flatten": OperatorCapability("supported"),
    "Reshape": OperatorCapability(
        "limited",
        "Reshape is implemented as a copy or CHW-to-flat layout conversion.",
    ),
    "Add": OperatorCapability(
        "limited",
        "General branched Add tensors are not yet resolved by the sequential HLS emitter.",
    ),
    "Concat": OperatorCapability("supported", "Static-axis tensor concatenation supports arbitrary runtime input fan-in in the shared DAG HLS emitter."),
    "Split": OperatorCapability("limited", "Static ONNX Split is canonicalized into ordinary Slice operators before backend lowering; dynamic split sizes remain unsupported."),
    "Slice": OperatorCapability("limited", "Static one-axis unit-step Slice is implemented by the shared DAG HLS emitter."),
    "Resize": OperatorCapability("limited", "Static NCHW nearest-neighbor Resize supports asymmetric, half_pixel, and align_corners coordinate transforms with ONNX nearest rounding modes; linear/cubic interpolation remains unsupported."),
    "Gather": OperatorCapability("limited", "Axis-0 row Gather/embedding lookup is implemented with typed integer index tensors; higher-rank/general-axis gather remains limited."),
    "Identity": OperatorCapability("supported", "Static identity is lowered as a typed copy."),
    "Cast": OperatorCapability("limited", "Static tensor cast between FPGAI numeric tensor types is lowered as a typed copy/conversion."),
    "Squeeze": OperatorCapability("limited", "Static squeeze with unchanged flattened element count is lowered as a typed copy."),
    "Unsqueeze": OperatorCapability("limited", "Static unsqueeze with unchanged flattened element count is lowered as a typed copy."),
    "Sub": OperatorCapability("limited", "Equal-shaped runtime tensors and runtime-minus-scalar forms are implemented."),
    "Div": OperatorCapability("limited", "Equal-shaped runtime tensors and runtime-divided-by-scalar forms are implemented."),
    "Sqrt": OperatorCapability("limited", "Static elementwise square root is implemented by the DAG HLS emitter."),
    "Pow": OperatorCapability("limited", "Runtime tensor raised to constant exponent 2 is implemented by the DAG HLS emitter."),
    "ReduceMean": OperatorCapability("limited", "Static last-axis ReduceMean is implemented by the DAG HLS emitter."),
    "ReduceSum": OperatorCapability("supported", "Static arbitrary-axis ReduceSum is implemented for inference and training HLS, including distribution-expectation reductions used by detection heads."),
}


def capability_for(
    op_type: str,
    pipeline_mode: str,
) -> OperatorCapability:
    inference = INFERENCE_HLS_CAPABILITIES.get(
        op_type,
        OperatorCapability(
            "unsupported",
            "No HLS emitter is registered for this operator.",
        ),
    )

    if pipeline_mode != "training_on_device":
        return inference

    training = OP_TRAINING_CAPS.get(op_type)

    if training is None or not training.backward_input:
        return OperatorCapability(
            "unsupported",
            "The training backend does not implement backward propagation "
            "for this operator.",
        )

    if inference.status == "unsupported":
        return inference

    if inference.status == "limited":
        return inference

    if training.backward_params and training.update:
        return OperatorCapability(
            "supported",
            "Forward, backward, and parameter update supported.",
        )

    return OperatorCapability(
        "supported",
        "Forward and backward-input propagation supported.",
    )