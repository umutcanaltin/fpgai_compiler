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
    "Conv": OperatorCapability(
        "limited",
        "Only the convolution shapes accepted by the HLS emitter are supported.",
    ),
    "MaxPool": OperatorCapability("supported"),
    "AvgPool": OperatorCapability("supported"),
    "BatchNormalization": OperatorCapability("supported"),
    "Relu": OperatorCapability("supported"),
    "LeakyRelu": OperatorCapability("supported"),
    "Sigmoid": OperatorCapability("supported"),
    "Softmax": OperatorCapability("supported"),
    "Flatten": OperatorCapability("supported"),
    "Reshape": OperatorCapability(
        "limited",
        "Reshape is implemented as a copy or CHW-to-flat layout conversion.",
    ),
    "Add": OperatorCapability(
        "limited",
        "General branched Add tensors are not yet resolved by the sequential HLS emitter.",
    ),
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