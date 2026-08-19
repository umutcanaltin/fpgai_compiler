from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fpgai.quantization.contracts import QuantizationParameters
from fpgai.quantization.ptq import fake_quantize


@dataclass(frozen=True)
class FakeQuantResult:
    values: np.ndarray
    parameters: QuantizationParameters


def qat_fake_quant_forward(values: np.ndarray, parameters: QuantizationParameters) -> FakeQuantResult:
    """Apply fake quantization while retaining floating-point tensor storage."""
    return FakeQuantResult(values=fake_quantize(values, parameters), parameters=parameters)


def straight_through_gradient(gradient: np.ndarray, *, enabled: bool = True) -> np.ndarray:
    """Reference STE used by the QAT contract.

    FPGAI training integration will route gradients through this contract when
    fake-quant nodes are present.  The reference form is identity when enabled.
    """
    grad = np.asarray(gradient, dtype=np.float32)
    if not enabled:
        return np.zeros_like(grad)
    return grad.copy()
