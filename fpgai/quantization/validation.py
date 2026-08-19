from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fpgai.numerics.fixed_emulation import cosine_similarity, mae, max_abs, mse
from fpgai.quantization.contracts import QuantizationParameters
from fpgai.quantization.ptq import fake_quantize


@dataclass(frozen=True)
class QuantizationValidationResult:
    mse: float
    mae: float
    max_abs: float
    cosine: float

    def to_dict(self) -> dict[str, float]:
        return {"mse": self.mse, "mae": self.mae, "max_abs": self.max_abs, "cosine": self.cosine}


def validate_fake_quantization(values: np.ndarray, parameters: QuantizationParameters) -> QuantizationValidationResult:
    reference = np.asarray(values, dtype=np.float32)
    candidate = fake_quantize(reference, parameters)
    return QuantizationValidationResult(
        mse=float(mse(reference, candidate)),
        mae=float(mae(reference, candidate)),
        max_abs=float(max_abs(reference, candidate)),
        cosine=float(cosine_similarity(reference, candidate)),
    )
