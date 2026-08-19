import numpy as np

from fpgai.quantization import (
    QuantizationParameters,
    QuantizationSpec,
    derive_requantization_contract,
    requantize_integer,
)


def _params(bits: int, scale: float, *, saturation: str = "saturate") -> QuantizationParameters:
    spec = QuantizationSpec(
        bits=bits,
        scheme="symmetric",
        granularity="per_tensor",
        rounding="nearest",
        saturation=saturation,
    )
    return QuantizationParameters(spec, scale, 0, -1.0, 1.0)


def test_requantization_widen_and_narrow_reference():
    q8 = _params(8, 0.5)
    q16 = _params(16, 0.25)
    up = derive_requantization_contract(q8, q16)
    down = derive_requantization_contract(q16, q8)

    assert up.source_bits == 8
    assert up.destination_bits == 16
    assert up.scale_ratio == 2.0
    assert up.lossy is False
    assert requantize_integer(np.asarray([7, -7]), up).tolist() == [14, -14]

    # 15 @ scale 0.25 = 3.75 -> nearest representable int8 @ scale 0.5 is 8.
    assert requantize_integer(np.asarray([15, -15]), down).tolist() == [8, -8]
    assert down.lossy is True


def test_requantization_saturates_to_destination_range():
    q16 = _params(16, 1.0)
    q8 = _params(8, 1.0)
    contract = derive_requantization_contract(q16, q8)
    values = requantize_integer(np.asarray([1000, -1000]), contract)
    assert values.tolist() == [127, -128]
