import pytest

from fpgai.quantization import QuantizationSpec, quantization_spec_from_mapping


def test_quantization_spec_requires_axis_for_per_channel():
    with pytest.raises(ValueError, match="requires axis"):
        QuantizationSpec(bits=8, granularity="per_channel")


def test_quantization_spec_mapping_is_deterministic():
    spec = quantization_spec_from_mapping({
        "bits": 8,
        "scheme": "symmetric",
        "granularity": "per_channel",
        "axis": 0,
        "rounding": "nearest",
        "saturation": "saturate",
    })
    assert spec.bits == 8
    assert spec.axis == 0
    assert spec.qmin == -128
    assert spec.qmax == 127
