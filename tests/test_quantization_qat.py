import numpy as np

from fpgai.quantization import QuantizationSpec, calibrate_ptq, qat_fake_quant_forward, straight_through_gradient


def test_qat_fake_quant_and_ste_reference_contract():
    params = calibrate_ptq([np.asarray([-1.0, 0.0, 1.0], dtype=np.float32)], QuantizationSpec(bits=8)).parameters
    result = qat_fake_quant_forward(np.asarray([-0.7, 0.3], dtype=np.float32), params)
    assert result.values.dtype == np.float32
    grad = np.asarray([1.0, -2.0], dtype=np.float32)
    np.testing.assert_array_equal(straight_through_gradient(grad), grad)
