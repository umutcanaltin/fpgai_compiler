import numpy as np

from fpgai.quantization import QuantizationSpec, calibrate_ptq, fake_quantize, quantize, validate_fake_quantization


def test_ptq_minmax_symmetric_per_tensor_round_trip():
    samples = [
        np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32),
        np.asarray([-1.5, 0.5], dtype=np.float32),
    ]
    result = calibrate_ptq(samples, QuantizationSpec(bits=8), method="min_max")
    assert result.sample_count == 2
    assert result.parameters.zero_point == 0
    q = quantize(samples[0], result.parameters)
    assert q.min() >= -128 and q.max() <= 127
    restored = fake_quantize(samples[0], result.parameters)
    assert restored.shape == samples[0].shape
    metrics = validate_fake_quantization(samples[0], result.parameters)
    assert metrics.cosine > 0.999
    assert metrics.max_abs < 0.02


def test_ptq_asymmetric_unsigned_uses_zero_point():
    spec = QuantizationSpec(bits=8, scheme="asymmetric", signed=False)
    result = calibrate_ptq([np.asarray([0.0, 1.0, 2.0, 4.0], dtype=np.float32)], spec)
    assert 0 <= result.parameters.zero_point <= 255
    q = quantize(np.asarray([0.0, 4.0], dtype=np.float32), result.parameters)
    assert q[0] >= 0 and q[1] <= 255


def test_ptq_per_channel_weight_calibration():
    weights = np.asarray([
        [[[-1.0, 1.0]]],
        [[[-4.0, 4.0]]],
    ], dtype=np.float32)
    spec = QuantizationSpec(bits=8, granularity="per_channel", axis=0)
    result = calibrate_ptq([weights], spec)
    scales = result.parameters.scale
    assert isinstance(scales, tuple)
    assert len(scales) == 2
    assert scales[1] > scales[0]
    restored = fake_quantize(weights, result.parameters)
    assert restored.shape == weights.shape


def test_percentile_calibration_reduces_outlier_range():
    values = np.concatenate([np.linspace(-1.0, 1.0, 1000), np.asarray([100.0])]).astype(np.float32)
    spec = QuantizationSpec(bits=8)
    minmax = calibrate_ptq([values], spec, method="min_max")
    percentile = calibrate_ptq([values], spec, method="percentile", percentile=99.0)
    assert percentile.parameters.scale < minmax.parameters.scale
