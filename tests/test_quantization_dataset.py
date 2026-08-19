from pathlib import Path

import numpy as np
import pytest

from fpgai.quantization import CalibrationDatasetError, load_calibration_samples


def test_load_npy_calibration_samples_uses_first_dimension(tmp_path: Path) -> None:
    path = tmp_path / "calibration.npy"
    np.save(path, np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32))
    samples = load_calibration_samples(path, limit=2)
    assert len(samples) == 2
    assert np.array_equal(samples[0], np.asarray([1.0, 2.0], dtype=np.float32))


def test_npz_with_multiple_arrays_requires_explicit_key(tmp_path: Path) -> None:
    path = tmp_path / "calibration.npz"
    np.savez(path, input=np.asarray([[1.0]], dtype=np.float32), labels=np.asarray([0]))
    with pytest.raises(CalibrationDatasetError, match="array_key"):
        load_calibration_samples(path)
    samples = load_calibration_samples(path, array_key="input")
    assert len(samples) == 1
