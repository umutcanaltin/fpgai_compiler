from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


class CalibrationDatasetError(ValueError):
    pass


def load_calibration_samples(
    path: str | Path,
    *,
    limit: int | None = None,
    array_key: str | None = None,
) -> tuple[np.ndarray, ...]:
    """Load representative calibration samples from .npy or .npz.

    The first dimension is treated as the sample dimension. A one-dimensional
    array is treated as one sample. NPZ inputs must either contain one array or
    provide ``array_key`` explicitly.
    """
    source = Path(path)
    if not source.is_file():
        raise CalibrationDatasetError(f"calibration dataset does not exist: {source}")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise CalibrationDatasetError("calibration sample limit must be a positive integer")

    suffix = source.suffix.lower()
    if suffix == ".npy":
        values = np.load(source, allow_pickle=False)
    elif suffix == ".npz":
        archive = np.load(source, allow_pickle=False)
        try:
            keys = list(archive.files)
            if array_key is not None:
                if array_key not in archive:
                    raise CalibrationDatasetError(
                        f"calibration NPZ key {array_key!r} was not found; available keys: {keys}"
                    )
                values = archive[array_key]
            elif len(keys) == 1:
                values = archive[keys[0]]
            else:
                raise CalibrationDatasetError(
                    "calibration NPZ contains multiple arrays; configure an explicit array_key"
                )
        finally:
            archive.close()
    else:
        raise CalibrationDatasetError("calibration dataset must use .npy or .npz format")

    array = np.asarray(values)
    if array.size == 0:
        raise CalibrationDatasetError("calibration dataset is empty")
    if not np.issubdtype(array.dtype, np.number):
        raise CalibrationDatasetError("calibration dataset must contain numeric values")
    array = np.asarray(array, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise CalibrationDatasetError("calibration dataset contains non-finite values")

    samples: Iterable[np.ndarray]
    if array.ndim <= 1:
        samples = (array,)
    else:
        samples = (array[index] for index in range(array.shape[0]))
    collected = tuple(samples)
    if limit is not None:
        collected = collected[:limit]
    if not collected:
        raise CalibrationDatasetError("calibration dataset produced no samples")
    return collected
