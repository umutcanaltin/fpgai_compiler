from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


class ObserverError(ValueError):
    pass


@dataclass
class MinMaxObserver:
    axis: int | None = None
    minimum: np.ndarray | float | None = None
    maximum: np.ndarray | float | None = None
    sample_count: int = 0

    def observe(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0:
            raise ObserverError("cannot observe an empty tensor")
        if not np.all(np.isfinite(array)):
            raise ObserverError("calibration tensor contains non-finite values")

        if self.axis is None:
            current_min: np.ndarray | float = float(np.min(array))
            current_max: np.ndarray | float = float(np.max(array))
        else:
            axis = self.axis if self.axis >= 0 else array.ndim + self.axis
            if axis < 0 or axis >= array.ndim:
                raise ObserverError(f"observer axis {self.axis} is invalid for rank-{array.ndim} tensor")
            reduce_axes = tuple(index for index in range(array.ndim) if index != axis)
            current_min = np.min(array, axis=reduce_axes)
            current_max = np.max(array, axis=reduce_axes)

        if self.minimum is None:
            self.minimum = current_min
            self.maximum = current_max
        else:
            self.minimum = np.minimum(self.minimum, current_min)
            self.maximum = np.maximum(self.maximum, current_max)
        self.sample_count += 1

    def range(self) -> tuple[np.ndarray | float, np.ndarray | float]:
        if self.minimum is None or self.maximum is None or self.sample_count == 0:
            raise ObserverError("observer has no calibration samples")
        return self.minimum, self.maximum


@dataclass
class PercentileObserver:
    percentile: float = 99.99
    axis: int | None = None
    _samples: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 < float(self.percentile) <= 100.0:
            raise ObserverError("percentile must be in (0, 100]")

    def observe(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0:
            raise ObserverError("cannot observe an empty tensor")
        if not np.all(np.isfinite(array)):
            raise ObserverError("calibration tensor contains non-finite values")
        self._samples.append(array.copy())

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    def range(self) -> tuple[np.ndarray | float, np.ndarray | float]:
        if not self._samples:
            raise ObserverError("observer has no calibration samples")
        if self.axis is None:
            flat = np.concatenate([sample.reshape(-1) for sample in self._samples])
            bound = float(np.percentile(np.abs(flat), self.percentile))
            return -bound, bound

        rank = self._samples[0].ndim
        axis = self.axis if self.axis >= 0 else rank + self.axis
        if axis < 0 or axis >= rank:
            raise ObserverError(f"observer axis {self.axis} is invalid for rank-{rank} tensor")
        if any(sample.ndim != rank or sample.shape[axis] != self._samples[0].shape[axis] for sample in self._samples):
            raise ObserverError("per-channel percentile samples must have compatible channel dimensions")
        channel_values: list[np.ndarray] = []
        for channel in range(self._samples[0].shape[axis]):
            chunks = [np.take(sample, channel, axis=axis).reshape(-1) for sample in self._samples]
            channel_values.append(np.concatenate(chunks))
        bounds = np.asarray([np.percentile(np.abs(values), self.percentile) for values in channel_values], dtype=np.float64)
        return -bounds, bounds


def observe_many(observer: MinMaxObserver | PercentileObserver, samples: Iterable[np.ndarray]) -> None:
    count = 0
    for sample in samples:
        observer.observe(sample)
        count += 1
    if count == 0:
        raise ObserverError("calibration requires at least one sample")
