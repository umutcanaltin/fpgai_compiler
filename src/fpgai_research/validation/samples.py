"""Canonical input sample layouts for fixed-batch ONNX research models.

For static batch=1 models, ordinary evaluation batches (N,C,H,W) are
accepted alongside explicit (N,1,C,H,W) samples. The canonical in-memory
representation is always (samples,*model_input_shape).
"""
from __future__ import annotations

def normalize_samples(values, model_shape):
    import numpy as np
    shape=tuple(model_shape)
    array=np.asarray(values)
    if shape and shape[0]==1 and tuple(array.shape)==shape[1:]:
        samples=array.reshape((1,*shape))
    elif tuple(array.shape)==shape:
        samples=array.reshape((1,*shape))
    elif array.ndim==len(shape)+1 and tuple(array.shape[1:])==shape:
        samples=array
    elif shape and shape[0]==1 and array.ndim==len(shape) and tuple(array.shape[1:])==shape[1:]:
        # Restore static ONNX batch=1 axis for each regular framework example.
        samples=array.reshape((array.shape[0],*shape))
    else:
        raise ValueError(f"Expected one sample {shape}, stacked (N,{shape}), "
                         f"or ordinary evaluation batch N+{shape[1:]} when model batch=1")
    if not 1<=len(samples)<=256:
        raise ValueError("Require 1..256 samples; partition larger experiments into shards")
    if not np.all(np.isfinite(samples)):
        raise ValueError("Require finite model inputs")
    return samples
