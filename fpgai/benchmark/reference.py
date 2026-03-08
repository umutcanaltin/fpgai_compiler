from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import onnxruntime as ort


@dataclass(frozen=True)
class ReferenceRunResult:
    input_npy: Path
    output_npy: Path
    input_name: str
    output_names: Tuple[str, ...]
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]


def _normalize_shape(shape) -> Tuple[int, ...]:
    dims = []
    for d in shape:
        if isinstance(d, int) and d > 0:
            dims.append(int(d))
        else:
            dims.append(1)
    return tuple(dims)


def _make_input_array(input_shape: Tuple[int, ...], seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(size=input_shape).astype(np.float32)
    return x


def run_onnx_reference(
    *,
    model_path: str | Path,
    out_dir: str | Path,
    seed: int = 0,
) -> ReferenceRunResult:
    model_path = Path(model_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    inputs = sess.get_inputs()
    outputs = sess.get_outputs()

    if len(inputs) != 1:
        raise ValueError(f"Expected exactly 1 ONNX input, got {len(inputs)}")
    if len(outputs) != 1:
        raise ValueError(f"Expected exactly 1 ONNX output for now, got {len(outputs)}")

    input_meta = inputs[0]
    output_meta = outputs[0]

    input_name = input_meta.name
    output_name = output_meta.name

    input_shape = _normalize_shape(input_meta.shape)
    x = _make_input_array(input_shape, seed=seed)

    y = sess.run([output_name], {input_name: x})[0]
    y = np.asarray(y, dtype=np.float32)

    input_npy = out_dir / "reference_input.npy"
    output_npy = out_dir / "reference_output.npy"

    np.save(input_npy, x)
    np.save(output_npy, y)

    if not input_npy.exists():
        raise RuntimeError(f"Failed to write reference input: {input_npy}")
    if not output_npy.exists():
        raise RuntimeError(f"Failed to write reference output: {output_npy}")

    return ReferenceRunResult(
        input_npy=input_npy,
        output_npy=output_npy,
        input_name=input_name,
        output_names=(output_name,),
        input_shape=tuple(int(v) for v in x.shape),
        output_shape=tuple(int(v) for v in y.shape),
    )