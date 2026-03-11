from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper


@dataclass(frozen=True)
class ReferenceIntermediateResult:
    out_dir: Path
    saved: Dict[str, Path]


def _clone_model_with_extra_outputs(model_path: Path, extra_output_names: List[str], temp_model_path: Path) -> Path:
    model = onnx.load(str(model_path))
    g = model.graph

    existing = {o.name for o in g.output}
    for name in extra_output_names:
        if name in existing:
            continue
        g.output.append(helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))

    onnx.save(model, str(temp_model_path))
    return temp_model_path


def run_onnx_intermediate_reference(
    *,
    model_path: str | Path,
    input_array: np.ndarray,
    graph,
    out_dir: str | Path,
) -> ReferenceIntermediateResult:
    model_path = Path(model_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # map stable op name -> first output tensor name
    wanted_tensor_names: List[str] = []
    tensor_to_op: Dict[str, str] = {}

    for op in graph.ops:
        if not op.outputs:
            continue
        tname = op.outputs[0]
        wanted_tensor_names.append(tname)
        tensor_to_op[tname] = op.name

    temp_model = out_dir / "_intermediate_model.onnx"
    _clone_model_with_extra_outputs(model_path, wanted_tensor_names, temp_model)

    sess = ort.InferenceSession(str(temp_model), providers=["CPUExecutionProvider"])
    inputs = sess.get_inputs()
    if len(inputs) != 1:
        raise ValueError(f"Expected 1 ONNX input, got {len(inputs)}")

    input_name = inputs[0].name
    output_names = list(wanted_tensor_names)
    values = sess.run(output_names, {input_name: input_array.astype(np.float32)})

    saved: Dict[str, Path] = {}
    for tname, arr in zip(output_names, values):
        op_name = tensor_to_op[tname]
        p = out_dir / f"{op_name}.npy"
        np.save(p, np.asarray(arr, dtype=np.float32))
        saved[op_name] = p

    return ReferenceIntermediateResult(
        out_dir=out_dir,
        saved=saved,
    )