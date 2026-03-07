from __future__ import annotations

from typing import Dict, Optional, Sequence, Any, List
import onnx
import onnxruntime as ort
import numpy as np
import onnx.numpy_helper as numpy_helper


def run_onnxruntime(
    onnx_model_path: str,
    input_data: np.ndarray,
    *,
    output_names: Optional[Sequence[str]] = None,
    verbose: bool = False,
) -> List[np.ndarray]:
    """
    Run ONNXRuntime inference.

    Args:
        onnx_model_path: Path to ONNX model.
        input_data: Input tensor (numpy).
        output_names: Which outputs to fetch. If None, fetches graph outputs.
        verbose: If True, prints output names/shapes and optionally values.

    Returns:
        List of numpy outputs in the same order as output_names (or graph outputs).
    """
    sess = ort.InferenceSession(onnx_model_path)

    input_name = sess.get_inputs()[0].name

    if output_names is None:
        # By default, only fetch graph outputs (not all intermediate tensors)
        output_names = [o.name for o in sess.get_outputs()]

    outputs = sess.run(list(output_names), {input_name: input_data})

    if verbose:
        for name, out in zip(output_names, outputs):
            # Print shape always; values only if small
            msg = f"[onnxruntime] output '{name}': shape={getattr(out, 'shape', None)}"
            print(msg)
            # Avoid spamming huge arrays unless you really want it:
            if isinstance(out, np.ndarray) and out.size <= 64:
                print(out)

    return outputs


def extract_weights_from_onnx(onnx_model_path: str) -> Dict[str, np.ndarray]:
    model = onnx.load(onnx_model_path)
    weights: Dict[str, np.ndarray] = {}
    for initializer in model.graph.initializer:
        weights[initializer.name] = numpy_helper.to_array(initializer)
    return weights


def update_weights_in_onnx(onnx_model_path: str, new_weights: Dict[str, np.ndarray], save_path: str, *, verbose: bool = False) -> None:
    model = onnx.load(onnx_model_path)
    for initializer in model.graph.initializer:
        if initializer.name in new_weights:
            updated_tensor = numpy_helper.from_array(new_weights[initializer.name], initializer.name)
            initializer.CopyFrom(updated_tensor)

    onnx.save(model, save_path)
    if verbose:
        print(f"[onnx] updated model saved at {save_path}")
