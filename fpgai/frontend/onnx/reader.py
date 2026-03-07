from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np
import onnx
from onnx import numpy_helper
from google.protobuf.json_format import MessageToDict


# -----------------------------------------------------------------------------
# Validation / IO
# -----------------------------------------------------------------------------
def verify_model(model: onnx.ModelProto) -> int | str:
    """Validates an ONNX model. Returns 1 on success, error string otherwise."""
    try:
        onnx.checker.check_model(model)
        return 1
    except onnx.checker.ValidationError as e:
        return f"The model is invalid: {e}"


def get_model_io_shapes(model: onnx.ModelProto) -> Tuple[Optional[List[int]], Optional[Tuple[int, ...]]]:
    """
    Extract runtime input and output tensor shapes from an ONNX model.
    Returns:
        input_shape: first real (non-initializer) input shape
        output_shape: first output tensor shape
    """
    initializer_names: Set[str] = {init.name for init in model.graph.initializer}

    input_shape: Optional[List[int]] = None
    output_shape: Optional[Tuple[int, ...]] = None

    # First real input tensor (not weights)
    for input_tensor in model.graph.input:
        if input_tensor.name in initializer_names:
            continue

        shape: List[int] = []
        for dim in input_tensor.type.tensor_type.shape.dim:
            shape.append(dim.dim_value if dim.HasField("dim_value") else 1)
        input_shape = shape
        break

    # First output tensor
    for output_tensor in model.graph.output:
        shape: List[int] = []
        for dim in output_tensor.type.tensor_type.shape.dim:
            shape.append(dim.dim_value if dim.HasField("dim_value") else 1)
        output_shape = tuple(shape)
        break

    return input_shape, output_shape


# -----------------------------------------------------------------------------
# Weights / initializers
# -----------------------------------------------------------------------------
def quantize_weights(weights_list: List[np.ndarray], scale_factor: float) -> List[np.ndarray]:
    """Scales and quantizes a list of weights to int8."""
    out: List[np.ndarray] = []
    for w in weights_list:
        scaled = w * scale_factor
        out.append(np.round(scaled).astype(np.int8))
    return out


def get_model_weights(model: onnx.ModelProto, quantization: bool = False) -> List[np.ndarray]:
    """Extracts weights (initializers) from an ONNX model. Optionally quantizes them."""
    weight_list: List[np.ndarray] = []
    for initializer in model.graph.initializer:
        weight_list.append(numpy_helper.to_array(initializer))

    if quantization:
        weight_list = quantize_weights(weights_list=weight_list, scale_factor=127)

    return weight_list


def get_initializer_map(model: onnx.ModelProto) -> Dict[str, np.ndarray]:
    """Returns a name->numpy mapping for initializers (weights/biases/constants)."""
    out: Dict[str, np.ndarray] = {}
    for initializer in model.graph.initializer:
        out[initializer.name] = numpy_helper.to_array(initializer)
    return out


# -----------------------------------------------------------------------------
# Graph inspection helpers
# -----------------------------------------------------------------------------
def get_nodes_by_op_type(model: onnx.ModelProto, op_type: str) -> List[onnx.NodeProto]:
    return [n for n in model.graph.node if n.op_type == op_type]


# -----------------------------------------------------------------------------
# Legacy architecture extraction (kept for compatibility)
# -----------------------------------------------------------------------------
def get_model_arch(model: onnx.ModelProto) -> list:
    """
    Legacy architecture extraction:
    Returns list entries in your old format:
      - Layer: [True, layer_type, weight_name, bias_name, kernel_shape, strides]
      - Activation: [False, activation_name]
    Note: this relies on name prefixes (/fc,/co) and is fragile; kept for backward compat.
    """
    layer_list = []

    for node in model.graph.node:
        dictionary_layer = MessageToDict(node)
        not_a_layer = False

        out0 = dictionary_layer["output"][0]
        name0 = dictionary_layer.get("name", "")

        if not (
            out0[:3] == "/co" or out0[:3] == "/fc" or out0[:2] == "co" or out0[:2] == "fc" or
            name0[:3] == "/co" or name0[:3] == "/fc" or name0[:2] == "co" or name0[:2] == "fc"
        ):
            not_a_layer = True

        if not not_a_layer:
            layer_stride = []
            layer_kernel_shape = []
            layer_type = "dense"

            if out0[:3] == "/co" or out0[:2] == "co":
                layer_type = "conv"
                for attr in dictionary_layer.get("attribute", []):
                    if attr["name"] == "strides":
                        layer_stride = attr.get("ints", [])
                    if attr["name"] == "kernel_shape":
                        layer_kernel_shape = attr.get("ints", [])

            layer_bias = dictionary_layer["input"][2]
            if (
                dictionary_layer["input"][1][:3] == "/co" or dictionary_layer["input"][1][:3] == "/fc" or
                dictionary_layer["input"][1][:2] == "co" or dictionary_layer["input"][1][:2] == "fc"
            ):
                layer_weight = dictionary_layer["input"][1]
            else:
                layer_weight = layer_bias[:-4] + "weight"

            layer_info = [True, layer_type, layer_weight, layer_bias, layer_kernel_shape, layer_stride]
            layer_list.append(layer_info)

        else:
            inp0 = dictionary_layer.get("input", [""])[0]
            if inp0[:3] == "/co" or inp0[:3] == "/fc" or inp0[:2] == "co" or inp0[:2] == "fc":
                out0 = dictionary_layer["output"][0]
                if out0[:5] == "/Relu" or out0[:4] == "relu":
                    layer_list.append([False, "relu"])
                elif out0[:3] == "sig":
                    layer_list.append([False, "sigmoid"])
                else:
                    raise Exception("Our tool does not support this activation function!")

    return layer_list
