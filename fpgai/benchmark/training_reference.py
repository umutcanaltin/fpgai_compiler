from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from fpgai.engine.training_graph_utils import (
    as_chw as _as_chw,
    get_tensor_shape as _get_tensor_shape,
    infer_conv_output_shape as _infer_conv_output_shape,
    infer_pool_output_shape as _infer_pool_output_shape,
    resolve_batchnorm_arrays as _resolve_bn_arrays,
    resolve_conv_arrays as _resolve_conv_arrays,
    resolve_dense_arrays as _resolve_dense_arrays,
)


@dataclass
class TrainingReferenceResult:
    out_dir: Path
    grads_flat_path: Path
    weights_before_flat_path: Path
    weights_after_flat_path: Path
    summary_json: Path
    summary_txt: Path
    loss_before: float
    loss_after: float
    layerwise_dir: Path
    optimizer_type: str = "sgd"
    optimizer_bias_correction: bool = False
    optimizer_state_before_flat_path: Path | None = None
    optimizer_state_after_flat_path: Path | None = None
    loss_type: str = "mse"
    logits_ref_path: Path | None = None
    softmax_ref_path: Path | None = None
    cross_entropy_loss_ref_json: Path | None = None
    dlogits_ref_path: Path | None = None
    tiled_inputs_ref_path: Path | None = None
    tiled_labels_ref_path: Path | None = None
    tiled_outputs_ref_path: Path | None = None
    tiled_gradients_ref_path: Path | None = None
    tiled_weights_after_ref_path: Path | None = None


def _write_f32(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(array, dtype=np.float32).reshape(-1).tofile(path)


def _training_tiled_io_requested(raw_cfg: Dict[str, Any]) -> bool:
    movement = (raw_cfg.get("data_movement", {}) or {}) if isinstance(raw_cfg, dict) else {}
    for tensor, direction in (("inputs", "import"), ("labels", "import"), ("outputs", "export")):
        spec = ((movement.get(tensor, {}) or {}).get(direction, {}) or {})
        if str(spec.get("policy", "")).strip().lower() == "tiled" and str(spec.get("interface", "")).strip().lower() in {"m_axi", "axi_stream"}:
            return True
    return False


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    shifted = x - np.max(x)
    exponentials = np.exp(shifted)
    total = np.sum(exponentials)

    if total == 0:
        return np.zeros_like(x)

    return (exponentials / total).astype(np.float32)


def _dense_forward(
    x: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    return (weights @ x + bias).astype(np.float32)


def _dense_weight_grad(
    x: np.ndarray,
    output_gradient: np.ndarray,
) -> np.ndarray:
    return np.outer(output_gradient, x).astype(np.float32)


def _dense_bias_grad(output_gradient: np.ndarray) -> np.ndarray:
    return output_gradient.astype(np.float32)


def _dense_backward_input(
    output_gradient: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    return (weights.T @ output_gradient).astype(np.float32)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0).astype(np.float32)


def _relu_backward_from_output(
    output: np.ndarray,
    output_gradient: np.ndarray,
) -> np.ndarray:
    mask = (output > 0).astype(np.float32)
    return (output_gradient * mask).astype(np.float32)


def _leaky_relu(
    x: np.ndarray,
    alpha: float,
) -> np.ndarray:
    return np.where(x > 0, x, alpha * x).astype(np.float32)


def _leaky_relu_backward_from_input(
    x: np.ndarray,
    output_gradient: np.ndarray,
    alpha: float,
) -> np.ndarray:
    return np.where(
        x > 0,
        output_gradient,
        alpha * output_gradient,
    ).astype(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def _sigmoid_backward_from_output(
    output: np.ndarray,
    output_gradient: np.ndarray,
) -> np.ndarray:
    return (
        output_gradient
        * output
        * (1.0 - output)
    ).astype(np.float32)


def _softmax_backward(
    output: np.ndarray,
    output_gradient: np.ndarray,
) -> np.ndarray:
    input_gradient = np.zeros_like(
        output,
        dtype=np.float32,
    )

    for row in range(output.size):
        accumulator = 0.0

        for column in range(output.size):
            if row == column:
                jacobian = (
                    output[row]
                    * (1.0 - output[row])
                )
            else:
                jacobian = (
                    -output[row]
                    * output[column]
                )

            accumulator += (
                jacobian
                * output_gradient[column]
            )

        input_gradient[row] = accumulator

    return input_gradient.astype(np.float32)


def _reshape_copy(x: np.ndarray) -> np.ndarray:
    return x.copy().astype(np.float32)


def _add_vectors(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    return (left + right).astype(np.float32)


def _conv_forward_hwc(
    x: np.ndarray,
    input_shape: Tuple[int, int, int],
    weights: np.ndarray,
    bias: np.ndarray,
    stride: int,
    pad: int,
    output_shape: Tuple[int, int, int],
) -> np.ndarray:
    channels_in, height_in, width_in = input_shape
    channels_out, height_out, width_out = output_shape
    kernel_size = weights.shape[2]

    output = np.zeros(
        (height_out * width_out * channels_out,),
        dtype=np.float32,
    )

    for output_row in range(height_out):
        for output_column in range(width_out):
            for output_channel in range(channels_out):
                accumulator = float(bias[output_channel])

                for input_channel in range(channels_in):
                    for kernel_row in range(kernel_size):
                        for kernel_column in range(kernel_size):
                            input_row = (
                                output_row * stride
                                + kernel_row
                                - pad
                            )
                            input_column = (
                                output_column * stride
                                + kernel_column
                                - pad
                            )

                            if (
                                0 <= input_row < height_in
                                and 0 <= input_column < width_in
                            ):
                                input_index = (
                                    (
                                        input_row * width_in
                                        + input_column
                                    )
                                    * channels_in
                                    + input_channel
                                )

                                accumulator += (
                                    float(x[input_index])
                                    * float(
                                        weights[
                                            output_channel,
                                            input_channel,
                                            kernel_row,
                                            kernel_column,
                                        ]
                                    )
                                )

                output_index = (
                    (
                        output_row * width_out
                        + output_column
                    )
                    * channels_out
                    + output_channel
                )

                output[output_index] = accumulator

    return output


def _conv_backward_input_hwc(
    output_gradient: np.ndarray,
    weights: np.ndarray,
    input_shape: Tuple[int, int, int],
    output_shape: Tuple[int, int, int],
    stride: int,
    pad: int,
) -> np.ndarray:
    channels_in, height_in, width_in = input_shape
    channels_out, height_out, width_out = output_shape
    kernel_size = weights.shape[2]

    input_gradient = np.zeros(
        (height_in * width_in * channels_in,),
        dtype=np.float32,
    )

    for output_row in range(height_out):
        for output_column in range(width_out):
            for output_channel in range(channels_out):
                output_index = (
                    (
                        output_row * width_out
                        + output_column
                    )
                    * channels_out
                    + output_channel
                )

                gradient = float(
                    output_gradient[output_index]
                )

                for input_channel in range(channels_in):
                    for kernel_row in range(kernel_size):
                        for kernel_column in range(kernel_size):
                            input_row = (
                                output_row * stride
                                + kernel_row
                                - pad
                            )
                            input_column = (
                                output_column * stride
                                + kernel_column
                                - pad
                            )

                            if (
                                0 <= input_row < height_in
                                and 0 <= input_column < width_in
                            ):
                                input_index = (
                                    (
                                        input_row * width_in
                                        + input_column
                                    )
                                    * channels_in
                                    + input_channel
                                )

                                input_gradient[input_index] += (
                                    gradient
                                    * float(
                                        weights[
                                            output_channel,
                                            input_channel,
                                            kernel_row,
                                            kernel_column,
                                        ]
                                    )
                                )

    return input_gradient.astype(np.float32)


def _conv_weight_grad_hwc(
    x: np.ndarray,
    output_gradient: np.ndarray,
    input_shape: Tuple[int, int, int],
    output_shape: Tuple[int, int, int],
    weight_shape: Tuple[int, int, int, int],
    stride: int,
    pad: int,
) -> np.ndarray:
    channels_in, height_in, width_in = input_shape
    channels_out, height_out, width_out = output_shape
    _, _, kernel_size, _ = weight_shape

    weight_gradient = np.zeros(
        weight_shape,
        dtype=np.float32,
    )

    for output_channel in range(channels_out):
        for input_channel in range(channels_in):
            for kernel_row in range(kernel_size):
                for kernel_column in range(kernel_size):
                    accumulator = 0.0

                    for output_row in range(height_out):
                        for output_column in range(width_out):
                            input_row = (
                                output_row * stride
                                + kernel_row
                                - pad
                            )
                            input_column = (
                                output_column * stride
                                + kernel_column
                                - pad
                            )

                            if (
                                0 <= input_row < height_in
                                and 0 <= input_column < width_in
                            ):
                                input_index = (
                                    (
                                        input_row * width_in
                                        + input_column
                                    )
                                    * channels_in
                                    + input_channel
                                )

                                output_index = (
                                    (
                                        output_row * width_out
                                        + output_column
                                    )
                                    * channels_out
                                    + output_channel
                                )

                                accumulator += (
                                    float(x[input_index])
                                    * float(
                                        output_gradient[
                                            output_index
                                        ]
                                    )
                                )

                    weight_gradient[
                        output_channel,
                        input_channel,
                        kernel_row,
                        kernel_column,
                    ] = accumulator

    return weight_gradient.reshape(-1).astype(np.float32)


def _conv_bias_grad_hwc(
    output_gradient: np.ndarray,
    output_shape: Tuple[int, int, int],
) -> np.ndarray:
    channels_out, height_out, width_out = output_shape

    bias_gradient = np.zeros(
        (channels_out,),
        dtype=np.float32,
    )

    for output_row in range(height_out):
        for output_column in range(width_out):
            for output_channel in range(channels_out):
                output_index = (
                    (
                        output_row * width_out
                        + output_column
                    )
                    * channels_out
                    + output_channel
                )

                bias_gradient[output_channel] += (
                    output_gradient[output_index]
                )

    return bias_gradient.astype(np.float32)


def _maxpool_forward_hwc(
    x: np.ndarray,
    input_shape: Tuple[int, int, int],
    kernel_size: int,
    stride: int,
    output_shape: Tuple[int, int, int],
) -> np.ndarray:
    channels_in, height_in, width_in = input_shape
    channels_out, height_out, width_out = output_shape

    output = np.zeros(
        (height_out * width_out * channels_out,),
        dtype=np.float32,
    )

    for output_row in range(height_out):
        for output_column in range(width_out):
            for channel in range(channels_in):
                first_index = (
                    (
                        output_row * stride * width_in
                        + output_column * stride
                    )
                    * channels_in
                    + channel
                )

                best = x[first_index]

                for kernel_row in range(kernel_size):
                    for kernel_column in range(kernel_size):
                        input_row = (
                            output_row * stride
                            + kernel_row
                        )
                        input_column = (
                            output_column * stride
                            + kernel_column
                        )

                        input_index = (
                            (
                                input_row * width_in
                                + input_column
                            )
                            * channels_in
                            + channel
                        )

                        if x[input_index] > best:
                            best = x[input_index]

                output_index = (
                    (
                        output_row * width_out
                        + output_column
                    )
                    * channels_out
                    + channel
                )

                output[output_index] = best

    return output.astype(np.float32)


def _avgpool_forward_hwc(
    x: np.ndarray,
    input_shape: Tuple[int, int, int],
    kernel_size: int,
    stride: int,
    output_shape: Tuple[int, int, int],
) -> np.ndarray:
    channels_in, height_in, width_in = input_shape
    channels_out, height_out, width_out = output_shape

    output = np.zeros(
        (height_out * width_out * channels_out,),
        dtype=np.float32,
    )

    for output_row in range(height_out):
        for output_column in range(width_out):
            for channel in range(channels_in):
                accumulator = 0.0

                for kernel_row in range(kernel_size):
                    for kernel_column in range(kernel_size):
                        input_row = (
                            output_row * stride
                            + kernel_row
                        )
                        input_column = (
                            output_column * stride
                            + kernel_column
                        )

                        input_index = (
                            (
                                input_row * width_in
                                + input_column
                            )
                            * channels_in
                            + channel
                        )

                        accumulator += float(x[input_index])

                output_index = (
                    (
                        output_row * width_out
                        + output_column
                    )
                    * channels_out
                    + channel
                )

                output[output_index] = (
                    accumulator
                    / float(kernel_size * kernel_size)
                )

    return output.astype(np.float32)


def _maxpool_backward_hwc(
    x: np.ndarray,
    output: np.ndarray,
    output_gradient: np.ndarray,
    input_shape: Tuple[int, int, int],
    kernel_size: int,
    stride: int,
    output_shape: Tuple[int, int, int],
) -> np.ndarray:
    channels_in, height_in, width_in = input_shape
    channels_out, height_out, width_out = output_shape

    input_gradient = np.zeros(
        (height_in * width_in * channels_in,),
        dtype=np.float32,
    )

    for output_row in range(height_out):
        for output_column in range(width_out):
            for channel in range(channels_in):
                output_index = (
                    (
                        output_row * width_out
                        + output_column
                    )
                    * channels_out
                    + channel
                )

                pooled_value = output[output_index]
                routed = False

                for kernel_row in range(kernel_size):
                    for kernel_column in range(kernel_size):
                        input_row = (
                            output_row * stride
                            + kernel_row
                        )
                        input_column = (
                            output_column * stride
                            + kernel_column
                        )

                        input_index = (
                            (
                                input_row * width_in
                                + input_column
                            )
                            * channels_in
                            + channel
                        )

                        if (
                            not routed
                            and x[input_index] == pooled_value
                        ):
                            input_gradient[input_index] += (
                                output_gradient[output_index]
                            )
                            routed = True

    return input_gradient.astype(np.float32)


def _avgpool_backward_hwc(
    output_gradient: np.ndarray,
    input_shape: Tuple[int, int, int],
    kernel_size: int,
    stride: int,
    output_shape: Tuple[int, int, int],
) -> np.ndarray:
    channels_in, height_in, width_in = input_shape
    channels_out, height_out, width_out = output_shape

    input_gradient = np.zeros(
        (height_in * width_in * channels_in,),
        dtype=np.float32,
    )

    scale = 1.0 / float(kernel_size * kernel_size)

    for output_row in range(height_out):
        for output_column in range(width_out):
            for channel in range(channels_in):
                output_index = (
                    (
                        output_row * width_out
                        + output_column
                    )
                    * channels_out
                    + channel
                )

                gradient = (
                    float(output_gradient[output_index])
                    * scale
                )

                for kernel_row in range(kernel_size):
                    for kernel_column in range(kernel_size):
                        input_row = (
                            output_row * stride
                            + kernel_row
                        )
                        input_column = (
                            output_column * stride
                            + kernel_column
                        )

                        input_index = (
                            (
                                input_row * width_in
                                + input_column
                            )
                            * channels_in
                            + channel
                        )

                        input_gradient[input_index] += gradient

    return input_gradient.astype(np.float32)


def _batchnorm_forward_hwc(
    x: np.ndarray,
    shape: Tuple[int, int, int],
    gamma: np.ndarray,
    beta: np.ndarray,
    epsilon: float = 1e-5,
):
    channels, height, width = shape
    spatial_size = height * width

    mean = np.zeros((channels,), dtype=np.float32)
    variance = np.zeros((channels,), dtype=np.float32)
    normalized = np.zeros_like(x, dtype=np.float32)
    output = np.zeros_like(x, dtype=np.float32)

    for channel in range(channels):
        values = np.array(
            [
                x[spatial_index * channels + channel]
                for spatial_index in range(spatial_size)
            ],
            dtype=np.float32,
        )

        channel_mean = float(np.mean(values))
        channel_variance = float(
            np.mean((values - channel_mean) ** 2)
        )

        mean[channel] = channel_mean
        variance[channel] = channel_variance

        inverse_standard_deviation = (
            1.0
            / np.sqrt(channel_variance + epsilon)
        )

        for spatial_index in range(spatial_size):
            index = spatial_index * channels + channel

            normalized_value = (
                (x[index] - channel_mean)
                * inverse_standard_deviation
            )

            normalized[index] = normalized_value
            output[index] = (
                gamma[channel] * normalized_value
                + beta[channel]
            )

    cache = {
        "mean": mean,
        "var": variance,
        "xhat": normalized,
        "eps": float(epsilon),
    }

    return output.astype(np.float32), cache


def _batchnorm_param_grad_hwc(
    output_gradient: np.ndarray,
    normalized: np.ndarray,
    shape: Tuple[int, int, int],
):
    channels, height, width = shape
    spatial_size = height * width

    gamma_gradient = np.zeros(
        (channels,),
        dtype=np.float32,
    )
    beta_gradient = np.zeros(
        (channels,),
        dtype=np.float32,
    )

    for channel in range(channels):
        gamma_accumulator = 0.0
        beta_accumulator = 0.0

        for spatial_index in range(spatial_size):
            index = spatial_index * channels + channel

            gamma_accumulator += (
                float(output_gradient[index])
                * float(normalized[index])
            )

            beta_accumulator += float(
                output_gradient[index]
            )

        gamma_gradient[channel] = gamma_accumulator
        beta_gradient[channel] = beta_accumulator

    return gamma_gradient, beta_gradient


def _batchnorm_backward_input_hwc(
    output_gradient: np.ndarray,
    normalized: np.ndarray,
    variance: np.ndarray,
    shape: Tuple[int, int, int],
    gamma: np.ndarray,
    epsilon: float = 1e-5,
) -> np.ndarray:
    channels, height, width = shape
    spatial_size = height * width

    input_gradient = np.zeros_like(
        output_gradient,
        dtype=np.float32,
    )

    for channel in range(channels):
        inverse_standard_deviation = (
            1.0
            / float(
                np.sqrt(
                    float(variance[channel])
                    + float(epsilon)
                )
            )
        )

        gradient_sum = 0.0
        gradient_normalized_sum = 0.0

        for spatial_index in range(spatial_size):
            index = spatial_index * channels + channel
            gradient = float(output_gradient[index])
            normalized_value = float(normalized[index])

            gradient_sum += gradient
            gradient_normalized_sum += (
                gradient * normalized_value
            )

        mean_gradient = (
            gradient_sum / float(spatial_size)
        )
        mean_gradient_normalized = (
            gradient_normalized_sum
            / float(spatial_size)
        )

        scale = (
            float(gamma[channel])
            * inverse_standard_deviation
        )

        for spatial_index in range(spatial_size):
            index = spatial_index * channels + channel
            normalized_value = float(normalized[index])

            input_gradient[index] = scale * (
                float(output_gradient[index])
                - mean_gradient
                - normalized_value
                * mean_gradient_normalized
            )

    return input_gradient.astype(np.float32)


def _forward_pass(
    graph,
    parameter_state: Dict[str, Dict[str, np.ndarray]],
    input_array: np.ndarray,
    layerwise_dir: Path,
):
    values: Dict[str, np.ndarray] = {}
    caches: Dict[str, Dict[str, Any]] = {}
    inferred_shapes: Dict[str, Tuple[int, ...]] = {}

    input_name = graph.inputs[0]
    values[input_name] = input_array.astype(np.float32)

    input_shape = _get_tensor_shape(graph, input_name)
    if input_shape:
        inferred_shapes[input_name] = input_shape

    def get_shape(name: str) -> Tuple[int, ...]:
        if (
            name in inferred_shapes
            and inferred_shapes[name]
        ):
            return inferred_shapes[name]

        shape = _get_tensor_shape(graph, name)

        if shape:
            inferred_shapes[name] = shape
            return shape

        return tuple()

    for op in graph.ops:
        input_name = op.inputs[0]
        output_name = op.outputs[0]

        input_value = values[input_name]
        output_value = None
        cache: Dict[str, Any] = {}

        if op.op_type == "Dense":
            parameters = parameter_state[op.name]

            output_value = _dense_forward(
                input_value,
                parameters["W"],
                parameters["B"],
            )

            cache["x"] = input_value

            inferred_shapes[output_name] = (
                int(parameters["B"].size),
            )

        elif op.op_type == "Conv":
            parameters = parameter_state[op.name]
            weights = parameters["W"]
            bias = parameters["B"]

            input_shape = get_shape(input_name)

            if not input_shape:
                raise RuntimeError(
                    f"Conv input shape unavailable for "
                    f"op '{op.name}' input '{input_name}'"
                )

            stride = int(
                op.attrs.get("strides", [1, 1])[0]
            )
            pad = int(
                op.attrs.get(
                    "pads",
                    [0, 0, 0, 0],
                )[0]
            )

            output_shape = get_shape(output_name)

            if not output_shape:
                output_shape = _infer_conv_output_shape(
                    _as_chw(input_shape),
                    tuple(
                        int(value)
                        for value in weights.shape
                    ),
                    stride,
                    pad,
                )

            output_value = _conv_forward_hwc(
                input_value,
                _as_chw(input_shape),
                weights,
                bias,
                stride,
                pad,
                _as_chw(output_shape),
            )

            cache["x"] = input_value
            cache["xshape"] = _as_chw(input_shape)
            cache["yshape"] = _as_chw(output_shape)
            cache["stride"] = stride
            cache["pad"] = pad
            cache["Wshape"] = weights.shape

            inferred_shapes[output_name] = tuple(
                int(value)
                for value in output_shape
            )

        elif op.op_type == "Relu":
            output_value = _relu(input_value)

            input_shape = get_shape(input_name)
            if input_shape:
                inferred_shapes[output_name] = input_shape

        elif op.op_type == "LeakyRelu":
            alpha = float(
                (
                    getattr(op, "attrs", {})
                    or {}
                ).get("alpha", 0.01)
            )

            output_value = _leaky_relu(
                input_value,
                alpha,
            )

            cache["alpha"] = alpha
            cache["x"] = input_value

            input_shape = get_shape(input_name)
            if input_shape:
                inferred_shapes[output_name] = input_shape

        elif op.op_type == "Sigmoid":
            output_value = _sigmoid(input_value)

            input_shape = get_shape(input_name)
            if input_shape:
                inferred_shapes[output_name] = input_shape

        elif op.op_type == "Softmax":
            output_value = _softmax(input_value)

            input_shape = get_shape(input_name)
            if input_shape:
                inferred_shapes[output_name] = input_shape

        elif op.op_type in ("Flatten", "Reshape"):
            output_value = _reshape_copy(input_value)
            inferred_shapes[output_name] = (
                int(output_value.size),
            )

        elif op.op_type == "Add":
            right_value = values[op.inputs[1]]

            output_value = _add_vectors(
                input_value,
                right_value,
            )

            input_shape = get_shape(input_name)

            if input_shape:
                inferred_shapes[output_name] = input_shape
            else:
                inferred_shapes[output_name] = (
                    int(output_value.size),
                )

        elif op.op_type == "MaxPool":
            input_shape = get_shape(input_name)

            if not input_shape:
                raise RuntimeError(
                    f"MaxPool input shape unavailable for "
                    f"op '{op.name}' input '{input_name}'"
                )

            kernel_size = int(
                op.attrs.get(
                    "kernel_shape",
                    [2, 2],
                )[0]
            )

            stride = int(
                op.attrs.get(
                    "strides",
                    [2, 2],
                )[0]
            )

            output_shape = get_shape(output_name)

            if not output_shape:
                output_shape = _infer_pool_output_shape(
                    _as_chw(input_shape),
                    kernel_size,
                    stride,
                )

            output_value = _maxpool_forward_hwc(
                input_value,
                _as_chw(input_shape),
                kernel_size,
                stride,
                _as_chw(output_shape),
            )

            cache["x"] = input_value
            cache["xshape"] = _as_chw(input_shape)
            cache["yshape"] = _as_chw(output_shape)
            cache["k"] = kernel_size
            cache["stride"] = stride

            inferred_shapes[output_name] = tuple(
                int(value)
                for value in output_shape
            )

        elif op.op_type == "AvgPool":
            input_shape = get_shape(input_name)

            if not input_shape:
                raise RuntimeError(
                    f"AvgPool input shape unavailable for "
                    f"op '{op.name}' input '{input_name}'"
                )

            kernel_size = int(
                op.attrs.get(
                    "kernel_shape",
                    [2, 2],
                )[0]
            )

            stride = int(
                op.attrs.get(
                    "strides",
                    [2, 2],
                )[0]
            )

            output_shape = get_shape(output_name)

            if not output_shape:
                output_shape = _infer_pool_output_shape(
                    _as_chw(input_shape),
                    kernel_size,
                    stride,
                )

            output_value = _avgpool_forward_hwc(
                input_value,
                _as_chw(input_shape),
                kernel_size,
                stride,
                _as_chw(output_shape),
            )

            cache["xshape"] = _as_chw(input_shape)
            cache["yshape"] = _as_chw(output_shape)
            cache["k"] = kernel_size
            cache["stride"] = stride

            inferred_shapes[output_name] = tuple(
                int(value)
                for value in output_shape
            )

        elif op.op_type == "BatchNormalization":
            parameters = parameter_state[op.name]
            gamma = parameters["gamma"]
            beta = parameters["beta"]

            shape = get_shape(output_name)
            if not shape:
                shape = get_shape(input_name)

            if not shape:
                raise RuntimeError(
                    "BatchNormalization shape unavailable "
                    f"for op '{op.name}'"
                )

            (
                output_value,
                batchnorm_cache,
            ) = _batchnorm_forward_hwc(
                input_value,
                _as_chw(shape),
                gamma,
                beta,
            )

            cache["shape"] = _as_chw(shape)
            cache.update(batchnorm_cache)

            inferred_shapes[output_name] = tuple(
                int(value)
                for value in shape
            )

        else:
            raise RuntimeError(
                "Unsupported training reference op: "
                f"{op.op_type}"
            )

        values[output_name] = output_value.astype(
            np.float32
        )
        caches[op.name] = cache

        _write_f32(
            layerwise_dir
            / f"{op.name}__fwd.bin",
            output_value,
        )

    return values, caches


def _mse_loss_and_grad(
    prediction: np.ndarray,
    target: np.ndarray,
):
    difference = (
        prediction.astype(np.float32)
        - target.astype(np.float32)
    )

    loss = float(
        np.mean(difference * difference)
    )

    gradient = (
        2.0
        / float(prediction.size)
    ) * difference

    return loss, gradient.astype(np.float32)




def _cross_entropy_loss_and_grad(
    logits: np.ndarray,
    target: np.ndarray,
):
    logits = logits.astype(np.float32).reshape(-1)
    target = target.astype(np.float32).reshape(-1)
    if target.size != logits.size:
        raise RuntimeError(
            "cross_entropy target size must match logits size: "
            f"target={target.size}, logits={logits.size}"
        )
    shifted = logits - np.max(logits)
    exp_values = np.exp(shifted).astype(np.float32)
    denom = float(np.sum(exp_values))
    if denom <= 0.0:
        probabilities = np.zeros_like(logits, dtype=np.float32)
    else:
        probabilities = (exp_values / denom).astype(np.float32)
    loss = float(-np.sum(target * np.log(probabilities + np.float32(1.0e-7))))
    gradient = (probabilities - target).astype(np.float32)
    return loss, gradient, probabilities

def _is_final_softmax_mse_case(
    graph,
    raw_config: Dict[str, Any],
) -> bool:
    training_config = (
        raw_config.get("training", {})
        or {}
    )
    loss_config = (
        training_config.get("loss", {})
        or {}
    )

    loss_type = str(
        loss_config.get("type", "mse")
    ).lower()

    if loss_type != "mse":
        return False

    if not getattr(graph, "ops", None):
        return False

    last_op = graph.ops[-1]

    return bool(
        last_op.op_type == "Softmax"
        and last_op.outputs
        and graph.outputs
        and last_op.outputs[0] == graph.outputs[0]
    )


def run_training_reference_step(
    *,
    graph,
    raw_cfg: Dict[str, Any],
    out_dir: Path,
    x_input: np.ndarray,
    target: np.ndarray,
) -> TrainingReferenceResult:
    reference_dir = (
        Path(out_dir)
        / "training_reference"
    )
    reference_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    layerwise_dir = reference_dir / "layerwise"
    layerwise_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_config = (
        raw_cfg.get("training", {})
        or {}
    )
    optimizer_config = (
        training_config.get("optimizer", {})
        or {}
    )
    loss_config = (
        training_config.get("loss", {})
        or {}
    )
    loss_type = str(loss_config.get("type", "mse")).strip().lower().replace("-", "_")
    tiled_io_requested = _training_tiled_io_requested(raw_cfg)

    learning_rate = float(
        optimizer_config.get(
            "learning_rate",
            0.01,
        )
    )
    optimizer_type = str(optimizer_config.get("type", "sgd")).lower().replace("-", "_")
    momentum = float(optimizer_config.get("momentum", 0.9))
    beta1 = float(optimizer_config.get("beta1", 0.9))
    beta2 = float(optimizer_config.get("beta2", 0.999))
    epsilon = float(optimizer_config.get("epsilon", 1.0e-8))
    bias_correction = bool(optimizer_config.get("bias_correction", False))

    tiled_inputs_ref_path = reference_dir / "tiled_inputs_ref.bin" if tiled_io_requested else None
    tiled_labels_ref_path = reference_dir / "tiled_labels_ref.bin" if tiled_io_requested else None
    tiled_outputs_ref_path = reference_dir / "tiled_outputs_ref.bin" if tiled_io_requested else None
    tiled_gradients_ref_path = reference_dir / "tiled_gradients_ref.bin" if tiled_io_requested else None
    tiled_weights_after_ref_path = reference_dir / "tiled_weights_after_ref.bin" if tiled_io_requested else None
    if tiled_io_requested:
        _write_f32(tiled_inputs_ref_path, x_input.astype(np.float32))
        _write_f32(tiled_labels_ref_path, target.astype(np.float32))

    bypass_final_softmax_backward = (
        _is_final_softmax_mse_case(
            graph,
            raw_cfg,
        )
    )

    parameter_state: Dict[
        str,
        Dict[str, np.ndarray],
    ] = {}

    trainable_order: List[
        Tuple[str, str]
    ] = []

    for op in graph.ops:
        if op.op_type == "Dense":
            (
                weights,
                bias,
                _,
                _,
            ) = _resolve_dense_arrays(
                graph,
                op,
            )

            parameter_state[op.name] = {
                "W": weights.copy(),
                "B": bias.copy(),
            }

            trainable_order.append(
                ("dense", op.name)
            )

        elif op.op_type == "Conv":
            (
                weights,
                bias,
                _,
            ) = _resolve_conv_arrays(
                graph,
                op,
            )

            parameter_state[op.name] = {
                "W": weights.copy(),
                "B": bias.reshape(-1).copy(),
            }

            trainable_order.append(
                ("conv", op.name)
            )

        elif op.op_type == "BatchNormalization":
            shape = _get_tensor_shape(
                graph,
                op.outputs[0],
            )

            if not shape:
                shape = _get_tensor_shape(
                    graph,
                    op.inputs[0],
                )

            if not shape:
                raise RuntimeError(
                    "BatchNormalization shape unavailable "
                    f"for op '{op.name}'"
                )

            channels, _, _ = _as_chw(shape)

            gamma, beta, _, _ = (
                _resolve_bn_arrays(
                    graph,
                    op,
                    channels,
                )
            )

            parameter_state[op.name] = {
                "gamma": gamma.copy(),
                "beta": beta.copy(),
            }

            trainable_order.append(
                ("bn", op.name)
            )

    weights_before_chunks: List[np.ndarray] = []

    for kind, name in trainable_order:
        parameters = parameter_state[name]

        if kind in ("dense", "conv"):
            first = parameters["W"].reshape(-1)
            second = parameters["B"].reshape(-1)
        else:
            first = parameters["gamma"].reshape(-1)
            second = parameters["beta"].reshape(-1)

        weights_before_chunks.extend(
            [first.copy(), second.copy()]
        )

        _write_f32(
            layerwise_dir
            / f"{name}__weights_before.bin",
            np.concatenate(
                [first, second]
            ).astype(np.float32),
        )

    values, caches = _forward_pass(
        graph,
        parameter_state,
        x_input.astype(np.float32),
        layerwise_dir,
    )

    prediction = values[graph.outputs[0]]
    if tiled_io_requested and tiled_outputs_ref_path is not None:
        _write_f32(tiled_outputs_ref_path, prediction.astype(np.float32))

    logits_ref_path = None
    softmax_ref_path = None
    cross_entropy_loss_ref_json = None
    dlogits_ref_path = None
    if loss_type in {"cross_entropy", "ce"}:
        loss_before, output_gradient, softmax_ref = _cross_entropy_loss_and_grad(
            prediction,
            target.astype(np.float32),
        )
        logits_ref_path = reference_dir / "logits_ref.bin"
        softmax_ref_path = reference_dir / "softmax_ref.bin"
        cross_entropy_loss_ref_json = reference_dir / "cross_entropy_loss_ref.json"
        dlogits_ref_path = reference_dir / "dlogits_ref.bin"
        _write_f32(logits_ref_path, prediction)
        _write_f32(softmax_ref_path, softmax_ref)
        _write_f32(dlogits_ref_path, output_gradient)
        cross_entropy_loss_ref_json.write_text(
            json.dumps({
                "loss_type": "cross_entropy",
                "loss": loss_before,
                "softmax_stable": True,
                "num_logits": int(np.asarray(prediction).size),
                "logits_ref_bin": str(logits_ref_path),
                "softmax_ref_bin": str(softmax_ref_path),
                "dlogits_ref_bin": str(dlogits_ref_path),
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        loss_before, output_gradient = (
            _mse_loss_and_grad(
                prediction,
                target.astype(np.float32),
            )
        )

    gradients_by_tensor: Dict[
        str,
        np.ndarray,
    ] = {
        graph.outputs[0]: output_gradient
    }

    parameter_gradients: Dict[
        str,
        Tuple[np.ndarray, np.ndarray],
    ] = {}

    for op_index in range(
        len(graph.ops) - 1,
        -1,
        -1,
    ):
        op = graph.ops[op_index]

        input_name = op.inputs[0]
        output_name = op.outputs[0]

        output_value = values[output_name]
        output_gradient = gradients_by_tensor[
            output_name
        ]

        if op.op_type == "Dense":
            parameters = parameter_state[op.name]

            weight_gradient = _dense_weight_grad(
                caches[op.name]["x"],
                output_gradient,
            )

            bias_gradient = _dense_bias_grad(
                output_gradient
            )

            input_gradient = _dense_backward_input(
                output_gradient,
                parameters["W"],
            )

            parameter_gradients[op.name] = (
                weight_gradient.reshape(-1),
                bias_gradient.reshape(-1),
            )

            gradients_by_tensor[input_name] = (
                input_gradient
            )

            _write_f32(
                layerwise_dir
                / f"{op.name}__param_grad_w.bin",
                weight_gradient,
            )
            _write_f32(
                layerwise_dir
                / f"{op.name}__param_grad_b.bin",
                bias_gradient,
            )

        elif op.op_type == "Conv":
            parameters = parameter_state[op.name]
            cache = caches[op.name]

            weight_gradient = _conv_weight_grad_hwc(
                cache["x"],
                output_gradient,
                cache["xshape"],
                cache["yshape"],
                cache["Wshape"],
                cache["stride"],
                cache["pad"],
            )

            bias_gradient = _conv_bias_grad_hwc(
                output_gradient,
                cache["yshape"],
            )

            input_gradient = _conv_backward_input_hwc(
                output_gradient,
                parameters["W"],
                cache["xshape"],
                cache["yshape"],
                cache["stride"],
                cache["pad"],
            )

            parameter_gradients[op.name] = (
                weight_gradient.reshape(-1),
                bias_gradient.reshape(-1),
            )

            gradients_by_tensor[input_name] = (
                input_gradient
            )

            _write_f32(
                layerwise_dir
                / f"{op.name}__param_grad_w.bin",
                weight_gradient,
            )
            _write_f32(
                layerwise_dir
                / f"{op.name}__param_grad_b.bin",
                bias_gradient,
            )

        elif op.op_type == "Relu":
            gradients_by_tensor[input_name] = (
                _relu_backward_from_output(
                    output_value,
                    output_gradient,
                )
            )

        elif op.op_type == "LeakyRelu":
            gradients_by_tensor[input_name] = (
                _leaky_relu_backward_from_input(
                    caches[op.name]["x"],
                    output_gradient,
                    caches[op.name]["alpha"],
                )
            )

        elif op.op_type == "Sigmoid":
            gradients_by_tensor[input_name] = (
                _sigmoid_backward_from_output(
                    output_value,
                    output_gradient,
                )
            )

        elif op.op_type == "Softmax":
            is_final_op = bool(
                op_index == len(graph.ops) - 1
                and output_name == graph.outputs[0]
            )

            if (
                bypass_final_softmax_backward
                and is_final_op
            ):
                gradients_by_tensor[input_name] = (
                    output_gradient.copy().astype(
                        np.float32
                    )
                )
            else:
                gradients_by_tensor[input_name] = (
                    _softmax_backward(
                        output_value,
                        output_gradient,
                    )
                )

        elif op.op_type in ("Flatten", "Reshape"):
            gradients_by_tensor[input_name] = (
                output_gradient.copy().astype(
                    np.float32
                )
            )

        elif op.op_type == "Add":
            gradients_by_tensor[input_name] = (
                output_gradient.copy().astype(
                    np.float32
                )
            )

            right_name = op.inputs[1]
            existing = gradients_by_tensor.get(
                right_name,
                np.zeros_like(
                    output_gradient,
                    dtype=np.float32,
                ),
            )

            gradients_by_tensor[right_name] = (
                existing
                + output_gradient.astype(np.float32)
            )

        elif op.op_type == "MaxPool":
            cache = caches[op.name]

            gradients_by_tensor[input_name] = (
                _maxpool_backward_hwc(
                    cache["x"],
                    output_value,
                    output_gradient,
                    cache["xshape"],
                    cache["k"],
                    cache["stride"],
                    cache["yshape"],
                )
            )

        elif op.op_type == "AvgPool":
            cache = caches[op.name]

            gradients_by_tensor[input_name] = (
                _avgpool_backward_hwc(
                    output_gradient,
                    cache["xshape"],
                    cache["k"],
                    cache["stride"],
                    cache["yshape"],
                )
            )

        elif op.op_type == "BatchNormalization":
            parameters = parameter_state[op.name]
            cache = caches[op.name]

            (
                gamma_gradient,
                beta_gradient,
            ) = _batchnorm_param_grad_hwc(
                output_gradient,
                cache["xhat"],
                cache["shape"],
            )

            input_gradient = (
                _batchnorm_backward_input_hwc(
                    output_gradient,
                    cache["xhat"],
                    cache["var"],
                    cache["shape"],
                    parameters["gamma"],
                    epsilon=float(
                        cache.get("eps", 1e-5)
                    ),
                )
            )

            parameter_gradients[op.name] = (
                gamma_gradient.reshape(-1),
                beta_gradient.reshape(-1),
            )

            gradients_by_tensor[input_name] = (
                input_gradient
            )

            _write_f32(
                layerwise_dir
                / f"{op.name}__param_grad_gamma.bin",
                gamma_gradient,
            )
            _write_f32(
                layerwise_dir
                / f"{op.name}__param_grad_beta.bin",
                beta_gradient,
            )

        else:
            raise RuntimeError(
                "Unsupported training reference op: "
                f"{op.op_type}"
            )

        _write_f32(
            layerwise_dir
            / f"{op.name}__bwd_in.bin",
            gradients_by_tensor[input_name],
        )

    gradient_chunks: List[np.ndarray] = []

    for _, name in trainable_order:
        first_gradient, second_gradient = (
            parameter_gradients[name]
        )

        gradient_chunks.extend(
            [
                first_gradient.reshape(-1),
                second_gradient.reshape(-1),
            ]
        )

    if gradient_chunks:
        gradients_flat = np.concatenate(
            gradient_chunks
        ).astype(np.float32)
    else:
        gradients_flat = np.zeros(
            (0,),
            dtype=np.float32,
        )

    if weights_before_chunks:
        weights_before_flat = np.concatenate(
            weights_before_chunks
        ).astype(np.float32)
    else:
        weights_before_flat = np.zeros(
            (0,),
            dtype=np.float32,
        )

    optimizer_state_before_chunks: List[np.ndarray] = []
    optimizer_state_after_chunks: List[np.ndarray] = []

    def _apply_optimizer_update(value: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        value_flat = value.reshape(-1).astype(np.float32)
        grad_flat = gradient.reshape(-1).astype(np.float32)
        if optimizer_type == "momentum":
            velocity_before = np.zeros_like(grad_flat, dtype=np.float32)
            velocity_after = (momentum * velocity_before - learning_rate * grad_flat).astype(np.float32)
            optimizer_state_before_chunks.append(velocity_before)
            optimizer_state_after_chunks.append(velocity_after)
            return (value_flat + velocity_after).reshape(value.shape).astype(np.float32)
        if optimizer_type == "adam":
            first_before = np.zeros_like(grad_flat, dtype=np.float32)
            second_before = np.zeros_like(grad_flat, dtype=np.float32)
            first_after = (beta1 * first_before + (1.0 - beta1) * grad_flat).astype(np.float32)
            second_after = (beta2 * second_before + (1.0 - beta2) * grad_flat * grad_flat).astype(np.float32)
            optimizer_state_before_chunks.extend([first_before, second_before])
            optimizer_state_after_chunks.extend([first_after, second_after])
            if bias_correction:
                first_used = (first_after / max(1.0e-12, 1.0 - beta1)).astype(np.float32)
                second_used = (second_after / max(1.0e-12, 1.0 - beta2)).astype(np.float32)
            else:
                first_used = first_after
                second_used = second_after
            updated = value_flat - learning_rate * first_used / (np.sqrt(second_used + epsilon).astype(np.float32))
            return updated.reshape(value.shape).astype(np.float32)
        return (value_flat - learning_rate * grad_flat).reshape(value.shape).astype(np.float32)

    for kind, name in trainable_order:
        parameters = parameter_state[name]
        first_gradient, second_gradient = (
            parameter_gradients[name]
        )

        if kind in ("dense", "conv"):
            parameters["W"] = _apply_optimizer_update(parameters["W"], first_gradient)
            parameters["B"] = _apply_optimizer_update(parameters["B"], second_gradient)

        else:
            parameters["gamma"] = _apply_optimizer_update(parameters["gamma"], first_gradient)
            parameters["beta"] = _apply_optimizer_update(parameters["beta"], second_gradient)

    if optimizer_state_before_chunks:
        optimizer_state_before_flat = np.concatenate(optimizer_state_before_chunks).astype(np.float32)
        optimizer_state_after_flat = np.concatenate(optimizer_state_after_chunks).astype(np.float32)
    else:
        optimizer_state_before_flat = np.zeros((0,), dtype=np.float32)
        optimizer_state_after_flat = np.zeros((0,), dtype=np.float32)

    weights_after_chunks: List[np.ndarray] = []

    for kind, name in trainable_order:
        parameters = parameter_state[name]

        if kind in ("dense", "conv"):
            first = parameters["W"].reshape(-1)
            second = parameters["B"].reshape(-1)
        else:
            first = parameters["gamma"].reshape(-1)
            second = parameters["beta"].reshape(-1)

        weights_after_chunks.extend(
            [first.copy(), second.copy()]
        )

        _write_f32(
            layerwise_dir
            / f"{name}__weights_after.bin",
            np.concatenate(
                [first, second]
            ).astype(np.float32),
        )

    if weights_after_chunks:
        weights_after_flat = np.concatenate(
            weights_after_chunks
        ).astype(np.float32)
    else:
        weights_after_flat = np.zeros(
            (0,),
            dtype=np.float32,
        )

    values_after, _ = _forward_pass(
        graph,
        parameter_state,
        x_input.astype(np.float32),
        layerwise_dir / "after_step",
    )

    prediction_after = values_after[
        graph.outputs[0]
    ]

    if loss_type in {"cross_entropy", "ce"}:
        loss_after, _, _ = _cross_entropy_loss_and_grad(
            prediction_after,
            target.astype(np.float32),
        )
    else:
        loss_after, _ = _mse_loss_and_grad(
            prediction_after,
            target.astype(np.float32),
        )

    gradients_path = (
        reference_dir / "grads_ref.bin"
    )
    weights_before_path = (
        reference_dir
        / "weights_before_ref.bin"
    )
    weights_after_path = (
        reference_dir
        / "weights_after_ref.bin"
    )
    optimizer_state_before_path = (
        reference_dir
        / "optimizer_state_before_ref.bin"
    )
    optimizer_state_after_path = (
        reference_dir
        / "optimizer_state_after_ref.bin"
    )
    summary_json = reference_dir / "summary.json"
    summary_txt = reference_dir / "summary.txt"

    _write_f32(
        gradients_path,
        gradients_flat,
    )
    _write_f32(
        weights_before_path,
        weights_before_flat,
    )
    _write_f32(
        weights_after_path,
        weights_after_flat,
    )
    if tiled_io_requested:
        if tiled_gradients_ref_path is not None:
            _write_f32(tiled_gradients_ref_path, gradients_flat)
        if tiled_weights_after_ref_path is not None:
            _write_f32(tiled_weights_after_ref_path, weights_after_flat)
    if optimizer_type in {"momentum", "adam"}:
        _write_f32(
            optimizer_state_before_path,
            optimizer_state_before_flat,
        )
        _write_f32(
            optimizer_state_after_path,
            optimizer_state_after_flat,
        )

    payload = {
        "loss_before": loss_before,
        "loss_after": loss_after,
        "num_grad_words": int(
            gradients_flat.size
        ),
        "num_weight_words": int(
            weights_before_flat.size
        ),
        "layerwise_dir": str(layerwise_dir),
        "bypass_final_softmax_backward": bool(
            bypass_final_softmax_backward
        ),
        "loss": {
            "type": "cross_entropy" if loss_type in {"cross_entropy", "ce"} else loss_type,
            "softmax_stable": bool(loss_type in {"cross_entropy", "ce"}),
            "logits_ref_bin": str(logits_ref_path) if logits_ref_path is not None else None,
            "softmax_ref_bin": str(softmax_ref_path) if softmax_ref_path is not None else None,
            "cross_entropy_loss_ref_json": str(cross_entropy_loss_ref_json) if cross_entropy_loss_ref_json is not None else None,
            "dlogits_ref_bin": str(dlogits_ref_path) if dlogits_ref_path is not None else None,
        },
        "tiled_io": {
            "requested": bool(tiled_io_requested),
            "inputs_ref_bin": str(tiled_inputs_ref_path) if tiled_inputs_ref_path is not None else None,
            "labels_ref_bin": str(tiled_labels_ref_path) if tiled_labels_ref_path is not None else None,
            "outputs_ref_bin": str(tiled_outputs_ref_path) if tiled_outputs_ref_path is not None else None,
            "gradients_ref_bin": str(tiled_gradients_ref_path) if tiled_gradients_ref_path is not None else None,
            "weights_after_ref_bin": str(tiled_weights_after_ref_path) if tiled_weights_after_ref_path is not None else None,
        },
        "optimizer": {
            "type": optimizer_type,
            "learning_rate": learning_rate,
            "momentum": momentum if optimizer_type == "momentum" else None,
            "beta1": beta1 if optimizer_type == "adam" else None,
            "beta2": beta2 if optimizer_type == "adam" else None,
            "epsilon": epsilon if optimizer_type == "adam" else None,
            "bias_correction": bias_correction if optimizer_type == "adam" else False,
            "state_words": int(optimizer_state_after_flat.size),
            "state_before_ref_bin": str(optimizer_state_before_path) if optimizer_type in {"momentum", "adam"} else None,
            "state_after_ref_bin": str(optimizer_state_after_path) if optimizer_type in {"momentum", "adam"} else None,
        },
    }

    summary_json.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    summary_txt.write_text(
        "\n".join(
            [
                "=========== FPGAI Training Reference ===========",
                f"loss_before : {loss_before}",
                f"loss_after  : {loss_after}",
                f"grad_words  : {int(gradients_flat.size)}",
                f"param_words : {int(weights_before_flat.size)}",
                f"optimizer   : {optimizer_type}",
                f"loss        : {loss_type}",
                f"tiled_io    : {bool(tiled_io_requested)}",
                f"optimizer_state_words : {int(optimizer_state_after_flat.size)}",
                f"layerwise   : {layerwise_dir}",
                (
                    "bypass_final_softmax_backward : "
                    f"{bool(bypass_final_softmax_backward)}"
                ),
                "================================================",
            ]
        ),
        encoding="utf-8",
    )

    return TrainingReferenceResult(
        out_dir=reference_dir,
        grads_flat_path=gradients_path,
        weights_before_flat_path=weights_before_path,
        weights_after_flat_path=weights_after_path,
        summary_json=summary_json,
        summary_txt=summary_txt,
        loss_before=loss_before,
        loss_after=loss_after,
        layerwise_dir=layerwise_dir,
        optimizer_type=optimizer_type,
        optimizer_bias_correction=bias_correction if optimizer_type == "adam" else False,
        optimizer_state_before_flat_path=(optimizer_state_before_path if optimizer_type in {"momentum", "adam"} else None),
        optimizer_state_after_flat_path=(optimizer_state_after_path if optimizer_type in {"momentum", "adam"} else None),
        loss_type=("cross_entropy" if loss_type in {"cross_entropy", "ce"} else loss_type),
        logits_ref_path=logits_ref_path,
        softmax_ref_path=softmax_ref_path,
        cross_entropy_loss_ref_json=cross_entropy_loss_ref_json,
        dlogits_ref_path=dlogits_ref_path,
    )