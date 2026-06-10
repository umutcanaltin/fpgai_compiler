from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np


def shape_without_batch(shape: Any) -> Tuple[int, ...]:
    if shape is None:
        return tuple()

    normalized = tuple(int(value) for value in shape)

    if len(normalized) > 1 and normalized[0] == 1:
        return normalized[1:]

    return normalized


def flat_size(shape: Tuple[int, ...]) -> int:
    if not shape:
        return 1

    size = 1

    for dimension in shape:
        size *= int(dimension)

    return int(size)


def as_chw(shape: Tuple[int, ...]) -> Tuple[int, int, int]:
    if len(shape) != 3:
        raise ValueError(f"Expected 3D shape, got {shape}")

    channels, height, width = shape

    return (
        int(channels),
        int(height),
        int(width),
    )


def as_numpy_numeric(value: Any) -> Optional[np.ndarray]:
    if value is None or isinstance(value, (str, bytes)):
        return None

    try:
        array = np.asarray(value)
    except Exception:
        return None

    if array.dtype.kind in ("U", "S", "O"):
        return None

    return array.astype(np.float32, copy=False)


def read_named_array(
    graph: Any,
    name: str | None,
) -> Optional[np.ndarray]:
    if name is None:
        return None

    constants = getattr(graph, "constants", {})

    if name in constants:
        array = as_numpy_numeric(constants[name])

        if array is not None:
            return array

    params = getattr(graph, "params", {})

    if name in params:
        array = as_numpy_numeric(params[name])

        if array is not None:
            return array

    try:
        tensor = graph.get_tensor(name)
    except Exception:
        tensor = None

    if tensor is None:
        return None

    for attribute in ("data", "initializer", "value", "values"):
        if not hasattr(tensor, attribute):
            continue

        array = as_numpy_numeric(
            getattr(tensor, attribute)
        )

        if array is not None:
            return array

    return None


def flatten_named_array(
    graph: Any,
    name: str | None,
) -> Optional[np.ndarray]:
    array = read_named_array(graph, name)

    if array is None:
        return None

    return np.asarray(
        array,
        dtype=np.float32,
    ).reshape(-1)


def get_tensor_shape(
    graph: Any,
    name: str,
) -> Tuple[int, ...]:
    try:
        tensor = graph.get_tensor(name)
    except Exception:
        tensor = None

    if (
        tensor is not None
        and getattr(tensor, "shape", None)
    ):
        return shape_without_batch(tensor.shape)

    tensors = getattr(graph, "tensors", {})

    if (
        name in tensors
        and getattr(tensors[name], "shape", None)
    ):
        return shape_without_batch(
            tensors[name].shape
        )

    return tuple()


def infer_conv_output_shape(
    input_shape: Tuple[int, int, int],
    weight_shape: Tuple[int, int, int, int],
    stride: int,
    pad: int,
) -> Tuple[int, int, int]:
    _, height_in, width_in = as_chw(input_shape)

    (
        channels_out,
        _,
        kernel_height,
        kernel_width,
    ) = tuple(
        int(value)
        for value in weight_shape
    )

    if kernel_height != kernel_width:
        raise RuntimeError(
            "Only square convolution kernels are supported, "
            f"got weight shape {weight_shape}"
        )

    height_out = (
        (height_in + 2 * pad - kernel_height)
        // stride
    ) + 1

    width_out = (
        (width_in + 2 * pad - kernel_width)
        // stride
    ) + 1

    return (
        int(channels_out),
        int(height_out),
        int(width_out),
    )


def infer_pool_output_shape(
    input_shape: Tuple[int, int, int],
    kernel_size: int,
    stride: int,
) -> Tuple[int, int, int]:
    channels, height, width = as_chw(input_shape)

    height_out = (
        (height - kernel_size)
        // stride
    ) + 1

    width_out = (
        (width - kernel_size)
        // stride
    ) + 1

    return (
        int(channels),
        int(height_out),
        int(width_out),
    )


def dense_input_preflatten_shape(
    graph: Any,
    op: Any,
) -> Tuple[int, ...]:
    input_name = op.inputs[0]

    producer = next(
        (
            previous
            for previous in graph.ops
            if previous.outputs
            and previous.outputs[0] == input_name
        ),
        None,
    )

    if producer is None:
        return get_tensor_shape(
            graph,
            input_name,
        )

    if producer.op_type not in ("Flatten", "Reshape"):
        return get_tensor_shape(
            graph,
            input_name,
        )

    if not producer.inputs:
        return get_tensor_shape(
            graph,
            input_name,
        )

    source_shape = get_tensor_shape(
        graph,
        producer.inputs[0],
    )

    if source_shape:
        return source_shape

    return get_tensor_shape(
        graph,
        input_name,
    )


def remap_dense_weights_chw_to_hwc(
    weights: np.ndarray,
    logical_input_shape: Tuple[int, ...],
) -> np.ndarray:
    if len(logical_input_shape) != 3:
        return weights

    if weights.ndim != 2:
        return weights

    channels = int(logical_input_shape[0])
    height = int(logical_input_shape[1])
    width = int(logical_input_shape[2])

    feature_count = channels * height * width

    if weights.shape[1] != feature_count:
        return weights

    chw_to_hwc = np.empty(
        (feature_count,),
        dtype=np.int64,
    )

    chw_index = 0

    for channel in range(channels):
        for row in range(height):
            for column in range(width):
                hwc_index = (
                    (row * width + column)
                    * channels
                    + channel
                )

                chw_to_hwc[chw_index] = hwc_index
                chw_index += 1

    remapped = np.empty_like(weights)
    remapped[:, chw_to_hwc] = weights

    return remapped


def _resolve_attr_candidate(
    graph: Any,
    value: Any,
) -> tuple[
    Optional[np.ndarray],
    Optional[str],
]:
    array = as_numpy_numeric(value)

    if array is not None and array.size > 0:
        return (
            array.reshape(-1),
            "direct_numeric_attr",
        )

    if isinstance(value, str):
        array = flatten_named_array(
            graph,
            value,
        )

        if array is not None:
            return (
                array.reshape(-1),
                f"graph_ref('{value}')",
            )

    return None, None


def _candidate_attr_arrays(
    graph: Any,
    op: Any,
    expected_weight_count: int,
    expected_bias_count: int,
) -> list[
    tuple[
        str,
        str,
        np.ndarray,
        Optional[str],
    ]
]:
    attributes = getattr(op, "attrs", {}) or {}
    candidates = []

    weight_keys = (
        "weights",
        "weight",
        "kernel",
        "W",
        "w",
        "weight_data",
        "weights_data",
        "kernel_data",
        "weight_values",
        "weights_values",
        "kernel_values",
        "initializer",
        "params",
    )

    bias_keys = (
        "bias",
        "biases",
        "B",
        "b",
        "bias_data",
        "bias_values",
    )

    structural_keys = {
        "in_features",
        "out_features",
        "axis",
        "alpha",
        "beta",
        "epsilon",
        "momentum",
        "group",
        "kernel_shape",
        "kernel_shape_full",
        "pads",
        "strides",
        "dilations",
        "perm",
        "shape",
        "weight_shape",
        "weights_shape",
    }

    for key in weight_keys:
        if key not in attributes:
            continue

        array, source = _resolve_attr_candidate(
            graph,
            attributes[key],
        )

        if array is not None and array.size > 0:
            candidates.append(
                (
                    "weight",
                    key,
                    array,
                    source,
                )
            )

    for key in bias_keys:
        if key not in attributes:
            continue

        array, source = _resolve_attr_candidate(
            graph,
            attributes[key],
        )

        if array is not None and array.size > 0:
            candidates.append(
                (
                    "bias",
                    key,
                    array,
                    source,
                )
            )

    known_parameter_keys = (
        set(weight_keys)
        | set(bias_keys)
    )

    for key, value in attributes.items():
        if (
            key in known_parameter_keys
            or key in structural_keys
        ):
            continue

        # Scalar metadata such as in_features=4 must
        # never be inferred as a one-element bias.
        if not isinstance(value, str):
            try:
                raw_array = np.asarray(value)
            except Exception:
                continue

            if raw_array.ndim == 0:
                continue

        array, source = _resolve_attr_candidate(
            graph,
            value,
        )

        if array is None or array.size == 0:
            continue

        if (
            expected_weight_count > 0
            and array.size == expected_weight_count
        ):
            candidates.append(
                (
                    "weight",
                    key,
                    array,
                    source,
                )
            )

        if (
            expected_bias_count > 0
            and array.size == expected_bias_count
        ):
            candidates.append(
                (
                    "bias",
                    key,
                    array,
                    source,
                )
            )

    return candidates


def _dense_expected_shapes(
    op: Any,
) -> Tuple[int, int, int, int]:
    input_features = int(
        op.attrs.get("in_features")
        or 0
    )

    output_features = int(
        op.attrs.get("out_features")
        or 0
    )

    if (
        input_features > 0
        and output_features > 0
    ):
        weight_count = (
            output_features
            * input_features
        )
    else:
        weight_count = 0

    bias_count = (
        output_features
        if output_features > 0
        else 0
    )

    return (
        input_features,
        output_features,
        weight_count,
        bias_count,
    )


def resolve_dense_arrays(
    graph: Any,
    op: Any,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    int,
    int,
]:
    (
        input_features,
        output_features,
        weight_count,
        bias_count,
    ) = _dense_expected_shapes(op)

    weight_array = None
    bias_array = None

    if len(op.inputs) > 1:
        weight_array = flatten_named_array(
            graph,
            op.inputs[1],
        )

    if len(op.inputs) > 2:
        bias_array = flatten_named_array(
            graph,
            op.inputs[2],
        )

    if (
        weight_array is None
        or (
            bias_array is None
            and bias_count > 0
        )
    ):
        attribute_weight = None
        attribute_bias = None

        candidates = _candidate_attr_arrays(
            graph,
            op,
            weight_count,
            bias_count,
        )

        for (
            kind,
            _key,
            array,
            _source,
        ) in candidates:
            if (
                kind == "weight"
                and attribute_weight is None
            ):
                attribute_weight = array

            elif (
                kind == "bias"
                and attribute_bias is None
            ):
                attribute_bias = array

        if weight_array is None:
            weight_array = attribute_weight

        if bias_array is None:
            bias_array = attribute_bias

    if weight_array is None:
        raise RuntimeError(
            f"Dense weights not found for op '{op.name}'"
        )

    flat_input_shape = get_tensor_shape(
        graph,
        op.inputs[0],
    )

    logical_input_shape = (
        dense_input_preflatten_shape(
            graph,
            op,
        )
    )

    output_shape = get_tensor_shape(
        graph,
        op.outputs[0],
    )

    if input_features <= 0:
        input_features = flat_size(
            flat_input_shape
        )

    if output_features <= 0:
        output_features = flat_size(
            output_shape
        )

    expected_size = (
        output_features
        * input_features
    )

    if weight_array.size != expected_size:
        raise RuntimeError(
            f"Dense weight size mismatch for '{op.name}': "
            f"expected {expected_size}, "
            f"got {weight_array.size}"
        )

    weights = np.asarray(
        weight_array,
        dtype=np.float32,
    ).reshape(
        output_features,
        input_features,
    )

    weights = remap_dense_weights_chw_to_hwc(
        weights,
        logical_input_shape,
    )

    if bias_array is None:
        biases = np.zeros(
            (output_features,),
            dtype=np.float32,
        )
    else:
        if bias_array.size != output_features:
            raise RuntimeError(
                f"Dense bias size mismatch for '{op.name}': "
                f"expected {output_features}, "
                f"got {bias_array.size}"
            )

        biases = np.asarray(
            bias_array,
            dtype=np.float32,
        ).reshape(output_features)

    return (
        weights,
        biases,
        input_features,
        output_features,
    )


def resolve_conv_arrays(
    graph: Any,
    op: Any,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    Tuple[int, int, int, int],
]:
    weight_array = None
    bias_array = None

    if len(op.inputs) > 1:
        weight_array = flatten_named_array(
            graph,
            op.inputs[1],
        )

    if len(op.inputs) > 2:
        bias_array = flatten_named_array(
            graph,
            op.inputs[2],
        )

    attributes = getattr(
        op,
        "attrs",
        {},
    ) or {}

    if weight_array is None:
        for key in (
            "weights",
            "weight",
            "kernel",
            "W",
            "w",
            "weight_data",
            "kernel_data",
        ):
            if key not in attributes:
                continue

            candidate, _source = (
                _resolve_attr_candidate(
                    graph,
                    attributes[key],
                )
            )

            if candidate is not None:
                weight_array = candidate
                break

    if weight_array is None:
        raise RuntimeError(
            f"Conv weight tensor not found for op '{op.name}'"
        )

    weight_shape = None

    if len(op.inputs) > 1:
        try:
            weight_tensor = graph.get_tensor(
                op.inputs[1]
            )
        except Exception:
            weight_tensor = None

        if (
            weight_tensor is not None
            and getattr(
                weight_tensor,
                "shape",
                None,
            )
        ):
            weight_shape = tuple(
                int(value)
                for value in weight_tensor.shape
            )

    constants = getattr(
        graph,
        "constants",
        {},
    )

    if (
        weight_shape is None
        and len(op.inputs) > 1
        and op.inputs[1] in constants
    ):
        weight_shape = tuple(
            int(value)
            for value in np.asarray(
                constants[op.inputs[1]]
            ).shape
        )

    if weight_shape is None:
        for key in (
            "kernel_shape_full",
            "weight_shape",
            "weights_shape",
        ):
            value = attributes.get(key)

            if (
                isinstance(
                    value,
                    (list, tuple),
                )
                and len(value) == 4
            ):
                weight_shape = tuple(
                    int(item)
                    for item in value
                )
                break

    if weight_shape is None:
        input_shape = get_tensor_shape(
            graph,
            op.inputs[0],
        )

        output_shape = get_tensor_shape(
            graph,
            op.outputs[0],
        )

        if input_shape and output_shape:
            channels_in, _, _ = as_chw(
                input_shape
            )

            channels_out, _, _ = as_chw(
                output_shape
            )

            kernel_shape = attributes.get(
                "kernel_shape",
                [3, 3],
            )

            kernel_size = int(
                kernel_shape[0]
            )

            weight_shape = (
                channels_out,
                channels_in,
                kernel_size,
                kernel_size,
            )

    if weight_shape is None:
        raise RuntimeError(
            f"Conv weight shape unavailable for op '{op.name}'"
        )

    expected_size = int(
        np.prod(weight_shape)
    )

    if weight_array.size != expected_size:
        raise RuntimeError(
            f"Conv weight size mismatch for '{op.name}': "
            f"expected {expected_size}, "
            f"got {weight_array.size}"
        )

    channels_out = int(
        weight_shape[0]
    )

    if bias_array is None:
        bias_array = np.zeros(
            (channels_out,),
            dtype=np.float32,
        )

    if bias_array.size != channels_out:
        raise RuntimeError(
            f"Conv bias size mismatch for '{op.name}': "
            f"expected {channels_out}, "
            f"got {bias_array.size}"
        )

    weights = np.asarray(
        weight_array,
        dtype=np.float32,
    ).reshape(weight_shape)

    biases = np.asarray(
        bias_array,
        dtype=np.float32,
    ).reshape(channels_out)

    return (
        weights,
        biases,
        weight_shape,
    )


def resolve_batchnorm_arrays(
    graph: Any,
    op: Any,
    channels: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    if channels <= 0:
        raise ValueError(
            "BatchNorm channels must be positive, "
            f"got {channels}"
        )

    def get_or_default(
        input_index: int,
        default: float,
    ) -> np.ndarray:
        if len(op.inputs) > input_index:
            array = flatten_named_array(
                graph,
                op.inputs[input_index],
            )

            if array is not None:
                if array.size != channels:
                    raise RuntimeError(
                        "BatchNorm parameter size mismatch "
                        f"for '{op.name}': expected "
                        f"{channels}, got {array.size}"
                    )

                return np.asarray(
                    array,
                    dtype=np.float32,
                ).reshape(channels)

        attribute_keys = {
            1: (
                "gamma",
                "scale",
                "weight",
                "weights",
            ),
            2: (
                "beta",
                "bias",
                "biases",
            ),
            3: (
                "mean",
                "running_mean",
            ),
            4: (
                "var",
                "variance",
                "running_var",
            ),
        }.get(input_index, ())

        attributes = getattr(
            op,
            "attrs",
            {},
        ) or {}

        for key in attribute_keys:
            if key not in attributes:
                continue

            array, _source = (
                _resolve_attr_candidate(
                    graph,
                    attributes[key],
                )
            )

            if array is None:
                continue

            if array.size != channels:
                raise RuntimeError(
                    "BatchNorm attribute size mismatch "
                    f"for '{op.name}.{key}': expected "
                    f"{channels}, got {array.size}"
                )

            return np.asarray(
                array,
                dtype=np.float32,
            ).reshape(channels)

        return np.full(
            (channels,),
            default,
            dtype=np.float32,
        )

    gamma = get_or_default(1, 1.0)
    beta = get_or_default(2, 0.0)
    mean = get_or_default(3, 0.0)
    variance = get_or_default(4, 1.0)

    return (
        gamma,
        beta,
        mean,
        variance,
    )