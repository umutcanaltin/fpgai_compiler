from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np

from fpgai.benchmark.training_reference import TrainingReferenceResult, run_training_reference_step
from fpgai.numerics.fixed_emulation import quantize_ap_fixed_array
from fpgai.engine.training import resolve_training_execution_schedule, training_record_order
from fpgai.engine.training_graph_utils import (
    as_chw,
    as_numpy_numeric,
    dense_input_preflatten_shape,
    get_tensor_shape,
    read_named_array,
    resolve_batchnorm_arrays,
    resolve_conv_arrays,
    resolve_dense_arrays,
)


def _write_f32(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(values, dtype=np.float32).reshape(-1).tofile(path)


def _attr_parameter_binding(
    graph: Any,
    op: Any,
    *,
    role: str,
    expected_count: int,
) -> tuple[str, str]:
    input_index = {"weight": 1, "bias": 2, "gamma": 1, "beta": 2}[role]
    if len(op.inputs) > input_index:
        tensor_name = op.inputs[input_index]
        array = read_named_array(graph, tensor_name)
        if array is not None and int(array.size) == int(expected_count):
            return ("named", tensor_name)

    preferred = {
        "weight": ("weights", "weight", "kernel", "W", "w", "weight_data", "weights_data", "kernel_data", "weight_values", "weights_values", "kernel_values"),
        "bias": ("bias", "biases", "B", "b", "bias_data", "bias_values"),
        "gamma": ("gamma", "scale", "weight", "weights"),
        "beta": ("beta", "bias", "biases"),
    }[role]
    attrs = getattr(op, "attrs", {}) or {}
    ordered_keys = list(preferred) + [key for key in attrs if key not in preferred]
    for key in ordered_keys:
        if key not in attrs:
            continue
        value = attrs[key]
        if isinstance(value, str):
            array = read_named_array(graph, value)
            if array is not None and int(array.size) == int(expected_count):
                return ("named", value)
            continue
        array = as_numpy_numeric(value)
        if array is not None and int(array.size) == int(expected_count):
            return ("attr", key)

    # Missing biases are represented as zero arrays by the resolver. Persist
    # the updated bias in a standard attribute so the second reference pass
    # sees the optimizer result rather than recreating zeros.
    if role in {"bias", "beta"}:
        return ("attr", "bias" if role == "bias" else "beta")
    raise RuntimeError(
        f"Dataset reference could not bind {role} parameters for '{op.name}' "
        f"with {expected_count} values."
    )


def _trainable_layout(graph: Any) -> list[tuple[str, str, str, str, tuple[int, ...], int]]:
    layout: list[tuple[str, str, str, str, tuple[int, ...], int]] = []
    for op in graph.ops:
        if op.op_type == "Dense":
            w, b, _, _ = resolve_dense_arrays(graph, op)
            w_kind, w_key = _attr_parameter_binding(graph, op, role="weight", expected_count=int(w.size))
            b_kind, b_key = _attr_parameter_binding(graph, op, role="bias", expected_count=int(b.size))
            layout.extend([
                (op.name, w_kind, w_key, "weight", tuple(w.shape), int(w.size)),
                (op.name, b_kind, b_key, "bias", tuple(b.shape), int(b.size)),
            ])
        elif op.op_type == "Conv":
            w, b, _ = resolve_conv_arrays(graph, op)
            w_kind, w_key = _attr_parameter_binding(graph, op, role="weight", expected_count=int(w.size))
            b_kind, b_key = _attr_parameter_binding(graph, op, role="bias", expected_count=int(b.size))
            layout.extend([
                (op.name, w_kind, w_key, "weight", tuple(w.shape), int(w.size)),
                (op.name, b_kind, b_key, "bias", tuple(b.shape), int(b.size)),
            ])
        elif op.op_type == "BatchNormalization":
            shape = get_tensor_shape(graph, op.outputs[0])
            channels, _, _ = as_chw(shape)
            gamma, beta, _, _ = resolve_batchnorm_arrays(graph, op, channels)
            g_kind, g_key = _attr_parameter_binding(graph, op, role="gamma", expected_count=int(gamma.size))
            b_kind, b_key = _attr_parameter_binding(graph, op, role="beta", expected_count=int(beta.size))
            layout.extend([
                (op.name, g_kind, g_key, "gamma", tuple(gamma.shape), int(gamma.size)),
                (op.name, b_kind, b_key, "beta", tuple(beta.shape), int(beta.size)),
            ])
    return layout


def _dense_inverse_storage_order(graph: Any, op: Any, weights: np.ndarray) -> np.ndarray:
    logical_shape = dense_input_preflatten_shape(graph, op)
    if len(logical_shape) != 3 or weights.ndim != 2:
        return weights
    channels, height, width = (int(v) for v in logical_shape)
    if weights.shape[1] != channels * height * width:
        return weights
    chw_to_hwc = []
    for channel in range(channels):
        for row in range(height):
            for column in range(width):
                chw_to_hwc.append((row * width + column) * channels + channel)
    return np.asarray(weights[:, np.asarray(chw_to_hwc, dtype=np.int64)], dtype=np.float32)


def _assign_flat_weights(
    graph: Any,
    flat: np.ndarray,
    layout: Iterable[tuple[str, str, str, str, tuple[int, ...], int]],
) -> None:
    cursor = 0
    op_by_name = {op.name: op for op in graph.ops}
    for op_name, binding_kind, binding_key, role, shape, count in layout:
        chunk = np.asarray(flat[cursor:cursor + count], dtype=np.float32).reshape(shape)
        op = op_by_name[op_name]
        if op.op_type == "Dense" and role == "weight":
            chunk = _dense_inverse_storage_order(graph, op, chunk)
        if binding_kind == "named":
            if binding_key in getattr(graph, "constants", {}):
                graph.constants[binding_key] = chunk.copy()
            elif binding_key in getattr(graph, "params", {}):
                graph.params[binding_key] = chunk.copy()
            else:
                graph.constants[binding_key] = chunk.copy()
        elif binding_kind == "attr":
            op.attrs[binding_key] = chunk.copy()
        else:
            raise RuntimeError(f"Unsupported dataset-reference binding kind: {binding_kind}")
        cursor += count
    if cursor != int(np.asarray(flat).size):
        raise RuntimeError(f"Dataset reference weight-layout mismatch: consumed {cursor}, got {np.asarray(flat).size}")




def _precision_spec(raw_cfg: Dict[str, Any], path: tuple[str, ...], fallback: Dict[str, Any]) -> Dict[str, Any]:
    node: Any = raw_cfg
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return dict(fallback)
        node = node[key]
    return dict(node) if isinstance(node, dict) else dict(fallback)




def _training_numeric_specs(raw_cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Resolve the same global numeric roles used by emit_types_h()."""
    activation = _precision_spec(
        raw_cfg,
        ("numerics", "defaults", "activation"),
        {"type": "ap_fixed", "total_bits": 16, "int_bits": 6},
    )
    weight = _precision_spec(
        raw_cfg,
        ("numerics", "defaults", "weight"),
        {"type": "ap_fixed", "total_bits": 16, "int_bits": 6},
    )
    bias = _precision_spec(
        raw_cfg,
        ("numerics", "defaults", "bias"),
        {"type": "ap_fixed", "total_bits": 24, "int_bits": 10},
    )
    accum = _precision_spec(
        raw_cfg,
        ("numerics", "defaults", "accum"),
        {"type": "ap_fixed", "total_bits": 24, "int_bits": 10},
    )
    generic_grad = _precision_spec(raw_cfg, ("numerics", "training", "grad"), activation)
    grad_activation = _precision_spec(
        raw_cfg,
        ("numerics", "training", "grad_activation"),
        generic_grad,
    )
    grad_weight = _precision_spec(
        raw_cfg,
        ("numerics", "training", "grad_weight"),
        weight,
    )
    grad_bias = _precision_spec(
        raw_cfg,
        ("numerics", "training", "grad_bias"),
        bias,
    )
    update_accum = _precision_spec(
        raw_cfg,
        ("numerics", "training", "update_accum"),
        accum,
    )
    optimizer_state = _precision_spec(
        raw_cfg,
        ("numerics", "training", "optimizer_state"),
        accum,
    )
    return {
        "activation": activation,
        "weight": weight,
        "bias": bias,
        "accum": accum,
        "grad_activation": grad_activation,
        "grad_weight": grad_weight,
        "grad_bias": grad_bias,
        "update_accum": update_accum,
        "optimizer_state": optimizer_state,
    }


def _q(values: Any, spec: Dict[str, Any]) -> np.ndarray:
    return quantize_ap_fixed_array(np.asarray(values, dtype=np.float32), spec)


def _hls_dense_forward_numeric(
    x: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    specs: Dict[str, Dict[str, Any]],
) -> np.ndarray:
    act_spec = specs["activation"]
    acc_spec = specs["accum"]
    x_q = _q(x, act_spec).reshape(-1)
    w_q = _q(weights, specs["weight"])
    b_q = _q(bias, specs["bias"]).reshape(-1)
    output = np.zeros((b_q.size,), dtype=np.float32)
    for output_index in range(b_q.size):
        accumulator = float(_q([b_q[output_index]], acc_spec)[0])
        for input_index in range(x_q.size):
            product = float(x_q[input_index]) * float(w_q[output_index, input_index])
            accumulator = float(_q([accumulator + product], acc_spec)[0])
        output[output_index] = _q([accumulator], act_spec)[0]
    return output


def _hls_softmax_numeric(
    x: np.ndarray,
    specs: Dict[str, Dict[str, Any]],
) -> np.ndarray:
    """Emulate layers_activations.h polynomial Softmax and cast boundaries."""
    act_spec = specs["activation"]
    acc_spec = specs["accum"]
    x_q = _q(x, act_spec).reshape(-1)
    maximum = float(_q([x_q[0]], acc_spec)[0])
    for value in x_q[1:]:
        candidate = float(_q([value], acc_spec)[0])
        if candidate > maximum:
            maximum = candidate

    temporary = np.zeros_like(x_q)
    total = 0.0
    for index, value in enumerate(x_q):
        shifted = float(_q([float(value) - maximum], acc_spec)[0])
        if shifted <= -8.0:
            exponential = 0.0
        elif shifted >= 0.0:
            exponential = 1.0
        else:
            squared = float(_q([shifted * shifted], acc_spec)[0])
            result = float(_q([1.0 + shifted + squared * 0.5], acc_spec)[0])
            exponential = max(result, 0.0)
        exponential = float(_q([exponential], act_spec)[0])
        temporary[index] = exponential
        total = float(_q([total + exponential], acc_spec)[0])
    if total <= 0.0:
        total = float(_q([1.0], acc_spec)[0])
    return _q(temporary / total, act_spec).reshape(-1)


def _hls_dense_backward_numeric(
    x: np.ndarray,
    output_gradient: np.ndarray,
    weights: np.ndarray,
    specs: Dict[str, Dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    act_spec = specs["activation"]
    grad_act_spec = specs["grad_activation"]
    grad_weight_spec = specs["grad_weight"]
    grad_bias_spec = specs["grad_bias"]
    acc_spec = specs["accum"]
    x_q = _q(x, act_spec).reshape(-1)
    dy_q = _q(output_gradient, grad_act_spec).reshape(-1)
    w_q = _q(weights, specs["weight"])

    d_weight = np.zeros(w_q.shape, dtype=np.float32)
    for output_index in range(dy_q.size):
        dy_acc = float(_q([dy_q[output_index]], acc_spec)[0])
        for input_index in range(x_q.size):
            x_acc = float(_q([x_q[input_index]], acc_spec)[0])
            d_weight[output_index, input_index] = _q(
                [dy_acc * x_acc], grad_weight_spec
            )[0]

    d_bias = _q(dy_q, grad_bias_spec).reshape(-1)
    d_input = np.zeros((x_q.size,), dtype=np.float32)
    for input_index in range(x_q.size):
        accumulator = 0.0
        for output_index in range(dy_q.size):
            product = float(_q([dy_q[output_index]], acc_spec)[0]) * float(
                _q([w_q[output_index, input_index]], acc_spec)[0]
            )
            accumulator = float(_q([accumulator + product], acc_spec)[0])
        d_input[input_index] = _q([accumulator], grad_act_spec)[0]
    return d_weight, d_bias, d_input


def _hls_softmax_backward_numeric(
    probabilities: np.ndarray,
    output_gradient: np.ndarray,
    specs: Dict[str, Dict[str, Any]],
) -> np.ndarray:
    act_spec = specs["activation"]
    grad_act_spec = specs["grad_activation"]
    acc_spec = specs["accum"]
    y = _q(probabilities, act_spec).reshape(-1)
    dy = _q(output_gradient, grad_act_spec).reshape(-1)
    result = np.zeros_like(dy)
    for output_index in range(y.size):
        accumulator = 0.0
        for input_index in range(y.size):
            if output_index == input_index:
                jacobian = float(y[output_index]) * (1.0 - float(y[output_index]))
            else:
                jacobian = -float(y[output_index]) * float(y[input_index])
            term = jacobian * float(dy[input_index])
            accumulator = float(_q([accumulator + term], acc_spec)[0])
        result[output_index] = _q([accumulator], grad_act_spec)[0]
    return result


def _run_hls_numeric_training_sample(
    *,
    graph: Any,
    raw_cfg: Dict[str, Any],
    x_input: np.ndarray,
    target: np.ndarray,
    layout: list[tuple[str, str, str, str, tuple[int, ...], int]],
    return_prediction: bool = False,
) -> Any:
    """Emulate the generated Dense/activation training arithmetic operation by operation.

    This is intentionally independent of the float reference.  It mirrors the
    global typedefs and the explicit cast points in the generated HLS kernels.
    Unsupported operators fail explicitly instead of silently falling back to a
    float gradient that could be mistaken for a hardware-domain reference.
    """
    specs = _training_numeric_specs(raw_cfg)
    values: Dict[str, np.ndarray] = {
        graph.inputs[0]: _q(np.asarray(x_input, dtype=np.float32).reshape(-1), specs["activation"])
    }
    caches: Dict[str, Dict[str, np.ndarray]] = {}
    parameters: Dict[str, Dict[str, np.ndarray]] = {}

    for op in graph.ops:
        input_name = op.inputs[0]
        output_name = op.outputs[0]
        input_value = values[input_name]
        if op.op_type == "Dense":
            weights, bias, _, _ = resolve_dense_arrays(graph, op)
            weights_q = _q(weights, specs["weight"])
            bias_q = _q(bias, specs["bias"])
            parameters[op.name] = {"W": weights_q, "B": bias_q}
            caches[op.name] = {"x": input_value.copy()}
            values[output_name] = _hls_dense_forward_numeric(
                input_value, weights_q, bias_q, specs
            )
        elif op.op_type == "Relu":
            values[output_name] = _q(np.maximum(input_value, 0.0), specs["activation"])
        elif op.op_type == "LeakyRelu":
            alpha = float((getattr(op, "attrs", {}) or {}).get("alpha", 0.01))
            alpha_q = float(_q([alpha], specs["activation"])[0])
            caches[op.name] = {"x": input_value.copy(), "alpha": np.asarray([alpha_q], dtype=np.float32)}
            values[output_name] = _q(
                np.where(input_value > 0.0, input_value, input_value * alpha_q),
                specs["activation"],
            )
        elif op.op_type == "Softmax":
            values[output_name] = _hls_softmax_numeric(input_value, specs)
        elif op.op_type in {"Flatten", "Reshape"}:
            values[output_name] = _q(input_value.reshape(-1), specs["activation"])
        else:
            raise NotImplementedError(
                f"Hardware-domain dataset reference does not yet support operator {op.op_type!r}."
            )

    output_name = graph.outputs[0]
    prediction = values[output_name].reshape(-1)
    target_q = _q(np.asarray(target, dtype=np.float32).reshape(-1), specs["activation"])
    loss_type = str(((raw_cfg.get("training", {}) or {}).get("loss", {}) or {}).get("type", "mse")).strip().lower()
    final_is_softmax = bool(graph.ops and graph.ops[-1].op_type == "Softmax" and graph.ops[-1].outputs[0] == output_name)
    if loss_type in {"cross_entropy", "ce"}:
        if final_is_softmax:
            probabilities = prediction
        else:
            probabilities = _hls_softmax_numeric(prediction, specs)
        output_gradient = _q(probabilities - target_q, specs["grad_activation"])
        loss = float(-np.sum(target_q * np.log(np.maximum(probabilities, np.float32(1.0e-7)))))
    elif loss_type == "mse":
        difference = _q(prediction - target_q, specs["accum"])
        output_gradient = _q(difference, specs["grad_activation"])
        loss = float(np.sum(difference * difference))
    else:
        raise NotImplementedError(
            f"Hardware-domain dataset reference does not yet support loss {loss_type!r}."
        )

    gradients_by_tensor: Dict[str, np.ndarray] = {output_name: output_gradient.reshape(-1)}
    parameter_gradients: Dict[tuple[str, str], np.ndarray] = {}
    for op_index in range(len(graph.ops) - 1, -1, -1):
        op = graph.ops[op_index]
        input_name = op.inputs[0]
        output_name_for_op = op.outputs[0]
        dy = gradients_by_tensor[output_name_for_op]
        if op.op_type == "Dense":
            d_weight, d_bias, d_input = _hls_dense_backward_numeric(
                caches[op.name]["x"], dy, parameters[op.name]["W"], specs
            )
            parameter_gradients[(op.name, "weight")] = d_weight.reshape(-1)
            parameter_gradients[(op.name, "bias")] = d_bias.reshape(-1)
            gradients_by_tensor[input_name] = d_input.reshape(-1)
        elif op.op_type == "Relu":
            output_value = values[output_name_for_op]
            gradients_by_tensor[input_name] = _q(
                np.where(output_value > 0.0, dy, 0.0), specs["grad_activation"]
            )
        elif op.op_type == "LeakyRelu":
            input_value = caches[op.name]["x"]
            alpha_q = float(caches[op.name]["alpha"][0])
            gradients_by_tensor[input_name] = _q(
                np.where(input_value > 0.0, dy, alpha_q * dy), specs["grad_activation"]
            )
        elif op.op_type == "Softmax":
            is_final = bool(op_index == len(graph.ops) - 1 and output_name_for_op == output_name)
            if final_is_softmax and is_final and loss_type in {"cross_entropy", "ce"}:
                gradients_by_tensor[input_name] = _q(dy, specs["grad_activation"])
            else:
                gradients_by_tensor[input_name] = _hls_softmax_backward_numeric(
                    values[output_name_for_op], dy, specs
                )
        elif op.op_type in {"Flatten", "Reshape"}:
            gradients_by_tensor[input_name] = _q(dy.reshape(-1), specs["grad_activation"])
        else:
            raise NotImplementedError(
                f"Hardware-domain backward reference does not yet support operator {op.op_type!r}."
            )

    chunks: list[np.ndarray] = []
    for op_name, _binding_kind, _binding_key, role, _shape, count in layout:
        key = (op_name, role)
        if key not in parameter_gradients:
            raise RuntimeError(f"Hardware-domain reference did not produce {role} gradient for {op_name!r}.")
        chunk = np.asarray(parameter_gradients[key], dtype=np.float32).reshape(-1)
        if chunk.size != count:
            raise RuntimeError(
                f"Hardware-domain gradient size mismatch for {op_name}.{role}: {chunk.size} != {count}."
            )
        chunks.append(chunk)
    gradient_vector = np.concatenate(chunks).astype(np.float32)
    prediction_vector = probabilities.astype(np.float32) if loss_type in {"cross_entropy", "ce"} else prediction.astype(np.float32)
    if return_prediction:
        return gradient_vector, loss, prediction_vector
    return gradient_vector, loss

def _parameter_specs_for_layout(
    raw_cfg: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    default_weight = {"type": "ap_fixed", "total_bits": 16, "int_bits": 6}
    default_bias = {"type": "ap_fixed", "total_bits": 24, "int_bits": 10}
    default_accum = {"type": "ap_fixed", "total_bits": 24, "int_bits": 10}
    weight_spec = _precision_spec(raw_cfg, ("numerics", "defaults", "weight"), default_weight)
    bias_spec = _precision_spec(raw_cfg, ("numerics", "defaults", "bias"), default_bias)
    accum_spec = _precision_spec(raw_cfg, ("numerics", "defaults", "accum"), default_accum)
    grad_weight_spec = _precision_spec(raw_cfg, ("numerics", "training", "grad_weight"), weight_spec)
    grad_bias_spec = _precision_spec(raw_cfg, ("numerics", "training", "grad_bias"), bias_spec)
    update_spec = _precision_spec(raw_cfg, ("numerics", "training", "update_accum"), accum_spec)
    return weight_spec, bias_spec, accum_spec, grad_weight_spec, grad_bias_spec, update_spec


def _quantize_parameter_vector(
    values: np.ndarray,
    layout: list[tuple[str, str, str, str, tuple[int, ...], int]],
    *,
    weight_spec: Dict[str, Any],
    bias_spec: Dict[str, Any],
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    result = np.zeros_like(values)
    cursor = 0
    for _op_name, _binding_kind, _binding_key, role, _shape, count in layout:
        parameter_spec = bias_spec if role in {"bias", "beta"} else weight_spec
        result[cursor:cursor + count] = quantize_ap_fixed_array(
            values[cursor:cursor + count], parameter_spec
        )
        cursor += count
    if cursor != values.size:
        raise RuntimeError(
            f"Hardware-domain parameter layout consumed {cursor} values, expected {values.size}."
        )
    return result


def _hardware_batch_update(
    *,
    graph: Any,
    raw_cfg: Dict[str, Any],
    inputs: np.ndarray,
    targets: np.ndarray,
    record_indices: list[int],
    current_weights: np.ndarray,
    layout: list[tuple[str, str, str, str, tuple[int, ...], int]],
    learning_rate: float,
    optimizer_type: str = "sgd",
    momentum: float = 0.0,
    optimizer_state_before: np.ndarray | None = None,
) -> Dict[str, Any]:
    (
        weight_spec,
        bias_spec,
        accum_spec,
        grad_weight_spec,
        grad_bias_spec,
        update_spec,
    ) = _parameter_specs_for_layout(raw_cfg)

    fixed_samples: list[np.ndarray] = []
    losses: list[float] = []
    for record_index in record_indices:
        fixed_gradient, loss = _run_hls_numeric_training_sample(
            graph=graph,
            raw_cfg=raw_cfg,
            x_input=np.asarray(inputs[record_index], dtype=np.float32),
            target=np.asarray(targets[record_index], dtype=np.float32),
            layout=layout,
        )
        fixed_samples.append(fixed_gradient)
        losses.append(float(loss))
    if not fixed_samples:
        raise RuntimeError("Hardware-domain batch update received no records.")

    sample_gradients = np.stack(fixed_samples, axis=0).astype(np.float32)
    current_weights = np.asarray(current_weights, dtype=np.float32).reshape(-1)
    q_grad = np.zeros(current_weights.shape, dtype=np.float32)
    q_accum_sum = np.zeros(current_weights.shape, dtype=np.float32)
    q_after = np.zeros(current_weights.shape, dtype=np.float32)
    q_per_sample = np.zeros(sample_gradients.shape, dtype=np.float32)
    q_accumulator_after = np.zeros(sample_gradients.shape, dtype=np.float32)
    lr_q = float(quantize_ap_fixed_array(np.asarray([learning_rate], dtype=np.float32), update_spec)[0])
    lr_acc = quantize_ap_fixed_array(np.asarray([lr_q], dtype=np.float32), accum_spec)[0]
    optimizer_state_spec = _training_numeric_specs(raw_cfg)["optimizer_state"]
    state_before = np.zeros(current_weights.shape, dtype=np.float32)
    if optimizer_type == "momentum":
        if optimizer_state_before is not None:
            candidate = np.asarray(optimizer_state_before, dtype=np.float32).reshape(-1)
            if candidate.shape != current_weights.shape:
                raise RuntimeError(
                    "Momentum optimizer-state shape mismatch: "
                    f"{candidate.shape} != {current_weights.shape}."
                )
            state_before = quantize_ap_fixed_array(candidate, optimizer_state_spec)
    elif optimizer_type != "sgd":
        raise ValueError(f"Hardware-domain dataset update does not support optimizer {optimizer_type!r}.")
    state_after = state_before.copy()
    momentum_q = float(
        quantize_ap_fixed_array(np.asarray([momentum], dtype=np.float32), update_spec)[0]
    ) if optimizer_type == "momentum" else 0.0

    cursor = 0
    for _op_name, _binding_kind, _binding_key, role, _shape, count in layout:
        sl = slice(cursor, cursor + count)
        is_bias = role in {"bias", "beta"}
        parameter_spec = bias_spec if is_bias else weight_spec
        gradient_spec = grad_bias_spec if is_bias else grad_weight_spec
        q_before_role = quantize_ap_fixed_array(current_weights[sl], parameter_spec)
        accumulator = np.zeros((count,), dtype=np.float32)
        for sample_index, row in enumerate(sample_gradients):
            per_sample = quantize_ap_fixed_array(row[sl], gradient_spec)
            accumulator = quantize_ap_fixed_array(accumulator + per_sample, accum_spec)
            q_per_sample[sample_index, sl] = per_sample
            q_accumulator_after[sample_index, sl] = accumulator
        q_accum_sum[sl] = accumulator
        mean_gradient = quantize_ap_fixed_array(
            accumulator / float(sample_gradients.shape[0]), gradient_spec
        )
        q_grad[sl] = mean_gradient
        gradient_acc = quantize_ap_fixed_array(mean_gradient, accum_spec)
        parameter_acc = quantize_ap_fixed_array(q_before_role, accum_spec)
        if optimizer_type == "momentum":
            velocity_before = quantize_ap_fixed_array(state_before[sl], optimizer_state_spec)
            velocity_acc = quantize_ap_fixed_array(velocity_before, accum_spec)
            velocity_updated = quantize_ap_fixed_array(
                (np.float32(momentum_q) * velocity_acc)
                - (np.float32(lr_acc) * gradient_acc),
                accum_spec,
            )
            velocity_after = quantize_ap_fixed_array(velocity_updated, optimizer_state_spec)
            state_after[sl] = velocity_after
            updated = quantize_ap_fixed_array(
                parameter_acc + quantize_ap_fixed_array(velocity_after, accum_spec),
                accum_spec,
            )
        else:
            updated = quantize_ap_fixed_array(
                parameter_acc - (np.float32(lr_acc) * gradient_acc), accum_spec
            )
        q_after[sl] = quantize_ap_fixed_array(updated, parameter_spec)
        cursor += count

    return {
        "gradient": q_grad,
        "accumulated_gradient": q_accum_sum,
        "weights_after": q_after,
        "per_sample_gradients": q_per_sample,
        "accumulators_after": q_accumulator_after,
        "mean_batch_loss": float(np.mean(losses)),
        "learning_rate_quantized": lr_q,
        "momentum_quantized": momentum_q if optimizer_type == "momentum" else None,
        "optimizer_state_before": state_before,
        "optimizer_state_after": state_after,
        "precision": {
            "weight": weight_spec,
            "bias": bias_spec,
            "accum": accum_spec,
            "grad_weight": grad_weight_spec,
            "grad_bias": grad_bias_spec,
            "update_accum": update_spec,
            "optimizer_state": optimizer_state_spec,
        },
    }


def _evaluate_float_dataset(
    *,
    graph: Any,
    zero_cfg: Dict[str, Any],
    root: Path,
    inputs: np.ndarray,
    targets: np.ndarray,
) -> tuple[float, np.ndarray, float]:
    losses: list[float] = []
    correct = 0
    weights: np.ndarray | None = None
    for index in range(int(inputs.shape[0])):
        result = run_training_reference_step(
            graph=graph,
            raw_cfg=zero_cfg,
            out_dir=root / f"sample_{index:04d}",
            x_input=np.asarray(inputs[index], dtype=np.float32).reshape(-1),
            target=np.asarray(targets[index], dtype=np.float32).reshape(-1),
        )
        losses.append(float(result.loss_before))
        prediction_path = result.softmax_ref_path or result.logits_ref_path
        if prediction_path is not None and Path(prediction_path).exists():
            prediction = np.fromfile(prediction_path, dtype=np.float32).reshape(-1)
            expected = np.asarray(targets[index], dtype=np.float32).reshape(-1)
            if prediction.size and expected.size and int(np.argmax(prediction)) == int(np.argmax(expected)):
                correct += 1
        if weights is None:
            weights = np.fromfile(result.weights_before_flat_path, dtype=np.float32)
    if weights is None:
        raise RuntimeError("Dataset evaluation produced no parameter vector.")
    return float(np.mean(losses)), weights, float(correct) / float(inputs.shape[0])


def _evaluate_hardware_dataset(
    *,
    graph: Any,
    raw_cfg: Dict[str, Any],
    inputs: np.ndarray,
    targets: np.ndarray,
    layout: list[tuple[str, str, str, str, tuple[int, ...], int]],
) -> tuple[float, float]:
    losses: list[float] = []
    correct = 0
    for index in range(int(inputs.shape[0])):
        _gradient, loss, prediction = _run_hls_numeric_training_sample(
            graph=graph,
            raw_cfg=raw_cfg,
            x_input=np.asarray(inputs[index], dtype=np.float32),
            target=np.asarray(targets[index], dtype=np.float32),
            layout=layout,
            return_prediction=True,
        )
        losses.append(float(loss))
        expected = np.asarray(targets[index], dtype=np.float32).reshape(-1)
        if prediction.size and expected.size and int(np.argmax(prediction)) == int(np.argmax(expected)):
            correct += 1
    return float(np.mean(losses)), float(correct) / float(inputs.shape[0])


def _write_training_curve_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "domain",
        "epoch",
        "optimizer_updates",
        "records_consumed",
        "dataset_loss",
        "gradient_l2_norm",
        "weight_update_l2_norm",
        "accuracy",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _hardware_domain_reference(
    *,
    graph: Any,
    raw_cfg: Dict[str, Any],
    inputs: np.ndarray,
    targets: np.ndarray,
    weights_before: np.ndarray,
    layout: list[tuple[str, str, str, str, tuple[int, ...], int]],
    learning_rate: float,
    out_dir: Path,
    schedule: Any,
) -> Dict[str, Any]:
    """Run the operation-level fixed-point reference across epochs and batches."""
    (
        weight_spec,
        bias_spec,
        accum_spec,
        grad_weight_spec,
        grad_bias_spec,
        update_spec,
    ) = _parameter_specs_for_layout(raw_cfg)
    hardware_graph = copy.deepcopy(graph)
    q_initial = _quantize_parameter_vector(
        weights_before,
        layout,
        weight_spec=weight_spec,
        bias_spec=bias_spec,
    )
    _assign_flat_weights(hardware_graph, q_initial, layout)
    current_weights = q_initial.copy()
    optimizer = ((raw_cfg.get("training", {}) or {}).get("optimizer", {}) or {})
    optimizer_type = str(optimizer.get("type", "sgd")).strip().lower().replace("-", "_")
    momentum = float(optimizer.get("momentum", 0.9))
    current_optimizer_state = np.zeros(current_weights.shape, dtype=np.float32)
    initial_optimizer_state = current_optimizer_state.copy()
    initial_loss, initial_accuracy = _evaluate_hardware_dataset(
        graph=hardware_graph,
        raw_cfg=raw_cfg,
        inputs=inputs,
        targets=targets,
        layout=layout,
    )

    total_updates = int(schedule.total_optimizer_updates or 0)
    batches_per_epoch = int(schedule.batches_per_epoch or 0)
    if total_updates <= 0 or batches_per_epoch <= 0:
        raise RuntimeError("Hardware-domain multi-epoch schedule has no optimizer updates.")

    last_update: Dict[str, Any] | None = None
    curve_rows: list[dict[str, Any]] = [{
        "domain": "hardware_fixed_point",
        "epoch": 0,
        "optimizer_updates": 0,
        "records_consumed": 0,
        "dataset_loss": initial_loss,
        "gradient_l2_norm": 0.0,
        "weight_update_l2_norm": 0.0,
        "accuracy": initial_accuracy,
    }]
    records_consumed = 0
    epoch_last_gradient_norm = 0.0
    epoch_update_norm = 0.0

    for update_index in range(total_updates):
        epoch_index, batch_index = divmod(update_index, batches_per_epoch)
        order = training_record_order(
            int(schedule.sample_count),
            epoch_index=epoch_index,
            shuffle=bool(schedule.shuffle),
            seed=int(schedule.seed),
        )
        start = batch_index * int(schedule.batch_size)
        stop = min(start + int(schedule.batch_size), int(schedule.sample_count))
        record_indices = order[start:stop]
        if schedule.drop_last and len(record_indices) < int(schedule.batch_size):
            continue
        before_update = current_weights.copy()
        last_update = _hardware_batch_update(
            graph=hardware_graph,
            raw_cfg=raw_cfg,
            inputs=inputs,
            targets=targets,
            record_indices=record_indices,
            current_weights=current_weights,
            layout=layout,
            learning_rate=learning_rate,
            optimizer_type=optimizer_type,
            momentum=momentum,
            optimizer_state_before=current_optimizer_state,
        )
        current_weights = np.asarray(last_update["weights_after"], dtype=np.float32)
        current_optimizer_state = np.asarray(
            last_update["optimizer_state_after"], dtype=np.float32
        )
        _assign_flat_weights(hardware_graph, current_weights, layout)
        records_consumed += len(record_indices)
        epoch_last_gradient_norm = float(np.linalg.norm(last_update["gradient"]))
        epoch_update_norm += float(np.linalg.norm(current_weights - before_update))

        end_of_epoch = batch_index == batches_per_epoch - 1
        end_of_run = update_index == total_updates - 1
        if end_of_epoch or end_of_run:
            epoch_loss, epoch_accuracy = _evaluate_hardware_dataset(
                graph=hardware_graph,
                raw_cfg=raw_cfg,
                inputs=inputs,
                targets=targets,
                layout=layout,
            )
            curve_rows.append({
                "domain": "hardware_fixed_point",
                "epoch": epoch_index + 1,
                "optimizer_updates": update_index + 1,
                "records_consumed": records_consumed,
                "dataset_loss": epoch_loss,
                "gradient_l2_norm": epoch_last_gradient_norm,
                "weight_update_l2_norm": epoch_update_norm,
                "accuracy": epoch_accuracy,
            })
            epoch_update_norm = 0.0

    if last_update is None:
        raise RuntimeError("Hardware-domain schedule executed no batch update.")

    root = Path(out_dir) / "hardware_domain"
    root.mkdir(parents=True, exist_ok=True)
    grads_path = root / "grads_ref.bin"
    accum_path = root / "gradient_accumulated_pre_reduce_ref.bin"
    reduced_path = root / "gradient_reduced_ref.bin"
    before_path = root / "weights_before_ref.bin"
    after_path = root / "weights_after_ref.bin"
    optimizer_state_before_path = root / "optimizer_state_before_ref.bin"
    optimizer_state_after_path = root / "optimizer_state_after_ref.bin"
    _write_f32(grads_path, last_update["gradient"])
    _write_f32(accum_path, last_update["accumulated_gradient"])
    _write_f32(reduced_path, last_update["gradient"])
    _write_f32(before_path, q_initial)
    _write_f32(after_path, current_weights)
    if optimizer_type == "momentum":
        _write_f32(optimizer_state_before_path, initial_optimizer_state)
        _write_f32(optimizer_state_after_path, current_optimizer_state)

    trace_root = root / "per_sample_trace"
    trace_root.mkdir(parents=True, exist_ok=True)
    per_sample_paths: list[str] = []
    accumulator_paths: list[str] = []
    for sample_index in range(int(last_update["per_sample_gradients"].shape[0])):
        sample_path = trace_root / f"per_sample_gradient_{sample_index:04d}_ref.bin"
        accum_sample_path = trace_root / f"accumulator_after_{sample_index:04d}_ref.bin"
        _write_f32(sample_path, last_update["per_sample_gradients"][sample_index])
        _write_f32(accum_sample_path, last_update["accumulators_after"][sample_index])
        per_sample_paths.append(str(sample_path))
        accumulator_paths.append(str(accum_sample_path))

    layer_map: list[dict[str, Any]] = []
    cursor = 0
    for op_name, _binding_kind, _binding_key, role, shape, count in layout:
        layer_map.append({
            "layer": op_name,
            "role": role,
            "offset": cursor,
            "count": count,
            "shape": list(shape),
        })
        cursor += count
    layer_map_path = trace_root / "parameter_layer_map.json"
    layer_map_path.write_text(
        json.dumps({"schema_version": 1, "entries": layer_map}, indent=2) + "\n",
        encoding="utf-8",
    )
    curve_path = root / "training_epoch_curve.csv"
    _write_training_curve_csv(curve_path, curve_rows)

    summary = {
        "artifact_kind": "fpgai_training_hardware_domain_reference",
        "schema_version": 2,
        "status": "available",
        "rounding_emulation": "AP_TRN",
        "overflow_emulation": "AP_WRAP",
        "update_cast_sequence": "upd_t_to_acc_t; grad_t_to_acc_t; expression_product; final_acc_t_cast; parameter_cast",
        "reference_method": "operation_level_fixed_point",
        "fallback_reason": None,
        "gradient_reduction": "quantize_each_sample_accumulate_then_mean",
        "learning_rate_quantized": last_update["learning_rate_quantized"],
        "optimizer_updates": total_updates,
        "epochs_completed": int(curve_rows[-1]["epoch"]),
        "records_consumed": records_consumed,
        "initial_dataset_loss": initial_loss,
        "final_dataset_loss": float(curve_rows[-1]["dataset_loss"]),
        "initial_accuracy": float(curve_rows[0]["accuracy"]),
        "final_accuracy": float(curve_rows[-1]["accuracy"]),
        "gradient_l2_norm": float(np.linalg.norm(last_update["gradient"])),
        "weight_update_l2_norm": float(np.linalg.norm(current_weights - q_initial)),
        "grads_ref_bin": str(grads_path),
        "gradient_accumulated_pre_reduce_ref_bin": str(accum_path),
        "gradient_reduced_ref_bin": str(reduced_path),
        "weights_before_ref_bin": str(before_path),
        "weights_after_ref_bin": str(after_path),
        "optimizer_type": optimizer_type,
        "optimizer_state_words": int(current_optimizer_state.size) if optimizer_type == "momentum" else 0,
        "optimizer_state_before_ref_bin": str(optimizer_state_before_path) if optimizer_type == "momentum" else None,
        "optimizer_state_after_ref_bin": str(optimizer_state_after_path) if optimizer_type == "momentum" else None,
        "per_sample_gradient_ref_bins": per_sample_paths,
        "accumulator_after_ref_bins": accumulator_paths,
        "parameter_layer_map_json": str(layer_map_path),
        "training_epoch_curve_csv": str(curve_path),
        "execution_schedule": schedule.to_dict(),
        "precision": last_update["precision"],
    }
    summary_path = root / "training_hardware_domain_reference.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    return summary

def _cfg_with_zero_lr(raw_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(raw_cfg)
    cfg.setdefault("training", {}).setdefault("optimizer", {})["learning_rate"] = 0.0
    return cfg


def run_training_dataset_reference(
    *,
    graph: Any,
    raw_cfg: Dict[str, Any],
    out_dir: Path,
    inputs: np.ndarray,
    targets: np.ndarray,
) -> TrainingReferenceResult:
    """Run deterministic accumulated mini-batch training across epochs.

    The software and operation-level fixed-point references share the same
    canonical record schedule.  The default one-batch/one-epoch workload remains
    identical to the previously validated P3D-F2 behavior.
    """
    inputs = np.asarray(inputs, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    if inputs.ndim < 2 or targets.ndim < 2:
        raise ValueError("Dataset training inputs and targets must include a sample dimension.")
    if int(inputs.shape[0]) != int(targets.shape[0]):
        raise ValueError(
            f"Dataset training sample/target count mismatch: {inputs.shape[0]} != {targets.shape[0]}."
        )

    training = raw_cfg.get("training", {}) or {}
    optimizer = training.get("optimizer", {}) or {}
    optimizer_type = str(optimizer.get("type", "sgd")).strip().lower().replace("-", "_")
    if optimizer_type not in {"sgd", "momentum"}:
        raise ValueError(
            "Dataset-wide multi-epoch training reference currently supports SGD and Momentum; "
            f"got optimizer.type={optimizer_type!r}."
        )
    learning_rate = float(optimizer.get("learning_rate", 0.01))
    momentum = float(optimizer.get("momentum", 0.9))
    schedule = resolve_training_execution_schedule(
        raw_cfg,
        sample_count=int(inputs.shape[0]),
    )
    accumulated_modes = {
        "accumulate",
        "accumulated",
        "true_minibatch",
        "true_minibatch",
        "mini_batch",
        "minibatch",
    }
    if schedule.batch_mode not in accumulated_modes:
        raise ValueError(
            "Dataset-wide multi-epoch reference requires accumulated mini-batches; "
            f"got batch_mode={schedule.batch_mode!r}."
        )

    root = Path(out_dir) / "training_dataset_reference"
    root.mkdir(parents=True, exist_ok=True)
    zero_cfg = _cfg_with_zero_lr(raw_cfg)
    float_graph = copy.deepcopy(graph)
    layout = _trainable_layout(float_graph)

    initial_loss, weights_before, initial_accuracy = _evaluate_float_dataset(
        graph=float_graph,
        zero_cfg=zero_cfg,
        root=root / "initial_evaluation",
        inputs=inputs,
        targets=targets,
    )
    current_weights = np.asarray(weights_before, dtype=np.float32).copy()
    optimizer_state_before = np.zeros(current_weights.shape, dtype=np.float32)
    current_optimizer_state = optimizer_state_before.copy()
    total_updates = int(schedule.total_optimizer_updates or 0)
    batches_per_epoch = int(schedule.batches_per_epoch or 0)
    if total_updates <= 0 or batches_per_epoch <= 0:
        raise RuntimeError("Resolved training schedule contains no optimizer updates.")

    float_curve_rows: list[dict[str, Any]] = [{
        "domain": "float_reference",
        "epoch": 0,
        "optimizer_updates": 0,
        "records_consumed": 0,
        "dataset_loss": initial_loss,
        "gradient_l2_norm": 0.0,
        "weight_update_l2_norm": 0.0,
        "accuracy": initial_accuracy,
    }]
    last_gradient: np.ndarray | None = None
    records_consumed = 0
    epoch_update_norm = 0.0
    epoch_last_gradient_norm = 0.0

    for update_index in range(total_updates):
        epoch_index, batch_index = divmod(update_index, batches_per_epoch)
        order = training_record_order(
            int(schedule.sample_count),
            epoch_index=epoch_index,
            shuffle=bool(schedule.shuffle),
            seed=int(schedule.seed),
        )
        start = batch_index * int(schedule.batch_size)
        stop = min(start + int(schedule.batch_size), int(schedule.sample_count))
        record_indices = order[start:stop]
        if schedule.drop_last and len(record_indices) < int(schedule.batch_size):
            continue
        if not record_indices:
            raise RuntimeError(
                f"Training schedule produced an empty batch at epoch={epoch_index}, batch={batch_index}."
            )

        sample_gradients: list[np.ndarray] = []
        batch_root = root / "batches" / f"epoch_{epoch_index + 1:04d}" / f"batch_{batch_index + 1:04d}"
        for slot, record_index in enumerate(record_indices):
            result = run_training_reference_step(
                graph=float_graph,
                raw_cfg=zero_cfg,
                out_dir=batch_root / f"slot_{slot:04d}_record_{record_index:04d}",
                x_input=np.asarray(inputs[record_index], dtype=np.float32).reshape(-1),
                target=np.asarray(targets[record_index], dtype=np.float32).reshape(-1),
            )
            current = np.fromfile(result.weights_before_flat_path, dtype=np.float32)
            if current.shape != current_weights.shape or not np.allclose(
                current, current_weights, atol=0.0, rtol=0.0
            ):
                raise RuntimeError(
                    "Dataset reference batch samples did not start from the same current weights."
                )
            sample_gradients.append(np.fromfile(result.grads_flat_path, dtype=np.float32))

        gradient_matrix = np.stack(sample_gradients, axis=0).astype(np.float32)
        last_gradient = np.mean(gradient_matrix, axis=0, dtype=np.float32).astype(np.float32)
        before_update = current_weights.copy()
        if optimizer_type == "momentum":
            current_optimizer_state = (
                momentum * current_optimizer_state - learning_rate * last_gradient
            ).astype(np.float32)
            current_weights = (current_weights + current_optimizer_state).astype(np.float32)
        else:
            current_weights = (current_weights - learning_rate * last_gradient).astype(np.float32)
        _assign_flat_weights(float_graph, current_weights, layout)
        records_consumed += len(record_indices)
        epoch_last_gradient_norm = float(np.linalg.norm(last_gradient))
        epoch_update_norm += float(np.linalg.norm(current_weights - before_update))

        end_of_epoch = batch_index == batches_per_epoch - 1
        end_of_run = update_index == total_updates - 1
        if end_of_epoch or end_of_run:
            epoch_loss, evaluated_weights, epoch_accuracy = _evaluate_float_dataset(
                graph=float_graph,
                zero_cfg=zero_cfg,
                root=root / "epoch_evaluation" / f"epoch_{epoch_index + 1:04d}",
                inputs=inputs,
                targets=targets,
            )
            if evaluated_weights.shape != current_weights.shape or not np.allclose(
                evaluated_weights, current_weights, atol=0.0, rtol=0.0
            ):
                raise RuntimeError("Epoch evaluation did not observe the current updated weights.")
            float_curve_rows.append({
                "domain": "float_reference",
                "epoch": epoch_index + 1,
                "optimizer_updates": update_index + 1,
                "records_consumed": records_consumed,
                "dataset_loss": epoch_loss,
                "gradient_l2_norm": epoch_last_gradient_norm,
                "weight_update_l2_norm": epoch_update_norm,
                "accuracy": epoch_accuracy,
            })
            epoch_update_norm = 0.0

    if last_gradient is None:
        raise RuntimeError("Dataset reference executed no optimizer update.")

    final_loss = float(float_curve_rows[-1]["dataset_loss"])
    hardware_domain = _hardware_domain_reference(
        graph=graph,
        raw_cfg=raw_cfg,
        inputs=inputs,
        targets=targets,
        weights_before=weights_before,
        layout=layout,
        learning_rate=learning_rate,
        out_dir=root,
        schedule=schedule,
    )

    grads_path = root / "grads_ref.bin"
    weights_before_path = root / "weights_before_ref.bin"
    weights_after_path = root / "weights_after_ref.bin"
    optimizer_state_before_path = root / "optimizer_state_before_ref.bin"
    optimizer_state_after_path = root / "optimizer_state_after_ref.bin"
    _write_f32(grads_path, last_gradient)
    _write_f32(weights_before_path, weights_before)
    _write_f32(weights_after_path, current_weights)
    if optimizer_type == "momentum":
        _write_f32(optimizer_state_before_path, optimizer_state_before)
        _write_f32(optimizer_state_after_path, current_optimizer_state)

    loss_change = final_loss - initial_loss
    loss_reduction = initial_loss - final_loss
    loss_reduction_pct = (
        100.0 * loss_reduction / abs(initial_loss) if initial_loss != 0.0 else None
    )
    weight_delta = current_weights - weights_before
    direction = (
        "decreased" if final_loss < initial_loss
        else ("increased" if final_loss > initial_loss else "unchanged")
    )
    curve_path = root / "training_epoch_curve.csv"
    _write_training_curve_csv(curve_path, float_curve_rows)

    summary = {
        "artifact_kind": "fpgai_training_dataset_reference",
        "schema_version": 2,
        "status": "available",
        "reference_scope": (
            "full_dataset_accumulated_update"
            if total_updates == 1
            and int(schedule.batch_size) == int(inputs.shape[0])
            and not bool(schedule.shuffle)
            else "deterministic_multi_epoch_accumulated_training"
        ),
        "sample_count": int(inputs.shape[0]),
        "optimizer_type": optimizer_type,
        "optimizer_updates": total_updates,
        "epochs_completed": int(float_curve_rows[-1]["epoch"]),
        "records_consumed": records_consumed,
        "learning_rate": learning_rate,
        "momentum": momentum if optimizer_type == "momentum" else None,
        "optimizer_state_words": int(current_optimizer_state.size) if optimizer_type == "momentum" else 0,
        "optimizer_state_before_ref_bin": str(optimizer_state_before_path) if optimizer_type == "momentum" else None,
        "optimizer_state_after_ref_bin": str(optimizer_state_after_path) if optimizer_type == "momentum" else None,
        "gradient_reduction": "mean_per_optimizer_batch",
        "initial_dataset_loss": initial_loss,
        "final_dataset_loss": final_loss,
        "initial_accuracy": float(float_curve_rows[0]["accuracy"]),
        "final_accuracy": float(float_curve_rows[-1]["accuracy"]),
        "loss_change": loss_change,
        "loss_reduction": loss_reduction,
        "loss_reduction_pct": loss_reduction_pct,
        "loss_direction": direction,
        "learning_observed": bool(final_loss < initial_loss),
        "convergence_claim": "not_evaluated",
        "gradient_l1_norm": float(np.sum(np.abs(last_gradient))),
        "gradient_l2_norm": float(np.linalg.norm(last_gradient)),
        "gradient_max_abs": float(np.max(np.abs(last_gradient))) if last_gradient.size else 0.0,
        "weight_update_l2_norm": float(np.linalg.norm(weight_delta)),
        "grads_ref_bin": str(grads_path),
        "weights_before_ref_bin": str(weights_before_path),
        "weights_after_ref_bin": str(weights_after_path),
        "training_epoch_curve_csv": str(curve_path),
        "execution_schedule": schedule.to_dict(),
        "hardware_domain_reference": hardware_domain,
    }
    summary_json = root / "training_dataset_reference.json"
    summary_txt = root / "training_dataset_reference.txt"
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary_txt.write_text(
        "\n".join([
            "FPGAI dataset-wide training reference",
            f"samples                    : {inputs.shape[0]}",
            f"epochs completed           : {summary['epochs_completed']}",
            f"optimizer updates          : {total_updates}",
            f"records consumed           : {records_consumed}",
            f"initial dataset loss       : {initial_loss:.9g}",
            f"final dataset loss         : {final_loss:.9g}",
            f"loss direction             : {direction}",
            f"last gradient L2 norm      : {summary['gradient_l2_norm']:.9g}",
            f"total weight update L2     : {summary['weight_update_l2_norm']:.9g}",
            "convergence claim          : not_evaluated",
        ]) + "\n",
        encoding="utf-8",
    )

    with (root / "training_loss_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["phase", "epoch", "loss"])
        for row in float_curve_rows:
            phase = "initial" if int(row["epoch"]) == 0 else "after_epoch"
            writer.writerow([phase, row["epoch"], row["dataset_loss"]])

    return TrainingReferenceResult(
        out_dir=root,
        grads_flat_path=grads_path,
        weights_before_flat_path=weights_before_path,
        weights_after_flat_path=weights_after_path,
        summary_json=summary_json,
        summary_txt=summary_txt,
        loss_before=initial_loss,
        loss_after=final_loss,
        layerwise_dir=root / "batches",
        optimizer_type=optimizer_type,
        optimizer_bias_correction=False,
        optimizer_state_before_flat_path=(optimizer_state_before_path if optimizer_type == "momentum" else None),
        optimizer_state_after_flat_path=(optimizer_state_after_path if optimizer_type == "momentum" else None),
        loss_type=str((training.get("loss", {}) or {}).get("type", "mse")),
    )

