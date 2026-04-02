from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import math
import numpy as np

from fpgai.util.fs import write_text


def _prod(shape) -> int:
    if not shape:
        return 1
    v = 1
    for d in shape:
        v *= int(d)
    return int(v)


def _shape_wo_batch(shape):
    if not shape:
        return tuple()
    shp = tuple(int(x) for x in shape)
    if len(shp) > 1 and shp[0] == 1:
        return shp[1:]
    return shp


def _bits_of(spec: Optional[Dict[str, Any]], default_bits: int = 32) -> int:
    if not spec:
        return int(default_bits)
    if "total_bits" in spec:
        return int(spec["total_bits"])
    return int(default_bits)


def _bytes_for_count(n: int, bits: int) -> int:
    return int(math.ceil((int(n) * int(bits)) / 8.0))


def _tensor_shape(graph, name: str):
    t = graph.get_tensor(name)
    if t is None:
        return tuple()
    return _shape_wo_batch(getattr(t, "shape", None))


def _try_to_numpy(value):
    if value is None:
        return None
    if isinstance(value, str):
        return None
    try:
        arr = np.asarray(value)
        if arr.dtype.kind in ("U", "S", "O"):
            return None
        return arr
    except Exception:
        return None


def _resolve_named_array(graph, name: str):
    if not name:
        return None

    if hasattr(graph, "constants") and name in getattr(graph, "constants", {}):
        arr = _try_to_numpy(graph.constants[name])
        if arr is not None:
            return arr

    if hasattr(graph, "params") and name in getattr(graph, "params", {}):
        arr = _try_to_numpy(graph.params[name])
        if arr is not None:
            return arr

    try:
        t = graph.get_tensor(name)
    except Exception:
        t = None

    if t is not None:
        for a in ("data", "initializer", "value", "values"):
            if hasattr(t, a):
                arr = _try_to_numpy(getattr(t, a))
                if arr is not None:
                    return arr

    return None


def _resolve_attr_array_or_ref(graph, op, *keys):
    attrs = getattr(op, "attrs", {}) or {}
    for k in keys:
        if k not in attrs:
            continue
        v = attrs[k]
        arr = _try_to_numpy(v)
        if arr is not None:
            return arr
        if isinstance(v, str):
            arr = _resolve_named_array(graph, v)
            if arr is not None:
                return arr
    return None


def _resolve_dense_param_elems(graph, op) -> int:
    w_arr = None
    b_arr = None

    if len(getattr(op, "inputs", [])) > 1:
        w_arr = _resolve_named_array(graph, op.inputs[1])
    if len(getattr(op, "inputs", [])) > 2:
        b_arr = _resolve_named_array(graph, op.inputs[2])

    if w_arr is None:
        w_arr = _resolve_attr_array_or_ref(graph, op, "weights", "weight", "W", "kernel", "weights_name", "weight_name")
    if b_arr is None:
        b_arr = _resolve_attr_array_or_ref(graph, op, "bias", "biases", "B", "bias_name")

    n = 0
    if w_arr is not None:
        n += int(np.asarray(w_arr).size)
    if b_arr is not None:
        n += int(np.asarray(b_arr).size)
    return n


def _resolve_conv_param_elems(graph, op) -> int:
    w_arr = None
    b_arr = None

    if len(getattr(op, "inputs", [])) > 1:
        w_arr = _resolve_named_array(graph, op.inputs[1])
    if len(getattr(op, "inputs", [])) > 2:
        b_arr = _resolve_named_array(graph, op.inputs[2])

    if w_arr is None:
        w_arr = _resolve_attr_array_or_ref(graph, op, "weights", "weight", "W", "kernel", "weights_name", "weight_name")
    if b_arr is None:
        b_arr = _resolve_attr_array_or_ref(graph, op, "bias", "biases", "B", "bias_name")

    n = 0
    if w_arr is not None:
        n += int(np.asarray(w_arr).size)
    if b_arr is not None:
        n += int(np.asarray(b_arr).size)
    return n


def _resolve_bn_param_elems(graph, op) -> int:
    arrays = []

    for idx in range(1, min(len(getattr(op, "inputs", [])), 5)):
        arr = _resolve_named_array(graph, op.inputs[idx])
        if arr is not None:
            arrays.append(arr)

    if not arrays:
        arrays.extend(
            x for x in [
                _resolve_attr_array_or_ref(graph, op, "scale", "gamma", "scale_name", "gamma_name"),
                _resolve_attr_array_or_ref(graph, op, "bias", "beta", "bias_name", "beta_name"),
                _resolve_attr_array_or_ref(graph, op, "mean", "running_mean", "mean_name", "running_mean_name"),
                _resolve_attr_array_or_ref(graph, op, "var", "running_var", "var_name", "running_var_name"),
            ]
            if x is not None
        )

    return int(sum(int(np.asarray(a).size) for a in arrays))


@dataclass(frozen=True)
class TrainingEstimateResult:
    out_dir: Path
    results_json: Path
    summary_txt: Path
    total_param_bytes: int
    total_activation_cache_bytes: int
    total_gradient_bytes: int
    total_optimizer_state_bytes: int
    total_forward_ops: int
    total_backward_input_ops: int
    total_backward_param_ops: int
    total_update_ops: int


def run_training_resource_estimate(
    *,
    graph,
    training_plan,
    out_dir: Path,
) -> TrainingEstimateResult:
    est_dir = out_dir / "training_estimate"
    est_dir.mkdir(parents=True, exist_ok=True)

    bits_act = _bits_of(training_plan.numerics.get("forward", {}).get("activation"), 32)
    bits_weight = _bits_of(training_plan.numerics.get("forward", {}).get("weight"), 32)
    bits_grad = _bits_of(training_plan.numerics.get("training", {}).get("grad"), 32)
    bits_optim = _bits_of(training_plan.numerics.get("training", {}).get("optimizer_state"), 32)

    total_param_elems = 0
    total_activation_cache_elems = 0
    total_gradient_elems = 0
    total_optimizer_state_elems = 0

    total_forward_ops = 0
    total_backward_input_ops = 0
    total_backward_param_ops = 0
    total_update_ops = 0

    layer_rows: List[Dict[str, Any]] = []

    for idx, op in enumerate(getattr(graph, "ops", [])):
        name = str(getattr(op, "name", f"op_{idx}"))
        op_type = str(getattr(op, "op_type", ""))

        xshape = _tensor_shape(graph, op.inputs[0]) if getattr(op, "inputs", None) else tuple()
        yshape = _tensor_shape(graph, op.outputs[0]) if getattr(op, "outputs", None) else tuple()

        in_elems = _prod(xshape)
        out_elems = _prod(yshape)

        caps_info = training_plan.op_capabilities.get(name, {})
        caps = caps_info.get("caps", {})
        cls = caps_info.get("classification", "unsupported")

        if op_type == "Dense":
            param_elems = _resolve_dense_param_elems(graph, op)
        elif op_type == "Conv":
            param_elems = _resolve_conv_param_elems(graph, op)
        elif op_type == "BatchNormalization":
            param_elems = _resolve_bn_param_elems(graph, op)
        else:
            param_elems = 0

        cache_elems = out_elems
        if op_type in ("Relu", "LeakyRelu", "Sigmoid"):
            cache_elems += out_elems
        if op_type == "MaxPool":
            cache_elems += out_elems

        grad_elems = 0
        if caps.get("backward_input", False):
            grad_elems += in_elems
        if caps.get("backward_params", False):
            grad_elems += param_elems

        optim_elems = param_elems if caps.get("update", False) else 0

        forward_ops = out_elems
        backward_input_ops = in_elems if caps.get("backward_input", False) else 0
        backward_param_ops = param_elems if caps.get("backward_params", False) else 0
        update_ops = param_elems if caps.get("update", False) else 0

        total_param_elems += param_elems
        total_activation_cache_elems += cache_elems
        total_gradient_elems += grad_elems
        total_optimizer_state_elems += optim_elems

        total_forward_ops += forward_ops
        total_backward_input_ops += backward_input_ops
        total_backward_param_ops += backward_param_ops
        total_update_ops += update_ops

        layer_rows.append(
            {
                "index": idx,
                "name": name,
                "op_type": op_type,
                "classification": cls,
                "input_elems": in_elems,
                "output_elems": out_elems,
                "param_elems": param_elems,
                "cache_elems": cache_elems,
                "gradient_elems": grad_elems,
                "optimizer_state_elems": optim_elems,
                "forward_ops_proxy": forward_ops,
                "backward_input_ops_proxy": backward_input_ops,
                "backward_param_ops_proxy": backward_param_ops,
                "update_ops_proxy": update_ops,
            }
        )

    total_param_bytes = _bytes_for_count(total_param_elems, bits_weight)
    total_activation_cache_bytes = _bytes_for_count(total_activation_cache_elems, bits_act)
    total_gradient_bytes = _bytes_for_count(total_gradient_elems, bits_grad)
    total_optimizer_state_bytes = _bytes_for_count(total_optimizer_state_elems, bits_optim)

    payload = {
        "storage_policy": {
            "weights_mode": training_plan.weights_mode,
            "weight_storage": training_plan.weight_storage,
            "activation_storage": training_plan.activation_storage,
            "gradient_storage": training_plan.gradient_storage,
            "optimizer_state_storage": training_plan.optimizer_state_storage,
        },
        "numerics_bits": {
            "activation_bits": bits_act,
            "weight_bits": bits_weight,
            "gradient_bits": bits_grad,
            "optimizer_state_bits": bits_optim,
        },
        "totals": {
            "param_bytes": total_param_bytes,
            "activation_cache_bytes": total_activation_cache_bytes,
            "gradient_bytes": total_gradient_bytes,
            "optimizer_state_bytes": total_optimizer_state_bytes,
            "forward_ops_proxy": total_forward_ops,
            "backward_input_ops_proxy": total_backward_input_ops,
            "backward_param_ops_proxy": total_backward_param_ops,
            "update_ops_proxy": total_update_ops,
        },
        "layers": layer_rows,
    }

    results_json = est_dir / "results.json"
    summary_txt = est_dir / "summary.txt"

    write_text(results_json, json.dumps(payload, indent=2))

    lines = []
    lines.append("=========== FPGAI Training Resource Estimate ===========")
    lines.append(f"weights_mode               : {training_plan.weights_mode}")
    lines.append(f"weight_storage             : {training_plan.weight_storage}")
    lines.append(f"activation_storage         : {training_plan.activation_storage}")
    lines.append(f"gradient_storage           : {training_plan.gradient_storage}")
    lines.append(f"optimizer_state_storage    : {training_plan.optimizer_state_storage}")
    lines.append(f"param_bytes                : {total_param_bytes}")
    lines.append(f"activation_cache_bytes     : {total_activation_cache_bytes}")
    lines.append(f"gradient_bytes             : {total_gradient_bytes}")
    lines.append(f"optimizer_state_bytes      : {total_optimizer_state_bytes}")
    lines.append(f"forward_ops_proxy          : {total_forward_ops}")
    lines.append(f"backward_input_ops_proxy   : {total_backward_input_ops}")
    lines.append(f"backward_param_ops_proxy   : {total_backward_param_ops}")
    lines.append(f"update_ops_proxy           : {total_update_ops}")
    lines.append("========================================================")

    write_text(summary_txt, "\n".join(lines))

    return TrainingEstimateResult(
        out_dir=est_dir,
        results_json=results_json,
        summary_txt=summary_txt,
        total_param_bytes=total_param_bytes,
        total_activation_cache_bytes=total_activation_cache_bytes,
        total_gradient_bytes=total_gradient_bytes,
        total_optimizer_state_bytes=total_optimizer_state_bytes,
        total_forward_ops=total_forward_ops,
        total_backward_input_ops=total_backward_input_ops,
        total_backward_param_ops=total_backward_param_ops,
        total_update_ops=total_update_ops,
    )