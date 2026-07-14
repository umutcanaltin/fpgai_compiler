from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np

from fpgai.benchmark.training_reference import TrainingReferenceResult, run_training_reference_step
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
    """Run a deterministic accumulated-mini-batch software reference.

    The current dataset reference supports one accumulated optimizer update and
    SGD. It evaluates every ordered sample, averages the parameter gradients,
    applies the configured SGD update once, and re-evaluates the full dataset.
    """
    training = raw_cfg.get("training", {}) or {}
    optimizer = training.get("optimizer", {}) or {}
    optimizer_type = str(optimizer.get("type", "sgd")).strip().lower().replace("-", "_")
    if optimizer_type != "sgd":
        raise ValueError(
            "Dataset-wide accumulated training reference currently supports SGD only; "
            f"got optimizer.type={optimizer_type!r}."
        )
    learning_rate = float(optimizer.get("learning_rate", 0.01))
    execution = training.get("execution", {}) or {}
    train_steps = int(execution.get("train_steps", 1))
    batch_size = int(execution.get("batch_size", inputs.shape[0]))
    batch_mode = str(execution.get("batch_mode", "replay")).strip().lower()
    if train_steps != 1 or batch_mode not in {"accumulate", "accumulated", "true_minibatch", "mini_batch", "minibatch"}:
        raise ValueError(
            "Dataset-wide reference currently requires train_steps=1 and an accumulated batch mode."
        )
    if int(inputs.shape[0]) != int(targets.shape[0]) or int(inputs.shape[0]) != batch_size:
        raise ValueError(
            "Dataset-wide reference requires sample_count == target_count == execution.batch_size; "
            f"got inputs={inputs.shape[0]}, targets={targets.shape[0]}, batch_size={batch_size}."
        )

    root = Path(out_dir) / "training_dataset_reference"
    root.mkdir(parents=True, exist_ok=True)
    sample_root = root / "samples_before"
    sample_root.mkdir(parents=True, exist_ok=True)
    zero_cfg = _cfg_with_zero_lr(raw_cfg)

    gradients: list[np.ndarray] = []
    losses_before: list[float] = []
    weights_before: np.ndarray | None = None
    first_result: TrainingReferenceResult | None = None

    for index in range(inputs.shape[0]):
        result = run_training_reference_step(
            graph=graph,
            raw_cfg=zero_cfg,
            out_dir=sample_root / f"sample_{index:04d}",
            x_input=np.asarray(inputs[index], dtype=np.float32).reshape(-1),
            target=np.asarray(targets[index], dtype=np.float32).reshape(-1),
        )
        first_result = first_result or result
        gradients.append(np.fromfile(result.grads_flat_path, dtype=np.float32))
        current_weights = np.fromfile(result.weights_before_flat_path, dtype=np.float32)
        if weights_before is None:
            weights_before = current_weights
        elif current_weights.shape != weights_before.shape or not np.allclose(current_weights, weights_before, atol=0.0, rtol=0.0):
            raise RuntimeError("Dataset reference samples did not start from identical weights.")
        losses_before.append(float(result.loss_before))

    assert first_result is not None and weights_before is not None
    gradient_matrix = np.stack(gradients, axis=0).astype(np.float32)
    avg_gradients = np.mean(gradient_matrix, axis=0, dtype=np.float32).astype(np.float32)
    weights_after = (weights_before - learning_rate * avg_gradients).astype(np.float32)

    updated_graph = copy.deepcopy(graph)
    layout = _trainable_layout(updated_graph)
    _assign_flat_weights(updated_graph, weights_after, layout)

    losses_after: list[float] = []
    after_root = root / "samples_after"
    after_root.mkdir(parents=True, exist_ok=True)
    for index in range(inputs.shape[0]):
        result_after = run_training_reference_step(
            graph=updated_graph,
            raw_cfg=zero_cfg,
            out_dir=after_root / f"sample_{index:04d}",
            x_input=np.asarray(inputs[index], dtype=np.float32).reshape(-1),
            target=np.asarray(targets[index], dtype=np.float32).reshape(-1),
        )
        losses_after.append(float(result_after.loss_before))

    grads_path = root / "grads_ref.bin"
    weights_before_path = root / "weights_before_ref.bin"
    weights_after_path = root / "weights_after_ref.bin"
    _write_f32(grads_path, avg_gradients)
    _write_f32(weights_before_path, weights_before)
    _write_f32(weights_after_path, weights_after)

    initial_loss = float(np.mean(losses_before))
    final_loss = float(np.mean(losses_after))
    loss_change = final_loss - initial_loss
    loss_reduction = initial_loss - final_loss
    loss_reduction_pct = (100.0 * loss_reduction / abs(initial_loss)) if initial_loss != 0.0 else None
    grad_l1 = float(np.sum(np.abs(avg_gradients)))
    grad_l2 = float(np.linalg.norm(avg_gradients))
    grad_max = float(np.max(np.abs(avg_gradients))) if avg_gradients.size else 0.0
    weight_delta = weights_after - weights_before
    update_l2 = float(np.linalg.norm(weight_delta))
    direction = "decreased" if final_loss < initial_loss else ("increased" if final_loss > initial_loss else "unchanged")

    summary = {
        "artifact_kind": "fpgai_training_dataset_reference",
        "schema_version": 1,
        "status": "available",
        "reference_scope": "full_dataset_accumulated_update",
        "sample_count": int(inputs.shape[0]),
        "optimizer_type": optimizer_type,
        "optimizer_updates": 1,
        "learning_rate": learning_rate,
        "gradient_reduction": "mean",
        "initial_dataset_loss": initial_loss,
        "final_dataset_loss": final_loss,
        "loss_change": loss_change,
        "loss_reduction": loss_reduction,
        "loss_reduction_pct": loss_reduction_pct,
        "loss_direction": direction,
        "learning_observed": bool(final_loss < initial_loss),
        "convergence_claim": "not_evaluated",
        "gradient_l1_norm": grad_l1,
        "gradient_l2_norm": grad_l2,
        "gradient_max_abs": grad_max,
        "weight_update_l2_norm": update_l2,
        "grads_ref_bin": str(grads_path),
        "weights_before_ref_bin": str(weights_before_path),
        "weights_after_ref_bin": str(weights_after_path),
    }
    summary_json = root / "training_dataset_reference.json"
    summary_txt = root / "training_dataset_reference.txt"
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary_txt.write_text(
        "\n".join([
            "FPGAI dataset-wide training reference",
            f"samples                    : {inputs.shape[0]}",
            f"optimizer updates          : 1",
            f"initial dataset loss       : {initial_loss:.9g}",
            f"final dataset loss         : {final_loss:.9g}",
            f"loss direction             : {direction}",
            f"gradient L2 norm           : {grad_l2:.9g}",
            f"weight update L2 norm      : {update_l2:.9g}",
            "convergence claim          : not_evaluated",
        ]) + "\n",
        encoding="utf-8",
    )

    with (root / "training_loss_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["phase", "sample_index", "loss"])
        for index, value in enumerate(losses_before):
            writer.writerow(["before_update", index, value])
        for index, value in enumerate(losses_after):
            writer.writerow(["after_update", index, value])

    return TrainingReferenceResult(
        out_dir=root,
        grads_flat_path=grads_path,
        weights_before_flat_path=weights_before_path,
        weights_after_flat_path=weights_after_path,
        summary_json=summary_json,
        summary_txt=summary_txt,
        loss_before=initial_loss,
        loss_after=final_loss,
        layerwise_dir=root / "samples_before",
        optimizer_type=optimizer_type,
        optimizer_bias_correction=False,
        loss_type=str((training.get("loss", {}) or {}).get("type", "mse")),
    )
