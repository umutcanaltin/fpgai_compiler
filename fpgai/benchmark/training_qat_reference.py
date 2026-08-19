from __future__ import annotations

import copy
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fpgai.benchmark.training_reference import TrainingReferenceResult, run_training_reference_step
from fpgai.benchmark.training_dataset_reference import _assign_flat_weights, _trainable_layout
from fpgai.engine.training import resolve_training_execution_schedule, training_record_order
from fpgai.quantization import (
    ModelQATResult,
    apply_model_qat_to_hls_graph,
    model_qat_session_from_config,
    write_model_qat_report,
)
from fpgai.quantization.ptq import dequantize, fake_quantize


def _zero_lr_config(raw_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(raw_cfg)
    cfg.setdefault("training", {}).setdefault("optimizer", {})["learning_rate"] = 0.0
    return cfg


def _flatten_named_trainables(
    graph: Any,
    layout: list[tuple[str, str, str, str, tuple[int, ...], int]],
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for op_name, binding_kind, binding_key, role, shape, count in layout:
        if binding_kind != "named":
            raise ValueError(
                "QAT dataset training currently requires named trainable tensor bindings; "
                f"{op_name}:{role} is bound through {binding_kind}:{binding_key}."
            )
        source = None
        if binding_key in getattr(graph, "constants", {}):
            source = graph.constants[binding_key]
        elif binding_key in getattr(graph, "params", {}):
            source = graph.params[binding_key]
        if source is None:
            raise ValueError(f"QAT trainable tensor {binding_key!r} is not present in graph constants/params")
        array = np.asarray(source, dtype=np.float32).reshape(shape)
        if int(array.size) != int(count):
            raise ValueError(f"QAT trainable tensor {binding_key!r} size mismatch")
        chunks.append(array.reshape(-1))
    return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros((0,), dtype=np.float32)


def _fake_quant_trainables(
    *,
    session: Any,
    master_weights: np.ndarray,
    layout: list[tuple[str, str, str, str, tuple[int, ...], int]],
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    cursor = 0
    for op_name, binding_kind, binding_key, role, shape, count in layout:
        if binding_kind != "named":
            raise ValueError(f"QAT fake-quant requires named binding for {op_name}:{role}")
        master = np.asarray(master_weights[cursor:cursor + count], dtype=np.float32).reshape(shape)
        fq_role = "bias" if role in {"bias", "beta"} else "weight"
        chunks.append(session.fake_quant_weight(binding_key, master, role=fq_role).reshape(-1))
        cursor += count
    if cursor != int(master_weights.size):
        raise RuntimeError(f"QAT parameter layout consumed {cursor} values from {master_weights.size}")
    return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros((0,), dtype=np.float32)


def _apply_optimizer(
    *,
    weights: np.ndarray,
    gradient: np.ndarray,
    state: np.ndarray,
    optimizer_type: str,
    learning_rate: float,
    momentum: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    bias_correction: bool,
    step: int,
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(weights, dtype=np.float32)
    gradient = np.asarray(gradient, dtype=np.float32)
    state = np.asarray(state, dtype=np.float32)
    if optimizer_type == "momentum":
        velocity = (np.float32(momentum) * state - np.float32(learning_rate) * gradient).astype(np.float32)
        return (weights + velocity).astype(np.float32), velocity
    if optimizer_type == "adam":
        n = weights.size
        m = state[:n]
        v = state[n:]
        m = (np.float32(beta1) * m + np.float32(1.0 - beta1) * gradient).astype(np.float32)
        v = (np.float32(beta2) * v + np.float32(1.0 - beta2) * gradient * gradient).astype(np.float32)
        if bias_correction:
            m_used = m / np.float32(max(1.0e-12, 1.0 - beta1 ** step))
            v_used = v / np.float32(max(1.0e-12, 1.0 - beta2 ** step))
        else:
            m_used, v_used = m, v
        updated = (weights - np.float32(learning_rate) * m_used / (np.sqrt(v_used) + np.float32(epsilon))).astype(np.float32)
        return updated, np.concatenate([m, v]).astype(np.float32)
    return (weights - np.float32(learning_rate) * gradient).astype(np.float32), state


def _write_curve(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "optimizer_update", "epoch", "batch", "records", "mean_batch_loss",
        "gradient_l2_norm", "weight_update_l2_norm", "observers_frozen",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@dataclass(frozen=True)
class QATTrainingReferenceResult:
    out_dir: Path
    summary_json: Path
    curve_csv: Path
    master_weights_before_path: Path
    master_weights_after_path: Path
    last_gradient_path: Path
    qat_report_path: Path
    qat_result: ModelQATResult
    optimizer_updates: int
    final_dataset_loss: float
    common_hls_lowering_status: str
    trained_graph: Any


def execute_frozen_qat_reference(
    *,
    graph: Any,
    qat_result: ModelQATResult,
    raw_cfg: dict[str, Any],
    out_dir: Path,
    x_input: np.ndarray,
) -> np.ndarray:
    """Execute one frozen fake-quant QAT forward pass on trained master weights.

    The evaluator reconstructs the fake-quantized constant view from the frozen
    exported QAT parameters and injects the same frozen activation fake-quant
    boundaries through the canonical training reference.  This provides a
    backend-independent software reference for comparing QAT against the
    exported integer hardware semantics.
    """
    eval_graph = copy.deepcopy(graph)
    parameter_by_tensor = {
        entry.tensor: entry.parameters
        for entry in (*qat_result.weights, *qat_result.biases)
    }
    for name, qvalues in qat_result.quantized_constants.items():
        params = parameter_by_tensor.get(str(name))
        if params is None:
            raise ValueError(f"QAT constant {name!r} has no frozen quantization parameters")
        eval_graph.constants[str(name)] = dequantize(np.asarray(qvalues), params).astype(np.float32)

    activation_parameters = {entry.tensor: entry.parameters for entry in qat_result.activations}

    def _activation_transform(name: str, values: np.ndarray) -> np.ndarray:
        params = activation_parameters.get(str(name))
        if params is None:
            return np.asarray(values, dtype=np.float32)
        return fake_quantize(np.asarray(values, dtype=np.float32), params)

    output_shape = tuple(int(v) for v in eval_graph.get_tensor(eval_graph.outputs[0]).shape)
    target = np.zeros(output_shape, dtype=np.float32)
    result = run_training_reference_step(
        graph=eval_graph,
        raw_cfg=_zero_lr_config(raw_cfg),
        out_dir=Path(out_dir),
        x_input=np.asarray(x_input, dtype=np.float32).reshape(-1),
        target=target.reshape(-1),
        activation_transform=_activation_transform,
        gradient_transform=lambda _name, grad: np.asarray(grad, dtype=np.float32),
    )
    if not getattr(eval_graph, "ops", None):
        raise ValueError("QAT frozen reference requires a non-empty graph")
    output_path = result.layerwise_dir / f"{eval_graph.ops[-1].name}__fwd.bin"
    if not output_path.exists():
        raise RuntimeError(f"QAT frozen reference output was not emitted: {output_path}")
    return np.fromfile(output_path, dtype=np.float32).reshape(output_shape)


def run_qat_training_dataset_reference(
    *,
    graph: Any,
    raw_cfg: dict[str, Any],
    out_dir: Path,
    inputs: np.ndarray,
    targets: np.ndarray,
) -> QATTrainingReferenceResult:
    """Execute multi-update QAT through FPGAI's existing training reference.

    Master parameters and optimizer state stay floating point.  Each sample is
    evaluated on a fake-quantized graph view, activation fake-quant boundaries
    are injected through the canonical training reference, gradients cross those
    boundaries through the QAT STE contract, and optimizer updates are applied
    to the master parameter vector only.
    """
    inputs = np.asarray(inputs, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    if inputs.ndim < 2 or targets.ndim < 2 or inputs.shape[0] != targets.shape[0]:
        raise ValueError("QAT dataset inputs/targets must have equal non-empty sample dimensions")

    root = Path(out_dir) / "training_qat_reference"
    root.mkdir(parents=True, exist_ok=True)
    session = model_qat_session_from_config(raw_cfg)
    master_graph = copy.deepcopy(graph)
    layout = _trainable_layout(master_graph)
    master_before = _flatten_named_trainables(master_graph, layout)
    master_weights = master_before.copy()

    training = raw_cfg.get("training", {}) or {}
    optimizer = training.get("optimizer", {}) or {}
    optimizer_type = str(optimizer.get("type", "sgd")).strip().lower().replace("-", "_")
    if optimizer_type not in {"sgd", "momentum", "adam"}:
        raise ValueError(f"QAT training supports SGD, Momentum, and Adam; got {optimizer_type!r}")
    learning_rate = float(optimizer.get("learning_rate", 0.01))
    momentum = float(optimizer.get("momentum", 0.9))
    beta1 = float(optimizer.get("beta1", 0.9))
    beta2 = float(optimizer.get("beta2", 0.999))
    epsilon = float(optimizer.get("epsilon", 1.0e-8))
    bias_correction = bool(optimizer.get("bias_correction", False))
    schedule = resolve_training_execution_schedule(raw_cfg, sample_count=int(inputs.shape[0]))
    if schedule.batch_mode not in {"accumulate", "accumulated", "true_minibatch", "mini_batch", "minibatch"}:
        raise ValueError(f"QAT dataset training requires accumulated mini-batches; got {schedule.batch_mode!r}")
    total_updates = int(schedule.total_optimizer_updates or 0)
    batches_per_epoch = int(schedule.batches_per_epoch or 0)
    if total_updates <= 0 or batches_per_epoch <= 0:
        raise ValueError("QAT training schedule resolves to zero optimizer updates")

    state_words = master_weights.size * (2 if optimizer_type == "adam" else 1)
    optimizer_state = np.zeros((state_words,), dtype=np.float32)
    zero_cfg = _zero_lr_config(raw_cfg)
    rows: list[dict[str, Any]] = []
    last_gradient = np.zeros_like(master_weights)

    for update_index in range(total_updates):
        epoch_index, batch_index = divmod(update_index, batches_per_epoch)
        order = training_record_order(
            int(schedule.sample_count), epoch_index=epoch_index,
            shuffle=bool(schedule.shuffle), seed=int(schedule.seed),
        )
        start = batch_index * int(schedule.batch_size)
        stop = min(start + int(schedule.batch_size), int(schedule.sample_count))
        record_indices = order[start:stop]
        if schedule.drop_last and len(record_indices) < int(schedule.batch_size):
            continue
        if not record_indices:
            raise RuntimeError("QAT training produced an empty optimizer batch")

        fake_weights = _fake_quant_trainables(session=session, master_weights=master_weights, layout=layout)
        sample_gradients: list[np.ndarray] = []
        sample_losses: list[float] = []
        batch_root = root / "updates" / f"update_{update_index + 1:04d}"
        for slot, record_index in enumerate(record_indices):
            step_graph = copy.deepcopy(master_graph)
            _assign_flat_weights(step_graph, fake_weights, layout)
            result = run_training_reference_step(
                graph=step_graph,
                raw_cfg=zero_cfg,
                out_dir=batch_root / f"slot_{slot:04d}_record_{record_index:04d}",
                x_input=np.asarray(inputs[record_index], dtype=np.float32).reshape(-1),
                target=np.asarray(targets[record_index], dtype=np.float32).reshape(-1),
                activation_transform=session.fake_quant_activation,
                gradient_transform=lambda _name, grad: session.backward_gradient(grad),
            )
            observed_weights = np.fromfile(result.weights_before_flat_path, dtype=np.float32)
            if not np.array_equal(observed_weights, fake_weights):
                raise RuntimeError("QAT forward/backward did not execute from the fake-quantized parameter view")
            grad = np.fromfile(result.grads_flat_path, dtype=np.float32)
            sample_gradients.append(session.backward_gradient(grad))
            sample_losses.append(float(result.loss_before))

        last_gradient = np.mean(np.stack(sample_gradients, axis=0), axis=0, dtype=np.float32).astype(np.float32)
        before = master_weights.copy()
        master_weights, optimizer_state = _apply_optimizer(
            weights=master_weights,
            gradient=last_gradient,
            state=optimizer_state,
            optimizer_type=optimizer_type,
            learning_rate=learning_rate,
            momentum=momentum,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            bias_correction=bias_correction,
            step=update_index + 1,
        )
        _assign_flat_weights(master_graph, master_weights, layout)
        session.complete_optimizer_update()
        rows.append({
            "optimizer_update": update_index + 1,
            "epoch": epoch_index + 1,
            "batch": batch_index + 1,
            "records": len(record_indices),
            "mean_batch_loss": float(np.mean(sample_losses)),
            "gradient_l2_norm": float(np.linalg.norm(last_gradient)),
            "weight_update_l2_norm": float(np.linalg.norm(master_weights - before)),
            "observers_frozen": bool(session.observers_frozen),
        })

    # Export freezes any still-live observers and materializes the standard FPGAI
    # tensor quantization metadata consumed by PTQ/QAT-common hardware lowering.
    qat_result = session.export_to_graph(master_graph)
    qat_report_path = write_model_qat_report(qat_result, root / "model_qat_export.json")

    # Evaluate the trained master weights using frozen fake-quant parameters.
    final_losses: list[float] = []
    frozen_fake_weights = _fake_quant_trainables(session=session, master_weights=master_weights, layout=layout)
    eval_root = root / "final_frozen_evaluation"
    for record_index in range(int(inputs.shape[0])):
        eval_graph = copy.deepcopy(master_graph)
        _assign_flat_weights(eval_graph, frozen_fake_weights, layout)
        result = run_training_reference_step(
            graph=eval_graph,
            raw_cfg=zero_cfg,
            out_dir=eval_root / f"record_{record_index:04d}",
            x_input=np.asarray(inputs[record_index], dtype=np.float32).reshape(-1),
            target=np.asarray(targets[record_index], dtype=np.float32).reshape(-1),
            activation_transform=session.fake_quant_activation,
            gradient_transform=lambda _name, grad: session.backward_gradient(grad),
        )
        final_losses.append(float(result.loss_before))

    lowering_status = "passed"
    lowering_summary: dict[str, Any]
    try:
        hardware_graph = copy.deepcopy(master_graph)
        lowering = apply_model_qat_to_hls_graph(hardware_graph, qat_result)
        lowering_summary = {
            "status": "passed",
            "quantized_conv_nodes": list(lowering.quantized_conv_nodes),
            "quantized_relu_nodes": list(lowering.quantized_relu_nodes),
            "quantized_add_nodes": list(lowering.quantized_add_nodes),
        }
    except (ValueError, RuntimeError) as exc:
        lowering_status = "unsupported"
        lowering_summary = {"status": "unsupported", "reason": str(exc)}

    before_path = root / "master_weights_before.bin"
    after_path = root / "master_weights_after.bin"
    grad_path = root / "last_gradient.bin"
    master_before.astype(np.float32).tofile(before_path)
    master_weights.astype(np.float32).tofile(after_path)
    last_gradient.astype(np.float32).tofile(grad_path)
    curve_path = root / "qat_training_curve.csv"
    _write_curve(curve_path, rows)
    summary = {
        "schema": "fpgai.qat-training-reference/v1",
        "status": "passed",
        "optimizer": {
            "type": optimizer_type,
            "learning_rate": learning_rate,
            "momentum": momentum if optimizer_type == "momentum" else None,
            "beta1": beta1 if optimizer_type == "adam" else None,
            "beta2": beta2 if optimizer_type == "adam" else None,
            "epsilon": epsilon if optimizer_type == "adam" else None,
            "bias_correction": bias_correction if optimizer_type == "adam" else False,
        },
        "execution_schedule": schedule.to_dict(),
        "optimizer_updates": total_updates,
        "observer_freeze_after_updates": session.schedule.freeze_after_updates,
        "observers_frozen": session.observers_frozen,
        "first_batch_loss": rows[0]["mean_batch_loss"],
        "last_training_batch_loss": rows[-1]["mean_batch_loss"],
        "final_frozen_dataset_loss": float(np.mean(final_losses)),
        "master_weight_update_l2_norm": float(np.linalg.norm(master_weights - master_before)),
        "last_gradient_l2_norm": float(np.linalg.norm(last_gradient)),
        "master_weights_before_bin": str(before_path),
        "master_weights_after_bin": str(after_path),
        "last_gradient_bin": str(grad_path),
        "qat_training_curve_csv": str(curve_path),
        "qat_export_json": str(qat_report_path),
        "common_hls_lowering": lowering_summary,
        "master_weight_policy": "float_master_weights_fake_quant_forward_ste_backward",
    }
    summary_path = root / "qat_training_reference.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return QATTrainingReferenceResult(
        out_dir=root,
        summary_json=summary_path,
        curve_csv=curve_path,
        master_weights_before_path=before_path,
        master_weights_after_path=after_path,
        last_gradient_path=grad_path,
        qat_report_path=qat_report_path,
        qat_result=qat_result,
        optimizer_updates=total_updates,
        final_dataset_loss=float(np.mean(final_losses)),
        common_hls_lowering_status=lowering_status,
        trained_graph=master_graph,
    )
