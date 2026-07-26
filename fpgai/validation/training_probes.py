"""Backend-neutral intermediate probes for training numerical validation."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROBE_SCHEMA_VERSION = 1
SUPPORTED_PROBE_STAGES = (
    "forward_input",
    "dense_forward_output",
    "activation_forward_output",
    "activation_upstream_gradient",
    "activation_backward_output",
    "backward_output_gradient",
    "parameter_gradient_term",
    "parameter_gradient_accumulated",
    "optimizer_m",
    "optimizer_v",
    "optimizer_delta",
    "parameter_before",
    "parameter_after",
)

ACTIVATION_BOUNDARY_STAGES = (
    "dense_forward_output",
    "activation_forward_output",
    "activation_upstream_gradient",
    "activation_backward_output",
)


def normalize_probe_config(raw_config: Mapping[str, Any] | None) -> dict[str, Any]:
    validation = (raw_config or {}).get("validation", {}) if isinstance(raw_config, Mapping) else {}
    numeric = validation.get("numeric", {}) if isinstance(validation, Mapping) else {}
    probes = numeric.get("probes", {}) if isinstance(numeric, Mapping) else {}
    if not isinstance(probes, Mapping) or not bool(probes.get("enabled", False)):
        return {"enabled": False, "selectors": [], "stages": []}
    selectors = probes.get("selectors", [])
    stages = [str(stage) for stage in probes.get("stages", list(SUPPORTED_PROBE_STAGES))]
    # Selected Dense probes always include the owning activation boundary.
    # Without these stages, a zero post-activation gradient cannot be
    # distinguished from a Dense-backward defect or a fixed-point branch
    # change. Preserve user-requested stages and append the required diagnostic
    # stages deterministically.
    for stage in ACTIVATION_BOUNDARY_STAGES:
        if stage not in stages:
            stages.append(stage)
    return {
        "enabled": True,
        "selectors": [dict(item) for item in selectors if isinstance(item, Mapping)],
        "stages": stages,
    }


def _read_f32(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32)


def _scalar(value: float) -> float:
    return float(np.float32(value))


def materialize_python_dense_probes(*, graph: Any, result: Any, probe_config: Mapping[str, Any], parameter_layout: Sequence[Mapping[str, Any]], output_path: str | Path) -> Path | None:
    if not probe_config.get("enabled"):
        return None
    entries: list[dict[str, Any]] = []
    stages = set(probe_config.get("stages", ()))
    layout_by_name = {str(item["name"]): item for item in parameter_layout}
    ops = {str(op.name): op for op in graph.ops}
    summary = json.loads(Path(result.summary_json).read_text(encoding="utf-8"))
    optimizer = str(getattr(result, "optimizer_type", "sgd")).lower()
    lr = float(summary.get("learning_rate", 0.01))
    beta1 = float(summary.get("beta1", 0.9)); beta2 = float(summary.get("beta2", 0.999)); eps = float(summary.get("epsilon", 1e-8))
    for selector in probe_config.get("selectors", []):
        operator = str(selector.get("operator", "")); parameter = str(selector.get("parameter", "weight"))
        index = selector.get("tensor_index", [])
        op = ops.get(operator)
        if op is None or op.op_type != "Dense" or parameter != "weight" or not isinstance(index, list) or len(index) != 2:
            continue
        row, col = int(index[0]), int(index[1]); layout = layout_by_name.get(f"{operator}.weight")
        if layout is None: continue
        shape = list(layout.get("shape", []))
        if len(shape) != 2 or row < 0 or col < 0 or row >= int(shape[0]) or col >= int(shape[1]): continue
        local = row * int(shape[1]) + col; flat = int(layout["offset"]) + local
        layer_dir = Path(result.layerwise_dir)
        input_path = layer_dir / f"{operator}__fwd_input.bin"
        out_grad_path = layer_dir / f"{operator}__bwd_output_grad.bin"
        grad_path = layer_dir / f"{operator}__param_grad_w.bin"
        before = _read_f32(Path(result.weights_before_flat_path)); after = _read_f32(Path(result.weights_after_flat_path)); grads = _read_f32(Path(result.grads_flat_path))
        activation = _scalar(_read_f32(input_path)[col]) if input_path.exists() else None
        output_grad = _scalar(_read_f32(out_grad_path)[row]) if out_grad_path.exists() else None
        gradient = _scalar(_read_f32(grad_path)[local]) if grad_path.exists() else _scalar(grads[flat])
        m = v = None
        state_path = getattr(result, "optimizer_state_after_flat_path", None)
        if optimizer == "adam" and state_path and Path(state_path).exists():
            state = _read_f32(Path(state_path)); total = len(before); m = _scalar(state[flat]); v = _scalar(state[total + flat])
        dense_forward_output = None
        activation_forward_output = None
        activation_upstream_gradient = None
        activation_backward_output = output_grad
        dense_output_path = layer_dir / f"{operator}__fwd.bin"
        if dense_output_path.exists():
            dense_values = _read_f32(dense_output_path)
            if row < len(dense_values):
                dense_forward_output = _scalar(dense_values[row])

        selected_output_name = str(op.outputs[0]) if getattr(op, "outputs", None) else ""
        activation_op = next((candidate for candidate in graph.ops
                              if candidate.op_type in {"Relu", "LeakyRelu", "Sigmoid"}
                              and getattr(candidate, "inputs", None)
                              and str(candidate.inputs[0]) == selected_output_name), None)
        if activation_op is not None:
            activation_forward_path = layer_dir / f"{activation_op.name}__fwd.bin"
            activation_upstream_path = layer_dir / f"{activation_op.name}__bwd_output_grad.bin"
            if activation_forward_path.exists():
                arr = _read_f32(activation_forward_path)
                if row < len(arr):
                    activation_forward_output = _scalar(arr[row])
            if activation_upstream_path.exists():
                arr = _read_f32(activation_upstream_path)
                if row < len(arr):
                    activation_upstream_gradient = _scalar(arr[row])

        values = {
            "forward_input": activation,
            "backward_output_gradient": output_grad,
            "parameter_gradient_term": None if activation is None or output_grad is None else _scalar(activation * output_grad),
            "parameter_gradient_accumulated": gradient,
            "optimizer_m": m, "optimizer_v": v,
            "optimizer_delta": _scalar(after[flat] - before[flat]),
            "parameter_before": _scalar(before[flat]), "parameter_after": _scalar(after[flat]),
            "dense_forward_output": dense_forward_output,
            "activation_forward_output": activation_forward_output,
            "activation_upstream_gradient": activation_upstream_gradient,
            "activation_backward_output": activation_backward_output,
        }
        for stage in SUPPORTED_PROBE_STAGES:
            if stage not in stages: continue
            value = values[stage]
            entries.append({"stage": stage, "operator": operator, "parameter": f"{operator}.weight", "tensor_index": [row, col], "flat_index": flat, "value": value, "status": "captured" if value is not None else "unavailable", "producer": "python_reference"})
    path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"artifact_kind":"fpgai_training_probe_capture","schema_version":PROBE_SCHEMA_VERSION,"producer":"python_reference","entries":entries}, indent=2)+"\n", encoding="utf-8")
    return path



def materialize_hardware_fixed_point_dense_probes(
    *,
    graph: Any,
    result: Any,
    raw_config: Mapping[str, Any],
    probe_config: Mapping[str, Any],
    parameter_layout: Sequence[Mapping[str, Any]],
    output_path: str | Path,
) -> Path | None:
    """Materialize the operation-level fixed-point reference for HLS probes.

    HLS CSim must be compared against the configured fixed-point arithmetic
    domain.  The float32 training reference is retained as a diagnostic, but it
    is not the numerical-equivalence owner because quantization can legitimately
    change activation branches such as ReLU.
    """
    if not probe_config.get("enabled"):
        return None

    input_path = getattr(result, "input_ref_path", None)
    target_path = getattr(result, "target_ref_path", None)
    if input_path is None or target_path is None or not Path(input_path).exists() or not Path(target_path).exists():
        raise RuntimeError("Hardware fixed-point probe reference requires persisted input_ref.bin and target_ref.bin artifacts.")

    from fpgai.benchmark.training_dataset_reference import (
        _assign_flat_weights,
        _hardware_batch_update,
        _parameter_specs_for_layout,
        _quantize_parameter_vector,
        _run_hls_numeric_training_sample,
        _trainable_layout,
    )

    hardware_graph = copy.deepcopy(graph)
    internal_layout = _trainable_layout(hardware_graph)
    weights_before_float = _read_f32(Path(result.weights_before_flat_path))
    weight_spec, bias_spec, _accum_spec, _grad_weight_spec, _grad_bias_spec, _update_spec = (
        _parameter_specs_for_layout(dict(raw_config))
    )
    weights_before = _quantize_parameter_vector(
        weights_before_float,
        internal_layout,
        weight_spec=weight_spec,
        bias_spec=bias_spec,
    )
    _assign_flat_weights(hardware_graph, weights_before, internal_layout)

    x_input = _read_f32(Path(input_path))
    target = _read_f32(Path(target_path))
    _gradient, _loss, _prediction, trace = _run_hls_numeric_training_sample(
        graph=hardware_graph,
        raw_cfg=dict(raw_config),
        x_input=x_input,
        target=target,
        layout=internal_layout,
        return_trace=True,
    )

    summary = json.loads(Path(result.summary_json).read_text(encoding="utf-8"))
    optimizer_cfg = ((raw_config.get("training", {}) or {}).get("optimizer", {}) or {})
    optimizer_type = str(optimizer_cfg.get("type", "sgd")).strip().lower().replace("-", "_")
    update = _hardware_batch_update(
        graph=hardware_graph,
        raw_cfg=dict(raw_config),
        inputs=x_input.reshape(1, -1),
        targets=target.reshape(1, -1),
        record_indices=[0],
        current_weights=weights_before,
        layout=internal_layout,
        learning_rate=float(optimizer_cfg.get("learning_rate", summary.get("learning_rate", 0.01))),
        optimizer_type=optimizer_type,
        momentum=float(optimizer_cfg.get("momentum", 0.9)),
        beta1=float(optimizer_cfg.get("beta1", 0.9)),
        beta2=float(optimizer_cfg.get("beta2", 0.999)),
        epsilon=float(optimizer_cfg.get("epsilon", 1.0e-8)),
        bias_correction=bool(optimizer_cfg.get("bias_correction", False)),
        optimizer_step=1,
    )

    layout_by_name = {str(item["name"]): item for item in parameter_layout}
    ops = {str(op.name): op for op in graph.ops}
    stages = set(probe_config.get("stages", ()))
    entries: list[dict[str, Any]] = []

    for selector in probe_config.get("selectors", []):
        operator = str(selector.get("operator", ""))
        index = selector.get("tensor_index", [])
        op = ops.get(operator)
        layout = layout_by_name.get(f"{operator}.weight")
        if op is None or layout is None or not isinstance(index, list) or len(index) != 2:
            continue
        row, col = int(index[0]), int(index[1])
        shape = list(layout.get("shape", []))
        local = row * int(shape[1]) + col
        flat = int(layout["offset"]) + local

        activation_op = next((candidate for candidate in graph.ops
                              if candidate.op_type in {"Relu", "LeakyRelu", "Sigmoid"}
                              and getattr(candidate, "inputs", None)
                              and str(candidate.inputs[0]) == str(op.outputs[0])), None)

        forward_input = _scalar(trace["forward_inputs"][operator][col])
        dense_forward_output = _scalar(trace["forward_outputs"][operator][row])
        backward_output_gradient = _scalar(trace["backward_output_gradients"][operator][row])
        parameter_gradient = _scalar(trace["parameter_gradients"][f"{operator}.weight"][local])
        activation_forward_output = activation_upstream_gradient = activation_backward_output = None
        if activation_op is not None:
            activation_forward_output = _scalar(trace["forward_outputs"][activation_op.name][row])
            activation_upstream_gradient = _scalar(trace["backward_output_gradients"][activation_op.name][row])
            activation_backward_output = _scalar(trace["backward_input_gradients"][activation_op.name][row])

        state = np.asarray(update.get("optimizer_state_after", []), dtype=np.float32).reshape(-1)
        total = int(weights_before.size)
        m = v = None
        if optimizer_type == "adam" and state.size >= total * 2:
            m = _scalar(state[flat])
            v = _scalar(state[total + flat])
        elif optimizer_type == "momentum" and state.size >= total:
            m = _scalar(state[flat])

        weights_after = np.asarray(update["weights_after"], dtype=np.float32).reshape(-1)
        values = {
            "forward_input": forward_input,
            "dense_forward_output": dense_forward_output,
            "activation_forward_output": activation_forward_output,
            "activation_upstream_gradient": activation_upstream_gradient,
            "activation_backward_output": activation_backward_output,
            "backward_output_gradient": backward_output_gradient,
            "parameter_gradient_term": parameter_gradient,
            "parameter_gradient_accumulated": _scalar(np.asarray(update["gradient"], dtype=np.float32)[flat]),
            "optimizer_m": m,
            "optimizer_v": v,
            "optimizer_delta": _scalar(weights_after[flat] - weights_before[flat]),
            "parameter_before": _scalar(weights_before[flat]),
            "parameter_after": _scalar(weights_after[flat]),
        }
        for stage in SUPPORTED_PROBE_STAGES:
            if stage not in stages:
                continue
            value = values[stage]
            entries.append({
                "stage": stage,
                "operator": operator,
                "parameter": f"{operator}.weight",
                "tensor_index": [row, col],
                "flat_index": flat,
                "value": value,
                "status": "captured" if value is not None else "unavailable",
                "producer": "hardware_fixed_point_reference",
            })

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "artifact_kind": "fpgai_training_probe_capture",
        "schema_version": PROBE_SCHEMA_VERSION,
        "producer": "hardware_fixed_point_reference",
        "reference_domain": "hardware_fixed_point",
        "entries": entries,
    }, indent=2) + "\n", encoding="utf-8")
    return path

def materialize_hls_observable_probes(*, hls_artifact_dir: str | Path, probe_config: Mapping[str, Any], parameter_layout: Sequence[Mapping[str, Any]], output_path: str | Path) -> Path | None:
    if not probe_config.get("enabled"):
        return None
    root = Path(hls_artifact_dir)
    entries: list[dict[str, Any]] = []
    layout_by_name = {str(item["name"]): item for item in parameter_layout}

    probe_path = next((path for path in root.rglob("training_probe_values.bin") if path.is_file()), None)
    probe_values = _read_f32(probe_path) if probe_path is not None else None
    stage_to_probe_index = {
        "forward_input": 0,
        "backward_output_gradient": 1,
        "parameter_gradient_term": 2,
        "parameter_gradient_accumulated": 3,
        "optimizer_m": 4,
        "optimizer_v": 5,
        "optimizer_delta": 6,
        "parameter_before": 7,
        "parameter_after": 8,
        "dense_forward_output": 12,
        "activation_forward_output": 13,
        "activation_upstream_gradient": 14,
        "activation_backward_output": 15,
    }
    probe_execution = {
        "loop_entered": bool(probe_values is not None and len(probe_values) >= 10 and probe_values[9] == np.float32(1.0)),
        "selected_index_hit": bool(probe_values is not None and len(probe_values) >= 11 and probe_values[10] == np.float32(1.0)),
        "capture_completed": bool(probe_values is not None and len(probe_values) >= 12 and probe_values[11] == np.float32(1.0)),
    }
    probe_capture_valid = all(probe_execution.values())

    gradients_path = next((path for path in root.rglob("gradients_after.bin") if path.is_file()), None)
    gradients = _read_f32(gradients_path) if gradients_path is not None else None

    for selector in probe_config.get("selectors", []):
        operator = str(selector.get("operator", ""))
        index = selector.get("tensor_index", [])
        layout = layout_by_name.get(f"{operator}.weight")
        if layout is None or not isinstance(index, list) or len(index) != 2:
            continue
        shape = list(layout.get("shape", []))
        row, col = int(index[0]), int(index[1])
        flat = int(layout["offset"]) + row * int(shape[1]) + col
        for stage in probe_config.get("stages", []):
            value = None
            probe_index = stage_to_probe_index.get(str(stage))
            if probe_capture_valid and probe_values is not None and probe_index is not None and probe_index < len(probe_values):
                value = _scalar(probe_values[probe_index])
            elif stage == "parameter_gradient_accumulated" and gradients is not None and flat < len(gradients):
                value = _scalar(gradients[flat])
            entries.append({
                "stage": stage,
                "operator": operator,
                "parameter": f"{operator}.weight",
                "tensor_index": [row, col],
                "flat_index": flat,
                "value": value,
                "status": "captured" if value is not None else "unavailable",
                "producer": "hls_csim",
            })
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "artifact_kind": "fpgai_training_probe_capture",
        "schema_version": PROBE_SCHEMA_VERSION,
        "producer": "hls_csim",
        "source_artifact": str(probe_path) if probe_path is not None else None,
        "execution": probe_execution,
        "capture_valid": probe_capture_valid,
        "entries": entries,
    }, indent=2) + "\n", encoding="utf-8")
    return path


def compare_training_probes(reference_path: str | Path, candidate_path: str | Path, output_path: str | Path, *, atol: float=1e-5, rtol: float=1e-4) -> Path:
    ref=json.loads(Path(reference_path).read_text()); got=json.loads(Path(candidate_path).read_text())
    key=lambda e:(e.get("stage"),e.get("operator"),e.get("parameter"),tuple(e.get("tensor_index",[])))
    rm={key(e):e for e in ref.get("entries",[])}; gm={key(e):e for e in got.get("entries",[])}; comparisons=[]; first=None
    for k in rm:
        a=rm[k]; b=gm.get(k); comparable=bool(b and a.get("status")=="captured" and b.get("status")=="captured")
        passed=None; abs_error=None
        if comparable:
            av=float(a["value"]); bv=float(b["value"]); abs_error=abs(av-bv); passed=abs_error <= atol + rtol*abs(av)
        item={"stage":k[0],"operator":k[1],"parameter":k[2],"tensor_index":list(k[3]),"comparable":comparable,"passed":passed,"reference_value":a.get("value"),"candidate_value":None if b is None else b.get("value"),"abs_error":abs_error}
        comparisons.append(item)
        if first is None and comparable and passed is False: first=item
    payload={"artifact_kind":"fpgai_training_probe_comparison","schema_version":1,"status":"diverged" if first else ("passed" if any(x["comparable"] for x in comparisons) else "incomplete"),"first_divergence":first,"comparisons":comparisons}
    out=Path(output_path); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8"); return out
