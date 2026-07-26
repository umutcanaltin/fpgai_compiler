"""Adapters that publish existing FPGAI numeric artifacts into capture contracts."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from fpgai.validation.capture_schema import NumericCaptureContract, write_capture_contract


def _ready(path: Path | None) -> dict[str, Any]:
    return {
        "path": str(path) if path is not None and path.exists() else None,
        "status": "captured" if path is not None and path.exists() else "missing",
    }


def _write_scalar(path: Path, value: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray([value], dtype=np.float32).tofile(path)
    return path


def publish_python_training_reference(
    *,
    result: Any,
    output_path: str | Path,
    workload_fingerprint_sha256: str,
    implementation_stack_fingerprint_sha256: str,
) -> Path:
    """Publish an existing ``TrainingReferenceResult`` without rerunning it."""
    root = Path(result.out_dir)
    loss_before = _write_scalar(root / "loss_before_ref.bin", float(result.loss_before))
    loss_after = _write_scalar(root / "loss_after_ref.bin", float(result.loss_after))
    step_before = _write_scalar(root / "optimizer_step_before_ref.bin", 0.0)
    step_after = _write_scalar(
        root / "optimizer_step_after_ref.bin",
        1.0 if str(getattr(result, "optimizer_type", "sgd")).lower() == "adam" else 0.0,
    )
    parameter_bundle = Path(result.weights_after_flat_path)
    captures = {
        "pre_update_loss": {**_ready(loss_before), "dtype": "float32", "layout": "scalar", "required": True},
        "post_update_loss": {**_ready(loss_after), "dtype": "float32", "layout": "scalar", "required": True},
        "parameter_gradients": {**_ready(Path(result.grads_flat_path)), "dtype": "float32", "layout": "flat_canonical_parameter_order", "required": True},
        "weights_after": {**_ready(parameter_bundle), "dtype": "float32", "layout": "canonical_parameter_bundle", "required": True, "tensor_map": {"select": "weight_like"}},
        "biases_after": {**_ready(parameter_bundle), "dtype": "float32", "layout": "canonical_parameter_bundle", "required": True, "tensor_map": {"select": "bias_like"}},
        "optimizer_step_before": {**_ready(step_before), "dtype": "float32", "layout": "scalar", "required": False},
        "optimizer_step_after": {**_ready(step_after), "dtype": "float32", "layout": "scalar", "required": str(getattr(result, "optimizer_type", "sgd")).lower() == "adam"},
    }
    state_after = getattr(result, "optimizer_state_after_flat_path", None)
    if state_after is not None:
        packed = Path(state_after)
        captures["optimizer_m_after"] = {**_ready(packed), "dtype": "float32", "layout": "packed_optimizer_state", "required": True, "tensor_map": {"component": "m"}}
        captures["optimizer_v_after"] = {**_ready(packed), "dtype": "float32", "layout": "packed_optimizer_state", "required": str(getattr(result, "optimizer_type", "sgd")).lower() == "adam", "tensor_map": {"component": "v"}}
    contract = NumericCaptureContract(
        workload_fingerprint_sha256=workload_fingerprint_sha256,
        implementation_stack_fingerprint_sha256=implementation_stack_fingerprint_sha256,
        producer_kind="python_reference",
        producer_id="fpgai.python.training_reference",
        captures=captures,
        metadata={"source_summary_json": str(result.summary_json), "adapter": "publish_python_training_reference"},
    )
    return write_capture_contract(output_path, contract)


def _first_named_recursive(root: Path, names: Iterable[str]) -> Path | None:
    for name in names:
        direct = root / name
        if direct.exists():
            return direct
    for name in names:
        matches = sorted(root.rglob(name)) if root.exists() else []
        if matches:
            return matches[0]
    return None


def _loss_endpoints(csv_path: Path) -> tuple[float, float] | None:
    if not csv_path.exists():
        return None
    values: list[float] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for key in ("loss", "dataset_loss", "average_loss"):
                if row.get(key) not in (None, ""):
                    values.append(float(row[key]))
                    break
    return (values[0], values[-1]) if values else None


def bind_hls_training_capture(
    *,
    artifact_dir: str | Path,
    output_path: str | Path,
    workload_fingerprint_sha256: str,
    implementation_stack_fingerprint_sha256: str,
    optimizer_type: str,
) -> Path:
    """Bind generated CSim files to semantic roles; absent files stay explicit."""
    root = Path(artifact_dir)
    loss_before = _first_named_recursive(root, ("loss_before.bin",))
    loss_after = _first_named_recursive(root, ("loss_after.bin",))
    loss_csv = _first_named_recursive(root, ("training_loss_curve.csv", "training_epoch_curve.csv", "training_batch_curve.csv", "loss_curve.csv"))
    if (loss_before is None or loss_after is None) and loss_csv is not None:
        endpoints = _loss_endpoints(loss_csv)
        if endpoints is not None:
            loss_before = loss_before or _write_scalar(root / "numeric_capture" / "loss_before.bin", endpoints[0])
            loss_after = loss_after or _write_scalar(root / "numeric_capture" / "loss_after.bin", endpoints[1])
    weights = _first_named_recursive(root, ("weights_after.bin",))
    gradients = _first_named_recursive(root, ("gradients_after.bin", "gradients_export.bin", "grads.bin"))
    state = _first_named_recursive(root, ("optimizer_state_after.bin",))
    captures = {
        "pre_update_loss": {**_ready(loss_before), "dtype": "float32", "layout": "scalar", "required": True},
        "post_update_loss": {**_ready(loss_after), "dtype": "float32", "layout": "scalar", "required": True},
        "parameter_gradients": {**_ready(gradients), "dtype": "float32", "layout": "flat_canonical_parameter_order", "required": True},
        "weights_after": {**_ready(weights), "dtype": "float32", "layout": "canonical_parameter_bundle", "required": True, "tensor_map": {"select": "weight_like"}},
        "biases_after": {**_ready(weights), "dtype": "float32", "layout": "canonical_parameter_bundle", "required": True, "tensor_map": {"select": "bias_like"}},
    }
    if str(optimizer_type).lower() == "adam":
        captures["optimizer_m_after"] = {**_ready(state), "dtype": "float32", "layout": "packed_adam_state_m_v_step", "required": True, "tensor_map": {"component": "m"}}
        captures["optimizer_v_after"] = {**_ready(state), "dtype": "float32", "layout": "packed_adam_state_m_v_step", "required": True, "tensor_map": {"component": "v"}}
        captures["optimizer_step_after"] = {**_ready(state), "dtype": "float32", "layout": "packed_adam_state_m_v_step", "required": True, "tensor_map": {"component": "step", "position": "last"}}
    contract = NumericCaptureContract(
        workload_fingerprint_sha256=workload_fingerprint_sha256,
        implementation_stack_fingerprint_sha256=implementation_stack_fingerprint_sha256,
        producer_kind="hls_csim",
        producer_id="fpgai.hls.training_testbench",
        captures=captures,
        metadata={"artifact_dir": str(root), "adapter": "bind_hls_training_capture"},
    )
    return write_capture_contract(output_path, contract)


def _tensor_index(flat_index: int, shape: Iterable[int] | None) -> list[int] | None:
    dims = [int(dim) for dim in (shape or [])]
    if not dims or any(dim <= 0 for dim in dims):
        return None
    count = int(np.prod(dims, dtype=np.int64))
    if flat_index < 0 or flat_index >= count:
        return None
    return [int(index) for index in np.unravel_index(flat_index, tuple(dims))]


def _localized_parameter_regions(capture: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return capture-local regions for canonical parameter localization.

    Role-filtered captures (weights or biases) are packed without gaps even
    though their source layout retains global parameter offsets. Therefore the
    comparison regions are intentionally rebuilt in capture-local order.
    """
    regions: list[dict[str, Any]] = []
    cursor = 0
    for raw in (capture or {}).get("parameter_layout", []) or []:
        count = int(raw.get("count", 0))
        region = {
            "name": str(raw.get("name", "parameter")),
            "layer": raw.get("layer"),
            "role": str(raw.get("role", "parameter")),
            "capture_offset": cursor,
            "canonical_offset": int(raw.get("offset", cursor)),
            "count": count,
            "shape": list(raw.get("shape", [])),
        }
        regions.append(region)
        cursor += count
    return regions


def _localize_mismatch(
    *, flat_index: int, reference_value: float, candidate_value: float,
    abs_error: float, capture: dict[str, Any] | None,
) -> dict[str, Any]:
    localized: dict[str, Any] = {
        "flat_index": int(flat_index),
        "reference_value": float(reference_value),
        "candidate_value": float(candidate_value),
        "abs_error": float(abs_error),
    }
    for region in _localized_parameter_regions(capture):
        start = int(region["capture_offset"])
        stop = start + int(region["count"])
        if start <= flat_index < stop:
            local_index = int(flat_index - start)
            localized.update({
                "parameter": region["name"],
                "layer": region["layer"],
                "parameter_role": region["role"],
                "parameter_flat_index": local_index,
                "tensor_index": _tensor_index(local_index, region.get("shape")),
                "canonical_parameter_offset": int(region["canonical_offset"]),
                "canonical_flat_index": int(region["canonical_offset"]) + local_index,
            })
            break
    return localized


def compare_flat_f32(
    reference_path: str | Path,
    candidate_path: str | Path,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-4,
    capture: dict[str, Any] | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    ref = np.fromfile(reference_path, dtype=np.float32)
    got = np.fromfile(candidate_path, dtype=np.float32)
    if ref.shape != got.shape:
        return {"status": "shape_mismatch", "reference_count": int(ref.size), "candidate_count": int(got.size), "passed": False}
    error = np.abs(ref - got)
    passed = bool(np.allclose(ref, got, atol=atol, rtol=rtol))
    report = {
        "status": "compared",
        "count": int(ref.size),
        "max_abs_error": float(error.max()) if error.size else 0.0,
        "mean_abs_error": float(error.mean()) if error.size else 0.0,
        "passed": passed,
        "atol": atol,
        "rtol": rtol,
    }
    if error.size:
        order = np.argsort(error, kind="stable")[::-1][:max(1, int(top_k))]
        localized = [
            _localize_mismatch(
                flat_index=int(index), reference_value=float(ref[index]),
                candidate_value=float(got[index]), abs_error=float(error[index]),
                capture=capture,
            )
            for index in order
        ]
        report["worst_mismatch"] = localized[0]
        report["top_mismatches"] = localized
    return report


def canonical_parameter_layout(entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate and normalize the canonical flattened parameter layout.

    Every entry names a contiguous region in the combined parameter vector.
    ``role`` is intentionally semantic (``weight`` or ``bias``), so the same
    layout can be consumed by Python, HLS, RTL, or board-runtime adapters.
    """
    normalized: list[dict[str, Any]] = []
    cursor = 0
    for index, raw in enumerate(entries):
        if not isinstance(raw, dict):
            raise TypeError(f"parameter layout entry {index} must be a mapping")
        offset = int(raw.get("offset", cursor))
        count = int(raw.get("count", 0))
        role = str(raw.get("role", "weight")).strip().lower()
        if role not in {"weight", "bias", "parameter"}:
            raise ValueError(f"unsupported parameter role {role!r}")
        if offset != cursor:
            raise ValueError(
                f"parameter layout must be contiguous: entry {index} starts at {offset}, expected {cursor}"
            )
        if count < 0:
            raise ValueError("parameter layout count must be non-negative")
        entry = {
            "name": str(raw.get("name", f"parameter_{index}")),
            "layer": raw.get("layer"),
            "role": role,
            "offset": offset,
            "count": count,
        }
        if "shape" in raw:
            entry["shape"] = list(raw["shape"])
        normalized.append(entry)
        cursor += count
    return normalized


def _select_parameter_role(values: np.ndarray, layout: list[dict[str, Any]], role: str) -> np.ndarray:
    chunks = [
        values[int(entry["offset"]): int(entry["offset"]) + int(entry["count"])]
        for entry in layout
        if entry.get("role") == role
    ]
    return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros((0,), dtype=np.float32)


def materialize_canonical_capture_files(
    *,
    manifest_path: str | Path,
    parameter_layout: Iterable[dict[str, Any]],
    optimizer_type: str,
    output_dir: str | Path | None = None,
) -> Path:
    """Materialize semantic files from bundled parameters and optimizer state."""
    manifest_file = Path(manifest_path)
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    layout = canonical_parameter_layout(parameter_layout)
    total_parameters = sum(int(entry["count"]) for entry in layout)
    root = Path(output_dir) if output_dir is not None else manifest_file.parent / "canonical_captures"
    root.mkdir(parents=True, exist_ok=True)
    captures = payload.get("captures", {})

    # Gradients already use the full canonical parameter order. Attach layout
    # metadata so mismatch localization can identify the exact layer/tensor.
    gradient_spec = captures.get("parameter_gradients", {})
    gradient_source = gradient_spec.get("path")
    if gradient_source and Path(gradient_source).exists():
        gradient_values = np.fromfile(gradient_source, dtype=np.float32)
        if gradient_values.size == total_parameters:
            gradient_spec.update({
                "count": int(gradient_values.size),
                "layout": "flat_canonical_parameter_order",
                "parameter_layout": layout,
            })
        else:
            gradient_spec.update({
                "status": "shape_mismatch",
                "count": int(gradient_values.size),
                "expected_count": total_parameters,
            })

    for semantic_role, parameter_role in (("weights_after", "weight"), ("biases_after", "bias")):
        spec = captures.get(semantic_role, {})
        source = spec.get("path")
        if source and Path(source).exists():
            values = np.fromfile(source, dtype=np.float32)
            if values.size != total_parameters:
                spec.update({"status": "shape_mismatch", "count": int(values.size), "expected_count": total_parameters})
                continue
            selected = _select_parameter_role(values, layout, parameter_role)
            out = root / f"{semantic_role}.bin"
            selected.tofile(out)
            spec.update({
                "source_path": str(source),
                "path": str(out),
                "status": "captured",
                "layout": "flat_canonical_parameter_role_order",
                "count": int(selected.size),
                "parameter_layout": [entry for entry in layout if entry["role"] == parameter_role],
            })

    optimizer = str(optimizer_type).lower().replace("-", "_")
    if optimizer == "adam":
        state_spec = captures.get("optimizer_m_after") or captures.get("optimizer_v_after") or captures.get("optimizer_step_after") or {}
        source = state_spec.get("path")
        if source and Path(source).exists():
            packed = np.fromfile(source, dtype=np.float32)
            # Python reference state is m||v; HLS export is m||v||step.
            expected_without_step = 2 * total_parameters
            if packed.size not in {expected_without_step, expected_without_step + 1}:
                for role in ("optimizer_m_after", "optimizer_v_after", "optimizer_step_after"):
                    if role in captures:
                        captures[role].update({"status": "shape_mismatch", "count": int(packed.size), "expected_count": [expected_without_step, expected_without_step + 1]})
            else:
                components = {
                    "optimizer_m_after": packed[:total_parameters],
                    "optimizer_v_after": packed[total_parameters:2 * total_parameters],
                }
                state_validation = {
                    "layout": "m_then_v_then_optional_step_canonical_parameter_order",
                    "actual_words": int(packed.size),
                    "expected_state_words": int(expected_without_step),
                    "step_present": bool(packed.size == expected_without_step + 1),
                    "finite": bool(np.all(np.isfinite(packed))),
                    "v_nonnegative": bool(np.all(components["optimizer_v_after"] >= 0.0)),
                    "v_min": float(components["optimizer_v_after"].min()) if total_parameters else 0.0,
                    "v_max": float(components["optimizer_v_after"].max()) if total_parameters else 0.0,
                }
                payload.setdefault("metadata", {})["optimizer_state_validation"] = state_validation
                for role, values in components.items():
                    out = root / f"{role}.bin"
                    values.astype(np.float32).tofile(out)
                    captures.setdefault(role, {}).update({
                        "source_path": str(source), "path": str(out), "status": "captured",
                        "layout": "flat_canonical_parameter_order", "count": int(values.size),
                        "parameter_layout": layout,
                    })
                step_value = float(packed[-1]) if packed.size == expected_without_step + 1 else None
                step_spec = captures.setdefault("optimizer_step_after", {"required": True, "dtype": "float32"})
                if step_value is not None:
                    out = _write_scalar(root / "optimizer_step_after.bin", step_value)
                    step_spec.update({"source_path": str(source), "path": str(out), "status": "captured", "layout": "scalar", "count": 1})
                elif step_spec.get("path") and Path(step_spec["path"]).exists():
                    step_spec.update({"status": "captured", "layout": "scalar", "count": 1})
                else:
                    step_spec.update({"path": None, "status": "missing"})

    payload["captures"] = captures
    payload.setdefault("metadata", {})["canonical_parameter_layout"] = layout
    payload["metadata"]["canonical_parameter_words"] = total_parameters
    payload["metadata"]["canonicalization_status"] = "completed"
    manifest_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest_file



def derive_canonical_parameter_layout_from_graph(graph: Any) -> list[dict[str, Any]]:
    """Derive the flattened trainable-parameter order used by FPGAI references/HLS.

    This is intentionally based on semantic IR operators rather than generated
    source declarations. It therefore remains stable across HLS, RTL, and future
    community backends.
    """
    from fpgai.engine.training_graph_utils import (
        as_chw, get_tensor_shape, resolve_batchnorm_arrays,
        resolve_conv_arrays, resolve_dense_arrays,
    )

    entries: list[dict[str, Any]] = []
    offset = 0

    def add(name: str, layer: str, role: str, array: Any) -> None:
        nonlocal offset
        values = np.asarray(array)
        count = int(values.size)
        entries.append({
            "name": name, "layer": layer, "role": role,
            "offset": offset, "count": count, "shape": list(values.shape),
        })
        offset += count

    for op in graph.ops:
        if op.op_type == "Dense":
            weights, bias, _, _ = resolve_dense_arrays(graph, op)
            add(f"{op.name}.weight", op.name, "weight", weights)
            add(f"{op.name}.bias", op.name, "bias", bias)
        elif op.op_type == "Conv":
            weights, bias, _ = resolve_conv_arrays(graph, op)
            add(f"{op.name}.weight", op.name, "weight", weights)
            add(f"{op.name}.bias", op.name, "bias", np.asarray(bias).reshape(-1))
        elif op.op_type == "BatchNormalization":
            shape = get_tensor_shape(graph, op.outputs[0]) or get_tensor_shape(graph, op.inputs[0])
            if not shape:
                raise RuntimeError(f"BatchNormalization shape unavailable for op {op.name!r}")
            channels, _, _ = as_chw(shape)
            gamma, beta, _, _ = resolve_batchnorm_arrays(graph, op, channels)
            add(f"{op.name}.gamma", op.name, "weight", gamma)
            add(f"{op.name}.beta", op.name, "bias", beta)
    return canonical_parameter_layout(entries)


def orchestrate_training_numeric_equivalence(
    *, graph: Any, training_reference_result: Any, hls_artifact_dir: str | Path | None,
    training_dir: str | Path, optimizer_type: str, raw_config: dict[str, Any] | None = None, atol: float = 1e-5, rtol: float = 1e-4,
) -> dict[str, Any]:
    """Publish, canonicalize, compare, and promote one training execution."""
    root = Path(training_dir)
    eq_path = root / "gradient_mechanism_equivalence.json"
    if not eq_path.exists():
        return {"status": "artifact_missing", "reason": "equivalence_contract_missing"}
    eq = json.loads(eq_path.read_text(encoding="utf-8"))
    workload = str(eq.get("workload_fingerprint_sha256", ""))
    implementation = str(eq.get("implementation_stack_fingerprint_sha256", ""))
    layout = derive_canonical_parameter_layout_from_graph(graph)
    layout_path = root / "canonical_parameter_layout.json"
    layout_path.write_text(json.dumps({
        "artifact_kind": "fpgai_canonical_parameter_layout",
        "schema_version": 1, "entries": layout,
        "total_parameter_words": sum(int(e["count"]) for e in layout),
    }, indent=2) + "\n", encoding="utf-8")

    ref_manifest = publish_python_training_reference(
        result=training_reference_result, output_path=root / "python_reference_capture.json",
        workload_fingerprint_sha256=workload,
        implementation_stack_fingerprint_sha256=implementation,
    )
    materialize_canonical_capture_files(
        manifest_path=ref_manifest, parameter_layout=layout, optimizer_type=optimizer_type,
        output_dir=root / "python_reference_canonical",
    )

    if hls_artifact_dir is None:
        report_path = root / "numeric_equivalence_report.json"
        report_path.write_text(json.dumps({
            "artifact_kind": "fpgai_numeric_equivalence_report",
            "schema_version": 1, "status": "artifact_missing", "passed": False,
            "reason": "hls_artifact_dir_missing",
        }, indent=2) + "\n", encoding="utf-8")
        promote_gradient_equivalence_status(eq_path, report_path)
        return {"status": "artifact_missing", "report": str(report_path), "parameter_layout": str(layout_path)}

    hls_manifest = bind_hls_training_capture(
        artifact_dir=hls_artifact_dir, output_path=root / "hls_csim_capture.json",
        workload_fingerprint_sha256=workload,
        implementation_stack_fingerprint_sha256=implementation,
        optimizer_type=optimizer_type,
    )
    materialize_canonical_capture_files(
        manifest_path=hls_manifest, parameter_layout=layout, optimizer_type=optimizer_type,
        output_dir=root / "hls_csim_canonical",
    )
    from fpgai.validation.training_probes import (
        compare_training_probes, materialize_hardware_fixed_point_dense_probes,
        materialize_hls_observable_probes, materialize_python_dense_probes,
        normalize_probe_config,
    )
    probe_config = normalize_probe_config(raw_config)
    float_probe_path = materialize_python_dense_probes(
        graph=graph, result=training_reference_result, probe_config=probe_config,
        parameter_layout=layout, output_path=root / "python_float_training_probes.json",
    )
    python_probe_path = materialize_hardware_fixed_point_dense_probes(
        graph=graph, result=training_reference_result, raw_config=raw_config or {},
        probe_config=probe_config, parameter_layout=layout,
        output_path=root / "python_training_probes.json",
    )
    hls_probe_path = materialize_hls_observable_probes(
        hls_artifact_dir=hls_artifact_dir, probe_config=probe_config,
        parameter_layout=layout, output_path=root / "hls_training_probes.json",
    )
    probe_comparison_path = None
    if python_probe_path is not None and hls_probe_path is not None:
        probe_comparison_path = compare_training_probes(
            python_probe_path, hls_probe_path, root / "training_probe_comparison.json",
            atol=atol, rtol=rtol,
        )
    report_path = write_numeric_equivalence_report(
        ref_manifest,
        hls_manifest,
        root / "numeric_equivalence_report.json",
        atol=atol,
        rtol=rtol,
        training_trace_output_path=root / "training_execution_trace.json",
        probe_comparison_path=probe_comparison_path,
    )
    promote_gradient_equivalence_status(eq_path, report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["parameter_layout"] = str(layout_path)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    trace_path = root / "training_execution_trace.json"
    return {"status": report.get("status"), "report": str(report_path), "parameter_layout": str(layout_path),
            "execution_trace": str(trace_path), "python_manifest": str(ref_manifest),
            "hls_manifest": str(hls_manifest),
            "python_probes": None if python_probe_path is None else str(python_probe_path),
            "python_float_probes": None if float_probe_path is None else str(float_probe_path),
            "hls_probes": None if hls_probe_path is None else str(hls_probe_path),
            "probe_comparison": None if probe_comparison_path is None else str(probe_comparison_path)}

def compare_capture_manifests(
    reference_manifest: str | Path,
    candidate_manifest: str | Path,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> dict[str, Any]:
    """Compare two canonical capture manifests and promote a final status."""
    from fpgai.validation.capture_schema import compare_capture_contracts

    ref = json.loads(Path(reference_manifest).read_text(encoding="utf-8"))
    got = json.loads(Path(candidate_manifest).read_text(encoding="utf-8"))
    comparability = compare_capture_contracts(ref, got)
    report: dict[str, Any] = {
        "artifact_kind": "fpgai_numeric_equivalence_report",
        "schema_version": 1,
        "reference_manifest": str(reference_manifest),
        "candidate_manifest": str(candidate_manifest),
        "comparability": comparability,
        "comparisons": {},
    }
    if comparability["status"] == "workload_mismatch":
        report["status"] = "workload_mismatch"
        report["passed"] = False
        return report
    if comparability["status"] != "ready_for_numeric_comparison":
        roles = comparability.get("roles", {})
        completed = sorted(role for role, item in roles.items() if item.get("comparable"))
        missing = sorted(comparability.get("missing_required_captures", []))
        required_total = sum(1 for item in roles.values() if item.get("required"))
        required_completed = sum(
            1 for item in roles.values()
            if item.get("required") and item.get("comparable")
        )
        report["status"] = "partial" if completed else "artifact_missing"
        report["passed"] = False
        report["comparison_possible"] = False
        report["completed_captures"] = completed
        report["missing_required_captures"] = missing
        report["required_capture_completion"] = {
            "completed": required_completed,
            "total": required_total,
            "percentage": (100.0 * required_completed / required_total) if required_total else 100.0,
        }
        return report

    all_passed = True
    for role, role_status in comparability["roles"].items():
        if not role_status.get("comparable"):
            continue
        ref_path = ref["captures"][role]["path"]
        got_path = got["captures"][role]["path"]
        comparison = compare_flat_f32(
            ref_path, got_path, atol=atol, rtol=rtol,
            capture=got["captures"].get(role) or ref["captures"].get(role),
        )
        report["comparisons"][role] = comparison
        all_passed = all_passed and bool(comparison.get("passed"))
    report["status"] = "passed" if all_passed else "failed"
    report["passed"] = all_passed
    return report


def write_numeric_equivalence_report(
    reference_manifest: str | Path,
    candidate_manifest: str | Path,
    output_path: str | Path,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-4,
    training_trace_output_path: str | Path | None = None,
    probe_comparison_path: str | Path | None = None,
) -> Path:
    """Write the numeric report and its canonical training execution trace.

    Training is identified from the semantic capture roles rather than the
    caller. This keeps direct and orchestrated report paths consistent while
    leaving inference-only comparisons unchanged.
    """
    report = compare_capture_manifests(reference_manifest, candidate_manifest, atol=atol, rtol=rtol)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    roles = set((report.get("comparability", {}) or {}).get("roles", {}))
    is_training = bool(roles & {
        "parameter_gradients", "optimizer_m_after", "optimizer_v_after",
        "optimizer_step_after", "post_update_loss",
    })
    if is_training:
        from fpgai.validation.execution_trace import write_training_execution_trace
        trace_path = Path(training_trace_output_path) if training_trace_output_path is not None else out.with_name("training_execution_trace.json")
        write_training_execution_trace(
            out,
            trace_path,
            probe_comparison_path=probe_comparison_path,
        )
    return out


def promote_gradient_equivalence_status(
    equivalence_path: str | Path,
    numeric_report_path: str | Path,
) -> Path:
    """Promote the mechanism artifact using a completed numeric report."""
    equivalence_file = Path(equivalence_path)
    numeric_file = Path(numeric_report_path)
    equivalence = json.loads(equivalence_file.read_text(encoding="utf-8"))
    report = json.loads(numeric_file.read_text(encoding="utf-8"))
    status = str(report.get("status", "artifact_missing"))
    allowed = {"passed", "failed", "partial", "artifact_missing", "workload_mismatch"}
    if status not in allowed:
        status = "artifact_missing"
    equivalence["numeric_equivalence_status"] = status
    equivalence["numeric_equivalence_report"] = str(numeric_file)
    equivalence["claim_status"] = (
        "numeric_equivalence_validated"
        if status == "passed"
        else "architectural_result_preliminary_until_numeric_equivalence_passes"
    )
    metrics = equivalence.setdefault("required_comparisons", {})
    comparisons = report.get("comparisons", {})
    role_to_metric = {
        "pre_update_loss": "pre_update_loss_abs_error",
        "post_update_loss": "post_update_loss_abs_error",
        "weights_after": "weights_max_abs_error",
        "biases_after": "biases_max_abs_error",
        "optimizer_m_after": "adam_m_max_abs_error",
        "optimizer_v_after": "adam_v_max_abs_error",
        "parameter_gradients": "exported_gradients_max_abs_error",
    }
    for role, metric_name in role_to_metric.items():
        if role in comparisons:
            metrics[metric_name] = comparisons[role].get("max_abs_error")
    if "optimizer_step_after" in comparisons:
        metrics["optimizer_step_match"] = bool(comparisons["optimizer_step_after"].get("passed"))
    equivalence_file.write_text(json.dumps(equivalence, indent=2) + "\n", encoding="utf-8")
    return equivalence_file
