"""Numeric validation between imported source semantics and functional FPGAI IR.

This module belongs to the compiler validation subsystem.  It deliberately
validates *before* architecture/backend lowering so FPGAI can distinguish an
ingress/IR semantic error from a backend/HLS/RTL error.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import json
import math
import numpy as np


def _metrics(reference: np.ndarray, generated: np.ndarray, *, limits: Mapping[str, float]) -> dict[str, Any]:
    ref = np.asarray(reference, dtype=np.float32).reshape(-1)
    got = np.asarray(generated, dtype=np.float32).reshape(-1)
    if ref.shape != got.shape:
        return {
            "status": "shape_mismatch",
            "passed": False,
            "reference_shape": list(np.asarray(reference).shape),
            "fpgai_ir_shape": list(np.asarray(generated).shape),
        }
    if ref.size == 0:
        return {"status": "empty", "passed": False, "num_values": 0}
    diff = got - ref
    abs_diff = np.abs(diff)
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(abs_diff))
    max_abs = float(np.max(abs_diff))
    rmse = float(math.sqrt(mse))
    nr = float(np.linalg.norm(ref))
    ng = float(np.linalg.norm(got))
    cosine = None if nr == 0.0 or ng == 0.0 else float(np.dot(ref, got) / (nr * ng))
    checks = [
        {"name": "max_abs_error", "value": max_abs, "limit": float(limits["max_abs_error_limit"]), "passed": max_abs <= float(limits["max_abs_error_limit"])},
        {"name": "mae", "value": mae, "limit": float(limits["mean_abs_error_limit"]), "passed": mae <= float(limits["mean_abs_error_limit"])},
        {"name": "rmse", "value": rmse, "limit": float(limits["rmse_limit"]), "passed": rmse <= float(limits["rmse_limit"])},
        {"name": "cosine_similarity", "value": cosine, "limit": float(limits["min_cosine_similarity"]), "passed": cosine is not None and cosine >= float(limits["min_cosine_similarity"])},
    ]
    return {
        "status": "compared",
        "passed": all(bool(item["passed"]) for item in checks),
        "num_values": int(ref.size),
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "max_abs_error": max_abs,
        "cosine_similarity": cosine,
        "checks": checks,
    }


def _reference_cfg(raw_config: dict[str, Any] | None) -> dict[str, Any]:
    numeric = dict(((raw_config or {}).get("validation") or {}).get("numeric") or {})
    value = numeric.get("reference", {})
    return dict(value) if isinstance(value, Mapping) else {}


def _reference_bundle_manifest(model_path: Path, raw_config: dict[str, Any] | None) -> Path | None:
    reference = _reference_cfg(raw_config)
    configured = reference.get("bundle")
    candidates: list[Path] = []
    if configured not in (None, ""):
        path = Path(str(configured)).expanduser()
        if not path.is_absolute():
            path = (model_path.parent / path).resolve()
        candidates.append(path / "reference_manifest.json" if path.is_dir() else path)
    candidates.extend([
        model_path.with_suffix(model_path.suffix + ".fpgai-reference.json"),
        model_path.with_name(model_path.stem + ".fpgai-reference.json"),
        model_path.with_name(model_path.stem + ".fpgai-reference") / "reference_manifest.json",
    ])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _bundle_entries(value: Any) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(value, Mapping):
        return [(str(name), dict(spec) if isinstance(spec, Mapping) else {"path": spec}) for name, spec in value.items()]
    if isinstance(value, list):
        result: list[tuple[str, dict[str, Any]]] = []
        for index, item in enumerate(value):
            if not isinstance(item, Mapping):
                continue
            spec = dict(item)
            name = str(spec.get("name") or spec.get("source_tensor") or spec.get("fpgai_tensor") or f"tensor_{index}")
            result.append((name, spec))
        return result
    return []


def _load_bundle_tensor(manifest_path: Path, spec: Mapping[str, Any]) -> np.ndarray:
    if "values" in spec:
        arr = np.asarray(spec.get("values"), dtype=np.dtype(str(spec.get("dtype", "float32"))))
    else:
        raw_path = spec.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("reference bundle tensor requires path or inline values")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = (manifest_path.parent / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"reference tensor artifact is missing: {path}")
        if path.suffix.lower() == ".npy":
            arr = np.load(path)
        elif path.suffix.lower() == ".npz":
            key = str(spec.get("key", ""))
            with np.load(path) as archive:
                if not key:
                    if len(archive.files) != 1:
                        raise ValueError(f"reference NPZ {path} requires a key because it contains {len(archive.files)} arrays")
                    key = archive.files[0]
                arr = np.asarray(archive[key])
        else:
            dtype = np.dtype(str(spec.get("dtype", "float32")))
            arr = np.fromfile(path, dtype=dtype)
    shape = spec.get("shape")
    if isinstance(shape, (list, tuple)) and shape:
        dims = tuple(int(v) for v in shape)
        if int(np.prod(dims, dtype=np.int64)) != int(arr.size):
            raise ValueError(f"reference tensor shape {dims} does not match {arr.size} values")
        arr = arr.reshape(dims)
    return np.asarray(arr, dtype=np.float32)


def _validate_reference_bundle(
    *, graph: Any, manifest_path: Path, levels: set[str], raw_config: dict[str, Any] | None, out_dir: str | Path
) -> dict[str, Any]:
    from fpgai.benchmark.graph_reference import execute_graph_reference_trace
    from .numeric_common import _precision_aware_inference_limits

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(payload.get("schema", "")) != "fpgai.frontend-reference/v1":
        return {"status": "invalid_reference_bundle", "passed": False, "reason": "unsupported reference bundle schema", "bundle": str(manifest_path)}

    inputs: dict[str, np.ndarray] = {}
    input_records: dict[str, Any] = {}
    by_target: dict[str, tuple[str, dict[str, Any]]] = {}
    for name, spec in _bundle_entries(payload.get("inputs", {})):
        target = str(spec.get("fpgai_tensor") or name)
        by_target[target] = (name, spec)
    missing_inputs = []
    for target in graph.inputs:
        target = str(target)
        record = by_target.get(target)
        if record is None:
            missing_inputs.append(target)
            continue
        source_name, spec = record
        arr = _load_bundle_tensor(manifest_path, spec)
        expected = graph.get_tensor(target)
        if expected is not None and getattr(expected, "shape", None):
            dims = tuple(int(v) for v in expected.shape)
            if dims and all(v > 0 for v in dims) and int(np.prod(dims)) == int(arr.size):
                arr = arr.reshape(dims)
        inputs[target] = arr
        input_records[target] = {"source_tensor": source_name, "shape": list(arr.shape)}
    if missing_inputs:
        return {
            "status": "reference_input_missing",
            "passed": False,
            "reason": f"reference bundle does not provide FPGAI input tensor(s): {missing_inputs}",
            "bundle": str(manifest_path),
        }

    try:
        trace = execute_graph_reference_trace(graph, inputs)
    except Exception as exc:
        return {"status": "fpgai_ir_execution_failed", "passed": False, "reason": str(exc), "bundle": str(manifest_path)}

    limits = _precision_aware_inference_limits(raw_config)

    def compare_group(group: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, spec in _bundle_entries(payload.get(group, {})):
            target = str(spec.get("fpgai_tensor") or name)
            if target not in trace:
                result[target] = {"status": "missing_fpgai_ir_tensor", "passed": False, "source_tensor": name}
                continue
            reference = _load_bundle_tensor(manifest_path, spec)
            result[target] = {"source_tensor": name, **_metrics(reference, trace[target], limits=limits)}
        return result

    outputs = compare_group("outputs")
    intermediates = compare_group("intermediates")
    states = compare_group("state")
    if not states:
        states = compare_group("states")

    graph_outputs = [str(x) for x in graph.outputs]
    missing_outputs = [name for name in graph_outputs if name not in outputs]
    requested_intermediate = bool(levels & {"layer", "layerwise", "intermediate"})
    graph_intermediates = [str(op.outputs[0]) for op in graph.ops if op.outputs and str(op.outputs[0]) not in graph_outputs]
    compared_intermediate_targets = set(intermediates)
    missing_intermediates = [name for name in graph_intermediates if name not in compared_intermediate_targets] if requested_intermediate else []

    state_tensors = []
    if "state" in levels:
        for name, spec in getattr(graph, "tensors", {}).items():
            state = getattr(getattr(spec, "semantics", None), "state", None)
            role = str(getattr(state, "role", "") or "") if state is not None else ""
            if role and role not in {"activation", "none", "unspecified"}:
                state_tensors.append(str(name))
    missing_states = [name for name in state_tensors if name not in states]

    comparisons = [*outputs.values(), *intermediates.values(), *states.values()]
    compared_pass = all(bool(item.get("passed")) for item in comparisons) if comparisons else False
    coverage_ok = not missing_outputs and not missing_intermediates and not missing_states
    passed = bool(compared_pass and coverage_ok)
    result = {
        "schema": "fpgai.frontend-ir-numeric-validation/v1",
        "status": "passed" if passed else ("insufficient_reference_coverage" if not coverage_ok else "failed_numeric_validation"),
        "passed": passed,
        "source_format": "reference_bundle",
        "source_framework": str(payload.get("source_framework", payload.get("framework", "unknown"))),
        "bundle": str(manifest_path),
        "requested_levels": sorted(levels),
        "inputs": input_records,
        "outputs": outputs,
        "intermediates": intermediates,
        "state": states,
        "missing_outputs": missing_outputs,
        "missing_intermediates": missing_intermediates,
        "missing_state": missing_states,
        "limits": limits,
    }
    work = Path(out_dir) / "reports" / "frontend_ir_validation"
    work.mkdir(parents=True, exist_ok=True)
    report = work / "frontend_to_fpgai_ir.json"
    report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["report"] = str(report)
    return result


def validate_frontend_to_fpgai_ir(
    *,
    graph: Any,
    model_path: str | Path | None,
    model_format: str | None,
    raw_config: dict[str, Any] | None,
    out_dir: str | Path,
) -> dict[str, Any]:
    """Compare executable source semantics with functional FPGAI IR.

    ONNX is currently executable through ONNX Runtime.  MLIR/StableHLO ingress is
    reported explicitly as unavailable unless its originating frontend supplies an
    executable source reference; it is never silently marked as passed.
    """
    numeric_cfg = dict(((raw_config or {}).get("validation") or {}).get("numeric") or {})
    if not bool(numeric_cfg.get("enabled", False)):
        return {"status": "not_requested", "passed": None, "reason": "validation.numeric.enabled is false"}

    levels = {str(x).strip().lower() for x in numeric_cfg.get("levels", ["model"]) or ["model"]}
    want_intermediate = bool(levels & {"layer", "intermediate", "state"})
    fmt = str(model_format or "").strip().lower()
    path = Path(model_path) if model_path not in (None, "") else None
    if path is None or not path.is_file():
        return {"status": "unavailable", "passed": None, "reason": "source model path is unavailable"}

    bundle = _reference_bundle_manifest(path, raw_config)
    if bundle is not None:
        try:
            return _validate_reference_bundle(
                graph=graph, manifest_path=bundle, levels=levels, raw_config=raw_config, out_dir=out_dir
            )
        except Exception as exc:
            return {"status": "reference_bundle_failed", "passed": False, "reason": str(exc), "bundle": str(bundle)}

    if fmt not in {"onnx", ""} and path.suffix.lower() != ".onnx":
        return {
            "status": "unsupported_source_execution",
            "passed": None,
            "reason": f"frontend-to-FPGAI-IR executable comparison is not yet available for source format {fmt or path.suffix}",
            "source_format": fmt or path.suffix.lstrip("."),
        }

    try:
        import onnxruntime as ort  # type: ignore
        import onnx  # type: ignore
        from onnx import helper  # type: ignore
    except Exception as exc:
        return {"status": "runtime_unavailable", "passed": None, "reason": f"ONNX reference runtime unavailable: {exc}"}

    from fpgai.benchmark.graph_reference import deterministic_graph_inputs, execute_graph_reference_trace
    from .numeric_common import _precision_aware_inference_limits

    try:
        graph_inputs = deterministic_graph_inputs(graph)
        if len(graph.inputs) != 1:
            return {"status": "unsupported_input_arity", "passed": None, "reason": "frontend-to-IR ONNX validation currently requires one model input"}
        input_tensor = str(graph.inputs[0])
        x = np.asarray(graph_inputs[input_tensor], dtype=np.float32)
        ir_trace = execute_graph_reference_trace(graph, graph_inputs)

        model = onnx.load(str(path))
        session_probe = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        source_inputs = session_probe.get_inputs()
        if len(source_inputs) != 1:
            return {"status": "unsupported_input_arity", "passed": None, "reason": f"ONNX source has {len(source_inputs)} inputs"}
        source_input_name = source_inputs[0].name
        source_shape = tuple(int(v) for v in x.shape)
        try:
            # The imported FPGAI tensor shape normally already matches ONNX.  Keep
            # a reshape-only reconciliation for flattened batch-one source inputs.
            meta_shape = tuple(int(v) for v in source_inputs[0].shape if isinstance(v, int) and v > 0)
            if meta_shape and int(np.prod(meta_shape)) == int(x.size):
                source_shape = meta_shape
        except Exception:
            pass
        source_x = x.reshape(source_shape).astype(np.float32)

        wanted = [str(graph.outputs[0])]
        if want_intermediate:
            wanted.extend(str(op.outputs[0]) for op in graph.ops if op.outputs and str(op.outputs[0]) not in wanted)
        existing_outputs = {str(item.name) for item in model.graph.output}
        known_values = {str(item.name) for item in model.graph.value_info} | existing_outputs
        known_values |= {str(item.name) for item in model.graph.input} | {str(item.name) for item in model.graph.initializer}
        source_wanted = [name for name in wanted if name in known_values]
        if str(graph.outputs[0]) not in source_wanted:
            source_wanted.insert(0, str(graph.outputs[0]))

        augmented = onnx.load(str(path))
        output_names = {str(item.name) for item in augmented.graph.output}
        for name in source_wanted:
            if name not in output_names:
                augmented.graph.output.append(helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
        work = Path(out_dir) / "reports" / "frontend_ir_validation"
        work.mkdir(parents=True, exist_ok=True)
        augmented_path = work / "source_with_validation_outputs.onnx"
        onnx.save(augmented, str(augmented_path))
        sess = ort.InferenceSession(str(augmented_path), providers=["CPUExecutionProvider"])
        source_values = sess.run(source_wanted, {source_input_name: source_x})
        limits = _precision_aware_inference_limits(raw_config)

        comparisons: dict[str, Any] = {}
        for name, source_value in zip(source_wanted, source_values):
            if name not in ir_trace:
                comparisons[name] = {"status": "missing_fpgai_ir_tensor", "passed": False}
                continue
            comparisons[name] = _metrics(np.asarray(source_value), ir_trace[name], limits=limits)

        final_name = str(graph.outputs[0])
        model_result = comparisons.get(final_name, {"status": "missing", "passed": False})
        intermediate_results = {name: value for name, value in comparisons.items() if name != final_name}
        requested_intermediates = [name for name in wanted if name != final_name]
        unavailable_intermediates = [name for name in requested_intermediates if name not in source_wanted]
        compared_intermediate_pass = all(bool(v.get("passed")) for v in intermediate_results.values()) if intermediate_results else True
        passed = bool(model_result.get("passed")) and compared_intermediate_pass
        payload = {
            "schema": "fpgai.frontend-ir-numeric-validation/v1",
            "status": "passed" if passed else "failed_numeric_validation",
            "passed": passed,
            "source_format": "onnx",
            "model": {"tensor": final_name, **model_result},
            "intermediates": intermediate_results,
            "unavailable_intermediates": unavailable_intermediates,
            "requested_levels": sorted(levels),
            "input_tensor": input_tensor,
            "source_input": source_input_name,
            "limits": limits,
        }
        report = work / "frontend_to_fpgai_ir.json"
        report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        payload["report"] = str(report)
        return payload
    except Exception as exc:
        return {"status": "execution_failed", "passed": False, "reason": str(exc)}


__all__ = ["validate_frontend_to_fpgai_ir"]
