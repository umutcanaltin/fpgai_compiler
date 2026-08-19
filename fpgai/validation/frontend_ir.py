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
