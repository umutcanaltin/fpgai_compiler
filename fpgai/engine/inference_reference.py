from __future__ import annotations

"""Inference reference artifact generation used by the compiler pipeline."""

from pathlib import Path
from typing import Any

import numpy as np

from fpgai.backends.hls.emit.params_h import _conv_sizes, _dense_sizes
from fpgai.validation.dataset import emit_dataset_artifacts

def _is_runtime_weight_mode(weights_mode: str) -> bool:
    return str(weights_mode).strip().lower() in {
        "stream",
        "streamed",
        "ddr",
        "dma_ddr",
        "uram",
        "bram_import_full",
        "bram_import_export_full",
        "uram_import_full",
        "uram_import_export_full",
        "ddr_tiled",
        "ddr_tiled_mutable",
    }


def _runtime_weight_word_count(graph) -> int:
    total = 0
    for op in graph.ops:
        if op.op_type == "Conv":
            weight_count, bias_count = _conv_sizes(graph, op)
            total += int(weight_count) + int(bias_count)
        elif op.op_type == "Dense":
            weight_count, bias_count = _dense_sizes(graph, op)
            total += int(weight_count) + int(bias_count)
    return int(total)



def _normalise_onnx_shape(shape: Any, fallback_size: int) -> tuple[int, ...]:
    dims: list[int] = []
    unknown_index: int | None = None
    known_product = 1
    for idx, raw_dim in enumerate(shape or []):
        try:
            dim = int(raw_dim)
        except Exception:
            dim = -1
        if dim > 0:
            dims.append(dim)
            known_product *= dim
        else:
            dims.append(1)
            if unknown_index is None:
                unknown_index = idx
    if not dims:
        return (int(fallback_size),)
    if unknown_index is not None and fallback_size > 0 and known_product > 0 and fallback_size % known_product == 0:
        dims[unknown_index] = max(1, int(fallback_size // known_product))
    return tuple(int(max(1, dim)) for dim in dims)


def _emit_inference_reference_artifacts(
    out_dir: str | Path,
    *,
    model_path: str | Path | None,
    hls_ok: bool | None,
    raw_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Emit Python/ONNX reference output for the generated HLS CSim output.

    This is deliberately an artifact-backed validation path: it only emits a
    successful comparison candidate when an HLS CSim output file exists.  If
    ONNX Runtime is unavailable, the model cannot be loaded, or the input shape
    cannot be reconciled, the returned payload records the reason and
    ``emit_numeric_validation_report`` keeps inference numeric status pending.
    """

    out = Path(out_dir)
    input_bin = out / "input.bin"
    output_bin = out / "output.bin"
    ref_dir = out / "reference"
    ref_output_bin = ref_dir / "outputs_ref.bin"
    ref_output_npy = ref_dir / "outputs_ref.npy"
    ref_input_npy = ref_dir / "inputs_ref.npy"
    status: dict[str, Any] = {
        "status": "not_run",
        "reason": "inference reference generation did not run",
        "inputs_bin": input_bin,
        "outputs_hw": output_bin,
        "outputs_ref": None,
    }
    if hls_ok is not True:
        status["reason"] = "HLS CSim did not pass, so inference numeric reference comparison is pending."
        return status

    dataset_artifacts = emit_dataset_artifacts(out, raw_config=raw_config)
    dataset_available = dataset_artifacts.get("status") == "available"
    resolved_input_bin = (
        Path(dataset_artifacts["inputs_bin"])
        if dataset_available
        else input_bin
    )

    if not resolved_input_bin.exists():
        status["reason"] = f"input artifact missing: {resolved_input_bin}"
        return status
    if not output_bin.exists():
        status["reason"] = f"HLS CSim output.bin missing: {output_bin}"
        return status
    if model_path in (None, ""):
        status["reason"] = "model path missing; cannot run ONNX reference"
        return status
    try:
        import onnxruntime as ort  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional runtime package
        status["reason"] = f"onnxruntime unavailable: {exc}"
        return status

    try:
        if dataset_available:
            x = np.asarray(dataset_artifacts["inputs_array"], dtype=np.float32)
            input_flat = x.reshape(-1)
        else:
            input_flat = np.fromfile(input_bin, dtype=np.float32)
            x = None

        sess = ort.InferenceSession(str(Path(model_path)), providers=["CPUExecutionProvider"])
        inputs = sess.get_inputs()
        outputs = sess.get_outputs()
        if len(inputs) != 1:
            status["reason"] = f"expected exactly one ONNX input, got {len(inputs)}"
            return status
        if len(outputs) != 1:
            status["reason"] = f"expected exactly one ONNX output, got {len(outputs)}"
            return status
        input_meta = inputs[0]
        output_meta = outputs[0]

        if dataset_available:
            sample_count = int(dataset_artifacts["sample_count"])
            sample_shape = tuple(int(v) for v in dataset_artifacts["input_shape"])
            if tuple(x.shape[1:]) != sample_shape:
                status["reason"] = f"normalized dataset input shape {tuple(x.shape[1:])} does not match manifest {sample_shape}"
                return status
            model_shape = list(getattr(input_meta, "shape", None) or [])
            sample_words = int(np.prod(sample_shape)) if sample_shape else 1

            def _static_product(dims: list[Any]) -> int | None:
                product = 1
                for dim in dims:
                    try:
                        value = int(dim)
                    except Exception:
                        return None
                    if value <= 0:
                        return None
                    product *= value
                return product

            full_product = _static_product(model_shape)
            trailing_product = _static_product(model_shape[1:]) if model_shape else None
            try:
                leading_dim = int(model_shape[0]) if model_shape else None
            except Exception:
                leading_dim = None

            # A dynamic leading dimension is a real dataset batch dimension when the
            # remaining ONNX dimensions describe one normalized sample.
            dynamic_batch = (
                bool(model_shape)
                and leading_dim in (None, -1, 0)
                and trailing_product == sample_words
            )

            # A static leading dimension equal to the selected sample count can also
            # consume the complete batch in one ONNX Runtime invocation.
            full_static_batch = (
                bool(model_shape)
                and leading_dim == sample_count
                and trailing_product == sample_words
                and sample_count > 1
            )

            if dynamic_batch or full_static_batch:
                model_sample_shape = tuple(int(v) for v in model_shape[1:])
                batch_input = x.reshape((sample_count, *model_sample_shape)).astype(np.float32)
                y_raw = sess.run([output_meta.name], {input_meta.name: batch_input})[0]
                y_arr = np.asarray(y_raw, dtype=np.float32)
                if y_arr.ndim == 1:
                    y_arr = y_arr.reshape((sample_count, -1))
            elif full_product == sample_words:
                # The ONNX model has a fixed per-invocation shape. This covers both
                # genuinely unbatched models such as [784] and static batch-one image
                # models such as [1, 1, 28, 28]. Dataset storage may be flattened; the
                # semantic compatibility requirement is equal element count.
                model_sample_shape = tuple(int(v) for v in model_shape)
                rows = []
                for sample in x:
                    sample_input = sample.reshape(model_sample_shape).astype(np.float32)
                    rows.append(
                        np.asarray(
                            sess.run([output_meta.name], {input_meta.name: sample_input})[0],
                            dtype=np.float32,
                        ).reshape(1, -1)
                    )
                y_arr = np.concatenate(rows, axis=0)
            elif leading_dim == 1 and trailing_product == sample_words:
                # Defensive compatibility for static batch-one metadata where the
                # complete shape could not otherwise be resolved.
                model_sample_shape = tuple(int(v) for v in model_shape[1:])
                rows = []
                for sample in x:
                    sample_input = sample.reshape((1, *model_sample_shape)).astype(np.float32)
                    rows.append(
                        np.asarray(
                            sess.run([output_meta.name], {input_meta.name: sample_input})[0],
                            dtype=np.float32,
                        ).reshape(1, -1)
                    )
                y_arr = np.concatenate(rows, axis=0)
            else:
                status["reason"] = (
                    f"normalized dataset sample shape {sample_shape} ({sample_words} values) is incompatible "
                    f"with ONNX input shape {model_shape}"
                )
                return status
            input_shape = tuple(int(v) for v in x.shape)
            output_shape_per_sample = tuple(int(v) for v in y_arr.shape[1:])
            y = y_arr.reshape(-1)
        else:
            input_shape = _normalise_onnx_shape(getattr(input_meta, "shape", None), int(input_flat.size))
            expected = int(np.prod(input_shape)) if input_shape else int(input_flat.size)
            if expected != int(input_flat.size):
                status["reason"] = (
                    f"input.bin word count {int(input_flat.size)} does not match ONNX input shape "
                    f"{input_shape} ({expected} words)"
                )
                return status
            x = input_flat.reshape(input_shape).astype(np.float32)
            y = np.asarray(sess.run([output_meta.name], {input_meta.name: x})[0], dtype=np.float32).reshape(-1)
            sample_count = 1
            output_shape_per_sample = (int(y.size),)

        ref_dir.mkdir(parents=True, exist_ok=True)
        np.save(ref_input_npy, x)
        np.save(ref_output_npy, y.reshape((sample_count, -1)) if dataset_available else y)
        y.tofile(ref_output_bin)
        status.update({
            "status": "available",
            "reason": "ONNX reference output generated from normalized dataset artifacts." if dataset_available else "ONNX reference output generated from the same input.bin used by HLS CSim.",
            "inputs_bin": dataset_artifacts.get("inputs_bin") if dataset_available else input_bin,
            "outputs_hw": output_bin,
            "outputs_ref": ref_output_bin,
            "outputs_ref_npy": ref_output_npy,
            "inputs_ref_npy": ref_input_npy,
            "input_shape": input_shape,
            "output_words": int(y.size),
            "sample_count": sample_count,
            "output_shape_per_sample": output_shape_per_sample,
            "dataset": {k: v for k, v in dataset_artifacts.items() if k != "inputs_array"},
            "labels_path": dataset_artifacts.get("labels_path"),
            "targets_path": dataset_artifacts.get("targets_path"),
        })
        return status
    except Exception as exc:
        status["reason"] = f"failed to generate ONNX reference output: {exc}"
        return status
