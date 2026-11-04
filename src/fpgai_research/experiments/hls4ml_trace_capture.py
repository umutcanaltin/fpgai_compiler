"""Capture actual hls4ml ModelGraph.trace(X) layer outputs for X07.

Use a separate trace-enabled native project clone. ModelGraph.trace may run
compilation; never invoke against immutable original Vitis evidence.
This capture is not independent E2 or original-weight attestation.
"""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np


def capture_hls4ml_trace(hls_model, inputs_file, original_onnx_file,
                         output_dir, *, expected_samples=16):
    paths = [Path(inputs_file), Path(original_onnx_file)]
    for path in paths:
        if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 128 * 1024 * 1024:
            raise ValueError("Missing or unsafe trace source")
    if type(expected_samples) is not int or not 1 <= expected_samples <= 10000:
        raise ValueError("Invalid sample count")
    x = np.load(paths[0], allow_pickle=False)
    if x.dtype != np.float32 or x.ndim < 2 or x.shape[0] != expected_samples or not np.isfinite(x).all():
        raise ValueError("Expected ordered float32 cohort")
    target = Path(output_dir)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Trace destination must be new")
    prediction, traces = hls_model.trace(x)
    if not isinstance(traces, dict) or not 1 <= len(traces) <= 128:
        raise ValueError("Trace-enabled layers missing")
    clean = {}
    for name, value in traces.items():
        if not isinstance(name, str) or not name or "/" in name or "\\" in name:
            raise ValueError("Invalid layer name")
        a = np.asarray(value)
        if a.ndim < 2 or a.shape[0] != expected_samples or a.size == 0 or a.dtype.kind not in "fiu" or not np.isfinite(a).all():
            raise ValueError("Invalid layer tensor")
        clean[name] = a
    pred = np.asarray(prediction)
    if pred.shape != (expected_samples, 10) or pred.dtype.kind != "f" or not np.isfinite(pred).all():
        raise ValueError("Invalid raw output shape or contents")
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".hls4ml-trace-", dir=target.parent) as tmp:
        folder = Path(tmp) / "export"
        folder.mkdir()
        np.savez(folder / "hls4ml_trace.npz", **clean)
        np.save(folder / "hls4ml_raw_predictions.npy", pred, allow_pickle=False)
        receipt = {
            "schema": "fpgai-research/hls4ml-operator-trace-capture/v1",
            "original_onnx_sha256": sha256(paths[1].read_bytes()).hexdigest(),
            "ordered_input_cohort_sha256": sha256(paths[0].read_bytes()).hexdigest(),
            "hls4ml_trace_npz_sha256": sha256((folder / "hls4ml_trace.npz").read_bytes()).hexdigest(),
            "raw_prediction_npy_sha256": sha256((folder / "hls4ml_raw_predictions.npy").read_bytes()).hexdigest(),
            "sample_count": len(x),
            "layer_names_and_shapes": {n: list(v.shape) for n, v in clean.items()},
            "native_log_origin_independently_attested": False,
            "original_to_native_weight_identity_attested": False,
            "operator_semantic_crosswalk_qualified": False,
            "numeric_e2_qualified": False,
            "causal_first_bad_operator_proven": False,
        }
        (folder / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
        if target.exists() or target.is_symlink():
            raise FileExistsError("Trace output appeared")
        folder.rename(target)
    return receipt
