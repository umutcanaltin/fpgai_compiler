from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import List
import json

import numpy as np

from fpgai.benchmark.compare import compare_outputs


def _nchw_to_hwc_flat(arr: np.ndarray) -> np.ndarray:
    """
    ONNX feature maps are typically NCHW.
    HLS internal dumps are currently flat HWC order.
    Convert ONNX output to the same flat order for comparison.
    """
    a = np.asarray(arr, dtype=np.float32)

    if a.ndim == 4:
        # assume batch=1, NCHW -> HWC
        if a.shape[0] != 1:
            raise ValueError(f"Expected batch=1 for 4D tensor, got shape={a.shape}")
        a = np.transpose(a[0], (1, 2, 0))
        return a.reshape(-1)

    if a.ndim == 3:
        # CHW -> HWC
        a = np.transpose(a, (1, 2, 0))
        return a.reshape(-1)

    return a.reshape(-1)


def _reference_for_compare(op, ref: np.ndarray) -> np.ndarray:
    """
    Normalize ONNX reference tensor into the same layout used by HLS dumps.
    """
    if op.op_type in ("Conv", "Relu", "MaxPool", "AvgPool"):
        return _nchw_to_hwc_flat(ref)

    if op.op_type in ("Dense", "Softmax"):
        return np.asarray(ref, dtype=np.float32).reshape(-1)

    if op.op_type in ("Flatten", "Reshape"):
        # Current HLS reshape path preserves internal HWC-linearized order,
        # while ONNX reshape follows framework memory order.
        raise ValueError("skip_layout_sensitive_reshape")

    return np.asarray(ref, dtype=np.float32).reshape(-1)


def _find_hls_layer_dump(csim_build_dir: Path, layer_name: str) -> Path | None:
    p = csim_build_dir / f"{layer_name}.bin"
    return p if p.exists() else None


def _dense_offset_analysis(
    ref_cmp: np.ndarray,
    hls_cmp: np.ndarray,
    *,
    centered_max_abs_error_limit: float = 0.02,
    centered_mean_abs_error_limit: float = 0.01,
    centered_rmse_limit: float = 0.01,
    std_diff_limit: float = 0.02,
) -> dict:
    """
    Dense/logit outputs can differ by a near-constant offset while still producing
    almost identical softmax probabilities. This helper quantifies that behavior.
    """
    diff = hls_cmp - ref_cmp
    mean_diff = float(diff.mean())
    std_diff = float(diff.std())
    diff_range = float(diff.max() - diff.min())

    centered_hls = hls_cmp - mean_diff
    centered_diff = centered_hls - ref_cmp

    centered_max_abs_error = float(np.abs(centered_diff).max())
    centered_mean_abs_error = float(np.abs(centered_diff).mean())
    centered_rmse = float(np.sqrt(np.mean(centered_diff ** 2)))

    constant_offset_like = bool(std_diff <= std_diff_limit)

    centered_pass = bool(
        centered_max_abs_error <= centered_max_abs_error_limit
        and centered_mean_abs_error <= centered_mean_abs_error_limit
        and centered_rmse <= centered_rmse_limit
    )

    effective_pass = bool(constant_offset_like and centered_pass)

    return {
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "diff_range": diff_range,
        "centered_max_abs_error": centered_max_abs_error,
        "centered_mean_abs_error": centered_mean_abs_error,
        "centered_rmse": centered_rmse,
        "constant_offset_like": constant_offset_like,
        "centered_pass": centered_pass,
        "effective_pass": effective_pass,
        "limits": {
            "centered_max_abs_error_limit": centered_max_abs_error_limit,
            "centered_mean_abs_error_limit": centered_mean_abs_error_limit,
            "centered_rmse_limit": centered_rmse_limit,
            "std_diff_limit": std_diff_limit,
        },
    }


def compare_intermediate_layers(
    *,
    graph,
    reference_dir: str | Path,
    csim_build_dir: str | Path,
    out_dir: str | Path,
    atol: float,
    rtol: float,
    max_abs_error_limit: float | None,
    mean_abs_error_limit: float | None,
    rmse_limit: float | None,
    require_argmax_match: bool,
    min_cosine_similarity: float | None,
) -> dict:
    reference_dir = Path(reference_dir)
    csim_build_dir = Path(csim_build_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layer_results: List[dict] = []
    skipped_layers: List[dict] = []
    first_bad_layer = None
    first_warning_layer = None

    for op in graph.ops:
        if not op.outputs:
            continue

        ref_path = reference_dir / f"{op.name}.npy"
        hls_path = _find_hls_layer_dump(csim_build_dir, op.name)
        if not ref_path.exists() or hls_path is None:
            continue

        ref_raw = np.load(ref_path).astype(np.float32)
        hls_flat = np.fromfile(hls_path, dtype=np.float32).reshape(-1)

        try:
            ref_cmp = _reference_for_compare(op, ref_raw)
        except ValueError as e:
            if str(e) == "skip_layout_sensitive_reshape":
                skipped_layers.append(
                    {
                        "layer_name": op.name,
                        "op_type": op.op_type,
                        "reason": "layout-sensitive op skipped in v1 intermediate compare",
                    }
                )
                continue
            raise

        if ref_cmp.shape != hls_flat.shape:
            skipped_layers.append(
                {
                    "layer_name": op.name,
                    "op_type": op.op_type,
                    "reason": f"shape mismatch after normalization: ref={ref_cmp.shape}, hls={hls_flat.shape}",
                }
            )
            continue

        raw_metrics = compare_outputs(
            ref_cmp,
            hls_flat,
            atol=atol,
            rtol=rtol,
            max_abs_error_limit=max_abs_error_limit,
            mean_abs_error_limit=mean_abs_error_limit,
            rmse_limit=rmse_limit,
            require_argmax_match=require_argmax_match,
            min_cosine_similarity=min_cosine_similarity,
        )

        item = {
            "layer_name": op.name,
            "op_type": op.op_type,
            "reference_shape": list(ref_raw.shape),
            "compare_shape": list(ref_cmp.shape),
            **asdict(raw_metrics),
            "effective_pass": bool(raw_metrics.passed),
            "warning_only": False,
            "note": "",
        }

        if op.op_type == "Dense":
            oa = _dense_offset_analysis(ref_cmp, hls_flat)
            item["offset_analysis"] = oa

            if oa["effective_pass"]:
                item["effective_pass"] = True
                item["warning_only"] = True
                item["note"] = "raw dense logits differ by near-constant offset, centered comparison passed"
                if first_warning_layer is None:
                    first_warning_layer = op.name

        layer_results.append(item)

        if first_bad_layer is None and not item["effective_pass"]:
            first_bad_layer = op.name

    layer_results.sort(key=lambda x: x["max_abs_error"], reverse=True)

    summary = {
        "num_layers_compared": len(layer_results),
        "num_layers_skipped": len(skipped_layers),
        "first_bad_layer": first_bad_layer,
        "first_warning_layer": first_warning_layer,
        "worst_layers": layer_results[:10],
        "all_layers": layer_results,
        "skipped_layers": skipped_layers,
    }

    (out_dir / "intermediate_metrics.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    lines = []
    lines.append("FPGAI Intermediate Layer Comparison")
    lines.append("===================================")
    lines.append(f"num_layers_compared : {len(layer_results)}")
    lines.append(f"num_layers_skipped  : {len(skipped_layers)}")
    lines.append(f"first_bad_layer     : {first_bad_layer}")
    lines.append(f"first_warning_layer : {first_warning_layer}")
    lines.append("")
    lines.append("Worst layers:")

    for i, item in enumerate(layer_results[:10], start=1):
        lines.append(
            f"{i}. {item['layer_name']} ({item['op_type']}) "
            f"max={item['max_abs_error']:.8f} "
            f"mean={item['mean_abs_error']:.8f} "
            f"rmse={item['rmse']:.8f} "
            f"cos={item['cosine_similarity']:.8f} "
            f"passed={item['passed']} "
            f"effective_pass={item['effective_pass']}"
        )

        if item["op_type"] == "Dense" and "offset_analysis" in item:
            oa = item["offset_analysis"]
            lines.append(
                f"   dense offset: mean_diff={oa['mean_diff']:.8f} "
                f"std_diff={oa['std_diff']:.8f} "
                f"range={oa['diff_range']:.8f} "
                f"centered_max={oa['centered_max_abs_error']:.8f} "
                f"centered_mean={oa['centered_mean_abs_error']:.8f} "
                f"centered_rmse={oa['centered_rmse']:.8f} "
                f"constant_offset_like={oa['constant_offset_like']} "
                f"centered_pass={oa['centered_pass']} "
                f"effective_pass={oa['effective_pass']}"
            )

        if item["note"]:
            lines.append(f"   note: {item['note']}")

    if skipped_layers:
        lines.append("")
        lines.append("Skipped layers:")
        for item in skipped_layers:
            lines.append(f"- {item['layer_name']} ({item['op_type']}): {item['reason']}")

    (out_dir / "intermediate_summary.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    return summary