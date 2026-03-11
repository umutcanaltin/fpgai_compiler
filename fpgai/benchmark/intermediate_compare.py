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
        # assume NCHW with batch=1
        if a.shape[0] != 1:
            raise ValueError(f"Expected batch=1 for 4D tensor, got shape={a.shape}")
        # [1, C, H, W] -> [H, W, C]
        a = np.transpose(a[0], (1, 2, 0))
        return a.reshape(-1)

    if a.ndim == 3:
        # ambiguous, but if encountered assume already CHW -> HWC
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
        # Comparing these directly is misleading, so skip for now.
        raise ValueError("skip_layout_sensitive_reshape")

    return np.asarray(ref, dtype=np.float32).reshape(-1)


def _find_hls_layer_dump(csim_build_dir: Path, layer_name: str) -> Path | None:
    p = csim_build_dir / f"{layer_name}.bin"
    return p if p.exists() else None


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

        metrics = compare_outputs(
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
            **asdict(metrics),
        }
        layer_results.append(item)

        if first_bad_layer is None and not metrics.passed:
            first_bad_layer = op.name

    layer_results.sort(key=lambda x: x["max_abs_error"], reverse=True)

    summary = {
        "num_layers_compared": len(layer_results),
        "num_layers_skipped": len(skipped_layers),
        "first_bad_layer": first_bad_layer,
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
    lines.append("")
    lines.append("Worst layers:")
    for i, item in enumerate(layer_results[:10], start=1):
        lines.append(
            f"{i}. {item['layer_name']} ({item['op_type']}) "
            f"max={item['max_abs_error']:.8f} "
            f"mean={item['mean_abs_error']:.8f} "
            f"rmse={item['rmse']:.8f} "
            f"cos={item['cosine_similarity']:.8f} "
            f"passed={item['passed']}"
        )

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