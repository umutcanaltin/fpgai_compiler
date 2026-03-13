from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import copy
import csv
import json

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper, TensorProto

from fpgai.numerics.fixed_emulation import (
    quantize_array,
    mse,
    mae,
    max_abs,
    cosine_similarity,
)


NUMERIC_WEIGHT_OPS = {
    "Conv",
    "Gemm",
    "MatMul",
    "BatchNormalization",
    "Add",
    "Sub",
    "Mul",
    "Div",
}


def _cfg_get(d: Dict[str, Any], path: str, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _default_spec(raw: Dict[str, Any], key: str, tb: int, ib: int) -> Dict[str, Any]:
    v = _cfg_get(raw, f"numerics.defaults.{key}", None)
    if isinstance(v, dict) and v.get("type") == "ap_fixed":
        return {
            "type": "ap_fixed",
            "total_bits": int(v.get("total_bits", tb)),
            "int_bits": int(v.get("int_bits", ib)),
        }
    return {"type": "ap_fixed", "total_bits": tb, "int_bits": ib}


def _match_rule(rule_match: Dict[str, Any], node, idx: int) -> bool:
    if "name" in rule_match and str(rule_match["name"]) != str(node.name):
        return False
    if "op_type" in rule_match and str(rule_match["op_type"]) != str(node.op_type):
        return False
    if "index" in rule_match and int(rule_match["index"]) != idx:
        return False
    return True


def _precision_for_node(raw_cfg: Dict[str, Any], node, idx: int) -> Dict[str, Any]:
    defaults = {
        "activation": _default_spec(raw_cfg, "activation", 16, 6),
        "weight": _default_spec(raw_cfg, "weight", 16, 6),
        "bias": _default_spec(raw_cfg, "bias", 24, 10),
        "accum": _default_spec(raw_cfg, "accum", 24, 10),
    }
    p = copy.deepcopy(defaults)
    rules = _cfg_get(raw_cfg, "numerics.layers", []) or []
    for rule in rules:
        match = rule.get("match", {})
        if not isinstance(match, dict):
            continue
        if not _match_rule(match, node, idx):
            continue
        for k in ["activation", "weight", "bias", "accum"]:
            if k in rule and isinstance(rule[k], dict):
                p[k] = {
                    "type": "ap_fixed",
                    "total_bits": int(rule[k]["total_bits"]),
                    "int_bits": int(rule[k]["int_bits"]),
                }
    return p


def _session_outputs_all_tensors(model: onnx.ModelProto) -> onnx.ModelProto:
    m = copy.deepcopy(model)
    existing = {o.name for o in m.graph.output}
    vi_names = {vi.name for vi in m.graph.value_info}
    inp_names = {i.name for i in m.graph.input}
    out_names = {o.name for o in m.graph.output}

    for node in m.graph.node:
        for out in node.output:
            if out and out not in existing and out in vi_names and out not in inp_names and out not in out_names:
                m.graph.output.append(helper.make_tensor_value_info(out, onnx.TensorProto.FLOAT, None))
                existing.add(out)
    return m


def _make_random_input_from_onnx(model: onnx.ModelProto, seed: int = 0) -> Tuple[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    if len(model.graph.input) == 0:
        raise RuntimeError("Model has no inputs")
    inp = model.graph.input[0]
    name = inp.name

    dims = []
    for d in inp.type.tensor_type.shape.dim:
        if d.dim_value and int(d.dim_value) > 0:
            dims.append(int(d.dim_value))
        else:
            dims.append(1)

    x = rng.standard_normal(size=dims).astype(np.float32)
    return name, x


def _load_or_make_input(raw_cfg: Dict[str, Any], model: onnx.ModelProto) -> Tuple[str, np.ndarray]:
    input_npy = _cfg_get(raw_cfg, "analysis.quantization_report.input_npy", None)
    seed = int(_cfg_get(raw_cfg, "analysis.quantization_report.seed", 0))
    if input_npy:
        x = np.load(str(input_npy)).astype(np.float32)
        name = model.graph.input[0].name
        return name, x
    return _make_random_input_from_onnx(model, seed=seed)


def _run_model_outputs(model: onnx.ModelProto, input_name: str, x: np.ndarray) -> Dict[str, np.ndarray]:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    output_names = [o.name for o in sess.get_outputs()]
    vals = sess.run(output_names, {input_name: x.astype(np.float32)})
    return {name: np.asarray(v, dtype=np.float32) for name, v in zip(output_names, vals)}


def _safe_top1(y: np.ndarray) -> int:
    z = np.asarray(y).reshape(-1)
    if z.size == 0:
        return -1
    return int(np.argmax(z))


def _initializer_map(model: onnx.ModelProto) -> Dict[str, onnx.TensorProto]:
    return {init.name: init for init in model.graph.initializer}


def _is_float_initializer(init: onnx.TensorProto) -> bool:
    return init.data_type in {
        TensorProto.FLOAT,
        TensorProto.FLOAT16,
        TensorProto.DOUBLE,
        TensorProto.BFLOAT16,
    }


def _quantize_numeric_initializers(raw_cfg: Dict[str, Any], model: onnx.ModelProto) -> onnx.ModelProto:
    m = copy.deepcopy(model)
    init_map = _initializer_map(m)

    for idx, node in enumerate(m.graph.node):
        if node.op_type not in NUMERIC_WEIGHT_OPS:
            continue

        p = _precision_for_node(raw_cfg, node, idx)

        # For numeric ops, only quantize initializer inputs that are already floating-point.
        # Never touch integer/control tensors used by structural ops like Reshape.
        float_init_inputs = []
        for name in node.input:
            init = init_map.get(name)
            if init is not None and _is_float_initializer(init):
                float_init_inputs.append(name)

        for j, init_name in enumerate(float_init_inputs):
            init = init_map[init_name]
            arr = numpy_helper.to_array(init)

            if not np.issubdtype(arr.dtype, np.floating):
                continue

            spec = p["weight"] if j == 0 else p["bias"]
            q = quantize_array(arr.astype(np.float32), spec)

            if arr.dtype == np.float16:
                q = q.astype(np.float16)
            elif arr.dtype == np.float64:
                q = q.astype(np.float64)
            else:
                q = q.astype(np.float32)

            new_init = numpy_helper.from_array(q, init_name)
            init.CopyFrom(new_init)

    return m


@dataclass(frozen=True)
class QuantizationReportResult:
    out_dir: Path
    metrics_json: Path
    summary_txt: Path
    layerwise_csv: Path
    passed: bool


def run_quantization_report(
    *,
    model_path: str | Path,
    raw_cfg: Dict[str, Any],
    out_dir: str | Path,
) -> QuantizationReportResult:
    out_dir = Path(out_dir).resolve()
    qdir = out_dir / "quant_report"
    if qdir.exists():
        for p in qdir.glob("**/*"):
            if p.is_file():
                p.unlink()
    qdir.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(model_path))
    model = onnx.shape_inference.infer_shapes(model)

    ref_model = _session_outputs_all_tensors(model)
    q_model_base = _quantize_numeric_initializers(raw_cfg, model)
    q_model = _session_outputs_all_tensors(q_model_base)

    input_name, x = _load_or_make_input(raw_cfg, model)
    act_spec = _default_spec(raw_cfg, "activation", 16, 6)
    xq = quantize_array(x, act_spec)

    ref_outs = _run_model_outputs(ref_model, input_name, x)
    q_outs = _run_model_outputs(q_model, input_name, xq)

    common_names = [n for n in ref_outs.keys() if n in q_outs]
    common_names_sorted = sorted(common_names)

    layer_rows: List[Dict[str, Any]] = []
    for idx, node in enumerate(model.graph.node):
        p = _precision_for_node(raw_cfg, node, idx)
        for out_name in node.output:
            if out_name not in ref_outs or out_name not in q_outs:
                continue
            a = ref_outs[out_name]
            b = q_outs[out_name]
            row = {
                "layer_index": idx,
                "layer_name": node.name or f"{node.op_type}_{idx}",
                "op_type": node.op_type,
                "tensor_name": out_name,
                "act_bits": int(p["activation"]["total_bits"]),
                "act_int_bits": int(p["activation"]["int_bits"]),
                "wgt_bits": int(p["weight"]["total_bits"]),
                "wgt_int_bits": int(p["weight"]["int_bits"]),
                "bias_bits": int(p["bias"]["total_bits"]),
                "bias_int_bits": int(p["bias"]["int_bits"]),
                "acc_bits": int(p["accum"]["total_bits"]),
                "acc_int_bits": int(p["accum"]["int_bits"]),
                "mse": mse(a, b),
                "mae": mae(a, b),
                "max_abs": max_abs(a, b),
                "cosine": cosine_similarity(a, b),
                "float_min": float(np.min(a)),
                "float_max": float(np.max(a)),
                "quant_min": float(np.min(b)),
                "quant_max": float(np.max(b)),
            }
            layer_rows.append(row)
            break

    ref_final_name = model.graph.output[0].name
    ref_y = ref_outs[ref_final_name]
    q_y = q_outs[ref_final_name]

    final_metrics = {
        "output_mse": mse(ref_y, q_y),
        "output_mae": mae(ref_y, q_y),
        "output_max_abs": max_abs(ref_y, q_y),
        "output_cosine": cosine_similarity(ref_y, q_y),
        "float_top1": _safe_top1(ref_y),
        "quant_top1": _safe_top1(q_y),
        "prediction_match": (_safe_top1(ref_y) == _safe_top1(q_y)),
    }

    worst_layer = None
    if layer_rows:
        worst_layer = max(layer_rows, key=lambda r: r["mse"])

    metrics = {
        "model_path": str(model_path),
        "quantized_input_used": True,
        "num_compared_tensors": len(common_names_sorted),
        "num_compared_layers": len(layer_rows),
        "defaults": {
            "activation": act_spec,
            "weight": _default_spec(raw_cfg, "weight", 16, 6),
            "bias": _default_spec(raw_cfg, "bias", 24, 10),
            "accum": _default_spec(raw_cfg, "accum", 24, 10),
        },
        "final": final_metrics,
        "worst_layer": worst_layer,
        "layers": layer_rows,
    }

    metrics_json = qdir / "metrics.json"
    summary_txt = qdir / "summary.txt"
    layerwise_csv = qdir / "layerwise.csv"

    metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    with layerwise_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "layer_index",
                "layer_name",
                "op_type",
                "tensor_name",
                "act_bits",
                "act_int_bits",
                "wgt_bits",
                "wgt_int_bits",
                "bias_bits",
                "bias_int_bits",
                "acc_bits",
                "acc_int_bits",
                "mse",
                "mae",
                "max_abs",
                "cosine",
                "float_min",
                "float_max",
                "quant_min",
                "quant_max",
            ],
        )
        writer.writeheader()
        for row in layer_rows:
            writer.writerow(row)

    lines = []
    lines.append("=============== FPGAI Quantization Report ===============")
    lines.append(f"Model path         : {model_path}")
    lines.append(f"Compared layers    : {len(layer_rows)}")
    lines.append(f"Compared tensors   : {len(common_names_sorted)}")
    lines.append("---------------------------------------------------------")
    lines.append(f"Final output MSE   : {final_metrics['output_mse']:.8f}")
    lines.append(f"Final output MAE   : {final_metrics['output_mae']:.8f}")
    lines.append(f"Final max abs      : {final_metrics['output_max_abs']:.8f}")
    lines.append(f"Final cosine       : {final_metrics['output_cosine']:.8f}")
    lines.append(f"Float top1         : {final_metrics['float_top1']}")
    lines.append(f"Quant top1         : {final_metrics['quant_top1']}")
    lines.append(f"Prediction match   : {final_metrics['prediction_match']}")
    if worst_layer is not None:
        lines.append("---------------------------------------------------------")
        lines.append(f"Worst layer        : {worst_layer['layer_name']} ({worst_layer['op_type']})")
        lines.append(f"Worst layer MSE    : {worst_layer['mse']:.8f}")
        lines.append(f"Worst layer MAE    : {worst_layer['mae']:.8f}")
        lines.append(f"Worst layer MaxAbs : {worst_layer['max_abs']:.8f}")
    lines.append("---------------------------------------------------------")
    lines.append(f"Metrics JSON       : {metrics_json}")
    lines.append(f"Layerwise CSV      : {layerwise_csv}")
    lines.append("=========================================================")

    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return QuantizationReportResult(
        out_dir=qdir,
        metrics_json=metrics_json,
        summary_txt=summary_txt,
        layerwise_csv=layerwise_csv,
        passed=True,
    )