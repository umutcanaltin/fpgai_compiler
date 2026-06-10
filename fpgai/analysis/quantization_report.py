from __future__ import annotations

import copy
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from fpgai.numerics.fixed_emulation import (
    cosine_similarity,
    mae,
    max_abs,
    mse,
    quantize_array,
)
from fpgai.numerics.precision_policy import (
    canonical_op_type,
    default_precision_policy,
    resolve_precision_for_op,
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


def _cfg_get(
    data: Dict[str, Any],
    path: str,
    default: Any = None,
) -> Any:
    current: Any = data

    for key in path.split("."):
        if (
            not isinstance(current, dict)
            or key not in current
        ):
            return default

        current = current[key]

    return current


def _stable_node_aliases(
    model: onnx.ModelProto,
) -> Dict[int, List[str]]:
    counters: Dict[str, int] = {}
    aliases: Dict[int, List[str]] = {}

    for index, node in enumerate(model.graph.node):
        canonical_type = canonical_op_type(
            node.op_type
        )
        key = canonical_type.lower()
        ordinal = counters.get(key, 0)
        counters[key] = ordinal + 1

        aliases[index] = [
            f"{key}{ordinal}",
        ]

    return aliases


def _precision_for_node(
    raw_cfg: Dict[str, Any],
    node,
    index: int,
    *,
    name_aliases: List[str] | None = None,
) -> Dict[str, Any]:
    resolved = resolve_precision_for_op(
        raw_cfg,
        node,
        index,
        name_aliases=name_aliases or (),
    )

    return resolved.specs


def _session_outputs_all_tensors(
    model: onnx.ModelProto,
) -> onnx.ModelProto:
    result = copy.deepcopy(model)

    existing = {
        output.name
        for output in result.graph.output
    }
    value_info_names = {
        value_info.name
        for value_info in result.graph.value_info
    }
    input_names = {
        graph_input.name
        for graph_input in result.graph.input
    }
    original_output_names = {
        output.name
        for output in result.graph.output
    }

    for node in result.graph.node:
        for output_name in node.output:
            if not output_name:
                continue

            if output_name in existing:
                continue

            if output_name not in value_info_names:
                continue

            if output_name in input_names:
                continue

            if output_name in original_output_names:
                continue

            result.graph.output.append(
                helper.make_tensor_value_info(
                    output_name,
                    TensorProto.FLOAT,
                    None,
                )
            )
            existing.add(output_name)

    return result


def _make_random_input_from_onnx(
    model: onnx.ModelProto,
    seed: int = 0,
) -> Tuple[str, np.ndarray]:
    if not model.graph.input:
        raise RuntimeError(
            "Model has no inputs"
        )

    initializer_names = {
        initializer.name
        for initializer in model.graph.initializer
    }

    graph_inputs = [
        graph_input
        for graph_input in model.graph.input
        if graph_input.name not in initializer_names
    ]

    if not graph_inputs:
        raise RuntimeError(
            "Model has no non-initializer inputs"
        )

    graph_input = graph_inputs[0]
    dimensions = []

    for dimension in (
        graph_input.type.tensor_type.shape.dim
    ):
        if (
            dimension.dim_value
            and int(dimension.dim_value) > 0
        ):
            dimensions.append(
                int(dimension.dim_value)
            )
        else:
            dimensions.append(1)

    if not dimensions:
        dimensions = [1]

    generator = np.random.default_rng(seed)
    values = generator.standard_normal(
        size=dimensions
    ).astype(np.float32)

    return graph_input.name, values


def _load_or_make_input(
    raw_cfg: Dict[str, Any],
    model: onnx.ModelProto,
) -> Tuple[str, np.ndarray]:
    input_npy = _cfg_get(
        raw_cfg,
        "analysis.quantization_report.input_npy",
    )
    seed = int(
        _cfg_get(
            raw_cfg,
            "analysis.quantization_report.seed",
            0,
        )
    )

    if input_npy:
        path = Path(str(input_npy))

        if not path.is_file():
            raise FileNotFoundError(
                "Quantization report input file "
                f"does not exist: {path}"
            )

        values = np.load(path).astype(
            np.float32
        )

        initializer_names = {
            initializer.name
            for initializer in model.graph.initializer
        }
        input_names = [
            graph_input.name
            for graph_input in model.graph.input
            if graph_input.name not in initializer_names
        ]

        if not input_names:
            raise RuntimeError(
                "Model has no non-initializer inputs"
            )

        return input_names[0], values

    return _make_random_input_from_onnx(
        model,
        seed=seed,
    )


def _run_model_outputs(
    model: onnx.ModelProto,
    input_name: str,
    values: np.ndarray,
) -> Dict[str, np.ndarray]:
    session = ort.InferenceSession(
        model.SerializeToString(),
        providers=[
            "CPUExecutionProvider",
        ],
    )

    output_names = [
        output.name
        for output in session.get_outputs()
    ]

    output_values = session.run(
        output_names,
        {
            input_name: values.astype(
                np.float32
            ),
        },
    )

    return {
        name: np.asarray(
            value,
            dtype=np.float32,
        )
        for name, value in zip(
            output_names,
            output_values,
        )
    }


def _safe_top1(
    values: np.ndarray,
) -> int:
    flattened = np.asarray(
        values
    ).reshape(-1)

    if flattened.size == 0:
        return -1

    return int(
        np.argmax(flattened)
    )


def _initializer_map(
    model: onnx.ModelProto,
) -> Dict[str, onnx.TensorProto]:
    return {
        initializer.name: initializer
        for initializer in model.graph.initializer
    }


def _is_float_initializer(
    initializer: onnx.TensorProto,
) -> bool:
    return initializer.data_type in {
        TensorProto.FLOAT,
        TensorProto.FLOAT16,
        TensorProto.DOUBLE,
        TensorProto.BFLOAT16,
    }


def _quantize_numeric_initializers(
    raw_cfg: Dict[str, Any],
    model: onnx.ModelProto,
) -> onnx.ModelProto:
    result = copy.deepcopy(model)
    initializers = _initializer_map(result)
    stable_aliases = _stable_node_aliases(
        result
    )

    for index, node in enumerate(
        result.graph.node
    ):
        if node.op_type not in NUMERIC_WEIGHT_OPS:
            continue

        precision = _precision_for_node(
            raw_cfg,
            node,
            index,
            name_aliases=stable_aliases.get(
                index,
            ),
        )

        float_initializer_inputs = []

        for input_name in node.input:
            initializer = initializers.get(
                input_name
            )

            if (
                initializer is not None
                and _is_float_initializer(
                    initializer
                )
            ):
                float_initializer_inputs.append(
                    input_name
                )

        for parameter_index, initializer_name in enumerate(
            float_initializer_inputs
        ):
            initializer = initializers[
                initializer_name
            ]
            values = numpy_helper.to_array(
                initializer
            )

            if not np.issubdtype(
                values.dtype,
                np.floating,
            ):
                continue

            if parameter_index == 0:
                spec = precision["weight"]
            else:
                spec = precision["bias"]

            quantized = quantize_array(
                values.astype(np.float32),
                spec,
            )

            if values.dtype == np.float16:
                quantized = quantized.astype(
                    np.float16
                )
            elif values.dtype == np.float64:
                quantized = quantized.astype(
                    np.float64
                )
            else:
                quantized = quantized.astype(
                    np.float32
                )

            replacement = numpy_helper.from_array(
                quantized,
                initializer_name,
            )
            initializer.CopyFrom(replacement)

    return result


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
    model_path = Path(model_path)
    output_root = Path(out_dir).resolve()
    report_dir = output_root / "quant_report"

    if report_dir.exists():
        for path in report_dir.glob("**/*"):
            if path.is_file():
                path.unlink()

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = onnx.load(
        str(model_path)
    )
    model = onnx.shape_inference.infer_shapes(
        model
    )

    stable_aliases = _stable_node_aliases(
        model
    )
    defaults = default_precision_policy(
        raw_cfg
    )

    reference_model = (
        _session_outputs_all_tensors(model)
    )

    quantized_model_base = (
        _quantize_numeric_initializers(
            raw_cfg,
            model,
        )
    )

    quantized_model = (
        _session_outputs_all_tensors(
            quantized_model_base
        )
    )

    input_name, input_values = (
        _load_or_make_input(
            raw_cfg,
            model,
        )
    )

    input_spec = defaults["activation"]
    quantized_input = quantize_array(
        input_values,
        input_spec,
    )

    reference_outputs = _run_model_outputs(
        reference_model,
        input_name,
        input_values,
    )

    quantized_outputs = _run_model_outputs(
        quantized_model,
        input_name,
        quantized_input,
    )

    common_names = sorted(
        name
        for name in reference_outputs
        if name in quantized_outputs
    )

    layer_rows: List[Dict[str, Any]] = []

    for index, node in enumerate(
        model.graph.node
    ):
        precision = _precision_for_node(
            raw_cfg,
            node,
            index,
            name_aliases=stable_aliases.get(
                index,
            ),
        )

        for output_name in node.output:
            if (
                output_name not in reference_outputs
                or output_name not in quantized_outputs
            ):
                continue

            reference_value = reference_outputs[
                output_name
            ]
            quantized_value = quantized_outputs[
                output_name
            ]

            activation_spec = precision[
                "activation"
            ]

            emulated_activation = quantize_array(
                quantized_value,
                activation_spec,
            )

            row = {
                "layer_index": index,
                "layer_name": (
                    node.name
                    or f"{node.op_type}_{index}"
                ),
                "stable_name": (
                    stable_aliases.get(
                        index,
                        [""],
                    )[0]
                ),
                "op_type": node.op_type,
                "canonical_op_type": (
                    canonical_op_type(
                        node.op_type
                    )
                ),
                "tensor_name": output_name,
                "act_bits": int(
                    activation_spec["total_bits"]
                ),
                "act_int_bits": int(
                    activation_spec["int_bits"]
                ),
                "wgt_bits": int(
                    precision["weight"][
                        "total_bits"
                    ]
                ),
                "wgt_int_bits": int(
                    precision["weight"][
                        "int_bits"
                    ]
                ),
                "bias_bits": int(
                    precision["bias"][
                        "total_bits"
                    ]
                ),
                "bias_int_bits": int(
                    precision["bias"][
                        "int_bits"
                    ]
                ),
                "acc_bits": int(
                    precision["accum"][
                        "total_bits"
                    ]
                ),
                "acc_int_bits": int(
                    precision["accum"][
                        "int_bits"
                    ]
                ),
                "mse": mse(
                    reference_value,
                    emulated_activation,
                ),
                "mae": mae(
                    reference_value,
                    emulated_activation,
                ),
                "max_abs": max_abs(
                    reference_value,
                    emulated_activation,
                ),
                "cosine": cosine_similarity(
                    reference_value,
                    emulated_activation,
                ),
                "float_min": float(
                    np.min(reference_value)
                ),
                "float_max": float(
                    np.max(reference_value)
                ),
                "quant_min": float(
                    np.min(emulated_activation)
                ),
                "quant_max": float(
                    np.max(emulated_activation)
                ),
            }

            layer_rows.append(row)
            break

    if not model.graph.output:
        raise RuntimeError(
            "Model has no graph outputs"
        )

    final_output_name = (
        model.graph.output[0].name
    )

    if final_output_name not in reference_outputs:
        raise RuntimeError(
            "Reference output is missing model "
            f"output: {final_output_name}"
        )

    if final_output_name not in quantized_outputs:
        raise RuntimeError(
            "Quantized output is missing model "
            f"output: {final_output_name}"
        )

    reference_final = reference_outputs[
        final_output_name
    ]
    quantized_final = quantized_outputs[
        final_output_name
    ]

    final_producer_index = None

    for index, node in enumerate(
        model.graph.node
    ):
        if final_output_name in node.output:
            final_producer_index = index

    if final_producer_index is not None:
        final_node = model.graph.node[
            final_producer_index
        ]
        final_precision = _precision_for_node(
            raw_cfg,
            final_node,
            final_producer_index,
            name_aliases=stable_aliases.get(
                final_producer_index,
            ),
        )
        quantized_final = quantize_array(
            quantized_final,
            final_precision["activation"],
        )

    final_metrics = {
        "output_mse": mse(
            reference_final,
            quantized_final,
        ),
        "output_mae": mae(
            reference_final,
            quantized_final,
        ),
        "output_max_abs": max_abs(
            reference_final,
            quantized_final,
        ),
        "output_cosine": cosine_similarity(
            reference_final,
            quantized_final,
        ),
        "float_top1": _safe_top1(
            reference_final
        ),
        "quant_top1": _safe_top1(
            quantized_final
        ),
        "prediction_match": (
            _safe_top1(reference_final)
            == _safe_top1(quantized_final)
        ),
    }

    worst_layer = None

    if layer_rows:
        worst_layer = max(
            layer_rows,
            key=lambda row: row["mse"],
        )

    metrics = {
        "model_path": str(model_path),
        "quantized_input_used": True,
        "initializer_quantization_used": True,
        "activation_output_quantization_used": True,
        "accumulator_emulation_used": False,
        "accumulator_emulation_note": (
            "Accumulator widths are reported but "
            "full operator-level accumulator "
            "emulation is not implemented yet."
        ),
        "num_compared_tensors": len(
            common_names
        ),
        "num_compared_layers": len(
            layer_rows
        ),
        "defaults": defaults,
        "final": final_metrics,
        "worst_layer": worst_layer,
        "layers": layer_rows,
    }

    metrics_json = (
        report_dir / "metrics.json"
    )
    summary_txt = (
        report_dir / "summary.txt"
    )
    layerwise_csv = (
        report_dir / "layerwise.csv"
    )

    metrics_json.write_text(
        json.dumps(
            metrics,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    csv_fields = [
        "layer_index",
        "layer_name",
        "stable_name",
        "op_type",
        "canonical_op_type",
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
    ]

    with layerwise_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=csv_fields,
        )

        writer.writeheader()
        writer.writerows(layer_rows)

    lines = [
        "=============== FPGAI Quantization Report ===============",
        f"Model path         : {model_path}",
        f"Compared layers    : {len(layer_rows)}",
        f"Compared tensors   : {len(common_names)}",
        "Input quantized    : True",
        "Weights quantized  : True",
        "Activations rounded: True",
        "Accumulator model  : Not yet implemented",
        "---------------------------------------------------------",
        (
            "Final output MSE   : "
            f"{final_metrics['output_mse']:.8f}"
        ),
        (
            "Final output MAE   : "
            f"{final_metrics['output_mae']:.8f}"
        ),
        (
            "Final max abs      : "
            f"{final_metrics['output_max_abs']:.8f}"
        ),
        (
            "Final cosine       : "
            f"{final_metrics['output_cosine']:.8f}"
        ),
        (
            "Float top1         : "
            f"{final_metrics['float_top1']}"
        ),
        (
            "Quant top1         : "
            f"{final_metrics['quant_top1']}"
        ),
        (
            "Prediction match   : "
            f"{final_metrics['prediction_match']}"
        ),
    ]

    if worst_layer is not None:
        lines.extend(
            [
                "---------------------------------------------------------",
                (
                    "Worst layer        : "
                    f"{worst_layer['layer_name']} "
                    f"({worst_layer['op_type']})"
                ),
                (
                    "Worst layer MSE    : "
                    f"{worst_layer['mse']:.8f}"
                ),
                (
                    "Worst layer MAE    : "
                    f"{worst_layer['mae']:.8f}"
                ),
                (
                    "Worst layer MaxAbs : "
                    f"{worst_layer['max_abs']:.8f}"
                ),
            ]
        )

    lines.extend(
        [
            "---------------------------------------------------------",
            f"Metrics JSON       : {metrics_json}",
            f"Layerwise CSV      : {layerwise_csv}",
            "=========================================================",
        ]
    )

    summary_txt.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    return QuantizationReportResult(
        out_dir=report_dir,
        metrics_json=metrics_json,
        summary_txt=summary_txt,
        layerwise_csv=layerwise_csv,
        passed=True,
    )