from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from fpgai.quantization.contracts import QuantizationParameters, QuantizationSpec
from fpgai.quantization.hardware import derive_requantization_contract, quantization_parameters_from_tensor
from fpgai.quantization.model_ptq import ModelPTQResult
from fpgai.quantization.model_qat import ModelQATResult
from fpgai.quantization.ptq import quantize


@dataclass(frozen=True)
class QuantizedHLSLoweringResult:
    quantized_constants: tuple[str, ...]
    quantized_conv_nodes: tuple[str, ...]
    quantized_add_nodes: tuple[str, ...]
    quantized_relu_nodes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "fpgai.quantized-hls-lowering/v1",
            "quantized_constants": list(self.quantized_constants),
            "quantized_conv_nodes": list(self.quantized_conv_nodes),
            "quantized_add_nodes": list(self.quantized_add_nodes),
            "quantized_relu_nodes": list(self.quantized_relu_nodes),
        }


def _as_tuple(value: Any) -> tuple[Any, ...]:
    return value if isinstance(value, tuple) else (value,)


def _int_precision(bits: int) -> dict[str, Any]:
    return {"type": "ap_int", "bits": int(bits), "total_bits": int(bits), "int_bits": int(bits)}


def _rounding_code(name: str) -> int:
    return {"nearest": 0, "floor": 1, "ceil": 2}[str(name)]


def _saturation_code(name: str) -> int:
    return {"saturate": 0, "wrap": 1}[str(name)]


def _scalar_param(parameters: QuantizationParameters, channel: int | None = None) -> tuple[float, int]:
    scales = _as_tuple(parameters.scale)
    zeros = _as_tuple(parameters.zero_point)
    index = 0 if channel is None else int(channel)
    if index >= len(scales) or index >= len(zeros):
        raise ValueError("quantization channel index exceeds parameter count")
    return float(scales[index]), int(zeros[index])


def _accumulator_parameters(scale: float) -> QuantizationParameters:
    spec = QuantizationSpec(bits=32, scheme="symmetric", granularity="per_tensor", signed=True)
    return QuantizationParameters(
        spec=spec,
        scale=float(scale),
        zero_point=0,
        observed_min=float(spec.qmin) * float(scale),
        observed_max=float(spec.qmax) * float(scale),
    )


def _conv_output_channels(weight: np.ndarray) -> int:
    array = np.asarray(weight)
    if array.ndim != 4:
        raise ValueError("quantized HLS Conv requires OIHW rank-4 weights")
    return int(array.shape[0])


def _result_parameters(result: ModelPTQResult | ModelQATResult, tensor_name: str) -> QuantizationParameters | None:
    for item in (*result.weights, *result.biases):
        if item.tensor == tensor_name:
            return item.parameters
    for item in result.activations:
        if item.tensor == tensor_name:
            return item.parameters
    return None


def apply_quantized_model_to_hls_graph(graph: Any, result: ModelPTQResult | ModelQATResult) -> QuantizedHLSLoweringResult:
    """Materialize exported PTQ/QAT semantics into the FPGAI HLS graph.

    This initial lowering supports signed integer Conv/Relu/Add inference. Conv
    weights may be per-channel on axis 0; activations remain per-tensor. Biases
    are converted into the integer accumulator domain instead of being quantized
    with the weight scale directly.
    """
    constants = getattr(graph, "constants", {}) or {}
    quantized_names: list[str] = []
    conv_nodes: list[str] = []
    add_nodes: list[str] = []
    relu_nodes: list[str] = []

    # Quantize non-bias constants first. Biases are handled in accumulator units
    # during Conv lowering below.
    conv_bias_names = {
        str(op.inputs[2])
        for op in getattr(graph, "ops", []) or []
        if str(getattr(op, "op_type", "")) == "Conv" and len(getattr(op, "inputs", []) or []) > 2
    }
    for name, values in list(constants.items()):
        if name in conv_bias_names:
            continue
        parameters = _result_parameters(result, name)
        if parameters is None:
            continue
        constants[name] = quantize(np.asarray(values, dtype=np.float32), parameters)
        quantized_names.append(name)

    for op in getattr(graph, "ops", []) or []:
        attrs = getattr(op, "attrs", {}) or {}
        op.attrs = attrs
        runtime_inputs = [str(name) for name in getattr(op, "inputs", []) or [] if str(name) not in constants]
        outputs = [str(name) for name in getattr(op, "outputs", []) or []]
        if not outputs:
            continue

        if op.op_type == "Conv":
            if len(runtime_inputs) != 1 or len(op.inputs) < 2:
                raise ValueError(f"quantized Conv {op.name!r} requires one runtime input and embedded weights")
            input_q = quantization_parameters_from_tensor(graph.get_tensor(runtime_inputs[0]))
            output_q = quantization_parameters_from_tensor(graph.get_tensor(outputs[0]))
            weight_name = str(op.inputs[1])
            weight_q = quantization_parameters_from_tensor(graph.get_tensor(weight_name))
            if input_q.spec.granularity != "per_tensor" or output_q.spec.granularity != "per_tensor":
                raise ValueError("quantized HLS Conv currently requires per-tensor activation quantization")
            if not input_q.spec.signed or not output_q.spec.signed or not weight_q.spec.signed:
                raise ValueError("quantized HLS Conv currently requires signed integer quantization")
            if weight_q.spec.granularity not in {"per_tensor", "per_channel"}:
                raise ValueError("unsupported Conv weight quantization granularity")
            if weight_q.spec.granularity == "per_channel" and int(weight_q.spec.axis or 0) != 0:
                raise ValueError("quantized HLS Conv requires per-channel weight axis 0")

            weight_values = np.asarray(constants[weight_name], dtype=np.int64)
            oc = _conv_output_channels(weight_values)
            weight_scales = _as_tuple(weight_q.scale)
            weight_zeros = _as_tuple(weight_q.zero_point)
            if len(weight_scales) not in {1, oc} or len(weight_zeros) not in {1, oc}:
                raise ValueError("Conv weight quantization parameters do not match output channels")

            input_scale, input_zero = _scalar_param(input_q)
            multipliers: list[int] = []
            shifts: list[int] = []
            expanded_weight_zeros: list[int] = []
            for channel in range(oc):
                weight_scale = float(weight_scales[channel if len(weight_scales) > 1 else 0])
                weight_zero = int(weight_zeros[channel if len(weight_zeros) > 1 else 0])
                accumulator_q = _accumulator_parameters(input_scale * weight_scale)
                contract = derive_requantization_contract(accumulator_q, output_q)
                multipliers.append(int(contract.multiplier))
                shifts.append(int(contract.shift))
                expanded_weight_zeros.append(weight_zero)

            bias_name = str(op.inputs[2]) if len(op.inputs) > 2 else None
            if bias_name and bias_name in constants:
                bias_float = np.asarray(constants[bias_name], dtype=np.float64).reshape(-1)
                if bias_float.size not in {1, oc}:
                    raise ValueError("Conv bias size does not match output channels")
                bias_int = []
                for channel in range(oc):
                    weight_scale = float(weight_scales[channel if len(weight_scales) > 1 else 0])
                    scale = input_scale * weight_scale
                    raw = float(bias_float[channel if bias_float.size > 1 else 0]) / scale
                    bias_int.append(int(np.rint(raw)))
                constants[bias_name] = np.asarray(bias_int, dtype=np.int64)
                graph.set_tensor_quantization(
                    bias_name,
                    _accumulator_parameters(input_scale * float(weight_scales[0])).to_dict(),
                )
                quantized_names.append(bias_name)

            attrs["precision"] = {
                "activation": _int_precision(output_q.spec.bits),
                "weight": _int_precision(weight_q.spec.bits),
                "bias": _int_precision(32),
                "accum": _int_precision(32),
            }
            attrs["quantized_conv"] = {
                "input_zero": int(input_zero),
                "weight_zero": expanded_weight_zeros,
                "output_zero": int(_scalar_param(output_q)[1]),
                "multipliers": multipliers,
                "shifts": shifts,
                "qmin": int(output_q.spec.qmin),
                "qmax": int(output_q.spec.qmax),
                "rounding_mode": _rounding_code(output_q.spec.rounding),
                "saturation_mode": _saturation_code(output_q.spec.saturation),
            }
            conv_nodes.append(str(op.name))

        elif op.op_type == "Add":
            if len(runtime_inputs) != 2:
                raise ValueError(f"quantized Add {op.name!r} requires two runtime inputs")
            left_q = quantization_parameters_from_tensor(graph.get_tensor(runtime_inputs[0]))
            right_q = quantization_parameters_from_tensor(graph.get_tensor(runtime_inputs[1]))
            output_q = quantization_parameters_from_tensor(graph.get_tensor(outputs[0]))
            for q in (left_q, right_q, output_q):
                if q.spec.granularity != "per_tensor" or not q.spec.signed:
                    raise ValueError("quantized HLS Add currently requires signed per-tensor quantization")
            left_contract = derive_requantization_contract(left_q, output_q)
            right_contract = derive_requantization_contract(right_q, output_q)
            attrs["precision"] = {
                "activation": _int_precision(output_q.spec.bits),
                "weight": _int_precision(output_q.spec.bits),
                "bias": _int_precision(32),
                "accum": _int_precision(32),
            }
            attrs["quantized_add"] = {
                "left_zero": int(_scalar_param(left_q)[1]),
                "left_multiplier": int(left_contract.multiplier),
                "left_shift": int(left_contract.shift),
                "right_zero": int(_scalar_param(right_q)[1]),
                "right_multiplier": int(right_contract.multiplier),
                "right_shift": int(right_contract.shift),
                "output_zero": int(_scalar_param(output_q)[1]),
                "qmin": int(output_q.spec.qmin),
                "qmax": int(output_q.spec.qmax),
                "rounding_mode": _rounding_code(output_q.spec.rounding),
                "saturation_mode": _saturation_code(output_q.spec.saturation),
            }
            add_nodes.append(str(op.name))

        elif op.op_type == "Relu":
            output_q = quantization_parameters_from_tensor(graph.get_tensor(outputs[0]))
            input_q = quantization_parameters_from_tensor(graph.get_tensor(runtime_inputs[0]))
            if input_q.spec.granularity != "per_tensor" or output_q.spec.granularity != "per_tensor":
                raise ValueError("quantized HLS Relu currently requires per-tensor quantization")
            contract = derive_requantization_contract(input_q, output_q)
            attrs["precision"] = {
                "activation": _int_precision(output_q.spec.bits),
                "weight": _int_precision(output_q.spec.bits),
                "bias": _int_precision(32),
                "accum": _int_precision(32),
            }
            attrs["quantized_relu"] = {
                "input_zero": int(_scalar_param(input_q)[1]),
                "multiplier": int(contract.multiplier),
                "shift": int(contract.shift),
                "output_zero": int(_scalar_param(output_q)[1]),
                "qmin": int(output_q.spec.qmin),
                "qmax": int(output_q.spec.qmax),
                "rounding_mode": _rounding_code(output_q.spec.rounding),
                "saturation_mode": _saturation_code(output_q.spec.saturation),
            }
            relu_nodes.append(str(op.name))

        else:
            raise ValueError(f"quantized HLS lowering does not yet support operator {op.op_type!r}")

    return QuantizedHLSLoweringResult(
        quantized_constants=tuple(sorted(set(quantized_names))),
        quantized_conv_nodes=tuple(conv_nodes),
        quantized_add_nodes=tuple(add_nodes),
        quantized_relu_nodes=tuple(relu_nodes),
    )



def apply_model_ptq_to_hls_graph(graph: Any, result: ModelPTQResult) -> QuantizedHLSLoweringResult:
    """Compatibility wrapper for the original PTQ API."""
    return apply_quantized_model_to_hls_graph(graph, result)


def apply_model_qat_to_hls_graph(graph: Any, result: ModelQATResult) -> QuantizedHLSLoweringResult:
    """Lower frozen QAT export through the same quantized HLS path as PTQ."""
    return apply_quantized_model_to_hls_graph(graph, result)

def _round_shift(value: np.ndarray, shift: int, rounding_mode: int) -> np.ndarray:
    values = np.asarray(value, dtype=np.int64)
    if shift <= 0:
        return values
    divisor = 1 << int(shift)
    if rounding_mode == 1:
        return np.floor_divide(values, divisor)
    if rounding_mode == 2:
        return -np.floor_divide(-values, divisor)
    positive = np.floor_divide(values + divisor // 2, divisor)
    negative = -np.floor_divide((-values) + divisor // 2, divisor)
    return np.where(values >= 0, positive, negative).astype(np.int64)


def _requant_array(centered: np.ndarray, *, multiplier: int, shift: int, output_zero: int, qmin: int, qmax: int, rounding_mode: int, saturation_mode: int) -> np.ndarray:
    shifted = _round_shift(np.asarray(centered, dtype=np.int64) * int(multiplier), int(shift), int(rounding_mode))
    result = shifted + int(output_zero)
    if int(saturation_mode) == 0:
        return np.clip(result, int(qmin), int(qmax)).astype(np.int64)
    width = int(qmax) - int(qmin) + 1
    return (((result - int(qmin)) % width) + int(qmin)).astype(np.int64)


def execute_quantized_hls_reference(graph: Any, quantized_input: np.ndarray) -> np.ndarray:
    """Execute the integer semantics emitted by ``apply_model_ptq_to_hls_graph``."""
    if len(getattr(graph, "inputs", []) or []) != 1 or len(getattr(graph, "outputs", []) or []) != 1:
        raise ValueError("quantized HLS reference requires one graph input and one graph output")
    input_name = str(graph.inputs[0])
    input_spec = graph.get_tensor(input_name)
    x = np.asarray(quantized_input, dtype=np.int64)
    if getattr(input_spec, "shape", None):
        x = x.reshape(tuple(int(v) for v in input_spec.shape))
    tensors: dict[str, np.ndarray] = {input_name: x}
    constants = getattr(graph, "constants", {}) or {}

    for op in getattr(graph, "ops", []) or []:
        runtime_inputs = [str(name) for name in op.inputs if str(name) not in constants]
        output_name = str(op.outputs[0])
        attrs = getattr(op, "attrs", {}) or {}
        if op.op_type == "Conv":
            qcfg = attrs.get("quantized_conv")
            if not isinstance(qcfg, Mapping):
                raise ValueError(f"Conv node {op.name!r} is missing quantized lowering metadata")
            source = np.asarray(tensors[runtime_inputs[0]], dtype=np.int64)
            weights = np.asarray(constants[str(op.inputs[1])], dtype=np.int64)
            bias = np.asarray(constants[str(op.inputs[2])], dtype=np.int64).reshape(-1) if len(op.inputs) > 2 else np.zeros((weights.shape[0],), dtype=np.int64)
            n, cin, ih, iw = source.shape
            cout, wcin, kh, kw = weights.shape
            if n != 1 or cin != wcin:
                raise ValueError("quantized HLS reference Conv expects N=1 and matching channels")
            strides = tuple(int(v) for v in attrs.get("strides", (1, 1)))
            pads = tuple(int(v) for v in attrs.get("pads", (0, 0, 0, 0)))
            sh, sw = strides
            pt, pl, pb, pr = pads
            oh = (ih + pt + pb - kh) // sh + 1
            ow = (iw + pl + pr - kw) // sw + 1
            padded = np.pad(source, ((0, 0), (0, 0), (pt, pb), (pl, pr)), constant_values=int(qcfg["input_zero"]))
            out = np.zeros((1, cout, oh, ow), dtype=np.int64)
            for co in range(cout):
                for oy in range(oh):
                    for ox in range(ow):
                        acc = int(bias[co])
                        for ci in range(cin):
                            for ky in range(kh):
                                for kx in range(kw):
                                    xv = int(padded[0, ci, oy * sh + ky, ox * sw + kx]) - int(qcfg["input_zero"])
                                    wv = int(weights[co, ci, ky, kx]) - int(qcfg["weight_zero"][co])
                                    acc += xv * wv
                        out[0, co, oy, ox] = _requant_array(
                            np.asarray([acc]), multiplier=int(qcfg["multipliers"][co]), shift=int(qcfg["shifts"][co]),
                            output_zero=int(qcfg["output_zero"]), qmin=int(qcfg["qmin"]), qmax=int(qcfg["qmax"]),
                            rounding_mode=int(qcfg["rounding_mode"]), saturation_mode=int(qcfg["saturation_mode"]),
                        )[0]
            tensors[output_name] = out
        elif op.op_type == "Relu":
            qcfg = attrs.get("quantized_relu")
            source = np.asarray(tensors[runtime_inputs[0]], dtype=np.int64)
            clamped = np.maximum(source, int(qcfg["input_zero"]))
            tensors[output_name] = _requant_array(
                clamped - int(qcfg["input_zero"]), multiplier=int(qcfg["multiplier"]), shift=int(qcfg["shift"]),
                output_zero=int(qcfg["output_zero"]), qmin=int(qcfg["qmin"]), qmax=int(qcfg["qmax"]),
                rounding_mode=int(qcfg["rounding_mode"]), saturation_mode=int(qcfg["saturation_mode"]),
            )
        elif op.op_type == "Add":
            qcfg = attrs.get("quantized_add")
            left = np.asarray(tensors[runtime_inputs[0]], dtype=np.int64)
            right = np.asarray(tensors[runtime_inputs[1]], dtype=np.int64)
            left_q = _requant_array(
                left - int(qcfg["left_zero"]), multiplier=int(qcfg["left_multiplier"]), shift=int(qcfg["left_shift"]),
                output_zero=int(qcfg["output_zero"]), qmin=int(qcfg["qmin"]), qmax=int(qcfg["qmax"]),
                rounding_mode=int(qcfg["rounding_mode"]), saturation_mode=int(qcfg["saturation_mode"]),
            )
            right_q = _requant_array(
                right - int(qcfg["right_zero"]), multiplier=int(qcfg["right_multiplier"]), shift=int(qcfg["right_shift"]),
                output_zero=int(qcfg["output_zero"]), qmin=int(qcfg["qmin"]), qmax=int(qcfg["qmax"]),
                rounding_mode=int(qcfg["rounding_mode"]), saturation_mode=int(qcfg["saturation_mode"]),
            )
            summed = left_q + right_q - int(qcfg["output_zero"])
            if int(qcfg["saturation_mode"]) == 0:
                summed = np.clip(summed, int(qcfg["qmin"]), int(qcfg["qmax"]))
            else:
                width = int(qcfg["qmax"]) - int(qcfg["qmin"]) + 1
                summed = ((summed - int(qcfg["qmin"])) % width) + int(qcfg["qmin"])
            tensors[output_name] = summed.astype(np.int64)
        else:
            raise ValueError(f"quantized HLS reference does not support {op.op_type!r}")
    return np.asarray(tensors[str(graph.outputs[0])], dtype=np.int64)
