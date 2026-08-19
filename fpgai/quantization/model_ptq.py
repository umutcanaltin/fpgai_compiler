from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from fpgai.quantization.contracts import QuantizationParameters, QuantizationSpec
from fpgai.quantization.ptq import calibrate_ptq, dequantize, quantize
from fpgai.quantization.validation import QuantizationValidationResult, validate_fake_quantization


TraceFunction = Callable[[Any, np.ndarray], Mapping[str, np.ndarray]]


@dataclass(frozen=True)
class TensorPTQResult:
    tensor: str
    role: str
    parameters: QuantizationParameters
    validation: QuantizationValidationResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "tensor": self.tensor,
            "role": self.role,
            "parameters": self.parameters.to_dict(),
            "validation": self.validation.to_dict(),
        }


@dataclass(frozen=True)
class ModelPTQResult:
    sample_count: int
    method: str
    activations: tuple[TensorPTQResult, ...]
    weights: tuple[TensorPTQResult, ...]
    biases: tuple[TensorPTQResult, ...]
    quantized_constants: Mapping[str, np.ndarray]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "fpgai.model-ptq/v1",
            "sample_count": self.sample_count,
            "method": self.method,
            "activations": [result.to_dict() for result in self.activations],
            "weights": [result.to_dict() for result in self.weights],
            "biases": [result.to_dict() for result in self.biases],
            "accumulators": {
                "policy": "operator_lowering",
                "description": "Conv biases are materialized in the accumulator domain during quantized hardware lowering.",
            },
            "quantized_constants": {
                name: {
                    "shape": list(np.asarray(values).shape),
                    "dtype": str(np.asarray(values).dtype),
                }
                for name, values in sorted(self.quantized_constants.items())
            },
        }


def _attach_quantization(graph: Any, tensor_name: str, parameters: QuantizationParameters) -> None:
    setter = getattr(graph, "set_tensor_quantization", None)
    if setter is None:
        raise ValueError("graph does not support tensor quantization metadata")
    if tensor_name not in getattr(graph, "tensors", {}):
        raise ValueError(f"quantized tensor {tensor_name!r} is missing from the FPGAI graph tensor table")
    setter(tensor_name, parameters.to_dict())


def calibrate_model_ptq(
    graph: Any,
    samples: Iterable[np.ndarray],
    *,
    trace_fn: TraceFunction,
    activation_spec: QuantizationSpec,
    weight_spec: QuantizationSpec,
    method: str = "min_max",
    percentile: float = 99.99,
    sample_limit: int | None = None,
) -> ModelPTQResult:
    """Calibrate a graph and attach resolved PTQ parameters to its tensors.

    ``trace_fn`` is intentionally caller supplied. It keeps PTQ independent from
    any one frontend/reference executor while still allowing the compiler to use
    the same execution owner it already trusts for model validation.
    """
    if activation_spec.granularity != "per_tensor":
        raise ValueError("activation PTQ currently requires per_tensor granularity")

    calibration_samples = tuple(np.asarray(sample, dtype=np.float32) for sample in samples)
    if sample_limit is not None:
        if type(sample_limit) is not int or sample_limit <= 0:
            raise ValueError("sample_limit must be a positive integer")
        calibration_samples = calibration_samples[:sample_limit]
    if not calibration_samples:
        raise ValueError("model PTQ calibration requires at least one sample")

    traces: list[Mapping[str, np.ndarray]] = []
    for sample in calibration_samples:
        trace = trace_fn(graph, sample)
        if not isinstance(trace, Mapping):
            raise ValueError("PTQ trace function must return a tensor-name mapping")
        traces.append(trace)

    constant_names = set(getattr(graph, "constants", {}) or {})
    activation_names = [
        name for name in getattr(graph, "tensors", {})
        if name not in constant_names
    ]

    activation_results: list[TensorPTQResult] = []
    for name in activation_names:
        observed = [np.asarray(trace[name], dtype=np.float32) for trace in traces if name in trace]
        if not observed:
            continue
        calibration = calibrate_ptq(observed, activation_spec, method=method, percentile=percentile)
        stacked = np.concatenate([value.reshape(-1) for value in observed])
        validation = validate_fake_quantization(stacked, calibration.parameters)
        _attach_quantization(graph, name, calibration.parameters)
        activation_results.append(TensorPTQResult(name, "activation", calibration.parameters, validation))

    conv_bias_names = {
        str(op.inputs[2])
        for op in getattr(graph, "ops", ()) or ()
        if str(getattr(op, "op_type", "")) == "Conv" and len(getattr(op, "inputs", ()) or ()) > 2
    }
    weight_results: list[TensorPTQResult] = []
    bias_results: list[TensorPTQResult] = []
    quantized_constants: dict[str, np.ndarray] = {}
    for name, value in (getattr(graph, "constants", {}) or {}).items():
        array = np.asarray(value, dtype=np.float32)
        calibration = calibrate_ptq((array,), weight_spec, method=method, percentile=percentile)
        validation = validate_fake_quantization(array, calibration.parameters)
        _attach_quantization(graph, name, calibration.parameters)
        quantized_constants[name] = quantize(array, calibration.parameters)
        if name in conv_bias_names:
            bias_results.append(TensorPTQResult(name, "bias", calibration.parameters, validation))
        else:
            weight_results.append(TensorPTQResult(name, "weight", calibration.parameters, validation))

    return ModelPTQResult(
        sample_count=len(calibration_samples),
        method=method,
        activations=tuple(activation_results),
        weights=tuple(weight_results),
        biases=tuple(bias_results),
        quantized_constants=quantized_constants,
    )


def dequantized_constant(result: ModelPTQResult, tensor_name: str) -> np.ndarray:
    params = next((entry.parameters for entry in (*result.weights, *result.biases) if entry.tensor == tensor_name), None)
    if params is None or tensor_name not in result.quantized_constants:
        raise KeyError(f"No calibrated PTQ constant {tensor_name!r}")
    return dequantize(result.quantized_constants[tensor_name], params)
