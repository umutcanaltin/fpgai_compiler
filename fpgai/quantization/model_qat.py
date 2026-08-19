from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from fpgai.quantization.contracts import QuantizationParameters, QuantizationSpec, quantization_spec_from_mapping
from fpgai.quantization.observers import MinMaxObserver, PercentileObserver
from fpgai.quantization.parameters import derive_quantization_parameters
from fpgai.quantization.ptq import fake_quantize, quantize
from fpgai.quantization.qat import straight_through_gradient
from fpgai.quantization.validation import QuantizationValidationResult, validate_fake_quantization


Observer = MinMaxObserver | PercentileObserver


@dataclass(frozen=True)
class QATSchedule:
    fake_quant: bool = True
    straight_through_estimator: bool = True
    freeze_after_updates: int | None = None

    def __post_init__(self) -> None:
        if self.freeze_after_updates is not None:
            if type(self.freeze_after_updates) is not int or self.freeze_after_updates < 0:
                raise ValueError("freeze_after_updates must be a non-negative integer or None")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "QATSchedule":
        raw = raw or {}
        if not isinstance(raw, Mapping):
            raise ValueError("QAT schedule must be a mapping")
        for key in ("fake_quant", "straight_through_estimator"):
            if key in raw and not isinstance(raw[key], bool):
                raise ValueError(f"QAT schedule {key} must be a boolean")
        freeze = raw.get("freeze_after_updates")
        if freeze is not None and type(freeze) is not int:
            raise ValueError("QAT schedule freeze_after_updates must be an integer or None")
        return cls(
            fake_quant=raw.get("fake_quant", True),
            straight_through_estimator=raw.get("straight_through_estimator", True),
            freeze_after_updates=freeze,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fake_quant": self.fake_quant,
            "straight_through_estimator": self.straight_through_estimator,
            "freeze_after_updates": self.freeze_after_updates,
        }


@dataclass
class QATTensorState:
    tensor: str
    role: str
    spec: QuantizationSpec
    observer: Observer
    parameters: QuantizationParameters | None = None
    observed_values: np.ndarray | None = None
    observer_updates: int = 0
    frozen: bool = False

    def observe(self, values: np.ndarray) -> QuantizationParameters:
        array = np.asarray(values, dtype=np.float32)
        if not self.frozen:
            self.observer.observe(array)
            minimum, maximum = self.observer.range()
            self.parameters = derive_quantization_parameters(minimum, maximum, self.spec)
            self.observed_values = array.copy()
            self.observer_updates += 1
        if self.parameters is None:
            raise ValueError(f"QAT tensor {self.tensor!r} has no observer-derived quantization parameters")
        return self.parameters

    def fake_quant(self, values: np.ndarray, *, enabled: bool = True) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        parameters = self.observe(array)
        if not enabled:
            return array.copy()
        return fake_quantize(array, parameters)

    def freeze(self) -> QuantizationParameters:
        if self.parameters is None:
            raise ValueError(f"cannot freeze QAT tensor {self.tensor!r} before it has been observed")
        self.frozen = True
        return self.parameters


@dataclass(frozen=True)
class TensorQATResult:
    tensor: str
    role: str
    parameters: QuantizationParameters
    validation: QuantizationValidationResult
    observer_updates: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "tensor": self.tensor,
            "role": self.role,
            "parameters": self.parameters.to_dict(),
            "validation": self.validation.to_dict(),
            "observer_updates": self.observer_updates,
        }


@dataclass(frozen=True)
class ModelQATResult:
    optimizer_updates: int
    schedule: QATSchedule
    activations: tuple[TensorQATResult, ...]
    weights: tuple[TensorQATResult, ...]
    biases: tuple[TensorQATResult, ...]
    quantized_constants: Mapping[str, np.ndarray]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "fpgai.model-qat/v1",
            "optimizer_updates": self.optimizer_updates,
            "schedule": self.schedule.to_dict(),
            "activations": [entry.to_dict() for entry in self.activations],
            "weights": [entry.to_dict() for entry in self.weights],
            "biases": [entry.to_dict() for entry in self.biases],
            "accumulators": {
                "policy": "operator_lowering",
                "description": "Conv biases are materialized in the accumulator domain during quantized hardware lowering.",
            },
            "quantized_constants": {
                name: {"shape": list(np.asarray(values).shape), "dtype": str(np.asarray(values).dtype)}
                for name, values in sorted(self.quantized_constants.items())
            },
        }


class ModelQATSession:
    """Stateful, backend-independent QAT owner.

    The session keeps graph/master tensors in floating point, exposes fake-quant
    views for forward execution, routes gradients through the existing STE
    contract, updates observers until the configured optimizer-update boundary,
    and exports ordinary FPGAI tensor quantization metadata.  Hardware lowering
    therefore consumes the same quantized IR contract used by PTQ.
    """

    def __init__(
        self,
        *,
        activation_spec: QuantizationSpec,
        weight_spec: QuantizationSpec,
        schedule: QATSchedule | None = None,
        observer_method: str = "min_max",
        percentile: float = 99.99,
    ) -> None:
        if activation_spec.granularity != "per_tensor":
            raise ValueError("QAT activations currently require per_tensor granularity")
        if observer_method not in {"min_max", "percentile"}:
            raise ValueError("QAT observer_method must be 'min_max' or 'percentile'")
        self.activation_spec = activation_spec
        self.weight_spec = weight_spec
        self.schedule = schedule or QATSchedule()
        self.observer_method = observer_method
        self.percentile = float(percentile)
        self.optimizer_updates = 0
        self._states: dict[str, QATTensorState] = {}

    def _observer(self, spec: QuantizationSpec) -> Observer:
        axis = spec.axis if spec.granularity == "per_channel" else None
        if self.observer_method == "min_max":
            return MinMaxObserver(axis=axis)
        return PercentileObserver(percentile=self.percentile, axis=axis)

    def register_tensor(self, tensor: str, *, role: str) -> QATTensorState:
        if role not in {"activation", "weight", "bias"}:
            raise ValueError("QAT tensor role must be activation, weight, or bias")
        spec = self.activation_spec if role == "activation" else self.weight_spec
        existing = self._states.get(tensor)
        if existing is not None:
            if existing.role != role or existing.spec != spec:
                raise ValueError(f"QAT tensor {tensor!r} is already registered with a different contract")
            return existing
        state = QATTensorState(tensor=tensor, role=role, spec=spec, observer=self._observer(spec))
        self._states[tensor] = state
        return state

    def fake_quant_activation(self, tensor: str, values: np.ndarray) -> np.ndarray:
        return self.register_tensor(tensor, role="activation").fake_quant(values, enabled=self.schedule.fake_quant)

    def observe_activation_trace(self, trace: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Observe/fake-quant a tensor-name activation trace without mutating it."""
        return {
            str(name): self.fake_quant_activation(str(name), np.asarray(values, dtype=np.float32))
            for name, values in trace.items()
        }

    def fake_quant_weight(self, tensor: str, master_values: np.ndarray, *, role: str = "weight") -> np.ndarray:
        # Never mutate master_values: optimizers continue to own floating-point
        # master weights while the forward path sees their fake-quantized view.
        return self.register_tensor(tensor, role=role).fake_quant(
            np.asarray(master_values, dtype=np.float32), enabled=self.schedule.fake_quant
        )

    def fake_quant_graph_constants(self, graph: Any) -> dict[str, np.ndarray]:
        """Return fake-quant views of trainable constants while preserving master floats."""
        conv_bias_names = {
            str(op.inputs[2])
            for op in getattr(graph, "ops", ()) or ()
            if str(getattr(op, "op_type", "")) == "Conv" and len(getattr(op, "inputs", ()) or ()) > 2
        }
        result: dict[str, np.ndarray] = {}
        for name, values in (getattr(graph, "constants", {}) or {}).items():
            role = "bias" if str(name) in conv_bias_names else "weight"
            result[str(name)] = self.fake_quant_weight(str(name), np.asarray(values, dtype=np.float32), role=role)
        return result

    def backward_gradient(self, gradient: np.ndarray) -> np.ndarray:
        return straight_through_gradient(gradient, enabled=self.schedule.straight_through_estimator)

    def complete_optimizer_update(self) -> None:
        self.optimizer_updates += 1
        freeze_at = self.schedule.freeze_after_updates
        if freeze_at is not None and self.optimizer_updates >= freeze_at:
            self.freeze_observers()

    def freeze_observers(self) -> None:
        for state in self._states.values():
            state.freeze()

    @property
    def observers_frozen(self) -> bool:
        return bool(self._states) and all(state.frozen for state in self._states.values())

    def _result_for_state(self, state: QATTensorState) -> TensorQATResult:
        parameters = state.freeze()
        if state.observed_values is None:
            raise ValueError(f"QAT tensor {state.tensor!r} has no observed values for validation")
        validation = validate_fake_quantization(state.observed_values, parameters)
        return TensorQATResult(
            tensor=state.tensor,
            role=state.role,
            parameters=parameters,
            validation=validation,
            observer_updates=state.observer_updates,
        )

    def export_to_graph(self, graph: Any) -> ModelQATResult:
        if not self._states:
            raise ValueError("QAT export requires at least one registered tensor")
        setter = getattr(graph, "set_tensor_quantization", None)
        if setter is None:
            raise ValueError("graph does not support tensor quantization metadata")

        activation_results: list[TensorQATResult] = []
        weight_results: list[TensorQATResult] = []
        bias_results: list[TensorQATResult] = []
        quantized_constants: dict[str, np.ndarray] = {}
        constants = getattr(graph, "constants", {}) or {}

        for name in sorted(self._states):
            state = self._states[name]
            if name not in getattr(graph, "tensors", {}):
                raise ValueError(f"QAT tensor {name!r} is missing from the FPGAI graph tensor table")
            result = self._result_for_state(state)
            setter(name, result.parameters.to_dict())
            if state.role == "activation":
                activation_results.append(result)
            elif state.role == "bias":
                bias_results.append(result)
            else:
                weight_results.append(result)
            if name in constants:
                quantized_constants[name] = quantize(np.asarray(constants[name], dtype=np.float32), result.parameters)

        return ModelQATResult(
            optimizer_updates=self.optimizer_updates,
            schedule=self.schedule,
            activations=tuple(activation_results),
            weights=tuple(weight_results),
            biases=tuple(bias_results),
            quantized_constants=quantized_constants,
        )


def model_qat_session_from_config(raw_cfg: Mapping[str, Any]) -> ModelQATSession:
    """Build the QAT state owner from the existing public YAML contract."""
    numerics = raw_cfg.get("numerics", {}) if isinstance(raw_cfg, Mapping) else {}
    quant = numerics.get("quantization", {}) if isinstance(numerics, Mapping) else {}
    if not isinstance(quant, Mapping) or str(quant.get("mode", "none")) != "qat":
        raise ValueError("model_qat_session_from_config requires numerics.quantization.mode=qat")
    activation_spec = quantization_spec_from_mapping(
        quant.get("activations", {}), path="numerics.quantization.activations"
    )
    weight_spec = quantization_spec_from_mapping(
        quant.get("weights", {}), path="numerics.quantization.weights"
    )
    schedule = QATSchedule.from_mapping(quant.get("qat", {}))
    calibration = quant.get("calibration", {})
    if calibration is None:
        calibration = {}
    if not isinstance(calibration, Mapping):
        raise ValueError("numerics.quantization.calibration must be a mapping when present")
    return ModelQATSession(
        activation_spec=activation_spec,
        weight_spec=weight_spec,
        schedule=schedule,
        observer_method=str(calibration.get("method", "min_max")),
        percentile=float(calibration.get("percentile", 99.99)),
    )
