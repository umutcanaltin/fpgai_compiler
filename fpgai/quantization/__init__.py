from fpgai.quantization.hardware import (
    RequantizationContract,
    derive_requantization_contract,
    quantization_parameters_from_tensor,
    requantize_integer,
)
from fpgai.quantization.reports import write_model_ptq_report, write_model_qat_report
from fpgai.quantization.dataset import CalibrationDatasetError, load_calibration_samples
from fpgai.quantization.model_ptq import ModelPTQResult, TensorPTQResult, calibrate_model_ptq, dequantized_constant
from fpgai.quantization.model_qat import ModelQATResult, ModelQATSession, QATSchedule, QATTensorState, TensorQATResult, model_qat_session_from_config
from fpgai.quantization.contracts import (
    QuantizationParameters,
    QuantizationSpec,
    quantization_spec_from_mapping,
)
from fpgai.quantization.observers import MinMaxObserver, ObserverError, PercentileObserver
from fpgai.quantization.parameters import derive_quantization_parameters
from fpgai.quantization.ptq import PTQCalibrationResult, calibrate_ptq, dequantize, fake_quantize, quantize
from fpgai.quantization.qat import FakeQuantResult, qat_fake_quant_forward, straight_through_gradient
from fpgai.quantization.validation import QuantizationValidationResult, validate_fake_quantization

__all__ = [
    "RequantizationContract",
    "derive_requantization_contract",
    "quantization_parameters_from_tensor",
    "requantize_integer",
    "write_model_ptq_report",
    "write_model_qat_report",
    "load_calibration_samples",
    "dequantized_constant",
    "calibrate_model_ptq",
    "TensorPTQResult",
    "ModelPTQResult",
    "CalibrationDatasetError",
    "FakeQuantResult",
    "ModelQATResult",
    "ModelQATSession",
    "QATSchedule",
    "QATTensorState",
    "TensorQATResult",
    "model_qat_session_from_config",
    "MinMaxObserver",
    "ObserverError",
    "PTQCalibrationResult",
    "PercentileObserver",
    "QuantizationParameters",
    "QuantizationSpec",
    "QuantizationValidationResult",
    "calibrate_ptq",
    "dequantize",
    "derive_quantization_parameters",
    "fake_quantize",
    "quantization_spec_from_mapping",
    "qat_fake_quant_forward",
    "quantize",
    "straight_through_gradient",
    "apply_model_qat_to_hls_graph",
    "apply_model_ptq_to_hls_graph",
    "apply_quantized_model_to_hls_graph",
    "validate_fake_quantization",
]

from .hls_lowering import (
    QuantizedHLSLoweringResult,
    apply_model_ptq_to_hls_graph,
    apply_model_qat_to_hls_graph,
    apply_quantized_model_to_hls_graph,
    execute_quantized_hls_reference,
)

from .partitioning import (
    TerminalQuantizedOperatorPartition,
    ResidualQuantizedOperatorPartition,
    partition_terminal_relu,
    partition_residual_add_and_terminal_relu,
)

from .vhdl_lowering import emit_quantized_add_int8x4_vhdl_package, emit_quantized_relu_int8x4_vhdl_package
