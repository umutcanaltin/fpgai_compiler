from .compatibility import CompatibilityRequest, CompatibilityResult, evaluate_implementation_compatibility
from .implementation_contract import (
    ImplementationContract,
    TrainingImplementationCapabilities,
    implementation_contract_from_manifest,
    validate_implementation_contract,
    resolve_architecture_parameters,
)
from .implementation_registry_adapter import implementation_contract_to_registry_entry
from .implementation_selector import select_implementation
from .selection_request import ImplementationSelectionRequest
from .selection_result import CandidateDecision, ImplementationSelectionResult
from .hls_integration import ExternalHLSProjectRequest, ExternalHLSProjectResult, emit_external_hls_operator_project

__all__ = [
    "CandidateDecision",
    "CompatibilityRequest",
    "CompatibilityResult",
    "ImplementationContract",
    "ImplementationSelectionRequest",
    "ImplementationSelectionResult",
    "TrainingImplementationCapabilities",
    "ExternalHLSProjectRequest",
    "ExternalHLSProjectResult",
    "emit_external_hls_operator_project",
    "evaluate_implementation_compatibility",
    "implementation_contract_from_manifest",
    "implementation_contract_to_registry_entry",
    "select_implementation",
    "validate_implementation_contract",
    "resolve_architecture_parameters",
]

from .hls_composition import (
    ExternalNodeBinding,
    HLSCompositionError,
    HLSCompositionPlan,
    build_hls_composition_plan,
    write_composition_report,
)

__all__ += [
    "ExternalNodeBinding",
    "HLSCompositionError",
    "HLSCompositionPlan",
    "build_hls_composition_plan",
    "write_composition_report",
]

from .vhdl_integration import (
    ExternalVHDLProjectRequest,
    ExternalVHDLProjectResult,
    VHDLScalarStreamABI,
    emit_external_vhdl_operator_project,
    parse_vhdl_scalar_stream_abi,
    validate_vhdl_integration_contract,
    run_external_vhdl_project,
)

__all__ += [
    "ExternalVHDLProjectRequest",
    "ExternalVHDLProjectResult",
    "VHDLScalarStreamABI",
    "emit_external_vhdl_operator_project",
    "parse_vhdl_scalar_stream_abi",
    "validate_vhdl_integration_contract", "run_external_vhdl_project",
]
