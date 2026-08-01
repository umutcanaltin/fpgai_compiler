from __future__ import annotations

from dataclasses import dataclass

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from .implementation_contract import ImplementationContract, validation_rank


@dataclass(frozen=True)
class CompatibilityRequest:
    mode: str
    backend: str | None = None
    language: str | None = None
    board: str | None = None
    toolchain_name: str | None = None
    toolchain_version: str | None = None
    precision: str | None = None
    input_protocol: str | None = None
    output_protocol: str | None = None
    weight_storage: str | None = None
    activation_storage: str | None = None
    minimum_validation_level: str | None = None


@dataclass(frozen=True)
class CompatibilityResult:
    compatible: bool
    reasons: tuple[str, ...]


def _toolchain_compatible(contract: ImplementationContract, request: CompatibilityRequest) -> bool:
    if not request.toolchain_name:
        return True
    declared = contract.toolchains.get(request.toolchain_name)
    if declared is None:
        return not contract.toolchains
    if not request.toolchain_version or not declared:
        return True
    for spec in declared:
        try:
            if Version(request.toolchain_version) in SpecifierSet(spec if any(ch in spec for ch in "<>=!~") else f"=={spec}"):
                return True
        except (InvalidSpecifier, InvalidVersion):
            if request.toolchain_version == spec:
                return True
    return False


def evaluate_implementation_compatibility(
    contract: ImplementationContract,
    request: CompatibilityRequest,
) -> CompatibilityResult:
    reasons: list[str] = []
    mode = request.mode.strip().lower()
    if mode == "inference" and not contract.inference:
        reasons.append("inference_not_supported")
    if mode == "training" and not contract.training.forward:
        reasons.append("training_forward_not_supported")
    if request.backend and contract.backend != request.backend:
        reasons.append("backend_mismatch")
    if request.language and contract.language != request.language:
        reasons.append("language_mismatch")
    if request.board and contract.boards and request.board not in contract.boards:
        reasons.append("board_not_supported")
    if request.precision and contract.precisions and request.precision not in contract.precisions:
        reasons.append("precision_not_supported")
    if not _toolchain_compatible(contract, request):
        reasons.append("toolchain_not_supported")
    protocols = {item.direction: item.protocol for item in contract.interfaces}
    if request.input_protocol and protocols.get("input") not in {None, request.input_protocol}:
        reasons.append("input_protocol_mismatch")
    if request.output_protocol and protocols.get("output") not in {None, request.output_protocol}:
        reasons.append("output_protocol_mismatch")
    if request.weight_storage and contract.weight_storage and request.weight_storage not in contract.weight_storage:
        reasons.append("weight_storage_not_supported")
    if request.activation_storage and contract.activation_storage and request.activation_storage not in contract.activation_storage:
        reasons.append("activation_storage_not_supported")
    if request.minimum_validation_level and validation_rank(contract.validation_level) < validation_rank(request.minimum_validation_level):
        reasons.append("validation_level_too_low")
    return CompatibilityResult(not reasons, tuple(reasons))
