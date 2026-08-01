from __future__ import annotations

from .compatibility import evaluate_implementation_compatibility
from .implementation_contract import ImplementationContract, validation_rank
from .selection_request import ImplementationSelectionRequest
from .selection_result import CandidateDecision, ImplementationSelectionResult


def _metric(contract: ImplementationContract, name: str, default: float) -> float:
    value = getattr(contract.metrics, name)
    return float(value) if value is not None else default


def _score(contract: ImplementationContract, policy: str) -> tuple[float, ...]:
    validation = float(validation_rank(contract.validation_level))
    latency = _metric(contract, "latency_cycles", 1e18)
    ii = _metric(contract, "initiation_interval", 1e18)
    area = sum(_metric(contract, name, 1e9) for name in ("lut", "ff", "dsp", "bram18", "uram"))
    power = _metric(contract, "power_w", 1e18)
    if policy == "validated_only":
        return (-validation, latency, area, contract.package_id)
    if policy == "latency":
        return (latency, ii, -validation, contract.package_id)
    if policy == "throughput":
        return (ii, latency, -validation, contract.package_id)
    if policy == "area":
        return (area, latency, -validation, contract.package_id)
    if policy == "power":
        return (power, latency, -validation, contract.package_id)
    return (latency, ii, area, power, -validation, contract.package_id)


def select_implementation(
    contracts: tuple[ImplementationContract, ...] | list[ImplementationContract],
    request: ImplementationSelectionRequest,
) -> ImplementationSelectionResult:
    matching = [item for item in contracts if item.operator_id == request.operator_id]
    decisions: list[CandidateDecision] = []
    compatible: list[ImplementationContract] = []
    for contract in sorted(matching, key=lambda item: (item.package_id, item.version)):
        result = evaluate_implementation_compatibility(contract, request.compatibility)
        if result.compatible:
            compatible.append(contract)
            decisions.append(CandidateDecision(contract, "compatible", (), _score(contract, request.policy)))
        else:
            decisions.append(CandidateDecision(contract, "rejected", result.reasons))

    preferred = [item for package_id in request.preferred_packages for item in compatible if item.package_id == package_id]
    selected = preferred[0] if preferred else None
    if selected is None and request.policy == "explicit_only":
        return ImplementationSelectionResult(request.operator_id, request.policy, None, tuple(decisions), ("explicit_selection_required",))
    if selected is None and request.preferred_packages and not request.allow_fallback:
        return ImplementationSelectionResult(request.operator_id, request.policy, None, tuple(decisions), ("preferred_implementation_unavailable",))
    if selected is None and compatible:
        selected = min(compatible, key=lambda item: _score(item, request.policy))
    if selected is None:
        return ImplementationSelectionResult(request.operator_id, request.policy, None, tuple(decisions), ("no_compatible_implementation",))

    final: list[CandidateDecision] = []
    for decision in decisions:
        if decision.contract.package_id == selected.package_id and decision.contract.version == selected.version:
            reason = "explicitly_preferred_and_compatible" if preferred else f"selected_by_{request.policy}"
            final.append(CandidateDecision(decision.contract, "selected", (reason,), decision.score))
        else:
            final.append(decision)
    return ImplementationSelectionResult(request.operator_id, request.policy, selected, tuple(final))
