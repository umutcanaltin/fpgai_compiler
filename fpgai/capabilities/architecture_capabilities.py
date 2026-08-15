from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

from fpgai.engine.models import CompilePlan, MemoryPlan


IMPLEMENTED = "implemented"
LIMITED = "limited"
PLANNING_ONLY = "planning_only"
UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class ArchitectureCapabilityIssue:
    layer_name: str
    feature: str
    status: str
    requested: Any
    effective: Any
    detail: str

    @property
    def blocks_strict_mode(self) -> bool:
        return self.status in {
            PLANNING_ONLY,
            UNSUPPORTED,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_name": self.layer_name,
            "feature": self.feature,
            "status": self.status,
            "requested": self.requested,
            "effective": self.effective,
            "detail": self.detail,
            "blocks_strict_mode": self.blocks_strict_mode,
        }


@dataclass
class ArchitectureCapabilityReport:
    strict: bool
    issues: List[ArchitectureCapabilityIssue] = field(
        default_factory=list
    )

    @property
    def blocking_issues(self) -> List[ArchitectureCapabilityIssue]:
        return [
            issue
            for issue in self.issues
            if issue.blocks_strict_mode
        ]

    @property
    def valid(self) -> bool:
        return not self.blocking_issues

    @property
    def status_counts(self) -> Dict[str, int]:
        counts = {
            IMPLEMENTED: 0,
            LIMITED: 0,
            PLANNING_ONLY: 0,
            UNSUPPORTED: 0,
        }
        for issue in self.issues:
            counts[issue.status] = (
                counts.get(issue.status, 0) + 1
            )
        return counts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": "fpgai.architecture_capabilities.v1",
            "strict": self.strict,
            "valid": self.valid,
            "status_counts": self.status_counts,
            "blocking_issue_count": len(
                self.blocking_issues
            ),
            "issues": [
                issue.to_dict()
                for issue in self.issues
            ],
        }

    def summary(self) -> str:
        counts = self.status_counts
        lines = [
            "FPGAI architecture capability report",
            f"strict                : {self.strict}",
            f"valid                 : {self.valid}",
            f"implemented           : {counts[IMPLEMENTED]}",
            f"limited               : {counts[LIMITED]}",
            f"planning_only         : {counts[PLANNING_ONLY]}",
            f"unsupported           : {counts[UNSUPPORTED]}",
            f"blocking_issue_count  : {len(self.blocking_issues)}",
        ]
        for issue in self.issues:
            lines.append(
                f"- {issue.layer_name}.{issue.feature}: "
                f"{issue.status} ({issue.detail})"
            )
        return "\n".join(lines) + "\n"


class ArchitectureCapabilityError(RuntimeError):
    def __init__(
        self,
        report: ArchitectureCapabilityReport,
    ) -> None:
        self.report = report
        details = "; ".join(
            f"{issue.layer_name}.{issue.feature}: "
            f"{issue.detail}"
            for issue in report.blocking_issues
        )
        super().__init__(
            "Architecture capability validation failed: "
            f"{details}"
        )


def _add(
    issues: List[ArchitectureCapabilityIssue],
    *,
    layer_name: str,
    feature: str,
    status: str,
    requested: Any,
    effective: Any,
    detail: str,
) -> None:
    issues.append(
        ArchitectureCapabilityIssue(
            layer_name=layer_name,
            feature=feature,
            status=status,
            requested=requested,
            effective=effective,
            detail=detail,
        )
    )


def _values_by_op(
    compile_plan: CompilePlan,
    getter,
) -> Dict[str, set[Any]]:
    result: Dict[str, set[Any]] = {}
    for layer in compile_plan.layer_plans:
        result.setdefault(layer.op_type, set()).add(
            getter(layer)
        )
    return result


def _validate_layer_features(
    compile_plan: CompilePlan,
    *,
    pipeline_mode: str,
) -> List[ArchitectureCapabilityIssue]:
    issues: List[ArchitectureCapabilityIssue] = []
    for layer in compile_plan.layer_plans:
        architecture = layer.architecture
        if architecture is None:
            _add(
                issues,
                layer_name=layer.node_name,
                feature="architecture",
                status=UNSUPPORTED,
                requested=None,
                effective=None,
                detail="No typed ArchitecturePlan is attached.",
            )
            continue

        _add(
            issues,
            layer_name=layer.node_name,
            feature="precision",
            status=IMPLEMENTED,
            requested=architecture.precision.to_dict(),
            effective=architecture.precision.to_dict(),
            detail=(
                "Layer-specific fixed-point typedefs are emitted "
                "into generated HLS."
            ),
        )

        forward_specialized = layer.op_type in {
            "Dense",
            "Conv",
        }
        _add(
            issues,
            layer_name=layer.node_name,
            feature="pipeline",
            status=(
                IMPLEMENTED
                if forward_specialized
                else PLANNING_ONLY
            ),
            requested=architecture.pipeline.to_dict(),
            effective={
                "ii": architecture.pipeline.ii
                if forward_specialized
                else "global_macro"
            },
            detail=(
                "Dense/Conv forward, backward, gradient, and update "
                "kernels receive the layer-specific pipeline II."
                if forward_specialized
                and pipeline_mode == "training_on_device"
                else "The generated forward kernel receives the "
                "layer-specific pipeline II."
                if forward_specialized
                else "This operator still uses a shared pipeline macro."
            ),
        )

        _add(
            issues,
            layer_name=layer.node_name,
            feature="parallelism",
            status=(
                IMPLEMENTED
                if forward_specialized
                else PLANNING_ONLY
            ),
            requested=architecture.parallelism.to_dict(),
            effective=(
                architecture.parallelism.to_dict()
                if forward_specialized
                else {"scope": "operator_global"}
            ),
            detail=(
                "Dense/Conv forward, backward-input, and weight-gradient "
                "kernels receive layer-specific PE/SIMD unroll values."
                if forward_specialized
                and pipeline_mode == "training_on_device"
                else "The generated forward kernel receives "
                "layer-specific PE/SIMD unroll values."
                if forward_specialized
                else "This operator still uses operator-global unroll."
            ),
        )

        partition = architecture.partitioning
        partition_requested = (
            partition.factor > 1
            or any(
                value > 1
                for value in partition.targets.values()
            )
        )
        _add(
            issues,
            layer_name=layer.node_name,
            feature="partitioning",
            status=(
                IMPLEMENTED
                if partition_requested
                and forward_specialized
                else PLANNING_ONLY
                if partition_requested
                else IMPLEMENTED
            ),
            requested=partition.to_dict(),
            effective=(
                partition.to_dict()
            ),
            detail=(
                "Forward, backward, gradient, and update arrays receive "
                "layer-specific partition pragmas."
                if partition_requested
                and forward_specialized
                and pipeline_mode == "training_on_device"
                else "Forward input, output, and weight arrays receive "
                "layer-specific partition pragmas."
                if partition_requested and forward_specialized
                else "No non-trivial partitioning was requested."
            ),
        )

        tiling = architecture.tiling
        has_tiling = bool(tiling.sizes)
        tiling_implemented = (
            has_tiling
            and forward_specialized
            and pipeline_mode in {"inference", "training_on_device"}
        )
        _add(
            issues,
            layer_name=layer.node_name,
            feature="tiling",
            status=(
                IMPLEMENTED
                if tiling_implemented or not has_tiling
                else PLANNING_ONLY
            ),
            requested=tiling.to_dict(),
            effective=(
                tiling.to_dict()
                if tiling_implemented
                else {"sizes": {}}
            ),
            detail=(
                "Dense/Conv code generation emits tiled call sites "
                "for the requested tile sizes."
                if tiling_implemented
                else "Tile sizes are recorded but generated compute loops "
                "are not tiled for this pipeline/operator yet."
                if has_tiling
                else "No tiling was requested."
            ),
        )

        buffering = architecture.buffering
        _add(
            issues,
            layer_name=layer.node_name,
            feature="buffering",
            status=(
                PLANNING_ONLY
                if buffering.double_buffer
                else IMPLEMENTED
            ),
            requested=buffering.to_dict(),
            effective={
                "mode": "single",
                "double_buffer": False,
            },
            detail=(
                "Double-buffer allocation and overlap are not emitted."
                if buffering.double_buffer
                else "Single buffering matches generated hardware."
            ),
        )

        memory = architecture.memory
        external_mode = (
            memory.weight_mode in {"ddr", "dma_ddr"}
            or memory.activation_mode == "ddr"
        )
        stream_mode = memory.weight_mode == "stream"
        _add(
            issues,
            layer_name=layer.node_name,
            feature="memory",
            status=(
                UNSUPPORTED
                if external_mode
                else LIMITED
                if stream_mode
                else IMPLEMENTED
            ),
            requested=memory.to_dict(),
            effective=(
                {"weight_mode": "embedded"}
                if external_mode
                else memory.to_dict()
            ),
            detail=(
                "DDR placement requires an AXI master interface that "
                "is not generated."
                if external_mode
                else "Runtime weight streaming is supported through "
                "the existing weight interface."
                if stream_mode
                else "Embedded weights and local activations are supported."
            ),
        )

    return issues


def _validate_memory_placements(
    memory_plan: Optional[MemoryPlan],
) -> Iterable[ArchitectureCapabilityIssue]:
    if memory_plan is None:
        return []

    issues: List[ArchitectureCapabilityIssue] = []
    for placement in memory_plan.placements:
        if (
            placement.region in {"DDR", "HOST"}
            and placement.kind not in {"input", "output"}
        ):
            _add(
                issues,
                layer_name=(
                    placement.consumer
                    or placement.producer
                    or placement.tensor_name
                ),
                feature="memory_placement",
                status=UNSUPPORTED,
                requested=placement.to_dict(),
                effective={"region": "BRAM"},
                detail=(
                    "Internal or weight tensors in DDR/HOST require "
                    "an external-memory interface; BRAM binding is "
                    "not semantically equivalent."
                ),
            )
    return issues


def validate_architecture_capabilities(
    compile_plan: CompilePlan,
    *,
    memory_plan: Optional[MemoryPlan] = None,
    pipeline_mode: str = "inference",
    strict: bool = False,
) -> ArchitectureCapabilityReport:
    report = ArchitectureCapabilityReport(
        strict=strict,
        issues=[
            *_validate_layer_features(
                compile_plan,
                pipeline_mode=pipeline_mode,
            ),
            *_validate_memory_placements(memory_plan),
        ],
    )

    if strict and not report.valid:
        raise ArchitectureCapabilityError(report)

    return report
