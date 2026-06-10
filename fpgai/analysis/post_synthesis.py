from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from fpgai.analysis.hls_estimate_compare import (
    EstimateVsHlsResult,
    run_estimate_vs_hls_compare,
)
from fpgai.analysis.hls_layer_validation import (
    HlsLayerValidationResult,
    run_hls_layer_validation,
)
from fpgai.analysis.hls_module_breakdown import (
    HlsModuleBreakdownResult,
    run_hls_module_breakdown,
)


@dataclass(frozen=True)
class PostSynthesisAnalysisResult:
    estimate_comparison: EstimateVsHlsResult
    module_breakdown: Optional[HlsModuleBreakdownResult]
    layer_validation: Optional[HlsLayerValidationResult]

    @property
    def available(self) -> bool:
        return (
            self.module_breakdown is not None
            and self.module_breakdown.available
        )

    @property
    def layer_validation_available(self) -> bool:
        return (
            self.layer_validation is not None
            and self.layer_validation.available
        )


def _load_mapping(
    source: Mapping[str, Any] | str | Path,
) -> Dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)

    path = Path(source)

    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Expected a JSON object in {path}"
        )

    return payload


def _candidate_name(
    design_space_summary: Mapping[str, Any],
) -> str | None:
    value = design_space_summary.get("name")

    if value is None:
        return None

    name = str(value).strip()
    return name or None


def _find_candidate_details(
    *,
    out_dir: Path,
    candidate_name: str | None,
) -> Tuple[
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
]:
    results_path = (
        out_dir
        / "design_space"
        / "results.json"
    )

    if not results_path.is_file():
        return None, None

    try:
        payload = json.loads(
            results_path.read_text(
                encoding="utf-8",
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ):
        return None, None

    detailed = payload.get(
        "detailed_results",
        [],
    )

    if not isinstance(detailed, list):
        return None, None

    selected = None

    if candidate_name is not None:
        for row in detailed:
            if not isinstance(row, Mapping):
                continue

            if str(
                row.get("name", "")
            ).strip() == candidate_name:
                selected = row
                break

    if selected is None and len(detailed) == 1:
        selected = detailed[0]

    if not isinstance(selected, Mapping):
        return None, None

    resource_estimate = selected.get(
        "resource_estimate"
    )
    performance_estimate = selected.get(
        "performance_estimate"
    )

    if not isinstance(
        resource_estimate,
        Mapping,
    ):
        resource_estimate = None

    if not isinstance(
        performance_estimate,
        Mapping,
    ):
        performance_estimate = None

    return (
        (
            dict(resource_estimate)
            if resource_estimate is not None
            else None
        ),
        (
            dict(performance_estimate)
            if performance_estimate is not None
            else None
        ),
    )


def _resolve_estimates(
    *,
    out_dir: Path,
    design_space_summary: Mapping[str, Any],
    resource_estimate: (
        Mapping[str, Any]
        | str
        | Path
        | None
    ),
    performance_estimate: (
        Mapping[str, Any]
        | str
        | Path
        | None
    ),
) -> Tuple[
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
]:
    resolved_resource = (
        _load_mapping(resource_estimate)
        if resource_estimate is not None
        else None
    )
    resolved_performance = (
        _load_mapping(performance_estimate)
        if performance_estimate is not None
        else None
    )

    if (
        resolved_resource is not None
        and resolved_performance is not None
    ):
        return (
            resolved_resource,
            resolved_performance,
        )

    discovered_resource, discovered_performance = (
        _find_candidate_details(
            out_dir=out_dir,
            candidate_name=_candidate_name(
                design_space_summary
            ),
        )
    )

    if resolved_resource is None:
        resolved_resource = (
            discovered_resource
        )

    if resolved_performance is None:
        resolved_performance = (
            discovered_performance
        )

    return (
        resolved_resource,
        resolved_performance,
    )


def run_post_synthesis_analysis(
    *,
    out_dir: str | Path,
    design_space_summary: Dict[str, float],
    csynth_report_path: str | Path | None,
    clock_mhz: float,
    top_name: str,
    resource_estimate: (
        Mapping[str, Any]
        | str
        | Path
        | None
    ) = None,
    performance_estimate: (
        Mapping[str, Any]
        | str
        | Path
        | None
    ) = None,
    print_terminal_summary: bool = True,
) -> PostSynthesisAnalysisResult:
    output_root = Path(out_dir).resolve()

    comparison = run_estimate_vs_hls_compare(
        out_dir=output_root,
        design_space_summary=design_space_summary,
        csynth_report_path=csynth_report_path,
        clock_mhz=clock_mhz,
    )

    module_breakdown = None
    layer_validation = None

    if csynth_report_path is not None:
        module_breakdown = (
            run_hls_module_breakdown(
                out_dir=output_root,
                report_path=csynth_report_path,
                top_name=top_name,
            )
        )

    (
        resolved_resource_estimate,
        resolved_performance_estimate,
    ) = _resolve_estimates(
        out_dir=output_root,
        design_space_summary=(
            design_space_summary
        ),
        resource_estimate=resource_estimate,
        performance_estimate=(
            performance_estimate
        ),
    )

    can_validate_layers = (
        module_breakdown is not None
        and module_breakdown.available
        and resolved_resource_estimate is not None
        and resolved_performance_estimate is not None
    )

    if can_validate_layers:
        layer_validation = (
            run_hls_layer_validation(
                out_dir=output_root,
                resource_estimate=(
                    resolved_resource_estimate
                ),
                performance_estimate=(
                    resolved_performance_estimate
                ),
                module_breakdown=(
                    module_breakdown.results_json
                ),
            )
        )

    if print_terminal_summary:
        print()
        print(comparison.terminal_summary)
        print()

        if module_breakdown is not None:
            print(
                module_breakdown.terminal_summary
            )
            print()

        if layer_validation is not None:
            print(
                layer_validation.terminal_summary
            )
            print()
        elif module_breakdown is not None:
            print(
                "[FPGAI] Layer validation skipped: "
                "the selected design-space candidate's "
                "detailed estimates were unavailable."
            )
            print()

    return PostSynthesisAnalysisResult(
        estimate_comparison=comparison,
        module_breakdown=module_breakdown,
        layer_validation=layer_validation,
    )