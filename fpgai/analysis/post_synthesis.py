from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from fpgai.analysis.hls_estimate_compare import (
    EstimateVsHlsResult,
    run_estimate_vs_hls_compare,
)
from fpgai.analysis.hls_module_breakdown import (
    HlsModuleBreakdownResult,
    run_hls_module_breakdown,
)


@dataclass(frozen=True)
class PostSynthesisAnalysisResult:
    estimate_comparison: EstimateVsHlsResult
    module_breakdown: Optional[HlsModuleBreakdownResult]

    @property
    def available(self) -> bool:
        return self.module_breakdown is not None and (
            self.module_breakdown.available
        )


def run_post_synthesis_analysis(
    *,
    out_dir: str | Path,
    design_space_summary: Dict[str, float],
    csynth_report_path: str | Path | None,
    clock_mhz: float,
    top_name: str,
    print_terminal_summary: bool = True,
) -> PostSynthesisAnalysisResult:
    comparison = run_estimate_vs_hls_compare(
        out_dir=out_dir,
        design_space_summary=design_space_summary,
        csynth_report_path=csynth_report_path,
        clock_mhz=clock_mhz,
    )

    module_breakdown = None

    if csynth_report_path is not None:
        module_breakdown = run_hls_module_breakdown(
            out_dir=out_dir,
            report_path=csynth_report_path,
            top_name=top_name,
        )

    if print_terminal_summary:
        print()
        print(comparison.terminal_summary)
        print()

        if module_breakdown is not None:
            print(module_breakdown.terminal_summary)
            print()

    return PostSynthesisAnalysisResult(
        estimate_comparison=comparison,
        module_breakdown=module_breakdown,
    )