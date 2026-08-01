from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from fpgai.implementations.implementation_contract import ImplementationContract

from .errors import HLSIntegrationIssue


@dataclass(frozen=True)
class ExternalHLSProjectRequest:
    out_dir: Path | str
    contract: ImplementationContract
    operator_name: str
    operator_attributes: Mapping[str, Any]
    input_words: int
    output_words: int
    top_name: str = "deeplearn"
    part: str = "xck26-sfvc784-2LV-c"
    clock_period_ns: float = 5.0


@dataclass(frozen=True)
class ExternalHLSProjectResult:
    ok: bool
    hls_dir: Path | None
    top_cpp: Path | None
    testbench_cpp: Path | None
    run_tcl: Path | None
    report_path: Path | None
    copied_sources: tuple[Path, ...] = ()
    copied_headers: tuple[Path, ...] = ()
    generated_files: tuple[Path, ...] = ()
    issues: tuple[HLSIntegrationIssue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "fpgai.external-hls-integration/v1",
            "status": "passed" if self.ok else "failed",
            "hls_dir": str(self.hls_dir) if self.hls_dir else None,
            "top_cpp": str(self.top_cpp) if self.top_cpp else None,
            "testbench_cpp": str(self.testbench_cpp) if self.testbench_cpp else None,
            "run_tcl": str(self.run_tcl) if self.run_tcl else None,
            "report_path": str(self.report_path) if self.report_path else None,
            "copied_sources": [str(path) for path in self.copied_sources],
            "copied_headers": [str(path) for path in self.copied_headers],
            "generated_files": [str(path) for path in self.generated_files],
            "issues": [issue.to_dict() for issue in self.issues],
        }
