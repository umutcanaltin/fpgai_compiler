from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


IMPORTANT_REPORTS = (
    "tiling_analysis.json",
    "tiling_resource_estimate.json",
    "tiling_performance_estimate.json",
)

OPTIONAL_REPORTS = (
    "tiling_sweep.json",
    "hls_validation.json",
)

TILED_KERNELS = (
    "dense_out_in_tiled",
    "conv2d_tiled",
)


@dataclass
class CheckItem:
    name: str
    ok: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "detail": self.detail,
        }


@dataclass
class OutputCheck:
    out_dir: str
    hls_dir: str
    ok: bool
    checks: list[CheckItem] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "fpgai.output_check.v1",
            "out_dir": self.out_dir,
            "hls_dir": self.hls_dir,
            "ok": self.ok,
            "summary": self.summary,
            "artifacts": self.artifacts,
            "checks": [
                check.to_dict()
                for check in self.checks
            ],
        }


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return loaded if isinstance(loaded, dict) else None


def _find_csynth_reports(hls_dir: Path) -> list[str]:
    return [
        str(path)
        for path in sorted(hls_dir.rglob("csynth.rpt"))
    ]


def _read_generated_sources(hls_dir: Path) -> str:
    src_dir = hls_dir / "src"
    include_dir = hls_dir / "include"

    text_parts: list[str] = []
    for directory in (src_dir, include_dir):
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if path.suffix.lower() not in {".cpp", ".h", ".hpp"}:
                continue
            try:
                text_parts.append(
                    path.read_text(
                        encoding="utf-8",
                        errors="ignore",
                    )
                )
            except Exception:
                continue

    return "\n".join(text_parts)


def _parse_csynth_summary(path: Path) -> dict[str, Any]:
    text = path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    summary: dict[str, Any] = {
        "path": str(path),
    }

    latency_match = re.search(
        r"Latency\s*\(cycles\).*?\n.*?\|\s*([0-9]+)\s*\|\s*([0-9]+)\s*\|",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if latency_match:
        summary["latency_cycles_min"] = int(latency_match.group(1))
        summary["latency_cycles_max"] = int(latency_match.group(2))

    interval_match = re.search(
        r"Interval\s*\(cycles\).*?\n.*?\|\s*([0-9]+)\s*\|\s*([0-9]+)\s*\|",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if interval_match:
        summary["interval_cycles_min"] = int(interval_match.group(1))
        summary["interval_cycles_max"] = int(interval_match.group(2))

    for resource in ("BRAM_18K", "DSP", "FF", "LUT", "URAM"):
        match = re.search(
            rf"\|\s*{re.escape(resource)}\s*\|[^|\n]*\|\s*([0-9]+)",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            summary.setdefault("resources", {})[resource] = int(match.group(1))

    return summary


def _compact_report_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    totals = report.get("totals")
    if isinstance(totals, Mapping):
        return dict(totals)

    summary: dict[str, Any] = {}
    for key in (
        "tiled_layer_count",
        "implemented_tiled_layer_count",
        "planning_only_tiled_layer_count",
        "local_buffer_elements",
        "estimated_macs",
        "estimated_memory_words",
        "tile_buffer_bits",
        "estimated_bram18",
        "overlapped_total_cycles",
        "overlapped_latency_us",
    ):
        if key in report:
            summary[key] = report[key]

    return summary


def check_output(
    out_dir: str | Path = "generated_files",
    *,
    require_hls_validation: bool = False,
    require_tiled_kernel: bool = True,
) -> OutputCheck:
    output_dir = Path(out_dir)
    hls_dir = output_dir / "hls"
    reports_dir = hls_dir / "reports"

    checks: list[CheckItem] = []
    artifacts: dict[str, Any] = {}
    summary: dict[str, Any] = {}

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append(
            CheckItem(
                name=name,
                ok=bool(ok),
                detail=detail,
            )
        )

    add(
        "hls_dir_exists",
        hls_dir.exists(),
        str(hls_dir),
    )

    meta_path = hls_dir / "codegen_meta.json"
    meta = _load_json(meta_path)
    add(
        "codegen_meta_exists",
        meta is not None,
        str(meta_path),
    )
    if meta is not None:
        artifacts["codegen_meta"] = str(meta_path)
        summary["pipeline_mode"] = meta.get("pipeline_mode")
        summary["top_name"] = meta.get("top_name")
        summary["clk_mhz"] = meta.get("clk_mhz")
        summary["compile_plan_present"] = meta.get("compile_plan_present")

    add(
        "reports_dir_exists",
        reports_dir.exists(),
        str(reports_dir),
    )

    for filename in IMPORTANT_REPORTS:
        path = reports_dir / filename
        report = _load_json(path)
        add(
            f"{filename}_exists",
            report is not None,
            str(path),
        )
        if report is not None:
            key = filename.replace(".json", "")
            artifacts[key] = str(path)
            summary[key] = _compact_report_summary(report)

    for filename in OPTIONAL_REPORTS:
        path = reports_dir / filename
        report = _load_json(path)
        if report is not None:
            key = filename.replace(".json", "")
            artifacts[key] = str(path)
            summary[key] = _compact_report_summary(report)
            add(
                f"{filename}_exists",
                True,
                str(path),
            )
        elif filename == "hls_validation.json" and require_hls_validation:
            add(
                "hls_validation.json_exists",
                False,
                str(path),
            )

    source_text = _read_generated_sources(hls_dir)
    kernel_hits = {
        kernel: kernel in source_text
        for kernel in TILED_KERNELS
    }
    summary["tiled_kernel_hits"] = kernel_hits

    if require_tiled_kernel:
        add(
            "at_least_one_tiled_kernel_present",
            any(kernel_hits.values()),
            json.dumps(kernel_hits, sort_keys=True),
        )
    else:
        add(
            "tiled_kernel_scan_completed",
            True,
            json.dumps(kernel_hits, sort_keys=True),
        )

    csynth_reports = _find_csynth_reports(hls_dir)
    summary["csynth_reports"] = csynth_reports
    add(
        "csynth_report_discovered",
        bool(csynth_reports),
        ", ".join(csynth_reports) if csynth_reports else "No csynth.rpt found yet.",
    )

    if csynth_reports:
        parsed = []
        for report_path in csynth_reports:
            try:
                parsed.append(
                    _parse_csynth_summary(Path(report_path))
                )
            except Exception as exc:
                parsed.append(
                    {
                        "path": report_path,
                        "error": str(exc),
                    }
                )
        summary["csynth_summary"] = parsed

    ok = all(check.ok for check in checks)

    return OutputCheck(
        out_dir=str(output_dir),
        hls_dir=str(hls_dir),
        ok=ok,
        checks=checks,
        artifacts=artifacts,
        summary=summary,
    )


def write_output_check(
    out_dir: str | Path = "generated_files",
    *,
    require_hls_validation: bool = False,
    require_tiled_kernel: bool = True,
) -> OutputCheck:
    result = check_output(
        out_dir,
        require_hls_validation=require_hls_validation,
        require_tiled_kernel=require_tiled_kernel,
    )

    hls_dir = Path(result.hls_dir)
    reports_dir = hls_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    json_path = reports_dir / "output_check.json"
    md_path = reports_dir / "output_check.md"

    json_path.write_text(
        json.dumps(
            result.to_dict(),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    md_lines = [
        "# FPGAI Sprint 4 real-output check",
        "",
        f"Overall status: {'PASS' if result.ok else 'FAIL'}",
        "",
        "## Checks",
        "",
    ]
    for check in result.checks:
        md_lines.append(
            f"- [{'x' if check.ok else ' '}] {check.name}: {check.detail}"
        )

    md_lines.extend(
        [
            "",
            "## Summary",
            "",
            "```json",
            json.dumps(
                result.summary,
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
        ]
    )

    md_path.write_text(
        "\n".join(md_lines),
        encoding="utf-8",
    )

    result.artifacts["output_check"] = str(json_path)
    result.artifacts["output_check_markdown"] = str(md_path)

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate real Sprint 4 FPGAI generated output artifacts.",
    )
    parser.add_argument(
        "out_dir",
        nargs="?",
        default="generated_files",
        help="Compiler output directory. Default: generated_files",
    )
    parser.add_argument(
        "--require-hls-validation",
        action="store_true",
        help="Fail if hls/reports/hls_validation.json is absent.",
    )
    parser.add_argument(
        "--no-require-tiled-kernel",
        action="store_true",
        help="Do not fail when no tiled kernel symbol is found.",
    )

    args = parser.parse_args()

    result = write_output_check(
        args.out_dir,
        require_hls_validation=args.require_hls_validation,
        require_tiled_kernel=not args.no_require_tiled_kernel,
    )

    print(
        json.dumps(
            result.to_dict(),
            indent=2,
            sort_keys=True,
        )
    )

    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
