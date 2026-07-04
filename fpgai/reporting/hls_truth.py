from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from fpgai.analysis.hls_estimate_compare import parse_hls_csynth_report


@dataclass(frozen=True)
class HlsTruthArtifacts:
    hls_synthesis_report_json: Path
    hls_synthesis_report_md: Path
    estimate_vs_hls_json: Path
    estimate_vs_hls_md: Path

    def as_dict(self) -> dict[str, Path]:
        return {
            "hls_synthesis_report_json": self.hls_synthesis_report_json,
            "hls_synthesis_report_md": self.hls_synthesis_report_md,
            "estimate_vs_hls_json": self.estimate_vs_hls_json,
            "estimate_vs_hls_md": self.estimate_vs_hls_md,
        }

    def items(self):
        # Compiler manifest code serializes artifact containers through .items().
        # Keep this dataclass compatible with the existing artifact contract.
        return self.as_dict().items()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _candidate_reports(root: Path) -> list[Path]:
    if not root.exists():
        return []
    patterns = [
        "**/*_csynth.xml",
        "**/csynth.xml",
        "**/*_csynth.rpt",
        "**/csynth.rpt",
        "**/utilization.rpt",
        "**/timing.rpt",
    ]
    seen: set[Path] = set()
    out: list[Path] = []
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            resolved = path.resolve()
            if resolved not in seen and path.is_file():
                seen.add(resolved)
                out.append(path)
    return out


def _select_report(*, hls_dir: Path | None, hls_run: Any | None) -> Path | None:
    explicit = getattr(hls_run, "csynth_report", None) if hls_run is not None else None
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return path
    if hls_dir is None:
        return None
    candidates = _candidate_reports(Path(hls_dir))
    return candidates[0] if candidates else None


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _prediction_from_design_space(design_space_summary: Mapping[str, Any] | None) -> dict[str, Any]:
    src = dict(design_space_summary or {})
    return {
        "lut": _safe_int(src.get("predicted_lut", src.get("lut", 0))),
        "ff": _safe_int(src.get("predicted_ff", src.get("ff", 0))),
        "dsp": _safe_int(src.get("predicted_dsp", src.get("dsp", 0))),
        "bram18": _safe_int(src.get("predicted_bram18", src.get("bram18", src.get("bram", 0)))),
        "uram": _safe_int(src.get("predicted_uram", src.get("uram", 0))),
        "latency_cycles": _safe_float(src.get("predicted_cycles", src.get("cycles", 0.0))),
        "latency_ms": _safe_float(src.get("predicted_latency_ms", src.get("latency_ms", 0.0))),
    }


def _actual_from_parsed(parsed: Mapping[str, Any], *, clock_mhz: float) -> dict[str, Any]:
    latency_cycles = _safe_float(parsed.get("actual_latency_cycles", 0.0))
    latency_ms = latency_cycles / max(float(clock_mhz), 1e-9) / 1000.0 if latency_cycles > 0.0 else 0.0
    return {
        "lut": _safe_int(parsed.get("actual_lut", 0)),
        "ff": _safe_int(parsed.get("actual_ff", 0)),
        "dsp": _safe_int(parsed.get("actual_dsp", 0)),
        "bram18": _safe_int(parsed.get("actual_bram18", 0)),
        "uram": _safe_int(parsed.get("actual_uram", 0)),
        "latency_cycles": latency_cycles,
        "latency_ms": latency_ms,
    }


def _comparison_row(estimated: float, hls: float) -> dict[str, Any]:
    absolute_error = estimated - hls
    relative = None if hls == 0 else (absolute_error / hls) * 100.0
    return {
        "estimated": estimated,
        "hls": hls,
        "absolute_error": absolute_error,
        "relative_error_percent": relative,
    }


def _compare(estimated: Mapping[str, Any], actual: Mapping[str, Any]) -> dict[str, Any]:
    resources: dict[str, Any] = {}
    for key in ["lut", "ff", "dsp", "bram18", "uram"]:
        resources[key] = _comparison_row(_safe_float(estimated.get(key, 0)), _safe_float(actual.get(key, 0)))
    return {
        "resources": resources,
        "latency": {
            "cycles": _comparison_row(_safe_float(estimated.get("latency_cycles", 0)), _safe_float(actual.get("latency_cycles", 0))),
            "ms": _comparison_row(_safe_float(estimated.get("latency_ms", 0)), _safe_float(actual.get("latency_ms", 0))),
        },
    }


def _load_design_space_best(design_result: Any | None) -> dict[str, Any] | None:
    if design_result is None:
        return None
    try:
        payload = json.loads(Path(design_result.results_json).read_text(encoding="utf-8"))
    except Exception:
        return None
    best = (
        payload.get("recommended_balanced")
        or payload.get("recommended_smallest_valid")
        or payload.get("recommended_best_accuracy")
    )
    return best if isinstance(best, dict) else None


def _md_from_report(payload: Mapping[str, Any]) -> str:
    lines = ["# HLS Synthesis Report", "", f"Status: {payload.get('status')}", f"Paper-safe: {payload.get('paper_safe')}"]
    reason = payload.get("reason")
    if reason:
        lines.extend(["", f"Reason: {reason}"])
    report_path = payload.get("report_path")
    if report_path:
        lines.extend(["", f"Report path: `{report_path}`"])
    actual = payload.get("actual")
    if isinstance(actual, Mapping):
        lines.extend(["", "## Parsed HLS values"])
        for key in ["lut", "ff", "dsp", "bram18", "uram", "latency_cycles", "latency_ms"]:
            lines.append(f"- {key}: {actual.get(key)}")
    return "\n".join(lines) + "\n"


def _md_from_comparison(payload: Mapping[str, Any]) -> str:
    lines = ["# Estimate vs HLS", "", f"Status: {payload.get('status')}", f"Paper-safe: {payload.get('paper_safe')}"]
    reason = payload.get("reason")
    if reason:
        lines.extend(["", f"Reason: {reason}"])
    comp = payload.get("comparison")
    if isinstance(comp, Mapping):
        lines.extend(["", "## Resources"])
        resources = comp.get("resources", {})
        if isinstance(resources, Mapping):
            for key, row in resources.items():
                if isinstance(row, Mapping):
                    lines.append(
                        f"- {key}: estimated={row.get('estimated')}, hls={row.get('hls')}, "
                        f"relative_error_percent={row.get('relative_error_percent')}"
                    )
        latency = comp.get("latency", {})
        if isinstance(latency, Mapping):
            lines.extend(["", "## Latency"])
            for key, row in latency.items():
                if isinstance(row, Mapping):
                    lines.append(
                        f"- {key}: estimated={row.get('estimated')}, hls={row.get('hls')}, "
                        f"relative_error_percent={row.get('relative_error_percent')}"
                    )
    return "\n".join(lines) + "\n"


def emit_hls_truth_reports(
    *,
    out_dir: str | Path,
    hls_dir: str | Path | None,
    build_stages: Mapping[str, bool],
    hls_run: Any | None,
    design_result: Any | None,
    clock_mhz: float,
) -> HlsTruthArtifacts:
    out_root = Path(out_dir)
    reports_dir = out_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    hls_report_json = reports_dir / "hls_synthesis_report.json"
    hls_report_md = reports_dir / "hls_synthesis_report.md"
    estimate_json = reports_dir / "estimate_vs_hls.json"
    estimate_md = reports_dir / "estimate_vs_hls.md"

    synthesis_requested = bool(build_stages.get("hls_synthesis", False))
    hls_project_requested = bool(build_stages.get("hls_project", False))

    selected = _select_report(hls_dir=Path(hls_dir) if hls_dir is not None else None, hls_run=hls_run)
    tool_returncode = getattr(hls_run, "returncode", None) if hls_run is not None else None
    tool_ok = getattr(hls_run, "ok", None) if hls_run is not None else None

    if not synthesis_requested:
        status = "not_requested"
        reason = "build.stages.hls_synthesis is false. No HLS synthesis success is claimed."
        actual = None
    elif hls_run is not None and tool_returncode == 127:
        status = "tool_missing"
        reason = "Vitis/Vivado HLS executable was not found or could not be launched."
        actual = None
    elif hls_run is not None and tool_ok is False and selected is None:
        status = "failed"
        reason = "HLS synthesis was requested, but the HLS run failed and no synthesis report was found."
        actual = None
    elif selected is None:
        status = "artifact_missing"
        reason = "No HLS synthesis report was found under the generated HLS project."
        actual = None
    else:
        parsed = parse_hls_csynth_report(selected)
        actual = _actual_from_parsed(parsed, clock_mhz=float(clock_mhz))
        if any(_safe_float(actual.get(k, 0)) > 0 for k in ["lut", "ff", "dsp", "bram18", "latency_cycles"]):
            status = "parsed"
            reason = None
        else:
            status = "failed"
            reason = "An HLS report was found, but no resource or latency values could be parsed."

    hls_payload: dict[str, Any] = {
        "status": status,
        "stage": "hls_synthesis",
        "hls_project_requested": hls_project_requested,
        "hls_synthesis_requested": synthesis_requested,
        "paper_safe": status == "parsed",
        "claimed_success": status == "parsed",
        "reason": reason,
        "report_path": str(selected) if selected is not None else None,
        "hls_run": None if hls_run is None else {
            "ok": getattr(hls_run, "ok", None),
            "returncode": getattr(hls_run, "returncode", None),
            "command": getattr(hls_run, "command", None),
            "stdout_log": getattr(hls_run, "stdout_log", None),
            "stderr_log": getattr(hls_run, "stderr_log", None),
        },
        "actual": actual,
    }

    estimated = _prediction_from_design_space(_load_design_space_best(design_result))
    if status == "parsed" and actual is not None:
        comparison = _compare(estimated, actual)
        estimate_payload = {
            "status": "compared",
            "paper_safe": True,
            "claimed_success": True,
            "reason": None,
            "report_path": str(selected) if selected is not None else None,
            "estimated": estimated,
            "hls": actual,
            "comparison": comparison,
        }
    else:
        estimate_payload = {
            "status": status if status != "parsed" else "failed",
            "paper_safe": False,
            "claimed_success": False,
            "reason": reason or "HLS estimate comparison requires a parsed synthesis report.",
            "report_path": str(selected) if selected is not None else None,
            "estimated": estimated,
            "hls": actual,
            "comparison": None,
        }

    _write_json(hls_report_json, hls_payload)
    hls_report_md.write_text(_md_from_report(hls_payload), encoding="utf-8")
    _write_json(estimate_json, estimate_payload)
    estimate_md.write_text(_md_from_comparison(estimate_payload), encoding="utf-8")

    return HlsTruthArtifacts(
        hls_synthesis_report_json=hls_report_json,
        hls_synthesis_report_md=hls_report_md,
        estimate_vs_hls_json=estimate_json,
        estimate_vs_hls_md=estimate_md,
    )
