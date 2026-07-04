from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


_REPORTS = {
    "resolved_config": "reports/resolved_config.json",
    "config_contract": "reports/config_contract.json",
    "generated_hls_explanation": "reports/generated_hls_explanation.json",
    "generated_cpp_readability": "reports/generated_cpp_readability.json",
    "generated_cpp_validation": "reports/generated_cpp_validation.json",
    "data_movement_plan": "reports/data_movement_plan.json",
    "ps_pl_transfer_plan": "reports/ps_pl_transfer_plan.json",
    "movement_contract_validation": "reports/movement_contract_validation.json",
    "board_fit": "reports/board_fit.json",
    "hls_synthesis_report": "reports/hls_synthesis_report.json",
    "estimate_vs_hls": "reports/estimate_vs_hls.json",
    "vivado_validation_report": "reports/vivado_validation_report.json",
    "vivado_bd_validation": "reports/vivado_bd_validation.json",
    "vivado_implementation_report": "reports/vivado_implementation_report.json",
    "bitstream_report": "reports/bitstream_report.json",
}


_ARTIFACTS = {
    "manifest": "manifest.json",
    "hls_dir": "hls",
    "hls_source_dir": "hls/src",
    "runtime_package": "runtime_package/package_manifest.json",
    "vivado_project_tcl": "vivado/project.tcl",
    "vivado_bd_tcl": "vivado/bd.tcl",
}


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _status_from_report(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"exists": False, "status": "missing", "paper_safe": False}
    data = _load_json(path)
    status = data.get("status") or data.get("evidence_status") or "present"
    paper_safe = bool(
        data.get("paper_safe")
        or data.get("paper_claim_allowed") is True
        or (isinstance(data.get("paper_claim_allowed"), Mapping) and any(bool(v) for v in data.get("paper_claim_allowed", {}).values()))
    )
    return {
        "exists": True,
        "status": str(status),
        "passed": data.get("passed"),
        "blocking_failure": data.get("blocking_failure"),
        "paper_safe": paper_safe,
    }


def audit_compile_artifacts(out_dir: str | Path) -> Dict[str, Any]:
    """Audit compiler outputs produced by one FPGAI compile directory.

    This is intentionally evidence-only: it does not run HLS, Vivado, or board
    execution. It reads artifacts/reports and summarizes which claims are only
    estimated, which have tool-backed evidence, and which are missing.
    """
    root = Path(out_dir)
    manifest = _load_json(root / "manifest.json")
    artifacts = {
        name: {"path": rel, "exists": (root / rel).exists()}
        for name, rel in _ARTIFACTS.items()
    }
    reports = {
        name: {"path": rel, **_status_from_report(root / rel)}
        for name, rel in _REPORTS.items()
    }
    hls_sources = sorted(str(p.relative_to(root)) for p in (root / "hls").rglob("*") if p.suffix in {".cpp", ".h", ".hpp"}) if (root / "hls").exists() else []
    build_stages = manifest.get("build_stages") or manifest.get("resolved_config_artifacts", {})
    movement = reports.get("movement_contract_validation", {})
    board_fit = reports.get("board_fit", {})
    hls_truth = reports.get("hls_synthesis_report", {})
    vivado_truth = reports.get("vivado_validation_report", {})
    bitstream = reports.get("bitstream_report", {})
    required_baseline = [
        "manifest",
        "hls_dir",
    ]
    runtime_requested = bool(build_stages.get("runtime_package", True))
    if runtime_requested:
        required_baseline.append("runtime_package")
    vivado_project_requested = bool(build_stages.get("vivado_project", False))
    if vivado_project_requested:
        required_baseline.extend(["vivado_project_tcl", "vivado_bd_tcl"])
    missing_required = [name for name in required_baseline if not artifacts.get(name, {}).get("exists")]
    missing_reports = [name for name, entry in reports.items() if not entry.get("exists") and name in {"resolved_config", "config_contract", "generated_hls_explanation", "data_movement_plan", "movement_contract_validation", "board_fit"}]
    blocking = list(missing_required) + [f"report:{name}" for name in missing_reports]
    if movement.get("blocking_failure") is True:
        blocking.append("movement_contract_validation")
    if board_fit.get("blocking_failure") is True:
        blocking.append("board_fit")
    return {
        "schema_version": 1,
        "artifact_kind": "artifact_smoke_audit",
        "out_dir": str(root),
        "status": "failed" if blocking else "passed",
        "passed": not bool(blocking),
        "blocking_failures": blocking,
        "manifest_summary": {
            "exists": bool((root / "manifest.json").exists()),
            "pipeline_mode": manifest.get("pipeline_mode"),
            "top_kernel_name": manifest.get("top_kernel_name"),
            "model_path": manifest.get("model_path"),
            "build_stages": build_stages,
            "runtime_package_requested": runtime_requested,
            "vivado_project_requested": vivado_project_requested,
        },
        "artifacts": artifacts,
        "reports": reports,
        "hls_sources": hls_sources,
        "evidence_levels": {
            "compiler_estimated": bool(reports.get("generated_hls_explanation", {}).get("exists") and reports.get("board_fit", {}).get("exists")),
            "hls_truth": hls_truth.get("status") in {"synthesized", "parsed", "compared", "passed"},
            "vivado_truth": vivado_truth.get("status") in {"validated", "implemented", "passed"},
            "bitstream_truth": bitstream.get("status") in {"validated", "generated", "passed"},
            "fpga_execution_truth": False,
        },
    }


def build_artifact_smoke_suite(out_dirs: Iterable[str | Path]) -> Dict[str, Any]:
    audits = [audit_compile_artifacts(path) for path in out_dirs]
    return {
        "schema_version": 1,
        "artifact_kind": "artifact_smoke_suite",
        "status": "failed" if any(not item.get("passed") for item in audits) else "passed",
        "passed": all(bool(item.get("passed")) for item in audits),
        "summary": {
            "runs": len(audits),
            "passed": sum(1 for item in audits if item.get("passed")),
            "failed": sum(1 for item in audits if not item.get("passed")),
        },
        "runs": audits,
    }


def write_artifact_smoke_suite(out_dirs: Iterable[str | Path], output: str | Path) -> Path:
    report = build_artifact_smoke_suite(out_dirs)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit FPGAI compile output artifacts without running tools.")
    parser.add_argument("out_dirs", nargs="+", help="Compile output directories to audit.")
    parser.add_argument("--output", default=None, help="Optional JSON path for the suite report.")
    args = parser.parse_args(argv)
    report = build_artifact_smoke_suite(args.out_dirs)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
