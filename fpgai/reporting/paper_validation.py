#!/usr/bin/env python3
"""Build a paper validation trace for compiled FPGAI examples.

The report links user/compiler knobs to generated artifacts, validation reports,
and validation boundaries.  It intentionally separates generated-artifact evidence from
HLS synthesis, Vivado implementation, bitstream, and board runtime validation.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


VALIDATION_STATIC_ARTIFACT = "static_artifact_validation"
VALIDATION_COMPILER_ESTIMATE = "compiler_estimate_validation_pending"
VALIDATION_HLS_SYNTHESIS = "hls_synthesis_validation"
VALIDATION_VIVADO_PROJECT = "vivado_project_generated"
VALIDATION_VIVADO_IMPLEMENTATION = "vivado_implementation_validation"
VALIDATION_BITSTREAM = "bitstream_validation"
VALIDATION_FPGA_RUNTIME = "fpga_runtime_validation"


@dataclass
class ValidationClaim:
    claim_id: str
    feature: str
    configuration_paths: list[str]
    compiler_decision: str
    generated_artifacts: list[str]
    validation_reports: list[str]
    validation_level: str
    paper_ready: bool
    paper_statement: str
    pending_validation: list[str]
    required_validation: list[str] = field(default_factory=list)


@dataclass
class ExampleValidation:
    example: str
    out_dir: str
    pipeline_mode: str | None
    support_level: str
    validation_status: dict[str, bool]
    artifact_status: dict[str, Any]
    claims: list[ValidationClaim]


@dataclass
class PaperValidationTrace:
    artifact_kind: str
    schema_version: int
    passed: bool
    summary: dict[str, Any]
    examples: list[ExampleValidation]
    validation_boundary: str


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _dig(obj: dict[str, Any], path: Iterable[str], default: Any = None) -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _status(report: dict[str, Any], default: str = "missing") -> str:
    if not report:
        return default
    value = report.get("status")
    if value is None and "passed" in report:
        value = "passed" if report.get("passed") is True else "failed"
    return str(value) if value is not None else default


def _passed(report: dict[str, Any]) -> bool:
    return report.get("passed") is True or report.get("status") == "passed"


def _exists(out_dir: Path, rel: str) -> bool:
    return (out_dir / rel).exists()


def _build_stages(manifest: dict[str, Any]) -> dict[str, Any]:
    stages = _dig(manifest, ("configuration", "effective", "build_stages"), {})
    return stages if isinstance(stages, dict) else {}


def _validation_status(out_dir: Path, manifest: dict[str, Any], reports: dict[str, dict[str, Any]]) -> dict[str, bool]:
    stages = _build_stages(manifest)
    hls_report = reports.get("hls_synthesis_report", {})
    vivado_impl = reports.get("vivado_implementation_report", {})
    vivado_validation = reports.get("vivado_validation_report", {})
    bitstream = reports.get("bitstream_report", {})

    hls_synthesis_passed = bool(stages.get("hls_synthesis")) and _passed(hls_report)
    vivado_implementation_passed = bool(stages.get("vivado_implementation")) and (_passed(vivado_impl) or _passed(vivado_validation))
    bitstream_validation = bool(stages.get("bitstream")) and _passed(bitstream)

    board_markers = (
        "reports/fpga_execution_report.json",
        "reports/board_execution_report.json",
        "runtime_package/fpga_run_report.json",
    )
    fpga_runtime_validation = any(_exists(out_dir, rel) for rel in board_markers)

    return {
        "compiler_estimated": bool(reports.get("resource_prediction") or reports.get("timing_prediction") or reports.get("board_fit")),
        "static_artifact_validation": bool(_passed(reports.get("movement_contract_validation", {})) or _exists(out_dir, "hls/src/deeplearn.cpp")),
        "hls_synthesis_passed": hls_synthesis_passed,
        "vivado_implementation_passed": vivado_implementation_passed,
        "bitstream_validation": bitstream_validation,
        "fpga_runtime_validation": fpga_runtime_validation,
    }


def _support_level(levels: dict[str, bool]) -> str:
    if levels.get("fpga_runtime_validation"):
        return "runtime_validated"
    if levels.get("bitstream_validation"):
        return "bitstream_package_validated"
    if levels.get("vivado_implementation_passed"):
        return "vivado_implementation_validated"
    if levels.get("hls_synthesis_passed"):
        return "hls_synthesis_validated"
    if levels.get("static_artifact_validation") or levels.get("compiler_estimated"):
        return "compiler_only"
    return "required_validation"


def _report_statuses(out_dir: Path) -> dict[str, dict[str, Any]]:
    names = [
        "board_fit",
        "config_contract",
        "data_movement_plan",
        "generated_cpp_readability",
        "generated_cpp_validation",
        "generated_hls_explanation",
        "hardware_knob_contract",
        "hls_synthesis_report",
        "movement_contract_validation",
        "ps_pl_transfer_plan",
        "resolved_config",
        "resource_prediction",
        "timing_prediction",
        "training_io_movement",
        "vivado_bd_validation",
        "vivado_implementation_report",
        "vivado_validation_report",
        "bitstream_report",
    ]
    reports: dict[str, dict[str, Any]] = {}
    for name in names:
        reports[name] = _read_json(out_dir / "reports" / f"{name}.json")
    return reports


def _artifact_status(out_dir: Path, manifest: dict[str, Any], reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    stages = _build_stages(manifest)
    return {
        "manifest": _exists(out_dir, "manifest.json"),
        "hls_sources": _exists(out_dir, "hls/src/deeplearn.cpp"),
        "host_cpp": _exists(out_dir, "hostcpp") or _exists(out_dir, "host_cpp"),
        "runtime_package": _exists(out_dir, "runtime_package/package_manifest.json"),
        "movement_contract_validation": _status(reports.get("movement_contract_validation", {})),
        "generated_cpp_validation": _status(reports.get("generated_cpp_validation", {})),
        "generated_cpp_readability": _status(reports.get("generated_cpp_readability", {})),
        "generated_hls_explanation": _status(reports.get("generated_hls_explanation", {})),
        "board_fit": _status(reports.get("board_fit", {})),
        "hls_synthesis": _status(reports.get("hls_synthesis_report", {}), "not_requested" if not stages.get("hls_synthesis") else "missing"),
        "vivado_project_requested": bool(stages.get("vivado_project")),
        "vivado_bd_tcl": _exists(out_dir, "vivado/bd.tcl"),
        "vivado_project_tcl": _exists(out_dir, "vivado/project.tcl"),
        "vivado_bd_validation": _status(reports.get("vivado_bd_validation", {}), "not_requested" if not stages.get("vivado_project") else "missing"),
        "vivado_implementation": _status(reports.get("vivado_implementation_report", {}), "not_requested" if not stages.get("vivado_implementation") else "missing"),
        "vivado_validation": _status(reports.get("vivado_validation_report", {}), "not_requested" if not stages.get("vivado_implementation") else "missing"),
        "bitstream": _status(reports.get("bitstream_report", {}), "not_requested" if not stages.get("bitstream") else "missing"),
    }


def _claim_build_backend(out_dir: Path, manifest: dict[str, Any], reports: dict[str, dict[str, Any]]) -> ValidationClaim:
    stages = _build_stages(manifest)
    requested = [f"build.stages.{k}={v}" for k, v in sorted(stages.items())]
    missing: list[str] = []
    evidence = ["manifest.json:configuration.effective.build_stages"]
    if stages.get("cpp") and _exists(out_dir, "hls/src/deeplearn.cpp"):
        evidence.append("hls/src/deeplearn.cpp")
    elif stages.get("cpp"):
        missing.append("hls/src/deeplearn.cpp")
    if stages.get("runtime_package") and _exists(out_dir, "runtime_package/package_manifest.json"):
        evidence.append("runtime_package/package_manifest.json")
    elif stages.get("runtime_package"):
        missing.append("runtime_package/package_manifest.json")
    return ValidationClaim(
        claim_id="build_backend_artifacts_materialized",
        feature="build_backend",
        configuration_paths=requested,
        compiler_decision=f"pipeline_mode={manifest.get('pipeline_mode')}; top_kernel={manifest.get('top_kernel_name')}",
        generated_artifacts=evidence,
        validation_reports=["manifest.json", "reports/generated_cpp_validation.json", "reports/generated_hls_explanation.json"],
        validation_level=VALIDATION_STATIC_ARTIFACT,
        paper_ready=not missing,
        paper_statement="FPGAI materialized the requested compile-stage artifacts and recorded the effective build-stage decisions in the manifest.",
        pending_validation=["HLS synthesis succeeded", "Vivado implementation succeeded", "a bitstream was generated", "the design executed on FPGA"],
        required_validation=missing,
    )


def _claim_movement(out_dir: Path, reports: dict[str, dict[str, Any]]) -> ValidationClaim:
    movement = reports.get("movement_contract_validation", {})
    data_plan = reports.get("data_movement_plan", {})
    status = _status(movement)
    safe = _passed(movement)
    missing = [] if safe else ["reports/movement_contract_validation.json:passed"]
    artifacts = ["reports/data_movement_plan.json", "reports/ps_pl_transfer_plan.json", "hls/src/deeplearn.cpp", "runtime_package/package_manifest.json"]
    return ValidationClaim(
        claim_id="data_movement_contract_materialized",
        feature="data_movement_ps_pl",
        configuration_paths=["data_movement.inputs", "data_movement.outputs", "data_movement.weights", "data_movement.labels", "data_movement.gradients"],
        compiler_decision=f"data_movement_plan.status={_status(data_plan)}; movement_contract_validation.status={status}",
        generated_artifacts=[a for a in artifacts if _exists(out_dir, a) or a.startswith("reports/")],
        validation_reports=["reports/movement_contract_validation.json", "reports/data_movement_plan.json", "reports/ps_pl_transfer_plan.json"],
        validation_level=VALIDATION_STATIC_ARTIFACT,
        paper_ready=safe,
        paper_statement="FPGAI generated and statically validated PS-PL movement paths against the resolved data-movement plan and runtime package.",
        pending_validation=["PS-PL transfers were measured on FPGA", "DMA bandwidth was measured", "runtime transfer correctness was proven on board"],
        required_validation=missing,
    )


def _claim_knobs(out_dir: Path, reports: dict[str, dict[str, Any]]) -> ValidationClaim:
    knobs_report = reports.get("hardware_knob_contract", {})
    knobs = knobs_report.get("knobs") if isinstance(knobs_report.get("knobs"), list) else []
    manual = [str(k.get("path")) for k in knobs if isinstance(k, dict) and k.get("source") == "manual_yaml"]
    changed = [str(k.get("path")) for k in knobs if isinstance(k, dict) and k.get("status") in {"changed", "clamped", "rejected"}]
    safe = bool(knobs)
    return ValidationClaim(
        claim_id="hardware_knob_decisions_traced",
        feature="optimization_precision_memory_knobs",
        configuration_paths=manual[:40],
        compiler_decision=f"hardware_knob_contract.knobs={len(knobs)}; changed_or_clamped={len(changed)}",
        generated_artifacts=["reports/hardware_knob_contract.json", "reports/resolved_config.json", "hls/include/fpgai_types.h", "hls/include/fpgai_params.h", "hls/src/deeplearn.cpp"],
        validation_reports=["reports/hardware_knob_contract.json", "reports/generated_cpp_validation.json", "reports/generated_hls_explanation.json"],
        validation_level=VALIDATION_STATIC_ARTIFACT,
        paper_ready=safe,
        paper_statement="FPGAI records hardware-relevant YAML knobs, their effective compiler decisions, and the generated artifact locations where those decisions are materialized.",
        pending_validation=["requested II was achieved by HLS", "parallelism improved latency by a measured amount", "precision reduced Vivado resources by a measured percentage"],
        required_validation=[] if safe else ["reports/hardware_knob_contract.json:knobs"],
    )


def _claim_estimates(reports: dict[str, dict[str, Any]]) -> ValidationClaim:
    present = [name for name in ("resource_prediction", "timing_prediction", "board_fit") if reports.get(name)]
    return ValidationClaim(
        claim_id="compiler_estimates_available_with_validation_boundary",
        feature="estimation",
        configuration_paths=["board", "clock_mhz", "optimization.*", "precision.*", "memory.*"],
        compiler_decision="; ".join(f"{p}=present" for p in present) or "no compiler estimate reports present",
        generated_artifacts=[f"reports/{p}.json" for p in present],
        validation_reports=["reports/board_fit.json", "reports/resource_prediction.json", "reports/timing_prediction.json"],
        validation_level=VALIDATION_COMPILER_ESTIMATE,
        paper_ready=bool(present),
        paper_statement="FPGAI emits compiler-estimated resource/timing/board-fit reports with a separate validation boundary from HLS and Vivado measurements.",
        pending_validation=["estimated resources are Vivado resources", "estimated latency is HLS latency", "board fit is proven by implementation"],
        required_validation=[] if present else ["resource_prediction/timing_prediction/board_fit report"],
    )


def _claim_training(out_dir: Path, manifest: dict[str, Any], reports: dict[str, dict[str, Any]]) -> ValidationClaim | None:
    if manifest.get("pipeline_mode") != "training_on_device":
        return None
    movement_ok = _passed(reports.get("movement_contract_validation", {}))
    training_plan = _exists(out_dir, "training/training_plan.json")
    buffer_plan = _exists(out_dir, "runtime_package/buffer_plan.json")
    missing = []
    if not movement_ok:
        missing.append("reports/movement_contract_validation.json:passed")
    if not training_plan:
        missing.append("training/training_plan.json")
    if not buffer_plan:
        missing.append("runtime_package/buffer_plan.json")
    return ValidationClaim(
        claim_id="training_pipeline_materialized",
        feature="training",
        configuration_paths=["pipeline.mode", "training.*", "data_movement.labels", "data_movement.weights", "data_movement.gradients"],
        compiler_decision="pipeline_mode=training_on_device",
        generated_artifacts=["manifest.json:pipeline_mode=training_on_device", "training/training_plan.json", "hls/src/deeplearn.cpp", "runtime_package/buffer_plan.json"],
        validation_reports=["reports/movement_contract_validation.json", "reports/training_io_movement.json", "reports/generated_cpp_validation.json"],
        validation_level=VALIDATION_STATIC_ARTIFACT,
        paper_ready=not missing,
        paper_statement="FPGAI generated and statically validated an on-device training HLS artifact with label, weight, and gradient movement paths.",
        pending_validation=["FPGA training converged", "training latency was measured", "gradient export bandwidth was measured", "optimizer correctness was proven on board"],
        required_validation=missing,
    )


def _claim_vivado(out_dir: Path, manifest: dict[str, Any], reports: dict[str, dict[str, Any]], levels: dict[str, bool]) -> ValidationClaim | None:
    stages = _build_stages(manifest)
    bd = reports.get("vivado_bd_validation", {})
    project_generated = _exists(out_dir, "vivado/project.tcl") and _exists(out_dir, "vivado/bd.tcl")
    vivado_requested = bool(stages.get("vivado_project"))

    # Many compiler runs emit a placeholder vivado_bd_validation.json with
    # status=not_requested.  That placeholder is useful status evidence, but it
    # must not create a failing Vivado handoff claim for examples that did not
    # request Vivado project generation.  Create the claim only when Vivado was
    # requested, artifacts exist, or structural validation actually passed.
    if not vivado_requested and not project_generated and not _passed(bd):
        return None

    missing = [] if project_generated and _passed(bd) else ["vivado/project.tcl + vivado/bd.tcl + reports/vivado_bd_validation.json:passed"]
    return ValidationClaim(
        claim_id="vivado_project_handoff_generated",
        feature="vivado_project_generation",
        configuration_paths=["build.stages.vivado_project", "board", "data_movement.*"],
        compiler_decision=f"vivado_project_requested={bool(stages.get('vivado_project'))}; vivado_bd_validation.status={_status(bd)}",
        generated_artifacts=["vivado/project.tcl", "vivado/bd.tcl", "vivado/run_vivado.tcl", "reports/vivado_bd_validation.json"],
        validation_reports=["reports/vivado_bd_validation.json", "manifest.json:configuration.effective.build_stages"],
        validation_level=VALIDATION_VIVADO_IMPLEMENTATION if levels.get("vivado_implementation_passed") else VALIDATION_VIVADO_PROJECT,
        paper_ready=not missing,
        paper_statement="FPGAI generated a Vivado project handoff and structurally validated the Tcl/BD generation for the requested interface topology.",
        pending_validation=["Vivado synthesis succeeded", "Vivado implementation met timing", "a bitstream was generated", "the overlay executed on FPGA"],
        required_validation=missing,
    )


def _claim_validation_boundary(levels: dict[str, bool]) -> ValidationClaim:
    missing = []
    if not levels.get("hls_synthesis_passed"):
        missing.append("HLS csynth report with passed status")
    if not levels.get("vivado_implementation_passed"):
        missing.append("Vivado implementation report with passed status")
    if not levels.get("bitstream_validation"):
        missing.append("bitstream report with passed status")
    if not levels.get("fpga_runtime_validation"):
        missing.append("FPGA runtime execution report")
    return ValidationClaim(
        claim_id="validation_boundary_declared",
        feature="validation_boundary",
        configuration_paths=["build.stages.hls_synthesis", "build.stages.vivado_implementation", "build.stages.bitstream", "runtime execution"],
        compiler_decision=f"hls_synthesis_passed={levels.get('hls_synthesis_passed')}; vivado_implementation_passed={levels.get('vivado_implementation_passed')}; bitstream_validation={levels.get('bitstream_validation')}; fpga_runtime_validation={levels.get('fpga_runtime_validation')}",
        generated_artifacts=[],
        validation_reports=["reports/hls_synthesis_report.json", "reports/vivado_implementation_report.json", "reports/bitstream_report.json"],
        validation_level=VALIDATION_STATIC_ARTIFACT,
        paper_ready=True,
        paper_statement="FPGAI explicitly separates compiler/static validation from HLS, Vivado, bitstream, and board runtime validation.",
        pending_validation=["compiler/static validation is enough for measured performance claims", "generated Tcl proves Vivado implementation"],
        required_validation=missing,
    )


def build_example_validation(out_dir: Path) -> ExampleValidation:
    out_dir = Path(out_dir)
    manifest = _read_json(out_dir / "manifest.json")
    reports = _report_statuses(out_dir)
    levels = _validation_status(out_dir, manifest, reports)
    statuses = _artifact_status(out_dir, manifest, reports)
    claims: list[ValidationClaim] = [
        _claim_build_backend(out_dir, manifest, reports),
        _claim_movement(out_dir, reports),
        _claim_knobs(out_dir, reports),
        _claim_estimates(reports),
        _claim_validation_boundary(levels),
    ]
    training = _claim_training(out_dir, manifest, reports)
    if training is not None:
        claims.append(training)
    vivado = _claim_vivado(out_dir, manifest, reports, levels)
    if vivado is not None:
        claims.append(vivado)

    return ExampleValidation(
        example=out_dir.name,
        out_dir=str(out_dir),
        pipeline_mode=manifest.get("pipeline_mode"),
        support_level=_support_level(levels),
        validation_status=levels,
        artifact_status=statuses,
        claims=claims,
    )


def build_paper_validation_trace(out_dirs: Iterable[Path]) -> dict[str, Any]:
    examples = [build_example_validation(Path(p)) for p in out_dirs]
    failed_examples = [e.example for e in examples if any(c.paper_ready is False for c in e.claims if c.claim_id not in {"compiler_estimates_available_with_validation_boundary"})]
    chain = PaperValidationTrace(
        artifact_kind="paper_validation_trace",
        schema_version=1,
        passed=not failed_examples,
        summary={
            "examples": len(examples),
            "passed_examples": len(examples) - len(failed_examples),
            "failed_examples": len(failed_examples),
            "failed_example_names": failed_examples,
            "claim_count": sum(len(e.claims) for e in examples),
        },
        examples=examples,
        validation_boundary="This report is a logical validation trace. It does not upgrade compiler/static validation into HLS synthesis, Vivado implementation, bitstream, or board runtime validation.",
    )
    return asdict(chain)


def _md_bool(value: bool) -> str:
    return "yes" if value else "no"


def render_markdown(chain: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# FPGAI Paper Validation Trace")
    lines.append("")
    lines.append(chain.get("validation_boundary", ""))
    lines.append("")
    summary = chain.get("summary", {})
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Passed: `{chain.get('passed')}`")
    lines.append(f"- Examples: `{summary.get('examples')}`")
    lines.append(f"- Claims: `{summary.get('claim_count')}`")
    lines.append(f"- Failed examples: `{summary.get('failed_example_names', [])}`")
    lines.append("")
    lines.append("## Example validation status")
    lines.append("")
    lines.append("| Example | Claim level | Compiler/static | HLS validation | Vivado validation | Bitstream validation | Runtime validation |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for ex in chain.get("examples", []):
        levels = ex.get("validation_status", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    str(ex.get("example")),
                    str(ex.get("support_level")),
                    _md_bool(bool(levels.get("static_artifact_validation") or levels.get("compiler_estimated"))),
                    _md_bool(bool(levels.get("hls_synthesis_passed"))),
                    _md_bool(bool(levels.get("vivado_implementation_passed"))),
                    _md_bool(bool(levels.get("bitstream_validation"))),
                    _md_bool(bool(levels.get("fpga_runtime_validation"))),
                ]
            )
            + " |"
        )
    lines.append("")
    for ex in chain.get("examples", []):
        lines.append(f"## {ex.get('example')}")
        lines.append("")
        lines.append(f"Pipeline mode: `{ex.get('pipeline_mode')}`")
        lines.append("")
        lines.append("| Claim | Feature | Validation level | Paper-ready | Paper statement | Missing validation evidence |")
        lines.append("|---|---|---|---:|---|---|")
        for claim in ex.get("claims", []):
            missing = "; ".join(claim.get("required_validation", []))
            safe = str(claim.get("paper_statement", "")).replace("|", "\\|")
            lines.append(
                f"| {claim.get('claim_id')} | {claim.get('feature')} | {claim.get('validation_level')} | {_md_bool(bool(claim.get('paper_ready')))} | {safe} | {missing} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def write_paper_validation_trace(out_dirs: Iterable[Path], output_json: Path, output_md: Path | None = None) -> dict[str, Any]:
    chain = build_paper_validation_trace(out_dirs)
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(chain, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if output_md is not None:
        output_md = Path(output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_markdown(chain), encoding="utf-8")
    return chain


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build FPGAI paper validation trace")
    parser.add_argument("out_dirs", nargs="+", help="Compiled FPGAI output directories")
    parser.add_argument("--output-json", required=True, help="Output JSON path")
    parser.add_argument("--output-md", default=None, help="Output Markdown path")
    args = parser.parse_args(argv)
    chain = write_paper_validation_trace([Path(p) for p in args.out_dirs], Path(args.output_json), Path(args.output_md) if args.output_md else None)
    print(json.dumps({"artifact_kind": chain["artifact_kind"], "passed": chain["passed"], "summary": chain["summary"]}, indent=2, sort_keys=True))
    return 0 if chain.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
