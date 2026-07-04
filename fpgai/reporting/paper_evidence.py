#!/usr/bin/env python3
"""Build a paper-safe logical evidence chain for compiled FPGAI examples.

The report links user/compiler knobs to generated artifacts, validation reports,
and truth boundaries.  It intentionally separates generated-artifact evidence from
HLS synthesis, Vivado implementation, bitstream, and board-execution truth.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


TRUTH_COMPILER_STATIC = "generated_artifact_static_validation"
TRUTH_COMPILER_ESTIMATE = "compiler_estimate_only"
TRUTH_HLS = "hls_synthesis_truth"
TRUTH_VIVADO_PROJECT = "compiler_generated_vivado_project"
TRUTH_VIVADO_IMPL = "vivado_implementation_truth"
TRUTH_BITSTREAM = "bitstream_truth"
TRUTH_BOARD = "fpga_execution_truth"


@dataclass
class EvidenceClaim:
    claim_id: str
    feature: str
    yaml_or_resolved_knobs: list[str]
    compiler_decision: str
    artifact_evidence: list[str]
    validation_reports: list[str]
    truth_level: str
    paper_safe: bool
    safe_wording: str
    unsafe_wording: list[str]
    missing_evidence: list[str] = field(default_factory=list)


@dataclass
class ExampleEvidence:
    example: str
    out_dir: str
    pipeline_mode: str | None
    paper_safe_claim_level: str
    evidence_levels: dict[str, bool]
    artifact_status: dict[str, Any]
    claims: list[EvidenceClaim]


@dataclass
class PaperEvidenceChain:
    artifact_kind: str
    schema_version: int
    passed: bool
    summary: dict[str, Any]
    examples: list[ExampleEvidence]
    truth_boundary: str


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


def _evidence_levels(out_dir: Path, manifest: dict[str, Any], reports: dict[str, dict[str, Any]]) -> dict[str, bool]:
    stages = _build_stages(manifest)
    hls_report = reports.get("hls_synthesis_report", {})
    vivado_impl = reports.get("vivado_implementation_report", {})
    vivado_validation = reports.get("vivado_validation_report", {})
    bitstream = reports.get("bitstream_report", {})

    hls_truth = bool(stages.get("hls_synthesis")) and _passed(hls_report)
    vivado_truth = bool(stages.get("vivado_implementation")) and (_passed(vivado_impl) or _passed(vivado_validation))
    bitstream_truth = bool(stages.get("bitstream")) and _passed(bitstream)

    board_markers = (
        "reports/fpga_execution_report.json",
        "reports/board_execution_report.json",
        "runtime_package/fpga_run_report.json",
    )
    fpga_execution_truth = any(_exists(out_dir, rel) for rel in board_markers)

    return {
        "compiler_estimated": bool(reports.get("resource_prediction") or reports.get("timing_prediction") or reports.get("board_fit")),
        "generated_artifact_static_validation": bool(_passed(reports.get("movement_contract_validation", {})) or _exists(out_dir, "hls/src/deeplearn.cpp")),
        "hls_truth": hls_truth,
        "vivado_truth": vivado_truth,
        "bitstream_truth": bitstream_truth,
        "fpga_execution_truth": fpga_execution_truth,
    }


def _claim_level(levels: dict[str, bool]) -> str:
    if levels.get("fpga_execution_truth"):
        return "board_validated"
    if levels.get("bitstream_truth"):
        return "bitstream_validated"
    if levels.get("vivado_truth"):
        return "vivado_validated"
    if levels.get("hls_truth"):
        return "hls_validated"
    if levels.get("generated_artifact_static_validation") or levels.get("compiler_estimated"):
        return "compiler_only"
    return "missing_evidence"


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


def _claim_build_backend(out_dir: Path, manifest: dict[str, Any], reports: dict[str, dict[str, Any]]) -> EvidenceClaim:
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
    return EvidenceClaim(
        claim_id="build_backend_artifacts_materialized",
        feature="build_backend",
        yaml_or_resolved_knobs=requested,
        compiler_decision=f"pipeline_mode={manifest.get('pipeline_mode')}; top_kernel={manifest.get('top_kernel_name')}",
        artifact_evidence=evidence,
        validation_reports=["manifest.json", "reports/generated_cpp_validation.json", "reports/generated_hls_explanation.json"],
        truth_level=TRUTH_COMPILER_STATIC,
        paper_safe=not missing,
        safe_wording="FPGAI materialized the requested compile-stage artifacts and recorded the effective build-stage decisions in the manifest.",
        unsafe_wording=["HLS synthesis succeeded", "Vivado implementation succeeded", "a bitstream was generated", "the design executed on FPGA"],
        missing_evidence=missing,
    )


def _claim_movement(out_dir: Path, reports: dict[str, dict[str, Any]]) -> EvidenceClaim:
    movement = reports.get("movement_contract_validation", {})
    data_plan = reports.get("data_movement_plan", {})
    status = _status(movement)
    safe = _passed(movement)
    missing = [] if safe else ["reports/movement_contract_validation.json:passed"]
    artifacts = ["reports/data_movement_plan.json", "reports/ps_pl_transfer_plan.json", "hls/src/deeplearn.cpp", "runtime_package/package_manifest.json"]
    return EvidenceClaim(
        claim_id="data_movement_contract_materialized",
        feature="data_movement_ps_pl",
        yaml_or_resolved_knobs=["data_movement.inputs", "data_movement.outputs", "data_movement.weights", "data_movement.labels", "data_movement.gradients"],
        compiler_decision=f"data_movement_plan.status={_status(data_plan)}; movement_contract_validation.status={status}",
        artifact_evidence=[a for a in artifacts if _exists(out_dir, a) or a.startswith("reports/")],
        validation_reports=["reports/movement_contract_validation.json", "reports/data_movement_plan.json", "reports/ps_pl_transfer_plan.json"],
        truth_level=TRUTH_COMPILER_STATIC,
        paper_safe=safe,
        safe_wording="FPGAI generated and statically validated PS-PL movement paths against the resolved data-movement plan and runtime package.",
        unsafe_wording=["PS-PL transfers were measured on FPGA", "DMA bandwidth was measured", "runtime transfer correctness was proven on board"],
        missing_evidence=missing,
    )


def _claim_knobs(out_dir: Path, reports: dict[str, dict[str, Any]]) -> EvidenceClaim:
    knobs_report = reports.get("hardware_knob_contract", {})
    knobs = knobs_report.get("knobs") if isinstance(knobs_report.get("knobs"), list) else []
    manual = [str(k.get("path")) for k in knobs if isinstance(k, dict) and k.get("source") == "manual_yaml"]
    changed = [str(k.get("path")) for k in knobs if isinstance(k, dict) and k.get("status") in {"changed", "clamped", "rejected"}]
    safe = bool(knobs)
    return EvidenceClaim(
        claim_id="hardware_knob_decisions_traced",
        feature="optimization_precision_memory_knobs",
        yaml_or_resolved_knobs=manual[:40],
        compiler_decision=f"hardware_knob_contract.knobs={len(knobs)}; changed_or_clamped={len(changed)}",
        artifact_evidence=["reports/hardware_knob_contract.json", "reports/resolved_config.json", "hls/include/fpgai_types.h", "hls/include/fpgai_params.h", "hls/src/deeplearn.cpp"],
        validation_reports=["reports/hardware_knob_contract.json", "reports/generated_cpp_validation.json", "reports/generated_hls_explanation.json"],
        truth_level=TRUTH_COMPILER_STATIC,
        paper_safe=safe,
        safe_wording="FPGAI records hardware-relevant YAML knobs, their effective compiler decisions, and the generated artifact locations where those decisions are materialized.",
        unsafe_wording=["requested II was achieved by HLS", "parallelism improved latency by a measured amount", "precision reduced Vivado resources by a measured percentage"],
        missing_evidence=[] if safe else ["reports/hardware_knob_contract.json:knobs"],
    )


def _claim_estimates(reports: dict[str, dict[str, Any]]) -> EvidenceClaim:
    present = [name for name in ("resource_prediction", "timing_prediction", "board_fit") if reports.get(name)]
    return EvidenceClaim(
        claim_id="compiler_estimates_available_with_truth_boundary",
        feature="estimation",
        yaml_or_resolved_knobs=["board", "clock_mhz", "optimization.*", "precision.*", "memory.*"],
        compiler_decision="; ".join(f"{p}=present" for p in present) or "no compiler estimate reports present",
        artifact_evidence=[f"reports/{p}.json" for p in present],
        validation_reports=["reports/board_fit.json", "reports/resource_prediction.json", "reports/timing_prediction.json"],
        truth_level=TRUTH_COMPILER_ESTIMATE,
        paper_safe=bool(present),
        safe_wording="FPGAI emits compiler-estimated resource/timing/board-fit reports with a separate truth boundary from HLS and Vivado measurements.",
        unsafe_wording=["estimated resources are Vivado resources", "estimated latency is HLS latency", "board fit is proven by implementation"],
        missing_evidence=[] if present else ["resource_prediction/timing_prediction/board_fit report"],
    )


def _claim_training(out_dir: Path, manifest: dict[str, Any], reports: dict[str, dict[str, Any]]) -> EvidenceClaim | None:
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
    return EvidenceClaim(
        claim_id="training_pipeline_materialized",
        feature="training",
        yaml_or_resolved_knobs=["pipeline.mode", "training.*", "data_movement.labels", "data_movement.weights", "data_movement.gradients"],
        compiler_decision="pipeline_mode=training_on_device",
        artifact_evidence=["manifest.json:pipeline_mode=training_on_device", "training/training_plan.json", "hls/src/deeplearn.cpp", "runtime_package/buffer_plan.json"],
        validation_reports=["reports/movement_contract_validation.json", "reports/training_io_movement.json", "reports/generated_cpp_validation.json"],
        truth_level=TRUTH_COMPILER_STATIC,
        paper_safe=not missing,
        safe_wording="FPGAI generated and statically validated an on-device training HLS artifact with label, weight, and gradient movement paths.",
        unsafe_wording=["FPGA training converged", "training latency was measured", "gradient export bandwidth was measured", "optimizer correctness was proven on board"],
        missing_evidence=missing,
    )


def _claim_vivado(out_dir: Path, manifest: dict[str, Any], reports: dict[str, dict[str, Any]], levels: dict[str, bool]) -> EvidenceClaim | None:
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
    return EvidenceClaim(
        claim_id="vivado_project_handoff_generated",
        feature="vivado_project_generation",
        yaml_or_resolved_knobs=["build.stages.vivado_project", "board", "data_movement.*"],
        compiler_decision=f"vivado_project_requested={bool(stages.get('vivado_project'))}; vivado_bd_validation.status={_status(bd)}",
        artifact_evidence=["vivado/project.tcl", "vivado/bd.tcl", "vivado/run_vivado.tcl", "reports/vivado_bd_validation.json"],
        validation_reports=["reports/vivado_bd_validation.json", "manifest.json:configuration.effective.build_stages"],
        truth_level=TRUTH_VIVADO_IMPL if levels.get("vivado_truth") else TRUTH_VIVADO_PROJECT,
        paper_safe=not missing,
        safe_wording="FPGAI generated a Vivado project handoff and structurally validated the Tcl/BD generation for the requested interface topology.",
        unsafe_wording=["Vivado synthesis succeeded", "Vivado implementation met timing", "a bitstream was generated", "the overlay executed on FPGA"],
        missing_evidence=missing,
    )


def _claim_truth_boundary(levels: dict[str, bool]) -> EvidenceClaim:
    missing = []
    if not levels.get("hls_truth"):
        missing.append("HLS csynth report with passed status")
    if not levels.get("vivado_truth"):
        missing.append("Vivado implementation report with passed status")
    if not levels.get("bitstream_truth"):
        missing.append("bitstream report with passed status")
    if not levels.get("fpga_execution_truth"):
        missing.append("FPGA runtime execution report")
    return EvidenceClaim(
        claim_id="truth_boundary_declared",
        feature="truth_boundary",
        yaml_or_resolved_knobs=["build.stages.hls_synthesis", "build.stages.vivado_implementation", "build.stages.bitstream", "runtime execution"],
        compiler_decision=f"hls_truth={levels.get('hls_truth')}; vivado_truth={levels.get('vivado_truth')}; bitstream_truth={levels.get('bitstream_truth')}; fpga_execution_truth={levels.get('fpga_execution_truth')}",
        artifact_evidence=[],
        validation_reports=["reports/hls_synthesis_report.json", "reports/vivado_implementation_report.json", "reports/bitstream_report.json"],
        truth_level=TRUTH_COMPILER_STATIC,
        paper_safe=True,
        safe_wording="FPGAI explicitly separates compiler/static evidence from HLS, Vivado, bitstream, and board-execution truth.",
        unsafe_wording=["compiler-only evidence is enough for measured performance claims", "generated Tcl is Vivado implementation truth"],
        missing_evidence=missing,
    )


def build_example_evidence(out_dir: Path) -> ExampleEvidence:
    out_dir = Path(out_dir)
    manifest = _read_json(out_dir / "manifest.json")
    reports = _report_statuses(out_dir)
    levels = _evidence_levels(out_dir, manifest, reports)
    statuses = _artifact_status(out_dir, manifest, reports)
    claims: list[EvidenceClaim] = [
        _claim_build_backend(out_dir, manifest, reports),
        _claim_movement(out_dir, reports),
        _claim_knobs(out_dir, reports),
        _claim_estimates(reports),
        _claim_truth_boundary(levels),
    ]
    training = _claim_training(out_dir, manifest, reports)
    if training is not None:
        claims.append(training)
    vivado = _claim_vivado(out_dir, manifest, reports, levels)
    if vivado is not None:
        claims.append(vivado)

    return ExampleEvidence(
        example=out_dir.name,
        out_dir=str(out_dir),
        pipeline_mode=manifest.get("pipeline_mode"),
        paper_safe_claim_level=_claim_level(levels),
        evidence_levels=levels,
        artifact_status=statuses,
        claims=claims,
    )


def build_paper_evidence_chain(out_dirs: Iterable[Path]) -> dict[str, Any]:
    examples = [build_example_evidence(Path(p)) for p in out_dirs]
    failed_examples = [e.example for e in examples if any(c.paper_safe is False for c in e.claims if c.claim_id not in {"compiler_estimates_available_with_truth_boundary"})]
    chain = PaperEvidenceChain(
        artifact_kind="paper_evidence_chain",
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
        truth_boundary="This report is a logical evidence chain. It does not upgrade compiler/static evidence into HLS, Vivado, bitstream, or board-execution truth.",
    )
    return asdict(chain)


def _md_bool(value: bool) -> str:
    return "yes" if value else "no"


def render_markdown(chain: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# FPGAI Paper Evidence Chain")
    lines.append("")
    lines.append(chain.get("truth_boundary", ""))
    lines.append("")
    summary = chain.get("summary", {})
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Passed: `{chain.get('passed')}`")
    lines.append(f"- Examples: `{summary.get('examples')}`")
    lines.append(f"- Claims: `{summary.get('claim_count')}`")
    lines.append(f"- Failed examples: `{summary.get('failed_example_names', [])}`")
    lines.append("")
    lines.append("## Example evidence levels")
    lines.append("")
    lines.append("| Example | Claim level | Compiler/static | HLS truth | Vivado truth | Bitstream truth | FPGA truth |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for ex in chain.get("examples", []):
        levels = ex.get("evidence_levels", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    str(ex.get("example")),
                    str(ex.get("paper_safe_claim_level")),
                    _md_bool(bool(levels.get("generated_artifact_static_validation") or levels.get("compiler_estimated"))),
                    _md_bool(bool(levels.get("hls_truth"))),
                    _md_bool(bool(levels.get("vivado_truth"))),
                    _md_bool(bool(levels.get("bitstream_truth"))),
                    _md_bool(bool(levels.get("fpga_execution_truth"))),
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
        lines.append("| Claim | Feature | Truth level | Paper-safe | Safe wording | Missing evidence |")
        lines.append("|---|---|---|---:|---|---|")
        for claim in ex.get("claims", []):
            missing = "; ".join(claim.get("missing_evidence", []))
            safe = str(claim.get("safe_wording", "")).replace("|", "\\|")
            lines.append(
                f"| {claim.get('claim_id')} | {claim.get('feature')} | {claim.get('truth_level')} | {_md_bool(bool(claim.get('paper_safe')))} | {safe} | {missing} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def write_paper_evidence_chain(out_dirs: Iterable[Path], output_json: Path, output_md: Path | None = None) -> dict[str, Any]:
    chain = build_paper_evidence_chain(out_dirs)
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(chain, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if output_md is not None:
        output_md = Path(output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_markdown(chain), encoding="utf-8")
    return chain


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build FPGAI paper-safe logical evidence chain")
    parser.add_argument("out_dirs", nargs="+", help="Compiled FPGAI output directories")
    parser.add_argument("--output-json", required=True, help="Output JSON path")
    parser.add_argument("--output-md", default=None, help="Output Markdown path")
    args = parser.parse_args(argv)
    chain = write_paper_evidence_chain([Path(p) for p in args.out_dirs], Path(args.output_json), Path(args.output_md) if args.output_md else None)
    print(json.dumps({"artifact_kind": chain["artifact_kind"], "passed": chain["passed"], "summary": chain["summary"]}, indent=2, sort_keys=True))
    return 0 if chain.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
